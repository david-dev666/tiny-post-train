"""Qwen3 LoRA SFT 最小可运行脚本。

用法见 docs/00-getting-started.md。
数据默认按 alpaca 字段 instruction / input / output 处理。
"""

import os
import sys
from pathlib import Path

# 本地权重强制离线。实现移到 offline.py —— eval.py 也要用同一份，
# 而它在 vllm 环境里 import 不了本模块（下面 import unsloth 会失败）。
from offline import LOCAL_MODEL_PATH  # noqa: E402

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from datasets import concatenate_datasets, load_dataset  # noqa: E402
from transformers import TrainerCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402

# 退化指标（复读率/多样性）的唯一实现在纯模块里，离线评测也要用同一份
from metrics import repetition_metrics  # noqa: E402

# 固定使用的干净模板。**不要改回 tokenizer 自带的那个。**
#
# tokenizer 自带的是 Qwen3 **instruct** 版模板，它会把每个 assistant 回复渲染成：
#     <|im_start|>assistant\n<think>\n\n</think>\n\n{回答}<|im_end|>\n
# 而推理时 add_generation_prompt=True 只给到 `<|im_start|>assistant\n`，**不预填**
# 那个思考块，于是模型必须自己"生成"它 —— 实测生成出来的是两个异常字节
# （token 124 = 裸字节 0xC0），decode 成乱码，挂在每次输出的最前面。
#
# 后果实测：20 条指令遵循里 8 条被这段乱码判错；probes 的复读率不可比；
# MMLU 的 logits 判分被污染。而且 base 模型本来没有思考能力，
# 教它输出一个空的思考块纯属浪费容量。
CLEAN_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# 兼容旧名字（eval.py 之前 import 的是这个）
FALLBACK_CHAT_TEMPLATE = CLEAN_CHAT_TEMPLATE

# 推理采样的固定问题。别改，改了就没法和之前步数对比了。
PROBE_PROMPTS = [
    "介绍一下你自己",
    "用一句话解释什么是 LoRA",
    "写一个 Python 函数，判断一个数是不是质数",
]


def stop_token_ids(tokenizer) -> list[int]:
    """生成时要认的停止符。

    **必须带上 `<|im_end|>`。** chat 模板里每个回合都以它结尾，模型学会的就是
    用它结束回答；但它**不在** tokenizer / model 的 eos 里（那是 `<|endoftext|>`）。
    不带上，生成就不会在那里停 —— 模型「说完了还在硬说」，
    实测每条输出末尾多出一段乱码（`לחלוט` / `NdrFc` / `аци`），
    把指令遵循的判分整片带偏：答案都对，被尾巴判错。

    实测影响：MMLU 那题输出 `9. לחלוט`（本该是 `9.`）、代码题把合法 Python
    变成非法、`END` 结尾题因为尾巴而不算结尾。
    """
    vocab = tokenizer.get_vocab()
    ids = []
    for name in ("<|im_end|>", "<|endoftext|>"):
        if name in vocab:
            ids.append(vocab[name])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids


class MetricsLogger(TrainerCallback):
    """把训练指标追加写到 metrics.jsonl，供 dashboard/ 读取。

    追加写、不加锁：进程被 kill 也只丢最后一行，不会毁掉已有数据。
    写出两个文件：
      metrics.jsonl   每行一条指标，粒度由 --logging-steps 决定
      run_meta.json   训练一开始就写，含总步数与起跑时间，看板用它算 ETA
    """

    def __init__(self, metrics_path, meta=None):
        self.metrics_path = Path(metrics_path)
        self.meta_path = self.metrics_path.with_name("run_meta.json")
        self.meta = dict(meta or {})

    def on_train_begin(self, args, state, control, **kwargs):
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "started_at": int(time.time()),
            "max_steps": state.max_steps,
            "num_train_epochs": state.num_train_epochs,
            **self.meta,
        }
        self.meta_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step, "ts": int(time.time())}
        if state.epoch is not None:
            record["epoch"] = round(float(state.epoch), 4)
        for key, value in logs.items():
            # 只留数字；字符串和布尔值（比如格式化过的 runtime 字段）丢掉
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            record[key] = float(value)
        # 有 eval_loss 就顺手算 perplexity。loss 大时 exp 会溢出，超过 20 就不算了。
        if "eval_loss" in record and record["eval_loss"] < 20:
            record["perplexity"] = round(math.exp(record["eval_loss"]), 3)
        try:
            with self.metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[MetricsLogger] 写指标失败：{exc}")


class ProbeCallback(TrainerCallback):
    """每隔 N 步用固定 prompt 让当前模型生成一次，看它到底在变好还是变坏。

    结果追加写到 probes.jsonl，看板读它展示「最新推理表现」。
    loss 只说明模型拟合得怎样，不说明输出像不像人话——这个才看得出。

    设计原则是「宁可少记，不能搞挂训练」：任何异常都被吞掉并自动停用本回调，
    绝不让采样把正在跑的训练弄崩。
    """

    def __init__(self, prompts, every, out_path, max_new_tokens=128):
        self.prompts = list(prompts)
        self.every = max(1, int(every))
        self.out_path = Path(out_path)
        self.max_new_tokens = int(max_new_tokens)
        self.disabled = False

    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if self.disabled or model is None or processing_class is None:
            return
        step = state.global_step
        if step <= 0 or step % self.every != 0:
            return
        try:
            records = self._generate(model, processing_class, step)
        except Exception as exc:  # noqa: BLE001 —— 采样失败绝不能影响训练
            self.disabled = True
            print(f"[ProbeCallback] 采样失败，已自动停用：{exc!r}")
            return

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[ProbeCallback] step {step} 采样 {len(records)} 条 → {self.out_path}")

    def _generate(self, model, tokenizer, step):
        was_training = model.training
        model.eval()
        records = []
        try:
            for prompt in self.prompts:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                started = time.time()
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=stop_token_ids(tokenizer),
                    )
                answer = tokenizer.decode(
                    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                records.append(
                    {
                        "step": step,
                        "ts": int(time.time()),
                        "prompt": prompt,
                        "output": answer.strip(),
                        "sec": round(time.time() - started, 2),
                        **self.score(output[0][inputs["input_ids"].shape[1]:]),
                    }
                )
        finally:
            if was_training:
                model.train()
        return records

    @staticmethod
    def score(generated_ids) -> dict:
        """对生成结果打客观分。

        **实现只剩一份**，在 `metrics.repetition_metrics`。原先这个方法自带一份，
        换 vLLM 引擎后 `eval.py` 要在 vllm 环境里跑（那环境装不了 unsloth），
        只能把实现下沉到纯模块 —— 顺手消掉了「训练曲线与离线评测各写一份」的隐患。

        指标含义见 `metrics.repetition_metrics` 的说明。
        """
        return repetition_metrics(generated_ids)


LETTERS = ("A", "B", "C", "D")


class BenchCallback(TrainerCallback):
    """每隔 N 步在一份固定的小子集上跑一次多选题，看知识有没有崩。

    为什么不用全量：MMLU 14042 题，4B 上跑一遍要一两个小时，比训练还慢。
    分层抽几百题后一次几十秒，才塞得进训练循环。

    为什么必须固定子集：每次换题的话，题目难度变化会被误读成模型退步。
    子集由 scripts/bench_subset.py 生成，训练时只读不抽。

    判分方式：只比 A/B/C/D 四个 token 的 logits，不做生成，一次前向就够。
    所以它测的是「模型更倾向输出哪个字母」，**不是**「模型会不会答题」。
    """

    def __init__(self, subset_path, every, out_path, tokenizer, batch_size=16, max_length=512):
        self.subset_path = Path(subset_path)
        self.every = max(1, int(every))
        self.out_path = Path(out_path)
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length)
        self.questions = self._load()
        self.letter_ids = self._letter_token_ids(tokenizer)
        self.disabled = False

    def _load(self) -> list[dict]:
        items = []
        with self.subset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        if not items:
            raise ValueError(f"{self.subset_path} 是空的")
        return items

    @staticmethod
    def _letter_token_ids(tokenizer) -> dict:
        """A/B/C/D 各自的候选 token id。

        有的分词器把 "A" 和 " A" 切成不同 token，两种都收着，判分时取较大的 logit。
        """
        ids = {}
        for letter in LETTERS:
            candidates = set()
            for text in (letter, f" {letter}"):
                encoded = tokenizer.encode(text, add_special_tokens=False)
                if len(encoded) == 1:
                    candidates.add(encoded[0])
            if not candidates:
                raise ValueError(f"分词器无法把 {letter} 切单个 token，这个模型不适合做 MCQ 判分")
            ids[letter] = sorted(candidates)
        return ids

    @staticmethod
    def _build_prompt(tokenizer, item) -> str:
        options = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(LETTERS, item["choices"])
        )
        content = (
            "以下是一道单项选择题，请直接回答正确选项的字母。\n\n"
            f"{item['question']}\n\n{options}\n\n答案："
        )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if self.disabled or model is None or processing_class is None:
            return
        step = state.global_step
        if step <= 0 or step % self.every != 0:
            return
        try:
            result = self.evaluate(model, processing_class, step)
        except Exception as exc:  # noqa: BLE001 —— 评测失败绝不能影响训练
            self.disabled = True
            print(f"[BenchCallback] 评测失败，已自动停用：{exc!r}")
            return

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            f"[BenchCallback] step {step} MMLU 子集 "
            f"{result['correct']}/{result['total']} = {result['accuracy']:.1%}"
            f"（{result['sec']}s，随机基准 25%）"
        )

    def evaluate(self, model, tokenizer, step) -> dict:
        """公开给 scripts/eval.py 用，保证离线评测和训练曲线是同一套判分口径。"""
        was_training = model.training
        saved_padding = tokenizer.padding_side
        # 取最后一个位置的 logits，所以必须左填充，否则末位是 pad
        tokenizer.padding_side = "left"
        started = time.time()
        correct = 0
        by_subject: dict[str, list[int]] = {}
        try:
            for start in range(0, len(self.questions), self.batch_size):
                chunk = self.questions[start : start + self.batch_size]
                prompts = [self._build_prompt(tokenizer, item) for item in chunk]
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(model.device)
                with torch.no_grad():
                    # logits_to_keep=1 只算最后一个位置，别把 16×512×151669 的
                    # 完整 logits 物化出来——实测这是显存尖峰的大头。
                    # 老版本 transformers 没这个参数，退回去就是了。
                    try:
                        output = model(**encoded, logits_to_keep=1)
                    except TypeError:
                        output = model(**encoded)
                    logits = output.logits[:, -1, :]
                # 每个选项取它所有候选 token 里最大的 logit
                per_letter = [
                    torch.stack([logits[:, tid] for tid in self.letter_ids[letter]]).max(dim=0).values
                    for letter in LETTERS
                ]
                predictions = torch.stack(per_letter, dim=1).argmax(dim=1).tolist()
                for item, prediction in zip(chunk, predictions):
                    hit = int(prediction == item["answer"])
                    correct += hit
                    by_subject.setdefault(item["subject"], []).append(hit)
        finally:
            tokenizer.padding_side = saved_padding
            if was_training:
                model.train()

        total = len(self.questions)
        return {
            "step": step,
            "ts": int(time.time()),
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "chance": 0.25,
            "sec": round(time.time() - started, 1),
            "by_subject": {
                subject: round(sum(hits) / len(hits), 3)
                for subject, hits in sorted(by_subject.items())
            },
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="本地基座权重路径或 HF/ModelScope id")
    p.add_argument("--data", required=True, help="本地数据目录或 HF 数据集名")
    p.add_argument("--output", required=True, help="输出目录")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--load-in-4bit", action="store_true", help="显存不足时打开")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--report-to", default="tensorboard",
                   help="日志后端：tensorboard / wandb / none")
    p.add_argument("--metrics-file", default=None,
                   help="指标 jsonl 路径，默认 <output>/metrics.jsonl")
    p.add_argument("--probe-every", type=int, default=0,
                   help="每 N 步做一次推理采样，0 表示关闭。冒烟可设 10")
    p.add_argument("--probe-file", default=None,
                   help="采样结果路径，默认 <output>/probes.jsonl")
    p.add_argument("--probe-max-new-tokens", type=int, default=128)
    p.add_argument("--eval-ratio", type=float, default=0.02,
                   help="从训练集切多少当验证集，0 表示不切")
    p.add_argument("--eval-steps", type=int, default=50,
                   help="每多少步在验证集上评一次（需要 --eval-ratio > 0）")
    p.add_argument("--bench-every", type=int, default=0,
                   help="每 N 步跑一次 MMLU 子集评测，0 表示关闭")
    p.add_argument("--bench-subset", default="evals/mmlu-subset.jsonl",
                   help="固定评测子集，由 scripts/bench_subset.py 生成")
    p.add_argument("--bench-file", default=None,
                   help="评测结果路径，默认 <output>/bench.jsonl")
    p.add_argument("--bench-batch-size", type=int, default=16)
    p.add_argument("--tokenizer-chat-template", action="store_true",
                   help="退回 tokenizer 自带模板（默认用项目内置的干净模板，见 CLEAN_CHAT_TEMPLATE 注释）")
    return p.parse_args()


METADATA_FILENAMES = {"dataset_infos.json", "dataset_info.json", ".gitattributes"}


def _looks_like_metadata(path: Path) -> bool:
    """判断一个文件是不是「跟着数据集一起下下来的元信息」。

    modelscope / HuggingFace 的快照里会带这些东西：
        dataset_infos.json          字段 schema
        <名字>.json                  指向真实数据文件的配置，形如
                                    {"default": {"train": {"meta": "train.csv"}}}
        README.md / .gitattributes
    把它们当训练数据读，schema 对不上，会直接崩在 DatasetGenerationError。

    判定依据：真正的指令数据顶层是**数组**；顶层是**字典**的基本都是元信息/配置。
    """
    if path.name in METADATA_FILENAMES or path.name.startswith("README"):
        return True
    if path.suffix.lower() != ".json":
        return False

    try:
        with path.open("r", encoding="utf-8") as f:
            head = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False

    if not isinstance(head, dict):
        return False
    # 配置文件的形态：{"default": {"train": {"meta": "train.csv", "file": ""}}}
    values = list(head.values())
    if not values or not all(isinstance(v, dict) for v in values):
        return False
    inner = [item for value in values for item in value.values()]
    return bool(inner) and all(isinstance(item, dict) for item in inner)


def _clean(value) -> str:
    """CSV 里的空字段读出来是 NaN 而不是空串，直接 .strip() 会炸。"""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def load_raw(data_path):
    """本地目录按后缀分组加载，csv / jsonl / json / parquet 都支持。

    会自动跳过元信息文件——alpaca 这份在 modelscope 上的快照，
    真实数据是 train.csv，json 全是配置，不排除就会读错。
    """
    if os.path.isdir(data_path):
        parts = []
        for ext in ("csv", "jsonl", "json", "parquet"):
            files = [
                path
                for path in sorted(glob.glob(os.path.join(data_path, f"*.{ext}")))
                if not _looks_like_metadata(Path(path))
            ]
            if not files:
                continue
            print(f"  读到 {ext}: {[os.path.basename(p) for p in files]}")
            parts.append(load_dataset(ext, data_files=files, split="train"))
        if not parts:
            raise FileNotFoundError(
                f"{data_path} 里没找到可用的数据文件"
                "（支持 csv / jsonl / json / parquet），或者文件全是元信息"
            )
        return parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    return load_dataset(data_path, split="train")


def build_user_text(example):
    user = _clean(example.get("instruction"))
    extra = _clean(example.get("input"))
    if extra:
        user = f"{user}\n{extra}"
    return user


def to_text(example, tokenizer, chat_template):
    messages = [
        {"role": "user", "content": build_user_text(example)},
        {"role": "assistant", "content": _clean(example.get("output"))},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, chat_template=chat_template
        )
    }


def _config_fields() -> set:
    """TrainingArguments 本身是 dataclass，直接问它支持哪些字段名。

    transformers 改过名字（max_seq_length → max_length、evaluation_strategy →
    eval_strategy），与其猜版本不如查自己。
    """
    try:
        return {f.name for f in dataclasses.fields(SFTConfig)}
    except TypeError:
        return set()


def build_config(args, common):
    fields = _config_fields()
    if fields:
        length_key = "max_seq_length" if "max_seq_length" in fields else "max_length"
        return SFTConfig(**{length_key: args.max_seq_len}, **common)
    try:
        return SFTConfig(max_seq_length=args.max_seq_len, **common)
    except TypeError:
        return SFTConfig(max_length=args.max_seq_len, **common)


def build_trainer(model, tokenizer, train_dataset, config, callbacks=None, eval_dataset=None):
    kwargs = dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=config,
        callbacks=callbacks,
    )
    try:
        return SFTTrainer(model=model, processing_class=tokenizer, **kwargs)
    except TypeError:
        return SFTTrainer(model=model, tokenizer=tokenizer, **kwargs)


def main():
    args = parse_args()

    if LOCAL_MODEL_PATH:
        print(f"本地权重 {LOCAL_MODEL_PATH}，已强制离线（HF_HUB_OFFLINE=1）")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    if args.tokenizer_chat_template:
        chat_template = getattr(tokenizer, "chat_template", None) or CLEAN_CHAT_TEMPLATE
        print("!! 使用 tokenizer 自带模板。Qwen3 instruct 模板会注入 <think> 空块，")
        print("   推理时那段会变成乱码挂在输出最前面，只在你明确知道后果时用。")
    else:
        chat_template = CLEAN_CHAT_TEMPLATE
        print("模板：使用项目内置的干净模板（不含 <think> 空块）")
    tokenizer.chat_template = chat_template

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = load_raw(args.data)
    dataset = dataset.map(
        lambda ex: to_text(ex, tokenizer, chat_template),
        remove_columns=dataset.column_names,
        desc="格式化数据",
    )
    eval_dataset = None
    if args.eval_ratio > 0:
        split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        dataset, eval_dataset = split["train"], split["test"]
        print(
            f"训练样本: {len(dataset)} | 验证样本: {len(eval_dataset)}"
            f"（seed={args.seed} 固定切分，换 seed 曲线就不可比）"
        )
    else:
        print(f"训练样本: {len(dataset)} | 无验证集（--eval-ratio 0）")
    print("抽查一条:\n" + dataset[0]["text"][:600])

    bf16_ok = torch.cuda.is_bf16_supported()
    common = dict(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        optim="adamw_8bit",
        seed=args.seed,
        report_to=args.report_to,
        logging_dir=str(Path(args.output) / "tb"),
        dataset_text_field="text",
    )

    if eval_dataset is not None:
        fields = _config_fields()
        eval_key = (
            "eval_strategy"
            if (not fields or "eval_strategy" in fields)
            else "evaluation_strategy"
        )
        common[eval_key] = "steps"
        common["eval_steps"] = args.eval_steps
        common["per_device_eval_batch_size"] = args.batch_size
        print(f"验证集评测: 每 {args.eval_steps} 步一次")

    metrics_path = (
        Path(args.metrics_file) if args.metrics_file else Path(args.output) / "metrics.jsonl"
    )
    metrics_logger = MetricsLogger(
        metrics_path,
        meta={
            "model": args.model,
            "data": args.data,
            "lora_r": args.lora_r,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.lr,
            "max_seq_len": args.max_seq_len,
            "load_in_4bit": args.load_in_4bit,
            "train_size": len(dataset),
            "eval_ratio": args.eval_ratio,
            "eval_size": len(eval_dataset) if eval_dataset is not None else 0,
            "eval_steps": args.eval_steps if eval_dataset is not None else 0,
            "bench_every": args.bench_every,
            "bench_subset": args.bench_subset if args.bench_every > 0 else "",
        },
    )
    print(f"指标写入: {metrics_path}")
    print("看板: bash scripts/start_dashboard.sh")

    callbacks = [metrics_logger]
    if args.probe_every > 0:
        probe_path = (
            Path(args.probe_file) if args.probe_file else Path(args.output) / "probes.jsonl"
        )
        callbacks.append(
            ProbeCallback(
                PROBE_PROMPTS, args.probe_every, probe_path, args.probe_max_new_tokens
            )
        )
        print(f"推理采样: 每 {args.probe_every} 步一次 → {probe_path}")

    if args.bench_every > 0:
        subset_path = Path(args.bench_subset)
        if not subset_path.exists():
            print(
                f"!! 找不到评测子集 {subset_path}，本次跳过基准评测。先生成：\n"
                "   python scripts/bench_subset.py"
            )
        else:
            bench_path = (
                Path(args.bench_file) if args.bench_file else Path(args.output) / "bench.jsonl"
            )
            callbacks.append(
                BenchCallback(
                    subset_path,
                    args.bench_every,
                    bench_path,
                    tokenizer,
                    args.bench_batch_size,
                )
            )
            print(
                f"基准评测: 每 {args.bench_every} 步一次，"
                f"{len(callbacks[-1].questions)} 题 → {bench_path}"
            )

    trainer = build_trainer(
        model,
        tokenizer,
        dataset,
        build_config(args, common),
        callbacks=callbacks,
        eval_dataset=eval_dataset,
    )
    stats = trainer.train()
    print(f"训练完成: {stats.metrics}")

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"已保存到 {args.output}")


if __name__ == "__main__":
    main()
