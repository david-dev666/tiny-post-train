"""DPO 偏好对齐：在 SFT 产物上再训一层 LoRA，把回答往「裁判更认可」的方向推。

最关键的一个设计：**ref 模型必须是 SFT 模型，不是基座**
----------------------------------------------------
DPO 的目标是让 policy 相对于 ref **提高**偏好对里 chosen 的似然。ref 定义的是
「起点」，所以它应该是「DPO 开始前的那份策略」= 我们训好的 SFT 模型。

如果偷懒让 TRL 用「基座 + 禁用 adapter」当 ref，等于把起点定在**微调之前** ——
DPO 会把 SFT 好不容易学到的指令遵循又往回拽（因为它会奖励「离基座更远」这个方向）。
这个错不会报错，只会让分数莫名下降，很难看出来。

做法（不需要落盘中间模型）：
    1. 加载基座 + SFT adapter，**在内存里合并** → 得到完整的 SFT 模型
    2. 在它上面挂一层新的 LoRA（DPO 层，随机初始化）
       policy = SFT模型 + DPO_LoRA
       ref    = 同一个 model 禁用 LoRA == SFT模型   ✓
    3. 训练完只存 DPO 层（约 200MB）

评测时怎么加载：`--model <SFT 合并后的目录> --adapter outputs/dpo-4b-v1`。
（合并目录用 `--export-merged` 一键生成，见下面的参数说明。）

训练数据
--------
`scripts/make_dpo_data.py` 造出来的 jsonl，字段 `prompt` / `chosen` / `rejected`。
**它不是下载的公开数据集**：prompt 取自训练池，回答由我们自己的 SFT 模型和
instruct-2507 分别给出，再让 instruct-2507 当裁判换位判两次定优劣 ——
所以哪边好哪边才是 chosen，不是无条件模仿 instruct。

用法
----
    # 冒烟（用 30 对的那份小数据，1~2 分钟）
    python scripts/train_dpo.py --data data/processed/dpo-zh/smoke.jsonl \
        --output outputs/dpo-4b-smoke --max-steps 10

    # 正式
    python scripts/train_dpo.py --data data/processed/dpo-zh/pairs.jsonl \
        --output outputs/dpo-4b-v1 --num-epochs 1

    # 生成评测用的合并基座（评测时 --model 用它）
    python scripts/train_dpo.py --export-merged outputs/sft-4b-v2-merged
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 本地权重强制离线，必须早于大件依赖 import
from offline import LOCAL_MODEL_PATH  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402

# unsloth 必须最先 import（它会 patch 一堆东西）
from unsloth import FastLanguageModel, PatchDPOTrainer  # noqa: E402

import train_sft as ts  # noqa: E402  复用 MetricsLogger / ProbeCallback / 模板

PROJECT_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="DPO 偏好对齐（在 SFT 产物上加一层 LoRA）")
    p.add_argument("--model", default="weights/Qwen3-4B-Base", help="基座权重")
    p.add_argument("--sft-adapter", default="outputs/sft-4b-v2",
                   help="要合并进基座的 SFT adapter。**它就是 DPO 的 ref**，别省")
    p.add_argument("--data", default=None, help="偏好数据 jsonl（prompt/chosen/rejected）")
    p.add_argument("--output", default=None, help="输出目录（只存 DPO 那层 LoRA）")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16,
                   help="DPO 层的秩。比 SFT 小 —— 只微调偏好方向，不需要重学能力")
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO 温度。越大越保守（离 ref 越近）；0.1 是常规起点")
    p.add_argument("--loss-type", default="sigmoid",
                   choices=["sigmoid", "hinge", "ipo", "kto_pair"],
                   help="sigmoid=原始 DPO；ipo 对偏好噪声更稳")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6,
                   help="比 SFT 小一个量级。DPO 是微调偏好，lr 大了容易把模型带崩")
    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--eval-ratio", type=float, default=0.02)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--probe-every", type=int, default=0,
                   help="每 N 步采样一次（看复读率有没有恶化），0 = 关")
    p.add_argument("--probe-max-new-tokens", type=int, default=128)
    p.add_argument("--export-merged", default=None,
                   help="不训练，只把 --model + --sft-adapter 合并到指定目录后退出。"
                        "评测时 --model 要指向这个目录")
    return p.parse_args()


def load_pairs(path: Path):
    """读偏好数据。顺手校验三个必需字段 —— TRL 在字段缺失时报的错很难懂。"""
    if not path.exists():
        sys.exit(f"!! 偏好数据不存在：{path}\n"
                 f"   先造：python scripts/make_dpo_data.py --n 3000 --out {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    if not rows:
        sys.exit(f"!! {path} 是空的")
    missing = [k for k in ("prompt", "chosen", "rejected") if k not in rows[0]]
    if missing:
        sys.exit(f"!! {path} 缺字段 {missing}（需要 prompt / chosen / rejected）")
    same = sum(1 for r in rows if r["chosen"] == r["rejected"])
    if same:
        # 两边一模一样的对，DPO 的梯度贡献是 0（logratio 差为 0），纯浪费算力
        print(f"!! 警告：{same}/{len(rows)} 对的 chosen 与 rejected 完全相同，应当剔除")
    return load_dataset("json", data_files=str(path), split="train")


def build_model(args):
    """基座 + SFT adapter 合到内存，再挂一层新 LoRA。返回 (model, tokenizer)。"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=False,
    )
    tokenizer.chat_template = ts.CLEAN_CHAT_TEMPLATE   # 和 SFT 一致，别换

    # 1) 把 SFT adapter 合并进权重 —— 合并后它就是「DPO 的起点」
    from peft import PeftModel

    sft_path = Path(args.sft_adapter)
    if not sft_path.exists():
        sys.exit(f"!! SFT adapter 不存在：{sft_path}")
    model = PeftModel.from_pretrained(model, str(sft_path))
    model = model.merge_and_unload()
    print(f"==> 已合并 SFT adapter（{sft_path}）—— 它就是 DPO 的 ref 模型")

    # 2) 在合并后的模型上挂新 LoRA。ref 就是「这个模型禁用 LoRA」，
    #    正好等于上面的 SFT 模型 —— 这是本脚本唯一要紧的设计点。
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"==> DPO 层：可训练 {n_train / 1e6:.1f}M / 总 {n_total / 1e6:.1f}M "
          f"({n_train / n_total * 100:.2f}%)")
    return model, tokenizer


def export_merged(args) -> int:
    """只做合并，供评测用（评测是 `--model <merged> --adapter <DPO 层>`）。"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_len,
        dtype=None, load_in_4bit=False,
    )
    tokenizer.chat_template = ts.CLEAN_CHAT_TEMPLATE
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(args.sft_adapter))
    out = Path(args.export_merged)
    out.mkdir(parents=True, exist_ok=True)
    model = model.merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    print(f"==> 已导出合并模型：{out}")
    print(f"    评测：python scripts/eval_pipeline.py --run dpo-4b-v1={out}+<DPO 层目录> ...")
    return 0


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_DIR)

    if args.export_merged:
        return export_merged(args)
    if not args.data or not args.output:
        sys.exit("!! 训练需要 --data 和 --output（只导出合并模型时加 --export-merged）")

    # PatchDPOTrainer 必须在 import DPOTrainer 之前调用，否则 unsloth 的优化不生效
    PatchDPOTrainer()
    from trl import DPOConfig, DPOTrainer

    dataset = load_pairs(Path(args.data))
    print(f"==> 偏好对 {len(dataset)} 条（{args.data}）")
    from_sft = sum(1 for r in dataset if r.get("chosen_from") == "sft")
    print(f"    chosen 来自 SFT 的 {from_sft} 条，来自 instruct 的 {len(dataset) - from_sft} 条")
    if from_sft == 0:
        # 全是 instruct 赢：这轮 DPO 实质是「学 instruct 的回答风格」。
        # 不是不能用，但要意识到它在做蒸馏 —— 训练后重点看风格类指标有没有被带偏。
        print("    !! chosen 全部来自 instruct —— 这轮是**蒸馏式**对齐，"
              "训练后重点看长度/格式类指标有没有被拉长带偏")

    eval_dataset = None
    if args.eval_ratio > 0 and len(dataset) > 50:
        split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        dataset, eval_dataset = split["train"], split["test"]
        print(f"    训练 {len(dataset)} / 验证 {len(eval_dataset)}")

    model, tokenizer = build_model(args)

    metrics_path = Path(args.output) / "metrics.jsonl"
    metrics_logger = ts.MetricsLogger(
        metrics_path,
        meta={
            "stage": "dpo",
            "model": args.model,
            "sft_adapter": args.sft_adapter,
            "data": args.data,
            "train_size": len(dataset),
            "lora_r": args.lora_r,
            "beta": args.beta,
            "loss_type": args.loss_type,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.lr,
            "max_seq_len": args.max_seq_len,
            "max_prompt_length": args.max_prompt_length,
        },
    )
    callbacks = [metrics_logger]
    if args.probe_every > 0:
        probe_path = Path(args.output) / "probes.jsonl"
        callbacks.append(ts.ProbeCallback(
            ts.PROBE_PROMPTS, args.probe_every, probe_path, args.probe_max_new_tokens))
        print(f"推理采样: 每 {args.probe_every} 步一次 → {probe_path}")

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
        save_total_limit=2,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        optim="adamw_8bit",
        seed=args.seed,
        report_to=args.report_to,
        logging_dir=str(Path(args.output) / "tb"),
        # DPO 特有的几个
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_seq_len,
        max_prompt_length=args.max_prompt_length,
        # ref 就是「当前模型禁用 LoRA」= 合并后的 SFT 模型，见文件头的说明
        precompute_ref_log_probs=False,
    )
    if eval_dataset is not None:
        fields = ts._config_fields()
        eval_key = "eval_strategy" if (not fields or "eval_strategy" in fields) else "evaluation_strategy"
        common[eval_key] = "steps"
        common["eval_steps"] = max(10, len(dataset) // 20)
        common["per_device_eval_batch_size"] = args.batch_size

    config = DPOConfig(**common)
    print(f"==> 指标写入: {metrics_path}")
    print(f"==> 看板: bash scripts/start_dashboard.sh")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # 见文件头：PEFT 下 ref = 本模型禁用 LoRA = SFT 模型
        args=config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"==> DPO 完成，LoRA 存在 {args.output}")
    print(f"    评测前先生成合并基座：python scripts/train_dpo.py "
          f"--export-merged outputs/sft-4b-v2-merged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
