"""造 DPO 偏好数据。两种模式，默认 `stop-token`。

模式一：`--mode stop-token`（默认，推荐）
----------------------------------------
靶心是**收尾那一个 token**：模型答完内容之后，不肯干净地结束回合，而是吐一个
乱码 token（希伯来语词 `לחלוט`、字节残缺的 `\\ufffd\\ufffd取` …），然后才停。

    prompt   : 请只回答一个数字，不要任何其他内容
    chosen   : 1024          ← 剪掉末尾乱码
    rejected : 1024. לחלוט   ← 模型原样输出

所以 **chosen 和 rejected 逐字相同、只差最后那一个 token**：

- 优劣是**客观**的，不需要裁判 → 没有标注噪声，也没有「裁判和对手是同一个
  模型」的固有偏差
- 只跑一个模型一次生成 → 成本是模式二的一半以下
- 不会退化成蒸馏（模式二的固有风险，见下面那段说明）
- 靶心已经量化过：指令遵循 61.5% → 76.5%，数值约束 0/25 里 18 条是乱码打掉的

实测「自然收尾」的乱码率：指令遵循 93.4%、GSM8K 93.0%、OpenQA 92.5%、
HumanEval 86.1% —— 只要模型自己决定收尾，就几乎必吐乱码。所以 prompt 不必挑，
从训练池随机抽即可；命中率取决于生成得够不够长（**被截断的题一条都没有**）。

模式二：`--mode judge`（原方案）
-------------------------------
同一个问题，让两个模型各答一份，用裁判判哪份更好。

为什么自己造，不下载公开偏好数据集
--------------------------------
1. **分布一致**：prompt 直接取自 SFT 的同一个池子（alpaca-gpt4-zh），
   不是「训练用 A 分布、对齐用 B 分布」——那种错配会让 DPO 学到无关的东西
2. **信号对准已知短板**：评测已经把差距量化了（指令遵循 61.5% vs 97.5%、
   长度约束 1/25 vs 23/25），裁判提示词里也明确写了「更长不等于更好」，
   正好压住我们最明显的毛病（啰嗦）
3. **不需要下载**：服务器数据盘只剩 14G，够用但要省

设计上最关键的一点：**不是无条件模仿 instruct**
--------------------------------------------
常见做法是「强模型答 = chosen，弱模型答 = rejected」，那等于在做蒸馏 ——
DPO 会把模型往对方的**说话风格**上拽，而不是往「更好的回答」上拽。

这里每对都**必须过裁判**，哪边好哪边才是 chosen：
instruct 更好就它是 chosen，我们自己的 SFT 更好就反过来。
所以落盘之后**先看 `裁判判定` 那行统计**：如果 SFT 一次都没赢，
说明裁判有偏（或差距确实大到没悬念），这种数据拿去 DPO 会退化成模仿，别用。

换位判两次，只有两次一致才收
--------------------------
裁判有位置偏置（倾向选前面那个）。所以每对判两次：`(SFT, instruct)` 与
`(instruct, SFT)`，只有结论一致才留下；不一致的记成「位置敏感」丢弃。
判定翻译用的是 `prompts.verdict` —— 和 `judge_openqa.py` 同一份实现，
换位翻译写错一次就会把一部分偏好对**标反**，而且完全看不出来。

两阶段，每个模型只加载一次
------------------------
显存只够放一个 4B（另加 KV），所以：
    阶段 1  加载 base+sft adapter → 生成 SFT 的回答 → 释放
    阶段 2  加载 instruct-2507    → 生成它的回答、再判所有对 → 落盘

红线：**prompt 只能来自训练池**
-----------------------------
`evals/*.jsonl` 是**测量工具**，一条都不能进训练数据 —— 那是最难发现、
后果也最严重的一种污染：分数会变好看，但那是背答案背出来的。
所以本脚本只从 `data/` 下读 prompt，并且显式拒绝 `evals/` 下的任何路径。

用法
----
    # 冒烟（30 条，约 1 分钟）
    python scripts/make_dpo_data.py --n 30 --out data/processed/dpo-zh/stop-smoke.jsonl

    # 正式（4000 条 prompt，命中约 3000 对，约 15 分钟）
    python scripts/make_dpo_data.py --n 4000 --out data/processed/dpo-zh/stop.jsonl

    # 模式二（裁判选优，成本高一倍）
    python scripts/make_dpo_data.py --mode judge --n 3000 \
        --out data/processed/dpo-zh/judge.jsonl

**必须在 vllm 环境里跑**（要 /root/autodl-tmp/envs/vllm/bin/python）。

`stop-token` 模式还会顺手落两份：
- `<out>.raw.jsonl` —— 原始生成记录（含 token id），改构造规则时可离线重造
- `<out>.junkids.json` —— 末尾乱码 token 的 id、出现次数与单 token 解码样例，
  用来核对这个缺陷的具体形态（实测只有 6 种，主力是希伯来语词 `לחלוט`）
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts as P  # noqa: E402  纯文本模块，两个环境都能用

# 乱码尾的唯一判据（纯文本模块，不需要 torch）
from answer_extract import is_junk_piece, trailing_junk_len  # noqa: E402

# 停止符裁剪的唯一实现。**不能省** —— vLLM 的 token_ids 带着停止符，
# 留着它 trailing_junk_len 会把停止符当成「最后那个 token」，
# 于是永远判不出末尾的乱码，偏好对一条也造不出来（而且不报错）。
from engines import strip_stop_tokens  # noqa: E402


def prepare_env() -> None:
    """绕掉和 `engines.VLLMEngine` 同样的两个环境坑。

    1) `envs/vllm/bin/python xxx.py` 调用时那个 bin 不在 PATH，
       flashinfer 要现场编译会报「找不到 ninja」（而 ninja 就装在同一个环境里）
    2) 本机 nvcc 是 CUDA 12.4，编不了 flashinfer 0.6.18 要的 `--compress-mode`
    """
    bindir = str(Path(sys.executable).resolve().parent)
    if bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


# ------------------------------------------------------------------ 读 prompt


def load_prompts(path: Path, n: int, seed: int = 0) -> list[str]:
    """从训练池里抽 prompt。`instruction` + `input` 拼起来，和 SFT 时的卷面一致。"""
    resolved = path.resolve()
    if "evals" in resolved.parts:
        sys.exit(f"!! 拒绝：{resolved} 在 evals/ 下。评测集不能进训练数据。")
    if "data" not in resolved.parts:
        sys.exit(f"!! 拒绝：{resolved} 不在 data/ 下。prompt 只能取自训练池。")

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    prompts = []
    for row in rows:
        user = (row.get("instruction") or "").strip()
        extra = (row.get("input") or "").strip()
        if extra:
            user = f"{user}\n{extra}"
        if user:
            prompts.append(user)
    random.Random(seed).shuffle(prompts)
    return prompts[:n]


# ------------------------------------------------------------------ vLLM 包装


class Engine:
    """一个模型只加载一次 —— 加载要 30 秒，分批重载是纯浪费。"""

    def __init__(self, model_ref: str, adapter: str | None = None, gpu_util: float = 0.85):
        prepare_env()
        from vllm import LLM

        self.llm = LLM(
            model=model_ref, max_model_len=2048, dtype="bfloat16",
            gpu_memory_utilization=gpu_util, enforce_eager=True,
            enable_prefix_caching=False, enable_lora=bool(adapter),
            max_lora_rank=64, disable_log_stats=True,
        )
        self.lora = None
        if adapter:
            from vllm.lora.request import LoRARequest

            self.lora = LoRARequest("adapter", 1, adapter)

    def gen_detailed(self, prompts: list[str], max_tokens: int, stop_ids: list[int],
                     temperature: float = 0.0, label: str = "", chunk: int = 1000) -> list[dict]:
        """分批提交，把 **token id** 和「有没有用满预算」也带回来。

        **提交粒度只影响进度可见性，不影响速度** —— vLLM 内部有自己的调度。
        为什么要分批：3000 条一次性丢进去要跑十几分钟，中途**一个字符都不输出**。
        这个项目已经因为「长循环看不见进度」吃过亏（MMLU/GSM8K 一开始也是这样，
        任务崩了才发现），所以宁可多几行日志。

        为什么造停止决策的数据非要 token id：乱码 token 解出来的**文本**可能以
        正常汉字收尾（字节级 BPE 的残缺片段解成 `\\ufffd\\ufffd取`），光看文本
        既定位不到它、也剪不干净。详见 `answer_extract.trailing_junk_len`。
        """
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature, top_p=1.0, max_tokens=max_tokens,
            stop_token_ids=stop_ids, skip_special_tokens=True,
        )
        records: list[dict] = []
        for start in range(0, len(prompts), chunk):
            part = prompts[start:start + chunk]
            outputs = self.llm.generate(part, params, lora_request=self.lora, use_tqdm=False)
            for one in outputs:
                completion = one.outputs[0]
                records.append({
                    "text": completion.text.strip(),
                    # vLLM 的 token_ids **含**停止符，必须去掉（和 HF 引擎同口径）——
                    # 见 engines.strip_stop_tokens 的说明：差这一个 token，
                    # 乱码尾判定会静默归零，而且看不出来
                    "ids": strip_stop_tokens(
                        [int(t) for t in completion.token_ids], stop_ids),
                    # 「用满预算」直接看 finish_reason，别拿 len(ids) 反推 ——
                    # 那要依赖 ids 里有没有停止符，正是上面刚踩过的坑
                    "truncated": completion.finish_reason == "length",
                })
            if label:
                print(f"   {label} {len(records)}/{len(prompts)}", flush=True)
        return records

    def gen(self, prompts: list[str], max_tokens: int, stop_ids: list[int],
            temperature: float = 0.0, label: str = "", chunk: int = 1000) -> list[str]:
        """只要文本。内部复用 gen_detailed —— 两条路径各自演化必然漂移。"""
        return [r["text"] for r in self.gen_detailed(
            prompts, max_tokens, stop_ids, temperature, label, chunk)]

    def close(self) -> None:
        del self.llm
        self.lora = None


def _render(tokenizer, texts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                      tokenize=False, add_generation_prompt=True)
        for text in texts
    ]


def _tokenizer(model_ref: str, template: str | None = None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if template is not None:
        tokenizer.chat_template = template
    return tokenizer


# ------------------------------------------------------------------ 主流程


def build_pairs(prompts, ours, theirs, judges, judge_tokenizer):
    """两次判定一致才算数。返回 (保留的对, 全部原始记录, 丢弃原因计数)。

    `judges` 是 2N 条判定结果：前 N 条是 (SFT, instruct) 顺序，
    后 N 条是 (instruct, SFT) 顺序。
    """
    reason = {"平局": 0, "位置敏感（换位就翻）": 0, "判定解析失败": 0}
    pairs, raw = [], []
    half = len(prompts)
    for index, prompt in enumerate(prompts):
        first = P.verdict(judges[index], swapped=False)              # 回答一 = SFT
        second = P.verdict(judges[half + index], swapped=True)       # 回答一 = instruct
        record = {
            "prompt": prompt,
            "ours": ours[index],
            "theirs": theirs[index],
            "verdict_first": first,
            "verdict_second": second,
            "judge": judge_tokenizer.name_or_path,
        }
        if first != second:
            reason["位置敏感（换位就翻）"] += 1
            record["kept"] = False
        elif first == "tie":
            reason["平局"] += 1
            record["kept"] = False
        elif first == "unparsed":
            reason["判定解析失败"] += 1
            record["kept"] = False
        else:
            chosen_is_ours = first == "a"
            record.update({
                "kept": True,
                "chosen": ours[index] if chosen_is_ours else theirs[index],
                "rejected": theirs[index] if chosen_is_ours else ours[index],
                "chosen_from": "sft" if chosen_is_ours else "instruct",
            })
            pairs.append({
                "prompt": prompt,
                "chosen": record["chosen"],
                "rejected": record["rejected"],
                # 来源记清楚：事后要能查出「这对是谁跟谁比出来的」
                "chosen_from": record["chosen_from"],
                "source": "alpaca-gpt4-zh + Qwen3-4B-Instruct-2507 裁判",
            })
        raw.append(record)
    return pairs, raw, reason


# ------------------------------------------------------------------ 模式二：停止决策

# chosen 末尾要**显式**写上的停止符。
#
# 🔴 必须显式写，绝不能指望 tokenizer 的 `eos_token_id` —— 那俩不是同一个 token：
#
#     tokenizer.eos_token_id = 151643 <|endoftext|>   收尾位置排第 **8958** 名（≈0）
#     训练数据真正教的是     151645 <|im_end|>         排第 **4**，只比第一名低 5%
#
# 第一轮 DPO 就是栽在这里：数据里 chosen 是纯文本，追加 EOS 的活儿交给了 TRL，
# 而它用的是 eos_token_id = 151643 —— 一个模型在那个位置根本吐不出来的 token。
# 于是 `rewards/chosen` 死活拉不动（+0.05）、`rewards/rejected` 一崩到底（-4.45），
# 训练曲线看着很健康，模型却毫无变化。
#
# 靶心对准之后的实测上界（把 <|im_end|> 的 logit 抬 +2.0）：
#     指令遵循 123/200（61.5%）→ **162/200（81.0%）**
STOP_TOKEN_TEXT = "<|im_end|>"


def build_stop_pairs(prompts, records, tokenizer):
    """把「答完之后吐乱码」翻成偏好对。返回 (保留的对, 统计, 乱码 token 计数)。

        chosen   = 剪掉末尾乱码的那一份
        rejected = 模型原样输出

    **为什么这条数据比「让 instruct 答、裁判选优」（--mode judge）更该先做：**

    1. 优劣是**客观**的 —— 末尾多吐了一个乱码 token，不需要裁判。既没有标注
       噪声，也避开了 `--mode judge` 那种「裁判和对手是同一个模型」的固有偏差
    2. 省掉整个阶段 2：不用生成对照答案，也不用跑 2N 次换位判定
    3. 不会退化成蒸馏 —— `--mode judge` 一旦 SFT 全输，DPO 就变成「模仿
       instruct 的说话风格」，本文件在它那条路上专门写了警告
    4. **靶心已经量化过**：把乱码剪掉再判，指令遵循 61.5% → 76.5%，而数值约束
       0/25 里有 18 条正是被乱码打掉的。训完哪一项会动、大概动多少，事先知道

    实测「自然收尾」的乱码率：指令遵循 93.4%、GSM8K 93.0%、OpenQA 92.5%、
    HumanEval 86.1% —— **只要模型自己决定收尾，就几乎必吐乱码**。
    唯一的例外是 MMLU（答案是单个字母，没有「要不要收尾」这一步）0.4%。
    所以 prompt 不必挑，从训练池随机抽即可；命中率取决于生成得够不够长。
    """
    stats = {"生成": len(records), "截断（丢弃）": 0, "干净收尾（丢弃）": 0,
             "剪完是空的（丢弃）": 0, "保留": 0}
    junk_ids: collections.Counter = collections.Counter()
    pairs = []
    for prompt, record in zip(prompts, records):
        if record["truncated"]:
            # 被预算切掉的题**一定**没有乱码尾（实测 0/19、0/10、0/56）：
            # 压根没走到「收尾决策」那一步，留着只会教模型「别写完」
            stats["截断（丢弃）"] += 1
            continue
        cut = trailing_junk_len(tokenizer, record["ids"])
        if cut == 0:
            stats["干净收尾（丢弃）"] += 1
            continue
        body = tokenizer.decode(record["ids"][:-cut], skip_special_tokens=True).strip()
        if not body:
            # 整条输出都是乱码，剪完什么都不剩 —— 这种对教不了任何东西
            stats["剪完是空的（丢弃）"] += 1
            continue
        # chosen 末尾**显式**补上 `<|im_end|>`。这一行就是整件事的靶心，
        # 见 STOP_TOKEN_TEXT 上面那段说明 —— 不能指望 tokenizer 自己追加。
        chosen = body + STOP_TOKEN_TEXT
        for token_id in record["ids"][-cut:]:
            # **只统计真正的乱码 token**：`cut` 里还含尾随的空白（剪的时候
            # 顺手带走），空白混进去会让「这个缺陷有几种形态」这份统计失真。
            if is_junk_piece(tokenizer.decode([int(token_id)], skip_special_tokens=False)):
                junk_ids[int(token_id)] += 1
        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            # rejected 是模型原样输出（末尾那个乱码 token 还在）
            "rejected": record["text"],
            # chosen **确实来自 SFT**（只是把尾巴上的乱码剪了），不是模仿别人答的。
            # train_dpo.py 靠这个字段判断「这轮是不是在蒸馏」，别写成别的值 ——
            # 写成空/别的会让它报「chosen 全部来自 instruct」那条误导性警告。
            "chosen_from": "sft",
            "source": "SFT 自采样 + 末尾乱码换成 <|im_end|>",
            # 一共剪了几个 token（**含**尾随空白，所以可能大于乱码本身的个数）
            "trimmed_tokens": cut,
        })
        stats["保留"] += 1
    return pairs, stats, junk_ids


def run_stop_token(args) -> int:
    """`--mode stop-token`：造「停止决策」偏好对。"""
    prompts = load_prompts(Path(args.source), args.n, args.seed)
    print(f"==> prompt {len(prompts)} 条（来源 {args.source}）", flush=True)
    began = time.time()

    # 训练时用的就是干净模板，这里必须一致 —— 换模板等于换卷面
    tokenizer = _tokenizer(args.model, P.CLEAN_CHAT_TEMPLATE)
    engine = Engine(args.model, adapter=args.adapter, gpu_util=args.gpu_util)
    print(f"\n== 生成：{args.model} + {args.adapter or '（无 adapter）'} ==", flush=True)
    records = engine.gen_detailed(
        _render(tokenizer, prompts), args.max_new_tokens,
        P.stop_token_ids(tokenizer), label="作答",
    )
    engine.close()

    pairs, stats, junk_ids = build_stop_pairs(prompts, records, tokenizer)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in pairs:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 原始记录（含 token id）也落盘 —— 它才是「造偏好对」的真正输入。
    # 改了 build_stop_pairs 的规则之后，可以直接拿它离线重造，不必再跑 6 分钟生成。
    raw_path = out_path.with_suffix(".raw.jsonl")
    with raw_path.open("w", encoding="utf-8") as handle:
        for prompt, record in zip(prompts, records):
            handle.write(json.dumps({"prompt": prompt, **record}, ensure_ascii=False) + "\n")

    # 乱码 token 的 id 落盘：用来核对这个缺陷的**具体形态**。
    # 实测只有 6 种，主力是第一行那个希伯来语词 —— 而词表里独立解码含 U+FFFD
    # 的 token 有 1457 个，所以「禁掉它们」这条路走不通，必须训练侧修。
    ids_path = out_path.with_suffix(".junkids.json")
    ids_path.write_text(json.dumps({
        "note": "末尾乱码 token 的 id、出现次数与单 token 解码样例（缺陷形态记录）",
        "tokenizer": args.model,
        "counts": dict(junk_ids.most_common()),
        "sample_decode": {str(i): tokenizer.decode([i], skip_special_tokens=False)
                          for i in junk_ids},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n== 结果 ==")
    for name, count in stats.items():
        print(f"  {name:<16} {count}")
    if stats["生成"]:
        print(f"  命中率 {stats['保留'] / stats['生成']:.1%}")
    print(f"  偏好对 → {out_path}")
    print(f"  乱码 token id → {ids_path}（共 {len(junk_ids)} 种）")
    for token_id, count in junk_ids.most_common(5):
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"      id={token_id:<8} {count:>5} 次   {piece!r}")
    if stats["保留"] < 200:
        print("  !! 保留的偏好对太少（<200）。命中率取决于**生成得够不够长** ——"
              "被截断的题一条乱码都没有。调大 --max-new-tokens 或换更长的 prompt。")
    print(f"  总用时 {(time.time() - began) / 60:.1f} 分钟")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="造 DPO 偏好数据。两种模式：stop-token（停止决策，推荐）/ judge（裁判选优）")
    parser.add_argument("--mode", default="stop-token", choices=["stop-token", "judge"],
                        help="stop-token：靶心是「答完之后吐乱码」的那一个 token —— "
                             "chosen 与 rejected 只差末尾一个 token，不需要裁判，"
                             "成本只有 judge 的一半，也不会退化成蒸馏；"
                             "judge：同题两答 + 裁判换位选优（原方案）")
    parser.add_argument("--source", default="data/raw/alpaca-gpt4-zh/train.csv",
                        help="prompt 来源。**必须在 data/ 下**，evals/ 会被直接拒绝")
    parser.add_argument("--model", default="weights/Qwen3-4B-Base")
    parser.add_argument("--adapter", default="outputs/sft-4b-v2")
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--judge", default="weights/Qwen3-4B-Instruct-2507")
    parser.add_argument("--n", type=int, default=30, help="造多少条偏好对")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-max-new-tokens", type=int, default=8)
    parser.add_argument("--batch", type=int, default=400, help="一次提交多少条给 vLLM")
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-out", default=None,
                        help="原始记录（含丢弃的）落盘路径，默认 <out>.raw.jsonl")
    args = parser.parse_args()

    if args.mode == "stop-token":
        return run_stop_token(args)

    prompts = load_prompts(Path(args.source), args.n, args.seed)
    print(f"==> prompt {len(prompts)} 条（来源 {args.source}）", flush=True)
    began = time.time()

    # ---- 阶段 1：SFT 模型的回答
    print(f"\n== 阶段 1/2：{args.model} + {args.adapter} 作答 ==", flush=True)
    ours_tokenizer = _tokenizer(args.model, P.CLEAN_CHAT_TEMPLATE)  # 训练时用的就是它
    ours_engine = Engine(args.model, adapter=args.adapter)
    ours = ours_engine.gen(_render(ours_tokenizer, prompts), args.max_new_tokens,
                           P.stop_token_ids(ours_tokenizer), label="SFT 作答")
    ours_engine.close()
    print(f"   生成 {len(ours)} 条，累计 {(time.time() - began) / 60:.1f} 分钟", flush=True)

    # ---- 阶段 2：裁判模型的回答，再由它自己判（同一个模型，只加载一次）
    print(f"\n== 阶段 2/2：{args.judge} 作答并当裁判 ==", flush=True)
    judge_tokenizer = _tokenizer(args.judge)
    if not getattr(judge_tokenizer, "chat_template", None):
        sys.exit("!! 裁判模型没有自带对话模板，无法评判")
    judge_engine = Engine(args.judge)
    theirs = judge_engine.gen(_render(judge_tokenizer, prompts), args.max_new_tokens,
                              P.stop_token_ids(judge_tokenizer), label="裁判作答")
    print(f"   生成 {len(theirs)} 条回答，开始判定（每对换位判两次）", flush=True)

    stop_ids = P.stop_token_ids(judge_tokenizer)
    # 两轮判定分开跑、分开攒：不换位的一轮 + 换位的一轮。
    # （早先写成把两半交错塞进一个 list，再用切片往下半段拼 —— 又绕又错，别再那样写。）
    first_verdicts: list[str] = []
    second_verdicts: list[str] = []
    for start in range(0, len(prompts), args.batch):
        chunk = list(range(start, min(start + args.batch, len(prompts))))
        for swapped in (False, True):
            batch_prompts = []
            for index in chunk:
                one, two = (ours[index], theirs[index])
                if swapped:
                    one, two = two, one
                batch_prompts.append(P.JUDGE_PROMPT.format(
                    question=prompts[index], answer_one=one, answer_two=two))
            replies = judge_engine.gen(_render(judge_tokenizer, batch_prompts),
                                       args.judge_max_new_tokens, stop_ids)
            target = second_verdicts if swapped else first_verdicts
            target.extend(P.parse_verdict(reply) for reply in replies)
        print(f"   判定 {min(start + args.batch, len(prompts))}/{len(prompts)}", flush=True)
    judge_engine.close()
    judges = first_verdicts + second_verdicts

    pairs, raw, reason = build_pairs(prompts, ours, theirs, judges, judge_tokenizer)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in pairs:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    raw_path = Path(args.raw_out) if args.raw_out else out_path.with_suffix(".raw.jsonl")
    with raw_path.open("w", encoding="utf-8") as handle:
        for record in raw:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n== 结果 ==")
    print(f"  保留 {len(pairs)}/{len(prompts)} 对 → {out_path}")
    print(f"  原始 {len(raw)} 条（含丢弃的）→ {raw_path}")
    for name, count in reason.items():
        print(f"  丢弃 {name:<18} {count}")
    if pairs:
        ours_win = sum(1 for r in pairs if r["chosen_from"] == "sft")
        print(f"\n  裁判判定：instruct 胜 {len(pairs) - ours_win}，SFT 胜 {ours_win}")
        if ours_win == 0:
            print("  !! SFT 一次都没赢。要么两者差距确实大，要么裁判有偏 —— "
                  "这种数据拿去 DPO 会退化成「模仿 instruct」，先人工看几条再决定。")
    print(f"  总用时 {(time.time() - began) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
