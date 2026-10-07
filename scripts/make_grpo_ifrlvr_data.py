"""把 AI2 的 IF-RLVR 训练数据转成 GRPO 能吃的 jsonl。

数据来源
--------
`allenai/IF_multi_constraints_upto5`（HF / ModelScope 镜像），ODC-BY-1.0。
字段：`key / messages / ground_truth / dataset / constraint_type / constraint`。
约束从 **IFEval(25) + IFBench-Train(29)** 采样，每条指令最多 5 条约束。

一个必须知道的坑
----------------
`ground_truth` 是**Python 字面量字符串**（单引号、`None`），不是 JSON：
`"[{'instruction_id': [...], 'kwargs': [None, {...}]}]"`。
用 `json.loads` 会当场炸，必须 `ast.literal_eval`。

输出字段（对齐 `ifrlvr_reward.py` 的期望）
------------------------------------------
    {"key", "prompt", "constraint_ids", "constraint_kwargs", "dataset"}

**与评测集零重叠**：本脚本会拿 `evals/ifollow-subset.jsonl` 一起做归一化指纹比对，
重叠不为 0 就直接报错退出 —— 训练集污染主指标是项目踩过的那类「假信号」。

用法：
    python scripts/make_grpo_ifrlvr_data.py                       # 全量
    python scripts/make_grpo_ifrlvr_data.py --limit 500 --output .../smoke.jsonl
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from open_instruct.IFEvalG import instructions_registry  # noqa: E402

DEFAULT_PARQUET = PROJECT_DIR / "data/raw/ifollow-open/if-multi/data/train-00000-of-00001.parquet"
DEFAULT_OUTPUT = PROJECT_DIR / "data/processed/grpo-ifrlvr/train.jsonl"
DEFAULT_EVAL = PROJECT_DIR / "evals/ifollow-subset.jsonl"

_WSP = re.compile(r"\s+")


def fingerprint(text: str) -> str:
    """归一化指纹：去空白、转小写后取 sha1 前 16 位。

    只用于「训练集与评测集是否撞题」的比对，不参与训练。
    """
    return hashlib.sha1(_WSP.sub("", (text or "").lower()).encode("utf-8")).hexdigest()[:16]


def parse_ground_truth(raw: str) -> tuple[list[str], list[dict | None]]:
    """`ground_truth` 字符串 → (instruction_id 列表, kwargs 列表)。"""
    parsed = ast.literal_eval(raw)
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    block = parsed[0]
    if isinstance(block, str):
        block = json.loads(block)
    return list(block["instruction_id"]), list(block["kwargs"])


def main() -> int:
    ap = argparse.ArgumentParser(description="IF-RLVR 训练数据转换")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL, help="用于零重叠校验")
    ap.add_argument("--limit", type=int, default=0, help="只转前 n 条（冒烟用，0 = 全量）")
    ap.add_argument("--tokenizer", default="weights/Qwen3-4B-Base",
                    help="用来量 prompt 长度（超长 prompt 会在 vllm 侧报 context length 错）")
    ap.add_argument("--max-prompt-tokens", type=int, default=1536,
                    help="超过这个 token 数的 prompt 直接丢。实测 p90 只有 418，"
                         "但 max 有 55928（坏数据），不丢会让 vllm 报 context length 错")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    if not args.parquet.exists():
        print(f"找不到 parquet：{args.parquet}", file=sys.stderr)
        print("先跑：modelscope download --dataset allenai/IF_multi_constraints_upto5 "
              f"--local_dir {args.parquet.parent.parent}", file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    ds = load_dataset("parquet", data_files=str(args.parquet), split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    eval_fps: dict[str, str] = {}
    if args.eval_set.exists():
        with args.eval_set.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    eval_fps[fingerprint(item["prompt"])] = item.get("id", "?")

    rows: list[dict] = []
    unknown = collections.Counter()
    kinds = collections.Counter()
    dataset_src = collections.Counter()
    per_prompt = collections.Counter()
    overlap: list[str] = []
    parse_fail = 0
    too_long = 0
    prompt_lens: list[int] = []

    for record in ds:
        try:
            ids, kws = parse_ground_truth(record["ground_truth"])
        except Exception as exc:  # 单条坏数据不该毁掉整批
            parse_fail += 1
            if parse_fail <= 3:
                print(f"  解析失败：{exc} | {record['ground_truth'][:120]!r}", file=sys.stderr)
            continue

        prompt = "\n".join(m["content"] for m in record["messages"] if m["role"] == "user")

        # 超长 prompt 直接丢：vllm 的 max_model_len 是硬上限，撞上就报错中断整批。
        # 实测 p90 只有 418 token，丢掉 1%/1.5% 的量对训练无感。
        n_tok = len(tokenizer(prompt)["input_ids"])
        prompt_lens.append(n_tok)
        if n_tok > args.max_prompt_tokens:
            too_long += 1
            continue

        fp = fingerprint(prompt)
        if fp in eval_fps:
            overlap.append(f"{record['key']} ↔ 评测集 {eval_fps[fp]}")

        for i in ids:
            kinds[i] += 1
            if i not in instructions_registry.INSTRUCTION_DICT:
                unknown[i] += 1
        dataset_src[record.get("dataset", "?")] += 1
        per_prompt[len(ids)] += 1

        rows.append({
            "key": record["key"],
            "prompt": prompt,
            "constraint_ids": ids,
            "constraint_kwargs": [k or {} for k in kws],
            "dataset": record.get("dataset", "?"),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"写入 {len(rows)} 条 → {args.output}")
    print(f"解析失败 {parse_fail} 条")
    if prompt_lens:
        s = sorted(prompt_lens)
        n = len(s)
        print(f"prompt token 长度：p50 {s[n // 2]} / p90 {s[int(n * 0.9)]} / "
              f"p99 {s[int(n * 0.99)]} / max {s[-1]}")
    print(f"超长丢弃 {too_long} 条（> {args.max_prompt_tokens} token）")
    print(f"数据来源分布：{dict(dataset_src)}")
    print(f"每条指令的约束个数分布：{dict(sorted(per_prompt.items()))}")
    print(f"约束类型 {len(kinds)} 种（训练数据实际用到的）")
    if unknown:
        print(f"⚠️ 判分器不认识的约束 {len(unknown)} 种（这些会被记 0 分）：{dict(unknown)}")
    else:
        print("✅ 全部约束都能被 vendor 的 registry 判分")
    if overlap:
        print(f"❌ 与评测集重叠 {len(overlap)} 条，拒绝继续：")
        for o in overlap[:10]:
            print("   ", o)
        return 1
    print("✅ 与 evals/ifollow-subset.jsonl 零重叠")
    return 0


if __name__ == "__main__":
    sys.exit(main())
