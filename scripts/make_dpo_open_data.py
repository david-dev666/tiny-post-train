"""把**开源、被广泛复用**的偏好数据集转成 `train_dpo.py` 要的 prompt/chosen/rejected。

为什么换数据（这一节是这次改动的全部理由）
------------------------------------------
项目此前的 DPO 数据（`stop.jsonl` / `last-token.jsonl`）**不是通用偏好数据**，
它是为「修收尾那一个 token」专门构造的定向数据 —— prompt 是「已到内容末尾」的前缀，
chosen/rejected 只差最后 1 个 token。那个目的已经 5 轮证否（见 notes/workflow.md）。
所以这一轮换的是**目的**（通用偏好对齐），不是"修数据"。

数据来源
--------
`opencsg/ultrafeedback-chinese`（ModelScope，54k，中文）
  - `ultrafeedback_zh_binarized_lowest.parquet`  chosen=四份里 overall_score 最高，rejected=最低
  - `ultrafeedback_zh_binarized_random.parquet` 随机配对
字段：instruction / source / chosen_response / rejected_response / chosen_rating / rejected_rating

**默认用 `lowest`**：chosen 与 rejected 的评分差最大，偏好信号最干净（`--min-gap` 再卡一道）。

一个必须做对的细节：prompt 要自己渲染成完整对话前缀
--------------------------------------------------
`train_dpo.py` 走的是「prompt 已经是完整前缀」这条路（见 `make_dpo_last_data.py` 的注释），
**它不会再套 chat 模板**。直接塞裸 instruction 会训出「user 回合缺失」或「模板套两遍」
的畸形样本 —— `make_rft_data.py` 踩过同一个坑（1632 条全中）。

    prompt = render_clean(instruction)
           = "<|im_start|>user\\n{instruction}<|im_end|>\\n<|im_start|>assistant\\n"

chosen / rejected 只放**回复正文**，结尾的 `<|im_end|>` 由 TRL 追加
（`train_dpo.py` 已把 `tokenizer.eos_token_id` 指到 `<|im_end|>`，那里有完整说明）。

用法（纯 CPU，tpt 环境即可）
--------------------------
    /root/miniconda3/envs/tpt/bin/python scripts/make_dpo_open_data.py \\
        --parquet data/raw/dpo-open/ultrafeedback-chinese/ultrafeedback_zh_binarized_lowest.parquet \\
        --output data/processed/dpo-open/pairs.jsonl \\
        --tokenizer weights/Qwen3-4B-Base \\
        --limit 20000 --min-gap 0.5

先看 `--stats-only` 的 token 分布再决定 `--limit`：超长样本会被 `--max-seq-len` 静默截断，
截断掉的偏好对等于噪声。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prompts import render_clean  # noqa: E402

# `chosen_rating` 是**字符串化的 dict**（`"{'helpfulness': '4', ..., 'overall_score': 4.6}"`），
# 不是 JSON，所以用正则取数，不用 eval / json.loads。
_OVERALL_RE = re.compile(r"['\"]overall_score['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_args():
    p = argparse.ArgumentParser(description="开源偏好数据 → train_dpo.py 格式")
    p.add_argument("--parquet", required=True)
    p.add_argument("--output", default="data/processed/dpo-open/pairs.jsonl")
    p.add_argument("--tokenizer", default="weights/Qwen3-4B-Base",
                   help="只用于统计 token 分布；传空字符串则跳过 token 统计")
    p.add_argument("--limit", type=int, default=0, help="最多取多少条，0=全部")
    p.add_argument("--min-gap", type=float, default=0.0,
                   help="chosen/rejected 的 overall_score 最小差距（默认 0，不过滤）")
    p.add_argument("--max-chars", type=int, default=4000, help="回复字符数上限")
    p.add_argument("--max-len-ratio", type=float, default=0.0,
                   help="chosen/rejected 字符长度比上限，防学到「更长=更好」。0=不过滤；"
                        "先跑 --stats-only 看存活率表再定")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="和训练时的 --max-seq-len 保持一致，用来统计会被截断的比例")
    p.add_argument("--max-prompt-len", type=int, default=512,
                   help="和训练时的 --max-prompt-length 保持一致")
    p.add_argument("--stats-only", action="store_true")
    return p.parse_args()


def overall_score(rating) -> float | None:
    if not rating:
        return None
    m = _OVERALL_RE.search(str(rating))
    return float(m.group(1)) if m else None


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


def main():
    args = parse_args()

    import pyarrow.parquet as pq

    rows = pq.read_table(args.parquet).to_pylist()
    print(f"读入 {len(rows)} 条（{Path(args.parquet).name}）")

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    kept, seen = [], set()
    stat = {"缺字段": 0, "评分差不足": 0, "太长": 0, "长度比过大": 0, "正负相同": 0, "重复题面": 0}
    gaps, ratios = [], []

    for r in rows:
        instruction = (r.get("instruction") or "").strip()
        chosen = (r.get("chosen_response") or "").strip()
        rejected = (r.get("rejected_response") or "").strip()
        if not instruction or not chosen or not rejected:
            stat["缺字段"] += 1
            continue
        if chosen == rejected:
            stat["正负相同"] += 1
            continue

        c, j = overall_score(r.get("chosen_rating")), overall_score(r.get("rejected_rating"))
        if args.min_gap > 0:
            if c is None or j is None or (c - j) < args.min_gap:
                stat["评分差不足"] += 1
                continue
        if c is not None and j is not None:
            gaps.append(c - j)

        if len(chosen) > args.max_chars or len(rejected) > args.max_chars:
            stat["太长"] += 1
            continue

        ratio = len(chosen) / max(len(rejected), 1)
        ratios.append(ratio)
        if args.max_len_ratio and ratio > args.max_len_ratio:
            stat["长度比过大"] += 1
            continue

        if instruction in seen:
            stat["重复题面"] += 1
            continue
        seen.add(instruction)

        kept.append({
            "prompt": render_clean(instruction),
            "chosen": chosen,
            "rejected": rejected,
        })
        if args.limit and len(kept) >= args.limit:
            break

    print(f"\n采用 {len(kept)} 条")
    for k, v in stat.items():
        print(f"  {k:10s} {v}")
    if gaps:
        print(f"  评分差 gap：中位 {sorted(gaps)[len(gaps) // 2]:.2f}｜最小 {min(gaps):.2f}｜最大 {max(gaps):.2f}")

    # —— 长度统计：这是「要不要缩 --limit / 调 --max-seq-len」的依据 ——
    cl = [len(x["chosen"]) for x in kept]
    rl = [len(x["rejected"]) for x in kept]
    print(f"\n字符长度  chosen  中位 {percentile(cl, .5)}  p95 {percentile(cl, .95)}  max {max(cl, default=0)}")
    print(f"           rejected 中位 {percentile(rl, .5)}  p95 {percentile(rl, .95)}  max {max(rl, default=0)}")
    # 长度差是个真实的坑：DPO 很容易学成「变啰嗦」，训练前先看两边差多少
    diff = [len(x["chosen"]) - len(x["rejected"]) for x in kept]
    print(f"          chosen - rejected 中位 {percentile(diff, .5)}（正数=偏好更长的那份）")

    # 长度偏置的取舍：卡得越紧，「更长=更好」这条捷径去得越干净，但数据也越少。
    # 这是个显式的取舍，没有免费选项 —— 所以把存活率摆出来再决定。
    if ratios:
        base = len(ratios)
        sr = sorted(ratios)
        print("\n--max-len-ratio 存活率（在当前 min-gap / max-chars 过滤之后）:")
        for q in (1.2, 1.5, 2.0, 3.0, 5.0):
            n = sum(1 for x in ratios if x <= q)
            print(f"  ratio <= {q:<4} 剩 {n:6d}/{base} ({n / base:.1%})")
        print(f"  ratio 中位 {sr[base // 2]:.2f}｜p95 {sr[min(base - 1, int(base * .95))]:.2f}")

    if tokenizer is not None and kept:
        pl = [len(tokenizer(x["prompt"], add_special_tokens=False)["input_ids"]) for x in kept]
        # 训练时 TRL 会给 chosen/rejected 各追加一个 <|im_end|>
        tl = [len(tokenizer(x["prompt"] + x["chosen"], add_special_tokens=False)["input_ids"]) + 1
              for x in kept]
        over = sum(1 for t in tl if t > args.max_seq_len)
        overp = sum(1 for p in pl if p > args.max_prompt_len)
        print(f"\ntoken 长度 prompt   中位 {percentile(pl, .5)}  p95 {percentile(pl, .95)}  max {max(pl)}")
        print(f"           prompt+chosen 中位 {percentile(tl, .5)}  p95 {percentile(tl, .95)}  max {max(tl)}")
        print(f"  超过 --max-seq-len {args.max_seq_len} 的：{over}/{len(tl)} ({over / len(tl):.1%})")
        print(f"  超过 --max-prompt-length {args.max_prompt_len} 的：{overp}/{len(pl)} ({overp / len(pl):.1%})")
        if over / len(tl) > 0.05:
            print("  ⚠️ 截断比例偏高：被截断的偏好对等于噪声，考虑调小 --limit 或调大 --max-seq-len")

    if args.stats_only:
        print("\n--stats-only：不写文件")
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n已写 {output}：{len(kept)} 条")

    sample = kept[0]
    print("\n样例：")
    print("  prompt 尾 : " + repr(sample["prompt"][-70:]))
    print("  chosen    : " + repr(sample["chosen"][:70]))
    print("  rejected  : " + repr(sample["rejected"][:70]))


if __name__ == "__main__":
    main()
