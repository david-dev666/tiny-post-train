"""量 GRPO 起点的「通过率」与「显存/长度」，决定能不能开训。

为什么要有这个脚本
------------------
GRPO 的梯度来自**组内相对优势**。如果某条 prompt 的 8 条采样**全对或全错**，
这一组的标准差是 0，优势也是 0 —— 这条数据对训练毫无贡献。

论文（arXiv 2507.02833）的经验区间是「准确率落在 **30%~70%**」，太低/太高都会让
大量组方差归零。本项目在 DPO 上已经演过一遍「曲线健康但模型没动」，所以
**开训前必须先量这个数**，而不是训完再看。

同时量出显存峰值与长度分布，避免又出现「32G 卡跑 DPO 到第 102 步 OOM」那种事
（那次就是没量峰值、跑了才知道）。

判据（脚本会自己下结论）
------------------------
* `reward 均值` 落在 **0.30 ~ 0.70** → 可以开训
* `zero_std 组比例` 越低越好；> 0.5 说明难度两极分化严重，需要改数据配比
* `clipped 比例` 高 → 说明 `--max-tokens` 给小了（「写不完」会被算成「不会」）

用法
----
    # vllm 环境（唯一装了 vllm 的那个）
    /root/autodl-tmp/envs/vllm/bin/python scripts/probe_grpo_start.py \
        --model outputs/dpo-4b-open-merged --n-prompts 200 --k 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

# RTX 5090 是 sm120（Blackwell），vllm 0.30 默认给采样器挂 FlashInfer，
# 而当前 FlashInfer 的架构检查不认 sm120 → 引擎初始化直接失败。
# 必须在 import vllm 之前设。见 notes/workflow.md「GRPO 环境连环失败」。
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import CLEAN_CHAT_TEMPLATE  # noqa: E402
from ifrlvr_reward import score_one  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="量 GRPO 起点的通过率与显存")
    ap.add_argument("--model", default="outputs/dpo-4b-open-merged")
    ap.add_argument("--data", default="data/processed/grpo-ifrlvr/train.jsonl")
    ap.add_argument("--n-prompts", type=int, default=200, help="抽多少条 prompt")
    ap.add_argument("--k", type=int, default=8, help="每条采样几条（与训练时的 num_generations 一致）")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = CLEAN_CHAT_TEMPLATE

    rows = []
    with (PROJECT_DIR / args.data).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n_prompts:
                break
            rows.append(json.loads(line))
    print(f"==> 抽 {len(rows)} 条 prompt，每条采样 {args.k} 次，max_tokens={args.max_tokens}")

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for r in rows
    ]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        disable_log_stats=True,
    )
    sp = SamplingParams(n=args.k, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sp)

    group_means: list[float] = []
    zero_std_groups = 0
    all_rewards: list[float] = []
    lengths: list[int] = []
    clipped = 0
    per_constraint_pass = Counter()
    per_constraint_total = Counter()

    for row, out in zip(rows, outputs):
        rewards = []
        for cand in out.outputs:
            text = cand.text
            lengths.append(len(text))
            if cand.finish_reason == "length":
                clipped += 1
            ratio, flags, _ = score_one(
                text, row["constraint_ids"], row["constraint_kwargs"], row["prompt"])
            rewards.append(ratio)
            all_rewards.append(ratio)
            for cid, ok in zip(row["constraint_ids"], flags):
                per_constraint_total[cid] += 1
                per_constraint_pass[cid] += int(ok)
        mean = sum(rewards) / len(rewards)
        group_means.append(mean)
        if len(set(rewards)) == 1:
            zero_std_groups += 1

    n_groups = len(group_means)
    overall = sum(all_rewards) / len(all_rewards)
    print()
    print("=" * 62)
    print(f"整体通过率（约束级平均）  {overall:.4f}   ← 判据 0.30~0.70")
    print(f"组均值 分布              min {min(group_means):.3f} / "
          f"中位 {statistics.median(group_means):.3f} / max {max(group_means):.3f}")
    print(f"组内方差为 0 的组         {zero_std_groups}/{n_groups} "
          f"({zero_std_groups / n_groups * 100:.1f}%)  ← 这些组对训练零贡献")
    print(f"生成长度                 均值 {sum(lengths) / len(lengths):.0f} / "
          f"max {max(lengths)}")
    print(f"撞上 max_tokens 的比例    {clipped}/{len(lengths)} "
          f"({clipped / len(lengths) * 100:.1f}%)  ← 高说明上限给小了")
    print("=" * 62)
    print()
    print("各约束通过率（低 → 高，头尾各 12 条）：")
    rates = sorted(
        ((per_constraint_pass[c] / per_constraint_total[c], c, per_constraint_total[c])
         for c in per_constraint_total),
        key=lambda x: x[0],
    )
    for rate, cid, tot in rates[:12]:
        print(f"  {rate:6.2%}  {cid:<45} n={tot}")
    print("  ...")
    for rate, cid, tot in rates[-6:]:
        print(f"  {rate:6.2%}  {cid:<45} n={tot}")

    verdict = "✅ 可以开训" if 0.30 <= overall <= 0.70 else "❌ 通过率不在 30%~70%，先调数据难度"
    print()
    print(f"结论：{verdict}（整体通过率 {overall:.3f}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
