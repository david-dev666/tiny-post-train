"""直接测「靶心」：在收尾位置，`<|im_end|>` 排第几。

为什么不能只看 probe
--------------------
probe 是 3 条**开放式**问题（介绍一下你自己 / 什么是 LoRA / 写质数函数），
而 DPO 的训练数据是 alpaca 多任务。**两者分布不同** —— probe 没变化，
既可能是"训练无效"，也可能只是"没迁移过去"，分不出来。

这个脚本直接拿**训练数据自己的上下文**去问模型：

    prompt（= 已经到「内容末尾」的那一段）
        ↓ 一次前向，只看最后位置的 logits
    <|im_end|> 的排名 / 是不是 argmax

- 如果 DPO 有效：`<|im_end|>` 应该从"排第 4"升到**第 1**（超过乱码）
- 如果没动：说明连训练分布上都没掰过来 —— 那就没有"等下去"的价值了

用法
----
    # 只测 ref（合并了 SFT 层的 v2），作基线
    python scripts/check_im_end_rank.py --n 200

    # 再测训练后的 DPO 层，两个数字直接比
    python scripts/check_im_end_rank.py --n 200 --dpo-adapter outputs/dpo-4b-last1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

IM_END = "<|im_end|>"


def parse_args():
    p = argparse.ArgumentParser(description="测收尾位置 <|im_end|> 的排名")
    p.add_argument("--model", default="outputs/sft-4b-v2-merged",
                   help="起点。默认是已合并 SFT 层的 v2（= DPO 的 ref）")
    p.add_argument("--dpo-adapter", default=None,
                   help="叠加的 DPO 层；不给就只测 ref")
    p.add_argument("--data", default="data/processed/dpo-zh/last-token.jsonl")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--tokenizer", default="weights/Qwen3-4B-Base")
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    tag = "ref（" + Path(args.model).name + "）"
    if args.dpo_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.dpo_adapter)
        tag += " + " + Path(args.dpo_adapter).name
    model.eval()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()][: args.n]
    print(f"模型：{tag}")
    print(f"样本：{len(rows)} 条（来自 {args.data}）\n")

    ranks, is_top, top_tokens = [], 0, {}
    for row in rows:
        ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1]
        # 只看排名，不看绝对概率：绝对概率受整段序列长度影响，不可比
        rank = int((logits > logits[im_end_id]).sum().item()) + 1
        ranks.append(rank)
        if rank == 1:
            is_top += 1
        best = int(logits.argmax().item())
        piece = tokenizer.convert_ids_to_tokens(best)
        top_tokens[piece] = top_tokens.get(piece, 0) + 1

    if not ranks:
        sys.exit("!! 一条都没测到")
    ranks.sort()
    n = len(ranks)
    print(f"<|im_end|> 排名：中位 {ranks[n // 2]}｜前 25% 分位 {ranks[n // 4]}｜最好 {ranks[0]}")
    print(f"**排在第一位（argmax）的比例：{is_top}/{n} = {is_top / n:.1%}**")
    print(f"排名 ≤ 4 的比例：{sum(1 for r in ranks if r <= 4) / n:.1%}")
    print("\n模型实际首选的前 8 个 token：")
    for tok, cnt in sorted(top_tokens.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {tok!r:24s} {cnt}")


if __name__ == "__main__":
    main()
