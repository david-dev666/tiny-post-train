"""RFT 第一步：用当前模型对**训练题**采样 N 条候选，落盘原始记录（含 token id）。

它在整条 RFT 里的位置
--------------------
    gen_rft_candidates.py   →  make_rft_data.py  →  train_sft.py
    （采样 N 条候选）           （按奖励筛选）        （只在干净样本上续训）

为什么要「拒绝采样」而不是直接上 GRPO
--------------------------------------
GRPO 要把 vLLM 和训练进程接在一起，而本项目两个环境的 torch 版本不同
（tpt 2.12.1 / vllm 2.13），同进程装不到一起，只能起服务 —— 工程量大。
拒绝采样把这件事拆成"先采样、再筛选、最后就是一次普通 SFT"，
**用最小的工程量拿到 RL 最主要的收益**：模型只在自己"答对且干净"的样本上继续学。

为什么必须是「训练题」而不是评测题
--------------------------------
GSM8K 评测用的是 test split（1319 条）。这里读的是 **train split（7473 条）**，
两边零重叠 —— 否则分数是自己喂出来的，评测就失去意义了。

采样参数为什么不是贪心
--------------------
评测走贪心（`temperature=0`）是为了可复现；但 RFT 需要**同一题的不同解法**，
贪心只会得到 N 条一样的输出。所以这里 `temperature>0`，并且**故意不开
enforce_eager** —— 那条约束是给「尺子」用的（同批输入两遍要逐字相同），
采样阶段要的恰恰是多样性。

输出格式
--------
每行一条题：question / gold / prompt / candidates[{text, ids, hit_cap}]
- `ids` 必存：乱码尾的唯一可靠判据是 `answer_extract.trailing_junk_len`，
  它吃 **token id 序列**（文本层判不了 `��取` 这种以正常汉字收尾的乱码）
- `hit_cap` 必存：打满 max_new_tokens 的样本**没走到收尾决策**，
  不能算作"干净"，筛选时要排除

用法（**必须用 vllm 环境**，tpt 环境里没有 vllm）
------------------------------------------------
    /root/autodl-tmp/envs/vllm/bin/python scripts/gen_rft_candidates.py \
        --data data/raw/benchmarks/gsm8k_train.jsonl \
        --model weights/Qwen3-4B-Base \
        --adapter outputs/sft-4b-v2 \
        --limit 2000 --n 8 \
        --output data/processed/rft/candidates.jsonl

中断了直接再跑一次即可（`--resume` 默认开）：已经采完的题会跳过。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prompts as P  # noqa: E402
from engines import _prepare_env, strip_stop_tokens  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="RFT：采样 N 条候选（拒绝采样的原料）")
    p.add_argument("--data", required=True,
                   help="训练题 jsonl（GSM8K train），每行 {question, answer}")
    p.add_argument("--model", default="weights/Qwen3-4B-Base")
    p.add_argument("--adapter", default="outputs/sft-4b-v2",
                   help="挂哪个 adapter 采样。空字符串 = 不挂（基座，只做对照用）")
    p.add_argument("--output", default="data/processed/rft/candidates.jsonl")
    p.add_argument("--limit", type=int, default=0, help="只取前 N 题，0 = 全部")
    p.add_argument("--n", type=int, default=8, help="每题采几条候选")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--batch", type=int, default=200,
                   help="每批多少题（每批落一次盘，中断不至于全丢）")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-util", type=float, default=0.85)
    p.add_argument("--no-resume", action="store_true",
                   help="不用断点续跑（默认：跳过输出里已经有的题）")
    return p.parse_args()


def load_questions(path: str, limit: int) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if limit:
        items = items[:limit]
    return items


def done_questions(output: Path) -> set[str]:
    """已经采完的题（断点续跑用）。按题面文本去重。"""
    done = set()
    if not output.exists():
        return done
    with open(output, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["question"])
            except (json.JSONDecodeError, KeyError):
                # 末行可能被中断写坏，忽略即可
                continue
    return done


def main():
    args = parse_args()
    _prepare_env()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    items = load_questions(args.data, args.limit)
    skipped = done_questions(output) if not args.no_resume else set()
    todo = [it for it in items if it["question"] not in skipped]
    print(f"题目 {len(items)} 条｜已完成 {len(skipped)}｜本次要采 {len(todo)}")

    # tokenizer 用**我们自己加载的**，不用 llm.get_tokenizer()：
    # 停止符与 `trailing_junk_len` 的判据都必须和评测同源，否则训练口径和评测口径漂移
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stop_ids = P.stop_token_ids(tokenizer)
    print(f"停止符 ids：{stop_ids}")

    adapter = args.adapter or None
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        dtype="bfloat16",
        # 采样阶段刻意不开 enforce_eager（那是给评测当尺子用的，见模块注释）
        enable_lora=bool(adapter),
        max_lora_rank=64,
        disable_log_stats=True,
    )
    lora_request = LoRARequest("adapter", 1, adapter) if adapter else None

    params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop_token_ids=stop_ids,
        skip_special_tokens=True,
    )

    mode = "a" if skipped else "w"
    written = 0
    with open(output, mode, encoding="utf-8") as f:
        for start in range(0, len(todo), args.batch):
            chunk = todo[start:start + args.batch]
            prompts = [P.gsm8k_prompt(it["question"]) for it in chunk]
            outs = llm.generate(prompts, params, lora_request=lora_request, use_tqdm=False)

            for item, out in zip(chunk, outs):
                candidates = []
                for completion in out.outputs:
                    ids = strip_stop_tokens(
                        [int(t) for t in completion.token_ids], stop_ids
                    )
                    candidates.append({
                        "text": completion.text.strip(),
                        "ids": ids,
                        # 「用满预算」直接看 finish_reason（见 engines.py 的同一条注释）
                        "hit_cap": completion.finish_reason == "length",
                    })
                f.write(json.dumps({
                    "question": item["question"],
                    "gold": item["answer"],
                    # `instruction` 是**没套过模板的裸指令**，给训练数据用；
                    # `prompt` 是渲染好的卷面（带 <|im_start|>user），只用于采样。
                    # 两者混用会训出「模板套两遍」的畸形样本，见 make_rft_data.py 的注释。
                    "instruction": P.GSM8K_INSTRUCTION.format(question=item["question"]),
                    "prompt": P.gsm8k_prompt(item["question"]),
                    "candidates": candidates,
                }, ensure_ascii=False) + "\n")
                written += 1
            f.flush()
            os.fsync(f.fileno())
            print(f"  已采 {written}/{len(todo)} 题（×{args.n} 条候选）", flush=True)

    print(f"\n完成：{output}（本次写入 {written} 题）")


if __name__ == "__main__":
    main()
