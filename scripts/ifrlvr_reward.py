"""IF-RLVR 的可验证奖励：**通过约束的比例**。

判分逻辑完全照抄官方实现
------------------------
来源：`allenai/open-instruct` → `open_instruct/ground_truth_utils.py` 的
`IFEvalVerifier.__call__`（本仓库 vendor 在 `scripts/open_instruct/IFEvalG/`）。

官方那段的核心（逐条约束判 0/1，最后取平均）：

    for instruction_key, args in zip(instruction_keys, args_list):
        args = {k: v for k, v in args.items() if v is not None}
        inst = INSTRUCTION_DICT[instruction_key](instruction_key)
        inst.build_description(**args)
        rewards.append(1.0 if inst.check_following(answer) else 0.0)
    score = sum(rewards) / max(len(rewards), 1)

**为什么用比例而不是 0/1**：GRPO 靠组内相对优势算梯度。8 条全 0 或全 1 时组内方差为 0，
梯度也是 0（论文 §4.1 的多约束设计正是为此）。比例天然给出分档：过 2/5 就是 0.4。

和 `rules.py` 的分工
--------------------
`rules.py` 只服务项目自制的 200 条评测集（24 种规则，中文模板），**不参与训练**。
本模块服务 IF-RLVR 训练（54 种约束 = IFEval 25 + IFTrain 29），两者互不影响。

一个刻意的**不作弊**设计
------------------------
`check_following` 只吃模型原始输出，不做任何剪裁、不抬 token、不看围栏 —— 与项目
「评测不允许任何推理技巧」的硬规则一致。也因此，输出尾部那点乱码会真实地判失败
（这正是把「乱码尾」算进 reward 的地方）。

用法（离线自检，不需要 GPU）：

    python scripts/ifrlvr_reward.py --data data/processed/grpo-ifrlvr/train.jsonl --n 200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from open_instruct.IFEvalG import instructions_registry

INSTRUCTION_DICT = instructions_registry.INSTRUCTION_DICT

# Qwen3 的思考段（本项目用 Qwen3-4B-Base 起步，正常情况下不会出现；
# 留着是为了万一某轮回滚到 instruct 版时不至于把 `...` 一起判进去）
THINK_RE = re.compile(r"^.*?<｜end▁of▁thinking｜>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """去掉 `...` 思考段。没有则原样返回。"""
    if "<｜end▁of▁thinking｜>" in text:
        return THINK_RE.sub("", text).strip()
    return text


def score_one(
    completion: str,
    instruction_ids: list[str],
    kwargs_list: list[dict | None],
    prompt: str | None = None,
) -> tuple[float, list[bool], list[str]]:
    """对一条输出判分。

    返回 `(通过比例, 每条约束是否通过, 每条约束的说明)`。
    空输出直接 0 分（与官方一致）。
    """
    answer = strip_thinking(completion or "")
    if not answer.strip():
        n = len(instruction_ids)
        return 0.0, [False] * n, ["空输出"] * n

    flags: list[bool] = []
    notes: list[str] = []
    for instruction_id, raw_kwargs in zip(instruction_ids, kwargs_list):
        cls = INSTRUCTION_DICT.get(instruction_id)
        if cls is None:
            # 数据里出现 registry 不认识的约束：记 0 并留痕，方便定位版本漂移
            flags.append(False)
            notes.append(f"未知约束 {instruction_id}")
            continue
        kwargs = {k: v for k, v in (raw_kwargs or {}).items() if v is not None}
        inst = cls(instruction_id)
        inst.build_description(**kwargs)
        # 少数约束（如 repeat_prompt）需要题目原文；官方 GRPO 路径靠 kwargs 自带，
        # 这里按 IFBench 的 evaluation_lib 再兜一层 prompt
        args = inst.get_instruction_args()
        if prompt is not None and args and "prompt" in args:
            inst.build_description(prompt=prompt)
        try:
            ok = bool(inst.check_following(answer))
        except Exception as exc:  # 判分器自身抛错时不能连累整批训练
            ok = False
            notes.append(f"{instruction_id} 判分异常: {exc}")
            flags.append(ok)
            continue
        flags.append(ok)
        notes.append(f"{instruction_id}:{'✓' if ok else '✗'}")
    return sum(flags) / max(len(flags), 1), flags, notes


def ifrlvr_reward(
    completions: list[str],
    constraint_ids: list[list[str]] | None = None,
    constraint_kwargs: list[list[dict | None]] | None = None,
    prompts: list[str] | None = None,
    length_penalty: float = 0.0,
    **_: object,
) -> list[float]:
    """TRL `GRPOTrainer(reward_funcs=[...])` 的奖励函数。

    数据集需要带 `constraint_ids` / `constraint_kwargs` 两列（列名刻意避开 `kwargs`，
    免得和 TRL 内部的透传参数撞名），`prompts` 由 GRPOTrainer 自动传入。

    `length_penalty`（字/步的惩罚系数）默认 0 —— **与论文一致**。论文没有长度惩罚，
    它靠「精确满足约束」本身抑制啰嗦。留着这个开关是为了消融，不是为了默认开启。
    """
    if constraint_ids is None or constraint_kwargs is None:
        raise ValueError("数据集必须带 constraint_ids / constraint_kwargs 两列")
    out: list[float] = []
    for i, completion in enumerate(completions):
        ids = constraint_ids[i]
        kws = constraint_kwargs[i]
        prompt = prompts[i] if prompts is not None else None
        ratio, _, _ = score_one(completion, ids, kws, prompt)
        if length_penalty:
            ratio -= length_penalty * len(completion or "")
        out.append(float(ratio))
    return out


# ------------------------------------------------------------------ 自检

def _selftest(data_path: Path, n: int) -> int:
    """离线自检：用「空输出」和「随机串」跑一遍全部约束，确认判分器不崩。

    这一步的价值是**提前暴露依赖缺失**（某条约束需要 nltk 数据 / langdetect 而环境里没有），
    而不是验证分数高低。
    """
    problems: list[str] = []
    total = 0
    with data_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            for label, text in (("空输出", ""), ("任意串", "The quick brown fox jumps.")):
                total += 1
                try:
                    score_one(text, row["constraint_ids"], row["constraint_kwargs"], row["prompt"])
                except Exception as exc:
                    problems.append(f"line {i} [{label}] {exc}")
    print(f"自检样本 {total} 次调用，异常 {len(problems)} 条")
    for p in problems[:20]:
        print("  ✗", p)
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="IF-RLVR 可验证奖励的自检")
    ap.add_argument("--data", required=True, help="转换后的 jsonl（含 constraint_ids / constraint_kwargs）")
    ap.add_argument("--n", type=int, default=200, help="只测前 n 条")
    args = ap.parse_args()
    return _selftest(Path(args.data), args.n)


if __name__ == "__main__":
    sys.exit(main())
