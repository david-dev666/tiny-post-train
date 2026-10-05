"""RFT 第二步：给候选打分、筛出「答对且干净」的，转成 SFT 训练数据。

奖励怎么定（每一项都能验证，没有主观判断）
------------------------------------------
    correct = 抽到的最后一个数字 == 标准答案（误差 < 1e-4，与 eval.py 同一口径）
    clean   = 不是打满上限截断的（走到收尾决策了）且 trailing_junk_len == 0
    repeat  = repeat_2gram 过高（复读）

    reward = 1.0 * correct + 0.3 * clean - 0.5 * (repeat > 0.5)

**为什么 clean 要进奖励、而且必须是被选中样本的硬门槛**
v2 的已知缺陷就是"答完不吐 `<|im_end|>`、先吐 1~2 个乱码 token"（评测 151/200）。
SFT 里三条硬修的路（末位 loss / 收尾加权 / eos 对齐）全部证否 —— 那个位置的梯度
要么推不动、要么一推就把内容带坏。**但"只在自己干净收尾的样本上继续学"是另一条路**：
不用去压某一个 token 的梯度，而是直接改**训练数据的分布**。

**为什么必须筛，不能将就**（这是 v3 第五轮栽过的地方）
上一轮把自生成数据**不做质量过滤**直接喂回去，结果复读被原样学进去了。
所以这里只保留 `correct and clean` 的样本；某题一条都没有就**整题丢掉**，
绝不退而求其次挑"最不脏"的那条。

用法（纯 CPU，tpt / vllm 环境都行）
-----------------------------------
    /root/miniconda3/envs/tpt/bin/python scripts/make_rft_data.py \
        --candidates data/processed/rft/candidates.jsonl \
        --output data/processed/rft/sft.jsonl \
        --mix-alpaca data/raw/alpaca-gpt4-zh --mix-n 300

`--mix-alpaca` 混入若干原始 SFT 数据，防止在 GSM8K 这一种分布上过拟合而遗忘通用能力。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from answer_extract import (  # noqa: E402
    JUNK_CHAR_RE,
    JUNK_TAIL_WINDOW,
    last_number,
    trailing_junk_len,
)
from metrics import repetition_metrics  # noqa: E402
from prompts import GSM8K_INSTRUCTION  # noqa: E402

REPEAT_PENALTY_THRESHOLD = 0.5
CLEAN_BONUS = 0.3
REPEAT_PENALTY = 0.5


def special_text_ids(tokenizer) -> set[int]:
    """词表里所有**形如 `<|...|>`** 的 token id。

    为什么必须单独挑出来：Qwen3 的 `<|fim_middle|>` / `<|repo_name|>` / `<|file_sep|>`
    这类 token **在词表里，却不在 `tokenizer.all_special_tokens` 里** ——
    于是 vLLM 的 `skip_special_tokens=True` **不会剔除它们**，会原样出现在
    `completion.text` 里（实测 1554 条训练数据里混进 28 条）。
    它们是预训练时给代码补全 / FIM 用的标记，**正常回答里绝不该出现**。

    判据放在 token 层的原因：文本层用 `<|...|>` 正则会在代码类回答里误报
    （比如 Python 的 `a < |b|`），token 层则不会 —— 只有真·词表 token 才命中。
    """
    return {
        index
        for token, index in tokenizer.get_vocab().items()
        if token.startswith("<|") and token.endswith("|>")
    }


def has_special_text(ids, special_ids: set[int]) -> bool:
    return any(int(t) in special_ids for t in ids)


def parse_args():
    p = argparse.ArgumentParser(description="RFT：按奖励筛选候选，产出 SFT 数据")
    p.add_argument("--candidates", default="data/processed/rft/candidates.jsonl")
    p.add_argument("--output", default="data/processed/rft/sft.jsonl")
    p.add_argument("--tokenizer", default="weights/Qwen3-4B-Base",
                   help="算 trailing_junk_len 用（乱码尾判据依赖 tokenizer 的切法）")
    p.add_argument("--max-per-question", type=int, default=1,
                   help="每题最多保留几条（默认 1，只留奖励最高的那条）")
    p.add_argument("--mix-alpaca", default=None,
                   help="混入原始 SFT 数据的目录（防止只学 GSM8K 一种分布）")
    p.add_argument("--mix-n", type=int, default=300)
    p.add_argument("--stats-only", action="store_true",
                   help="只统计奖励分布、不写文件（先看数据够不够再决定）")
    return p.parse_args()


def score_candidate(cand: dict, gold: float, tokenizer, special_ids: set[int]) -> dict:
    """给一条候选打分。返回带诊断字段的 dict。"""
    text = cand.get("text") or ""
    ids = cand.get("ids") or []
    hit_cap = bool(cand.get("hit_cap"))

    predicted = last_number(text)
    correct = predicted is not None and gold is not None and abs(predicted - gold) < 1e-4
    junk_n = trailing_junk_len(tokenizer, ids) if ids else 0
    # 文本层**兜底**：`trailing_junk_len` 只吃「末尾连续的乱码 token」，
    # 而实测它会漏 —— `離`(U+F9AA) 那类兼容区字符不在乱码字符集里时，
    # token 层判 0、样本就被当成"干净"混进训练集（第一轮混进 1 条）。
    # 两道判据都过才算干净：宁可选少，不可喂脏。
    text_junk = bool(JUNK_CHAR_RE.search((text or "")[-JUNK_TAIL_WINDOW:]))
    # 第三道：`<|fim_middle|>` 那类「在词表里但不算 special」的标记 token。
    # 它们会被 vLLM 原样解进 text（见 special_text_ids 的注释）。
    special = has_special_text(ids, special_ids)
    clean = (junk_n == 0) and not hit_cap and not text_junk and not special
    repeat = repetition_metrics(ids)["repeat_2gram"] if ids else 0.0

    reward = (1.0 if correct else 0.0) + (CLEAN_BONUS if clean else 0.0)
    if repeat > REPEAT_PENALTY_THRESHOLD:
        reward -= REPEAT_PENALTY

    return {
        "reward": round(reward, 4),
        "correct": correct,
        "clean": clean,
        "junk_tokens": junk_n,
        "hit_cap": hit_cap,
        "repeat_2gram": repeat,
        "predicted": predicted,
        "text": text,
    }


def load_alpaca(directory: str, n: int) -> list[dict]:
    """从 alpaca 目录里抽 n 条，转成同样的 {instruction, input, output} 形状。"""
    if not n:
        return []
    path = Path(directory)
    csv_files = sorted(path.glob("*.csv"))
    if not csv_files:
        print(f"!! {directory} 下没有 csv，跳过混合")
        return []
    rows = []
    with open(csv_files[0], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "instruction": (row.get("instruction") or "").strip(),
                "input": (row.get("input") or "").strip(),
                "output": (row.get("output") or "").strip(),
            })
            if len(rows) >= n:
                break
    print(f"混合 alpaca：{len(rows)} 条（{csv_files[0].name}）")
    return rows


def main():
    args = parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    special_ids = special_text_ids(tokenizer)
    print(f"词表里的 `<|...|>` 标记 token：{len(special_ids)} 个（命中即判脏）")

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        sys.exit(f"!! 找不到 {candidates_path}，先跑 gen_rft_candidates.py")
    with open(candidates_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    print(f"读入 {len(items)} 题候选")

    total_cand = 0
    stat = {"correct": 0, "clean": 0, "both": 0, "hit_cap": 0, "junk": 0, "repeated": 0}
    kept = []
    no_good = 0

    for item in items:
        gold = last_number(item.get("gold") or "")
        scored = [
            score_candidate(c, gold, tokenizer, special_ids)
            for c in item["candidates"]
        ]
        scored.sort(key=lambda c: -c["reward"])

        for c in scored:
            total_cand += 1
            stat["correct"] += int(c["correct"])
            stat["clean"] += int(c["clean"])
            stat["both"] += int(c["correct"] and c["clean"])
            stat["hit_cap"] += int(c["hit_cap"])
            stat["junk"] += int(c["junk_tokens"] > 0)
            stat["repeated"] += int(c["repeat_2gram"] > REPEAT_PENALTY_THRESHOLD)

        # 硬门槛：只要「答对且干净」的。一条都没有就整题丢掉（不将就）
        good = [c for c in scored if c["correct"] and c["clean"]]
        if not good:
            no_good += 1
            continue
        # ⚠️ instruction 必须是**没套过对话模板**的裸指令：
        # 采样时用的 `prompt` 字段是 `render_clean()` 渲染过的（带 `<|im_start|>user`），
        # 而 train_sft.py 的 to_text() 会**再套一次**模板 —— 直接拿 prompt 当 instruction
        # 会训出「<|im_start|>user 出现两遍」的畸形样本（实测 1632 条全中）。
        instruction = item.get("instruction") or GSM8K_INSTRUCTION.format(
            question=item["question"]
        )
        for c in good[:args.max_per_question]:
            kept.append({
                "instruction": instruction,
                "input": "",
                "output": c["text"],
                "_reward": c["reward"],
            })

    if total_cand:
        print(f"\n候选总数 {total_cand}")
        for name in ("correct", "clean", "both", "hit_cap", "junk", "repeated"):
            print(f"  {name:9s} {stat[name]:6d}  ({stat[name] / total_cand:.1%})")
    print(f"\n选中的样本 {len(kept)} 条｜整题无可用样本被丢弃 {no_good} 题")

    if args.stats_only:
        print("\n--stats-only：不写文件")
        return

    extra = load_alpaca(args.mix_alpaca, args.mix_n) if args.mix_alpaca else []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in kept + extra:
            # _reward 只是诊断字段，不带进训练数据（train_sft 会按字段名取用）
            row = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_out = len(kept) + len(extra)
    print(f"\n已写 {output}：{n_out} 条（RFT {len(kept)} + 混合 {len(extra)}）")
    if n_out == 0:
        sys.exit("!! 一条都没选出来，别往下训 —— 先看上面的奖励分布")
    if len(kept) < 200:
        print("⚠️ RFT 样本少于 200 条，训练量可能不够（考虑把题目数或采样数调大）")


if __name__ == "__main__":
    main()
