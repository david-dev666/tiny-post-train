"""把「答完吐乱码」翻成**只差最后一个 token** 的偏好对（末位 DPO 的数据）。

⚠️ **已证否，留档用。** 末位 DPO 跑了 2 轮（`dpo-4b-last1` / `last2`）：差分压到只剩最后 1 个
token、`margins` 推到 **5.19**，**生成行为依然没变**（ifollow 乱码尾 177/200）。根因是那个位置的
窗口只有 0.13 nats，序列级训练信号落不到单点上。**不要再跑这个方向。**

和已有的 `make_dpo_data.py --mode stop-token` 差在哪
---------------------------------------------------
已有的那份是：

    chosen   = 整条回答剪掉末尾乱码
    rejected = 模型原样输出

两条**从回答的第一个字就可能不同**（各自采样出来的），DPO 的 loss 于是摊在整条
序列上 —— 三轮 DPO 白训的根因就是这个：要用 ~150 个「两条完全一样的位置」的噪声，
去推收尾那一个位置 0.13 nats 的差距，推不动。

这份数据把它压到最小：

    prompt   = 内容（两条**逐 token 完全相同**的共同前缀）
    chosen   = <|im_end|>
    rejected = 那个乱码 token

于是「哪一部分算 loss」这件事由 **completion 的长度**自动决定 —— DPO 的
completion mask 只覆盖最后 1 个 token，梯度精确落在收尾决策上，别的位置一个都不碰。
不需要改任何 loss 实现。

为什么这样做是有依据的（而不是又一次拍脑袋）
--------------------------------------------
- 靶心已量化：收尾位置 P(`<|im_end|>`)=0.000426（第 4 名）vs 第一名乱码 0.000450，
  差 **0.13 nats** —— 要推的量极小
- 模型**已经知道该停**（im_end 就排在第 4），不是要它学新能力
- 所以需要的是「在那一个位置上把相对概率掰回来」，而不是重塑整个分布
- 上界已知：把 im_end 的 logit 抬 +2.0，指令遵循 61.5% → **80.5%**（解码侧实测）
  —— 那份数字是这轮的验收线（无 boost 时也该校到 ~80%）

数据从哪来
----------
`data/processed/rft/candidates.jsonl`（2000 题 × 8 候选，**带 token id**）。
用 token id 而不是文本，是因为「末尾有几个乱码 token」只有 token 层判得准
（`trailing_junk_len`，文本层剪不掉 `��取` 那种以正常汉字收尾的乱码）。

只在**答案正确**的候选上构造：先把「答错」这一类排除掉，否则 DPO 会连
「答对 vs 答错」一起学，那就不是我们要动的变量了。

用法（纯 CPU）
--------------
    /root/miniconda3/envs/tpt/bin/python scripts/make_dpo_last_data.py \
        --candidates data/processed/rft/candidates.jsonl \
        --output data/processed/dpo-zh/last-token.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from answer_extract import last_number, trailing_junk_len  # noqa: E402
import prompts as P  # noqa: E402

# 需要「材料」才完整的指示代词。alpaca 里这类题不少，但**它的 input 是空的**：
#     '提取这段文字中的观点。'          ← 没给"这段文字"
#     '用四到六句话概括以下文本。'       ← 没给"以下文本"
# 抽到它们，就是在"没有材料"的上下文里学收尾 —— 数据本身就脏。
# 判据：出现了这些词、却没有 `输入：`，就判题目不完整。
CONTENT_CUES = ("以下", "这段", "上述", "给定", "如下", "下面", "该文本", "此文本", "本文")

# 模型对「缺材料」的题会回答"抱歉，您没有提供…"。这类回答**没有实质内容**，
# 也就没有「内容写完该不该停」这个决策 —— 对我们要学的偏好没有信号。
REFUSAL_CUES = ("抱歉", "对不起", "请提供", "无法", "没有提供", "未能提供", "您没有")

STOP_TOKEN_TEXT = "<|im_end|>"


def parse_args():
    p = argparse.ArgumentParser(description="构造「只差最后一个 token」的 DPO 偏好对")
    p.add_argument("--candidates", default=None,
                   help="RFT 候选文件（含 gold，会做「答案对」过滤）")
    p.add_argument("--raw", default=None,
                   help="stop.raw.jsonl（prompt/text/ids/truncated，**不做 gold 过滤**）")
    p.add_argument("--output", default="data/processed/dpo-zh/last-token.jsonl")
    p.add_argument("--tokenizer", default="weights/Qwen3-4B-Base")
    p.add_argument("--max-per-question", type=int, default=2,
                   help="candidates 模式下每题最多构造几对")
    p.add_argument("--keep-incomplete", action="store_true",
                   help="不筛「题目缺材料」「回答是道歉」的样本（默认筛掉）")
    p.add_argument("--min-answer-chars", type=int, default=60,
                   help="回答短于这个长度就丢掉（太短的没有「收尾决策」可学）")
    return p.parse_args()


def special_text_ids(tokenizer) -> set[int]:
    """词表里形如 `<|...|>` 的 token（`<|fim_middle|>` 那类，见 make_rft_data.py）。"""
    return {
        index
        for token, index in tokenizer.get_vocab().items()
        if token.startswith("<|") and token.endswith("|>")
    }


def build_pair(tokenizer, ids: list[int], junk_n: int):
    """把一条脏候选拆成 (prompt, chosen, rejected)。

    ⚠️ 为什么 refused 不做「还原成那一个 token」：乱码 token 是**字节级 BPE**，
    1 个 token ↔ 多字符且**不可逆**。实测：

        原 ids 末尾 : [22, 17, 13, 140461, 198]      ← 140461 一个 token
        单独解码     : ' ได้แก่\\n'
        重新 tokenize: [127196, 19841, 124920, 18625, 198]   ← 变成 4 个

    所以 rejected 直接用**乱码的文本形态**（多个乱码 token 解出来的那段字符串）。
    重新切出来的是 2~4 个乱码 token —— 它们**仍然是模型会吐的东西**，DPO 照样
    压低它们；而 mask 只覆盖这几个 token，不是整条 150 个。

    prompt 则**必须自洽**：`tokenize(prompt) == ids[:-junk_n]`，否则 DPO 对齐的
    位置就不是我们以为的那一个（这类错位不报错、只是悄悄训错地方）。
    """
    prefix_ids = ids[: len(ids) - junk_n]
    junk_ids = ids[len(ids) - junk_n:]

    # 保留特殊 token 的字面文本：train_dpo.py 走的是「prompt 已是完整对话前缀」
    # 这条路（不再套 chat 模板）
    prompt = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    if tokenizer(prompt, add_special_tokens=False)["input_ids"] != list(prefix_ids):
        return None
    rejected = tokenizer.decode(junk_ids, skip_special_tokens=False)
    if not rejected.strip():
        return None
    # 🔴 **只留第一个非空白字符**（关键，有实测依据）
    #
    # 若整段乱码都当 rejected（这里是 6 个 token），DPO 学到的是「这段乱码不该出现」——
    # 它压低的是**后续那几个位置**，而「分叉点该选谁」根本没有被约束。
    # 实测（2656 对、163 步、margins 涨到 5.19）：靶心 argmax 只从 12.7% 动到 17.3%，
    # 83% 的样本第一位仍然是乱码。幅度全花在"整段概率"上，没花在"第一位选谁"上。
    #
    # 压到 1 个 token 之后，chosen（1 token）与 rejected（1 token）长度对称，
    # logratio 的差就只剩「那一位选谁」这一件事。
    rejected = rejected.lstrip()[:1]
    return prompt, STOP_TOKEN_TEXT, rejected


def main():
    args = parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ss = special_text_ids(tokenizer)
    print(f"词表里 `<|...|>` 标记 token：{len(ss)} 个（出现在正文即判脏）")

    if args.raw:
        source = Path(args.raw)
        if not source.exists():
            sys.exit(f"!! 找不到 {source}")
        raw = [json.loads(l) for l in open(source, encoding="utf-8") if l.strip()]
        # ⚠️ `stop.raw.jsonl` 的 `ids` **只是「回答」的 token**（不含用户指令、
        # 不含对话模板）。必须用 `render_clean` 把那一半拼回来 —— 否则模型会在
        # 「没有用户提问」的上下文里学收尾，而这个上下文推理时根本不存在
        # （第一版就是这么错的，所有样本都得作废）。
        # 拼字符串再 tokenize，而不是重新 tokenize 整条：后者会把原始生成的
        # 乱码 token 切碎（字节级 BPE 不可逆）。
        records = []
        for r in raw:
            prefix = P.render_clean(r.get("prompt") or "")
            records.append((prefix, r.get("ids") or [], bool(r.get("truncated"))))
        print(f"读入 {len(records)} 条原始记录（{source.name}，已补回用户回合前缀）")
    elif args.candidates:
        source = Path(args.candidates)
        if not source.exists():
            sys.exit(f"!! 找不到 {source}")
        items = [json.loads(l) for l in open(source, encoding="utf-8") if l.strip()]
        print(f"读入 {len(items)} 题候选")
        records = []
        for item in items:
            gold = last_number(item.get("gold") or "")
            # 存下来的 `prompt` 已经是渲染好的卷面（含 <|im_start|>user …），
            # 直接用作前缀；`ids` 同样只是回答部分
            prefix = item.get("prompt") or ""
            for cand in item["candidates"]:
                ids = cand.get("ids") or []
                if not ids or cand.get("hit_cap"):
                    continue
                predicted = last_number(cand.get("text") or "")
                if gold is None or predicted is None or abs(predicted - gold) >= 1e-4:
                    continue
                records.append((prefix, ids, False))
    else:
        sys.exit("!! 要给 --raw 或 --candidates 之一")

    pairs = []
    stat = {"记录": 0, "题目不完整": 0, "回答无实质": 0, "有乱码尾": 0,
            "含标记token": 0, "构造失败": 0, "采用": 0}
    for prefix, out_ids, truncated in records:
        stat["记录"] += 1
        if not out_ids or truncated:
            continue

        # —— 题目不完整就丢掉（缺材料还硬问"概括以下文本"）——
        if not args.keep_incomplete:
            user_part = prefix.replace("<|im_start|>user\n", "").split("<|im_end|>")[0]
            if "输入：" not in user_part and any(c in user_part for c in CONTENT_CUES):
                stat["题目不完整"] += 1
                continue
            # —— 回答是"抱歉，您没有提供…"，没有实质内容，也就没有收尾决策 ——
            answer = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            if len(answer) < args.min_answer_chars or answer.startswith(REFUSAL_CUES):
                stat["回答无实质"] += 1
                continue
        # 前缀 tokenize + 原始输出 token：**不重新 tokenize 输出**，
        # 保留原始生成的乱码 token
        ids = tokenizer(prefix, add_special_tokens=False)["input_ids"] + list(out_ids)
        junk_n = trailing_junk_len(tokenizer, ids)
        if junk_n <= 0:
            continue
        stat["有乱码尾"] += 1
        # 乱码 token 里若混着 `<|...|>` 标记，说明这不是"收尾吐了个乱码字符"
        # 那种干净可用的偏好信号，丢掉
        if any(int(t) in ss for t in ids[len(ids) - junk_n:]):
            stat["含标记token"] += 1
            continue
        built = build_pair(tokenizer, ids, junk_n)
        if built is None:
            stat["构造失败"] += 1
            continue
        prompt, chosen, rejected = built
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        stat["采用"] += 1

    print("\n统计：")
    for k, v in stat.items():
        print(f"  {k:9s} {v}")
    print(f"\n构造出 {len(pairs)} 对（只差最后一个 token）")

    if not pairs:
        sys.exit("!! 一对都没构造出来，别往下训")
    if stat["构造失败"]:
        print(f"⚠️ 有 {stat['构造失败']} 条自检不通过（拼回去不还原），已丢弃")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已写 {output}")

    sample = pairs[0]
    print("\n样例：")
    print("  prompt 尾 : " + repr(sample["prompt"][-60:]))
    print("  chosen    : " + repr(sample["chosen"]))
    print("  rejected  : " + repr(sample["rejected"]))


if __name__ == "__main__":
    main()
