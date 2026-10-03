"""OpenQA 成对裁判：拿两份 eval.py 的结果，判 B 相对 A 的胜率。

为什么必须成对、必须换位
------------------------
开放式问题没有唯一答案，规则判不了。两个选择：

  1. 绝对打分（给每条回答打 1-5 分）—— 分数不可比、不可复现，
     换个裁判就变一套刻度，跨模型比较基本没有意义
  2. **成对偏好**（同一道题的两份回答，哪个更好）—— 判断的是相对好坏，
     比绝对打分稳得多

选 2。但成对比较有个已知的坑：**位置偏差**（裁判倾向于选前面那个）。
所以每道题判两次：`(A,B)` 和 `(B,A)`，只有两次结论**一致**才计入。
这样得到的胜率才有意义，不一致的记为「位置敏感」单独报出来。

用法
----
    python scripts/judge_openqa.py \
        --a evals/results/base-4b.json \
        --b evals/results/sft-4b-v2.json \
        --judge weights/Qwen3-4B-Instruct-2507

裁判模型用的是官方 instruct 版。注意：**不能拿被测模型自己当裁判**，
否则等于自己给自己打分。
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402

import eval as eval_module  # noqa: E402 —— 复用同一套 Wilson 区间，免得两个脚本算法漂移
import train_sft as ts  # noqa: E402

_wilson_ci = eval_module._wilson_ci

JUDGE_PROMPT = """你是一个严格的评审。下面是一道问题和两份回答，请判断哪一份更好。

评判标准，按重要性排序：
1. 是否正确——有没有事实错误、有没有编造
2. 是否切题——有没有答非所问
3. 是否遵守了问题里的约束（字数、格式、结构等）
4. 是否完整、清楚、有条理

注意：更长的回答不等于更好。啰嗦、套话、重复都是缺点。
如果两份回答质量确实相当，判平局。

【问题】
{question}

【回答一】
{answer_one}

【回答二】
{answer_two}

请只输出一个字母：一 / 二 / 平"""


def _load_judge(path: str, max_seq_len: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path, max_seq_length=max_seq_len, dtype=None, load_in_4bit=False
    )
    # 裁判用官方 instruct 版自己的对话模板
    own = getattr(tokenizer, "chat_template", None)
    if own:
        tokenizer.chat_template = own
    model.eval()
    return model, tokenizer


def _ask(model, tokenizer, question: str, answer_one: str, answer_two: str, max_new_tokens: int) -> str:
    prompt = JUDGE_PROMPT.format(question=question, answer_one=answer_one, answer_two=answer_two)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=ts.stop_token_ids(tokenizer),
        )
    reply = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    for char in reply:
        if char in "一二平":
            return char
    return "?"


def _verdict(choice: str, swapped: bool) -> str:
    """把「一/二/平」翻译成不依赖顺序的结论。"""
    if choice == "平":
        return "tie"
    if choice == "?":
        return "unparsed"
    first_is_b = swapped  # 交换过的话，回答一就是 B
    if choice == "一":
        return "b" if first_is_b else "a"
    return "a" if first_is_b else "b"


def draw(ok: int, total: int, width: int = 30) -> str:
    if not total:
        return "░" * width
    filled = round(ok / total * width)
    return "█" * filled + "░" * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="OpenQA 成对裁判（含换位一致性检查）")
    parser.add_argument("--a", required=True, help="对照方结果 json（含 openqa.records）")
    parser.add_argument("--b", required=True, help="实验方结果 json")
    parser.add_argument("--judge", required=True, help="裁判模型路径（别用被测模型自己）")
    parser.add_argument("--out", default=None, help="输出 json，默认 evals/results/judge-<a>-vs-<b>.json")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="只判前 N 条（调试用）")
    args = parser.parse_args()

    a_data = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b_data = json.loads(Path(args.b).read_text(encoding="utf-8"))
    for tag, data in (("--a", a_data), ("--b", b_data)):
        if "openqa" not in data:
            sys.exit(f"!! {tag} 的结果里没有 openqa（先跑 eval.py --only openqa）")

    a_records = {r["id"]: r for r in a_data["openqa"]["records"]}
    b_records = {r["id"]: r for r in b_data["openqa"]["records"]}
    shared = [i for i in a_records if i in b_records]
    if not shared:
        sys.exit("!! 两份结果的题目 id 完全对不上，确认跑的是同一版 openqa.jsonl")
    if args.limit:
        shared = shared[: args.limit]

    print(f"==> 裁判: {args.judge}")
    print(f"==> 对照 A: {args.a}")
    print(f"==> 实验 B: {args.b}")
    print(f"==> 共同题目: {len(shared)} 条（每题判两次，交换顺序消除位置偏差）\n")

    model, tokenizer = _load_judge(args.judge, args.max_seq_len)

    started = time.time()
    details = []
    tally = {"a": 0, "b": 0, "tie": 0, "unparsed": 0, "position_sensitive": 0}
    for index, identifier in enumerate(shared, 1):
        question = a_records[identifier]["prompt"]
        answer_a = a_records[identifier]["output"]
        answer_b = b_records[identifier]["output"]
        forward = _verdict(_ask(model, tokenizer, question, answer_a, answer_b, args.max_new_tokens), swapped=False)
        backward = _verdict(_ask(model, tokenizer, question, answer_b, answer_a, args.max_new_tokens), swapped=True)
        if forward == backward:
            verdict = forward
        else:
            verdict = "position_sensitive"
        tally[verdict] = tally.get(verdict, 0) + 1
        details.append({
            "id": identifier, "category": a_records[identifier].get("category", ""),
            "forward": forward, "backward": backward, "verdict": verdict,
            "question": question, "answer_a": answer_a, "answer_b": answer_b,
        })
        if index % 10 == 0 or index == len(shared):
            print(f"   {index}/{len(shared)}  A={tally['a']} B={tally['b']} 平={tally['tie']} "
                  f"位置敏感={tally['position_sensitive']}")

    decisive = tally["a"] + tally["b"]
    win_rate = tally["b"] / decisive if decisive else 0.0
    low, high = _wilson_ci(tally["b"], decisive) if decisive else (0.0, 0.0)
    by_category: dict[str, dict] = {}
    for row in details:
        if row["verdict"] not in ("a", "b"):
            continue
        slot = by_category.setdefault(row["category"], {"a": 0, "b": 0})
        slot[row["verdict"]] += 1

    result = {
        "schema_version": 1,
        "judge": args.judge,
        "note": "换裁判后结果不可直接比，所以裁判路径要一起记下来",
        "a": {"path": args.a, "label": a_data.get("label"),
              "openqa_sha": a_data.get("subsets", {}).get("openqa", {}).get("sha256", "")},
        "b": {"path": args.b, "label": b_data.get("label")},
        "judged_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": tally,
        "decisive": decisive,
        "b_win_rate": round(win_rate, 4),
        "b_win_rate_ci95": [low, high],
        "by_category": {
            category: {
                **slot,
                "b_win_rate": round(slot["b"] / (slot["a"] + slot["b"]), 4) if slot["a"] + slot["b"] else 0.0,
            }
            for category, slot in sorted(by_category.items())
        },
        "elapsed": round(time.time() - started, 1),
        "details": details,
    }

    out_path = Path(args.out) if args.out else (
        PROJECT_DIR / "evals" / "results"
        / f"judge-{a_data.get('label', 'a')}-vs-{b_data.get('label', 'b')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"{'B 胜':<12}: {tally['b']:<4} {draw(tally['b'], decisive, 20)}")
    print(f"{'A 胜':<12}: {tally['a']:<4} {draw(tally['a'], decisive, 20)}")
    print(f"{'平局':<12}: {tally['tie']}")
    print(f"{'位置敏感':<12}: {tally['position_sensitive']}  ← 交换顺序后结论翻转，已在胜率里剔除")
    print(f"{'抽不出结论':<12}: {tally['unparsed']}")
    print("-" * 60)
    print(f"B 的胜率（仅计胜负分明）: {win_rate:.1%}  95% CI [{low:.1%}, {high:.1%}]   样本 {decisive}")
    for category, slot in result["by_category"].items():
        print(f"    {category:<6} B 胜率 {slot['b_win_rate']:.0%}（{slot['b']}/{slot['a'] + slot['b']}）")
    print(f"{'耗时':<12}: {result['elapsed']}s")
    print(f"{'结果写入':<12}: {out_path}")
    print("=" * 60)
    print("\n怎么读：CI 跨过 50% 就是「分不出高下」。样本只有几十条时，")
    print("        胜率差 10pp 以内基本都在噪声范围内 —— 别过度解读。")


if __name__ == "__main__":
    main()
