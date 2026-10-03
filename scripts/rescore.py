"""离线重判：判分口径改了，**不必重跑模型**。

为什么需要它
------------
生成是贵的那一步（占显存、占时间、还要和其它任务抢卡），判分是廉价的。
结果 json 里把每条 `raw_output` / `output` 都存了下来，题目文件里也有规则和单测，
所以**抽取逻辑或判分逻辑一改，拿存档重判即可** —— 零显存、几秒钟、结论立刻更新。

它救回过两次
------------
1. **HumanEval**：`code_extract.strip_fence` 的闭围栏正则锚定了 `$`，而模型输出末尾
   粘着乱码 token（`` ``` לחלוט ``），围栏没剥掉 → 拼出来的程序第一行是 ```` ``` ````
   → SyntaxError。164 题里挂了 80 题，真实 73.2% 被压成 35.4%，
   差点写出「SFT 把代码能力练废了」这个完全相反的结论。

2. **指令遵循**：同一个根因。200 条里 sft-4b-v2 有 26 条判定「不是合法 Python」，
   重判后 **25 条恢复**（它的输出是完整的代码，只是围栏后面拖了乱码）。
   base 的 23 条是同症状但**真失败** —— 它确实在代码后面加了中文解释，
   违反「只输出 Python 代码，不要解释」。所以重判不是「一律放水」，它只修抽取。

覆盖范围
--------
| 评测项 | 重判依据 | 说明 |
| --- | --- | --- |
| HumanEval | 重新抽取 + 重新执行单测 | 会真的跑代码，有超时保护 |
| 指令遵循 | 重新跑 200 条规则判定 | 规则从 `evals/ifollow-subset.jsonl` 读，和线上同一份 |

MMLU / GSM8K / OpenQA **不在范围内**：它们的判分只看生成文本，重判就等于重抽一次
最后一个数字/字母，价值不大（GSM8K 那类问题的根因是生成被截断，只能重跑生成）。

用法
----
    python scripts/rescore.py evals/results/*.json            # 只对比，不落盘
    python scripts/rescore.py --write evals/results/*.json    # 写回

注意：本脚本要 import eval（复用 `_run_python` 的执行方式、`check_all` 的判定、
`_attach_ci` 的置信区间口径），而 eval 依赖 unsloth —— 所以**在装了项目环境的机器上跑**。
抽取逻辑本身在 `code_extract.py`，那个是纯文本、随处可跑。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval as ev  # noqa: E402  复用执行方式与判分口径
from answer_extract import (  # noqa: E402
    follows_hash_format,
    has_other_script,
    mmlu_letter,
    trim_junk_tail,
)
from code_extract import extract_code  # noqa: E402

SUITES = ("mmlu_gen", "mmlu_gen_plain", "ifollow", "humaneval")


def load_items(path: Path) -> dict[str, dict]:
    return {item["id"]: item for item in ev._read_items(path)}


def rescore_mmlu(data: dict, key: str) -> tuple[int, int]:
    """重新抽选项字母。结果 json 里存了生成原文，所以纯离线。

    `by_rule` 会一并重算 —— 它记录有多少题是靠**兜底规则**（可能抓到题干里的
    选项标号）蒙出来的。这个比例高，说明这个口径对当前模型不可信。
    """
    slot = data[key]
    records = slot["records"]
    before = slot["correct"]
    correct = 0
    by_rule: dict[str, int] = {}
    for record in records:
        letter, rule = mmlu_letter(record["generated"])
        record["predicted"] = letter
        record["rule"] = rule
        hit = letter is not None and letter == record["answer"]
        record["correct"] = bool(hit)
        correct += int(hit)
        by_rule[rule] = by_rule.get(rule, 0) + 1
    slot.update({
        "total": len(records), "correct": correct,
        "accuracy": round(correct / len(records), 4) if records else 0.0,
        "unparsed": by_rule.get("none", 0),
        "by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1])),
    })
    ev._attach_ci(slot)
    return before, correct


def rescore_ifollow(data: dict, items: dict[str, dict]) -> tuple[int, int]:
    """按当前规则判定重算指令遵循。规则和线上是同一份文件。"""
    slot = data["ifollow"]
    records = slot["records"]
    before = slot["passed"]
    hit = 0
    trimmed_ok = 0
    flipped: list[str] = []
    by_category: dict[str, dict] = {}
    for record in records:
        item = items[record["id"]]
        passed, detail = ev.check_all(item["rules"], record["output"])
        record["passed"] = passed
        record["detail"] = detail
        hit += int(passed)
        # 对照：去掉尾部乱码再判一次（判分本身仍看原始输出）
        trimmed, _ = ev.check_all(item["rules"], trim_junk_tail(record["output"]))
        trimmed_ok += int(trimmed)
        if trimmed and not passed:
            flipped.append(record["id"])
        bucket = by_category.setdefault(record["category"], {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(passed)
    for bucket in by_category.values():
        bucket["rate"] = round(bucket["passed"] / bucket["total"], 4) if bucket["total"] else 0.0
    total = len(records)
    slot.update({
        "total": total, "correct": hit, "passed": hit,
        "rate": round(hit / total, 4) if total else 0.0,
        "by_category": by_category,
        "passed_junk_trimmed": trimmed_ok,
        "rate_junk_trimmed": round(trimmed_ok / total, 4) if total else 0.0,
        "flipped_by_junk_tail": flipped,
    })
    ev._attach_ci(slot)
    return before, hit


def rescore_humaneval(data: dict, items: dict[str, dict], timeout: int) -> tuple[int, int]:
    """重新抽取 + 重新执行单测。"""
    slot = data["humaneval"]
    records = slot["records"]
    before = slot["correct"]
    correct = 0
    for record in records:
        item = items[record["id"]]
        completion = extract_code(record["raw_output"])
        program = (
            item["prompt"] + completion + "\n" + item["test"]
            + f"\ncheck({item['entry_point']})\n"
        )
        passed, why = ev._run_python(program, timeout)
        record["passed"] = passed
        record["detail"] = why
        record["completion"] = completion
        correct += int(passed)
    slot.update({
        "total": len(records), "correct": correct,
        "accuracy": round(correct / len(records), 4) if records else 0.0,
    })
    ev._attach_ci(slot)
    return before, correct


def refresh_diagnostics(data: dict) -> list[str]:
    """从存档输出重算**能重算**的诊断位。

    `junk_tail` 和 `follows_format` 完全由输出文本决定，所以早先跑的结果也能补上，
    不必重跑模型；`truncated` 需要 token 数，旧版本没存，只能留空 ——
    **不知道就留空，不猜**。
    """
    touched = []
    for key, field in (("mmlu_gen", "generated"), ("mmlu_gen_plain", "generated"),
                       ("ifollow", "output"), ("gsm8k", "output"),
                       ("humaneval", "raw_output")):
        slot = data.get(key)
        if not slot or not slot.get("records"):
            continue
        junk = 0
        for record in slot["records"]:
            text = record.get(field) or ""
            flag = has_other_script(text)
            record["junk_tail"] = flag
            junk += int(flag)
            if key == "gsm8k":
                record["follows_format"] = follows_hash_format(text)
        slot["junk_tail"] = junk
        if key == "ifollow":
            slot["failed_with_junk_tail"] = sum(
                1 for r in slot["records"] if r["junk_tail"] and not r["passed"]
            )
        if key == "gsm8k":
            slot["follows_format"] = sum(1 for r in slot["records"] if r["follows_format"])
        touched.append(key)
    return touched


def rescore_one(path: Path, items: dict, timeout: int, write: bool) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    touched = []
    for suite in SUITES:
        slot = data.get(suite)
        if not slot or not slot.get("records"):
            continue
        # 每个评测项靠哪个字段重判
        field = "raw_output" if suite == "humaneval" else ("generated" if suite.startswith("mmlu") else "output")
        missing = [r["id"] for r in slot["records"] if not r.get(field)]
        if missing:
            print(f"  {path.stem}: 跳过 {suite}（{len(missing)} 条没存生成文本，"
                  "是旧版本跑的结果，只能重跑）")
            continue

        total = len(slot["records"])
        if suite.startswith("mmlu"):
            before, after = rescore_mmlu(data, suite)
        elif suite == "ifollow":
            before, after = rescore_ifollow(data, items["ifollow"])
        else:
            before, after = rescore_humaneval(data, items["humaneval"], timeout)
        delta = after - before
        arrow = "  (无变化)" if delta == 0 else f"  ({delta:+d} 条)"
        print(f"  {path.stem:<16} {suite:<15} {before:>3}/{total} = {before / total * 100:5.1f}%"
              f"  →  {after:>3}/{total} = {after / total * 100:5.1f}%{arrow}")
        touched.append(suite)

    diag = refresh_diagnostics(data)
    if diag:
        print(f"  {path.stem:<16} 诊断位已重算：{'、'.join(diag)}")

    if write and (touched or diag):
        data["rescored_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data["rescore_note"] = (
            f"离线重判过：{'、'.join(touched)}（scripts/rescore.py）。"
            "修掉了闭围栏正则锚定 $ 导致把完整代码误判为语法错误的问题，模型未重新生成。"
        )
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 只要有一项变了就算「有变化」
    return bool(touched)


def main():
    parser = argparse.ArgumentParser(description="离线重判（改判分口径不必重跑模型）")
    parser.add_argument("results", nargs="+", help="结果 json（可用通配符，shell 展开）")
    parser.add_argument("--ifollow", default=str(PROJECT_DIR / "evals" / "ifollow-subset.jsonl"))
    parser.add_argument("--humaneval", default=str(PROJECT_DIR / "evals" / "humaneval.jsonl"))
    parser.add_argument("--timeout", type=int, default=10, help="HumanEval 单题执行超时秒数")
    parser.add_argument("--write", action="store_true", help="把重判结果写回 json")
    args = parser.parse_args()

    items = {
        "ifollow": load_items(Path(args.ifollow)),
        "humaneval": load_items(Path(args.humaneval)),
    }
    print(f"题目：指令遵循 {len(items['ifollow'])} 条　HumanEval {len(items['humaneval'])} 条")
    print("重判：")
    changed = 0
    for raw in args.results:
        path = Path(raw)
        if not path.exists():
            print(f"  跳过 {raw}：文件不存在")
            continue
        if rescore_one(path, items, args.timeout, args.write):
            changed += 1
    print(f"\n{len(args.results)} 份结果里 {changed} 份被重判"
          + ("（已写回）" if args.write else "（未写回，加 --write 落盘）"))


if __name__ == "__main__":
    main()
