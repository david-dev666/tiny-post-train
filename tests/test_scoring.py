"""判分逻辑的回归测试。

为什么必须有
------------
判分代码的 bug 不会报错，它只会**悄悄给出错误的分数**。今天一天里，两个这样的 bug
各自制造了一个看起来很合理的假结论：

- 闭围栏正则锚定 `$` → HumanEval 73.2% 变成 35.4%，差点写成「SFT 把代码练废了」
- MMLU 只做 `LETTER_RE.search()` 取第一个字母 → 复读题干时抓到选项标号，
  算出「chat 口径 10%」这种没有意义的数

两者都不会抛异常、不会有日志、不会让 pipeline 变红。**只有测试能拦住它们。**

这些测试**不依赖 torch / unsloth**（只 import 两个纯文本模块），所以 Mac 上直接跑：

    python tests/test_scoring.py
    python -m pytest tests/ -q        # 装了 pytest 也行
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from answer_extract import (  # noqa: E402
    follows_hash_format,
    has_junk_tail,
    last_number,
    mmlu_letter,
    trailing_junk_len,
)
from code_extract import extract_code, strip_fence  # noqa: E402

CASES: list[tuple[str, object, object]] = []


def case(name: str, got, want):
    CASES.append((name, got, want))


# ============================================================ 数字抽取

case("千分位不被拆碎", last_number("$80,000 + $50,000 = $130,000"), 130000.0)
case("千分位小数", last_number("答案是 1,250.5"), 1250.5)
case("普通整数", last_number("总共 42 个"), 42.0)
case("负数", last_number("结果是 -7"), -7.0)
case("没有数字", last_number("我不知道"), None)
# 曾经把 "80,000" 拆成 80 和 000，取最后一个得到 0
case("千分位不会退化成 0", last_number("房子值 $80,000"), 80000.0)


# ============================================================ 格式遵从度（只做诊断）

case("末尾 #### 数字", follows_hash_format("先算一下\n#### 42"), True)
case("末尾 #### 带逗号", follows_hash_format("所以\n#### 1,250"), True)
# 实测两个模型都把提示里的「数字」当字面量写进去了
case("#### 数字: 18.（字面量写法）", follows_hash_format("...#### 数字: 18. (Janet makes $18)"), True)
case("#### 数字 换行再给数", follows_hash_format("...#### 数字\n18"), True)
case("末尾有解释文字不影响", follows_hash_format("#### 42\n希望有帮助！"), True)
case("没用 #### 标记", follows_hash_format("答案是 42"), False)
case("空文本", follows_hash_format(""), False)
# 关键回归：基座把 #### 当章节编号用，标记后面没有数字 → 不算「按格式作答」
case("章节编号不算格式遵从", follows_hash_format("#### 第一步\n结果见上"), False)
# 标记出现在很靠前的位置，不算「末尾用标记作答」
case("标记在很早的位置", follows_hash_format("#### 42\n" + "补充说明。" * 40), False)
# 尾部乱码不该影响格式判定（sft 的输出就是这样）
case("尾部乱码不影响判定", follows_hash_format("#### 数字: 18. (解释) לחלוט"), True)


# ============================================================ MMLU 字母

case("只有字母", mmlu_letter("C"), (2, "only"))
case("字母加句点", mmlu_letter("C."), (2, "only"))
case("加粗字母", mmlu_letter("**B**"), (1, "only"))
case("前导空格", mmlu_letter(" C"), (2, "only"))
case("全角字母", mmlu_letter("Ｃ"), (2, "only"))
case("以字母开头带后文", mmlu_letter(" C\n\nQuestion: ..."), (2, "leading"))
case("答案标记", mmlu_letter("这道题的答案是 B。"), (1, "marked"))
case("英文标记", mmlu_letter("Answer: D"), (3, "marked"))

# 关键回归：复读题干时，第一个 A-D 是**选项标号**，不是答案
echo = "以下是一道单项选择题，请直接回答正确选项的字母。\n\nA. 0\nB. 1\nC. 4\nD. 6"
letter, rule = mmlu_letter(echo)
case("复读题干时规则名是 fallback（可信度最低）", rule, "fallback")

case("空输出", mmlu_letter("")[0], None)
case("没有字母", mmlu_letter("12345")[0], None)


# ============================================================ 代码围栏

case("纯代码不动", strip_fence("def f():\n    pass"), "def f():\n    pass")
case("带语言标签", strip_fence("```python\ndef f():\n    pass\n```"), "def f():\n    pass")
case("无标签", strip_fence("```\ndef f():\n    pass\n```"), "def f():\n    pass")

# 关键回归：闭围栏后面粘着乱码 token（实测 sft-4b-v2 的输出）
# 旧实现用 `\s*```$` 锚定末尾，匹配不上 → 围栏没剥掉 → 代码被判语法错误
junk_tail = "```python\ndef reverse_string(s):\n    return s[::-1]\n``` לחלוט"
case("围栏后有乱码也能剥掉", strip_fence(junk_tail), "def reverse_string(s):\n    return s[::-1]")
case(
    "乱码尾巴不参与判分",
    extract_code(junk_tail),
    "def reverse_string(s):\n    return s[::-1]",
)

# 函数后面多写的调用示例要被截掉
case(
    "截掉示例调用",
    extract_code("```python\ndef f():\n    return 1\n```\n\nif __name__ == '__main__':\n    print(f())"),
    "def f():\n    return 1",
)


# ============================================================ 乱码尾检测

case("检出希伯来乱码", has_junk_tail("return s[::-1]\n``` לחלוט"), True)
case("检出西里尔乱码", has_junk_tail("结果\nаци"), True)
case("正常中文不误报", has_junk_tail("这个函数返回反转后的字符串"), False)
case("正常英文不误报", has_junk_tail("the answer is 42"), False)
case("代码不误报", has_junk_tail("def f():\n    return x[::-1]"), False)

# 关键回归：替换字符 U+FFFD。原先的字符类只覆盖「其它文字系统」，漏了这一类，
# 于是 GSM8K 的乱码尾被报成 65.3%（真实 92.3%）、指令遵循 75.5%（真实 84.5%）。
case("检出替换字符乱码", has_junk_tail("#### 数字：18.�"), True)
case("检出替换字符+真汉字", has_junk_tail("#### 数字：14.��取"), True)


# --- token 层：乱码尾的唯一可靠判据 -----------------------------------------
#
# 为什么非要有这一层：`��取` 是**一个** token（字节级 BPE 的残缺片段，
# 前两个无效字节解成 U+FFFD、后三个字节解成「取」）。文本层看它「以正常汉字
# 收尾」，`[乱码字符]+$` 匹配不到 —— 既漏判也剪不干净。只有回头看 token 才准。
class _FakeTokenizer:
    """id → 字符串直接查表，够 trailing_junk_len 用。不需要真模型。"""

    def __init__(self, table: dict[int, str]):
        self.table = table

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(self.table[int(i)] for i in ids)


_tok = _FakeTokenizer({
    0: "你好", 1: "，世界", 2: " לחלוט", 3: " ", 4: "��取", 5: "。", 6: " פייסב",
})

case("token 层：干净收尾不剪", trailing_junk_len(_tok, [0, 1, 5]), 0)
case("token 层：希伯来乱码剪 1 个", trailing_junk_len(_tok, [0, 1, 2]), 1)
case("token 层：另一个希伯来乱码也剪", trailing_junk_len(_tok, [0, 1, 6]), 1)
# 这条是回归的核心：文本层认不出它，token 层必须认得
case("token 层：��取 剪得掉（文本层剪不掉）", trailing_junk_len(_tok, [0, 1, 4]), 1)
# 乱码后面还多吐了个空格 → 乱码和空格一起剪，别在答案里留个尾空格
case("token 层：乱码+尾空白一起剪", trailing_junk_len(_tok, [0, 1, 2, 3]), 2)
# 反过来：末尾只有一个空格、没有乱码 → 一个都不剪（不能白剪掉正常内容）
case("token 层：只有尾空白不剪", trailing_junk_len(_tok, [0, 1, 5, 3]), 0)
# 连续两个乱码 token 都要剪掉，但 max_scan 兜住「整段都是乱码」的极端情况
case("token 层：连续两个乱码", trailing_junk_len(_tok, [0, 1, 2, 6]), 2)
case("token 层：max_scan 上限", trailing_junk_len(_tok, [2, 6, 2, 6, 2, 6], max_scan=4), 4)
case("token 层：空序列", trailing_junk_len(_tok, []), 0)
case("token 层：全是空白", trailing_junk_len(_tok, [3, 3]), 0)


# ============================================================ 指令遵循规则引擎

import rules  # noqa: E402

for _rule, _text, _want, _name in [
    # --- 长度 / 计数类
    ({"type": "max_chars", "n": 3}, "你好吗", True, "字数不超"),
    ({"type": "max_chars", "n": 3}, "你好吗啊", False, "字数超了"),
    ({"type": "char_count_eq", "n": 2}, "你好", True, "恰好 2 字"),
    ({"type": "line_count_eq", "n": 2}, "a\nb", True, "恰好 2 行"),
    ({"type": "line_count_eq", "n": 2}, "a\nb\nc", False, "3 行了"),
    # 关键回归：_lines 曾用 splitlines()，U+2028 会被当成换行 → 1 行变 2 行
    ({"type": "line_count_eq", "n": 1}, "第一行\u2028第二行", True, "U+2028 不算换行"),
    ({"type": "single_line"}, "第一行\u2028第二行", True, "U+2028 仍是单行"),
    # --- 结构类
    ({"type": "line_prefix_count", "prefix": "- ", "n": 2}, "- a\n- b", True, "2 个要点"),
    # 关键回归：题面写「用三个要点」，所以是**恰好**，不是「至少」
    ({"type": "line_prefix_count", "prefix": "- ", "n": 2}, "- a\n- b\n- c", False, "给多了要点"),
    ({"type": "line_starts_with_all", "prefixes": ["# ", "## "]}, "# a\n## b", True, "两种前缀都有"),
    ({"type": "starts_with", "text": "答"}, "答：是", True, "以「答」开头"),
    ({"type": "ends_with", "text": "。"}, "结束。", True, "以句号结尾"),
    # --- 内容 / 禁忌
    ({"type": "contains_all", "words": ["甲", "乙"]}, "甲乙", True, "都含"),
    ({"type": "contains_all", "words": ["甲", "乙"]}, "只有甲", False, "缺一个"),
    ({"type": "not_contains", "words": ["```"]}, "纯文本", True, "不含围栏"),
    ({"type": "not_contains", "words": ["```"]}, "```py", False, "命中围栏"),
    ({"type": "no_prefix", "prefixes": ["当然"]}, "首先", True, "开头干净"),
    ({"type": "no_prefix", "prefixes": ["当然"]}, "当然可以", False, "踩了禁忌开头"),
    # --- 语言类
    ({"type": "all_upper"}, "ABC", True, "全大写"),
    ({"type": "all_upper"}, "AbC", False, "混了小写"),
    ({"type": "all_lower"}, "abc", True, "全小写"),
    ({"type": "ascii_letters_max", "n": 2}, "ab你", True, "英文字母 2 个"),
    ({"type": "ascii_letters_max", "n": 2}, "abc你", False, "英文字母 3 个"),
    ({"type": "no_cjk"}, "hello", True, "没有汉字"),
    ({"type": "no_cjk"}, "hello你", False, "有汉字"),
    ({"type": "word_count_eq", "n": 2}, "hello world", True, "2 个词"),
    ({"type": "word_count_eq", "n": 2}, "hello there world", False, "3 个词"),
    # --- 解析类
    ({"type": "regex_fullmatch", "pattern": r"\d{4}"}, "2026", True, "四位数"),
    ({"type": "regex_fullmatch", "pattern": r"\d{4}"}, "2026年", False, "多了字"),
    ({"type": "numeric_eq", "n": 3.14}, "3.14", True, "纯数字相等"),
    ({"type": "numeric_eq", "n": 3.14}, "3.14 左右", False, "不是纯数字"),
    ({"type": "equals_ci", "text": "ok"}, "OK", True, "忽略大小写"),
    ({"type": "comma_fields", "n": 3}, "a, b, c", True, "3 个字段"),
    ({"type": "json_object"}, '{"a": 1}', True, "合法 JSON 对象"),
    ({"type": "json_object"}, "[1, 2]", False, "是数组不是对象"),
    ({"type": "json_keys", "keys": ["a"]}, '{"a": 1}', True, "键齐全"),
    ({"type": "json_array"}, "[1, 2]", True, "合法数组"),
    ({"type": "python_code"}, "def add(a, b):\n    return a + b", True, "真函数"),
    (
        {"type": "python_code"},
        "```python\ndef add(a, b):\n    return a + b\n``` לחלוט",
        True,
        "围栏+乱码尾仍算代码",
    ),
    ({"type": "python_code"}, 'print("hi")', True, "调用也算代码"),
    ({"type": "python_code"}, "import os", True, "import 也算代码"),
    # 关键回归：Python 允许中文标识符，「这不是代码」能 parse 成功
    ({"type": "python_code"}, "这不是代码", False, "中文裸标识符不是代码"),
    ({"type": "python_code"}, "以下是一道单项选择题", False, "复读题干不是代码"),
    ({"type": "python_code"}, "# 只是个注释", False, "只有注释不是代码"),
]:
    case(f"规则 {_name}", rules.check_rule(_rule, _text)[0], _want)

# 未实现的规则类型必须明确失败，不能静默通过
case("未知规则类型 → 失败", rules.check_rule({"type": "不存在的规则"}, "任意")[0], False)

# check_all：多条规则按「全部通过」聚合
case(
    "多条规则全通过",
    rules.check_all([{"type": "max_chars", "n": 5}, {"type": "no_cjk"}], "hello")[0],
    True,
)
case(
    "一条不过就整体不过",
    rules.check_all([{"type": "max_chars", "n": 2}, {"type": "no_cjk"}], "hello")[0],
    False,
)


# ============================================================ 置信区间

# 用线上那份实现，**不在测试里复刻公式** —— 复刻就等于又开一份实现，
# 两边一起错还测不出来（这个坑今天在 last_number 上踩过）
from metrics import wilson_ci  # noqa: E402

# 与标准 Wilson 区间对照（50/100 的已知值是 0.4038~0.5962）
case("Wilson 50/100 下界", wilson_ci(50, 100)[0], 0.4038)
case("Wilson 50/100 上界", wilson_ci(50, 100)[1], 0.5962)
case("Wilson 两端不越界（0/400 上界<1）", wilson_ci(0, 400)[1] < 1, True)
case("Wilson 两端不越界（10/10 下界>0）", wilson_ci(10, 10)[0] > 0, True)
case("Wilson 空样本不炸", wilson_ci(0, 0), (0.0, 0.0))


# ============================================================ 与真实题库对齐


def dataset_rule_problems() -> list[str]:
    """题库里出现的每一种规则，引擎都必须实现。

    漏实现一种规则不会报错 —— `check_rule` 的兜底会返回 False，
    于是那一整批题**永远判不过**，而且看起来像「模型不行」。
    所以这里拿题库里真实出现的规则各跑一次，确认引擎认得它们。
    """
    import json

    path = Path(__file__).resolve().parent.parent / "evals" / "ifollow-subset.jsonl"
    if not path.exists():
        return [f"找不到题库 {path}（这条检查跳过）"]

    samples: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            for rule in json.loads(line)["rules"]:
                samples.setdefault(rule["type"], rule)

    problems = []
    for kind, sample in sorted(samples.items()):
        _, detail = rules.check_rule(sample, "占位输出")
        if detail.startswith("未知规则"):
            problems.append(f"题库用了规则 `{kind}`，但 rules.py 没实现")
    print(f"  （题库覆盖 {len(samples)} 种规则类型）")
    return problems


# ============================================================ 跑起来


def additional_asserts() -> list[str]:
    """需要真的解析 Python 的断言，单独放这里。"""
    problems = []
    import ast

    body = extract_code(junk_tail)
    try:
        ast.parse(body)
    except SyntaxError as exc:
        problems.append(f"剥壳后仍是非法 Python：{exc}")
    return problems


def main() -> int:
    failed = 0
    for name, got, want in CASES:
        if got != want:
            failed += 1
            print(f"  ✗ {name}\n      得到 {got!r}\n      期望 {want!r}")
    for problem in additional_asserts():
        failed += 1
        print(f"  ✗ {problem}")
    for problem in dataset_rule_problems():
        failed += 1
        print(f"  ✗ {problem}")

    total = len(CASES)
    if failed:
        print(f"\n{failed}/{total} 条失败")
        return 1
    print(f"{total} 条全过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
