"""指令遵循的硬约束判定引擎。**纯文本，不依赖 torch / unsloth。**

为什么单独成模块
----------------
`eval.py` 线上判分、`rescore.py` 离线重判、`tests/test_scoring.py` 单测三处都要用。
放在 `eval.py` 里的话，单测就得 import 整个 eval（会拉起 unsloth），
在 Mac 上跑不了 —— 而「跑不了的测试等于没有测试」。
本项目的规矩是：**判分逻辑一律抽成可单测的纯模块**（另有 code_extract / answer_extract）。

24 种规则类型，全部来自 `evals/ifollow-subset.jsonl` 的 `rules` 字段：

    contains_all / not_contains / python_code / max_chars / numeric_eq /
    line_prefix_count / no_cjk / word_count_eq / regex_fullmatch / ascii_letters_max /
    no_prefix / starts_with / line_count_eq / all_upper / single_line /
    line_starts_with_all / ends_with / equals_ci / json_object / comma_fields /
    all_lower / json_keys / json_array / char_count_eq

**判分口径的一条原则**：多数规则看**原始输出**（这样「不要用代码围栏」这类约束才有效），
只有「解析类」规则（json_* / python_code）先剥掉围栏再解析 —— 目的是把
「内容对但格式错」和「内容就不对」区分开。
"""

from __future__ import annotations

import ast
import json
import re

from code_extract import strip_fence as _strip_fence  # 剥壳逻辑与 HumanEval 共用一份
from answer_extract import NUMBER_TOKEN  # 数字定义与 GSM8K 共用一份

# python_code 判定里「算得上代码」的节点类型。
# 为什么不能只看 ast.parse 是否抛异常：Python 的标识符允许 CJK，
# 所以「这不是代码」会被解析成一个裸 Name 表达式，parse 是成功的。
CODE_NODE_TYPES = (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.Import, ast.ImportFrom,
    ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.For, ast.While, ast.If, ast.With, ast.Try,
    ast.Return, ast.Call, ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp,
)


# ------------------------------------------------------------------ 指令遵循判分

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
# 复用 answer_extract 的数字定义（带千分位），别在这里再写一份
NUMBER_RE = re.compile(r"(?:" + NUMBER_TOKEN + r")")
TRAILING_PUNCT = "。．.!！?？,，;；:： \t\n"


def _lines(text: str) -> list[str]:
    """按 `\\n` 切行，去掉空行。

    **不要用 `splitlines()`**：它按 Unicode 定义的**所有**行边界切，包括
    `\\x0b` / `\\x0c` / `\\x85`(NEL) / `\\u2028` / `\\u2029`。而「一行」在本项目里
    就是 `\\n` 分隔的东西 —— `single_line` / `line_count_eq` / `starts_with` /
    `line_prefix_count` / `line_starts_with_all` 五种规则都走这里，被切开就会误判。

    实测 200 条指令遵循输出里 **0 条** 含这类字符，所以这是个**潜伏** bug
    （和 JSONL 那次不同，那次真的炸了），但规则里「只输出一行」「恰好 3 行」
    会因为一个看不见的字符判错，不能留。
    """
    return [line for line in (text or "").split("\n") if line.strip()]


def check_rule(rule: dict, raw: str) -> tuple[bool, str]:
    """判定一条硬约束。返回 (是否通过, 说明)。"""
    kind = rule["type"]
    text = (raw or "").strip()

    if kind == "max_chars":
        n = len(re.sub(r"\s", "", text))
        return n <= rule["n"], f"{n}/{rule['n']} 字"

    if kind == "char_count_eq":
        n = len(re.sub(r"\s", "", text))
        return n == rule["n"], f"{n}/{rule['n']} 字"

    if kind == "single_line":
        return "\n" not in text, "1 行" if "\n" not in text else f"{len(_lines(text))} 行"

    if kind == "line_count_eq":
        n = len(_lines(text))
        return n == rule["n"], f"{n} 行"

    if kind == "json_object":
        try:
            parsed = json.loads(_strip_fence(text))
        except (json.JSONDecodeError, TypeError):
            return False, "不是合法 JSON"
        return isinstance(parsed, dict), "是 JSON 对象" if isinstance(parsed, dict) else "JSON 但不是对象"

    if kind == "json_keys":
        try:
            parsed = json.loads(_strip_fence(text))
        except (json.JSONDecodeError, TypeError):
            return False, "不是合法 JSON"
        if not isinstance(parsed, dict):
            return False, "不是 JSON 对象"
        miss = [k for k in rule["keys"] if k not in parsed]
        return not miss, "键齐全" if not miss else f"缺键 {miss}"

    if kind == "json_array":
        try:
            parsed = json.loads(_strip_fence(text))
        except (json.JSONDecodeError, TypeError):
            return False, "不是合法 JSON"
        if not isinstance(parsed, list):
            return False, "不是 JSON 数组"
        want = rule.get("n")
        if want is not None and len(parsed) != want:
            return False, f"数组长度 {len(parsed)}，期望 {want}"
        return True, f"数组长度 {len(parsed)}"

    if kind == "comma_fields":
        parts = [p for p in text.split(",") if p.strip()]
        return len(parts) == rule["n"], f"{len(parts)}/{rule['n']} 个字段"

    if kind == "all_upper":
        letters = ASCII_LETTER_RE.findall(text)
        ok = bool(letters) and not re.search(r"[a-z]", text)
        return ok, f"{len(letters)} 个字母全大写" if ok else "含小写字母或没有字母"

    if kind == "all_lower":
        letters = ASCII_LETTER_RE.findall(text)
        ok = bool(letters) and not re.search(r"[A-Z]", text)
        return ok, f"{len(letters)} 个字母全小写" if ok else "含大写字母或没有字母"

    if kind == "regex_fullmatch":
        ok = re.fullmatch(rule["pattern"], text) is not None
        return ok, f"匹配 {rule['pattern']}" if ok else f"不匹配 {rule['pattern']}：{text[:24]!r}"

    if kind == "python_code":
        # 内部容忍围栏，好跟 not_contains ["```"] 搭配出「内容对但格式错」这种区分
        try:
            tree = ast.parse(_strip_fence(text))
        except (SyntaxError, ValueError):
            return False, "不是合法 Python"
        # **只判断「能 parse」不够**：Python 允许中文标识符，所以「这不是代码」
        # 也能 parse 成功（ast 解析成一个裸 Name 表达式）。必须要求真的出现
        # 像代码的节点 —— 本规则的题面全是「写一个函数 xxx(...)」。
        # 判定集合放宽到 import / 赋值 / 调用等，免得把合法答案判错。
        if not any(isinstance(node, CODE_NODE_TYPES) for node in ast.walk(tree)):
            return False, "能解析但没有代码结构（像纯文本）"
        return True, "可解析"

    if kind == "not_contains":
        hit = [w for w in rule["words"] if w in text]
        return not hit, "干净" if not hit else f"命中 {hit}"

    if kind == "contains_all":
        miss = [w for w in rule["words"] if w not in text]
        return not miss, "齐全" if not miss else f"缺 {miss}"

    if kind == "no_prefix":
        head = text.lstrip()
        hit = [p for p in rule["prefixes"] if head.startswith(p)]
        return not hit, "开头干净" if not hit else f"以 {hit} 开头"

    if kind == "starts_with":
        head = _lines(text)[0].lstrip() if _lines(text) else ""
        return head.startswith(rule["text"]), head[:12]

    if kind == "ends_with":
        return text.rstrip().endswith(rule["text"]), text[-12:]

    if kind == "line_prefix_count":
        n = sum(1 for line in _lines(text) if line.lstrip().startswith(rule["prefix"]))
        # **恰好** n 行，不是「至少」n 行。原先是 `n >= rule["n"]`，而 7 道题的题面
        # 全是「用三个/四个/两个要点…」—— 给 5 个要点照样判通过。
        # 这是对啰嗦模型的放水（sft-4b-v2 正是啰嗦的那个），所以改成相等。
        # 实测当前 0/7 条触发，属潜伏 bug；改完分数不变，但判定口径对了。
        return n == rule["n"], f"{n}/{rule['n']} 行"

    if kind == "line_starts_with_all":
        miss = [
            p for p in rule["prefixes"]
            if not any(line.lstrip().startswith(p) for line in _lines(text))
        ]
        return not miss, "齐全" if not miss else f"缺 {miss}"

    if kind == "ascii_letters_max":
        n = len(ASCII_LETTER_RE.findall(text))
        return n <= rule["n"], f"{n}/{rule['n']} 个英文字母"

    if kind == "no_cjk":
        n = len(CJK_RE.findall(text))
        return n == 0, f"{n} 个汉字"

    if kind == "word_count_eq":
        n = len(text.split())
        return n == rule["n"], f"{n}/{rule['n']} 词"

    if kind == "numeric_eq":
        body = text.rstrip(TRAILING_PUNCT)
        if not NUMBER_RE.fullmatch(body):
            return False, f"不是纯数字：{body[:16]!r}"
        return abs(float(body) - rule["n"]) < 1e-6, body

    if kind == "equals_ci":
        body = text.rstrip(TRAILING_PUNCT)
        return body.upper() == rule["text"].upper(), body[:16]

    return False, f"未知规则 {kind}"


def check_all(rules: list[dict], raw: str) -> tuple[bool, str]:
    parts = []
    passed = True
    for rule in rules:
        ok, why = check_rule(rule, raw)
        passed = passed and ok
        parts.append(("✓ " if ok else "✗ ") + why)
    return passed, " | ".join(parts)
