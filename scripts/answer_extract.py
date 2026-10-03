"""从模型输出里抽出「可判分的答案」。纯文本处理，**不依赖 torch / unsloth**。

为什么要单独成一个模块
----------------------
三处要用同一份实现，各写一遍必然漂移：

- `eval.py` —— 线上判分
- `rescore.py` —— 离线重判（拿存档重新抽，不重跑模型）
- `tests/test_scoring.py` —— 单元测试

本项目已经因为「两处实现不一致」栽过两次，不再重复：

1. 本地校验用 `for line in f`、线上用 `read_text().splitlines()`，切法不同，
   校验通过而线上必崩（`\\x85` 被当成换行，JSONL 被从字符串中间切开）
2. `_strip_fence` 的闭围栏正则锚定 `$`，模型输出末尾带乱码时围栏剥不掉，
   把完整代码判成语法错误，真实 73.2% 被压成 35.4%

每条抽取函数都返回 **(值, 用了哪条规则)**。规则名会统计进结果 json ——
「有多少题是靠兜底规则蒙出来的」是判断一份分数可不可信的关键信息。
"""

from __future__ import annotations

import re

# ------------------------------------------------------------------ 数字

# ⚠️ 千分位必须写进正则。原来写的是 `-?\d+(?:\.\d+)?`，于是：
#     "$80,000"      → ['80', '000']       取最后一个 = 0     ✗
#     "答案是 1,250"  → ['1', '250']        取最后一个 = 250   ✗
# 这类题在 GSM8K 里不少（钱、人数、距离），答案≥1000 的题正确率被系统性压低。
NUMBER_TOKEN = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?|\d*\.\d+"
NUMBER_RE = re.compile(NUMBER_TOKEN)


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def all_numbers(text: str) -> list[float]:
    out = []
    for token in NUMBER_RE.findall(text or ""):
        value = _to_float(token)
        if value is not None:
            out.append(value)
    return out


def last_number(text: str) -> float | None:
    """GSM8K 判分用的答案：全文最后一个数字。

    **不要再改成「优先取 `####` 后面的数字」。** 实测过 4 种候选规则
    （最后一个数字 / 最后一行 / 结尾 200 字内的 #### / 全文最后一个 ####），
    在线上的 400 题上跑出来：

        规则                      base-4b   sft-v2  instruct
        最后一个数字（现状）           55.8%    83.0%    51.7%
        最后一行                     53.5%    82.8%    38.5%
        结尾 ####                    55.8%    83.0%    51.7%
        全文最后一个 ####              55.8%    83.0%    51.7%

    原因很具体：**基座把 `####` 当章节编号用**（`#### 1. 计算每天的总产蛋量`），
    于是「取最后一个 #### 后面的数字」拿到的是**步骤号**，不是答案 ——
    基座会从 55.8% 掉到 45.2%。

    提示里确实要求「最后一行用 #### 数字」，但那只是**格式要求**；
    判分不该依赖模型遵守格式（这正是 MMLU 那个假 0 分的教训）。
    """
    values = all_numbers(text)
    return values[-1] if values else None


# 提示要求「在最后一行用「#### 数字」的格式给出最终答案」。
# 实测两个模型都把「数字」当成了**字面量**，写成：
#     base    : `#### 数字\n18`
#     sft     : `#### 数字: 18. (Janet makes $18 ...)`
# 所以判据不能是「最后一行严格等于 `#### <数>`」——那样两个模型都会被判 0
# （实测 sft 因此显示 0/400，看起来很严重，其实只是我的判据写窄了）。
# 改成：**末尾一小段里出现了 `####` 标记**。这就是「有没有用要求的标记作答」。
# 末尾窗口内「`####` 后面 40 字内跟着数字」才算用了标记作答。
# 只找 `####` 不够 —— 基座把 `####` 当章节编号用（`#### 1. 计算…`），
# 那种也会命中。要求后面跟着数字能把「标题」和「给答案」区分开。
# 用 DOTALL：数字可能写在标记的**下一行**（基座就是 `#### 数字\n18` 这种）
HASH_TAIL_RE = re.compile(r"#{2,4}.{0,40}?(?:" + NUMBER_TOKEN + r")", re.DOTALL)
HASH_TAIL_WINDOW = 120


def follows_hash_format(text: str) -> bool:
    """模型有没有按提示要求用 `####` 标记给出最终答案。

    **只作为格式遵从度的诊断指标，不参与判分。** 它量的是「模型有多听话」；
    而 last_number 保证「哪怕它不听话，只要写了数字就能判对」——
    判分不该依赖模型遵守格式（这是 MMLU 那个假 0 分的教训）。
    """
    return bool(HASH_TAIL_RE.search(trim_junk_tail(text)[-HASH_TAIL_WINDOW:]))


# ------------------------------------------------------------------ MMLU

LETTER_RE = re.compile(r"[ABCD]")
_FULLWIDTH = str.maketrans("ＡＢＣＤａｂｃｄ", "ABCDABCD")
# 输出**整体**就是一个字母（最理想）。加粗记号 `**B**` 也算 —— 那也是「只回答了字母」
ONLY_LETTER_RE = re.compile(r"^\s*\**\s*([ABCD])\s*\**\s*[.。)）、,，:：]?\s*$")
# 输出**以**字母开头
LEADING_LETTER_RE = re.compile(r"^\s*\**\s*([ABCD])(?![A-Za-z0-9_])")
# 带标记：「答案是 B」
MARKED_LETTER_RE = re.compile(
    r"(?:答案是|答案为|答案|Answer|answer|ANSWER)\s*[:：=]?\s*\**\s*([ABCD])(?![A-Za-z0-9_])"
)


def mmlu_letter(text: str, letters: str = "ABCD") -> tuple[int | None, str]:
    """抽选项字母，返回 (下标, 规则名)。

    同样是「标记优先」。原来只做 `LETTER_RE.search()` 取**第一个** A-D ——
    模型一旦复读题干，第一个字母会是选项标号 `A.` 而不是答案，
    于是「出字母 20/20 但只有 10% 正确」这种假分数就出来了。
    """
    raw = (text or "").strip()
    if not raw:
        return None, "none"
    body = raw.translate(_FULLWIDTH)
    index = {letter: n for n, letter in enumerate(letters)}

    for pattern, rule in (
        (ONLY_LETTER_RE, "only"),
        (LEADING_LETTER_RE, "leading"),
    ):
        found = pattern.match(body)
        if found and found.group(1) in index:
            return index[found.group(1)], rule

    found = MARKED_LETTER_RE.search(body)
    if found and found.group(1) in index:
        return index[found.group(1)], "marked"

    found = LETTER_RE.search(body)
    if found and found.group(0) in index:
        # 兜底：可能抓到的是题干里的选项标号，可信度最低，单独计数
        return index[found.group(0)], "fallback"
    return None, "none"


# ------------------------------------------------------------------ 乱码尾

# 其它文字系统的字符（西里尔 / 希伯来 / 阿拉伯 / 泰文…）。
# 我们的 SFT 模型会把稀有 token 吐在回合结束之前 —— 实测 ifollow 140/200 条
# 以 `לחלוט` 结尾、human eval 84/164 条、gsm8k 174/400 条，而 base 和 instruct
# 一条都没有。这是**模型的真实输出**（token 序列里它就在停止符前面），
# 不是解码 bug，所以判分时**不偷偷删掉**，而是如实上报比率，
# 免得它悄悄污染「字数上限」「结尾必须是 X」这类规则却无人察觉。
OTHER_SCRIPT_RE = re.compile(
    r"[\u0400-\u04FF\u0530-\u058F\u0590-\u05FF\u0600-\u06FF"
    r"\u0E00-\u0E7F\uAC00-\uD7AF\u3040-\u30FF]"
)


def has_other_script(text: str, window: int = 16) -> bool:
    """末尾一小段里有没有其它文字系统的字符。"""
    return bool(OTHER_SCRIPT_RE.search((text or "")[-window:]))


JUNK_TAIL_RE = re.compile(
    r"[\s" + OTHER_SCRIPT_RE.pattern[1:-1] + r"]+$"
)


def trim_junk_tail(text: str) -> str:
    """去掉末尾的乱码 token 与空白。

    **只用于「看看不算乱码会是多少分」的对照，不用于线上判分。**
    线上判分看原始输出 —— 乱码是模型的真实输出，删掉等于掩盖缺陷。
    但把「含乱码」和「不含乱码」两个数都摆出来，读者才知道这 1~3 个 token
    到底值多少分（实测 sft-4b-v2 的指令遵循：61.0% → 77.0%，差 16pp）。
    """
    return JUNK_TAIL_RE.sub("", (text or "").rstrip())
