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
#
# 「乱码尾」= 模型答完内容之后、回合结束之前吐出的那一个稀有 token。
#
# 它是**模型的真实输出**（token 序列里它就在停止符前面），不是解码 bug，
# 所以判分时**不偷偷删掉**，而是如实上报比率 —— 免得它悄悄污染
# 「字数上限」「结尾必须是 X」这类规则却无人察觉。
#
# 乱码字符分两类：
#
# 1. 其它文字系统（西里尔 / 亚美尼亚 / 希伯来 / 阿拉伯 / 泰 / 韩 / 假名）。
#    实测以希伯来语词 `לחלוט` 为主。已核验评测集（200+1319+40 条）的题面与
#    规则里**没有任何一条合法要求这些文字的输出**，所以当作乱码不会误报。
#
# 2. **U+FFFD 替换字符**。原先漏的就是这一类，导致乱码尾被系统性低估
#    （GSM8K 实报 65.3%，真实 92.3%；指令遵循实报 75.5%，真实 84.5%）。
#    成因是字节级 BPE：某个 token 的原始字节形如 `A0 A1 E5 8F 96`，
#    独立解码时前面两个无效字节变成 `��`、后面 `E5 8F 96` 正常解出 `取`，
#    于是**整块解成 `��取`**。它在文本层看着像「结尾是个正常汉字」，
#    因此 `[乱码字符]+$` 这种正则**匹配不到、剪不掉** —— 只能回到 token 层定位。
_OTHER_SCRIPT_CLASS = (
    r"\u0400-\u04FF"      # 西里尔
    r"\u0530-\u058F"      # 亚美尼亚
    r"\u0590-\u05FF"      # 希伯来
    r"\u0600-\u06FF"      # 阿拉伯
    r"\u0E00-\u0E7F"      # 泰文
    r"\uAC00-\uD7AF"      # 谚文
    r"\u3040-\u30FF"      # 平假名 / 片假名
)
OTHER_SCRIPT_RE = re.compile("[" + _OTHER_SCRIPT_CLASS + "]")
# 乱码字符全集：其它文字系统 + 替换字符
JUNK_CHAR_CLASS = _OTHER_SCRIPT_CLASS + r"\ufffd"
JUNK_CHAR_RE = re.compile("[" + JUNK_CHAR_CLASS + "]")
JUNK_TAIL_WINDOW = 16
JUNK_TAIL_RE = re.compile(r"[\s" + JUNK_CHAR_CLASS + r"]+$")


def has_junk_tail(text: str, window: int = JUNK_TAIL_WINDOW) -> bool:
    """末尾一小段里有没有乱码字符。"""
    return bool(JUNK_CHAR_RE.search((text or "")[-window:]))


def is_junk_piece(piece: str) -> bool:
    """一段文本（通常是**单个 token 单独解码**的结果）是不是乱码。"""
    return bool(JUNK_CHAR_RE.search(piece or ""))


def trailing_junk_len(tokenizer, ids, max_scan: int = 4) -> int:
    """末尾该剪掉几个 token。0 = 干净收尾。

    **这是乱码尾的唯一可靠判据** —— 文本层判不了，原因见上面 U+FFFD 那段：
    `��取` 解码后以正常汉字收尾，正则剪不掉它。造 DPO 的停止决策偏好数据
    也靠这个函数，所以**评测口径和训练靶心是同一个定义**，不会各量各的。

    `max_scan` 防止某天吐出一长串乱码时把答案正文也一起吃掉。

    ⚠️ 结果依赖 tokenizer 的切法（同一个字符可能被切成不同数量的 token），
    所以只在同一个 tokenizer 内部可比。
    """
    ids = [int(t) for t in ids]
    total = len(ids)

    def piece(index: int) -> str:
        return tokenizer.decode([ids[index]], skip_special_tokens=False)

    # 1) 从尾部跳过纯空白，定位最后一个「有内容」的 token
    last = total - 1
    while last >= 0 and not piece(last).strip():
        last -= 1
    if last < 0:
        return 0

    # 2) 从它往回吃掉连续的乱码 token
    index = last
    eaten = 0
    while index >= 0 and eaten < max_scan and is_junk_piece(piece(index)):
        eaten += 1
        index -= 1

    # 3) 没乱码就不动；有乱码则「乱码 + 它后面的空白」一起剪
    return 0 if eaten == 0 else total - index - 1


def trim_junk_tail(text: str) -> str:
    """去掉末尾的乱码与空白（**文本层，只在拿不到 token 时用**）。

    **只用于「看看不算乱码会是多少分」的对照，不用于线上判分。**
    线上判分看原始输出 —— 乱码是模型的真实输出，删掉等于掩盖缺陷。

    它剪不掉 `��取` 这类「以正常汉字收尾」的乱码，所以给出的对照分是**下界**。
    能拿到 token 时请用 `trailing_junk_len` 配 `tokenizer.decode(ids[:-n])`。
    """
    return JUNK_TAIL_RE.sub("", (text or "").rstrip())
