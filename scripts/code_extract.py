"""把模型生成的一段文本，变成「可以交给解释器执行」的 Python。

为什么单独成一个文件
--------------------
这段逻辑有两个调用方：

- `eval.py` —— 跑 HumanEval 时，拿到输出立刻抽取并执行
- `rescore_humaneval.py` —— harness 修好后，拿存档的 raw_output 离线重判

**两份实现一定会漂移。** 本项目已经在 `splitlines()` 上吃过一次亏：
本地校验脚本用 `for line in f`、线上用 `read_text().splitlines()`，
两边切法不同，校验通过而线上必崩。所以这里只保留一份实现。
本模块是纯文本处理，**不依赖 torch / unsloth**，可以单独跑、单独测。

踩过的坑（别再改回去）
---------------------
原来剥闭围栏写的是 `re.sub(r"\\s*```$", "", text)` —— `$` 要求围栏落在字符串**最末尾**。

实测 sft-4b-v2 的输出长这样：

    ```python
    def truncate_number(number: float) -> float:
        ...
        return number - int(number)
    ``` לחלוט

围栏后面还跟着 ` לחלוט` 这种**词表外的乱码 token**（模型收尾没收干净，训练采样里
也见过同样现象）。`$` 匹配不上，围栏就没被剥掉，拼出来的程序第一行是
`` ``` `` → SyntaxError。

164 题里 80 题这么挂的，把真实成绩 73.2% 硬压成 35.4%，
差点得出「SFT 把代码能力练废了」这个完全错误的结论。
"""

from __future__ import annotations

import re

# HumanEval 官方做法的截断标记：模型常会在函数后面多写一段调用示例
CODE_STOP_MARKERS = ("\nif __name__", "\nclass ", "\nprint(", "\n# ")


def strip_fence(text: str) -> str:
    """剥掉 Markdown 代码围栏。

    只给「解析类」规则用，**其它规则一律看原始输出**。曾经这里把所有规则都比对
    剥壳后的文本，结果 `not_contains ["```"]` 永远命中不了 —— 壳已经被剥掉了，
    「不要用代码块包裹」这条约束形同虚设（单测抓出来的）。

    闭围栏按「**从它出现的位置起全部丢弃**」处理，不做 `$` 锚定：
    围栏后面经常粘着乱码 token，锚定末尾就会漏。
    """
    text = (text or "").strip()
    if text.startswith("```"):
        # 开围栏可能带语言标签（```python / ```py）
        text = re.sub(r"^```[A-Za-z0-9_+-]*[ \t]*\n?", "", text)
    cut = text.find("```")
    if cut != -1:
        text = text[:cut]
    return text.strip()


def extract_code(text: str) -> str:
    """剥壳 + 截掉函数后面的示例代码，得到可 execute 的函数体。"""
    body = strip_fence(text)
    for marker in CODE_STOP_MARKERS:
        position = body.find(marker)
        if position != -1:
            body = body[:position]
    return body.rstrip()
