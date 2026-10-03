"""生成指令遵循评测集：evals/ifollow-subset.jsonl

设计原则
--------
1. **前 20 条是初版手写的，一字不改地保留。** 这样旧结果在「原 20 条」这个子集上
   仍然可比，历史数据不会作废。
2. 其余 180 条按约束类型批量补到每类 25 条，共 200 条。
3. 每条都必须能被 eval.py 的 checker **程序化判定**，没有主观分、没有参考答案。

为什么按类型分层
----------------
20 条时 ±1 条 = ±5pp，统计噪声比信号还大，得出「+10pp」这种结论根本不敢采信。
分层到每类 25 条后（噪声降到 ±2pp 量级）既能看总分，也能定位「哪一类没学会」。

为什么不用 LLM 生成题目
----------------------
题目本身是模板化的，确定性生成更可控、可复现、可 diff；LLM 生成会引入随机性，
而且评测集一旦变动就无法和历史对比。真正需要 LLM 的是**训练数据**，不是评测集。

用法：
    python scripts/make_ifollow_subset.py            # 写到 evals/ifollow-subset.jsonl
    python scripts/make_ifollow_subset.py --check    # 只校验现有评测集是否完整可执行
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_DIR / "evals" / "ifollow-subset.jsonl"

# ------------------------------------------------------------------ 初版手写 20 条（原样保留）

ORIGINAL = [
    {"id": "len-20", "category": "长度约束", "desc": "不超过 20 字（不计空白）",
     "prompt": "用不超过 20 个字介绍一下北京。", "rules": [{"type": "max_chars", "n": 20}]},
    {"id": "len-50-http", "category": "长度约束", "desc": "不超过 50 字（不计空白）",
     "prompt": "用 50 个字以内解释什么是 HTTP。", "rules": [{"type": "max_chars", "n": 50}]},
    {"id": "one-line", "category": "格式约束", "desc": "输出只有一行，不含换行",
     "prompt": "只回答一行，不要换行：1+1 等于几？", "rules": [{"type": "single_line"}]},
    {"id": "json-only", "category": "格式约束", "desc": "输出是合法 JSON 对象，且没有代码围栏",
     "prompt": "只输出一个 JSON 对象，包含 name 和 age 两个字段，值分别是 \"Tom\" 和 18。不要输出任何其他文字。",
     "rules": [{"type": "json_object"}, {"type": "not_contains", "words": ["```"]}]},
    {"id": "code-no-fence", "category": "格式约束", "desc": "输出是可解析的 Python 代码，且没有 ``` 围栏",
     "prompt": "只输出 Python 代码，不要用 ``` 代码块包裹，不要任何解释：写一个函数 add(a, b) 返回两数之和。",
     "rules": [{"type": "python_code"}, {"type": "not_contains", "words": ["```"]}]},
    {"id": "no-ai-prefix", "category": "禁忌词", "desc": "开头不得出现「作为一个 AI」「当然」等套话",
     "prompt": "回答时不要以「作为一个 AI」或「当然」开头。问题：什么是机器学习？",
     "rules": [{"type": "no_prefix", "prefixes": ["作为一个", "作为 AI", "作为AI", "当然", "首先"]}]},
    {"id": "no-apology", "category": "禁忌词", "desc": "不得出现道歉用语",
     "prompt": "不要道歉，不要说「抱歉」「对不起」，直接回答：天空为什么是蓝色的？",
     "rules": [{"type": "not_contains", "words": ["抱歉", "对不起", "遗憾"]}]},
    {"id": "no-markdown", "category": "禁忌词", "desc": "不得出现 Markdown 标记 # * ` >",
     "prompt": "不要使用任何 Markdown 标记（#、*、`、>），用纯文本介绍什么是 Python。",
     "rules": [{"type": "not_contains", "words": ["#", "*", "`", ">"]}]},
    {"id": "three-bullets", "category": "结构约束", "desc": "至少三行以「- 」开头",
     "prompt": "用三个要点列出学习编程的建议，每个要点单独一行，并且以「- 」开头。",
     "rules": [{"type": "line_prefix_count", "prefix": "- ", "n": 3}]},
    {"id": "numbered-three", "category": "结构约束", "desc": "1. / 2. / 3. 三个编号各自出现在行首",
     "prompt": "用 1. 2. 3. 的编号形式列出三种水果，每行一个。",
     "rules": [{"type": "line_starts_with_all", "prefixes": ["1.", "2.", "3."]}]},
    {"id": "starts-with-yes", "category": "结构约束", "desc": "第一行以「是」开头",
     "prompt": "先在第一行只回答「是」，然后再解释原因。问题：Python 是解释型语言吗？",
     "rules": [{"type": "starts_with", "text": "是"}]},
    {"id": "end-with-marker", "category": "结构约束", "desc": "以 END 结尾",
     "prompt": "回答结束时必须以「END」结尾，END 后面不要有任何字符。问题：什么是二叉树？",
     "rules": [{"type": "ends_with", "text": "END"}]},
    {"id": "exactly-three-lines", "category": "结构约束", "desc": "恰好三行非空文本",
     "prompt": "用恰好三行回答：你喜欢哪种水果？每行一句话，不要有空行。",
     "rules": [{"type": "line_count_eq", "n": 3}]},
    {"id": "keywords-three", "category": "内容约束", "desc": "同时包含 猫 / 狗 / 宠物",
     "prompt": "用一句话同时提到「猫」「狗」和「宠物」这三个词。",
     "rules": [{"type": "contains_all", "words": ["猫", "狗", "宠物"]}]},
    {"id": "chinese-only", "category": "语言约束", "desc": "不出现任何英文字母",
     "prompt": "只用中文回答，不要出现任何英文字母：介绍一下太阳系。",
     "rules": [{"type": "ascii_letters_max", "n": 0}]},
    {"id": "english-only", "category": "语言约束", "desc": "不出现任何中日韩汉字",
     "prompt": "Answer in English only. Do not use any Chinese characters: what is a variable?",
     "rules": [{"type": "no_cjk"}]},
    {"id": "exactly-5-words", "category": "语言约束", "desc": "恰好 5 个英文单词",
     "prompt": "Write exactly 5 words about the sun. Output only those 5 words.",
     "rules": [{"type": "word_count_eq", "n": 5}]},
    {"id": "numeric-9", "category": "数值约束", "desc": "输出就是数字 9，无单位无解释",
     "prompt": "只输出一个数字，不要单位、不要解释：3 的平方是多少？",
     "rules": [{"type": "numeric_eq", "n": 9}]},
    {"id": "roman-vii", "category": "数值约束", "desc": "输出就是 VII",
     "prompt": "用罗马数字表示 7，只输出罗马数字本身，不要任何其他内容。",
     "rules": [{"type": "equals_ci", "text": "VII"}]},
    {"id": "code-def-name", "category": "代码约束", "desc": "可解析的 Python 代码，且包含 def reverse_string",
     "prompt": "只输出 Python 代码，不要解释：写一个函数 reverse_string(s)，返回反转后的字符串。",
     "rules": [{"type": "python_code"}, {"type": "contains_all", "words": ["def reverse_string"]}]},
]


def item(identifier, category, desc, prompt, rules):
    return {"id": identifier, "category": category, "desc": desc, "prompt": prompt, "rules": rules}


# ------------------------------------------------------------------ 长度约束（补到 25）

def gen_length():
    specs = [
        (10, "中国的首都是哪里？", "只回答城市名，不要解释"),
        (15, "解释什么是 CPU。", "越短越好"),
        (15, "列出两种编程语言。", "只列名字"),
        (20, "说明水的化学式。", "只回答这一件事"),
        (20, "介绍一下杭州。", "不要展开"),
        (25, "解释什么是 API。", "不要举例"),
        (25, "说明为什么天空是蓝色的。", "一句话说完"),
        (30, "介绍 Python 这门语言。", "不要列优点"),
        (30, "解释什么是数据库。", "不要类比"),
        (30, "说明 1+1=2 的原因。", "不要展开数学定义"),
        (35, "解释什么是操作系统。", "不要列举例子"),
        (35, "介绍长江。", "不要写数据"),
        (40, "解释什么是算法。", "不要举代码例子"),
        (40, "说明什么是云计算。", "不要讲历史"),
        (40, "介绍太阳系。", "不要逐个行星展开"),
        (45, "解释什么是机器学习。", "不要分类展开"),
        (45, "说明什么是区块链。", "不要讲币"),
        (50, "解释什么是 LoRA。", "不要写公式"),
        (50, "介绍黄河。", "不要写长度数据"),
        (50, "说明什么是 HTTP 状态码。", "不要列举全部"),
        (60, "解释什么是神经网络。", "不要讲训练过程"),
        (60, "介绍咖啡。", "不要讲冲泡方法"),
        (60, "说明什么是光合作用。", "不要写化学式"),
    ]
    out = []
    for index, (limit, topic, extra) in enumerate(specs, 1):
        out.append(item(
            f"len-g{index:02d}",
            "长度约束",
            f"不超过 {limit} 字（不计空白）",
            f"用不超过 {limit} 个字回答：{topic}（{extra}）",
            [{"type": "max_chars", "n": limit}],
        ))
    return out


# ------------------------------------------------------------------ 数值约束（补到 25）

def gen_numeric():
    numbers = [
        ("只输出一个数字，不要单位、不要解释：7 乘 8 是多少？", 56),
        ("只输出一个数字，不要单位、不要解释：12 加 15 是多少？", 27),
        ("只输出一个数字，不要单位、不要解释：100 减 37 是多少？", 63),
        ("只输出一个数字，不要单位、不要解释：144 除以 12 是多少？", 12),
        ("只输出一个数字，不要单位、不要解释：2 的 10 次方是多少？", 1024),
        ("只输出一个数字，不要单位、不要解释：从 1 加到 10 的和是多少？", 55),
        ("只输出一个数字，不要单位、不要解释：9 的平方是多少？", 81),
        ("只输出一个数字，不要单位、不要解释：一个正方体有多少条棱？", 12),
        ("只输出一个数字，不要单位、不要解释：一年有多少个月？", 12),
        ("只输出一个数字，不要单位、不要解释：一周有多少小时？", 168),
        ("只输出一个数字，不要单位、不要解释：半小时是多少分钟？", 30),
        ("只输出一个数字，不要单位、不要解释：3 的阶乘是多少？", 6),
        ("只输出一个数字，不要单位、不要解释：三角形内角和是多少度？", 180),
        ("只输出一个数字，不要单位、不要解释：闰年的二月有多少天？", 29),
        ("只输出一个数字，不要单位、不要解释：一个季度有几个月？", 3),
        ("只输出一个数字，不要单位、不要解释：1000 除以 8 是多少？", 125),
        ("只输出一个数字，不要单位、不要解释：13 的平方是多少？", 169),
        ("只输出一个数字，不要单位、不要解释：17 加 28 是多少？", 45),
        ("只输出一个数字，不要单位、不要解释：81 的平方根是多少？", 9),
        ("只输出一个数字，不要单位、不要解释：365 减 100 是多少？", 265),
    ]
    out = []
    for index, (prompt, answer) in enumerate(numbers, 1):
        out.append(item(
            f"num-{index:02d}", "数值约束", f"输出就是数字 {answer}，无单位无解释",
            prompt, [{"type": "numeric_eq", "n": answer}],
        ))
    romans = [(9, "IX"), (40, "XL"), (2024, "MMXXIV")]
    for index, (value, roman) in enumerate(romans, 1):
        out.append(item(
            f"roman-{index:02d}", "数值约束", f"输出就是 {roman}",
            f"用罗马数字表示 {value}，只输出罗马数字本身，不要任何其他内容。",
            [{"type": "equals_ci", "text": roman}],
        ))
    return out


# ------------------------------------------------------------------ 代码约束（补到 25）

def gen_code():
    specs = [
        ("square", "square(x)", "返回 x 的平方"),
        ("is_even", "is_even(n)", "判断 n 是否为偶数，返回布尔值"),
        ("max_of_two", "max_of_two(a, b)", "返回两个数中较大的那个"),
        ("sum_list", "sum_list(nums)", "返回列表里所有数字的和"),
        ("count_vowels", "count_vowels(s)", "返回字符串里元音字母的个数"),
        ("factorial", "factorial(n)", "返回 n 的阶乘"),
        ("fibonacci", "fibonacci(n)", "返回斐波那契数列第 n 项"),
        ("to_upper", "to_upper(s)", "把字符串转成大写"),
        ("abs_value", "abs_value(x)", "返回 x 的绝对值"),
        ("min_of_three", "min_of_three(a, b, c)", "返回三个数中最小的那个"),
        ("repeat_string", "repeat_string(s, n)", "把字符串 s 重复 n 次"),
        ("list_length", "list_length(items)", "返回列表长度"),
        ("first_element", "first_element(items)", "返回列表第一个元素"),
        ("last_element", "last_element(items)", "返回列表最后一个元素"),
        ("contains_word", "contains_word(text, word)", "判断 text 里是否包含 word"),
        ("average", "average(nums)", "返回列表的平均值"),
        ("double_all", "double_all(nums)", "返回每个元素都乘 2 的新列表"),
        ("remove_duplicates", "remove_duplicates(items)", "去掉列表里的重复元素"),
        ("is_palindrome", "is_palindrome(s)", "判断字符串是否为回文"),
        ("celsius_to_fahrenheit", "celsius_to_fahrenheit(c)", "把摄氏度转成华氏度"),
        ("gcd", "gcd(a, b)", "返回两个数的最大公约数"),
        ("clamp", "clamp(x, lo, hi)", "把 x 限制在 lo 和 hi 之间"),
        ("flatten", "flatten(matrix)", "把二维列表摊平成一维"),
        ("char_count", "char_count(s)", "返回字符串的长度"),
    ]
    out = []
    for name, signature, purpose in specs:
        out.append(item(
            f"code-{name}", "代码约束", f"可解析的 Python 代码，且包含 def {name}",
            f"只输出 Python 代码，不要解释：写一个函数 {signature}，{purpose}。",
            [{"type": "python_code"}, {"type": "contains_all", "words": [f"def {name}"]}],
        ))
    return out


# ------------------------------------------------------------------ 格式约束（补到 25）

def gen_format():
    out = [
        item("fmt-json-city", "格式约束", "JSON 对象且含 city / population 两个键",
             "只输出一个 JSON 对象，包含 city 和 population 两个字段，值分别是 \"Tokyo\" 和 37000000。不要任何其他文字。",
             [{"type": "json_object"}, {"type": "json_keys", "keys": ["city", "population"]},
              {"type": "not_contains", "words": ["```"]}]),
        item("fmt-json-ok", "格式约束", "JSON 对象且含键 ok",
             "只输出一个 JSON 对象，包含一个布尔字段 ok，值为 true。不要任何其他文字。",
             [{"type": "json_object"}, {"type": "json_keys", "keys": ["ok"]},
              {"type": "not_contains", "words": ["```"]}]),
        item("fmt-json-array", "格式约束", "JSON 数组且长度为 3",
             "只输出一个 JSON 数组，按顺序包含 1、2、3 三个数字。不要任何其他文字。",
             [{"type": "json_array", "n": 3}, {"type": "not_contains", "words": ["```"]}]),
        item("fmt-json-colors", "格式约束", "JSON 数组且长度为 3",
             "只输出一个 JSON 数组，元素是 \"red\"、\"green\"、\"blue\" 三个字符串。不要任何其他文字。",
             [{"type": "json_array", "n": 3}, {"type": "not_contains", "words": ["```"]}]),
        item("fmt-csv-3", "格式约束", "英文逗号分隔且恰好 3 个字段",
             "只输出三个值，用英文逗号分隔，不要空格、不要任何其他文字：a、b、c。",
             [{"type": "comma_fields", "n": 3}]),
        item("fmt-csv-4", "格式约束", "英文逗号分隔且恰好 4 个字段",
             "只输出四个值，用英文逗号分隔，不要空格、不要任何其他文字：1、2、3、4。",
             [{"type": "comma_fields", "n": 4}]),
        item("fmt-csv-weekdays", "格式约束", "英文逗号分隔且恰好 5 个字段",
             "只输出周一到周五的英文缩写，用英文逗号分隔，不要空格、不要任何其他文字。",
             [{"type": "comma_fields", "n": 5}]),
        item("fmt-lower", "格式约束", "输出全为小写英文字母",
             "把 \"HELLO\" 转成小写，只输出结果本身，不要任何其他字符。",
             [{"type": "all_lower"}]),
        item("fmt-upper", "格式约束", "输出全为大写英文字母",
             "把 \"world\" 转成大写，只输出结果本身，不要任何其他字符。",
             [{"type": "all_upper"}]),
        item("fmt-upper-py", "格式约束", "输出全为大写英文字母",
             "把 \"python\" 转成大写，只输出结果本身，不要任何其他字符。",
             [{"type": "all_upper"}]),
        item("fmt-digits-4", "格式约束", "输出恰好四位数字",
             "只输出一个四位数，不要任何其他字符：1234。",
             [{"type": "regex_fullmatch", "pattern": r"\d{4}"}]),
        item("fmt-date", "格式约束", "输出 YYYY-MM-DD 格式",
             "只输出一个 YYYY-MM-DD 格式的日期，表示 2026 年 3 月 5 日，不要任何其他字符。",
             [{"type": "regex_fullmatch", "pattern": r"\d{4}-\d{2}-\d{2}"}]),
        item("fmt-time", "格式约束", "输出 HH:MM 格式",
             "用 HH:MM 的 24 小时格式表示下午三点整，只输出这个时间，不要任何其他字符。",
             [{"type": "regex_fullmatch", "pattern": r"\d{2}:\d{2}"}]),
        item("fmt-percent", "格式约束", "输出百分数形式",
             "把四分之一写成百分数，只输出结果本身（带 % 号），不要任何其他字符。",
             [{"type": "regex_fullmatch", "pattern": r"\d+(\.\d+)?%"}]),
        item("fmt-hex", "格式约束", "输出 0x 开头的小写十六进制",
             "把十进制 255 写成小写十六进制，带 0x 前缀，只输出结果本身。",
             [{"type": "regex_fullmatch", "pattern": r"0x[0-9a-f]+"}]),
        item("fmt-quoted", "格式约束", "输出被英文双引号包住的词",
             "只输出被一对英文双引号包住的单词 hello，不要任何其他字符。",
             [{"type": "regex_fullmatch", "pattern": r"\"[A-Za-z]+\""}]),
        item("fmt-filename", "格式约束", "输出小写文件名且以 .txt 结尾",
             "给一个名为 note 的文本文件取文件名，只输出文件名本身，小写，以 .txt 结尾。",
             [{"type": "regex_fullmatch", "pattern": r"[a-z0-9_]+\.txt"}]),
        item("fmt-one-word", "格式约束", "恰好一个英文单词",
             "只输出一个英文单词，表示「书」这个意思，不要任何其他内容。",
             [{"type": "word_count_eq", "n": 1}, {"type": "no_cjk"}]),
        item("fmt-single-line", "格式约束", "输出只有一行",
             "只回答一行，不要换行：一年有多少个季节？",
             [{"type": "single_line"}]),
        item("fmt-single-line-2", "格式约束", "输出只有一行",
             "只回答一行，不要换行：水在多少摄氏度结冰？",
             [{"type": "single_line"}]),
        item("fmt-code-no-fence", "格式约束", "Python 代码且无 ``` 围栏",
             "只输出 Python 代码，不要 ``` 围栏、不要解释：写一个函数 square(x) 返回平方。",
             [{"type": "python_code"}, {"type": "not_contains", "words": ["```"]}]),
        item("fmt-plain-one-line", "格式约束", "无 Markdown 标记且只有一行",
             "不要使用任何 Markdown 标记，只用一行纯文本回答：什么是 Git？",
             [{"type": "single_line"}, {"type": "not_contains", "words": ["#", "*", "`", ">"]}]),
    ]
    return out


# ------------------------------------------------------------------ 结构约束（补到 25）

def gen_structure():
    out = [
        item("struct-bullets-3", "结构约束", "至少 3 行以「- 」开头",
             "用三个要点说明如何保持健康，每个要点单独一行并以「- 」开头。",
             [{"type": "line_prefix_count", "prefix": "- ", "n": 3}]),
        item("struct-bullets-4", "结构约束", "至少 4 行以「- 」开头",
             "用四个要点说明为什么要写单元测试，每行以「- 」开头。",
             [{"type": "line_prefix_count", "prefix": "- ", "n": 4}]),
        item("struct-bullets-2", "结构约束", "至少 2 行以「* 」开头",
             "用两个要点说明咖啡的好处，每行以「* 」开头。",
             [{"type": "line_prefix_count", "prefix": "* ", "n": 2}]),
        item("struct-numbered-3", "结构约束", "1./2./3. 都在行首",
             "用 1. 2. 3. 的编号形式列出三种运动，每行一个。",
             [{"type": "line_starts_with_all", "prefixes": ["1.", "2.", "3."]}]),
        item("struct-numbered-4", "结构约束", "1.~4. 都在行首",
             "用 1. 2. 3. 4. 的编号形式列出四个季节，每行一个。",
             [{"type": "line_starts_with_all", "prefixes": ["1.", "2.", "3.", "4."]}]),
        item("struct-numdot-2", "结构约束", "1.2. 都在行首",
             "用 1. 2. 的编号形式列出两种颜色，每行一个。",
             [{"type": "line_starts_with_all", "prefixes": ["1.", "2."]}]),
        item("struct-lines-2", "结构约束", "恰好 2 行",
             "用恰好两行回答：你喜欢的季节和原因，每行一句话。",
             [{"type": "line_count_eq", "n": 2}]),
        item("struct-lines-4", "结构约束", "恰好 4 行",
             "用恰好四行回答：列出四个方向，每行一个，不要有空行。",
             [{"type": "line_count_eq", "n": 4}]),
        item("struct-lines-5", "结构约束", "恰好 5 行",
             "用恰好五行回答：列出五种颜色，每行一个，不要有空行。",
             [{"type": "line_count_eq", "n": 5}]),
        item("struct-start-answer", "结构约束", "第一行以「答案：」开头",
             "回答的第一行必须以「答案：」开头，然后另起一行解释：1 加 1 等于几？",
             [{"type": "starts_with", "text": "答案："}]),
        item("struct-start-conclusion", "结构约束", "第一行以「结论：」开头",
             "回答的第一行必须以「结论：」开头，然后再说明理由：Python 是解释型语言吗？",
             [{"type": "starts_with", "text": "结论："}]),
        item("struct-start-no", "结构约束", "第一行以「不是」开头",
             "回答的第一行必须以「不是」开头，然后解释：地球是方的吗？",
             [{"type": "starts_with", "text": "不是"}]),
        item("struct-end-thanks", "结构约束", "以「谢谢」结尾",
             "回答的最后必须以「谢谢」两个字结尾，后面不要有任何字符。问题：什么是函数？",
             [{"type": "ends_with", "text": "谢谢"}]),
        item("struct-end-done", "结构约束", "以 DONE 结尾",
             "回答的最后必须以「DONE」结尾，后面不要有任何字符。问题：什么是循环？",
             [{"type": "ends_with", "text": "DONE"}]),
        item("struct-end-bracket", "结构约束", "以 ] 结尾",
             "回答的最后必须以英文右方括号「]」结尾，后面不要有任何字符。问题：什么是数组？",
             [{"type": "ends_with", "text": "]"}]),
        item("struct-quote-prefix", "结构约束", "至少 2 行以「> 」开头",
             "用引述格式写两句关于阅读的句子，每行以「> 」开头。",
             [{"type": "line_prefix_count", "prefix": "> ", "n": 2}]),
        item("struct-marker-3", "结构约束", "至少 3 行以「### 」开头",
             "列出三个小标题，每行以「### 」开头。",
             [{"type": "line_prefix_count", "prefix": "### ", "n": 3}]),
        item("struct-space-bullet", "结构约束", "至少 3 行以「• 」开头",
             "用三个圆点要点列出三种水果，每行以「• 」开头。",
             [{"type": "line_prefix_count", "prefix": "• ", "n": 3}]),
        item("struct-lines-3-b", "结构约束", "恰好 3 行",
             "用恰好三行回答：什么时候该用列表、什么时候该用字典、什么时候该用集合，每行一句。",
             [{"type": "line_count_eq", "n": 3}]),
        item("struct-start-first", "结构约束", "第一行以「首先」开头",
             "回答的第一行必须以「首先」两个字开头，然后继续说明如何学习编程。",
             [{"type": "starts_with", "text": "首先"}]),
    ]
    return out


# ------------------------------------------------------------------ 语言约束（补到 25）

def gen_language():
    out = []
    chinese_topics = ["介绍一下机器学习", "解释什么是操作系统", "介绍中国的四大发明",
                      "说明为什么要喝水", "介绍一下大熊猫"]
    for index, topic in enumerate(chinese_topics, 1):
        out.append(item(
            f"lang-zh-{index}", "语言约束", "不出现任何英文字母",
            f"只用中文回答，不要出现任何英文字母：{topic}。",
            [{"type": "ascii_letters_max", "n": 0}],
        ))
    english_topics = ["what is a function", "what is a database", "what is an API",
                      "what is a compiler", "what is an operating system"]
    for index, topic in enumerate(english_topics, 1):
        out.append(item(
            f"lang-en-{index}", "语言约束", "不出现任何中日韩汉字",
            f"Answer in English only. Do not use any Chinese characters: {topic}.",
            [{"type": "no_cjk"}],
        ))
    word_specs = [(3, "the moon"), (5, "a cat"), (7, "winter"), (10, "the ocean"), (4, "coffee")]
    for index, (count, topic) in enumerate(word_specs, 1):
        out.append(item(
            f"lang-wc-{index}", "语言约束", f"恰好 {count} 个英文单词",
            f"Write exactly {count} words about {topic}. Output only those words.",
            [{"type": "word_count_eq", "n": count}],
        ))
    upper_words = ["hello world", "machine learning", "database"]
    for index, text in enumerate(upper_words, 1):
        out.append(item(
            f"lang-upper-{index}", "语言约束", "输出全为大写英文字母",
            f"把 \"{text}\" 全部转成大写，只输出结果本身，不要任何其他字符。",
            [{"type": "all_upper"}],
        ))
    lower_words = ["PYTHON", "DATA"]
    for index, text in enumerate(lower_words, 1):
        out.append(item(
            f"lang-lower-{index}", "语言约束", "输出全为小写英文字母",
            f"把 \"{text}\" 全部转成小写，只输出结果本身，不要任何其他字符。",
            [{"type": "all_lower"}],
        ))
    char_specs = [(10, "只用十个字说明什么是水。"), (20, "用恰好二十个字介绍春天。")]
    for index, (count, prompt) in enumerate(char_specs, 1):
        out.append(item(
            f"lang-cc-{index}", "语言约束", f"恰好 {count} 字（不计空白）",
            prompt, [{"type": "char_count_eq", "n": count}],
        ))
    return out


# ------------------------------------------------------------------ 内容约束（补到 25）

def gen_content():
    triples = [
        ["苹果", "香蕉", "水果"], ["太阳", "月亮", "星球"], ["跑步", "游泳", "运动"],
        ["书", "知识", "学习"], ["咖啡", "茶叶", "饮品"], ["钢琴", "吉他", "乐器"],
        ["桌子", "椅子", "家具"], ["蜜蜂", "蝴蝶", "昆虫"], ["火车", "飞机", "交通"],
        ["雪山", "大海", "风景"], ["面包", "米饭", "主食"], ["医生", "教师", "职业"],
        ["春天", "秋天", "季节"], ["红色", "蓝色", "颜色"], ["鼠标", "键盘", "外设"],
        ["路由器", "交换机", "网络"], ["函数", "变量", "编程"], ["数据库", "缓存", "存储"],
        ["密码", "指纹", "安全"],
    ]
    out = []
    for index, words in enumerate(triples, 1):
        joined = "」「".join(words)
        out.append(item(
            f"content-{index:02d}", "内容约束", "同时包含 " + " / ".join(words),
            f"用一句话同时提到「{joined}」这三个词。",
            [{"type": "contains_all", "words": words}],
        ))
    pairs = [
        ["米饭", "筷子"], ["键盘", "打字"], ["地图", "导航"], ["电池", "充电"], ["雨伞", "下雨"],
    ]
    for index, words in enumerate(pairs, 1):
        joined = "」「".join(words)
        out.append(item(
            f"content-p{index}", "内容约束", "同时包含 " + " / ".join(words),
            f"用一句话同时提到「{joined}」这两个词。",
            [{"type": "contains_all", "words": words}],
        ))
    return out


# ------------------------------------------------------------------ 禁忌词（补到 25）

def gen_negative():
    bans = [
        (["很抱歉", "非常抱歉"], "不要道歉，直接回答问题：什么是素数？"),
        (["作为一个人工智能", "作为AI助手"], "回答时不要出现「作为一个人工智能」这类说法。问题：什么是变量？"),
        (["首先", "其次"], "回答中不要使用「首先」「其次」这样的序词。问题：如何学习编程？"),
        (["总之", "综上所述"], "回答中不要使用「总之」「综上所述」。问题：什么是递归？"),
        (["值得注意的是"], "回答中不要使用「值得注意的是」。问题：什么是内存？"),
        (["希望这对你有帮助", "希望对你有帮助"], "回答中不要出现「希望这对你有帮助」这类客套话。问题：什么是 Git 分支？"),
        (["如有疑问", "如有问题"], "回答中不要出现「如有疑问」这类结尾。问题：什么是索引？"),
        (["毋庸置疑", "显而易见"], "回答中不要使用「毋庸置疑」「显而易见」这类词。问题：为什么需要测试？"),
        (["众所周知"], "回答中不要使用「众所周知」。问题：什么是熵？"),
        (["个人认为", "我觉得"], "回答中不要出现「个人认为」「我觉得」。问题：Python 和 Java 有什么区别？"),
        (["总而言之"], "回答中不要使用「总而言之」。问题：什么是 HTTP 请求？"),
        (["让我们", "我们来"], "回答中不要使用「让我们」「我们来」这类说法。问题：什么是排序算法？"),
        (["值得注意的是", "需要注意的是"], "回答中不要使用「值得注意的是」「需要注意的是」。问题：什么是死锁？"),
        (["你可以", "你可以试试"], "回答中不要出现「你可以试试」这类建议。问题：什么是缓存？"),
        (["祝你好运", "祝你"], "回答中不要出现祝福语。问题：什么是编译原理？"),
        (["嗯", "呃"], "回答中不要出现「嗯」「呃」这类语气词。问题：什么是数据结构？"),
        (["?", "？"], "回答结尾不要用问号，直接陈述。问题：什么是机器学习？"),
        (["!", "！"], "回答中不要使用感叹号。问题：什么是深度学习？"),
    ]
    out = []
    for index, (words, prompt) in enumerate(bans, 1):
        out.append(item(
            f"neg-{index:02d}", "禁忌词", "不得出现 " + " / ".join(words),
            prompt, [{"type": "not_contains", "words": words}],
        ))
    prefixes = [
        (["以下是", "下面"], "回答时不要以「以下是」或「下面」开头。问题：什么是数组？"),
        (["好的", "当然"], "回答时不要以「好的」或「当然」开头。问题：什么是队列？"),
        (["我"], "回答时不要以「我」字开头。问题：什么是栈？"),
        (["这是一个"], "回答时不要以「这是一个」开头。问题：什么是哈希表？"),
    ]
    for index, (words, prompt) in enumerate(prefixes, 1):
        out.append(item(
            f"neg-p{index}", "禁忌词", "开头不得出现 " + " / ".join(words),
            prompt, [{"type": "no_prefix", "prefixes": words}],
        ))
    return out


GENERATORS = {
    "长度约束": gen_length,
    "数值约束": gen_numeric,
    "代码约束": gen_code,
    "格式约束": gen_format,
    "结构约束": gen_structure,
    "语言约束": gen_language,
    "内容约束": gen_content,
    "禁忌词": gen_negative,
}


def build() -> list[dict]:
    items = list(ORIGINAL)
    for category, generator in GENERATORS.items():
        items.extend(generator())
    return items


def write(items: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, ensure_ascii=False) for item in items]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(items: list[dict]) -> None:
    from collections import Counter

    counts = Counter(item["category"] for item in items)
    print(f"共 {len(items)} 条")
    for category in sorted(counts):
        print(f"  {category:<6} {counts[category]:>3} 条")
    ids = [item["id"] for item in items]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    if duplicates:
        sys.exit(f"!! id 重复：{duplicates}")
    print(f"id 唯一性：通过（{len(set(ids))} 个不同 id）")


def check(out_path: Path) -> None:
    """校验现有评测集：条数、id 唯一、规则类型是否都能被 rules.py 执行。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # ⚠️ 按 `\n` 切，**不用 splitlines()** —— 它按 Unicode 定义的所有行边界切
    # （`\x85` / `\u2028` …），会把一行 JSON 从字符串中间切开。
    # 这正是「校验脚本和线上切法不同 → 校验通过而线上必崩」的坑，别再犯。
    items = [json.loads(l) for l in out_path.read_text(encoding="utf-8").split("\n") if l.strip()]
    report(items)
    try:
        # 规则引擎在纯模块 rules.py 里 —— 校验不需要拉起 torch/unsloth
        import rules as rule_engine

        bad = []
        for entry in items:
            for rule in entry["rules"]:
                try:
                    rule_engine.check_all([rule], "占位输出")
                except Exception as exc:  # noqa: BLE001
                    bad.append((entry["id"], rule["type"], repr(exc)))
        print("规则可执行性：" + ("全部通过" if not bad else f"!! 有问题 {bad}"))
        if bad:
            sys.exit(1)
    except ImportError as exc:
        print(f"（跳过规则校验：{exc}）")


def main():
    parser = argparse.ArgumentParser(description="生成指令遵循评测集")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--check", action="store_true", help="不生成，只校验现有文件")
    args = parser.parse_args()

    out_path = Path(args.out)
    if args.check:
        check(out_path)
        return

    items = build()
    report(items)
    write(items, out_path)
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()
