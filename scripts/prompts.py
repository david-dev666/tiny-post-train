"""生成相关的公共定义：卷面怎么拼、什么 token 该停。**纯文本，不 import torch。**

为什么单独成模块
----------------
换推理引擎（vLLM）之后，**同一个脚本要在两套环境里跑**：

- `tpt` 环境：unsloth + torch 2.12.1（训练用，HF 推理路径也走这套）
- `vllm` 环境：vLLM + torch 2.13.0（评测的批量推理）

而原先这些东西都在 `train_sft.py` 里，那个模块 `import unsloth` —— 在 vllm 环境里
**根本导入不了**。所以要有一份两边都能用的纯定义。

更要紧的是：**A/B 对比两个引擎时，喂进去的 prompt 必须逐字节一致**，
否则「换引擎」和「换卷面」两个变量一起动，分数差异根本没法归因。
放在这里就保证了只有一个来源。

（`train_sft.py` 仍然保留同名符号用于训练侧，但**数值必须与这里一致**；
两边一旦分叉，训练曲线和离线评测就不可比了。）
"""

from __future__ import annotations

# ------------------------------------------------------------------ 常量

LETTERS = ("A", "B", "C", "D")

# 固定使用的干净模板。**不要改回 tokenizer 自带的那个。**
#
# tokenizer 自带的是 Qwen3 **instruct** 版模板，它会把每个 assistant 回复渲染成：
#     <|im_start|>assistant\n<think>\n\n</think>\n\n{回答}<|im_end|>\n
# 而推理时 add_generation_prompt=True 只给到 `<|im_start|>assistant\n`，**不预填**
# 那个思考块，于是模型必须自己「生成」它 —— 实测生成出来的是两个异常字节，
# decode 成乱码，挂在每次输出的最前面。
#
# 注意：官方 instruct 模型要**用它自带模板**，换成这个反而是发错卷子。
# 判据见 eval.py 的 pick_chat_template。
CLEAN_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# 推理采样的固定问题。别改，改了就没法和之前步数对比了。
PROBE_PROMPTS = [
    "介绍一下你自己",
    "用一句话解释什么是 LoRA",
    "写一个 Python 函数，判断一个数是不是质数",
]

GSM8K_INSTRUCTION = (
    "请一步一步思考，并在最后一行用「#### 数字」的格式给出最终答案。\n\n问题：{question}"
)
CODE_INSTRUCTION = (
    "补全下面的 Python 函数。只输出完整的 Python 代码，"
    "不要任何解释，不要 Markdown 代码围栏。\n\n{prompt}"
)


# ------------------------------------------------------------------ 拼卷面


def render_clean(content: str, role: str = "user") -> str:
    """用项目内置的干净模板拼一轮对话（不用 tokenizer，纯字符串）。"""
    return (
        f"<|im_start|>{role}\n{content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def mmlu_prompt(item: dict, style: str = "chat") -> str:
    """MMLU 的卷面。两种口径的差别见 eval.py 的 `_mmlu_prompt`（那里有完整说明）。

    **注意**：`chat` 口径官方 instruct 模型要改用令牌器自带模板，
    那种情况调 `mmlu_content()` 拿到正文后自己 `apply_chat_template`。
    """
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(LETTERS, item["choices"])
    )
    if style == "plain":
        return mmlu_plain(item)
    return render_clean(mmlu_content(item, options))


def mmlu_content(item: dict, options: str | None = None) -> str:
    if options is None:
        options = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(LETTERS, item["choices"])
        )
    return (
        "以下是一道单项选择题，请直接回答正确选项的字母。\n\n"
        + item["question"] + "\n\n" + options + "\n\n答案："
    )


def mmlu_plain(item: dict) -> str:
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(LETTERS, item["choices"])
    )
    return f"Question: {item['question']}\n{options}\nAnswer:"


def gsm8k_prompt(question: str) -> str:
    return render_clean(GSM8K_INSTRUCTION.format(question=question))


def code_prompt(humaneval_prompt: str) -> str:
    return render_clean(CODE_INSTRUCTION.format(prompt=humaneval_prompt))


# ------------------------------------------------------------------ 停止符


def stop_token_ids(tokenizer) -> list[int]:
    """生成时要认的停止符。

    **必须带上 `<|im_end|>`。** chat 模板里每个回合都以它结尾，模型学会的就是
    用它结束回答；但它**不在** tokenizer / model 的 eos 里（那是 `<|endoftext|>`）。
    不带上，生成就不会在那里停 —— 模型「说完了还在硬说」，
    实测每条输出末尾多出一段乱码（`לחלוט` / `NdrFc` / `аци`），
    把指令遵循的判分整片带偏：答案都对，被尾巴判错。
    """
    vocab = tokenizer.get_vocab()
    ids = []
    for name in ("<|im_end|>", "<|endoftext|>"):
        if name in vocab:
            ids.append(vocab[name])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids
