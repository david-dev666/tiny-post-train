"""客观指标：置信区间 + 退化信号。**纯数学，不依赖 torch / unsloth。**

为什么单独成模块
----------------
三处要用同一份：

- `eval.py` —— 每个评测项的准确率
- `judge_openqa.py` —— 裁判胜率（它原本就是 `eval_module._wilson_ci` 复用，没有第二份实现）
- `tests/test_scoring.py` —— 数值校验

放在 `eval.py` 里的话单测就得拉起 unsloth，在 Mac 上跑不了。
**跑不了的测试等于没有测试** —— 而置信区间算错是那种不会报错、只会让人
把 ±9pp 的数当成精确值的错。

顺带记一笔：`_wilson_ci` 本来就没有重复实现（只有 `judge_openqa` 一处复用），
和 `last_number` / `_strip_fence` 那两次「多地各写一份」不一样。

`repetition_metrics` 原先是 `train_sft.ProbeCallback.score` 的静态方法。
换 vLLM 引擎后 `eval.py` 要在 **vllm 环境**里跑，而那个环境装不了 unsloth，
`import train_sft` 直接失败 —— 所以下沉到这里。`ProbeCallback.score` 现在
只是转发，**训练曲线和离线评测仍然共用同一份实现**。
"""

from __future__ import annotations

import math


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """准确率的 95% 置信区间（Wilson 区间）。

    **为什么必须给区间**：400 题上是 ±4.5pp、14042 题上是 ±0.83pp。
    只报一个点估计（"26.5%"）会让读者以为精确到 0.1pp，据此下的结论多半是错的。

    Wilson 比正态近似好在两端不会越界 —— 准确率接近 0 或 1 时（base 的 MMLU 就是）
    正态近似会算出负的下界，Wilson 不会。
    """
    if total <= 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (
        round(max(0.0, (centre - margin) / denominator), 4),
        round(min(1.0, (centre + margin) / denominator), 4),
    )


def attach_ci(result: dict) -> dict:
    """给一个含 `correct` / `total` 的结果块补上置信区间，原地改并返回。"""
    low, high = wilson_ci(result["correct"], result["total"])
    result["ci95"] = [low, high]
    result["ci95_half_pp"] = round((high - low) / 2 * 100, 2)
    return result


def repetition_metrics(generated_ids) -> dict:
    """对生成结果打客观分。

    故意不算 ROUGE / BLEU：固定话题都是开放式的，没有唯一正确答案，
    硬套一个参考答案算出来的分数看着精确、其实没有意义，容易误导。

    能算的是「退化信号」——模型崩坏时最典型的表现是复读和长度失控：
      repeat_2gram   重复的 2-gram 占比，越高越像复读机
      distinct_ratio 不同 token 占比，越低越单调
      tokens         生成长度，突然暴涨/暴跌都是异常

    注意：它吃的是 **token id 列表**，不是文本。换引擎后 id 的来源会变
    （HF 是 generate 返回的原生 id，vLLM 只能拿回文本再重新编码），
    所以跨引擎比这个指标时要留意这一点。
    """
    ids = [int(t) for t in generated_ids]
    n = len(ids)
    if n == 0:
        return {"tokens": 0, "repeat_2gram": 0.0, "distinct_ratio": 0.0}

    grams = [tuple(ids[i : i + 2]) for i in range(n - 1)]
    repeat = 1 - len(set(grams)) / len(grams) if grams else 0.0
    return {
        "tokens": n,
        "repeat_2gram": round(repeat, 4),
        "distinct_ratio": round(len(set(ids)) / n, 4),
    }
