"""推理引擎抽象：同一份评测逻辑，可换 HF / vLLM 两条生成后端。

为什么要有这一层
----------------
换引擎 = **换了一把尺子**。所以不能直接切，必须先能 A/B（见 `ab_engine.py`），
而且切换之后的结果要能标清楚「这份分数是哪把尺子量的」—— 否则
`results/*.json` 里几份 json 混着两个引擎的分数，谁也不知道差异从哪来。

契约
----
    engine.run(prompts, max_new_tokens, on_progress) -> list[GenOut]

**为什么接口是批量的，而不是逐条**
vLLM 的加速全部来自 continuous batching。逐条调用等于每条自成一个 batch，
吞吐掉到 1/16 —— 那就等于没换引擎。所以一次把所有卷面交出去。

HF 后端内部仍然是逐条 `generate()`（它本来就只能这样），
行为与换引擎前**逐字节一致**，这样 HF 口径的历史结果依然可比。

卷面由调用方拼好
----------------
engine 只负责「给一段文本，生成一段文本」。
对话模板、few-shot、`Question/Answer` 这些**卷面问题**留在 `eval.py` /`prompts.py`。
理由：一旦引擎内部替你套 `chat()`，换引擎和换卷面两个变量就一起动了，
分数差异再也归因不了（`ab_engine.py` 的注释里记了这条）。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import prompts as P


def _prepare_env() -> None:
    """进入 vLLM 之前把环境铺好。两件事，都是被坑过之后补的。

    1) 把当前解释器所在目录挂到 PATH 前面
       flashinfer（vLLM 的采样核）初始化时会 `subprocess.run(["ninja", ...])`
       现场 JIT 编译。**用 `envs/vllm/bin/python scripts/eval.py` 直接调用时，
       那个目录不在 PATH 里**（只有 `conda activate` 才会加），于是报一句莫名其妙的
       `FileNotFoundError: ... 'ninja'` —— 而 ninja 就装在同一个环境里。
       报错指向 flashinfer 的 C++ 编译流程，跟「环境没激活」八竿子打不着，很难查。

    2) 关掉 flashinfer 采样核（`VLLM_USE_FLASHINFER_SAMPLER=0`）
       它默认是开的，但要现场用 nvcc 编一个 CUDA 核。本机 `/usr/local/cuda` 是
       **CUDA 12.4**，而 flashinfer 0.6.18 传了 `--compress-mode=size`（CUDA 12.8+
       才有的选项），编译直接 `nvcc fatal: Unknown option`，引擎初始化当场失败。
       换新 CUDA toolkit 要拉几个 G，不值得 —— 关掉它走 vLLM 自带的采样实现即可。
       代价：采样 kernel 换了实现。**所以开关这一项之后必须重新验证可复现性**
       （同一批输入跑两遍比对），不能沿用之前的结论。
    """
    bindir = str(Path(sys.executable).resolve().parent)
    if bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


@dataclass
class GenOut:
    """一条生成结果。

    `text` / `ids` 都**已按停止符截断**（停止符本身不算内容），
    两个后端语义一致 —— 否则 `truncated`、复读率这些诊断量在两个引擎之间不可比。

    `hit_cap`：是否用满了 `max_new_tokens` 还没结束（= 被截断）。
    """

    text: str
    ids: list[int]
    hit_cap: bool


class BaseEngine:
    name = "base"

    def __init__(self) -> None:
        # 生成过程的异常计数。分数之外还要看这些 —— 它们说明「分数为什么不可信」。
        self.stats: dict[str, int] = {"stop_token_in_middle": 0}
        self.config: dict = {"engine": self.name}

    def run(self, prompts: list[str], max_new_tokens: int, on_progress=None) -> list[GenOut]:
        raise NotImplementedError

    # 换引擎前没有这个契约，加它是为了让两个后端的进度输出长得一样：
    # 全量评测一次几十分钟，中途看不见进度是之前吃过亏的地方。
    @staticmethod
    def _tick(on_progress, done: int, total: int) -> None:
        if on_progress is not None:
            on_progress(done, total)


class HFEngine(BaseEngine):
    """unsloth / HuggingFace 路径。**行为与换引擎前完全一致**，用来做基准与兜底。

    它是唯一能做 `mmlu_logit` 口径的后端：那个口径要读「答案：」后面那个位置
    任意 token 的 logits，vLLM 只给 top-k logprobs，取不到。
    """

    name = "hf"

    def __init__(self, model_ref: str, max_seq_len: int, load_in_4bit: bool = False) -> None:
        import torch
        from unsloth import FastLanguageModel

        self.torch = torch
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_ref,
            max_seq_length=max_seq_len,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.stop_ids = set(P.stop_token_ids(tokenizer))
        super().__init__()
        self.config.update({
            "backend": "unsloth FastLanguageModel.generate",
            "do_sample": False,
            "load_in_4bit": bool(load_in_4bit),
            "max_seq_len": max_seq_len,
        })

    def run(self, prompts: list[str], max_new_tokens: int, on_progress=None) -> list[GenOut]:
        outputs = []
        torch = self.torch
        for index, prompt in enumerate(prompts, 1):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    # 必须带 <|im_end|>，否则模型说完了不会停，末尾吐乱码（见 stop_token_ids 注释）
                    eos_token_id=list(self.stop_ids),
                )
            raw = [int(t) for t in generated[0][inputs["input_ids"].shape[1]:]]
            # 在**第一个**停止符处截断，而不是「把停止符删掉、保留它后面的内容」。
            # 后面那种写法会把「模型回合结束之后」的 token 也算成回答的一部分。
            cut = next((n for n, t in enumerate(raw) if t in self.stop_ids), len(raw))
            if len(raw) - cut > 1:
                # 停止符后面居然还有内容（不止那一个停止符本身），记录下来供诊断
                self.stats["stop_token_in_middle"] += 1
            kept = raw[:cut]
            outputs.append(GenOut(
                text=self.tokenizer.decode(kept, skip_special_tokens=True).strip(),
                ids=kept,
                hit_cap=cut >= max_new_tokens,
            ))
            if index % 50 == 0 or index == len(prompts):
                BaseEngine._tick(on_progress, index, len(prompts))
        return outputs


class VLLMEngine(BaseEngine):
    """vLLM 批量推理。实测比 HF 快 16.7×（9.43 → 0.57 s/题，50 题 GSM8K）。

    **必须开 `enforce_eager`**，否则同一批输入跑两遍结果不一样：
    实测默认配置下 50 题里有 10 题预测翻转、命中 27 vs 28，
    而 `enforce_eager=True` 之后两次跑**逐字完全相同**（0/50）。
    原因是 CUDA graphs 按运行时 batch 大小选不同的图，而 continuous batching
    的 batch 组成每一步都在变。关掉它慢约 12%，换完全可复现 —— 评测是尺子，
    这个交换是必须的。

    注意：开了 enforce_eager 只能保证**同一个引擎内**可复现，
    跨引擎的差异（HF vs vLLM 50 题里 13 题翻转）是 bf16 两条数值路径的
    自回归误差发散，消不掉。**所以不能混用两个引擎的分数做对比。**
    """

    name = "vllm"

    def __init__(
        self,
        model_ref: str,
        max_seq_len: int,
        adapter: str | None = None,
        gpu_util: float = 0.85,
        enforce_eager: bool = True,
        enable_prefix_caching: bool = False,
        max_lora_rank: int = 64,
    ) -> None:
        import vllm
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        _prepare_env()

        self.llm = LLM(
            model=model_ref,
            max_model_len=max_seq_len,
            gpu_memory_utilization=gpu_util,
            dtype="bfloat16",
            enforce_eager=enforce_eager,
            # prefix caching 会让 KV 块布局随调度/淘汰顺序变化 → attention 走不同路径。
            # 实测它**不是**不可复现的主因（主因是 CUDA graphs），但既然不占便宜，就关掉。
            enable_prefix_caching=enable_prefix_caching,
            enable_lora=bool(adapter),
            max_lora_rank=max_lora_rank,
            disable_log_stats=True,
        )
        self.adapter = adapter
        self.lora_request = (
            LoRARequest("adapter", 1, adapter) if adapter else None
        )
        # 用**我们自己**的 tokenizer 而不是 llm.get_tokenizer()：卷面模板由
        # eval.py 决定（可能要覆盖成项目内置的干净模板），不能让引擎偷偷换一套。
        self.tokenizer = AutoTokenizer.from_pretrained(model_ref)
        self.stop_ids = P.stop_token_ids(self.tokenizer)
        super().__init__()
        self.config.update({
            "backend": "vllm.LLM.generate",
            "vllm_version": vllm.__version__,
            "do_sample": False,
            "dtype": "bfloat16",
            "max_seq_len": max_seq_len,
            "gpu_memory_utilization": gpu_util,
            "enforce_eager": enforce_eager,
            "enable_prefix_caching": enable_prefix_caching,
            "lora": adapter,
        })

    def run(self, prompts: list[str], max_new_tokens: int, on_progress=None) -> list[GenOut]:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=0,          # greedy，和 HF 的 do_sample=False 对齐
            top_p=1.0,
            max_tokens=max_new_tokens,
            stop_token_ids=self.stop_ids,
            skip_special_tokens=True,
        )
        outputs = self.llm.generate(
            prompts, params, lora_request=self.lora_request, use_tqdm=False,
        )
        results = []
        for out in outputs:
            completion = out.outputs[0]
            ids = [int(t) for t in completion.token_ids]
            results.append(GenOut(
                text=completion.text.strip(),
                ids=ids,
                # vLLM 的 token_ids **不含**停止符，所以「用满预算」就是 finish_reason=length
                hit_cap=completion.finish_reason == "length",
            ))
        BaseEngine._tick(on_progress, len(prompts), len(prompts))
        return results


def make_engine(name: str, model_ref: str, **kwargs) -> BaseEngine:
    if name == "hf":
        return HFEngine(model_ref, **kwargs)
    if name == "vllm":
        return VLLMEngine(model_ref, **kwargs)
    raise SystemExit(f"!! 不认识引擎 {name!r}（可选 hf / vllm）")


# 两个后端在结果 json 里留下的 `engine` 字段取值。报告靠它判断
# 「这几行分数是不是同一把尺子量的」，所以只能从这里取，不许各写各的字面量。
ENGINES = ("hf", "vllm")

# 只有 HF 能做 mmlu_logit：它要读「答案：」后面那个位置**任意** token 的 logits，
# vLLM 只暴露 top-k logprobs，取不到。跑 logit 口径时用 vllm 直接报错，
# 而不是悄悄跳过 —— 悄悄跳过会让人以为「这个口径也算过了」。
LOGIT_ONLY_HF = "mmlu"
