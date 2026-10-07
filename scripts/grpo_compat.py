"""TRL 0.24.0 × transformers 5.x 的兼容补丁（GRPO 前必须调用）。

问题现象
--------
`from trl import GRPOTrainer` 直接崩：

    RuntimeError: Failed to import trl.trainer.grpo_trainer because of the following
    error: No module named 'mergekit'

即使把 mergekit 装上，下一个报错变成 `No module named 'llm_blender'`，
再下一个是别的可选依赖 —— 永远装不完。

根因（已定位，不是环境问题）
----------------------------
`trl/import_utils.py`：

    _mergekit_available = _is_package_available("mergekit")
    ...
    def is_mergekit_available() -> bool:
        return _mergekit_available

而 **transformers 5.x 的 `_is_package_available()` 返回元组 `(exists, version)`**，
不是 bool。于是 `is_mergekit_available()` 返回 `(False, None)` ——
`if (False, None):` 在 Python 里是 **truthy**（非空元组恒为真）。

结果：`trl/mergekit_utils.py` 的 `if is_mergekit_available():` 这道 guard 形同虚设，
无条件去 `from mergekit.config import ...` → 包没装就崩。
同理受害的还有 `llm_blender` / `weave` / `liger_kernel` / `math_verify` / `deepspeed` 等
**12 个**可选依赖探测。

实测：tpt 环境 trl 0.24.0 + transformers 5.5.0，12 个 `_*_available` 全是元组。

处置
----
把模块级 `_*_available` 从元组改回 bool（只改 trl 内存里的变量，不动 site-packages
一个字节）。补丁后 `is_*_available()` 返回值正确，guard 恢复生效，缺包时走「不可用」
分支而不是崩。

为什么不在环境里装 mergekit
---------------------------
试过 `pip install --no-deps mergekit`，能装上，但 mergekit 0.1.4 是给 pydantic 2.10 写的，
当前环境是 pydantic 2.13.5 → `Unable to generate pydantic-core schema for torch.Tensor`。
而按正常依赖装 mergekit 会**降级** accelerate(1.15→1.6) 和 huggingface_hub(1.33→1.16)，
那会破坏训练环境。所以补丁是这里的正解。

用法（必须在 import trl 的 GRPOTrainer 之前）
---------------------------------------------

    from grpo_compat import patch_trl_optional_deps
    patch_trl_optional_deps()
    from trl import GRPOTrainer, GRPOConfig

`train_grpo.py` 已经这么做了，单独成文件是为了能写清这段根因、也方便单测。
"""

from __future__ import annotations


def patch_trl_optional_deps(verbose: bool = True) -> list[str]:
    """把 `trl.import_utils` 里被误写成元组的 `_*_available` 改回 bool。

    返回被修补的变量名列表（便于冒烟时打印核对）。
    """
    import trl.import_utils as iu

    fixed: list[str] = []
    for name in list(vars(iu)):
        if not name.endswith("_available"):
            continue
        value = getattr(iu, name)
        if isinstance(value, tuple):
            setattr(iu, name, value[0])  # (exists, version) → exists
            fixed.append(name)

    if verbose:
        print(f"[grpo_compat] 修补 trl 可选依赖探测 {len(fixed)} 个：{sorted(fixed)}")
        if not fixed:
            print("[grpo_compat] 无需修补（trl 已修此 bug，或 transformers < 5）")
    return fixed


def describe_env() -> str:
    """一行环境摘要，写进训练日志/run_meta 用。"""
    import transformers
    import trl

    return f"trl {trl.__version__} / transformers {transformers.__version__}"
