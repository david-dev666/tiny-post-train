"""本地权重时强制离线。**纯 stdlib，不 import torch。**

为什么单独成模块
----------------
`_force_offline_for_local_model` 原本住在 `train_sft.py` 里，靠
「`eval.py` 先 `import train_sft`」这个副作用生效（它在模块顶层就执行）。

换 vLLM 引擎之后这条路断了：`train_sft` 顶层 `import unsloth`，
而 vllm 环境装不了 unsloth —— 在 vllm 环境里跑 `eval.py`，
那句 import 会直接 **ImportError**，整个评测连启动都启动不了。

所以把这一步提出来单独成模块，训练脚本和评测脚本各调一次同一个函数。
（本项目已经因为「一处实现抄成两处」栽过两次：`splitlines()` 和闭围栏正则。
 这里就不再抄第三遍。）

必须在**大件依赖 import 之前**调用 —— `huggingface_hub` 是导入时读环境变量的。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def force_offline_for_local_model(argv) -> str:
    """模型给的是本地目录时，把 HuggingFace hub 关掉，返回那个目录（没给就返回空串）。

    不关的话，即使 `--model` 传的是本地路径，unsloth / transformers 仍会去
    huggingface.co 查一次元信息；国内机器连不上就无限重试，
    表现是「加载模型卡死」，很难看出问题在哪。

    实测：同一份权重，不加这个 10 分钟不动，加了 2.3 秒载入。
    """
    for index, arg in enumerate(argv):
        value = ""
        if arg == "--model" and index + 1 < len(argv):
            value = argv[index + 1]
        elif arg.startswith("--model="):
            value = arg.split("=", 1)[1]
        if value and Path(value).expanduser().is_dir():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            return value
    return ""


LOCAL_MODEL_PATH = force_offline_for_local_model(sys.argv)
