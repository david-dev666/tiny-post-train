#!/usr/bin/env bash
# tiny-post-train 环境搭建脚本（面向 AutoDL 等自带 conda 的镜像）
#
# 用法：
#   bash scripts/setup_env.sh                    # 新建独立 conda 环境再装依赖
#   bash scripts/setup_env.sh --base             # 直接装进当前环境(base)，最快
#   bash scripts/setup_env.sh --download         # 顺带下载 0.6B 模型和数据集
#   bash scripts/setup_env.sh --base --download  # 两个一起用
#
# 什么时候用 --base：
#   选了 PyTorch 基础镜像、且这台机器只跑本项目、不介意环境不隔离时用它，
#   可以复用镜像预装的 torch，省掉一次约 3GB 的下载。
#   选了 Miniconda3 镜像时不要用 --base，那种镜像的 base 是白纸，没意义。
#
# 可覆盖的环境变量：
#   ENV_NAME      conda 环境名，默认 tpt（--base 模式下忽略）
#   PY_VER        python 版本，默认 3.11（--base 模式下忽略）
#   PROJECT_DIR   项目根目录，默认脚本所在项目的根目录
#   PIP_INDEX     pip 源，默认清华
set -euo pipefail

ENV_NAME="${ENV_NAME:-tpt}"
PY_VER="${PY_VER:-3.11}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

USE_BASE=0
DO_DOWNLOAD=0
for arg in "$@"; do
    case "$arg" in
        --base) USE_BASE=1 ;;
        --download) DO_DOWNLOAD=1 ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "未知参数: $arg"; exit 1 ;;
    esac
done

cd "$PROJECT_DIR"
echo "==> 项目目录: $PROJECT_DIR"

# ---------- 1. 环境 ----------
if ! command -v conda >/dev/null 2>&1; then
    echo "!! 没找到 conda。先装 miniconda，或改用镜像自带的 python。"
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "$USE_BASE" == "1" ]]; then
    conda activate base
    echo "==> --base 模式：直接装进 (base)，不新建环境"
    echo "    python: $(command -v python)"
    python -c "import torch; print('    已有 torch:', torch.__version__)" 2>/dev/null \
        || echo "    未检测到 torch，稍后由 unsloth 连带安装"
else
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "==> 环境 $ENV_NAME 已存在，跳过创建"
    else
        echo "==> 创建 conda 环境 $ENV_NAME (python $PY_VER)"
        conda create -n "$ENV_NAME" python="$PY_VER" -y
    fi
    conda activate "$ENV_NAME"
    echo "==> 已激活 $ENV_NAME"
fi

# ---------- 2. pip 换国内源 ----------
echo "==> pip 源: $PIP_INDEX"
pip config set global.index-url "$PIP_INDEX" >/dev/null
python -m pip install --upgrade pip >/dev/null 2>&1 || echo "    (pip 升级跳过，不影响)"

# ---------- 3. 装依赖 ----------
# 顺序很重要：unsloth 会锁定 torch 版本，必须先装它。
# base 模式下如果镜像预装的 torch 版本已满足要求，pip 会直接跳过，这是最省时的情况。
echo "==> 安装 unsloth"
pip install unsloth

# vllm 对 torch 的要求和 unsloth 冲突，默认不装，留给部署阶段单独环境。
echo "==> 安装训练依赖（跳过 vllm）"
grep -vE '^[[:space:]]*vllm[[:space:]]*$' requirements.txt > /tmp/tpt-req-train.txt
pip install -r /tmp/tpt-req-train.txt

if [[ "${INSTALL_VLLM:-0}" == "1" ]]; then
    echo "==> INSTALL_VLLM=1，额外安装 vllm（可能破坏训练环境，慎用）"
    pip install vllm
fi

# ---------- 4. 自检 ----------
echo "==> 自检"
python - <<'PY'
import torch, transformers, unsloth, trl, peft, datasets
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("trl / peft  :", trl.__version__, "/", peft.__version__)
print("cuda 可用   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
    print("显存(GB)    :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
PY

# ---------- 5. 可选：下模型和数据 ----------
if [[ "$DO_DOWNLOAD" == "1" ]]; then
    echo "==> 下载 0.6B 冒烟模型与数据集"
    mkdir -p weights data/raw
    modelscope download --model Qwen/Qwen3-0.6B-Base --local_dir weights/Qwen3-0.6B-Base
    modelscope download --dataset AI-ModelScope/alpaca-gpt4-data-zh --local_dir data/raw/alpaca-gpt4-zh
    echo "==> 4B 基座按需自己下（体积大，冒烟通过后再下）："
    echo "    modelscope download --model Qwen/Qwen3-4B-Base --local_dir weights/Qwen3-4B-Base"
fi

echo
if [[ "$USE_BASE" == "1" ]]; then
    echo "==> 完成。环境: base，直接用，不用 activate"
else
    echo "==> 完成。激活环境： conda activate $ENV_NAME"
fi
echo "==> 冒烟命令见 docs/00-getting-started.md 第 3 节"
