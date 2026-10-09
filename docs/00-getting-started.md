# 极简上手指南（Getting Started）

目标：单卡环境下，在最短周期内跑通模型后训练全流程（环境搭建 → 冒烟测试 → SFT 训练 → 结果产出）。

---

## 0. 算力与存储准备

- **硬件推荐**：单卡 RTX 4090 (24G) 起步；预算充足可选 A100 (40G)。
- **镜像选择**：预装 PyTorch 2.x + CUDA 12.x 的 Linux 环境（推荐自带 unsloth 的镜像）。
- **存储路径**：**必须将代码、模型与数据挂载到数据盘**（如 `/root/autodl-tmp`），避免系统盘撑爆。

---

## 1. 环境搭建与隔离

训练与评测环境依赖不同（unsloth 锁定 torch 2.12.1，而 vLLM 评测需 torch 2.13），建议优先配置训练专用环境 `tpt`：

```bash
cd /root/autodl-tmp
# 克隆或进入项目目录
cd tiny-post-train

conda create -n tpt python=3.11 -y
conda activate tpt

pip install unsloth
pip install -r requirements.txt
```

**环境快速校验**（避免训练中途崩坏）：

```bash
python -c "import torch, unsloth, trl, transformers; print(f'CUDA={torch.cuda.is_available()}, PyTorch={torch.__version__}, Transformers={transformers.__version__}')"
```

---

## 2. 权重与数据集拉取

通过 ModelScope 极速下载 Base 模型与微调数据集（统一存入数据盘）：

```bash
mkdir -p weights data/raw

# 冒烟验证小模型 + 4B 主模型
modelscope download --model Qwen/Qwen3-0.6B-Base --local_dir weights/Qwen3-0.6B-Base
modelscope download --model Qwen/Qwen3-4B-Base --local_dir weights/Qwen3-4B-Base

# SFT 训练集（Alpaca GPT-4 中文高质量子集）
modelscope download --dataset AI-ModelScope/alpaca-gpt4-data-zh --local_dir data/raw/alpaca-gpt4-zh
```

---

## 3. 冒烟测试（必跑）

**严禁跳过冒烟直接跑 4B**。先用 0.6B 跑 30 步，验证数据加载、分词、反向传播与显存占用链路：

```bash
python scripts/train_sft.py \
  --model weights/Qwen3-0.6B-Base \
  --data data/raw/alpaca-gpt4-zh \
  --output outputs/smoke-sft-0.6b \
  --max-steps 30
```

*校验指标：检查 `outputs/smoke-sft-0.6b/` 下是否正常生成 LoRA adapter 权重文件。*

---

## 4. 正式 SFT 训练（4B）

```bash
python scripts/train_sft.py \
  --model weights/Qwen3-4B-Base \
  --data data/raw/alpaca-gpt4-zh \
  --output outputs/sft-4b-v2 \
  --lora-r 32 --lr 2e-4 --num-epochs 1 \
  --batch-size 4 --grad-accum 4 --max-seq-len 2048
```

### 关键超参结论与避坑
- **`--num-epochs 1`（切勿设为 2）**：实测 2 epoch 时，`eval_loss` 在 ~2900 步（≈1 epoch）即触底走平，第 2 个 epoch 毫无增益，徒增 1.3 小时电费（耗时 2.4h vs 1.1h）。
- **显存保护机制**：若遇到 OOM，将 `--batch-size` 降至 1，并将 `--grad-accum` 提升至 16（保持等效 batch size = 16 不变）。

---

## 5. 一周推进节奏

| 阶段 | 核心任务 | 核心产出 | 耗时预估 |
| --- | --- | --- | --- |
| **Day 1** | 环境安装、权重拉取、0.6B 冒烟跑通 | 验证端到端流水线正常 | ~2 小时 |
| **Day 2** | Qwen3-4B SFT (v2) 训练 | `outputs/sft-4b-v2` | 1.1 GPU 小时 |
| **Day 3** | SFT 离线评测（vLLM 引擎） | 产出基线横评报表 | 0.35 GPU 小时 |
| **Day 4-5** | DPO 偏好对齐训练与评测 | `outputs/dpo-4b-open` | 1.5 GPU 小时 |
| **Day 6** | GRPO 强化学习验证与归因 | 收敛曲线与负结果归因 | 1~2 GPU 小时 |
| **Day 7** | 评测报告生成、整理文档开源 | 交互式 HTML 报告发布 | 无卡操作 |

*注：若进度受阻，优先保证 SFT 与 DPO 的全量评测完备性，压缩 GRPO 规模，但务必在仓库中保留真实实验记录。*

---

## 6. 实战工程铁律

1. **坚持单一变量**：每次实验仅调整一个超参或一组数据，否则成败无法归因。
2. **严禁自制低质数据**：首周务必使用成熟、经过清洗的开源数据集（如 Alpaca-GPT4-zh、UltraFeedback），避免把时间浪费在数据脏污上。
3. **环境坏了直接重建**：遇复杂依赖冲突立即新建 conda 环境，切勿在破损环境中打补丁。
4. **训练异常排查顺序**：Loss 不降先抽样肉眼检查训练数据格式，再核对学习率与梯度；切忌盲目加大网络容量。
