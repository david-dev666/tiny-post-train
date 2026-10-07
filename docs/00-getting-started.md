# 快速上手

本文给出从租卡到发布的最短路径，请按顺序执行。

## 0. 租卡

- 平台：AutoDL 或同类服务
- 卡型：4090 24G 起，预算充足可选 A100 40G
- 镜像：选择预装 PyTorch 2.x + CUDA 12.x 的社区镜像，建议自带 unsloth
- 项目应放在数据盘 `/root/autodl-tmp`，系统盘容量有限

## 1. 环境安装

```bash
cd /root/autodl-tmp
# 将本地 tiny-post-train 上传至此，或在服务器上直接创建

conda create -n tpt python=3.11 -y
conda activate tpt

pip install unsloth
pip install -r requirements.txt
```

安装完成后应立即验证，避免在训练阶段才暴露问题：

```bash
python -c "import torch, unsloth, trl, transformers; print(torch.__version__, transformers.__version__)"
```

## 2. 下载模型与数据

```bash
mkdir -p weights data/raw

modelscope download --model Qwen/Qwen3-0.6B-Base --local_dir weights/Qwen3-0.6B-Base
modelscope download --model Qwen/Qwen3-4B-Base --local_dir weights/Qwen3-4B-Base
modelscope download --dataset AI-ModelScope/alpaca-gpt4-data-zh --local_dir data/raw/alpaca-gpt4-zh
```

模型与数据均放在数据盘，不放在系统盘。

## 3. 冒烟测试

先用 0.6B 模型运行数十步，确认脚本、数据与显存均正常后，再运行 4B。

```bash
python scripts/train_sft.py \
  --model weights/Qwen3-0.6B-Base \
  --data data/raw/alpaca-gpt4-zh \
  --output outputs/smoke-sft-0.6b \
  --max-steps 30
```

运行结束后检查 `outputs/smoke-sft-0.6b` 中是否生成 adapter 文件。若存在，则说明流程已打通。

## 4. 正式训练

```bash
python scripts/train_sft.py \
  --model weights/Qwen3-4B-Base \
  --data data/raw/alpaca-gpt4-zh \
  --output outputs/sft-4b-v2 \
  --lora-r 32 --lr 2e-4 --num-epochs 1 \
  --batch-size 4 --grad-accum 4 --max-seq-len 2048
```

**`--num-epochs 1`，请勿改为 2。** 该取值为实测结论而非简化处理：运行 2 epoch 时，eval loss 在约 step 2900（约 1 epoch 处）触底后趋于平稳，**第二个 epoch 无额外收益**，并额外消耗 1.3 GPU 小时（2.4 h 对 1.1 h）。

显存不足时，将 `--batch-size` 降至 1，并将 `--grad-accum` 提高至 16。

## 5. 一周节奏

| 天 | 目标 | 产出 |
| --- | --- | --- |
| 1 | 租卡、安装环境、0.6B 冒烟 | 流程打通 |
| 2 | 完成 4B SFT | outputs/sft-4b |
| 3 | SFT 评测 | evals 中的基线数据 |
| 4-5 | DPO | outputs/dpo-4b |
| 6 | GRPO，小规模 | reward 曲线 |
| 7 | 部署、README、开源 | 可发布仓库 |

若进度不及预期，可缩减 GRPO 规模，但脚本与至少一次真实运行记录必须保留在仓库中。

## 6. 常见问题

- 显存不足：降低 batch size、开启 gradient checkpointing、改用 4bit 加载
- 数据格式报错：先运行数据格式化脚本，并人工抽查 3 条
- 训练 loss 不下降：优先排查数据，其次检查学习率
- 环境安装失败：删除环境后重建，不在已损坏的环境中修复

## 7. 注意事项

- 不要自行构造数据集，首轮全部使用开源现成数据
- 不要同时修改多个变量，每次仅调整一个，否则结果无法解释
- 不要省略冒烟测试；直接运行 4B 一旦报错，将浪费一整天
