# 极简上手

目标：一周内让仓库有可发布的真实结果。按顺序做，别跳步。

## 0. 租卡

- 平台：AutoDL 或同类
- 卡型：4090 24G 起步，预算够就 A100 40G
- 镜像：选预装 PyTorch 2.x + CUDA 12.x 的社区镜像，最好自带 unsloth
- 项目放数据盘 `/root/autodl-tmp`，系统盘很小

## 1. 装环境

```bash
cd /root/autodl-tmp
# 把本地 tiny-post-train 上传到这里，或直接在服务器上建

conda create -n tpt python=3.11 -y
conda activate tpt

pip install unsloth
pip install -r requirements.txt
```

装完立刻验一遍，不要等训练时报错才发现：

```bash
python -c "import torch, unsloth, trl, transformers; print(torch.__version__, transformers.__version__)"
```

## 2. 下模型和数据

```bash
mkdir -p weights data/raw

modelscope download --model Qwen/Qwen3-0.6B-Base --local_dir weights/Qwen3-0.6B-Base
modelscope download --model Qwen/Qwen3-4B-Base --local_dir weights/Qwen3-4B-Base
modelscope download --dataset AI-ModelScope/alpaca-gpt4-data-zh --local_dir data/raw/alpaca-gpt4-zh
```

模型和数据都放数据盘，别放系统盘。

## 3. 冒烟测试

先拿 0.6B 跑几十步，确认脚本、数据、显存都正常，再动 4B。

```bash
python scripts/train_sft.py \
  --model weights/Qwen3-0.6B-Base \
  --data data/raw/alpaca-gpt4-zh \
  --output outputs/smoke-sft-0.6b \
  --max-steps 30
```

跑完看 `outputs/smoke-sft-0.6b` 里有没有 adapter 文件。有就说明管道通了。

## 4. 正式训练

```bash
python scripts/train_sft.py \
  --model weights/Qwen3-4B-Base \
  --data data/raw/alpaca-gpt4-zh \
  --output outputs/sft-4b-v2 \
  --lora-r 32 --lr 2e-4 --num-epochs 1 \
  --batch-size 4 --grad-accum 4 --max-seq-len 2048
```

**`--num-epochs 1`，别照抄成 2。** 这是实测结论不是省事：跑 2 epoch 时 eval loss 在约 step 2900（≈1 epoch 处）触底就走平了，**第二个 epoch 是白跑**，白烧 1.3 GPU 小时（2.4 h vs 1.1 h）。

显存不够就把 `--batch-size` 降到 1，把 `--grad-accum` 提到 16。

## 5. 一周节奏

| 天 | 目标 | 产出 |
| --- | --- | --- |
| 1 | 租卡、装环境、0.6B 冒烟 | 管道打通 |
| 2 | 4B SFT 跑完 | outputs/sft-4b |
| 3 | SFT 评测 | evals 里的基线表 |
| 4-5 | DPO | outputs/dpo-4b |
| 6 | GRPO，小规模 | reward 曲线 |
| 7 | 部署、README、开源 | 可发布仓库 |

进度崩了就砍 GRPO 的规模，但脚本和一次真实 run 必须留在仓库里。

## 6. 常见问题

- 显存爆了：降 batch size、开 gradient checkpointing、改 4bit 加载
- 数据格式报错：先把数据格式化脚本跑一遍，肉眼抽查 3 条
- 训练 loss 不降：先怀疑数据，再怀疑学习率
- 环境装不上：删掉环境重建，不要在一个坏环境里修

## 7. 不要做的事

- 不要自己造数据集，第一周全部用开源现成的
- 不要同时改多个变量，一次只动一个，否则结果没法解释
- 不要跳冒烟测试，4B 直接开跑，报错会浪费一整天
