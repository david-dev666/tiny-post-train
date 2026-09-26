# tiny-post-train

Qwen3-4B 的后训练全流程开源项目：SFT -> DPO -> GRPO -> 评测 -> 部署。

## 目标

在单卡 24G 显存上，用一套可复现的脚本跑通小模型后训练的完整链路，并给出真实评测对比，而不是只贴一个 loss 曲线。

## 流程

| 阶段 | 方法 | 脚本 | 状态 |
| --- | --- | --- | --- |
| 0 | 环境搭建 | scripts/setup_env.sh | 可用 |
| 1 | SFT 监督微调 | scripts/train_sft.py | 脚本就绪，待跑 |
| 2 | DPO 偏好对齐 | scripts/train_dpo.py | 未实现 |
| 3 | GRPO 可验证奖励强化 | scripts/train_grpo.py | 未实现 |
| 4 | 评测 | scripts/eval.py | 未实现 |
| 5 | 推理部署 | scripts/serve_vllm.sh | 未实现 |

## 实验设计

- 基座：Qwen/Qwen3-4B-Base
- 对照基线：Qwen/Qwen3-4B-Instruct-2507，官方对齐版，用于说明自训流程与官方流程的差距
- 冒烟模型：Qwen/Qwen3-0.6B-Base，用于验证脚本管道，几十分钟就能跑完一轮

从 Base 起步而不是从 Instruct 起步，是为了让 SFT 这一步是真实可控的，整条链路才是完整的后训练。

## 快速开始

见 docs/00-getting-started.md。

## 结果

待填：SFT / DPO / GRPO 三阶段在各评测集上的分数对比表。

## 硬件与成本

待填：卡型、GPU 小时数、实际花费。

## 目录结构

```text
tiny-post-train/
├── README.md
├── requirements.txt
├── configs/            训练配置
├── data/
│   ├── raw/            原始数据，不进版本库
│   └── processed/      清洗后的数据
├── docs/               文档
├── evals/              评测结果与日志
├── logs/               训练日志
├── outputs/            权重输出，不进版本库
├── scripts/            训练、评测、部署脚本
└── weights/            基座权重，不进版本库
```

## 许可

Apache-2.0
