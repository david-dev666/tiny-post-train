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
| 辅助 | 训练看板 | dashboard/server.py | 可用 |

## 实验设计

- 基座：Qwen/Qwen3-4B-Base
- 对照基线：Qwen/Qwen3-4B-Instruct-2507，官方对齐版，用于说明自训流程与官方流程的差距
- 冒烟模型：Qwen/Qwen3-0.6B-Base，用于验证脚本管道，几十分钟就能跑完一轮

从 Base 起步而不是从 Instruct 起步，是为了让 SFT 这一步是真实可控的，整条链路才是完整的后训练。

## 快速开始

见 docs/00-getting-started.md。

## 训练看板

训练脚本会把指标追加写到 `outputs/<run>/metrics.jsonl`，看板读这个文件实时画曲线。

```bash
bash scripts/start_dashboard.sh          # 起服务
bash scripts/start_dashboard.sh stop     # 停
bash scripts/start_dashboard.sh token    # 只看口令
```

看板默认跑在 **6006**，并自动生成访问口令（存在 `logs/dashboard.token`，已被 gitignore）。每次启动复用同一个口令，所以手机书签长期有效。

**手机 / 任意设备**：把 AutoDL 实例卡片「自定义服务」里 **6006 那条**域名接上口令打开：

```
https://<你的实例域名>/?token=<口令>
```

**Mac 走 SSH 隧道**（不经过公网，等价可用）：

```bash
ssh -N -L 6006:localhost:6006 autodl
# http://localhost:6006/?token=<口令>
```

看板显示：

| 类别 | 内容 |
| --- | --- |
| 曲线 | train loss、eval loss、perplexity、learning rate、MMLU 子集准确率 |
| 进度 | 当前步数 / 总步数、预计剩余时间 |
| 硬件 | GPU 利用率、显存、温度、功耗 |
| 健康 | 训练进程是否存活 |
| 配置 | 基座模型、数据路径、超参、验证集大小 |
| 数据 | 训练数据抽样 3 条（真实喂进去的样本） |
| 推理 | 固定问题的生成结果（最新 + 上一次对比），附复读率 / 生成长度 |

### 指标说明（别误读）

| 指标 | 含义 | 别当它是 |
| --- | --- | --- |
| `train loss` | 对训练集的拟合程度 | 回答质量 |
| `eval loss` | 对验证集的拟合程度 | 回答质量 |
| `perplexity` | `exp(eval_loss)`，同样只反映拟合度 | 回答质量。**过拟合时它照样降** |
| `复读率` | 生成里重复 2-gram 的占比 | 质量分。只用来发现**退化**（复读率飙升 = 崩了） |
| `词面多样` | 不同 token 的占比 | 质量分。太低说明输出单调 |
| `MMLU 子集准确率` | 固定题目上的多选题正确率 | 综合能力。**用指令数据做 SFT 通常不会提升它，甚至下降**（灾难性遗忘），这是常见现象不是 bug |

**故意没有做的**：不自动算 ROUGE / BLEU，也不给生成结果打质量分。采样的三个问题都是开放式的，没有唯一正确答案，硬套参考答案算出来的数字看着精确、实则误导。真正的效果要看生成结果本身，以及后续 `eval.py` 跑出来的基准评测。

### 基准评测（训练中）

MMLU 全量 14042 题，4B 上跑一遍要一两个小时，比训练还慢，所以用**固定子集**：

```bash
python scripts/bench_subset.py     # 生成 evals/mmlu-subset.jsonl（57 学科 × 8 题 = 456 题）
```

```bash
--bench-every 100                  # 每 100 步在子集上评一次
```

几个要点：

- **子集必须固定**（`--seed` + 只抽一次落盘）。每次重抽的话，题目难度变化会被误读成模型退步。
- **`evals/mmlu-subset.jsonl` 要提交进版本库**，SFT / DPO / GRPO 三阶段的分数才可比。
- 判分只比 A/B/C/D 四个 token 的 logits，不做生成，一次前向就够——所以它测的是「模型更倾向输出哪个字母」，**不等于模型会答题**。
- 国内直连 `huggingface.co` 会超时，脚本默认走 `hf-mirror.com`，可用 `--hf-endpoint` 换。
- 也可以 `--source /path/to/local.jsonl` 完全绕开网络。

### 实时评测指标怎么产生

```bash
--eval-ratio 0.02     # 从训练集切 2% 当验证集（默认）
--eval-steps 50       # 每 50 步评一次
--probe-every 50      # 每 50 步用固定问题采样一次生成，0 表示关闭
```

验证集切分用 `--seed` 固定，**换 seed 曲线就不可比**。

### 关于安全

- 服务**只绑 `127.0.0.1`**。AutoDL 的公网映射转发的是容器内 localhost，所以手机照样能访问，但容器内不会多开一层监听面。
- **口令是必须的**。6006 公网可达，没口令等于把训练数据挂在网上。口令在 URL 里传一次，之后前端存 localStorage 并改用 `X-Token` 请求头。
- 只在隧道场景下想省掉口令：`ENABLE_AUTH=0 bash scripts/start_dashboard.sh`。
- 6008 是另一条映射端口，留给 TensorBoard 用，别和看板抢。

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
├── dashboard/          训练看板：后端 + 单页前端
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
