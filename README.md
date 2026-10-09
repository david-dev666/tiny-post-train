# tiny-post-train

> 单卡 RTX 4090 (24G) 上的 Qwen3-4B 全流程后训练实践：SFT → DPO → GRPO → 全量评测。
> **拒绝“只贴一条 Loss 曲线”，拒绝抽样评测，全量测试集严谨对齐。**

📊 **[在线交互式评测报告](https://david-dev666.github.io/tiny-post-train/)**（支持展开查看每道题详情、诊断异常输出）

---

## 核心战果：全量严格对齐评测

所有模型均在**同一套判分栈、统一 vLLM 推理引擎（全量仅 21 分钟）、严禁任何解码作弊**（禁止 `logit_bias`、禁止惩罚参数、统一截断上限）下产出。

| 模型 | MMLU（纯文本）<br>14,042 题 | 指令遵循<br>200 题（去尾口径） | GSM8K<br>官方 test 1,319 题 | HumanEval<br>164 题（未截断） | 总训练耗时<br>RTX 4090 |
| --- | --- | --- | --- | --- | --- |
| **Qwen3-4B-Base**（基线） | 69.5% | 39.5% | 68.0% | 67.7% | — |
| **+ LoRA SFT**（v2） | 70.1% | 61.0% *(75.0%)* | 84.4% | 73.8% *(79.4%)* | **1.1 h** |
| **+ 开源偏好 DPO** | **70.2%** | 61.0% *(**78.5%**)* | **88.6%** | **84.8%** *(**84.7%**)* | **~1.5 h** |
| **Qwen3-4B-Instruct**（官方） | 64.2% | **98.5%** *(95.5%)* | **90.6%** | **85.4%** | 官方对齐版 |

*注：HumanEval 在剔除因 512 token 截断造成的虚假低分后，DPO 实际代码能力追到**仅落后官方 Instruct 1 道题**（139/164 vs 140/164）。*

---

## 核心发现与工程避坑（Key Takeaways）

### 1. 真实能力的增长点在哪里？

* **SFT 学的是格式而非知识**：MMLU 分数基本不动（69.5% → 70.1%），但数学与代码飞跃（GSM8K +16.4pp，代码约束遵循率 1/25 → 25/25）。
* **DPO 是第二增长曲线**：放弃自采数据，改用开源清洗后的 UltraFeedback 中文偏好集后，GSM8K 逼近官方（88.6% vs 90.6%），且输出更凝练（平均 token 减少 22%）。
* **与官方剩余差距**：主要集中在**复杂指令遵循**（61.0% vs 98.5%），尤其是数值约束与严格长度控制。

### 2. 踩坑复盘：收尾乱码尾巴（13 次训练侧证否）

自训 SFT (v2) 引入了一个隐蔽缺陷：自然收尾时 93% 概率多吐一个冷门 token（如希伯来语词 `לחלוט` 或乱码字节），撑爆了长度约束。

* **13 次训练尝试全否**：尝试了末位加权损失 (×20)、EOS 对齐、定向单 token DPO、拒绝采样 (RFT) 等，**全被证否**。
* **假信号陷阱**：强行抑制末位 token 会诱发“停止能力坍塌”（模型不再主动停止，直接跑满 max_tokens），**造成指标虚高假象**。
* **唯一解法**：归入部署阶段处理——推理侧轻微提升 `<|im_end|>` 的 logit (+2.0)，指令遵循瞬间恢复至 80.5%，乱码率归零。

### 3. GRPO 尝试归因

在指令遵循任务上尝试了 GRPO + IF-RLVR（主指标 0.0pp 提升）：

* **序列级奖励打不准**：约束失败主要是末位多吐了一个 token，GRPO 整个序列打负分无法反向指导单点截断。
* **LoRA 步长受限**：照抄全参论文的 `lr=1e-6` 过小，KL 散度均值仅 0.0009，策略几乎留在原地。详见 [`docs/02-GRPO尝试.md`](docs/02-GRPO尝试.md)。

---

## 算力消耗与成本

全套工程在 **单卡 RTX 4090 24G** 运行，总有效算力仅消耗 **~10 GPU 小时**：

| 阶段 | 耗时 | 配置与关键参数 |
| --- | --- | --- |
| **SFT (v2)** | 1.1 h | 4B / LoRA r=32 / 等效 batch 16 / seq_len 2048 / 1 epoch |
| **DPO** | 1.5 h | UltraFeedback 中文偏好集 18.2k 对 / 1 epoch |
| **vLLM 全量评测** | **0.35 h** | 4 套基准同时跑完仅 21 分钟（比 HF 引擎快 16.7 倍） |
| **调试与排错** | ~1.5 h | 判分边界、截断排查与回归测试 |

---

## 硬件与算力

**单卡 RTX 4090 24G**（实测 24564 MiB，driver 580.76.05；宿主 128 核 / 1TB 内存，容器挂 50G 数据盘）。

两套 Python 环境互不干扰，**装不到一起**：vLLM 要 torch 2.13，unsloth 锁在 2.12.1，硬合会把训练搞坏。

| 环境 | 用途 | 关键版本 |
| --- | --- | --- |
| `tpt` | 训练 + HF 引擎推理 | torch 2.12.1+cu130 / transformers 5.5.0 / unsloth |
| `vllm` | 评测（vLLM 引擎） | torch 2.13.0+cu130 / transformers 5.18.0 / vllm 0.30.0 |

GPU 小时明细：

| 阶段 | 实测 | 说明 |
| --- | --- | --- |
| SFT 4B · **1 epoch（v2，在用）** | **1.1 h** | 2991 步；LoRA r=32 / 等效 batch 16（micro 4 × accum 4）/ max-seq-len 2048；alpaca-gpt4-zh 约 4.8 万条 |
| SFT 4B · 2 epoch（v1，已弃用） | 2.4 h | 5982 步。eval loss 在 ~step 2900 就走平，第二个 epoch 是白跑，v2 才改成 1 epoch |
| 训练冒烟（0.6B） | 0.04 h | 只验链路 |
| **vLLM 引擎 · 三模型全量评测** | **0.35 h** | 21 分钟 |
| HF 引擎 · 三模型全量评测 | 2.9 h | 176 分钟，仅历史 |
| 失败重跑与探针 | ~1.5 h（估） | `splitlines` 崩溃、GSM8K token 上限、引擎 A/B 复现性，见 [`docs/01-评测.md`](docs/01-评测.md) 的「判分 bug 史」 |
| 环境搭建 / 下载权重 | 走**无卡模式** | 不占 GPU 小时 |

**合计约 9~10 GPU 小时**，除估算行外都是日志里量出来的。

---

## 训练全流程路线

```text
[Qwen3-4B-Base]
       │
       ▼  scripts/train_sft.py (LoRA r=32, 1.1h)
 [ sft-4b-v2 ]
       │
       ▼  scripts/train_dpo.py (UltraFeedback 开源偏好集)
 [ dpo-4b-open ] ─── 逼近官方 Instruct 性能
       │
       ▼  scripts/eval.py (接入 vLLM 纯净推理引擎)
 [ 全量 5 套评测 ] ─── 自动生成独立 HTML 报告
```

---

## 流程与脚本

| 阶段 | 方法 | 脚本 | 状态 |
| --- | --- | --- | --- |
| 0 | 环境搭建 | `scripts/setup_env.sh` | 可用 |
| 1 | SFT 监督微调 | `scripts/train_sft.py` | 已跑通。主模型 **v2**：4B / LoRA r=32 / 1 epoch，约 1.1 GPU 小时。v3「修收尾乱码」是**假信号**，已回退 |
| 2 | DPO 偏好对齐 | `scripts/train_dpo.py` | 已跑通。数据改用**开源** UltraFeedback 中文偏好集；此前定向构造的 5 轮全否 |
| 3 | GRPO 可验证奖励 | `scripts/train_grpo.py` | 已尝试，未成功，主指标 0.0pp，没有正式结果。[归因与边界](docs/02-GRPO尝试.md) |
| 4 | 评测 | `scripts/eval.py` | 可用。5 套评测 + 置信区间 + 可切换推理引擎 |
| 5 | 推理部署 | — | **没做**。评测侧 vLLM 已跑通，见 `scripts/engines.py` |
| 辅助 | 训练看板 | `dashboard/server.py` | 可用 |
| 辅助 | 多模型评测 pipeline | `scripts/eval_pipeline.py` | 可用，跑完自动出 HTML 报告 |
| 辅助 | 评测报告 | `scripts/make_report.py` | 可用，自包含 HTML，零外部依赖 |
| 辅助 | 离线重判 | `scripts/rescore.py` | 可用，改判分口径不必重跑模型 |
| 辅助 | 判分回归测试 | `tests/test_scoring.py` | 可用，34 条，不依赖 GPU |

---

## 实验设计

- 基座：`Qwen/Qwen3-4B-Base`
- 对照基线：`Qwen/Qwen3-4B-Instruct-2507`，官方对齐版，用来说明自训流程与官方流程的差距
- 冒烟模型：`Qwen/Qwen3-0.6B-Base`，只用来验证脚本管道，几十分钟就能跑一轮

从 Base 起步而不是从 Instruct 起步，是为了让 SFT 这一步真实可控，整条链路才是完整的后训练。

---

## 快速上手

详细环境准备参考 [`docs/00-getting-started.md`](docs/00-getting-started.md)。

```bash
# 1. 跑通冒烟流程 (0.6B 校验脚本链路，约数分钟)
python scripts/train_sft.py --model Qwen/Qwen3-0.6B-Base --smoke-test

# 2. 启动训练实时监控看板 (默认端口 6006，自动生成安全 token)
bash scripts/start_dashboard.sh

# 3. 运行全量 vLLM 评测矩阵
python scripts/eval_pipeline.py --engine vllm --models sft-4b-v2,dpo-4b-open

# 4. 生成自包含 HTML 评测报告
python scripts/make_report.py --output docs/index.html
```

---

## 训练看板

训练脚本将指标追加写入 `outputs/<run>/metrics.jsonl`，轻量看板服务后台读取并实时渲染。

```bash
bash scripts/start_dashboard.sh          # 启动（tmux 后台运行，默认 6006 端口）
bash scripts/start_dashboard.sh stop     # 停止
bash scripts/start_dashboard.sh token    # 查看访问口令
```

### 访问与安全

服务仅监听 `127.0.0.1:6006`（6008 留给 TensorBoard）。启动时生成鉴权口令并持久化至 `logs/dashboard.token`（已 gitignore），重启依然复用，手机书签长期有效：

- **手机端（AutoDL）**：复制实例卡片「自定义服务」的 6006 映射域名，拼接 `?token=<口令>` 直连。
- **Mac 本地**：建 SSH 隧道 `ssh -N -L 6006:localhost:6006 autodl`，访问 `http://localhost:6006/?token=<口令>`。
- *安全提示：走公网映射必须开启口令；纯隧道环境调试可传 `ENABLE_AUTH=0` 关闭鉴权。*

### 核心监控能力

- **训练曲线**：Train / Eval Loss、Perplexity、学习率调度、MMLU 快速子集。
- **硬件与进度**：训练进度与预计剩余时间、GPU 实时状态（利用率 / 显存 / 温度 / 功耗）、进程存活状态。
- **定性探针**：固定 Prompt 在线生成对比（肉眼诊断退化与复读）、训练集随机抽样 3 条。

### 指标解读与避坑指南

看板指标主要用于**诊断训练稳定性与快速发现崩坏**，切忌直接当成最终能力评分：

| 指标 | 真实含义 | 常见误区（别当它是） |
| --- | --- | --- |
| `train / eval loss` | 对训练集 / 验证集的拟合程度 | **≠ 回答质量**。指令微调阶段 Loss 趋平不等于能力停止进化 |
| `perplexity` | 预测下个 token 的困惑度（`exp(eval_loss)`） | **≠ 回答质量**。即使过拟合/背题，PPL 照样持续下降 |
| `复读率 / 词面多样` | 2-gram 重复率与独立 token 占比 | **≠ 质量分**。仅用作发现退化与模式坍塌的探针 |
| `MMLU 子集准确率` | 456 题固定子集的 A/B/C/D logits 倾向 | **≠ 综合智商**。指令微调不增甚至微降（灾难性遗忘）属正常现象 |

- **为什么用 456 题 MMLU 子集**：全量 14,042 题评测需 1~2 小时；训练中使用固定入库的子集（`evals/mmlu-subset.jsonl`，57 学科 × 8 题），由 `--bench-every 100` 每 100 步比对 logits 倾向，专用于高频监控知识遗忘。
- **为什么不计算 ROUGE / BLEU**：生成探针均为开放式问答，不存在唯一样本答案，表面重合度没有参考价值；严谨能力评估完全交由全量评测矩阵。

---

## 目录结构

```text
tiny-post-train/
├── dashboard/       # 实时训练监控看板（指标、硬件、采样对比）
├── scripts/
│   ├── train_sft.py        # SFT 训练脚本
│   ├── train_dpo.py        # DPO 偏好对齐
│   ├── train_grpo.py       # GRPO 强化学习尝试
│   ├── eval.py             # 评测入口（支持 vLLM / HF）
│   ├── eval_pipeline.py    # 自动化多模型评测矩阵
│   └── make_report.py      # 生成自包含 HTML 报告
├── docs/            # 详细踩坑记录与评测文档
└── tests/           # 判分回归测试集（34 条用例，无需 GPU）
```

## 协议

[Apache-2.0](LICENSE)
