# IFBench（vendored）

本目录是 [allenai/IFBench](https://github.com/allenai/IFBench) 的**原样副本**，用于 GRPO 的可验证奖励判分。

| 项 | 值 |
| --- | --- |
| 上游仓库 | `https://github.com/allenai/IFBench` |
| 取回 commit | `1c40f0c`（main 分支，2026-10-06 取回） |
| 取回方式 | `curl -sL https://codeload.github.com/allenai/IFBench/tar.gz/refs/heads/main` |
| 许可证 | Apache-2.0（见同目录 `LICENSE`） |
| 论文 | *Generalizing Verifiable Instruction Following*，arXiv 2507.02833（NeurIPS 2025） |

## 为什么放在 `scripts/` 而不是 `third_party/`

项目用黑名单式 rsync 同步代码（排除 `weights/` `data/` `outputs/` `logs/` 等）。判分器和测试集要能上服务器，所以放在 `scripts/ifbench/`。

## 复制了哪些文件

| 文件 | 用途 |
| --- | --- |
| `instructions.py` | 58 个 IFBench OOD 约束的验证类 |
| `classic_instructions.py` | 25 个 IFEval 经典约束的验证类 |
| `instructions_registry.py` | `INSTRUCTION_DICT`：约束 id → 验证类 |
| `instructions_util.py` | 分词、句子切分等工具 |
| `__init__.py` | 包入口，含 `data_path()` |
| `data/IFBench_test.jsonl` | **泛化验证集**，299 条 held-out prompt（58 个全新约束） |

上游的 `run_eval.py` / `generate_responses.py` / `config.py`（要 API key 的评测 CLI）没有复制。本项目只用判分逻辑，另外要写 `eval_ifbench.py` 接进既有评测栈（尚未实现）。

## 改动记录

**无改动。** 与上游逐字节一致，升级时直接覆盖整个目录。

## 依赖

上游 `requirements.txt`：`absl-py langdetect nltk immutabledict spacy emoji syllapy>=0.8.0`。

实际判分路径用到的是 `absl-py emoji immutabledict langdetect nltk syllapy`（从源码 import 实测得出，没用 spacy），已写进根目录 `requirements.txt`。
