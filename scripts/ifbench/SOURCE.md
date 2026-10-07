# IFBench（vendored）

本目录是 [allenai/IFBench](https://github.com/allenai/IFBench) 的**原样副本**，用于 GRPO 的可验证奖励判分。

| 项 | 值 |
| --- | --- |
| 上游仓库 | `https://github.com/allenai/IFBench` |
| 取回 commit | `1c40f0c`（main 分支，2026-10-06 取回） |
| 取回方式 | `curl -sL https://codeload.github.com/allenai/IFBench/tar.gz/refs/heads/main` |
| 许可证 | Apache-2.0（见同目录 `LICENSE`） |
| 论文 | *Generalizing Verifiable Instruction Following*，arXiv 2507.02833（NeurIPS 2025） |

## 放置位置

项目的代码同步采用黑名单式 rsync（排除 `weights/`、`data/`、`outputs/`、`logs/` 等目录）。判分器与测试集必须能够同步至服务器，因此放在 `scripts/ifbench/`。

## 复制的文件

| 文件 | 用途 |
| --- | --- |
| `instructions.py` | 58 个 IFBench OOD 约束的验证类 |
| `classic_instructions.py` | 25 个 IFEval 经典约束的验证类 |
| `instructions_registry.py` | `INSTRUCTION_DICT`：约束 id → 验证类 |
| `instructions_util.py` | 分词、句子切分等工具 |
| `__init__.py` | 包入口，含 `data_path()` |
| `data/IFBench_test.jsonl` | **泛化验证集**，299 条 held-out prompt（58 个全新约束） |

上游的 `run_eval.py` / `generate_responses.py` / `config.py`（需要 API key 的评测 CLI）未复制。本项目仅复用判分逻辑，另需实现 `eval_ifbench.py` 接入既有评测栈（该脚本尚未实现）。

## 改动记录

**无改动。** 与上游逐字节一致，升级时直接覆盖整个目录。

## 依赖

上游 `requirements.txt`：`absl-py langdetect nltk immutabledict spacy emoji syllapy>=0.8.0`。

本项目实际判分路径的依赖为 `absl-py emoji immutabledict langdetect nltk syllapy`（由源码 import 实测得出，未使用 spacy），已列入根目录 `requirements.txt`。
