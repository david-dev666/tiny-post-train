"""模型评测 pipeline：一条命令把多个模型跑完整套评测，再生成 HTML 报告。

为什么要有它
------------
单跑一次 `eval.py` 只能得到一个模型的数字。但项目真正要回答的问题是
「**基座 / 自训 SFT / 官方 instruct 谁更好、差在哪**」—— 那需要：

1. 同一套评测、同一批题目、同一套生成参数，跑多个模型（否则不可比）
2. 中间任何一个模型崩了，不能把整批带下去
3. 跑完直接把结果摆成一张对照表，而不是让人去翻 json

这三件事就是本脚本的全部内容。

用法
----
    # 一次跑三个模型，跑完自动出报告
    python scripts/eval_pipeline.py \
        --run base-4b=weights/Qwen3-4B-Base \
        --run sft-4b-v2=weights/Qwen3-4B-Base+outputs/sft-4b-v2 \
        --run instruct-2507=weights/Qwen3-4B-Instruct-2507 \
        --mmlu evals/mmlu-full.jsonl --only all

    # 只重跑其中一个，跑完仍然出完整报告（报告读的是 evals/results/ 下全部结果）
    python scripts/eval_pipeline.py --run sft-4b-v2=weights/Qwen3-4B-Base+outputs/sft-4b-v2 --force

    # 换 vLLM 引擎（快 16.7×）。注意要整批重跑，别只补一部分 ——
    # 两个引擎的分数不能混着比
    python scripts/eval_pipeline.py --engine vllm --force \
        --run base-4b=weights/Qwen3-4B-Base \
        --run sft-4b-v2=weights/Qwen3-4B-Base+outputs/sft-4b-v2 \
        --run instruct-2507=weights/Qwen3-4B-Instruct-2507 \
        --mmlu evals/mmlu-full.jsonl --only all

`--run` 的格式是 `标签=模型路径`，要挂 adapter 就在后面接 `+适配器路径`。
用 `+` 而不是 `:`，因为路径里本来就有冒号的情况更常见。

设计取舍
--------
- **每个模型一个子进程**：单个模型 OOM / 报错不会带走整批，日志各自独立
- **默认跳过已完成的结果，但要看引擎**：12 小时的批量任务必须能断点续跑；
  而「已存在」不只看项全不全，还要看它是不是**同一个引擎**跑的（见
  `result_is_complete`）—— 否则一批结果里会悄悄混进两把尺子
- **评测口径全部落在 eval.py 里**：本脚本不碰判分，只做调度，避免出现两套标准
- **按引擎挑解释器**：vLLM 要独立环境（见 `VLLM_PYTHON` 的注释）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_DIR)
SCRIPTS = PROJECT_DIR / "scripts"
RESULTS = PROJECT_DIR / "evals" / "results"
LOGS = PROJECT_DIR / "logs"

# 报告里用什么名字显示每个评测项（顺序即表格顺序）
SUITE_TITLES = {
    "mmlu_gen": "MMLU（生成式）",
    "ifollow": "指令遵循",
    "gsm8k": "GSM8K（数学）",
    "humaneval": "HumanEval（代码）",
}
# 只收集回答、没有分数的项
COLLECT_ONLY = {"openqa": "OpenQA（待裁判）", "probes": "固定话题采样"}

# vLLM 必须跑在**独立环境**里：它要 torch 2.13 + transformers 5.18，
# 而训练环境（unsloth）锁在 torch 2.12.1 —— 装不到一起，硬装会把训练搞坏。
# 所以 pipeline 得按引擎挑解释器，不能一律用 `sys.executable`。
VLLM_PYTHON = os.environ.get("TPT_VLLM_PYTHON", "/root/autodl-tmp/envs/vllm/bin/python")


def parse_run(spec: str) -> tuple[str, str, str | None]:
    """把 `标签=模型路径[+adapter路径]` 拆开。"""
    if "=" not in spec:
        sys.exit(f"!! --run 格式应为 `标签=模型路径[+adapter]`，收到：{spec}")
    label, target = spec.split("=", 1)
    label = label.strip()
    if "+" in target:
        model, adapter = target.split("+", 1)
    else:
        model, adapter = target, None
    if not label:
        sys.exit(f"!! --run 缺少标签：{spec}")
    return label, model.strip(), (adapter.strip() if adapter else None)


def result_is_complete(path: Path, wanted: set[str], engine: str) -> bool:
    """已完成的结果里，请求的每一项都应该在，**且是同一个引擎跑的**。

    引擎这一条不能省：HF 和 vLLM 在 bf16 下是两条数值路径，
    50 题里逐题预测有 13 题会翻转。只看「项全不全」会把一份 HF 的结果
    当成「已经跑过了」跳过 —— 于是一批结果里混着两把尺子，
    而且没有任何迹象能看出来。
    """
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    # 老结果没有 engine 字段（那时候只有 hf），按 hf 算
    if (data.get("engine") or "hf") != engine:
        return False
    for name in wanted:
        key = {"mmlu": "mmlu_gen", "probes": "probes"}.get(name, name)
        if key not in data:
            return False
    return True


def run_one(label: str, model: str, adapter: str | None, args) -> bool:
    command = [
        args.python, str(SCRIPTS / "eval.py"),
        "--engine", args.engine,
        "--model", model,
        "--label", label,
        "--mmlu", args.mmlu,
        "--only", args.only,
    ]
    if adapter:
        command += ["--adapter", adapter]
    for extra in args.extra:
        command.append(extra)

    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"eval-{label}.log"
    print(f"\n{'=' * 70}")
    print(f">>> {label}")
    print(f"    模型   : {model}" + (f"\n    adapter: {adapter}" if adapter else ""))
    print(f"    日志   : {log_path}")
    print(f"    命令   : {' '.join(command)}")
    print("=" * 70, flush=True)

    began = time.time()
    environment = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_DIR), env=environment,
        )
    spent = time.time() - began
    status = "成功" if proc.returncode == 0 else f"失败（退出码 {proc.returncode}）"
    print(f"<<< {label} {status}，用时 {spent / 60:.1f} 分钟", flush=True)
    if proc.returncode != 0:
        print(f"    看日志尾部：tail -30 {log_path}")
    return proc.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="多模型评测 pipeline（跑完自动出 HTML 报告）")
    parser.add_argument("--run", action="append", default=[], required=True,
                        metavar="标签=模型路径[+adapter]",
                        help="要评测的模型，可重复传多次")
    parser.add_argument("--mmlu", default="evals/mmlu-subset.jsonl",
                        help="默认小份子集；出正式结论用 evals/mmlu-full.jsonl")
    parser.add_argument("--only", default="all",
                        help="沿用 eval.py 的 --only：all / mmlu / ifollow / gsm8k / humaneval / openqa / probes")
    parser.add_argument("--extra", action="append", default=[],
                        help="额外透传给 eval.py 的参数，如 --extra=--limit=20")
    parser.add_argument("--force", action="store_true",
                        help="已完成的结果也重跑（默认跳过，方便断点续跑）")
    parser.add_argument("--parallel", type=int, default=1,
                        help="同时跑几个模型（默认 1 串行）。24G 卡建议 2，再多会抢带宽并逼近显存上限")
    parser.add_argument("--engine", default="vllm", choices=["hf", "vllm"],
                        help="生成后端。vllm=批量推理快 16.7×（默认，需要独立环境，"
                             "见 TPT_VLLM_PYTHON）；hf=unsloth（需要 tpt 环境，慢）。"
                             "**两个引擎的分数不能混着比**，换引擎要整批重跑")
    parser.add_argument("--python", default=None,
                        help="跑 eval.py 用的解释器。不给就按 --engine 自动挑")
    parser.add_argument("--report", action="store_true", default=True,
                        help="跑完生成 HTML 报告（默认生成）")
    parser.add_argument("--no-report", dest="report", action="store_false")
    args = parser.parse_args()
    if not args.python:
        args.python = VLLM_PYTHON if args.engine == "vllm" else sys.executable
    if args.engine == "vllm" and not Path(args.python).exists():
        sys.exit(f"!! 找不到 vLLM 环境的解释器：{args.python}\n"
                 f"   用 TPT_VLLM_PYTHON 指定，或 --python 直接给路径，"
                 f"或改 --engine hf（慢 16.7×）")
    if args.engine == "vllm" and args.parallel > 1:
        # vLLM 单进程就会把整张卡喂满（continuous batching 自己会调度），
        # 再并行只会互相抢显存。这和 HF 路径（单进程只有 9% 利用率）正好相反。
        print("!! vLLM 单进程已能喂满 GPU，--parallel 强制按 1 处理")
        args.parallel = 1

    runs = [parse_run(spec) for spec in args.run]
    wanted = {p.strip() for p in args.only.split(",") if p.strip()}
    # "all" 必须先展开：下面判断「结果是否完整」是按项名去 json 里找 key 的，
    # 留着字面量 "all" 会导致永远判定为不完整 —— 12 小时的任务一崩就得从头跑
    if "all" in wanted:
        wanted = {"mmlu", "ifollow", "probes", "gsm8k", "humaneval", "openqa"}

    print(f"计划评测 {len(runs)} 个模型（引擎 {args.engine}）：")
    for label, model, adapter in runs:
        print(f"  - {label:<16} {model}" + (f" + {adapter}" if adapter else ""))
    print(f"评测集：{args.mmlu}　评测项：{sorted(wanted)}")
    print(f"解释器：{args.python}")
    print(f"结果目录：{RESULTS}")

    overall = time.time()
    outcomes = {}
    pending = []
    for label, model, adapter in runs:
        out_path = RESULTS / f"{label}.json"
        if not args.force and result_is_complete(out_path, wanted, args.engine):
            print(f"\n>>> {label} 已有完整结果，跳过。要重跑加 --force")
            outcomes[label] = "跳过（已有结果）"
            continue
        pending.append((label, model, adapter))

    # 显存足够时并行跑多个模型：单进程生成只吃 8~9G，24G 卡塞得下 2 个。
    # 注意生成是**显存带宽瓶颈**，并行两份不会真的快一倍，实测约 1.6 倍。
    # 3 个以上会开始抢带宽 + 逼近显存上限，不建议。
    if args.parallel > 1 and len(pending) > 1:
        print(f"\n>>> 并行跑 {len(pending)} 个模型（并发 {args.parallel}）")
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(run_one, label, model, adapter, args): label
                for label, model, adapter in pending
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    outcomes[label] = "成功" if future.result() else "失败"
                except Exception as exc:  # 单个模型崩了不影响其它
                    outcomes[label] = f"异常（{exc}）"
    else:
        for label, model, adapter in pending:
            outcomes[label] = "成功" if run_one(label, model, adapter, args) else "失败"

    print(f"\n{'=' * 70}\n全部完成，总用时 {(time.time() - overall) / 60:.1f} 分钟")
    for label, status in outcomes.items():
        print(f"  {label:<18} {status}")

    if args.report:
        report = SCRIPTS / "make_report.py"
        if not report.exists():
            print(f"\n!! 没找到 {report}，跳过报告生成")
            return
        print("\n>>> 生成 HTML 报告")
        subprocess.run(
            [sys.executable, str(report), "--results", str(RESULTS)],
            cwd=str(PROJECT_DIR),
        )


if __name__ == "__main__":
    main()
