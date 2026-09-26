"""从 MMLU 里抽一份固定的子集，供训练中做快速评测。

为什么要子集：
    MMLU 全量 14042 题，4B 模型上跑一遍要一两个小时，比训练还慢。
    分层抽 456 题（57 学科 × 8 题）后，一次只要几十秒到几分钟，
    每 50~100 步跑一次才现实。

为什么必须固定：
    每次随机抽新题的话，题目难度本身就在变，曲线抖动会被误读成模型退步。
    所以只抽一次、落盘，之后所有训练复用同一份文件。

为什么不用 datasets 库下载：
    实测在 AutoDL 上跑 `datasets.load_dataset("cais/mmlu")` 会卡在 20MiB 不动，
    而同一个镜像站的 parquet 直链是通的。所以这里自己用 HTTP 拉，
    顺带能断点续传、能看清进度。

用法：
    python scripts/bench_subset.py
    python scripts/bench_subset.py --per-subject 12 --seed 3407
    python scripts/bench_subset.py --source /path/to/mmlu.parquet        # 用本地文件
    python scripts/bench_subset.py --hf-endpoint https://huggingface.co  # 换端点

产物：
    evals/mmlu-subset.jsonl   每行一题，提交进版本库
    data/raw/mmlu/*.parquet   下载缓存，已被 gitignore
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

DEFAULT_SOURCE = "cais/mmlu"
DEFAULT_HF_PATH = "all/test-00000-of-00001.parquet"
# AutoDL 这类国内机器直连 huggingface.co 会超时，hf-mirror 是常用镜像
DEFAULT_ENDPOINT = "https://hf-mirror.com"


# ------------------------------------------------------------------ 读取


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_parquet(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("!! 读 parquet 需要 pyarrow：pip install pyarrow")
    return pq.read_table(path).to_pylist()


def read_records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return _read_jsonl(path)
    if suffix == ".parquet":
        return _read_parquet(path)
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    sys.exit(f"!! 不认识的格式：{path}（支持 .jsonl / .json / .parquet）")


# ------------------------------------------------------------------ 下载


def download(url: str, dest: Path, attempts: int = 3) -> Path:
    """带断点续传的下载。镜像站偶尔会在整 MiB 处断流，重试能接上。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, attempts + 1):
        offset = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(url)
        if offset:
            request.add_header("Range", f"bytes={offset}-")
            print(f"    第 {attempt}/{attempts} 次尝试，从 {offset / 1024 / 1024:.1f} MB 续传")

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                # 服务端不支持 Range 却回了 200，那就得从头来
                if offset and response.status != 206:
                    offset = 0
                mode = "ab" if offset else "wb"
                total = response.headers.get("Content-Length")
                total = offset + int(total) if total else None
                written = offset
                last_report = time.time()
                with part.open(mode) as out:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        written += len(chunk)
                        if time.time() - last_report >= 3:
                            last_report = time.time()
                            if total:
                                print(
                                    f"    {written / 1024 / 1024:.1f} / "
                                    f"{total / 1024 / 1024:.1f} MB"
                                )
                            else:
                                print(f"    {written / 1024 / 1024:.1f} MB")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"    中断：{exc!r}")
            continue

        if total is not None and part.stat().st_size < total:
            print(f"    传完了但大小不对（{part.stat().st_size} < {total}），重试")
            continue

        part.replace(dest)
        return dest

    sys.exit(f"!! 下载失败：{url}\n   文件留在 {part}，可以重跑本脚本继续续传")


# ------------------------------------------------------------------ 组装


def load_mmlu(source: str, hf_path: str, endpoint: str, download_dir: Path) -> list[dict]:
    local = Path(source)
    if local.exists():
        print(f"==> 读本地文件: {local}")
        return read_records(local)

    url = f"{endpoint.rstrip('/')}/datasets/{source}/resolve/main/{hf_path}"
    cached = download_dir / Path(hf_path).name

    print(f"==> 数据源: {url}")
    if cached.exists():
        print(f"    命中缓存: {cached}（{cached.stat().st_size / 1024 / 1024:.1f} MB）")
    else:
        print(f"    下载到: {cached}")
        download(url, cached)

    return read_records(cached)


def build_subset(records: list[dict], per_subject: int, seed: int) -> tuple[list[dict], dict]:
    by_subject: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for row in records:
        choices = row.get("choices") or []
        if len(choices) != 4 or not isinstance(row.get("answer"), int) or not row.get("question"):
            skipped += 1
            continue
        by_subject[row.get("subject") or "unknown"].append(row)

    if not by_subject:
        sys.exit("!! 一条可用样本都没有，检查数据格式")

    rng = random.Random(seed)
    picked: list[dict] = []
    counts: dict[str, int] = {}
    for subject in sorted(by_subject):
        items = list(by_subject[subject])
        rng.shuffle(items)
        chosen = items[:per_subject]
        counts[subject] = len(chosen)
        for row in chosen:
            picked.append(
                {
                    "subject": subject,
                    "question": row["question"],
                    "choices": list(row["choices"]),
                    "answer": int(row["answer"]),
                }
            )

    picked.sort(key=lambda r: (r["subject"], r["question"]))
    stats = {
        "total": len(picked),
        "subjects": len(counts),
        "per_subject": per_subject,
        "seed": seed,
        "skipped": skipped,
    }
    return picked, stats


def main():
    parser = argparse.ArgumentParser(description="抽一份固定的 MMLU 子集")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help=f"本地文件（json/jsonl/parquet）或 HF 数据集 id，默认 {DEFAULT_SOURCE}")
    parser.add_argument("--hf-path", default=DEFAULT_HF_PATH,
                        help=f"仓库内 parquet 路径，默认 {DEFAULT_HF_PATH}")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT,
                        help=f"HuggingFace 镜像端点，默认 {DEFAULT_ENDPOINT}")
    parser.add_argument("--download-dir", default="data/raw/mmlu",
                        help="parquet 缓存目录，默认 data/raw/mmlu（已 gitignore）")
    parser.add_argument("--out", default="evals/mmlu-subset.jsonl")
    parser.add_argument("--per-subject", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--force", action="store_true", help="已存在也重新抽")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        existing = sum(1 for line in out.open(encoding="utf-8") if line.strip())
        print(f"==> {out} 已存在（{existing} 题），跳过。要重抽加 --force")
        return

    records = load_mmlu(args.source, args.hf_path, args.hf_endpoint, Path(args.download_dir))
    print(f"==> 原始样本: {len(records)}")

    picked, stats = build_subset(records, args.per_subject, args.seed)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in picked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"==> 已写出 {out}\n"
        f"    {stats['total']} 题 / {stats['subjects']} 学科"
        f"（每科 {stats['per_subject']} 题，seed={stats['seed']}）"
    )
    if stats["skipped"]:
        print(f"    跳过 {stats['skipped']} 条（选项数不是 4 或字段缺失）")
    print("    随机猜的基准线是 25%，曲线从这附近起步是正常的")
    print("    这份文件要提交进版本库，SFT / DPO / GRPO 三阶段的分数才可比")


if __name__ == "__main__":
    main()
