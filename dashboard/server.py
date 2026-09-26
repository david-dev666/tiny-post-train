"""训练看板后端。

只做一件事：把训练脚本写出的 metrics.jsonl 读出来，通过 HTTP 暴露给前端。

用法：
    python dashboard/server.py --port 8000

为什么绑 127.0.0.1：
    服务器上只监听本机，公网访问不到，安全。
    Mac 上用 SSH 隧道连过来：
        ssh -N -L 8000:localhost:8000 autodl
    然后浏览器打开 http://localhost:8000

数据来源（由 scripts/train_sft.py 的 MetricsLogger 写出）：
    outputs/<run>/metrics.jsonl    每行一条指标，追加写
    outputs/<run>/run_meta.json    训练开始的元信息（总步数、起跑时间等）
"""

from __future__ import annotations

import argparse
import csv
import hmac
import json
import os
import re
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUTS_DIR = Path(os.environ.get("TPT_OUTPUTS") or (ROOT / "outputs"))

# run 名会拼进文件路径，必须严格校验，防止 ../ 越权读文件
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._\-\u4e00-\u9fff]+$")

ACTIVE_WINDOW_SEC = 300  # 指标文件 5 分钟内更新过，就算这个 run 还活着

# 设了就开启鉴权。走 AutoDL 6006 公网映射时必须设，否则谁拿到链接谁都能看。
TOKEN = os.environ.get("TPT_TOKEN", "").strip()

app = FastAPI(title="tiny-post-train dashboard")


@app.middleware("http")
async def require_token(request: Request, call_next):
    """token 从 X-Token 头或 ?token= 查询参数取。

    ?token= 是为了让浏览器首次打开页面就能带上；页面加载后前端会把 token
    存进 localStorage，后续请求改用 X-Token 头。
    """
    if not TOKEN:
        return await call_next(request)
    supplied = request.headers.get("x-token") or request.query_params.get("token") or ""
    if not hmac.compare_digest(supplied, TOKEN):
        return JSONResponse({"detail": "token 不对"}, status_code=401)
    return await call_next(request)


# ---------------------------------------------------------------- 数据读取


def _run_dir(run: str) -> Path:
    if not RUN_NAME_RE.match(run):
        raise HTTPException(status_code=400, detail="非法 run 名")
    path = OUTPUTS_DIR / run
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"找不到 run: {run}")
    return path


def _read_records(path: Path) -> list[dict]:
    """逐行读 jsonl。坏行直接跳过——训练正在写时最后一行可能是半行。"""
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _read_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------- 系统状态


def _gpu_info() -> dict:
    """直接用 nvidia-smi 拿 GPU 状态。无卡模式下这里会失败，属于正常。"""
    fields = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": str(exc)[:200]}

    if proc.returncode != 0:
        return {"available": False, "reason": (proc.stderr or "nvidia-smi 失败").strip()[:200]}

    gpus = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue

        def num(text: str):
            try:
                return float(text)
            except ValueError:
                return None

        gpus.append(
            {
                "name": parts[0],
                "util": num(parts[1]),
                "mem_used": num(parts[2]),
                "mem_total": num(parts[3]),
                "temp": num(parts[4]),
                "power": num(parts[5]),
            }
        )
    return {"available": bool(gpus), "gpus": gpus}


def _trainer_processes() -> dict:
    """看看还有没有在跑的 train_sft.py。"""
    try:
        proc = subprocess.run(["pgrep", "-af", "train_sft.py"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return {"alive": False, "procs": []}
    procs = [line for line in proc.stdout.strip().splitlines() if line]
    return {"alive": bool(procs), "procs": procs[:5]}


# ---------------------------------------------------------------- 接口


@app.get("/")
def index():
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail=f"缺少前端文件: {page}")
    return FileResponse(page)


@app.get("/api/runs")
def list_runs():
    runs = []
    if OUTPUTS_DIR.is_dir():
        for child in sorted(OUTPUTS_DIR.iterdir()):
            metrics = child / "metrics.jsonl"
            if not (child.is_dir() and metrics.exists()):
                continue
            records = _read_records(metrics)
            last = records[-1] if records else {}
            # 最后一条可能是 eval 记录（只有 eval_loss），所以往回找最后一次 train loss
            last_train = next((r for r in reversed(records) if "loss" in r), {})
            age = time.time() - metrics.stat().st_mtime
            runs.append(
                {
                    "name": child.name,
                    "records": len(records),
                    "last_step": last.get("step"),
                    "last_loss": last_train.get("loss"),
                    "updated_ago": round(age, 1),
                    "active": age < ACTIVE_WINDOW_SEC,
                }
            )
    return {"runs": runs, "outputs_dir": str(OUTPUTS_DIR), "now": int(time.time())}


@app.get("/api/metrics")
def get_metrics(run: str = Query(..., description="run 名，即 outputs 下的目录名"), offset: int = 0):
    """增量返回。

    offset = 前端已经拿到多少条记录。因为 jsonl 只追加不修改，
    返回 records[offset:] 就够，省流量也省序列化。
    """
    run_dir = _run_dir(run)
    records = _read_records(run_dir / "metrics.jsonl")
    offset = max(0, offset)
    # bench.jsonl 记录不多（每 N 步一条），整份返回；by_subject 太大，前端用不到，丢掉
    bench = [
        {
            "step": r.get("step"),
            "ts": r.get("ts"),
            "accuracy": r.get("accuracy"),
            "correct": r.get("correct"),
            "total": r.get("total"),
            "sec": r.get("sec"),
        }
        for r in _read_records(run_dir / "bench.jsonl")
    ]
    return {
        "run": run,
        "total": len(records),
        "offset": offset,
        "records": records[offset:],
        "bench": bench,
        "meta": _read_meta(run_dir),
        "now": int(time.time()),
    }


@app.get("/api/status")
def get_status():
    return {
        "now": int(time.time()),
        "gpu": _gpu_info(),
        "trainer": _trainer_processes(),
        "outputs_dir": str(OUTPUTS_DIR),
    }


# ---------------------------------------------------------------- 数据抽样

_SAMPLE_CACHE: dict = {}


# 数据文件格式，按优先级排：真正的数据在前，配置在后
DATA_EXTENSIONS = ("csv", "jsonl", "json", "parquet")
METADATA_FILENAMES = {"dataset_infos.json", "dataset_info.json", ".gitattributes"}


def _looks_like_metadata(path: Path) -> bool:
    """跳过随数据集一起下下来的元信息文件。

    alpaca 在 modelscope 上的快照里，真实数据是 train.csv，
    两个 json 分别是字段 schema 和指向 train.csv 的配置。
    不排除的话会挑中 116 字节的配置文件，抽样出来是空的。
    """
    if path.name in METADATA_FILENAMES or path.name.startswith("README"):
        return True
    if path.suffix.lower() != ".json":
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            head = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    if not isinstance(head, dict):
        return False
    # 真正的指令数据顶层是数组；顶层是「字典套字典」的基本都是配置
    values = list(head.values())
    if not values or not all(isinstance(v, dict) for v in values):
        return False
    inner = [item for value in values for item in value.values()]
    return bool(inner) and all(isinstance(item, dict) for item in inner)


def _pick_data_file(path: Path) -> Path | None:
    """数据可能是一个目录，也可能直接是一个文件。取第一个像数据的。"""
    if path.is_file():
        return path
    if path.is_dir():
        for ext in DATA_EXTENSIONS:
            for candidate in sorted(path.glob(f"*.{ext}")):
                if not _looks_like_metadata(candidate):
                    return candidate
    return None


def _read_samples(path: Path, n: int) -> list:
    """只读前 n 条，不把整个数据集读进内存。

    csv 逐行进、jsonl 逐行解析、parquet 只读前 n 行，都是够数就停。
    只有 json 是数组，必须整体 parse，所以外面加了 mtime 缓存。
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        out = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                out.append({key: (value or "").strip() for key, value in row.items()})
                if len(out) >= n:
                    break
        return out

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return []
        return pq.read_table(path).slice(0, n).to_pylist()

    out = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        if suffix in (".jsonl", ".ndjson"):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(out) >= n:
                    break
            return out

        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
        return data[:n] if isinstance(data, list) else []


@app.get("/api/samples")
def get_samples(run: str = Query(...), n: int = 3):
    """从这次训练用的数据里抽几条，看看到底喂了什么进去。"""
    run_dir = _run_dir(run)
    meta = _read_meta(run_dir)
    data_ref = meta.get("data")
    if not data_ref:
        raise HTTPException(status_code=404, detail="run_meta.json 里没有 data 字段")

    path = Path(data_ref)
    if not path.is_absolute():
        path = ROOT / path

    data_file = _pick_data_file(path)
    if data_file is None:
        raise HTTPException(status_code=404, detail=f"找不到数据文件: {path}")

    n = max(1, min(n, 20))
    key = (str(data_file), data_file.stat().st_mtime, n)
    if _SAMPLE_CACHE.get("key") != key:
        _SAMPLE_CACHE["key"] = key
        _SAMPLE_CACHE["data"] = _read_samples(data_file, n)

    return {
        "run": run,
        "data_ref": data_ref,
        "file": str(data_file),
        "size_mb": round(data_file.stat().st_size / 1024 / 1024, 1),
        "samples": _SAMPLE_CACHE["data"],
    }


# ---------------------------------------------------------------- 推理采样


@app.get("/api/probes")
def get_probes(run: str = Query(...)):
    """训练脚本每 N 步用固定 prompt 生成一次，这里按问题分组返回。

    同一个问题在训练过程中的多次输出放在一起，才能看出模型是不是真的在变好。
    """
    run_dir = _run_dir(run)
    path = run_dir / "probes.jsonl"
    if not path.exists():
        return {"run": run, "total": 0, "latest": [], "file": str(path)}

    grouped: dict[str, list] = {}
    for record in _read_records(path):
        grouped.setdefault(str(record.get("prompt", "")), []).append(record)

    latest = []
    for prompt, items in grouped.items():
        items.sort(key=lambda r: r.get("step", 0))
        latest.append(
            {
                "prompt": prompt,
                "count": len(items),
                "current": items[-1],
                "previous": items[-2] if len(items) > 1 else None,
            }
        )
    latest.sort(key=lambda p: p["prompt"])
    return {
        "run": run,
        "total": sum(p["count"] for p in latest),
        "latest": latest,
        "file": str(path),
    }


# ---------------------------------------------------------------- 入口


def main():
    parser = argparse.ArgumentParser(description="tiny-post-train 训练看板")
    parser.add_argument("--host", default="127.0.0.1", help="默认只监听本机，配合 SSH 隧道使用")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--outputs", default=None, help="outputs 目录，默认项目根下的 outputs")
    parser.add_argument("--token", default=None, help="访问口令，留空则用环境变量 TPT_TOKEN；都为空则不鉴权")
    args = parser.parse_args()

    global OUTPUTS_DIR, TOKEN
    if args.outputs:
        OUTPUTS_DIR = Path(args.outputs).resolve()
    if args.token is not None:
        TOKEN = args.token.strip()

    import uvicorn

    print(f"看板启动中： http://{args.host}:{args.port}")
    print(f"数据目录： {OUTPUTS_DIR}")
    if TOKEN:
        print(f"鉴权已开启，带 token 访问： http://{args.host}:{args.port}/?token={TOKEN}")
        if args.port == 6006:
            print("公网地址见 AutoDL 实例卡片的「自定义服务」，把 ?token=... 接在后面")
    else:
        print("!! 未设 token，任何能访问到这个端口的人都能看到数据")
    print("Mac 上不走公网的话也可以建隧道： ssh -N -L %d:localhost:%d autodl" % (args.port, args.port))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
