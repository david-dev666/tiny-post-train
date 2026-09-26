#!/usr/bin/env bash
# 在 tmux 里启动训练看板。
#
# 用法：
#   bash scripts/start_dashboard.sh          # 起服务（默认 6006，自动带 token）
#   bash scripts/start_dashboard.sh stop     # 停掉
#   bash scripts/start_dashboard.sh token    # 只打印当前口令
#
# 为什么默认 6006：
#   AutoDL 把每个实例的 6006 / 6008 映射到了公网，手机浏览器可以直接打开，
#   不用隧道、不用装 App。服务只绑 127.0.0.1，该映射同样能转发进来（实测过）。
#
# 可覆盖的环境变量：
#   PORT        监听端口，默认 6006
#   TPT_TOKEN   访问口令，默认从 logs/dashboard.token 读，没有就生成
#   PYTHON      python 可执行文件，默认 python
#   SESSION     tmux 会话名，默认 dash
set -euo pipefail

SESSION="${SESSION:-dash}"
PORT="${PORT:-6006}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# 非交互 SSH 里 conda 不在 PATH，所以按顺序挑一个能用的 python，别写死
if [[ -z "${PYTHON:-}" ]]; then
    for cand in python python3 /root/miniconda3/bin/python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            PYTHON="$cand"
            break
        fi
    done
fi

if [[ -z "${PYTHON:-}" ]]; then
    echo "!! 找不到 python，请显式指定，例如 PYTHON=/root/miniconda3/bin/python3"
    exit 1
fi

TOKEN_FILE="${TOKEN_FILE:-$PROJECT_DIR/logs/dashboard.token}"

cd "$PROJECT_DIR"

is_up() { tmux has-session -t "$SESSION" 2>/dev/null; }

# 口令优先级：环境变量 > logs/dashboard.token > 现场生成并落盘
# 落盘是为了让手机上的书签在服务重启后依然有效。logs/ 已在 .gitignore 里。
resolve_token() {
    if [[ -n "${TPT_TOKEN:-}" ]]; then
        printf '%s' "$TPT_TOKEN"
        return
    fi
    if [[ -s "$TOKEN_FILE" ]]; then
        tr -d '[:space:]' < "$TOKEN_FILE"
        return
    fi
    mkdir -p "$(dirname "$TOKEN_FILE")"
    "$PYTHON" -c "import secrets; print(secrets.token_urlsafe(16))" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
    tr -d '[:space:]' < "$TOKEN_FILE"
}

case "${1:-start}" in
    stop)
        if is_up; then
            tmux kill-session -t "$SESSION"
            echo "==> 已停止看板会话 $SESSION"
        else
            echo "==> 看板没在跑"
        fi
        exit 0
        ;;
    token)
        echo "$(resolve_token)"
        exit 0
        ;;
    start) ;;
    *)
        echo "未知参数: $1（支持 start / stop / token）"
        exit 1
        ;;
esac

if [[ "${ENABLE_AUTH:-1}" != "1" ]]; then
    TOKEN=""
else
    TOKEN="$(resolve_token)"
fi

if is_up; then
    echo "==> 会话 $SESSION 已存在，先看日志： tmux attach -t $SESSION"
    [[ -n "$TOKEN" ]] && echo "    口令： $TOKEN"
    exit 0
fi

if ! "$PYTHON" -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "!! 缺少依赖，先装："
    echo "   $PYTHON -m pip install fastapi uvicorn"
    exit 1
fi

mkdir -p logs

# token 直接写进命令串，因为 tmux server 的环境变量不一定从当前 shell 继承
tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR" \
    "TPT_TOKEN='$TOKEN' $PYTHON dashboard/server.py --host 127.0.0.1 --port $PORT 2>&1 | tee -a logs/dashboard.log"

sleep 2

if ! is_up; then
    echo "!! 启动失败，看 logs/dashboard.log"
    exit 1
fi

SUFFIX=""
[[ -n "$TOKEN" ]] && SUFFIX="?token=$TOKEN"

echo "==> 看板已启动（tmux 会话 $SESSION，端口 $PORT）"
echo "    日志：logs/dashboard.log      进去看：tmux attach -t $SESSION"
echo
echo "==> 公网访问（域名从 AutoDL 实例卡片的「自定义服务」复制，6006 那条）："
echo "    https://<你的实例域名>/$SUFFIX"
echo
echo "==> 或者 Mac 走 SSH 隧道："
echo "    ssh -N -L $PORT:localhost:$PORT autodl"
echo "    http://localhost:$PORT/$SUFFIX"
echo
if [[ -z "$TOKEN" ]]; then
    echo "!! ENABLE_AUTH=0，没有鉴权。走公网映射时请不要这样开。"
fi
