#!/usr/bin/env bash
# GRPO 训练结束后自动接力：等它退出 → 合并 adapter → 五模型同一判分栈评测 → 出报告。
#
# 为什么需要它：GRPO 单轮要跑 3~4 小时，训练结束的时刻没法盯着；
# 而评测必须**等训练真正释放显存**才能起（32G 单卡塞不下「训练 + 评测」两个 4B）。
#
# 用法（服务器上，tmux 里）：
#   tmux new -d -s eval "cd /root/autodl-tmp/tiny-post-train && bash scripts/after_grpo_eval.sh"
#
# 产物：
#   outputs/grpo-4b-merged/        合并后的完整模型（评测与部署用）
#   evals/results/grpo-4b.json     该模型的评测结果
#   evals/results/report.html      全模型报告；同时同步 docs/index.html
#   logs/after-grpo-eval.log       这一整套的日志
set -uo pipefail

cd /root/autodl-tmp/tiny-post-train

VENV=/root/autodl-tmp/envs/vllm/bin/python   # 有 vllm（评测 + 合并都用得到 peft）
TPT=/root/miniconda3/envs/tpt/bin/python     # 有 unsloth（train_dpo.py --export-merged 要用它）
RUN="${RUN:-outputs/grpo-4b-ifrlvr}"
MERGED="${MERGED:-outputs/grpo-4b-merged}"
LOG=logs/after-grpo-eval.log

exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo "=== $(date '+%F %T') 等待 GRPO 训练结束 ==="

waited=0
while pgrep -f "train_grpo.py" > /dev/null 2>&1; do
    sleep 30
    waited=$((waited + 30))
    # 每 10 分钟报一次，免得看起来像卡住
    if [ $((waited % 600)) -eq 0 ]; then
        echo "    ...仍在训练（已等 $((waited / 60)) 分钟）"
    fi
done
echo "=== $(date '+%F %T') 训练进程已退出（等了 $((waited / 60)) 分钟）==="

# 判据：adapter 落盘才算成功。训练崩掉时没有这个文件，别拿半成品去评测。
if [ ! -f "$RUN/adapter_model.safetensors" ]; then
    echo "!! $RUN/adapter_model.safetensors 不存在 —— 训练可能失败，终止接力。"
    echo "!! 先看 logs/grpo-train.log 尾部。"
    exit 1
fi
echo "==> adapter 已落盘：$(ls -la "$RUN/adapter_model.safetensors" | awk '{print $5}') bytes"

# 训练进程退出后显存不一定立刻释放，等一等再合并
sleep 20

echo "=========================================================="
echo "=== $(date '+%F %T') 合并 adapter → $MERGED ==="
rm -rf "$MERGED"
# 复用 train_dpo.py 的 --export-merged（已验证过：把 base + adapter 合并并写入干净 chat template）
$TPT scripts/train_dpo.py \
    --model outputs/dpo-4b-open-merged \
    --sft-adapter "$RUN" \
    --export-merged "$MERGED"
if [ ! -f "$MERGED/config.json" ]; then
    echo "!! 合并失败，终止。"
    exit 1
fi
echo "==> 合并完成：$(du -sh "$MERGED" | cut -f1)"

echo "=========================================================="
echo "=== $(date '+%F %T') 五模型全量评测（同一判分栈）==="
# 关键：**整批重跑**（--force），不拿历史 json 拼表 —— 判分口径变了之后跨版本并列不成立。
$VENV scripts/eval_pipeline.py --engine vllm --force \
    --mmlu evals/mmlu-full.jsonl --only all --extra=--mmlu-style=both \
    --run base-4b=weights/Qwen3-4B-Base \
    --run sft-4b-v2=outputs/sft-4b-v2-merged \
    --run dpo-4b-open=outputs/sft-4b-v2-merged+outputs/dpo-4b-open \
    --run instruct-2507=weights/Qwen3-4B-Instruct-2507 \
    --run grpo-4b="$MERGED"

echo "=========================================================="
echo "=== $(date '+%F %T') 全部完成 ==="
echo "报告：evals/results/report.html（已同步 docs/index.html）"
echo "下一步（在 Mac 上拉回产出）："
echo "  rsync -avz westb:/root/autodl-tmp/tiny-post-train/evals/results/ ./evals/results/"
echo "  rsync -avz westb:/root/autodl-tmp/tiny-post-train/docs/ ./docs/"
