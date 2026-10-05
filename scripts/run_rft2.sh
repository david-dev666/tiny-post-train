#!/bin/bash
# RFT 第二轮：**只改第一轮被证否的那两处**，其余逐项相同。
#
# 第一轮（sft-4b-rft1）实测失败，根因已定位（不是「RFT 这条路不行」）：
#   数据 = 1509 条，全是 GSM8K 自采样、output 中位 347 字符（纯长解答）
#   配置 = 3 epoch / lr 1e-4 / r=32
#   → 模型学成「什么题都写长解答」：ifollow 平均 64→222 token、GSM8K 175→925 token，
#     88% 的样本停不下来（用满上限）。乱码尾看着从 169/200 掉到 13/200，
#     但那是**没走到收尾决策**造成的假信号 —— 真走到收尾的样本里 69% 仍是乱码尾。
#     和 workflow.md 里 v3 那条是同一个病（「不是修好乱码尾，是换成了不会收尾」）。
#
# 这一轮只动两处：
#   1. 数据混入 alpaca（--mix-alpaca，第一轮从没被用过）→ 治「单域 → 只会写长解答」
#   2. 1 epoch / lr 2e-5（第一轮 3 epoch / 1e-4）        → 治「过训坍塌」
# 起点、采样候选池、筛选口径、LoRA rank、batch 全部保持不变，结果好坏都能归因。
#
# 验收判据（**硬门槛，先看这个再看别的指标**）——见 step_verdict：
#   A 不退化：用满上限的样本数不得超过 v2（第一轮 ifollow 19→161、GSM8K 10→1164，直接判死）
#   B 靶心真动：**自然收尾样本里**的乱码尾率要比 v2 低 10pp 以上
#     （总 junk_tail 会被「不收尾」刷低，不能单独用 —— 第一轮就是这么被骗的）
#   C 不倒退：指令遵循不低于 v2
#
# 用法（每步都能单独跑，也能一次跑完）：
#   bash scripts/run_rft2.sh data      # 1. 造数据（CPU，约 1 分钟）
#   bash scripts/run_rft2.sh train     # 2. 训练（约 10 分钟）
#   bash scripts/run_rft2.sh eval      # 3. 评测（约 5 分钟）
#   bash scripts/run_rft2.sh verdict   # 4. 判读（纯 CPU，几秒）
#   bash scripts/run_rft2.sh all       # 全流程
#
# 两个环境不能混：造数据/判读用 tpt，评测要 vllm（见 run_dpo.sh 顶部同一段说明）。

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY_VLLM=${TPT_VLLM_PYTHON:-/root/autodl-tmp/envs/vllm/bin/python}
PY_TPT=${TPT_PYTHON:-/root/miniconda3/envs/tpt/bin/python}

START=outputs/sft-4b-v2-merged          # 起点 = v2（项目当前主模型）
CAND=data/processed/rft/candidates.jsonl   # 与第一轮**同一个**候选池，不重采
DATA=data/processed/rft/sft2.jsonl
OUT=outputs/sft-4b-rft2
LABEL=sft-4b-rft2
MMLU=evals/mmlu-subset.jsonl            # 灾难性遗忘监控位（子集，快）

EPOCHS=${RFT2_EPOCHS:-1}
LR=${RFT2_LR:-2e-5}
MIX_N=${RFT2_MIX_N:-1500}               # 与 RFT 样本量约 1:1

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

step_data() {
  echo "== 1/4 造数据（候选池与第一轮相同，只加混合）=="
  echo "   起点 $START｜候选 $CAND"
  "$PY_TPT" scripts/make_rft_data.py \
    --candidates "$CAND" --output "$DATA" \
    --tokenizer weights/Qwen3-4B-Base \
    --mix-alpaca data/raw/alpaca-gpt4-zh --mix-n "$MIX_N" || return 1
}

step_train() {
  echo "== 2/4 训练（1 epoch / lr $LR / r=32）=="
  [ -f "$DATA" ] || { echo "!! 缺 $DATA，先跑：bash scripts/run_rft2.sh data"; return 1; }
  "$PY_TPT" scripts/train_sft.py \
    --model "$START" --data "$DATA" --output "$OUT" \
    --num-epochs "$EPOCHS" --lr "$LR" --lora-r 32 \
    --batch-size 4 --grad-accum 4 --max-seq-len 2048 \
    --save-steps 50 --probe-every 50 \
    --eval-ratio 0.02 --eval-steps 25 || return 1
  echo "   训练完：$OUT（probe 看 outputs/.../probes.jsonl，长度暴涨就是又退化了）"
}

step_eval() {
  echo "== 3/4 评测（同一引擎、同一套题，才和 v2 可比）=="
  [ -d "$OUT" ] || { echo "!! 缺 $OUT，先跑：bash scripts/run_rft2.sh train"; return 1; }
  "$PY_VLLM" scripts/eval_pipeline.py --engine vllm \
    --run "$LABEL=$START+$OUT" \
    --mmlu "$MMLU" --only mmlu,ifollow,gsm8k || return 1
}

step_verdict() {
  echo "== 4/4 判读（硬门槛）=="
  "$PY_TPT" - <<'PY' || return 1
import json, sys
from pathlib import Path

def load(label):
    p = Path(f"evals/results/{label}.json")
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def stats(d):
    out = {}
    for k in ("ifollow", "gsm8k"):
        s = d.get(k)
        if not s:
            out[k] = None
            continue
        recs = s["records"]
        nat = [x for x in recs if not x.get("truncated")]
        junk_nat = sum(1 for x in nat if x.get("junk_tail"))
        out[k] = dict(
            n=len(recs),
            trunc=s.get("truncated", 0),
            nat=len(nat),
            junk_nat=junk_nat,
            junk_rate=(junk_nat / len(nat)) if nat else float("nan"),
            junk=s.get("junk_tail", 0),
            rate=s.get("rate", s.get("accuracy")),
            rate_trim=s.get("rate_junk_trimmed"),
        )
    return out

v2, new = load("sft-4b-v2"), load("sft-4b-rft2")
if v2 is None or new is None:
    sys.exit("!! 缺 sft-4b-v2.json 或 sft-4b-rft2.json，先跑 eval")
a, b = stats(v2), stats(new)

print(f"{'':8s} {'用满上限':>10s} {'自然收尾':>10s} {'收尾里乱码尾':>14s} {'总junk':>8s} {'主指标':>9s}")
for k in ("ifollow", "gsm8k"):
    for tag, s in (("v2", a[k]), ("rft2", b[k])):
        print(f"{k+'/'+tag:8s} {s['trunc']:>10d} {s['nat']:>10d} "
              f"{s['junk_nat']:>6d} ({s['junk_rate']:>5.1%}) {s['junk']:>8d} {s['rate']:>8.3f}")
    print()

fails, passes = [], []
for k in ("ifollow", "gsm8k"):
    if b[k]["trunc"] > a[k]["trunc"]:
        fails.append(f"A 退化：{k} 用满上限 {a[k]['trunc']} → {b[k]['trunc']}（模型开始停不下来）")
for k in ("ifollow", "gsm8k"):
    drop = a[k]["junk_rate"] - b[k]["junk_rate"]
    tag = f"B {k} 自然收尾乱码尾率 {a[k]['junk_rate']:.1%} → {b[k]['junk_rate']:.1%}（{drop:+.1%}）"
    (passes if drop >= 0.10 else fails).append(tag + "" if drop >= 0.10 else "  ← 未达 10pp")
if b["ifollow"]["rate"] < a["ifollow"]["rate"]:
    fails.append(f"C 倒退：指令遵循 {a['ifollow']['rate']:.3f} → {b['ifollow']['rate']:.3f}")
else:
    passes.append(f"C 指令遵循 {a['ifollow']['rate']:.3f} → {b['ifollow']['rate']:.3f}")
if b["gsm8k"]["rate"] < a["gsm8k"]["rate"] - 0.01:
    fails.append(f"C 倒退：GSM8K {a['gsm8k']['rate']:.4f} → {b['gsm8k']['rate']:.4f}")
else:
    passes.append(f"C GSM8K {a['gsm8k']['rate']:.4f} → {b['gsm8k']['rate']:.4f}")

print("满足：")
for p in passes:
    print("  ✓", p)
print("不满足：")
for f in fails:
    print("  ✗", f)
print()
if fails:
    print("★ 结论：**这一轮仍然失败** —— 按 .codebuddy 规则先写进 notes/workflow.md 的失败记录，再谈下一轮")
    sys.exit(1)
print("★ 结论：**通过** —— 这是项目第一次「训练侧」真正动到收尾靶心且不退化")
PY
}

case "${1:-}" in
  data)    step_data ;;
  train)   step_train ;;
  eval)    step_eval ;;
  verdict) step_verdict ;;
  all)
    step_data    || exit 1
    step_train   || exit 1
    step_eval    || exit 1
    step_verdict || exit 1
    ;;
  *) sed -n '24,32p' "$0"; exit 1 ;;
esac
