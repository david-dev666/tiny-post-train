#!/bin/bash
# DPO 全流程。四个步骤分开跑，每步都能单独看结果 ——
# 30 分钟起步的任务做成黑盒，崩了只能靠「几点开始」估（这个项目吃过这个亏）。
#
#   bash scripts/run_dpo.sh check    0. 冒烟：验三个 API 假设（约 1 分钟）★ 开机先跑
#   bash scripts/run_dpo.sh data     1. 造偏好数据（约 15 分钟）
#   bash scripts/run_dpo.sh probe    2. 上界探针（约 2 分钟）★ 决定值不值得训
#   bash scripts/run_dpo.sh train    3. DPO 训练（约 30 分钟）+ 导出合并模型
#   bash scripts/run_dpo.sh eval     4. 评测 + 报告（约 10 分钟）
#
# 靶心 = 「收尾那一个 token」：模型答完内容后不肯干净结束，而是吐一个乱码 token。
#   prompt   : 请只回答一个数字，不要任何其他内容
#   chosen   : 1024
#   rejected : 1024. לחלוט        ← 只差末尾一个 token
# 设计理由见 scripts/make_dpo_data.py 的文件头。
#
# 两个环境不能混：造数据/评测要 vllm 环境，训练要 tpt 环境（unsloth 只有它有）。

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY_VLLM=${TPT_VLLM_PYTHON:-/root/autodl-tmp/envs/vllm/bin/python}
PY_TPT=${TPT_PYTHON:-/root/miniconda3/envs/tpt/bin/python}

BASE=weights/Qwen3-4B-Base
SFT=outputs/sft-4b-v2
MERGED=outputs/sft-4b-v2-merged
DPO=outputs/dpo-4b-v1
PAIRS=data/processed/dpo-zh/stop.jsonl
JUNKIDS=data/processed/dpo-zh/stop.junkids.json

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

step_check() {
  echo "== 0/4 冒烟：验证三个 API 假设（约 1 分钟）=="
  echo "（这些假设在 Mac 上验不了 —— 没装 vllm。让它们花掉 15 分钟不值得）"
  "$PY_VLLM" - <<'PY'
import sys
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
from answer_extract import trailing_junk_len
from make_dpo_data import Engine, _render, _tokenizer
import prompts as P

MODEL, ADAPTER = "weights/Qwen3-4B-Base", "outputs/sft-4b-v2"

# 假设 1/2：SamplingParams 收 logit_bias（上界探针靠它）
from vllm import SamplingParams
try:
    params = SamplingParams(max_tokens=1, logit_bias={0: -100.0})
    print("  ✓ SamplingParams 接受 logit_bias（上界探针可用）")
except Exception as exc:
    print("  ✗ SamplingParams 不接受 logit_bias：%s" % exc)
    print("    → 上界探针要换实现（vLLM 版本差异），先别跑 probe")

# 假设 3：token_ids / finish_reason 拿得到，且 trailing_junk_len 能定位乱码
tok = _tokenizer(MODEL, P.CLEAN_CHAT_TEMPLATE)
engine = Engine(MODEL, adapter=ADAPTER)
records = engine.gen_detailed(
    _render(tok, ["请只回答一个数字，不要任何其他内容"] * 4), 128,
    P.stop_token_ids(tok),
)
engine.close()
for index, record in enumerate(records, 1):
    ids = record["ids"]
    cut = trailing_junk_len(tok, ids)
    last = repr(tok.decode([ids[-1]], skip_special_tokens=False)) if ids else "（空）"
    print("  #%d 截断=%-5s 剪掉=%d 个  末 token=%s" % (index, record["truncated"], cut, last))
    print("      尾部 %r" % record["text"][-60:])
if not any(trailing_junk_len(tok, r["ids"]) for r in records):
    print("  !! 4 条都没检出乱码。可能这条 prompt 太短 —— 检查 answer_extract 的字符类")
print("  过完这步再跑 data")
PY
}

step_data() {
  echo "== 1/4 造偏好数据（SFT 自采样，末尾乱码剪成 chosen）=="
  "$PY_VLLM" scripts/make_dpo_data.py --mode stop-token \
    --model "$BASE" --adapter "$SFT" \
    --n 4000 --max-new-tokens 512 --out "$PAIRS"
  echo
  echo "看完命中率再往下走。低于 ~50% 说明生成太短（被截断的题一条乱码都没有）。"
}

step_probe() {
  echo "== 2/4 上界探针：禁掉已知乱码 token，量「只修这一个毛病」值多少分 =="
  echo "（免费的先量上界。DPO 训完达不到这个数，就说明还有别的东西在拖）"
  if [ ! -f "$JUNKIDS" ]; then
    echo "!! 缺 $JUNKIDS，先跑：bash scripts/run_dpo.sh data"; return 1
  fi

  # 只禁出现 >=2 次的：偶发一次的 id 可能是检测误报，禁错了会让探针低估
  IDS=$("$PY_VLLM" - "$JUNKIDS" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
counts = data["counts"]
common = [k for k, v in sorted(counts.items(), key=lambda kv: -kv[1]) if v >= 2]
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print("      id=%-8s %5d 次   %r" % (k, v, data["sample_decode"].get(str(k))), file=sys.stderr)
print(",".join(common))
PY
)
  echo "  禁用 $(echo "$IDS" | tr ',' '\n' | wc -l) 个 id: $IDS"

  # 只跑指令遵循：它是唯一被乱码直接打分的项（数值约束 0/25 就是被它打掉的）
  rm -f evals/results/probe-ban.json
  "$PY_VLLM" scripts/eval.py --engine vllm --model "$BASE" --adapter "$SFT" \
    --label probe-ban --only ifollow --ban-token-ids "$IDS" 2>&1 | grep -E '^  [0-9]+/|去掉'

  echo
  echo "对照：sft-4b-v2 现状 61.5%（123/200），剪掉乱码再判的上界是 76.5%（153/200）。"
}

step_train() {
  echo "== 3/4 DPO 训练（在 SFT 产物上加一层 LoRA，ref 就是 SFT 模型）=="
  if [ ! -f "$PAIRS" ]; then
    echo "!! 缺 $PAIRS，先跑：bash scripts/run_dpo.sh data"; return 1
  fi
  "$PY_TPT" scripts/train_dpo.py --data "$PAIRS" --output "$DPO" --num-epochs 1
  echo
  echo "== 导出合并基座（评测时 --model 要指向它）=="
  "$PY_TPT" scripts/train_dpo.py --model "$BASE" --sft-adapter "$SFT" \
    --export-merged "$MERGED"
}

step_eval() {
  echo "== 4/4 评测（同一个引擎、同一套题，才能和 sft-4b-v2 比）=="
  if [ ! -d "$DPO" ]; then
    echo "!! 缺 $DPO，先跑：bash scripts/run_dpo.sh train"; return 1
  fi
  "$PY_VLLM" scripts/eval_pipeline.py --engine vllm \
    --run "dpo-4b-v1=$MERGED+$DPO" \
    --mmlu evals/mmlu-full.jsonl --only all
  echo
  echo "报告：evals/results/report.html"
  echo "和 sft-4b-v2 比这几个数（都在同一次输出里）："
  echo "  指令遵循   61.5% → ?    ← 主指标，目标 ≥76.5%"
  echo "  数值约束    0/25 → ?    ← 18 条是乱码打掉的，应该大幅回升"
  echo "  GSM8K      83.2% → ?    ← 监控位，不能掉"
  echo "  HumanEval  75.0% → ?    ← 监控位，不能掉"
  echo "  乱码尾     169/200 → ?  ← 直接看这个毛病修掉没有"
}

case "${1:-}" in
  check) step_check ;;
  data)  step_data ;;
  probe) step_probe ;;
  train) step_train ;;
  eval)  step_eval ;;
  *) sed -n '2,17p' "$0"; exit 1 ;;
esac
