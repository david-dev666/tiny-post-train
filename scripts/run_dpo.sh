#!/bin/bash
# DPO 全流程。四个步骤分开跑，每步都能单独看结果 ——
# 30 分钟起步的任务做成黑盒，崩了只能靠「几点开始」估（这个项目吃过这个亏）。
#
#   bash scripts/run_dpo.sh check    0. 冒烟：验三个 API 假设（约 1 分钟）★ 开机先跑
#   bash scripts/run_dpo.sh data     1. 造偏好数据（约 6 分钟）
#   bash scripts/run_dpo.sh train    2. DPO 训练（约 12 分钟）+ 导出合并模型
#   bash scripts/run_dpo.sh eval     3. 评测 + 报告（约 10 分钟）
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

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

step_check() {
  echo "== 0/3 冒烟：验证两个 API 假设（约 1 分钟）=="
  echo "（这些假设在 Mac 上验不了 —— 没装 vllm。让它们花掉 15 分钟不值得）"
  "$PY_VLLM" - <<'PY'
import sys
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
from answer_extract import trailing_junk_len
from make_dpo_data import Engine, _render, _tokenizer
import prompts as P

MODEL, ADAPTER = "weights/Qwen3-4B-Base", "outputs/sft-4b-v2"

# 假设：token_ids / finish_reason 拿得到，停止符被裁掉，乱码尾能被检出。
#
# 探测 prompt **必须是模型会自然收尾的**。含糊的提问（「请只回答一个数字」
# 却不给具体问题）会让它退化成复读、一路撞到预算上 —— 被截断的题压根走不到
# 「收尾决策」那一步，于是永远探测不出乱码。第一次就是被这个骗过去的。
PROBES = [
    "计算 7×8，只回答数字，不要任何其他内容。",
    "用一句话说明什么是递归。",
    "把「你好」翻译成英语，只输出翻译结果。",
    "列出三种水果，用顿号分隔。",
]
tok = _tokenizer(MODEL, P.CLEAN_CHAT_TEMPLATE)
stops = P.stop_token_ids(tok)
engine = Engine(MODEL, adapter=ADAPTER)
records = engine.gen_detailed(_render(tok, PROBES), 256, stops)
engine.close()

hits = 0
for prompt, record in zip(PROBES, records):
    ids = record["ids"]
    cut = trailing_junk_len(tok, ids)
    hits += int(cut > 0)
    print("  %-13s 截断=%-5s %3d tok  剪掉=%d  尾部 %r"
          % (prompt[:6], record["truncated"], len(ids), cut, record["text"][-28:]))

# 停止符必须已经不在 ids 里 —— 留着它，乱码尾判定会**静默归零**
leaked = [r for r in records if r["ids"] and r["ids"][-1] in set(stops)]
print("  %s ids 里残留停止符：%d 条" % ("✓" if not leaked else "✗", len(leaked)))
if hits == 0:
    print("  ✗ 一条乱码都没检出 —— 别往下跑，先查 answer_extract / strip_stop_tokens")
else:
    print("  ✓ %d/%d 条检出乱码尾，链路通了；可以跑 data" % (hits, len(records)))
PY
}

step_data() {
  echo "== 1/3 造偏好数据（SFT 自采样，末尾乱码换 <|im_end|>）=="
  "$PY_VLLM" scripts/make_dpo_data.py --mode stop-token \
    --model "$BASE" --adapter "$SFT" \
    --n 4000 --max-new-tokens 512 --out "$PAIRS"
  echo
  echo "看完命中率再往下走。低于 ~50% 说明生成太短（被截断的题一条乱码都没有）。"
}

step_train() {
  echo "== 2/3 DPO 训练（在 SFT 产物上加一层 LoRA，ref 就是 SFT 模型）=="
  if [ ! -f "$PAIRS" ]; then
    echo "!! 缺 $PAIRS，先跑：bash scripts/run_dpo.sh data"; return 1
  fi
  "$PY_TPT" scripts/train_dpo.py --data "$PAIRS" --output "$DPO" --num-epochs 1
  echo
  echo "== 导出合并基座（评测时 --model 要指向它）=="
  echo "   注意要 8G 空闲磁盘，满了会报 No space left on device"
  "$PY_TPT" scripts/train_dpo.py --model "$BASE" --sft-adapter "$SFT" \
    --export-merged "$MERGED"
}

step_eval() {
  echo "== 3/3 评测（同一个引擎、同一套题，才能和 sft-4b-v2 比）=="
  if [ ! -d "$DPO" ]; then
    echo "!! 缺 $DPO，先跑：bash scripts/run_dpo.sh train"; return 1
  fi
  "$PY_VLLM" scripts/eval_pipeline.py --engine vllm \
    --run "dpo-4b-v1=$MERGED+$DPO" \
    --mmlu evals/mmlu-full.jsonl --only all
  echo
  echo "报告：evals/results/report.html"
  echo "和 sft-4b-v2 比这几个数："
  echo "  指令遵循   61.5% → ?    ← 主指标"
  echo "  数值约束    0/25 → ?    ← 24 条是被末尾乱码打掉的，应该大幅回升"
  echo "  GSM8K      83.2% → ?    ← 监控位，不能掉"
  echo "  HumanEval  75.0% → ?    ← 监控位，不能掉"
  echo "  乱码尾     169/200 → ?  ← 直接看这个毛病修掉没有"
  echo
  echo "★ 但**别只看 loss 和 rewards**：三轮实测它们全都健康（acc 到 1.0）"
  echo "  而生成行为一点没变。真要判有没有学会，量这个："
  echo "  收尾位置上 P(<|im_end|>) 有没有超过第一名的乱码"
}

case "${1:-}" in
  check) step_check ;;
  data)  step_data ;;
  train) step_train ;;
  eval)  step_eval ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
