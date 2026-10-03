"""换推理引擎的 A/B 验证：同一批卷面，HF 与 vLLM 各跑一遍，逐题比对。

为什么必须先做这个
------------------
**换引擎 = 换了一把尺子。** greedy 解码理论上逐样本独立、结果应该一样，
但两个引擎实现差得很远：

- HF：自己写循环，batch=1，每生成一个 token 读一遍权重
- vLLM：continuous batching + paged attention + 自己的采样核

数值路径不同、kernel 不同、batch 组成还会随调度变化 —— **输出可能有差异**。
不验证就切，就会出现「分数变了，但不知道是引擎的锅还是模型的锅」。

为什么两个引擎能对比同一份 prompt
--------------------------------
卷面由 `prompts.py` 拼（纯文本模块，两套环境都能 import），
**不让 vLLM 用它自己的 `chat()`** —— 否则「换引擎」和「换卷面」两个变量一起动，
差异没法归因。

用法
----
两套环境分别跑，再比对：

    # tpt 环境（unsloth / torch 2.12.1）—— 慢，50 题约 8 分钟
    python scripts/ab_engine.py run --engine hf --suite gsm8k \
        --model weights/Qwen3-4B-Base --n 50 --max-new-tokens 1024 --out /tmp/ab-hf.json

    # vllm 环境（vLLM / torch 2.13.0）—— 快，50 题约 30 秒
    python scripts/ab_engine.py run --engine vllm --suite gsm8k \
        --model weights/Qwen3-4B-Base --n 50 --max-new-tokens 1024 --out /tmp/ab-vllm.json

    python scripts/ab_engine.py compare /tmp/ab-hf.json /tmp/ab-vllm.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts as P  # noqa: E402  纯文本模块，两套环境都能用
from answer_extract import last_number, mmlu_letter  # noqa: E402

SUITES = ("gsm8k", "mmlu_chat", "mmlu_plain")


# ------------------------------------------------------------------ 卷面与判分


def build_prompts(suite: str, n: int) -> tuple[list[str], list[dict]]:
    """构造卷面。返回 (prompts, items)。"""
    if suite == "gsm8k":
        items = _read("evals/gsm8k.jsonl", n)
        return [P.gsm8k_prompt(it["question"]) for it in items], items
    style = "chat" if suite == "mmlu_chat" else "plain"
    items = _read("evals/mmlu-full.jsonl", n)
    return [P.mmlu_prompt(it, style) for it in items], items


def _read(relative: str, n: int) -> list[dict]:
    path = PROJECT_DIR / relative
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()][:n]


def predict(suite: str, text: str):
    """与 eval.py 相同的判分口径（判分层是纯模块，两个引擎共用）。"""
    if suite == "gsm8k":
        return last_number(text)
    return mmlu_letter(text)[0]


def reference(suite: str, item: dict):
    return item["answer"]  # gsm8k 是数值，mmlu 是选项下标，两者的 answer 字段语义不同


# ------------------------------------------------------------------ 两个引擎


def run_hf(prompts: list[str], model_ref: str, stop_ids: list[int], max_new: int):
    """HF 路径：逐条生成，和 eval.py 现在的做法一致（含停止符截断语义）。"""
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_ref, max_seq_length=2048, dtype=None, load_in_4bit=False
    )
    model.eval()
    stop = set(stop_ids)
    results = []
    began = time.time()   # 只计生成，模型加载不计
    for index, prompt in enumerate(prompts, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=stop_ids,
            )
        ids = [int(t) for t in generated[0][inputs["input_ids"].shape[1]:]]
        # 在第一个停止符处截断（与 eval.py 的 generate() 一致）
        cut = next((i for i, t in enumerate(ids) if t in stop), len(ids))
        kept = ids[:cut]
        results.append({
            "text": tokenizer.decode(kept, skip_special_tokens=True).strip(),
            "tokens": len(kept),
            "hit_cap": cut >= max_new,
        })
        if index % 10 == 0:
            print(f"  {index}/{len(prompts)}  {time.time() - began:.0f}s", flush=True)
    return results, time.time() - began


def run_vllm(prompts: list[str], model_ref: str, stop_ids: list[int],
             max_new: int, gpu_util: float, force_full_length: bool = False,
             tokenizer=None, no_prefix_cache: bool = False,
             enforce_eager: bool = False):
    """vLLM 路径：一次把整批丢进去，内部 continuous batching。

    `force_full_length=True` 是**为了可复现**：
    vLLM 默认的 continuous batching 里，序列长短不一、陆续结束，
    每一步 batch 的组成都在变 → GEMM 形状变 → 数值有微小差异 → 偶尔翻转 argmax。
    实测同一批 50 题跑两遍，逐题预测有 **10 条不一致**、命中数 27 vs 28。
    强制所有序列跑满 max_tokens（`ignore_eos=True`）后批次组成恒定，
    数值路径固定 —— 代价是浪费一些算力（写完的序列还要继续生成），
    换来的是「同一批输入两次跑出同一个结果」，这对评测尺子是必需的。
    尾巴要自己按停止符截断（vLLM 的 text 会带上停止符之后的内容）。
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_ref, max_model_len=2048,
        gpu_memory_utilization=gpu_util, dtype="bfloat16",
        # 试过的两个可复现开关：
        #   prefix caching 会让 KV 块布局随调度/淘汰顺序变化 → attention 走不同路径
        #   CUDA graphs 会按运行时 batch 大小选不同的图
        enable_prefix_caching=not no_prefix_cache,
        enforce_eager=enforce_eager,
    )
    params = SamplingParams(
        temperature=0, top_p=1.0, max_tokens=max_new,
        # 关闭 sampler 层的停止，让所有序列跑满；停止符在下面自己处理
        stop_token_ids=None if force_full_length else stop_ids,
        ignore_eos=force_full_length,
        skip_special_tokens=True,
    )
    # 只计生成时间：模型加载 / CUDA graph 捕获 / 显存 profiling 要几十秒，
    # 混进去会把「每题耗时」算大好几倍（全量评测时这部分是摊薄的）
    began = time.time()
    outputs = llm.generate(prompts, params)
    generation_seconds = time.time() - began
    stop = set(stop_ids)

    results = []
    for out in outputs:
        completion = out.outputs[0]
        if force_full_length:
            ids = [int(t) for t in completion.token_ids]
            cut = next((i for i, t in enumerate(ids) if t in stop), len(ids))
            text = tokenizer.decode(ids[:cut], skip_special_tokens=True).strip()
            results.append({
                "text": text, "tokens": cut,
                # 判「被截断」要看停止符之前有没有内容被切掉，即 cut 是否等于 max_new
                "hit_cap": cut >= max_new,
            })
        else:
            results.append({
                # vLLM 默认把停止符从输出里剔除，和 HF 路径的「截断」语义一致
                "text": completion.text.strip(),
                "tokens": len(completion.token_ids),
                "hit_cap": completion.finish_reason == "length",
            })
    return results, generation_seconds


# ------------------------------------------------------------------ 命令


def cmd_run(args) -> int:
    prompts, items = build_prompts(args.suite, args.n)
    _, tokenizer = _tokenizer(args.model)
    stop_ids = P.stop_token_ids(tokenizer)
    print(f"卷面 {len(prompts)} 条 | 停止符 {stop_ids} | max_new_tokens {args.max_new_tokens}")

    began = time.time()
    if args.engine == "hf":
        raw, gen_seconds = run_hf(prompts, args.model, stop_ids, args.max_new_tokens)
    else:
        raw, gen_seconds = run_vllm(prompts, args.model, stop_ids, args.max_new_tokens,
                                    args.gpu_util, args.force_full_length, tokenizer,
                                    args.no_prefix_cache, args.enforce_eager)
    spent = time.time() - began

    records = []
    for index, (item, gen) in enumerate(zip(items, raw)):
        records.append({
            "index": index,
            "predicted": predict(args.suite, gen["text"]),
            "answer": item["answer"],
            "tokens": gen["tokens"],
            "hit_cap": gen["hit_cap"],
            "text": gen["text"],
        })

    payload = {
        "engine": args.engine,
        "suite": args.suite,
        "model": args.model,
        "n": len(records),
        "max_new_tokens": args.max_new_tokens,
        "elapsed": round(spent, 1),
        "generation_seconds": round(gen_seconds, 2),
        "sec_per_item": round(gen_seconds / max(1, len(records)), 3),
        "records": records,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    right = sum(1 for r in records if r["predicted"] is not None and r["predicted"] == r["answer"])
    print(f"{args.engine}: {len(records)} 题 | 纯生成 {gen_seconds:.1f}s"
          f"（{payload['sec_per_item']}s/题）| 含加载 {spent:.1f}s | 命中 {right}/{len(records)}")
    print(f"写入 {args.out}")
    return 0


def _tokenizer(model_ref: str):
    from transformers import AutoTokenizer

    return None, AutoTokenizer.from_pretrained(model_ref)


def cmd_compare(args) -> int:
    a = json.loads(Path(args.left).read_text(encoding="utf-8"))
    b = json.loads(Path(args.right).read_text(encoding="utf-8"))
    if a["suite"] != b["suite"] or a["n"] != b["n"]:
        print(f"!! 不可比：{a['suite']}/{a['n']} vs {b['suite']}/{b['n']}")
        return 1

    print(f"对比 {a['suite']}｜{a['n']} 题｜max_new_tokens={a['max_new_tokens']}")
    print(f"  {a['engine']:<5} {a['elapsed']:>7.1f}s  {a['sec_per_item']:>5.2f}s/题")
    print(f"  {b['engine']:<5} {b['elapsed']:>7.1f}s  {b['sec_per_item']:>5.2f}s/题"
          f"   ← 提速 {a['sec_per_item'] / max(b['sec_per_item'], 1e-9):.1f}×")

    for tag, x in (("左", a), ("右", b)):
        hit = sum(1 for r in x["records"] if r["predicted"] is not None and r["predicted"] == r["answer"])
        cap = sum(1 for r in x["records"] if r["hit_cap"])
        print(f"  {tag}[{x['engine']}] 命中 {hit}/{x['n']}，被截断 {cap}")

    differ = []
    for ra, rb in zip(a["records"], b["records"]):
        if ra["predicted"] != rb["predicted"]:
            differ.append((ra, rb))
    print(f"\n逐题预测不一致：{len(differ)}/{a['n']}")
    for ra, rb in differ[:5]:
        print(f"  #{ra['index']} 期望={ra['answer']}")
        print(f"      {a['engine']:<5} 预测={ra['predicted']} ({ra['tokens']} tok) 尾: {ra['text'][-60:]!r}")
        print(f"      {b['engine']:<5} 预测={rb['predicted']} ({rb['tokens']} tok) 尾: {rb['text'][-60:]!r}")

    if not differ:
        print("  → 两个引擎逐题一致，可以切换")
        return 0
    print(f"  → 有差异，需要归因后再决定是否切换")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="推理引擎 A/B 验证")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="用某个引擎跑一批并落盘")
    run.add_argument("--engine", choices=["hf", "vllm"], required=True)
    run.add_argument("--suite", choices=SUITES, default="gsm8k")
    run.add_argument("--model", default="weights/Qwen3-4B-Base")
    run.add_argument("--n", type=int, default=50)
    run.add_argument("--max-new-tokens", type=int, default=1024)
    run.add_argument("--gpu-util", type=float, default=0.85)
    run.add_argument("--force-full-length", action="store_true",
                     help="vLLM 专用：所有序列跑满 max_tokens，让 batch 组成恒定以换取可复现")
    run.add_argument("--no-prefix-cache", action="store_true",
                     help="vLLM 专用：关掉 prefix caching（KV 块布局会影响 attention 路径）")
    run.add_argument("--enforce-eager", action="store_true",
                     help="vLLM 专用：关掉 CUDA graphs")
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", help="比对两份结果")
    cmp_.add_argument("left")
    cmp_.add_argument("right")
    cmp_.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
