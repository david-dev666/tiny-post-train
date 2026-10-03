"""统一离线评测：对「基座」或「基座 + adapter」跑同一套评测，产出可对比的 json。

为什么必须有这个脚本
--------------------
训练脚本里的 `--bench-every` / `--probe-every` 只能给「训练过程中的快照」，
最早的观测点已经在若干步之后。4B 那次的第一颗采样点在 step 100/200，
那时已经训过 1600~3200 条样本 —— **微调前的基线根本不存在**，
所以「这次微调到底有没有用」这个问题当时无法回答，只能靠嘴说。

有了它，四个阶段就是四条命令，结果落在 `evals/results/` 下直接对照。

三套评测的分工
--------------
| 评测 | 测什么 | 怎么读数 |
| --- | --- | --- |
| MMLU（gen） | **默认口径**：让模型自己生成，取第一个 A/B/C/D 字母 | 端到端知识能力。默认跑小份子集（456 题，快、做回归）；**出正式结论要跑 `evals/mmlu-full.jsonl`（14042 题）** |
| MMLU（logit） | 只比「答案：」后面那个位置四个字母的 logits | 参考值。它测的是「想不想吐字母」，base 的偏置纯属预训练残留，SFT 会把它冲掉，**不等于不会答题** |
| 指令遵循 | 200 条带硬约束的题目，逐条程序化判分 | SFT 在「听不听话」上的效果，分 8 类看 |
| **GSM8K** | 1319 道小学数学题，答案唯一 | **可验证**：抽最后一个数字比对，不依赖主观判断 |
| **HumanEval** | 164 道写函数题，**真的执行 + 跑官方单测** | **可验证**：通过了才算会 |
| **OpenQA** | 40 条开放式问答，**本脚本只收集回答** | 质量维度。判分靠 `scripts/judge_openqa.py` 的成对偏好 |
| 固定话题采样 | 同样三个问题的生成结果 + 复读率 | 看输出像不像人话、有没有崩坏 |

**每个准确率都带 95% 置信区间（Wilson）**。这不是装饰：456 题上区间有 ±4.6pp，
只报点估计（"26.5%"）会让人以为精确到 0.1pp，据此下的结论多半是错的。

**这里没有 logits 口径。** 曾经有过一个「只比 `答案：` 后四个字母的 logits、
完全不生成」的参考口径，已从评测流程移除：它只有 HF 引擎能算（要读全词表 logits，
vLLM 只给 top-k logprobs），留着就会变成「一份 vLLM 结果里嵌一个 HF 算的块」，
破掉「整批结果必须同一把尺子」这条底线。它想回答的问题也已经用别的方式答了
（MMLU 两列 + 探针打印的生成原文）。训练看板上的 MMLU 曲线仍是那个口径，
读那条曲线时注意它的局限。

采样判分**刻意复用 `metrics.repetition_metrics`**（训练侧的 `ProbeCallback.score`
是它的转发），保证离线评测和训练曲线是同一把尺子，不会出现
「训练图上是 26%，离线测出来 30%」这种对不上的情况。

用法
----
    # 基线：只加载基座，不挂 adapter
    python scripts/eval.py --model weights/Qwen3-4B-Base --label base

    # 微调后：同一份基座 + adapter
    python scripts/eval.py --model weights/Qwen3-4B-Base \
        --adapter outputs/sft-4b --label sft-4b

    # 只跑其中几项
    python scripts/eval.py --model weights/Qwen3-4B-Base --label base --only ifollow
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
# 相对路径（weights/... evals/...）一律按项目根解析，从哪个目录调用都一样
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 本地权重强制离线，必须早于大件依赖 import —— huggingface_hub 是导入时读环境变量。
# 原先这行是 `import train_sft as ts` 的副作用；现在直接调用 offline.py 里的同一份实现，
# 因为 **vllm 环境装不了 unsloth**，import train_sft 会让评测在 vllm 环境里直接起不来。
from offline import LOCAL_MODEL_PATH  # noqa: E402

import argparse  # noqa: E402
import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

# 代码抽取单独成模块：rescore_humaneval.py 离线重判时要用**同一份**实现。
# 两处各写一份必然漂移 —— 本项目在 splitlines() 上已经栽过一次。
from code_extract import extract_code as _extract_code  # noqa: E402
from code_extract import strip_fence as _strip_fence  # noqa: E402

import torch  # noqa: E402  （vllm / tpt 两个环境都有；unsloth 则只有 tpt 有，
# 所以 unsloth 的 import 下沉到 engines.HFEngine 里 —— 见下面的说明）

# 推理引擎抽象：hf / vllm 两条生成后端，卷面与判分完全共用。
# **不要**在别处 import unsloth：vllm 环境没有它，`--engine vllm` 会当场 ImportError。
import prompts as P  # noqa: E402
from engines import ENGINES, make_engine  # noqa: E402

# 答案抽取（数字 / 选项字母 / 乱码尾检测）统一放在 answer_extract.py：
# eval.py 线上判分、rescore.py 离线重判、tests/ 单测都用同一份实现。
# 曾经这里 import 的是 make_benchmarks.last_number —— 又一份独立实现，
# 正是「多处实现必然漂移」的隐患。
from answer_extract import (  # noqa: E402
    follows_hash_format,
    has_junk_tail,
    last_number,
    mmlu_letter,
    trailing_junk_len,
    trim_junk_tail,
)

# 所有可跑的评测项。--only 里写 all 就是全跑。
SUITES = ("mmlu", "ifollow", "probes", "gsm8k", "humaneval", "openqa")

# 只跑前 N 条，给冒烟测试用（main() 会按 --limit 设置）。
# 注意：跑出来的结果**不完整**，只用来验证链路通不通，别写进 README。
LIMIT = 0


def _jsonl_lines(path: Path) -> list[str]:
    """按 `\\n` 切 JSONL。

    **绝对不能用 `str.splitlines()`**：它按 Unicode 定义的所有行边界切，
    包括 `\\x85`(NEL) / `\\x0b` / `\\x0c` / `\\u2028` 这些。而 JSONL 的换行就是 `\\n`。
    MMLU 题库第 6956 行里含一个 `\\x85`，用 splitlines 会把那行从字符串中间切开，
    json 解析直接报 `Unterminated string`，整个评测当场崩 —— 实测踩到过。
    """
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _read_items(path: Path) -> list[dict]:
    items = [json.loads(line) for line in _jsonl_lines(path)]
    return items[:LIMIT] if LIMIT else items


# 指令遵循的硬约束判定在 rules.py（纯文本模块，Mac 上可单测）
from rules import check_all  # noqa: E402


# ------------------------------------------------------------------ 模型

def stage_adapter(adapter_dir: Path, base: str) -> Path:
    """把 adapter 软链到临时目录，并把 base_model_name_or_path 改写成 --model。

    adapter_config.json 里记的是**训练当时**的基座路径：可能是相对路径，也可能是
    服务器上的绝对路径。换台机器或换个工作目录就找不到了。统一以 --model 为准，
    免得到时候报一个「找不到基座」的错又得回来翻配置。
    """
    if not (adapter_dir / "adapter_config.json").exists():
        sys.exit(f"!! {adapter_dir} 里没有 adapter_config.json，不是 LoRA 产物")

    resolved_base = Path(base).expanduser()
    tmp = Path(tempfile.mkdtemp(prefix="tpt-adapter-"))
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = str(resolved_base.resolve())
    (tmp / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name in (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        source = adapter_dir / name
        if source.exists():
            (tmp / name).symlink_to(source.resolve())
    print(f"==> adapter 就绪：{adapter_dir}（基座写死为 {config['base_model_name_or_path']}）")
    return tmp


def pick_chat_template(
    tokenizer, mode: str, has_adapter: bool, model_hint: str = ""
) -> tuple[str, str]:
    """选模板。返回 (模板, 说明)。

    这决定了「发给模型的卷子长什么样」，选错就是拿错格式考试：

    - 我们自训的产物（base + 自己的 LoRA）必须用 CLEAN —— 训练时就是它，
      用别的等于换了格式，而且 instruct 模板的 <think> 空块会变成开头乱码
    - **官方 instruct 版必须用它自带的模板**，那是它训练时用的格式，
      换成我们的干净模板反而是发错卷子
    """
    own = getattr(tokenizer, "chat_template", None)
    if mode == "clean":
        return P.CLEAN_CHAT_TEMPLATE, "项目内置干净模板（--chat-template clean 指定）"
    if mode == "tokenizer":
        if not own:
            sys.exit("!! 这个 tokenizer 没有自带模板，改用 --chat-template clean")
        return own, "tokenizer 自带模板（--chat-template tokenizer 指定）"

    # auto
    if has_adapter:
        return P.CLEAN_CHAT_TEMPLATE, "auto → 干净模板（挂了 adapter，说明是我们自己训的）"
    # 判据必须用**模型名**，不能用「tokenizer 模板里有没有 thinking 分支」：
    # Qwen3 的 base 版和 instruct 版共用同一份 tokenizer，模板也带 thinking 分支，
    # 按模板判会把 base 误判成 instruct（实测踩到）。
    if "instruct" in model_hint.lower() and own:
        return own, f"auto → tokenizer 自带模板（模型名含 instruct：{Path(model_hint).name}）"
    return P.CLEAN_CHAT_TEMPLATE, f"auto → 干净模板（非 instruct 版：{Path(model_hint).name}）"


def build_engine(
    name: str, model_ref: str, adapter: Path | None, max_seq_len: int,
    load_in_4bit: bool = False, gpu_util: float = 0.85,
    enforce_eager: bool = True, enable_prefix_caching: bool = False,
    ban_token_ids: tuple[int, ...] = (),
    template_mode: str = "auto", has_adapter: bool = False, model_hint: str = "",
):
    """按 `--engine` 造生成后端，并把对话模板钉到 tokenizer 上。

    模板在这里决定、不交给引擎内部自己套 —— 否则就成了「引擎决定卷面」，
    换引擎和换卷面两个变量一起动，分数差异归因不了（详见 engines.py 的说明）。
    """
    if name == "hf":
        if ban_token_ids:
            sys.exit("!! --ban-token-ids 只支持 vLLM 引擎（HF 那边要自己改 logits 处理器）")
        engine = make_engine("hf", model_ref, max_seq_len=max_seq_len, load_in_4bit=load_in_4bit)
    else:
        engine = make_engine(
            "vllm", model_ref, max_seq_len=max_seq_len,
            adapter=str(adapter) if adapter else None,
            gpu_util=gpu_util, enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching,
            ban_token_ids=ban_token_ids,
        )
    tokenizer = engine.tokenizer
    template, why = pick_chat_template(tokenizer, template_mode, has_adapter, model_hint)
    tokenizer.chat_template = template
    print(f"==> 对话模板：{why}")
    print(f"==> 推理引擎：{name}")
    return engine, tokenizer, why


def _chat(tokenizer, content: str) -> str:
    """把一段用户内容渲染成最终卷面。所有走对话模板的题都从这里出。"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )


def _progress(label: str):
    """生成阶段的进度回调。

    为什么要有：全量评测一次几十分钟，而 MMLU / GSM8K 这两个长循环原先
    **只在跑完时打一行汇总**，中途完全看不见跑到哪了（踩过：12 小时的任务
    崩了两次，每次都只能靠「几点开始」估）。
    """
    def report(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"  生成 {label} {done}/{total}", flush=True)

    return report


def _junk_info(tokenizer, ids: list[int], text: str) -> tuple[bool, str, int]:
    """(末尾有没有乱码, 假设它在最后一个正常 token 处停下会输出什么, 剪掉几个 token)。

    **按 token 判，不按文本判。** 乱码 token 独立解码后可能以正常汉字收尾
    （字节级 BPE 的残缺片段解成 `\\ufffd\\ufffd取`），文本层的
    `[乱码字符]+$` 匹配不到它 —— 既会漏报，也剪不干净。详见 answer_extract.py。

    第二项是「去掉乱码再判」的对照输入：**只用于对照，不改判分**
    （线上看原始输出，乱码是模型的真实输出，删掉等于掩盖缺陷）。
    它是按 token 重建的，所以是真上界；`answer_extract.trim_junk_tail`
    那个文本版给出的只是下界。

    拿不到 token 时（理论上不会，两个引擎都回传 ids）退回文本层 —— 宁可低估。
    """
    if ids:
        cut = trailing_junk_len(tokenizer, ids)
        if cut:
            return True, tokenizer.decode(ids[:-cut], skip_special_tokens=True).strip(), cut
        return False, text, 0
    return has_junk_tail(text), trim_junk_tail(text), 0


def _parse_token_ids(spec: str) -> tuple[int, ...]:
    """把 `--ban-token-ids` 的 `"139941,139942"` 解析成 id 元组。空串 = 不禁。"""
    out = []
    for piece in (spec or "").replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        if not piece.isdigit():
            sys.exit(f"!! --ban-token-ids 里 {piece!r} 不是整数 id")
        out.append(int(piece))
    return tuple(out)


# ------------------------------------------------------------------ 三套评测

# 生成过程中的异常计数（`stop_token_in_middle` 之类）现在由引擎自己维护：
# 详见 engines.BaseEngine.stats。分数之外还要看这些 —— 它们说明「分数为什么不可信」。


def run_ifollow(engine, tokenizer, path: Path, max_new_tokens: int) -> dict:
    items = _read_items(path)
    print(f"\n== 指令遵循（{len(items)} 条）==")
    # 先生成完再判分：vLLM 的加速全靠一次把整批丢进去（详见 engines.py）。
    # HF 后端内部仍是逐条，行为与改动前一致。
    outputs = engine.run(
        [_chat(tokenizer, item["prompt"]) for item in items],
        max_new_tokens, on_progress=_progress("指令遵循"),
    )
    truncated = junk_tail = 0
    records = []
    for index, (item, gen) in enumerate(zip(items, outputs), 1):
        began = time.time()
        output, ids = gen.text, gen.ids
        passed, detail = check_all(item["rules"], output)
        # 诊断位：被截断 / 尾巴有乱码。都**不参与判分**，但它们是「这条为什么没过」
        # 的常见误判来源 —— 比如答案写对了，却因为尾巴的乱码超出字数上限
        junk, output_trimmed, cut_tokens = _junk_info(tokenizer, ids, output)
        # 对照分在这里算：它要用 token 重建「去掉乱码的输出」，循环外已经拿不到了
        passed_trimmed, _ = check_all(item["rules"], output_trimmed)
        truncated += int(len(ids) >= max_new_tokens)
        junk_tail += int(junk)
        records.append(
            {
                "id": item["id"],
                "category": item["category"],
                "desc": item["desc"],
                "prompt": item["prompt"],
                "passed": passed,
                "detail": detail,
                "output": output,
                "sec": round(time.time() - began, 2),
                "tokens": len(ids),
                "truncated": len(ids) >= max_new_tokens,
                "junk_tail": junk,
                "junk_tokens": cut_tokens,
                # 假设它在最后一个正常 token 处停下，这条规则过不过（不参与判分）
                "passed_trimmed": passed_trimmed,
            }
        )
        print(f"  [{'✓' if passed else '✗'}] {item['id']:<18} {detail}")

    total = len(records)
    hit = sum(1 for r in records if r["passed"])
    by_category: dict[str, dict] = {}
    for r in records:
        slot = by_category.setdefault(r["category"], {"passed": 0, "total": 0})
        slot["total"] += 1
        slot["passed"] += int(r["passed"])
    for slot in by_category.values():
        slot["rate"] = round(slot["passed"] / slot["total"], 4) if slot["total"] else 0.0

    # 对照：把尾部乱码去掉再判一遍。
    # **不是为了改判分**（线上看原始输出，乱码是模型的真实输出），
    # 而是让「这一两个乱码 token 到底值多少分」变成可见的数字 ——
    # 实测 sft-4b-v2 因此差 15pp（61.5% → 76.5%），不摆出来读者会以为是能力问题。
    # 逐条的对照判定在生成循环里就算好了（要 token 才能重建输出），这里只汇总。
    trimmed_ok = sum(1 for r in records if r["passed_trimmed"])
    flipped = [r["id"] for r in records if r["passed_trimmed"] and not r["passed"]]

    result = _attach_ci({
        "total": total,
        "correct": hit,
        "passed": hit,
        "rate": round(hit / total, 4) if total else 0.0,
        "by_category": by_category,
        "records": records,
        "truncated": truncated,
        "junk_tail": junk_tail,
        # 因为尾巴乱码而失分的条数（含乱码尾且未通过）—— 这是「分数被污染」的直接度量
        "failed_with_junk_tail": sum(1 for r in records if r["junk_tail"] and not r["passed"]),
        "passed_junk_trimmed": trimmed_ok,
        "rate_junk_trimmed": round(trimmed_ok / total, 4) if total else 0.0,
        "flipped_by_junk_tail": flipped,
    })
    print(f"  {hit}/{total} = {result['rate']:.1%} ± {result['ci95_half_pp']}pp")
    return result


# MMLU 生成式的提问口径。chat 给指令微调过的模型，plain 给基座 —— 详见 _mmlu_prompt。
# 用 dict 而不是直接判断字符串，是为了让 `--mmlu-style both` 只写一遍展开逻辑。
MMLU_STYLES = {"chat": ("chat",), "plain": ("plain",), "both": ("chat", "plain")}

# 结果 json 里的「块」。--update 合并时靠它区分「这次重算的」和「上次沿用的」，
# 沿用来的块代码指纹对不上当前代码，必须能在报告里看出来。
RESULT_BLOCKS = ("mmlu_gen", "mmlu_gen_plain", "ifollow",
                 "gsm8k", "humaneval", "probes", "openqa")

# 结果文件的格式版本。以后字段有变动就 +1，方便一眼看出新旧。
SCHEMA_VERSION = 3


def _file_sha(path: Path, length: int = 16) -> str:
    """文件内容指纹。用来锁定「这份结果是在哪一版题目上跑的」。"""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:length]
    except OSError:
        return ""


# **影响结果**的每个模块都要进指纹。
#
# 重构之后判分逻辑分散在多个文件里：规则判定在 rules.py、字母/数字抽取在
# answer_extract.py、置信区间在 metrics.py、剥围栏在 code_extract.py。
# 只记 eval.py 的 sha 会出现「改了 rules.py 但指纹一点没变」——
# 那正是指纹要防的事：两份结果声称同一版代码，实际判分规则不同。
#
# prompts.py 也在内：它现在供给 GSM8K / 代码题的**卷面原文**，
# 改一个字的措辞，分数就可能变 —— 那是比判分更上游的变量，不记指纹等于没记。
# engines.py 同理：生成参数（采样、停止符截断语义）在那里。
# train_sft.py **不在内**：它已经不影响评测结果了（那个复用它的 logits 口径被移除，
# 采样指标下沉到了 metrics）。留着它只会让「改了训练代码」被误报成「判分栈变了」。
SCORING_MODULES = (
    "eval.py", "rules.py", "metrics.py",
    "answer_extract.py", "code_extract.py",
    "prompts.py", "engines.py",
)


def _code_version() -> dict:
    """代码版本。服务器上不是 git 仓库，所以脚本指纹才是真正管用的那个。

    元信息是给半年后的自己看的：没有它，一堆 json 摆在 results/ 里
    谁也不知道哪份是什么条件下跑出来的。
    """
    here = Path(__file__).resolve()
    modules = {}
    for name in SCORING_MODULES:
        sha = _file_sha(here.parent / name, length=12)
        if sha:
            modules[name] = sha
    # 整栈一个聚合指纹：比对「两次评测是不是同一版判分代码」时看这一个就够
    stack = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(modules.items())).encode()
    ).hexdigest()[:12]
    info = {
        "eval_py_sha": modules.get("eval.py", ""),          # 旧字段，报告仍在用
        "scoring_modules": modules,
        "scoring_stack_sha": stack,
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_DIR, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=PROJECT_DIR, timeout=5,
        ).stdout.strip()
        if commit:
            info["git_commit"] = commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass  # 服务器上项目目录不是 git 仓库，正常
    return info


def _subset_info(path: Path) -> dict:
    count = 0
    if path.exists():
        count = len(_jsonl_lines(path))
    return {"path": str(path), "count": count, "sha256": _file_sha(path)}


# 置信区间在 metrics.py（纯数学模块，Mac 上可单测）。judge_openqa.py 也复用这一份。
from metrics import attach_ci as _attach_ci  # noqa: E402
from metrics import repetition_metrics as _repetition_metrics  # noqa: E402
from metrics import wilson_ci as _wilson_ci  # noqa: E402




def _mmlu_prompt(tokenizer, item, style: str = "chat") -> str:
    """构造 MMLU 的卷面。`style` 决定用哪种提问形式。

    为什么要有 `plain`
    -----------------
    `chat` 形式把「答案：」放在 **user 回合内**，模型真正开始生成时已经另起
    `<|im_start|>assistant` 回合：

        <|im_start|>user
        以下是一道单项选择题…答案：<|im_end|>
        <|im_start|>assistant

    instruct / 自训 adapter 知道「assistant 该回答」。但**基座不知道** ——
    它看到「答案：」+ 回合结束符，判定这是一份**已写完的问答文档**，于是接着
    开下一题（复读题干）。实测 400 题里一个字母都没吐出来，得 0.0%。

    `plain` 形式去掉 chat 包装，只用基座在预训练里见过的自然文档模式：

        Question: …
        A. …  B. …  C. …  D. …
        Answer:

    同一个基座、同一批题，从 0% 变成 85%，**且 0-shot 就够**（实测 0/1/2/3/5-shot
    完全一样，说明差别全在 chat 包装，不在示例数量）。

    所以两种口径都要留：`plain` 给基座，`chat` 给指令微调过的模型。
    **只跑一种，总有一方是被错怪或者被放水的。**
    """
    options = "\n".join(
        letter + ". " + choice for letter, choice in zip(P.LETTERS, item["choices"])
    )
    if style == "plain":
        return f"Question: {item['question']}\n{options}\nAnswer:"
    content = (
        "以下是一道单项选择题，请直接回答正确选项的字母。\n\n"
        + item["question"] + "\n\n" + options + "\n\n答案："
    )
    return _chat(tokenizer, content)


def run_mmlu_gen(engine, tokenizer, path: Path, max_new_tokens: int, style: str = "chat") -> dict:
    """生成式判分：让模型自己生成，取第一个 A/B/C/D。

    这是端到端能力：既要知道答案，也要按格式把它吐出来。
    （曾经的 logits 口径只比「答案：」后四个字母的 logits、完全不生成，
     和这里可以给出相反结论 —— 已从评测流程移除，理由见 docs/01-评测.md。）

    `style` 见 `_mmlu_prompt`：`plain` 给基座用，`chat` 给指令微调过的模型用。
    """
    print(f"\n== MMLU 子集 · 生成式（{path.name}，{style} 口径）==")
    items = _read_items(path)
    started = time.time()
    outputs = engine.run(
        [_mmlu_prompt(tokenizer, item, style) for item in items],
        max_new_tokens, on_progress=_progress(f"MMLU-{style}"),
    )
    correct = 0
    noparse = 0
    by_subject: dict[str, list[int]] = {}
    by_rule: dict[str, int] = {}
    truncated = 0
    records = []
    for index, (item, gen) in enumerate(zip(items, outputs)):
        ids, generated = gen.ids, gen.text
        letter, rule = mmlu_letter(generated)
        by_rule[rule] = by_rule.get(rule, 0) + 1
        truncated += int(len(ids) >= max_new_tokens)
        if letter is None:
            noparse += 1
            hit = 0
        else:
            hit = int(letter == item["answer"])
        correct += hit
        by_subject.setdefault(item["subject"], []).append(hit)
        # 存生成原文：判分口径一改就能离线重判，不必重跑模型（rescore.py）
        # ⚠️ MMLU 的题目文件**没有 id 字段**（其它几套都有），所以用下标兜底 ——
        # 曾经这里直接 item["id"]，导致刚加完 records 就 KeyError 秒退
        records.append({
            "id": item.get("id") or f"mmlu-{index:05d}",
            "subject": item["subject"], "answer": item["answer"],
            "predicted": letter, "rule": rule, "correct": bool(hit),
            "generated": generated, "tokens": len(ids),
        })

    total = len(items)
    result = _attach_ci({
        "mode": "gen",
        "style": style,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "chance": 0.25,
        "unparsed": noparse,
        "sec": round(time.time() - started, 1),
        "by_subject": {
            subject: round(sum(hits) / len(hits), 3) for subject, hits in sorted(by_subject.items())
        },
        # 诊断：分数可不可信，看这几个数
        "by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1])),
        "truncated": truncated,
        "records": records,
    })
    print(
        f"  {correct}/{total} = {result['accuracy']:.1%} ± {result['ci95_half_pp']}pp"
        f"（解析不出字母 {noparse} 条，{result['sec']}s，随机基准 25%）"
    )
    return result


def run_probes(engine, tokenizer, max_new_tokens: int) -> dict:
    prompts = P.PROBE_PROMPTS
    print(f"\n== 固定话题采样（{len(prompts)} 条）==")
    # 注意：ProbeCallback 原先是逐条 generate，这里改成整批 —— 采样只有 3 条，
    # 批量对 HF 后端没有差别（它内部还是逐条），对 vLLM 则省下两次调度开销。
    outputs = engine.run(
        [_chat(tokenizer, prompt) for prompt in prompts],
        max_new_tokens, on_progress=_progress("采样"),
    )
    records = []
    for prompt, gen in zip(prompts, outputs):
        record = {
            "prompt": prompt,
            "output": gen.text,
            # sec 是逐条生成耗时，批量后端给不出单条值，记 0 并在报告里说明
            "sec": 0.0,
            **_repetition_metrics(gen.ids),
        }
        records.append(record)
        print(
            f"  {prompt[:16]:<18} {record['tokens']:>4} tokens  "
            f"复读率 {record['repeat_2gram']:.1%}  多样 {record['distinct_ratio']:.1%}"
        )
    return {"records": records}


# ------------------------------------------------------------ 可验证答案（数学 / 代码）

# 卷面文案的唯一来源是 prompts.py（ab_engine.py 做引擎 A/B 时也用那一份）。
# 这里只是转个手 —— 抄一份就会漂移，本项目已经栽过两次。
GSM8K_INSTRUCTION = P.GSM8K_INSTRUCTION
CODE_INSTRUCTION = P.CODE_INSTRUCTION
def _run_python(program: str, timeout: int) -> tuple[bool, str]:
    """在子进程里跑一段 Python，返回 (是否通过, 说明)。

    ⚠️ 这里执行的是**模型生成的代码**。本项目只做了三件事：
    独立解释器（-I：不加载 site-packages、不读用户目录）、超时、临时工作目录。
    **没有真正的沙箱**（没做文件系统隔离、没断网）。
    所以：不要在有敏感数据或凭据的机器上跑，也不要拿不可信的代码当输入。
    """
    with tempfile.TemporaryDirectory(prefix="tpt-code-") as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", program],
                capture_output=True, text=True, timeout=timeout, cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return False, f"超时（>{timeout}s）"
        except OSError as exc:
            return False, f"启动失败：{exc!r}"
    if proc.returncode == 0:
        return True, "通过"
    lines = [line for line in (proc.stderr or "").strip().splitlines() if line.strip()]
    return False, (lines[-1][:120] if lines else f"退出码 {proc.returncode}")


def run_gsm8k(engine, tokenizer, path: Path, max_new_tokens: int) -> dict:
    """GSM8K：答案唯一，抽最后一个数字比对 —— 不依赖任何主观判断。"""
    print(f"\n== GSM8K（{path.name}，可验证）==")
    items = _read_items(path)
    started = time.time()
    outputs = engine.run(
        [_chat(tokenizer, GSM8K_INSTRUCTION.format(question=item["question"])) for item in items],
        max_new_tokens, on_progress=_progress("GSM8K"),
    )
    correct = 0
    unparsed = 0
    truncated = follows_format = junk_tail = 0
    records = []
    for item, gen in zip(items, outputs):
        output, ids = gen.text, gen.ids
        predicted = last_number(output)
        if predicted is None:
            unparsed += 1
        hit = predicted is not None and abs(predicted - item["answer"]) < 1e-4
        correct += int(hit)
        # 三个诊断位：被截断 / 按格式作答 / 尾巴有乱码。
        # 它们**不参与判分**，但决定这份分数可不可信 —— 截断率高说明生成长度给少了。
        junk, _, cut_tokens = _junk_info(tokenizer, ids, output)
        truncated += int(len(ids) >= max_new_tokens)
        follows_format += int(follows_hash_format(output))
        junk_tail += int(junk)
        records.append({"id": item["id"], "correct": hit, "predicted": predicted,
                        "answer": item["answer"], "output": output,
                        "tokens": len(ids), "truncated": len(ids) >= max_new_tokens,
                        "follows_format": follows_hash_format(output),
                        "junk_tail": junk, "junk_tokens": cut_tokens})
    total = len(items)
    result = _attach_ci({
        "total": total, "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "unparsed": unparsed, "sec": round(time.time() - started, 1),
        "records": records,
        "truncated": truncated, "follows_format": follows_format, "junk_tail": junk_tail,
    })
    print(f"  {correct}/{total} = {result['accuracy']:.1%} ± {result['ci95_half_pp']}pp"
          f"（抽不出数字 {unparsed} 条，{result['sec']}s）")
    return result


def run_humaneval(engine, tokenizer, path: Path, max_new_tokens: int, timeout: int) -> dict:
    """HumanEval：真的执行模型写的函数 + 跑官方单元测试。"""
    print(f"\n== HumanEval（{path.name}，可执行验证）==")
    items = _read_items(path)
    started = time.time()
    outputs = engine.run(
        [_chat(tokenizer, CODE_INSTRUCTION.format(prompt=item["prompt"])) for item in items],
        max_new_tokens, on_progress=_progress("HumanEval"),
    )
    correct = 0
    truncated = junk_tail = 0
    records = []
    for item, gen in zip(items, outputs):
        output, ids = gen.text, gen.ids
        completion = _extract_code(output)
        program = (
            item["prompt"] + completion + "\n" + item["test"]
            + f"\ncheck({item['entry_point']})\n"
        )
        passed, why = _run_python(program, timeout)
        correct += int(passed)
        junk, _, cut_tokens = _junk_info(tokenizer, ids, output)
        truncated += int(len(ids) >= max_new_tokens)
        junk_tail += int(junk)
        records.append({"id": item["id"], "passed": passed, "detail": why,
                        "completion": completion, "raw_output": output,
                        "tokens": len(ids), "truncated": len(ids) >= max_new_tokens,
                        "junk_tail": junk, "junk_tokens": cut_tokens})
        print(f"  [{'✓' if passed else '✗'}] {item['id']:<14} {why}")
    total = len(items)
    result = _attach_ci({
        "total": total, "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "sec": round(time.time() - started, 1),
        "records": records,
        "truncated": truncated, "junk_tail": junk_tail,
    })
    print(f"  {correct}/{total} = {result['accuracy']:.1%} ± {result['ci95_half_pp']}pp（{result['sec']}s）")
    return result


def run_openqa(engine, tokenizer, path: Path, max_new_tokens: int) -> dict:
    """开放式质量：**只收集回答，不打分**。

    为什么不在这一步打分：开放式问题没有唯一答案，规则判不了。
    必须靠裁判做**成对偏好**（A/B 哪个更好），见 scripts/judge_openqa.py。
    把「采回答」和「判分」拆开的好处是：换裁判、加裁判、改评分维度
    都不用重新跑模型 —— 模型生成才是慢的那一步。
    """
    print(f"\n== OpenQA（{path.name}，只收集回答）==")
    items = _read_items(path)
    started = time.time()
    outputs = engine.run(
        [_chat(tokenizer, item["prompt"]) for item in items],
        max_new_tokens, on_progress=_progress("OpenQA"),
    )
    records = []
    for item, gen in zip(items, outputs):
        records.append({
            "id": item["id"], "category": item["category"], "prompt": item["prompt"],
            "output": gen.text, **_repetition_metrics(gen.ids),
        })
    result = {
        "total": len(records), "sec": round(time.time() - started, 1), "records": records,
    }
    print(f"  收集 {len(records)} 条回答（{result['sec']}s）。判分用：")
    print("    python scripts/judge_openqa.py --a <结果A>.json --b <结果B>.json --judge weights/Qwen3-4B-Instruct-2507")
    return result


# ------------------------------------------------------------------ 入口

def parse_args():
    parser = argparse.ArgumentParser(description="统一离线评测（基座 / 基座+adapter）")
    parser.add_argument("--model", required=True, help="基座权重路径，如 weights/Qwen3-4B-Base")
    parser.add_argument("--adapter", default=None, help="LoRA 产物目录，如 outputs/sft-4b；不给就是纯基座基线")
    parser.add_argument("--label", default=None, help="结果文件名（默认按 model+adapter 推）")
    parser.add_argument("--mmlu", default="evals/mmlu-subset.jsonl",
                        help="默认是小份子集（快）；全量用 evals/mmlu-full.jsonl")
    parser.add_argument("--ifollow", default="evals/ifollow-subset.jsonl")
    parser.add_argument("--gsm8k", default="evals/gsm8k.jsonl")
    parser.add_argument("--humaneval", default="evals/humaneval.jsonl")
    parser.add_argument("--openqa", default="evals/openqa.jsonl")
    parser.add_argument("--only", default="all",
                        help=f"逗号分隔，写 all 表示全跑。可选：all / {' / '.join(SUITES)}")
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=1024,
                        help="GSM8K 单题生成长度上限。**不要调回 256**：实测 256 时 "
                             "instruct-2507 有 11/16 题被截断在最终答案之前，"
                             "分数从 87.5% 掉到 31.2%。模型越啰嗦被扣得越狠 —— "
                             "这是风格差异被当成了能力差异（详见 notes 失败记录）")
    parser.add_argument("--code-max-new-tokens", type=int, default=512)
    parser.add_argument("--openqa-max-new-tokens", type=int, default=512)
    parser.add_argument("--code-timeout", type=int, default=10,
                        help="HumanEval 每道题执行模型代码的超时秒数")
    parser.add_argument("--limit", type=int, default=0,
                        help="每套只跑前 N 条，冒烟测试用。结果不完整，别写进 README")
    parser.add_argument("--output-dir", default="evals/results")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="指令遵循题目的生成长度上限")
    parser.add_argument("--probe-max-new-tokens", type=int, default=128,
                        help="与训练时 --probe-max-new-tokens 保持一致，否则复读率不可比")
    parser.add_argument("--mmlu-style", default="chat", choices=["chat", "plain", "both"],
                        help="MMLU 生成式的提问口径。chat=带对话模板（指令微调模型用）；"
                             "plain=纯文本 Question/Answer（**基座必须用这个**，否则复读题干得 0 分）；"
                             "both=两种都跑并存两份")
    parser.add_argument("--update", action="store_true",
                        help="把这次算出来的块**合并**进已存在的结果 json，保留没重算的块"
                             "（如只补 MMLU 口径时用，免得把 gsm8k/humaneval 洗掉）")
    parser.add_argument("--mmlu-max-new-tokens", type=int, default=8,
                        help="生成式判分里每题只生成这么长，够吐一个字母就行")
    parser.add_argument("--chat-template", default="auto", choices=["auto", "clean", "tokenizer"],
                        help="auto：挂 adapter 用干净模板，官方 instruct 用其自带模板；"
                             "clean：强制项目内置干净模板；tokenizer：强制 tokenizer 自带")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--engine", default="vllm", choices=list(ENGINES),
                        help="生成后端。vllm=批量推理，实测快 16.7×（默认）；"
                             "hf=unsloth 逐条，慢但与换引擎前逐字节一致（备用/对照）。"
                             "**装了 unsloth 的是 tpt 环境，vllm 要用 "
                             "/root/autodl-tmp/envs/vllm/bin/python 跑**。"
                             "两个引擎的分数不能混着比（数值路径不同，逐题会翻转）")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.85)
    parser.add_argument("--vllm-no-eager", action="store_true",
                        help="vLLM：**开了就别指望可复现**。默认 enforce_eager=True，"
                             "关掉 CUDA graphs 换来同一批输入两次跑逐字相同。"
                             "实测关掉它（即本开关）后 50 题里有 10 题预测翻转")
    parser.add_argument("--vllm-prefix-caching", action="store_true",
                        help="vLLM：默认关闭。prefix caching 会让 KV 块布局随调度变化，"
                             "换个 attention 路径，对可复现性没好处")
    parser.add_argument("--ban-token-ids", default="",
                        help="**上界探针**：禁掉这些 token（逗号分隔的 id），量一量"
                             "「只修乱码尾」到底值多少分。id 列表来自 "
                             "make_dpo_data.py --mode stop-token 顺手落的 .junkids.json。"
                             "注意这不是修复方案 —— 只能挡住已经见过的 id")
    return parser.parse_args()


def main():
    args = parse_args()
    global LIMIT
    LIMIT = max(0, args.limit)
    if LIMIT:
        print(f"!! --limit {LIMIT}：冒烟模式，结果不完整，不要写进 README")
    wanted = {part.strip() for part in args.only.split(",") if part.strip()}
    if "all" in wanted:
        wanted = set(SUITES)
    unknown = wanted - set(SUITES)
    if unknown:
        sys.exit(f"!! --only 里有不认识的名字：{sorted(unknown)}（可选 {list(SUITES)}）")

    label = args.label
    if not label:
        label = (Path(args.adapter).name if args.adapter else Path(args.model).name) or "run"

    print(f"==> 项目根：{PROJECT_DIR}")
    if LOCAL_MODEL_PATH:
        print(f"==> 本地权重 {LOCAL_MODEL_PATH}，已强制离线（HF_HUB_OFFLINE=1）")

    # adapter：HF 走 unsloth 的适配器加载；vLLM 走它自己的 LoRA 支持。
    # 两边都需要一个「base_model_name_or_path 指向 --model」的目录，
    # 所以 stage_adapter 两边共用。
    model_ref = args.model
    staged_adapter = None
    if args.adapter:
        staged_adapter = stage_adapter(Path(args.adapter), args.model)
        if args.engine == "hf":
            model_ref = str(staged_adapter)

    began = time.time()
    engine, tokenizer, template_note = build_engine(
        args.engine, model_ref, staged_adapter, args.max_seq_len,
        load_in_4bit=args.load_in_4bit,
        gpu_util=args.vllm_gpu_util,
        enforce_eager=not args.vllm_no_eager,
        enable_prefix_caching=args.vllm_prefix_caching,
        ban_token_ids=_parse_token_ids(args.ban_token_ids),
        template_mode=args.chat_template,
        has_adapter=bool(args.adapter),
        model_hint=args.model,
    )
    print(f"==> 模型加载完成（{time.time() - began:.1f}s）：{'基座 + adapter' if args.adapter else '纯基座'}")

    # 元信息是给半年后的自己看的。没有它，results/ 里一堆 json 谁也不知道
    # 哪份是什么条件下跑出来的 —— 模板、判分口径、题目指纹缺一不可。
    result = {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "model": args.model,
        "adapter": args.adapter,
        "base_model_used": str(model_ref) if args.adapter else args.model,
        "evaluated_at": int(time.time()),
        "evaluated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "chat_template": {"mode": args.chat_template, "resolved": template_note},
        # 引擎与它的配置。**缺了它，results/ 里混着两个引擎的分数就再也归因不了**：
        # HF 和 vLLM 在 bf16 下是两条数值路径，50 题里逐题预测有 13 题会翻转，
        # 聚合分差 3 题 —— 不记下来，半年后看到两份不同的 json 只能猜。
        "engine": engine.name,
        "engine_config": dict(engine.config),
        "code": _code_version(),
        "subsets": {
            name: _subset_info(Path(getattr(args, name)))
            for name in ("mmlu", "ifollow", "gsm8k", "humaneval", "openqa")
        },
        "generation": {
            "do_sample": False,
            "stop_token_ids": P.stop_token_ids(tokenizer),
            "mmlu_max_new_tokens": args.mmlu_max_new_tokens,
            "ifollow_max_new_tokens": args.max_new_tokens,
            "probe_max_new_tokens": args.probe_max_new_tokens,
            "gsm8k_max_new_tokens": args.gsm8k_max_new_tokens,
            "code_max_new_tokens": args.code_max_new_tokens,
            "openqa_max_new_tokens": args.openqa_max_new_tokens,
            "code_timeout": args.code_timeout,
        },
        # limit 必须记进来。**之前漏了它，导致 make_report 的「截断档」告警是死代码**：
        # 报告读的是 requested["limit"]，永远取不到 → 一批只跑了前 400 题的结果
        # 在页面上和全量结果长得一模一样，没有任何提示。
        "requested": {"only": sorted(wanted),
                      "mmlu_style": args.mmlu_style, "limit": LIMIT or None},
    }

    if "mmlu" in wanted:
        # chat / plain 各存一份，互不覆盖：base 与 instruct 需要不同口径，
        # 只留一种总有一方被错怪。--mmlu-style both 就是两种都跑。
        for style in MMLU_STYLES[args.mmlu_style]:
            key = "mmlu_gen" if style == "chat" else f"mmlu_gen_{style}"
            result[key] = run_mmlu_gen(
                engine, tokenizer, Path(args.mmlu), args.mmlu_max_new_tokens, style
            )
    if "ifollow" in wanted:
        result["ifollow"] = run_ifollow(engine, tokenizer, Path(args.ifollow), args.max_new_tokens)
    if "gsm8k" in wanted:
        result["gsm8k"] = run_gsm8k(
            engine, tokenizer, Path(args.gsm8k), args.gsm8k_max_new_tokens
        )
    if "humaneval" in wanted:
        result["humaneval"] = run_humaneval(
            engine, tokenizer, Path(args.humaneval), args.code_max_new_tokens, args.code_timeout
        )
    if "probes" in wanted:
        result["probes"] = run_probes(engine, tokenizer, args.probe_max_new_tokens)
    if "openqa" in wanted:
        result["openqa"] = run_openqa(
            engine, tokenizer, Path(args.openqa), args.openqa_max_new_tokens
        )

    result["elapsed"] = round(time.time() - began, 1)
    # 判分之外的异常：分数对不对是其次，「这个分数该不该信」才是第一位的。
    # 计数由引擎自己维护 —— vLLM 在引擎内部处理停止符，观测不到这个量，恒为 0。
    result["diagnostics"] = dict(engine.stats)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}.json"

    # --update：只补跑某一项（比如只补 MMLU 的 plain 口径）时，别把上次算好的
    # gsm8k / humaneval 洗掉 —— 那些是花了几小时生成出来的。
    # 沿用来的块要记账：它们是用**当时的代码**算的，和这次重算的块代码指纹不同，
    # 不标出来，报告里「三个模型代码指纹不一致」会让人以为跑的时候换了版本。
    if args.update and out_path.exists():
        try:
            previous = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"!! --update 读不了已有结果（{exc}），这次按全新结果写")
            previous = {}
        carried = [key for key in RESULT_BLOCKS if key in previous and key not in result]
        for key in carried:
            result[key] = previous[key]
        if carried:
            result["carried_over"] = {
                "blocks": sorted(carried),
                "from_evaluated_at": previous.get("evaluated_at_local"),
                "from_eval_py_sha": (previous.get("code") or {}).get("eval_py_sha"),
                "note": "这些块沿用自上一次评测，未用当前代码重算",
            }
            print(f"==> --update 沿用上次结果：{'、'.join(sorted(carried))}")

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"{'label':<16}: {label}")

    def line(name: str, slot: dict, key: str = "accuracy") -> None:
        value = slot.get(key, slot.get("rate", 0.0))
        half = slot.get("ci95_half_pp")
        suffix = f" ± {half}pp" if half is not None else ""
        print(f"{name:<16}: {value:.1%}{suffix}  （{slot.get('correct', slot.get('passed'))}/{slot['total']}）")

    if "mmlu_gen" in result:
        line("MMLU(chat)", result["mmlu_gen"])
        print(f"{'':<16}  解析不出字母 {result['mmlu_gen']['unparsed']} 条")
    if "mmlu_gen_plain" in result:
        line("MMLU(plain)", result["mmlu_gen_plain"])
        print(f"{'':<16}  解析不出字母 {result['mmlu_gen_plain']['unparsed']} 条")
    if "ifollow" in result:
        line("指令遵循", result["ifollow"], key="rate")
        for category, slot in sorted(result["ifollow"]["by_category"].items()):
            print(f"{'':<16}    {category:<6} {slot['passed']}/{slot['total']}")
    if "gsm8k" in result:
        line("GSM8K(数学)", result["gsm8k"])
        print(f"{'':<16}  抽不出数字 {result['gsm8k']['unparsed']} 条")
    if "humaneval" in result:
        line("HumanEval(代码)", result["humaneval"])
    if "probes" in result:
        records = result["probes"]["records"]
        avg = sum(r["repeat_2gram"] for r in records) / len(records)
        print(f"{'采样复读率':<16}: 均值 {avg:.1%}")
    if "openqa" in result:
        print(f"{'OpenQA':<16}: 收集了 {result['openqa']['total']} 条回答，待裁判成对比较")

    print(f"{'耗时':<16}: {result['elapsed']}s")
    print(f"{'结果写入':<16}: {out_path}")
    print("=" * 60)

    missing = [name for name in ("mmlu_gen", "ifollow", "gsm8k", "humaneval", "openqa")
               if name not in result]
    if missing and not LIMIT:
        print(f"\n提示：这次没跑 {'、'.join(missing)}。要跑全量用 --only all")

    if not args.adapter:
        print("\n提示：以上是**基线**。跑完训练后用同样命令加 --adapter 再评一次，两组 json 才是可比的。")


if __name__ == "__main__":
    try:
        main()
    finally:
        # 临时 adapter 目录里只有软链，删掉本体不受影响
        for leftover in Path(tempfile.gettempdir()).glob("tpt-adapter-*"):
            shutil.rmtree(leftover, ignore_errors=True)
