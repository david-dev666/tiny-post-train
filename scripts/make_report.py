"""把 evals/results/ 下的评测结果汇总成一张 HTML 报告。

为什么不用 TensorBoard / W&B
----------------------------
因为要展示的东西它们都不管：**多个模型 × 多个评测项的对照表 + 置信区间**。
而且这份报告是要拿去写 README、发给别人的，需要能离线打开、能一眼看出
「数据是不是在同一套题目上跑的」。

设计要点
--------
1. **零外部依赖**：图表是内联 SVG，不引 CDN。报告可以离线打开、可以进版本库
   （dashboard 那套 ECharts 走 CDN，这里刻意不跟它一样）
2. **可比性检查**：每条结果都带评测集指纹（sha256）。指纹不一致会在页面顶部
   报警 —— 拿不同的题目跑出来的分数并排放，是最容易犯也最致命的错
3. **顺带产出 Markdown 表格**：README 的「结果」表直接复制，不用手抄数字

用法
----
    python scripts/make_report.py                        # 扫 evals/results/
    python scripts/make_report.py --results evals/results --out /tmp/report.html
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# 主表里的评测项：(结果里的 key, 显示名, 越小越好?, 说明)
MAIN_SUITES = [
    ("mmlu_gen", "MMLU（chat）", False, "带对话模板 · 指令微调模型的口径"),
    ("mmlu_gen_plain", "MMLU（纯文本）", False, "同一批题、去掉 chat 包装 · 基座的口径"),
    ("ifollow", "指令遵循", False, "带硬约束的指令，看听不听话"),
    ("gsm8k", "GSM8K（数学）", False, "答案唯一，可验证"),
    ("humaneval", "HumanEval（代码）", False, "真执行 + 跑单测"),
]

# 每个模型固定一个颜色，方便跨图对照
PALETTE = ["#4c8dff", "#3fb950", "#e3b341", "#db61a2", "#a371f7", "#39c5cf"]

# 展示顺序 = **训练顺序**（base → SFT → DPO → …），官方 instruct 永远排最后当参考。
#
# 为什么用显式清单而不是字母序：字母序排出来是 base / instruct / sft-4b-v2 这种
# 跟训练过程毫无关系的顺序，读者看不出「这一行是在上一行基础上加了什么」——
# 而这张表的全部意义恰恰是看每一步的增量。
LABEL_ORDER = ("base-4b", "sft-4b-v2", "dpo-4b-open")

# 参考基线（官方对齐版）：排最后、单独隔一行、**不参与「领先」标记**。
# 它是标尺不是选手 —— 让它抢走 ▲ 就看不出「我们自己训的哪个最好」了。
REFERENCE_PREFIXES = ("instruct-",)


def is_reference(label: str) -> bool:
    return any(label.startswith(p) for p in REFERENCE_PREFIXES)


def order_results(results: list[dict]) -> list[dict]:
    def key(r: dict):
        label = r["label"]
        if is_reference(label):
            return (3, 0, label)
        if label in LABEL_ORDER:
            return (1, LABEL_ORDER.index(label), "")
        # 不在清单里的（新实验、截断跑）排在中间，按字母序保持确定性
        return (2, 0, label)

    return sorted(results, key=key)


def load_results(directory: Path) -> list[dict]:
    """读结果目录。judge-*.json 是裁判结果，单独处理。

    **探针一律不读**（`kind == "probe"`，由 `eval.py --probe` 打在 json 里）。
    探针是「换了解码参数」的对照实验，不是模型评测 —— 摆进同一张表会被读成
    「模型之间的差异」，而这正是本项目最忌讳的那种错（看不出来，因为不报错）。
    探针跑完写在 `evals/probes/`，一般根本不会出现在这个目录里；
    这里再拦一道是防它被手工挪进来。
    """
    results = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("judge-"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  跳过 {path.name}：{exc}")
            continue
        if "label" not in data:
            continue
        if data.get("kind") == "probe":
            print(f"  跳过 {path.name}：探针（换解码参数的对照实验），不进评测表")
            continue
        data["_file"] = path.name
        # 带 --limit 的是「截断跑」。**不能直接从表里排除** ——
        # 400 题的快速档同样是有效可比数据（三个模型截的是同一批），
        # 排除了就等于把结论藏起来了。改成标注题数，让人自己判断够不够。
        data["_partial"] = (data.get("requested") or {}).get("limit") or None
        results.append(data)
    return results


def load_judges(directory: Path) -> list[dict]:
    judges = []
    for path in sorted(directory.glob("judge-*.json")):
        try:
            judges.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return judges


# ------------------------------------------------------------------ 小工具


def pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def score_cell(slot: dict, best: float | None) -> str:
    """一个评测项在某个模型上的表现：百分比 + 置信区间 + 领先标记。"""
    if not slot:
        return '<span class="muted">—</span>'
    value = slot.get("accuracy", slot.get("rate"))
    half = slot.get("ci95_half_pp")
    correct = slot.get("correct", slot.get("passed"))
    total = slot.get("total")
    if value is None:
        return f'<span class="muted">{total} 条已收集</span>'
    lead = ' <span class="lead">▲</span>' if best is not None and abs(value - best) < 1e-9 else ""
    ci = f'<span class="ci">±{half}</span>' if half is not None else ""
    return (f'<b>{pct(value)}</b>{ci}{lead}'
            f'<div class="frac">{correct}/{total}</div>')


def bar_svg(value: float | None, half: float | None, color: str, width: int = 150) -> str:
    """横向条形图。内联 SVG，不依赖任何库。"""
    if value is None:
        return ""
    filled = max(2, round(value * width))
    ci_left = max(0, round((value - (half or 0) / 100) * width))
    ci_right = min(width, round((value + (half or 0) / 100) * width))
    return (
        f'<svg class="bar" width="{width}" height="12" viewBox="0 0 {width} 12">'
        f'<rect x="0" y="0" width="{width}" height="12" fill="#e9edf3" rx="3"/>'
        f'<rect x="0" y="0" width="{filled}" height="12" fill="{color}" rx="3"/>'
        f'<line x1="{ci_left}" y1="1" x2="{ci_right}" y2="1" stroke="#2b3138" stroke-width="2"/>'
        f"</svg>"
    )


def consistency_warning(results: list[dict]) -> str:
    """题目指纹不一致就报警 —— 不同题目的分数并排是最容易犯的错。"""
    problems = []
    for suite in ("ifollow", "mmlu"):
        shas = {}
        for data in results:
            info = (data.get("subsets") or {}).get(suite) or {}
            digest = info.get("sha256")
            count = info.get("count")
            if digest:
                shas.setdefault((digest, count), []).append(data["label"])
        if len(shas) > 1:
            detail = "；".join(
                f"{count} 条（{', '.join(labels)}）" for (_digest, count), labels in shas.items()
            )
            problems.append(f"{suite} 评测集不一致：{detail}")
    if not problems:
        return ""
    items = "".join(f"<li>{html.escape(p)}</li>" for p in problems)
    return (
        '<div class="alert"><b>可比性告警</b>'
        "<div>下列结果的评测集不是同一份，并排比较不成立：</div>"
        f"<ul>{items}</ul></div>"
    )


def engine_warning(results: list[dict]) -> str:
    """结果里混了不止一个推理引擎就报警。

    为什么这条和题目指纹一样要命：HF 与 vLLM 在 bf16 下是**两条数值路径**，
    实测同一批 50 题里逐题预测有 13 题翻转（聚合分差 3 题）。
    把两个引擎的分数并排列出来，读者会把它当成「模型之间的差异」——
    而实际上一部分差异是尺子造成的。这种错看不出来，因为它不报错。
    """
    engines = {}
    for data in results:
        name = data.get("engine") or "hf"   # 老结果没有这个字段，那时只有 hf
        engines.setdefault(name, []).append(data["label"])
    if len(engines) < 2:
        return ""
    detail = "；".join(
        f"<b>{html.escape(name)}</b>（{html.escape(', '.join(labels))}）"
        for name, labels in sorted(engines.items())
    )
    return (
        '<div class="alert"><b>推理引擎不一致</b>'
        "<div>本批结果的生成后端不统一。不同后端在 bf16 下逐题预测存在翻转"
        "（实测 50 题中 13 题，聚合分差 3 题），<b>并排比较会把后端差异计入模型差异</b>。</div>"
        f"<div>{detail}</div>"
        "<div>变更引擎须<b>整批重跑</b>，不支持部分补跑后拼接："
        "<code>python scripts/eval_pipeline.py --engine vllm --force ...</code></div></div>"
    )


STYLE_NAMES = {"mmlu_gen": "MMLU（chat）", "mmlu_gen_plain": "MMLU（纯文本）"}


# 「这个口径对它无效」的判定阈值：解析失败率 ≥ 99%。
#
# **不能写成 `unparsed == total`。** 400 题那会儿基座在 chat 口径下是干净的 400/400
# 全失败，等值判断能用；换到全量 14042 题之后有 **7 条**碰巧吐出了字母
# （14035/14042 失败，0.0%）—— 等值判断当场落空，而表上那一格照样是「0.0%」，
# 没有任何提示。这类 0 分最容易被当成「模型不会」，必须靠比例判。
UNUSABLE_UNPARSED_RATIO = 0.99


def _unusable_hits(results: list[dict]) -> list[tuple[str, str, int, int]]:
    """找出「某个模型在某个口径下几乎抽不出答案」的格子。

    返回 (标签, 口径名, 抽不出的题数, 总题数)。
    """
    hits = []
    for data in results:
        for key, name in STYLE_NAMES.items():
            slot = data.get(key) or {}
            total, unparsed = slot.get("total"), slot.get("unparsed")
            if total and unparsed is not None and unparsed / total >= UNUSABLE_UNPARSED_RATIO:
                hits.append((data["label"], name, unparsed, total))
    return hits


def style_footnote(results: list[dict]) -> str:
    """主表下方的注解：读完数字之后再解释「这一格为什么不算数」。

    刻意不放在页面顶部 —— 读者要先看到数字，再读判读说明。
    措辞保持陈述式：只写「观测到什么、成因是什么、该怎么读」，
    不写操作指引（重跑命令属于 `docs/01-评测.md`）。
    """
    hits = _unusable_hits(results)
    if not hits:
        return ""
    listed = "、".join(
        f"{html.escape(label)} 的 {html.escape(name)}" for label, name, _u, _t in hits
    )
    counts = "；".join(f"{unparsed}/{total}" for _l, _n, unparsed, total in hits)
    return (
        '<div class="note">'
        f"<p><b>口径无效：{listed}</b>。{counts} 题未抽出选项字母，按 0 分计。"
        "该结果表示口径不适用，不代表模型得分低。</p>"
        "<p>该口径将「答案：」置于 user 回合内，基座按续写文档处理；"
        "仅指令微调模型会输出选项字母。此项度量格式适配度，不代表知识水平。</p>"
        "<p>基座以「MMLU（纯文本）」列为准；「MMLU（chat）」仅在指令微调模型之间可比。"
        "口径定义见 <code>docs/01-评测.md</code>。</p>"
        "</div>"
    )


# 诊断要看的评测项。顺序 = 报告里的顺序。
DIAGNOSTIC_SUITES = (
    ("mmlu_gen", "MMLU（chat）"),
    ("mmlu_gen_plain", "MMLU（纯文本）"),
    ("ifollow", "指令遵循"),
    ("gsm8k", "GSM8K（数学）"),
    ("humaneval", "HumanEval（代码）"),
)

# 截断率超过这个数就提示。**但「截断」在不同评测项下含义完全不同**，
# 所以文案分项写 —— 统一说「生成长度给少了」是错的（实测 HumanEval 的截断是
# 模型写完了函数还在自测，MMLU 更是只要吐一个字母、截断根本无意义）。
# 详见 docs/01-评测.md 的「被截断在两个评测项里是两件事」。
TRUNCATION_ALARM = 0.20
TRUNCATION_MEANING = {
    "gsm8k": ("生成长度不足",
              "答案可能被截断在最终数值之前，<b>分数被系统性压低</b>；输出越长的模型受影响越大"),
    "ifollow": ("输出偏长",
                "多为模型输出冗长所致（字数类约束因此超出），并非生成预算不足"),
    "humaneval": ("函数已完成",
                  "模型在继续编写示例调用与自测；抽取逻辑在 <code>print(</code> 处截断，"
                  "<b>不影响判分</b>"),
}
# 只要吐一个字母的项（上限 8 token），「截断」是常态、没有诊断价值 —— 不提示。
TRUNCATION_SILENT = {"mmlu_gen", "mmlu_gen_plain"}


def suite_notes(key: str, slot: dict) -> list[str]:
    """每个评测项特有的可信度备注。"""
    total = slot.get("total") or 0
    notes = []
    if key.startswith("mmlu"):
        fallback = (slot.get("by_rule") or {}).get("fallback", 0)
        if total and slot.get("unparsed") == total:
            notes.append("全部题目解析失败，该口径不适用")
        elif fallback:
            notes.append(f"{fallback}/{total} 题使用兜底抽取（可能命中题干中的选项标号）")
    elif key == "gsm8k":
        follows = slot.get("follows_format")
        if follows is not None:
            notes.append(f"按指定格式作答 {follows}/{total}")
    elif key == "ifollow":
        bad = slot.get("failed_with_junk_tail")
        if bad:
            notes.append(f"{bad} 条含乱码尾且未通过")
        # 把「不含乱码会是多少分」摆出来 —— 乱码是模型的真实输出（判分照算），
        # 但它值多少分必须可见，否则 16pp 的差距会被读成能力问题
        raw, trimmed = slot.get("rate"), slot.get("rate_junk_trimmed")
        if raw is not None and trimmed is not None and abs(trimmed - raw) > 1e-6:
            notes.append(f"去除尾部乱码后 {trimmed:.1%}（{trimmed - raw:+.1%}）")
    return notes


def diagnostics_section(results: list[dict]) -> str:
    """「这个分数该不该信」—— 比分数本身更该先看的东西。

    分数对不对是第二位的：被截断的答案、靠兜底规则蒙出来的字母、拖在尾巴上的乱码，
    都会让一个看起来正常的数字失真。这些指标**不参与判分**，但要摆在明面上。
    """
    rows = []
    for data in results:
        for key, name in DIAGNOSTIC_SUITES:
            slot = data.get(key)
            if not slot:
                continue
            total = slot.get("total") or 0
            truncated = slot.get("truncated")
            junk = slot.get("junk_tail")
            notes = suite_notes(key, slot)
            alarming = False
            if truncated and total and truncated / total >= TRUNCATION_ALARM:
                short, why = TRUNCATION_MEANING.get(key, ("截断偏高", "生成长度可能不足"))
                if key not in TRUNCATION_SILENT:
                    notes.append(f"<b>{short} · {truncated / total:.0%}</b>：{why}")
                    # 只有 GSM8K 的截断会真的压低分数，那里才标红；
                    # 其余项标红等于把「正常现象」报成事故
                    alarming = key == "gsm8k"
            rows.append((data["label"], name, total, truncated, junk, notes, alarming))

    if not rows:
        return ""

    parts = ["<h2>可信度诊断</h2>",
             '<div class="sub">以下指标不参与判分，用于判断上表的可信度。'
             "被截断：生成用满 token 上限（<b>含义随评测项而异</b>，见备注）。"
             "乱码尾：回合结束前输出一个稀有 token"
             "（本项目 SFT 模型的已知缺陷，base 与 instruct 无此现象）。</div>",
             '<div class="card"><table><thead><tr>'
             "<th>模型</th><th>评测项</th><th>样本</th><th>被截断</th><th>乱码尾</th>"
             "<th>备注</th></tr></thead><tbody>"]
    for label, name, total, truncated, junk, notes, alarming in rows:
        def cell(value, red=False):
            if value is None:
                return '<span class="muted">—</span>'
            ratio = value / total if total else 0
            cls = ' style="color:#cf222e;font-weight:600"' if red else ""
            return f"<span{cls}>{value}</span>" + (f'<div class="frac">{ratio:.0%}</div>' if total else "")
        note_html = "；".join(notes) if notes else '<span class="muted">—</span>'
        parts.append(
            f"<tr><td>{html.escape(label)}</td><td>{html.escape(name)}</td>"
            f'<td class="num">{total}</td><td class="num">{cell(truncated, alarming)}</td>'
            f'<td class="num">{cell(junk)}</td><td class="meta">{note_html}</td></tr>'
        )
    parts.append("</tbody></table></div>")
    return "".join(parts)


def markdown_table(results: list[dict]) -> str:
    """给 README 用的 Markdown 表格。"""
    active = results
    suites = [(k, name) for k, name, _lower, _note in MAIN_SUITES if any(k in r for r in active)]
    if not suites:
        return ""
    partial = next((r["_partial"] for r in active if r.get("_partial")), None)
    lines = ["| 模型 | " + " | ".join(name for _k, name in suites) + " |",
             "| --- | " + " | ".join("---" for _ in suites) + " |"]
    for data in active:
        cells = []
        for key, _name in suites:
            slot = data.get(key)
            value = None if not slot else slot.get("accuracy", slot.get("rate"))
            half = None if not slot else slot.get("ci95_half_pp")
            cells.append("—" if value is None else f"{value * 100:.1f}%" + (f" ±{half}" if half else ""))
        lines.append(f"| `{data['label']}` | " + " | ".join(cells) + " |")

    # ⚠️ 这里必须报**实际评测题数**，不是题目文件行数。
    # `subsets.*.count` 是文件里的条目数（MMLU 全量 14042），但我们一直用
    # `--limit 400` 跑，早先的注解写成「mmlu_gen=14042 题」—— 读者会以为跑的是全量。
    evaluated = []
    for key, name in suites:
        slot = next((r.get(key) for r in active if r.get(key)), None)
        if slot and slot.get("total") is not None:
            evaluated.append(f"{name} {slot['total']}")
    note = f"> 实际评测题数：{'、'.join(evaluated)}（与随机基准比较时按各自的 n 算区间）。"

    # 题目文件指纹另一行：判断几次评测用的是不是同一份题
    first = active[0] if active else {}
    fingerprints = []
    for key, _name in suites:
        info = (first.get("subsets") or {}).get("mmlu" if key.startswith("mmlu") else key) or {}
        if info.get("count"):
            fingerprints.append(f"{key} {info['count']} 题 / {info['sha256'][:8]}")
    if fingerprints:
        note += f"\n>\n> 题目文件：{'、'.join(fingerprints)}"
    note += "\n>\n> 括号内为 Wilson 95% 置信区间半宽（pp）。"
    if partial:
        note += f"\n>\n> 本表为截断档（各项仅前 {partial} 条），不作最终结论。"
    return "\n".join(lines) + "\n\n" + note


# ------------------------------------------------------------------ 页面


CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin:0; padding:32px 28px 80px; background:#f6f8fa; color:#1f2328;
  font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB",sans-serif; }
h1 { font-size:24px; margin:0 0 6px; }
h2 { font-size:17px; margin:36px 0 12px; padding-bottom:6px; border-bottom:1px solid #d8dee4; }
.sub { color:#656d76; font-size:13px; margin-bottom:20px; }
.alert { background:#fff8e5; border:1px solid #e3b341; border-radius:8px; padding:14px 18px; margin:18px 0; }
.alert ul { margin:8px 0 0; padding-left:20px; }
.alert.compact { padding:8px 14px; margin:14px 0; font-size:13px; }
.note { background:#fff; border-left:3px solid #d0d7de; border-radius:0 6px 6px 0;
  padding:12px 16px; margin:14px 0 0; font-size:12.5px; color:#57606a; line-height:1.75; }
.note p { margin:0 0 7px; }
.note p:first-child { margin-bottom:9px; }
.note p:last-child { margin-bottom:0; }
.note code { background:#eaeef2; padding:1px 5px; border-radius:4px; }
.card { background:#fff; border:1px solid #d8dee4; border-radius:10px; padding:4px 0; overflow:auto; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { padding:10px 14px; text-align:left; border-bottom:1px solid #eaeef2; white-space:nowrap; }
th { background:#f6f8fa; font-weight:600; color:#57606a; position:sticky; top:0; }
tr:last-child td { border-bottom:none; }
td.num { font-variant-numeric:tabular-nums; }
.ci { color:#8c959f; font-size:11px; margin-left:5px; }
.frac { color:#8c959f; font-size:11px; }
.lead { background:#dafbe1; color:#116329; font-size:11px; font-weight:600;
  border-radius:10px; padding:1px 6px; margin-left:4px; white-space:nowrap; }
.muted { color:#b1b8c0; }
.bar { display:block; }
td .bar { margin-bottom:3px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; vertical-align:middle; }
pre { background:#0d1117; color:#c9d1d9; padding:16px; border-radius:8px; overflow:auto; font-size:12px; line-height:1.5; }
.meta { font-size:12px; color:#57606a; }
.meta code { background:#eaeef2; padding:1px 5px; border-radius:4px; }
.tag { display:inline-block; background:#eaeef2; color:#57606a; border-radius:20px; padding:1px 9px; font-size:11px; margin-left:6px; }
.tag.warn { background:#fff1c1; color:#8a6100; }
.tag.ref { background:#eef2ff; color:#3b4a9e; }

/* 居中 + 限宽：宽屏上表格不被拉成一条，窄屏上也不会贴边 */
.wrap { max-width:1240px; margin:0 auto; }

/* 参考基线（官方 instruct）整行去饱和 —— 一眼分清「标尺」和「选手」 */
tr.ref-row td { background:#fbfcfe; }
tr.ref-row td:first-child { border-left:3px solid #c8d2f5; }
tr.sep td { background:#f6f8fa; color:#57606a; font-size:12px;
  border-top:2px solid #d8dee4; border-bottom:1px solid #eaeef2; padding:8px 14px; }

/* 表变宽（手机上）以后全靠这两条才读得下去：悬浮高亮 + 首列吸附 */
tbody tr:hover td { background:#f3f7ff; }
tbody tr.ref-row:hover td { background:#f4f7fd; }
th:first-child, td:first-child { position:sticky; left:0; background:#fff; z-index:1; }
th:first-child { background:#f6f8fa; z-index:2; }
tr.sep td:first-child { background:#f6f8fa; }
tr.ref-row td:first-child { background:#fbfcfe; }

/* 小屏收紧内边距，别让两列就撑满屏幕 */
@media (max-width:720px) {
  body { padding:20px 12px 60px; }
  th, td { padding:8px 10px; font-size:12px; }
}
"""


def render(results: list[dict], judges: list[dict]) -> str:
    # 截断跑（--limit）也进主表：三个模型截的是同一批题，是有效可比数据，
    # 排掉就等于把结论藏起来了。截断这件事改用标注提示。
    active = order_results(results)
    colors = {data["label"]: PALETTE[i % len(PALETTE)] for i, data in enumerate(active)}

    parts = [
        "<!doctype html>", '<html lang="zh-CN"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>tiny-post-train · 模型评测报告</title>",
        f'<style>{CSS}</style></head><body><div class="wrap">',
        "<h1>模型评测报告</h1>",
        f'<div class="sub">生成时间　{time.strftime("%Y-%m-%d %H:%M:%S")}　·　'
        f"模型数　{len(active)}　·　数据源　<code>evals/results/</code></div>",
        consistency_warning(active),
        engine_warning(active),
    ]

    if not active:
        parts.append('<div class="alert">未找到完整结果。请先执行：'
                     "<code>python scripts/eval_pipeline.py --run ...</code></div>")
        return "\n".join(parts) + "</div></body></html>"

    # ---- 主对照表
    suites = [(k, name, note) for k, name, _lower, note in MAIN_SUITES if any(k in r for r in active)]
    if suites:
        parts.append("<h2>主指标对照</h2>")
        parts.append(
            '<div class="sub">按<b>训练顺序</b>排列（base → SFT → DPO），便于逐阶段观察增量。'
            "末行 <code>instruct-2507</code> 为官方对齐版，作上限参照，"
            "<b>不参与领先标记</b>。</div>"
        )
        parts.append('<div class="card"><table><thead><tr><th>模型</th>')
        for _key, name, note in suites:
            parts.append(f'<th>{html.escape(name)}<div class="frac">{html.escape(note)}</div></th>')
        parts.append("</tr></thead><tbody>")
        for idx, data in enumerate(active):
            ref = is_reference(data["label"])
            if ref and idx and not is_reference(active[idx - 1]["label"]):
                # 参考基线单独隔一行：它是标尺，不是选手
                parts.append(
                    f'<tr class="sep"><td colspan="{len(suites) + 1}">'
                    "以下为 <b>参考基线</b>（官方对齐版）—— 用来量差距，不参与「▲ 领先」标记</td></tr>"
                )
            parts.append(
                f'<tr class="{"ref-row" if ref else ""}">'
                f'<td><span class="dot" style="background:{colors[data["label"]]}"></span>'
                f'<b>{html.escape(data["label"])}</b>'
                + (f'<span class="tag warn">截断 {data["_partial"]} 条</span>'
                   if data.get("_partial") else "")
                + ('<span class="tag ref">参考</span>' if ref else "")
                + f'<div class="frac">{html.escape(Path(str(data.get("base_model_used", data.get("model", "")))).name)}'
                + (f' + {html.escape(Path(str(data["adapter"])).name)}' if data.get("adapter") else "")
                + "</div></td>"
            )
            for key, _name, _note in suites:
                slot = data.get(key) or {}
                # 「领先」只在**我们自己训的模型**之间比 —— 参考基线不参赛
                values = [r.get(key, {}).get("accuracy", r.get(key, {}).get("rate"))
                          for r in active if r.get(key) and not is_reference(r["label"])]
                best = None if ref else max([v for v in values if v is not None], default=None)
                value = slot.get("accuracy", slot.get("rate"))
                parts.append(
                    '<td class="num">'
                    + bar_svg(value, slot.get("ci95_half_pp"), colors[data["label"]])
                    + score_cell(slot, best)
                    + "</td>"
                )
            parts.append("</tr>")
        parts.append("</tbody></table></div>")
        # 读完数字再给「为什么这一格不算数」，免得读者对着一个 0 分发懵
        parts.append(style_footnote(active))

    # ---- 指令遵循分类别
    if any("by_category" in r.get("ifollow", {}) for r in active):
        categories = sorted({c for r in active for c in r.get("ifollow", {}).get("by_category", {})})
        parts.append("<h2>指令遵循 · 分类别</h2>")
        parts.append('<div class="card"><table><thead><tr><th>类别</th>')
        for data in active:
            parts.append(f"<th>{html.escape(data['label'])}</th>")
        parts.append("</tr></thead><tbody>")
        for category in categories:
            parts.append(f"<td>{html.escape(category)}</td>")
            for data in active:
                slot = (data.get("ifollow") or {}).get("by_category", {}).get(category)
                if not slot:
                    parts.append('<td class="num"><span class="muted">—</span></td>')
                    continue
                rate = slot["passed"] / slot["total"] if slot["total"] else 0.0
                parts.append(
                    f'<td class="num">{bar_svg(rate, None, colors[data["label"]], 90)}'
                    f'{slot["passed"]}/{slot["total"]}</td>'
                )
            parts.append("</tr>")
        parts.append("</tbody></table></div>")

    # ---- 可信度诊断（放在分数后面，但比分数更该先看）
    parts.append(diagnostics_section(active))

    # ---- OpenQA 裁判
    if judges:
        parts.append("<h2>OpenQA · 成对裁判</h2>")
        parts.append('<div class="card"><table><thead><tr>'
                     "<th>对照 A</th><th>实验 B</th><th>B 胜率</th><th>95% CI</th>"
                     "<th>胜/负/平</th><th>位置敏感</th><th>裁判</th></tr></thead><tbody>")
        for judge in judges:
            counts = judge.get("counts", {})
            low, high = (judge.get("b_win_rate_ci95") or [0, 0])
            spans = " CI 跨过 50%" if low <= 0.5 <= high else ""
            parts.append(
                f"<tr><td>{html.escape(str(judge.get('a', {}).get('label')))}</td>"
                f"<td>{html.escape(str(judge.get('b', {}).get('label')))}</td>"
                f'<td class="num"><b>{pct(judge.get("b_win_rate"))}</b></td>'
                f'<td class="num">{pct(low)} ~ {pct(high)}'
                f'<div class="frac">{"无区分度" if spans else "有区分度"}</div></td>'
                f'<td class="num">{counts.get("b", 0)}/{counts.get("a", 0)}/{counts.get("tie", 0)}</td>'
                f'<td class="num">{counts.get("position_sensitive", 0)}'
                f'<div class="frac">交换顺序后翻转</div></td>'
                f"<td class=\"meta\">{html.escape(Path(str(judge.get('judge', ''))).name)}</td></tr>"
            )
        parts.append("</tbody></table></div>")
        parts.append('<div class="sub">位置敏感率偏高表示裁判本身不可靠；'
                     "交换顺序后结论翻转的样本已剔除，不计入胜率。</div>")

    # ---- Markdown
    markdown = markdown_table(results)
    if markdown:
        parts.append("<h2>Markdown 表格（供 README 使用）</h2>")
        parts.append(f"<pre>{html.escape(markdown)}</pre>")

    # ---- 元信息
    parts.append("<h2>运行配置与可复现信息</h2>")
    parts.append('<div class="card"><table><thead><tr>'
                 "<th>模型</th><th>推理引擎</th><th>对话模板</th><th>评测集指纹</th>"
                 "<th>生成长度</th><th>时间</th></tr></thead><tbody>")
    for data in active:
        subsets = data.get("subsets") or {}
        fingerprints = "　".join(
            f"{key}:{info['count']}题/{info['sha256'][:8]}"
            for key, info in subsets.items() if info.get("count")
        )
        generation = data.get("generation") or {}
        lengths = "　".join(
            f"{key.replace('_max_new_tokens', '')}={value}"
            for key, value in generation.items() if key.endswith("_max_new_tokens")
        )
        template = (data.get("chat_template") or {}).get("resolved", "")
        # 离线重判过的结果，eval_py_sha 已经对不上当时代码了，必须标出来，
        # 否则「三个模型代码指纹不一致」会被误读成跑的时候用了不同版本
        rescored = ""
        if data.get("rescored_at"):
            title = html.escape(str(data.get("rescore_note", "")))
            rescored = (f'<span class="tag warn" title="{title}">'
                        f'{html.escape(str(data["rescored_at"]))} 离线重判</span>')
        carried = data.get("carried_over")
        if carried:
            title = html.escape(str(carried.get("note", "")))
            rescored += (f'<span class="tag" title="{title}">'
                         f'{len(carried.get("blocks", []))} 块沿用上次</span>')
        engine_name = str(data.get("engine") or "hf")
        engine_cfg = data.get("engine_config") or {}
        # 引擎的关键旋钮要露出来：vLLM 关掉 enforce_eager 就不可复现了，
        # 这个信息不该藏在 json 里
        knobs = "　".join(
            f"{key}={engine_cfg[key]}"
            for key in ("vllm_version", "enforce_eager", "enable_prefix_caching", "lora")
            if key in engine_cfg
        )
        parts.append(
            f'<tr><td><b>{html.escape(data["label"])}</b>{rescored}'
            f'<div class="frac">{html.escape(data["_file"])}</div></td>'
            f'<td class="meta"><b>{html.escape(engine_name)}</b>'
            f'<div class="frac">{html.escape(knobs)}</div></td>'
            f'<td class="meta">{html.escape(template)}</td>'
            f'<td class="meta">{html.escape(fingerprints)}</td>'
            f'<td class="meta">{html.escape(lengths)}</td>'
            f'<td class="meta">{html.escape(str(data.get("evaluated_at_local", "")))}'
            f'<div class="frac">判分栈 '
            f'{html.escape(str((data.get("code") or {}).get("scoring_stack_sha") or (data.get("code") or {}).get("eval_py_sha", ""))[:12])}'
            f'</div></td></tr>'
        )
    parts.append("</tbody></table></div>")

    partial = next((r["_partial"] for r in active if r.get("_partial")), None)
    if partial:
        parts.append(
            f'<div class="alert"><b>截断档（非最终结果）</b>'
            f"<div>各项仅评测前 {partial} 条，用于快速判断强弱序。"
            f"置信区间显著变宽，<b>不适用于最终结论</b>；"
            f"正式结论须去掉 <code>--limit</code> 跑全量。</div></div>"
        )

    parts.append("</div></body></html>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="把评测结果汇总成 HTML 报告")
    parser.add_argument("--results", default=str(PROJECT_DIR / "evals" / "results"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    directory = Path(args.results)
    if not directory.exists():
        raise SystemExit(f"!! 目录不存在：{directory}")

    results = load_results(directory)
    judges = load_judges(directory)
    print(f"==> 读到 {len(results)} 份结果、{len(judges)} 份裁判结果")

    page = render(results, judges)
    out_path = Path(args.out) if args.out else directory / "report.html"
    out_path.write_text(page, encoding="utf-8")
    print(f"==> 报告写入 {out_path}")
    print(f"    打开：open {out_path}")

    # GitHub Pages 用「main 分支 / docs」这个源时，**站点根目录就是 docs/**，
    # 所以首页必须是 docs/index.html（放在 evals/ 里在站点上永远访问不到）。
    # 报告本身是自包含 HTML（内联 CSS + 内联 SVG、零外部资源），可以直接发布；
    # docs/.nojekyll 关掉 Jekyll 处理，免得页面里的花括号被当成模板语法吃掉。
    # 两处都是同一个 render() 的产物 —— 不会出现「本地和线上不是同一份」。
    pages_path = PROJECT_DIR / "docs" / "index.html"
    pages_path.parent.mkdir(parents=True, exist_ok=True)
    pages_path.write_text(page, encoding="utf-8")
    print(f"==> GitHub Pages 首页写入 {pages_path}")

    markdown = markdown_table(order_results(results))
    if markdown:
        print("\n---- 复制到 README ----")
        print(markdown)


if __name__ == "__main__":
    main()
