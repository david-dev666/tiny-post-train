"""GRPO + 可验证奖励：复现 AI2 的 IF-RLVR（README 阶段 3）。

方法出处
--------
*Generalizing Verifiable Instruction Following*（AI2, NeurIPS 2025, arXiv 2507.02833）。
官方实现 = `allenai/open-instruct` 的 GRPO + `open_instruct/ground_truth_utils.py` 的
`IFEvalVerifier`（本仓库 vendor 在 `scripts/open_instruct/IFEvalG/`）。

论文给本项目的三条直接结论
--------------------------
1. **GRPO 优于 DPO**：同 prompt / 同起点 / 同验证函数，IFEval strict **89.65 vs 79.67**。
2. **`GRPO after DPO` 是最优前置**（89.65 > `GRPO after SFT` 85.77）→ 起点就是本项目
   的 `dpo-4b-open`，不用换。
3. **每条指令叠多条约束**（论文实测 5~6 条最好）→ 官方训练数据
   `IF_multi_constraints_upto5` 正是为此构造（本项目 95373 条，1~5 条约束）。

奖励 = 通过约束的比例（`ifrlvr_reward.score_one`），公式与官方逐行对齐：
`Σ per-constraint 0/1 ÷ 约束条数`。**不是 0/1 整题**，因为组内全对/全错时 GRPO 的
优势为 0、梯度为 0 —— 分档才有梯度，这也是「多约束」在论文里有效的同一件事。

工程约束（本项目特有，两条都是踩坑换来的）
------------------------------------------
**本脚本必须用 `vllm` 环境跑**（`/root/autodl-tmp/envs/vllm/bin/python`），不能用 `tpt`：

1. `tpt` 的 trl 0.24.0 × transformers 5.5.0 不兼容 —— `GRPOTrainer.__init__` 访问
   `model.warnings_issued`，该属性在 transformers 5 已被移除 → 当场 `AttributeError`
   （在此之前还有 12 个可选依赖探测因元组返回值而失效，见 `grpo_compat.py`）。
2. TRL 的 **server 模式**需要 `/generate` + `/init_communicator` + `/update_named_param`
   一整套端点（后两个用于把更新后的权重同步给推理端），**只有 TRL 自带的 `trl vllm-serve`
   提供**，普通的 `vllm serve` 没有这些路由 → server 模式在本项目走不通。
   实测 `vllm serve` 的 `/generate` 直接 404。

所以走 **colocate**：训练与 rollout 同进程，`vllm` 环境里 trl 1.14.1 + vllm 0.30 齐备
（装 trl 时 torch 2.13 / transformers 5.18 / vllm 0.30 版本均未变动，已核对）。

跑法
----
    VENV=/root/autodl-tmp/envs/vllm/bin/python

    # 冒烟：先看链路（判分调通、显存不爆、生成内容真的变），**不看 reward**
    $VENV scripts/train_grpo.py --limit 32 --max-steps 2 --num-generations 4 \\
        --max-completion-length 256 --output outputs/grpo-4b-smoke

    # 正式训练
    $VENV scripts/train_grpo.py --output outputs/grpo-4b-ifrlvr

    # 兜底（不依赖 vllm 的纯 HF 生成，慢，只在排查时用）
    $VENV scripts/train_grpo.py --vllm-mode none --limit 16 --max-steps 1 ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grpo_compat import patch_trl_optional_deps  # noqa: E402

patch_trl_optional_deps()  # 必须在 import trl 之前

from prompts import CLEAN_CHAT_TEMPLATE  # noqa: E402
from ifrlvr_reward import ifrlvr_reward  # noqa: E402

import json as _json  # noqa: E402
import time as _time  # noqa: E402

from transformers import TrainerCallback  # noqa: E402


class MetricsLogger(TrainerCallback):
    """把训练指标追加写到 `metrics.jsonl`，供 `dashboard/` 读取。

    为什么不用 `train_sft.MetricsLogger`：那个模块 import unsloth，而本脚本跑在
    `vllm` 环境（装了 vllm 但没有 unsloth）—— `import train_sft` 会直接失败。
    所以这里重写一份，**输出格式与它逐字段一致**（见 `dashboard/server.py` 的
    `_read_records` / `_read_meta`）。

    追加写、不加锁：进程被 kill 也只丢最后一行，不会毁掉已有数据。
    两个文件：
      metrics.jsonl   每行一条指标
      run_meta.json   开训时写一次，含 started_at / max_steps / data，看板用它算 ETA 和抽数据
    """

    def __init__(self, metrics_path: Path, meta: dict | None = None) -> None:
        self.metrics_path = Path(metrics_path)
        self.meta_path = self.metrics_path.with_name("run_meta.json")
        self.meta = dict(meta or {})

    def on_train_begin(self, args, state, control, **kwargs):
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "started_at": int(_time.time()),
            "max_steps": state.max_steps,
            "num_train_epochs": state.num_train_epochs,
            **self.meta,
        }
        self.meta_path.write_text(
            _json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step, "ts": int(_time.time())}
        if state.epoch is not None:
            record["epoch"] = round(float(state.epoch), 4)
        for key, value in logs.items():
            # 只留数字：TRL 的日志里混着 runtime / 提示信息，看板画不了
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            record[key] = float(value)
        try:
            with self.metrics_path.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(record) + "\n")
        except OSError as exc:
            print(f"[MetricsLogger] 写指标失败：{exc}")

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO + IF-RLVR 可验证奖励")
    p.add_argument("--model", default="outputs/dpo-4b-open-merged",
                   help="起点（合并好的完整模型；DPO 之后 = 论文最优前置）")
    p.add_argument("--data", default="data/processed/grpo-ifrlvr/train.jsonl")
    p.add_argument("--output", default="outputs/grpo-4b-ifrlvr")
    p.add_argument("--limit", type=int, default=0, help="只用前 n 条（冒烟用）")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--num-generations", type=int, default=8,
                   help="每条 prompt 采样几条（论文 16；单卡砍半）")
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--vllm-max-model-length", type=int, default=2048)
    p.add_argument("--vllm-enable-sleep-mode", action="store_true", default=True,
                   help="colocate 时让 vllm 在 rollout 间隙释放显存。32G 单卡强烈建议开")
    p.add_argument("--no-vllm-sleep-mode", dest="vllm_enable_sleep_mode",
                   action="store_false")
    p.add_argument("--lr", type=float, default=1e-6,
                   help="论文 5e-7（全参 8B）。这里 LoRA，取 1e-6 起")
    p.add_argument("--beta", type=float, default=0.04, help="KL 系数")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--length-penalty", type=float, default=0.0,
                   help="长度惩罚系数。**默认 0 = 与论文一致**，只为消融留着")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--vllm-mode", choices=["server", "colocate", "none"], default="colocate",
                   help="默认 colocate（vllm 同进程）。**改成 none 会退化成 HF 逐条生成，慢约 10 倍** —— "
                        "实测漏传参数就会静默走这条路（24.99s/step vs vllm 的 ~2s/step）")
    p.add_argument("--vllm-server-port", type=int, default=8000)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.30,
                   help="仅 colocate 模式使用")
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


def load_policy(args: argparse.Namespace):
    """加载合并好的起点模型，再挂一层新 LoRA。"""
    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"!! 起点模型不存在：{model_path}\n"
                 f"   先合并：python scripts/train_dpo.py --model outputs/sft-4b-v2-merged "
                 f"--sft-adapter outputs/dpo-4b-open --export-merged {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    # 模板必须与 SFT/DPO 一致，否则等于换了一份卷子
    tokenizer.chat_template = CLEAN_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    model.gradient_checkpointing_enable()

    # LoRA 交给 GRPOTrainer（TRL 1.x 的 `peft_config` 参数）去挂：它会自己 get_peft_model，
    # 并保证 rollout 与训练用的是同一份 adapter。手动包一层反而容易和 TRL 的内部假设打架。
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
    )
    return model, tokenizer, peft_config


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_DIR)

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"!! 训练数据不存在：{data_path}\n"
                 f"   先造：python scripts/make_grpo_ifrlvr_data.py")

    model, tokenizer, peft_config = load_policy(args)

    ds = load_dataset("json", data_files=str(data_path), split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    needed = {"prompt", "constraint_ids", "constraint_kwargs"}
    missing = needed - set(ds.column_names)
    if missing:
        sys.exit(f"!! 数据缺列 {sorted(missing)}（需要 {sorted(needed)}）")
    print(f"==> 训练 {len(ds)} 条；约束数分布 "
          f"{sorted({len(x) for x in ds[:2000]['constraint_ids']})}")

    use_vllm = args.vllm_mode != "none"
    # 把这行打出来。曾经因为漏传 --vllm-mode 静默退化成 HF 逐条生成，
    # 从 ~2s/step 变成 24.99s/step，而日志里没有任何提示。
    print(f"==> 生成后端：{'vLLM (' + args.vllm_mode + ')' if use_vllm else 'transformers（HF 逐条，慢约 10 倍）'}"
          f" | num_generations={args.num_generations}"
          f" | max_completion_length={args.max_completion_length}")

    cfg = GRPOConfig(
        output_dir=args.output,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=args.temperature,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        vllm_max_model_length=args.vllm_max_model_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=use_vllm,
        vllm_mode=args.vllm_mode if use_vllm else "colocate",
        vllm_server_port=args.vllm_server_port,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        log_completions=True,          # 冒烟要肉眼看生成变了没（判据不看 reward）
        report_to=["tensorboard"],
        seed=args.seed,
    )

    def reward_fn(completions, **reward_kwargs):
        """把 length_penalty 绑进去；默认 0，即与论文一致的纯「通过比例」。"""
        return ifrlvr_reward(completions, length_penalty=args.length_penalty, **reward_kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[MetricsLogger(Path(args.output) / "metrics.jsonl", meta={
            "recipe": "IF-RLVR (arXiv 2507.02833)",
            "start_from": args.model,
            # 看板的「数据抽样」靠这个字段（dashboard/server.py 的 /api/data）。
            # 存用户传入的原样字符串：脚本开头 chdir 到了 PROJECT_DIR，看板那边
            # 遇到相对路径会自己拼 ROOT，所以这里**不能**做 resolve/relative_to
            # （data_path 是相对的、PROJECT_DIR 是绝对的，relative_to 会直接抛错）。
            "data": args.data,
            "n_train": len(ds),
            "num_generations": args.num_generations,
            "lr": args.lr,
            "beta": args.beta,
            "length_penalty": args.length_penalty,
            "vllm_mode": args.vllm_mode,
            "max_completion_length": args.max_completion_length,
            "args": vars(args),
        })],
    )

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("==> 开始 GRPO 训练")
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"==> 完成：{args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
