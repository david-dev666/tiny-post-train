"""Qwen3 LoRA SFT 最小可运行脚本。

用法见 docs/00-getting-started.md。
数据默认按 alpaca 字段 instruction / input / output 处理。
"""

import argparse
import glob
import os

import torch
from datasets import concatenate_datasets, load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

FALLBACK_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="本地基座权重路径或 HF/ModelScope id")
    p.add_argument("--data", required=True, help="本地数据目录或 HF 数据集名")
    p.add_argument("--output", required=True, help="输出目录")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--load-in-4bit", action="store_true", help="显存不足时打开")
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


def load_raw(data_path):
    """目录按后缀分组加载，json 和 jsonl 混放也不会解析错。"""
    if os.path.isdir(data_path):
        parts = []
        for ext in ("json", "jsonl"):
            files = sorted(glob.glob(os.path.join(data_path, f"*.{ext}")))
            if files:
                parts.append(load_dataset(ext, data_files=files, split="train"))
        if not parts:
            raise FileNotFoundError(f"{data_path} 里没找到 json/jsonl 文件")
        return parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    return load_dataset(data_path, split="train")


def build_user_text(example):
    user = (example.get("instruction") or "").strip()
    extra = (example.get("input") or "").strip()
    if extra:
        user = f"{user}\n{extra}"
    return user


def to_text(example, tokenizer, chat_template):
    messages = [
        {"role": "user", "content": build_user_text(example)},
        {"role": "assistant", "content": (example.get("output") or "").strip()},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, chat_template=chat_template
        )
    }


def build_config(args, common):
    try:
        return SFTConfig(max_seq_length=args.max_seq_len, **common)
    except TypeError:
        return SFTConfig(max_length=args.max_seq_len, **common)


def build_trainer(model, tokenizer, dataset, config):
    try:
        return SFTTrainer(
            model=model, processing_class=tokenizer,
            train_dataset=dataset, args=config,
        )
    except TypeError:
        return SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=dataset, args=config,
        )


def main():
    args = parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    chat_template = getattr(tokenizer, "chat_template", None) or FALLBACK_CHAT_TEMPLATE
    tokenizer.chat_template = chat_template

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = load_raw(args.data)
    dataset = dataset.map(
        lambda ex: to_text(ex, tokenizer, chat_template),
        remove_columns=dataset.column_names,
        desc="格式化数据",
    )
    print(f"样本数: {len(dataset)}")
    print("抽查一条:\n" + dataset[0]["text"][:600])

    bf16_ok = torch.cuda.is_bf16_supported()
    common = dict(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        optim="adamw_8bit",
        seed=args.seed,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = build_trainer(model, tokenizer, dataset, build_config(args, common))
    stats = trainer.train()
    print(f"训练完成: {stats.metrics}")

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"已保存到 {args.output}")


if __name__ == "__main__":
    main()
