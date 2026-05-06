"""
Vortaz Labs — Fine-Tune Llama-3.2-1B on Science Birds Data (Unsloth)
======================================================================
Fair baseline: train a 1B LLM on the SAME 50K trajectories as the JEPA.
Smallest viable model — tests if scale matters for physics prediction.

Uses Unsloth + QLoRA (4-bit) for RTX 3050 compatibility (~3-4GB VRAM).

Usage: python training/finetune_llama_1b.py
"""

import os
import sys
import time
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    t0 = time.time()
    print("=" * 60)
    print("  VORTAZ LABS — Llama-3.2-1B Fine-Tuning (Unsloth)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("\n  WARNING: No CUDA GPU detected. Training will be very slow.")

    # Check for training data
    data_dir = Path(__file__).resolve().parent.parent / "data"
    train_file = data_dir / "finetune_train.jsonl"
    val_file = data_dir / "finetune_val.jsonl"

    if not train_file.exists():
        print(f"\n  ERROR: {train_file} not found.")
        print("  Run: python training/finetune_llm_data_prep.py first")
        sys.exit(1)

    with open(train_file) as f:
        n_train = sum(1 for _ in f)
    with open(val_file) as f:
        n_val = sum(1 for _ in f)
    print(f"  Train samples: {n_train:,}")
    print(f"  Val samples:   {n_val:,}")

    # --- Unsloth Setup ---
    print(f"\n  Loading Unsloth + Llama-3.2-1B-Instruct (4-bit)...")
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("\n  ERROR: Unsloth not installed.")
        print("  Install: pip install unsloth")
        sys.exit(1)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"  Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    # --- LoRA Config ---
    print(f"\n  Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- Dataset ---
    print(f"\n  Loading dataset...")
    from datasets import load_dataset

    dataset = load_dataset("json", data_files={
        "train": str(train_file),
        "validation": str(val_file),
    })

    def format_chat(example):
        text = tokenizer.apply_chat_template([
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    print(f"  Formatting with chat template...")
    dataset = dataset.map(format_chat, num_proc=1)
    print(f"  Train: {len(dataset['train']):,} | Val: {len(dataset['validation']):,}")

    # --- Training ---
    print(f"\n  Starting training...")
    from trl import SFTTrainer
    from transformers import TrainingArguments

    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints" / "llama1b_finetuned"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=50,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=1024,
        dataset_num_proc=1,
        packing=False,
        args=training_args,
    )

    print(f"\n  {'='*50}")
    print(f"  Training Llama-3.2-1B with LoRA")
    print(f"  Batch: 1 x 16 grad_accum = 16 effective")
    print(f"  Epochs: 1 | LR: 2e-4")
    print(f"  {'='*50}\n")

    train_result = trainer.train()

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed/60:.1f} minutes")
    print(f"  Final loss: {train_result.training_loss:.4f}")

    # --- Save ---
    print(f"\n  Saving model to {ckpt_dir}...")
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    merged_dir = ckpt_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    print(f"  Saving merged model to {merged_dir}...")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    total_time = time.time() - t0
    print(f"\n  {'='*50}")
    print(f"  DONE — Llama-3.2-1B fine-tuned in {total_time/60:.1f} min")
    print(f"  Adapter: {ckpt_dir}")
    print(f"  Merged:  {merged_dir}")
    print(f"  {'='*50}")


if __name__ == "__main__":
    main()
