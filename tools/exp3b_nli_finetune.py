#!/usr/bin/env python3
"""
exp3b_nli_finetune.py — Fine-tune nli-roberta-base on HotpotQA evidence pairs

Takes the train.jsonl / val.jsonl from data prep and fine-tunes the existing
nli-roberta-base model. Short training (2-3 epochs) since we're domain-adapting
a model that already understands NLI — we just need it to handle multi-hop
evidence patterns better.

Key design choices:
  - Freeze nothing — full fine-tuning (model is small, 125M params)
  - Low learning rate (2e-5) — standard for fine-tuning pretrained NLI
  - Early stopping on val loss — prevents overfitting
  - Save best checkpoint by val loss

Output: fine-tuned model saved to --out_model_dir

Usage:
    python3 tools/exp3b_nli_finetune.py \
        --base_model    /var/tmp/u24sf51014/sro/models/nli-roberta-base \
        --train_data    exp3b/nli_finetune/data/train.jsonl \
        --val_data      exp3b/nli_finetune/data/val.jsonl \
        --out_model_dir exp3b/nli_finetune/model \
        --epochs        3 \
        --batch_size    32 \
        --lr            2e-5 \
        --warmup_ratio  0.1 \
        --seed          42
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score


class NLIPairDataset(Dataset):
    """Simple dataset for (premise, hypothesis, label) triples."""

    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.pairs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.pairs.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        enc = self.tokenizer(
            p["premise"],
            p["hypothesis"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(p["label"], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    """Compute accuracy + macro F1 for trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro")

    # Per-class accuracy
    per_class = {}
    for cls in [0, 1, 2]:
        mask = labels == cls
        if mask.sum() > 0:
            per_class[f"acc_class_{cls}"] = float((preds[mask] == cls).mean())

    return {"accuracy": acc, "macro_f1": f1, **per_class}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model",    required=True,
                    help="Path to nli-roberta-base")
    ap.add_argument("--train_data",    required=True,
                    help="train.jsonl from data prep")
    ap.add_argument("--val_data",      required=True,
                    help="val.jsonl from data prep")
    ap.add_argument("--out_model_dir", required=True,
                    help="Where to save the fine-tuned model")
    ap.add_argument("--epochs",        type=int,   default=3)
    ap.add_argument("--batch_size",    type=int,   default=32)
    ap.add_argument("--lr",            type=float, default=2e-5)
    ap.add_argument("--warmup_ratio",  type=float, default=0.1)
    ap.add_argument("--weight_decay",  type=float, default=0.01)
    ap.add_argument("--max_length",    type=int,   default=256,
                    help="Max token length (256 is enough for sentence+answer)")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--fp16",          action="store_true",
                    help="Use mixed precision (recommended for GPU)")
    ap.add_argument("--patience",      type=int,   default=2,
                    help="Early stopping patience (in eval steps)")
    args = ap.parse_args()

    os.makedirs(args.out_model_dir, exist_ok=True)

    print(f"[finetune] Base model:  {args.base_model}")
    print(f"[finetune] Train data:  {args.train_data}")
    print(f"[finetune] Val data:    {args.val_data}")
    print(f"[finetune] Output:      {args.out_model_dir}")
    print(f"[finetune] Epochs={args.epochs}  BS={args.batch_size}  "
          f"LR={args.lr}  Warmup={args.warmup_ratio}")

    # ── load model + tokenizer ──
    print("[finetune] Loading model and tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=3
    )

    # Verify label mapping
    print(f"[finetune] Label mapping: {model.config.id2label}")

    # ── load datasets ──
    print("[finetune] Loading datasets ...")
    train_dataset = NLIPairDataset(args.train_data, tokenizer, args.max_length)
    val_dataset   = NLIPairDataset(args.val_data,   tokenizer, args.max_length)
    print(f"[finetune] Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    # ── training arguments ──
    log_dir = os.path.join(args.out_model_dir, "logs")
    training_args = TrainingArguments(
        output_dir=args.out_model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=log_dir,
        logging_steps=100,
        seed=args.seed,
        fp16=args.fp16 or torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",  # no wandb
        disable_tqdm=False,
    )

    # ── trainer ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # ── train ──
    print("[finetune] Starting training ...")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0
    print(f"[finetune] Training completed in {elapsed / 60:.1f} min")

    # ── evaluate on val ──
    print("[finetune] Evaluating on validation set ...")
    val_metrics = trainer.evaluate()
    print(f"[finetune] Val metrics: {json.dumps(val_metrics, indent=2)}")

    # ── save best model ──
    print(f"[finetune] Saving best model to {args.out_model_dir} ...")
    trainer.save_model(args.out_model_dir)
    tokenizer.save_pretrained(args.out_model_dir)

    # ── save training summary ──
    summary = {
        "base_model":     args.base_model,
        "train_pairs":    len(train_dataset),
        "val_pairs":      len(val_dataset),
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "lr":             args.lr,
        "warmup_ratio":   args.warmup_ratio,
        "elapsed_min":    round(elapsed / 60, 1),
        "train_loss":     round(train_result.training_loss, 4),
        "val_metrics":    val_metrics,
        "best_model_dir": args.out_model_dir,
    }
    summary_path = os.path.join(args.out_model_dir, "finetune_summary.json")
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"[finetune] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()