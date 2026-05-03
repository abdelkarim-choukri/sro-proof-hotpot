#!/usr/bin/env python3
"""
sfav_train_one_run.py — Train + evaluate ONE (arch, seed, fold) for the SFAV experiment.

Trains either:
  arch=A    : Architecture A on diverse candidates (EM-only baseline, no sup head)
  arch=SFAV : Architecture A + supporting-fact auxiliary head (λ weighted)

This script is a fork of train_one_run.py adapted for the SFAV experiment.
Key differences:
  1. Uses sfav_model.SFAVVerifier (both for arch=A as a fallback and arch=SFAV)
  2. Uses sfav_data.SFAVDataset (which loads supporting-fact labels)
  3. For arch=A: passes no sup_labels → l_sup = 0 → identical to train_one_run.py
  4. For arch=SFAV: passes sup_labels → dual loss L_main + λ * L_sup
  5. Logs L_main and L_sup separately per epoch
  6. Reads from exp_distractor/ by default (new diverse candidate pool)

Session-aware: skips immediately if metrics.json exists. Resumes from
checkpoint_latest.pt if training was interrupted.

Usage:
  # Architecture A baseline (EM only, on diverse candidates)
  CUDA_VISIBLE_DEVICES=0 \\
  /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/sfav_train_one_run.py \\
      --arch A \\
      --seed 42 --fold 0 \\
      --gold       data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
      --candidates exp_distractor/candidates/dev_M5_diverse.jsonl \\
      --chains     exp_distractor/evidence/dev_distractor_chains.jsonl \\
      --out_dir    exp_sfav/runs \\
      --n_epochs 8

  # SFAV (Architecture A + supporting-fact head, λ=0.3)
  CUDA_VISIBLE_DEVICES=0 \\
  /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/sfav_train_one_run.py \\
      --arch SFAV \\
      --lam 0.3 \\
      --seed 42 --fold 0 \\
      --gold       data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
      --candidates exp_distractor/candidates/dev_M5_diverse.jsonl \\
      --chains     exp_distractor/evidence/dev_distractor_chains.jsonl \\
      --out_dir    exp_sfav/runs \\
      --n_epochs 8

Output layout (same structure as train_one_run.py):
  exp_sfav/runs/{arch}_lam{lam}_s{seed}_f{fold}/
    metrics.json          ← run complete
    oof_preds.jsonl       ← one row per question
    epoch_log.jsonl       ← per-epoch loss + EM + L_sup
    checkpoint_best.pt
    config.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from sfav_model import SFAVVerifier
from sfav_data import (
    load_supporting_facts, load_hop_texts_distractor,
    build_sfav_instances, SFAVDataset, collate_sfav,
    compute_label_stats, make_question_folds, train_val_split_from_folds,
)
from crosshop_data import load_hotpot_gold, load_candidates


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: MATHS UTILITIES  (copied from train_one_run.py)
# ═══════════════════════════════════════════════════════════════════════

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    if X.shape[0] < 2:
        return float("nan")
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic_xy = np.linalg.norm(X @ Y.T, "fro") ** 2
    hsic_xx = np.linalg.norm(X @ X.T, "fro") ** 2
    hsic_yy = np.linalg.norm(Y @ Y.T, "fro") ** 2
    denom = math.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else float("nan")


def verifier_ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(scores)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores >= lo) & (scores < hi)
        if lo == edges[-2]:
            mask = (scores >= lo) & (scores <= hi)
        cnt = mask.sum()
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(scores[mask].mean() - labels[mask].mean())
    return float(ece)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: TRAINING STEP
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(
    model: SFAVVerifier,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    label_smoothing: float,
    grad_clip: float,
    arch: str,
    log_every: int = 100,
) -> Tuple[float, float]:
    """Returns (avg_main_loss, avg_sup_loss)."""
    model.train()
    sum_main = sum_sup = 0.0
    n_steps = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        h1_ids  = batch["h1_input_ids"].to(device)
        h1_mask = batch["h1_attention_mask"].to(device)
        h2_ids  = batch["h2_input_ids"].to(device)
        h2_mask = batch["h2_attention_mask"].to(device)
        labels  = batch["label"].to(device)

        # Supporting-fact labels (only present for training dataset, not eval)
        h1_sup   = batch.get("h1_sup_labels")
        h2_sup   = batch.get("h2_sup_labels")
        h1_ev    = batch.get("h1_evidence_mask")
        h2_ev    = batch.get("h2_evidence_mask")

        use_sup  = (arch == "SFAV") and (h1_sup is not None)

        if use_sup:
            h1_sup = h1_sup.to(device)
            h2_sup = h2_sup.to(device)
            h1_ev  = h1_ev.to(device)  if h1_ev  is not None else None
            h2_ev  = h2_ev.to(device)  if h2_ev  is not None else None

            logits, l_sup = model(
                h1_ids, h1_mask, h2_ids, h2_mask,
                h1_sup_labels=h1_sup, h2_sup_labels=h2_sup,
                h1_evidence_mask=h1_ev, h2_evidence_mask=h2_ev,
            )
        else:
            # arch=A: call model without sup_labels → l_sup = 0
            logits = model(h1_ids, h1_mask, h2_ids, h2_mask)
            l_sup  = torch.tensor(0.0, device=device)

        # Main verifier loss with label smoothing
        y = labels * (1 - label_smoothing) + 0.5 * label_smoothing
        l_main = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss   = l_main + l_sup   # l_sup already includes λ for SFAV; 0 for A

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        sum_main += l_main.item()
        sum_sup  += l_sup.item()
        n_steps  += 1

        if (step + 1) % log_every == 0:
            elapsed = (time.time() - t0) / 60
            print(f"  step {step+1:4d}  l_main={l_main.item():.4f}  "
                  f"l_sup={l_sup.item():.4f}  elapsed={elapsed:.1f}m")

    return sum_main / max(n_steps, 1), sum_sup / max(n_steps, 1)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: EVALUATION  (identical logic to train_one_run.py)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model: SFAVVerifier,
    instances,
    tokenizer,
    sup_facts, para_sents,
    device, max_length, batch_size,
    mode: str = "main",   # main | flat | mask_h1 | mask_h2
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str],
           Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray]]:
    model.eval()
    collect_repr = (mode == "main")

    # Build dataset with appropriate hop overrides
    from sfav_data import SFAVInstance
    overridden: List[SFAVInstance] = []
    for inst in instances:
        if mode == "flat":
            oi = SFAVInstance(
                **{**inst.__dict__,
                   "hop1": inst.hop1 + " " + inst.hop2,
                   "hop2": inst.hop1 + " " + inst.hop2})
        elif mode == "mask_h1":
            oi = SFAVInstance(**{**inst.__dict__, "hop1": ""})
        elif mode == "mask_h2":
            oi = SFAVInstance(**{**inst.__dict__, "hop2": ""})
        else:
            oi = inst
        overridden.append(oi)

    ds = SFAVDataset(
        overridden, tokenizer, sup_facts, para_sents,
        max_length=max_length, at_inference=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_sfav, num_workers=0)

    all_logits, all_labels = [], []
    all_qids, all_cands, all_qtypes = [], [], []
    z1_l, z2_l, z1p_l, z2p_l = [], [], [], []

    for batch in loader:
        h1_ids  = batch["h1_input_ids"].to(device)
        h1_mask = batch["h1_attention_mask"].to(device)
        h2_ids  = batch["h2_input_ids"].to(device)
        h2_mask = batch["h2_attention_mask"].to(device)

        if collect_repr:
            result = model(h1_ids, h1_mask, h2_ids, h2_mask,
                           return_representations=True)
            logits, (z1, z2, z1p, z2p) = result
            z1_l.append(z1.cpu().numpy())
            z2_l.append(z2.cpu().numpy())
            z1p_l.append(z1p.cpu().numpy())
            z2p_l.append(z2p.cpu().numpy())
        else:
            logits = model(h1_ids, h1_mask, h2_ids, h2_mask)

        all_logits.extend(logits.cpu().numpy().tolist())
        all_labels.extend(batch["label"].numpy().tolist())
        all_qids.extend(batch["qid"])
        all_cands.extend(batch["candidate"])
        all_qtypes.extend(batch["qtype"])

    logits_arr = np.array(all_logits, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.float32)
    z1  = np.concatenate(z1_l,  0) if collect_repr else None
    z2  = np.concatenate(z2_l,  0) if collect_repr else None
    z1p = np.concatenate(z1p_l, 0) if collect_repr else None
    z2p = np.concatenate(z2p_l, 0) if collect_repr else None

    return logits_arr, labels_arr, all_qids, all_cands, all_qtypes, z1, z2, z1p, z2p


def compute_metrics(
    model, val_instances, tokenizer, sup_facts, para_sents,
    device, max_length, batch_size,
) -> Tuple[dict, List[dict]]:
    """Full 4-probe diagnostic evaluation."""
    print("    scoring: main ...")
    main_log, main_lbl, qids, cands, qtypes, z1, z2, z1p, z2p = run_inference(
        model, val_instances, tokenizer, sup_facts, para_sents,
        device, max_length, batch_size, "main")
    print("    scoring: flat ...")
    flat_log, *_ = run_inference(
        model, val_instances, tokenizer, sup_facts, para_sents,
        device, max_length, batch_size, "flat")
    print("    scoring: mask_h1 ...")
    mh1_log, *_ = run_inference(
        model, val_instances, tokenizer, sup_facts, para_sents,
        device, max_length, batch_size, "mask_h1")
    print("    scoring: mask_h2 ...")
    mh2_log, *_ = run_inference(
        model, val_instances, tokenizer, sup_facts, para_sents,
        device, max_length, batch_size, "mask_h2")

    p_main = 1 / (1 + np.exp(-main_log))
    p_flat = 1 / (1 + np.exp(-flat_log))
    p_mh1  = 1 / (1 + np.exp(-mh1_log))
    p_mh2  = 1 / (1 + np.exp(-mh2_log))

    qid_rows: Dict[str, List[int]] = defaultdict(list)
    for i, q in enumerate(qids):
        qid_rows[q].append(i)

    em_total = n_q = 0
    v_scores, v_labels = [], []
    oof_preds = []
    type_em: Dict[str, List[int]] = defaultdict(list)
    anchor_corr, anchor_wrong = [], []

    for qid, rows in qid_rows.items():
        best = rows[int(np.argmax(p_main[rows]))]
        correct = int(main_lbl[best] > 0.5)
        em_total += correct
        n_q += 1
        v_scores.append(float(p_main[best]))
        v_labels.append(float(main_lbl[best]))
        type_em[qtypes[best]].append(correct)
        diff = float(p_mh1[best]) - float(p_mh2[best])  # hop2only - hop1only
        (anchor_corr if main_lbl[best] > 0.5 else anchor_wrong).append(diff)
        oof_preds.append({
            "qid": qid, "pred": cands[best],
            "score": float(p_main[best]), "label": correct,
            "qtype": qtypes[best], "cand_idx": val_instances[best].cand_idx,
        })

    em  = em_total / max(n_q, 1)
    ece = verifier_ece(np.array(v_scores), np.array(v_labels))
    pr  = pearson(p_flat, np.minimum(p_mh1, p_mh2))
    cka = linear_cka(z1p, z2p) if z1p is not None else float("nan")
    cka_pre = linear_cka(z1, z2) if z1 is not None else float("nan")
    anc_c = float(np.mean(anchor_corr)) if anchor_corr else float("nan")
    anc_w = float(np.mean(anchor_wrong)) if anchor_wrong else float("nan")

    metrics = {
        "n_questions":          n_q,
        "em":                   round(em, 4),
        "ece_verifier":         round(ece, 4),
        "pearson_flat_minhop":  round(pr, 4) if not math.isnan(pr) else None,
        "cka_pre":              round(cka_pre, 4) if not math.isnan(cka_pre) else None,
        "cka_post":             round(cka, 4) if not math.isnan(cka) else None,
        "anchor_correct":       round(anc_c, 4) if not math.isnan(anc_c) else None,
        "anchor_wrong":         round(anc_w, 4) if not math.isnan(anc_w) else None,
        "anchor_delta":         round(anc_c - anc_w, 4)
                                if not (math.isnan(anc_c) or math.isnan(anc_w)) else None,
        "type_em": {
            qt: round(sum(v) / len(v), 4)
            for qt, v in type_em.items() if v
        },
        "type_counts": {qt: len(v) for qt, v in type_em.items()},
    }
    return metrics, oof_preds


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: MAIN
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch",       required=True, choices=["A", "SFAV"])
    ap.add_argument("--seed",       type=int, required=True)
    ap.add_argument("--fold",       type=int, required=True)
    ap.add_argument("--gold",       default="data/hotpotqa/raw/hotpot_dev_distractor_v1.json")
    ap.add_argument("--candidates", default="exp_distractor/candidates/dev_M5_diverse.jsonl")
    ap.add_argument("--chains",     default="exp_distractor/evidence/dev_distractor_chains.jsonl")
    ap.add_argument("--out_dir",    default="exp_sfav/runs")
    ap.add_argument("--encoder",    default="microsoft/deberta-v3-base")
    ap.add_argument("--n_folds",    type=int, default=5)
    ap.add_argument("--n_epochs",   type=int, default=8)
    ap.add_argument("--batch_candidates", type=int, default=16)
    ap.add_argument("--encoder_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr",    type=float, default=1e-4)
    ap.add_argument("--warmup_frac",type=float, default=0.1)
    ap.add_argument("--grad_clip",  type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--lam",        type=float, default=0.3,
                    help="λ: supporting-fact loss weight (SFAV only)")
    ap.add_argument("--pos_weight", type=float, default=8.0,
                    help="Class weight for positive tokens in sup-fact head")
    ap.add_argument("--gpu",        type=int, default=0)
    ap.add_argument("--validate_labels", action="store_true",
                    help="Run label-quality diagnostic before training and exit")
    return ap.parse_args()


def main():
    args = parse_args()
    lam_str = f"_lam{args.lam:.2f}" if args.arch == "SFAV" else ""
    run_tag = f"{args.arch}{lam_str}_s{args.seed}_f{args.fold}"
    run_dir = Path(args.out_dir) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "metrics.json"

    if done_path.exists():
        print(f"[sfav_train] SKIP — already complete: {done_path}")
        return

    print(f"\n{'='*72}")
    print(f"  arch={args.arch}  seed={args.seed}  fold={args.fold}/{args.n_folds}"
          + (f"  λ={args.lam}" if args.arch == "SFAV" else ""))
    print(f"  out: {run_dir}")
    print(f"{'='*72}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading data ...")
    gold_map   = load_hotpot_gold(args.gold)
    cand_map   = load_candidates(args.candidates)
    hop_map    = load_hop_texts_distractor(args.chains)
    sup_facts, para_sents = load_supporting_facts(args.gold)

    instances, data_stats = build_sfav_instances(gold_map, cand_map, hop_map)
    print(f"  questions : {data_stats['n_questions_q'] if 'n_questions_q' in data_stats else data_stats['n_common_qids']:,}")
    print(f"  instances : {data_stats['n_instances']:,}  "
          f"pos_rate={data_stats['positive_rate']:.3f}  "
          f"correct_filtered={data_stats['n_correct_filtered_out']}")
    assert data_stats["n_correct_filtered_out"] == 0, \
        "Stage 1 filtered correct answers — check Stage 1 rules"

    # ── Label quality diagnostic (optional) ───────────────────────────
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    if args.validate_labels:
        print("\nRunning supporting-fact label diagnostic on 200 samples ...")
        stats = compute_label_stats(
            instances, sup_facts, para_sents, tokenizer,
            max_length=args.max_length, n_sample=200,
        )
        print(json.dumps(stats, indent=2))
        print("\nLabel validation done. Re-run without --validate_labels to train.")
        return

    # ── CV folds ───────────────────────────────────────────────────────
    folds = make_question_folds(instances, n_folds=args.n_folds, seed=args.seed)
    trn_idx, val_idx = train_val_split_from_folds(folds, args.fold)
    trn_instances = [instances[i] for i in trn_idx]
    val_instances = [instances[i] for i in val_idx]
    print(f"  train: {len(trn_instances):,}  val: {len(val_instances):,}")

    # ── Model ──────────────────────────────────────────────────────────
    print(f"Loading model ({args.arch}, encoder={args.encoder}) ...")
    lam = args.lam if args.arch == "SFAV" else 0.0
    model = SFAVVerifier(
        encoder_name=args.encoder,
        lam=lam,
        pos_weight=args.pos_weight,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    # ── Optimiser + scheduler ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.encoder_lr, args.head_lr),
        weight_decay=0.01,
    )
    trn_ds = SFAVDataset(
        trn_instances, tokenizer, sup_facts, para_sents,
        max_length=args.max_length,
        at_inference=(args.arch != "SFAV"),
          # load sup_labels
    )
    trn_loader = DataLoader(
        trn_ds, batch_size=args.batch_candidates,
        shuffle=True, collate_fn=collate_sfav, num_workers=0,
    )
    n_steps_total = len(trn_loader) * args.n_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(n_steps_total * args.warmup_frac),
        num_training_steps=n_steps_total,
    )

    # ── Checkpoint resume ──────────────────────────────────────────────
    start_epoch = 0
    best_em = -1.0
    epoch_log = []
    best_ckpt  = run_dir / "checkpoint_best.pt"
    latest_ckpt = run_dir / "checkpoint_latest.pt"

    if latest_ckpt.exists():
        print(f"  Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_em     = ckpt.get("best_em", -1.0)
        epoch_log   = ckpt.get("epoch_log", [])
        print(f"  Resuming from epoch {start_epoch}  best_em={best_em:.4f}")

    # ── Training loop ──────────────────────────────────────────────────
    print("Training ...")
    for epoch in range(start_epoch, args.n_epochs):
        ep_t0 = time.time()
        print(f"\n  Epoch {epoch+1}/{args.n_epochs}")

        avg_main, avg_sup = train_epoch(
            model, trn_loader, optimizer, scheduler,
            device, args.label_smoothing, args.grad_clip, args.arch,
        )

        # Quick val EM for checkpoint selection
        model.eval()
        qid_best: Dict[str, Tuple[float, float]] = {}
        val_ds = SFAVDataset(
            val_instances, tokenizer, sup_facts, para_sents,
            max_length=args.max_length, at_inference=True,
        )
        val_ldr = DataLoader(
            val_ds, batch_size=args.batch_candidates, shuffle=False,
            collate_fn=collate_sfav, num_workers=0,
        )
        with torch.no_grad():
            for batch in val_ldr:
                logits = model(
                    batch["h1_input_ids"].to(device),
                    batch["h1_attention_mask"].to(device),
                    batch["h2_input_ids"].to(device),
                    batch["h2_attention_mask"].to(device),
                )
                probs = torch.sigmoid(logits).cpu().numpy()
                for i, qid in enumerate(batch["qid"]):
                    p = float(probs[i])
                    lbl = float(batch["label"][i])
                    if qid not in qid_best or p > qid_best[qid][0]:
                        qid_best[qid] = (p, lbl)
        val_em = sum(int(lbl > 0.5) for _, (p, lbl) in qid_best.items()) \
                 / max(len(qid_best), 1)
        ep_min = (time.time() - ep_t0) / 60

        print(f"  Epoch {epoch+1}  l_main={avg_main:.4f}  "
              f"l_sup={avg_sup:.4f}  val_em={val_em:.4f}  time={ep_min:.1f}m")

        entry = {
            "epoch": epoch + 1, "avg_main_loss": round(avg_main, 5),
            "avg_sup_loss": round(avg_sup, 5), "val_em": round(val_em, 4),
            "time_min": round(ep_min, 1),
        }
        epoch_log.append(entry)
        with open(run_dir / "epoch_log.jsonl", "w") as f:
            for e in epoch_log:
                f.write(json.dumps(e) + "\n")

        if val_em > best_em:
            best_em = val_em
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ new best  em={best_em:.4f}")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best_em": best_em, "epoch_log": epoch_log,
        }, latest_ckpt)

    # ── Final evaluation ───────────────────────────────────────────────
    print(f"\nLoading best checkpoint (em={best_em:.4f}) for final eval ...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print("Running full diagnostic evaluation ...")
    val_metrics, oof_preds = compute_metrics(
        model, val_instances, tokenizer, sup_facts, para_sents,
        device, args.max_length, args.batch_candidates,
    )

    # ── Save results ───────────────────────────────────────────────────
    config = {
        "arch": args.arch, "lam": args.lam, "seed": args.seed, "fold": args.fold,
        "n_folds": args.n_folds, "n_epochs": args.n_epochs,
        "encoder": args.encoder, "n_params": n_params,
        "data_stats": data_stats, "best_em": round(best_em, 4),
        "candidates": args.candidates, "chains": args.chains,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    final_metrics = {
        **val_metrics, "arch": args.arch, "lam": args.lam,
        "seed": args.seed, "fold": args.fold,
        "best_epoch_em": round(best_em, 4), "epoch_log": epoch_log,
    }
    with open(done_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    with open(run_dir / "oof_preds.jsonl", "w") as f:
        for row in oof_preds:
            f.write(json.dumps(row) + "\n")

    # Clean up latest checkpoint
    if latest_ckpt.exists():
        latest_ckpt.unlink()

    # ── Summary ────────────────────────────────────────────────────────
    W = 72
    print(f"\n{'='*W}")
    print(f"  FINAL  arch={args.arch}  seed={args.seed}  fold={args.fold}"
          + (f"  λ={args.lam}" if args.arch == "SFAV" else ""))
    print(f"{'='*W}")
    print(f"  EM (verifier)       : {val_metrics['em']:.4f}")
    print(f"  ECE                 : {val_metrics['ece_verifier']:.4f}")
    print(f"  Pearson(flat,minhop): {val_metrics['pearson_flat_minhop']}")
    print(f"  CKA pre/post        : {val_metrics['cka_pre']} / {val_metrics['cka_post']}")
    print(f"  Anchor delta        : {val_metrics['anchor_delta']}")
    print(f"{'='*W}")
    print(f"  Saved: {done_path}")


if __name__ == "__main__":
    main()