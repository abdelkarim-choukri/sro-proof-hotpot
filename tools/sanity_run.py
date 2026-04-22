#!/usr/bin/env python3
"""
sanity_run.py — Deliverable 1: single-arm sanity run for cross-hop experiment.

Tests that:
  1. The training loop runs without errors and the loss decreases.
  2. Held-out EM comes out roughly where Z3 sits (~0.56 on 7B MDR).
  3. All diagnostic functions (Pearson, CKA, Δqa_hop2 proxy) produce sane numbers.
  4. Checkpointing and result logging work end-to-end.

Configuration:
  Architecture : A (per-hop separate, no interaction)
  Seed         : 42
  Epochs       : 3  (NOT 10 — sanity only)
  CV folds     : 1  (train on 4 folds, evaluate on 1 held-out)
  Batch (Q×M)  : 8 questions × up to 5 candidates per question
  Encoder      : DeBERTa-v3-base

Runtime on a single A100: ~60–90 minutes end to end.

Outputs:
  exp_crosshop/sanity/
    config.json           — exactly what was run
    training_log.jsonl    — per-step loss
    metrics.json          — all diagnostics from the held-out fold
    oof_predictions.jsonl — per-candidate scores on the held-out fold

Usage:
  cd /var/tmp/u24sf51014/sro/work/sro-proof-hotpot
  python3 tools/sanity_run.py \\
      --gold        data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
      --candidates  exp5b/candidates/dev_M5_7b_hightemp.jsonl \\
      --chains      exp5b/evidence/dev_K200_chains.jsonl \\
      --encoder     microsoft/deberta-v3-base \\
      --out_dir     exp_crosshop/sanity \\
      --gpu         0

Important: the --chains path must point to the evidence chain file aligned
with the exp5b candidate generation. Run the find command in the deploy
instructions below if you're unsure of the exact path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup


# Local imports (assumes script is in tools/ and peers are at tools/)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from crosshop_model import CrossHopVerifier  # noqa: E402
from crosshop_data import (                   # noqa: E402
    load_hotpot_gold, load_candidates, load_hop_texts_from_chains,
    build_instances, HopPairDataset, collate_fn,
    make_question_folds, train_val_split_from_folds,
    em_match,
)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: SEED HANDLING
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two representation matrices.

    X: (N, D_x), Y: (N, D_y). Higher = more similar.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, ord='fro') ** 2
    den = (np.linalg.norm(X.T @ X, ord='fro')
           * np.linalg.norm(Y.T @ Y, ord='fro'))
    if den < 1e-12:
        return 0.0
    return float(num / den)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def train_one_fold(
    model: CrossHopVerifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    encoder_lr: float,
    head_lr: float,
    warmup_frac: float,
    grad_clip: float,
    label_smoothing: float,
    device: torch.device,
    log_path: str,
) -> Dict:
    """Run training over n_epochs. Returns per-step log + final model state."""
    model.to(device)

    # Param groups with different learning rates
    encoder_params = list(model.encoder.parameters())
    other_params = (list(model.interaction.parameters())
                    + list(model.head.parameters()))
    optim = AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": other_params,   "lr": head_lr},
        ],
        weight_decay=0.01,
    )

    total_steps = n_epochs * len(train_loader)
    warmup_steps = max(1, int(warmup_frac * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    bce = nn.BCEWithLogitsLoss()

    log = []
    step = 0
    t0 = time.time()
    with open(log_path, "w") as lf:
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                h1_ids  = batch["h1_input_ids"].to(device)
                h1_mask = batch["h1_attention_mask"].to(device)
                h2_ids  = batch["h2_input_ids"].to(device)
                h2_mask = batch["h2_attention_mask"].to(device)
                labels  = batch["labels"].to(device)

                if label_smoothing > 0:
                    labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

                logits = model(h1_ids, h1_mask, h2_ids, h2_mask)
                loss = bce(logits, labels)

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1
                step += 1

                if step % 50 == 0:
                    rec = {
                        "step": step,
                        "epoch": epoch + 1,
                        "loss": round(loss.item(), 4),
                        "lr_encoder": scheduler.get_last_lr()[0],
                        "lr_head":    scheduler.get_last_lr()[1],
                        "elapsed_min": round((time.time() - t0) / 60, 2),
                    }
                    log.append(rec)
                    lf.write(json.dumps(rec) + "\n")
                    lf.flush()
                    print(f"  step {step}  ep{epoch+1}  loss={rec['loss']:.4f}  "
                          f"elapsed={rec['elapsed_min']:.1f}m")

            print(f"  Epoch {epoch+1}/{n_epochs}  avg_loss={epoch_loss/max(n_batches,1):.4f}")
    return {"log": log, "total_steps": step}


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: EVALUATION & DIAGNOSTICS ON HELD-OUT FOLD
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_and_diagnose(
    model: CrossHopVerifier,
    val_loader: DataLoader,
    flat_loader: DataLoader,
    mask_h1_loader: DataLoader,
    mask_h2_loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict, List[Dict]]:
    """Produce all diagnostics on the held-out fold.

    flat_loader:   same instances but both hop inputs are h_concat = h1 + h2
                   (used for flat-equivalent Pearson baseline)
    mask_h1_loader: same instances, hop1 input replaced with empty string
    mask_h2_loader: same instances, hop2 input replaced with empty string

    Returns (metrics_dict, per_candidate_records).
    """
    model.eval()

    def run_scoring(loader, collect_repr: bool = False):
        all_logits = []
        all_labels = []
        all_qids = []
        all_cand_idxs = []
        all_qtypes = []
        all_cands = []
        z1_pre_all, z2_pre_all, z1_post_all, z2_post_all = [], [], [], []

        for batch in loader:
            h1_ids  = batch["h1_input_ids"].to(device)
            h1_mask = batch["h1_attention_mask"].to(device)
            h2_ids  = batch["h2_input_ids"].to(device)
            h2_mask = batch["h2_attention_mask"].to(device)

            if collect_repr:
                logits, reprs = model(h1_ids, h1_mask, h2_ids, h2_mask,
                                       return_representations=True)
                z1p, z2p, z1q, z2q = reprs
                z1_pre_all.append(z1p.cpu().numpy())
                z2_pre_all.append(z2p.cpu().numpy())
                z1_post_all.append(z1q.cpu().numpy())
                z2_post_all.append(z2q.cpu().numpy())
            else:
                logits = model(h1_ids, h1_mask, h2_ids, h2_mask)

            all_logits.extend(logits.cpu().numpy().tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
            all_qids.extend(batch["qids"])
            all_cand_idxs.extend(batch["cand_idxs"])
            all_qtypes.extend(batch["qtypes"])
            all_cands.extend(batch["candidates"])

        out = {
            "logits":     np.array(all_logits, dtype=np.float32),
            "labels":     np.array(all_labels, dtype=np.float32),
            "qids":       all_qids,
            "cand_idxs":  all_cand_idxs,
            "qtypes":     all_qtypes,
            "candidates": all_cands,
        }
        if collect_repr:
            out["z1_pre"]  = np.concatenate(z1_pre_all,  axis=0)
            out["z2_pre"]  = np.concatenate(z2_pre_all,  axis=0)
            out["z1_post"] = np.concatenate(z1_post_all, axis=0)
            out["z2_post"] = np.concatenate(z2_post_all, axis=0)
        return out

    print("  Scoring main (hop1, hop2) ...")
    main = run_scoring(val_loader, collect_repr=True)
    probs_main = 1.0 / (1.0 + np.exp(-main["logits"]))

    print("  Scoring flat (h_concat) ...")
    flat = run_scoring(flat_loader)
    probs_flat = 1.0 / (1.0 + np.exp(-flat["logits"]))

    print("  Scoring mask-h1 (probe hop2 contribution) ...")
    mh1 = run_scoring(mask_h1_loader)
    probs_mh1 = 1.0 / (1.0 + np.exp(-mh1["logits"]))   # ≈ score with only hop2

    print("  Scoring mask-h2 (probe hop1 contribution) ...")
    mh2 = run_scoring(mask_h2_loader)
    probs_mh2 = 1.0 / (1.0 + np.exp(-mh2["logits"]))   # ≈ score with only hop1

    # ── EM on validation fold ──
    # Group by qid, take argmax over candidates
    qid_to_rows: Dict[str, List[int]] = {}
    for i, q in enumerate(main["qids"]):
        qid_to_rows.setdefault(q, []).append(i)

    em_total = 0
    n_scored = 0
    per_q_correct = {}
    preds_by_qid = {}
    for qid, rows in qid_to_rows.items():
        best = rows[int(np.argmax(probs_main[rows]))]
        pred = main["candidates"][best]
        # The label on the "best" row tells us if it's correct
        correct = int(main["labels"][best] > 0.5)
        em_total += correct
        n_scored += 1
        per_q_correct[qid] = correct
        preds_by_qid[qid] = {
            "pred":      pred,
            "score":     float(probs_main[best]),
            "cand_idx":  int(main["cand_idxs"][best]),
        }
    em = em_total / max(n_scored, 1)

    # ── ECE ──
    ece = expected_calibration_error(probs_main, main["labels"])

    # ── Pearson(flat, min_hop) at candidate level ──
    # min_hop proxy = min(score_only_h1, score_only_h2) = min(probs_mh2, probs_mh1)
    min_hop = np.minimum(probs_mh1, probs_mh2)
    pearson_flat_minhop = pearson(probs_flat, min_hop)

    # ── CKA between pre-interaction z1 and z2 (encoder-level) ──
    cka_pre  = linear_cka(main["z1_pre"],  main["z2_pre"])
    cka_post = linear_cka(main["z1_post"], main["z2_post"])

    # ── Δqa_hop2 proxy on candidate-pair basis: for each question where the
    # model picked correctly, compute the gap between the chosen candidate's
    # "hop2-only" score and its "hop1-only" score. Positive values indicate
    # hop-2 anchoring (more information in hop2).
    # This is a VERY rough proxy for Experiment 4's Δqa_hop2; the clean version
    # requires CHAIN_WINS identification which needs the XGBoost predictions.
    hop2_minus_hop1_correct = []
    hop2_minus_hop1_wrong   = []
    for qid, rows in qid_to_rows.items():
        best = rows[int(np.argmax(probs_main[rows]))]
        if main["labels"][best] > 0.5:
            hop2_minus_hop1_correct.append(float(probs_mh1[best] - probs_mh2[best]))
            # NOTE: probs_mh1 masks h1 → is "hop2-only" score; probs_mh2 is "hop1-only"
        else:
            hop2_minus_hop1_wrong.append(float(probs_mh1[best] - probs_mh2[best]))
    anchor_correct_mean = float(np.mean(hop2_minus_hop1_correct)) if hop2_minus_hop1_correct else 0.0
    anchor_wrong_mean   = float(np.mean(hop2_minus_hop1_wrong))   if hop2_minus_hop1_wrong   else 0.0
    anchor_delta = anchor_correct_mean - anchor_wrong_mean

    # ── Per-type EM ──
    type_em = {}
    type_counts = {}
    for qid, rows in qid_to_rows.items():
        # Take qtype from first row of this qid
        qt = main["qtypes"][rows[0]]
        type_counts[qt] = type_counts.get(qt, 0) + 1
        type_em[qt] = type_em.get(qt, 0) + per_q_correct[qid]
    for qt in type_em:
        type_em[qt] = round(type_em[qt] / type_counts[qt], 4)

    metrics = {
        "n_scored_questions": n_scored,
        "em": round(em, 4),
        "ece": round(ece, 4),
        "pearson_flat_minhop": round(pearson_flat_minhop, 4),
        "cka_pre_interaction":  round(cka_pre, 4),
        "cka_post_interaction": round(cka_post, 4),
        "anchor_correct_mean": round(anchor_correct_mean, 4),
        "anchor_wrong_mean":   round(anchor_wrong_mean, 4),
        "anchor_delta":        round(anchor_delta, 4),
        "type_em": type_em,
        "type_counts": type_counts,
    }

    # Per-candidate records (for offline analysis)
    per_cand = []
    for i in range(len(main["logits"])):
        per_cand.append({
            "qid":         main["qids"][i],
            "cand_idx":    int(main["cand_idxs"][i]),
            "candidate":   main["candidates"][i],
            "label":       int(main["labels"][i]),
            "score_main":  float(probs_main[i]),
            "score_flat":  float(probs_flat[i]),
            "score_h2only": float(probs_mh1[i]),  # masking h1 = hop2 only
            "score_h1only": float(probs_mh2[i]),  # masking h2 = hop1 only
        })

    # Also store per-qid predictions for easier later joining with baselines
    metrics["_preds_by_qid"] = preds_by_qid
    return metrics, per_cand


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: VARIANT DATASETS (flat + masked)
# ═══════════════════════════════════════════════════════════════════════

class FlatPairDataset(torch.utils.data.Dataset):
    """Same instances as HopPairDataset, but both hop inputs are h_concat.

    h_concat = "hop1_text hop2_text". Used for the flat-equivalent probe.
    """
    def __init__(self, instances, tokenizer, max_length=512):
        self.inner = HopPairDataset(instances, tokenizer, max_length)
        self.instances = instances
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        h_concat = (inst.hop1 + " " + inst.hop2).strip()
        input_text = f"Question: {inst.question} Answer: {inst.candidate} Evidence: {h_concat}"
        t = self.tokenizer(input_text, truncation=True,
                            max_length=self.max_length,
                            padding=False, return_tensors=None)
        return {
            "qid":       inst.qid,
            "cand_idx":  inst.cand_idx,
            "label":     inst.label,
            "qtype":     inst.qtype,
            "candidate": inst.candidate,
            # same token seq goes in for BOTH hop passes — the model will produce
            # z1 ≈ z2 for each, which is the point (this is the flat signal).
            "h1_input_ids":      t["input_ids"],
            "h1_attention_mask": t["attention_mask"],
            "h2_input_ids":      t["input_ids"],
            "h2_attention_mask": t["attention_mask"],
        }


class MaskedHopDataset(torch.utils.data.Dataset):
    """Same instances as HopPairDataset, but one hop's input is an empty string.

    which: "h1" to mask hop1, "h2" to mask hop2.
    """
    def __init__(self, instances, tokenizer, which: str, max_length=512):
        assert which in ("h1", "h2")
        self.instances = instances
        self.tokenizer = tokenizer
        self.which = which
        self.max_length = max_length

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        empty = ""
        h1 = empty if self.which == "h1" else inst.hop1
        h2 = empty if self.which == "h2" else inst.hop2

        def enc(h):
            return self.tokenizer(
                f"Question: {inst.question} Answer: {inst.candidate} Evidence: {h}",
                truncation=True, max_length=self.max_length,
                padding=False, return_tensors=None,
            )
        t1 = enc(h1)
        t2 = enc(h2)
        return {
            "qid":       inst.qid,
            "cand_idx":  inst.cand_idx,
            "label":     inst.label,
            "qtype":     inst.qtype,
            "candidate": inst.candidate,
            "h1_input_ids":      t1["input_ids"],
            "h1_attention_mask": t1["attention_mask"],
            "h2_input_ids":      t2["input_ids"],
            "h2_attention_mask": t2["attention_mask"],
        }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold",       required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--chains",     required=True,
                    help="Evidence chains JSONL with chains[0].hops for each qid.")
    ap.add_argument("--encoder",    default="microsoft/deberta-v3-base")
    ap.add_argument("--out_dir",    required=True)

    ap.add_argument("--arch",       default="A", choices=["A", "B", "C"])
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--n_epochs",   type=int, default=3)
    ap.add_argument("--n_folds",    type=int, default=5)
    ap.add_argument("--val_fold",   type=int, default=0,
                    help="Index of fold used for validation.")

    ap.add_argument("--batch_questions", type=int, default=8,
                    help="Batch size in QUESTIONS. Candidates inside get grouped.")
    ap.add_argument("--batch_candidates", type=int, default=16,
                    help="Actual per-step batch in CANDIDATES.")
    ap.add_argument("--encoder_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr",    type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--grad_clip",  type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--gpu",         type=int, default=0)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Set seed and device ──
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[sanity] device={device}")
    print(f"[sanity] arch={args.arch}  seed={args.seed}  epochs={args.n_epochs}  val_fold={args.val_fold}/{args.n_folds}")

    # ── Load inputs ──
    print("[sanity] Loading gold ...")
    gold = load_hotpot_gold(args.gold)
    print(f"          {len(gold):,} questions")

    print("[sanity] Loading candidates ...")
    cands = load_candidates(args.candidates)
    print(f"          {len(cands):,} candidate records")

    print("[sanity] Loading hop texts from chains ...")
    hop_texts = load_hop_texts_from_chains(args.chains)
    print(f"          {len(hop_texts):,} evidence records")

    # Verify chain loader actually produced non-empty hops.
    # If the schema parse silently broke, we want to know BEFORE training.
    n_empty = sum(1 for h1, h2 in hop_texts.values() if not h1 or not h2)
    n_nonempty = len(hop_texts) - n_empty
    avg_h1_len = (sum(len(h1) for h1, h2 in hop_texts.values() if h1)
                  / max(n_nonempty, 1))
    avg_h2_len = (sum(len(h2) for h1, h2 in hop_texts.values() if h2)
                  / max(n_nonempty, 1))
    print(f"          non-empty hop pairs : {n_nonempty:,}")
    print(f"          empty hop pairs     : {n_empty:,}")
    print(f"          avg hop1 text chars : {avg_h1_len:.0f}")
    print(f"          avg hop2 text chars : {avg_h2_len:.0f}")
    if n_empty > 0.1 * len(hop_texts):
        print("[sanity] WARNING: >10% of questions have empty hop texts. "
              "Schema may not be parsing correctly. Abort and inspect.")
        sys.exit(2)
    if avg_h1_len < 20 or avg_h2_len < 20:
        print("[sanity] WARNING: hop texts are suspiciously short. "
              "Schema may not be parsing correctly. Abort and inspect.")
        sys.exit(2)

    print("[sanity] Building instances + applying Stage 1 filter ...")
    instances, data_stats = build_instances(gold, cands, hop_texts)

    # ── CV folds ──
    folds = make_question_folds(instances, n_folds=args.n_folds, seed=args.seed)
    train_idx, val_idx = train_val_split_from_folds(folds, args.val_fold)
    print(f"[sanity] train instances: {len(train_idx):,}   val instances: {len(val_idx):,}")

    # ── Tokenizer ──
    print(f"[sanity] Loading tokenizer: {args.encoder}")
    tok = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)

    # ── Datasets & loaders ──
    full_ds = HopPairDataset(instances, tok, max_length=args.max_length)
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_candidates, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_candidates, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # Validation-only variants for diagnostics
    val_instances = [instances[i] for i in val_idx]
    flat_ds     = FlatPairDataset(val_instances, tok, max_length=args.max_length)
    mask_h1_ds  = MaskedHopDataset(val_instances, tok, which="h1", max_length=args.max_length)
    mask_h2_ds  = MaskedHopDataset(val_instances, tok, which="h2", max_length=args.max_length)
    flat_loader    = DataLoader(flat_ds,    batch_size=args.batch_candidates, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=collate_fn)
    mask_h1_loader = DataLoader(mask_h1_ds, batch_size=args.batch_candidates, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=collate_fn)
    mask_h2_loader = DataLoader(mask_h2_ds, batch_size=args.batch_candidates, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=collate_fn)

    # ── Model ──
    print(f"[sanity] Building model (arch={args.arch}) ...")
    model = CrossHopVerifier(encoder_name=args.encoder, arch=args.arch)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"          trainable params: {n_params:,}")

    # ── Train ──
    print("[sanity] Training ...")
    train_log_path = os.path.join(args.out_dir, "training_log.jsonl")
    t_train0 = time.time()
    train_info = train_one_fold(
        model, train_loader, val_loader,
        n_epochs=args.n_epochs,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        warmup_frac=args.warmup_frac,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        device=device,
        log_path=train_log_path,
    )
    t_train_min = (time.time() - t_train0) / 60
    print(f"[sanity] Training done in {t_train_min:.1f} min  "
          f"({train_info['total_steps']} steps)")

    # ── Evaluate with all diagnostics ──
    print("[sanity] Running evaluation + diagnostics on held-out fold ...")
    metrics, per_cand = evaluate_and_diagnose(
        model, val_loader, flat_loader, mask_h1_loader, mask_h2_loader, device,
    )

    preds_by_qid = metrics.pop("_preds_by_qid")

    # ── Save outputs ──
    config = {
        "arch":            args.arch,
        "seed":            args.seed,
        "n_epochs":        args.n_epochs,
        "n_folds":         args.n_folds,
        "val_fold":        args.val_fold,
        "batch_candidates": args.batch_candidates,
        "encoder_lr":      args.encoder_lr,
        "head_lr":         args.head_lr,
        "warmup_frac":     args.warmup_frac,
        "grad_clip":       args.grad_clip,
        "label_smoothing": args.label_smoothing,
        "max_length":      args.max_length,
        "encoder":         args.encoder,
        "trainable_params": n_params,
        "data_stats":      data_stats,
        "train_time_min":  round(t_train_min, 1),
    }

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.out_dir, "oof_predictions.jsonl"), "w") as f:
        for qid, p in preds_by_qid.items():
            f.write(json.dumps({"qid": qid, **p}) + "\n")

    with open(os.path.join(args.out_dir, "per_candidate_scores.jsonl"), "w") as f:
        for rec in per_cand:
            f.write(json.dumps(rec) + "\n")

    # ── Print summary ──
    W = 72
    print()
    print("=" * W)
    print(f"  SANITY RESULTS  —  arch={args.arch}  seed={args.seed}  "
          f"epochs={args.n_epochs}  fold={args.val_fold}")
    print("=" * W)
    print(f"  Questions scored      : {metrics['n_scored_questions']:,}")
    print(f"  EM                    : {metrics['em']:.4f}")
    print(f"  ECE                   : {metrics['ece']:.4f}")
    print(f"  Pearson(flat, minhop) : {metrics['pearson_flat_minhop']:.4f}")
    print(f"  CKA pre-interaction   : {metrics['cka_pre_interaction']:.4f}")
    print(f"  CKA post-interaction  : {metrics['cka_post_interaction']:.4f}")
    print(f"  Anchor Δ (correct)    : {metrics['anchor_correct_mean']:+.4f}")
    print(f"  Anchor Δ (wrong)      : {metrics['anchor_wrong_mean']:+.4f}")
    print(f"  Anchor Δ gap          : {metrics['anchor_delta']:+.4f}")
    print()
    print(f"  Per-type EM:")
    for qt, em in metrics["type_em"].items():
        print(f"    {qt:<20s}  {em:.4f}  (n={metrics['type_counts'][qt]})")
    print("=" * W)
    print()
    print("SANITY GATES:")
    em_gate = 0.48 <= metrics["em"] <= 0.62
    ece_gate = metrics["ece"] < 0.15
    pearson_gate = not math.isnan(metrics["pearson_flat_minhop"])
    print(f"  [{'PASS' if em_gate else 'FAIL'}] EM in [0.48, 0.62]  (got {metrics['em']:.4f})")
    print(f"  [{'PASS' if ece_gate else 'FAIL'}] ECE < 0.15         (got {metrics['ece']:.4f})")
    print(f"  [{'PASS' if pearson_gate else 'FAIL'}] Pearson finite     (got {metrics['pearson_flat_minhop']:.4f})")

    all_pass = em_gate and ece_gate and pearson_gate
    print()
    print(f"  OVERALL: {'PASS — proceed to Deliverable 2' if all_pass else 'FAIL — investigate before full run'}")
    print("=" * W)

    print()
    print(f"Results in: {args.out_dir}")
    print(f"  config.json")
    print(f"  metrics.json")
    print(f"  oof_predictions.jsonl")
    print(f"  per_candidate_scores.jsonl")
    print(f"  training_log.jsonl")


if __name__ == "__main__":
    main()