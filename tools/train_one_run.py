#!/usr/bin/env python3
"""
train_one_run.py — Train + evaluate ONE (arch, seed, fold) combination.

Session-aware:
  - Skips immediately if metrics.json already exists in out_dir.
  - Resumes from the latest epoch checkpoint if training was interrupted.
  - Saves best checkpoint (by val EM) and latest checkpoint only — deletes
    intermediate checkpoints to keep disk usage low.

ECE is computed at VERIFIER LEVEL (argmax score per question vs EM correctness),
matching the project standard from exp2_q8_calibration.py. NOT candidate-level.

2Wiki zero-shot eval runs the best-epoch checkpoint against the 2Wiki dev set
without any retraining. Pass --skip_wiki2 if the 2Wiki files are not ready.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 tools/train_one_run.py \\
      --arch A --seed 42 --fold 0 \\
      --gold        data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
      --candidates  exp5b/candidates/dev_M5_7b_hightemp.jsonl \\
      --chains      exp0c/evidence/dev_K200_chains.jsonl \\
      --wiki2_gold  data/wiki2/raw/dev.json \\
      --wiki2_candidates exp_wiki2/candidates/dev_M5_sampling.jsonl \\
      --wiki2_chains     exp_wiki2/evidence/dev_wiki2_chains.jsonl \\
      --encoder     microsoft/deberta-v3-base \\
      --out_dir     exp_crosshop/runs \\
      --n_folds 5 --n_epochs 8 --batch_candidates 16 --gpu 0

Output layout:
  exp_crosshop/runs/{arch}_s{seed}_f{fold}/
    metrics.json          ← present = run is complete
    oof_preds.jsonl       ← one row per question {qid, pred, score, label, qtype}
    epoch_log.jsonl       ← per-epoch val EM
    checkpoint_best.pt    ← model state dict at best val EM
    config.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import string
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import project modules (must live in the same tools/ directory)
sys.path.insert(0, os.path.dirname(__file__))
from crosshop_model import CrossHopVerifier
from crosshop_data import (
    normalize, em_match, survives_stage1,
    iter_jsonl, load_hotpot_gold, load_2wiki_gold,
    load_candidates, load_hop_texts_from_chains,
)


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — MATHS UTILITIES
# ══════════════════════════════════════════════════════════════════════

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centred Kernel Alignment between rows of X and Y (n_samples × d)."""
    if X.shape[0] < 2:
        return float("nan")
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic_xy = np.linalg.norm(X @ Y.T, "fro") ** 2
    hsic_xx = np.linalg.norm(X @ X.T, "fro") ** 2
    hsic_yy = np.linalg.norm(Y @ Y.T, "fro") ** 2
    denom = math.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else float("nan")


def verifier_ece(scores: np.ndarray, labels: np.ndarray,
                 n_bins: int = 10) -> float:
    """ECE at VERIFIER level — must receive one (score, label) per question."""
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


def bootstrap_ci(vals: np.ndarray, n_boot: int = 2000,
                 seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = [rng.choice(vals, size=len(vals), replace=True).mean()
            for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET
# ══════════════════════════════════════════════════════════════════════

class HopPairDataset(Dataset):
    """Each item = one (question, candidate, hop1, hop2) tuple + label."""

    def __init__(
        self,
        instances: List[dict],
        tokenizer,
        max_length: int = 512,
        hop1_override: Optional[str] = None,   # None = use normal hop1
        hop2_override: Optional[str] = None,   # "" = mask that hop
        flat_mode: bool = False,               # concat hops into single pass
    ):
        self.instances = instances
        self.tok = tokenizer
        self.max_length = max_length
        self.h1_override = hop1_override
        self.h2_override = hop2_override
        self.flat = flat_mode

    def _encode(self, q: str, a: str, h: str) -> dict:
        text = f"Question: {q} Answer: {a} Evidence: {h}"
        return self.tok(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict:
        inst = self.instances[idx]
        q, a = inst["question"], inst["candidate"]
        h1 = inst["hop1"] if self.h1_override is None else self.h1_override
        h2 = inst["hop2"] if self.h2_override is None else self.h2_override

        if self.flat:
            # Concat both hops into a single evidence string
            enc_h1 = self._encode(q, a, h1 + " " + h2)
            enc_h2 = enc_h1   # Same sequence; model sees concat for both passes
        else:
            enc_h1 = self._encode(q, a, h1)
            enc_h2 = self._encode(q, a, h2)

        return {
            "h1_input_ids":      enc_h1["input_ids"].squeeze(0),
            "h1_attention_mask": enc_h1["attention_mask"].squeeze(0),
            "h2_input_ids":      enc_h2["input_ids"].squeeze(0),
            "h2_attention_mask": enc_h2["attention_mask"].squeeze(0),
            "label":    torch.tensor(inst["label"], dtype=torch.float32),
            "qid":      inst["qid"],
            "candidate":inst["candidate"],
            "cand_idx": inst["cand_idx"],
            "qtype":    inst.get("qtype", "bridge"),
        }


def collate_fn(batch: List[dict]) -> dict:
    keys_tensor = ["h1_input_ids", "h1_attention_mask",
                   "h2_input_ids", "h2_attention_mask", "label"]
    keys_list   = ["qid", "candidate", "cand_idx", "qtype"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys_tensor}
    out.update({k: [b[k] for b in batch] for k in keys_list})
    return out


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA BUILDING
# ══════════════════════════════════════════════════════════════════════

def build_instances(
    gold_map: Dict[str, dict],
    cand_map: Dict[str, List[str]],
    hop_map:  Dict[str, Tuple[str, str]],
) -> Tuple[List[dict], dict]:
    """Apply Stage 1 filter and build per-candidate instance dicts."""
    total = survived = correct_filtered = 0
    instances = []
    for qid, cands in cand_map.items():
        if qid not in gold_map or qid not in hop_map:
            continue
        gold   = gold_map[qid]["answer"]
        q      = gold_map[qid]["question"]
        qtype  = gold_map[qid].get("type", "bridge")
        h1, h2 = hop_map[qid]
        if not h1 and not h2:
            continue
        for ci, ans in enumerate(cands):
            total += 1
            if not survives_stage1(ans):
                if em_match(ans, gold):
                    correct_filtered += 1
                continue
            survived += 1
            instances.append({
                "qid": qid, "question": q, "candidate": ans,
                "hop1": h1, "hop2": h2, "label": em_match(ans, gold),
                "cand_idx": ci, "qtype": qtype,
            })
    stats = {
        "n_total": total, "n_survived": survived,
        "n_correct_filtered": correct_filtered,
        "n_questions": len({i["qid"] for i in instances}),
        "positive_rate": round(
            sum(i["label"] for i in instances) / max(len(instances), 1), 4),
    }
    return instances, stats


def make_folds(instances: List[dict], n_folds: int,
               seed: int) -> List[List[int]]:
    """Question-level stratified folds (all candidates of a question → same fold)."""
    qids = sorted({i["qid"] for i in instances})
    rng  = random.Random(seed)
    rng.shuffle(qids)
    qid_fold = {q: i % n_folds for i, q in enumerate(qids)}
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for idx, inst in enumerate(instances):
        folds[qid_fold[inst["qid"]]].append(idx)
    return folds


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING
# ══════════════════════════════════════════════════════════════════════

def train_epoch(
    model: CrossHopVerifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    label_smoothing: float,
    grad_clip: float,
    log_every: int = 100,
) -> float:
    model.train()
    total_loss, n_steps = 0.0, 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        h1_ids  = batch["h1_input_ids"].to(device)
        h1_mask = batch["h1_attention_mask"].to(device)
        h2_ids  = batch["h2_input_ids"].to(device)
        h2_mask = batch["h2_attention_mask"].to(device)
        labels  = batch["label"].to(device)

        logits = model(h1_ids, h1_mask, h2_ids, h2_mask)
        # Label smoothing
        y = labels * (1 - label_smoothing) + 0.5 * label_smoothing
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_steps += 1
        if (step + 1) % log_every == 0:
            elapsed = (time.time() - t0) / 60
            print(f"  step {step+1:4d}  loss={loss.item():.4f}  "
                  f"elapsed={elapsed:.1f}m")
    return total_loss / max(n_steps, 1)


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_eval(
    model: CrossHopVerifier,
    instances: List[dict],
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    mode: str = "main",       # main | flat | mask_h1 | mask_h2
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str],
           Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """Returns (logits, labels, qids, candidates, qtypes,
                z1_pre, z2_pre, z1_post, z2_post).
    z* arrays only for mode='main'; None otherwise."""
    model.eval()
    collect_repr = (mode == "main")

    flat  = (mode == "flat")
    h1_ov = "" if mode == "mask_h1" else None
    h2_ov = "" if mode == "mask_h2" else None

    ds = HopPairDataset(instances, tokenizer, max_length=max_length,
                        hop1_override=h1_ov, hop2_override=h2_ov, flat_mode=flat)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    all_logits, all_labels = [], []
    all_qids, all_cands, all_qtypes = [], [], []
    z1_pre_l, z2_pre_l, z1_post_l, z2_post_l = [], [], [], []

    for batch in loader:
        h1_ids  = batch["h1_input_ids"].to(device)
        h1_mask = batch["h1_attention_mask"].to(device)
        h2_ids  = batch["h2_input_ids"].to(device)
        h2_mask = batch["h2_attention_mask"].to(device)

        if collect_repr:
            logits, reprs = model(h1_ids, h1_mask, h2_ids, h2_mask,
                                  return_representations=True)
            z1_pre_l.append(reprs[0].cpu().numpy())
            z2_pre_l.append(reprs[1].cpu().numpy())
            z1_post_l.append(reprs[2].cpu().numpy())
            z2_post_l.append(reprs[3].cpu().numpy())
        else:
            logits = model(h1_ids, h1_mask, h2_ids, h2_mask)

        all_logits.extend(logits.cpu().numpy().tolist())
        all_labels.extend(batch["label"].numpy().tolist())
        all_qids.extend(batch["qid"])
        all_cands.extend(batch["candidate"])
        all_qtypes.extend(batch["qtype"])

    logits_arr = np.array(all_logits, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.float32)

    z1_pre  = np.concatenate(z1_pre_l,  axis=0) if collect_repr else None
    z2_pre  = np.concatenate(z2_pre_l,  axis=0) if collect_repr else None
    z1_post = np.concatenate(z1_post_l, axis=0) if collect_repr else None
    z2_post = np.concatenate(z2_post_l, axis=0) if collect_repr else None

    return (logits_arr, labels_arr, all_qids, all_cands, all_qtypes,
            z1_pre, z2_pre, z1_post, z2_post)


def compute_metrics(
    model: CrossHopVerifier,
    val_instances: List[dict],
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    arch_name: str,
) -> Tuple[dict, List[dict]]:
    """Full diagnostic evaluation. Returns (metrics_dict, oof_preds_list)."""
    # — Four scoring passes —
    print("    scoring: main ...")
    main_log, main_lbl, qids, cands, qtypes, z1_pre, z2_pre, z1_post, z2_post = \
        run_eval(model, val_instances, tokenizer, device, max_length, batch_size, "main")
    print("    scoring: flat ...")
    flat_log, *_ = run_eval(model, val_instances, tokenizer, device, max_length, batch_size, "flat")
    print("    scoring: mask_h1 (hop2 only) ...")
    mh1_log,  *_ = run_eval(model, val_instances, tokenizer, device, max_length, batch_size, "mask_h1")
    print("    scoring: mask_h2 (hop1 only) ...")
    mh2_log,  *_ = run_eval(model, val_instances, tokenizer, device, max_length, batch_size, "mask_h2")

    probs_main = 1.0 / (1.0 + np.exp(-main_log))
    probs_flat = 1.0 / (1.0 + np.exp(-flat_log))
    probs_mh1  = 1.0 / (1.0 + np.exp(-mh1_log))   # hop-2-only contribution
    probs_mh2  = 1.0 / (1.0 + np.exp(-mh2_log))   # hop-1-only contribution

    # — Group by question — argmax for EM + verifier-level ECE —
    qid_rows: Dict[str, List[int]] = defaultdict(list)
    for i, q in enumerate(qids):
        qid_rows[q].append(i)

    em_total = n_scored = 0
    v_scores, v_labels = [], []
    oof_preds = []
    type_em: Dict[str, List[int]] = defaultdict(list)

    for qid, rows in qid_rows.items():
        best = rows[int(np.argmax(probs_main[rows]))]
        correct = int(main_lbl[best] > 0.5)
        em_total += correct
        n_scored += 1
        v_scores.append(float(probs_main[best]))
        v_labels.append(float(main_lbl[best]))
        type_em[qtypes[best]].append(correct)
        oof_preds.append({
            "qid":     qid,
            "pred":    cands[best],
            "score":   float(probs_main[best]),
            "label":   correct,
            "qtype":   qtypes[best],
            "cand_idx": val_instances[best]["cand_idx"],
        })

    em = em_total / max(n_scored, 1)
    ece = verifier_ece(np.array(v_scores), np.array(v_labels))

    # — Pearson (flat vs min_hop) — candidate level —
    min_hop = np.minimum(probs_mh1, probs_mh2)
    pearson_flat_minhop = pearson(probs_flat, min_hop)

    # — CKA pre- and post-interaction —
    cka_pre  = linear_cka(z1_pre,  z2_pre)
    cka_post = linear_cka(z1_post, z2_post)

    # — Anchor delta: hop-2 anchoring signal —
    # For correct picks: score_hop2only - score_hop1only should be positive (answer in hop2)
    anchor_correct, anchor_wrong = [], []
    for qid, rows in qid_rows.items():
        best = rows[int(np.argmax(probs_main[rows]))]
        diff = float(probs_mh1[best]) - float(probs_mh2[best])  # hop2only - hop1only
        if main_lbl[best] > 0.5:
            anchor_correct.append(diff)
        else:
            anchor_wrong.append(diff)

    anc_corr = float(np.mean(anchor_correct)) if anchor_correct else float("nan")
    anc_wrong = float(np.mean(anchor_wrong))   if anchor_wrong  else float("nan")
    anc_delta = (anc_corr - anc_wrong) if not math.isnan(anc_corr) else float("nan")

    metrics = {
        "n_questions":        n_scored,
        "em":                 round(em, 4),
        "ece_verifier":       round(ece, 4),
        "pearson_flat_minhop":round(pearson_flat_minhop, 4),
        "cka_pre":            round(cka_pre, 4) if not math.isnan(cka_pre) else None,
        "cka_post":           round(cka_post, 4) if not math.isnan(cka_post) else None,
        "anchor_correct":     round(anc_corr,  4) if not math.isnan(anc_corr)  else None,
        "anchor_wrong":       round(anc_wrong, 4) if not math.isnan(anc_wrong) else None,
        "anchor_delta":       round(anc_delta, 4) if not math.isnan(anc_delta) else None,
        "type_em": {
            qt: round(sum(vals) / len(vals), 4)
            for qt, vals in type_em.items() if vals
        },
        "type_counts": {qt: len(vals) for qt, vals in type_em.items()},
    }
    return metrics, oof_preds


def eval_zero_shot(
    model: CrossHopVerifier,
    wiki2_gold_path: str,
    wiki2_cands_path: str,
    wiki2_chains_path: str,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> Optional[dict]:
    """2Wiki zero-shot evaluation. Returns metrics dict or None on error."""
    try:
        print("  [2Wiki zero-shot] Loading data ...")
        gold_map  = load_2wiki_gold(wiki2_gold_path)
        cand_map  = load_candidates(wiki2_cands_path)
        hop_map   = load_hop_texts_from_chains(wiki2_chains_path)
        instances, stats = build_instances(gold_map, cand_map, hop_map)
        print(f"    questions: {stats['n_questions']:,}  "
              f"candidates: {stats['n_survived']:,}  "
              f"positive_rate: {stats['positive_rate']:.3f}")
        metrics, _ = compute_metrics(
            model, instances, tokenizer, device, max_length, batch_size, "wiki2"
        )
        return {**metrics, "build_stats": stats}
    except Exception as e:
        print(f"  [2Wiki zero-shot] ERROR: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser()
    # Required
    ap.add_argument("--arch",        required=True, choices=["A","B","C"])
    ap.add_argument("--seed",        type=int, required=True)
    ap.add_argument("--fold",        type=int, required=True)
    ap.add_argument("--gold",        required=True)
    ap.add_argument("--candidates",  required=True)
    ap.add_argument("--chains",      required=True)
    ap.add_argument("--out_dir",     required=True)
    # 2Wiki (optional)
    ap.add_argument("--wiki2_gold", default="data/wiki2/raw/dev_normalized.json")
    ap.add_argument("--wiki2_candidates", default="exp_wiki2/candidates/dev_M5_sampling.jsonl")
    ap.add_argument("--wiki2_chains",     default="exp_wiki2/evidence/dev_wiki2_chains.jsonl")
    ap.add_argument("--skip_wiki2",  action="store_true")
    # Model / training
    ap.add_argument("--encoder",     default="microsoft/deberta-v3-base")
    ap.add_argument("--n_folds",     type=int, default=5)
    ap.add_argument("--n_epochs",    type=int, default=8)
    ap.add_argument("--batch_candidates", type=int, default=16)
    ap.add_argument("--encoder_lr",  type=float, default=2e-5)
    ap.add_argument("--head_lr",     type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--grad_clip",   type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--max_length",  type=int, default=512)
    ap.add_argument("--gpu",         type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()

    # ── output directory ──
    run_tag = f"{args.arch}_s{args.seed}_f{args.fold}"
    run_dir = Path(args.out_dir) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "metrics.json"

    # ── session-aware skip ──
    if done_path.exists():
        print(f"[train_one_run] SKIP — already complete: {done_path}")
        return

    print(f"\n{'='*72}")
    print(f"  arch={args.arch}  seed={args.seed}  fold={args.fold}/{args.n_folds}")
    print(f"  out: {run_dir}")
    print(f"{'='*72}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # ── seeds ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── load data ──
    print("Loading data ...")
    gold_map = load_hotpot_gold(args.gold)
    cand_map = load_candidates(args.candidates)
    hop_map  = load_hop_texts_from_chains(args.chains)
    instances, data_stats = build_instances(gold_map, cand_map, hop_map)

    print(f"  questions: {data_stats['n_questions']:,}  "
          f"candidates: {data_stats['n_survived']:,}  "
          f"positive_rate: {data_stats['positive_rate']:.3f}  "
          f"correct_filtered: {data_stats['n_correct_filtered']}")
    assert data_stats["n_correct_filtered"] == 0, \
        "Stage 1 filtered out correct answers — check Stage 1 rules"

    # ── CV folds ──
    folds   = make_folds(instances, args.n_folds, seed=args.seed)
    val_idx = folds[args.fold]
    trn_idx = [i for f, idxs in enumerate(folds) for i in idxs if f != args.fold]
    trn_instances = [instances[i] for i in trn_idx]
    val_instances = [instances[i] for i in val_idx]
    print(f"  train: {len(trn_instances):,}  val: {len(val_instances):,}")

    # ── tokenizer + model ──
    print(f"Loading tokenizer + model ({args.arch}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = CrossHopVerifier(
        encoder_name=args.encoder,
        arch=args.arch,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    # ── optimizer ──
    enc_params  = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": enc_params,  "lr": args.encoder_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=0.01)

    trn_ds = HopPairDataset(trn_instances, tokenizer, max_length=args.max_length)
    trn_loader = DataLoader(trn_ds, batch_size=args.batch_candidates,
                            shuffle=True, collate_fn=collate_fn, num_workers=0)
    n_steps_total = len(trn_loader) * args.n_epochs
    warmup_steps  = int(n_steps_total * args.warmup_frac)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=n_steps_total,
    )

    # ── checkpoint resume ──
    start_epoch = 0
    best_em = -1.0
    epoch_log = []
    best_ckpt_path = run_dir / "checkpoint_best.pt"
    latest_ckpt_path = run_dir / "checkpoint_latest.pt"

    if latest_ckpt_path.exists():
        print(f"  Resuming from checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_em     = ckpt.get("best_em", -1.0)
        epoch_log   = ckpt.get("epoch_log", [])
        print(f"  Resuming from epoch {start_epoch}  best_em={best_em:.4f}")

    # ── training loop ──
    print("Training ...")
    for epoch in range(start_epoch, args.n_epochs):
        ep_t0 = time.time()
        print(f"\n  Epoch {epoch+1}/{args.n_epochs}")
        avg_loss = train_epoch(
            model, trn_loader, optimizer, scheduler, device,
            args.label_smoothing, args.grad_clip,
        )
        ep_min = (time.time() - ep_t0) / 60

        # Quick val EM (no full diagnostics — those run at the end)
        model.eval()
        qid_best_probs: Dict[str, Tuple[float, float]] = {}
        val_ds = HopPairDataset(val_instances, tokenizer, max_length=args.max_length)
        val_loader = DataLoader(val_ds, batch_size=args.batch_candidates,
                                shuffle=False, collate_fn=collate_fn, num_workers=0)
        with torch.no_grad():
            for batch in val_loader:
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
                    if qid not in qid_best_probs or p > qid_best_probs[qid][0]:
                        qid_best_probs[qid] = (p, lbl)
        # qid_best_probs[qid] = (best_prob, label_of_best_candidate)
        # built above with argmax logic — this is verifier-level EM.
        val_em = sum(int(lbl > 0.5) for _, (p, lbl) in qid_best_probs.items()) \
                 / max(len(qid_best_probs), 1)

        print(f"  Epoch {epoch+1}  avg_loss={avg_loss:.4f}  "
              f"val_em={val_em:.4f}  time={ep_min:.1f}m")
        entry = {"epoch": epoch+1, "avg_loss": round(avg_loss, 5),
                 "val_em": round(val_em, 4), "time_min": round(ep_min, 1)}
        epoch_log.append(entry)
        # Write epoch log
        with open(run_dir / "epoch_log.jsonl", "w") as f:
            for e in epoch_log:
                f.write(json.dumps(e) + "\n")

        # Save best checkpoint
        if val_em > best_em:
            best_em = val_em
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  ✓ new best checkpoint  em={best_em:.4f}")

        # Save latest checkpoint (for resuming)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_em": best_em,
            "epoch_log": epoch_log,
        }, latest_ckpt_path)

    # ── final evaluation with best checkpoint ──
    print(f"\nLoading best checkpoint (em={best_em:.4f}) for final eval ...")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    print("Running full diagnostic evaluation ...")
    val_metrics, oof_preds = compute_metrics(
        model, val_instances, tokenizer, device,
        args.max_length, args.batch_candidates, args.arch,
    )

    # ── 2Wiki zero-shot ──
    wiki2_metrics = None
    if not args.skip_wiki2 and all(
        os.path.exists(p) for p in [
            args.wiki2_gold, args.wiki2_candidates, args.wiki2_chains
        ]
    ):
        print("\nRunning 2Wiki zero-shot eval ...")
        wiki2_metrics = eval_zero_shot(
            model, args.wiki2_gold, args.wiki2_candidates, args.wiki2_chains,
            tokenizer, device, args.max_length, args.batch_candidates,
        )
    else:
        print("\n2Wiki eval skipped.")

    # ── save results ──
    config = {
        "arch": args.arch, "seed": args.seed, "fold": args.fold,
        "n_folds": args.n_folds, "n_epochs": args.n_epochs,
        "encoder": args.encoder, "n_params": n_params,
        "batch_candidates": args.batch_candidates,
        "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
        "label_smoothing": args.label_smoothing,
        "data_stats": data_stats,
        "best_em": round(best_em, 4),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    final_metrics = {
        **val_metrics,
        "arch": args.arch, "seed": args.seed, "fold": args.fold,
        "best_epoch_em": round(best_em, 4),
        "epoch_log": epoch_log,
        "wiki2_zero_shot": wiki2_metrics,
    }
    with open(done_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    with open(run_dir / "oof_preds.jsonl", "w") as f:
        for row in oof_preds:
            f.write(json.dumps(row) + "\n")

    # ── clean up latest checkpoint (run is complete) ──
    if latest_ckpt_path.exists():
        latest_ckpt_path.unlink()

    # ── print summary ──
    W = 72
    print(f"\n{'='*W}")
    print(f"  FINAL RESULTS  —  arch={args.arch}  seed={args.seed}  fold={args.fold}")
    print(f"{'='*W}")
    print(f"  EM (verifier)       : {val_metrics['em']:.4f}")
    print(f"  ECE (verifier-lvl)  : {val_metrics['ece_verifier']:.4f}")
    print(f"  Pearson(flat,minhop): {val_metrics['pearson_flat_minhop']:.4f}")
    print(f"  CKA pre/post        : {val_metrics['cka_pre']} / {val_metrics['cka_post']}")
    print(f"  Anchor delta        : {val_metrics['anchor_delta']}")
    if wiki2_metrics:
        print(f"  2Wiki zero-shot EM  : {wiki2_metrics['em']:.4f}")
    print(f"{'='*W}")
    print(f"  Saved to: {done_path}")


if __name__ == "__main__":
    main()