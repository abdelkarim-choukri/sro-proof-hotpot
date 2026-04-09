#!/usr/bin/env python3
"""
exp_b1_stage1_filter.py  —  Phase B1: Stage 1 Garbage Filter

A logistic regression filter that removes garbage candidates before
the evidence-quality ranker (Stage 2) sees them.

Why this matters:
  In the monolithic model, is_bad consumes 71.5% of capacity.
  By handling garbage filtering in a dedicated Stage 1, Stage 2 can
  allocate full capacity to chain-aware evidence features.

Filter logic:
  Features: is_bad, is_unknown, answer_len_words, answer_len_chars
  Model: logistic regression (simple, interpretable hard wall)
  Label: 0 = garbage (is_bad OR is_unknown), 1 = survives

Critical validation:
  FALSE NEGATIVE RATE must be ~0%: the filter must NEVER remove a
  correct answer. Oracle EM before and after filtering must be identical.

Inputs:
  --hop_scores   exp0c/preds/dev_hop_scores.jsonl
  --gold         data/hotpotqa/raw/hotpot_dev_distractor_v1.json
  --out_jsonl    exp_phaseB/B1.1/filter_output/dev_stage1_filtered.jsonl
  --out_json     exp_phaseB/B1.1/filter_output/summary.json

Output JSONL — one record per question:
{
  "qid": "...",
  "gold": "...",
  "oracle_before": 1,
  "oracle_after": 1,
  "n_before": 5,
  "n_after": 3,
  "surviving_ids": [0, 2, 4],
  "filtered_ids": [1, 3]
}
"""

import argparse
import collections
import json
import math
import os
import re
import string
import sys
from typing import Any, Dict, List, Set

import numpy as np
from sklearn.linear_model import LogisticRegression


# ─────────────────────────── text utils ─────────────────────────────

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))

def is_bad_answer(ans: str) -> bool:
    """Identical to exp2_q2q3q4_chain_verifier.py."""
    a = ans.strip()
    if not a: return True
    low = a.lower()
    if low.startswith("[chain"): return True
    if "if the evidence does not contain" in low: return True
    if low.startswith("the evidence provided"): return True
    if low.startswith(("okay,", "alright,", "so,")): return True
    if low in {"unknown", "unk"}: return True
    if len(a) > 120: return True
    return False

def is_unknown_answer(ans: str) -> bool:
    return normalize(ans) in {"unknown", "unk", ""}


# ─────────────────────────── I/O utils ──────────────────────────────

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def extract_gold(gold_field: Any) -> str:
    if isinstance(gold_field, dict):
        return gold_field.get("answer", "")
    return str(gold_field) if gold_field else ""


# ─────────────────────────── feature extraction ──────────────────────

def stage1_features(answer: str) -> List[float]:
    """Four features for Stage 1 filter."""
    return [
        float(is_bad_answer(answer)),       # is_bad
        float(is_unknown_answer(answer)),   # is_unknown
        float(len(answer.split())),         # answer_len_words
        math.log1p(len(answer)),            # answer_len_chars (log-scaled)
    ]

STAGE1_FEATURE_NAMES = ["is_bad", "is_unknown", "answer_len_words", "answer_len_chars"]


# ─────────────────────────── main ────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Phase B1.1 — Stage 1 garbage filter"
    )
    ap.add_argument("--hop_scores",  required=True,
                    help="exp0c/preds/dev_hop_scores.jsonl")
    ap.add_argument("--gold",        required=True,
                    help="data/hotpotqa/raw/hotpot_dev_distractor_v1.json")
    ap.add_argument("--out_jsonl",   required=True,
                    help="Output: per-question filter decisions")
    ap.add_argument("--out_json",    required=True,
                    help="Output: summary + validation stats")
    ap.add_argument("--threshold",   type=float, default=0.5,
                    help="Filter threshold (default 0.5 — conservative)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)),  exist_ok=True)

    # ── Load gold ────────────────────────────────────────────────────
    print("[B1.1] Loading gold answers...")
    gold_map: Dict[str, str] = {}
    for ex in json.load(open(args.gold)):
        gold_map[str(ex["_id"])] = ex["answer"]
    print(f"       {len(gold_map)} questions")

    # ── Load hop scores ───────────────────────────────────────────────
    print("[B1.1] Loading hop scores...")
    hop_records: Dict[str, Dict] = {}
    for rec in iter_jsonl(args.hop_scores):
        hop_records[str(rec["qid"])] = rec
    print(f"       {len(hop_records)} questions")

    # ── Build training data for Stage 1 ──────────────────────────────
    # Label: 1 = survives (not garbage), 0 = garbage
    # Garbage = is_bad OR is_unknown (hard rule from existing is_bad fn)
    print("[B1.1] Building Stage 1 training data...")
    X_all:    List[List[float]] = []
    y_all:    List[int]         = []
    qid_rows: Dict[str, List[int]] = collections.defaultdict(list)  # qid → row indices

    for qid in sorted(hop_records.keys()):
        rec   = hop_records[qid]
        cands = rec["candidates"]
        for ci, cd in enumerate(cands):
            answer  = cd["answer"]
            is_garb = int(is_bad_answer(answer) or is_unknown_answer(answer))
            label   = 1 - is_garb  # 1 = survives, 0 = garbage
            feats   = stage1_features(answer)
            row_idx = len(X_all)
            X_all.append(feats)
            y_all.append(label)
            qid_rows[qid].append(row_idx)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)

    print(f"       {len(X)} candidate rows")
    print(f"       Garbage rate: {(1-y).mean():.1%}  "
          f"Survive rate: {y.mean():.1%}")

    # ── Train logistic regression ─────────────────────────────────────
    print("[B1.1] Training logistic regression filter...")
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight="balanced",  # handle imbalance
    )
    clf.fit(X, y)

    probs    = clf.predict_proba(X)[:, 1]  # P(survives)
    preds_05 = (probs >= args.threshold).astype(int)

    train_acc = (preds_05 == y.astype(int)).mean()
    print(f"       Training accuracy @ τ={args.threshold}: {train_acc:.4f}")

    # Feature coefficients
    coef_dict = {
        name: round(float(c), 4)
        for name, c in zip(STAGE1_FEATURE_NAMES, clf.coef_[0])
    }
    print(f"       Coefficients: {coef_dict}")

    # ── Apply filter and validate ─────────────────────────────────────
    print("[B1.1] Applying filter and validating oracle preservation...")

    n_total          = 0
    n_surviving_total = 0
    oracle_before    = 0
    oracle_after     = 0
    false_negatives  = 0  # correct answers removed by filter
    all_filtered     = 0  # questions where ALL candidates got filtered

    if os.path.exists(args.out_jsonl):
        os.remove(args.out_jsonl)

    for qid in sorted(hop_records.keys()):
        rec   = hop_records[qid]
        cands = rec["candidates"]
        gold  = gold_map.get(qid, extract_gold(rec.get("gold", "")))
        rows  = qid_rows[qid]

        # Oracle before filter
        oracle_b = int(any(em(cd["answer"], gold) for cd in cands))

        # Surviving candidates
        surviving_ids = []
        filtered_ids  = []
        for ci, (cd, row_idx) in enumerate(zip(cands, rows)):
            if probs[row_idx] >= args.threshold:
                surviving_ids.append(ci)
            else:
                filtered_ids.append(ci)
                if em(cd["answer"], gold):
                    false_negatives += 1

        # Oracle after filter
        oracle_a = int(any(
            em(cands[ci]["answer"], gold) for ci in surviving_ids
        ))

        if not surviving_ids:
            all_filtered += 1

        n_total           += len(cands)
        n_surviving_total += len(surviving_ids)
        oracle_before     += oracle_b
        oracle_after      += oracle_a

        append_jsonl(args.out_jsonl, {
            "qid":           qid,
            "gold":          gold,
            "oracle_before": oracle_b,
            "oracle_after":  oracle_a,
            "n_before":      len(cands),
            "n_after":       len(surviving_ids),
            "surviving_ids": surviving_ids,
            "filtered_ids":  filtered_ids,
        })

    # ── Summary ───────────────────────────────────────────────────────
    n_questions      = len(hop_records)
    filter_rate      = 1 - (n_surviving_total / max(n_total, 1))
    fn_rate          = false_negatives / max(n_total, 1)
    oracle_em_before = oracle_before / n_questions
    oracle_em_after  = oracle_after  / n_questions
    oracle_delta     = oracle_em_after - oracle_em_before

    print(f"\n[B1.1] Validation results:")
    print(f"       Questions:              {n_questions:,}")
    print(f"       Candidates before:      {n_total:,}")
    print(f"       Candidates after:       {n_surviving_total:,}")
    print(f"       Filter rate:            {filter_rate:.1%}")
    print(f"       All-filtered questions: {all_filtered}")
    print(f"       False negatives:        {false_negatives}  "
          f"(correct answers removed)")
    print(f"       FN rate:                {fn_rate:.4%}")
    print(f"       Oracle EM before:       {oracle_em_before:.4f}")
    print(f"       Oracle EM after:        {oracle_em_after:.4f}")
    print(f"       Oracle delta:           {oracle_delta:+.4f}")

    # Safety check
    if false_negatives > 0:
        print(f"\n  WARNING: {false_negatives} correct answers removed.")
        print(f"  Consider lowering threshold below {args.threshold}.")
    else:
        print(f"\n  SAFETY CHECK PASSED: 0 correct answers removed.")

    summary = {
        "script":             "exp_b1_stage1_filter.py",
        "hop_scores":         args.hop_scores,
        "threshold":          args.threshold,
        "n_questions":        n_questions,
        "n_candidates_before": n_total,
        "n_candidates_after": n_surviving_total,
        "filter_rate":        round(filter_rate, 4),
        "all_filtered_questions": all_filtered,
        "false_negatives":    false_negatives,
        "fn_rate":            round(fn_rate, 6),
        "oracle_em_before":   round(oracle_em_before, 4),
        "oracle_em_after":    round(oracle_em_after,  4),
        "oracle_delta":       round(oracle_delta, 4),
        "lr_coefficients":    coef_dict,
        "garbage_rate":       round(float((1-y).mean()), 4),
        "training_accuracy":  round(float(train_acc), 4),
        "timestamp":          __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[B1.1] Summary written: {args.out_json}")
    print(f"[B1.1] Filter output:   {args.out_jsonl}")


if __name__ == "__main__":
    main()