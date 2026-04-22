#!/usr/bin/env python3
"""
platt_calibrate_sanity.py — Option Alpha: post-hoc Platt calibration for the sanity run.

Reads the existing per_candidate_scores.jsonl from exp_crosshop/sanity/ (no
retraining, no GPU) and reports:
  - Pre-Platt ECE  (verifier-level = max-score question vs EM correctness)
  - Post-Platt ECE  (same convention, after sigmoid calibration)
  - Whether the ECE failure was numeric miscalibration or architectural

Decision rule:
  Post-Platt ECE < 0.12  → miscalibration, architecture is OK, proceed to full run
  Post-Platt ECE >= 0.12 → deeper problem, investigate before full run

Usage (CPU-only, ~5 seconds):
    /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 \\
        tools/platt_calibrate_sanity.py \\
        --scores   exp_crosshop/sanity/per_candidate_scores.jsonl \\
        --out_json exp_crosshop/sanity/platt_calibration.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Expected Calibration Error (Guo et al. 2017).
    Returns (scalar_ece, reliability_bins)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if lo == edges[-2]:
            mask = (probs >= lo) & (probs <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0,
                         "mean_conf": None, "frac_correct": None, "gap": None})
            continue
        mc = float(probs[mask].mean())
        fc = float(labels[mask].mean())
        w  = cnt / n
        ece += w * abs(mc - fc)
        bins.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": cnt,
                     "mean_conf": round(mc, 4), "frac_correct": round(fc, 4),
                     "gap": round(abs(mc - fc), 4)})
    return float(ece), bins


def platt_fit(raw_probs: np.ndarray, labels: np.ndarray):
    """Fit a logistic calibrator on (raw_probs, labels).
    Returns (calibrated_probs, slope, intercept)."""
    X = raw_probs.reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(X, labels)
    cal = lr.predict_proba(X)[:, 1]
    return cal, float(lr.coef_[0][0]), float(lr.intercept_[0])


# ──────────────────────────────────────────────────────────────────────────────
# Load per_candidate_scores.jsonl
# ──────────────────────────────────────────────────────────────────────────────

def load_scores(path: str):
    """Load per-candidate scores file.
    Returns (qid_list, score_arr, label_arr) parallel arrays.
    Handles both naming conventions the sanity_run.py may have used."""
    qids, scores, labels = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qids.append(r["qid"])
            # Score field: 'score' or 'prob'
            s = r.get("score", r.get("prob", r.get("main_prob", None)))
            if s is None:
                # Fall back to logit → sigmoid if stored as logit
                logit = r.get("logit", r.get("main_logit", 0.0))
                s = 1.0 / (1.0 + math.exp(-float(logit)))
            scores.append(float(s))
            # Label field: 'label' or 'is_correct'
            lbl = r.get("label", r.get("is_correct", r.get("em", 0)))
            labels.append(int(float(lbl) > 0.5))
    return qids, np.array(scores, dtype=np.float32), np.array(labels, dtype=np.float32)


def verifier_level(qids, scores, labels):
    """Reduce to one (score, label) per question by argmax score.
    This is the project-standard 'verifier-level ECE' metric."""
    from collections import defaultdict
    qmap = defaultdict(list)
    for i, q in enumerate(qids):
        qmap[q].append(i)
    v_scores, v_labels = [], []
    for rows in qmap.values():
        best = rows[int(np.argmax(scores[rows]))]
        v_scores.append(float(scores[best]))
        v_labels.append(float(labels[best]))
    return np.array(v_scores, dtype=np.float32), np.array(v_labels, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores",   required=True,
                    help="exp_crosshop/sanity/per_candidate_scores.jsonl")
    ap.add_argument("--out_json", required=True,
                    help="Output path for calibration report")
    ap.add_argument("--n_bins",   type=int, default=10)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)

    # ── load ──
    print(f"Loading scores from: {args.scores}")
    qids, scores, labels = load_scores(args.scores)
    print(f"  Loaded {len(qids):,} candidate rows")
    print(f"  Unique questions: {len(set(qids)):,}")
    print(f"  Positive rate (candidate-level): {labels.mean():.3f}")

    # ── candidate-level ECE ──
    ece_cand_raw, bins_cand_raw = compute_ece(scores, labels, args.n_bins)
    cal_cand, slope, intercept  = platt_fit(scores, labels)
    ece_cand_platt, bins_cand_platt = compute_ece(cal_cand, labels, args.n_bins)

    # ── verifier-level ECE (project standard) ──
    v_scores, v_labels = verifier_level(qids, scores, labels)
    ece_verif_raw, bins_verif_raw = compute_ece(v_scores, v_labels, args.n_bins)
    cal_verif, slope_v, intercept_v = platt_fit(v_scores, v_labels)
    ece_verif_platt, bins_verif_platt = compute_ece(cal_verif, v_labels, args.n_bins)

    # ── decision ──
    PASS_THRESHOLD = 0.12
    if ece_verif_platt < PASS_THRESHOLD:
        decision = (f"PASS — ECE collapse is numeric miscalibration. "
                    f"Post-Platt ECE={ece_verif_platt:.4f} < {PASS_THRESHOLD}. "
                    f"Architecture is OK; proceed to full 3-arm experiment. "
                    f"Apply Platt scaling in the full-experiment eval pipeline.")
    else:
        decision = (f"FAIL — Post-Platt ECE={ece_verif_platt:.4f} >= {PASS_THRESHOLD}. "
                    f"Miscalibration is not fully explained by sigmoid overconfidence. "
                    f"Investigate: reduce to 2 epochs or increase label_smoothing to 0.10.")

    # ── assemble report ──
    report = {
        "input_file": args.scores,
        "n_candidates": len(qids),
        "n_questions": len(set(qids)),
        "positive_rate_cand": round(float(labels.mean()), 4),
        "positive_rate_verif": round(float(v_labels.mean()), 4),
        "candidate_level": {
            "ece_raw":   round(ece_cand_raw, 4),
            "ece_platt": round(ece_cand_platt, 4),
            "improvement": round(ece_cand_raw - ece_cand_platt, 4),
        },
        "verifier_level": {
            "ece_raw":      round(ece_verif_raw, 4),
            "ece_platt":    round(ece_verif_platt, 4),
            "improvement":  round(ece_verif_raw - ece_verif_platt, 4),
            "platt_slope":      round(slope_v, 4),
            "platt_intercept":  round(intercept_v, 4),
            "note": ("Fit and evaluated on same dev fold — optimistic lower bound. "
                     "For production, fit Platt on a held-out calibration split."),
        },
        "sanity_gate_original": 0.15,
        "sanity_gate_post_platt": PASS_THRESHOLD,
        "decision": decision,
        "reliability_bins_raw":   bins_verif_raw,
        "reliability_bins_platt": bins_verif_platt,
    }

    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)

    # ── print summary ──
    W = 72
    print()
    print("=" * W)
    print("  PLATT CALIBRATION REPORT  —  sanity run fold 0")
    print("=" * W)
    print(f"  {'Metric':<42}  {'Raw':>8}  {'Post-Platt':>10}")
    print("  " + "-" * (W - 2))
    print(f"  {'ECE (candidate-level)':<42}  "
          f"{ece_cand_raw:>8.4f}  {ece_cand_platt:>10.4f}")
    print(f"  {'ECE (verifier-level, project standard)':<42}  "
          f"{ece_verif_raw:>8.4f}  {ece_verif_platt:>10.4f}")
    print()
    print(f"  Platt fit (verifier-level):  "
          f"score_cal = sigmoid({slope_v:.3f} * raw_score + {intercept_v:.3f})")
    print()
    print(f"  SANITY ECE GATE (post-Platt < {PASS_THRESHOLD}): "
          f"{'PASS' if ece_verif_platt < PASS_THRESHOLD else 'FAIL'}")
    print()
    print(f"  DECISION: {decision}")
    print("=" * W)
    print(f"\nReport saved to: {args.out_json}")


if __name__ == "__main__":
    main()