#!/usr/bin/env python3
"""
exp2_q8_calibration.py — Q8: Is calibration preserved after adding chain-aware features?

Baseline: surface XGBoost (exp1b) ECE = 0.0161  (well calibrated)
Question: does the chain-aware verifier maintain this, or does ECE degrade?

Two ECE measures (both computed for completeness):
  verifier-level ECE  — max(probs) per question vs EM correctness
                        (matches exp1_xgb_verifier.py methodology exactly)
  classifier-level ECE — all 5 (prob, label) pairs per question
                         (measures per-candidate calibration)

If verifier-level ECE > 0.05 (3x the baseline), apply Platt scaling
(sigmoid/logistic calibration) as the post-hoc fix specified in the pipeline doc.

Outputs:
  exp1b/metrics/q8_calibration.json  — ECE scores, reliability diagram data,
                                        pre/post Platt comparison (git-tracked)
  exp1b/logs/q8_calibration.log

Usage (from project root, no GPU):
    /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 \\
        tools/exp2_q8_calibration.py \\
        --verifier_preds  exp1b/preds/dev_chain_verifier_min_preds.jsonl \\
        --gold            data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
        --out_json        exp1b/metrics/q8_calibration.json \\
        --log             exp1b/logs/q8_calibration.log
"""

import argparse
import json
import logging
import os
import re
import string
import sys

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


# ─────────────────────────── ECE ─────────────────────────────────────

def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """
    Expected Calibration Error (Guo et al. 2017).
    Returns (ece_scalar, reliability_diagram_bins).
    Each bin dict: {bin_lo, bin_hi, mean_conf, frac_correct, n, weight}
    """
    bins_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(probs)
    diagram = []

    for lo, hi in zip(bins_edges[:-1], bins_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if lo == bins_edges[-2]:          # include right edge in last bin
            mask = (probs >= lo) & (probs <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            diagram.append({
                "bin_lo": round(float(lo), 2),
                "bin_hi": round(float(hi), 2),
                "mean_conf": None, "frac_correct": None,
                "n": 0, "weight": 0.0, "gap": None,
            })
            continue
        mean_conf    = float(probs[mask].mean())
        frac_correct = float(labels[mask].mean())
        weight       = cnt / n
        gap          = abs(mean_conf - frac_correct)
        ece         += weight * gap
        diagram.append({
            "bin_lo":       round(float(lo), 2),
            "bin_hi":       round(float(hi), 2),
            "mean_conf":    round(mean_conf,    4),
            "frac_correct": round(frac_correct, 4),
            "n":            cnt,
            "weight":       round(weight, 4),
            "gap":          round(gap,    4),
        })

    return float(ece), diagram


# ─────────────────────────── Platt scaling ───────────────────────────

def platt_scale(
    raw_probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Fit a logistic (Platt) calibrator on (raw_probs, labels).
    Returns (calibrated_probs, intercept, coef).

    Note: fitting and evaluating on the same dev set is optimistic.
    Since we have no separate calibration split, this gives a lower
    bound on calibrated ECE — report as 'best-case Platt'.
    """
    X = raw_probs.reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(X, labels)
    cal_probs = lr.predict_proba(X)[:, 1]
    return cal_probs, float(lr.intercept_[0]), float(lr.coef_[0][0])


# ─────────────────────────── main ────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier_preds", required=True,
                    help="exp1b/preds/dev_chain_verifier_min_preds.jsonl")
    ap.add_argument("--gold",           required=True,
                    help="data/hotpotqa/raw/hotpot_dev_distractor_v1.json")
    ap.add_argument("--out_json",       required=True)
    ap.add_argument("--log",            required=True)
    ap.add_argument("--n_bins",         type=int, default=10)
    ap.add_argument("--platt_threshold", type=float, default=0.05,
                    help="Apply Platt scaling if verifier-level ECE exceeds this")
    args = ap.parse_args()

    for p in [args.out_json, args.log]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # ── logging ──
    log = logging.getLogger("q8")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(args.log, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)

    log.info("=== Q8 Calibration Check ===")
    log.info(f"verifier_preds   : {args.verifier_preds}")
    log.info(f"platt_threshold  : {args.platt_threshold}")
    log.info(f"baseline ECE     : 0.0161  (surface XGBoost exp1b)")

    # ── load gold ──
    log.info("Loading gold answers ...")
    gold_map: dict[str, str] = {
        str(ex["_id"]): ex["answer"]
        for ex in json.load(open(args.gold))
    }

    # ── load verifier predictions ──
    log.info("Loading verifier predictions ...")
    preds_map: dict[str, dict] = {}
    for line in open(args.verifier_preds):
        r = json.loads(line)
        preds_map[str(r["qid"])] = r
    log.info(f"  {len(preds_map)} records")

    all_qids = sorted(set(preds_map) & set(gold_map))
    log.info(f"  Aligned: {len(all_qids)}")

    # ── build arrays ──
    # Verifier-level: one (max_prob, correctness) per question
    verif_confs:  list[float] = []
    verif_labels: list[int]   = []

    # Classifier-level: one (prob, label) per (question, candidate)
    clf_probs:  list[float] = []
    clf_labels: list[int]   = []

    for qid in all_qids:
        rec   = preds_map[qid]
        probs = rec.get("probs", [])
        pred  = rec.get("pred", "")
        gold  = gold_map[qid]

        max_conf   = max(probs) if probs else 0.0
        is_correct = em(pred, gold)

        verif_confs.append(max_conf)
        verif_labels.append(is_correct)

        # all 5 per-candidate probs with binary correct label
        # (a candidate is "correct" if it matches gold exactly)
        # We need candidate answers for this — reconstruct from probs only.
        # Since we only have pred + probs (not all 5 answers), use:
        #   best_idx candidate → label = is_correct,
        #   others → label = 0
        best_idx = int(np.argmax(probs)) if probs else 0
        for ci, p in enumerate(probs):
            # We only know the picked answer; approximate other labels as 0.
            # This is conservative — the classifier-level ECE may be slightly
            # optimistic for picked candidates and pessimistic for others.
            clf_probs.append(p)
            clf_labels.append(is_correct if ci == best_idx else 0)

    verif_conf_arr  = np.array(verif_confs,  dtype=np.float64)
    verif_label_arr = np.array(verif_labels, dtype=np.float64)
    clf_prob_arr    = np.array(clf_probs,    dtype=np.float64)
    clf_label_arr   = np.array(clf_labels,   dtype=np.float64)

    # ── compute pre-Platt ECE ──
    log.info("Computing ECE (pre-Platt) ...")
    ece_verif,  diagram_verif  = compute_ece(verif_conf_arr,  verif_label_arr, args.n_bins)
    ece_clf,    diagram_clf    = compute_ece(clf_prob_arr,    clf_label_arr,    args.n_bins)

    BASELINE_ECE = 0.0161
    delta_vs_baseline = ece_verif - BASELINE_ECE

    # ── Platt scaling ──
    platt_applied = ece_verif > args.platt_threshold
    platt_result  = None

    if platt_applied:
        log.info(f"ECE={ece_verif:.4f} > threshold={args.platt_threshold} "
                 f"— applying Platt scaling ...")
        cal_probs, intercept, coef = platt_scale(verif_conf_arr, verif_label_arr)
        ece_platt, diagram_platt   = compute_ece(cal_probs, verif_label_arr, args.n_bins)

        platt_result = {
            "ece_after_platt":      round(ece_platt,   4),
            "ece_improvement":      round(ece_verif - ece_platt, 4),
            "platt_intercept":      round(intercept,   6),
            "platt_coef":           round(coef,        6),
            "reliability_diagram":  diagram_platt,
            "note": (
                "Platt fit on same dev set — optimistic lower bound. "
                "For production use, fit on a held-out calibration split."
            ),
        }
        log.info(f"  ECE after Platt : {ece_platt:.4f}  "
                 f"(improvement: {ece_verif - ece_platt:.4f})")
    else:
        log.info(f"ECE={ece_verif:.4f} ≤ threshold={args.platt_threshold} "
                 f"— Platt scaling not needed")

    # ── calibration decision ──
    if ece_verif <= 0.02:
        q8_decision = f"PRESERVED: ECE={ece_verif:.4f} ≤ 0.02 (baseline={BASELINE_ECE})"
    elif ece_verif <= 0.05:
        q8_decision = (f"MODERATE DEGRADATION: ECE={ece_verif:.4f} "
                       f"(+{delta_vs_baseline:.4f} vs baseline={BASELINE_ECE}) — "
                       f"acceptable without calibration")
    else:
        q8_decision = (f"SIGNIFICANT DEGRADATION: ECE={ece_verif:.4f} "
                       f"(+{delta_vs_baseline:.4f} vs baseline={BASELINE_ECE}) — "
                       f"Platt scaling {'applied' if platt_applied else 'recommended'}")

    # ── assemble summary ──
    summary = {
        "baseline": {
            "method":  "surface XGBoost (exp1b)",
            "ece":     BASELINE_ECE,
        },
        "chain_aware_verifier": {
            "method":  "chain-aware XGBoost [min pooling]",
            "verifier_level_ece":    round(ece_verif, 4),
            "classifier_level_ece":  round(ece_clf,   4),
            "delta_vs_baseline":     round(delta_vs_baseline, 4),
            "reliability_diagram":   diagram_verif,
        },
        "platt_scaling": platt_result,
        "q8_decision":   q8_decision,
        "n_questions":   len(all_qids),
        "positive_rate": round(float(verif_label_arr.mean()), 4),
        "mean_confidence": round(float(verif_conf_arr.mean()), 4),
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to {args.out_json}")

    # ── print results ──
    W = 68
    log.info("=" * W)
    log.info("  Q8 CALIBRATION RESULTS")
    log.info("=" * W)
    log.info(f"  {'Method':<38}  {'ECE':>8}  {'Delta':>8}")
    log.info("  " + "-" * (W - 2))
    log.info(f"  {'Surface XGBoost (exp1b baseline)':<38}  "
             f"{BASELINE_ECE:>8.4f}  {'—':>8}")
    log.info(f"  {'Chain-aware XGB [min] (verifier-level)':<38}  "
             f"{ece_verif:>8.4f}  {delta_vs_baseline:>+8.4f}")
    log.info(f"  {'Chain-aware XGB [min] (classifier-level)':<38}  "
             f"{ece_clf:>8.4f}  {'':>8}")
    if platt_result:
        log.info(f"  {'Chain-aware + Platt (verifier-level)':<38}  "
                 f"{platt_result['ece_after_platt']:>8.4f}  "
                 f"{platt_result['ece_after_platt'] - BASELINE_ECE:>+8.4f}")
    log.info("=" * W)
    log.info(f"  DECISION: {q8_decision}")
    log.info("=" * W)

    # ── reliability diagram ──
    log.info("\n  Reliability diagram (chain-aware, verifier-level):")
    log.info(f"  {'Bin':>12}  {'Mean conf':>10}  {'Frac correct':>13}  "
             f"{'Gap':>7}  {'N':>6}")
    log.info("  " + "-" * 54)
    for b in diagram_verif:
        if b["n"] == 0:
            continue
        over_under = "over" if b["mean_conf"] > b["frac_correct"] else "under"
        log.info(
            f"  [{b['bin_lo']:.1f}, {b['bin_hi']:.1f})  "
            f"{b['mean_conf']:>10.4f}  {b['frac_correct']:>13.4f}  "
            f"{b['gap']:>7.4f}  {b['n']:>6}  ({over_under})"
        )


if __name__ == "__main__":
    main()