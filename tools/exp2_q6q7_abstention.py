#!/usr/bin/env python3
"""
exp2_q6q7_abstention.py — Q6 + Q7: Abstention curve analysis

Q6: Can the verifier reliably detect when none of the 5 candidates is correct?
    Signal: max(probs) across 5 candidates per question.
    Test: do "has-correct" vs "all-wrong" questions have separable distributions?
    Metric: AUROC treating "has-correct" as positive class.

Q7: What is the accuracy/coverage tradeoff curve?
    Sort questions by max(probs) descending. Compute EM on the top-X%.
    Plot how accuracy improves as we abstain on more questions.

Inputs (no inference — pure analysis on existing outputs):
  dev_chain_verifier_min_preds.jsonl  — Q2 verifier: {qid, pred, probs, best_idx}
  oracle_M5_dev_perqid.jsonl          — {qid, best_em, per_candidate, ...}
  hotpot_dev_distractor_v1.json       — gold answers + question type

Outputs:
  exp1b/metrics/q6q7_abstention.json  — full results (tracked by git)
  exp1b/logs/q6q7_abstention.log

Usage (from project root, no GPU needed):
    /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 \\
        tools/exp2_q6q7_abstention.py \\
        --verifier_preds  exp1b/preds/dev_chain_verifier_min_preds.jsonl \\
        --oracle_perqid   exp1b/metrics/oracle_M5_dev_perqid.jsonl \\
        --gold            data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
        --out_json        exp1b/metrics/q6q7_abstention.json \\
        --log             exp1b/logs/q6q7_abstention.log
"""

import argparse
import collections
import json
import logging
import os
import re
import string
import sys

import numpy as np


# ─────────────────────────── text utils ─────────────────────────────

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


# ─────────────────────────── AUROC ──────────────────────────────────

def compute_auroc(scores: list[float], labels: list[int]) -> float:
    """
    AUROC via trapezoidal rule.
    labels: 1 = positive (has-correct), 0 = negative (all-wrong)
    scores: max(probs) — higher = more likely to be positive
    """
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = fp = 0
    auc = 0.0
    prev_fp = 0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
            # Each time FP increases, add a trapezoid strip
            auc += tp / n_pos * (1 / n_neg)
    return float(auc)


# ─────────────────────────── threshold sweep ─────────────────────────

def threshold_sweep(
    max_confs: list[float],
    em_correct: list[int],
    oracle_flags: list[int],
    thresholds: list[float],
) -> list[dict]:
    """
    For each threshold τ:
      - Questions with max_conf >= τ are answered
      - Questions with max_conf <  τ are abstained

    Returns per-threshold stats.
    """
    results = []
    n_total = len(max_confs)
    for tau in thresholds:
        answered_mask = [c >= tau for c in max_confs]
        n_answered    = sum(answered_mask)
        n_abstained   = n_total - n_answered

        if n_answered == 0:
            results.append({
                "tau":             round(tau, 2),
                "n_answered":      0,
                "n_abstained":     n_total,
                "coverage":        0.0,
                "abstain_rate":    1.0,
                "em_on_answered":  None,
                # How many abstained questions actually had a correct candidate?
                "abstain_has_correct_rate": None,
            })
            continue

        em_ans      = [em_correct[i] for i, a in enumerate(answered_mask) if a]
        oracle_abs  = [oracle_flags[i] for i, a in enumerate(answered_mask) if not a]

        em_on_answered = sum(em_ans) / n_answered
        abstain_has_correct = (
            sum(oracle_abs) / n_abstained if n_abstained > 0 else None
        )

        results.append({
            "tau":                    round(tau, 2),
            "n_answered":             n_answered,
            "n_abstained":            n_abstained,
            "coverage":               round(n_answered / n_total, 4),
            "abstain_rate":           round(n_abstained / n_total, 4),
            "em_on_answered":         round(em_on_answered, 4),
            # Fraction of abstained questions that actually had a correct candidate
            # (we want this LOW — abstaining on winnable questions is waste)
            "abstain_has_correct_rate": (
                round(abstain_has_correct, 4)
                if abstain_has_correct is not None else None
            ),
        })
    return results


# ─────────────────────────── coverage curve (Q7) ─────────────────────

def coverage_curve(
    max_confs: list[float],
    em_correct: list[int],
    n_steps: int = 20,
) -> list[dict]:
    """
    Sort questions by max_conf descending.
    For each coverage level X% (top X% by confidence), compute EM.
    Returns list of {coverage, em, n} dicts.
    """
    order = sorted(range(len(max_confs)), key=lambda i: -max_confs[i])
    n_total = len(order)
    curve = []

    # Always include 100% coverage as anchor
    coverages = [i / n_steps for i in range(1, n_steps + 1)]
    if 1.0 not in coverages:
        coverages.append(1.0)

    for cov in sorted(coverages):
        k = max(1, int(round(cov * n_total)))
        top_k = order[:k]
        em_k  = sum(em_correct[i] for i in top_k) / k
        curve.append({
            "coverage":  round(k / n_total, 4),
            "n":         k,
            "em":        round(em_k, 4),
        })
    return curve


# ─────────────────────────── distribution stats ───────────────────────

def dist_stats(values: list[float]) -> dict:
    a = np.array(values)
    return {
        "n":      len(a),
        "mean":   round(float(a.mean()), 4),
        "std":    round(float(a.std()),  4),
        "p25":    round(float(np.percentile(a, 25)), 4),
        "median": round(float(np.median(a)), 4),
        "p75":    round(float(np.percentile(a, 75)), 4),
        "p90":    round(float(np.percentile(a, 90)), 4),
    }


# ─────────────────────────── main ────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier_preds", required=True,
                    help="exp1b/preds/dev_chain_verifier_min_preds.jsonl")
    ap.add_argument("--oracle_perqid",  required=True,
                    help="exp1b/metrics/oracle_M5_dev_perqid.jsonl")
    ap.add_argument("--gold",           required=True,
                    help="data/hotpotqa/raw/hotpot_dev_distractor_v1.json")
    ap.add_argument("--out_json",       required=True)
    ap.add_argument("--log",            required=True)
    args = ap.parse_args()

    for p in [args.out_json, args.log]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # ── logging ──
    log = logging.getLogger("q6q7")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(args.log, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)

    log.info("=== Q6/Q7 Abstention Analysis ===")

    # ── load gold answers + question types ──
    log.info("Loading gold ...")
    gold_map:  dict[str, str] = {}
    qtype_map: dict[str, str] = {}
    for ex in json.load(open(args.gold)):
        qid = str(ex["_id"])
        gold_map[qid]  = ex["answer"]
        qtype_map[qid] = ex.get("type", "bridge")

    # ── load oracle per-question flags ──
    log.info("Loading oracle flags ...")
    # best_em=1 means at least one of the 5 candidates matched gold
    oracle_map: dict[str, int] = {}
    for line in open(args.oracle_perqid):
        r = json.loads(line)
        oracle_map[str(r["qid"])] = int(r["best_em"])
    log.info(f"  {len(oracle_map)} oracle records")
    n_has_correct = sum(oracle_map.values())
    n_all_wrong   = len(oracle_map) - n_has_correct
    log.info(f"  has-correct (oracle_em=1): {n_has_correct} "
             f"({100*n_has_correct/len(oracle_map):.1f}%)")
    log.info(f"  all-wrong   (oracle_em=0): {n_all_wrong} "
             f"({100*n_all_wrong/len(oracle_map):.1f}%)")

    # ── load verifier predictions ──
    log.info("Loading verifier predictions ...")
    preds_map: dict[str, dict] = {}
    for line in open(args.verifier_preds):
        r = json.loads(line)
        preds_map[str(r["qid"])] = r
    log.info(f"  {len(preds_map)} verifier records")

    # ── align all three sources ──
    all_qids = sorted(set(preds_map) & set(oracle_map) & set(gold_map))
    log.info(f"  Aligned qids: {len(all_qids)}")

    # Build parallel arrays
    max_confs:   list[float] = []
    em_correct:  list[int]   = []   # did verifier pick correctly?
    oracle_flags: list[int]  = []   # did any candidate exist?
    qtype_list:  list[str]   = []

    for qid in all_qids:
        pred_rec = preds_map[qid]
        probs    = pred_rec.get("probs", [])
        pred_ans = pred_rec.get("pred", "")
        gold_ans = gold_map[qid]

        max_conf = max(probs) if probs else 0.0
        max_confs.append(max_conf)
        em_correct.append(em(pred_ans, gold_ans))
        oracle_flags.append(oracle_map[qid])
        qtype_list.append(qtype_map.get(qid, "bridge"))

    # ── Q6: distribution separation ──
    log.info("Computing Q6 — distribution separation ...")

    has_correct_confs = [max_confs[i] for i, o in enumerate(oracle_flags) if o == 1]
    all_wrong_confs   = [max_confs[i] for i, o in enumerate(oracle_flags) if o == 0]

    auroc = compute_auroc(max_confs, oracle_flags)

    # Distribution stats per group
    stats_has_correct = dist_stats(has_correct_confs)
    stats_all_wrong   = dist_stats(all_wrong_confs)

    # Separation ratio: mean(has_correct) / mean(all_wrong)
    sep_ratio = (stats_has_correct["mean"] / stats_all_wrong["mean"]
                 if stats_all_wrong["mean"] > 0 else float("inf"))

    # Q6 decision
    if auroc >= 0.70:
        q6_decision = f"VIABLE: AUROC={auroc:.4f} ≥ 0.70 — abstention is reliable"
    elif auroc >= 0.60:
        q6_decision = f"PARTIAL: AUROC={auroc:.4f} in [0.60, 0.70) — modest abstention value"
    else:
        q6_decision = f"NOT VIABLE: AUROC={auroc:.4f} < 0.60 — distributions overlap too much"

    # ── Q7: threshold sweep ──
    log.info("Computing Q7 — threshold sweep ...")
    thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05).tolist()]
    sweep = threshold_sweep(max_confs, em_correct, oracle_flags, thresholds)

    # ── Q7: coverage curve ──
    log.info("Computing Q7 — coverage curve ...")
    curve = coverage_curve(max_confs, em_correct, n_steps=20)

    # Full-coverage EM (= overall verifier EM, our baseline)
    em_full = sum(em_correct) / len(em_correct)

    # Find the point where abstaining 30% gives best EM gain
    curve_70 = next((p for p in curve if p["coverage"] <= 0.71), None)

    # Find optimal tau by best EM on answered (min 50% coverage)
    valid_sweep = [s for s in sweep
                   if s["coverage"] is not None and s["coverage"] >= 0.50
                   and s["em_on_answered"] is not None]
    best_tau_entry = max(valid_sweep, key=lambda s: s["em_on_answered"],
                         default=None)

    # ── per question-type abstention analysis ──
    log.info("Computing per-type abstention stats ...")
    bridge_confs     = [max_confs[i] for i, t in enumerate(qtype_list) if t == "bridge"]
    comparison_confs = [max_confs[i] for i, t in enumerate(qtype_list) if t == "comparison"]
    bridge_oracle    = [oracle_flags[i] for i, t in enumerate(qtype_list) if t == "bridge"]
    comparison_oracle= [oracle_flags[i] for i, t in enumerate(qtype_list) if t == "comparison"]

    auroc_bridge     = compute_auroc(bridge_confs, bridge_oracle)
    auroc_comparison = compute_auroc(comparison_confs, comparison_oracle)

    # ── assemble summary ──
    summary = {
        # ── Q6 ──
        "q6_separation": {
            "auroc":         round(auroc, 4),
            "auroc_bridge":  round(auroc_bridge, 4),
            "auroc_comparison": round(auroc_comparison, 4),
            "separation_ratio_means": round(sep_ratio, 4),
            "has_correct_conf_stats": stats_has_correct,
            "all_wrong_conf_stats":   stats_all_wrong,
            "n_has_correct":          len(has_correct_confs),
            "n_all_wrong":            len(all_wrong_confs),
            "decision":               q6_decision,
        },
        # ── Q7 ──
        "q7_coverage_curve": {
            "em_at_100pct_coverage": round(em_full, 4),
            "em_at_70pct_coverage":  curve_70["em"] if curve_70 else None,
            "gain_30pct_abstain":    (
                round(curve_70["em"] - em_full, 4)
                if curve_70 else None
            ),
            "optimal_tau":           (
                best_tau_entry["tau"] if best_tau_entry else None
            ),
            "em_at_optimal_tau":     (
                best_tau_entry["em_on_answered"] if best_tau_entry else None
            ),
            "coverage_at_optimal_tau": (
                best_tau_entry["coverage"] if best_tau_entry else None
            ),
            "curve": curve,
        },
        "threshold_sweep": sweep,
        "verifier_preds_source": args.verifier_preds,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to {args.out_json}")

    # ── print results ──
    W = 68
    log.info("=" * W)
    log.info("  Q6 — Distribution Separation")
    log.info("=" * W)
    log.info(f"  AUROC (overall)    : {auroc:.4f}")
    log.info(f"  AUROC (bridge)     : {auroc_bridge:.4f}")
    log.info(f"  AUROC (comparison) : {auroc_comparison:.4f}")
    log.info(f"  Mean conf (has-correct) : {stats_has_correct['mean']:.4f}  "
             f"± {stats_has_correct['std']:.4f}")
    log.info(f"  Mean conf (all-wrong)   : {stats_all_wrong['mean']:.4f}  "
             f"± {stats_all_wrong['std']:.4f}")
    log.info(f"  Separation ratio        : {sep_ratio:.3f}x")
    log.info(f"  DECISION: {q6_decision}")
    log.info("=" * W)
    log.info("  Q7 — Accuracy / Coverage")
    log.info("=" * W)
    log.info(f"  EM at 100% coverage : {em_full:.4f}  (full verifier EM)")
    if curve_70:
        log.info(f"  EM at ~70% coverage : {curve_70['em']:.4f}  "
                 f"(+{curve_70['em'] - em_full:.4f} from abstaining 30%)")
    if best_tau_entry:
        log.info(f"  Best τ (≥50% cov)   : {best_tau_entry['tau']}  →  "
                 f"EM={best_tau_entry['em_on_answered']:.4f}  "
                 f"coverage={best_tau_entry['coverage']:.2%}")
    log.info("-" * W)
    log.info(f"  {'Coverage':>10}  {'N':>6}  {'EM':>8}")
    log.info(f"  {'-'*28}")
    for pt in curve:
        marker = " ◀" if abs(pt["coverage"] - 1.0) < 0.01 else ""
        log.info(f"  {pt['coverage']:>10.0%}  {pt['n']:>6}  "
                 f"{pt['em']:>8.4f}{marker}")
    log.info("=" * W)
    log.info("  Threshold sweep (key rows):")
    log.info(f"  {'τ':>6}  {'Abstain%':>10}  {'Coverage':>10}  "
             f"{'EM answered':>12}  {'Abstain has-correct%':>22}")
    log.info(f"  {'-'*66}")
    key_taus = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7}
    for s in sweep:
        if s["tau"] in key_taus and s["em_on_answered"] is not None:
            log.info(
                f"  {s['tau']:>6.2f}  {s['abstain_rate']:>10.1%}  "
                f"{s['coverage']:>10.1%}  "
                f"{s['em_on_answered']:>12.4f}  "
                f"{(s['abstain_has_correct_rate'] or 0):>22.1%}"
            )
    log.info("=" * W)


if __name__ == "__main__":
    main()