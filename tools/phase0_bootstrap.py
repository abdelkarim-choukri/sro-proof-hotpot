#!/usr/bin/env python3
"""
phase0_bootstrap.py — Bootstrap Significance Tests

PURPOSE:
  Determine whether the differences observed in the Phase 0 ablation
  are statistically significant, or whether they could arise from
  random variation in the 5-fold CV question splits.

  Also designed to handle the future distractor-setting experiment,
  where we need to decompose gains into:
    (a) self-consistency bonus (M=1 → M=5 majority vote)
    (b) chain-aware verification bonus (M=5 majority → M=5 + verifier)
  Only (b) is our claimed contribution.

HOW IT WORKS:
  Bootstrap resampling at the QUESTION level (not candidate level):
    1. Load per-question predictions from each system (z1, z2, z3, z_full)
    2. Load gold answers
    3. Repeat B=10,000 times:
       a. Sample 7,405 question IDs WITH replacement
       b. Compute EM for each system on this bootstrap sample
       c. Compute all pairwise deltas (e.g., Z_full - Z2)
    4. From the B bootstrap deltas, compute:
       - 95% confidence intervals (percentile method)
       - Two-sided p-value: fraction of bootstrap samples where delta ≤ 0
         (i.e., how often does the "better" system fail to beat the "worse" one)

  Why question-level resampling?
    Because our 5-fold CV produces predictions at the question level.
    Two candidates from the same question are NOT independent — they
    share the same evidence, same gold answer, same difficulty.
    Resampling at question level preserves this dependence structure.

WHAT THE TESTS TELL YOU:
  - If Z_full vs Z2 has p < 0.05 → chain features significantly help
  - If Z1 vs Z2 has p > 0.05 → surface features don't significantly differ
    from majority voting (confirms the "surface hurts" finding)
  - If Z3 vs Monolithic has p > 0.05 → chain-only matches monolithic
    (confirms chain features alone capture the full signal)

  For the distractor-setting paper table:
  - The ONLY delta that matters for your contribution claim is:
    "M=5 + verifier" minus "M=5 majority vote"
  - The "M=1 → M=5 majority" delta is self-consistency (not your claim)
  - Both need to be significant, but reported separately

INPUT:
  --results_json   Phase 0 results JSON (has per-system EM)
  --preds_dir      Directory containing z1_majority_preds.jsonl, etc.
  --gold           HotpotQA gold file
  --mono_preds     Monolithic predictions (for the monolithic baseline row)

  Optional (for distractor-setting, run later):
  --m1_preds       M=1 greedy predictions (single answer, no verifier)
  --distractor     Flag to enable distractor-setting comparison mode

OUTPUT:
  --out_dir/bootstrap_results.json   — all CIs and p-values
  --out_dir/bootstrap_report.txt     — human-readable significance table

Usage:
  # For Phase 0 results:
  python3 tools/phase0_bootstrap.py \\
      --preds_dir  exp_phase0/results \\
      --gold       data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
      --mono_preds exp0c/preds/dev_chain_verifier_mean_preds.jsonl \\
      --out_dir    exp_phase0/results \\
      --n_bootstrap 10000 --seed 42

  # Later, for distractor-setting with M=1 baseline:
  python3 tools/phase0_bootstrap.py \\
      --preds_dir     exp_distractor/results \\
      --gold          data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
      --mono_preds    exp0c/preds/dev_chain_verifier_mean_preds.jsonl \\
      --m1_preds      exp_distractor/preds/dev_M1_greedy_preds.jsonl \\
      --distractor \\
      --out_dir       exp_distractor/results \\
      --n_bootstrap 10000 --seed 42
"""

import argparse
import collections
import json
import os
import re
import string
import sys
import time

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1: TEXT UTILITIES (same as phase0_ablations.py)
# ═══════════════════════════════════════════════════════════════════════

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def em_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2: DATA LOADING
#  Loads per-question predictions from each system and computes
#  per-question binary correctness vectors.
# ═══════════════════════════════════════════════════════════════════════

def load_gold(path: str) -> dict:
    """Returns {qid: gold_answer}."""
    data = json.load(open(path))
    return {str(ex["_id"]): ex["answer"] for ex in data}


def load_preds_jsonl(path: str) -> dict:
    """Returns {qid: predicted_answer}."""
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                result[str(rec["qid"])] = rec.get("pred", "")
    return result


def build_correctness_vector(preds: dict, gold_map: dict, qid_order: list) -> np.ndarray:
    """
    Build a binary vector aligned to qid_order.
    correctness[i] = 1 if preds[qid_order[i]] matches gold, else 0.

    This alignment is critical — all systems must be evaluated on
    the exact same question ordering so bootstrap resampling is paired.
    """
    vec = np.zeros(len(qid_order), dtype=np.int32)
    for i, qid in enumerate(qid_order):
        pred = preds.get(qid, "")
        gold = gold_map.get(qid, "")
        vec[i] = em_match(pred, gold)
    return vec


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3: BOOTSTRAP ENGINE
#  Performs paired bootstrap resampling at question level.
#
#  "Paired" means: for each bootstrap sample, we resample the SAME
#  set of question indices for ALL systems, then compare their EM
#  on that identical sample. This is the standard method for
#  comparing NLP systems (Koehn, 2004; Berg-Kirkpatrick et al., 2012;
#  Dror et al., 2018).
#
#  Why not permutation test or sign test?
#  Bootstrap is more general and gives confidence intervals in addition
#  to p-values. The paired structure already handles correlation.
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_paired_test(
    vec_a: np.ndarray,     # binary correctness, system A
    vec_b: np.ndarray,     # binary correctness, system B
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    """
    Paired bootstrap significance test.

    Tests H0: EM(A) = EM(B) against H1: EM(A) ≠ EM(B).

    Returns:
      observed_delta: EM(A) - EM(B) on original data
      ci_lower, ci_upper: 95% CI of the delta
      p_value: two-sided p-value (fraction of bootstraps where
               the sign of delta flips, i.e., delta ≤ 0 when observed > 0
               or delta ≥ 0 when observed < 0)
      bootstrap_deltas: the raw array (for plotting if desired)
    """
    n = len(vec_a)
    assert len(vec_b) == n

    observed_a = vec_a.mean()
    observed_b = vec_b.mean()
    observed_delta = observed_a - observed_b

    # Bootstrap resampling
    boot_deltas = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_a = vec_a[idx].mean()
        boot_b = vec_b[idx].mean()
        boot_deltas[b] = boot_a - boot_b

    # 95% CI (percentile method)
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))

    # Two-sided p-value
    # Count how often the bootstrap delta is on the opposite side of zero
    # from the observed delta. If observed > 0, count boot <= 0 and vice versa.
    if observed_delta > 0:
        p_value = float(np.mean(boot_deltas <= 0))
    elif observed_delta < 0:
        p_value = float(np.mean(boot_deltas >= 0))
    else:
        p_value = 1.0  # exactly zero delta → not significant

    # Also compute bootstrap std for reporting
    boot_std = float(np.std(boot_deltas))

    return {
        "observed_delta": float(observed_delta),
        "observed_delta_pp": round(float(observed_delta) * 100, 2),
        "ci_95_lower": round(ci_lower, 6),
        "ci_95_upper": round(ci_upper, 6),
        "ci_95_lower_pp": round(ci_lower * 100, 2),
        "ci_95_upper_pp": round(ci_upper * 100, 2),
        "p_value": round(p_value, 6),
        "boot_std": round(boot_std, 6),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_single_ci(
    vec: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    """
    Bootstrap confidence interval for a single system's EM.
    """
    n = len(vec)
    observed = float(vec.mean())

    boot_ems = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_ems[b] = vec[idx].mean()

    return {
        "em": round(observed, 6),
        "ci_95_lower": round(float(np.percentile(boot_ems, 2.5)), 6),
        "ci_95_upper": round(float(np.percentile(boot_ems, 97.5)), 6),
        "boot_std": round(float(np.std(boot_ems)), 6),
    }


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 4: MAIN — ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Bootstrap significance tests for Phase 0 ablations")

    ap.add_argument("--preds_dir", required=True,
        help="Directory with z1_majority_preds.jsonl, z2_surface_preds.jsonl, etc.")
    ap.add_argument("--gold", required=True,
        help="data/hotpotqa/raw/hotpot_dev_distractor_v1.json")
    ap.add_argument("--mono_preds", required=True,
        help="Monolithic verifier predictions for baseline comparison")
    ap.add_argument("--out_dir", required=True,
        help="Output directory")

    # Optional: M=1 greedy preds for distractor-setting comparison
    ap.add_argument("--m1_preds", default=None,
        help="M=1 greedy predictions (for distractor-setting comparison)")
    ap.add_argument("--distractor", action="store_true",
        help="Enable distractor-setting comparison mode")

    # Bootstrap params
    ap.add_argument("--n_bootstrap", type=int, default=10000,
        help="Number of bootstrap samples (default: 10000)")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── Load gold ──
    print("Loading gold answers ...")
    gold_map = load_gold(args.gold)
    qid_order = sorted(gold_map.keys())  # fixed ordering for all systems
    n_questions = len(qid_order)
    print(f"  {n_questions} questions")

    # ── Load system predictions ──
    print("Loading predictions ...")

    systems = {}

    # Phase 0 ablation systems
    phase0_files = {
        "Z1_majority":  os.path.join(args.preds_dir, "z1_majority_preds.jsonl"),
        "Z2_surface":   os.path.join(args.preds_dir, "z2_surface_preds.jsonl"),
        "Z3_chain":     os.path.join(args.preds_dir, "z3_chain_preds.jsonl"),
        "Z_full":       os.path.join(args.preds_dir, "z_full_preds.jsonl"),
    }

    for name, path in phase0_files.items():
        if os.path.isfile(path):
            preds = load_preds_jsonl(path)
            vec = build_correctness_vector(preds, gold_map, qid_order)
            systems[name] = vec
            print(f"  ✓ {name}: {path} (EM={vec.mean():.4f})")
        else:
            print(f"  ✗ {name}: {path} NOT FOUND — skipping")

    # Monolithic baseline
    if os.path.isfile(args.mono_preds):
        preds = load_preds_jsonl(args.mono_preds)
        vec = build_correctness_vector(preds, gold_map, qid_order)
        systems["Monolithic"] = vec
        print(f"  ✓ Monolithic: {args.mono_preds} (EM={vec.mean():.4f})")

    # M=1 greedy (for distractor-setting, if provided)
    if args.m1_preds and os.path.isfile(args.m1_preds):
        preds = load_preds_jsonl(args.m1_preds)
        vec = build_correctness_vector(preds, gold_map, qid_order)
        systems["M1_greedy"] = vec
        print(f"  ✓ M1_greedy: {args.m1_preds} (EM={vec.mean():.4f})")

    if len(systems) < 2:
        print("ERROR: Need at least 2 systems to compare. Check paths.")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════
    #  Run bootstrap CIs for each system
    # ═══════════════════════════════════════════════════════════════

    print(f"\nRunning bootstrap ({args.n_bootstrap} samples) ...")
    t0 = time.time()

    results = {"system_cis": {}, "pairwise_tests": {}, "meta": {}}

    # Individual system CIs
    for name, vec in systems.items():
        ci = bootstrap_single_ci(vec, args.n_bootstrap, rng)
        results["system_cis"][name] = ci
        print(f"  {name}: {ci['em']:.4f} "
              f"[{ci['ci_95_lower']:.4f}, {ci['ci_95_upper']:.4f}]")

    # ═══════════════════════════════════════════════════════════════
    #  Pairwise significance tests
    #  We test the comparisons that matter for the paper:
    #
    #  KEY TESTS (Phase 0):
    #  1. Z_full vs Z2      — chain marginal (the decisive test)
    #  2. Z_full vs Z1      — full model vs self-consistency
    #  3. Z_full vs Z3      — surface feature marginal
    #  4. Z1 vs Z2          — does surface XGB beat majority vote?
    #  5. Z3 vs Monolithic  — chain-only vs full monolithic
    #  6. Z_full vs Mono    — two-stage vs monolithic
    #
    #  KEY TESTS (Distractor-setting, when available):
    #  7. M=5 verifier vs M=5 majority  — YOUR contribution
    #  8. M=5 majority vs M=1 greedy    — self-consistency (not your claim)
    #  9. M=5 verifier vs M=1 greedy    — total pipeline gain
    # ═══════════════════════════════════════════════════════════════

    # Define comparisons as (system_A, system_B, label)
    # Delta = EM(A) - EM(B), so A should be the "better" system
    comparisons = []

    # Phase 0 comparisons (if systems available)
    phase0_pairs = [
        ("Z_full",    "Z2_surface",  "Chain marginal (Z_full - Z2)"),
        ("Z_full",    "Z1_majority", "Full model vs majority vote"),
        ("Z_full",    "Z3_chain",    "Surface marginal (Z_full - Z3)"),
        ("Z1_majority", "Z2_surface","Majority vote vs surface XGB"),
        ("Z3_chain",  "Monolithic",  "Chain-only vs monolithic"),
        ("Z_full",    "Monolithic",  "Two-stage vs monolithic"),
        ("Z3_chain",  "Z1_majority", "Chain-only vs majority vote"),
        ("Z3_chain",  "Z2_surface",  "Chain-only vs surface-only"),
    ]

    for a, b, label in phase0_pairs:
        if a in systems and b in systems:
            comparisons.append((a, b, label))

    # Distractor-setting comparisons
    if args.distractor and "M1_greedy" in systems:
        distractor_pairs = [
            ("Z_full",     "Z1_majority", "Verifier vs self-consistency (YOUR claim)"),
            ("Z1_majority","M1_greedy",   "Self-consistency bonus (M=5 vs M=1)"),
            ("Z_full",     "M1_greedy",   "Total pipeline gain (M=5+verifier vs M=1)"),
        ]
        for a, b, label in distractor_pairs:
            if a in systems and b in systems:
                comparisons.append((a, b, label))

    print(f"\n  Running {len(comparisons)} pairwise tests ...")

    for sys_a, sys_b, label in comparisons:
        test = bootstrap_paired_test(
            systems[sys_a], systems[sys_b],
            args.n_bootstrap, rng
        )
        key = f"{sys_a}_vs_{sys_b}"
        test["label"] = label
        test["system_a"] = sys_a
        test["system_b"] = sys_b
        results["pairwise_tests"][key] = test

        sig_marker = "***" if test["significant_at_01"] else \
                     "**"  if test["significant_at_05"] else "ns"
        print(f"    {label}")
        print(f"      Δ = {test['observed_delta_pp']:+.2f}pp  "
              f"95%CI [{test['ci_95_lower_pp']:+.2f}, {test['ci_95_upper_pp']:+.2f}]  "
              f"p = {test['p_value']:.4f}  {sig_marker}")

    elapsed = time.time() - t0
    print(f"\n  Bootstrap completed in {elapsed:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    #  Generate human-readable report
    # ═══════════════════════════════════════════════════════════════

    report_lines = []
    report_lines.append("=" * 78)
    report_lines.append("  BOOTSTRAP SIGNIFICANCE TESTS — Phase 0 Ablations")
    report_lines.append(f"  B = {args.n_bootstrap} bootstrap samples, seed = {args.seed}")
    report_lines.append(f"  N = {n_questions} questions")
    report_lines.append("=" * 78)

    # System CIs table
    report_lines.append("")
    report_lines.append("  SYSTEM CONFIDENCE INTERVALS")
    report_lines.append("  " + "-" * 74)
    report_lines.append(f"  {'System':<25} {'EM':>8} {'95% CI':>22} {'±':>8}")
    report_lines.append("  " + "-" * 74)

    for name in ["M1_greedy", "Z1_majority", "Z2_surface",
                  "Z3_chain", "Z_full", "Monolithic"]:
        if name in results["system_cis"]:
            ci = results["system_cis"][name]
            half_width = (ci["ci_95_upper"] - ci["ci_95_lower"]) / 2
            report_lines.append(
                f"  {name:<25} {ci['em']:>8.4f} "
                f"[{ci['ci_95_lower']:.4f}, {ci['ci_95_upper']:.4f}] "
                f"±{half_width:.4f}"
            )

    # Pairwise tests table
    report_lines.append("")
    report_lines.append("  PAIRWISE SIGNIFICANCE TESTS")
    report_lines.append("  " + "-" * 74)
    report_lines.append(
        f"  {'Comparison':<42} {'Δ (pp)':>8} {'95% CI':>18} {'p':>8} {'Sig':>5}")
    report_lines.append("  " + "-" * 74)

    for key, test in results["pairwise_tests"].items():
        sig = "***" if test["significant_at_01"] else \
              "**"  if test["significant_at_05"] else "ns"
        report_lines.append(
            f"  {test['label']:<42} "
            f"{test['observed_delta_pp']:>+8.2f} "
            f"[{test['ci_95_lower_pp']:>+.2f},{test['ci_95_upper_pp']:>+.2f}] "
            f"{test['p_value']:>8.4f} "
            f"{sig:>5}"
        )

    # Interpretation
    report_lines.append("")
    report_lines.append("  " + "-" * 74)
    report_lines.append("  INTERPRETATION FOR THE PAPER")
    report_lines.append("  " + "-" * 74)

    chain_test = results["pairwise_tests"].get("Z_full_vs_Z2_surface")
    if chain_test:
        if chain_test["significant_at_05"]:
            report_lines.append(
                f"  ✓ Chain marginal ({chain_test['observed_delta_pp']:+.2f}pp) is "
                f"SIGNIFICANT (p={chain_test['p_value']:.4f})")
            report_lines.append(
                f"    → The chain-aware claim holds under bootstrap testing")
        else:
            report_lines.append(
                f"  ⚠ Chain marginal ({chain_test['observed_delta_pp']:+.2f}pp) is "
                f"NOT significant (p={chain_test['p_value']:.4f})")
            report_lines.append(
                f"    → Cannot claim chain features significantly help")

    mv_test = results["pairwise_tests"].get("Z1_majority_vs_Z2_surface")
    if mv_test:
        if not mv_test["significant_at_05"]:
            report_lines.append(
                f"  ✓ Z1 vs Z2 NOT significant (p={mv_test['p_value']:.4f})")
            report_lines.append(
                f"    → Confirms surface XGB does not significantly beat majority vote")
        else:
            report_lines.append(
                f"  Note: Z1 vs Z2 is significant (p={mv_test['p_value']:.4f})")
            if mv_test["observed_delta"] > 0:
                report_lines.append(
                    f"    → Majority vote significantly beats surface XGB!")

    z3_mono_test = results["pairwise_tests"].get("Z3_chain_vs_Monolithic")
    if z3_mono_test:
        if not z3_mono_test["significant_at_05"]:
            report_lines.append(
                f"  ✓ Z3 vs Mono NOT significant (p={z3_mono_test['p_value']:.4f})")
            report_lines.append(
                f"    → Chain-only ≈ monolithic (9 features match 19+is_bad)")

    report_lines.append("")
    report_lines.append("  PAPER-READY FORMATTING:")
    report_lines.append("  Report CIs as: 'EM = 0.XXXX (95% CI: [0.XXXX, 0.XXXX])'")
    report_lines.append("  Report deltas as: '+X.XXpp (p < 0.05, bootstrap B=10000)'")
    report_lines.append("")

    # Distractor-setting guidance
    if args.distractor:
        report_lines.append("  DISTRACTOR-SETTING PAPER TABLE:")
        report_lines.append("  " + "-" * 74)
        report_lines.append("  Row 1: M=1 greedy       → comparable to extractive SOTA")
        report_lines.append("  Row 2: M=5 majority     → self-consistency gain (cite Wang et al.)")
        report_lines.append("  Row 3: M=5 + verifier   → YOUR contribution (chain-aware)")
        report_lines.append("  Only claim credit for Row 3 minus Row 2.")
        report_lines.append("")
    else:
        report_lines.append("  FUTURE: When you run the distractor-setting experiment,")
        report_lines.append("  re-run this script with --m1_preds and --distractor to get")
        report_lines.append("  the decomposed significance tests for the paper table.")
        report_lines.append("")

    report_lines.append("=" * 78)

    # Print and save report
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = os.path.join(args.out_dir, "bootstrap_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"\nReport saved to {report_path}")

    # Save full results
    results["meta"] = {
        "n_questions": n_questions,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "distractor_mode": args.distractor,
        "systems_loaded": list(systems.keys()),
    }

    results_path = os.path.join(args.out_dir, "bootstrap_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {results_path}")


if __name__ == "__main__":
    main()