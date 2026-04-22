#!/usr/bin/env python3
"""
aggregate_results.py — Aggregate all 45 (arch × seed × fold) results and
apply pre-registered decision rules.

Computes:
  - OOF EM per (arch, seed): aggregate val EM across 5 folds
  - Mean EM per arch (across 3 seeds) with bootstrap CI
  - Pearson delta (B−A, C−A): mean across seeds×folds
  - CKA delta (B−A, C−A)
  - Anchor delta comparison
  - 2Wiki zero-shot EM comparison
  - Decision verdict per comparison (B vs A, C vs A, C vs B)

Pre-registered decision rules (verbatim from the experiment spec):
  Full update (cross-hop recommended):
    EM(B) >= EM(A) + 0.003    (0.3pp)
    Pearson_delta(B−A) < +0.15
    Anchor_delta(B) >= Anchor_delta(A)
    2Wiki EM(B) >= 2Wiki EM(A)

  Partial update (cross-hop optional):
    EM(B) >= EM(A) (bootstrap CI doesn't exclude 0 on negative side)
    Pearson_delta < +0.15

  No update (per-hop separate stays):
    EM(B) < EM(A)  [bootstrap 95% CI excludes 0 on negative side]  OR
    Pearson_delta > +0.15  OR
    Anchor_delta(B) < Anchor_delta(A) × 0.70   [drops >30%]

Usage:
    python3 tools/aggregate_results.py \\
        --runs_dir exp_crosshop/runs \\
        --out_json exp_crosshop/verdict.json
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════

def bootstrap_mean_ci(vals: List[float], n_boot: int = 2000,
                      seed: int = 0) -> Tuple[float, float, float]:
    """Returns (mean, ci_lo, ci_hi) at 95% with paired bootstrap."""
    arr = np.array(vals, dtype=float)
    rng = np.random.default_rng(seed)
    boot = [rng.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(n_boot)]
    return float(arr.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def bootstrap_diff_ci(vals_a: List[float], vals_b: List[float],
                      n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """Bootstrap CI on mean(B) − mean(A) using paired resampling.
    Pairs are indexed by seed×fold position assumed to match across archs."""
    a = np.array(vals_a, dtype=float)
    b = np.array(vals_b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(b[idx].mean() - a[idx].mean())
    diff_obs = b.mean() - a.mean()
    return float(diff_obs), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def safe_mean(vals: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    return float(np.mean(clean)) if clean else None


# ══════════════════════════════════════════════════════════════════════
# LOAD RESULTS
# ══════════════════════════════════════════════════════════════════════

def load_all_results(runs_dir: str) -> Dict[str, List[dict]]:
    """Load all metrics.json files. Returns {arch: [metrics_dict, ...]}."""
    results: Dict[str, List[dict]] = defaultdict(list)
    runs_path = Path(runs_dir)
    for p in sorted(runs_path.glob("*/metrics.json")):
        with open(p) as f:
            m = json.load(f)
        arch = m.get("arch") or p.parent.name[0]
        results[arch].append(m)
    return dict(results)


# ══════════════════════════════════════════════════════════════════════
# AGGREGATION
# ══════════════════════════════════════════════════════════════════════

def aggregate_arch(runs: List[dict]) -> dict:
    """Aggregate all (seed, fold) runs for one architecture."""
    oof_em_by_seed: Dict[int, List[float]] = defaultdict(list)
    pearson_vals, cka_pre_vals, cka_post_vals, anchor_vals = [], [], [], []
    wiki2_em_vals = []
    type_em_by_type: Dict[str, List[float]] = defaultdict(list)

    for r in runs:
        seed = r.get("seed", 0)
        oof_em_by_seed[seed].append(r["em"])

        if r.get("pearson_flat_minhop") is not None:
            pearson_vals.append(r["pearson_flat_minhop"])
        if r.get("cka_pre") is not None:
            cka_pre_vals.append(r["cka_pre"])
        if r.get("cka_post") is not None:
            cka_post_vals.append(r["cka_post"])
        if r.get("anchor_delta") is not None:
            anchor_vals.append(r["anchor_delta"])
        if r.get("wiki2_zero_shot") and r["wiki2_zero_shot"].get("em") is not None:
            wiki2_em_vals.append(r["wiki2_zero_shot"]["em"])
        for qt, em in (r.get("type_em") or {}).items():
            type_em_by_type[qt].append(em)

    # OOF EM per seed (mean across folds for that seed)
    oof_em_per_seed = {s: float(np.mean(v)) for s, v in oof_em_by_seed.items()}
    all_oof = list(oof_em_per_seed.values())
    em_mean, em_ci_lo, em_ci_hi = bootstrap_mean_ci(all_oof)

    return {
        "oof_em_per_seed":  oof_em_per_seed,
        "oof_em_per_fold":  [r["em"] for r in runs],
        "em_mean":          round(em_mean, 4),
        "em_ci":            [round(em_ci_lo, 4), round(em_ci_hi, 4)],
        "pearson_mean":     round(safe_mean(pearson_vals), 4)
                            if safe_mean(pearson_vals) is not None else None,
        "cka_pre_mean":     round(safe_mean(cka_pre_vals), 4)
                            if safe_mean(cka_pre_vals) is not None else None,
        "cka_post_mean":    round(safe_mean(cka_post_vals), 4)
                            if safe_mean(cka_post_vals) is not None else None,
        "anchor_delta_mean":round(safe_mean(anchor_vals), 4)
                            if safe_mean(anchor_vals) is not None else None,
        "wiki2_em_mean":    round(safe_mean(wiki2_em_vals), 4)
                            if safe_mean(wiki2_em_vals) is not None else None,
        "wiki2_em_n":       len(wiki2_em_vals),
        "type_em": {qt: round(float(np.mean(v)), 4)
                    for qt, v in type_em_by_type.items()},
        "n_runs": len(runs),
    }


# ══════════════════════════════════════════════════════════════════════
# DECISION RULES
# ══════════════════════════════════════════════════════════════════════

PEARSON_BAN_DELTA   = 0.15    # Pearson(B) − Pearson(A) > this → ban
EM_FULL_UPDATE_PP   = 0.003   # EM gain required for "full update" (0.3pp)
ANCHOR_DROP_THRESH  = 0.30    # anchor_delta(B) drops > 30% vs A → ban signal

def compare_archs(agg_a: dict, agg_b: dict,
                  label_a: str = "A", label_b: str = "B") -> dict:
    """Apply pre-registered rules to compare arch B against arch A."""
    em_diff, ci_lo, ci_hi = bootstrap_diff_ci(
        agg_a["oof_em_per_fold"], agg_b["oof_em_per_fold"]
    )

    pearson_a = agg_a["pearson_mean"]
    pearson_b = agg_b["pearson_mean"]
    pearson_delta = (pearson_b - pearson_a) if (
        pearson_a is not None and pearson_b is not None) else None

    anchor_a = agg_a["anchor_delta_mean"]
    anchor_b = agg_b["anchor_delta_mean"]
    anchor_ratio = (anchor_b / anchor_a) if (
        anchor_a is not None and anchor_b is not None
        and abs(anchor_a) > 1e-6) else None

    wiki2_a = agg_a["wiki2_em_mean"]
    wiki2_b = agg_b["wiki2_em_mean"]
    wiki2_diff = (wiki2_b - wiki2_a) if (wiki2_a and wiki2_b) else None

    # Gate evaluations
    em_positive       = em_diff >= 0                        # B not worse
    em_full_update    = em_diff >= EM_FULL_UPDATE_PP        # B meaningfully better
    em_ban_boot       = ci_hi < 0                           # CI entirely negative → B worse
    pearson_ban       = (pearson_delta is not None and pearson_delta > PEARSON_BAN_DELTA)
    anchor_ban        = (anchor_ratio is not None and anchor_ratio < (1 - ANCHOR_DROP_THRESH))
    wiki2_ok          = (wiki2_diff is None or wiki2_diff >= 0)

    # Apply decision tree
    if pearson_ban or em_ban_boot or anchor_ban:
        reasons = []
        if pearson_ban:
            reasons.append(f"Pearson_delta={pearson_delta:.3f} > {PEARSON_BAN_DELTA}")
        if em_ban_boot:
            reasons.append(f"EM boot CI=[{ci_lo:.4f},{ci_hi:.4f}] < 0")
        if anchor_ban:
            reasons.append(f"anchor_ratio={anchor_ratio:.3f} < {1-ANCHOR_DROP_THRESH:.2f}")
        verdict = f"NO UPDATE ({label_b} ruled out): " + "; ".join(reasons)
        verdict_code = "no_update"

    elif em_full_update and not pearson_ban and wiki2_ok and (anchor_ratio is None or anchor_ratio >= 1.0):
        verdict = (f"FULL UPDATE ({label_b} recommended): "
                   f"EM_diff=+{em_diff:.4f} >= {EM_FULL_UPDATE_PP}, "
                   f"Pearson_delta={pearson_delta:.3f} < {PEARSON_BAN_DELTA}, "
                   f"anchor OK, 2Wiki OK")
        verdict_code = "full_update"

    elif em_positive and not pearson_ban:
        verdict = (f"PARTIAL UPDATE ({label_b} optional ablation): "
                   f"EM_diff={em_diff:+.4f} (positive but < {EM_FULL_UPDATE_PP}), "
                   f"Pearson_delta={pearson_delta:.3f if pearson_delta else 'N/A'}")
        verdict_code = "partial_update"

    else:
        verdict = (f"NO UPDATE ({label_b} not recommended): "
                   f"EM_diff={em_diff:+.4f}")
        verdict_code = "no_update"

    return {
        "comparison":    f"{label_b}_vs_{label_a}",
        "em_diff":       round(em_diff, 4),
        "em_diff_ci":    [round(ci_lo, 4), round(ci_hi, 4)],
        "pearson_delta": round(pearson_delta, 4) if pearson_delta is not None else None,
        "pearson_a":     pearson_a,
        "pearson_b":     pearson_b,
        "anchor_ratio":  round(anchor_ratio, 3) if anchor_ratio is not None else None,
        "wiki2_diff":    round(wiki2_diff, 4) if wiki2_diff is not None else None,
        "gate_em_ban_boot":  em_ban_boot,
        "gate_pearson_ban":  pearson_ban,
        "gate_anchor_ban":   anchor_ban,
        "verdict_code":  verdict_code,
        "verdict":       verdict,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--min_runs", type=int, default=5,
                    help="Warn if fewer than this many runs found per arch")
    args = ap.parse_args()

    results = load_all_results(args.runs_dir)

    W = 72
    print(f"\n{'='*W}")
    print("  CROSS-HOP ATTENTION EXPERIMENT — AGGREGATED RESULTS")
    print(f"{'='*W}")

    # ── check completeness ──
    all_archs = ["A", "B", "C"]
    missing = []
    for arch in all_archs:
        n = len(results.get(arch, []))
        expected = 15  # 3 seeds × 5 folds
        status = "✓" if n == expected else f"⚠ {n}/{expected}"
        print(f"  Arch {arch}: {n} runs  {status}")
        if n < args.min_runs:
            missing.append(arch)
    print()

    if missing:
        print(f"  WARNING: Archs {missing} have fewer than {args.min_runs} runs.")
        print(f"  Results may be incomplete. Proceeding anyway.\n")

    # ── aggregate per arch ──
    agg: Dict[str, dict] = {}
    for arch in all_archs:
        runs = results.get(arch, [])
        if not runs:
            print(f"  Arch {arch}: no results found — skipping.")
            continue
        agg[arch] = aggregate_arch(runs)

    # ── print arch summary table ──
    print(f"  {'Arch':<6}  {'EM mean':>8}  {'95% CI':>18}  "
          f"{'Pearson':>8}  {'CKA_post':>9}  {'Anchor_Δ':>10}  {'2Wiki':>7}")
    print("  " + "-" * (W - 2))
    for arch in all_archs:
        if arch not in agg:
            continue
        a = agg[arch]
        ci_str = f"[{a['em_ci'][0]:.4f},{a['em_ci'][1]:.4f}]"
        p_str  = f"{a['pearson_mean']:.4f}" if a['pearson_mean'] is not None else "   N/A"
        ck_str = f"{a['cka_post_mean']:.4f}" if a['cka_post_mean'] is not None else "    N/A"
        an_str = f"{a['anchor_delta_mean']:+.4f}" if a['anchor_delta_mean'] is not None else "      N/A"
        w2_str = f"{a['wiki2_em_mean']:.4f}" if a['wiki2_em_mean'] is not None else "    N/A"
        print(f"  {arch:<6}  {a['em_mean']:>8.4f}  {ci_str:>18}  "
              f"{p_str:>8}  {ck_str:>9}  {an_str:>10}  {w2_str:>7}")
    print()

    # ── apply decision rules ──
    comparisons = []
    comparison_pairs = [("A","B"), ("A","C"), ("B","C")]
    for arch_a, arch_b in comparison_pairs:
        if arch_a not in agg or arch_b not in agg:
            continue
        comp = compare_archs(agg[arch_a], agg[arch_b],
                             label_a=arch_a, label_b=arch_b)
        comparisons.append(comp)
        print(f"  {comp['comparison']}: {comp['verdict']}")
        print(f"    EM diff={comp['em_diff']:+.4f}  "
              f"CI=[{comp['em_diff_ci'][0]:+.4f},{comp['em_diff_ci'][1]:+.4f}]  "
              f"Pearson_Δ={comp['pearson_delta'] if comp['pearson_delta'] else 'N/A'}  "
              f"2Wiki_diff={comp['wiki2_diff'] if comp['wiki2_diff'] is not None else 'N/A'}")
        print()

    # ── Pearson baseline re-calibration note ──
    if "A" in agg and agg["A"]["pearson_mean"] is not None:
        p_a = agg["A"]["pearson_mean"]
        print(f"  NOTE: Arch A Pearson baseline = {p_a:.4f}")
        print(f"        (sanity run showed 0.6568 — consistent with end-to-end")
        print(f"         EM training collapsing per-hop orthogonality)")
        print(f"        Ban threshold for B and C: Pearson > {p_a:.4f} + 0.15 = {p_a+0.15:.4f}")
        print()

    # ── overall recommendation ──
    def best_arch() -> str:
        codes = {c["comparison"]: c["verdict_code"] for c in comparisons}
        b_vs_a = codes.get("B_vs_A", "no_update")
        c_vs_a = codes.get("C_vs_A", "no_update")
        c_vs_b = codes.get("C_vs_B", "no_update")
        if c_vs_a == "full_update":
            return "C (standard cross-hop recommended)"
        elif b_vs_a == "full_update" and c_vs_b != "full_update":
            return "B (lightweight cross-hop recommended)"
        elif b_vs_a == "no_update" and c_vs_a == "no_update":
            return "A (per-hop separate — cross-hop not beneficial)"
        else:
            return "A (partial updates only — keep per-hop as default)"

    recommendation = best_arch()
    print(f"{'='*W}")
    print(f"  OVERALL RECOMMENDATION: {recommendation}")
    print(f"{'='*W}")

    # ── save ──
    report = {
        "runs_dir":      args.runs_dir,
        "per_arch":      agg,
        "comparisons":   comparisons,
        "recommendation": recommendation,
        "n_runs_per_arch": {a: len(results.get(a, [])) for a in all_archs},
        "preregistered_thresholds": {
            "em_full_update_pp":   EM_FULL_UPDATE_PP,
            "pearson_ban_delta":   PEARSON_BAN_DELTA,
            "anchor_drop_thresh":  ANCHOR_DROP_THRESH,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to: {args.out_json}")


if __name__ == "__main__":
    main()