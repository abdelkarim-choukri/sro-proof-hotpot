#!/usr/bin/env python3
"""
sfav_aggregate.py — Aggregate results from all 30 SFAV experiment runs.

Reads every metrics.json in exp_sfav/runs/ and produces:
  1. Per-run table (arch, seed, fold, EM, Pearson, CKA)
  2. Aggregated mean ± std per architecture (across 3 seeds × 5 folds)
  3. Verdict: does SFAV beat Architecture A on EM? Does Pearson improve?
  4. JSON output: exp_sfav/results/aggregate.json

Decision rule (pre-registered, matches cross-hop experiment):
  FULL UPDATE if:
    mean_EM(SFAV) - mean_EM(A) >= +0.3pp
    AND Pearson(SFAV) does not WORSEN vs Pearson(A) by more than +0.15
      (lower Pearson = less collapse = better structural property)
  PARTIAL UPDATE if EM gap < 0.3pp but Pearson improves by > 0.10
  NO UPDATE if neither condition holds

Usage:
  cd /var/tmp/u24sf51014/sro/work/sro-proof-hotpot
  /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/sfav_aggregate.py

  # Print only — no file write:
  /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/sfav_aggregate.py --dry_run

  # Check progress mid-run (shows completed runs only):
  /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/sfav_aggregate.py --progress
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ── Configuration ──────────────────────────────────────────────────────────────
RUNS_DIR  = "exp_sfav/runs"
OUT_DIR   = "exp_sfav/results"
LAM       = 0.3
ARCHS     = ["A", "SFAV"]
SEEDS     = [42, 123, 456]
N_FOLDS   = 5

# Decision thresholds
EM_THRESHOLD      = 0.003   # +0.3pp minimum for full update
PEARSON_MAX_WORSE = 0.15    # Pearson may not worsen by more than this
PEARSON_MIN_GAIN  = 0.10    # Minimum Pearson gain for partial update


# ── Helpers ────────────────────────────────────────────────────────────────────

def run_tag(arch: str, seed: int, fold: int) -> str:
    if arch == "SFAV":
        return f"SFAV_lam{LAM:.2f}_s{seed}_f{fold}"
    return f"A_s{seed}_f{fold}"


def load_run(arch: str, seed: int, fold: int) -> Optional[dict]:
    tag  = run_tag(arch, seed, fold)
    path = Path(RUNS_DIR) / tag / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        m = json.load(f)
    m["_tag"]  = tag
    m["_arch"] = arch
    m["_seed"] = seed
    m["_fold"] = fold
    return m


def mean_std(vals: List[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=RUNS_DIR)
    ap.add_argument("--out_dir",  default=OUT_DIR)
    ap.add_argument("--dry_run",  action="store_true")
    ap.add_argument("--progress", action="store_true",
                    help="Show completed runs only, useful mid-experiment")
    args = ap.parse_args()

    # ── Load all runs ─────────────────────────────────────────────────────────
    all_runs: List[dict] = []
    missing: List[str]  = []

    for arch in ARCHS:
        for seed in SEEDS:
            for fold in range(N_FOLDS):
                m = load_run(arch, seed, fold)
                if m is None:
                    missing.append(run_tag(arch, seed, fold))
                else:
                    all_runs.append(m)

    total   = len(ARCHS) * len(SEEDS) * N_FOLDS
    n_done  = len(all_runs)
    n_miss  = len(missing)

    print(f"\n{'='*72}")
    print(f"  SFAV EXPERIMENT RESULTS")
    print(f"  {n_done}/{total} runs complete  ({n_miss} missing)")
    print(f"{'='*72}")

    if args.progress:
        print(f"\n  Completed runs ({n_done}):")
        for r in sorted(all_runs, key=lambda x: (x["_arch"], x["_seed"], x["_fold"])):
            print(f"    {r['_tag']:40s}  EM={r.get('em','?'):.4f}  "
                  f"Pearson={r.get('pearson_flat_minhop','?')}"
                  f"  CKA={r.get('cka_post','?')}")
        if missing:
            print(f"\n  Missing runs ({n_miss}):")
            for t in sorted(missing):
                print(f"    {t}")
        return

    if n_done == 0:
        print("  No completed runs found. Run sfav_run_experiment.sh first.")
        return

    # ── Per-run table ─────────────────────────────────────────────────────────
    print(f"\n  Per-run results:")
    print(f"  {'tag':42s} {'EM':>7} {'Pearson':>9} {'CKA_pre':>9} {'CKA_post':>9} {'bridge':>8} {'comp':>8}")
    print(f"  {'─'*100}")

    by_arch: Dict[str, List[dict]] = defaultdict(list)
    for r in sorted(all_runs, key=lambda x: (x["_arch"], x["_seed"], x["_fold"])):
        em      = r.get("em", float("nan"))
        pearson = r.get("pearson_flat_minhop") or float("nan")
        cka_pre = r.get("cka_pre") or float("nan")
        cka_pst = r.get("cka_post") or float("nan")
        type_em = r.get("type_em", {})
        bridge  = type_em.get("bridge", float("nan"))
        comp    = type_em.get("comparison", float("nan"))
        print(f"  {r['_tag']:42s} {em:7.4f} {pearson:9.4f} {cka_pre:9.4f} "
              f"{cka_pst:9.4f} {bridge:8.4f} {comp:8.4f}")
        by_arch[r["_arch"]].append(r)

    # ── Aggregated stats per architecture ─────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"\n  Aggregated (mean ± std across {N_FOLDS} folds × {len(SEEDS)} seeds):")
    print(f"\n  {'arch':8s}  {'n':>4}  {'EM':>12}  {'Pearson':>12}  {'CKA_post':>12}  {'bridge_EM':>10}  {'comp_EM':>9}")
    print(f"  {'─'*80}")

    agg: Dict[str, dict] = {}
    for arch in ARCHS:
        runs = by_arch.get(arch, [])
        if not runs:
            continue
        ems      = [r.get("em", float("nan")) for r in runs]
        pearsons = [r.get("pearson_flat_minhop") or float("nan") for r in runs]
        ckas     = [r.get("cka_post") or float("nan") for r in runs]
        bridges  = [r.get("type_em", {}).get("bridge", float("nan")) for r in runs]
        comps    = [r.get("type_em", {}).get("comparison", float("nan")) for r in runs]

        # Filter NaNs
        ems_f = [x for x in ems if not math.isnan(x)]
        prs_f = [x for x in pearsons if not math.isnan(x)]
        cks_f = [x for x in ckas if not math.isnan(x)]
        brs_f = [x for x in bridges if not math.isnan(x)]
        cos_f = [x for x in comps if not math.isnan(x)]

        em_m, em_s    = mean_std(ems_f)
        pr_m, pr_s    = mean_std(prs_f)
        ck_m, ck_s    = mean_std(cks_f)
        br_m, br_s    = mean_std(brs_f)
        co_m, co_s    = mean_std(cos_f)

        agg[arch] = {
            "n": len(runs),
            "em_mean": em_m, "em_std": em_s,
            "pearson_mean": pr_m, "pearson_std": pr_s,
            "cka_mean": ck_m, "cka_std": ck_s,
            "bridge_em_mean": br_m, "comp_em_mean": co_m,
        }

        print(f"  {arch:8s}  {len(runs):4d}  "
              f"{em_m:.4f}±{em_s:.4f}  "
              f"{pr_m:.4f}±{pr_s:.4f}  "
              f"{ck_m:.4f}±{ck_s:.4f}  "
              f"{br_m:.4f}       {co_m:.4f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"\n  VERDICT:")

    if "A" not in agg or "SFAV" not in agg:
        print("  Cannot compute verdict — both A and SFAV must have completed runs.")
    else:
        em_a    = agg["A"]["em_mean"]
        em_sfav = agg["SFAV"]["em_mean"]
        pr_a    = agg["A"]["pearson_mean"]
        pr_sfav = agg["SFAV"]["pearson_mean"]

        em_delta     = em_sfav - em_a
        pearson_diff = pr_sfav - pr_a   # positive = more collapse = worse
        # Lower Pearson = less collapse = structurally better

        print(f"\n  EM delta (SFAV − A)      : {em_delta:+.4f}  "
              f"(threshold ≥ +{EM_THRESHOLD:.3f})")
        print(f"  Pearson delta (SFAV − A) : {pearson_diff:+.4f}  "
              f"(negative = less collapse = better)")
        print(f"  CKA (A / SFAV)           : "
              f"{agg['A']['cka_mean']:.4f} / {agg['SFAV']['cka_mean']:.4f}")
        print(f"  Bridge EM (A / SFAV)     : "
              f"{agg['A']['bridge_em_mean']:.4f} / {agg['SFAV']['bridge_em_mean']:.4f}")
        print(f"  Comp EM (A / SFAV)       : "
              f"{agg['A']['comp_em_mean']:.4f} / {agg['SFAV']['comp_em_mean']:.4f}")

        em_pass      = em_delta >= EM_THRESHOLD
        pearson_ok   = pearson_diff <= PEARSON_MAX_WORSE
        pearson_gain = pearson_diff <= -PEARSON_MIN_GAIN   # SFAV Pearson is lower = better

        print()
        if em_pass and pearson_ok:
            verdict = "FULL UPDATE"
            detail  = (f"EM +{em_delta*100:.2f}pp ≥ +0.3pp  AND  "
                       f"Pearson Δ={pearson_diff:+.3f} (not worsened)")
        elif pearson_gain and not em_pass:
            verdict = "PARTIAL UPDATE"
            detail  = (f"EM +{em_delta*100:.2f}pp < +0.3pp  BUT  "
                       f"Pearson improved by {-pearson_diff:.3f} (structural gain)")
        elif em_pass and not pearson_ok:
            verdict = "PARTIAL UPDATE"
            detail  = (f"EM +{em_delta*100:.2f}pp ≥ +0.3pp  BUT  "
                       f"Pearson worsened by {pearson_diff:.3f}")
        else:
            verdict = "NO UPDATE"
            detail  = (f"EM +{em_delta*100:.2f}pp < +0.3pp  AND  "
                       f"Pearson Δ={pearson_diff:+.3f}")

        print(f"  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │  {verdict:<57}│")
        print(f"  │  {detail:<57}│")
        print(f"  └─────────────────────────────────────────────────────────┘")

    # ── Missing runs reminder ─────────────────────────────────────────────────
    if missing:
        print(f"\n  WARNING: {n_miss} runs not yet complete:")
        for t in sorted(missing):
            print(f"    {t}")
        print(f"\n  Re-run sfav_run_experiment.sh to complete missing runs.")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
        out = {
            "n_complete": n_done,
            "n_total":    total,
            "missing":    sorted(missing),
            "per_run": [
                {
                    "tag":      r["_tag"],
                    "arch":     r["_arch"],
                    "seed":     r["_seed"],
                    "fold":     r["_fold"],
                    "em":       r.get("em"),
                    "pearson":  r.get("pearson_flat_minhop"),
                    "cka_pre":  r.get("cka_pre"),
                    "cka_post": r.get("cka_post"),
                    "bridge_em": r.get("type_em", {}).get("bridge"),
                    "comp_em":   r.get("type_em", {}).get("comparison"),
                    "anchor_delta": r.get("anchor_delta"),
                }
                for r in sorted(all_runs,
                                key=lambda x: (x["_arch"], x["_seed"], x["_fold"]))
            ],
            "aggregated": agg,
        }
        out_path = Path(args.out_dir) / "aggregate.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved: {out_path}")

    print()


if __name__ == "__main__":
    main()