#!/usr/bin/env python3
"""
experiments/check_status.py — Zero-dependency file status checker

Run this before anything else to see what exists across all 3 settings.
No numpy, no scipy, no matplotlib required.

Usage:
    python3 experiments/check_status.py \
        --proj_root /var/tmp/u24sf51014/sro/work/sro-proof-hotpot
"""

import argparse
import json
import os


def fmt_size(path: str) -> str:
    if not os.path.exists(path):
        return ""
    kb = os.path.getsize(path) // 1024
    if kb >= 1024:
        return f"({kb//1024:,}MB)"
    return f"({kb:,}KB)"


def check(path: str) -> tuple:
    exists = os.path.exists(path)
    size   = fmt_size(path)
    icon   = "✓" if exists else "✗"
    return exists, icon, size


def count_lines(path: str) -> str:
    if not os.path.exists(path):
        return ""
    try:
        n = sum(1 for _ in open(path))
        return f"  [{n:,} lines]"
    except:
        return ""


def peek_bootstrap(path: str) -> None:
    """Print the key comparisons from a bootstrap results JSON."""
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            data = json.load(f)

        comparisons_of_interest = {
            "Z_full_vs_Z2_surface":    "Chain marginal (Z_full − Z2)",
            "Z3_chain_vs_Z1_majority": "Z3 vs Z1 (no voting)",
            "Z3_chain_vs_Z2_surface":  "Z3 vs Z2",
        }

        pairwise = data.get("pairwise_tests", {})
        print(f"      Key results:")
        for key, label in comparisons_of_interest.items():
            test = pairwise.get(key)
            if test is None:
                # Try case-insensitive match
                for k, v in pairwise.items():
                    if key.lower().replace("_", "") in k.lower().replace("_", ""):
                        test = v
                        break
            if test is None:
                print(f"        {label:<35}  NOT IN REPORT")
                continue
            delta_pp = test.get("observed_delta_pp",
                       round(test.get("observed_delta", 0) * 100, 2))
            p = test.get("p_value", 1.0)
            sig = ("***" if p < 0.001 else
                   "**"  if p < 0.01  else
                   "*"   if p < 0.05  else
                   f"ns (p={p:.3f})")
            print(f"        {label:<35}  {delta_pp:>+6.2f}pp  {sig}")
    except Exception as e:
        print(f"      Could not read: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root",
                    default="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot")
    args = ap.parse_args()
    R = args.proj_root

    settings = [
        {
            "name":     "HotpotQA MDR  (exp_phase0 / exp0c)",
            "key":      "hotpotqa_mdr",
            "files": {
                "bootstrap_results.json":  f"{R}/exp_phase0/results/bootstrap_results.json",
                "bootstrap_report.txt":    f"{R}/exp_phase0/results/bootstrap_report.txt",
                "z1_majority_preds":       f"{R}/exp_phase0/results/z1_majority_preds.jsonl",
                "z2_surface_preds":        f"{R}/exp_phase0/results/z2_surface_preds.jsonl",
                "z3_chain_preds":          f"{R}/exp_phase0/results/z3_chain_preds.jsonl",
                "z_full_preds":            f"{R}/exp_phase0/results/z_full_preds.jsonl",
                "dev_hop_scores":          f"{R}/exp0c/preds/dev_hop_scores.jsonl",
                "dev_qa_scores":           f"{R}/exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl",
                "dev_lex_features":        f"{R}/exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl",
                "evidence (K200)":         f"{R}/exp0c/evidence/dev_K200_chains.jsonl",
                "gold (hotpot)":           f"{R}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json",
            },
            "bootstrap": f"{R}/exp_phase0/results/bootstrap_results.json",
        },
        {
            "name":     "HotpotQA Distractor  (exp_distractor)",
            "key":      "hotpotqa_distractor",
            "files": {
                "bootstrap_results.json":  f"{R}/exp_distractor/results/bootstrap_results.json",
                "z1_majority_preds":       f"{R}/exp_distractor/results/z1_majority_preds.jsonl",
                "z2_surface_preds":        f"{R}/exp_distractor/results/z2_surface_preds.jsonl",
                "z3_chain_preds":          f"{R}/exp_distractor/results/z3_chain_preds.jsonl",
                "z_full_preds":            f"{R}/exp_distractor/results/z_full_preds.jsonl",
                "dev_hop_scores":          f"{R}/exp_distractor/preds/dev_hop_scores.jsonl",
                "dev_qa_scores":           f"{R}/exp_distractor/preds/dev_qa_hop_scores.jsonl",
                "dev_lex_features":        f"{R}/exp_distractor/preds/dev_lex_features.jsonl",
                "evidence (distractor)":   f"{R}/exp_distractor/evidence/dev_distractor_chains.jsonl",
                "gold (hotpot)":           f"{R}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json",
            },
            "bootstrap": f"{R}/exp_distractor/results/bootstrap_results.json",
        },
        {
            "name":     "2WikiMultiHopQA  (exp_wiki2)",
            "key":      "wiki2",
            "files": {
                "bootstrap_results.json":  f"{R}/exp_wiki2/results/bootstrap_results.json",
                "z1_majority_preds":       f"{R}/exp_wiki2/results/z1_majority_preds.jsonl",
                "z2_surface_preds":        f"{R}/exp_wiki2/results/z2_surface_preds.jsonl",
                "z3_chain_preds":          f"{R}/exp_wiki2/results/z3_chain_preds.jsonl",
                "z_full_preds":            f"{R}/exp_wiki2/results/z_full_preds.jsonl",
                "dev_hop_scores":          f"{R}/exp_wiki2/preds/dev_hop_scores.jsonl",
                "dev_qa_scores":           f"{R}/exp_wiki2/preds/dev_qa_hop_scores.jsonl",
                "dev_lex_features":        f"{R}/exp_wiki2/preds/dev_lex_features.jsonl",
                "evidence (wiki2)":        f"{R}/exp_wiki2/evidence/dev_wiki2_chains.jsonl",
                "gold (wiki2)":            f"{R}/data/wiki2/raw/dev_normalized.json",
            },
            "bootstrap": f"{R}/exp_wiki2/results/bootstrap_results.json",
        },
    ]

    print("\n" + "━"*70)
    print("  FILE STATUS CHECK — No dependencies required")
    print("━"*70)

    all_missing = []
    all_present = []

    for s in settings:
        print(f"\n  {'─'*60}")
        print(f"  {s['name']}")
        print(f"  {'─'*60}")

        for label, path in s["files"].items():
            exists, icon, size = check(path)
            lines = count_lines(path) if exists and path.endswith(".jsonl") else ""
            status = f"{icon}  {label:<30}  {size}{lines}"
            print(f"    {status}")
            if exists:
                all_present.append(path)
            else:
                all_missing.append((s["name"], label, path))

        # Peek at bootstrap results if they exist
        if os.path.exists(s["bootstrap"]):
            print()
            peek_bootstrap(s["bootstrap"])

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print(f"  SUMMARY")
    print(f"{'━'*70}")
    print(f"  Present : {len(all_present)}")
    print(f"  Missing : {len(all_missing)}")

    if all_missing:
        print(f"\n  MISSING FILES:")
        for setting_name, label, path in all_missing:
            print(f"    [{setting_name}]  {label}")
            print(f"      {path}")

    # ── Install instructions ───────────────────────────────────────────
    print(f"\n{'━'*70}")
    print(f"  TO RUN THE FULL EXPERIMENT SCRIPT")
    print(f"{'━'*70}")
    print(f"""
  Install required packages first:

    pip install numpy scipy matplotlib --break-system-packages

    OR if using conda:
    conda install numpy scipy matplotlib

  Then run:

    python3 experiments/run_hypothesis_experiments.py \\
        --proj_root /var/tmp/u24sf51014/sro/work/sro-proof-hotpot \\
        --out_dir   experiments/results

  If matplotlib is unavailable (text output + JSON only, no plots):

    python3 experiments/run_hypothesis_experiments.py \\
        --proj_root /var/tmp/u24sf51014/sro/work/sro-proof-hotpot \\
        --out_dir   experiments/results \\
        --no_plots
""")


if __name__ == "__main__":
    main()