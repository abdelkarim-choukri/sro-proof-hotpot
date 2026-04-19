#!/usr/bin/env python3
"""
experiments/run_b2_fixed.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
B2 Fixed: Paired Within-Question Delta Analysis

WHY THE PREVIOUS B2 FAILED
    The previous analysis compared two different groups across two different
    question sets:
      - CHAIN_WINS:   |nli_hop1 - nli_hop2| of Z2's wrong pick
      - FLAT_CORRECT: |nli_hop1 - nli_hop2| of the gold answer

    Both groups have similar mean imbalance (~0.55) because BOTH bridge
    questions and comparison questions naturally produce high-imbalance
    candidates. The signal washes out.

THE CORRECT ANALYSIS — Paired within CHAIN_WINS
    For each of the 241 CHAIN_WINS questions:
      wrong_cand  = the candidate Z2 picked (wrong answer)
      right_cand  = the correct candidate (gold answer match)

    Compute Δ = right_score − wrong_score for each feature:
      Δ_nli_flat   → near 0 or NEGATIVE (flat NLI was fooled — Z2 chose wrong)
      Δ_nli_hop1   → could go either way
      Δ_nli_hop2   → POSITIVE (correct answer has higher hop2 score)
      Δ_min_hop    → POSITIVE (correct answer more grounded in BOTH hops)
      Δ_imbalance  → NEGATIVE (correct answer is more balanced)

    If Δ_nli_flat ≈ 0 while Δ_min_hop > 0, that IS the mechanism proof:
    flat NLI cannot distinguish correct from wrong in these cases,
    but min-hop (per-hop decomposition) can.

    This is what the project's exp_b2b_delta_features.py already does.
    Run that script directly. This file just documents the correct usage
    and adds a quick inline version you can run immediately.

USAGE — Use the existing project script (recommended):
    python3 tools/exp_b2b_delta_features.py \
        --hop_scores  exp_wiki2/preds/dev_hop_scores.jsonl \
        --z2_preds    exp_wiki2/results/z2_surface_preds.jsonl \
        --zfull_preds exp_wiki2/results/z_full_preds.jsonl \
        --gold        data/wiki2/raw/dev_normalized.json \
        --type_field  type \
        --label       wiki2 \
        --out_dir     experiments/results/b2_fixed/wiki2

    python3 tools/exp_b2b_delta_features.py \
        --hop_scores  exp0c/preds/dev_hop_scores.jsonl \
        --z2_preds    exp_phase0/results/z2_surface_preds.jsonl \
        --zfull_preds exp_phase0/results/z_full_preds.jsonl \
        --gold        data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
        --label       hotpotqa_mdr \
        --out_dir     experiments/results/b2_fixed/hotpotqa_mdr

OR run this script which does the same analysis inline:
    python3 experiments/run_b2_fixed.py \
        --proj_root /var/tmp/u24sf51014/sro/work/sro-proof-hotpot \
        --out_dir   experiments/results/b2_fixed
"""

import argparse
import json
import os
import re
import string
from typing import Dict, Iterator, List, Optional

import numpy as np


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def em(p: str, g: str) -> bool:
    return normalize(p) == normalize(g)

def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_preds(path: str) -> Dict[str, str]:
    result = {}
    for rec in iter_jsonl(path):
        result[str(rec["qid"])] = str(rec.get("pred", ""))
    return result

def load_hop_scores(path: str) -> Dict[str, List[dict]]:
    result = {}
    for rec in iter_jsonl(path):
        result[str(rec["qid"])] = rec.get("candidates", [])
    return result

def load_gold(path: str) -> Dict[str, dict]:
    """Handles HotpotQA (list with _id) and 2Wiki (list with id or _id)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read(1)
        f.seek(0)
        if raw.strip() == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    result = {}
    for ex in data:
        qid = str(ex.get("_id", ex.get("id", "")))
        ans = ex.get("answer", "")
        if isinstance(ans, dict):
            ans = ans.get("answer", "")
        result[qid] = {
            "answer": str(ans),
            "type":   ex.get("type", ex.get("q_type", "bridge")),
        }
    return result

def find_candidate(cands: List[dict], answer: str) -> Optional[dict]:
    """Return first candidate whose answer matches (normalized)."""
    a = normalize(answer)
    for c in cands:
        if normalize(c.get("answer", c.get("answer_text", ""))) == a:
            return c
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CORE ANALYSIS — Paired delta within CHAIN_WINS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FEATURES = ["nli_flat", "nli_hop1", "nli_hop2", "min_hop", "mean_hop", "imbalance"]

def run_b2(
    label:     str,
    hop_path:  str,
    z2_path:   str,
    zf_path:   str,
    gold_path: str,
    out_dir:   str,
    no_plots:  bool = False,
) -> None:

    print(f"\n{'━'*70}")
    print(f"  B2 Fixed: {label}")
    print(f"{'━'*70}")

    for p, n in [(hop_path, "hop_scores"), (z2_path, "z2_preds"),
                 (zf_path, "zfull_preds"), (gold_path, "gold")]:
        if not os.path.exists(p):
            print(f"  ✗ {n} not found: {p}")
            return

    gold       = load_gold(gold_path)
    hop_scores = load_hop_scores(hop_path)
    z2_preds   = load_preds(z2_path)
    zf_preds   = load_preds(zf_path)

    print(f"  gold: {len(gold):,}  hop_scores: {len(hop_scores):,}  "
          f"z2: {len(z2_preds):,}  zf: {len(zf_preds):,}")

    # ── Classify questions and collect PAIRED scores ──────────────────
    # Key change from broken B2:
    # We work WITHIN CHAIN_WINS questions only.
    # For each CHAIN_WINS question we extract BOTH:
    #   - wrong_cand scores (what Z2 picked — incorrect)
    #   - right_cand scores (the correct answer — what Z_full picked)
    # Then compute Δ = right − wrong for every feature.

    chain_wins_pairs = []   # list of {qid, type, wrong_scores, right_scores, delta}
    skipped_no_wrong = 0
    skipped_no_right = 0
    n_chain_wins = 0
    n_chain_hurts = 0
    n_both_right = 0
    n_both_wrong = 0

    for qid, gold_info in gold.items():
        gold_ans = gold_info["answer"]
        qtype    = gold_info["type"]

        z2_pred = z2_preds.get(qid, "")
        zf_pred = zf_preds.get(qid, "")

        z2_correct = em(z2_pred, gold_ans)
        zf_correct = em(zf_pred, gold_ans)

        if not z2_correct and zf_correct:
            n_chain_wins += 1
        elif z2_correct and not zf_correct:
            n_chain_hurts += 1
            continue
        elif z2_correct and zf_correct:
            n_both_right += 1
            continue
        else:
            n_both_wrong += 1
            continue

        # CHAIN_WINS: get both wrong and correct candidate scores
        cands = hop_scores.get(qid, [])

        wrong_cand = find_candidate(cands, z2_pred)
        right_cand = find_candidate(cands, gold_ans)

        if wrong_cand is None:
            skipped_no_wrong += 1
            continue
        if right_cand is None:
            skipped_no_right += 1
            continue

        def get_scores(c: dict) -> dict:
            h1 = float(c.get("nli_hop1", 0) or 0)
            h2 = float(c.get("nli_hop2", 0) or 0)
            fl = float(c.get("nli_flat", 0) or 0)
            return {
                "nli_flat":  fl,
                "nli_hop1":  h1,
                "nli_hop2":  h2,
                "min_hop":   min(h1, h2),
                "mean_hop":  (h1 + h2) / 2,
                "imbalance": abs(h1 - h2),
            }

        wrong_scores = get_scores(wrong_cand)
        right_scores = get_scores(right_cand)
        delta = {f: right_scores[f] - wrong_scores[f] for f in FEATURES}

        chain_wins_pairs.append({
            "qid":          qid,
            "type":         qtype,
            "z2_pred":      z2_pred,
            "zf_pred":      zf_pred,
            "gold_ans":     gold_ans,
            "wrong_scores": wrong_scores,
            "right_scores": right_scores,
            "delta":        delta,
        })

    print(f"\n  Confusion matrix:")
    print(f"    CHAIN_WINS  (Z2 wrong, Z_full right) : {n_chain_wins:,}")
    print(f"    CHAIN_HURTS (Z2 right, Z_full wrong)  : {n_chain_hurts:,}")
    print(f"    BOTH_RIGHT                            : {n_both_right:,}")
    print(f"    BOTH_WRONG                            : {n_both_wrong:,}")
    print(f"    Helps/Hurts ratio                     : "
          f"{n_chain_wins/max(n_chain_hurts,1):.2f}×")
    print(f"\n  CHAIN_WINS pairs extracted: {len(chain_wins_pairs):,}")
    if skipped_no_wrong:
        print(f"  Skipped (wrong candidate not in hop_scores): {skipped_no_wrong}")
    if skipped_no_right:
        print(f"  Skipped (gold candidate not in hop_scores) : {skipped_no_right}")

    if not chain_wins_pairs:
        print("  No pairs to analyse — check file paths and schemas")
        return

    # ── Compute delta statistics ──────────────────────────────────────
    print(f"\n  PAIRED DELTA (right − wrong) within CHAIN_WINS questions:")
    print(f"  Positive Δ means the correct answer scores HIGHER than wrong answer.")
    print(f"  Negative Δ means the feature is being FOOLED by the wrong answer.")
    print()
    print(f"  {'Feature':<14}  {'Mean Δ':>9}  {'Median Δ':>10}  "
          f"{'% Δ>0':>8}  {'p-value':>10}  Sig")
    print(f"  {'─'*62}")

    delta_stats = {}
    for feat in FEATURES:
        deltas = np.array([p["delta"][feat] for p in chain_wins_pairs])
        pct_positive = (deltas > 0).mean() * 100

        try:
            from scipy import stats
            _, p_val = stats.wilcoxon(deltas)
        except ImportError:
            p_val = None

        sig = ""
        if p_val is not None:
            sig = ("***" if p_val < 0.001 else
                   "**"  if p_val < 0.01  else
                   "*"   if p_val < 0.05  else
                   "ns")
            p_str = f"{p_val:.4f}"
        else:
            p_str = "N/A"

        print(f"  {feat:<14}  {deltas.mean():>+9.4f}  "
              f"{np.median(deltas):>+10.4f}  "
              f"{pct_positive:>7.1f}%  {p_str:>10}  {sig}")

        delta_stats[feat] = {
            "mean_delta":    float(deltas.mean()),
            "median_delta":  float(np.median(deltas)),
            "pct_positive":  float(pct_positive),
            "p_value":       float(p_val) if p_val is not None else None,
        }

    # ── Interpret ─────────────────────────────────────────────────────
    print()
    print(f"  KEY INTERPRETATION:")
    nf_delta = delta_stats["nli_flat"]["mean_delta"]
    mh_delta = delta_stats["min_hop"]["mean_delta"]
    h2_delta = delta_stats["nli_hop2"]["mean_delta"]
    imb_delta = delta_stats["imbalance"]["mean_delta"]

    if nf_delta < 0.01:
        print(f"  ✓ nli_flat Δ = {nf_delta:+.4f}: flat NLI is FOOLED — "
              f"cannot distinguish correct from wrong")
    else:
        print(f"  ✗ nli_flat Δ = {nf_delta:+.4f}: flat NLI IS discriminating — "
              f"something unexpected")

    if mh_delta > 0:
        print(f"  ✓ min_hop Δ  = {mh_delta:+.4f}: per-hop min-score IS higher "
              f"for correct answers — chain signal works")
    else:
        print(f"  ✗ min_hop Δ  = {mh_delta:+.4f}: min_hop does NOT discriminate")

    if h2_delta > 0:
        print(f"  ✓ nli_hop2 Δ = {h2_delta:+.4f}: hop2 score IS higher for "
              f"correct answers — correct answers grounded in answer doc")
    else:
        print(f"  ~ nli_hop2 Δ = {h2_delta:+.4f}: hop2 not clearly higher")

    if imb_delta < 0:
        print(f"  ✓ imbalance Δ = {imb_delta:+.4f}: correct answers ARE more "
              f"balanced than wrong answers")
    else:
        print(f"  ✗ imbalance Δ = {imb_delta:+.4f}: imbalance not lower for "
              f"correct answers — old B2 hypothesis was wrong")

    # ── Per-type breakdown ────────────────────────────────────────────
    type_groups = {}
    for p in chain_wins_pairs:
        t = p["type"] or "unknown"
        if t not in type_groups:
            type_groups[t] = []
        type_groups[t].append(p)

    if len(type_groups) > 1:
        print(f"\n  PER-TYPE DELTA (mean Δ nli_flat vs Δ min_hop):")
        print(f"  {'Type':<20}  {'N':>5}  {'Δ nli_flat':>11}  "
              f"{'Δ min_hop':>10}  {'Δ nli_hop2':>11}")
        print(f"  {'─'*62}")
        for qtype, pairs in sorted(type_groups.items(),
                                    key=lambda x: -len(x[1])):
            fl = np.mean([p["delta"]["nli_flat"] for p in pairs])
            mh = np.mean([p["delta"]["min_hop"]  for p in pairs])
            h2 = np.mean([p["delta"]["nli_hop2"] for p in pairs])
            print(f"  {qtype:<20}  {len(pairs):>5}  "
                  f"{fl:>+11.4f}  {mh:>+10.4f}  {h2:>+11.4f}")

    # ── Save results ──────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    stats_path = os.path.join(out_dir, f"b2_delta_stats_{label}.json")
    with open(stats_path, "w") as f:
        json.dump({
            "label":         label,
            "n_chain_wins":  n_chain_wins,
            "n_pairs_used":  len(chain_wins_pairs),
            "delta_stats":   delta_stats,
            "per_type":      {
                t: {feat: float(np.mean([p["delta"][feat] for p in pairs]))
                    for feat in FEATURES}
                for t, pairs in type_groups.items()
            },
        }, f, indent=2)
    print(f"\n  Saved → {stats_path}")

    # ── Plot ──────────────────────────────────────────────────────────
    if not no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(14, 5))

            for ax, feat, color, title in [
                (axes[0], "nli_flat",  "#9E9E9E",
                 "Δ nli_flat\n(flat NLI — should be ~0)"),
                (axes[1], "min_hop",   "#2196F3",
                 "Δ min_hop\n(per-hop min — should be +)"),
                (axes[2], "nli_hop2",  "#4CAF50",
                 "Δ nli_hop2\n(hop2 score — should be +)"),
            ]:
                deltas = [p["delta"][feat] for p in chain_wins_pairs]
                ax.hist(deltas, bins=40, color=color, alpha=0.8, edgecolor="white")
                ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
                ax.axvline(np.mean(deltas), color="red", linestyle="-",
                           linewidth=2, label=f"mean={np.mean(deltas):+.3f}")
                ax.set_xlabel(f"Δ {feat} (correct − wrong)", fontsize=11)
                ax.set_ylabel("Count", fontsize=11)
                ax.set_title(title, fontsize=11)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

            fig.suptitle(
                f"B2 Fixed: Feature Deltas within CHAIN_WINS Questions — {label}\n"
                f"N={len(chain_wins_pairs)} questions where Z2 wrong, Z_full right",
                fontsize=12, fontweight="bold"
            )
            plt.tight_layout()
            plot_path = os.path.join(out_dir, f"b2_delta_{label}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Plot  → {plot_path}")
        except ImportError:
            print(f"  (matplotlib not available)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proj_root",
                    default="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot")
    ap.add_argument("--out_dir",   default="experiments/results/b2_fixed")
    ap.add_argument("--no_plots",  action="store_true")
    args = ap.parse_args()
    R = args.proj_root

    datasets = [
        {
            "label":    "hotpotqa_mdr",
            "hop":      f"{R}/exp0c/preds/dev_hop_scores.jsonl",
            "z2":       f"{R}/exp_phase0/results/z2_surface_preds.jsonl",
            "zf":       f"{R}/exp_phase0/results/z_full_preds.jsonl",
            "gold":     f"{R}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json",
        },
        {
            "label":    "hotpotqa_distractor",
            "hop":      f"{R}/exp_distractor/preds/dev_hop_scores.jsonl",
            "z2":       f"{R}/exp_distractor/results/z2_surface_preds.jsonl",
            "zf":       f"{R}/exp_distractor/results/z_full_preds.jsonl",
            "gold":     f"{R}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json",
        },
        {
            "label":    "wiki2",
            "hop":      f"{R}/exp_wiki2/preds/dev_hop_scores.jsonl",
            "z2":       f"{R}/exp_wiki2/results/z2_surface_preds.jsonl",
            "zf":       f"{R}/exp_wiki2/results/z_full_preds.jsonl",
            "gold":     f"{R}/data/wiki2/raw/dev_normalized.json",
        },
    ]

    print("\n" + "━"*70)
    print("  B2 FIXED — Paired Within-Question Delta Analysis")
    print("  Replaces the broken between-group imbalance comparison")
    print("━"*70)

    for d in datasets:
        run_b2(
            label    = d["label"],
            hop_path = d["hop"],
            z2_path  = d["z2"],
            zf_path  = d["zf"],
            gold_path= d["gold"],
            out_dir  = args.out_dir,
            no_plots = args.no_plots,
        )

    print(f"\n  All outputs → {args.out_dir}/")


if __name__ == "__main__":
    main()
    