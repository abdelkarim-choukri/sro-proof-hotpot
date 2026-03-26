#!/usr/bin/env python3
"""
exp3_compare_oracle.py — Per-question oracle comparison: exp1b (flat) vs exp3 (chain-aware)

Answers:
  1. Overall oracle@5 delta (EM + F1)
  2. Bridge vs comparison breakdown (expect bridge gains, comparison ~flat)
  3. Feasible subset breakdown
  4. Per-question win/loss/tie counts
  5. Candidate quality stats: bad rate, unique count, UNKNOWN rate

Usage:
    python3 tools/exp3_compare_oracle.py \
        --baseline_candidates  exp1b/candidates/dev_M5_candidates_qwen.jsonl \
        --exp3_candidates      exp3/candidates/dev_M5_candidates_chain_aware.jsonl \
        --baseline_oracle      exp1b/metrics/oracle_M5_dev_perqid.jsonl \
        --exp3_oracle          exp3/metrics/oracle_M5_dev_perqid.jsonl \
        --evidence             exp1b/evidence/dev_K100_chains.jsonl \
        --gold                 data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
        --out_json             exp3/metrics/exp3_vs_exp1b_comparison.json
"""

import argparse
import collections
import json
import os
import re
import string
import sys


# ─────────────────────────── text utils ─────────────────────────────

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))

def f1_score(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    common = collections.Counter(p_toks) & collections.Counter(g_toks)
    n = sum(common.values())
    if not n:
        return 0.0
    p = n / len(p_toks)
    r = n / len(g_toks)
    return 2 * p * r / (p + r)

def is_bad(ans: str) -> bool:
    a = ans.strip()
    if not a:
        return True
    low = a.lower()
    if low.startswith("[chain"):
        return True
    if "if the evidence does not contain" in low:
        return True
    if low.startswith("the evidence provided"):
        return True
    if low.startswith(("okay,", "alright,", "so,")):
        return True
    if low in {"unknown", "unk"}:
        return True
    if len(a) > 120:
        return True
    return False


# ─────────────────────────── loaders ────────────────────────────────

def load_oracle_perqid(path: str) -> dict:
    """Returns {qid: {"best_em": int, "best_f1": float}}"""
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[str(r["qid"])] = r
    return out

def load_candidates(path: str) -> dict:
    """Returns {qid: [answer_text, ...]}"""
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[str(r["qid"])] = [c["answer_text"] for c in r["candidates"]]
    return out


# ─────────────────────────── main ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_candidates", required=True)
    ap.add_argument("--exp3_candidates",     required=True)
    ap.add_argument("--baseline_oracle",     required=True)
    ap.add_argument("--exp3_oracle",         required=True)
    ap.add_argument("--evidence",            required=True)
    ap.add_argument("--gold",                required=True)
    ap.add_argument("--out_json",            required=True)
    args = ap.parse_args()

    # ── load gold + question types ──
    gold_map = {}
    qtype_map = {}
    for ex in json.load(open(args.gold)):
        qid = str(ex["_id"])
        gold_map[qid] = ex["answer"]
        qtype_map[qid] = ex.get("type", "bridge")

    # ── feasible subset ──
    feasible = set()
    for line in open(args.evidence):
        r = json.loads(line)
        if r.get("flags", {}).get("doc_recall_at_k", False):
            feasible.add(str(r["qid"]))

    # ── load oracle per-qid ──
    base_oracle = load_oracle_perqid(args.baseline_oracle)
    exp3_oracle = load_oracle_perqid(args.exp3_oracle)

    # ── load candidates for quality stats ──
    base_cands = load_candidates(args.baseline_candidates)
    exp3_cands = load_candidates(args.exp3_candidates)

    # ── align ──
    all_qids = sorted(set(base_oracle) & set(exp3_oracle) & set(gold_map))
    print(f"Aligned: {len(all_qids)} questions")

    # ── per-question comparison ──
    wins = losses = ties = 0
    bridge_wins = bridge_losses = bridge_ties = 0
    comp_wins = comp_losses = comp_ties = 0

    # Aggregate EM/F1 by split
    splits = {
        "overall":    lambda qid: True,
        "feasible":   lambda qid: qid in feasible,
        "bridge":     lambda qid: qtype_map.get(qid) == "bridge",
        "comparison": lambda qid: qtype_map.get(qid) == "comparison",
        "bridge_feasible": lambda qid: qid in feasible and qtype_map.get(qid) == "bridge",
        "comp_feasible":   lambda qid: qid in feasible and qtype_map.get(qid) == "comparison",
    }

    results = {}
    for split_name, pred_fn in splits.items():
        base_em = exp3_em = base_f1 = exp3_f1 = 0.0
        n = 0
        for qid in all_qids:
            if not pred_fn(qid):
                continue
            n += 1
            b = base_oracle[qid]
            e = exp3_oracle[qid]
            base_em += b["best_em"]
            exp3_em += e["best_em"]
            base_f1 += b["best_f1"]
            exp3_f1 += e["best_f1"]
        if n > 0:
            results[split_name] = {
                "n": n,
                "baseline_oracle_em": round(base_em / n, 4),
                "exp3_oracle_em":     round(exp3_em / n, 4),
                "delta_em":           round((exp3_em - base_em) / n, 4),
                "baseline_oracle_f1": round(base_f1 / n, 4),
                "exp3_oracle_f1":     round(exp3_f1 / n, 4),
                "delta_f1":           round((exp3_f1 - base_f1) / n, 4),
            }

    # Per-question win/loss/tie
    for qid in all_qids:
        b_em = base_oracle[qid]["best_em"]
        e_em = exp3_oracle[qid]["best_em"]
        qtype = qtype_map.get(qid, "bridge")

        if e_em > b_em:
            wins += 1
            if qtype == "bridge":
                bridge_wins += 1
            else:
                comp_wins += 1
        elif e_em < b_em:
            losses += 1
            if qtype == "bridge":
                bridge_losses += 1
            else:
                comp_losses += 1
        else:
            ties += 1
            if qtype == "bridge":
                bridge_ties += 1
            else:
                comp_ties += 1

    # ── candidate quality stats ──
    def quality_stats(cands_map):
        total = bad = unknown = empty = 0
        unique_counts = []
        for qid, cands in cands_map.items():
            unique_counts.append(len(set(cands)))
            for a in cands:
                total += 1
                if is_bad(a):
                    bad += 1
                if normalize(a) in {"unknown", "unk"}:
                    unknown += 1
                if not a.strip():
                    empty += 1
        return {
            "total_candidates": total,
            "bad_rate":     round(bad / max(total, 1), 4),
            "unknown_rate": round(unknown / max(total, 1), 4),
            "empty_rate":   round(empty / max(total, 1), 4),
            "mean_unique":  round(sum(unique_counts) / max(len(unique_counts), 1), 2),
        }

    base_quality = quality_stats(base_cands)
    exp3_quality = quality_stats(exp3_cands)

    # ── assemble ──
    summary = {
        "description": "Exp3 (chain-aware prompt v3) vs Exp1b (flat prompt v2)",
        "n_aligned": len(all_qids),
        "splits": results,
        "per_question_wins_losses": {
            "overall": {"wins": wins, "losses": losses, "ties": ties,
                        "net_gain": wins - losses},
            "bridge":  {"wins": bridge_wins, "losses": bridge_losses,
                        "ties": bridge_ties, "net_gain": bridge_wins - bridge_losses},
            "comparison": {"wins": comp_wins, "losses": comp_losses,
                           "ties": comp_ties, "net_gain": comp_wins - comp_losses},
        },
        "candidate_quality": {
            "baseline_flat":  base_quality,
            "exp3_chain_aware": exp3_quality,
        },
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # ── print ──
    W = 72
    print()
    print("=" * W)
    print("  Exp3 vs Exp1b — Oracle@5 Comparison")
    print("=" * W)
    print(f"  {'Split':<22} {'Baseline EM':>12} {'Exp3 EM':>10} {'Delta':>10}")
    print("  " + "-" * (W - 2))
    for split_name in ["overall", "feasible", "bridge", "comparison",
                       "bridge_feasible", "comp_feasible"]:
        if split_name in results:
            r = results[split_name]
            d = r["delta_em"]
            marker = " ★" if d >= 0.015 else (" ✗" if d < -0.01 else "")
            print(f"  {split_name:<22} {r['baseline_oracle_em']:>12.4f} "
                  f"{r['exp3_oracle_em']:>10.4f} {d:>+10.4f}{marker}")
    print("=" * W)

    print(f"\n  Per-question oracle EM changes:")
    print(f"    Overall:    +{wins} wins  -{losses} losses  ={ties} ties  "
          f"(net: {wins - losses:+d})")
    print(f"    Bridge:     +{bridge_wins} wins  -{bridge_losses} losses  "
          f"={bridge_ties} ties  (net: {bridge_wins - bridge_losses:+d})")
    print(f"    Comparison: +{comp_wins} wins  -{comp_losses} losses  "
          f"={comp_ties} ties  (net: {comp_wins - comp_losses:+d})")

    print(f"\n  Candidate quality (bad rate / unknown rate / mean unique):")
    print(f"    Baseline:  bad={base_quality['bad_rate']:.1%}  "
          f"unk={base_quality['unknown_rate']:.1%}  "
          f"unique={base_quality['mean_unique']:.1f}")
    print(f"    Exp3:      bad={exp3_quality['bad_rate']:.1%}  "
          f"unk={exp3_quality['unknown_rate']:.1%}  "
          f"unique={exp3_quality['mean_unique']:.1f}")

    print()
    print(f"  Results saved to: {args.out_json}")
    print("=" * W)


if __name__ == "__main__":
    main()