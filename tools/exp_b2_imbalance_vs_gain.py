#!/usr/bin/env python3
"""
exp_b2_imbalance_vs_gain.py — B2: Mechanism Proof Figure

For each dataset, identifies CHAIN_WINS questions (Z2 wrong, Z_full correct)
and BOTH_RIGHT questions (both Z2 and Z_full correct), then plots the
distribution of hop imbalance |nli_hop1 - nli_hop2| of the candidate each
verifier picked.

The mechanism claim: CHAIN_WINS clusters at HIGH imbalance (flat scoring
fooled by one strong hop hiding a weak one); BOTH_RIGHT clusters at LOW
imbalance (no asymmetry to exploit, flat scoring fine).

Inputs (per dataset):
  --hop_scores      dev_hop_scores.jsonl  (must contain candidate.imbalance)
  --z2_preds        z2_surface_preds.jsonl
  --zfull_preds     z_full_preds.jsonl
  --gold            normalized dev json with {"_id", "answer"} or list of objects
  --type_field      optional: question-type field for per-type breakdown
  --label           short tag e.g. "wiki2" or "hotpot_distractor"
  --out_dir         where to write outputs

Outputs:
  imbalance_distributions_<label>.png    KDE plot of two distributions
  imbalance_stats_<label>.json           means, medians, p-values
  imbalance_per_type_<label>.json        per-type breakdown (if type_field given)
  chain_wins_examples_<label>.jsonl      241ish rows for B4 (failure cases)

Usage:
  python3 tools/exp_b2_imbalance_vs_gain.py \\
      --hop_scores  exp_wiki2/preds/dev_hop_scores.jsonl \\
      --z2_preds    exp_wiki2/results/z2_surface_preds.jsonl \\
      --zfull_preds exp_wiki2/results/z_full_preds.jsonl \\
      --gold        data/wiki2/raw/dev_normalized.json \\
      --type_field  type \\
      --label       wiki2 \\
      --out_dir     exp_wiki2/analysis/b2_imbalance_vs_gain
"""

import argparse
import json
import os
import re
import string
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# ─── text normalization (matches phase0_bootstrap.py) ───
def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def em(p: str, g: str) -> int:
    return int(normalize(p) == normalize(g))


def load_preds(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[str(r["qid"])] = {
                "pred":     r.get("pred", ""),
                "fallback": r.get("fallback", False),
            }
    return out


def load_gold(path: str, type_field: str | None) -> dict:
    """Returns {qid: {"answer": str, "type": str|None}}."""
    data = json.load(open(path))
    if not isinstance(data, list):
        # some HotpotQA dumps wrap in a key
        data = data.get("data") or list(data.values())[0]
    out = {}
    for ex in data:
        qid = str(ex.get("_id") or ex.get("id") or ex.get("qid"))
        out[qid] = {
            "answer": ex.get("answer", ""),
            "type":   (ex.get(type_field) if type_field else None),
        }
    return out


def load_hop_scores(path: str) -> dict:
    """Returns {qid: [candidate_dict, ...]}."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[str(r["qid"])] = r.get("candidates", [])
    return out


def find_picked_imbalance(candidates: list, picked_answer: str):
    """Return (imbalance, candidate_dict) for the candidate matching picked_answer.
    If multiple candidates share the same answer, take the first.
    Returns (None, None) if no match."""
    pa = normalize(picked_answer)
    for c in candidates:
        if normalize(c.get("answer", "")) == pa:
            return c.get("imbalance", None), c
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop_scores",  required=True)
    ap.add_argument("--z2_preds",    required=True)
    ap.add_argument("--zfull_preds", required=True)
    ap.add_argument("--gold",        required=True)
    ap.add_argument("--type_field",  default=None,
                    help="Optional type field name in gold JSON")
    ap.add_argument("--label",       required=True,
                    help="Short tag for output filenames, e.g. 'wiki2'")
    ap.add_argument("--out_dir",     required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n=== B2 Imbalance vs Gain — dataset: {args.label} ===\n")
    print("Loading inputs ...")
    gold      = load_gold(args.gold, args.type_field)
    hop       = load_hop_scores(args.hop_scores)
    z2_preds  = load_preds(args.z2_preds)
    zf_preds  = load_preds(args.zfull_preds)
    print(f"  gold:        {len(gold):,} questions")
    print(f"  hop_scores:  {len(hop):,} questions")
    print(f"  z2_preds:    {len(z2_preds):,} questions")
    print(f"  zfull_preds: {len(zf_preds):,} questions")

    common = set(gold) & set(hop) & set(z2_preds) & set(zf_preds)
    print(f"  intersection: {len(common):,} questions\n")

    # ── classify and collect imbalance ──
    chain_wins  = []   # (qid, type, imbalance, z2_pred, gold_ans)
    both_right  = []   # (qid, type, imbalance, picked, gold_ans)
    chain_hurts = []
    both_wrong  = []
    skipped_no_match = 0

    for qid in sorted(common):
        g_ans = gold[qid]["answer"]
        q_type = gold[qid]["type"]
        z2  = z2_preds[qid]
        zf  = zf_preds[qid]

        z2_correct = bool(em(z2["pred"], g_ans))
        zf_correct = bool(em(zf["pred"], g_ans))

        cands = hop[qid]
        if not cands:
            continue

        if not z2_correct and zf_correct:
            # CHAIN_WINS: record the imbalance of the wrong candidate Z2 picked
            imb, _ = find_picked_imbalance(cands, z2["pred"])
            if imb is None:
                skipped_no_match += 1
                continue
            chain_wins.append({
                "qid": qid, "type": q_type, "imbalance": float(imb),
                "z2_pred": z2["pred"], "zfull_pred": zf["pred"],
                "gold": g_ans,
            })
        elif z2_correct and not zf_correct:
            imb, _ = find_picked_imbalance(cands, zf["pred"])
            if imb is None:
                skipped_no_match += 1
                continue
            chain_hurts.append({
                "qid": qid, "type": q_type, "imbalance": float(imb),
            })
        elif z2_correct and zf_correct:
            # BOTH_RIGHT (control): imbalance of the (correct) candidate Z2 picked
            imb, _ = find_picked_imbalance(cands, z2["pred"])
            if imb is None:
                skipped_no_match += 1
                continue
            both_right.append({
                "qid": qid, "type": q_type, "imbalance": float(imb),
            })
        else:
            both_wrong.append({"qid": qid, "type": q_type})

    print(f"  CHAIN_WINS   : {len(chain_wins):>5}")
    print(f"  CHAIN_HURTS  : {len(chain_hurts):>5}")
    print(f"  BOTH_RIGHT   : {len(both_right):>5}")
    print(f"  BOTH_WRONG   : {len(both_wrong):>5}")
    print(f"  skipped (no answer-match): {skipped_no_match}")
    total = len(chain_wins) + len(chain_hurts) + len(both_right) + len(both_wrong)
    print(f"  total accounted: {total:,} / {len(common):,}\n")

    # ── arrays for plotting + stats ──
    imb_wins = np.array([r["imbalance"] for r in chain_wins])
    imb_both = np.array([r["imbalance"] for r in both_right])
    imb_hurts = np.array([r["imbalance"] for r in chain_hurts])

    def describe(name, arr):
        if len(arr) == 0:
            return {"name": name, "n": 0}
        return {
            "name":   name,
            "n":      int(len(arr)),
            "mean":   round(float(arr.mean()), 4),
            "std":    round(float(arr.std()),  4),
            "median": round(float(np.median(arr)), 4),
            "q25":    round(float(np.percentile(arr, 25)), 4),
            "q75":    round(float(np.percentile(arr, 75)), 4),
            "max":    round(float(arr.max()), 4),
        }

    desc_wins  = describe("CHAIN_WINS", imb_wins)
    desc_both  = describe("BOTH_RIGHT", imb_both)
    desc_hurts = describe("CHAIN_HURTS", imb_hurts)

    # Mann-Whitney U: is CHAIN_WINS imbalance stochastically larger than BOTH_RIGHT?
    if len(imb_wins) and len(imb_both):
        u, p_mwu = stats.mannwhitneyu(imb_wins, imb_both, alternative="greater")
        # Also a simple effect size: rank-biserial correlation
        n1, n2 = len(imb_wins), len(imb_both)
        rbc = 1 - (2 * u) / (n1 * n2)
        rbc = -rbc  # alternative='greater' => reverse sign convention
    else:
        u, p_mwu, rbc = None, None, None

    print("Distribution stats:")
    for d in [desc_wins, desc_both, desc_hurts]:
        if d["n"]:
            print(f"  {d['name']:12s} n={d['n']:>5}  mean={d['mean']:.3f}  "
                  f"median={d['median']:.3f}  q25–q75=[{d['q25']:.3f},{d['q75']:.3f}]")

    if p_mwu is not None:
        print(f"\nMann-Whitney U (CHAIN_WINS > BOTH_RIGHT):")
        print(f"  U = {u:.0f}   p = {p_mwu:.4g}   rank-biserial r = {rbc:+.3f}")

    # ── per-type breakdown ──
    per_type = {}
    if args.type_field:
        for group_name, rows in [("CHAIN_WINS", chain_wins),
                                 ("BOTH_RIGHT", both_right),
                                 ("CHAIN_HURTS", chain_hurts)]:
            buckets = defaultdict(list)
            for r in rows:
                t = r.get("type") or "unknown"
                buckets[t].append(r["imbalance"])
            per_type[group_name] = {
                t: describe(t, np.array(v)) for t, v in buckets.items()
            }

    # ── plot ──
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(0, max(0.01, max(
        imb_wins.max() if len(imb_wins) else 0,
        imb_both.max() if len(imb_both) else 0,
    )), 40)

    if len(imb_both):
        ax.hist(imb_both, bins=bins, density=True, alpha=0.45,
                color="#3b82f6", label=f"BOTH_RIGHT (n={len(imb_both)})",
                edgecolor="white", linewidth=0.4)
    if len(imb_wins):
        ax.hist(imb_wins, bins=bins, density=True, alpha=0.65,
                color="#ef4444", label=f"CHAIN_WINS (n={len(imb_wins)})",
                edgecolor="white", linewidth=0.4)

    # vertical lines at means
    if len(imb_both):
        ax.axvline(imb_both.mean(), color="#1e40af", linestyle="--",
                   linewidth=1.5, alpha=0.8)
    if len(imb_wins):
        ax.axvline(imb_wins.mean(), color="#991b1b", linestyle="--",
                   linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Hop imbalance  |nli_hop1 − nli_hop2|  of picked candidate",
                  fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    title = f"Imbalance vs Gain — {args.label}\n"
    if p_mwu is not None:
        title += (f"CHAIN_WINS mean={imb_wins.mean():.3f}  "
                  f"BOTH_RIGHT mean={imb_both.mean():.3f}  "
                  f"Mann-Whitney p={p_mwu:.2e}")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig_path = os.path.join(args.out_dir, f"imbalance_distributions_{args.label}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure → {fig_path}")

    # ── write outputs ──
    stats_out = {
        "label":       args.label,
        "n_questions": len(common),
        "groups": {
            "CHAIN_WINS":  desc_wins,
            "BOTH_RIGHT":  desc_both,
            "CHAIN_HURTS": desc_hurts,
            "BOTH_WRONG":  {"name": "BOTH_WRONG", "n": len(both_wrong)},
        },
        "mann_whitney_u_chain_wins_gt_both_right": {
            "U":              float(u) if u is not None else None,
            "p_value":        float(p_mwu) if p_mwu is not None else None,
            "rank_biserial":  float(rbc) if rbc is not None else None,
            "alternative":    "greater",
        },
        "skipped_no_answer_match": skipped_no_match,
    }
    stats_path = os.path.join(args.out_dir, f"imbalance_stats_{args.label}.json")
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"  Stats  → {stats_path}")

    if per_type:
        pt_path = os.path.join(args.out_dir, f"imbalance_per_type_{args.label}.json")
        with open(pt_path, "w") as f:
            json.dump(per_type, f, indent=2)
        print(f"  Types  → {pt_path}")

    # CHAIN_WINS examples — useful for B4 (failure case studies)
    ex_path = os.path.join(args.out_dir, f"chain_wins_examples_{args.label}.jsonl")
    with open(ex_path, "w") as f:
        for r in sorted(chain_wins, key=lambda x: -x["imbalance"]):  # most-imbalanced first
            f.write(json.dumps(r) + "\n")
    print(f"  CHAIN_WINS examples → {ex_path}")
    print(f"\nDone.\n")


if __name__ == "__main__":
    main()