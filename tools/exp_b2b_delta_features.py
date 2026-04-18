#!/usr/bin/env python3
"""
exp_b2b_delta_features.py — B2 Option B: Comparative Mechanism Proof

For each CHAIN_WINS question (Z2 wrong, Z_full correct), compute delta
features between the two competing candidates:
    Δ = feature(Z_full's pick = correct) − feature(Z2's pick = wrong)

Hypothesis (Option B): chain features distinguish the right candidate from
the wrong one even when flat scoring cannot. Predicts:
    Δ_flat       ≈ 0       (flat indifferent — that's why Z2 picked wrong)
    Δ_min_hop    > 0       (chain sees right is stronger on bottleneck hop)
    Δ_mean_hop   > 0       (chain sees right is stronger on average)

Control group: CHAIN_HURTS (Z2 right, Z_full wrong) — should mirror.

Inputs (per dataset):
  --hop_scores      dev_hop_scores.jsonl (with candidates + nli features)
  --z2_preds        z2_surface_preds.jsonl
  --zfull_preds     z_full_preds.jsonl
  --gold            normalized dev json
  --type_field      optional question-type field for per-type breakdown
  --label           short tag e.g. "wiki2"
  --out_dir         output directory

Outputs:
  delta_distributions_<label>.png    two-panel figure
  delta_stats_<label>.json           all numbers
  delta_per_type_<label>.json        per-type breakdown
  chain_wins_with_deltas_<label>.jsonl   enriched CHAIN_WINS for B4
"""

import argparse
import json
import os
import re
import string
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# ─── text normalization ─────────────────────────────────────────
def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def em(p: str, g: str) -> int:
    return int(normalize(p) == normalize(g))


# ─── loaders ────────────────────────────────────────────────────
def load_preds(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            out[str(r["qid"])] = r.get("pred", "")
    return out


def load_gold(path: str, type_field):
    data = json.load(open(path))
    if not isinstance(data, list):
        data = data.get("data") or list(data.values())[0]
    out = {}
    for ex in data:
        qid = str(ex.get("_id") or ex.get("id") or ex.get("qid"))
        out[qid] = {
            "answer": ex.get("answer", ""),
            "type":   ex.get(type_field) if type_field else None,
        }
    return out


def load_hop(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            out[str(r["qid"])] = r.get("candidates", [])
    return out


def find_cand(cands: list, answer: str):
    """Return first candidate whose answer matches `answer` (normalized).
    None if no match."""
    a = normalize(answer)
    for c in cands:
        if normalize(c.get("answer", "")) == a:
            return c
    return None


# ─── delta feature names (matched to dev_hop_scores schema) ─────
FEATURES = ["nli_flat", "nli_hop1", "nli_hop2", "min_hop", "mean_hop", "imbalance"]


def compute_deltas(right_cand: dict, wrong_cand: dict) -> dict:
    """Δ = right − wrong for each feature."""
    return {f"delta_{k}": float(right_cand[k]) - float(wrong_cand[k])
            for k in FEATURES}


# ─── stats helpers ──────────────────────────────────────────────
def describe_delta(name: str, arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {"name": name, "n": 0}
    # Wilcoxon signed-rank: H0 median(Δ) = 0
    try:
        w_stat, p_two = stats.wilcoxon(arr, alternative="two-sided")
        _,      p_gt  = stats.wilcoxon(arr, alternative="greater")
        _,      p_lt  = stats.wilcoxon(arr, alternative="less")
    except ValueError:
        w_stat = p_two = p_gt = p_lt = None
    return {
        "name":   name,
        "n":      int(len(arr)),
        "mean":   round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "std":    round(float(arr.std()), 4),
        "frac_positive": round(float((arr > 0).mean()), 4),
        "wilcoxon_W":            float(w_stat) if w_stat is not None else None,
        "p_two_sided":           float(p_two)  if p_two  is not None else None,
        "p_greater_than_zero":   float(p_gt)   if p_gt   is not None else None,
        "p_less_than_zero":      float(p_lt)   if p_lt   is not None else None,
    }


# ─── main ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop_scores",  required=True)
    ap.add_argument("--z2_preds",    required=True)
    ap.add_argument("--zfull_preds", required=True)
    ap.add_argument("--gold",        required=True)
    ap.add_argument("--type_field",  default=None)
    ap.add_argument("--label",       required=True)
    ap.add_argument("--out_dir",     required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n=== B2 Option B — delta features — dataset: {args.label} ===\n")
    gold = load_gold(args.gold, args.type_field)
    hop  = load_hop(args.hop_scores)
    z2   = load_preds(args.z2_preds)
    zf   = load_preds(args.zfull_preds)
    common = set(gold) & set(hop) & set(z2) & set(zf)
    print(f"  intersection: {len(common):,} questions\n")

    # ── classify and compute deltas ──
    chain_wins  = []   # rows with deltas
    chain_hurts = []
    n_skip_no_match = 0
    n_skip_same_pick = 0   # cases where Z2 and Z_full picked same answer (irrelevant)

    for qid in sorted(common):
        g_ans = gold[qid]["answer"]
        q_type = gold[qid]["type"]
        z2_pred, zf_pred = z2[qid], zf[qid]
        z2_ok = bool(em(z2_pred, g_ans))
        zf_ok = bool(em(zf_pred, g_ans))
        cands = hop[qid]
        if not cands:
            continue
        if normalize(z2_pred) == normalize(zf_pred):
            # both verifiers picked the same candidate — no contrast to study
            n_skip_same_pick += 1
            continue

        if not z2_ok and zf_ok:
            wrong = find_cand(cands, z2_pred)
            right = find_cand(cands, zf_pred)
            if wrong is None or right is None:
                n_skip_no_match += 1
                continue
            row = {
                "qid": qid, "type": q_type,
                "wrong_pick": z2_pred, "right_pick": zf_pred, "gold": g_ans,
                **compute_deltas(right, wrong),
                # also keep raw for B4 sanity later
                "wrong_features": {k: float(wrong[k]) for k in FEATURES},
                "right_features": {k: float(right[k]) for k in FEATURES},
            }
            chain_wins.append(row)
        elif z2_ok and not zf_ok:
            # mirror direction: Z2 was right, Z_full was wrong
            # Δ = right − wrong = z2_pick − zfull_pick (so chain HURTS = negative Δs expected)
            right = find_cand(cands, z2_pred)
            wrong = find_cand(cands, zf_pred)
            if wrong is None or right is None:
                n_skip_no_match += 1
                continue
            row = {
                "qid": qid, "type": q_type,
                **compute_deltas(right, wrong),
            }
            chain_hurts.append(row)

    print(f"  CHAIN_WINS  : {len(chain_wins):>5}")
    print(f"  CHAIN_HURTS : {len(chain_hurts):>5}")
    print(f"  skipped (same pick by both): {n_skip_same_pick:,}")
    print(f"  skipped (answer not in hop_scores): {n_skip_no_match}\n")

    # ── distribution stats per Δ feature ──
    def collect(rows, key):
        return np.array([r[key] for r in rows]) if rows else np.array([])

    stats_wins  = {f: describe_delta(f, collect(chain_wins, f"delta_{f}"))
                   for f in FEATURES}
    stats_hurts = {f: describe_delta(f, collect(chain_hurts, f"delta_{f}"))
                   for f in FEATURES}

    print("CHAIN_WINS  (Δ = right − wrong, expect chain features > 0, flat ≈ 0):")
    for f in FEATURES:
        d = stats_wins[f]
        if d["n"]:
            print(f"  Δ {f:10s}  mean={d['mean']:+.3f}  median={d['median']:+.3f}  "
                  f"frac>0={d['frac_positive']*100:.1f}%  "
                  f"p(Δ>0)={d['p_greater_than_zero']:.3g}")

    print("\nCHAIN_HURTS (Δ = right − wrong, expect chain features < 0):")
    for f in FEATURES:
        d = stats_hurts[f]
        if d["n"]:
            print(f"  Δ {f:10s}  mean={d['mean']:+.3f}  median={d['median']:+.3f}  "
                  f"frac>0={d['frac_positive']*100:.1f}%  "
                  f"p(Δ<0)={d['p_less_than_zero']:.3g}")

    # ── per-type breakdown ──
    per_type = {}
    if args.type_field:
        for group_name, rows in [("CHAIN_WINS", chain_wins),
                                 ("CHAIN_HURTS", chain_hurts)]:
            buckets = defaultdict(list)
            for r in rows:
                buckets[r.get("type") or "unknown"].append(r)
            per_type[group_name] = {}
            for t, group_rows in buckets.items():
                per_type[group_name][t] = {
                    "n": len(group_rows),
                    "delta_features": {
                        f: describe_delta(f, collect(group_rows, f"delta_{f}"))
                        for f in FEATURES
                    },
                }

        # Print compact per-type summary for the two key features
        for group_name in ["CHAIN_WINS", "CHAIN_HURTS"]:
            print(f"\nPer-type summary — {group_name} (Δ_min_hop, Δ_flat):")
            for t, d in sorted(per_type[group_name].items(),
                               key=lambda x: -x[1]["n"]):
                mh = d["delta_features"]["min_hop"]
                fl = d["delta_features"]["nli_flat"]
                print(f"  {t:20s}  n={mh['n']:>4}  "
                      f"Δmin_hop mean={mh['mean']:+.3f}  "
                      f"Δflat mean={fl['mean']:+.3f}  "
                      f"frac(Δmin>0)={mh['frac_positive']*100:.0f}%")

    # ── figure: two panels ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1 — bar chart of mean Δ per feature, CHAIN_WINS vs CHAIN_HURTS
    x = np.arange(len(FEATURES))
    width = 0.38
    means_w = [stats_wins[f]["mean"]  if stats_wins[f]["n"]  else 0 for f in FEATURES]
    means_h = [stats_hurts[f]["mean"] if stats_hurts[f]["n"] else 0 for f in FEATURES]
    err_w   = [stats_wins[f].get("std", 0) / np.sqrt(max(stats_wins[f]["n"], 1))
               if stats_wins[f]["n"] else 0 for f in FEATURES]
    err_h   = [stats_hurts[f].get("std", 0) / np.sqrt(max(stats_hurts[f]["n"], 1))
               if stats_hurts[f]["n"] else 0 for f in FEATURES]

    bw = ax1.bar(x - width/2, means_w, width, yerr=err_w, capsize=3,
                 color="#22c55e", alpha=0.85, label=f"CHAIN_WINS (n={len(chain_wins)})",
                 edgecolor="white", linewidth=0.8)
    bh = ax1.bar(x + width/2, means_h, width, yerr=err_h, capsize=3,
                 color="#ef4444", alpha=0.85, label=f"CHAIN_HURTS (n={len(chain_hurts)})",
                 edgecolor="white", linewidth=0.8)
    ax1.axhline(0, color="black", linewidth=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.replace("nli_", "") for f in FEATURES],
                        rotation=20, ha="right")
    ax1.set_ylabel("Mean Δ  (right − wrong),  ± SEM", fontsize=11)
    ax1.set_title(f"Mean delta features — {args.label}\n"
                  "If chain mechanism holds: chain features positive in WINS, "
                  "negative in HURTS;  flat ≈ 0", fontsize=10)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax1.grid(True, axis="y", alpha=0.25, linestyle=":")
    for sp in ("top", "right"): ax1.spines[sp].set_visible(False)

    # Panel 2 — scatter Δ_flat vs Δ_min_hop for CHAIN_WINS
    if chain_wins:
        dflat = np.array([r["delta_nli_flat"] for r in chain_wins])
        dmin  = np.array([r["delta_min_hop"]  for r in chain_wins])
        ax2.scatter(dflat, dmin, s=30, alpha=0.55, c="#22c55e",
                    edgecolor="#166534", linewidth=0.4, label="CHAIN_WINS")
    if chain_hurts:
        dflat_h = np.array([r["delta_nli_flat"] for r in chain_hurts])
        dmin_h  = np.array([r["delta_min_hop"]  for r in chain_hurts])
        ax2.scatter(dflat_h, dmin_h, s=30, alpha=0.55, c="#ef4444",
                    edgecolor="#7f1d1d", linewidth=0.4, label="CHAIN_HURTS")
    ax2.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax2.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax2.set_xlabel("Δ nli_flat   (right − wrong)", fontsize=11)
    ax2.set_ylabel("Δ min_hop    (right − wrong)", fontsize=11)
    ax2.set_title(f"Per-question deltas — {args.label}\n"
                  "Upper-left quadrant = chain saw what flat missed",
                  fontsize=10)
    ax2.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.25, linestyle=":")
    for sp in ("top", "right"): ax2.spines[sp].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f"delta_distributions_{args.label}.png")
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure → {fig_path}")

    # ── write outputs ──
    out_stats = {
        "label":           args.label,
        "n_chain_wins":    len(chain_wins),
        "n_chain_hurts":   len(chain_hurts),
        "n_skip_same_pick": n_skip_same_pick,
        "n_skip_no_match":  n_skip_no_match,
        "delta_features_chain_wins":  stats_wins,
        "delta_features_chain_hurts": stats_hurts,
    }
    s_path = os.path.join(args.out_dir, f"delta_stats_{args.label}.json")
    with open(s_path, "w") as f:
        json.dump(out_stats, f, indent=2)
    print(f"  Stats  → {s_path}")

    if per_type:
        pt_path = os.path.join(args.out_dir, f"delta_per_type_{args.label}.json")
        with open(pt_path, "w") as f:
            json.dump(per_type, f, indent=2)
        print(f"  Types  → {pt_path}")

    # CHAIN_WINS examples enriched with deltas — useful for B4
    ex_path = os.path.join(args.out_dir, f"chain_wins_with_deltas_{args.label}.jsonl")
    with open(ex_path, "w") as f:
        # sort by largest |Δ_min_hop| — most striking cases first
        for r in sorted(chain_wins, key=lambda x: -abs(x["delta_min_hop"])):
            f.write(json.dumps(r) + "\n")
    print(f"  CHAIN_WINS w/ deltas → {ex_path}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()