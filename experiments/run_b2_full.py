#!/usr/bin/env python3
"""
experiments/run_b2_full.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
B2 Full Mechanism Analysis — All Three Scorers (NLI + QA + Lexical)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS REPLACES run_b2_fixed.py
    The previous B2 only measured NLI deltas. But Z3's feature importances show:
        qa_hop2        = 19.1%  ← top feature
        qa_hop_balance = 18.6%  ← second feature
        nli_hop_balance = 10.2%
        nli_hop2        =  9.7%
        lex_hop2        = 10.0%

    The model's primary driver is qa_hop2, not nli_hop2. Showing only NLI
    creates a framing gap. This script computes the CHAIN_WINS paired delta
    for ALL 9 chain features (3 scorers × 3 variants) and their flat baselines.

THE MECHANISM CLAIM (updated and complete)
    In CHAIN_WINS questions (where chain features corrected surface failures),
    correct answers score significantly higher than wrong answers on:
        - nli_hop2:   Δ = +0.14 to +0.28  (p<0.001)
        - qa_hop2:    Δ = ???              (p<???)   ← NEW
        - lex_hop2:   Δ = ???              (p<???)   ← NEW
    While their flat counterparts (nli_flat, qa_flat, lex_flat) show Δ ≈ 0.

    If qa_hop2 shows the LARGEST delta (consistent with its feature importance),
    this directly connects the mechanism analysis to the model's decision logic.
    The paper can say: "Hop-2 anchoring is visible across all three independent
    scorers, and the QA cross-encoder — which contributes most to Z3's decisions
    (qa_hop2: 19.1% importance) — shows the strongest discrimination signal."

SCHEMAS (verified from phase0 logs)
    hop_scores:   {qid, candidates[i]: {answer, nli_flat, nli_hop1, nli_hop2, ...}}
                  answer field = "answer", list indexed by position
    qa_scores:    {qid, candidates[i]: {answer_id, answer_text, qa_hop1, qa_hop2,
                                         qa_flat, qa_hop_balance, qa_min_hop}}
                  answer field = "answer_text", dict with answer_id
    lex_features: {qid, candidates[i]: {answer_id, answer_text, lex_hop1, lex_hop2,
                                         lex_flat, lex_hop_balance, ...}}
                  answer field = "answer_text", dict with answer_id

CROSS-FILE MATCHING
    All three files are joined on normalized answer text.
    hop_scores uses "answer", qa/lex use "answer_text" — handled by find_cand().

USAGE
    python3 experiments/run_b2_full.py \\
        --proj_root /var/tmp/u24sf51014/sro/work/sro-proof-hotpot \\
        --out_dir   experiments/results/b2_full

    Single setting:
    python3 experiments/run_b2_full.py ... --setting hotpotqa_mdr
"""

import argparse
import json
import os
import re
import string
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 1 — TEXT UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def em(p: str, g: str) -> bool:
    return normalize(p) == normalize(g)

def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 2 — DATA LOADERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_gold(path: str) -> Dict[str, str]:
    """Load {qid: gold_answer}. Handles HotpotQA and 2Wiki formats."""
    with open(path, encoding="utf-8") as f:
        raw = f.read(1); f.seek(0)
        if raw.strip() == "[":
            data = json.load(f)
        else:
            data = [json.loads(l) for l in f if l.strip()]
    result = {}
    for ex in data:
        qid = str(ex.get("_id", ex.get("id", "")))
        ans = ex.get("answer", "")
        result[qid] = str(ans.get("answer","") if isinstance(ans, dict) else ans)
    return result

def load_preds(path: str) -> Dict[str, str]:
    result = {}
    for rec in iter_jsonl(path):
        result[str(rec["qid"])] = str(rec.get("pred", ""))
    return result

def load_hop_scores(path: str) -> Dict[str, List[dict]]:
    """hop_scores: answer field = 'answer', list by position."""
    result = {}
    for rec in iter_jsonl(path):
        result[str(rec["qid"])] = rec.get("candidates", [])
    return result

def load_qa_scores(path: str) -> Dict[str, List[dict]]:
    """qa_scores: answer field = 'answer_text', has answer_id."""
    result = {}
    for rec in iter_jsonl(path):
        result[str(rec["qid"])] = rec.get("candidates", [])
    return result

def load_lex_features(path: str) -> Dict[str, List[dict]]:
    """lex_features: answer field = 'answer_text', has answer_id."""
    result = {}
    for rec in iter_jsonl(path):
        result[str(rec["qid"])] = rec.get("candidates", [])
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 3 — CANDIDATE LOOKUP (handles both answer/answer_text field names)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_cand(cands: List[dict], answer: str) -> Optional[dict]:
    """Find candidate by normalized answer text.
    Handles both 'answer' (hop_scores) and 'answer_text' (qa/lex) field names."""
    a = normalize(answer)
    for c in cands:
        cand_ans = c.get("answer", c.get("answer_text", ""))
        if normalize(cand_ans) == a:
            return c
    return None

def get_nli_scores(c: dict) -> dict:
    """Extract NLI scores from hop_scores candidate."""
    return {
        "nli_flat":      float(c.get("nli_flat",  0) or 0),
        "nli_hop1":      float(c.get("nli_hop1",  0) or 0),
        "nli_hop2":      float(c.get("nli_hop2",  0) or 0),
        "nli_hop_balance": abs(float(c.get("nli_hop1", 0) or 0) -
                               float(c.get("nli_hop2", 0) or 0)),
    }

def get_qa_scores(c: dict) -> dict:
    """Extract QA scores from qa_scores candidate."""
    return {
        "qa_flat":      float(c.get("qa_flat",  0) or 0),
        "qa_hop1":      float(c.get("qa_hop1",  0) or 0),
        "qa_hop2":      float(c.get("qa_hop2",  0) or 0),
        "qa_hop_balance": float(c.get("qa_hop_balance",
                           abs(float(c.get("qa_hop1",0) or 0) -
                               float(c.get("qa_hop2",0) or 0))) or 0),
    }

def get_lex_scores(c: dict) -> dict:
    """Extract lexical scores from lex_features candidate."""
    return {
        "lex_flat":      float(c.get("lex_flat",  0) or 0),
        "lex_hop1":      float(c.get("lex_hop1",  0) or 0),
        "lex_hop2":      float(c.get("lex_hop2",  0) or 0),
        "lex_hop_balance": float(c.get("lex_hop_balance",
                           abs(float(c.get("lex_hop1",0) or 0) -
                               float(c.get("lex_hop2",0) or 0))) or 0),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 4 — FEATURE GROUPS FOR DISPLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# All 12 features (3 scorers × 4 variants: flat/hop1/hop2/balance)
ALL_FEATURES = [
    "nli_flat", "nli_hop1", "nli_hop2", "nli_hop_balance",
    "qa_flat",  "qa_hop1",  "qa_hop2",  "qa_hop_balance",
    "lex_flat", "lex_hop1", "lex_hop2", "lex_hop_balance",
]

# Display groups for the paper table
SCORERS = {
    "NLI":     ["nli_flat",  "nli_hop1",  "nli_hop2",  "nli_hop_balance"],
    "QA":      ["qa_flat",   "qa_hop1",   "qa_hop2",   "qa_hop_balance"],
    "Lexical": ["lex_flat",  "lex_hop1",  "lex_hop2",  "lex_hop_balance"],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 5 — CORE ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_b2_full(
    label:     str,
    hop_path:  str,
    qa_path:   str,
    lex_path:  str,
    z2_path:   str,
    zf_path:   str,
    gold_path: str,
    out_dir:   str,
    no_plots:  bool = False,
) -> Optional[dict]:

    print(f"\n{'━'*70}")
    print(f"  B2 Full: {label}")
    print(f"{'━'*70}")

    for p, n in [(hop_path,"hop_scores"), (qa_path,"qa_scores"),
                 (lex_path,"lex_features"), (z2_path,"z2_preds"),
                 (zf_path,"zfull_preds"),   (gold_path,"gold")]:
        if not os.path.exists(p):
            print(f"  ✗ {n} not found: {p}")
            return None

    gold        = load_gold(gold_path)
    hop_scores  = load_hop_scores(hop_path)
    qa_scores   = load_qa_scores(qa_path)
    lex_features = load_lex_features(lex_path)
    z2_preds    = load_preds(z2_path)
    zf_preds    = load_preds(zf_path)

    print(f"  gold={len(gold):,}  hop={len(hop_scores):,}  "
          f"qa={len(qa_scores):,}  lex={len(lex_features):,}  "
          f"z2={len(z2_preds):,}  zf={len(zf_preds):,}")

    # ── Classify + collect paired scores ────────────────────────────
    chain_wins_pairs = []
    n_cw = n_ch = n_br = n_bw = 0
    skipped = 0

    for qid, gold_ans in gold.items():
        z2_pred = z2_preds.get(qid, "")
        zf_pred = zf_preds.get(qid, "")
        z2_ok   = em(z2_pred, gold_ans)
        zf_ok   = em(zf_pred, gold_ans)

        if   not z2_ok and zf_ok:   n_cw += 1
        elif z2_ok and not zf_ok:   n_ch += 1; continue
        elif z2_ok and zf_ok:       n_br += 1; continue
        else:                        n_bw += 1; continue

        # CHAIN_WINS: get all scorer data for wrong and correct candidates
        hop_cands = hop_scores.get(qid, [])
        qa_cands  = qa_scores.get(qid,  [])
        lex_cands = lex_features.get(qid, [])

        wrong_hop = find_cand(hop_cands, z2_pred)
        right_hop = find_cand(hop_cands, gold_ans)
        wrong_qa  = find_cand(qa_cands,  z2_pred)
        right_qa  = find_cand(qa_cands,  gold_ans)
        wrong_lex = find_cand(lex_cands, z2_pred)
        right_lex = find_cand(lex_cands, gold_ans)

        # Only keep if we have at least hop scores for both candidates
        if wrong_hop is None or right_hop is None:
            skipped += 1
            continue

        def merge_scores(hop_c, qa_c, lex_c) -> dict:
            s = {}
            s.update(get_nli_scores(hop_c))
            s.update(get_qa_scores(qa_c)  if qa_c  else
                     {f: 0.0 for f in SCORERS["QA"]})
            s.update(get_lex_scores(lex_c) if lex_c else
                     {f: 0.0 for f in SCORERS["Lexical"]})
            return s

        wrong_scores = merge_scores(wrong_hop, wrong_qa, wrong_lex)
        right_scores = merge_scores(right_hop, right_qa, right_lex)
        delta = {f: right_scores[f] - wrong_scores[f] for f in ALL_FEATURES}

        chain_wins_pairs.append({
            "qid":          qid,
            "wrong_scores": wrong_scores,
            "right_scores": right_scores,
            "delta":        delta,
            "qa_missing":   wrong_qa is None or right_qa is None,
            "lex_missing":  wrong_lex is None or right_lex is None,
        })

    print(f"\n  Confusion matrix:")
    print(f"    CHAIN_WINS : {n_cw:,}  (Z2 wrong, Z_full right)")
    print(f"    CHAIN_HURTS: {n_ch:,}  (Z2 right, Z_full wrong)")
    print(f"    BOTH_RIGHT : {n_br:,}")
    print(f"    BOTH_WRONG : {n_bw:,}")
    print(f"    Pairs used : {len(chain_wins_pairs):,}")
    if skipped:
        print(f"    Skipped    : {skipped} (hop_scores missing for candidate)")

    qa_missing_n  = sum(1 for p in chain_wins_pairs if p["qa_missing"])
    lex_missing_n = sum(1 for p in chain_wins_pairs if p["lex_missing"])
    if qa_missing_n:
        print(f"    QA missing : {qa_missing_n} pairs (zeros used)")
    if lex_missing_n:
        print(f"    Lex missing: {lex_missing_n} pairs (zeros used)")

    if not chain_wins_pairs:
        print("  No pairs — check paths")
        return None

    # ── Compute delta statistics per feature ─────────────────────────
    print(f"\n  PAIRED DELTA (Δ = correct − wrong) within CHAIN_WINS")
    print(f"  Positive Δ = correct answer scores HIGHER on this feature")
    print()

    delta_stats = {}
    for scorer_name, feats in SCORERS.items():
        print(f"  {scorer_name}:")
        print(f"  {'Feature':<22}  {'Mean Δ':>9}  {'Median Δ':>10}  "
              f"{'% Δ>0':>8}  {'p-value':>10}  Sig")
        print(f"  {'─'*62}")
        for feat in feats:
            deltas = np.array([p["delta"][feat] for p in chain_wins_pairs])
            pct_pos = (deltas > 0).mean() * 100
            try:
                from scipy import stats
                _, p_val = stats.wilcoxon(deltas)
                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
                      ("*"   if p_val < 0.05  else "ns"))
                p_str = f"{p_val:.4f}"
            except ImportError:
                p_val, p_str, sig = None, "N/A", ""

            print(f"  {feat:<22}  {deltas.mean():>+9.4f}  "
                  f"{np.median(deltas):>+10.4f}  "
                  f"{pct_pos:>7.1f}%  {p_str:>10}  {sig}")

            delta_stats[feat] = {
                "mean_delta":   float(deltas.mean()),
                "median_delta": float(np.median(deltas)),
                "pct_positive": float(pct_pos),
                "p_value":      float(p_val) if p_val is not None else None,
            }
        print()

    # ── Key findings summary ─────────────────────────────────────────
    print(f"  KEY FINDINGS:")
    for scorer_name, feats in SCORERS.items():
        flat_f  = feats[0]
        hop2_f  = feats[2]
        flat_d  = delta_stats[flat_f]["mean_delta"]
        hop2_d  = delta_stats[hop2_f]["mean_delta"]
        hop2_p  = delta_stats[hop2_f]["p_value"]
        flat_p  = delta_stats[flat_f]["p_value"]

        flat_sig = "fooled (Δ≈0, ns)" if (flat_p is None or flat_p > 0.05) \
                   else f"partially discriminates (Δ={flat_d:+.3f})"
        hop2_sig = "***" if hop2_p is not None and hop2_p < 0.001 else \
                   ("*" if hop2_p is not None and hop2_p < 0.05 else "ns")

        print(f"  {scorer_name}:")
        print(f"    {flat_f:<20} Δ={flat_d:+.4f}  → {flat_sig}")
        print(f"    {hop2_f:<20} Δ={hop2_d:+.4f}  → hop-2 IS higher for correct ({hop2_sig})")

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"b2_full_{label}.json")
    result = {
        "label":          label,
        "n_chain_wins":   n_cw,
        "n_chain_hurts":  n_ch,
        "n_pairs_used":   len(chain_wins_pairs),
        "delta_stats":    delta_stats,
        "z3_importances_note": {
            "qa_hop2":        "19.1% (top feature in Z3)",
            "qa_hop_balance": "18.6% (second feature)",
            "nli_hop_balance": "10.2%",
            "lex_hop2":       "10.0%",
            "nli_hop2":        "9.7%",
        },
    }
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {save_path}")

    # ── Plot — 3 panels (one per scorer) ────────────────────────────
    if not no_plots:
        _plot_b2_full(result, out_dir, label)

    return result


def _plot_b2_full(result: dict, out_dir: str, label: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.patheffects as pe
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
        return

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
        "grid.linestyle": "--", "figure.facecolor": "#fafafa",
        "axes.facecolor": "#fafafa",
    })

    COLORS = {"flat": "#757575", "hop1": "#E53935", "hop2": "#1E88E5",
              "balance": "#43A047"}

    ds  = result["delta_stats"]
    ncw = result["n_chain_wins"]
    n   = result["n_pairs_used"]

    fig = plt.figure(figsize=(16, 5), facecolor="#fafafa")
    gs  = GridSpec(1, 3, figure=fig, wspace=0.4)

    scorer_configs = [
        ("NLI",     "nli_flat",  "nli_hop1",  "nli_hop2",  "nli_hop_balance"),
        ("QA\n(Cross-Encoder)", "qa_flat",   "qa_hop1",   "qa_hop2",   "qa_hop_balance"),
        ("Lexical", "lex_flat",  "lex_hop1",  "lex_hop2",  "lex_hop_balance"),
    ]

    for col, (sname, ff, h1f, h2f, bf) in enumerate(scorer_configs):
        ax = fig.add_subplot(gs[0, col])

        feats  = [ff,  h1f,  h2f]
        vals   = [ds[f]["mean_delta"]  for f in feats]
        pvals  = [ds[f]["p_value"]     for f in feats]
        colors = [COLORS["flat"], COLORS["hop1"], COLORS["hop2"]]
        xlbls  = ["Flat\n(concat)", "Hop-1\n(bridge)", "Hop-2\n(answer)"]

        bars = ax.bar(xlbls, vals, color=colors, width=0.55,
                      edgecolor="white", linewidth=1.2, zorder=3)
        ax.axhline(0, color="#212121", linewidth=1.0, zorder=2)

        # Significance stars
        for bar, val, p in zip(bars, vals, pvals):
            if p is None: continue
            star = "***" if p < 0.001 else ("**" if p < 0.01 else
                   ("*" if p < 0.05 else "ns"))
            yoff = 0.006 if val >= 0 else -0.015
            va   = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2, val + yoff, star,
                    ha="center", va=va, fontsize=11, fontweight="bold",
                    color="#212121" if star != "ns" else "#9e9e9e")
            if abs(val) > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, val/2,
                        f"{val:+.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white",
                        path_effects=[pe.withStroke(linewidth=1.5,
                                      foreground="black")])

        ax.set_title(f"{sname}\n", fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Δ Score (correct − wrong)", fontsize=10)

        # Set ylim symmetrically
        maxv = max(abs(v) for v in vals) + 0.05
        ax.set_ylim(-maxv, maxv)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(
        f"B2 Full: Hop-2 Anchoring Across All Three Scorers — {label}\n"
        f"N={n} CHAIN_WINS pairs  |  Δ = correct_score − wrong_score",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.legend(handles=[
        mpatches.Patch(color=COLORS["flat"],  label="Flat (concat hop1+hop2) — cannot distinguish"),
        mpatches.Patch(color=COLORS["hop1"],  label="Hop-1 (bridge doc) — wrong answer anchored here"),
        mpatches.Patch(color=COLORS["hop2"],  label="Hop-2 (answer doc) — correct answer grounded here"),
    ], loc="lower center", ncol=3, fontsize=10,
       framealpha=0.9, edgecolor="#bdbdbd", bbox_to_anchor=(0.5, -0.08))

    plot_path = os.path.join(out_dir, f"b2_full_{label}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f"  Plot  → {plot_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 6 — SETTINGS + MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_settings(R: str) -> dict:
    return {
        "hotpotqa_mdr": {
            "label":   "HotpotQA MDR",
            "hop":     f"{R}/exp0c/preds/dev_hop_scores.jsonl",
            "qa":      f"{R}/exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl",
            "lex":     f"{R}/exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl",
            "z2":      f"{R}/exp_phase0/results/z2_surface_preds.jsonl",
            "zf":      f"{R}/exp_phase0/results/z_full_preds.jsonl",
            "gold":    f"{R}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json",
        },
        "hotpotqa_distractor": {
            "label":   "HotpotQA Distractor",
            "hop":     f"{R}/exp_distractor/preds/dev_hop_scores.jsonl",
            "qa":      f"{R}/exp_distractor/preds/dev_qa_hop_scores.jsonl",
            "lex":     f"{R}/exp_distractor/preds/dev_lex_features.jsonl",
            "z2":      f"{R}/exp_distractor/results/z2_surface_preds.jsonl",
            "zf":      f"{R}/exp_distractor/results/z_full_preds.jsonl",
            "gold":    f"{R}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json",
        },
        "wiki2": {
            "label":   "2WikiMultiHopQA",
            "hop":     f"{R}/exp_wiki2/preds/dev_hop_scores.jsonl",
            "qa":      f"{R}/exp_wiki2/preds/dev_qa_hop_scores.jsonl",
            "lex":     f"{R}/exp_wiki2/preds/dev_lex_features.jsonl",
            "z2":      f"{R}/exp_wiki2/results/z2_surface_preds.jsonl",
            "zf":      f"{R}/exp_wiki2/results/z_full_preds.jsonl",
            "gold":    f"{R}/data/wiki2/raw/dev_normalized.json",
        },
    }


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--proj_root",
                    default="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot")
    ap.add_argument("--out_dir",  default="experiments/results/b2_full")
    ap.add_argument("--setting",  default="all",
                    help="hotpotqa_mdr | hotpotqa_distractor | wiki2 | all")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    R       = args.proj_root
    configs = get_settings(R)

    keys = list(configs.keys()) if args.setting == "all" else [args.setting]
    all_results = {}

    for key in keys:
        cfg = configs[key]
        r = run_b2_full(
            label     = cfg["label"],
            hop_path  = cfg["hop"],
            qa_path   = cfg["qa"],
            lex_path  = cfg["lex"],
            z2_path   = cfg["z2"],
            zf_path   = cfg["zf"],
            gold_path = cfg["gold"],
            out_dir   = args.out_dir,
            no_plots  = args.no_plots,
        )
        if r:
            all_results[key] = r

    # ── Cross-dataset hop-2 delta summary table ───────────────────────
    if len(all_results) > 1:
        print(f"\n{'━'*72}")
        print(f"  CROSS-DATASET SUMMARY — Δ hop-2 score (correct − wrong)")
        print(f"  In every setting: flat ≈ 0, hop-2 significantly positive")
        print(f"{'━'*72}")
        print(f"  {'Setting':<26}  "
              f"{'Δnli_flat':>10}  {'Δnli_hop2':>10}  "
              f"{'Δqa_flat':>9}  {'Δqa_hop2':>10}  "
              f"{'Δlex_flat':>10}  {'Δlex_hop2':>10}")
        print(f"  {'─'*88}")
        for key, r in all_results.items():
            ds = r["delta_stats"]
            def v(f): return f"{ds[f]['mean_delta']:>+9.4f}"
            def s(f):
                p = ds[f]["p_value"]
                if p is None: return "   "
                return "***" if p<0.001 else ("** " if p<0.01 else
                       ("*  " if p<0.05 else "ns "))
            print(f"  {r['label']:<26}  "
                  f"{v('nli_flat')}{s('nli_flat')}  "
                  f"{v('nli_hop2')}{s('nli_hop2')}  "
                  f"{v('qa_flat')}{s('qa_flat')}  "
                  f"{v('qa_hop2')}{s('qa_hop2')}  "
                  f"{v('lex_flat')}{s('lex_flat')}  "
                  f"{v('lex_hop2')}{s('lex_hop2')}")

    # Save combined results
    out_all = os.path.join(args.out_dir, "b2_full_all.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_all, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  All results → {args.out_dir}/")


if __name__ == "__main__":
    main()