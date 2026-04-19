#!/usr/bin/env python3
"""
experiments/plot_b2.py — Generate B2 figures from b2_delta_stats JSON files

Usage:
    python3 experiments/plot_b2.py \
        --data_dir  experiments/results/b2_fixed \
        --out_dir   experiments/results/b2_fixed/figures
"""
import argparse, json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

C_HOP1 = "#E53935"
C_HOP2 = "#1E88E5"
C_FLAT = "#757575"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "grid.linestyle": "--", "grid.linewidth": 0.6,
    "figure.facecolor": "#fafafa", "axes.facecolor": "#fafafa",
})

def load_stats(data_dir: str) -> dict:
    """Load all b2_delta_stats_*.json files from data_dir."""
    result = {}
    order = ["hotpotqa_mdr", "hotpotqa_distractor", "wiki2"]
    for key in order:
        path = os.path.join(data_dir, f"b2_delta_stats_{key}.json")
        if os.path.exists(path):
            with open(path) as f:
                result[key] = json.load(f)
    return result

def sig_star(p):
    if p is None: return ""
    return "***" if p < 0.001 else ("**" if p < 0.01 else
           ("*" if p < 0.05 else "ns"))

def add_bar_labels(ax, bars, vals, pvals):
    for bar, val, p in zip(bars, vals, pvals):
        star = sig_star(p)
        yoff = 0.008 if val >= 0 else -0.018
        va   = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, val + yoff, star,
                ha="center", va=va, fontsize=11, fontweight="bold",
                color="#212121" if star != "ns" else "#9e9e9e")
        if abs(val) > 0.015:
            ax.text(bar.get_x() + bar.get_width()/2, val/2,
                    f"{val:+.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                    path_effects=[pe.withStroke(linewidth=1.5,
                                  foreground="black")])

def plot_main(stats: dict, out_path: str):
    """Figure 1: Three-panel, one per dataset."""
    keys = [k for k in ["hotpotqa_mdr", "hotpotqa_distractor", "wiki2"]
            if k in stats]
    name_map = {
        "hotpotqa_mdr":        "HotpotQA MDR",
        "hotpotqa_distractor": "HotpotQA Distractor",
        "wiki2":               "2WikiMultiHopQA",
    }

    fig = plt.figure(figsize=(14, 5), facecolor="#fafafa")
    gs  = GridSpec(1, len(keys), figure=fig, wspace=0.38)

    for col, key in enumerate(keys):
        d  = stats[key]
        ax = fig.add_subplot(gs[0, col])
        ds = d["delta_stats"]

        vals   = [ds["nli_flat"]["mean_delta"],
                  ds["nli_hop1"]["mean_delta"],
                  ds["nli_hop2"]["mean_delta"]]
        pvals  = [ds["nli_flat"].get("p_value"),
                  ds["nli_hop1"].get("p_value"),
                  ds["nli_hop2"].get("p_value")]
        colors = [C_FLAT, C_HOP1, C_HOP2]
        xlbls  = ["Flat NLI\n(baseline)", "Hop-1 NLI\n(bridge doc)",
                  "Hop-2 NLI\n(answer doc)"]

        bars = ax.bar(xlbls, vals, color=colors, width=0.55,
                      edgecolor="white", linewidth=1.2, zorder=3)
        ax.axhline(0, color="#212121", linewidth=1.0, zorder=2)
        add_bar_labels(ax, bars, vals, pvals)

        n = d.get("n_chain_wins", d.get("n_pairs_used", "?"))
        ax.set_title(f"{name_map.get(key, key)}\n(N={n} CHAIN_WINS)",
                     fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Δ Score (correct − wrong)" if col == 0 else "",
                      fontsize=10)
        ax.set_ylim(-0.42, 0.42)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

    fig.legend(handles=[
        mpatches.Patch(color=C_FLAT,  label="Flat NLI — cannot distinguish"),
        mpatches.Patch(color=C_HOP1,  label="Hop-1 NLI — wrong answer anchored in bridge doc"),
        mpatches.Patch(color=C_HOP2,  label="Hop-2 NLI — correct answer in answer doc"),
    ], loc="lower center", ncol=3, fontsize=10,
       framealpha=0.9, edgecolor="#bdbdbd", bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(
        "B2: Per-Hop NLI Score Deltas within CHAIN_WINS Questions\n"
        "Δ = correct_answer_score − wrong_answer_score  |  *** p<0.001",
        fontsize=12, fontweight="bold", y=1.02)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_pertype(stats: dict, out_path: str):
    """Figure 2: 2Wiki per-type breakdown."""
    if "wiki2" not in stats:
        print("  wiki2 data not found — skipping per-type figure")
        return

    d  = stats["wiki2"]
    pt = d.get("per_type", {})
    if not pt:
        print("  No per_type data — skipping")
        return

    type_labels = {
        "bridge_comparison": "Bridge-\nComparison",
        "comparison":        "Comparison",
        "compositional":     "Compositional",
        "inference":         "Inference",
    }
    types = [t for t in ["bridge_comparison","comparison","compositional","inference"]
             if t in pt]

    ns      = [pt[t].get("n",   pt[t].get("nli_hop2", 0)) for t in types]
    h1_vals = [pt[t]["nli_hop1"] for t in types]
    h2_vals = [pt[t]["nli_hop2"] for t in types]

    # ns might not be in per_type if saved differently — fallback
    # re-read from chain_wins count if n key missing
    has_n = all(isinstance(pt[t].get("n"), int) for t in types)
    if not has_n:
        ns = ["?" for _ in types]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#fafafa")
    x = np.arange(len(types))
    w = 0.32

    bars1 = ax.bar(x - w/2, h1_vals, w, color=C_HOP1, alpha=0.88,
                   edgecolor="white", linewidth=1.2,
                   label="Δ nli_hop1 (bridge doc)", zorder=3)
    bars2 = ax.bar(x + w/2, h2_vals, w, color=C_HOP2, alpha=0.88,
                   edgecolor="white", linewidth=1.2,
                   label="Δ nli_hop2 (answer doc)", zorder=3)

    ax.axhline(0, color="#212121", linewidth=1.0, zorder=2)

    for bar, val in list(zip(bars1, h1_vals)) + list(zip(bars2, h2_vals)):
        if abs(val) > 0.03:
            ax.text(bar.get_x() + bar.get_width()/2, val/2,
                    f"{val:+.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                    path_effects=[pe.withStroke(linewidth=1.5,
                                  foreground="black")])

    xlbls = [f"{type_labels.get(t,t)}\n(n={ns[i]})"
             for i, t in enumerate(types)]
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=10)
    ax.set_ylabel("Δ Score (correct − wrong)", fontsize=11)
    ax.set_title(
        "B2: Per-Type Breakdown — 2WikiMultiHopQA CHAIN_WINS\n"
        "Inference: mechanism weaker — correct answers require reasoning, not extraction",
        fontsize=11, fontweight="bold")
    ax.set_ylim(-0.55, 0.55)
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="#bdbdbd")

    if "inference" in types:
        inf_idx = types.index("inference")
        ax.annotate("Correct answers require\nreasoning beyond literal\nentailment in hop-2",
                    xy=(inf_idx + w/2, h2_vals[inf_idx]),
                    xytext=(inf_idx + 0.55, -0.38),
                    fontsize=8.5, color="#555",
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff",
                              edgecolor="#bbb", alpha=0.9))

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f"  Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="experiments/results/b2_fixed")
    ap.add_argument("--out_dir",
                    default="experiments/results/b2_fixed/figures")
    args = ap.parse_args()

    print("Loading delta stats ...")
    stats = load_stats(args.data_dir)
    print(f"  Found: {list(stats.keys())}")

    print("\nGenerating Figure 1 — main 3-panel ...")
    plot_main(stats, os.path.join(args.out_dir, "b2_main.png"))

    print("Generating Figure 2 — 2Wiki per-type ...")
    plot_pertype(stats, os.path.join(args.out_dir, "b2_pertype.png"))

    print(f"\nDone. Figures → {args.out_dir}/")


if __name__ == "__main__":
    main()