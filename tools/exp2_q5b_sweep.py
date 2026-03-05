#!/usr/bin/env python3
"""
exp2_q5b_sweep.py — Q5 Phase 2 (fast, no inference)

Reads dev_multichain_scores.jsonl (from exp2_q5a_score_chains.py) and
runs 5-fold XGBoost CV for each N in --chain_counts.

At each N, per-(question, candidate) chain-aggregate features are built
by taking the top-N chains (by chain_id order, which is MDR rank order)
and computing:
  coverage_max    max_j   min(hop1_j, hop2_j)
  coverage_mean   mean_j  min(hop1_j, hop2_j)
  stability_gap   coverage_j1 - coverage_j2  (0 when N=1)
  nli_hop1_best   hop1 score of best-coverage chain
  nli_hop2_best   hop2 score of best-coverage chain
  hop_balance_best |hop1 - hop2| of best-coverage chain

+ the 12 surface features from exp1_xgb_verifier.py (unchanged)
= 18 features total

Outputs:
  exp1b/metrics/q5_chain_sweep.json   — EM/F1 for each N + feature importances
  exp1b/logs/q5b_sweep.log

Usage (from project root):
    /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 \\
        tools/exp2_q5b_sweep.py \\
        --multichain_scores  exp1b/preds/dev_multichain_scores.jsonl \\
        --evidence           exp1b/evidence/dev_K100_chains.jsonl \\
        --gold               data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
        --out_json           exp1b/metrics/q5_chain_sweep.json \\
        --log                exp1b/logs/q5b_sweep.log \\
        --chain_counts       1 4 8 16 32 \\
        --n_folds 5 --seed 42
"""

import argparse
import collections
import json
import logging
import math
import os
import re
import string
import sys

import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb


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
    if not a: return True
    low = a.lower()
    if low.startswith("[chain"): return True
    if "if the evidence does not contain" in low: return True
    if low.startswith("the evidence provided"): return True
    if low.startswith(("okay,", "alright,", "so,")): return True
    if low in {"unknown", "unk"}: return True
    if len(a) > 120: return True
    return False


# ─────────────────────────── ECE ─────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


# ─────────────────────────── features ───────────────────────────────

SURFACE_FEATURES = [
    "nli_score", "nli_rank", "nli_score_gap",
    "answer_freq", "is_majority",
    "answer_len_chars", "answer_len_words",
    "is_empty", "is_unknown", "is_bad",
    "cand_idx", "unique_count",
]
CHAIN_AGG_FEATURES = [
    "coverage_max",       # max over N chains of min(hop1,hop2)
    "coverage_mean",      # mean over N chains of min(hop1,hop2)
    "stability_gap",      # coverage_j1 - coverage_j2 (0 at N=1)
    "nli_hop1_best",      # hop1 score of best-coverage chain
    "nli_hop2_best",      # hop2 score of best-coverage chain
    "hop_balance_best",   # |hop1-hop2| of best-coverage chain
]
FEATURE_NAMES = SURFACE_FEATURES + CHAIN_AGG_FEATURES  # 18 total


def build_features_for_N(cands_data: list, N: int) -> np.ndarray:
    """
    Build (m, 18) feature matrix for one question's candidates,
    using only the top-N chains.
    """
    m = len(cands_data)
    norms = [normalize(cd["answer"]) for cd in cands_data]
    freq_counter  = collections.Counter(norms)
    majority_norm = freq_counter.most_common(1)[0][0]
    unique_count  = len(freq_counter) / m

    nli_arr = np.array([cd["nli_flat"] for cd in cands_data], dtype=np.float64)
    nli_max = nli_arr.max()
    nli_ranks = np.argsort(-nli_arr)
    nli_rank_map = np.empty(m, dtype=np.float64)
    nli_rank_map[nli_ranks] = np.arange(m) / max(m - 1, 1)

    rows = []
    for i, cd in enumerate(cands_data):
        ans  = cd["answer"]
        norm = norms[i]

        # ── surface (identical to exp1_xgb_verifier.py) ──
        freq    = freq_counter[norm] / m
        is_maj  = int(norm == majority_norm)
        alen_c  = math.log1p(len(ans))
        alen_w  = len(ans.split())
        i_empty = int(not ans.strip())
        i_unk   = int(norm in {"unknown", "unk", ""})
        i_bad   = int(is_bad(ans))
        c_idx   = i / max(m - 1, 1)
        nli_gap = float(nli_max - nli_arr[i])

        # ── chain aggregate over top-N chains ──
        chains = cd.get("chains", [])[:N]
        if not chains:
            # fallback: all zeros for chain features
            cov_max = cov_mean = stab = h1b = h2b = hbal = 0.0
        else:
            coverages = [
                min(float(ch.get("nli_hop1", 0.0)),
                    float(ch.get("nli_hop2", 0.0)))
                for ch in chains
            ]
            cov_sorted = sorted(coverages, reverse=True)
            cov_max  = cov_sorted[0]
            cov_mean = sum(coverages) / len(coverages)
            stab     = cov_sorted[0] - (cov_sorted[1] if len(cov_sorted) > 1 else 0.0)

            best_j    = int(np.argmax(coverages))
            best_chain = chains[best_j]
            h1b  = float(best_chain.get("nli_hop1", 0.0))
            h2b  = float(best_chain.get("nli_hop2", 0.0))
            hbal = abs(h1b - h2b)

        rows.append([
            float(nli_arr[i]),       # nli_score
            float(nli_rank_map[i]),  # nli_rank
            nli_gap,                 # nli_score_gap
            freq,                    # answer_freq
            float(is_maj),           # is_majority
            alen_c,                  # answer_len_chars
            float(alen_w),           # answer_len_words
            float(i_empty),          # is_empty
            float(i_unk),            # is_unknown
            float(i_bad),            # is_bad
            c_idx,                   # cand_idx
            unique_count,            # unique_count
            cov_max,                 # coverage_max
            cov_mean,                # coverage_mean
            stab,                    # stability_gap
            h1b,                     # nli_hop1_best
            h2b,                     # nli_hop2_best
            hbal,                    # hop_balance_best
        ])
    return np.array(rows, dtype=np.float32)


# ─────────────────────────── CV + metrics ────────────────────────────

def run_cv_and_eval(
    unique_qids, qids, X, y,
    qid_cands, gold_map, qtype_map, feasible,
    n_folds, seed, xgb_params, N, log,
) -> dict:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(X), dtype=np.float32)
    fold_imps = []

    for fold, (train_qi, val_qi) in enumerate(kf.split(unique_qids)):
        train_set = set(unique_qids[train_qi])
        val_set   = set(unique_qids[val_qi])
        tm = np.array([q in train_set for q in qids])
        vm = np.array([q in val_set   for q in qids])
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X[tm], y[tm])
        oof_probs[vm] = clf.predict_proba(X[vm])[:, 1]
        fold_imps.append(clf.feature_importances_)

    # pick best candidate per question
    qid_rows: dict[str, list] = collections.defaultdict(list)
    for i, qid in enumerate(qids):
        qid_rows[qid].append(i)

    preds: dict[str, str] = {}
    conf:  dict[str, float] = {}
    for qid, idxs in qid_rows.items():
        cands = qid_cands[qid]
        probs = [float(oof_probs[i]) for i in idxs]
        best  = int(np.argmax(probs))
        preds[qid] = cands[best]
        conf[qid]  = max(probs)

    def _em_f1(qid_filter=None, type_filter=None):
        e = f = n = 0
        for qid, pred in preds.items():
            if qid_filter is not None and qid not in qid_filter:
                continue
            if type_filter is not None and qtype_map.get(qid) != type_filter:
                continue
            g = gold_map.get(qid, "")
            e += em(pred, g)
            f += f1_score(pred, g)
            n += 1
        return ({"n": n, "em": round(e/n, 4), "f1": round(f/n, 4)}
                if n else {"n": 0, "em": 0.0, "f1": 0.0})

    conf_arr  = np.array([conf[q] for q in preds])
    label_arr = np.array([em(preds[q], gold_map.get(q, "")) for q in preds])
    ece = compute_ece(conf_arr, label_arr)

    mean_imp = np.mean(fold_imps, axis=0)
    imps = {name: round(float(v), 4)
            for name, v in sorted(zip(FEATURE_NAMES, mean_imp),
                                  key=lambda x: -x[1])}

    return {
        "N":          N,
        "overall":    _em_f1(),
        "feasible":   _em_f1(qid_filter=feasible),
        "bridge":     _em_f1(type_filter="bridge"),
        "comparison": _em_f1(type_filter="comparison"),
        "ece":        round(ece, 4),
        "feature_importances": imps,
    }


# ─────────────────────────── main ────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multichain_scores", required=True,
                    help="exp1b/preds/dev_multichain_scores.jsonl")
    ap.add_argument("--evidence",   required=True)
    ap.add_argument("--gold",       required=True)
    ap.add_argument("--out_json",   required=True)
    ap.add_argument("--log",        required=True)
    ap.add_argument("--chain_counts", type=int, nargs="+",
                    default=[1, 4, 8, 16, 32],
                    help="Values of N to sweep. Must not exceed max_chains "
                         "used in phase 1. Add 100 if phase 1 used --max_chains 100.")
    ap.add_argument("--n_folds",    type=int, default=5)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--xgb_n_estimators", type=int,   default=300)
    ap.add_argument("--xgb_max_depth",    type=int,   default=4)
    ap.add_argument("--xgb_lr",           type=float, default=0.1)
    ap.add_argument("--xgb_subsample",    type=float, default=0.8)
    ap.add_argument("--xgb_colsample",    type=float, default=0.8)
    args = ap.parse_args()

    for p in [args.out_json, args.log]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    log = logging.getLogger("q5b")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    log.addHandler(logging.FileHandler(args.log, mode="w"))
    log.addHandler(logging.StreamHandler(sys.stdout))
    for h in log.handlers:
        h.setFormatter(fmt)

    log.info("=== Q5b Chain Count Sweep ===")
    log.info(f"chain_counts : {args.chain_counts}")
    log.info(f"n_folds={args.n_folds}  seed={args.seed}")

    xgb_params = dict(
        n_estimators     = args.xgb_n_estimators,
        max_depth        = args.xgb_max_depth,
        learning_rate    = args.xgb_lr,
        subsample        = args.xgb_subsample,
        colsample_bytree = args.xgb_colsample,
        objective        = "binary:logistic",
        eval_metric      = "logloss",
        random_state     = args.seed,
        n_jobs           = -1,
        verbosity        = 0,
    )

    # ── load gold + types ──
    log.info("Loading gold + question types ...")
    gold_map:  dict[str, str] = {}
    qtype_map: dict[str, str] = {}
    for ex in json.load(open(args.gold)):
        qid = str(ex["_id"])
        gold_map[qid]  = ex["answer"]
        qtype_map[qid] = ex.get("type", "bridge")

    # ── feasible subset ──
    log.info("Loading feasible subset ...")
    feasible: set[str] = set()
    for line in open(args.evidence):
        r = json.loads(line)
        if r.get("flags", {}).get("doc_recall_at_k", False):
            feasible.add(str(r["qid"]))
    log.info(f"  feasible={len(feasible)}")

    # ── load multichain scores ──
    log.info("Loading multichain scores ...")
    mc_records: dict[str, dict] = {}
    for line in open(args.multichain_scores):
        r = json.loads(line)
        mc_records[str(r["qid"])] = r
    log.info(f"  {len(mc_records)} records")

    # Validate: warn if any N in chain_counts exceeds available chains
    max_available = max(
        (len(cd["chains"])
         for r in mc_records.values()
         for cd in r["candidates"]),
        default=0,
    )
    log.info(f"  Max chains available per candidate: {max_available}")
    for N in args.chain_counts:
        if N > max_available:
            log.warning(f"  N={N} exceeds max_available={max_available} — "
                        f"re-run phase 1 with --max_chains {N}")

    # ── build qid_cands map ──
    qid_cands: dict[str, list[str]] = {
        qid: [cd["answer"] for cd in r["candidates"]]
        for qid, r in mc_records.items()
        if qid in gold_map
    }

    # ── sweep ──
    sweep_results = []

    # Baselines for reference (from exp1b)
    BASELINE_NLI_EM  = 0.3011
    BASELINE_XGB_EM  = 0.3409   # surface-only XGBoost

    log.info(f"\n  {'N':>4}  {'Overall EM':>11}  {'Bridge EM':>10}  "
             f"{'Comp EM':>9}  {'Feasible EM':>12}  {'ECE':>6}")
    log.info("  " + "-" * 60)

    for N in sorted(args.chain_counts):
        log.info(f"  Building features for N={N} ...")

        all_qids_list: list[str] = []
        all_X: list  = []
        all_y: list  = []

        for qid in sorted(mc_records):
            if qid not in gold_map:
                continue
            r     = mc_records[qid]
            cands = r["candidates"]
            gold  = gold_map[qid]

            X_q = build_features_for_N(cands, N)
            y_q = [em(cd["answer"], gold) for cd in cands]

            for ci in range(len(cands)):
                all_qids_list.append(qid)
                all_X.append(X_q[ci].tolist())
                all_y.append(y_q[ci])

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.float32)
        qids = np.array(all_qids_list)
        unique_qids = np.unique(qids)

        result = run_cv_and_eval(
            unique_qids, qids, X, y,
            qid_cands, gold_map, qtype_map, feasible,
            args.n_folds, args.seed, xgb_params, N, log,
        )
        sweep_results.append(result)

        log.info(
            f"  {N:>4}  {result['overall']['em']:>11.4f}  "
            f"{result['bridge']['em']:>10.4f}  "
            f"{result['comparison']['em']:>9.4f}  "
            f"{result['feasible']['em']:>12.4f}  "
            f"{result['ece']:>6.4f}"
        )

    # ── decision ──
    ems = [r["overall"]["em"] for r in sweep_results]
    plateau_N = None
    for i in range(1, len(ems)):
        if ems[i] - ems[i-1] < 0.001:
            plateau_N = sweep_results[i]["N"]
            break

    decision = (
        f"Curve plateaus at N={plateau_N} "
        f"(gain <0.1pp from N={sweep_results[i-1]['N']} to N={plateau_N})"
        if plateau_N else
        f"Curve has not plateaued by N={max(args.chain_counts)} — "
        f"consider extending to N=100 with phase 1 --max_chains 100"
    )

    summary = {
        "sweep":          sweep_results,
        "baselines": {
            "flat_nli_em":  BASELINE_NLI_EM,
            "xgb_surface_em": BASELINE_XGB_EM,
        },
        "plateau_decision": decision,
        "chain_counts_used": sorted(args.chain_counts),
        "xgb_params": xgb_params,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nSummary saved to {args.out_json}")

    log.info("=" * 64)
    log.info(f"  DECISION: {decision}")
    log.info("=" * 64)

    # ── feature importance trend: does stability_gap grow with N? ──
    log.info("\n  stability_gap importance by N:")
    for r in sweep_results:
        imp = r["feature_importances"].get("stability_gap", 0.0)
        log.info(f"    N={r['N']:>3}  stability_gap={imp:.4f}")


if __name__ == "__main__":
    main() 