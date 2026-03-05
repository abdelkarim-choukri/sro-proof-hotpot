#!/usr/bin/env python3
"""
exp2_q2q3q4_chain_verifier.py — Q2 + Q3 + Q4 in one CV run

Q2: Does min(sup1, sup2) outperform mean(sup1, sup2) for chain coverage?
Q3: Is the improvement asymmetric — bridge gains, comparison does not?
Q4: Which chain features matter most?

Inputs (no NLI inference needed — reads from dev_hop_scores.jsonl):
  dev_hop_scores.jsonl   — per-(qid, candidate): nli_flat, nli_hop1, nli_hop2,
                           min_hop, mean_hop, imbalance  (produced by Q1 script)
  hotpot_dev_distractor  — gold answers + question type (bridge / comparison)
  dev_K100_chains.jsonl  — for feasible-subset flag (doc_recall_at_k)

Feature set (17 features total):
  Surface features (identical to exp1_xgb_verifier.py):
    nli_score, nli_rank, nli_score_gap, answer_freq, is_majority,
    answer_len_chars, answer_len_words, is_empty, is_unknown, is_bad,
    cand_idx, unique_count
  Chain-aware additions:
    nli_hop1, nli_hop2, hop_balance (|hop1−hop2|),
    coverage (min_hop OR mean_hop), coverage_rank, coverage_gap

Two XGBoost variants run identically except for the coverage feature:
  "min"  — coverage = min(nli_hop1, nli_hop2)
  "mean" — coverage = mean(nli_hop1, nli_hop2)

Decision rule for Q2:
  bridge_em(min) > bridge_em(mean) by ≥0.5pp  → min pooling preferred
  otherwise                                     → pooling choice is secondary

Decision rule for Q3:
  bridge gain > 1.0pp  AND  comparison gain < 0.5pp  → asymmetry confirmed

Usage (no server, no LLM env — scipy + xgboost only):
    /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/exp2_q2q3q4_chain_verifier.py \\
        --hop_scores   exp1b/preds/dev_hop_scores.jsonl \\
        --evidence     exp1b/evidence/dev_K100_chains.jsonl \\
        --gold         data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
        --out_json     exp1b/metrics/q2q3q4_chain_verifier.json \\
        --out_preds_min   exp1b/preds/dev_chain_verifier_min_preds.jsonl \\
        --out_preds_mean  exp1b/preds/dev_chain_verifier_mean_preds.jsonl \\
        --log          exp1b/logs/q2q3q4_chain_verifier.log \\
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
    """Identical to exp1_xgb_verifier.py."""
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
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)

# ─────────────────────────── feature extraction ──────────────────────

# Feature names — order must match extract_features() output
SURFACE_FEATURES = [
    "nli_score",        # 0  flat entailment probability
    "nli_rank",         # 1  rank among 5 candidates (normalised 0–1)
    "nli_score_gap",    # 2  max_nli - this_nli
    "answer_freq",      # 3  fraction of candidates with same normalised answer
    "is_majority",      # 4  1 if plurality answer
    "answer_len_chars", # 5  log(1 + char length)
    "answer_len_words", # 6  word count
    "is_empty",         # 7
    "is_unknown",       # 8
    "is_bad",           # 9
    "cand_idx",         # 10 position (0–4, normalised)
    "unique_count",     # 11 distinct answers / M
]
CHAIN_FEATURES = [
    "nli_hop1",         # 12 hop-1 entailment probability
    "nli_hop2",         # 13 hop-2 entailment probability
    "hop_balance",      # 14 |hop1 - hop2|
    "coverage",         # 15 min_hop OR mean_hop (the ablation axis)
    "coverage_rank",    # 16 rank of coverage among candidates (normalised)
    "coverage_gap",     # 17 max_coverage - this_coverage
]
FEATURE_NAMES = SURFACE_FEATURES + CHAIN_FEATURES  # length 18


def extract_features(cands_data: list, coverage_mode: str) -> np.ndarray:
    """
    Build feature matrix for one question's candidates.
    cands_data: list of dicts from dev_hop_scores.jsonl candidates array,
                each has keys: answer, nli_flat, nli_hop1, nli_hop2,
                               min_hop, mean_hop, imbalance
    coverage_mode: "min" or "mean"
    Returns np.ndarray of shape (m, 18).
    """
    m = len(cands_data)
    norms = [normalize(cd["answer"]) for cd in cands_data]
    freq_counter  = collections.Counter(norms)
    majority_norm = freq_counter.most_common(1)[0][0]
    unique_count  = len(freq_counter) / m

    # NLI flat ranks
    nli_arr  = np.array([cd["nli_flat"] for cd in cands_data], dtype=np.float64)
    nli_max  = nli_arr.max()
    nli_ranks = np.argsort(-nli_arr)
    nli_rank_map = np.empty(m, dtype=np.float64)
    nli_rank_map[nli_ranks] = np.arange(m) / max(m - 1, 1)

    # Coverage ranks
    cov_arr = np.array(
        [cd["min_hop"] if coverage_mode == "min" else cd["mean_hop"]
         for cd in cands_data],
        dtype=np.float64,
    )
    cov_max  = cov_arr.max()
    cov_ranks = np.argsort(-cov_arr)
    cov_rank_map = np.empty(m, dtype=np.float64)
    cov_rank_map[cov_ranks] = np.arange(m) / max(m - 1, 1)

    rows = []
    for i, cd in enumerate(cands_data):
        ans  = cd["answer"]
        norm = norms[i]

        # ── surface ──
        freq     = freq_counter[norm] / m
        is_maj   = int(norm == majority_norm)
        alen_c   = math.log1p(len(ans))
        alen_w   = len(ans.split())
        i_empty  = int(not ans.strip())
        i_unk    = int(norm in {"unknown", "unk", ""})
        i_bad    = int(is_bad(ans))
        c_idx    = i / max(m - 1, 1)
        nli_gap  = float(nli_max - nli_arr[i])

        # ── chain-aware ──
        s1       = float(cd.get("nli_hop1") or 0.0)
        s2       = float(cd.get("nli_hop2") or 0.0)
        h_bal    = abs(s1 - s2)
        coverage = float(cov_arr[i])
        cov_gap  = float(cov_max - coverage)

        rows.append([
            float(nli_arr[i]),     #  0 nli_score
            float(nli_rank_map[i]),#  1 nli_rank
            nli_gap,               #  2 nli_score_gap
            freq,                  #  3 answer_freq
            float(is_maj),         #  4 is_majority
            alen_c,                #  5 answer_len_chars
            float(alen_w),         #  6 answer_len_words
            float(i_empty),        #  7 is_empty
            float(i_unk),          #  8 is_unknown
            float(i_bad),          #  9 is_bad
            c_idx,                 # 10 cand_idx
            unique_count,          # 11 unique_count
            s1,                    # 12 nli_hop1
            s2,                    # 13 nli_hop2
            h_bal,                 # 14 hop_balance
            coverage,              # 15 coverage
            float(cov_rank_map[i]),# 16 coverage_rank
            cov_gap,               # 17 coverage_gap
        ])
    return np.array(rows, dtype=np.float32)


# ─────────────────────────── CV runner ───────────────────────────────

def run_cv(
    all_qids: np.ndarray,
    qids: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
    xgb_params: dict,
    log: logging.Logger,
    tag: str,
) -> tuple[np.ndarray, list]:
    """
    5-fold CV split at question level.
    Returns (oof_probs, fold_importances).
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(X), dtype=np.float32)
    fold_importances = []

    for fold, (train_qi, val_qi) in enumerate(kf.split(all_qids)):
        train_set = set(all_qids[train_qi])
        val_set   = set(all_qids[val_qi])

        train_mask = np.array([q in train_set for q in qids])
        val_mask   = np.array([q in val_set   for q in qids])

        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X[train_mask], y[train_mask])
        oof_probs[val_mask] = clf.predict_proba(X[val_mask])[:, 1]
        fold_importances.append(clf.feature_importances_)

        log.info(f"  [{tag}] fold {fold+1}/{n_folds} — "
                 f"train={len(train_set)}q  val={len(val_set)}q")

    return oof_probs, fold_importances


# ─────────────────────────── metrics ─────────────────────────────────

def compute_metrics(
    oof_probs: np.ndarray,
    qids: np.ndarray,
    qid_cands: dict,      # qid → list of answer strings
    gold_map: dict,       # qid → gold answer
    qtype_map: dict,      # qid → "bridge" | "comparison"
    feasible: set,        # qids with doc_recall_at_k = True
    tag: str,
) -> tuple[dict, dict]:
    """
    Returns (metrics_dict, preds_dict).
    preds_dict: qid → {pred, probs, best_idx}
    """
    # Group OOF probs by question
    qid_to_rows: dict[str, list[int]] = collections.defaultdict(list)
    for i, qid in enumerate(qids):
        qid_to_rows[qid].append(i)

    preds: dict[str, dict] = {}
    for qid, row_idxs in qid_to_rows.items():
        cands = qid_cands[qid]
        probs = [float(oof_probs[i]) for i in row_idxs]
        best_i = int(np.argmax(probs))
        preds[qid] = {
            "pred":     cands[best_i],
            "probs":    [round(p, 6) for p in probs],
            "best_idx": best_i,
        }

    def _em_f1(qid_filter=None, type_filter=None):
        e = f = n = 0
        for qid, p in preds.items():
            if qid_filter is not None and qid not in qid_filter:
                continue
            if type_filter is not None and qtype_map.get(qid) != type_filter:
                continue
            g = gold_map.get(qid, "")
            e += em(p["pred"], g)
            f += f1_score(p["pred"], g)
            n += 1
        if n == 0:
            return {"n": 0, "em": 0.0, "f1": 0.0}
        return {"n": n, "em": round(e / n, 4), "f1": round(f / n, 4)}

    # ECE
    conf_list  = [max(preds[q]["probs"]) for q in preds]
    label_list = [em(preds[q]["pred"], gold_map.get(q, "")) for q in preds]
    ece = compute_ece(np.array(conf_list), np.array(label_list))

    metrics = {
        "tag":        tag,
        "overall":    _em_f1(),
        "feasible":   _em_f1(qid_filter=feasible),
        "bridge":     _em_f1(type_filter="bridge"),
        "comparison": _em_f1(type_filter="comparison"),
        "ece":        round(ece, 4),
    }
    return metrics, preds


# ─────────────────────────── main ────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop_scores",     required=True,
                    help="exp1b/preds/dev_hop_scores.jsonl  (from Q1)")
    ap.add_argument("--evidence",       required=True,
                    help="exp1b/evidence/dev_K100_chains.jsonl")
    ap.add_argument("--gold",           required=True,
                    help="data/hotpotqa/raw/hotpot_dev_distractor_v1.json")
    ap.add_argument("--out_json",       required=True)
    ap.add_argument("--out_preds_min",  required=True)
    ap.add_argument("--out_preds_mean", required=True)
    ap.add_argument("--log",            required=True)
    ap.add_argument("--n_folds",    type=int, default=5)
    ap.add_argument("--seed",       type=int, default=42)
    # XGBoost hyper-parameters — same as exp1_xgb_verifier.py
    ap.add_argument("--xgb_n_estimators", type=int,   default=300)
    ap.add_argument("--xgb_max_depth",    type=int,   default=4)
    ap.add_argument("--xgb_lr",           type=float, default=0.1)
    ap.add_argument("--xgb_subsample",    type=float, default=0.8)
    ap.add_argument("--xgb_colsample",    type=float, default=0.8)
    args = ap.parse_args()

    for p in [args.out_json, args.out_preds_min, args.out_preds_mean, args.log]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # ── logging ──
    log = logging.getLogger("q2q3q4")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(args.log, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)

    log.info("=== Q2/Q3/Q4 Chain-Aware Verifier ===")
    log.info(f"hop_scores : {args.hop_scores}")
    log.info(f"evidence   : {args.evidence}")
    log.info(f"n_folds    : {args.n_folds}  seed={args.seed}")

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

    # ── load gold + question types ──
    log.info("Loading gold + question types ...")
    gold_map:  dict[str, str] = {}
    qtype_map: dict[str, str] = {}
    for ex in json.load(open(args.gold)):
        qid = str(ex["_id"])
        gold_map[qid]  = ex["answer"]
        qtype_map[qid] = ex.get("type", "bridge")   # "bridge" or "comparison"
    n_bridge     = sum(1 for t in qtype_map.values() if t == "bridge")
    n_comparison = sum(1 for t in qtype_map.values() if t == "comparison")
    log.info(f"  bridge={n_bridge}  comparison={n_comparison}  total={len(gold_map)}")

    # ── feasible subset (doc_recall_at_k = True) ──
    log.info("Loading feasible subset from evidence pack ...")
    feasible: set[str] = set()
    for line in open(args.evidence):
        r = json.loads(line)
        qid = str(r["qid"])
        flag = r.get("flags", {}).get("doc_recall_at_k", False)
        if flag:
            feasible.add(qid)
    log.info(f"  feasible={len(feasible)}/7405")

    # ── load hop scores ──
    log.info("Loading hop scores ...")
    # Schema: {qid, gold, flat_em, min_em, mean_em, oracle_em, candidates: [...]}
    hop_records: dict[str, dict] = {}
    for line in open(args.hop_scores):
        r = json.loads(line)
        hop_records[str(r["qid"])] = r
    log.info(f"  {len(hop_records)} records")

    # ── build dataset ──
    log.info("Building feature matrix ...")
    all_qids_list: list[str] = []
    all_X_min:  list[list] = []
    all_X_mean: list[list] = []
    all_y:      list[int]  = []
    qid_cands:  dict[str, list[str]] = {}   # qid → answer strings

    for qid in sorted(hop_records):
        if qid not in gold_map:
            continue
        rec   = hop_records[qid]
        cands = rec["candidates"]
        gold  = gold_map[qid]

        X_min_q  = extract_features(cands, "min")
        X_mean_q = extract_features(cands, "mean")
        labels   = [em(cd["answer"], gold) for cd in cands]

        for ci in range(len(cands)):
            all_qids_list.append(qid)
            all_X_min.append(X_min_q[ci].tolist())
            all_X_mean.append(X_mean_q[ci].tolist())
            all_y.append(labels[ci])

        qid_cands[qid] = [cd["answer"] for cd in cands]

    X_min  = np.array(all_X_min,  dtype=np.float32)
    X_mean = np.array(all_X_mean, dtype=np.float32)
    y      = np.array(all_y,      dtype=np.float32)
    qids   = np.array(all_qids_list)
    unique_qids = np.unique(qids)

    log.info(f"  {len(unique_qids)} questions, {len(X_min)} rows, "
             f"{X_min.shape[1]} features")
    log.info(f"  Positive rate: {y.mean():.3f}")

    # ── run CV for both variants ──
    log.info("Running CV — min pooling variant ...")
    oof_min, imps_min = run_cv(
        unique_qids, qids, X_min, y,
        args.n_folds, args.seed, xgb_params, log, "min"
    )

    log.info("Running CV — mean pooling variant ...")
    oof_mean, imps_mean = run_cv(
        unique_qids, qids, X_mean, y,
        args.n_folds, args.seed, xgb_params, log, "mean"
    )

    # ── compute metrics ──
    log.info("Computing metrics ...")
    metrics_min,  preds_min  = compute_metrics(
        oof_min,  qids, qid_cands, gold_map, qtype_map, feasible, "min")
    metrics_mean, preds_mean = compute_metrics(
        oof_mean, qids, qid_cands, gold_map, qtype_map, feasible, "mean")

    # ── feature importances ──
    def importance_dict(fold_imps):
        mean_imp = np.mean(fold_imps, axis=0)
        return {
            name: round(float(v), 4)
            for name, v in sorted(
                zip(FEATURE_NAMES, mean_imp), key=lambda x: -x[1]
            )
        }

    imps_min_dict  = importance_dict(imps_min)
    imps_mean_dict = importance_dict(imps_mean)

    # ── decisions ──
    bridge_gain_min  = metrics_min["bridge"]["em"]  - metrics_min["overall"]["em"]
    bridge_gain_mean = metrics_mean["bridge"]["em"] - metrics_mean["overall"]["em"]
    min_vs_mean_bridge = (metrics_min["bridge"]["em"]
                          - metrics_mean["bridge"]["em"])

    # Baseline numbers from exp1b for delta calculation
    BASELINE_FLAT_NLI_EM = 0.3011
    BASELINE_XGB_EM      = 0.3409

    def decision_q2():
        gap = metrics_min["bridge"]["em"] - metrics_mean["bridge"]["em"]
        if gap >= 0.005:
            return f"MIN pooling preferred on bridge (+{gap:.4f})"
        elif gap <= -0.005:
            return f"MEAN pooling preferred on bridge ({gap:.4f})"
        else:
            return f"Pooling choice negligible on bridge (gap={gap:.4f})"

    def decision_q3():
        # Compare best chain-aware variant vs flat NLI baseline (0.3011)
        best = metrics_min if (metrics_min["overall"]["em"]
                               >= metrics_mean["overall"]["em"]) else metrics_mean
        bridge_delta     = best["bridge"]["em"]  - BASELINE_FLAT_NLI_EM
        comparison_delta = best["comparison"]["em"] - BASELINE_FLAT_NLI_EM
        asymmetric = (bridge_delta > 0.010) and (abs(comparison_delta) < 0.005)
        label = "ASYMMETRY CONFIRMED" if asymmetric else "ASYMMETRY NOT CONFIRMED"
        return (f"{label}: bridge Δ={bridge_delta:+.4f}  "
                f"comparison Δ={comparison_delta:+.4f} vs flat NLI baseline")

    def decision_q4():
        top3 = list(imps_min_dict.items())[:3]
        top3_str = ", ".join(f"{k}={v:.3f}" for k, v in top3)
        chain_rank = {k: i for i, k in enumerate(imps_min_dict)}
        chain_ranks = {f: chain_rank.get(f, -1)
                       for f in CHAIN_FEATURES}
        return (f"Top-3 features (min model): {top3_str} | "
                f"Chain feature ranks: "
                + ", ".join(f"{f}=#{chain_ranks[f]+1}" for f in CHAIN_FEATURES))

    # ── assemble summary ──
    summary = {
        "exp1b_baselines": {
            "flat_nli_em": BASELINE_FLAT_NLI_EM,
            "xgb_surface_em": BASELINE_XGB_EM,
        },
        "min_pooling": {
            **metrics_min,
            "feature_importances": imps_min_dict,
        },
        "mean_pooling": {
            **metrics_mean,
            "feature_importances": imps_mean_dict,
        },
        "q2_decision": decision_q2(),
        "q3_decision": decision_q3(),
        "q4_decision": decision_q4(),
        "xgb_params": xgb_params,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to {args.out_json}")

    # ── save predictions ──
    for preds, path in [(preds_min,  args.out_preds_min),
                        (preds_mean, args.out_preds_mean)]:
        with open(path, "w") as f:
            for qid in sorted(preds):
                f.write(json.dumps({"qid": qid, **preds[qid]}) + "\n")
    log.info(f"Predictions saved to {args.out_preds_min} / {args.out_preds_mean}")

    # ── print results table ──
    W = 72
    log.info("=" * W)
    log.info("  Q2/Q3/Q4 RESULTS")
    log.info("=" * W)
    log.info(f"  {'Method':<28} {'Overall EM':>10} {'Bridge EM':>10} "
             f"{'Comp EM':>9} {'Feasible EM':>12}")
    log.info("  " + "-" * (W - 2))

    rows = [
        ("Flat NLI (exp1b baseline)",
         BASELINE_FLAT_NLI_EM, "—", "—", "—"),
        ("XGBoost surface (exp1b)",
         BASELINE_XGB_EM, "—", "—", "—"),
        ("Chain-aware XGB [min]",
         metrics_min["overall"]["em"],
         metrics_min["bridge"]["em"],
         metrics_min["comparison"]["em"],
         metrics_min["feasible"]["em"]),
        ("Chain-aware XGB [mean]",
         metrics_mean["overall"]["em"],
         metrics_mean["bridge"]["em"],
         metrics_mean["comparison"]["em"],
         metrics_mean["feasible"]["em"]),
    ]
    for label, ov, br, co, fe in rows:
        log.info(f"  {label:<28} {str(ov):>10} {str(br):>10} "
                 f"{str(co):>9} {str(fe):>12}")

    log.info("=" * W)
    log.info(f"  Q2: {decision_q2()}")
    log.info(f"  Q3: {decision_q3()}")
    log.info(f"  Q4: {decision_q4()}")
    log.info("=" * W)

    log.info("\n  Feature importances (min model, top-10):")
    for i, (feat, imp) in enumerate(list(imps_min_dict.items())[:10]):
        marker = " ◀ chain" if feat in CHAIN_FEATURES else ""
        log.info(f"    #{i+1:>2}  {feat:<22}  {imp:.4f}{marker}")


if __name__ == "__main__":
    main()