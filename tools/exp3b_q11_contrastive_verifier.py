#!/usr/bin/env python3
"""
exp3b_q11_contrastive_verifier.py — Q11: Contrastive Cross-Candidate Features

Hypothesis: the verifier looks at each candidate in isolation. C2 failures
(plausible-wrong picked over correct) might be reduced by adding features
that describe where each candidate sits *relative to its peers*.

New features (5 total, added to the existing 18):
  nli_hop1_rank       — rank of this candidate's hop1 score among all M (0=best)
  nli_hop2_rank       — rank of hop2 score (0=best)
  hop_balance_rank    — rank by hop_balance (0=most balanced, smallest |hop1-hop2|)
  delta_to_best_hop1  — best_hop1 - this_hop1  (how far behind the leader)
  delta_to_best_hop2  — best_hop2 - this_hop2

Total features: 18 (existing) + 5 (contrastive) = 23

Two ablation variants run in one script:
  "baseline" — original 18 features (reproduces exp3b chain-aware results)
  "contrastive" — 23 features (the new thing)

Decision rules:
  EM(contrastive) > EM(baseline) by ≥0.5pp  → contrastive features help
  Any contrastive feature in top-5 importances → verifier is using them
  C2 count drops → the right mechanism (relative ranking) is working

Inputs (no inference — reads existing hop scores):
  dev_hop_scores.jsonl   — from exp3b pipeline (M=10 candidates)
  dev_K100_chains.jsonl  — feasible flags
  hotpot_dev_distractor  — gold answers + question type

Usage (CPU only, ~2 min):
    /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 \\
        tools/exp3b_q11_contrastive_verifier.py \\
        --hop_scores   exp3b/preds/dev_hop_scores.jsonl \\
        --evidence     exp1b/evidence/dev_K100_chains.jsonl \\
        --gold         data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
        --out_json     exp3b/metrics/q11_contrastive_verifier.json \\
        --out_preds    exp3b/preds/dev_q11_contrastive_preds.jsonl \\
        --log          exp3b/logs/q11_contrastive.log \\
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
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


# ─────────────────────────── feature extraction ──────────────────────

# Original 18 features
SURFACE_FEATURES = [
    "nli_score", "nli_rank", "nli_score_gap",
    "answer_freq", "is_majority",
    "answer_len_chars", "answer_len_words",
    "is_empty", "is_unknown", "is_bad",
    "cand_idx", "unique_count",
]
CHAIN_FEATURES = [
    "nli_hop1", "nli_hop2", "hop_balance",
    "coverage", "coverage_rank", "coverage_gap",
]

# NEW: contrastive cross-candidate features
CONTRASTIVE_FEATURES = [
    "nli_hop1_rank",        # rank of hop1 score (0=best, normalised)
    "nli_hop2_rank",        # rank of hop2 score (0=best, normalised)
    "hop_balance_rank",     # rank by balance (0=most balanced, normalised)
    "delta_to_best_hop1",   # best_hop1 - this_hop1
    "delta_to_best_hop2",   # best_hop2 - this_hop2
]

BASELINE_FEATURES = SURFACE_FEATURES + CHAIN_FEATURES            # 18
ALL_FEATURES = SURFACE_FEATURES + CHAIN_FEATURES + CONTRASTIVE_FEATURES  # 23


def extract_features(cands_data: list, coverage_mode: str,
                     include_contrastive: bool) -> np.ndarray:
    """
    Build feature matrix for one question's candidates.

    cands_data: list of dicts from dev_hop_scores.jsonl candidates array
    coverage_mode: "min" or "mean"
    include_contrastive: if True, append 5 contrastive features (23 total)
                         if False, return original 18 features

    Returns np.ndarray of shape (m, 18) or (m, 23).
    """
    m = len(cands_data)
    norms = [normalize(cd["answer"]) for cd in cands_data]
    freq_counter  = collections.Counter(norms)
    majority_norm = freq_counter.most_common(1)[0][0]
    unique_count  = len(freq_counter) / m

    # ── arrays for ranking ──
    nli_arr = np.array([cd["nli_flat"] for cd in cands_data], dtype=np.float64)
    nli_max = nli_arr.max()

    # Flat NLI ranks
    nli_ranks = np.argsort(-nli_arr)
    nli_rank_map = np.empty(m, dtype=np.float64)
    nli_rank_map[nli_ranks] = np.arange(m) / max(m - 1, 1)

    # Coverage
    cov_arr = np.array(
        [cd["min_hop"] if coverage_mode == "min" else cd["mean_hop"]
         for cd in cands_data],
        dtype=np.float64,
    )
    cov_max = cov_arr.max()
    cov_ranks = np.argsort(-cov_arr)
    cov_rank_map = np.empty(m, dtype=np.float64)
    cov_rank_map[cov_ranks] = np.arange(m) / max(m - 1, 1)

    # ── NEW: per-hop arrays for contrastive features ──
    hop1_arr = np.array([float(cd.get("nli_hop1") or 0.0)
                         for cd in cands_data], dtype=np.float64)
    hop2_arr = np.array([float(cd.get("nli_hop2") or 0.0)
                         for cd in cands_data], dtype=np.float64)
    hop1_max = hop1_arr.max()
    hop2_max = hop2_arr.max()

    # Hop1 ranks (descending — 0 = best hop1 score)
    hop1_ranks = np.argsort(-hop1_arr)
    hop1_rank_map = np.empty(m, dtype=np.float64)
    hop1_rank_map[hop1_ranks] = np.arange(m) / max(m - 1, 1)

    # Hop2 ranks
    hop2_ranks = np.argsort(-hop2_arr)
    hop2_rank_map = np.empty(m, dtype=np.float64)
    hop2_rank_map[hop2_ranks] = np.arange(m) / max(m - 1, 1)

    # Hop balance = |hop1 - hop2|. Lower = more balanced.
    # Rank ascending: 0 = most balanced
    balance_arr = np.abs(hop1_arr - hop2_arr)
    balance_ranks = np.argsort(balance_arr)  # ascending — smallest first
    balance_rank_map = np.empty(m, dtype=np.float64)
    balance_rank_map[balance_ranks] = np.arange(m) / max(m - 1, 1)

    rows = []
    for i, cd in enumerate(cands_data):
        ans  = cd["answer"]
        norm = norms[i]

        # ── surface (0–11) ──
        freq     = freq_counter[norm] / m
        is_maj   = int(norm == majority_norm)
        alen_c   = math.log1p(len(ans))
        alen_w   = len(ans.split())
        i_empty  = int(not ans.strip())
        i_unk    = int(norm in {"unknown", "unk", ""})
        i_bad    = int(is_bad(ans))
        c_idx    = i / max(m - 1, 1)
        nli_gap  = float(nli_max - nli_arr[i])

        # ── chain-aware (12–17) ──
        s1       = float(hop1_arr[i])
        s2       = float(hop2_arr[i])
        h_bal    = float(balance_arr[i])
        coverage = float(cov_arr[i])
        cov_gap  = float(cov_max - coverage)

        row = [
            float(nli_arr[i]),      #  0 nli_score
            float(nli_rank_map[i]), #  1 nli_rank
            nli_gap,                #  2 nli_score_gap
            freq,                   #  3 answer_freq
            float(is_maj),          #  4 is_majority
            alen_c,                 #  5 answer_len_chars
            float(alen_w),          #  6 answer_len_words
            float(i_empty),         #  7 is_empty
            float(i_unk),           #  8 is_unknown
            float(i_bad),           #  9 is_bad
            c_idx,                  # 10 cand_idx
            unique_count,           # 11 unique_count
            s1,                     # 12 nli_hop1
            s2,                     # 13 nli_hop2
            h_bal,                  # 14 hop_balance
            coverage,               # 15 coverage
            float(cov_rank_map[i]), # 16 coverage_rank
            cov_gap,                # 17 coverage_gap
        ]

        if include_contrastive:
            row.extend([
                float(hop1_rank_map[i]),        # 18 nli_hop1_rank
                float(hop2_rank_map[i]),        # 19 nli_hop2_rank
                float(balance_rank_map[i]),     # 20 hop_balance_rank
                float(hop1_max - hop1_arr[i]),  # 21 delta_to_best_hop1
                float(hop2_max - hop2_arr[i]),  # 22 delta_to_best_hop2
            ])

        rows.append(row)

    return np.array(rows, dtype=np.float32)


# ─────────────────────────── CV runner ───────────────────────────────

def run_cv(
    all_qids_unique: np.ndarray,
    qids: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    qid_cands: dict,
    gold_map: dict,
    qtype_map: dict,
    feasible: set,
    n_folds: int,
    seed: int,
    xgb_params: dict,
    feature_names: list,
    log,
    tag: str,
):
    """Run 5-fold CV, return metrics + preds + importances."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(X), dtype=np.float32)
    fold_importances = []

    for fold, (train_qi, val_qi) in enumerate(kf.split(all_qids_unique)):
        train_qids_set = set(all_qids_unique[train_qi])
        val_qids_set   = set(all_qids_unique[val_qi])

        train_mask = np.array([q in train_qids_set for q in qids])
        val_mask   = np.array([q in val_qids_set   for q in qids])

        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X[train_mask], y[train_mask])
        oof_probs[val_mask] = clf.predict_proba(X[val_mask])[:, 1]
        fold_importances.append(clf.feature_importances_)

        log.info(f"  [{tag}] fold {fold+1}/{n_folds} — "
                 f"train={len(train_qids_set)}q  val={len(val_qids_set)}q")

    # ── pick best candidate per question ──
    qid_to_rows = collections.defaultdict(list)
    for i, qid in enumerate(qids):
        qid_to_rows[qid].append(i)

    preds = {}
    for qid, row_idxs in qid_to_rows.items():
        cands = qid_cands[qid]
        probs = [float(oof_probs[i]) for i in row_idxs]
        best_i = int(np.argmax(probs))
        preds[qid] = {
            "pred":     cands[best_i],
            "probs":    [round(p, 6) for p in probs],
            "best_idx": best_i,
        }

    # ── metrics ──
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

    # Feature importances
    mean_imp = np.mean(fold_importances, axis=0)
    imp_dict = {name: round(float(imp), 4)
                for name, imp in sorted(zip(feature_names, mean_imp),
                                        key=lambda x: -x[1])}

    metrics = {
        "tag":        tag,
        "n_features": X.shape[1],
        "overall":    _em_f1(),
        "feasible":   _em_f1(qid_filter=feasible),
        "bridge":     _em_f1(type_filter="bridge"),
        "comparison": _em_f1(type_filter="comparison"),
        "ece":        round(ece, 4),
        "feature_importances": imp_dict,
    }

    return metrics, preds


# ─────────────────────────── main ────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop_scores",   required=True)
    ap.add_argument("--evidence",     required=True)
    ap.add_argument("--gold",         required=True)
    ap.add_argument("--out_json",     required=True)
    ap.add_argument("--out_preds",    required=True)
    ap.add_argument("--log",          required=True)
    ap.add_argument("--n_folds",      type=int, default=5)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--xgb_n_estimators", type=int,   default=300)
    ap.add_argument("--xgb_max_depth",    type=int,   default=4)
    ap.add_argument("--xgb_lr",           type=float, default=0.1)
    ap.add_argument("--xgb_subsample",    type=float, default=0.8)
    ap.add_argument("--xgb_colsample",    type=float, default=0.8)
    args = ap.parse_args()

    for p in [args.out_json, args.out_preds, args.log]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    log = logging.getLogger("q11")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(args.log, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)

    log.info("=== Q11 Contrastive Cross-Candidate Verifier ===")

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
    log.info("Loading gold ...")
    gold_map  = {}
    qtype_map = {}
    for ex in json.load(open(args.gold)):
        qid = str(ex["_id"])
        gold_map[qid]  = ex["answer"]
        qtype_map[qid] = ex.get("type", "bridge")

    # ── feasible subset ──
    log.info("Loading feasible subset ...")
    feasible = set()
    for line in open(args.evidence):
        r = json.loads(line)
        if r.get("flags", {}).get("doc_recall_at_k", False):
            feasible.add(str(r["qid"]))
    log.info(f"  feasible={len(feasible)}/7405")

    # ── load hop scores ──
    log.info("Loading hop scores ...")
    hop_records = {}
    for line in open(args.hop_scores):
        r = json.loads(line)
        hop_records[str(r["qid"])] = r
    log.info(f"  {len(hop_records)} records")

    # ── build datasets (baseline + contrastive) ──
    log.info("Building feature matrices ...")
    all_qids_list = []
    all_X_base = []
    all_X_cont = []
    all_y      = []
    qid_cands  = {}

    for qid in sorted(hop_records):
        if qid not in gold_map:
            continue
        rec   = hop_records[qid]
        cands = rec["candidates"]
        gold  = gold_map[qid]

        X_base = extract_features(cands, "min", include_contrastive=False)
        X_cont = extract_features(cands, "min", include_contrastive=True)
        labels = [em(cd["answer"], gold) for cd in cands]

        qid_cands[qid] = [cd["answer"] for cd in cands]

        for ci in range(len(cands)):
            all_qids_list.append(qid)
            all_X_base.append(X_base[ci].tolist())
            all_X_cont.append(X_cont[ci].tolist())
            all_y.append(labels[ci])

    X_base = np.array(all_X_base, dtype=np.float32)
    X_cont = np.array(all_X_cont, dtype=np.float32)
    y      = np.array(all_y, dtype=np.float32)
    qids   = np.array(all_qids_list)

    unique_qids = np.unique(qids)
    log.info(f"  {len(unique_qids)} questions, {len(X_base)} rows")
    log.info(f"  Baseline features: {X_base.shape[1]}")
    log.info(f"  Contrastive features: {X_cont.shape[1]}")
    log.info(f"  Positive rate: {y.mean():.3f}")

    # ── run baseline (18 features) ──
    log.info("Running baseline (18 features) ...")
    metrics_base, preds_base = run_cv(
        unique_qids, qids, X_base, y, qid_cands, gold_map, qtype_map,
        feasible, args.n_folds, args.seed, xgb_params,
        BASELINE_FEATURES, log, "baseline",
    )

    # ── run contrastive (23 features) ──
    log.info("Running contrastive (23 features) ...")
    metrics_cont, preds_cont = run_cv(
        unique_qids, qids, X_cont, y, qid_cands, gold_map, qtype_map,
        feasible, args.n_folds, args.seed, xgb_params,
        ALL_FEATURES, log, "contrastive",
    )

    # ── C2 analysis: where does contrastive change the outcome? ──
    log.info("Computing C2 differential ...")
    wins = losses = ties = 0
    for qid in preds_base:
        b_em = em(preds_base[qid]["pred"], gold_map.get(qid, ""))
        c_em = em(preds_cont[qid]["pred"], gold_map.get(qid, ""))
        if c_em > b_em:
            wins += 1
        elif c_em < b_em:
            losses += 1
        else:
            ties += 1

    # ── save contrastive preds ──
    with open(args.out_preds, "w") as f:
        for qid in sorted(preds_cont):
            f.write(json.dumps({"qid": qid, **preds_cont[qid]}) + "\n")

    # ── decisions ──
    delta_em = metrics_cont["overall"]["em"] - metrics_base["overall"]["em"]
    delta_feasible = metrics_cont["feasible"]["em"] - metrics_base["feasible"]["em"]

    # Check if any contrastive feature in top-5
    top5_cont = list(metrics_cont["feature_importances"].keys())[:5]
    contrastive_in_top5 = [f for f in top5_cont
                           if f in CONTRASTIVE_FEATURES]

    if delta_em >= 0.005 and contrastive_in_top5:
        decision = (f"SUCCESS: +{delta_em:.4f} EM, contrastive features "
                    f"in top-5: {contrastive_in_top5}. "
                    f"Relative ranking helps the verifier.")
    elif delta_em >= 0.003:
        decision = (f"MARGINAL: +{delta_em:.4f} EM. Small gain — "
                    f"contrastive features offer modest help.")
    elif delta_em >= -0.003:
        decision = (f"NO EFFECT: {delta_em:+.4f} EM. "
                    f"C2 failures are NOT a ranking problem. "
                    f"NLI quality is the bottleneck → proceed to NLI fine-tuning.")
    else:
        decision = (f"REGRESSED: {delta_em:+.4f} EM. "
                    f"Extra features hurt (overfitting or noise).")

    # ── assemble summary ──
    summary = {
        "baseline_18feat": metrics_base,
        "contrastive_23feat": metrics_cont,
        "delta": {
            "overall_em":  round(delta_em, 4),
            "feasible_em": round(delta_feasible, 4),
            "overall_f1":  round(metrics_cont["overall"]["f1"] -
                                 metrics_base["overall"]["f1"], 4),
        },
        "per_question_changes": {
            "wins":   wins,
            "losses": losses,
            "ties":   ties,
            "net":    wins - losses,
        },
        "contrastive_features_in_top5": contrastive_in_top5,
        "decision": decision,
        "xgb_params": xgb_params,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to {args.out_json}")

    # ── print results ──
    W = 72
    log.info("=" * W)
    log.info("  Q11 RESULTS — Contrastive Cross-Candidate Features")
    log.info("=" * W)
    log.info(f"  {'Variant':<28} {'Overall EM':>10} {'Feasible EM':>12} "
             f"{'Bridge EM':>10} {'ECE':>8}")
    log.info("  " + "-" * (W - 2))
    for m, label in [(metrics_base, "Baseline (18 feat)"),
                     (metrics_cont, "Contrastive (23 feat)")]:
        log.info(f"  {label:<28} {m['overall']['em']:>10.4f} "
                 f"{m['feasible']['em']:>12.4f} "
                 f"{m['bridge']['em']:>10.4f} {m['ece']:>8.4f}")

    log.info("")
    log.info(f"  Delta EM:  {delta_em:+.4f}  (overall)   "
             f"{delta_feasible:+.4f}  (feasible)")
    log.info(f"  Per-question: +{wins} wins  -{losses} losses  "
             f"={ties} ties  (net: {wins - losses:+d})")

    log.info("")
    log.info("  Feature importances (contrastive model, top-10):")
    for i, (feat, imp) in enumerate(
            list(metrics_cont["feature_importances"].items())[:10]):
        marker = " ◀ NEW" if feat in CONTRASTIVE_FEATURES else ""
        log.info(f"    #{i+1:>2}  {feat:<24}  {imp:.4f}{marker}")

    log.info("")
    log.info(f"  DECISION: {decision}")
    log.info("=" * W)


if __name__ == "__main__":
    main()