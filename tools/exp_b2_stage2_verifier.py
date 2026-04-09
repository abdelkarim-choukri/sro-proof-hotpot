#!/usr/bin/env python3
"""
exp_b2_stage2_verifier.py  —  Phase B2: Stage 2 Chain-Aware Verifier

XGBoost verifier trained ONLY on candidates that survived Stage 1.
With garbage removed, chain-aware features should dominate capacity
instead of being starved by is_bad.

Feature set (19 features — is_bad and is_unknown deliberately excluded):
  Surface (10):
    nli_score, nli_rank, nli_score_gap,
    answer_freq, is_majority,
    answer_len_chars, answer_len_words,
    cand_idx, unique_count, answer_freq_filtered
  NLI chain (3):    nli_hop1, nli_hop2, nli_hop_balance
  QA chain (3):     qa_hop1,  qa_hop2,  qa_hop_balance
  Lex chain (3):    lex_hop1, lex_hop2, lex_hop_balance

Baseline for comparison: 0.4666 (exp0c monolithic XGBoost)

Inputs:
  --hop_scores   exp0c/preds/dev_hop_scores.jsonl
  --qa_scores    exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl
  --lex_scores   exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl
  --stage1_filter exp_phaseB/B1.1/filter_output/dev_stage1_filtered.jsonl
  --evidence     exp0c/evidence/dev_K200_chains.jsonl
  --gold         data/hotpotqa/raw/hotpot_dev_distractor_v1.json
  --out_json     exp_phaseB/B2.1/results/stage2_verifier.json
  --out_preds    exp_phaseB/B2.1/results/dev_stage2_preds.jsonl
  --log          exp_phaseB/B2.1/results/stage2_verifier.log
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
    if not n: return 0.0
    p = n / len(p_toks); r = n / len(g_toks)
    return 2 * p * r / (p + r)


# ─────────────────────────── I/O utils ──────────────────────────────

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def extract_gold(gold_field) -> str:
    if isinstance(gold_field, dict):
        return gold_field.get("answer", "")
    return str(gold_field) if gold_field else ""


# ─────────────────────────── ECE ─────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0; n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0: continue
        ece += mask.sum() / n * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


# ─────────────────────────── feature names ───────────────────────────

SURFACE_FEATURES = [
    "nli_score",           #  0
    "nli_rank",            #  1
    "nli_score_gap",       #  2
    "answer_freq",         #  3  (in full M=5 pool)
    "is_majority",         #  4
    "answer_len_chars",    #  5
    "answer_len_words",    #  6
    "cand_idx",            #  7
    "unique_count",        #  8  (in full M=5 pool)
    "answer_freq_filtered",#  9  (freq within filtered pool only)
]
CHAIN_FEATURES = [
    "nli_hop1",            # 10
    "nli_hop2",            # 11
    "nli_hop_balance",     # 12
    "qa_hop1",             # 13
    "qa_hop2",             # 14
    "qa_hop_balance",      # 15
    "lex_hop1",            # 16
    "lex_hop2",            # 17
    "lex_hop_balance",     # 18
]
FEATURE_NAMES = SURFACE_FEATURES + CHAIN_FEATURES  # 19 total


# ─────────────────────────── feature extraction ──────────────────────

def build_feature_row(
    answer:       str,
    cand_idx_raw: int,      # original position 0-4
    n_full:       int,      # M=5 (full pool size)
    nli_arr_full: np.ndarray,  # NLI scores for all 5 candidates
    freq_counter_full:  collections.Counter,
    majority_norm_full: str,
    unique_count_full:  float,
    freq_counter_filt:  collections.Counter,  # freq in filtered pool only
    n_filt: int,
    nli_hop1:     float,
    nli_hop2:     float,
    qa_hop1:      float,
    qa_hop2:      float,
    qa_hop_balance: float,
    lex_hop1:     float,
    lex_hop2:     float,
    lex_hop_balance: float,
) -> list:
    norm    = normalize(answer)
    nli_val = float(nli_arr_full[cand_idx_raw])
    nli_max = nli_arr_full.max()

    nli_ranks    = np.argsort(-nli_arr_full)
    nli_rank_map = np.empty(n_full, dtype=np.float64)
    nli_rank_map[nli_ranks] = np.arange(n_full) / max(n_full - 1, 1)

    nli_hop_balance = abs(nli_hop1 - nli_hop2)

    return [
        # ── Surface ──────────────────────────────────────────────────
        nli_val,                                         #  0 nli_score
        float(nli_rank_map[cand_idx_raw]),               #  1 nli_rank
        float(nli_max - nli_val),                        #  2 nli_score_gap
        freq_counter_full[norm] / n_full,                #  3 answer_freq
        float(norm == majority_norm_full),               #  4 is_majority
        math.log1p(len(answer)),                         #  5 answer_len_chars
        float(len(answer.split())),                      #  6 answer_len_words
        cand_idx_raw / max(n_full - 1, 1),               #  7 cand_idx
        unique_count_full,                               #  8 unique_count
        freq_counter_filt[norm] / max(n_filt, 1),        #  9 answer_freq_filtered
        # ── Chain features ───────────────────────────────────────────
        nli_hop1,                                        # 10
        nli_hop2,                                        # 11
        nli_hop_balance,                                 # 12
        qa_hop1,                                         # 13
        qa_hop2,                                         # 14
        qa_hop_balance,                                  # 15
        lex_hop1,                                        # 16
        lex_hop2,                                        # 17
        lex_hop_balance,                                 # 18
    ]


# ─────────────────────────── CV runner ───────────────────────────────
# Identical to exp2_q2q3q4_chain_verifier.py

def run_cv(
    all_qids, qids, X, y,
    n_folds, seed, xgb_params, log, tag,
):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(X), dtype=np.float32)
    fold_importances = []

    for fold, (train_qi, val_qi) in enumerate(kf.split(all_qids)):
        train_set  = set(all_qids[train_qi])
        val_set    = set(all_qids[val_qi])
        train_mask = np.array([q in train_set for q in qids])
        val_mask   = np.array([q in val_set   for q in qids])

        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X[train_mask], y[train_mask])
        oof_probs[val_mask] = clf.predict_proba(X[val_mask])[:, 1]
        fold_importances.append(clf.feature_importances_)
        log.info(f"  [{tag}] fold {fold+1}/{n_folds} done")

    return oof_probs, fold_importances


# ─────────────────────────── metrics ─────────────────────────────────

def compute_metrics(oof_probs, qids, qid_cands, gold_map,
                    qtype_map, feasible, tag):
    qid_to_rows = collections.defaultdict(list)
    for i, qid in enumerate(qids):
        qid_to_rows[qid].append(i)

    preds = {}
    for qid, row_idxs in qid_to_rows.items():
        cands  = qid_cands[qid]
        probs  = [float(oof_probs[i]) for i in row_idxs]
        best_i = int(np.argmax(probs))
        preds[qid] = {
            "pred":     cands[best_i],
            "probs":    [round(p, 6) for p in probs],
            "best_idx": best_i,
        }

    def _em_f1(qid_filter=None, type_filter=None):
        e = f = n = 0
        for qid, p in preds.items():
            if qid_filter is not None and qid not in qid_filter: continue
            if type_filter is not None and qtype_map.get(qid) != type_filter: continue
            g = gold_map.get(qid, "")
            e += em(p["pred"], g); f += f1_score(p["pred"], g); n += 1
        if n == 0: return {"n": 0, "em": 0.0, "f1": 0.0}
        return {"n": n, "em": round(e/n, 4), "f1": round(f/n, 4)}

    conf_list  = [max(preds[q]["probs"]) for q in preds]
    label_list = [em(preds[q]["pred"], gold_map.get(q, "")) for q in preds]
    ece = compute_ece(np.array(conf_list), np.array(label_list))

    return {
        "tag":        tag,
        "overall":    _em_f1(),
        "feasible":   _em_f1(qid_filter=feasible),
        "bridge":     _em_f1(type_filter="bridge"),
        "comparison": _em_f1(type_filter="comparison"),
        "ece":        round(ece, 4),
    }, preds


# ─────────────────────────── main ────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Phase B2.1 — Stage 2 chain-aware verifier"
    )
    ap.add_argument("--hop_scores",    required=True)
    ap.add_argument("--qa_scores",     required=True)
    ap.add_argument("--lex_scores",    required=True)
    ap.add_argument("--stage1_filter", required=True,
                    help="exp_phaseB/B1.1/filter_output/dev_stage1_filtered.jsonl")
    ap.add_argument("--evidence",      required=True)
    ap.add_argument("--gold",          required=True)
    ap.add_argument("--out_json",      required=True)
    ap.add_argument("--out_preds",     required=True)
    ap.add_argument("--log",           required=True)
    ap.add_argument("--n_folds",       type=int, default=5)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--xgb_n_estimators", type=int,   default=300)
    ap.add_argument("--xgb_max_depth",    type=int,   default=4)
    ap.add_argument("--xgb_lr",           type=float, default=0.1)
    ap.add_argument("--xgb_subsample",    type=float, default=0.8)
    ap.add_argument("--xgb_colsample",    type=float, default=0.8)
    args = ap.parse_args()

    for p in [args.out_json, args.out_preds, args.log]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # ── logging ──────────────────────────────────────────────────────
    log = logging.getLogger("b2_stage2")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(args.log, mode="w"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout);       sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)

    log.info("=== Phase B2.1: Stage 2 Chain-Aware Verifier ===")
    log.info(f"Features: {len(FEATURE_NAMES)} total  "
             f"({len(SURFACE_FEATURES)} surface + {len(CHAIN_FEATURES)} chain)")
    log.info("Note: is_bad and is_unknown excluded — handled by Stage 1")

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

    # ── Load all inputs ───────────────────────────────────────────────
    log.info("Loading gold + question types...")
    gold_map:  dict = {}
    qtype_map: dict = {}
    for ex in json.load(open(args.gold)):
        qid = str(ex["_id"])
        gold_map[qid]  = ex["answer"]
        qtype_map[qid] = ex.get("type", "bridge")

    log.info("Loading feasible subset...")
    feasible: set = set()
    for line in open(args.evidence):
        r = json.loads(line)
        if r.get("flags", {}).get("doc_recall_at_k", False):
            feasible.add(str(r["qid"]))
    log.info(f"  feasible={len(feasible)}")

    log.info("Loading Stage 1 filter decisions...")
    filter_map: dict = {}  # qid → {surviving_ids: [...]}
    for rec in iter_jsonl(args.stage1_filter):
        filter_map[str(rec["qid"])] = rec
    log.info(f"  {len(filter_map)} questions")

    log.info("Loading NLI hop scores...")
    nli_map: dict = {}
    for rec in iter_jsonl(args.hop_scores):
        nli_map[str(rec["qid"])] = rec["candidates"]

    log.info("Loading QA scores...")
    qa_map: dict = {}
    for rec in iter_jsonl(args.qa_scores):
        qa_map[str(rec["qid"])] = {c["answer_id"]: c for c in rec["candidates"]}

    log.info("Loading lexical features...")
    lex_map: dict = {}
    for rec in iter_jsonl(args.lex_scores):
        lex_map[str(rec["qid"])] = {c["answer_id"]: c for c in rec["candidates"]}

    log.info(f"  nli={len(nli_map)}  qa={len(qa_map)}  lex={len(lex_map)}")

    # ── Build Stage 2 dataset ─────────────────────────────────────────
    log.info("Building Stage 2 feature matrix (filtered candidates only)...")

    valid_qids = (set(nli_map.keys()) & set(qa_map.keys())
                  & set(lex_map.keys()) & set(filter_map.keys())
                  & set(gold_map.keys()))
    log.info(f"  Valid qids: {len(valid_qids)}")

    X_list:   list = []
    y_list:   list = []
    qids_list: list = []
    qid_cands: dict = {}   # qid → surviving answer strings (for prediction)
    n_skipped_all_filtered = 0

    for qid in sorted(valid_qids):
        filt_rec      = filter_map[qid]
        surviving_ids = filt_rec["surviving_ids"]

        if not surviving_ids:
            n_skipped_all_filtered += 1
            continue

        nli_cands = nli_map[qid]   # all 5, indexed by position
        qa_cands  = qa_map[qid]    # dict: answer_id → feats
        lex_cands = lex_map[qid]   # dict: answer_id → feats
        gold      = gold_map[qid]
        n_full    = len(nli_cands)

        # Surface precomputation over full pool (M=5)
        answers_full   = [cd["answer"] for cd in nli_cands]
        norms_full     = [normalize(a) for a in answers_full]
        freq_full      = collections.Counter(norms_full)
        majority_norm  = freq_full.most_common(1)[0][0]
        unique_count   = len(freq_full) / n_full
        nli_arr_full   = np.array([cd["nli_flat"] for cd in nli_cands],
                                   dtype=np.float64)

        # Frequency counter for filtered pool only
        answers_filt  = [answers_full[ci] for ci in surviving_ids]
        norms_filt    = [normalize(a) for a in answers_filt]
        freq_filt     = collections.Counter(norms_filt)
        n_filt        = len(surviving_ids)

        surviving_answers = []
        for ci in surviving_ids:
            answer = answers_full[ci]
            label  = em(answer, gold)

            cd_qa  = qa_cands.get(ci, {})
            cd_lex = lex_cands.get(ci, {})
            cd_nli = nli_cands[ci]

            row = build_feature_row(
                answer       = answer,
                cand_idx_raw = ci,
                n_full       = n_full,
                nli_arr_full = nli_arr_full,
                freq_counter_full  = freq_full,
                majority_norm_full = majority_norm,
                unique_count_full  = unique_count,
                freq_counter_filt  = freq_filt,
                n_filt       = n_filt,
                nli_hop1     = float(cd_nli.get("nli_hop1", 0.0)),
                nli_hop2     = float(cd_nli.get("nli_hop2", 0.0)),
                qa_hop1      = float(cd_qa.get("qa_hop1",   0.0)),
                qa_hop2      = float(cd_qa.get("qa_hop2",   0.0)),
                qa_hop_balance = float(cd_qa.get("qa_hop_balance", 0.0)),
                lex_hop1     = float(cd_lex.get("lex_hop1",  0.0)),
                lex_hop2     = float(cd_lex.get("lex_hop2",  0.0)),
                lex_hop_balance = float(cd_lex.get("lex_hop_balance", 0.0)),
            )
            X_list.append(row)
            y_list.append(label)
            qids_list.append(qid)
            surviving_answers.append(answer)

        qid_cands[qid] = surviving_answers

    X        = np.array(X_list,  dtype=np.float32)
    y        = np.array(y_list,  dtype=np.float32)
    qids_arr = np.array(qids_list)
    unique_q = np.unique(qids_arr)

    log.info(f"  {len(unique_q)} questions  {len(X)} rows  "
             f"{X.shape[1]} features")
    log.info(f"  Skipped (all filtered): {n_skipped_all_filtered}")
    log.info(f"  Positive rate: {y.mean():.3f}")

    # ── Run CV ────────────────────────────────────────────────────────
    log.info("Running 5-fold CV...")
    oof_probs, fold_importances = run_cv(
        unique_q, qids_arr, X, y,
        args.n_folds, args.seed, xgb_params, log, "stage2",
    )

    # ── Compute metrics ───────────────────────────────────────────────
    log.info("Computing metrics...")
    metrics, preds = compute_metrics(
        oof_probs, qids_arr, qid_cands,
        gold_map, qtype_map, feasible, "stage2",
    )

    # ── Feature importances ───────────────────────────────────────────
    mean_imp  = np.mean(fold_importances, axis=0)
    imp_dict  = {
        name: round(float(v), 4)
        for name, v in sorted(
            zip(FEATURE_NAMES, mean_imp), key=lambda x: -x[1]
        )
    }
    names_sorted  = list(imp_dict.keys())
    chain_feat_set = set(CHAIN_FEATURES)

    # Rank of each chain feature
    chain_ranks = {
        f: names_sorted.index(f) + 1
        for f in CHAIN_FEATURES if f in names_sorted
    }

    # ── Compare to baseline ───────────────────────────────────────────
    BASELINE_EM = 0.4666
    delta_pp    = round((metrics["overall"]["em"] - BASELINE_EM) * 100, 2)
    sign        = "+" if delta_pp >= 0 else ""

    # ── Save predictions ──────────────────────────────────────────────
    with open(args.out_preds, "w") as f:
        for qid in sorted(preds):
            f.write(json.dumps({"qid": qid, **preds[qid]}) + "\n")

    # ── Save summary ──────────────────────────────────────────────────
    output = {
        "baseline_em":    BASELINE_EM,
        "stage2_em":      metrics["overall"]["em"],
        "delta_pp":       delta_pp,
        "metrics":        metrics,
        "feature_importances": imp_dict,
        "chain_feature_ranks": chain_ranks,
        "n_questions_trained": int(len(unique_q)),
        "n_skipped_all_filtered": n_skipped_all_filtered,
        "xgb_params":     xgb_params,
    }
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)

    # ── Print results ─────────────────────────────────────────────────
    W = 72
    log.info("")
    log.info("=" * W)
    log.info("  PHASE B2.1 — STAGE 2 RESULTS")
    log.info("=" * W)
    log.info(f"  Monolithic baseline (exp0c):  {BASELINE_EM:.4f}")
    log.info(f"  Stage 2 chain-aware:          "
             f"{metrics['overall']['em']:.4f}  "
             f"({sign}{delta_pp}pp)")
    log.info(f"  Bridge EM:    {metrics['bridge']['em']:.4f}")
    log.info(f"  Comparison EM:{metrics['comparison']['em']:.4f}")
    log.info(f"  Feasible EM:  {metrics['feasible']['em']:.4f}")
    log.info(f"  ECE:          {metrics['ece']:.4f}")
    log.info("=" * W)

    log.info("\n  Feature importances (top-12):")
    for i, (feat, val) in enumerate(list(imp_dict.items())[:12]):
        marker = " ◀ chain" if feat in chain_feat_set else ""
        log.info(f"    #{i+1:>2}  {feat:<25}  {val:.4f}{marker}")

    log.info(f"\n  Chain feature ranks:")
    for feat, rank in sorted(chain_ranks.items(), key=lambda x: x[1]):
        log.info(f"    #{rank:>2}  {feat}")

    log.info(f"\n  Results saved: {args.out_json}")


if __name__ == "__main__":
    main()