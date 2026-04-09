#!/usr/bin/env python3
"""
exp_a2_principle_table.py  —  Phase A2.1: Flat vs Chain-Aware Ablation Table

Trains 6 XGBoost models (3 scorers × 2 conditions) and produces the
principle table for the paper:

  | Scorer  | Flat EM | Chain-Aware EM | Δ (pp) | hop_balance Rank |
  |---------|---------|----------------|--------|-----------------|
  | NLI     | ...     | ...            | ...    | ...             |
  | QA      | ...     | ...            | ...    | ...             |
  | Lexical | ...     | ...            | ...    | ...             |

SUCCESS CRITERIA (Phase A decision gate):
  PRIMARY:   All three Δ values positive → chain-aware beats flat for every scorer
  SECONDARY: hop_balance (or equiv) in top-5 for ≥2 of 3 scorers

Same XGBoost setup as exp2_q2q3q4_chain_verifier.py:
  5-fold CV, 300 estimators, depth=4, seed=42, question-level splits

Surface features are identical across all 6 models — the ONLY thing
that changes between flat and chain-aware is the scorer's features.

Inputs:
  --hop_scores   exp0c/preds/dev_hop_scores.jsonl        (NLI features)
  --qa_scores    exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl
  --lex_scores   exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl
  --evidence     exp0c/evidence/dev_K200_chains.jsonl    (feasible flag)
  --gold         data/hotpotqa/raw/hotpot_dev_distractor_v1.json
  --out_json     exp_phaseA/A2.1/principle_table.json
  --log          exp_phaseA/A2.1/principle_table.log

Usage:
    python3 tools/exp_a2_principle_table.py \
        --hop_scores  exp0c/preds/dev_hop_scores.jsonl \
        --qa_scores   exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl \
        --lex_scores  exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl \
        --evidence    exp0c/evidence/dev_K200_chains.jsonl \
        --gold        data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
        --out_json    exp_phaseA/A2.1/principle_table.json \
        --log         exp_phaseA/A2.1/principle_table.log
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
# Identical to exp2_q2q3q4_chain_verifier.py

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

def is_bad_answer(ans: str) -> bool:
    """Identical to exp2_q2q3q4_chain_verifier.py."""
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
    ece  = 0.0
    n    = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


# ─────────────────────────── feature sets ────────────────────────────

# Surface features — identical in all 6 models
SURFACE_FEATURES = [
    "nli_score",         # flat NLI entailment probability (from hop_scores)
    "nli_rank",          # rank among candidates (normalised 0-1)
    "nli_score_gap",     # max_nli - this_nli
    "answer_freq",       # fraction of candidates with same normalised answer
    "is_majority",       # 1 if plurality answer
    "answer_len_chars",  # log(1 + char length)
    "answer_len_words",  # word count
    "is_empty",
    "is_unknown",
    "is_bad",
    "cand_idx",          # position 0-4 normalised
    "unique_count",      # distinct answers / M
]

# Per-scorer feature names for flat and chain-aware conditions
SCORER_FEATURES = {
    "nli": {
        "flat":        ["nli_flat_score"],
        "chain_aware": ["nli_hop1", "nli_hop2", "nli_hop_balance"],
        "balance_key": "nli_hop_balance",
    },
    "qa": {
        "flat":        ["qa_flat"],
        "chain_aware": ["qa_hop1", "qa_hop2", "qa_hop_balance"],
        "balance_key": "qa_hop_balance",
    },
    "lex": {
        "flat":        ["lex_flat"],
        "chain_aware": ["lex_hop1", "lex_hop2", "lex_hop_balance"],
        "balance_key": "lex_hop_balance",
    },
}


# ─────────────────────────── feature extraction ──────────────────────

def build_surface_row(
    answer: str,
    cand_idx: int,
    n_cands: int,
    nli_arr: np.ndarray,
    freq_counter: collections.Counter,
    majority_norm: str,
    unique_count: float,
) -> list:
    """Compute the 12 surface features for one candidate.
    Identical logic to exp2_q2q3q4_chain_verifier.py."""
    norm     = normalize(answer)
    nli_max  = nli_arr.max()
    nli_val  = nli_arr[cand_idx]

    # NLI rank (normalised)
    nli_ranks    = np.argsort(-nli_arr)
    nli_rank_map = np.empty(n_cands, dtype=np.float64)
    nli_rank_map[nli_ranks] = np.arange(n_cands) / max(n_cands - 1, 1)

    freq    = freq_counter[norm] / n_cands
    is_maj  = int(norm == majority_norm)
    alen_c  = math.log1p(len(answer))
    alen_w  = len(answer.split())
    i_empty = int(not answer.strip())
    i_unk   = int(norm in {"unknown", "unk", ""})
    i_bad   = int(is_bad_answer(answer))
    c_idx   = cand_idx / max(n_cands - 1, 1)
    nli_gap = float(nli_max - nli_val)

    return [
        float(nli_val),              #  0 nli_score
        float(nli_rank_map[cand_idx]),#  1 nli_rank
        nli_gap,                     #  2 nli_score_gap
        freq,                        #  3 answer_freq
        float(is_maj),               #  4 is_majority
        alen_c,                      #  5 answer_len_chars
        float(alen_w),               #  6 answer_len_words
        float(i_empty),              #  7 is_empty
        float(i_unk),                #  8 is_unknown
        float(i_bad),                #  9 is_bad
        c_idx,                       # 10 cand_idx
        unique_count,                # 11 unique_count
    ]


def build_scorer_row_flat(scorer: str, cand_feats: dict) -> list:
    """One flat scorer feature for this candidate."""
    if scorer == "nli":
        return [float(cand_feats.get("nli_flat", 0.0))]
    elif scorer == "qa":
        return [float(cand_feats.get("qa_flat", 0.0))]
    else:  # lex
        return [float(cand_feats.get("lex_flat", 0.0))]


def build_scorer_row_chain(scorer: str, cand_feats: dict) -> list:
    """Three chain-aware scorer features for this candidate."""
    if scorer == "nli":
        h1  = float(cand_feats.get("nli_hop1", 0.0))
        h2  = float(cand_feats.get("nli_hop2", 0.0))
        bal = abs(h1 - h2)
        return [h1, h2, bal]
    elif scorer == "qa":
        return [
            float(cand_feats.get("qa_hop1", 0.0)),
            float(cand_feats.get("qa_hop2", 0.0)),
            float(cand_feats.get("qa_hop_balance", 0.0)),
        ]
    else:  # lex
        return [
            float(cand_feats.get("lex_hop1", 0.0)),
            float(cand_feats.get("lex_hop2", 0.0)),
            float(cand_feats.get("lex_hop_balance", 0.0)),
        ]


# ─────────────────────────── CV runner ───────────────────────────────
# Identical to exp2_q2q3q4_chain_verifier.py

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
) -> tuple:
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
        log.info(f"  [{tag}] fold {fold+1}/{n_folds} done")

    return oof_probs, fold_importances


# ─────────────────────────── metrics ─────────────────────────────────

def compute_metrics(
    oof_probs: np.ndarray,
    qids: np.ndarray,
    qid_cands: dict,
    gold_map: dict,
    qtype_map: dict,
    feasible: set,
    tag: str,
) -> tuple:
    qid_to_rows = collections.defaultdict(list)
    for i, qid in enumerate(qids):
        qid_to_rows[qid].append(i)

    preds = {}
    for qid, row_idxs in qid_to_rows.items():
        cands  = qid_cands[qid]
        probs  = [float(oof_probs[i]) for i in row_idxs]
        best_i = int(np.argmax(probs))
        preds[qid] = {"pred": cands[best_i], "probs": probs}

    def _em_f1(qid_filter=None, type_filter=None):
        e = f = n = 0
        for qid, p in preds.items():
            if qid_filter is not None and qid not in qid_filter:
                continue
            if type_filter is not None and qtype_map.get(qid) != type_filter:
                continue
            g  = gold_map.get(qid, "")
            e += em(p["pred"], g)
            f += f1_score(p["pred"], g)
            n += 1
        if n == 0:
            return {"n": 0, "em": 0.0, "f1": 0.0}
        return {"n": n, "em": round(e / n, 4), "f1": round(f / n, 4)}

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
        description="Phase A2.1 — Flat vs Chain-Aware Ablation Table"
    )
    ap.add_argument("--hop_scores",  required=True,
                    help="exp0c/preds/dev_hop_scores.jsonl")
    ap.add_argument("--qa_scores",   required=True,
                    help="exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl")
    ap.add_argument("--lex_scores",  required=True,
                    help="exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl")
    ap.add_argument("--evidence",    required=True,
                    help="exp0c/evidence/dev_K200_chains.jsonl")
    ap.add_argument("--gold",        required=True,
                    help="data/hotpotqa/raw/hotpot_dev_distractor_v1.json")
    ap.add_argument("--out_json",    required=True)
    ap.add_argument("--log",         required=True)
    ap.add_argument("--n_folds",     type=int, default=5)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--xgb_n_estimators", type=int,   default=300)
    ap.add_argument("--xgb_max_depth",    type=int,   default=4)
    ap.add_argument("--xgb_lr",           type=float, default=0.1)
    ap.add_argument("--xgb_subsample",    type=float, default=0.8)
    ap.add_argument("--xgb_colsample",    type=float, default=0.8)
    args = ap.parse_args()

    for p in [args.out_json, args.log]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # ── logging ──────────────────────────────────────────────────────
    log = logging.getLogger("a2_principle")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(args.log, mode="w"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout);       sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)

    log.info("=== Phase A2.1: Flat vs Chain-Aware Ablation Table ===")

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

    # ── Load gold + question types ────────────────────────────────────
    log.info("Loading gold + question types...")
    gold_map:  dict = {}
    qtype_map: dict = {}
    for ex in json.load(open(args.gold)):
        qid = str(ex["_id"])
        gold_map[qid]  = ex["answer"]
        qtype_map[qid] = ex.get("type", "bridge")
    log.info(f"  {len(gold_map)} questions  "
             f"bridge={sum(1 for t in qtype_map.values() if t=='bridge')}  "
             f"comparison={sum(1 for t in qtype_map.values() if t=='comparison')}")

    # ── Load feasible subset ──────────────────────────────────────────
    log.info("Loading feasible subset...")
    feasible: set = set()
    for line in open(args.evidence):
        r   = json.loads(line)
        qid = str(r["qid"])
        if r.get("flags", {}).get("doc_recall_at_k", False):
            feasible.add(qid)
    log.info(f"  feasible={len(feasible)}/7405")

    # ── Load NLI hop scores ───────────────────────────────────────────
    # Schema: {qid, candidates[i]: {cand_idx, answer, nli_flat, nli_hop1,
    #          nli_hop2, min_hop, mean_hop, imbalance}}
    log.info("Loading NLI hop scores...")
    nli_map: dict = {}  # qid → list of candidate dicts (by position)
    for line in open(args.hop_scores):
        r   = json.loads(line)
        qid = str(r["qid"])
        nli_map[qid] = r["candidates"]
    log.info(f"  {len(nli_map)} questions")

    # ── Load QA hop scores ────────────────────────────────────────────
    # Schema: {qid, candidates[i]: {answer_id, answer_text, qa_hop1,
    #          qa_hop2, qa_flat, qa_hop_balance, qa_min_hop}}
    log.info("Loading QA hop scores...")
    qa_map: dict = {}   # qid → {answer_id → feature dict}
    for line in open(args.qa_scores):
        r   = json.loads(line)
        qid = str(r["qid"])
        qa_map[qid] = {c["answer_id"]: c for c in r["candidates"]}
    log.info(f"  {len(qa_map)} questions")

    # ── Load lexical features ─────────────────────────────────────────
    # Schema: {qid, candidates[i]: {answer_id, answer_text, ans_in_hop1,
    #          ans_in_hop2, lex_hop1, lex_hop2, lex_flat, lex_hop_balance, ...}}
    log.info("Loading lexical features...")
    lex_map: dict = {}  # qid → {answer_id → feature dict}
    for line in open(args.lex_scores):
        r   = json.loads(line)
        qid = str(r["qid"])
        lex_map[qid] = {c["answer_id"]: c for c in r["candidates"]}
    log.info(f"  {len(lex_map)} questions")

    # ── Build datasets for all 6 models ──────────────────────────────
    log.info("Building feature matrices for 6 models...")

    # All qids present in all four sources
    valid_qids = (set(nli_map.keys()) & set(qa_map.keys())
                  & set(lex_map.keys()) & set(gold_map.keys()))
    log.info(f"  Valid qids (in all sources): {len(valid_qids)}")

    # Storage: 6 models × rows
    # Model keys: nli_flat, nli_chain, qa_flat, qa_chain, lex_flat, lex_chain
    model_keys = [
        "nli_flat", "nli_chain",
        "qa_flat",  "qa_chain",
        "lex_flat", "lex_chain",
    ]
    X_all = {k: [] for k in model_keys}
    y_all       = []
    qids_all    = []
    qid_cands   = {}  # qid → list of answer strings (for prediction)

    for qid in sorted(valid_qids):
        nli_cands = nli_map[qid]   # list, indexed by position
        qa_cands  = qa_map[qid]    # dict: answer_id → feats
        lex_cands = lex_map[qid]   # dict: answer_id → feats

        n = len(nli_cands)
        if n == 0:
            continue

        # Answer strings for this question (from NLI file, field = "answer")
        answers = [cd["answer"] for cd in nli_cands]
        gold    = gold_map[qid]
        labels  = [em(a, gold) for a in answers]

        # Surface feature precomputation (shared across all 6 models)
        norms         = [normalize(a) for a in answers]
        freq_counter  = collections.Counter(norms)
        majority_norm = freq_counter.most_common(1)[0][0]
        unique_count  = len(freq_counter) / n
        nli_arr       = np.array([cd["nli_flat"] for cd in nli_cands],
                                  dtype=np.float64)

        for ci, cd_nli in enumerate(nli_cands):
            answer = answers[ci]

            # Surface row (same for all 6 models)
            surf = build_surface_row(
                answer, ci, n, nli_arr,
                freq_counter, majority_norm, unique_count,
            )

            # NLI features
            nli_feats = {
                "nli_flat":  float(cd_nli.get("nli_flat",  0.0)),
                "nli_hop1":  float(cd_nli.get("nli_hop1",  0.0)),
                "nli_hop2":  float(cd_nli.get("nli_hop2",  0.0)),
            }

            # QA features — join by answer_id = ci
            cd_qa = qa_cands.get(ci, {})
            qa_feats = {
                "qa_flat":        float(cd_qa.get("qa_flat",        0.0)),
                "qa_hop1":        float(cd_qa.get("qa_hop1",        0.0)),
                "qa_hop2":        float(cd_qa.get("qa_hop2",        0.0)),
                "qa_hop_balance": float(cd_qa.get("qa_hop_balance", 0.0)),
            }

            # Lex features — join by answer_id = ci
            cd_lex = lex_cands.get(ci, {})
            lex_feats = {
                "lex_flat":        float(cd_lex.get("lex_flat",        0.0)),
                "lex_hop1":        float(cd_lex.get("lex_hop1",        0.0)),
                "lex_hop2":        float(cd_lex.get("lex_hop2",        0.0)),
                "lex_hop_balance": float(cd_lex.get("lex_hop_balance", 0.0)),
            }

            # Build rows for all 6 models
            X_all["nli_flat"].append(surf  + [nli_feats["nli_flat"]])
            X_all["nli_chain"].append(surf + [nli_feats["nli_hop1"],
                                              nli_feats["nli_hop2"],
                                              abs(nli_feats["nli_hop1"]
                                                  - nli_feats["nli_hop2"])])
            X_all["qa_flat"].append(surf   + [qa_feats["qa_flat"]])
            X_all["qa_chain"].append(surf  + [qa_feats["qa_hop1"],
                                              qa_feats["qa_hop2"],
                                              qa_feats["qa_hop_balance"]])
            X_all["lex_flat"].append(surf  + [lex_feats["lex_flat"]])
            X_all["lex_chain"].append(surf + [lex_feats["lex_hop1"],
                                              lex_feats["lex_hop2"],
                                              lex_feats["lex_hop_balance"]])
            y_all.append(labels[ci])
            qids_all.append(qid)

        qid_cands[qid] = answers

    # Convert to numpy
    X = {k: np.array(v, dtype=np.float32) for k, v in X_all.items()}
    y         = np.array(y_all, dtype=np.float32)
    qids_arr  = np.array(qids_all)
    unique_q  = np.unique(qids_arr)

    log.info(f"  {len(unique_q)} questions, {len(y)} rows")
    log.info(f"  Feature dims: flat=13  chain-aware=14  "
             f"(12 surface + 1 flat  OR  12 surface + 3 chain)")
    log.info(f"  Positive rate: {y.mean():.3f}")

    # ── Run CV for all 6 models ───────────────────────────────────────
    results = {}
    feat_names = {
        "nli_flat":  SURFACE_FEATURES + ["nli_flat_score"],
        "nli_chain": SURFACE_FEATURES + ["nli_hop1", "nli_hop2", "nli_hop_balance"],
        "qa_flat":   SURFACE_FEATURES + ["qa_flat"],
        "qa_chain":  SURFACE_FEATURES + ["qa_hop1", "qa_hop2", "qa_hop_balance"],
        "lex_flat":  SURFACE_FEATURES + ["lex_flat"],
        "lex_chain": SURFACE_FEATURES + ["lex_hop1", "lex_hop2", "lex_hop_balance"],
    }

    for key in model_keys:
        log.info(f"Running CV: {key} ...")
        oof, imps = run_cv(
            unique_q, qids_arr, X[key], y,
            args.n_folds, args.seed, xgb_params, log, key,
        )
        metrics, preds = compute_metrics(
            oof, qids_arr, qid_cands,
            gold_map, qtype_map, feasible, key,
        )
        # Feature importances
        mean_imp = np.mean(imps, axis=0)
        imp_dict = {
            name: round(float(v), 4)
            for name, v in sorted(
                zip(feat_names[key], mean_imp), key=lambda x: -x[1]
            )
        }
        # Rank of hop_balance feature (1-indexed)
        names_sorted = list(imp_dict.keys())
        balance_key  = {"nli_flat": None, "nli_chain": "nli_hop_balance",
                        "qa_flat":  None, "qa_chain":  "qa_hop_balance",
                        "lex_flat": None, "lex_chain": "lex_hop_balance"}[key]
        balance_rank = (names_sorted.index(balance_key) + 1
                        if balance_key and balance_key in names_sorted else None)

        results[key] = {
            "metrics":       metrics,
            "importances":   imp_dict,
            "balance_rank":  balance_rank,
        }
        log.info(f"  {key}: overall_em={metrics['overall']['em']:.4f}  "
                 f"balance_rank={balance_rank}")

    # ── Compute deltas and build principle table ──────────────────────
    BASELINE_EM = 0.4666   # exp0c mean_pooling — our locked baseline

    scorers = ["nli", "qa", "lex"]
    table_rows = []
    for scorer in scorers:
        flat_em  = results[f"{scorer}_flat"]["metrics"]["overall"]["em"]
        chain_em = results[f"{scorer}_chain"]["metrics"]["overall"]["em"]
        delta_pp = round((chain_em - flat_em) * 100, 2)
        b_rank   = results[f"{scorer}_chain"]["balance_rank"]
        table_rows.append({
            "scorer":      scorer,
            "flat_em":     flat_em,
            "chain_em":    chain_em,
            "delta_pp":    delta_pp,
            "balance_rank": b_rank,
        })

    # ── Decision gate ─────────────────────────────────────────────────
    all_positive    = all(r["delta_pp"] > 0 for r in table_rows)
    balance_top5    = sum(1 for r in table_rows
                         if r["balance_rank"] is not None
                         and r["balance_rank"] <= 5)
    gate_passed     = all_positive and balance_top5 >= 2

    gate_str = "PASSED" if gate_passed else "FAILED"
    reasons  = []
    if not all_positive:
        failing = [r["scorer"] for r in table_rows if r["delta_pp"] <= 0]
        reasons.append(f"Non-positive Δ for: {failing}")
    if balance_top5 < 2:
        reasons.append(f"hop_balance top-5 in only {balance_top5}/3 scorers")
    if gate_passed:
        reasons.append("All Δ positive + hop_balance top-5 in ≥2 scorers → proceed to Phase B")

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "baseline_em":    BASELINE_EM,
        "principle_table": table_rows,
        "decision_gate":  {
            "result":  gate_str,
            "reasons": reasons,
            "all_positive_delta":  all_positive,
            "balance_top5_count":  balance_top5,
        },
        "full_results": {k: {
            "metrics":     v["metrics"],
            "importances": v["importances"],
            "balance_rank": v["balance_rank"],
        } for k, v in results.items()},
        "xgb_params": xgb_params,
    }

    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results saved to {args.out_json}")

    # ── Print principle table ─────────────────────────────────────────
    W = 72
    log.info("")
    log.info("=" * W)
    log.info("  PHASE A2.1 — PRINCIPLE TABLE")
    log.info("  Chain-aware decomposition vs flat scoring")
    log.info("=" * W)
    log.info(f"  {'Scorer':<10} {'Flat EM':>9} {'Chain EM':>10} "
             f"{'Δ (pp)':>8} {'bal_rank':>10}")
    log.info("  " + "-" * (W - 2))
    for r in table_rows:
        sign   = "+" if r["delta_pp"] > 0 else ""
        br_str = f"#{r['balance_rank']}" if r["balance_rank"] else "N/A"
        log.info(f"  {r['scorer']:<10} {r['flat_em']:>9.4f} "
                 f"{r['chain_em']:>10.4f} "
                 f"{sign}{r['delta_pp']:>7.2f}pp "
                 f"{br_str:>10}")
    log.info("  " + "-" * (W - 2))
    log.info(f"  Baseline (exp0c, locked): {BASELINE_EM:.4f}")
    log.info("=" * W)
    log.info(f"  DECISION GATE: {gate_str}")
    for r in reasons:
        log.info(f"    → {r}")
    log.info("=" * W)

    # ── Print top-5 features per chain-aware model ────────────────────
    log.info("")
    log.info("  Feature importances — chain-aware models (top-8):")
    for scorer in scorers:
        key  = f"{scorer}_chain"
        imps = results[key]["importances"]
        log.info(f"  [{scorer}]")
        for i, (feat, val) in enumerate(list(imps.items())[:8]):
            marker = " ◀" if any(feat.endswith(s) for s in
                                  ["hop1", "hop2", "balance", "flat"]) else ""
            log.info(f"    #{i+1:>2}  {feat:<25}  {val:.4f}{marker}")


if __name__ == "__main__":
    main()