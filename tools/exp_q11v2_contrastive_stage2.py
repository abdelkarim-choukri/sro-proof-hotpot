#!/usr/bin/env python3
"""
exp_q11v2_contrastive_stage2.py — Corrected Q11: Contrastive Cross-Candidate
                                  Features on Stage-2 Filtered Pool

Original Q11 had four methodological flaws (see Q11v2 design notes):
  1. Run on monolithic XGBoost where is_bad/is_unknown consumed 67-89% of
     model capacity, starving chain features.
  2. Contrastive features were NLI-only, missing QA (the strongest scorer per
     Z3 importances: qa_hop2 = 16.5%, qa_hop_balance = 11.9%).
  3. "Best" reference for delta features was computed over UNFILTERED
     candidates, including garbage. delta_to_best was a noise-floor compare.
  4. Decision threshold was set on global EM (insensitive to C2-only effects).

This corrected version:
  - Trains and evaluates on Stage-1 SURVIVING candidates only (no is_bad/is_unknown).
  - Adds 12 contrastive features spanning all THREE scoring modalities (NLI/QA/lex).
  - "Best" computed only over surviving candidates (competitive baseline).
  - Reports overall ΔEM AND C2-specific ΔEM AND feature importance ranks.

Two ablation variants run per setting:
  baseline_z3       — 9 chain features (cand_idx + nli/qa/lex hop1/hop2/balance)
                      This is the Z3 schema from phase0_ablations.py
  contrastive_z3plus — 9 chain features + 12 contrastive = 21 total

Settings (run twice — once per setting):
  hotpotqa_mdr_7b_m5    (deployment regime)
  hotpotqa_mdr_1p5b_m5  (noisier-generator regime where chain features matter most)

Inputs (no GPU, no inference — reads precomputed scores):
  --hop_scores      dev_hop_scores.jsonl   (NLI per-hop)
  --qa_scores       dev_qa_hop_scores.jsonl  (QA per-hop)
  --lex_features    dev_lex_features.jsonl   (lexical per-hop)
  --gold            hotpot_dev_distractor_v1.json
  --candidates      dev_M5_candidates_*.jsonl  (raw answer strings, for is_bad)
  --setting_label   short tag for output JSON

Output:
  --out_json        per-setting metrics JSON in the schema agreed with reviewer

Usage:
  python3 tools/exp_q11v2_contrastive_stage2.py \\
      --hop_scores      exp5b/preds/dev_hop_scores.jsonl \\
      --qa_scores       exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl \\
      --lex_features    exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl \\
      --candidates      exp5b/candidates/dev_M5_7b_hightemp.jsonl \\
      --gold            data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
      --setting_label   hotpotqa_mdr_7b_m5 \\
      --out_json        exp_q11v2/hotpotqa_mdr_7b_m5_results.json \\
      --n_folds         5 --seed 42

Runs in 2-3 minutes per setting on CPU.
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
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Soft import — fail with a clear message if xgboost missing
try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. pip install xgboost", file=sys.stderr)
    sys.exit(1)

from sklearn.model_selection import KFold


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: TEXT NORMALIZATION (matches phase0_ablations.py exactly)
# ═══════════════════════════════════════════════════════════════════════

def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: GARBAGE FILTER (Stage 1 rules, applied deterministically)
# ═══════════════════════════════════════════════════════════════════════
# These are the exact rules from exp_b1_stage1_filter.py / phase0_ablations.py.
# Logistic regression on these 4 features hits training accuracy 1.0000
# with FN=0 in the project's own validation. Apply rules directly = same result.

def is_bad_answer(ans: str) -> bool:
    """Lifted from exp2_q2q3q4_chain_verifier.py."""
    a = (ans or "").strip()
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


def is_unknown_answer(ans: str) -> bool:
    """Raw-string check (the post-fix version from chain_aware_verification_report)."""
    a = (ans or "").strip().lower()
    return a in {"unknown", "unk", ""}


def survives_stage1(ans: str) -> bool:
    return not (is_bad_answer(ans) or is_unknown_answer(ans))


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def iter_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_gold(path: str) -> Dict[str, Dict]:
    """HotpotQA distractor JSON. Returns {qid: {answer, type}}."""
    with open(path) as f:
        data = json.load(f)
    out = {}
    for ex in data:
        qid = ex.get("_id") or ex.get("id")
        out[str(qid)] = {
            "answer": ex.get("answer", ""),
            "type":   ex.get("type", "bridge"),
        }
    return out


def load_hop_scores(path: str) -> Dict[str, Dict]:
    """NLI per-hop scores. Returns {qid: record}."""
    out = {}
    for rec in iter_jsonl(path):
        out[str(rec["qid"])] = rec
    return out


def load_qa_scores(path: str) -> Dict[str, Dict]:
    """QA per-hop scores. Returns {qid: record}."""
    out = {}
    for rec in iter_jsonl(path):
        out[str(rec["qid"])] = rec
    return out


def load_lex_features(path: str) -> Dict[str, Dict]:
    """Lex per-hop features. Returns {qid: record}."""
    out = {}
    for rec in iter_jsonl(path):
        out[str(rec["qid"])] = rec
    return out


def load_candidates(path: str) -> Dict[str, List[str]]:
    """Raw candidate answer strings. Returns {qid: [ans0, ans1, ...]}.

    Supports two schemas:
      schema A (exp1b/exp5b): {qid, candidates: [{answer_id, answer_text}, ...]}
      schema B (exp3b/exp4_7b): {qid, candidates: [{cand_idx, answer}, ...]}
    """
    out = {}
    for rec in iter_jsonl(path):
        qid = str(rec["qid"])
        cands = rec.get("candidates", [])
        # Try schema A first, fall back to B
        answers = []
        for c in cands:
            if "answer_text" in c:
                answers.append(c["answer_text"])
            elif "answer" in c:
                answers.append(c["answer"])
            else:
                answers.append("")
        out[qid] = answers
    return out


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: FEATURE BUILDING
# ═══════════════════════════════════════════════════════════════════════

# Z3 chain features (matches phase0_ablations.py CHAIN_FEATURES exactly,
# plus cand_idx so the model has positional info — Z3 uses these 10).
CHAIN_FEATURES = [
    "cand_idx",
    "nli_hop1", "nli_hop2", "nli_hop_balance",
    "qa_hop1",  "qa_hop2",  "qa_hop_balance",
    "lex_hop1", "lex_hop2", "lex_hop_balance",
]

# 12 NEW contrastive features spanning all three modalities
CONTRASTIVE_FEATURES = [
    # NLI ranks & deltas
    "nli_hop1_rank",
    "nli_hop2_rank",
    "nli_balance_rank",
    "delta_to_best_nli_hop1",
    "delta_to_best_nli_hop2",
    # QA ranks & deltas (the strongest modality per Z3)
    "qa_hop1_rank",
    "qa_hop2_rank",
    "qa_balance_rank",
    "delta_to_best_qa_hop1",
    "delta_to_best_qa_hop2",
    # Lex (only hop2 — strongest lex feature in Z3)
    "lex_hop2_rank",
    "delta_to_best_lex_hop2",
]

ALL_FEATURES = CHAIN_FEATURES + CONTRASTIVE_FEATURES  # 22 features


def safe_get(rec: Dict, candidates_key: str, idx: int, field: str,
             default: float = 0.0) -> float:
    """Pull a per-candidate field with safe fallback."""
    cands = rec.get(candidates_key, [])
    if idx >= len(cands):
        return default
    val = cands[idx].get(field, default)
    if val is None:
        return default
    return float(val)


def build_per_question_features(
    qid: str,
    answers: List[str],
    hop_rec: Dict,
    qa_rec: Dict,
    lex_rec: Dict,
    surviving_idx: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Build feature matrix for SURVIVING candidates only.

    Returns:
      X_chain  (n_surv, 10) — Z3 chain features
      X_contra (n_surv, 12) — contrastive features (computed over survivors)
      original_idx (list)   — position in original M=5 list
    """
    M = len(answers)

    # Pull raw per-hop scores for ALL M candidates from each source
    nli_hop1   = [safe_get(hop_rec, "candidates", i, "nli_hop1", 0.0) for i in range(M)]
    nli_hop2   = [safe_get(hop_rec, "candidates", i, "nli_hop2", 0.0) for i in range(M)]
    qa_hop1    = [safe_get(qa_rec,  "candidates", i, "qa_hop1",  0.0) for i in range(M)]
    qa_hop2    = [safe_get(qa_rec,  "candidates", i, "qa_hop2",  0.0) for i in range(M)]

    # Lex features: schema may differ. Read the whole record once.
    lex_cands = lex_rec.get("candidates", [])
    def lex_field(i, key):
        if i >= len(lex_cands):
            return 0.0
        v = lex_cands[i].get(key, 0.0)
        return 0.0 if v is None else float(v)

    lex_hop1 = [lex_field(i, "lex_hop1") for i in range(M)]
    lex_hop2 = [lex_field(i, "lex_hop2") for i in range(M)]

    # Balance features per scorer
    nli_balance = [abs(nli_hop1[i] - nli_hop2[i]) for i in range(M)]
    qa_balance  = [abs(qa_hop1[i]  - qa_hop2[i])  for i in range(M)]
    lex_balance = [abs(lex_hop1[i] - lex_hop2[i]) for i in range(M)]

    # ── Stage-2 restriction: compute ranks/deltas over SURVIVORS ONLY ──
    if not surviving_idx:
        # No survivors — return empty matrices (caller will skip this qid)
        return np.zeros((0, 10), dtype=np.float32), np.zeros((0, 12), dtype=np.float32), []

    n_surv = len(surviving_idx)

    def rank_among_survivors(values: List[float], higher_is_better: bool = True) -> Dict[int, float]:
        """Return {orig_idx: rank in [0,1]} over survivors. 0 = best."""
        sub = [(i, values[i]) for i in surviving_idx]
        sub.sort(key=lambda x: -x[1] if higher_is_better else x[1])
        out = {}
        for r, (i, _) in enumerate(sub):
            out[i] = r / max(n_surv - 1, 1)
        return out

    def best_among_survivors(values: List[float]) -> float:
        return max(values[i] for i in surviving_idx)

    rank_nli_h1   = rank_among_survivors(nli_hop1, higher_is_better=True)
    rank_nli_h2   = rank_among_survivors(nli_hop2, higher_is_better=True)
    # Balance: lower is better (more balanced = better)
    rank_nli_bal  = rank_among_survivors(nli_balance, higher_is_better=False)
    rank_qa_h1    = rank_among_survivors(qa_hop1, higher_is_better=True)
    rank_qa_h2    = rank_among_survivors(qa_hop2, higher_is_better=True)
    rank_qa_bal   = rank_among_survivors(qa_balance, higher_is_better=False)
    rank_lex_h2   = rank_among_survivors(lex_hop2, higher_is_better=True)

    best_nli_h1 = best_among_survivors(nli_hop1)
    best_nli_h2 = best_among_survivors(nli_hop2)
    best_qa_h1  = best_among_survivors(qa_hop1)
    best_qa_h2  = best_among_survivors(qa_hop2)
    best_lex_h2 = best_among_survivors(lex_hop2)

    chain_rows = []
    contra_rows = []
    out_idx = []

    for i in surviving_idx:
        # Z3 chain features (10)
        chain_rows.append([
            i / max(M - 1, 1),  # cand_idx normalized
            nli_hop1[i], nli_hop2[i], nli_balance[i],
            qa_hop1[i],  qa_hop2[i],  qa_balance[i],
            lex_hop1[i], lex_hop2[i], lex_balance[i],
        ])

        # 12 contrastive features
        contra_rows.append([
            rank_nli_h1[i], rank_nli_h2[i], rank_nli_bal[i],
            best_nli_h1 - nli_hop1[i],
            best_nli_h2 - nli_hop2[i],
            rank_qa_h1[i], rank_qa_h2[i], rank_qa_bal[i],
            best_qa_h1 - qa_hop1[i],
            best_qa_h2 - qa_hop2[i],
            rank_lex_h2[i],
            best_lex_h2 - lex_hop2[i],
        ])
        out_idx.append(i)

    return (np.array(chain_rows, dtype=np.float32),
            np.array(contra_rows, dtype=np.float32),
            out_idx)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: CROSS-VALIDATED TRAINING
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_XGB = {
    "n_estimators":     200,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.85,
    "colsample_bytree": 0.85,
    "reg_lambda":       1.0,
    "objective":        "binary:logistic",
    "tree_method":      "hist",
    "verbosity":        0,
}


def run_cv(
    qid_list: np.ndarray,    # (N_rows,)  qid string per row
    X: np.ndarray,           # (N_rows, n_features)
    y: np.ndarray,           # (N_rows,)
    qid_cands: Dict[str, List[Tuple[int, str]]],  # qid -> [(orig_idx, ans_text), ...]
    gold_map: Dict[str, Dict],
    n_folds: int,
    seed: int,
    feature_names: List[str],
    log: logging.Logger,
    tag: str,
) -> Dict[str, Any]:
    """
    K-fold CV at the question level. Returns metrics dict + per-qid predictions.
    """
    unique_qids = np.unique(qid_list)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_probs = np.zeros(len(y), dtype=np.float32)
    fold_importances = []

    for fold_idx, (train_q_idx, val_q_idx) in enumerate(kf.split(unique_qids)):
        train_qids = set(unique_qids[train_q_idx])
        val_qids   = set(unique_qids[val_q_idx])

        train_mask = np.array([q in train_qids for q in qid_list])
        val_mask   = np.array([q in val_qids   for q in qid_list])

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va       = X[val_mask]

        clf = xgb.XGBClassifier(**DEFAULT_XGB, random_state=seed + fold_idx)
        clf.fit(X_tr, y_tr)

        oof_probs[val_mask] = clf.predict_proba(X_va)[:, 1]
        fold_importances.append(clf.feature_importances_)
        log.info(f"  [{tag}] fold {fold_idx+1}/{n_folds}  "
                 f"train_rows={len(y_tr):,}  val_rows={val_mask.sum():,}")

    # Aggregate feature importances across folds
    mean_imp = np.mean(fold_importances, axis=0)
    imp_dict = dict(sorted(
        zip(feature_names, [float(round(x, 4)) for x in mean_imp]),
        key=lambda x: -x[1]
    ))

    # Per-qid prediction = argmax over OOF probs of that qid's rows
    preds = {}
    for qid in unique_qids:
        rows = np.where(qid_list == qid)[0]
        if len(rows) == 0:
            continue
        best_local = int(np.argmax(oof_probs[rows]))
        # Map back to original candidate text using qid_cands
        cand_list = qid_cands[str(qid)]
        if best_local >= len(cand_list):
            continue
        orig_idx, ans_text = cand_list[best_local]
        preds[str(qid)] = {
            "pred": ans_text,
            "best_local_idx": best_local,
            "best_orig_idx": orig_idx,
            "score": float(oof_probs[rows[best_local]]),
        }

    # Compute EM
    correct = 0
    total = 0
    for qid, p in preds.items():
        gold = gold_map.get(qid, {}).get("answer", "")
        correct += em(p["pred"], gold)
        total += 1

    return {
        "n_questions_scored": total,
        "em": round(correct / total, 4) if total > 0 else 0.0,
        "feature_importances": imp_dict,
        "preds": preds,
    }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop_scores",     required=True)
    ap.add_argument("--qa_scores",      required=True)
    ap.add_argument("--lex_features",   required=True)
    ap.add_argument("--candidates",     required=True)
    ap.add_argument("--gold",           required=True)
    ap.add_argument("--setting_label",  required=True)
    ap.add_argument("--out_json",       required=True)
    ap.add_argument("--n_folds",        type=int, default=5)
    ap.add_argument("--seed",           type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    # ── Logging ──
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(args.setting_label)

    W = 76
    log.info("=" * W)
    log.info(f"  Q11v2 — Contrastive Features on Stage-2 Pool  ({args.setting_label})")
    log.info("=" * W)

    # ── Load all inputs ──
    log.info("Loading gold answers ...")
    gold_map = load_gold(args.gold)
    log.info(f"  {len(gold_map):,} questions in gold")

    log.info("Loading NLI hop scores ...")
    hop_records = load_hop_scores(args.hop_scores)

    log.info("Loading QA hop scores ...")
    qa_records  = load_qa_scores(args.qa_scores)

    log.info("Loading lex features ...")
    lex_records = load_lex_features(args.lex_features)

    log.info("Loading raw candidate strings ...")
    cand_records = load_candidates(args.candidates)

    common_qids = (set(hop_records) & set(qa_records) & set(lex_records)
                   & set(cand_records) & set(gold_map))
    log.info(f"  Common qids across all sources: {len(common_qids):,}")

    # ── Apply Stage 1 filter rules deterministically ──
    log.info("Applying Stage 1 filter (deterministic rules) ...")
    n_total_cands     = 0
    n_surv_cands      = 0
    n_all_filtered_qs = 0
    surviving_per_qid: Dict[str, List[int]] = {}

    for qid in common_qids:
        answers = cand_records[qid]
        n_total_cands += len(answers)
        surv = [i for i, a in enumerate(answers) if survives_stage1(a)]
        if not surv:
            n_all_filtered_qs += 1
            surviving_per_qid[qid] = []
        else:
            surviving_per_qid[qid] = surv
            n_surv_cands += len(surv)

    log.info(f"  Total candidates       : {n_total_cands:,}")
    log.info(f"  Surviving Stage 1      : {n_surv_cands:,}  "
             f"({100*n_surv_cands/max(n_total_cands,1):.1f}%)")
    log.info(f"  All-filtered questions : {n_all_filtered_qs:,}")

    # ── Build feature matrices ──
    log.info("Building feature matrices for surviving candidates ...")
    qid_list_all   = []
    X_chain_all    = []
    X_contra_all   = []
    y_all          = []
    qid_cands: Dict[str, List[Tuple[int, str]]] = {}

    n_no_correct_in_pool = 0  # questions where gold not in any candidate (pre-filter)
    n_correct_filtered_out = 0  # gold WAS in pool but Stage 1 removed it

    for qid in sorted(common_qids):
        answers = cand_records[qid]
        gold = gold_map[qid]["answer"]
        gold_in_full_pool = any(em(a, gold) for a in answers)
        if not gold_in_full_pool:
            n_no_correct_in_pool += 1

        surv = surviving_per_qid[qid]
        if not surv:
            continue
        gold_in_surv_pool = any(em(answers[i], gold) for i in surv)
        if gold_in_full_pool and not gold_in_surv_pool:
            n_correct_filtered_out += 1

        X_ch, X_co, orig_idx = build_per_question_features(
            qid, answers,
            hop_records[qid], qa_records[qid], lex_records[qid],
            surv,
        )
        if X_ch.shape[0] == 0:
            continue

        local_cands = []
        for li, oi in enumerate(orig_idx):
            ans_text = answers[oi]
            qid_list_all.append(qid)
            X_chain_all.append(X_ch[li].tolist())
            X_contra_all.append(X_co[li].tolist())
            y_all.append(em(ans_text, gold))
            local_cands.append((oi, ans_text))
        qid_cands[qid] = local_cands

    qid_list = np.array(qid_list_all)
    X_chain  = np.array(X_chain_all,  dtype=np.float32)
    X_full   = np.concatenate([X_chain, np.array(X_contra_all, dtype=np.float32)], axis=1)
    y        = np.array(y_all, dtype=np.float32)

    log.info(f"  Stage-2 training pool : {len(y):,} candidate rows  "
             f"across {len(qid_cands):,} questions")
    log.info(f"  Positive rate         : {y.mean():.3f}")
    log.info(f"  Gold-not-in-full-pool : {n_no_correct_in_pool:,}  "
             f"(unrecoverable for any verifier)")
    log.info(f"  Gold-filtered-by-S1   : {n_correct_filtered_out:,}  "
             f"(Stage 1 false negatives — should be 0)")

    # ── Run baseline (Z3, 10 chain features) ──
    log.info("")
    log.info("Running BASELINE (Z3 chain features only, 10 features) ...")
    res_base = run_cv(
        qid_list, X_chain, y, qid_cands, gold_map,
        args.n_folds, args.seed, CHAIN_FEATURES, log, "z3_baseline",
    )
    log.info(f"  Baseline EM (over {res_base['n_questions_scored']:,} scored qs): "
             f"{res_base['em']:.4f}")

    # ── Run contrastive (Z3 + 12 contrastive = 22 features) ──
    log.info("")
    log.info("Running CONTRASTIVE (Z3 + 12 contrastive features, 22 total) ...")
    res_cont = run_cv(
        qid_list, X_full, y, qid_cands, gold_map,
        args.n_folds, args.seed, ALL_FEATURES, log, "z3_plus_contrastive",
    )
    log.info(f"  Contrastive EM (over {res_cont['n_questions_scored']:,} scored qs): "
             f"{res_cont['em']:.4f}")

    # ── Compute deltas and C2-specific analysis ──
    # C2 = questions where gold is in the Stage-1-surviving pool, but baseline
    # picked a wrong candidate. This is where ranking SHOULD help.
    c2_qids = []
    for qid, p in res_base["preds"].items():
        gold = gold_map.get(qid, {}).get("answer", "")
        if not em(p["pred"], gold):
            # Baseline wrong. Was gold in surviving pool?
            cands = qid_cands.get(qid, [])
            if any(em(ans, gold) for _, ans in cands):
                c2_qids.append(qid)

    n_c2 = len(c2_qids)
    c2_baseline_correct = 0  # by definition 0
    c2_contrastive_correct = sum(
        em(res_cont["preds"].get(qid, {}).get("pred", ""),
           gold_map.get(qid, {}).get("answer", ""))
        for qid in c2_qids
    )

    delta_overall_em = round(res_cont["em"] - res_base["em"], 4)
    delta_c2_em      = round(c2_contrastive_correct / max(n_c2, 1), 4)
    # C2 baseline EM is 0 by construction, so contrastive's EM on C2 IS the delta.

    # Per-question wins/losses/ties between baseline and contrastive
    wins = losses = ties = 0
    for qid in res_base["preds"]:
        gold = gold_map.get(qid, {}).get("answer", "")
        b = em(res_base["preds"][qid]["pred"], gold)
        c = em(res_cont["preds"].get(qid, {}).get("pred", ""), gold)
        if c > b: wins += 1
        elif c < b: losses += 1
        else: ties += 1

    # Top-8 importance check for contrastive features
    cont_imp = res_cont["feature_importances"]
    top8_features = list(cont_imp.keys())[:8]
    contrastive_in_top8 = [f for f in top8_features if f in CONTRASTIVE_FEATURES]

    # ── Decision ──
    if delta_c2_em >= 0.015 and delta_overall_em >= 0.002 and contrastive_in_top8:
        decision = (f"CONTRASTIVE HELPS: ΔC2={delta_c2_em*100:+.2f}pp, "
                    f"Δoverall={delta_overall_em*100:+.2f}pp, "
                    f"top-8: {contrastive_in_top8}. "
                    f"Listwise/cross-cand signal is justified.")
    elif delta_c2_em >= 0.005 and delta_overall_em >= 0:
        decision = (f"MARGINAL: ΔC2={delta_c2_em*100:+.2f}pp, "
                    f"Δoverall={delta_overall_em*100:+.2f}pp. "
                    f"Design listwise as optional ablation.")
    elif delta_c2_em < 0.005 and delta_overall_em <= 0:
        decision = (f"NO EFFECT: ΔC2={delta_c2_em*100:+.2f}pp, "
                    f"Δoverall={delta_overall_em*100:+.2f}pp. "
                    f"Confirms original Q11 — strict pointwise stays.")
    else:
        decision = (f"MIXED: ΔC2={delta_c2_em*100:+.2f}pp, "
                    f"Δoverall={delta_overall_em*100:+.2f}pp, "
                    f"top-8: {contrastive_in_top8}. Inspect manually.")

    # ── Write JSON ──
    summary = {
        "setting": args.setting_label,
        "n_questions_total": len(common_qids),
        "n_questions_post_stage1": len(qid_cands),
        "n_all_filtered_questions": n_all_filtered_qs,
        "n_total_candidates": n_total_cands,
        "n_surviving_candidates": n_surv_cands,
        "stage1_survival_rate": round(n_surv_cands / max(n_total_cands, 1), 4),
        "n_correct_filtered_out_by_stage1": n_correct_filtered_out,
        "n_c2_questions": n_c2,
        "baseline_z3": {
            "em": res_base["em"],
            "n_scored": res_base["n_questions_scored"],
            "feature_importances": res_base["feature_importances"],
        },
        "contrastive_z3plus": {
            "em": res_cont["em"],
            "n_scored": res_cont["n_questions_scored"],
            "feature_importances": res_cont["feature_importances"],
        },
        "delta": {
            "overall_em": delta_overall_em,
            "c2_em":      delta_c2_em,
        },
        "per_question_changes": {
            "wins":   wins,
            "losses": losses,
            "ties":   ties,
            "net":    wins - losses,
        },
        "contrastive_in_top8": contrastive_in_top8,
        "all_contrastive_ranks": {
            f: list(cont_imp.keys()).index(f) + 1 if f in cont_imp else None
            for f in CONTRASTIVE_FEATURES
        },
        "decision": decision,
        "xgb_params": DEFAULT_XGB,
        "n_folds": args.n_folds,
        "seed": args.seed,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print results ──
    log.info("")
    log.info("=" * W)
    log.info(f"  Q11v2 RESULTS — {args.setting_label}")
    log.info("=" * W)
    log.info(f"  N questions total         : {len(common_qids):,}")
    log.info(f"  N post-Stage-1            : {len(qid_cands):,}")
    log.info(f"  N C2 questions            : {n_c2:,}")
    log.info("")
    log.info(f"  {'System':<32} {'EM':>10}")
    log.info(f"  {'─' * 44}")
    log.info(f"  {'Z3 baseline (10 features)':<32} {res_base['em']:>10.4f}")
    log.info(f"  {'Z3 + contrastive (22 feat)':<32} {res_cont['em']:>10.4f}")
    log.info("")
    log.info(f"  ΔEM overall        : {delta_overall_em*100:+.2f}pp")
    log.info(f"  ΔEM on C2 subset   : {delta_c2_em*100:+.2f}pp  "
             f"({c2_contrastive_correct}/{n_c2} C2 recovered)")
    log.info(f"  Per-question       : +{wins} wins  -{losses} losses  "
             f"={ties} ties  (net: {wins - losses:+d})")
    log.info("")
    log.info(f"  Contrastive features in top-8 importance:")
    if contrastive_in_top8:
        for f in contrastive_in_top8:
            log.info(f"    ◀ {f}  (importance={cont_imp[f]})")
    else:
        log.info(f"    (none)")
    log.info("")
    log.info(f"  Top-12 features overall (contrastive model):")
    for i, (f, imp) in enumerate(list(cont_imp.items())[:12]):
        marker = " ◀ NEW" if f in CONTRASTIVE_FEATURES else ""
        log.info(f"    #{i+1:>2}  {f:<28}  {imp:.4f}{marker}")
    log.info("")
    log.info(f"  DECISION: {decision}")
    log.info("=" * W)
    log.info(f"  Saved: {args.out_json}")


if __name__ == "__main__":
    main()