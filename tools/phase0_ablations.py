#!/usr/bin/env python3
"""
phase0_ablations.py — The Decisive Ablations

PURPOSE:
  Determine whether chain-aware features actually contribute to EM,
  or whether the two-stage architecture's gain comes entirely from
  garbage filtering + majority voting (self-consistency).

  This script runs FOUR systems on the same Stage-1-filtered candidate pool:

    Z1  — Pure majority vote on filtered pool (no model at all)
          Picks the most frequent answer among Stage 1 survivors.
          This is operationally identical to Wang et al. (2023) self-consistency.

    Z2  — Two-stage, SURFACE features only (10 features, no chain features)
          Tests whether answer_freq + NLI flat + length heuristics suffice.

    Z3  — Two-stage, CHAIN features only (9 chain features + cand_idx)
          Tests what chain features can do WITHOUT frequency voting help.

    Z_full — Two-stage, ALL 19 features (should reproduce 0.4708 EM)
          Serves as a sanity check that our reimplementation matches.

  The critical comparison:
    - If Z2 ≈ Z_full  →  chain features are ornamental
    - If Z1 ≈ Z2      →  XGBoost adds nothing over majority voting
    - If Z3 << Z2     →  chain features can't stand alone
    - If Z_full > Z2 by ≥0.2pp  →  chain features contribute meaningfully

ARCHITECTURE:
  For each system (Z2, Z3, Z_full):
    1. Load candidates, apply Stage 1 filter (re-derived from is_bad/is_unknown)
    2. Load NLI hop scores, QA hop scores, lexical features
    3. Build feature matrix (surface / chain / all depending on ablation)
    4. 5-fold CV XGBoost, pick best candidate per question from OOF probs
    5. For all-filtered questions (1,289), fall back to monolithic prediction
    6. Evaluate EM against gold

  For Z1:
    Just pick argmax(answer_freq) among survivors — no model needed.

INPUT FILES (all relative to --proj_root):
  exp0c/candidates/dev_M5_7b_K200.jsonl       — M=5 candidate answers
  exp0c/preds/dev_hop_scores.jsonl             — NLI hop scores per candidate
  exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl  — QA cross-encoder scores
  exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl — Lexical features
  exp0c/preds/dev_chain_verifier_mean_preds.jsonl     — Monolithic fallback preds
  data/hotpotqa/raw/hotpot_dev_distractor_v1.json     — Gold answers

OUTPUT:
  --out_dir/phase0_results.json   — all metrics in one file
  --out_dir/phase0_report.txt     — human-readable comparison table
  --out_dir/z{1,2,3,full}_preds.jsonl — per-question predictions for each system

Usage:
  python3 tools/phase0_ablations.py \\
      --proj_root /var/tmp/u24sf51014/sro/work/sro-proof-hotpot \\
      --out_dir   exp_phase0/results \\
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
import time

import numpy as np
from sklearn.model_selection import KFold

# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1: TEXT UTILITIES
#  These must match your existing pipeline exactly — same normalize(),
#  same is_bad(), same EM. Copied from exp1_xgb_verifier.py.
# ═══════════════════════════════════════════════════════════════════════

def normalize(s: str) -> str:
    """Lowercase, strip articles/punctuation, collapse whitespace.
    Used for EM comparison and answer frequency counting."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def em(pred: str, gold: str) -> int:
    """Exact match after normalization. Returns 1 or 0."""
    return int(normalize(pred) == normalize(gold))


def is_bad(ans: str) -> bool:
    """Garbage detector — identifies malformed / hedging / too-long answers.
    Same logic as your existing is_bad() in exp1_xgb_verifier.py."""
    a = ans.strip()
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


def is_unknown_safe(ans: str) -> bool:
    """Safe is_unknown check — uses raw stripped lowercase, NOT normalize().
    This avoids the bug where normalize() destroyed punctuation-only
    and article-only gold answers like '!!!' or 'The The'."""
    raw = ans.strip().lower()
    return raw in {"unknown", "unk", ""}


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2: STAGE 1 FILTER (re-derived, not loaded from file)
#  We re-derive rather than loading dev_stage1_filtered.jsonl because:
#  (a) we know the exact logic, and (b) it avoids schema-guessing.
#  A candidate survives if it is NOT is_bad AND NOT is_unknown_safe.
# ═══════════════════════════════════════════════════════════════════════

def apply_stage1_filter(candidates: list) -> list:
    """Returns list of (original_index, answer_text) for survivors."""
    survivors = []
    for i, ans in enumerate(candidates):
        if not is_bad(ans) and not is_unknown_safe(ans):
            survivors.append((i, ans))
    return survivors


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3: FEATURE EXTRACTION
#  Builds the 19-feature vector used by the full Stage 2 model.
#  Features are grouped so we can select subsets for ablations.
#
#  SURFACE features (10):
#    0  nli_score           — flat NLI entailment probability
#    1  nli_rank            — rank among candidates (0=best, normalized)
#    2  nli_score_gap       — max_nli - this_nli
#    3  answer_freq         — frequency of this answer in original M=5 pool
#    4  is_majority         — 1 if this is the plurality answer
#    5  answer_len_chars    — log1p(char length)
#    6  answer_len_words    — word count
#    7  cand_idx            — position in candidate list, normalized
#    8  unique_count        — distinct answers / M
#    9  answer_freq_filtered — frequency in the filtered pool only
#
#  CHAIN features (9):
#    10  nli_hop1           — NLI entailment vs hop 1 evidence
#    11  nli_hop2           — NLI entailment vs hop 2 evidence
#    12  nli_hop_balance    — |nli_hop1 - nli_hop2|
#    13  qa_hop1            — QA cross-encoder confidence, hop 1
#    14  qa_hop2            — QA cross-encoder confidence, hop 2
#    15  qa_hop_balance     — |qa_hop1 - qa_hop2|
#    16  lex_hop1           — lexical grounding, hop 1
#    17  lex_hop2           — lexical grounding, hop 2
#    18  lex_hop_balance    — |lex_hop1 - lex_hop2|
# ═══════════════════════════════════════════════════════════════════════

# Feature name lists for each group
SURFACE_FEATURES = [
    "nli_score", "nli_rank", "nli_score_gap",
    "answer_freq", "is_majority",
    "answer_len_chars", "answer_len_words",
    "cand_idx", "unique_count",
    "answer_freq_filtered",
]

CHAIN_FEATURES = [
    "nli_hop1", "nli_hop2", "nli_hop_balance",
    "qa_hop1", "qa_hop2", "qa_hop_balance",
    "lex_hop1", "lex_hop2", "lex_hop_balance",
]

ALL_FEATURES = SURFACE_FEATURES + CHAIN_FEATURES


def build_features_for_question(
    survivors,        # list of (orig_idx, answer_text)
    all_answers,      # list of ALL M candidates (for original freq)
    nli_cands,        # list of dicts from hop_scores, aligned by position
    qa_scores,        # dict with qa_hop1/qa_hop2 lists, aligned by position
    lex_feats,        # dict with lex_hop1/lex_hop2 lists, aligned by position
):
    """
    Build feature matrix for surviving candidates of one question.

    Returns:
      X: np.array of shape (n_survivors, 19)
      feature_names: list of 19 feature names
    """
    n_surv = len(survivors)
    if n_surv == 0:
        return np.empty((0, len(ALL_FEATURES))), ALL_FEATURES

    # --- Pre-compute answer frequencies ---

    # Original pool frequency (among all M candidates)
    all_norms = [normalize(a) for a in all_answers]
    orig_freq_counter = collections.Counter(all_norms)
    m_total = len(all_answers)
    orig_majority_norm = orig_freq_counter.most_common(1)[0][0]

    # Filtered pool frequency (among survivors only)
    surv_norms = [normalize(ans) for _, ans in survivors]
    filt_freq_counter = collections.Counter(surv_norms)
    m_filt = len(survivors)

    # Unique count (original pool)
    unique_count = len(orig_freq_counter) / max(m_total, 1)

    # --- NLI scores for survivors ---
    # nli_cands is aligned to original candidate positions
    surv_nli_scores = []
    for orig_idx, _ in survivors:
        cand_data = nli_cands[orig_idx]
        # Try different field names for the flat NLI score
        nli_val = cand_data.get("nli_flat",
                  cand_data.get("nli_score",
                  cand_data.get("flat_nli", 0.0)))
        surv_nli_scores.append(float(nli_val))

    nli_arr = np.array(surv_nli_scores, dtype=np.float32)
    nli_max = nli_arr.max() if len(nli_arr) > 0 else 0.0

    # NLI ranks among survivors
    if n_surv > 1:
        rank_order = np.argsort(-nli_arr)
        rank_map = np.empty(n_surv, dtype=np.float32)
        rank_map[rank_order] = np.arange(n_surv) / (n_surv - 1)
    else:
        rank_map = np.array([0.0], dtype=np.float32)

    # --- Build per-candidate feature vectors ---
    rows = []
    for si, (orig_idx, ans) in enumerate(survivors):
        norm = normalize(ans)
        cand_nli = nli_cands[orig_idx]

        # Surface features
        nli_score     = surv_nli_scores[si]
        nli_rank      = float(rank_map[si])
        nli_gap       = float(nli_max - nli_score)
        ans_freq_orig = orig_freq_counter[norm] / max(m_total, 1)
        is_maj        = int(norm == orig_majority_norm)
        ans_len_c     = math.log1p(len(ans))
        ans_len_w     = len(ans.split())
        c_idx         = orig_idx / max(m_total - 1, 1)
        ans_freq_filt = filt_freq_counter[norm] / max(m_filt, 1)

        # NLI chain features
        nli_h1  = float(cand_nli.get("nli_hop1", 0.0))
        nli_h2  = float(cand_nli.get("nli_hop2", 0.0))
        nli_bal = abs(nli_h1 - nli_h2)

        # QA chain features
        qa_h1  = float(qa_scores[orig_idx].get("qa_hop1", 0.0))
        qa_h2  = float(qa_scores[orig_idx].get("qa_hop2", 0.0))
        qa_bal = abs(qa_h1 - qa_h2)

        # Lexical chain features
        lex_h1  = float(lex_feats[orig_idx].get("lex_hop1",
                        lex_feats[orig_idx].get("ans_in_hop1", 0.0)))
        lex_h2  = float(lex_feats[orig_idx].get("lex_hop2",
                        lex_feats[orig_idx].get("ans_in_hop2", 0.0)))
        lex_bal = abs(lex_h1 - lex_h2)

        row = [
            # 10 surface features
            nli_score, nli_rank, nli_gap,
            ans_freq_orig, is_maj,
            ans_len_c, ans_len_w,
            c_idx, unique_count,
            ans_freq_filt,
            # 9 chain features
            nli_h1, nli_h2, nli_bal,
            qa_h1, qa_h2, qa_bal,
            lex_h1, lex_h2, lex_bal,
        ]
        rows.append(row)

    return np.array(rows, dtype=np.float32), ALL_FEATURES


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 4: DATA LOADING
#  Loads all input files and aligns them by question ID.
#  Includes schema auto-detection with helpful error messages.
# ═══════════════════════════════════════════════════════════════════════

def load_jsonl(path: str) -> list:
    """Load a JSONL file, return list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_candidates(path: str) -> dict:
    """Load candidates file.
    Returns: {qid: [answer_text_0, answer_text_1, ...]}"""
    result = {}
    for rec in load_jsonl(path):
        qid = str(rec["qid"])
        cands = rec["candidates"]
        # Handle both {answer_text: ...} and plain strings
        if isinstance(cands[0], dict):
            answers = [c.get("answer_text", c.get("answer", "")) for c in cands]
        else:
            answers = [str(c) for c in cands]
        result[qid] = answers
    return result


def load_hop_scores(path: str) -> dict:
    """Load NLI hop scores.
    Returns: {qid: [cand_dict_0, cand_dict_1, ...]}
    where each cand_dict has nli_flat, nli_hop1, nli_hop2."""
    result = {}
    for rec in load_jsonl(path):
        qid = str(rec["qid"])
        result[qid] = rec["candidates"]
    return result


def load_qa_scores(path: str) -> dict:
    """Load QA cross-encoder hop scores.
    Returns: {qid: [cand_dict_0, ...]} with qa_hop1, qa_hop2."""
    result = {}
    for rec in load_jsonl(path):
        qid = str(rec["qid"])
        # Handle both flat list and nested candidates
        if "candidates" in rec:
            result[qid] = rec["candidates"]
        else:
            # Might be per-candidate records; group by qid
            if qid not in result:
                result[qid] = []
            result[qid].append(rec)
    return result


def load_lex_features(path: str) -> dict:
    """Load lexical grounding features.
    Returns: {qid: [cand_dict_0, ...]} with lex_hop1, lex_hop2."""
    result = {}
    for rec in load_jsonl(path):
        qid = str(rec["qid"])
        if "candidates" in rec:
            result[qid] = rec["candidates"]
        else:
            if qid not in result:
                result[qid] = []
            result[qid].append(rec)
    return result


def load_gold(path: str) -> dict:
    """Load gold answers from HotpotQA distractor file.
    Returns: {qid: gold_answer_string}"""
    data = json.load(open(path))
    return {str(ex["_id"]): ex["answer"] for ex in data}


def load_monolithic_preds(path: str) -> dict:
    """Load monolithic verifier predictions for fallback.
    Returns: {qid: predicted_answer_string}"""
    result = {}
    for rec in load_jsonl(path):
        qid = str(rec["qid"])
        result[qid] = rec.get("pred", "")
    return result


def load_greedy_preds(path: str) -> dict:
    """Load M=1 greedy predictions for same-setting fallback.
    Handles both candidate-format (from distractor_generate.py) and
    pred-format (from step8 m1_greedy_preds.jsonl).
    Returns: {qid: predicted_answer_string}"""
    result = {}
    for rec in load_jsonl(path):
        qid = str(rec["qid"])
        if "candidates" in rec:
            # Candidate format: {qid, candidates: [{answer_text: ...}]}
            cands = rec["candidates"]
            if cands and isinstance(cands[0], dict):
                result[qid] = cands[0].get("answer_text", "")
            elif cands:
                result[qid] = str(cands[0])
            else:
                result[qid] = ""
        elif "pred" in rec:
            # Prediction format: {qid, pred: "..."}
            result[qid] = rec["pred"]
        else:
            result[qid] = ""
    return result


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 5: SCHEMA INSPECTOR
#  Prints the first record of each file so you can verify field names
#  match before running the full experiment.
# ═══════════════════════════════════════════════════════════════════════

def inspect_schemas(args, log):
    """Print first record of each input file for sanity checking."""
    log.info("=" * 70)
    log.info("  SCHEMA INSPECTION — verify these match your data")
    log.info("=" * 70)

    files = {
        "candidates":      args.candidates_path,
        "hop_scores":      args.hop_scores_path,
        "qa_scores":       args.qa_scores_path,
        "lex_features":    args.lex_features_path,
        "monolithic_preds": args.mono_preds_path,
    }

    for name, path in files.items():
        log.info(f"\n  --- {name}: {path} ---")
        try:
            with open(path) as f:
                first = json.loads(f.readline().strip())
            # Show top-level keys
            log.info(f"  Keys: {list(first.keys())}")
            # If there's a 'candidates' list, show first candidate's keys
            if "candidates" in first and isinstance(first["candidates"], list):
                c0 = first["candidates"][0]
                if isinstance(c0, dict):
                    log.info(f"  candidates[0] keys: {list(c0.keys())}")
                else:
                    log.info(f"  candidates[0] type: {type(c0).__name__}")
        except Exception as e:
            log.error(f"  ERROR reading {path}: {e}")

    log.info("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 6: EXPERIMENT RUNNERS
# ═══════════════════════════════════════════════════════════════════════

def run_majority_vote(qid_survivors, gold_map, fallback_preds, log):
    """
    Z1: Pure majority vote on filtered pool.

    For each question with survivors, pick the most frequent answer
    (by normalized string). Ties broken by first occurrence.
    For all-filtered questions, fall back to same-setting prediction.

    This is equivalent to Wang et al. (2023) self-consistency.
    """
    log.info("Running Z1: majority vote on filtered pool ...")
    correct = 0
    total = 0
    preds = {}

    for qid in sorted(gold_map.keys()):
        gold = gold_map[qid]
        total += 1

        survivors = qid_survivors.get(qid, [])
        if not survivors:
            # All-filtered: fall back to same-setting prediction
            pred = fallback_preds.get(qid, "")
        else:
            # Count normalized answer frequencies among survivors
            surv_norms = [normalize(ans) for _, ans in survivors]
            freq = collections.Counter(surv_norms)
            majority_norm = freq.most_common(1)[0][0]
            # Pick the first candidate with that normalized answer
            pred = ""
            for _, ans in survivors:
                if normalize(ans) == majority_norm:
                    pred = ans
                    break

        preds[qid] = {"pred": pred}
        correct += em(pred, gold)

    em_score = correct / total
    log.info(f"  Z1 result: {correct}/{total} = {em_score:.4f} EM")
    return em_score, preds


def run_xgb_ablation(
    name,             # e.g. "Z2_surface", "Z3_chain", "Z_full"
    feature_indices,  # which columns of the 19-feature matrix to use
    feature_names_used,  # names for logging
    qid_features,     # {qid: (X_matrix, survivor_list)}
    gold_map,
    fallback_preds,
    n_folds,
    seed,
    xgb_params,
    log,
):
    """
    Run one XGBoost ablation with a specific feature subset.

    1. Build global arrays (X, y, qids) from per-question feature matrices
    2. 5-fold CV at question level
    3. Pick best candidate per question from out-of-fold probabilities
    4. Fall back to same-setting prediction for all-filtered questions
    5. Return EM and per-question predictions
    """
    import xgboost as xgb_lib

    log.info(f"Running {name}: {len(feature_names_used)} features ...")
    log.info(f"  Features: {feature_names_used}")

    # --- Step 1: Build global arrays ---
    all_qids = []       # qid per candidate row
    all_X = []          # feature vectors
    all_y = []          # binary labels (EM match)
    all_answers = []    # answer strings for prediction
    qid_row_ranges = {} # {qid: (start_idx, end_idx)}

    for qid in sorted(qid_features.keys()):
        X_q, survivors = qid_features[qid]
        if X_q.shape[0] == 0:
            continue

        gold = gold_map.get(qid, "")
        start = len(all_X)

        for si in range(X_q.shape[0]):
            _, ans = survivors[si]
            all_qids.append(qid)
            all_X.append(X_q[si, feature_indices])
            all_y.append(em(ans, gold))
            all_answers.append(ans)

        end = len(all_X)
        qid_row_ranges[qid] = (start, end)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)
    qids_arr = np.array(all_qids)

    log.info(f"  Matrix: {X.shape[0]} candidates × {X.shape[1]} features")
    log.info(f"  Positive rate: {y.mean():.4f} ({y.sum()}/{len(y)})")

    # --- Step 2: 5-fold CV at question level ---
    unique_qids = sorted(set(all_qids))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_probs = np.zeros(len(all_X), dtype=np.float64)
    fold_importances = []

    for fold, (train_qi, val_qi) in enumerate(kf.split(unique_qids)):
        train_qid_set = set(unique_qids[i] for i in train_qi)
        val_qid_set   = set(unique_qids[i] for i in val_qi)

        train_mask = np.array([q in train_qid_set for q in all_qids])
        val_mask   = np.array([q in val_qid_set   for q in all_qids])

        clf = xgb_lib.XGBClassifier(**xgb_params)
        clf.fit(X[train_mask], y[train_mask])

        oof_probs[val_mask] = clf.predict_proba(X[val_mask])[:, 1]
        fold_importances.append(clf.feature_importances_)

        log.info(f"  Fold {fold+1}/{n_folds}: "
                 f"train={train_mask.sum()}cands({len(train_qid_set)}q) "
                 f"val={val_mask.sum()}cands({len(val_qid_set)}q)")

    # --- Step 3: Pick best candidate per question ---
    preds = {}
    for qid, (start, end) in qid_row_ranges.items():
        probs = oof_probs[start:end]
        best_i = int(np.argmax(probs))
        preds[qid] = {
            "pred": all_answers[start + best_i],
            "prob": float(probs[best_i]),
        }

    # --- Step 4: Fallback for all-filtered questions ---
    for qid in gold_map:
        if qid not in preds:
            preds[qid] = {
                "pred": fallback_preds.get(qid, ""),
                "prob": 0.0,
                "fallback": True,
            }

    # --- Step 5: Evaluate ---
    correct = sum(em(preds[qid]["pred"], gold_map[qid]) for qid in gold_map)
    total = len(gold_map)
    em_score = correct / total

    # Feature importances
    mean_imp = np.mean(fold_importances, axis=0)
    imp_pairs = sorted(zip(feature_names_used, mean_imp), key=lambda x: -x[1])

    log.info(f"  {name} result: {correct}/{total} = {em_score:.4f} EM")
    log.info(f"  Feature importances (top-10):")
    for i, (fname, imp) in enumerate(imp_pairs[:10]):
        log.info(f"    #{i+1:>2}  {fname:<24}  {imp:.4f}")

    return em_score, preds, imp_pairs


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 7: MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Phase 0: Decisive Ablations")

    # Paths — either specify individually or use --proj_root for defaults
    ap.add_argument("--proj_root", required=True,
        help="Project root, e.g. /var/tmp/u24sf51014/sro/work/sro-proof-hotpot")
    ap.add_argument("--out_dir", required=True,
        help="Output directory for results")

    # Override individual paths if needed
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--hop_scores", default=None)
    ap.add_argument("--qa_scores", default=None)
    ap.add_argument("--lex_features", default=None)
    ap.add_argument("--mono_preds", default=None)
    ap.add_argument("--greedy_preds", default=None,
        help="M=1 greedy predictions for same-setting fallback (prevents cross-setting leak)")
    ap.add_argument("--gold", default=None)

    # XGBoost params (same as existing pipeline)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--xgb_n_estimators", type=int, default=300)
    ap.add_argument("--xgb_max_depth", type=int, default=4)
    ap.add_argument("--xgb_lr", type=float, default=0.1)
    ap.add_argument("--xgb_subsample", type=float, default=0.8)
    ap.add_argument("--xgb_colsample", type=float, default=0.8)

    # Flags
    ap.add_argument("--inspect_only", action="store_true",
        help="Only print file schemas, don't run experiments")

    args = ap.parse_args()

    # --- Resolve paths ---
    R = args.proj_root
    args.candidates_path = args.candidates or f"{R}/exp0c/candidates/dev_M5_7b_K200.jsonl"
    args.hop_scores_path = args.hop_scores or f"{R}/exp0c/preds/dev_hop_scores.jsonl"
    args.qa_scores_path  = args.qa_scores  or f"{R}/exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl"
    args.lex_features_path = args.lex_features or f"{R}/exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl"
    args.mono_preds_path = args.mono_preds or f"{R}/exp0c/preds/dev_chain_verifier_mean_preds.jsonl"
    args.gold_path       = args.gold       or f"{R}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"

    # --- Setup ---
    os.makedirs(args.out_dir, exist_ok=True)

    log = logging.getLogger("phase0")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(f"{args.out_dir}/phase0.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)

    log.info("=" * 70)
    log.info("  PHASE 0: THE DECISIVE ABLATIONS")
    log.info("=" * 70)

    # --- Schema inspection ---
    inspect_schemas(args, log)
    if args.inspect_only:
        log.info("--inspect_only flag set. Exiting.")
        return

    # --- Verify all files exist ---
    for name, path in [
        ("candidates", args.candidates_path),
        ("hop_scores", args.hop_scores_path),
        ("qa_scores",  args.qa_scores_path),
        ("lex_features", args.lex_features_path),
        ("mono_preds", args.mono_preds_path),
        ("gold",       args.gold_path),
    ]:
        if not os.path.isfile(path):
            log.error(f"MISSING FILE: {name} at {path}")
            log.error("Fix the path or use --{name} to override.")
            sys.exit(1)
        log.info(f"  ✓ {name}: {path}")

    # ── Load data ──
    log.info("\nLoading data ...")
    t0 = time.time()

    gold_map = load_gold(args.gold_path)
    log.info(f"  Gold: {len(gold_map)} questions")

    candidates = load_candidates(args.candidates_path)
    log.info(f"  Candidates: {len(candidates)} questions")

    hop_scores = load_hop_scores(args.hop_scores_path)
    log.info(f"  Hop scores: {len(hop_scores)} questions")

    qa_scores = load_qa_scores(args.qa_scores_path)
    log.info(f"  QA scores: {len(qa_scores)} questions")

    lex_feats = load_lex_features(args.lex_features_path)
    log.info(f"  Lex features: {len(lex_feats)} questions")

    mono_preds = load_monolithic_preds(args.mono_preds_path)
    log.info(f"  Monolithic preds: {len(mono_preds)} questions")

    # Load greedy preds if provided (same-setting fallback)
    greedy_preds = {}
    if args.greedy_preds and os.path.isfile(args.greedy_preds):
        greedy_preds = load_greedy_preds(args.greedy_preds)
        log.info(f"  Greedy preds: {len(greedy_preds)} questions (same-setting fallback)")
    elif args.greedy_preds:
        log.warning(f"  ⚠ Greedy preds file not found: {args.greedy_preds}")
        log.warning(f"    Falling back to monolithic preds (may cause cross-setting leak)")

    # Build fallback predictions:
    # If greedy_preds provided → use greedy (same-setting, no cross-contamination)
    # Otherwise → use mono_preds (legacy behavior, MDR setting)
    if greedy_preds:
        fallback_preds = greedy_preds
        log.info(f"  Fallback source: GREEDY (same-setting, {len(fallback_preds)} questions)")
    else:
        fallback_preds = mono_preds
        log.info(f"  Fallback source: MONOLITHIC (legacy, {len(fallback_preds)} questions)")

    log.info(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Apply Stage 1 filter ──
    log.info("\nApplying Stage 1 garbage filter ...")
    qid_survivors = {}      # {qid: [(orig_idx, answer_text), ...]}
    total_before = 0
    total_after = 0
    all_filtered_qids = []

    for qid in sorted(gold_map.keys()):
        cands = candidates.get(qid, [])
        total_before += len(cands)
        survivors = apply_stage1_filter(cands)
        total_after += len(survivors)

        if not survivors:
            all_filtered_qids.append(qid)
        qid_survivors[qid] = survivors

    filter_rate = 1 - (total_after / max(total_before, 1))
    log.info(f"  Before: {total_before} candidates")
    log.info(f"  After:  {total_after} candidates")
    log.info(f"  Filter rate: {filter_rate:.1%}")
    log.info(f"  All-filtered questions: {len(all_filtered_qids)}")

    # Verify oracle preservation (critical safety check)
    # Oracle must match the system's actual answer sources:
    #   - For questions with survivors: can the verifier pick a correct one?
    #   - For all-filtered questions: is the fallback prediction correct?
    oracle_before = 0
    oracle_after = 0
    oracle_system = 0    # NEW: oracle that matches what the system can access
    fallback_correct = 0  # how many all-filtered questions get correct fallback

    for qid in gold_map:
        gold = gold_map[qid]
        cands = candidates.get(qid, [])

        # Oracle before filter (any of M candidates correct?)
        if any(em(c, gold) for c in cands):
            oracle_before += 1

        # Oracle after filter (any survivor correct?)
        survivors = qid_survivors.get(qid, [])
        if any(em(ans, gold) for _, ans in survivors):
            oracle_after += 1

        # System oracle: matches actual answer sources
        if survivors:
            # System picks from survivors
            if any(em(ans, gold) for _, ans in survivors):
                oracle_system += 1
        else:
            # System uses fallback
            fb = fallback_preds.get(qid, "")
            if em(fb, gold):
                oracle_system += 1
                fallback_correct += 1

    log.info(f"  Oracle EM before filter: {oracle_before}/{len(gold_map)} "
             f"= {oracle_before/len(gold_map):.4f}")
    log.info(f"  Oracle EM after filter:  {oracle_after}/{len(gold_map)} "
             f"= {oracle_after/len(gold_map):.4f}")
    log.info(f"  Oracle EM (system):      {oracle_system}/{len(gold_map)} "
             f"= {oracle_system/len(gold_map):.4f}")
    log.info(f"  False negatives (correct answers filtered): "
             f"{oracle_before - oracle_after}")
    log.info(f"  Fallback correct among {len(all_filtered_qids)} all-filtered: "
             f"{fallback_correct}")

    if oracle_before - oracle_after > 0:
        log.warning("  ⚠ WARNING: Stage 1 filter removed some correct answers!")
        log.warning("  Check is_unknown_safe() logic against your pipeline.")

    # ── Build feature matrices ──
    log.info("\nBuilding feature matrices ...")
    qid_features = {}  # {qid: (X_matrix, survivors_list)}

    missing_qa = 0
    missing_lex = 0

    for qid in sorted(gold_map.keys()):
        survivors = qid_survivors.get(qid, [])
        if not survivors:
            continue

        cands = candidates.get(qid, [])
        nli_c = hop_scores.get(qid, [{}] * len(cands))

        # QA scores — might be missing for some questions
        qa_c = qa_scores.get(qid, [{}] * len(cands))
        if not qa_c:
            qa_c = [{}] * len(cands)
            missing_qa += 1

        # Lex features — might be missing
        lex_c = lex_feats.get(qid, [{}] * len(cands))
        if not lex_c:
            lex_c = [{}] * len(cands)
            missing_lex += 1

        X_q, _ = build_features_for_question(
            survivors, cands, nli_c, qa_c, lex_c
        )
        qid_features[qid] = (X_q, survivors)

    log.info(f"  Feature matrices built for {len(qid_features)} questions")
    if missing_qa > 0:
        log.warning(f"  ⚠ {missing_qa} questions missing QA scores (using zeros)")
    if missing_lex > 0:
        log.warning(f"  ⚠ {missing_lex} questions missing lex features (using zeros)")

    # ── XGBoost params ──
    xgb_params = dict(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_lr,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.seed,
        n_jobs=-1,
        verbosity=0,
    )

    # Feature index mappings for ablations
    surface_idx = list(range(len(SURFACE_FEATURES)))               # 0-9
    chain_idx   = list(range(len(SURFACE_FEATURES),
                             len(SURFACE_FEATURES) + len(CHAIN_FEATURES)))  # 10-18
    # Z3 needs cand_idx (index 7) as a position feature alongside chain
    chain_plus_idx = [7] + chain_idx   # cand_idx + 9 chain features
    all_idx     = list(range(len(ALL_FEATURES)))                   # 0-18

    # ═══════════════════════════════════════════════════════════════
    #  RUN ALL FOUR ABLATIONS
    # ═══════════════════════════════════════════════════════════════

    results = {}

    # --- Z1: Majority vote ---
    z1_em, z1_preds = run_majority_vote(
        qid_survivors, gold_map, fallback_preds, log
    )
    results["Z1_majority_vote"] = {"em": z1_em}

    log.info("")

    # --- Z2: Surface features only ---
    z2_em, z2_preds, z2_imp = run_xgb_ablation(
        "Z2_surface_only", surface_idx, SURFACE_FEATURES,
        qid_features, gold_map, fallback_preds,
        args.n_folds, args.seed, xgb_params, log
    )
    results["Z2_surface_only"] = {
        "em": z2_em,
        "features": SURFACE_FEATURES,
        "importances": {k: round(float(v), 4) for k, v in z2_imp},
    }

    log.info("")

    # --- Z3: Chain features only (+ cand_idx for position) ---
    z3_names = ["cand_idx"] + CHAIN_FEATURES
    z3_em, z3_preds, z3_imp = run_xgb_ablation(
        "Z3_chain_only", chain_plus_idx, z3_names,
        qid_features, gold_map, fallback_preds,
        args.n_folds, args.seed, xgb_params, log
    )
    results["Z3_chain_only"] = {
        "em": z3_em,
        "features": z3_names,
        "importances": {k: round(float(v), 4) for k, v in z3_imp},
    }

    log.info("")

    # --- Z_full: All 19 features (should reproduce 0.4708) ---
    zf_em, zf_preds, zf_imp = run_xgb_ablation(
        "Z_full_all_features", all_idx, ALL_FEATURES,
        qid_features, gold_map, fallback_preds,
        args.n_folds, args.seed, xgb_params, log
    )
    results["Z_full"] = {
        "em": zf_em,
        "features": ALL_FEATURES,
        "importances": {k: round(float(v), 4) for k, v in zf_imp},
    }

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 8: REPORT
    # ═══════════════════════════════════════════════════════════════

    log.info("")
    log.info("=" * 70)
    log.info("  PHASE 0 RESULTS — THE DECISIVE TABLE")
    log.info("=" * 70)

    baseline_em = 0.4666   # monolithic XGBoost (hardcoded reference)
    log.info(f"  {'System':<40} {'EM':>8} {'Δ vs mono':>10} {'Δ vs Z1':>10}")
    log.info("  " + "-" * 68)

    rows = [
        ("Monolithic XGBoost (reference)",  baseline_em),
        ("Z1: Stage1 + majority vote",     z1_em),
        ("Z2: Stage1 + XGB surface only",  z2_em),
        ("Z3: Stage1 + XGB chain only",    z3_em),
        ("Z_full: Stage1 + XGB all 19",    zf_em),
    ]
    for label, score in rows:
        delta_mono = score - baseline_em
        delta_z1   = score - z1_em
        log.info(f"  {label:<40} {score:>8.4f} "
                 f"{delta_mono:>+10.4f} {delta_z1:>+10.4f}")

    log.info("  " + "-" * 68)

    # --- Interpretation guide ---
    chain_marginal = zf_em - z2_em
    surface_marginal = zf_em - z3_em
    mv_vs_full = zf_em - z1_em

    log.info("")
    log.info("  INTERPRETATION:")
    log.info(f"  Chain feature marginal contribution: "
             f"{chain_marginal:+.4f} ({chain_marginal*100:+.2f}pp)")
    log.info(f"  Surface feature marginal contribution: "
             f"{surface_marginal:+.4f} ({surface_marginal*100:+.2f}pp)")
    log.info(f"  Full model vs majority vote: "
             f"{mv_vs_full:+.4f} ({mv_vs_full*100:+.2f}pp)")

    if chain_marginal < 0.001:
        log.info("  ⚠ CHAIN FEATURES APPEAR ORNAMENTAL (<0.1pp gain)")
        log.info("    → Reframe paper around diagnostic contribution")
    elif chain_marginal < 0.002:
        log.info("  ⚡ CHAIN FEATURES CONTRIBUTE MODESTLY (0.1-0.2pp)")
        log.info("    → Defensible but needs careful framing")
    else:
        log.info("  ✓ CHAIN FEATURES CONTRIBUTE MEANINGFULLY (≥0.2pp)")
        log.info("    → Chain-aware claim survives")

    log.info("")
    log.info("=" * 70)

    # --- Save results ---
    results["meta"] = {
        "baseline_em": baseline_em,
        "chain_marginal_pp": round(chain_marginal * 100, 2),
        "surface_marginal_pp": round(surface_marginal * 100, 2),
        "majority_vote_vs_full_pp": round(mv_vs_full * 100, 2),
        "n_questions": len(gold_map),
        "n_all_filtered": len(all_filtered_qids),
        "filter_rate": round(filter_rate, 4),
        "oracle_before": oracle_before,
        "oracle_after": oracle_after,
        "oracle_system": oracle_system,
        "oracle_system_em": round(oracle_system / len(gold_map), 4),
        "fallback_correct_in_filtered": fallback_correct,
        "fallback_source": "greedy" if greedy_preds else "monolithic",
        "greedy_preds_path": args.greedy_preds or "none",
    }

    with open(f"{args.out_dir}/phase0_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"  Results saved to {args.out_dir}/phase0_results.json")

    # Save predictions for each system
    for tag, preds_dict in [
        ("z1_majority", z1_preds),
        ("z2_surface", z2_preds),
        ("z3_chain", z3_preds),
        ("z_full", zf_preds),
    ]:
        path = f"{args.out_dir}/{tag}_preds.jsonl"
        with open(path, "w") as f:
            for qid in sorted(preds_dict.keys()):
                f.write(json.dumps({"qid": qid, **preds_dict[qid]}) + "\n")
    log.info(f"  Predictions saved to {args.out_dir}/z*_preds.jsonl")


if __name__ == "__main__":
    main()