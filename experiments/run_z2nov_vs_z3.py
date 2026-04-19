#!/usr/bin/env python3
"""
experiments/run_z2nov_vs_z3.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Experiment 3 — Fixed: Z2-NoV vs Z3 (genuinely symmetric comparison)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS EXISTS
    The original Experiment 3 compared Z3 (no voting) vs Z2 (surface WITH
    voting). That was asymmetric — Z2 had answer_freq, is_majority,
    unique_count, answer_freq_filtered while Z3 had none of those.

    This script creates Z2-NoV: surface features with voting removed,
    keeping only the pure evidence-quality signals:
        nli_score, nli_rank, nli_score_gap,
        answer_len_chars, answer_len_words, cand_idx

    Then Z3 vs Z2-NoV is the symmetric question:
    "Does scoring hop-1 and hop-2 separately (Z3) beat scoring them
     concatenated as flat NLI (Z2-NoV), when both sides have EQUAL
     information and neither knows what the other candidates said?"

FEATURE COMPARISON
    Z2-NoV  (6 features, no voting):
        nli_score       — flat NLI on hop1+hop2 concatenated
        nli_rank        — rank among candidates (normalized)
        nli_score_gap   — max_nli - this_nli
        answer_len_chars — log(1 + char length)
        answer_len_words — word count
        cand_idx        — position in list

    Z3      (10 features, no voting):
        cand_idx        — position in list
        nli_hop1        — NLI vs hop-1 only (bridge doc)
        nli_hop2        — NLI vs hop-2 only (answer doc)
        nli_hop_balance — |nli_hop1 - nli_hop2|
        qa_hop1         — QA confidence hop-1
        qa_hop2         — QA confidence hop-2
        qa_hop_balance  — |qa_hop1 - qa_hop2|
        lex_hop1        — lexical overlap hop-1
        lex_hop2        — lexical overlap hop-2
        lex_hop_balance — |lex_hop1 - lex_hop2|

    Both: XGBoost 5-fold CV, same Stage 1 filter, same fallback preds.
    Neither: answer_freq, is_majority, unique_count, answer_freq_filtered.

USAGE
    # 7B MDR
    python3 experiments/run_z2nov_vs_z3.py \\
        --proj_root /var/tmp/u24sf51014/sro/work/sro-proof-hotpot \\
        --setting   hotpotqa_mdr \\
        --out_dir   experiments/results/z2nov

    # All settings at once
    python3 experiments/run_z2nov_vs_z3.py \\
        --proj_root /var/tmp/u24sf51014/sro/work/sro-proof-hotpot \\
        --out_dir   experiments/results/z2nov

    # 1.5B MDR (still running ablations — run after those finish)
    python3 experiments/run_z2nov_vs_z3.py \\
        --proj_root /var/tmp/u24sf51014/sro/work/sro-proof-hotpot \\
        --setting   hotpotqa_1p5b \\
        --out_dir   experiments/results/z2nov
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
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 1 — TEXT UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))

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

def is_unknown_safe(ans: str) -> bool:
    return ans.strip().lower() in {"unknown", "unk", ""}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 2 — FEATURE DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Z2-NoV: surface features with all voting features REMOVED
# Only pure evidence-quality signals — no cross-candidate pooling
Z2_NOV_FEATURES = [
    "nli_score",        # flat NLI (evidence quality, not voting)
    "nli_rank",         # rank among survivors
    "nli_score_gap",    # gap from best NLI score
    "answer_len_chars", # log(1 + char length)
    "answer_len_words", # word count
    "cand_idx",         # position in list
]

# Z3: chain-aware, no voting features
Z3_FEATURES = [
    "cand_idx",
    "nli_hop1", "nli_hop2", "nli_hop_balance",
    "qa_hop1",  "qa_hop2",  "qa_hop_balance",
    "lex_hop1", "lex_hop2", "lex_hop_balance",
]

# Full 19-feature list (same order as phase0_ablations_v2.py)
ALL_19 = [
    # surface (10)
    "nli_score", "nli_rank", "nli_score_gap",
    "answer_freq", "is_majority",
    "answer_len_chars", "answer_len_words",
    "cand_idx", "unique_count", "answer_freq_filtered",
    # chain (9)
    "nli_hop1", "nli_hop2", "nli_hop_balance",
    "qa_hop1",  "qa_hop2",  "qa_hop_balance",
    "lex_hop1", "lex_hop2", "lex_hop_balance",
]
ALL_19_IDX = {f: i for i, f in enumerate(ALL_19)}

Z2_NOV_IDX = [ALL_19_IDX[f] for f in Z2_NOV_FEATURES]
Z3_IDX     = [ALL_19_IDX[f] for f in Z3_FEATURES]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 3 — I/O
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_candidates(path: str) -> Dict[str, List[str]]:
    result = {}
    for rec in iter_jsonl(path):
        qid = str(rec["qid"])
        raw = rec.get("candidates", [])
        if raw and isinstance(raw[0], dict):
            answers = [c.get("answer_text", c.get("answer", "")) for c in raw]
        else:
            answers = [str(c) for c in raw]
        result[qid] = answers
    return result

def load_hop_scores(path: str) -> Dict[str, List[dict]]:
    result = {}
    for rec in iter_jsonl(path):
        result[str(rec["qid"])] = rec.get("candidates", [])
    return result

def load_qa_lex(path: str) -> Dict[str, List[dict]]:
    result = {}
    for rec in iter_jsonl(path):
        result[str(rec["qid"])] = rec.get("candidates", [])
    return result

def load_gold_hotpot(path: str) -> Dict[str, str]:
    data = json.load(open(path))
    return {str(ex["_id"]): ex["answer"] for ex in data}

def load_gold_wiki2(path: str) -> Dict[str, str]:
    with open(path, encoding="utf-8") as f:
        raw = f.read(1)
        f.seek(0)
        data = json.load(f) if raw.strip() == "[" else \
               [json.loads(l) for l in f if l.strip()]
    result = {}
    for ex in data:
        qid = str(ex.get("_id", ex.get("id", "")))
        ans = ex.get("answer", "")
        result[qid] = str(ans.get("answer", "") if isinstance(ans, dict) else ans)
    return result

def load_preds(path: str) -> Dict[str, str]:
    result = {}
    for rec in iter_jsonl(path):
        result[str(rec["qid"])] = str(rec.get("pred", ""))
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 4 — FEATURE BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_features(
    survivors: List[Tuple[int, str]],
    all_answers: List[str],
    nli_cands: List[dict],
    qa_cands:  List[dict],
    lex_cands: List[dict],
) -> np.ndarray:
    """Build full 19-feature matrix for survivors of one question."""
    n_surv = len(survivors)
    if n_surv == 0:
        return np.empty((0, len(ALL_19)), dtype=np.float32)

    all_norms = [normalize(a) for a in all_answers]
    freq_orig  = collections.Counter(all_norms)
    m_total    = len(all_answers)
    majority_norm = freq_orig.most_common(1)[0][0]

    surv_norms = [normalize(ans) for _, ans in survivors]
    freq_filt  = collections.Counter(surv_norms)
    m_filt     = len(survivors)
    unique_count = len(freq_orig) / max(m_total, 1)

    # NLI scores for survivors
    surv_nli = []
    for orig_idx, _ in survivors:
        c = nli_cands[orig_idx] if orig_idx < len(nli_cands) else {}
        surv_nli.append(float(c.get("nli_flat", c.get("nli_score", 0)) or 0))

    nli_arr = np.array(surv_nli, dtype=np.float32)
    nli_max = nli_arr.max() if len(nli_arr) > 0 else 0.0

    if n_surv > 1:
        rank_order = np.argsort(-nli_arr)
        rank_map   = np.empty(n_surv, dtype=np.float32)
        rank_map[rank_order] = np.arange(n_surv) / (n_surv - 1)
    else:
        rank_map = np.array([0.0], dtype=np.float32)

    rows = []
    for si, (orig_idx, ans) in enumerate(survivors):
        norm = normalize(ans)
        cn   = nli_cands[orig_idx] if orig_idx < len(nli_cands) else {}
        cq   = qa_cands[orig_idx]  if orig_idx < len(qa_cands)  else {}
        cl   = lex_cands[orig_idx] if orig_idx < len(lex_cands) else {}

        # Surface features (10)
        nli_score  = surv_nli[si]
        nli_rank   = float(rank_map[si])
        nli_gap    = float(nli_max - nli_score)
        ans_freq   = freq_orig[norm] / max(m_total, 1)
        is_maj     = int(norm == majority_norm)
        len_c      = math.log1p(len(ans))
        len_w      = len(ans.split())
        c_idx      = orig_idx / max(m_total - 1, 1)
        u_count    = unique_count
        ans_freq_f = freq_filt[norm] / max(m_filt, 1)

        # Chain features (9)
        h1n = float(cn.get("nli_hop1", 0) or 0)
        h2n = float(cn.get("nli_hop2", 0) or 0)
        h1q = float(cq.get("qa_hop1",  0) or 0)
        h2q = float(cq.get("qa_hop2",  0) or 0)
        h1l = float(cl.get("lex_hop1", cl.get("ans_in_hop1", 0)) or 0)
        h2l = float(cl.get("lex_hop2", cl.get("ans_in_hop2", 0)) or 0)

        rows.append([
            nli_score, nli_rank, nli_gap,
            ans_freq, is_maj, len_c, len_w, c_idx, u_count, ans_freq_f,
            h1n, h2n, abs(h1n - h2n),
            h1q, h2q, abs(h1q - h2q),
            h1l, h2l, abs(h1l - h2l),
        ])

    return np.array(rows, dtype=np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 5 — XGB RUNNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_xgb(name, feat_idx, feat_names, qid_features, gold_map,
            fallback_preds, n_folds, seed, xgb_params, log):
    import xgboost as xgb_lib

    log.info(f"Running {name}: {len(feat_names)} features — {feat_names}")

    all_qids, all_X, all_y, all_ans = [], [], [], []
    qid_ranges = {}

    for qid in sorted(qid_features):
        X_q, survivors = qid_features[qid]
        if X_q.shape[0] == 0:
            continue
        gold = gold_map.get(qid, "")
        start = len(all_X)
        for si in range(X_q.shape[0]):
            _, ans = survivors[si]
            all_qids.append(qid)
            all_X.append(X_q[si, feat_idx])
            all_y.append(em(ans, gold))
            all_ans.append(ans)
        qid_ranges[qid] = (start, len(all_X))

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)
    log.info(f"  Matrix: {X.shape[0]} × {X.shape[1]}  "
             f"positive rate={y.mean():.4f}")

    unique_qids = sorted(set(all_qids))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(all_X), dtype=np.float64)
    fold_imps  = []

    for fold, (tr_qi, va_qi) in enumerate(kf.split(unique_qids)):
        tr_set = set(unique_qids[i] for i in tr_qi)
        va_set = set(unique_qids[i] for i in va_qi)
        tr_m = np.array([q in tr_set for q in all_qids])
        va_m = np.array([q in va_set for q in all_qids])
        clf  = xgb_lib.XGBClassifier(**xgb_params)
        clf.fit(X[tr_m], y[tr_m])
        oof_probs[va_m] = clf.predict_proba(X[va_m])[:, 1]
        fold_imps.append(clf.feature_importances_)
        log.info(f"  Fold {fold+1}/{n_folds}")

    preds = {}
    for qid, (s, e) in qid_ranges.items():
        best_i = int(np.argmax(oof_probs[s:e]))
        preds[qid] = {"pred": all_ans[s + best_i]}

    for qid in gold_map:
        if qid not in preds:
            preds[qid] = {"pred": fallback_preds.get(qid, ""), "fallback": True}

    correct = sum(em(preds[q]["pred"], gold_map[q]) for q in gold_map)
    em_score = correct / len(gold_map)

    mean_imp = np.mean(fold_imps, axis=0)
    imp_pairs = sorted(zip(feat_names, mean_imp), key=lambda x: -x[1])

    log.info(f"  {name}: {correct}/{len(gold_map)} = {em_score:.4f} EM")
    log.info(f"  Top features:")
    for i, (f, v) in enumerate(imp_pairs[:6]):
        log.info(f"    #{i+1}  {f:<24}  {v:.4f}")

    return em_score, preds, imp_pairs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 6 — BOOTSTRAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def bootstrap_delta(vec_a, vec_b, qid_order, gold_map, n_boot=10000, seed=42):
    rng   = np.random.default_rng(seed)
    a_arr = np.array([em(vec_a.get(q, {}).get("pred",""), gold_map[q])
                      for q in qid_order], dtype=np.int8)
    b_arr = np.array([em(vec_b.get(q, {}).get("pred",""), gold_map[q])
                      for q in qid_order], dtype=np.int8)
    obs   = a_arr.mean() - b_arr.mean()
    n     = len(qid_order)
    boot_deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_deltas[i] = a_arr[idx].mean() - b_arr[idx].mean()
    lo, hi = np.percentile(boot_deltas, [2.5, 97.5])
    p = float((boot_deltas <= 0).mean()) if obs >= 0 else float((boot_deltas >= 0).mean())
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    return {"delta_pp": round(obs*100, 2), "ci_lo": round(lo*100, 2),
            "ci_hi": round(hi*100, 2), "p": round(p, 4), "sig": sig}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 7 — SETTING CONFIGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_settings(R):
    return {
        "hotpotqa_mdr": {
            "label":      "HotpotQA MDR (7B)",
            "candidates": f"{R}/exp0c/candidates/dev_M5_7b_K200.jsonl",
            "hop":        f"{R}/exp0c/preds/dev_hop_scores.jsonl",
            "qa":         f"{R}/exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl",
            "lex":        f"{R}/exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl",
            "fallback":   f"{R}/exp0c/preds/dev_chain_verifier_mean_preds.jsonl",
            "gold":       f"{R}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json",
            "gold_type":  "hotpot",
        },
        "hotpotqa_distractor": {
            "label":      "HotpotQA Distractor (7B)",
            "candidates": f"{R}/exp_distractor/candidates/dev_M5_sampling.jsonl",
            "hop":        f"{R}/exp_distractor/preds/dev_hop_scores.jsonl",
            "qa":         f"{R}/exp_distractor/preds/dev_qa_hop_scores.jsonl",
            "lex":        f"{R}/exp_distractor/preds/dev_lex_features.jsonl",
            "fallback":   f"{R}/exp0c/preds/dev_chain_verifier_mean_preds.jsonl",
            "gold":       f"{R}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json",
            "gold_type":  "hotpot",
        },
        "wiki2": {
            "label":      "2WikiMultiHopQA (7B)",
            "candidates": f"{R}/exp_wiki2/candidates/dev_M5_sampling.jsonl",
            "hop":        f"{R}/exp_wiki2/preds/dev_hop_scores.jsonl",
            "qa":         f"{R}/exp_wiki2/preds/dev_qa_hop_scores.jsonl",
            "lex":        f"{R}/exp_wiki2/preds/dev_lex_features.jsonl",
            "fallback":   f"{R}/exp0c/preds/dev_chain_verifier_mean_preds.jsonl",
            "gold":       f"{R}/data/wiki2/raw/dev_normalized.json",
            "gold_type":  "wiki2",
        },
        "hotpotqa_1p5b": {
            "label":      "HotpotQA MDR (1.5B)",
            "candidates": f"{R}/exp1b/candidates/dev_M5_candidates_qwen.jsonl",
            "hop":        f"{R}/exp1b/preds/dev_hop_scores.jsonl",
            "qa":         f"{R}/exp1b/preds/dev_qa_hop_scores.jsonl",
            "lex":        f"{R}/exp1b/preds/dev_lex_features.jsonl",
            "fallback":   f"{R}/exp0c/preds/dev_chain_verifier_mean_preds.jsonl",
            "gold":       f"{R}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json",
            "gold_type":  "hotpot",
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 8 — MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--proj_root",
                    default="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot")
    ap.add_argument("--out_dir",   default="experiments/results/z2nov")
    ap.add_argument("--setting",   default="all",
                    help="hotpotqa_mdr | hotpotqa_distractor | wiki2 | "
                         "hotpotqa_1p5b | all")
    ap.add_argument("--n_folds",   type=int, default=5)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--n_boot",    type=int, default=10000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    R = args.proj_root

    log = logging.getLogger("z2nov")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    sh  = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh  = logging.FileHandler(f"{args.out_dir}/z2nov.log", mode="w")
    fh.setFormatter(fmt)
    log.addHandler(sh); log.addHandler(fh)

    xgb_params = dict(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        random_state=args.seed, n_jobs=-1, verbosity=0,
    )

    all_settings = get_settings(R)
    if args.setting == "all":
        to_run = list(all_settings.keys())
    else:
        to_run = [args.setting]

    all_results = {}

    for key in to_run:
        cfg = all_settings[key]
        log.info(f"\n{'━'*65}")
        log.info(f"  {cfg['label']}")
        log.info(f"{'━'*65}")

        # Check files
        missing = [n for n, p in [
            ("candidates", cfg["candidates"]), ("hop",  cfg["hop"]),
            ("qa",         cfg["qa"]),          ("lex",  cfg["lex"]),
            ("fallback",   cfg["fallback"]),    ("gold", cfg["gold"]),
        ] if not os.path.exists(p)]
        if missing:
            log.warning(f"  SKIPPED — missing files: {missing}")
            continue

        # Load
        load_gold = load_gold_hotpot if cfg["gold_type"] == "hotpot" \
                    else load_gold_wiki2
        gold       = load_gold(cfg["gold"])
        candidates = load_candidates(cfg["candidates"])
        hop        = load_hop_scores(cfg["hop"])
        qa         = load_qa_lex(cfg["qa"])
        lex        = load_qa_lex(cfg["lex"])
        fallback   = load_preds(cfg["fallback"])
        log.info(f"  Loaded: {len(gold):,} questions")

        # Stage 1 filter
        qid_survivors = {}
        all_filtered  = []
        for qid in sorted(gold):
            cands    = candidates.get(qid, [])
            survs    = [(i, a) for i, a in enumerate(cands)
                        if not is_bad(a) and not is_unknown_safe(a)]
            qid_survivors[qid] = survs
            if not survs:
                all_filtered.append(qid)

        log.info(f"  All-filtered questions: {len(all_filtered):,}")

        # Build full 19-feature matrices
        qid_features = {}
        for qid in sorted(gold):
            survs = qid_survivors[qid]
            if not survs:
                continue
            cands   = candidates.get(qid, [])
            nli_c   = hop.get(qid, [{}]*len(cands))
            qa_c    = qa.get(qid,  [{}]*len(cands))
            lex_c   = lex.get(qid, [{}]*len(cands))
            X_q     = build_features(survs, cands, nli_c, qa_c, lex_c)
            qid_features[qid] = (X_q, survs)

        log.info(f"  Feature matrices: {len(qid_features):,} questions")

        # Run Z2-NoV
        z2nov_em, z2nov_preds, z2nov_imp = run_xgb(
            "Z2_NoV", Z2_NOV_IDX, Z2_NOV_FEATURES,
            qid_features, gold, fallback,
            args.n_folds, args.seed, xgb_params, log
        )

        # Run Z3
        z3_em, z3_preds, z3_imp = run_xgb(
            "Z3_chain", Z3_IDX, Z3_FEATURES,
            qid_features, gold, fallback,
            args.n_folds, args.seed, xgb_params, log
        )

        # Bootstrap Z3 vs Z2-NoV
        qid_order = sorted(gold.keys())
        boot = bootstrap_delta(z3_preds, z2nov_preds, qid_order, gold,
                               n_boot=args.n_boot, seed=args.seed)

        log.info(f"\n  ── RESULT: Z3 vs Z2-NoV ──")
        log.info(f"  Z2-NoV (flat NLI, no voting) : {z2nov_em:.4f} EM")
        log.info(f"  Z3     (chain, no voting)    : {z3_em:.4f} EM")
        log.info(f"  Δ Z3 − Z2-NoV = {boot['delta_pp']:+.2f}pp  "
                 f"95%CI [{boot['ci_lo']:+.2f}, {boot['ci_hi']:+.2f}]  "
                 f"p={boot['p']:.4f}  {boot['sig']}")
        log.info(f"  Interpretation: {'Z3 (per-hop) IS stronger than flat NLI alone' if boot['delta_pp'] > 0 else 'Z3 does NOT beat flat NLI alone'}")

        # Save preds
        for tag, pd in [("z2nov", z2nov_preds), ("z3", z3_preds)]:
            path = os.path.join(args.out_dir, f"{key}_{tag}_preds.jsonl")
            with open(path, "w") as f:
                for qid in sorted(pd):
                    f.write(json.dumps({"qid": qid, **pd[qid]}) + "\n")

        # Save stats
        result = {
            "setting": key, "label": cfg["label"],
            "z2nov_em": z2nov_em, "z3_em": z3_em,
            "z2nov_features": Z2_NOV_FEATURES,
            "z3_features":    Z3_FEATURES,
            "z2nov_importances": {k: round(float(v),4) for k,v in z2nov_imp},
            "z3_importances":    {k: round(float(v),4) for k,v in z3_imp},
            "z3_vs_z2nov": boot,
        }
        stats_path = os.path.join(args.out_dir, f"{key}_z2nov_stats.json")
        with open(stats_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"  Saved → {stats_path}")
        all_results[key] = result

    # Cross-setting summary
    log.info(f"\n{'━'*65}")
    log.info(f"  CROSS-SETTING SUMMARY — Z3 vs Z2-NoV (symmetric, no voting either side)")
    log.info(f"{'━'*65}")
    log.info(f"  {'Setting':<26}  {'Z2-NoV':>8}  {'Z3':>8}  "
             f"{'Δ(pp)':>8}  {'95% CI':>18}  Sig")
    log.info(f"  {'─'*72}")
    for key, r in all_results.items():
        b = r["z3_vs_z2nov"]
        log.info(f"  {r['label']:<26}  "
                 f"{r['z2nov_em']:>8.4f}  {r['z3_em']:>8.4f}  "
                 f"{b['delta_pp']:>+7.2f}pp  "
                 f"[{b['ci_lo']:+.2f},{b['ci_hi']:+.2f}]  {b['sig']}")

    log.info(f"\n  Z2-NoV features (6): {Z2_NOV_FEATURES}")
    log.info(f"  Z3     features (10): {Z3_FEATURES}")
    log.info(f"\n  All outputs → {args.out_dir}/")

    # Save all results
    with open(os.path.join(args.out_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()