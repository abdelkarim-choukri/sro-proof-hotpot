#!/usr/bin/env python3
"""
exp1_xgb_verifier.py — Step 6: Chain-aware XGBoost verifier (Option 1)

Approach:
  - Extract per-(question, candidate) features: NLI score, answer frequency,
    length, position, diversity, etc.
  - Label: EM match with gold answer (binary)
  - Train/eval via 5-fold cross-validation on dev set (no separate train set)
  - At inference: pick candidate with highest verifier score per question
  - Reports EM/F1 overall + feasible subset + ECE calibration

No tuning on full dev — predictions come from held-out CV folds only.

Usage:
    python3 tools/exp1_xgb_verifier.py \
        --candidates  exp1/candidates/dev_M5_candidates_qwen_v4.jsonl \
        --evidence    exp1/evidence/dev_K20_chains.jsonl \
        --gold        data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
        --nli_preds   exp1/preds/dev_nli_preds.jsonl \
        --out_metrics exp1/metrics/exp1_step6_xgb.json \
        --out_preds   exp1/preds/dev_xgb_preds.jsonl \
        --n_folds     5 \
        --seed        42
"""
import argparse, collections, json, math, os, re, string
import numpy as np
from sklearn.model_selection import KFold
from sklearn.calibration import calibration_curve
import xgboost as xgb

# ──────────────────────────── text utils ────────────────────────────
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

# ──────────────────────────── ECE ────────────────────────────────────
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (Guo et al. 2017)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)

# ──────────────────────────── feature extraction ────────────────────
def extract_features(candidates: list, nli_scores: list) -> np.ndarray:
    """
    Returns array of shape (n_cands, n_features).

    Features per candidate:
      0  nli_score          — entailment probability from cross-encoder
      1  nli_rank           — rank by nli_score (0 = best), normalised to [0,1]
      2  nli_score_gap      — max_nli - this_nli  (how far below the best)
      3  answer_freq        — fraction of candidates with same normalised answer
      4  is_majority        — 1 if this is the plurality answer
      5  answer_len_chars   — character length (log-scaled)
      6  answer_len_words   — word count
      7  is_empty           — 1 if empty
      8  is_unknown         — 1 if answer is "unknown"
      9  is_bad             — 1 if flagged by is_bad()
      10 cand_idx           — position in candidate list (0-4), normalised
      11 unique_count       — number of distinct answers / M (diversity)
    """
    m = len(candidates)
    norms = [normalize(a) for a in candidates]

    # frequency counts
    freq_counter = collections.Counter(norms)
    majority_norm = freq_counter.most_common(1)[0][0]
    unique_count  = len(freq_counter) / m

    # NLI ranks
    nli_arr  = np.array(nli_scores, dtype=np.float32)
    nli_max  = nli_arr.max()
    ranks    = np.argsort(-nli_arr)          # descending
    rank_map = np.empty(m, dtype=np.float32)
    rank_map[ranks] = np.arange(m) / max(m - 1, 1)

    rows = []
    for i, (ans, norm) in enumerate(zip(candidates, norms)):
        freq     = freq_counter[norm] / m
        is_maj   = int(norm == majority_norm)
        alen_c   = math.log1p(len(ans))
        alen_w   = len(ans.split())
        i_empty  = int(not ans.strip())
        i_unk    = int(norm in {"unknown", "unk", ""})
        i_bad    = int(is_bad(ans))
        c_idx    = i / max(m - 1, 1)
        nli_gap  = float(nli_max - nli_arr[i])

        rows.append([
            float(nli_arr[i]),   # 0
            float(rank_map[i]),  # 1
            nli_gap,             # 2
            freq,                # 3
            is_maj,              # 4
            alen_c,              # 5
            alen_w,              # 6
            i_empty,             # 7
            i_unk,               # 8
            i_bad,               # 9
            c_idx,               # 10
            unique_count,        # 11
        ])
    return np.array(rows, dtype=np.float32)

FEATURE_NAMES = [
    "nli_score", "nli_rank", "nli_score_gap",
    "answer_freq", "is_majority",
    "answer_len_chars", "answer_len_words",
    "is_empty", "is_unknown", "is_bad",
    "cand_idx", "unique_count",
]

# ──────────────────────────── main ──────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",   required=True)
    ap.add_argument("--evidence",     required=True)
    ap.add_argument("--gold",         required=True)
    ap.add_argument("--nli_preds",    required=True)
    ap.add_argument("--out_metrics",  required=True)
    ap.add_argument("--out_preds",    required=True)
    ap.add_argument("--n_folds",      type=int, default=5)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--xgb_n_estimators", type=int,   default=300)
    ap.add_argument("--xgb_max_depth",    type=int,   default=4)
    ap.add_argument("--xgb_lr",           type=float, default=0.05)
    ap.add_argument("--xgb_subsample",    type=float, default=0.8)
    ap.add_argument("--xgb_colsample",    type=float, default=0.8)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_preds),   exist_ok=True)

    # ── load inputs ──
    gold_map = {q['_id']: q['answer']
                for q in json.load(open(args.gold))}

    feasible = set()
    for line in open(args.evidence):
        r = json.loads(line)
        gold_titles = set(r['gold']['supporting_titles'])
        retrieved   = set(hop['title']
                          for ch in r['chains'] for hop in ch['hops'])
        if gold_titles.issubset(retrieved):
            feasible.add(r['qid'])
    print(f"[xgb] Feasible subset: {len(feasible)}/7405")

    nli_map = {}
    for line in open(args.nli_preds):
        r = json.loads(line)
        nli_map[r['qid']] = r['scores']   # list of 5 floats

    # ── build dataset ──
    # one row per (question, candidate)
    all_qids   = []
    all_X      = []   # feature rows
    all_y      = []   # EM label
    all_cands  = {}   # qid -> list of answer strings

    for line in open(args.candidates):
        r      = json.loads(line)
        qid    = r['qid']
        cands  = [c['answer_text'] for c in r['candidates']]
        gold   = gold_map.get(qid, '')
        scores = nli_map.get(qid, [0.0] * len(cands))

        feats  = extract_features(cands, scores)   # (m, n_feat)
        labels = [em(a, gold) for a in cands]

        all_qids.extend([qid] * len(cands))
        all_X.extend(feats.tolist())
        all_y.extend(labels)
        all_cands[qid] = cands

    X = np.array(all_X,  dtype=np.float32)
    y = np.array(all_y,  dtype=np.float32)
    qids = np.array(all_qids)

    print(f"[xgb] Dataset: {len(np.unique(qids))} questions, "
          f"{len(X)} rows, {X.shape[1]} features")
    print(f"[xgb] Positive rate: {y.mean():.3f}")

    # ── 5-fold cross-validation ──
    # Split at question level (not row level) to avoid leakage
    unique_qids = np.unique(qids)
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    oof_probs = np.zeros(len(X), dtype=np.float32)   # out-of-fold probabilities
    fold_importances = []

    for fold, (train_qi, val_qi) in enumerate(kf.split(unique_qids)):
        train_qids_set = set(unique_qids[train_qi])
        val_qids_set   = set(unique_qids[val_qi])

        train_mask = np.array([q in train_qids_set for q in qids])
        val_mask   = np.array([q in val_qids_set   for q in qids])

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_val      = X[val_mask]

        clf = xgb.XGBClassifier(
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
        clf.fit(X_tr, y_tr)
        oof_probs[val_mask] = clf.predict_proba(X_val)[:, 1]
        fold_importances.append(clf.feature_importances_)

        val_q_count = len(val_qids_set)
        print(f"[xgb] Fold {fold+1}/{args.n_folds} — "
              f"train={len(train_qids_set)}q  val={val_q_count}q")

    # ── pick best candidate per question using OOF probs ──
    preds = {}
    qid_to_rows = collections.defaultdict(list)
    for i, qid in enumerate(qids):
        qid_to_rows[qid].append(i)

    for qid, row_idxs in qid_to_rows.items():
        cands  = all_cands[qid]
        probs  = [float(oof_probs[i]) for i in row_idxs]
        best_i = int(np.argmax(probs))
        preds[qid] = {
            "pred":     cands[best_i],
            "probs":    probs,
            "best_idx": best_i,
        }

    # ── save predictions ──
    with open(args.out_preds, 'w') as f:
        for qid, p in preds.items():
            f.write(json.dumps({"qid": qid, **p}) + "\n")
    print(f"[xgb] Predictions saved to {args.out_preds}")

    # ── metrics ──
    results = {}
    for split, qid_filter in [('overall', None), ('feasible', feasible)]:
        xgb_em = xgb_f1 = n = 0
        for qid, p in preds.items():
            if qid_filter is not None and qid not in qid_filter:
                continue
            g = gold_map.get(qid, '')
            xgb_em += em(p['pred'], g)
            xgb_f1 += f1_score(p['pred'], g)
            n += 1
        results[split] = {
            'n':      n,
            'xgb_em': round(xgb_em / n, 4),
            'xgb_f1': round(xgb_f1 / n, 4),
        }

    # ── ECE on overall ──
    # Use max prob per question as confidence, EM as correctness
    conf_list  = []
    label_list = []
    for qid, p in preds.items():
        conf_list.append(max(p['probs']))
        label_list.append(em(p['pred'], gold_map.get(qid, '')))
    conf_arr  = np.array(conf_list)
    label_arr = np.array(label_list)
    ece = compute_ece(conf_arr, label_arr)
    results['ece'] = round(ece, 4)

    # ── feature importances ──
    mean_imp = np.mean(fold_importances, axis=0)
    results['feature_importances'] = {
        name: round(float(imp), 4)
        for name, imp in sorted(
            zip(FEATURE_NAMES, mean_imp),
            key=lambda x: -x[1]
        )
    }

    json.dump(results, open(args.out_metrics, 'w'), indent=2)
    print(json.dumps(results, indent=2))
    print(f"[xgb] Metrics saved to {args.out_metrics}")

    # ── summary table ──
    print("\n" + "="*55)
    print(f"{'Method':<20} {'Overall EM':>10} {'Overall F1':>10}")
    print("-"*55)
    print(f"{'Baseline-0':<20} {'0.3028':>10} {'0.4294':>10}")
    print(f"{'Majority vote':<20} {'0.3070':>10} {'0.4264':>10}")
    print(f"{'NLI cross-enc':<20} {'0.3041':>10} {'0.4374':>10}")
    print(f"{'XGBoost verifier':<20} {results['overall']['xgb_em']:>10.4f} "
          f"{results['overall']['xgb_f1']:>10.4f}")
    print(f"{'Oracle@5':<20} {'0.4004':>10} {'0.5369':>10}")
    print("="*55)
    print(f"ECE (calibration): {ece:.4f}")


if __name__ == "__main__":
    main()