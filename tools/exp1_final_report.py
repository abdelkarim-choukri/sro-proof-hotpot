#!/usr/bin/env python3
"""
exp1_final_report.py — Step 7: Consolidated Exp1 metrics report

Produces:
  - exp1/metrics/exp1_final_report.json   — all metrics in one file
  - exp1/metrics/exp1_accuracy_coverage.json — accuracy-coverage curve data
  - Prints a clean summary table to stdout

Usage:
    python3 tools/exp1_final_report.py \
        --candidates  exp1/candidates/dev_M5_candidates_qwen_v4.jsonl \
        --evidence    exp1/evidence/dev_K20_chains.jsonl \
        --gold        data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
        --nli_preds   exp1/preds/dev_nli_preds.jsonl \
        --xgb_preds   exp1/preds/dev_xgb_preds.jsonl \
        --metrics_dir exp1/metrics \
        --out_report  exp1/metrics/exp1_final_report.json \
        --out_curve   exp1/metrics/exp1_accuracy_coverage.json
"""
import argparse, collections, json, os, re, string, time
import numpy as np

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
    if not n: return 0.0
    p = n / len(p_toks); r = n / len(g_toks)
    return 2 * p * r / (p + r)

def majority_vote(answers: list) -> str:
    counts = collections.Counter(normalize(a) for a in answers if a.strip())
    if not counts:
        return answers[0] if answers else ""
    top_norm = counts.most_common(1)[0][0]
    for a in answers:
        if normalize(a) == top_norm:
            return a
    return answers[0]

# ──────────────────────────── ECE ───────────────────────────────────
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0: continue
        ece += mask.sum() / n * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)

# ──────────────────────────── accuracy-coverage ─────────────────────
def accuracy_coverage_curve(confidences: list, correct: list, n_steps: int = 20):
    """
    Abstention curve: sort by confidence descending, compute accuracy
    on the top-k% of questions (coverage = k/total).
    Returns list of {coverage, accuracy, n} dicts.
    """
    pairs = sorted(zip(confidences, correct), key=lambda x: -x[0])
    total = len(pairs)
    curve = []
    for step in range(1, n_steps + 1):
        k = max(1, int(total * step / n_steps))
        subset = pairs[:k]
        acc = sum(c for _, c in subset) / k
        curve.append({
            "coverage":  round(k / total, 4),
            "accuracy":  round(acc, 4),
            "n":         k,
        })
    return curve

# ──────────────────────────── main ──────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",  required=True)
    ap.add_argument("--evidence",    required=True)
    ap.add_argument("--gold",        required=True)
    ap.add_argument("--nli_preds",   required=True)
    ap.add_argument("--xgb_preds",   required=True)
    ap.add_argument("--metrics_dir", required=True)
    ap.add_argument("--out_report",  required=True)
    ap.add_argument("--out_curve",   required=True)
    args = ap.parse_args()

    os.makedirs(args.metrics_dir, exist_ok=True)

    # ── load gold ──
    gold_map = {q['_id']: q['answer']
                for q in json.load(open(args.gold))}

    # ── feasible subset ──
    feasible = set()
    for line in open(args.evidence):
        r = json.loads(line)
        gold_titles = set(r['gold']['supporting_titles'])
        retrieved   = set(hop['title']
                          for ch in r['chains'] for hop in ch['hops'])
        if gold_titles.issubset(retrieved):
            feasible.add(r['qid'])
    print(f"[report] Feasible subset: {len(feasible)}/7405")

    # ── load candidates ──
    cand_map = {}
    for line in open(args.candidates):
        r = json.loads(line)
        cand_map[r['qid']] = [c['answer_text'] for c in r['candidates']]

    # ── load NLI preds ──
    nli_map = {}
    for line in open(args.nli_preds):
        r = json.loads(line)
        nli_map[r['qid']] = r

    # ── load XGB preds ──
    xgb_map = {}
    for line in open(args.xgb_preds):
        r = json.loads(line)
        xgb_map[r['qid']] = r

    # ── compute all methods ──
    methods = {
        "baseline0":    {"em": 0, "f1": 0},
        "majority_vote":{"em": 0, "f1": 0},
        "nli":          {"em": 0, "f1": 0},
        "xgb":          {"em": 0, "f1": 0},
        "oracle5":      {"em": 0, "f1": 0},
    }
    methods_feasible = {k: {"em": 0, "f1": 0} for k in methods}

    # for ECE and coverage curves (XGB only — it has calibrated probs)
    xgb_conf_overall  = []
    xgb_corr_overall  = []
    xgb_conf_feasible = []
    xgb_corr_feasible = []

    n_overall = n_feasible = 0

    for qid, cands in cand_map.items():
        g = gold_map.get(qid, '')

        # predictions
        pred_b0  = cands[0]
        pred_mv  = majority_vote(cands)
        pred_nli = nli_map[qid]['pred']  if qid in nli_map  else cands[0]
        pred_xgb = xgb_map[qid]['pred']  if qid in xgb_map  else cands[0]

        # oracle
        best_em  = max(em(c, g)       for c in cands)
        best_f1  = max(f1_score(c, g) for c in cands)

        # XGB confidence = max prob across candidates
        xgb_conf = max(xgb_map[qid]['probs']) if qid in xgb_map else 0.5
        xgb_corr = em(pred_xgb, g)

        for split in ['overall', 'feasible' if qid in feasible else None]:
            if split is None: continue
            d = methods if split == 'overall' else methods_feasible

            d['baseline0']['em']    += em(pred_b0, g)
            d['baseline0']['f1']    += f1_score(pred_b0, g)
            d['majority_vote']['em']+= em(pred_mv, g)
            d['majority_vote']['f1']+= f1_score(pred_mv, g)
            d['nli']['em']          += em(pred_nli, g)
            d['nli']['f1']          += f1_score(pred_nli, g)
            d['xgb']['em']          += em(pred_xgb, g)
            d['xgb']['f1']          += f1_score(pred_xgb, g)
            d['oracle5']['em']      += best_em
            d['oracle5']['f1']      += best_f1

        n_overall += 1
        xgb_conf_overall.append(xgb_conf)
        xgb_corr_overall.append(xgb_corr)
        if qid in feasible:
            n_feasible += 1
            xgb_conf_feasible.append(xgb_conf)
            xgb_corr_feasible.append(xgb_corr)

    # ── normalise ──
    def norm_dict(d, n):
        return {k: {"em": round(v["em"]/n, 4), "f1": round(v["f1"]/n, 4)}
                for k, v in d.items()}

    overall_metrics  = norm_dict(methods,          n_overall)
    feasible_metrics = norm_dict(methods_feasible, n_feasible)

    # ── ECE ──
    ece_overall  = compute_ece(np.array(xgb_conf_overall),
                               np.array(xgb_corr_overall))
    ece_feasible = compute_ece(np.array(xgb_conf_feasible),
                               np.array(xgb_corr_feasible))

    # ── accuracy-coverage curves ──
    curve_overall  = accuracy_coverage_curve(xgb_conf_overall,  xgb_corr_overall)
    curve_feasible = accuracy_coverage_curve(xgb_conf_feasible, xgb_corr_feasible)

    # ── assemble report ──
    report = {
        "meta": {
            "experiment":    "exp1",
            "description":   "Mode-2 reranking sanity check — multi-candidate answers",
            "generated_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_overall":     n_overall,
            "n_feasible":    n_feasible,
            "model":         "Qwen2.5-1.5B-Instruct",
            "M":             5,
            "retriever":     "MDR K=20",
        },
        "overall":  overall_metrics,
        "feasible": feasible_metrics,
        "calibration": {
            "xgb_ece_overall":  round(ece_overall,  4),
            "xgb_ece_feasible": round(ece_feasible, 4),
            "note": "ECE computed on max(probs) per question vs EM correctness",
        },
        "exp0_retrieval_reference": {
            "doc_recall_union":   0.8684672518568535,
            "chain_hit":          0.7954085077650236,
            "sf_recall_lenient":  0.8615940016662038,
        },
    }

    json.dump(report, open(args.out_report, 'w'), indent=2)
    print(f"[report] Report saved to {args.out_report}")

    curve_out = {"overall": curve_overall, "feasible": curve_feasible}
    json.dump(curve_out, open(args.out_curve, 'w'), indent=2)
    print(f"[report] Accuracy-coverage curve saved to {args.out_curve}")

    # ── print summary ──
    W = 62
    print("\n" + "=" * W)
    print(f" Exp1 Final Results  —  Overall (n={n_overall})")
    print("=" * W)
    print(f"{'Method':<22} {'EM':>8} {'F1':>8}  {'vs Base0':>9}")
    print("-" * W)
    base_em = overall_metrics['baseline0']['em']
    base_f1 = overall_metrics['baseline0']['f1']
    labels = {
        'baseline0':     'Baseline-0 (cand#0)',
        'majority_vote': 'Majority vote',
        'nli':           'NLI cross-encoder',
        'xgb':           'XGBoost verifier ★',
        'oracle5':       'Oracle@5 (ceiling)',
    }
    for key, label in labels.items():
        m   = overall_metrics[key]
        gap = f"+{m['em']-base_em:.4f}" if m['em'] > base_em else f"{m['em']-base_em:.4f}"
        print(f"  {label:<20} {m['em']:>8.4f} {m['f1']:>8.4f}  {gap:>9}")
    print("=" * W)

    print(f"\n Feasible subset (DocRecall@20=1, n={n_feasible})")
    print("-" * W)
    base_em_f = feasible_metrics['baseline0']['em']
    for key, label in labels.items():
        m   = feasible_metrics[key]
        gap = f"+{m['em']-base_em_f:.4f}" if m['em'] > base_em_f else f"{m['em']-base_em_f:.4f}"
        print(f"  {label:<20} {m['em']:>8.4f} {m['f1']:>8.4f}  {gap:>9}")
    print("=" * W)

    print(f"\n Calibration (XGBoost verifier)")
    print(f"  ECE overall:  {ece_overall:.4f}  (0=perfect, lower is better)")
    print(f"  ECE feasible: {ece_feasible:.4f}")

    print(f"\n Accuracy-Coverage (XGBoost, overall, top-K% by confidence)")
    print(f"  {'Coverage':>10}  {'Accuracy':>10}")
    for pt in curve_overall[::4]:
        print(f"  {pt['coverage']:>10.0%}  {pt['accuracy']:>10.4f}")
    print("=" * W)


if __name__ == "__main__":
    main()