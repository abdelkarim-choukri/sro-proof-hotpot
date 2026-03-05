#!/usr/bin/env python3
"""
exp2_q0_hop_diagnostic.py — Q0: Does flat NLI systematically pick answers
                             supported by ONE hop but not both?

For every question where NLI chose the wrong answer, we re-score that wrong
answer independently against:
  - hop 1 text of the top chain  (premise=hop1_text, hypothesis=wrong_answer)
  - hop 2 text of the top chain  (premise=hop2_text, hypothesis=wrong_answer)

We then compute |nli_hop1 - nli_hop2| (the imbalance) and check what fraction
of NLI failures show a gap above a threshold. The same scoring is done for
NLI-correct questions to give a meaningful baseline for comparison.

Decision rule (threshold=0.3):
  pct_imbalanced >= 25%  → hypothesis confirmed, proceed to chain-aware verifier
  pct_imbalanced  15-25% → discuss before proceeding
  pct_imbalanced <= 15%  → central claim needs rethinking

Usage (from project root, in llm conda env):
    /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/exp2_q0_hop_diagnostic.py \
        --nli_preds   exp1b/preds/dev_nli_preds.jsonl \
        --evidence    exp1b/evidence/dev_K100_chains.jsonl \
        --gold        data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
        --model       /var/tmp/u24sf51014/sro/models/nli-roberta-base \
        --out_json    exp1b/metrics/q0_hop_diagnostic.json \
        --out_jsonl   exp1b/metrics/q0_hop_diagnostic_perqid.jsonl \
        --log         exp1b/logs/q0_diagnostic.log \
        --batch_size  64
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

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ─────────────────────────── text utils ─────────────────────────────

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def em(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


# ─────────────────────────── NLI batch ──────────────────────────────

def score_batch(model, tokenizer, device: str,
                premises: list, hypotheses: list,
                entail_idx: int) -> list:
    """Return entailment probabilities for a batch of (premise, hypothesis) pairs."""
    enc = tokenizer(
        premises, hypotheses,
        truncation=True, max_length=512, padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[:, entail_idx].tolist()


# ─────────────────────────── helpers ────────────────────────────────

def get_hop_texts(chains: list):
    """
    Extract (hop1_text, hop2_text) from the top-ranked chain (index 0).
    Returns (None, None) if the chain is missing or has fewer than 2 hops.
    hop_text = "Title: passage_text" — same format used by flatten_evidence.
    """
    if not chains:
        return None, None
    top_chain = chains[0]
    hops = top_chain.get("hops", [])
    if len(hops) < 2:
        return None, None
    h1 = hops[0]
    h2 = hops[1]
    hop1_text = f"{h1.get('title', '')}: {h1.get('text', '')}".strip()
    hop2_text = f"{h2.get('title', '')}: {h2.get('text', '')}".strip()
    return hop1_text, hop2_text


# ─────────────────────────── main ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli_preds",   required=True,
                    help="exp1b/preds/dev_nli_preds.jsonl")
    ap.add_argument("--evidence",    required=True,
                    help="exp1b/evidence/dev_K100_chains.jsonl")
    ap.add_argument("--gold",        required=True,
                    help="data/hotpotqa/raw/hotpot_dev_distractor_v1.json")
    ap.add_argument("--model",       required=True,
                    help="Path to nli-roberta-base")
    ap.add_argument("--out_json",    required=True,
                    help="Summary JSON output path")
    ap.add_argument("--out_jsonl",   required=True,
                    help="Per-question JSONL output path")
    ap.add_argument("--log",         required=True,
                    help="Log file path")
    ap.add_argument("--batch_size",  type=int, default=64)
    ap.add_argument("--threshold",   type=float, default=0.3,
                    help="Primary imbalance threshold for reporting")
    args = ap.parse_args()

    # ── output dirs ──
    for p in [args.out_json, args.out_jsonl, args.log]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # ── logging ──
    log = logging.getLogger("q0")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(args.log, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)

    log.info("=== Q0 Hop Imbalance Diagnostic ===")
    log.info(f"nli_preds : {args.nli_preds}")
    log.info(f"evidence  : {args.evidence}")
    log.info(f"gold      : {args.gold}")
    log.info(f"model     : {args.model}")
    log.info(f"threshold : {args.threshold}")
    log.info(f"batch_size: {args.batch_size}")

    # ── load model ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading NLI model on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    # Detect entailment label index dynamically (roberta-base-nli: 2 = entailment)
    label2id = {k.lower(): v for k, v in model.config.label2id.items()}
    entail_idx = label2id.get("entailment", 2)
    log.info(f"Label mapping: {model.config.id2label}  →  entailment_idx={entail_idx}")

    # ── load gold answers ──
    log.info("Loading gold answers ...")
    gold_map: dict[str, str] = {}
    with open(args.gold) as f:
        for ex in json.load(f):
            gold_map[str(ex["_id"])] = ex["answer"]
    log.info(f"  {len(gold_map)} gold answers loaded")

    # ── load evidence pack: qid → chains list ──
    log.info("Loading evidence pack ...")
    ev_map: dict[str, list] = {}
    n_ev_loaded = 0
    with open(args.evidence) as f:
        for line in f:
            r = json.loads(line)
            qid = str(r["qid"])
            # exp1b schema: evidence.chains
            chains = (r.get("evidence", {}).get("chains")
                      or r.get("chains", []))
            ev_map[qid] = chains
            n_ev_loaded += 1
    log.info(f"  {n_ev_loaded} evidence records loaded")

    # ── load NLI predictions ──
    log.info("Loading NLI predictions ...")
    nli_map: dict[str, dict] = {}
    with open(args.nli_preds) as f:
        for line in f:
            r = json.loads(line)
            nli_map[str(r["qid"])] = r
    log.info(f"  {len(nli_map)} NLI prediction records loaded")

    # ─────────────────────────────────────────────────────────────────
    # Build batch scoring job
    # For EVERY question (wrong and correct) we score:
    #   (hop1_text, pred_answer)  and  (hop2_text, pred_answer)
    # This lets us compare imbalance in wrong vs correct cases.
    # ─────────────────────────────────────────────────────────────────

    # We'll accumulate pairs in a flat list, track (qid, hop_idx) per pair
    premises_list:    list[str] = []
    hypotheses_list:  list[str] = []
    pair_meta:        list[tuple[str, int]] = []   # (qid, 0=hop1|1=hop2)

    skipped_no_hops    = 0
    skipped_no_pred    = 0
    skipped_no_gold    = 0
    n_nli_wrong        = 0
    n_nli_correct      = 0

    # Track per-qid metadata before scoring
    qid_meta: dict[str, dict] = {}

    all_qids = sorted(set(nli_map.keys()) & set(ev_map.keys()) & set(gold_map.keys()))
    log.info(f"Questions in all three sources: {len(all_qids)}")

    for qid in all_qids:
        pred_rec   = nli_map[qid]
        pred_text  = pred_rec.get("pred", "").strip()
        gold_text  = gold_map[qid]
        chains     = ev_map[qid]

        if not pred_text:
            skipped_no_pred += 1
            continue

        hop1_text, hop2_text = get_hop_texts(chains)
        if hop1_text is None or hop2_text is None:
            skipped_no_hops += 1
            continue

        is_wrong = not em(pred_text, gold_text)
        if is_wrong:
            n_nli_wrong += 1
        else:
            n_nli_correct += 1

        # Record metadata for this question (scores filled in after inference)
        qid_meta[qid] = {
            "qid":       qid,
            "pred":      pred_text,
            "gold":      gold_text,
            "is_wrong":  is_wrong,
            "flat_score": max(pred_rec.get("scores", [0.0])),  # best flat NLI score
            "nli_hop1":  None,
            "nli_hop2":  None,
        }

        # Enqueue scoring pairs: hop1 first, then hop2
        pair_meta.append((qid, 0))  # hop1
        premises_list.append(hop1_text)
        hypotheses_list.append(pred_text)

        pair_meta.append((qid, 1))  # hop2
        premises_list.append(hop2_text)
        hypotheses_list.append(pred_text)

    log.info(f"Questions to score : {len(qid_meta)} "
             f"(wrong={n_nli_wrong}, correct={n_nli_correct})")
    log.info(f"Skipped (no hops)  : {skipped_no_hops}")
    log.info(f"Skipped (no pred)  : {skipped_no_pred}")
    log.info(f"Total NLI pairs    : {len(pair_meta)}")

    # ── batch inference ──
    log.info(f"Running NLI inference in batches of {args.batch_size} ...")
    t0 = time.time()
    n_pairs = len(pair_meta)
    hop_scores: list[float] = []

    for i in range(0, n_pairs, args.batch_size):
        batch_p = premises_list[i: i + args.batch_size]
        batch_h = hypotheses_list[i: i + args.batch_size]
        scores  = score_batch(model, tokenizer, device,
                              batch_p, batch_h, entail_idx)
        hop_scores.extend(scores)

        if (i // args.batch_size + 1) % 100 == 0:
            done = min(i + args.batch_size, n_pairs)
            elapsed = time.time() - t0
            eta = elapsed / done * (n_pairs - done)
            log.info(f"  {done}/{n_pairs} pairs  |  ETA {eta/60:.1f} min")

    elapsed = time.time() - t0
    log.info(f"Inference done in {elapsed/60:.1f} min  "
             f"({n_pairs / elapsed:.0f} pairs/sec)")

    # ── distribute scores back to qid_meta ──
    assert len(hop_scores) == len(pair_meta), \
        f"Score count mismatch: {len(hop_scores)} vs {len(pair_meta)}"

    for (qid, hop_idx), score in zip(pair_meta, hop_scores):
        key = "nli_hop1" if hop_idx == 0 else "nli_hop2"
        qid_meta[qid][key] = round(float(score), 6)

    # ── compute per-question imbalance ──
    log.info("Computing imbalance statistics ...")
    imbalances_wrong:   list[float] = []
    imbalances_correct: list[float] = []
    hop2_dominated_wrong = 0    # hop2 score >> hop1 score (gap > threshold)
    hop1_dominated_wrong = 0    # hop1 score >> hop2 score (gap > threshold)

    # threshold sweep accumulators  {threshold: count_above}
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    sweep_counts: dict[float, int] = {t: 0 for t in thresholds}

    for qid, meta in qid_meta.items():
        s1 = meta["nli_hop1"]
        s2 = meta["nli_hop2"]
        if s1 is None or s2 is None:
            continue   # should not happen after the skip above, but be safe

        imbalance = abs(s1 - s2)
        meta["imbalance"] = round(imbalance, 6)
        meta["hop1_higher"] = s1 >= s2

        if meta["is_wrong"]:
            imbalances_wrong.append(imbalance)
            for t in thresholds:
                if imbalance > t:
                    sweep_counts[t] += 1
            if imbalance > args.threshold:
                if s2 > s1:
                    hop2_dominated_wrong += 1
                else:
                    hop1_dominated_wrong += 1
        else:
            imbalances_correct.append(imbalance)

    # ── build summary ──
    n_wrong_scored = len(imbalances_wrong)
    n_above_thresh = sweep_counts[args.threshold]
    pct_imbalanced = (n_above_thresh / n_wrong_scored * 100.0
                      if n_wrong_scored > 0 else 0.0)
    mean_imb_wrong   = (sum(imbalances_wrong) / len(imbalances_wrong)
                        if imbalances_wrong else 0.0)
    mean_imb_correct = (sum(imbalances_correct) / len(imbalances_correct)
                        if imbalances_correct else 0.0)

    threshold_sweep = {
        str(t): round(sweep_counts[t] / n_wrong_scored * 100.0, 2)
        for t in thresholds
    }

    summary = {
        "n_questions_total":        len(all_qids),
        "n_scored":                 len(qid_meta),
        "n_nli_wrong":              n_nli_wrong,
        "n_nli_correct":            n_nli_correct,
        "n_wrong_scored":           n_wrong_scored,
        "n_imbalanced_above_0.3":   n_above_thresh,
        "pct_imbalanced":           round(pct_imbalanced, 2),
        "hop2_dominated_count":     hop2_dominated_wrong,
        "hop1_dominated_count":     hop1_dominated_wrong,
        "mean_imbalance_when_wrong":    round(mean_imb_wrong, 4),
        "mean_imbalance_when_correct":  round(mean_imb_correct, 4),
        "threshold_sweep":          threshold_sweep,
        "decision": (
            "CONFIRMED: proceed to chain-aware verifier"
            if pct_imbalanced >= 25.0 else
            "BORDERLINE: discuss before proceeding"
            if pct_imbalanced >= 15.0 else
            "NOT CONFIRMED: central claim needs rethinking"
        ),
        "skipped_no_hops": skipped_no_hops,
        "skipped_no_pred": skipped_no_pred,
        "elapsed_sec":     round(elapsed, 1),
    }

    # ── save outputs ──
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to {args.out_json}")

    with open(args.out_jsonl, "w") as f:
        for qid in sorted(qid_meta.keys()):
            f.write(json.dumps(qid_meta[qid]) + "\n")
    log.info(f"Per-question JSONL saved to {args.out_jsonl}")

    # ── print summary ──
    W = 60
    log.info("=" * W)
    log.info(f"  Q0 RESULTS  (threshold = {args.threshold})")
    log.info("=" * W)
    log.info(f"  Questions scored    : {n_wrong_scored} wrong + "
             f"{len(imbalances_correct)} correct")
    log.info(f"  NLI wrong total     : {n_nli_wrong}")
    log.info(f"  Imbalanced (>{args.threshold})  : "
             f"{n_above_thresh}  ({pct_imbalanced:.1f}%)")
    log.info(f"  Hop-2 dominated     : {hop2_dominated_wrong}")
    log.info(f"  Hop-1 dominated     : {hop1_dominated_wrong}")
    log.info(f"  Mean imb (wrong)    : {mean_imb_wrong:.4f}")
    log.info(f"  Mean imb (correct)  : {mean_imb_correct:.4f}")
    log.info(f"  Threshold sweep (% wrong questions above threshold):")
    for t in thresholds:
        log.info(f"    >{t:.1f} : {threshold_sweep[str(t)]:.1f}%")
    log.info("-" * W)
    log.info(f"  DECISION: {summary['decision']}")
    log.info("=" * W)


if __name__ == "__main__":
    main()