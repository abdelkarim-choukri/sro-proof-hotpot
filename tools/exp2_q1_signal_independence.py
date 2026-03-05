#!/usr/bin/env python3
"""
exp2_q1_signal_independence.py — Q1: Does per-hop NLI carry different
                                  information than flat NLI?

For every (question, candidate) pair we compute three scores:
  nli_flat  — already computed in dev_nli_preds.jsonl (scores array)
  nli_hop1  — entailment of (hop1_text, candidate)
  nli_hop2  — entailment of (hop2_text, candidate)

Then for each candidate:
  min_hop  = min(nli_hop1, nli_hop2)   (chain coverage signal)
  mean_hop = mean(nli_hop1, nli_hop2)

We measure Pearson + Spearman correlation between min_hop and nli_flat
across all (question, candidate) pairs.

Decision rule:
  corr(min_hop, nli_flat) < 0.70  → signals are DIFFERENT, Q2–Q4 are worth building
  corr(min_hop, nli_flat) ≥ 0.70  → signals collapse; per-hop features may be redundant

Secondary outputs:
  • EM comparison: what EM do you get picking by nli_flat vs min_hop vs mean_hop?
  • dev_hop_scores.jsonl — REUSABLE artifact for Q2, Q3, Q4 (no need to re-run NLI)

Usage (from project root, llm conda env):
    /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/exp2_q1_signal_independence.py \\
        --candidates  exp1b/candidates/dev_M5_candidates_qwen.jsonl \\
        --nli_preds   exp1b/preds/dev_nli_preds.jsonl \\
        --evidence    exp1b/evidence/dev_K100_chains.jsonl \\
        --gold        data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
        --model       /var/tmp/u24sf51014/sro/models/nli-roberta-base \\
        --out_hop_scores  exp1b/preds/dev_hop_scores.jsonl \\
        --out_json        exp1b/metrics/q1_signal_independence.json \\
        --out_jsonl       exp1b/metrics/q1_signal_independence_perqid.jsonl \\
        --log             exp1b/logs/q1_signal_independence.log \\
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

import numpy as np
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

def is_bad(ans: str) -> bool:
    """Mirror of exp1_xgb_verifier.py — skip these for NLI."""
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


# ─────────────────────────── correlation ────────────────────────────

def pearson(x: list, y: list) -> float:
    a, b = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def spearman(x: list, y: list) -> float:
    from scipy.stats import spearmanr
    r, _ = spearmanr(x, y)
    return float(r)


# ─────────────────────────── hop text ───────────────────────────────

def get_hop_texts(chains: list):
    """Return (hop1_text, hop2_text) from the top chain, or (None, None)."""
    if not chains:
        return None, None
    hops = chains[0].get("hops", [])
    if len(hops) < 2:
        return None, None
    h1, h2 = hops[0], hops[1]
    t1 = f"{h1.get('title', '')}: {h1.get('text', '')}".strip()
    t2 = f"{h2.get('title', '')}: {h2.get('text', '')}".strip()
    return t1, t2


# ─────────────────────────── main ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",     required=True,
                    help="exp1b/candidates/dev_M5_candidates_qwen.jsonl")
    ap.add_argument("--nli_preds",      required=True,
                    help="exp1b/preds/dev_nli_preds.jsonl")
    ap.add_argument("--evidence",       required=True,
                    help="exp1b/evidence/dev_K100_chains.jsonl")
    ap.add_argument("--gold",           required=True,
                    help="data/hotpotqa/raw/hotpot_dev_distractor_v1.json")
    ap.add_argument("--model",          required=True,
                    help="Path to nli-roberta-base")
    ap.add_argument("--out_hop_scores", required=True,
                    help="Reusable per-(qid,cand) hop scores JSONL for Q2-Q4")
    ap.add_argument("--out_json",       required=True,
                    help="Q1 summary JSON")
    ap.add_argument("--out_jsonl",      required=True,
                    help="Q1 per-question JSONL")
    ap.add_argument("--log",            required=True)
    ap.add_argument("--batch_size",     type=int, default=64)
    args = ap.parse_args()

    # ── dirs ──
    for p in [args.out_hop_scores, args.out_json, args.out_jsonl, args.log]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # ── logging ──
    log = logging.getLogger("q1")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(args.log, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)

    log.info("=== Q1 Signal Independence Diagnostic ===")
    log.info(f"candidates    : {args.candidates}")
    log.info(f"nli_preds     : {args.nli_preds}")
    log.info(f"evidence      : {args.evidence}")
    log.info(f"model         : {args.model}")
    log.info(f"out_hop_scores: {args.out_hop_scores}")
    log.info(f"batch_size    : {args.batch_size}")

    # ── load model ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading NLI model on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()
    label2id = {k.lower(): v for k, v in model.config.label2id.items()}
    entail_idx = label2id.get("entailment", 2)
    log.info(f"Labels: {model.config.id2label}  →  entailment_idx={entail_idx}")

    # ── load gold ──
    log.info("Loading gold answers ...")
    gold_map: dict[str, str] = {
        str(ex["_id"]): ex["answer"]
        for ex in json.load(open(args.gold))
    }
    log.info(f"  {len(gold_map)} gold answers")

    # ── load evidence ──
    log.info("Loading evidence pack ...")
    ev_map: dict[str, list] = {}
    for line in open(args.evidence):
        r = json.loads(line)
        qid = str(r["qid"])
        ev_map[qid] = (r.get("evidence", {}).get("chains")
                       or r.get("chains", []))
    log.info(f"  {len(ev_map)} evidence records")

    # ── load flat NLI preds ──
    log.info("Loading flat NLI predictions ...")
    # schema: {"qid": "...", "pred": "...", "scores": [5 floats], "best_idx": int}
    flat_map: dict[str, dict] = {}
    for line in open(args.nli_preds):
        r = json.loads(line)
        flat_map[str(r["qid"])] = r
    log.info(f"  {len(flat_map)} NLI pred records")

    # ── load candidates ──
    log.info("Loading candidates ...")
    cand_map: dict[str, list[str]] = {}
    for line in open(args.candidates):
        r = json.loads(line)
        qid = str(r["qid"])
        # candidates is a list of dicts with answer_text
        cands = r.get("candidates", [])
        cands_sorted = sorted(cands, key=lambda c: c.get("answer_id", 0))
        cand_map[qid] = [c.get("answer_text", "") for c in cands_sorted]
    log.info(f"  {len(cand_map)} candidate records")

    # ─────────────────────────────────────────────────────────────────
    # Build scoring job: (hop1_text, candidate) and (hop2_text, candidate)
    # for all 5 candidates × all questions
    # ─────────────────────────────────────────────────────────────────
    all_qids = sorted(
        set(cand_map) & set(flat_map) & set(ev_map) & set(gold_map)
    )
    log.info(f"Questions in all four sources: {len(all_qids)}")

    # pair_meta: (qid, cand_idx, hop_idx)   hop_idx 0=hop1 1=hop2
    pair_meta:    list[tuple[str, int, int]] = []
    premises_list:   list[str] = []
    hypotheses_list: list[str] = []

    skipped_no_hops = 0
    skipped_bad_cand = 0

    # qid → per-candidate data (filled in after inference)
    # Structure: {qid: {"gold": str, "candidates": [{answer, nli_flat, nli_hop1, nli_hop2}]}}
    qid_data: dict[str, dict] = {}

    for qid in all_qids:
        answers = cand_map[qid]       # list of str, up to 5
        flat_rec = flat_map[qid]
        flat_scores = flat_rec.get("scores", [0.0] * len(answers))
        chains = ev_map[qid]
        gold = gold_map[qid]

        hop1_text, hop2_text = get_hop_texts(chains)
        if hop1_text is None or hop2_text is None:
            skipped_no_hops += 1
            continue

        qid_data[qid] = {
            "qid":  qid,
            "gold": gold,
            "candidates": []
        }

        for ci, ans in enumerate(answers):
            flat_score = flat_scores[ci] if ci < len(flat_scores) else 0.0

            # Record placeholder — nli_hop1/nli_hop2 filled after inference
            qid_data[qid]["candidates"].append({
                "cand_idx": ci,
                "answer":   ans,
                "nli_flat": round(float(flat_score), 6),
                "nli_hop1": None,
                "nli_hop2": None,
            })

            if is_bad(ans):
                # Assign 0.0 without a forward pass (same as original baseline)
                skipped_bad_cand += 1
                qid_data[qid]["candidates"][ci]["nli_hop1"] = 0.0
                qid_data[qid]["candidates"][ci]["nli_hop2"] = 0.0
                continue

            # Enqueue hop1 then hop2
            pair_meta.append((qid, ci, 0))
            premises_list.append(hop1_text)
            hypotheses_list.append(ans)

            pair_meta.append((qid, ci, 1))
            premises_list.append(hop2_text)
            hypotheses_list.append(ans)

    log.info(f"Questions queued    : {len(qid_data)}")
    log.info(f"Skipped (no hops)   : {skipped_no_hops}")
    log.info(f"Bad cands (scored 0): {skipped_bad_cand}")
    log.info(f"Total NLI pairs     : {len(pair_meta)}")

    # ── batch inference ──
    log.info(f"Running NLI inference in batches of {args.batch_size} ...")
    t0 = time.time()
    n_pairs = len(pair_meta)
    hop_scores_flat: list[float] = []

    for i in range(0, n_pairs, args.batch_size):
        batch_p = premises_list[i: i + args.batch_size]
        batch_h = hypotheses_list[i: i + args.batch_size]
        scores  = score_batch(model, tokenizer, device,
                              batch_p, batch_h, entail_idx)
        hop_scores_flat.extend(scores)

        if (i // args.batch_size + 1) % 100 == 0:
            done = min(i + args.batch_size, n_pairs)
            eta = (time.time() - t0) / done * (n_pairs - done)
            log.info(f"  {done}/{n_pairs} pairs  |  ETA {eta/60:.1f} min")

    elapsed = time.time() - t0
    log.info(f"Inference done in {elapsed/60:.1f} min  "
             f"({n_pairs / elapsed:.0f} pairs/sec)")

    # ── distribute hop scores back ──
    for (qid, ci, hop_idx), score in zip(pair_meta, hop_scores_flat):
        key = "nli_hop1" if hop_idx == 0 else "nli_hop2"
        qid_data[qid]["candidates"][ci][key] = round(float(score), 6)

    # ── compute derived scores + selection ──
    log.info("Computing derived signals and EM ...")

    # Accumulators for correlation
    all_nli_flat:  list[float] = []
    all_min_hop:   list[float] = []
    all_mean_hop:  list[float] = []

    # EM accumulators
    em_flat = em_min = em_mean = em_oracle = 0
    n_q = 0

    for qid, data in qid_data.items():
        gold = data["gold"]
        cands_data = data["candidates"]

        flat_scores_q  = []
        min_hop_scores = []
        mean_hop_scores = []

        for cd in cands_data:
            s1 = cd["nli_hop1"] if cd["nli_hop1"] is not None else 0.0
            s2 = cd["nli_hop2"] if cd["nli_hop2"] is not None else 0.0
            min_h  = min(s1, s2)
            mean_h = (s1 + s2) / 2.0
            imb    = abs(s1 - s2)

            cd["min_hop"]   = round(min_h,  6)
            cd["mean_hop"]  = round(mean_h, 6)
            cd["imbalance"] = round(imb,    6)

            flat_scores_q.append(cd["nli_flat"])
            min_hop_scores.append(min_h)
            mean_hop_scores.append(mean_h)

            all_nli_flat.append(cd["nli_flat"])
            all_min_hop.append(min_h)
            all_mean_hop.append(mean_h)

        # Pick best candidate by each selector
        def pick(scores):
            return max(range(len(scores)), key=lambda i: scores[i])

        flat_best  = pick(flat_scores_q)
        min_best   = pick(min_hop_scores)
        mean_best  = pick(mean_hop_scores)
        oracle_hit = any(em(cd["answer"], gold) for cd in cands_data)

        data["flat_best_idx"]  = flat_best
        data["min_best_idx"]   = min_best
        data["mean_best_idx"]  = mean_best
        data["flat_em"]        = int(em(cands_data[flat_best]["answer"],  gold))
        data["min_em"]         = int(em(cands_data[min_best]["answer"],   gold))
        data["mean_em"]        = int(em(cands_data[mean_best]["answer"],  gold))
        data["oracle_em"]      = int(oracle_hit)

        em_flat   += data["flat_em"]
        em_min    += data["min_em"]
        em_mean   += data["mean_em"]
        em_oracle += data["oracle_em"]
        n_q += 1

    # ── correlations ──
    log.info("Computing correlations ...")
    try:
        pear_min  = pearson(all_min_hop,  all_nli_flat)
        pear_mean = pearson(all_mean_hop, all_nli_flat)
        spear_min  = spearman(all_min_hop,  all_nli_flat)
        spear_mean = spearman(all_mean_hop, all_nli_flat)
    except ImportError:
        log.warning("scipy not found — Spearman will be NaN")
        pear_min  = pearson(all_min_hop,  all_nli_flat)
        pear_mean = pearson(all_mean_hop, all_nli_flat)
        spear_min = spear_mean = float("nan")

    # Decision based on Pearson(min_hop, flat) < 0.70
    threshold = 0.70
    if pear_min < threshold:
        decision = "DIFFERENT SIGNAL: per-hop features carry new information — proceed to Q2-Q4"
    else:
        decision = "SAME SIGNAL: min_hop collapses to flat NLI — per-hop features may be redundant"

    # ── save hop scores (reusable for Q2-Q4) ──
    log.info(f"Saving hop scores to {args.out_hop_scores} ...")
    with open(args.out_hop_scores, "w") as f:
        for qid in sorted(qid_data):
            data = qid_data[qid]
            rec = {
                "qid":        qid,
                "gold":       data["gold"],
                "flat_em":    data["flat_em"],
                "min_em":     data["min_em"],
                "mean_em":    data["mean_em"],
                "oracle_em":  data["oracle_em"],
                "candidates": data["candidates"],
            }
            f.write(json.dumps(rec) + "\n")
    log.info(f"  {len(qid_data)} records written")

    # ── build summary ──
    summary = {
        "n_questions":      n_q,
        "n_pairs":          len(all_nli_flat),
        # Correlation between per-hop signals and flat NLI
        "pearson_minhop_vs_flat":   round(pear_min,  4),
        "pearson_meanhop_vs_flat":  round(pear_mean, 4),
        "spearman_minhop_vs_flat":  round(spear_min,  4),
        "spearman_meanhop_vs_flat": round(spear_mean, 4),
        # EM under each selector (on full dev set, picking from 5 candidates)
        "em_comparison": {
            "flat_nli_em":  round(em_flat   / n_q, 4),
            "min_hop_em":   round(em_min    / n_q, 4),
            "mean_hop_em":  round(em_mean   / n_q, 4),
            "oracle_em":    round(em_oracle / n_q, 4),
        },
        "correlation_threshold": threshold,
        "decision": decision,
        # Descriptive stats on the distributions
        "signal_stats": {
            "nli_flat_mean":  round(float(np.mean(all_nli_flat)),  4),
            "nli_flat_std":   round(float(np.std(all_nli_flat)),   4),
            "min_hop_mean":   round(float(np.mean(all_min_hop)),   4),
            "min_hop_std":    round(float(np.std(all_min_hop)),    4),
            "mean_hop_mean":  round(float(np.mean(all_mean_hop)),  4),
            "mean_hop_std":   round(float(np.std(all_mean_hop)),   4),
        },
        "elapsed_sec": round(elapsed, 1),
        "hop_scores_path": args.out_hop_scores,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to {args.out_json}")

    # ── save per-question JSONL ──
    with open(args.out_jsonl, "w") as f:
        for qid in sorted(qid_data):
            f.write(json.dumps(qid_data[qid]) + "\n")
    log.info(f"Per-question JSONL saved to {args.out_jsonl}")

    # ── print summary ──
    W = 64
    log.info("=" * W)
    log.info("  Q1 RESULTS")
    log.info("=" * W)
    log.info(f"  Questions / candidate pairs : {n_q} / {len(all_nli_flat)}")
    log.info(f"  Pearson  (min_hop  vs flat) : {pear_min:.4f}")
    log.info(f"  Pearson  (mean_hop vs flat) : {pear_mean:.4f}")
    log.info(f"  Spearman (min_hop  vs flat) : {spear_min:.4f}")
    log.info(f"  Spearman (mean_hop vs flat) : {spear_mean:.4f}")
    log.info("-" * W)
    log.info("  EM comparison (picking best candidate by each signal):")
    log.info(f"    flat NLI selector  : {em_flat  / n_q:.4f}")
    log.info(f"    min_hop selector   : {em_min   / n_q:.4f}")
    log.info(f"    mean_hop selector  : {em_mean  / n_q:.4f}")
    log.info(f"    oracle@5           : {em_oracle / n_q:.4f}")
    log.info("-" * W)
    log.info(f"  DECISION: {decision}")
    log.info("=" * W)
    log.info(f"  Hop scores saved to: {args.out_hop_scores}")
    log.info(f"  (This file is the input for Q2, Q3, Q4 — no re-inference needed)")


if __name__ == "__main__":
    main()