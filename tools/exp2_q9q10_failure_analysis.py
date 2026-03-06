#!/usr/bin/env python3
"""
exp2_q9q10_failure_analysis.py — Q9 + Q10: Failure taxonomy and differential

Q9: Systematic failure patterns in the chain-aware verifier.
    Every question gets bucketed into exactly one failure category:

    BUCKET A — Retrieval failure (feasible=0)
               Gold evidence was never retrieved. Verifier can't fix this.

    BUCKET B — Oracle failure (feasible=1, oracle_em=0)
               Evidence was retrieved but the generator produced 5 wrong
               candidates. Generator is the binding constraint.

    BUCKET C — Verifier failure (feasible=1, oracle_em=1, verifier wrong)
               Correct candidate existed but verifier didn't pick it.
               Sub-buckets:
                 C1 — verifier picked a BAD answer (is_bad=True)
                 C2 — verifier picked wrong-but-plausible answer
                       (is_bad=False, but answer is wrong)

    BUCKET D — Success (verifier correct)

Q10: Chain-aware vs flat NLI differential.
    Four outcome groups per question:
      BOTH_CORRECT   — neither method needs improving here
      CHAIN_WINS     — chain-aware correct, flat NLI wrong
      FLAT_WINS      — flat NLI correct, chain-aware wrong
      BOTH_WRONG     — neither method works here

Inputs (no inference):
  dev_chain_verifier_min_preds.jsonl  — chain-aware verifier (Q2)
  dev_nli_preds.jsonl                 — flat NLI baseline
  oracle_M5_dev_perqid.jsonl          — oracle flags per question
  dev_K100_chains.jsonl               — feasible flags
  hotpot_dev_distractor_v1.json       — gold answers + question type

Outputs:
  exp1b/metrics/q9q10_failure_analysis.json   (git-tracked)
  exp1b/logs/q9q10_failure_analysis.log

Usage (from project root, no GPU):
    /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 \\
        tools/exp2_q9q10_failure_analysis.py \\
        --chain_preds     exp1b/preds/dev_chain_verifier_min_preds.jsonl \\
        --nli_preds       exp1b/preds/dev_nli_preds.jsonl \\
        --oracle_perqid   exp1b/metrics/oracle_M5_dev_perqid.jsonl \\
        --evidence        exp1b/evidence/dev_K100_chains.jsonl \\
        --gold            data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
        --out_json        exp1b/metrics/q9q10_failure_analysis.json \\
        --log             exp1b/logs/q9q10_failure_analysis.log
"""

import argparse
import collections
import json
import logging
import os
import re
import string
import sys

import numpy as np


# ─────────────────────────── text utils ─────────────────────────────

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

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


# ─────────────────────────── descriptive stats ───────────────────────

def group_stats(values: list) -> dict:
    if not values:
        return {"n": 0}
    a = np.array(values, dtype=float)
    return {
        "n":      len(a),
        "mean":   round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "std":    round(float(a.std()), 4),
    }

def counter_top(counter: dict, k: int = 8) -> list:
    total = sum(counter.values())
    return [
        {"value": v, "count": c, "pct": round(c / total, 4)}
        for v, c in sorted(counter.items(), key=lambda x: -x[1])[:k]
    ]


# ─────────────────────────── main ────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain_preds",   required=True)
    ap.add_argument("--nli_preds",     required=True)
    ap.add_argument("--oracle_perqid", required=True)
    ap.add_argument("--evidence",      required=True)
    ap.add_argument("--gold",          required=True)
    ap.add_argument("--out_json",      required=True)
    ap.add_argument("--log",           required=True)
    args = ap.parse_args()

    for p in [args.out_json, args.log]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    log = logging.getLogger("q9q10")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(args.log, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)

    log.info("=== Q9/Q10 Failure Analysis ===")

    # ── load gold ──
    log.info("Loading gold answers + question types ...")
    gold_map:  dict[str, str] = {}
    qtype_map: dict[str, str] = {}
    for ex in json.load(open(args.gold)):
        qid = str(ex["_id"])
        gold_map[qid]  = ex["answer"]
        qtype_map[qid] = ex.get("type", "bridge")
    log.info(f"  {len(gold_map)} questions")

    # ── load feasible flags ──
    log.info("Loading feasible flags ...")
    feasible: set[str] = set()
    for line in open(args.evidence):
        r = json.loads(line)
        if r.get("flags", {}).get("doc_recall_at_k", False):
            feasible.add(str(r["qid"]))
    log.info(f"  feasible: {len(feasible)}")

    # ── load oracle flags + per-candidate info ──
    log.info("Loading oracle flags ...")
    oracle_map: dict[str, dict] = {}
    for line in open(args.oracle_perqid):
        r = json.loads(line)
        oracle_map[str(r["qid"])] = r
    log.info(f"  {len(oracle_map)} oracle records")

    # ── load chain-aware verifier preds ──
    log.info("Loading chain-aware verifier preds ...")
    chain_map: dict[str, dict] = {}
    for line in open(args.chain_preds):
        r = json.loads(line)
        chain_map[str(r["qid"])] = r
    log.info(f"  {len(chain_map)} records")

    # ── load flat NLI preds ──
    log.info("Loading flat NLI preds ...")
    nli_map: dict[str, dict] = {}
    for line in open(args.nli_preds):
        r = json.loads(line)
        nli_map[str(r["qid"])] = r
    log.info(f"  {len(nli_map)} records")

    # ── align ──
    all_qids = sorted(
        set(chain_map) & set(nli_map) & set(oracle_map) & set(gold_map)
    )
    log.info(f"  Aligned: {len(all_qids)}")

    # ─────────────────────────────────────────────────────────────────
    # Q9 — Failure taxonomy
    # ─────────────────────────────────────────────────────────────────

    # Bucket counters
    buckets = {"A": [], "B": [], "C1": [], "C2": [], "D": []}

    # Per-bucket characterisation collectors
    bucket_qtype:   dict[str, list] = {k: [] for k in buckets}
    bucket_conf:    dict[str, list] = {k: [] for k in buckets}
    bucket_gold_len: dict[str, list] = {k: [] for k in buckets}
    # For C2 only: what did the verifier pick instead?
    c2_picked_answers: list[str] = []
    c2_gold_answers:   list[str] = []

    for qid in all_qids:
        gold  = gold_map[qid]
        qtype = qtype_map.get(qid, "bridge")
        chain_rec = chain_map[qid]
        pred      = chain_rec.get("pred", "")
        probs     = chain_rec.get("probs", [])
        max_conf  = max(probs) if probs else 0.0
        oracle    = oracle_map[qid]
        oracle_em_flag = int(oracle.get("best_em", 0))
        is_feasible    = qid in feasible
        verifier_correct = em(pred, gold)
        gold_len = len(gold.split())

        if not is_feasible:
            bucket = "A"
        elif oracle_em_flag == 0:
            bucket = "B"
        elif verifier_correct:
            bucket = "D"
        elif is_bad(pred):
            bucket = "C1"
        else:
            bucket = "C2"
            c2_picked_answers.append(pred)
            c2_gold_answers.append(gold)

        buckets[bucket].append(qid)
        bucket_qtype[bucket].append(qtype)
        bucket_conf[bucket].append(max_conf)
        bucket_gold_len[bucket].append(gold_len)

    n_total = len(all_qids)

    def bucket_summary(key: str) -> dict:
        qids  = buckets[key]
        n     = len(qids)
        qt    = collections.Counter(bucket_qtype[key])
        return {
            "n":          n,
            "pct":        round(n / n_total, 4),
            "qtype_dist": dict(qt),
            "bridge_pct": round(qt.get("bridge", 0) / max(n, 1), 4),
            "comp_pct":   round(qt.get("comparison", 0) / max(n, 1), 4),
            "confidence": group_stats(bucket_conf[key]),
            "gold_len_words": group_stats(bucket_gold_len[key]),
        }

    # For C2: what are the most common wrong answers picked?
    c2_wrong_counter = collections.Counter(
        normalize(a) for a in c2_picked_answers
    )
    # Gold length distribution for C2 vs D
    c2_gold_len_dist = group_stats(
        [len(g.split()) for g in c2_gold_answers]
    )

    q9_result = {
        "A_retrieval_failure": bucket_summary("A"),
        "B_oracle_failure":    bucket_summary("B"),
        "C1_verifier_bad":     bucket_summary("C1"),
        "C2_verifier_plausible_wrong": {
            **bucket_summary("C2"),
            "most_common_wrong_picks": counter_top(c2_wrong_counter, 8),
            "gold_len_for_c2": c2_gold_len_dist,
        },
        "D_success":           bucket_summary("D"),
        "total":               n_total,
        "interpretation": {
            "A_pct_label":  "retrieval failure — unfixable by verifier",
            "B_pct_label":  "generator failure — correct answer never generated",
            "C_pct_label":  "verifier failure — correct answer existed, verifier missed it",
            "D_pct_label":  "success",
        },
    }

    # ─────────────────────────────────────────────────────────────────
    # Q10 — Chain-aware vs flat NLI differential
    # ─────────────────────────────────────────────────────────────────

    # outcome groups
    groups = {
        "BOTH_CORRECT": [],
        "CHAIN_WINS":   [],
        "FLAT_WINS":    [],
        "BOTH_WRONG":   [],
    }

    group_qtype:   dict[str, list] = {k: [] for k in groups}
    group_conf_chain: dict[str, list] = {k: [] for k in groups}
    group_conf_nli:   dict[str, list] = {k: [] for k in groups}
    group_gold_len:   dict[str, list] = {k: [] for k in groups}

    # For CHAIN_WINS: what did flat NLI pick instead?
    chain_wins_nli_picked:   list[str] = []
    chain_wins_chain_picked: list[str] = []
    chain_wins_gold:         list[str] = []

    # For FLAT_WINS: what did chain-aware pick instead?
    flat_wins_chain_picked: list[str] = []
    flat_wins_nli_picked:   list[str] = []
    flat_wins_gold:         list[str] = []

    for qid in all_qids:
        gold  = gold_map[qid]
        qtype = qtype_map.get(qid, "bridge")

        chain_rec = chain_map[qid]
        chain_pred  = chain_rec.get("pred", "")
        chain_probs = chain_rec.get("probs", [])
        chain_conf  = max(chain_probs) if chain_probs else 0.0
        chain_ok    = em(chain_pred, gold)

        nli_rec   = nli_map[qid]
        # NLI preds schema: scores list + pred field
        nli_pred  = nli_rec.get("pred", "")
        nli_scores = nli_rec.get("scores", [])
        nli_conf  = max(nli_scores) if nli_scores else 0.0
        # If pred not stored, reconstruct from scores
        if not nli_pred and nli_scores:
            best_ci = int(np.argmax(nli_scores))
            # Can't reconstruct answer text without candidate list here
            # Mark as unknown — will lower FLAT_WINS count slightly
            nli_pred = ""
        nli_ok = em(nli_pred, gold)

        if chain_ok and nli_ok:
            grp = "BOTH_CORRECT"
        elif chain_ok and not nli_ok:
            grp = "CHAIN_WINS"
            chain_wins_chain_picked.append(chain_pred)
            chain_wins_nli_picked.append(nli_pred)
            chain_wins_gold.append(gold)
        elif not chain_ok and nli_ok:
            grp = "FLAT_WINS"
            flat_wins_chain_picked.append(chain_pred)
            flat_wins_nli_picked.append(nli_pred)
            flat_wins_gold.append(gold)
        else:
            grp = "BOTH_WRONG"

        groups[grp].append(qid)
        group_qtype[grp].append(qtype)
        group_conf_chain[grp].append(chain_conf)
        group_conf_nli[grp].append(nli_conf)
        group_gold_len[grp].append(len(gold.split()))

    def group_summary(key: str) -> dict:
        n  = len(groups[key])
        qt = collections.Counter(group_qtype[key])
        return {
            "n":          n,
            "pct":        round(n / n_total, 4),
            "qtype_dist": dict(qt),
            "bridge_pct": round(qt.get("bridge", 0) / max(n, 1), 4),
            "comp_pct":   round(qt.get("comparison", 0) / max(n, 1), 4),
            "chain_conf": group_stats(group_conf_chain[key]),
            "nli_conf":   group_stats(group_conf_nli[key]),
            "gold_len_words": group_stats(group_gold_len[key]),
        }

    # characterise CHAIN_WINS: were the NLI picks bad/garbage?
    cw_nli_bad_rate = (
        sum(1 for a in chain_wins_nli_picked if is_bad(a))
        / max(len(chain_wins_nli_picked), 1)
    )
    cw_nli_wrong_counter = collections.Counter(
        normalize(a) for a in chain_wins_nli_picked if not is_bad(a)
    )
    # characterise FLAT_WINS: were the chain picks bad/garbage?
    fw_chain_bad_rate = (
        sum(1 for a in flat_wins_chain_picked if is_bad(a))
        / max(len(flat_wins_chain_picked), 1)
    )
    fw_chain_wrong_counter = collections.Counter(
        normalize(a) for a in flat_wins_chain_picked if not is_bad(a)
    )

    q10_result = {
        "BOTH_CORRECT": group_summary("BOTH_CORRECT"),
        "CHAIN_WINS":   {
            **group_summary("CHAIN_WINS"),
            "nli_bad_pick_rate":    round(cw_nli_bad_rate, 4),
            "nli_common_wrong_picks": counter_top(cw_nli_wrong_counter, 5),
        },
        "FLAT_WINS": {
            **group_summary("FLAT_WINS"),
            "chain_bad_pick_rate":    round(fw_chain_bad_rate, 4),
            "chain_common_wrong_picks": counter_top(fw_chain_wrong_counter, 5),
        },
        "BOTH_WRONG":   group_summary("BOTH_WRONG"),
        "total":        n_total,
        "net_gain_chain_over_flat": len(groups["CHAIN_WINS"]) - len(groups["FLAT_WINS"]),
    }

    # ─────────────────────────────────────────────────────────────────
    # Assemble and write output
    # ─────────────────────────────────────────────────────────────────

    summary = {"q9_failure_taxonomy": q9_result, "q10_differential": q10_result}

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to {args.out_json}")

    # ── print Q9 ──
    W = 70
    log.info("=" * W)
    log.info("  Q9 — Failure Taxonomy")
    log.info("=" * W)
    log.info(f"  Total questions: {n_total}")
    log.info(f"  {'Bucket':<40}  {'N':>6}  {'%':>6}  {'Bridge%':>8}  {'Comp%':>7}")
    log.info("  " + "-" * (W - 2))

    def fmt_bucket(label, key):
        s = bucket_summary(key)
        log.info(f"  {label:<40}  {s['n']:>6}  "
                 f"{s['pct']:>6.1%}  {s['bridge_pct']:>8.1%}  "
                 f"{s['comp_pct']:>7.1%}")

    fmt_bucket("A — Retrieval failure (feasible=0)", "A")
    fmt_bucket("B — Oracle failure (no correct cand)", "B")
    fmt_bucket("C1 — Verifier: picked bad answer", "C1")
    fmt_bucket("C2 — Verifier: picked plausible wrong", "C2")
    fmt_bucket("D — Success", "D")

    log.info("")
    log.info("  C2 most common wrong picks (plausible failures):")
    for row in counter_top(c2_wrong_counter, 5):
        log.info(f"    '{row['value']}'  n={row['count']}  ({row['pct']:.1%})")

    # ── print Q10 ──
    log.info("=" * W)
    log.info("  Q10 — Chain-Aware vs Flat NLI Differential")
    log.info("=" * W)
    log.info(f"  {'Group':<30}  {'N':>6}  {'%':>6}  {'Bridge%':>8}  {'Comp%':>7}")
    log.info("  " + "-" * (W - 2))

    for grp in ["BOTH_CORRECT", "CHAIN_WINS", "FLAT_WINS", "BOTH_WRONG"]:
        s = group_summary(grp)
        log.info(f"  {grp:<30}  {s['n']:>6}  "
                 f"{s['pct']:>6.1%}  {s['bridge_pct']:>8.1%}  "
                 f"{s['comp_pct']:>7.1%}")

    log.info("")
    net = q10_result["net_gain_chain_over_flat"]
    log.info(f"  Net gain (CHAIN_WINS - FLAT_WINS): {net:+d} questions")
    log.info(f"  CHAIN_WINS: NLI bad-pick rate = "
             f"{cw_nli_bad_rate:.1%}")
    log.info(f"  FLAT_WINS:  Chain bad-pick rate = "
             f"{fw_chain_bad_rate:.1%}")

    log.info("=" * W)
    log.info("  Confidence profile per group:")
    log.info(f"  {'Group':<20}  {'Chain conf':>12}  {'NLI conf':>10}")
    log.info("  " + "-" * 46)
    for grp in ["BOTH_CORRECT", "CHAIN_WINS", "FLAT_WINS", "BOTH_WRONG"]:
        cc = group_stats(group_conf_chain[grp])
        nc = group_stats(group_conf_nli[grp])
        log.info(f"  {grp:<20}  "
                 f"{cc.get('mean', 0):.4f} ± {cc.get('std', 0):.3f}  "
                 f"{nc.get('mean', 0):.4f} ± {nc.get('std', 0):.3f}")
    log.info("=" * W)


if __name__ == "__main__":
    main()