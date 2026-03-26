#!/usr/bin/env python3
"""
exp3b_7b_sample_select.py — Select 300 Bucket B questions for 7B generator test

Bucket B = feasible (retrieval succeeded) but oracle_em=0 (1.5B never produced
the correct answer in M=10 tries). These are the questions where a better
generator could make a difference.

Outputs:
  - sample_qids.json: list of 300 qids
  - sample_evidence.jsonl: evidence records for just these 300 questions
    (same format as dev_K100_chains.jsonl, so the generator script reads it directly)

Usage:
    python3 tools/exp3b_7b_sample_select.py \
        --oracle_perqid   exp3b/metrics/oracle_M10_dev_perqid.jsonl \
        --evidence        exp1b/evidence/dev_K100_chains.jsonl \
        --gold            data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
        --out_qids        exp3b_7b/sample_qids.json \
        --out_evidence    exp3b_7b/evidence/sample_300_chains.jsonl \
        --n_sample        300 \
        --seed            42
"""

import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle_perqid", required=True)
    ap.add_argument("--evidence",      required=True)
    ap.add_argument("--gold",          required=True)
    ap.add_argument("--out_qids",      required=True)
    ap.add_argument("--out_evidence",  required=True)
    ap.add_argument("--n_sample",      type=int, default=300)
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # ── load feasible flags ──
    feasible = set()
    ev_by_qid = {}
    for line in open(args.evidence):
        r = json.loads(line)
        qid = str(r["qid"])
        ev_by_qid[qid] = line.strip()
        if r.get("flags", {}).get("doc_recall_at_k", False):
            feasible.add(qid)
    print(f"[select] Feasible: {len(feasible)}")

    # ── load oracle flags (M=10) ──
    oracle_map = {}
    for line in open(args.oracle_perqid):
        r = json.loads(line)
        oracle_map[str(r["qid"])] = int(r["best_em"])

    # ── load question types ──
    qtype_map = {}
    for ex in json.load(open(args.gold)):
        qtype_map[str(ex["_id"])] = ex.get("type", "bridge")

    # ── Bucket B: feasible AND oracle_em=0 ──
    bucket_b = [qid for qid in feasible
                if oracle_map.get(qid, 0) == 0]
    print(f"[select] Bucket B (feasible, oracle=0): {len(bucket_b)}")

    # ── stratified sample: preserve bridge/comparison ratio ──
    bridge = [q for q in bucket_b if qtype_map.get(q) == "bridge"]
    comp   = [q for q in bucket_b if qtype_map.get(q) == "comparison"]
    print(f"[select] Bucket B breakdown: bridge={len(bridge)}  comparison={len(comp)}")

    random.shuffle(bridge)
    random.shuffle(comp)

    # Proportional sampling
    bridge_ratio = len(bridge) / len(bucket_b)
    n_bridge = int(args.n_sample * bridge_ratio)
    n_comp   = args.n_sample - n_bridge

    sample = bridge[:n_bridge] + comp[:n_comp]
    random.shuffle(sample)
    print(f"[select] Sample: {len(sample)}  "
          f"(bridge={n_bridge}  comparison={n_comp})")

    # ── write outputs ──
    os.makedirs(os.path.dirname(args.out_qids), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_evidence), exist_ok=True)

    json.dump(sample, open(args.out_qids, "w"), indent=2)
    print(f"[select] QIDs saved to {args.out_qids}")

    sample_set = set(sample)
    n_written = 0
    with open(args.out_evidence, "w") as f:
        for line in open(args.evidence):
            r = json.loads(line)
            if str(r["qid"]) in sample_set:
                f.write(line.strip() + "\n")
                n_written += 1
    print(f"[select] Evidence written: {n_written} to {args.out_evidence}")


if __name__ == "__main__":
    main()