#!/usr/bin/env python3
"""
exp2_q5a_score_chains.py — Q5 Phase 1 (inference only)

For every (question, candidate, chain_j), score hop-1 and hop-2 of
chain_j against the candidate answer using nli-roberta-base.

Saves dev_multichain_scores.jsonl — the reusable artifact that
exp2_q5b_sweep.py reads to run the XGBoost sweep for N in {1,4,8,16,32,100}
without any further NLI inference.

Schema per line:
{
  "qid": "...",
  "gold": "...",
  "candidates": [
    {
      "cand_idx": 0,
      "answer":   "...",
      "nli_flat": 0.82,
      "is_bad":   false,
      "chains": [
        {"chain_id": 0, "nli_hop1": 0.71, "nli_hop2": 0.65},
        ...
      ]
    },
    ...   (5 candidates)
  ]
}

Runtime estimate (A100, batch=64):
  max_chains=32  →  ~1.8M pairs  →  ~85 min
  max_chains=100 →  ~5.5M pairs  →  ~4.5 hr

Start with --max_chains 32. Run to 100 only if EM hasn't plateaued by 32.

Usage (from project root, llm conda env):
    /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 \\
        tools/exp2_q5a_score_chains.py \\
        --candidates   exp1b/candidates/dev_M5_candidates_qwen.jsonl \\
        --nli_preds    exp1b/preds/dev_nli_preds.jsonl \\
        --evidence     exp1b/evidence/dev_K100_chains.jsonl \\
        --model        /var/tmp/u24sf51014/sro/models/nli-roberta-base \\
        --out          exp1b/preds/dev_multichain_scores.jsonl \\
        --log          exp1b/logs/q5a_score_chains.log \\
        --max_chains   32 \\
        --batch_size   64
"""

import argparse
import json
import logging
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


# ─────────────────────────── NLI ────────────────────────────────────

def score_batch(model, tokenizer, device: str,
                premises: list, hypotheses: list,
                entail_idx: int) -> list:
    enc = tokenizer(
        premises, hypotheses,
        truncation=True, max_length=512, padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=-1)[:, entail_idx].tolist()


# ─────────────────────────── hop text ───────────────────────────────

def hop_text(hop: dict) -> str:
    return f"{hop.get('title', '')}: {hop.get('text', '')}".strip()


# ─────────────────────────── main ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",  required=True)
    ap.add_argument("--nli_preds",   required=True)
    ap.add_argument("--evidence",    required=True)
    ap.add_argument("--model",       required=True)
    ap.add_argument("--out",         required=True,
                    help="dev_multichain_scores.jsonl")
    ap.add_argument("--log",         required=True)
    ap.add_argument("--max_chains",  type=int, default=32,
                    help="Score top-K chains per question (default 32; "
                         "use 100 only if curve hasn't plateaued)")
    ap.add_argument("--batch_size",  type=int, default=64)
    ap.add_argument("--resume",      action="store_true",
                    help="Skip qids already written to --out")
    args = ap.parse_args()

    for p in [args.out, args.log]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # ── logging ──
    log = logging.getLogger("q5a")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    log.addHandler(logging.FileHandler(args.log, mode="a" if args.resume else "w"))
    log.addHandler(logging.StreamHandler(sys.stdout))
    for h in log.handlers:
        h.setFormatter(fmt)

    log.info("=== Q5a Multi-Chain NLI Scoring ===")
    log.info(f"max_chains : {args.max_chains}")
    log.info(f"batch_size : {args.batch_size}")
    log.info(f"out        : {args.out}")

    # ── resume: load already-done qids ──
    done_qids: set[str] = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done_qids.add(str(json.loads(line)["qid"]))
                except Exception:
                    pass
        log.info(f"Resume: {len(done_qids)} qids already done")

    # ── load model ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading NLI model on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_nli = AutoModelForSequenceClassification.from_pretrained(
        args.model).to(device)
    model_nli.eval()
    label2id   = {k.lower(): v for k, v in model_nli.config.label2id.items()}
    entail_idx = label2id.get("entailment", 2)
    log.info(f"Labels: {model_nli.config.id2label}  →  entailment_idx={entail_idx}")

    # ── load data ──
    # gold is read from the evidence pack (gold.answer field per record)
    gold_map = {}
    ev_map: dict[str, list] = {}
    log.info("Loading evidence pack ...")
    with open(args.evidence) as f:
        for line in f:
            r = json.loads(line)
            qid = str(r["qid"])
            gold_map[qid] = r["gold"]["answer"]
            chains = (r.get("evidence", {}).get("chains")
                      or r.get("chains", []))
            ev_map[qid] = chains[:args.max_chains]
    log.info(f"  {len(ev_map)} questions, capped at {args.max_chains} chains each")

    log.info("Loading flat NLI preds ...")
    flat_map: dict[str, list] = {}
    with open(args.nli_preds) as f:
        for line in f:
            r = json.loads(line)
            flat_map[str(r["qid"])] = r.get("scores", [])

    log.info("Loading candidates ...")
    cand_map: dict[str, list[str]] = {}
    with open(args.candidates) as f:
        for line in f:
            r = json.loads(line)
            qid = str(r["qid"])
            cands_sorted = sorted(r.get("candidates", []),
                                  key=lambda c: c.get("answer_id", 0))
            cand_map[qid] = [c.get("answer_text", "") for c in cands_sorted]

    all_qids = sorted(set(ev_map) & set(cand_map) & set(flat_map))
    remaining = [q for q in all_qids if q not in done_qids]
    log.info(f"Questions total={len(all_qids)}  remaining={len(remaining)}")

    # ─────────────────────────────────────────────────────────────────
    # Build the flat batch.
    # For each (qid, cand_idx, chain_id, hop_idx) we need one NLI pair.
    # bad candidates get score 0.0 without a forward pass.
    # ─────────────────────────────────────────────────────────────────
    # pair_meta: (qid, cand_idx, chain_id, hop_idx)
    pair_meta:    list[tuple[str, int, int, int]] = []
    premises_list:   list[str] = []
    hypotheses_list: list[str] = []

    # Per-qid scaffold: records[qid]["candidates"][ci]["chains"][j] = {hop1, hop2}
    records: dict[str, dict] = {}

    skipped_bad  = 0
    skipped_hops = 0

    for qid in remaining:
        answers = cand_map[qid]
        chains  = ev_map[qid]
        flat_scores = flat_map.get(qid, [0.0] * len(answers))

        cands_out = []
        for ci, ans in enumerate(answers):
            flat_s = flat_scores[ci] if ci < len(flat_scores) else 0.0
            bad    = is_bad(ans)

            chain_slots = []
            for j, chain in enumerate(chains):
                hops = chain.get("hops", [])
                if len(hops) < 2:
                    skipped_hops += 1
                    chain_slots.append({
                        "chain_id": chain.get("chain_id", j),
                        "nli_hop1": 0.0,
                        "nli_hop2": 0.0,
                    })
                    continue

                slot = {
                    "chain_id": chain.get("chain_id", j),
                    "nli_hop1": 0.0,
                    "nli_hop2": 0.0,
                }
                chain_slots.append(slot)

                if bad:
                    continue  # leave 0.0 — no forward pass

                h1t = hop_text(hops[0])
                h2t = hop_text(hops[1])

                pair_meta.append((qid, ci, j, 0))
                premises_list.append(h1t)
                hypotheses_list.append(ans)

                pair_meta.append((qid, ci, j, 1))
                premises_list.append(h2t)
                hypotheses_list.append(ans)

            if bad:
                skipped_bad += 1

            cands_out.append({
                "cand_idx": ci,
                "answer":   ans,
                "nli_flat": round(float(flat_s), 6),
                "is_bad":   bad,
                "chains":   chain_slots,
            })

        records[qid] = {
            "qid":        qid,
            "gold":       gold_map[qid],
            "candidates": cands_out,
        }

    total_pairs = len(pair_meta)
    log.info(f"Bad cand slots skipped : {skipped_bad}")
    log.info(f"Chains with <2 hops    : {skipped_hops}")
    log.info(f"Total NLI pairs        : {total_pairs}")

    # ── batch inference ──
    log.info(f"Running NLI inference in batches of {args.batch_size} ...")
    t0 = time.time()
    hop_scores_flat: list[float] = []

    for i in range(0, total_pairs, args.batch_size):
        batch_p = premises_list[i: i + args.batch_size]
        batch_h = hypotheses_list[i: i + args.batch_size]
        hop_scores_flat.extend(
            score_batch(model_nli, tokenizer, device,
                        batch_p, batch_h, entail_idx)
        )
        if (i // args.batch_size + 1) % 200 == 0:
            done = min(i + args.batch_size, total_pairs)
            eta  = (time.time() - t0) / done * (total_pairs - done)
            log.info(f"  {done}/{total_pairs} pairs  |  ETA {eta/60:.1f} min")

    elapsed = time.time() - t0
    log.info(f"Inference done in {elapsed/60:.1f} min  "
             f"({total_pairs / max(elapsed,1):.0f} pairs/sec)")

    # ── distribute scores back ──
    for (qid, ci, chain_j, hop_idx), score in zip(pair_meta, hop_scores_flat):
        key = "nli_hop1" if hop_idx == 0 else "nli_hop2"
        records[qid]["candidates"][ci]["chains"][chain_j][key] = round(
            float(score), 6)

    # ── write output (append mode for resume support) ──
    mode = "a" if args.resume else "w"
    with open(args.out, mode) as f:
        for qid in sorted(records):
            f.write(json.dumps(records[qid]) + "\n")
    log.info(f"Wrote {len(records)} records to {args.out}")
    log.info("Phase 1 complete — run exp2_q5b_sweep.py next")


if __name__ == "__main__":
    main()