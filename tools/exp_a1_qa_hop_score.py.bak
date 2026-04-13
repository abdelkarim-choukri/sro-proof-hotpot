#!/usr/bin/env python3
"""
exp_a1_qa_hop_score.py  —  Phase A1.1: QA Cross-Encoder Hop Scoring

For each question, scores every candidate answer against three contexts
using deepset/deberta-v3-base-squad2 (SQuAD 2.0 extractive QA):
  - hop1 text only         → qa_hop1
  - hop2 text only         → qa_hop2
  - hop1 + hop2 concat     → qa_flat

Derived features (computed in-script, stored in output):
  - qa_hop_balance  = |qa_hop1 − qa_hop2|
  - qa_min_hop      = min(qa_hop1, qa_hop2)

Verified input schemas:
  --evidence    exp0c/evidence/dev_K200_chains.jsonl
                  keys: qid, question, gold (dict with "answer"), evidence.chains
                  chains[0].hops[0/1]: {title, text}

  --candidates  exp0c/candidates/dev_M5_7b_K200.jsonl
                  keys: qid, candidates
                  candidates[i]: {answer_id, answer_text}

Output JSONL — one record per question:
{
  "qid": "...",
  "question": "...",
  "gold": "...",
  "hop1_len": 312,
  "hop2_len": 289,
  "candidates": [
    {
      "answer_id": 0,
      "answer_text": "Paris",
      "qa_hop1": 0.823,
      "qa_hop2": 0.041,
      "qa_flat": 0.791,
      "qa_hop_balance": 0.782,
      "qa_min_hop": 0.041
    }
  ]
}

Resumable: skips qids already written to out_jsonl.
Progress: every 100 questions with rate + eta.

Usage:
    python3 tools/exp_a1_qa_hop_score.py \
        --evidence   exp0c/evidence/dev_K200_chains.jsonl \
        --candidates exp0c/candidates/dev_M5_7b_K200.jsonl \
        --model      /var/tmp/u24sf51014/sro/models/deberta-v3-base-squad2 \
        --out_jsonl  exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl \
        --out_json   exp_phaseA/A1.1/qa_scores/summary.json \
        --batch_size 32
"""

import argparse
import json
import os
import re
import string
import time
from typing import Any, Dict, List, Optional, Set, Tuple


# ─────────────────────────── text utils ─────────────────────────────

def normalize_answer(s: str) -> str:
    """Lower, strip articles/punctuation, collapse whitespace.
    Identical to every other tool in this repo."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def em(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


# ─────────────────────────── I/O utils ──────────────────────────────

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_qids(out_jsonl: str) -> Set[str]:
    """Return set of qids already written — for resumability."""
    done: Set[str] = set()
    if not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["qid"]))
            except Exception:
                continue
    return done


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ─────────────────────────── evidence utils ─────────────────────────

def extract_gold(gold_field: Any) -> str:
    """gold is either a plain string or a dict with 'answer' key."""
    if isinstance(gold_field, dict):
        return gold_field.get("answer", "")
    return str(gold_field) if gold_field else ""


def get_hop_texts(ev_record: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Extract (hop1_text, hop2_text) from the top-scoring chain.
    Returns (None, None) if fewer than 2 hops.
    Format: 'Title: passage_text' — same as exp2_q0_hop_diagnostic.py."""
    chains = (ev_record.get("evidence", {}).get("chains")
              or ev_record.get("chains", []))
    if not chains:
        return None, None
    hops = chains[0].get("hops", [])
    if len(hops) < 2:
        return None, None
    h1 = hops[0]
    h2 = hops[1]
    return (f"{h1.get('title','')}: {h1.get('text','')}".strip(),
            f"{h2.get('title','')}: {h2.get('text','')}".strip())


# ─────────────────────────── QA scoring ─────────────────────────────

def score_candidate_qa(
    qa_pipe,
    question: str,
    context: str,
    candidate: str,
    null_cap: float = 0.10,
) -> float:
    """
    Score P(candidate is the answer | question, context).

    If the model's top span matches the candidate → return span confidence.
    If not → return min(1 - span_confidence, null_cap).

    null_cap=0.10 keeps the feature near-zero when the candidate is not
    grounded, matching NLI calibration so features are on the same scale.
    """
    if not context.strip() or not question.strip():
        return 0.0
    try:
        result = qa_pipe(question=question, context=context)
    except Exception:
        return 0.0

    predicted = result.get("answer", "") or ""
    score     = float(result.get("score", 0.0))

    if em(predicted, candidate):
        return score
    else:
        return min(1.0 - score, null_cap)


def score_all_contexts(
    qa_pipe,
    question: str,
    hop1_text: str,
    hop2_text: str,
    candidate: str,
) -> Dict[str, float]:
    """Score hop1, hop2, flat. Compute derived features."""
    qa_hop1 = score_candidate_qa(qa_pipe, question, hop1_text, candidate)
    qa_hop2 = score_candidate_qa(qa_pipe, question, hop2_text, candidate)
    qa_flat  = score_candidate_qa(qa_pipe, question,
                                  hop1_text + " " + hop2_text, candidate)
    return {
        "qa_hop1":        round(qa_hop1, 6),
        "qa_hop2":        round(qa_hop2, 6),
        "qa_flat":        round(qa_flat, 6),
        "qa_hop_balance": round(abs(qa_hop1 - qa_hop2), 6),
        "qa_min_hop":     round(min(qa_hop1, qa_hop2), 6),
    }


# ─────────────────────────── main ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Phase A1.1 — QA cross-encoder hop scoring"
    )
    ap.add_argument("--evidence",   required=True,
                    help="exp0c/evidence/dev_K200_chains.jsonl")
    ap.add_argument("--candidates", required=True,
                    help="exp0c/candidates/dev_M5_7b_K200.jsonl")
    ap.add_argument("--model",      required=True,
                    help="Path to deepset/deberta-v3-base-squad2")
    ap.add_argument("--out_jsonl",  required=True,
                    help="Output: per-question QA scores")
    ap.add_argument("--out_json",   required=True,
                    help="Output: summary JSON with run metadata")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device",     type=int, default=0,
                    help="CUDA device index (default: 0; -1 for CPU)")
    args = ap.parse_args()

    # ── Load evidence → question + gold + hop texts ──────────────────
    print("[A1.1] Loading evidence...")
    ev_map: Dict[str, Dict] = {}
    for rec in iter_jsonl(args.evidence):
        ev_map[str(rec["qid"])] = rec
    print(f"       {len(ev_map):,} questions")

    # ── Load candidates → answer_id + answer_text ────────────────────
    print("[A1.1] Loading candidates...")
    cand_map: Dict[str, List[Dict]] = {}
    for rec in iter_jsonl(args.candidates):
        cand_map[str(rec["qid"])] = rec.get("candidates", [])
    print(f"       {len(cand_map):,} questions")

    # ── Schema check ─────────────────────────────────────────────────
    first_qid   = next(iter(cand_map))
    first_cands = cand_map[first_qid]
    assert first_cands, \
        "ERROR: first question has empty candidates list"
    assert "answer_text" in first_cands[0], \
        f"ERROR: expected 'answer_text', got keys: {list(first_cands[0].keys())}"
    first_ev = ev_map.get(first_qid, {})
    assert "question" in first_ev, \
        f"ERROR: 'question' not found in evidence record keys: {list(first_ev.keys())}"
    print(f"       Schema OK — qid={first_qid}  "
          f"n_cands={len(first_cands)}  "
          f"first_ans={first_cands[0]['answer_text']!r}  "
          f"gold={extract_gold(first_ev.get('gold',''))!r}")

    # ── Resumability ─────────────────────────────────────────────────
    done = load_done_qids(args.out_jsonl)
    if done:
        print(f"[A1.1] Resuming: {len(done):,} already scored")

    # ── Build QA pipeline ────────────────────────────────────────────
    print(f"[A1.1] Loading model: {args.model}")
    t_load = time.time()
    from transformers import pipeline as hf_pipeline
    qa_pipe = hf_pipeline(
        "question-answering",
        model=args.model,
        tokenizer=args.model,
        handle_impossible_answer=True,
        device=args.device,
        max_answer_len=50,
    )
    print(f"       Loaded in {time.time() - t_load:.1f}s  device={args.device}")

    # ── Scoring loop ──────────────────────────────────────────────────
    all_qids = sorted(set(ev_map.keys()) & set(cand_map.keys()))
    to_score = [q for q in all_qids if q not in done]

    print(f"\n[A1.1] To score: {len(to_score):,}  "
          f"(total: {len(all_qids):,}  "
          f"~{len(to_score)*5*3:,} forward passes)\n")

    t0              = time.time()
    n_written       = 0
    n_skip_hops     = 0
    n_skip_cands    = 0

    for i, qid in enumerate(to_score):
        ev_rec = ev_map[qid]

        hop1_text, hop2_text = get_hop_texts(ev_rec)
        if hop1_text is None or hop2_text is None:
            n_skip_hops += 1
            continue

        candidates = cand_map.get(qid, [])
        if not candidates:
            n_skip_cands += 1
            continue

        question = ev_rec.get("question", "")
        gold     = extract_gold(ev_rec.get("gold", ""))

        scored = []
        for cand in candidates:
            scores = score_all_contexts(
                qa_pipe, question, hop1_text, hop2_text,
                cand["answer_text"],
            )
            scored.append({
                "answer_id":   cand["answer_id"],
                "answer_text": cand["answer_text"],
                **scores,
            })

        append_jsonl(args.out_jsonl, {
            "qid":        qid,
            "question":   question,
            "gold":       gold,
            "hop1_len":   len(hop1_text),
            "hop2_len":   len(hop2_text),
            "candidates": scored,
        })
        n_written += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate    = n_written / elapsed * 60
            eta     = (len(to_score) - (i + 1)) / max(n_written / elapsed, 1e-9)
            print(f"  [{i+1:5d}/{len(to_score):5d}]  "
                  f"written={n_written:,}  "
                  f"{rate:.1f} q/min  "
                  f"eta={eta/60:.1f} min")

    elapsed_total = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n[A1.1] Done.")
    print(f"       Written:            {n_written:,}")
    print(f"       Skipped (no hops):  {n_skip_hops:,}")
    print(f"       Skipped (no cands): {n_skip_cands:,}")
    print(f"       Total time:         {elapsed_total/60:.1f} min")

    summary = {
        "script":          "exp_a1_qa_hop_score.py",
        "model":           args.model,
        "evidence":        args.evidence,
        "candidates":      args.candidates,
        "out_jsonl":       args.out_jsonl,
        "n_written":       n_written,
        "n_skip_hops":     n_skip_hops,
        "n_skip_cands":    n_skip_cands,
        "elapsed_min":     round(elapsed_total / 60, 2),
        "device":          args.device,
        "features_output": ["qa_hop1", "qa_hop2", "qa_flat",
                            "qa_hop_balance", "qa_min_hop"],
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"       Summary:            {args.out_json}")

    # ── Sanity check — first 3 records ───────────────────────────────
    print("\n[A1.1] Sanity check — first 3 records:")
    count = 0
    for rec in iter_jsonl(args.out_jsonl):
        if count >= 3:
            break
        print(f"  qid={rec['qid']}  gold={rec['gold']!r}  "
              f"n_cands={len(rec['candidates'])}")
        for c in rec["candidates"][:2]:
            print(f"    ans={c['answer_text']!r:25s}  "
                  f"hop1={c['qa_hop1']:.3f}  "
                  f"hop2={c['qa_hop2']:.3f}  "
                  f"flat={c['qa_flat']:.3f}  "
                  f"bal={c['qa_hop_balance']:.3f}  "
                  f"min={c['qa_min_hop']:.3f}")
        count += 1


if __name__ == "__main__":
    main()