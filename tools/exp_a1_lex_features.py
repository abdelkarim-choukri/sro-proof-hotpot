#!/usr/bin/env python3
"""
exp_a1_lex_features.py  —  Phase A1.2: Lexical Grounding Features

Pure string-matching features — no model, no GPU, CPU only.
Measures whether candidate answers and question entities are literally
present in each hop's text.

Verified input schemas:
  --evidence    exp0c/evidence/dev_K200_chains.jsonl
                  keys: qid, question, gold (dict with "answer"),
                        evidence.chains[0].hops[0/1]: {title, text}

  --candidates  exp0c/candidates/dev_M5_7b_K200.jsonl
                  keys: qid, candidates[i]: {answer_id, answer_text}

Output features per candidate:
  ans_in_hop1       Binary: normalized candidate appears in normalized hop1 text
  ans_in_hop2       Binary: normalized candidate appears in normalized hop2 text
  ans_in_both       Binary: appears in both hops
  ans_in_neither    Binary: appears in neither hop (hallucination signal)
  lex_hop1          Token-level Jaccard(candidate_tokens, hop1_tokens)
  lex_hop2          Token-level Jaccard(candidate_tokens, hop2_tokens)
  lex_flat          Token-level Jaccard(candidate_tokens, hop1+hop2_tokens)
  lex_hop_balance   |lex_hop1 - lex_hop2|
  lex_min_hop       min(lex_hop1, lex_hop2)
  q_entity_in_hop1  Fraction of question named entities found in hop1 text
  q_entity_in_hop2  Fraction of question named entities found in hop2 text

Named entity extraction: capitalized non-stop-word tokens from question,
excluding the first token (sentence-initial capital). No NER model needed.

Output JSONL — one record per question:
{
  "qid": "...",
  "question": "...",
  "gold": "...",
  "q_entities": ["Scott Derrickson", "Ed Wood"],
  "hop1_len": 312,
  "hop2_len": 289,
  "candidates": [
    {
      "answer_id": 0,
      "answer_text": "yes",
      "ans_in_hop1": 0,
      "ans_in_hop2": 0,
      "ans_in_both": 0,
      "ans_in_neither": 1,
      "lex_hop1": 0.0,
      "lex_hop2": 0.0,
      "lex_flat": 0.0,
      "lex_hop_balance": 0.0,
      "lex_min_hop": 0.0,
      "q_entity_in_hop1": 1.0,
      "q_entity_in_hop2": 1.0
    }
  ]
}

Resumable: skips qids already written to out_jsonl.
Progress: every 500 questions (fast CPU run, ~10 min total).

Usage:
    python3 tools/exp_a1_lex_features.py \
        --evidence   exp0c/evidence/dev_K200_chains.jsonl \
        --candidates exp0c/candidates/dev_M5_7b_K200.jsonl \
        --out_jsonl  exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl \
        --out_json   exp_phaseA/A1.2/lex_features/summary.json
"""

import argparse
import json
import os
import re
import string
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# ─────────────────────────── constants ──────────────────────────────

# Common English stop words — capitalized versions filtered out of
# named entity extraction. Kept small and focused on question words
# that appear capitalized mid-sentence in HotpotQA questions.
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are",
    "were", "be", "been", "have", "has", "had", "do", "did", "does",
    "will", "would", "could", "should", "may", "might", "shall",
    "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "each", "than", "that", "this", "these", "those", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "if",
    "though", "although", "because", "since", "while", "after",
    "before", "until", "unless", "also", "just", "more", "most",
    "same", "such", "than", "then", "there", "their", "they",
    "he", "she", "it", "we", "you", "i", "his", "her", "its",
    "our", "your", "my", "its", "his", "her",
}


# ─────────────────────────── text utils ─────────────────────────────

def normalize_answer(s: str) -> str:
    """Lower, strip articles/punctuation, collapse whitespace.
    Identical to every other tool in this repo."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def tokenize(s: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace.
    Returns non-empty tokens."""
    s = s.lower()
    s = "".join(c if c not in string.punctuation else " " for c in s)
    return [t for t in s.split() if t]


def jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Token-level Jaccard similarity between two token sets."""
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def extract_question_entities(question: str) -> List[str]:
    """
    Extract named entity candidates from a question using capitalization
    heuristics. No NER model — pure string operations.

    Strategy:
    1. Tokenize preserving original case
    2. Skip the first token (sentence-initial capital is not an entity)
    3. Keep tokens that start with a capital letter and are not stop words
    4. Merge consecutive capitalized tokens into multi-word entities

    Example: "Were Scott Derrickson and Ed Wood of the same nationality?"
    → ["Scott Derrickson", "Ed Wood"]
    """
    # Split on whitespace, keep punctuation attached for now
    raw_tokens = question.split()
    if not raw_tokens:
        return []

    # Strip punctuation from each token but preserve case
    cleaned = []
    for tok in raw_tokens:
        t = tok.strip(string.punctuation)
        if t:
            cleaned.append(t)

    if not cleaned:
        return []

    # Skip index 0 (sentence-initial capital)
    entities = []
    current_entity = []

    for i, tok in enumerate(cleaned):
        if i == 0:
            # Always skip first token regardless of case
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
            continue

        is_capitalized = tok and tok[0].isupper()
        is_stopword    = tok.lower() in STOP_WORDS

        if is_capitalized and not is_stopword:
            current_entity.append(tok)
        else:
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []

    if current_entity:
        entities.append(" ".join(current_entity))

    return entities


# ─────────────────────────── I/O utils ──────────────────────────────

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_qids(out_jsonl: str) -> Set[str]:
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


def extract_gold(gold_field: Any) -> str:
    if isinstance(gold_field, dict):
        return gold_field.get("answer", "")
    return str(gold_field) if gold_field else ""


def get_hop_texts(ev_record: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Extract (hop1_text, hop2_text) from top chain.
    Format: 'Title: passage_text' — same as all other tools."""
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


# ─────────────────────────── lexical scoring ────────────────────────

def lex_score_candidate(
    candidate: str,
    hop1_text: str,
    hop2_text: str,
    q_entities: List[str],
) -> Dict[str, Any]:
    """
    Compute all lexical features for one (candidate, hop1, hop2) triple.
    Pure string operations — no model calls.
    """
    norm_cand  = normalize_answer(candidate)
    norm_hop1  = normalize_answer(hop1_text)
    norm_hop2  = normalize_answer(hop2_text)
    norm_flat  = norm_hop1 + " " + norm_hop2

    # ── Binary grounding ──────────────────────────────────────────────
    # Use normalized substring match — same as the em() check everywhere
    # but we want substring (candidate IN hop), not equality
    ans_in_hop1 = int(bool(norm_cand) and norm_cand in norm_hop1)
    ans_in_hop2 = int(bool(norm_cand) and norm_cand in norm_hop2)
    ans_in_both    = int(ans_in_hop1 == 1 and ans_in_hop2 == 1)
    ans_in_neither = int(ans_in_hop1 == 0 and ans_in_hop2 == 0)

    # ── Token-level Jaccard ───────────────────────────────────────────
    cand_toks  = set(tokenize(candidate))
    hop1_toks  = set(tokenize(hop1_text))
    hop2_toks  = set(tokenize(hop2_text))
    flat_toks  = hop1_toks | hop2_toks

    lex_hop1 = jaccard(cand_toks, hop1_toks)
    lex_hop2 = jaccard(cand_toks, hop2_toks)
    lex_flat = jaccard(cand_toks, flat_toks)

    lex_hop_balance = abs(lex_hop1 - lex_hop2)
    lex_min_hop     = min(lex_hop1, lex_hop2)

    # ── Question entity grounding ─────────────────────────────────────
    # Fraction of question entities literally found in each hop
    if q_entities:
        hop1_lower = hop1_text.lower()
        hop2_lower = hop2_text.lower()
        found_hop1 = sum(1 for e in q_entities if e.lower() in hop1_lower)
        found_hop2 = sum(1 for e in q_entities if e.lower() in hop2_lower)
        q_entity_in_hop1 = found_hop1 / len(q_entities)
        q_entity_in_hop2 = found_hop2 / len(q_entities)
    else:
        # No extractable entities (e.g. pure yes/no question with no NEs)
        q_entity_in_hop1 = 0.0
        q_entity_in_hop2 = 0.0

    return {
        "ans_in_hop1":      ans_in_hop1,
        "ans_in_hop2":      ans_in_hop2,
        "ans_in_both":      ans_in_both,
        "ans_in_neither":   ans_in_neither,
        "lex_hop1":         round(lex_hop1, 6),
        "lex_hop2":         round(lex_hop2, 6),
        "lex_flat":         round(lex_flat, 6),
        "lex_hop_balance":  round(lex_hop_balance, 6),
        "lex_min_hop":      round(lex_min_hop, 6),
        "q_entity_in_hop1": round(q_entity_in_hop1, 6),
        "q_entity_in_hop2": round(q_entity_in_hop2, 6),
    }


# ─────────────────────────── main ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Phase A1.2 — Lexical grounding features (CPU only)"
    )
    ap.add_argument("--evidence",   required=True,
                    help="exp0c/evidence/dev_K200_chains.jsonl")
    ap.add_argument("--candidates", required=True,
                    help="exp0c/candidates/dev_M5_7b_K200.jsonl")
    ap.add_argument("--out_jsonl",  required=True,
                    help="Output: per-question lexical features")
    ap.add_argument("--out_json",   required=True,
                    help="Output: summary JSON")
    args = ap.parse_args()

    # ── Load evidence ─────────────────────────────────────────────────
    print("[A1.2] Loading evidence...")
    ev_map: Dict[str, Dict] = {}
    for rec in iter_jsonl(args.evidence):
        ev_map[str(rec["qid"])] = rec
    print(f"       {len(ev_map):,} questions")

    # ── Load candidates ───────────────────────────────────────────────
    print("[A1.2] Loading candidates...")
    cand_map: Dict[str, List[Dict]] = {}
    for rec in iter_jsonl(args.candidates):
        cand_map[str(rec["qid"])] = rec.get("candidates", [])
    print(f"       {len(cand_map):,} questions")

    # ── Schema check ─────────────────────────────────────────────────
    first_qid   = next(iter(cand_map))
    first_cands = cand_map[first_qid]
    assert first_cands, "ERROR: first question has empty candidates"
    assert "answer_text" in first_cands[0], \
        f"ERROR: expected 'answer_text', got: {list(first_cands[0].keys())}"
    first_ev = ev_map.get(first_qid, {})
    assert "question" in first_ev, \
        f"ERROR: 'question' missing from evidence, keys: {list(first_ev.keys())}"

    # Quick entity extraction check on first question
    test_q       = first_ev["question"]
    test_entities = extract_question_entities(test_q)
    print(f"       Schema OK — question={test_q!r}")
    print(f"       Entities extracted: {test_entities}")

    # ── Resumability ──────────────────────────────────────────────────
    done = load_done_qids(args.out_jsonl)
    if done:
        print(f"[A1.2] Resuming: {len(done):,} already scored")

    # ── Main loop ─────────────────────────────────────────────────────
    all_qids = sorted(set(ev_map.keys()) & set(cand_map.keys()))
    to_score = [q for q in all_qids if q not in done]

    print(f"\n[A1.2] Questions to score: {len(to_score):,}  "
          f"(total: {len(all_qids):,})  CPU only\n")

    t0           = time.time()
    n_written    = 0
    n_skip_hops  = 0
    n_skip_cands = 0

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

        question   = ev_rec.get("question", "")
        gold       = extract_gold(ev_rec.get("gold", ""))
        q_entities = extract_question_entities(question)

        scored = []
        for cand in candidates:
            feats = lex_score_candidate(
                cand["answer_text"], hop1_text, hop2_text, q_entities
            )
            scored.append({
                "answer_id":   cand["answer_id"],
                "answer_text": cand["answer_text"],
                **feats,
            })

        append_jsonl(args.out_jsonl, {
            "qid":        qid,
            "question":   question,
            "gold":       gold,
            "q_entities": q_entities,
            "hop1_len":   len(hop1_text),
            "hop2_len":   len(hop2_text),
            "candidates": scored,
        })
        n_written += 1

        # Progress every 500 questions — fast enough that 100 is noisy
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate    = n_written / elapsed * 60
            eta     = (len(to_score) - (i + 1)) / max(n_written / elapsed, 1e-9)
            print(f"  [{i+1:5d}/{len(to_score):5d}]  "
                  f"written={n_written:,}  "
                  f"{rate:.0f} q/min  "
                  f"eta={eta/60:.1f} min")

    elapsed_total = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n[A1.2] Done.")
    print(f"       Written:            {n_written:,}")
    print(f"       Skipped (no hops):  {n_skip_hops:,}")
    print(f"       Skipped (no cands): {n_skip_cands:,}")
    print(f"       Total time:         {elapsed_total/60:.1f} min")

    summary = {
        "script":          "exp_a1_lex_features.py",
        "evidence":        args.evidence,
        "candidates":      args.candidates,
        "out_jsonl":       args.out_jsonl,
        "n_written":       n_written,
        "n_skip_hops":     n_skip_hops,
        "n_skip_cands":    n_skip_cands,
        "elapsed_min":     round(elapsed_total / 60, 2),
        "features_output": [
            "ans_in_hop1", "ans_in_hop2", "ans_in_both", "ans_in_neither",
            "lex_hop1", "lex_hop2", "lex_flat", "lex_hop_balance",
            "lex_min_hop", "q_entity_in_hop1", "q_entity_in_hop2",
        ],
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"       Summary:            {args.out_json}")

    # ── Sanity check — first 3 records ────────────────────────────────
    print("\n[A1.2] Sanity check — first 3 records:")
    count = 0
    for rec in iter_jsonl(args.out_jsonl):
        if count >= 3:
            break
        print(f"  qid={rec['qid']}  gold={rec['gold']!r}  "
              f"entities={rec['q_entities']}")
        for c in rec["candidates"][:2]:
            print(f"    ans={c['answer_text']!r:25s}  "
                  f"in_h1={c['ans_in_hop1']}  in_h2={c['ans_in_hop2']}  "
                  f"neither={c['ans_in_neither']}  "
                  f"lex_h1={c['lex_hop1']:.3f}  lex_h2={c['lex_hop2']:.3f}  "
                  f"q_ent_h1={c['q_entity_in_hop1']:.2f}  "
                  f"q_ent_h2={c['q_entity_in_hop2']:.2f}")
        count += 1


if __name__ == "__main__":
    main()
