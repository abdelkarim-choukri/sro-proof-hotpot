#!/usr/bin/env python3
"""
exp4_llm_verifier.py — LLM-as-Verifier with Chain-of-Thought

Two modes:
  Option 1 (--mode scores_in_prompt):
    Show ALL candidates + their hop scores to the LLM.
    LLM reasons about which is best supported by both hops.

  Option 2 (--mode prefilter):
    Pre-filter to top 2 candidates by XGB confidence.
    LLM reasons between only those 2.

Both modes use chain-of-thought: the LLM must reason step by step
before outputting a final answer and confidence score.

Input: pilot_questions.jsonl (from exp4_pilot_select.py)
Output: predictions JSONL with {qid, pred, confidence, reasoning, mode}

Usage:
    python3 tools/exp4_llm_verifier.py \
        --pilot       exp4_7b/pilot/pilot_questions.jsonl \
        --mode        scores_in_prompt \
        --out_preds   exp4_7b/pilot/preds_option1.jsonl \
        --llm_base_url http://127.0.0.1:8000/v1 \
        --llm_model_id /var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct \
        --temperature 0.0 \
        --max_new_tokens 512
"""

import argparse
import json
import os
import re
import string
import time
import collections
from typing import List, Dict, Any
from urllib import request, error


# ─────────────────────────── text utils ─────────────────────────────

def normalize(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def em(pred, gold):
    return int(normalize(pred) == normalize(gold))

def f1_score(pred, gold):
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    common = collections.Counter(p_toks) & collections.Counter(g_toks)
    n = sum(common.values())
    if not n: return 0.0
    p = n / len(p_toks)
    r = n / len(g_toks)
    return 2 * p * r / (p + r)


# ─────────────────────────── LLM client ─────────────────────────────

def call_llm(base_url, model, prompt, temperature, max_tokens, timeout=120):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data,
                          headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body_text[:300]}") from e

    choices = body.get("choices", [])
    if choices:
        return choices[0]["message"]["content"]
    return ""


def retry_call(fn, retries=3, sleep_s=2.0):
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(sleep_s * (attempt + 1))


# ─────────────────────────── prompt builders ────────────────────────

def build_prompt_option1(rec):
    """Option 1: All candidates + hop scores in prompt."""
    cands = rec["candidates"]
    hop1_text = rec.get("hop1_text") or "(not available)"
    hop2_text = rec.get("hop2_text") or "(not available)"
    question = rec.get("question", "")

    # Filter to non-bad candidates
    valid_cands = []
    for i, c in enumerate(cands):
        ans = c.get("answer", "").strip()
        if not ans or ans.lower() in ("unknown", "unk"):
            continue
        if len(ans) > 120:
            continue
        valid_cands.append((i, c))

    if not valid_cands:
        return None, []

    cand_block = []
    for idx, (orig_i, c) in enumerate(valid_cands):
        letter = chr(65 + idx)  # A, B, C, ...
        h1 = c.get("nli_hop1", 0) or 0
        h2 = c.get("nli_hop2", 0) or 0
        bal = abs(h1 - h2)
        cand_block.append(
            f"  {letter}) \"{c['answer']}\"\n"
            f"     Hop1 entailment: {h1:.3f}  |  Hop2 entailment: {h2:.3f}  |  "
            f"Balance: {bal:.3f}"
        )

    prompt = f"""You are verifying answers to a multi-hop question using two evidence documents.

EVIDENCE:
Document 1 (bridge): {hop1_text}

Document 2 (answer): {hop2_text}

QUESTION: {question}

CANDIDATE ANSWERS (with per-hop NLI entailment scores):
{chr(10).join(cand_block)}

A good answer should be supported by BOTH documents (high scores for both Hop1 and Hop2).
If one hop has a very low score, the answer may only be partially supported.

Think step by step:
1. Which candidate has strong support from BOTH hops?
2. Is there a candidate where the evidence clearly entails the answer?
3. Consider the balance — an answer supported by only one hop is weaker.

After your reasoning, output EXACTLY two lines at the end:
ANSWER: [your chosen answer, copied exactly from the candidates]
CONFIDENCE: [a number between 0.0 and 1.0]"""

    return prompt, valid_cands


def build_prompt_option2(rec):
    """Option 2: Pre-filter to top 2 by XGB, then LLM reasons between them."""
    cands = rec["candidates"]
    xgb_probs = rec.get("xgb_probs", [])
    hop1_text = rec.get("hop1_text") or "(not available)"
    hop2_text = rec.get("hop2_text") or "(not available)"
    question = rec.get("question", "")

    # Rank candidates by XGB probability
    indexed = [(i, xgb_probs[i] if i < len(xgb_probs) else 0.0)
               for i in range(len(cands))]
    indexed.sort(key=lambda x: -x[1])

    # Take top 2 non-bad candidates
    top2 = []
    for orig_i, prob in indexed:
        c = cands[orig_i]
        ans = c.get("answer", "").strip()
        if not ans or ans.lower() in ("unknown", "unk") or len(ans) > 120:
            continue
        top2.append((orig_i, c, prob))
        if len(top2) == 2:
            break

    if len(top2) < 2:
        # Not enough valid candidates for comparison
        if top2:
            return None, top2  # just return the one
        return None, []

    cand_block = []
    for idx, (orig_i, c, prob) in enumerate(top2):
        letter = chr(65 + idx)
        h1 = c.get("nli_hop1", 0) or 0
        h2 = c.get("nli_hop2", 0) or 0
        cand_block.append(
            f"  {letter}) \"{c['answer']}\"\n"
            f"     Hop1 entailment: {h1:.3f}  |  Hop2 entailment: {h2:.3f}  |  "
            f"Verifier confidence: {prob:.3f}"
        )

    prompt = f"""You are choosing between two candidate answers to a multi-hop question.
A previous verifier narrowed it down to these two, but may have chosen wrong.

EVIDENCE:
Document 1 (bridge): {hop1_text}

Document 2 (answer): {hop2_text}

QUESTION: {question}

TOP 2 CANDIDATES:
{chr(10).join(cand_block)}

Think step by step:
1. Read both documents carefully.
2. For each candidate, check: does Document 1 connect to it? Does Document 2 support it?
3. Which candidate is more clearly supported by the combined evidence?

After your reasoning, output EXACTLY two lines at the end:
ANSWER: [your chosen answer, copied exactly from the candidates]
CONFIDENCE: [a number between 0.0 and 1.0]"""

    return prompt, top2


# ─────────────────────────── response parsing ───────────────────────

def parse_response(text, valid_answers):
    """Extract answer and confidence from LLM response."""
    lines = text.strip().splitlines()

    answer = ""
    confidence = 0.5

    for line in reversed(lines):
        line_stripped = line.strip()
        if line_stripped.upper().startswith("ANSWER:"):
            answer = line_stripped[7:].strip().strip('"').strip("'").strip()
        elif line_stripped.upper().startswith("CONFIDENCE:"):
            try:
                confidence = float(
                    re.search(r'[\d.]+', line_stripped[11:]).group())
                confidence = max(0.0, min(1.0, confidence))
            except:
                confidence = 0.5

    # If answer not parsed cleanly, try fuzzy match against valid answers
    if answer and valid_answers:
        norm_answer = normalize(answer)
        best_match = None
        best_f1 = 0
        for va in valid_answers:
            f = f1_score(answer, va)
            if f > best_f1:
                best_f1 = f
                best_match = va
        if best_match and best_f1 > 0.5:
            answer = best_match

    return answer, confidence, text


# ─────────────────────────── main ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot",        required=True)
    ap.add_argument("--mode",         required=True,
                    choices=["scores_in_prompt", "prefilter"])
    ap.add_argument("--out_preds",    required=True)
    ap.add_argument("--llm_base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--llm_model_id", required=True)
    ap.add_argument("--temperature",  type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--timeout",      type=int, default=120)
    ap.add_argument("--resume",       action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_preds), exist_ok=True)

    # Load pilot questions
    pilot = []
    for line in open(args.pilot):
        pilot.append(json.loads(line))
    print(f"[llm-verifier] Mode: {args.mode}")
    print(f"[llm-verifier] Pilot questions: {len(pilot)}")
    print(f"[llm-verifier] Model: {args.llm_model_id}")

    # Resume support
    done_qids = set()
    if args.resume and os.path.exists(args.out_preds):
        for line in open(args.out_preds):
            r = json.loads(line)
            done_qids.add(r["qid"])
        print(f"[llm-verifier] Resuming: {len(done_qids)} already done")

    wrote = skipped = errors = 0
    t0 = time.time()

    with open(args.out_preds, "a" if args.resume else "w") as f:
        for i, rec in enumerate(pilot):
            qid = rec["qid"]
            if qid in done_qids:
                skipped += 1
                continue

            gold = rec["gold"]
            cands = rec["candidates"]
            valid_answers = [c["answer"] for c in cands
                            if c.get("answer", "").strip()]

            # Build prompt
            if args.mode == "scores_in_prompt":
                prompt, valid_cands = build_prompt_option1(rec)
            else:
                prompt, valid_cands = build_prompt_option2(rec)

            if prompt is None:
                # Fallback: use XGB prediction
                pred_rec = {
                    "qid": qid,
                    "pred": rec.get("xgb_pred", ""),
                    "confidence": 0.0,
                    "reasoning": "(no valid candidates for LLM)",
                    "mode": args.mode,
                    "fallback": True,
                }
                f.write(json.dumps(pred_rec) + "\n")
                wrote += 1
                continue

            # Call LLM
            try:
                response = retry_call(
                    lambda: call_llm(
                        args.llm_base_url, args.llm_model_id,
                        prompt, args.temperature,
                        args.max_new_tokens, args.timeout
                    ),
                    retries=2, sleep_s=2.0
                )

                answer, confidence, reasoning = parse_response(
                    response, valid_answers)

                pred_rec = {
                    "qid": qid,
                    "pred": answer,
                    "confidence": round(confidence, 4),
                    "reasoning": reasoning[:1000],  # truncate for storage
                    "mode": args.mode,
                    "fallback": False,
                    "n_candidates_shown": len(valid_cands),
                }

            except Exception as e:
                errors += 1
                pred_rec = {
                    "qid": qid,
                    "pred": rec.get("xgb_pred", ""),
                    "confidence": 0.0,
                    "reasoning": f"ERROR: {str(e)[:200]}",
                    "mode": args.mode,
                    "fallback": True,
                }

            f.write(json.dumps(pred_rec) + "\n")
            f.flush()
            wrote += 1

            if (wrote + skipped) % 50 == 0:
                elapsed = time.time() - t0
                rate = wrote / max(elapsed, 1) * 60
                print(f"[llm-verifier] {wrote} wrote  {skipped} skipped  "
                      f"{errors} errors  ({rate:.0f}/min)")

    elapsed = time.time() - t0
    print(f"\n[llm-verifier] DONE: {wrote} wrote  {skipped} skipped  "
          f"{errors} errors  ({elapsed/60:.1f} min)")

    # ── compute metrics ──
    preds = {}
    for line in open(args.out_preds):
        r = json.loads(line)
        preds[r["qid"]] = r

    # Load pilot for gold + bucket info
    pilot_map = {r["qid"]: r for r in pilot}

    # Metrics
    total_em = total_f1 = n = 0
    c2_correct = c2_total = 0
    d_correct = d_total = 0

    for qid, p in preds.items():
        rec = pilot_map.get(qid, {})
        gold = rec.get("gold", "")
        bucket = rec.get("bucket", "")

        is_correct = em(p["pred"], gold)
        total_em += is_correct
        total_f1 += f1_score(p["pred"], gold)
        n += 1

        if bucket == "C2":
            c2_total += 1
            c2_correct += is_correct
        elif bucket == "D":
            d_total += 1
            d_correct += is_correct

    xgb_em_c2 = 0  # XGB is always wrong on C2 by definition
    xgb_em_d = d_total  # XGB is always right on D by definition

    print(f"\n{'='*60}")
    print(f"  {args.mode.upper()} RESULTS")
    print(f"{'='*60}")
    print(f"  Overall EM:     {total_em/max(n,1):.4f}  ({total_em}/{n})")
    print(f"  Overall F1:     {total_f1/max(n,1):.4f}")
    print(f"")
    print(f"  C2 recovery:    {c2_correct}/{c2_total}  "
          f"({c2_correct/max(c2_total,1):.1%})")
    print(f"  C2 baseline:    0/{c2_total}  (XGB always wrong on C2)")
    print(f"")
    print(f"  D retention:    {d_correct}/{d_total}  "
          f"({d_correct/max(d_total,1):.1%})")
    print(f"  D baseline:     {d_total}/{d_total}  (XGB always right on D)")
    print(f"")
    net_gain = c2_correct - (d_total - d_correct)
    print(f"  Net gain vs XGB: {c2_correct} C2 recovered "
          f"- {d_total - d_correct} D lost = {net_gain:+d}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()