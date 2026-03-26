#!/usr/bin/env python3
"""
exp4_hybrid_verifier.py — Hybrid XGB + LLM Verifier

Strategy:
  - If XGB confidence ≥ threshold → trust XGB (keep its prediction)
  - If XGB confidence < threshold → send to LLM for chain-of-thought reasoning

This preserves D questions (high-confidence XGB is usually right) while
targeting C2 questions (low-confidence XGB is where it gets confused).

Runs on the full pilot set (300 questions) with multiple thresholds to
find the optimal operating point.

Input: pilot_questions.jsonl + LLM verifier (uses vLLM)
Output: predictions at each threshold + comparison table

Usage:
    python3 tools/exp4_hybrid_verifier.py \
        --pilot         exp4_7b/pilot/pilot_questions.jsonl \
        --out_dir       exp4_7b/pilot/hybrid \
        --llm_base_url  http://127.0.0.1:8000/v1 \
        --llm_model_id  /var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct \
        --temperature   0.0 \
        --max_new_tokens 512
"""

import argparse
import collections
import json
import os
import re
import string
import time
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
        "messages": [{"role": "user", "content": prompt}],
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
    return choices[0]["message"]["content"] if choices else ""


def retry_call(fn, retries=2, sleep_s=2.0):
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(sleep_s * (attempt + 1))


# ─────────────────────────── prompt builder ─────────────────────────

def build_hybrid_prompt(rec):
    """
    Option 1 style prompt (all valid candidates + hop scores).
    Used only for low-confidence questions.
    """
    cands = rec["candidates"]
    hop1_text = rec.get("hop1_text") or "(not available)"
    hop2_text = rec.get("hop2_text") or "(not available)"
    question = rec.get("question", "")

    valid_cands = []
    for i, c in enumerate(cands):
        ans = c.get("answer", "").strip()
        if not ans or ans.lower() in ("unknown", "unk") or len(ans) > 120:
            continue
        valid_cands.append((i, c))

    if not valid_cands:
        return None, []

    cand_block = []
    for idx, (orig_i, c) in enumerate(valid_cands):
        letter = chr(65 + idx)
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


def parse_response(text, valid_answers):
    lines = text.strip().splitlines()
    answer = ""
    confidence = 0.5

    for line in reversed(lines):
        ls = line.strip()
        if ls.upper().startswith("ANSWER:"):
            answer = ls[7:].strip().strip('"').strip("'").strip()
        elif ls.upper().startswith("CONFIDENCE:"):
            try:
                confidence = float(re.search(r'[\d.]+', ls[11:]).group())
                confidence = max(0.0, min(1.0, confidence))
            except:
                confidence = 0.5

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

    return answer, confidence


# ─────────────────────────── main ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot",        required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--llm_base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--llm_model_id", required=True)
    ap.add_argument("--temperature",  type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--timeout",      type=int, default=120)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load pilot
    pilot = []
    for line in open(args.pilot):
        pilot.append(json.loads(line))
    print(f"[hybrid] Pilot questions: {len(pilot)}")

    # ── Step 1: Score ALL questions with LLM (cache for threshold sweep) ──
    # We only need to call LLM once per question, then reuse at different thresholds
    llm_cache_path = os.path.join(args.out_dir, "llm_cache.jsonl")

    llm_cache = {}
    if os.path.exists(llm_cache_path):
        for line in open(llm_cache_path):
            r = json.loads(line)
            llm_cache[r["qid"]] = r
        print(f"[hybrid] LLM cache: {len(llm_cache)} already scored")

    # Score any missing questions
    n_to_score = sum(1 for rec in pilot if rec["qid"] not in llm_cache)
    print(f"[hybrid] Need to score: {n_to_score} questions with LLM")

    if n_to_score > 0:
        t0 = time.time()
        wrote = 0
        with open(llm_cache_path, "a") as f:
            for rec in pilot:
                qid = rec["qid"]
                if qid in llm_cache:
                    continue

                cands = rec["candidates"]
                valid_answers = [c["answer"] for c in cands
                                if c.get("answer", "").strip()]

                prompt, valid_cands = build_hybrid_prompt(rec)

                if prompt is None:
                    result = {
                        "qid": qid,
                        "llm_pred": rec.get("xgb_pred", ""),
                        "llm_confidence": 0.0,
                        "fallback": True,
                    }
                else:
                    try:
                        response = retry_call(
                            lambda: call_llm(
                                args.llm_base_url, args.llm_model_id,
                                prompt, args.temperature,
                                args.max_new_tokens, args.timeout
                            )
                        )
                        answer, confidence = parse_response(
                            response, valid_answers)
                        result = {
                            "qid": qid,
                            "llm_pred": answer,
                            "llm_confidence": round(confidence, 4),
                            "fallback": False,
                        }
                    except Exception as e:
                        result = {
                            "qid": qid,
                            "llm_pred": rec.get("xgb_pred", ""),
                            "llm_confidence": 0.0,
                            "fallback": True,
                            "error": str(e)[:200],
                        }

                f.write(json.dumps(result) + "\n")
                f.flush()
                llm_cache[qid] = result
                wrote += 1

                if wrote % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"[hybrid] {wrote}/{n_to_score} scored  "
                          f"({wrote/max(elapsed,1)*60:.0f}/min)")

        print(f"[hybrid] LLM scoring done: {wrote} new  "
              f"({(time.time()-t0)/60:.1f} min)")

    # ── Step 2: Sweep thresholds ──
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70]

    pilot_map = {r["qid"]: r for r in pilot}
    c2_qids = set(r["qid"] for r in pilot if r["bucket"] == "C2")
    d_qids  = set(r["qid"] for r in pilot if r["bucket"] == "D")

    print(f"\n[hybrid] Sweeping {len(thresholds)} thresholds...")
    print(f"  C2 questions: {len(c2_qids)}  D questions: {len(d_qids)}")

    W = 78
    print()
    print("=" * W)
    print(f"  {'Threshold':<10} {'Sent to LLM':>12} {'C2 Recov':>10} "
          f"{'D Lost':>8} {'Net':>6} {'Overall EM':>11} {'EM Rate':>8}")
    print("-" * W)

    results = []

    for tau in thresholds:
        total_em = 0
        c2_rec = 0
        d_lost = 0
        n_sent_llm = 0
        n = 0

        for rec in pilot:
            qid = rec["qid"]
            gold = rec["gold"]
            xgb_pred = rec["xgb_pred"]
            xgb_conf = max(rec.get("xgb_probs", [0.0]))
            llm_result = llm_cache.get(qid, {})
            llm_pred = llm_result.get("llm_pred", xgb_pred)

            # Hybrid decision
            if xgb_conf >= tau:
                final_pred = xgb_pred
                used_llm = False
            else:
                final_pred = llm_pred
                used_llm = True
                n_sent_llm += 1

            correct = em(final_pred, gold)
            total_em += correct
            n += 1

            if qid in c2_qids and correct:
                c2_rec += 1
            if qid in d_qids and not correct:
                d_lost += 1

        net = c2_rec - d_lost
        em_rate = total_em / max(n, 1)

        marker = ""
        if net == max(r["net"] for r in results) if results else True:
            marker = " ★"

        results.append({
            "tau": tau,
            "n_sent_llm": n_sent_llm,
            "c2_recovered": c2_rec,
            "d_lost": d_lost,
            "net": net,
            "total_em": total_em,
            "em_rate": round(em_rate, 4),
        })

        print(f"  τ={tau:<8.2f} {n_sent_llm:>12} {c2_rec:>10} "
              f"{d_lost:>8} {net:>+6} {total_em:>11} {em_rate:>8.1%}")

    # Find best threshold by net gain
    best = max(results, key=lambda r: r["net"])
    # Among ties, pick the one with highest EM
    ties = [r for r in results if r["net"] == best["net"]]
    best = max(ties, key=lambda r: r["em_rate"])

    print("-" * W)
    print(f"\n  Best threshold: τ = {best['tau']}")
    print(f"  Sent to LLM: {best['n_sent_llm']}/{len(pilot)} "
          f"({best['n_sent_llm']/len(pilot):.0%})")
    print(f"  C2 recovered: {best['c2_recovered']}/{len(c2_qids)}")
    print(f"  D lost: {best['d_lost']}/{len(d_qids)}")
    print(f"  Net gain: {best['net']:+d}")
    print(f"  Overall EM: {best['total_em']}/{len(pilot)} ({best['em_rate']:.1%})")

    # XGB-only baseline
    xgb_em = sum(em(r["xgb_pred"], r["gold"]) for r in pilot)
    print(f"\n  XGB-only baseline: {xgb_em}/{len(pilot)} "
          f"({xgb_em/len(pilot):.1%})")
    print(f"  Hybrid improvement: {best['total_em'] - xgb_em:+d} questions "
          f"({best['em_rate'] - xgb_em/len(pilot):+.1%})")

    # ── Save ──
    summary = {
        "n_pilot": len(pilot),
        "n_c2": len(c2_qids),
        "n_d": len(d_qids),
        "xgb_baseline_em": xgb_em,
        "best_threshold": best["tau"],
        "best_result": best,
        "all_results": results,
    }
    out_path = os.path.join(args.out_dir, "hybrid_results.json")
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"\n  Results saved to {out_path}")
    print("=" * W)


if __name__ == "__main__":
    main()