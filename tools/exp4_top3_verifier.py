#!/usr/bin/env python3
"""
exp4_top3_verifier.py — LLM Verifier with Top-3 XGB Pre-filter

Tests two things:
  1. Standalone LLM with top 3 candidates by XGB confidence
  2. Hybrid: XGB when confident, LLM with top 3 when uncertain (threshold sweep)

Rationale:
  - Top 2 (Option 2) was too restrictive: 13.8% C2 recovery, correct answer
    often ranked 3rd by XGB and never shown to LLM
  - All candidates (Option 1) was too noisy: 35.1% C2 recovery but lost 35 D
    questions because LLM overthinks with too many options
  - Top 3 is the sweet spot hypothesis: enough to include the correct answer,
    narrow enough to avoid LLM confusion

Usage:
    python3 tools/exp4_top3_verifier.py \
        --pilot         exp4_7b/pilot/pilot_questions.jsonl \
        --out_dir       exp4_7b/pilot/top3 \
        --llm_base_url  http://127.0.0.1:8000/v1 \
        --llm_model_id  /var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct
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

def build_top3_prompt(rec):
    """
    Pre-filter to top 3 candidates by XGB confidence, then show to LLM
    with hop scores and evidence text.
    """
    cands = rec["candidates"]
    xgb_probs = rec.get("xgb_probs", [])
    hop1_text = rec.get("hop1_text") or "(not available)"
    hop2_text = rec.get("hop2_text") or "(not available)"
    question = rec.get("question", "")

    # Rank candidates by XGB probability
    indexed = [(i, xgb_probs[i] if i < len(xgb_probs) else 0.0)
               for i in range(len(cands))]
    indexed.sort(key=lambda x: -x[1])

    # Take top 3 non-bad candidates
    top3 = []
    for orig_i, prob in indexed:
        c = cands[orig_i]
        ans = c.get("answer", "").strip()
        if not ans or ans.lower() in ("unknown", "unk") or len(ans) > 120:
            continue
        top3.append((orig_i, c, prob))
        if len(top3) == 3:
            break

    if not top3:
        return None, []

    cand_block = []
    for idx, (orig_i, c, prob) in enumerate(top3):
        letter = chr(65 + idx)
        h1 = c.get("nli_hop1", 0) or 0
        h2 = c.get("nli_hop2", 0) or 0
        bal = abs(h1 - h2)
        cand_block.append(
            f"  {letter}) \"{c['answer']}\"\n"
            f"     Hop1 entailment: {h1:.3f}  |  Hop2 entailment: {h2:.3f}  |  "
            f"Balance: {bal:.3f}  |  Verifier confidence: {prob:.3f}"
        )

    prompt = f"""You are choosing the best answer to a multi-hop question from 3 candidates.
A previous verifier narrowed it down to these three, ranked by confidence.

EVIDENCE:
Document 1 (bridge): {hop1_text}

Document 2 (answer): {hop2_text}

QUESTION: {question}

TOP 3 CANDIDATES (ranked by verifier confidence):
{chr(10).join(cand_block)}

Think step by step:
1. Read both documents carefully.
2. For each candidate, check: does Document 1 connect to it? Does Document 2 support it?
3. A good answer needs support from BOTH documents — not just one.
4. The verifier's top pick may be wrong if it only considered one hop.

After your reasoning, output EXACTLY two lines at the end:
ANSWER: [your chosen answer, copied exactly from the candidates]
CONFIDENCE: [a number between 0.0 and 1.0]"""

    return prompt, top3


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

    # Fuzzy match
    if answer and valid_answers:
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

    pilot_map = {r["qid"]: r for r in pilot}
    c2_qids = set(r["qid"] for r in pilot if r["bucket"] == "C2")
    d_qids  = set(r["qid"] for r in pilot if r["bucket"] == "D")

    print(f"[top3] Pilot: {len(pilot)} questions "
          f"(C2={len(c2_qids)}  D={len(d_qids)})")

    # ── Check: how often is the correct answer in top 3? ──
    correct_in_top3 = 0
    correct_in_top2 = 0
    for rec in pilot:
        if rec["bucket"] != "C2":
            continue
        cands = rec["candidates"]
        xgb_probs = rec.get("xgb_probs", [])
        gold = rec["gold"]

        indexed = [(i, xgb_probs[i] if i < len(xgb_probs) else 0.0)
                   for i in range(len(cands))]
        indexed.sort(key=lambda x: -x[1])

        top3_answers = []
        top2_answers = []
        count = 0
        for orig_i, prob in indexed:
            ans = cands[orig_i].get("answer", "").strip()
            if not ans or ans.lower() in ("unknown", "unk") or len(ans) > 120:
                continue
            count += 1
            if count <= 2:
                top2_answers.append(ans)
            if count <= 3:
                top3_answers.append(ans)
            if count >= 3:
                break

        if any(em(a, gold) for a in top3_answers):
            correct_in_top3 += 1
        if any(em(a, gold) for a in top2_answers):
            correct_in_top2 += 1

    print(f"[top3] C2 correct answer reachable:")
    print(f"  In top 2: {correct_in_top2}/{len(c2_qids)} "
          f"({correct_in_top2/max(len(c2_qids),1):.1%})")
    print(f"  In top 3: {correct_in_top3}/{len(c2_qids)} "
          f"({correct_in_top3/max(len(c2_qids),1):.1%})")
    print(f"  Gain from 3rd candidate: "
          f"+{correct_in_top3 - correct_in_top2} reachable C2 questions")

    # ── Score all questions with LLM (top-3 prompt) ──
    cache_path = os.path.join(args.out_dir, "llm_top3_cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line)
            cache[r["qid"]] = r
        print(f"[top3] Cache: {len(cache)} already scored")

    n_to_score = sum(1 for r in pilot if r["qid"] not in cache)
    print(f"[top3] Need to score: {n_to_score}")

    if n_to_score > 0:
        t0 = time.time()
        wrote = 0
        with open(cache_path, "a") as f:
            for rec in pilot:
                qid = rec["qid"]
                if qid in cache:
                    continue

                cands = rec["candidates"]
                valid_answers = [c["answer"] for c in cands
                                if c.get("answer", "").strip()]

                prompt, top3 = build_top3_prompt(rec)

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
                            "n_shown": len(top3),
                        }
                    except Exception as e:
                        result = {
                            "qid": qid,
                            "llm_pred": rec.get("xgb_pred", ""),
                            "llm_confidence": 0.0,
                            "fallback": True,
                        }

                f.write(json.dumps(result) + "\n")
                f.flush()
                cache[qid] = result
                wrote += 1

                if wrote % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"[top3] {wrote}/{n_to_score}  "
                          f"({wrote/max(elapsed,1)*60:.0f}/min)")

        print(f"[top3] Done: {wrote} scored ({(time.time()-t0)/60:.1f} min)")

    # ── Standalone LLM-top3 results ──
    W = 78
    print()
    print("=" * W)
    print("  STANDALONE LLM-TOP3 RESULTS")
    print("=" * W)

    total_em = c2_rec = d_lost = 0
    n = 0
    for rec in pilot:
        qid = rec["qid"]
        gold = rec["gold"]
        llm_pred = cache[qid]["llm_pred"]
        correct = em(llm_pred, gold)
        total_em += correct
        n += 1
        if qid in c2_qids and correct:
            c2_rec += 1
        if qid in d_qids and not correct:
            d_lost += 1

    xgb_em = sum(em(r["xgb_pred"], r["gold"]) for r in pilot)
    net = c2_rec - d_lost

    print(f"  Overall EM:  {total_em}/{n} ({total_em/n:.1%})")
    print(f"  C2 recovery: {c2_rec}/{len(c2_qids)} ({c2_rec/len(c2_qids):.1%})")
    print(f"  D retention: {len(d_qids)-d_lost}/{len(d_qids)} "
          f"({(len(d_qids)-d_lost)/len(d_qids):.1%})")
    print(f"  D lost:      {d_lost}/{len(d_qids)}")
    print(f"  Net gain:    {net:+d}")

    # ── Compare all standalone options ──
    print()
    print("-" * W)
    print("  STANDALONE COMPARISON (all 300 questions)")
    print("-" * W)

    # Load Option 1 and Option 2 from previous pilot
    opt1_preds, opt2_preds = {}, {}
    opt1_path = os.path.join(os.path.dirname(args.out_dir), "preds_option1.jsonl")
    opt2_path = os.path.join(os.path.dirname(args.out_dir), "preds_option2.jsonl")

    if os.path.exists(opt1_path):
        for line in open(opt1_path):
            r = json.loads(line)
            opt1_preds[r["qid"]] = r
    if os.path.exists(opt2_path):
        for line in open(opt2_path):
            r = json.loads(line)
            opt2_preds[r["qid"]] = r

    def compute_stats(preds_map, label):
        te = cr = dl = 0
        nn = 0
        for rec in pilot:
            qid = rec["qid"]
            gold = rec["gold"]
            if qid in preds_map:
                pred = preds_map[qid].get("pred", "")
            else:
                pred = rec["xgb_pred"]
            c = em(pred, gold)
            te += c
            nn += 1
            if qid in c2_qids and c: cr += 1
            if qid in d_qids and not c: dl += 1
        return te, cr, dl

    print(f"  {'Method':<30} {'EM':>6} {'C2 Rec':>8} {'D Lost':>7} {'Net':>6}")
    print(f"  " + "-" * (W - 2))
    print(f"  {'XGB baseline':<30} {xgb_em:>6} {'0':>8} {'0':>7} {'0':>6}")

    if opt1_preds:
        te, cr, dl = compute_stats(opt1_preds, "opt1")
        print(f"  {'Option 1 (all cands)':<30} {te:>6} {cr:>8} {dl:>7} "
              f"{cr-dl:>+6}")
    if opt2_preds:
        te, cr, dl = compute_stats(opt2_preds, "opt2")
        print(f"  {'Option 2 (top 2)':<30} {te:>6} {cr:>8} {dl:>7} "
              f"{cr-dl:>+6}")
    print(f"  {'Option 3 (top 3) ★':<30} {total_em:>6} {c2_rec:>8} "
          f"{d_lost:>7} {net:>+6}")

    # ── Hybrid threshold sweep with top-3 LLM ──
    print()
    print("=" * W)
    print("  HYBRID SWEEP (XGB when confident, LLM-top3 when uncertain)")
    print("=" * W)

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70]

    print(f"  {'τ':<6} {'→LLM':>6} {'C2 Rec':>8} {'D Lost':>7} "
          f"{'Net':>6} {'EM':>6} {'Rate':>7}")
    print("  " + "-" * (W - 2))

    results = []
    for tau in thresholds:
        te = c2r = dl = n_llm = 0
        for rec in pilot:
            qid = rec["qid"]
            gold = rec["gold"]
            xgb_pred = rec["xgb_pred"]
            xgb_conf = max(rec.get("xgb_probs", [0.0]))
            llm_pred = cache[qid]["llm_pred"]

            if xgb_conf >= tau:
                final = xgb_pred
            else:
                final = llm_pred
                n_llm += 1

            correct = em(final, gold)
            te += correct
            if qid in c2_qids and correct: c2r += 1
            if qid in d_qids and not correct: dl += 1

        net_val = c2r - dl
        rate = te / len(pilot)
        results.append({"tau": tau, "n_llm": n_llm, "c2r": c2r,
                        "dl": dl, "net": net_val, "em": te, "rate": rate})
        print(f"  {tau:<6.2f} {n_llm:>6} {c2r:>8} {dl:>7} "
              f"{net_val:>+6} {te:>6} {rate:>7.1%}")

    best = max(results, key=lambda r: r["net"])
    ties = [r for r in results if r["net"] == best["net"]]
    best = max(ties, key=lambda r: r["rate"])

    print("  " + "-" * (W - 2))
    print(f"  Best τ={best['tau']}:  LLM calls={best['n_llm']}  "
          f"C2={best['c2r']}  D_lost={best['dl']}  "
          f"net={best['net']:+d}  EM={best['em']}/{len(pilot)} ({best['rate']:.1%})")
    print(f"  XGB baseline: {xgb_em}/{len(pilot)} ({xgb_em/len(pilot):.1%})")
    print(f"  Improvement: {best['em']-xgb_em:+d} ({best['rate']-xgb_em/len(pilot):+.1%})")

    # ── Also compare hybrid-top3 vs hybrid-all (from previous run) ──
    hybrid_all_path = os.path.join(os.path.dirname(args.out_dir),
                                   "hybrid", "hybrid_results.json")
    if os.path.exists(hybrid_all_path):
        hybrid_all = json.load(open(hybrid_all_path))
        ha_best = hybrid_all["best_result"]
        print()
        print("  " + "=" * (W - 2))
        print(f"  HYBRID COMPARISON (best threshold each)")
        print(f"  {'Variant':<35} {'τ':>5} {'Net':>6} {'EM':>6} {'Rate':>7}")
        print(f"  " + "-" * (W - 2))
        print(f"  {'Hybrid + all cands (Option 1)':<35} "
              f"{ha_best['tau']:>5.2f} {ha_best['net']:>+6} "
              f"{ha_best['total_em']:>6} {ha_best['em_rate']:>7.1%}")
        print(f"  {'Hybrid + top 3 (Option 3) ★':<35} "
              f"{best['tau']:>5.2f} {best['net']:>+6} "
              f"{best['em']:>6} {best['rate']:>7.1%}")

    # Save
    summary = {
        "standalone_top3": {
            "em": total_em, "c2_recovered": c2_rec,
            "d_lost": d_lost, "net": net,
        },
        "correct_in_top3": correct_in_top3,
        "correct_in_top2": correct_in_top2,
        "hybrid_sweep": results,
        "hybrid_best": best,
        "xgb_baseline_em": xgb_em,
    }
    out_path = os.path.join(args.out_dir, "top3_results.json")
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"\n  Saved to {out_path}")
    print("=" * W)


if __name__ == "__main__":
    main()