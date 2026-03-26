#!/usr/bin/env python3
"""
exp4_guided_verifier.py — LLM Verifier with Guided Attention

Shows ALL candidates but marks the top 3 as "recommended" by the XGB verifier.
The LLM is told to focus on those first but consider all options.

This avoids:
  - Top-2/Top-3 reachability problem (correct answer always visible)
  - All-candidates noise problem (LLM has guidance on where to focus)

Usage:
    python3 tools/exp4_guided_verifier.py \
        --pilot         exp4_7b/pilot/pilot_questions.jsonl \
        --out_dir       exp4_7b/pilot/guided \
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


def build_guided_prompt(rec):
    """
    Show ALL valid candidates, but mark top 3 by XGB as 'recommended'.
    """
    cands = rec["candidates"]
    xgb_probs = rec.get("xgb_probs", [])
    hop1_text = rec.get("hop1_text") or "(not available)"
    hop2_text = rec.get("hop2_text") or "(not available)"
    question = rec.get("question", "")

    # Rank by XGB confidence
    indexed = [(i, xgb_probs[i] if i < len(xgb_probs) else 0.0)
               for i in range(len(cands))]
    indexed.sort(key=lambda x: -x[1])

    # Identify top 3 non-bad indices
    top3_set = set()
    count = 0
    for orig_i, prob in indexed:
        c = cands[orig_i]
        ans = c.get("answer", "").strip()
        if not ans or ans.lower() in ("unknown", "unk") or len(ans) > 120:
            continue
        top3_set.add(orig_i)
        count += 1
        if count >= 3:
            break

    # Build candidate list (all valid, with markers)
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
        marker = "  ★ RECOMMENDED" if orig_i in top3_set else ""
        cand_block.append(
            f"  {letter}) \"{c['answer']}\"{marker}\n"
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

Candidates marked ★ RECOMMENDED were ranked highest by a previous verifier.
Start by examining those first, but DO NOT ignore the other candidates — 
the previous verifier sometimes makes mistakes. If a non-recommended candidate 
is clearly better supported by BOTH documents, choose it instead.

Think step by step:
1. First check the ★ RECOMMENDED candidates — are any strongly supported by both hops?
2. Then check the remaining candidates — does any have better evidence support?
3. A good answer needs support from BOTH documents, not just one.
4. Choose the answer with the strongest combined evidence, regardless of recommendation.

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

    pilot = []
    for line in open(args.pilot):
        pilot.append(json.loads(line))

    pilot_map = {r["qid"]: r for r in pilot}
    c2_qids = set(r["qid"] for r in pilot if r["bucket"] == "C2")
    d_qids  = set(r["qid"] for r in pilot if r["bucket"] == "D")

    print(f"[guided] Pilot: {len(pilot)} (C2={len(c2_qids)}  D={len(d_qids)})")

    # ── Score all questions ──
    cache_path = os.path.join(args.out_dir, "llm_guided_cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line)
            cache[r["qid"]] = r
        print(f"[guided] Cache: {len(cache)} already scored")

    n_to_score = sum(1 for r in pilot if r["qid"] not in cache)
    print(f"[guided] Need to score: {n_to_score}")

    if n_to_score > 0:
        t0 = time.time()
        wrote = 0
        with open(cache_path, "a") as f:
            for rec in pilot:
                qid = rec["qid"]
                if qid in cache:
                    continue

                valid_answers = [c["answer"] for c in rec["candidates"]
                                if c.get("answer", "").strip()]
                prompt, valid_cands = build_guided_prompt(rec)

                if prompt is None:
                    result = {"qid": qid, "llm_pred": rec.get("xgb_pred", ""),
                              "llm_confidence": 0.0, "fallback": True}
                else:
                    try:
                        response = retry_call(
                            lambda: call_llm(
                                args.llm_base_url, args.llm_model_id,
                                prompt, args.temperature,
                                args.max_new_tokens, args.timeout))
                        answer, confidence = parse_response(response, valid_answers)
                        result = {"qid": qid, "llm_pred": answer,
                                  "llm_confidence": round(confidence, 4),
                                  "fallback": False}
                    except Exception as e:
                        result = {"qid": qid, "llm_pred": rec.get("xgb_pred", ""),
                                  "llm_confidence": 0.0, "fallback": True}

                f.write(json.dumps(result) + "\n")
                f.flush()
                cache[qid] = result
                wrote += 1

                if wrote % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"[guided] {wrote}/{n_to_score}  "
                          f"({wrote/max(elapsed,1)*60:.0f}/min)")

        print(f"[guided] Done: {wrote} scored ({(time.time()-t0)/60:.1f} min)")

    # ── Standalone guided results ──
    W = 78
    total_em = c2_rec = d_lost = 0
    n = 0
    for rec in pilot:
        qid = rec["qid"]
        gold = rec["gold"]
        llm_pred = cache[qid]["llm_pred"]
        correct = em(llm_pred, gold)
        total_em += correct
        n += 1
        if qid in c2_qids and correct: c2_rec += 1
        if qid in d_qids and not correct: d_lost += 1

    xgb_em = sum(em(r["xgb_pred"], r["gold"]) for r in pilot)
    net = c2_rec - d_lost

    print()
    print("=" * W)
    print("  GUIDED ATTENTION RESULTS (standalone)")
    print("=" * W)
    print(f"  Overall EM:  {total_em}/{n} ({total_em/n:.1%})")
    print(f"  C2 recovery: {c2_rec}/{len(c2_qids)} ({c2_rec/len(c2_qids):.1%})")
    print(f"  D retained:  {len(d_qids)-d_lost}/{len(d_qids)} "
          f"({(len(d_qids)-d_lost)/len(d_qids):.1%})")
    print(f"  D lost:      {d_lost}/{len(d_qids)}")
    print(f"  Net gain:    {net:+d}")

    # ── Compare all methods ──
    print()
    print("-" * W)
    print("  ALL STANDALONE METHODS COMPARISON")
    print("-" * W)
    print(f"  {'Method':<35} {'EM':>6} {'C2 Rec':>8} {'D Lost':>7} {'Net':>6} {'D Ret%':>7}")
    print(f"  " + "-" * (W - 2))
    print(f"  {'XGB baseline':<35} {xgb_em:>6} {'0':>8} {'0':>7} {'0':>6} {'100%':>7}")

    # Load previous results
    for label, path in [
        ("Option 1 (all cands)", "preds_option1.jsonl"),
        ("Option 2 (top 2)", "preds_option2.jsonl"),
    ]:
        fpath = os.path.join(os.path.dirname(args.out_dir), path)
        if os.path.exists(fpath):
            preds = {}
            for line in open(fpath):
                r = json.loads(line)
                preds[r["qid"]] = r
            te = cr = dl = 0
            for rec in pilot:
                qid = rec["qid"]
                gold = rec["gold"]
                pred = preds.get(qid, {}).get("pred", rec["xgb_pred"])
                c = em(pred, gold)
                te += c
                if qid in c2_qids and c: cr += 1
                if qid in d_qids and not c: dl += 1
            dr = (len(d_qids)-dl)/len(d_qids)
            print(f"  {label:<35} {te:>6} {cr:>8} {dl:>7} {cr-dl:>+6} {dr:>7.1%}")

    # Top 3
    top3_path = os.path.join(os.path.dirname(args.out_dir), "top3", "llm_top3_cache.jsonl")
    if os.path.exists(top3_path):
        preds = {}
        for line in open(top3_path):
            r = json.loads(line)
            preds[r["qid"]] = r
        te = cr = dl = 0
        for rec in pilot:
            qid = rec["qid"]
            gold = rec["gold"]
            pred = preds.get(qid, {}).get("llm_pred", rec["xgb_pred"])
            c = em(pred, gold)
            te += c
            if qid in c2_qids and c: cr += 1
            if qid in d_qids and not c: dl += 1
        dr = (len(d_qids)-dl)/len(d_qids)
        print(f"  {'Option 3 (top 3)':<35} {te:>6} {cr:>8} {dl:>7} {cr-dl:>+6} {dr:>7.1%}")

    # Guided (this run)
    dr = (len(d_qids)-d_lost)/len(d_qids)
    print(f"  {'Guided attention ★':<35} {total_em:>6} {c2_rec:>8} {d_lost:>7} {net:>+6} {dr:>7.1%}")

    # ── Hybrid sweep with guided ──
    print()
    print("=" * W)
    print("  HYBRID SWEEP (XGB when confident, guided LLM when uncertain)")
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
    print(f"  Best τ={best['tau']}:  net={best['net']:+d}  "
          f"EM={best['em']}/{len(pilot)} ({best['rate']:.1%})")
    print(f"  XGB baseline: {xgb_em}/{len(pilot)} ({xgb_em/len(pilot):.1%})")

    # Compare with previous hybrid results
    for label, path in [
        ("Hybrid all-cands", "hybrid/hybrid_results.json"),
        ("Hybrid top-3", "top3/top3_results.json"),
    ]:
        fpath = os.path.join(os.path.dirname(args.out_dir), path)
        if os.path.exists(fpath):
            prev = json.load(open(fpath))
            pb = prev.get("hybrid_best", prev.get("best_result", {}))
            print(f"  {label}: τ={pb.get('tau',0)}  net={pb.get('net',0):+d}  "
                  f"EM={pb.get('em', pb.get('total_em',0))}")

    print(f"  Guided hybrid: τ={best['tau']}  net={best['net']:+d}  EM={best['em']}")

    # Save
    summary = {
        "standalone": {"em": total_em, "c2_recovered": c2_rec,
                       "d_lost": d_lost, "net": net},
        "hybrid_sweep": results,
        "hybrid_best": best,
        "xgb_baseline_em": xgb_em,
    }
    out_path = os.path.join(args.out_dir, "guided_results.json")
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"\n  Saved to {out_path}")
    print("=" * W)


if __name__ == "__main__":
    main()