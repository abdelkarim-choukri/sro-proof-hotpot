#!/usr/bin/env python3
"""
exp4_topK_sweep.py — Sweep pre-filter sizes: K=2,3,5,7,all

For each K, builds prompt with top-K candidates by XGB confidence,
runs LLM, then sweeps hybrid thresholds. One run, all comparisons.

Usage:
    python3 tools/exp4_topK_sweep.py \
        --pilot         exp4_7b/pilot/pilot_questions.jsonl \
        --out_dir       exp4_7b/pilot/topK_sweep \
        --llm_base_url  http://127.0.0.1:8000/v1 \
        --llm_model_id  /var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct
"""

import argparse, collections, json, os, re, string, time
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
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": float(temperature), "max_tokens": int(max_tokens)}
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}") from e
    choices = body.get("choices", [])
    return choices[0]["message"]["content"] if choices else ""

def retry_call(fn, retries=2, sleep_s=2.0):
    for attempt in range(retries + 1):
        try: return fn()
        except Exception as e:
            if attempt == retries: raise
            time.sleep(sleep_s * (attempt + 1))

def build_topK_prompt(rec, K):
    """Top-K prompt. K=None means all valid candidates."""
    cands = rec["candidates"]
    xgb_probs = rec.get("xgb_probs", [])
    hop1_text = rec.get("hop1_text") or "(not available)"
    hop2_text = rec.get("hop2_text") or "(not available)"
    question = rec.get("question", "")

    indexed = [(i, xgb_probs[i] if i < len(xgb_probs) else 0.0)
               for i in range(len(cands))]
    indexed.sort(key=lambda x: -x[1])

    selected = []
    for orig_i, prob in indexed:
        c = cands[orig_i]
        ans = c.get("answer", "").strip()
        if not ans or ans.lower() in ("unknown", "unk") or len(ans) > 120:
            continue
        selected.append((orig_i, c, prob))
        if K is not None and len(selected) >= K:
            break

    if not selected:
        return None, []

    cand_block = []
    for idx, (orig_i, c, prob) in enumerate(selected):
        letter = chr(65 + idx)
        h1 = c.get("nli_hop1", 0) or 0
        h2 = c.get("nli_hop2", 0) or 0
        bal = abs(h1 - h2)
        cand_block.append(
            f"  {letter}) \"{c['answer']}\"\n"
            f"     Hop1 entailment: {h1:.3f}  |  Hop2 entailment: {h2:.3f}  |  "
            f"Balance: {bal:.3f}  |  Verifier confidence: {prob:.3f}")

    k_desc = f"top {K}" if K else "all valid"
    prompt = f"""You are choosing the best answer to a multi-hop question from {len(selected)} candidates.
A previous verifier ranked these as the {k_desc} candidates.

EVIDENCE:
Document 1 (bridge): {hop1_text}

Document 2 (answer): {hop2_text}

QUESTION: {question}

CANDIDATES (ranked by verifier confidence):
{chr(10).join(cand_block)}

Think step by step:
1. Read both documents carefully.
2. For each candidate, check: does Document 1 connect to it? Does Document 2 support it?
3. A good answer needs support from BOTH documents, not just one.
4. The verifier's ranking may be wrong — choose based on evidence, not ranking.

After your reasoning, output EXACTLY two lines at the end:
ANSWER: [your chosen answer, copied exactly from the candidates]
CONFIDENCE: [a number between 0.0 and 1.0]"""

    return prompt, selected

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
            except: confidence = 0.5
    if answer and valid_answers:
        best_match, best_f1 = None, 0
        for va in valid_answers:
            f = f1_score(answer, va)
            if f > best_f1: best_f1 = f; best_match = va
        if best_match and best_f1 > 0.5: answer = best_match
    return answer, confidence

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--llm_base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--llm_model_id", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pilot = [json.loads(line) for line in open(args.pilot)]
    pilot_map = {r["qid"]: r for r in pilot}
    c2_qids = set(r["qid"] for r in pilot if r["bucket"] == "C2")
    d_qids = set(r["qid"] for r in pilot if r["bucket"] == "D")
    xgb_em_total = sum(em(r["xgb_pred"], r["gold"]) for r in pilot)

    print(f"[topK] Pilot: {len(pilot)} (C2={len(c2_qids)}  D={len(d_qids)})")
    print(f"[topK] XGB baseline: {xgb_em_total}/{len(pilot)} ({xgb_em_total/len(pilot):.1%})")

    # ── Reachability for each K ──
    K_values = [2, 3, 5, 7, None]  # None = all
    print(f"\n  Reachability (correct answer in top-K for C2 questions):")
    for K in K_values:
        reachable = 0
        for rec in pilot:
            if rec["bucket"] != "C2": continue
            cands = rec["candidates"]
            xgb_probs = rec.get("xgb_probs", [])
            gold = rec["gold"]
            indexed = sorted([(i, xgb_probs[i] if i < len(xgb_probs) else 0.0)
                              for i in range(len(cands))], key=lambda x: -x[1])
            sel = []
            for orig_i, prob in indexed:
                ans = cands[orig_i].get("answer", "").strip()
                if not ans or ans.lower() in ("unknown", "unk") or len(ans) > 120:
                    continue
                sel.append(ans)
                if K is not None and len(sel) >= K: break
            if any(em(a, gold) for a in sel): reachable += 1
        label = f"K={K}" if K else "K=all"
        print(f"    {label:<8}  {reachable}/{len(c2_qids)}  ({reachable/len(c2_qids):.1%})")

    # ── Score each K ──
    all_results = {}
    for K in K_values:
        label = f"top{K}" if K else "all"
        cache_path = os.path.join(args.out_dir, f"cache_{label}.jsonl")
        cache = {}
        if os.path.exists(cache_path):
            for line in open(cache_path):
                r = json.loads(line)
                cache[r["qid"]] = r

        n_to_score = sum(1 for r in pilot if r["qid"] not in cache)
        print(f"\n[topK] K={K or 'all'}: {len(cache)} cached, {n_to_score} to score")

        if n_to_score > 0:
            t0 = time.time()
            wrote = 0
            with open(cache_path, "a") as f:
                for rec in pilot:
                    qid = rec["qid"]
                    if qid in cache: continue
                    valid_answers = [c["answer"] for c in rec["candidates"]
                                    if c.get("answer", "").strip()]
                    prompt, sel = build_topK_prompt(rec, K)
                    if prompt is None:
                        result = {"qid": qid, "llm_pred": rec.get("xgb_pred", ""),
                                  "llm_confidence": 0.0, "fallback": True}
                    else:
                        try:
                            response = retry_call(lambda: call_llm(
                                args.llm_base_url, args.llm_model_id,
                                prompt, args.temperature, args.max_new_tokens, args.timeout))
                            answer, confidence = parse_response(response, valid_answers)
                            result = {"qid": qid, "llm_pred": answer,
                                      "llm_confidence": round(confidence, 4), "fallback": False}
                        except:
                            result = {"qid": qid, "llm_pred": rec.get("xgb_pred", ""),
                                      "llm_confidence": 0.0, "fallback": True}
                    f.write(json.dumps(result) + "\n"); f.flush()
                    cache[qid] = result; wrote += 1
                    if wrote % 100 == 0:
                        print(f"    {wrote}/{n_to_score}  ({wrote/(time.time()-t0)*60:.0f}/min)")
            print(f"    Done: {wrote} in {(time.time()-t0)/60:.1f} min")

        all_results[label] = cache

    # ══════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════
    W = 82

    # ── Standalone ──
    print("\n" + "=" * W)
    print("  STANDALONE RESULTS")
    print("=" * W)
    print(f"  {'Method':<20} {'EM':>6} {'C2 Rec':>8} {'D Lost':>7} {'Net':>6} {'D Ret%':>7} {'Rate':>7}")
    print("  " + "-" * (W - 2))
    print(f"  {'XGB baseline':<20} {xgb_em_total:>6} {'0':>8} {'0':>7} {'0':>6} {'100%':>7} {xgb_em_total/len(pilot):>7.1%}")

    standalone = {}
    for K in K_values:
        label = f"top{K}" if K else "all"
        cache = all_results[label]
        te = c2r = dl = 0
        for rec in pilot:
            qid = rec["qid"]
            gold = rec["gold"]
            pred = cache[qid]["llm_pred"]
            c = em(pred, gold); te += c
            if qid in c2_qids and c: c2r += 1
            if qid in d_qids and not c: dl += 1
        net = c2r - dl
        dr = (len(d_qids) - dl) / len(d_qids)
        name = f"LLM K={K}" if K else "LLM K=all"
        standalone[label] = {"em": te, "c2r": c2r, "dl": dl, "net": net}
        marker = " ★" if net == max(s["net"] for s in standalone.values()) else ""
        print(f"  {name:<20} {te:>6} {c2r:>8} {dl:>7} {net:>+6} {dr:>7.1%} {te/len(pilot):>7.1%}{marker}")

    # ── Hybrid sweep for each K ──
    print("\n" + "=" * W)
    print("  HYBRID SWEEP (best threshold per K)")
    print("=" * W)
    print(f"  {'K':<8} {'Best τ':>7} {'→LLM':>6} {'C2 Rec':>8} {'D Lost':>7} {'Net':>6} {'EM':>6} {'Rate':>7}")
    print("  " + "-" * (W - 2))

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    hybrid_bests = {}

    for K in K_values:
        label = f"top{K}" if K else "all"
        cache = all_results[label]
        best = None
        for tau in thresholds:
            te = c2r = dl = n_llm = 0
            for rec in pilot:
                qid = rec["qid"]
                gold = rec["gold"]
                xgb_pred = rec["xgb_pred"]
                xgb_conf = max(rec.get("xgb_probs", [0.0]))
                if xgb_conf >= tau:
                    final = xgb_pred
                else:
                    final = cache[qid]["llm_pred"]; n_llm += 1
                c = em(final, gold); te += c
                if qid in c2_qids and c: c2r += 1
                if qid in d_qids and not c: dl += 1
            net = c2r - dl
            rate = te / len(pilot)
            if best is None or net > best["net"] or (net == best["net"] and rate > best["rate"]):
                best = {"tau": tau, "n_llm": n_llm, "c2r": c2r, "dl": dl,
                        "net": net, "em": te, "rate": rate}
        hybrid_bests[label] = best
        name = f"K={K}" if K else "K=all"
        print(f"  {name:<8} {best['tau']:>7.2f} {best['n_llm']:>6} {best['c2r']:>8} "
              f"{best['dl']:>7} {best['net']:>+6} {best['em']:>6} {best['rate']:>7.1%}")

    # ── Winner ──
    best_label = max(hybrid_bests, key=lambda k: hybrid_bests[k]["net"])
    b = hybrid_bests[best_label]
    print("  " + "-" * (W - 2))
    print(f"  Best hybrid: {best_label}  τ={b['tau']}  net={b['net']:+d}  "
          f"EM={b['em']}/{len(pilot)} ({b['rate']:.1%})")
    print(f"  XGB baseline: {xgb_em_total}/{len(pilot)} ({xgb_em_total/len(pilot):.1%})")

    # Save
    summary = {"standalone": standalone, "hybrid_bests": hybrid_bests,
               "xgb_baseline": xgb_em_total}
    json.dump(summary, open(os.path.join(args.out_dir, "topK_sweep.json"), "w"), indent=2)
    print(f"\n  Saved to {args.out_dir}/topK_sweep.json")
    print("=" * W)

if __name__ == "__main__":
    main()