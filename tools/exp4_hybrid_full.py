#!/usr/bin/env python3
"""
exp4_hybrid_full.py — Full-scale Hybrid XGB + LLM Verifier

For all 7405 questions:
  - If XGB confidence ≥ threshold → keep XGB prediction
  - If XGB confidence < threshold → run LLM with all valid candidates + hop scores

Uses the Option 1 prompt (all candidates with scores) since the hybrid
threshold already protects high-confidence D questions.

Phase 1: Prepare context for all questions (CPU, ~1 min)
Phase 2: Run LLM on uncertain questions (GPU via vLLM)
Phase 3: Merge predictions and compute metrics

Usage:
    python3 tools/exp4_hybrid_full.py \
        --chain_preds   exp4_7b/preds/dev_chain_verifier_mean_preds.jsonl \
        --hop_scores    exp4_7b/preds/dev_hop_scores.jsonl \
        --evidence      exp1b/evidence/dev_K100_chains.jsonl \
        --gold          data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
        --oracle_perqid exp4_7b/metrics/oracle_M5_dev_perqid.jsonl \
        --out_dir       exp4_7b/hybrid_full \
        --threshold     0.55 \
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

def is_bad(ans):
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

def get_hop_texts(chains):
    if not chains: return None, None
    top = chains[0]
    hops = top.get("hops", [])
    if len(hops) < 2: return None, None
    h1 = hops[0]
    h2 = hops[1]
    return (f"{h1.get('title','')}: {h1.get('text','')}".strip(),
            f"{h2.get('title','')}: {h2.get('text','')}".strip())


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

def build_prompt(hop1_text, hop2_text, question, cands):
    """Option 1 style: all valid candidates + hop scores."""
    valid = []
    for i, c in enumerate(cands):
        ans = c.get("answer", "").strip()
        if not ans or ans.lower() in ("unknown", "unk") or len(ans) > 120:
            continue
        valid.append((i, c))

    if not valid:
        return None, []

    cand_block = []
    for idx, (orig_i, c) in enumerate(valid):
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
Document 1 (bridge): {hop1_text or '(not available)'}

Document 2 (answer): {hop2_text or '(not available)'}

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

    return prompt, valid


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


# ─────────────────────────── main ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain_preds",   required=True)
    ap.add_argument("--hop_scores",    required=True)
    ap.add_argument("--evidence",      required=True)
    ap.add_argument("--gold",          required=True)
    ap.add_argument("--oracle_perqid", required=True)
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--threshold",     type=float, default=0.55)
    ap.add_argument("--llm_base_url",  default="http://127.0.0.1:8000/v1")
    ap.add_argument("--llm_model_id",  required=True)
    ap.add_argument("--temperature",   type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--timeout",       type=int, default=120)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tau = args.threshold

    # ── Phase 1: Load everything ──
    print(f"[hybrid-full] Loading data...")

    # Gold
    gold_map, qtype_map = {}, {}
    for ex in json.load(open(args.gold)):
        qid = str(ex["_id"])
        gold_map[qid] = ex["answer"]
        qtype_map[qid] = ex.get("type", "bridge")

    # Feasible
    feasible = set()
    ev_map = {}
    for line in open(args.evidence):
        r = json.loads(line)
        qid = str(r["qid"])
        chains = r.get("evidence", {}).get("chains") or r.get("chains", [])
        ev_map[qid] = chains
        if r.get("flags", {}).get("doc_recall_at_k", False):
            feasible.add(qid)

    # Oracle
    oracle_map = {}
    for line in open(args.oracle_perqid):
        r = json.loads(line)
        oracle_map[str(r["qid"])] = int(r.get("best_em", 0))

    # XGB preds
    xgb_map = {}
    for line in open(args.chain_preds):
        r = json.loads(line)
        xgb_map[str(r["qid"])] = r

    # Hop scores
    hop_map = {}
    for line in open(args.hop_scores):
        r = json.loads(line)
        hop_map[str(r["qid"])] = r

    all_qids = sorted(set(xgb_map) & set(gold_map))
    print(f"[hybrid-full] Total questions: {len(all_qids)}")

    # ── Count how many need LLM ──
    need_llm = []
    keep_xgb = []
    for qid in all_qids:
        xgb = xgb_map[qid]
        max_conf = max(xgb.get("probs", [0.0]))
        if max_conf >= tau:
            keep_xgb.append(qid)
        else:
            need_llm.append(qid)

    print(f"[hybrid-full] Threshold τ = {tau}")
    print(f"[hybrid-full] Keep XGB (conf ≥ {tau}): {len(keep_xgb)}")
    print(f"[hybrid-full] Send to LLM (conf < {tau}): {len(need_llm)}")

    # ── Phase 2: Run LLM on uncertain questions ──
    cache_path = os.path.join(args.out_dir, "llm_cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line)
            cache[r["qid"]] = r
        print(f"[hybrid-full] LLM cache: {len(cache)} already scored")

    n_to_score = sum(1 for qid in need_llm if qid not in cache)
    print(f"[hybrid-full] Need to score: {n_to_score} questions")

    if n_to_score > 0:
        t0 = time.time()
        wrote = errors = 0

        with open(cache_path, "a") as f:
            for qid in need_llm:
                if qid in cache:
                    continue

                hop = hop_map.get(qid, {})
                cands = hop.get("candidates", [])
                question = hop.get("question", "")
                chains = ev_map.get(qid, [])
                hop1_text, hop2_text = get_hop_texts(chains)

                valid_answers = [c["answer"] for c in cands
                                if c.get("answer", "").strip()]

                prompt, valid_cands = build_prompt(
                    hop1_text, hop2_text, question, cands)

                if prompt is None:
                    result = {
                        "qid": qid,
                        "llm_pred": xgb_map[qid].get("pred", ""),
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
                        errors += 1
                        result = {
                            "qid": qid,
                            "llm_pred": xgb_map[qid].get("pred", ""),
                            "llm_confidence": 0.0,
                            "fallback": True,
                        }

                f.write(json.dumps(result) + "\n")
                f.flush()
                cache[qid] = result
                wrote += 1

                if wrote % 200 == 0:
                    elapsed = time.time() - t0
                    pct = wrote / n_to_score * 100
                    rate = wrote / max(elapsed, 1) * 60
                    eta = (n_to_score - wrote) / max(rate / 60, 0.01)
                    print(f"[hybrid-full] {wrote}/{n_to_score} ({pct:.1f}%)  "
                          f"{rate:.0f}/min  ETA {eta:.1f} min  "
                          f"errors={errors}")

        elapsed = time.time() - t0
        print(f"[hybrid-full] LLM scoring done: {wrote} questions  "
              f"{errors} errors  ({elapsed/60:.1f} min)")

    # ── Phase 3: Merge and compute metrics ──
    print(f"\n[hybrid-full] Computing metrics...")

    # Build final predictions
    total_em = total_f1 = 0
    n = 0
    n_llm_used = 0

    # Per-bucket counters
    buckets = {"A": 0, "B": 0, "C1": 0, "C2": 0, "D": 0}
    bucket_correct = {"A": 0, "B": 0, "C1": 0, "C2": 0, "D": 0}

    # Per-split
    splits = {
        "overall": {"em": 0, "f1": 0, "n": 0},
        "feasible": {"em": 0, "f1": 0, "n": 0},
        "bridge": {"em": 0, "f1": 0, "n": 0},
        "comparison": {"em": 0, "f1": 0, "n": 0},
    }

    # XGB-only baseline for comparison
    xgb_only_em = 0

    preds_out = []

    for qid in all_qids:
        gold = gold_map[qid]
        xgb = xgb_map[qid]
        xgb_pred = xgb.get("pred", "")
        xgb_conf = max(xgb.get("probs", [0.0]))
        is_feas = qid in feasible
        oracle_em_flag = oracle_map.get(qid, 0)
        qtype = qtype_map.get(qid, "bridge")

        # Determine bucket
        if not is_feas:
            bucket = "A"
        elif oracle_em_flag == 0:
            bucket = "B"
        elif em(xgb_pred, gold):
            bucket = "D"
        elif is_bad(xgb_pred):
            bucket = "C1"
        else:
            bucket = "C2"
        buckets[bucket] += 1

        # Hybrid decision
        if xgb_conf >= tau:
            final_pred = xgb_pred
            used_llm = False
        else:
            llm_result = cache.get(qid, {})
            final_pred = llm_result.get("llm_pred", xgb_pred)
            used_llm = True
            n_llm_used += 1

        correct = em(final_pred, gold)
        f1 = f1_score(final_pred, gold)
        xgb_correct = em(xgb_pred, gold)

        total_em += correct
        total_f1 += f1
        xgb_only_em += xgb_correct
        n += 1

        bucket_correct[bucket] += correct

        # Splits
        splits["overall"]["em"] += correct
        splits["overall"]["f1"] += f1
        splits["overall"]["n"] += 1
        if is_feas:
            splits["feasible"]["em"] += correct
            splits["feasible"]["f1"] += f1
            splits["feasible"]["n"] += 1
        if qtype == "bridge":
            splits["bridge"]["em"] += correct
            splits["bridge"]["f1"] += f1
            splits["bridge"]["n"] += 1
        else:
            splits["comparison"]["em"] += correct
            splits["comparison"]["f1"] += f1
            splits["comparison"]["n"] += 1

        preds_out.append({
            "qid": qid,
            "pred": final_pred,
            "used_llm": used_llm,
            "xgb_pred": xgb_pred,
            "xgb_conf": round(xgb_conf, 4),
            "bucket": bucket,
        })

    # Save predictions
    preds_path = os.path.join(args.out_dir, "hybrid_preds.jsonl")
    with open(preds_path, "w") as f:
        for p in preds_out:
            f.write(json.dumps(p) + "\n")

    # ── Print results ──
    W = 76
    print()
    print("=" * W)
    print(f"  HYBRID VERIFIER RESULTS (τ = {tau}, n = {n})")
    print("=" * W)

    print(f"\n  LLM calls: {n_llm_used}/{n} ({n_llm_used/n:.1%})")
    print(f"  XGB kept:  {n - n_llm_used}/{n} ({(n-n_llm_used)/n:.1%})")

    # Baselines
    EXP1B_XGB = 0.3409
    EXP3B_XGB = 0.3622
    EXP4_XGB  = xgb_only_em / n

    hybrid_em = total_em / n
    hybrid_f1 = total_f1 / n

    print(f"\n  {'Method':<40} {'EM':>8} {'F1':>8}")
    print("  " + "-" * (W - 2))
    print(f"  {'1.5B M=5 XGB (exp1b)':<40} {EXP1B_XGB:>8.4f}")
    print(f"  {'1.5B M=10 XGB (exp3b)':<40} {EXP3B_XGB:>8.4f}")
    print(f"  {'7B M=5 XGB only (exp4)':<40} {EXP4_XGB:>8.4f} {0:>8}")
    print(f"  {'7B M=5 Hybrid (τ={tau}) ★':<40} {hybrid_em:>8.4f} {hybrid_f1:>8.4f}")
    print(f"  {'Delta vs XGB-only':<40} {hybrid_em - EXP4_XGB:>+8.4f}")

    # Per-split
    print(f"\n  {'Split':<20} {'N':>6} {'EM':>8} {'F1':>8}")
    print("  " + "-" * (W - 2))
    for split_name in ["overall", "feasible", "bridge", "comparison"]:
        s = splits[split_name]
        if s["n"] > 0:
            print(f"  {split_name:<20} {s['n']:>6} "
                  f"{s['em']/s['n']:>8.4f} {s['f1']/s['n']:>8.4f}")

    # Per-bucket impact
    print(f"\n  BUCKET IMPACT (hybrid vs XGB-only):")
    print(f"  {'Bucket':<35} {'N':>6} {'Hybrid correct':>15} {'Rate':>8}")
    print("  " + "-" * (W - 2))
    for b in ["A", "B", "C1", "C2", "D"]:
        bn = buckets[b]
        bc = bucket_correct[b]
        rate = bc / max(bn, 1)
        print(f"  {b:<35} {bn:>6} {bc:>15} {rate:>8.1%}")

    # C2 recovery
    c2_n = buckets["C2"]
    c2_c = bucket_correct["C2"]
    d_n = buckets["D"]
    d_c = bucket_correct["D"]
    d_lost = d_n - d_c

    print(f"\n  C2 recovered: {c2_c}/{c2_n} ({c2_c/max(c2_n,1):.1%})")
    print(f"  D retained:   {d_c}/{d_n} ({d_c/max(d_n,1):.1%})")
    print(f"  D lost:       {d_lost}")
    print(f"  Net gain:     {c2_c - d_lost:+d}")

    # Save summary
    summary = {
        "threshold": tau,
        "n_total": n,
        "n_llm_calls": n_llm_used,
        "hybrid_em": round(hybrid_em, 4),
        "hybrid_f1": round(hybrid_f1, 4),
        "xgb_only_em": round(EXP4_XGB, 4),
        "delta_em": round(hybrid_em - EXP4_XGB, 4),
        "splits": {k: {"n": v["n"],
                        "em": round(v["em"]/max(v["n"],1), 4),
                        "f1": round(v["f1"]/max(v["n"],1), 4)}
                   for k, v in splits.items()},
        "buckets": {b: {"n": buckets[b], "correct": bucket_correct[b]}
                    for b in buckets},
        "c2_recovered": c2_c,
        "d_lost": d_lost,
        "net_gain": c2_c - d_lost,
    }
    summary_path = os.path.join(args.out_dir, "hybrid_full_results.json")
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"\n  Results saved to {summary_path}")
    print("=" * W)


if __name__ == "__main__":
    main()