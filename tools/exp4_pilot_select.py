#!/usr/bin/env python3
"""
exp4_pilot_select.py — Select C2 + D questions for LLM verifier pilot

Identifies the 188 C2 questions (verifier picked plausible-wrong when correct
existed) and ~112 D questions (successes) for a balanced 300-question pilot.

Outputs a single pilot.jsonl with all context the LLM verifier needs per question:
  {qid, question, gold, qtype, bucket, candidates: [{answer, nli_flat, nli_hop1, 
   nli_hop2, min_hop, hop_balance}], hop1_text, hop2_text, xgb_pred, xgb_probs,
   xgb_best_idx, oracle_best_idx}

Usage:
    python3 tools/exp4_pilot_select.py \
        --chain_preds   exp4_7b/preds/dev_chain_verifier_mean_preds.jsonl \
        --oracle_perqid exp4_7b/metrics/oracle_M5_dev_perqid.jsonl \
        --hop_scores    exp4_7b/preds/dev_hop_scores.jsonl \
        --evidence      exp1b/evidence/dev_K100_chains.jsonl \
        --gold          data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
        --out_jsonl     exp4_7b/pilot/pilot_questions.jsonl \
        --out_summary   exp4_7b/pilot/pilot_summary.json \
        --n_d_questions 112 \
        --seed 42
"""

import argparse, collections, json, os, random, re, string


def normalize(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def em(pred, gold):
    return int(normalize(pred) == normalize(gold))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain_preds",   required=True)
    ap.add_argument("--oracle_perqid", required=True)
    ap.add_argument("--hop_scores",    required=True)
    ap.add_argument("--evidence",      required=True)
    ap.add_argument("--gold",          required=True)
    ap.add_argument("--out_jsonl",     required=True)
    ap.add_argument("--out_summary",   required=True)
    ap.add_argument("--n_d_questions", type=int, default=112)
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # Load gold
    gold_map, qtype_map = {}, {}
    for ex in json.load(open(args.gold)):
        qid = str(ex["_id"])
        gold_map[qid] = ex["answer"]
        qtype_map[qid] = ex.get("type", "bridge")

    # Load feasible
    feasible = set()
    ev_map = {}
    for line in open(args.evidence):
        r = json.loads(line)
        qid = str(r["qid"])
        chains = r.get("evidence", {}).get("chains") or r.get("chains", [])
        ev_map[qid] = chains
        if r.get("flags", {}).get("doc_recall_at_k", False):
            feasible.add(qid)

    # Load oracle
    oracle_map = {}
    for line in open(args.oracle_perqid):
        r = json.loads(line)
        oracle_map[str(r["qid"])] = r

    # Load XGB preds
    xgb_map = {}
    for line in open(args.chain_preds):
        r = json.loads(line)
        xgb_map[str(r["qid"])] = r

    # Load hop scores
    hop_map = {}
    for line in open(args.hop_scores):
        r = json.loads(line)
        hop_map[str(r["qid"])] = r

    # Identify buckets
    c2_qids, d_qids = [], []
    for qid in sorted(set(xgb_map) & set(oracle_map) & set(gold_map)):
        gold = gold_map[qid]
        pred = xgb_map[qid].get("pred", "")
        oracle_em = int(oracle_map[qid].get("best_em", 0))
        is_feas = qid in feasible
        correct = em(pred, gold)

        if not is_feas:
            continue  # Bucket A
        if oracle_em == 0:
            continue  # Bucket B
        if correct:
            d_qids.append(qid)
        elif is_bad(pred):
            continue  # C1
        else:
            c2_qids.append(qid)

    print(f"[pilot] C2 questions: {len(c2_qids)}")
    print(f"[pilot] D questions: {len(d_qids)}")

    # Sample D questions
    random.shuffle(d_qids)
    d_sample = d_qids[:args.n_d_questions]
    pilot_qids = c2_qids + d_sample
    random.shuffle(pilot_qids)

    print(f"[pilot] Pilot set: {len(pilot_qids)} "
          f"(C2={len(c2_qids)}  D={len(d_sample)})")

    # Build pilot JSONL
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    records = []
    for qid in pilot_qids:
        gold = gold_map[qid]
        xgb = xgb_map[qid]
        oracle = oracle_map[qid]
        hop = hop_map.get(qid, {})
        chains = ev_map.get(qid, [])

        hop1_text, hop2_text = get_hop_texts(chains)
        bucket = "C2" if qid in c2_qids else "D"

        # Find oracle best candidate index
        oracle_best = oracle.get("best_answer_id", 0)

        cands = hop.get("candidates", [])

        rec = {
            "qid": qid,
            "question": hop.get("question", ""),
            "gold": gold,
            "qtype": qtype_map.get(qid, "bridge"),
            "bucket": bucket,
            "hop1_text": hop1_text,
            "hop2_text": hop2_text,
            "candidates": cands,
            "xgb_pred": xgb.get("pred", ""),
            "xgb_probs": xgb.get("probs", []),
            "xgb_best_idx": xgb.get("best_idx", 0),
            "oracle_best_idx": oracle_best,
        }
        records.append(rec)

    with open(args.out_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary = {
        "n_total": len(records),
        "n_c2": len(c2_qids),
        "n_d": len(d_sample),
        "c2_bridge": sum(1 for q in c2_qids if qtype_map.get(q) == "bridge"),
        "c2_comp": sum(1 for q in c2_qids if qtype_map.get(q) == "comparison"),
    }
    json.dump(summary, open(args.out_summary, "w"), indent=2)
    print(f"[pilot] Written to {args.out_jsonl}")
    print(f"[pilot] Summary: {json.dumps(summary)}")


if __name__ == "__main__":
    main()