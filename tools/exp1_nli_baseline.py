#!/usr/bin/env python3
"""
exp1_nli_baseline.py — Step 5: NLI reranking baseline
Uses cross-encoder/nli-roberta-base to score each candidate answer
against the evidence, then picks the highest-entailment candidate.

Usage:
    python3 tools/exp1_nli_baseline.py \
        --candidates exp1/candidates/dev_M5_candidates_qwen_v4.jsonl \
        --evidence   exp1/evidence/dev_K20_chains.jsonl \
        --gold       data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
        --model      /var/tmp/$USER/sro/models/nli-roberta-base \
        --out_metrics exp1/metrics/exp1_step5_nli.json \
        --out_preds   exp1/preds/dev_nli_preds.jsonl \
        --batch_size  32
"""
import argparse, collections, json, os, re, string, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ──────────────────────────── text utils ────────────────────────────
def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))

def f1(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    common = collections.Counter(p_toks) & collections.Counter(g_toks)
    n = sum(common.values())
    if not n:
        return 0.0
    p = n / len(p_toks)
    r = n / len(g_toks)
    return 2 * p * r / (p + r)

# ──────────────────────────── evidence ──────────────────────────────
def flatten_evidence(chains: list, max_chars: int = 2000) -> str:
    parts = []
    for ch in chains[:4]:
        for hop in ch.get("hops", []):
            parts.append(f"{hop.get('title', '')}: {hop.get('text', '')}")
    return " ".join(parts)[:max_chars]

# ──────────────────────────── NLI scoring ───────────────────────────
def score_batch(
    model, tokenizer, device: str,
    premises: list, hypotheses: list,
    entail_idx: int,
) -> list:
    """Score a batch of (premise, hypothesis) pairs. Returns entailment probs."""
    enc = tokenizer(
        premises, hypotheses,
        truncation=True, max_length=512, padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[:, entail_idx].tolist()

# ──────────────────────────── main ──────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",   required=True)
    ap.add_argument("--evidence",     required=True)
    ap.add_argument("--gold",         required=True)
    ap.add_argument("--model",        required=True)
    ap.add_argument("--out_metrics",  required=True)
    ap.add_argument("--out_preds",    required=True)
    ap.add_argument("--max_ev_chars", type=int, default=2000)
    ap.add_argument("--batch_size",   type=int, default=32,
                    help="Number of (premise, hypothesis) pairs per GPU batch")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_preds),   exist_ok=True)

    # ── load model ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[nli] Loading model from {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    entail_idx = model.config.label2id.get("entailment", 1)
    print(f"[nli] Labels: {model.config.id2label}  →  entailment idx={entail_idx}")

    # ── load data ──
    gold_map = {q['_id']: q['answer']
                for q in json.load(open(args.gold))}

    feasible = set()
    ev_map   = {}
    for line in open(args.evidence):
        r = json.loads(line)
        ev_map[r['qid']] = r['chains']
        gold_titles = set(r['gold']['supporting_titles'])
        retrieved   = set(hop['title']
                          for ch in r['chains']
                          for hop in ch['hops'])
        if gold_titles.issubset(retrieved):
            feasible.add(r['qid'])

    print(f"[nli] Feasible subset: {len(feasible)}/7405")

    # ── build flat batch list ──
    # We score all (premise, candidate) pairs in large batches for GPU efficiency
    records   = []   # list of dicts with qid, candidates, premise
    pair_meta = []   # (record_idx, cand_idx) for each pair
    premises  = []
    hypotheses= []

    for line in open(args.candidates):
        r      = json.loads(line)
        qid    = r['qid']
        cands  = [c['answer_text'] for c in r['candidates']]
        chains = ev_map.get(qid, [])
        prem   = flatten_evidence(chains, args.max_ev_chars)

        rec_idx = len(records)
        records.append({"qid": qid, "candidates": cands,
                         "scores": [0.0] * len(cands)})

        for ci, ans in enumerate(cands):
            if not ans.strip() or ans.lower() == "unknown":
                # don't waste a forward pass — score stays 0.0
                continue
            pair_meta.append((rec_idx, ci))
            premises.append(prem)
            hypotheses.append(ans)

    print(f"[nli] Scoring {len(pair_meta)} pairs "
          f"({len(records)} questions × ~5 candidates) "
          f"in batches of {args.batch_size}...")

    t0 = time.time()
    for i in range(0, len(pair_meta), args.batch_size):
        batch_p = premises[i: i + args.batch_size]
        batch_h = hypotheses[i: i + args.batch_size]
        batch_scores = score_batch(model, tokenizer, device,
                                   batch_p, batch_h, entail_idx)
        for j, score in enumerate(batch_scores):
            rec_idx, ci = pair_meta[i + j]
            records[rec_idx]["scores"][ci] = score

        if (i // args.batch_size + 1) % 50 == 0:
            done = min(i + args.batch_size, len(pair_meta))
            eta  = (time.time() - t0) / done * (len(pair_meta) - done)
            print(f"[nli]   {done}/{len(pair_meta)} pairs "
                  f"| ETA {eta/60:.1f} min")

    print(f"[nli] Scoring done in {(time.time()-t0)/60:.1f} min")

    # ── pick best candidate per question ──
    preds = {}
    for rec in records:
        qid    = rec['qid']
        cands  = rec['candidates']
        scores = rec['scores']
        best_i = max(range(len(scores)), key=lambda i: scores[i])
        preds[qid] = {
            "pred":     cands[best_i],
            "scores":   scores,
            "best_idx": best_i,
        }

    # ── save predictions ──
    with open(args.out_preds, 'w') as f:
        for qid, p in preds.items():
            f.write(json.dumps({"qid": qid, **p}) + "\n")
    print(f"[nli] Predictions saved to {args.out_preds}")

    # ── compute metrics ──
    results = {}
    for split, qid_filter in [('overall', None), ('feasible', feasible)]:
        nli_em = nli_f1 = n = 0
        for qid, p in preds.items():
            if qid_filter is not None and qid not in qid_filter:
                continue
            g = gold_map.get(qid, '')
            nli_em += em(p['pred'], g)
            nli_f1 += f1(p['pred'], g)
            n += 1
        results[split] = {
            'n':      n,
            'nli_em': round(nli_em / n, 4),
            'nli_f1': round(nli_f1 / n, 4),
        }

    json.dump(results, open(args.out_metrics, 'w'), indent=2)
    print(json.dumps(results, indent=2))
    print(f"[nli] Metrics saved to {args.out_metrics}")


if __name__ == "__main__":
    main()