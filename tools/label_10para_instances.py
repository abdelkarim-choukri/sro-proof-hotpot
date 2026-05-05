#!/usr/bin/env python3
"""
label_10para_instances.py — Build labeled instances for the 10-paragraph verifier.

Each instance = (question, one unique answer, 10 paragraphs, label).
Deduplicates candidates per question by normalized answer text.

Input:
  --candidates  exp_distractor/candidates/dev_M5_diverse_10para.jsonl
  --gold        data/hotpotqa/raw/hotpot_dev_distractor_v1.json
  --chains      exp_distractor/evidence/dev_distractor_chains.jsonl

Output:
  --out         exp_10para/instances/dev_10para_instances.jsonl

Schema per line:
  {
    "qid":          "5a70f0e7",
    "instance_id":  "5a70f0e7_0",
    "question":     "Which city...",
    "answer":       "Paris",
    "label":        1,
    "qtype":        "bridge",
    "paragraphs":   [{"title": "...", "text": "..."}, ...],  # all 10
    "gold_titles":  ["Eiffel Tower", "Paris"],                # which 2 are gold
    "n_paragraphs": 10
  }

Usage:
  cd /var/tmp/u24sf51014/sro/work/sro-proof-hotpot
  /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/label_10para_instances.py
"""

import argparse
import json
import os
import re
import string
import sys
from collections import Counter


def normalize(s):
    s = (s or "").lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def em_match(pred, gold):
    return int(normalize(pred) == normalize(gold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",
                    default="exp_distractor/candidates/dev_M5_diverse_10para.jsonl")
    ap.add_argument("--gold",
                    default="data/hotpotqa/raw/hotpot_dev_distractor_v1.json")
    ap.add_argument("--chains",
                    default="exp_distractor/evidence/dev_distractor_chains.jsonl")
    ap.add_argument("--out",
                    default="exp_10para/instances/dev_10para_instances.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # ── Load gold ──────────────────────────────────────────────────────
    print("Loading gold answers...")
    gold_data = json.load(open(args.gold))
    gold_map = {}
    for ex in gold_data:
        qid = str(ex["_id"])
        # Extract gold paragraph titles from supporting_facts
        sf = ex.get("supporting_facts", [])
        if isinstance(sf, list) and sf and isinstance(sf[0], (list, tuple)):
            sf_titles = [s[0] for s in sf]
        elif isinstance(sf, dict):
            sf_titles = sf.get("title", [])
        else:
            sf_titles = []
        # Deduplicate while preserving order
        seen = set()
        gold_titles = []
        for t in sf_titles:
            if t not in seen:
                seen.add(t)
                gold_titles.append(t)

        gold_map[qid] = {
            "answer": ex["answer"],
            "question": ex["question"],
            "type": ex.get("type", "bridge"),
            "gold_titles": gold_titles,
        }
    print(f"  {len(gold_map)} questions")

    # ── Load 10 paragraphs ─────────────────────────────────────────────
    print("Loading paragraphs from chains file...")
    para_map = {}
    for line in open(args.chains):
        rec = json.loads(line.strip())
        qid = str(rec.get("qid", ""))
        paras = rec.get("all_paragraphs", [])
        if qid and paras:
            para_map[qid] = paras
    print(f"  {len(para_map)} questions with paragraphs")

    # ── Load candidates ────────────────────────────────────────────────
    print("Loading candidates...")
    cand_recs = [json.loads(l) for l in open(args.candidates) if l.strip()]
    print(f"  {len(cand_recs)} questions with candidates")

    # ── Build instances ────────────────────────────────────────────────
    print("Building instances...")
    n_instances = 0
    n_positive = 0
    n_questions_with_positive = 0
    n_questions_skipped = 0
    unique_per_q = []

    with open(args.out, "w") as fout:
        for rec in cand_recs:
            qid = str(rec["qid"])

            if qid not in gold_map or qid not in para_map:
                n_questions_skipped += 1
                continue

            g = gold_map[qid]
            gold_answer = g["answer"]
            question = g["question"]
            qtype = g["type"]
            gold_titles = g["gold_titles"]
            paragraphs = para_map[qid]

            # Deduplicate candidates by normalized answer
            seen_norm = set()
            unique_answers = []
            for c in rec["candidates"]:
                ans = c["answer_text"].strip()
                if not ans:
                    continue
                n = normalize(ans)
                if n not in seen_norm:
                    seen_norm.add(n)
                    unique_answers.append(ans)

            has_positive = False
            for idx, ans in enumerate(unique_answers):
                label = em_match(ans, gold_answer)
                if label:
                    n_positive += 1
                    has_positive = True

                instance = {
                    "qid": qid,
                    "instance_id": f"{qid}_{idx}",
                    "question": question,
                    "answer": ans,
                    "label": label,
                    "qtype": qtype,
                    "paragraphs": paragraphs,
                    "gold_titles": gold_titles,
                    "n_paragraphs": len(paragraphs),
                }

                fout.write(json.dumps(instance, ensure_ascii=False) + "\n")
                n_instances += 1

            if has_positive:
                n_questions_with_positive += 1
            unique_per_q.append(len(unique_answers))

    # ── Report ─────────────────────────────────────────────────────────
    n_questions = len(cand_recs) - n_questions_skipped
    print()
    print("=" * 60)
    print("  LABELING COMPLETE")
    print("=" * 60)
    print(f"  Questions processed:       {n_questions}")
    print(f"  Questions skipped:         {n_questions_skipped}")
    print(f"  Total instances:           {n_instances}")
    print(f"  Avg unique per question:   {sum(unique_per_q)/len(unique_per_q):.1f}")
    print(f"  Positive instances:        {n_positive} ({100*n_positive/n_instances:.1f}%)")
    print(f"  Questions with ≥1 correct: {n_questions_with_positive} ({100*n_questions_with_positive/n_questions:.1f}%)")
    print(f"  Questions with 0 correct:  {n_questions - n_questions_with_positive} ({100*(n_questions - n_questions_with_positive)/n_questions:.1f}%)")
    print()
    print(f"  Unique answers per question:")
    dist = Counter(unique_per_q)
    for k in sorted(dist):
        print(f"    {k} unique: {dist[k]:>5} ({100*dist[k]/len(unique_per_q):.1f}%)")
    print()
    print(f"  Output: {args.out}")
    print(f"  File size: {os.path.getsize(args.out) / 1024 / 1024:.1f} MB")
    print("=" * 60)

    # ── Sanity check — print 2 examples ────────────────────────────────
    print()
    print("  Sample instances:")
    with open(args.out) as f:
        for i, line in enumerate(f):
            if i >= 6:
                break
            inst = json.loads(line)
            print(f"    {inst['instance_id']:20s}  label={inst['label']}  "
                  f"answer={inst['answer'][:40]:40s}  "
                  f"qtype={inst['qtype']:12s}  "
                  f"n_para={inst['n_paragraphs']}")


if __name__ == "__main__":
    main()