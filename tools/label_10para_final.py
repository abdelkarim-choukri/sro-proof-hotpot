#!/usr/bin/env python3
"""
label_10para_final.py — Build clean labeled instances for the 10-paragraph verifier.

Each instance = (question, one answer, 10 paragraphs, label).

Positive answers:  from dev_M5_diverse.jsonl (2-gold generation, clean short answers)
Negative answers:  1. wrong generated answers (clean only)
                   2. distractor paragraph titles (always clean, hard negatives)

Filters:
  - direct + skeptical prompts: always kept
  - decomposed/entity/quote: kept only if short and clean
  - title negatives: stripped of parentheticals, max 6 words

Output schema:
  {
    "qid":          "5a70f0e7",
    "instance_id":  "5a70f0e7_0",
    "question":     "Which actress...",
    "answer":       "Kareena Kapoor",
    "label":        1,
    "qtype":        "bridge",
    "source":       "generated" | "title_negative",
    "paragraphs":   [{"title": "...", "text": "..."}, ...],
    "gold_titles":  ["Kareena Kapoor filmography", "Kabhi Khushi..."],
    "n_paragraphs": 10
  }

Usage:
  cd /var/tmp/u24sf51014/sro/work/sro-proof-hotpot
  /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/label_10para_final.py
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


# Garbage detection patterns
GARBAGE_PREFIXES = [
    'Step', 'Relevant', 'The question', 'Most relevant',
    'Entities', 'Attribute', '- ', 'Based on', 'From the',
    'Looking at', 'To answer', 'We need', 'Let me',
    'According to', 'The entities', 'The answer',
    'Passage ', 'The most', 'First,',
]


def is_clean_answer(ans, prompt_id):
    """Check if an answer is clean enough for training."""
    ans = ans.strip()
    if not ans:
        return False

    # Direct and skeptical are almost always clean
    if prompt_id in ('direct', 'skeptical'):
        # Only reject truly broken ones
        if len(ans.split()) > 20:
            return False
        if any(ans.startswith(p) for p in GARBAGE_PREFIXES):
            return False
        if '**' in ans:
            return False
        return True

    # Other prompts: strict filtering
    if len(ans.split()) > 10:
        return False
    if any(ans.startswith(p) for p in GARBAGE_PREFIXES):
        return False
    if '**' in ans or ':' in ans and len(ans) > 30:
        return False
    if ans[0].isdigit() and '.' in ans[:3]:
        return False
    return True


def clean_title(title):
    """Strip parentheticals from paragraph titles for use as negatives."""
    # "Ed Wood (film)" → "Ed Wood"
    # "Golden Globe Award for Best Original Score" → keep as is
    cleaned = re.sub(r'\s*\([^)]*\)\s*$', '', title).strip()
    return cleaned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",
                    default="exp_distractor/candidates/dev_M5_diverse.jsonl")
    ap.add_argument("--gold",
                    default="data/hotpotqa/raw/hotpot_dev_distractor_v1.json")
    ap.add_argument("--chains",
                    default="exp_distractor/evidence/dev_distractor_chains.jsonl")
    ap.add_argument("--out",
                    default="exp_10para/instances/dev_10para_final.jsonl")
    ap.add_argument("--max_title_negatives", type=int, default=6,
                    help="Max title negatives per question (to limit imbalance)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # ── Load gold ──────────────────────────────────────────────────────
    print("Loading gold answers...")
    gold_data = json.load(open(args.gold))
    gold_map = {}
    for ex in gold_data:
        qid = str(ex["_id"])
        sf = ex.get("supporting_facts", [])
        seen = set()
        gold_titles = []
        for s in sf:
            t = s[0]
            if t not in seen:
                seen.add(t)
                gold_titles.append(t)

        gold_map[qid] = {
            "answer": ex["answer"],
            "question": ex["question"],
            "type": ex.get("type", "bridge"),
            "gold_titles": gold_titles,
            "context": ex["context"],  # all 10 paragraphs
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
    cand_recs = {json.loads(l)["qid"]: json.loads(l)
                 for l in open(args.candidates) if l.strip()}
    print(f"  {len(cand_recs)} questions with candidates")

    # ── Build instances ────────────────────────────────────────────────
    print("Building instances...")
    n_instances = 0
    n_pos = 0
    n_neg_generated = 0
    n_neg_title = 0
    n_questions = 0
    n_q_with_pos = 0
    n_garbage_filtered = 0
    unique_per_q = []
    source_counts = Counter()

    with open(args.out, "w") as fout:
        for qid in sorted(gold_map.keys()):
            if qid not in para_map or qid not in cand_recs:
                continue

            g = gold_map[qid]
            gold_answer = g["answer"]
            question = g["question"]
            qtype = g["type"]
            gold_titles_list = g["gold_titles"]
            gold_titles_set = set(gold_titles_list)
            paragraphs = para_map[qid]
            rec = cand_recs[qid]

            n_questions += 1
            seen_norm = set()
            instances_this_q = []

            # ── Step 1: Clean generated answers ────────────────────────
            for c in rec["candidates"]:
                ans = c["answer_text"].strip()
                pid = c["prompt_id"]

                if not is_clean_answer(ans, pid):
                    n_garbage_filtered += 1
                    continue

                # Strip leading "- " or "* "
                ans = re.sub(r'^[-*•]\s+', '', ans).strip()
                if not ans:
                    continue

                n = normalize(ans)
                if n and n not in seen_norm:
                    seen_norm.add(n)
                    label = int(n == normalize(gold_answer))
                    instances_this_q.append({
                        "answer": ans,
                        "label": label,
                        "source": "generated",
                    })

            # ── Step 2: Title negatives from distractor paragraphs ─────
            all_titles = [c[0] for c in g["context"]]
            title_negs_added = 0
            for title in all_titles:
                if title in gold_titles_set:
                    continue
                ct = clean_title(title)
                if not ct or len(ct.split()) > 6:
                    continue
                n = normalize(ct)
                if n and n not in seen_norm:
                    seen_norm.add(n)
                    label = int(n == normalize(gold_answer))
                    instances_this_q.append({
                        "answer": ct,
                        "label": label,
                        "source": "title_negative",
                    })
                    title_negs_added += 1
                    if title_negs_added >= args.max_title_negatives:
                        break

            # ── Write instances ────────────────────────────────────────
            has_pos = False
            for idx, inst in enumerate(instances_this_q):
                if inst["label"] == 1:
                    has_pos = True
                    n_pos += 1
                elif inst["source"] == "generated":
                    n_neg_generated += 1
                else:
                    n_neg_title += 1

                source_counts[inst["source"]] += 1

                fout.write(json.dumps({
                    "qid": qid,
                    "instance_id": f"{qid}_{idx}",
                    "question": question,
                    "answer": inst["answer"],
                    "label": inst["label"],
                    "qtype": qtype,
                    "source": inst["source"],
                    "paragraphs": paragraphs,
                    "gold_titles": gold_titles_list,
                    "n_paragraphs": len(paragraphs),
                }, ensure_ascii=False) + "\n")
                n_instances += 1

            if has_pos:
                n_q_with_pos += 1
            unique_per_q.append(len(instances_this_q))

    # ── Report ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  LABELING COMPLETE")
    print("=" * 65)
    print(f"  Questions:                 {n_questions}")
    print(f"  Total instances:           {n_instances}")
    print(f"  Avg per question:          {sum(unique_per_q)/len(unique_per_q):.1f}")
    print()
    print(f"  Positive (correct):        {n_pos} ({100*n_pos/n_instances:.1f}%)")
    print(f"  Negative (generated):      {n_neg_generated}")
    print(f"  Negative (title):          {n_neg_title}")
    print(f"  Garbage filtered:          {n_garbage_filtered}")
    print()
    print(f"  Questions with correct:    {n_q_with_pos} ({100*n_q_with_pos/n_questions:.1f}%)")
    print(f"  Questions without correct: {n_questions - n_q_with_pos} ({100*(n_questions-n_q_with_pos)/n_questions:.1f}%)")
    print()
    print(f"  Source breakdown:")
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src}: {cnt}")
    print()
    dist = Counter(unique_per_q)
    print(f"  Instances per question distribution:")
    for k in sorted(dist):
        print(f"    {k}: {dist[k]:>5} ({100*dist[k]/len(unique_per_q):.1f}%)")
    print()
    pct_2plus = 100 * sum(v for k, v in dist.items() if k >= 2) / len(unique_per_q)
    pct_3plus = 100 * sum(v for k, v in dist.items() if k >= 3) / len(unique_per_q)
    print(f"  2+ instances: {pct_2plus:.1f}%")
    print(f"  3+ instances: {pct_3plus:.1f}%")
    print()
    print(f"  Output: {args.out}")
    print(f"  File size: {os.path.getsize(args.out) / 1024 / 1024:.1f} MB")
    print("=" * 65)

    # ── Sanity check ───────────────────────────────────────────────────
    print()
    print("  Sample instances (first 2 questions):")
    shown_qids = set()
    with open(args.out) as f:
        for line in f:
            inst = json.loads(line)
            if len(shown_qids) >= 2 and inst["qid"] not in shown_qids:
                break
            shown_qids.add(inst["qid"])
            print(f"    {inst['instance_id']:25s}  label={inst['label']}  "
                  f"src={inst['source']:15s}  "
                  f"answer={inst['answer'][:40]}")


if __name__ == "__main__":
    main()