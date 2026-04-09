#!/usr/bin/env python3
"""
distractor_prepare_evidence.py — Convert HotpotQA distractor format to pipeline evidence JSONL

PURPOSE:
  The existing pipeline expects evidence in the MDR chain format:
    {qid, question, chains: [{chain_id, hops: [{hop, title, text}]}], flags: {doc_recall_at_k}}

  The distractor setting provides 10 paragraphs per question (2 gold + 8 distractors),
  with supporting_facts identifying which paragraphs and sentences are gold.

  This script converts the distractor format into the pipeline's evidence format by:
    1. Identifying the 2 gold paragraphs from supporting_facts
    2. Structuring them as hop1 (bridge) and hop2 (answer) in a chain
    3. Including ALL 10 paragraphs as flat evidence for the generator prompt
    4. Setting doc_recall_at_k = True for ALL questions (gold is always present)

  This means:
    - Bucket A (retrieval failure) = 0% → gold is guaranteed
    - The generator sees all 10 paragraphs as context
    - The NLI/QA/lex scorers see hop1 and hop2 separately (just like MDR chains)

INPUT:
  --gold  data/hotpotqa/raw/hotpot_dev_distractor_v1.json
    HotpotQA dev distractor file with fields:
      _id, question, answer, supporting_facts, context, type, level

OUTPUT:
  --out_evidence  exp_distractor/evidence/dev_distractor_chains.jsonl
    One line per question in the pipeline's chain format

  --out_stats  exp_distractor/evidence/prep_stats.json
    Statistics about the conversion (sanity checks)

Usage:
  python3 tools/distractor_prepare_evidence.py \
      --gold      data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
      --out_evidence exp_distractor/evidence/dev_distractor_chains.jsonl \
      --out_stats    exp_distractor/evidence/prep_stats.json
"""

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="Prepare distractor evidence for pipeline")
    ap.add_argument("--gold", required=True, help="HotpotQA distractor dev JSON")
    ap.add_argument("--out_evidence", required=True, help="Output evidence JSONL")
    ap.add_argument("--out_stats", required=True, help="Output stats JSON")
    args = ap.parse_args()

    for p in [args.out_evidence, args.out_stats]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    print("Loading HotpotQA distractor dev set ...")
    data = json.load(open(args.gold))
    print(f"  {len(data)} questions loaded")

    # ── Statistics tracking ──
    stats = {
        "n_questions": len(data),
        "n_gold_found_2": 0,      # both gold paragraphs identified
        "n_gold_found_1": 0,      # only 1 gold paragraph found
        "n_gold_found_0": 0,      # no gold paragraphs (shouldn't happen)
        "n_bridge": 0,
        "n_comparison": 0,
        "hop_order_issues": 0,    # cases where hop ordering is ambiguous
    }

    out_f = open(args.out_evidence, "w")
    n_written = 0

    for ex in data:
        qid = str(ex["_id"])
        question = ex["question"]
        answer = ex["answer"]
        qtype = ex.get("type", "bridge")
        context = ex["context"]             # list of [title, [sent0, sent1, ...]]
        sup_facts = ex["supporting_facts"]  # {title: [...], sent_id: [...]}

        if qtype == "bridge":
            stats["n_bridge"] += 1
        else:
            stats["n_comparison"] += 1

        # ── Build paragraph lookup ──
        # context is a list of [title, sentences_list]
        para_map = {}  # title → full text
        para_order = []  # preserve original ordering
        for title, sentences in context:
            full_text = " ".join(sentences)
            para_map[title] = full_text
            para_order.append(title)

        # ── Identify gold paragraphs from supporting_facts ──
        # supporting_facts has two parallel lists: title and sent_id
        if isinstance(sup_facts, dict):
            sf_titles = sup_facts.get("title", [])
        elif isinstance(sup_facts, list):
            # Some versions have list of [title, sent_id] pairs
            sf_titles = [sf[0] for sf in sup_facts]
        else:
            sf_titles = []

        gold_titles = list(dict.fromkeys(sf_titles))  # unique, preserve order

        # ── Assign hop1 and hop2 ──
        # For bridge questions: hop1 = first gold paragraph (bridge), hop2 = second (answer)
        # For comparison questions: both paragraphs are somewhat parallel,
        #   but we still assign them as hop1/hop2 in the order they appear in supporting_facts

        if len(gold_titles) >= 2:
            stats["n_gold_found_2"] += 1
            hop1_title = gold_titles[0]
            hop2_title = gold_titles[1]
        elif len(gold_titles) == 1:
            stats["n_gold_found_1"] += 1
            hop1_title = gold_titles[0]
            # Use a dummy for hop2 — pick the first non-gold paragraph
            hop2_title = None
            for t in para_order:
                if t != hop1_title:
                    hop2_title = t
                    break
            if hop2_title is None:
                hop2_title = hop1_title  # fallback to same paragraph
        else:
            stats["n_gold_found_0"] += 1
            # Shouldn't happen — use first two paragraphs
            hop1_title = para_order[0] if len(para_order) > 0 else ""
            hop2_title = para_order[1] if len(para_order) > 1 else hop1_title

        hop1_text = para_map.get(hop1_title, "")
        hop2_text = para_map.get(hop2_title, "")

        # ── Build flat evidence text (all 10 paragraphs for the generator) ──
        flat_parts = []
        for ci, title in enumerate(para_order):
            text = para_map[title]
            flat_parts.append(f"[chain 0 hop {ci}] {title}: {text}")

        flat_evidence = "\n".join(flat_parts)

        # ── Build the chain structure (for NLI/QA/lex scoring) ──
        chain = {
            "chain_id": 0,
            "hops": [
                {"hop": 1, "title": hop1_title, "text": hop1_text},
                {"hop": 2, "title": hop2_title, "text": hop2_text},
            ],
        }

        # ── Build the output record matching the pipeline's expected format ──
        record = {
            "qid": qid,
            "question": question,
            "gold": answer,
            "type": qtype,
            "chains": [chain],
            "flat_evidence": flat_evidence,
            "all_paragraphs": [
                {"title": title, "text": para_map[title]}
                for title in para_order
            ],
            "gold_titles": gold_titles,
            "flags": {
                "doc_recall_at_k": True,  # gold is ALWAYS present in distractor setting
                "distractor_setting": True,
            },
        }

        out_f.write(json.dumps(record) + "\n")
        n_written += 1

    out_f.close()

    stats["n_written"] = n_written

    print(f"\n  Written: {n_written} questions to {args.out_evidence}")
    print(f"  Gold paragraphs found (2): {stats['n_gold_found_2']}")
    print(f"  Gold paragraphs found (1): {stats['n_gold_found_1']}")
    print(f"  Gold paragraphs found (0): {stats['n_gold_found_0']}")
    print(f"  Bridge: {stats['n_bridge']}  Comparison: {stats['n_comparison']}")

    with open(args.out_stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to {args.out_stats}")

    # Sanity checks
    if stats["n_gold_found_0"] > 0:
        print(f"\n  ⚠ WARNING: {stats['n_gold_found_0']} questions have 0 gold paragraphs!")
    if stats["n_gold_found_1"] > 10:
        print(f"\n  ⚠ WARNING: {stats['n_gold_found_1']} questions have only 1 gold paragraph")
    if n_written != len(data):
        print(f"\n  ⚠ WARNING: wrote {n_written} but expected {len(data)}")

    print("\n  ✓ Evidence preparation complete")


if __name__ == "__main__":
    main()