#!/usr/bin/env python3
"""
wiki2_prepare_evidence.py — Convert 2WikiMultiHopQA format to pipeline evidence JSONL

PURPOSE:
  The pipeline expects evidence in the schema produced by distractor_prepare_evidence.py:
    {qid, question, gold, type, chains:[{chain_id, hops:[{hop,title,text}]}],
     flat_evidence, all_paragraphs, gold_titles, flags}

  2WikiMultiHopQA provides:
    - context: [[title, [sent0, sent1, ...]], ...]   (same structure as HotpotQA)
    - supporting_facts: [[title, sent_id], ...]       (same structure as HotpotQA)
    - type: bridge | comparison | inference | compositional
    - evidences: [[title, sent_id, text], ...]        (explicit reasoning sentences)

  This script converts 2Wiki into the EXACT same schema as
  distractor_prepare_evidence.py so ALL downstream tools
  (distractor_generate.py, exp2_q1_signal_independence.py,
   exp_a1_qa_hop_score.py, exp_a1_lex_features.py,
   phase0_ablations_v2.py, phase0_bootstrap.py)
  work unchanged.

  Key properties of the output (matching distractor setting):
    - flags.doc_recall_at_k = True for ALL questions
      (gold paragraphs are guaranteed present in 2Wiki dev, same as
       HotpotQA distractor)
    - Bucket A (retrieval failure) = 0%
    - The generator sees ALL context paragraphs as evidence
    - NLI/QA/lex scorers see hop1 and hop2 separately via chains[0].hops

HOP ORDERING LOGIC:
  For all question types, we identify the 2 gold paragraphs from
  supporting_facts and order them as follows:

  bridge / inference / compositional:
    hop1 = the paragraph whose title appears more in the question
           (the bridge entity paragraph — same heuristic as
            distractor_prepare_evidence.py)
    hop2 = the other gold paragraph (the answer paragraph)

  comparison:
    hop1, hop2 in supporting_facts order (both are equally "answering"
    paragraphs; order is arbitrary, kept stable for reproducibility)

  Rationale: downstream NLI/QA scorers compute per-hop scores and their
  balance (|hop1 - hop2|). For bridge questions the balance signal is
  most informative when hop1 is consistently the bridge paragraph.

INPUT:
  --gold   data/wiki2/raw/dev.json
    2WikiMultiHopQA dev file. Each record has fields:
      _id, question, answer, type, supporting_facts, context, evidences

OUTPUT:
  --out_evidence   exp_wiki2/evidence/dev_wiki2_chains.jsonl
    One record per question, schema matches distractor_prepare_evidence.py

  --out_stats      exp_wiki2/evidence/prep_stats.json
    Conversion statistics and sanity-check counts

Usage:
  python3 tools/wiki2_prepare_evidence.py \\
      --gold         data/wiki2/raw/dev.json \\
      --out_evidence exp_wiki2/evidence/dev_wiki2_chains.jsonl \\
      --out_stats    exp_wiki2/evidence/prep_stats.json
"""

import argparse
import json
import os
import re
import sys


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1: TEXT UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _stop_words():
    return {
        "a", "an", "the", "is", "was", "were", "are", "be", "been",
        "of", "in", "on", "at", "to", "for", "with", "by", "from",
        "and", "or", "but", "not", "it", "this", "that", "which",
        "who", "what", "when", "where", "how", "did", "do", "does",
    }

STOP_WORDS = _stop_words()


def tokenize(text: str) -> list:
    """Lower-case word tokenizer, strips punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


def title_overlap(title: str, question: str) -> int:
    """
    Count question tokens that appear in the title (case-insensitive).
    Used to decide hop1 vs hop2 for bridge/inference/compositional.
    Higher overlap → more likely to be the bridge paragraph (hop1).
    """
    title_tokens = set(tokenize(title)) - STOP_WORDS
    question_tokens = set(tokenize(question)) - STOP_WORDS
    return len(title_tokens & question_tokens)


def format_flat_evidence(all_paragraphs: list, max_chars: int = 14000) -> str:
    """
    Format all paragraphs into the flat evidence string.
    Uses the same style as distractor_generate.py:
      [paragraph 1] Title: text
      [paragraph 2] Title: text
    Truncates at max_chars to stay within context limits.
    """
    parts = []
    for i, para in enumerate(all_paragraphs):
        title = para.get("title", f"Paragraph {i+1}")
        text = para.get("text", "")
        parts.append(f"[paragraph {i+1}] {title}: {text}")
    result = "\n\n".join(parts)
    if len(result) > max_chars:
        result = result[:max_chars].rsplit("\n", 1)[0]
    return result


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2: SUPPORTING FACTS PARSER
#  2WikiMultiHopQA supporting_facts has the same format as HotpotQA:
#  either a dict {"title": [...], "sent_id": [...]} or a list of
#  [title, sent_id] pairs.
# ═══════════════════════════════════════════════════════════════════════

def parse_supporting_facts(sup_facts) -> list:
    """
    Returns list of (title, sent_id) pairs, same parsing logic as
    distractor_prepare_evidence.py.
    """
    if isinstance(sup_facts, dict):
        titles = sup_facts.get("title", [])
        sent_ids = sup_facts.get("sent_id", [])
        return list(zip(titles, sent_ids))
    elif isinstance(sup_facts, list):
        if not sup_facts:
            return []
        if isinstance(sup_facts[0], (list, tuple)):
            # [[title, sent_id], ...] format
            return [(sf[0], sf[1]) for sf in sup_facts]
        else:
            # flat list shouldn't happen, but guard it
            return []
    return []


def get_unique_gold_titles(sf_pairs: list) -> list:
    """Deduplicate gold titles while preserving order."""
    seen = set()
    result = []
    for title, _ in sf_pairs:
        if title not in seen:
            seen.add(title)
            result.append(title)
    return result


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3: HOP ORDERING
# ═══════════════════════════════════════════════════════════════════════

def order_hops(gold_titles: list, question: str, qtype: str) -> tuple:
    """
    Decide (hop1_title, hop2_title) from the (at most 2) gold titles.

    bridge / inference / compositional:
      hop1 = title with higher overlap with question tokens
             (the bridge entity paragraph)
      hop2 = the other title (the answer paragraph)
      Tie-break: use supporting_facts order (first title = hop1)

    comparison:
      hop1, hop2 in supporting_facts order (stable, no ordering heuristic)

    Returns (hop1_title, hop2_title, hop_order_swapped: bool)
    hop_order_swapped is True if we reversed the supporting_facts order.
    """
    if len(gold_titles) < 2:
        # Only one gold title found — use it for both hops (degenerate)
        t = gold_titles[0] if gold_titles else ""
        return t, t, False

    t0, t1 = gold_titles[0], gold_titles[1]

    if qtype == "comparison":
        # Comparison: both titles score the entities; preserve SF order
        return t0, t1, False

    # bridge / inference / compositional: use overlap heuristic
    overlap0 = title_overlap(t0, question)
    overlap1 = title_overlap(t1, question)

    if overlap1 > overlap0:
        # t1 is more question-like → it is the bridge paragraph → hop1
        return t1, t0, True
    else:
        # t0 has higher (or equal) overlap → it is hop1 (SF order preserved)
        return t0, t1, False


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 4: MAIN CONVERSION LOGIC
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Convert 2WikiMultiHopQA dev set to pipeline evidence JSONL"
    )
    ap.add_argument("--gold", required=True,
        help="2WikiMultiHopQA dev JSON (e.g. data/wiki2/raw/dev.json)")
    ap.add_argument("--out_evidence", required=True,
        help="Output evidence JSONL (e.g. exp_wiki2/evidence/dev_wiki2_chains.jsonl)")
    ap.add_argument("--out_stats", required=True,
        help="Output stats JSON (e.g. exp_wiki2/evidence/prep_stats.json)")
    ap.add_argument("--max_evidence_chars", type=int, default=14000,
        help="Max chars for flat_evidence field (default: 14000, slightly larger "
             "than distractor 12000 since 2Wiki context varies in size)")
    args = ap.parse_args()

    for path in [args.out_evidence, args.out_stats]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # ── Load 2WikiMultiHopQA dev set ──
    print(f"Loading 2WikiMultiHopQA dev set from {args.gold} ...")
    with open(args.gold, encoding="utf-8") as f:
        raw = f.read(1)
        f.seek(0)
        if raw.strip() == "[":
            data = json.load(f)
        else:
            # JSONL fallback
            data = [json.loads(line) for line in f if line.strip()]
    print(f"  {len(data):,} questions loaded")

    # ── Statistics tracking ──
    stats = {
        "n_questions": len(data),
        "n_written": 0,
        # Question type counts
        "n_bridge": 0,
        "n_comparison": 0,
        "n_inference": 0,
        "n_compositional": 0,
        "n_other_type": 0,
        # Gold paragraph recovery
        "n_gold_found_2": 0,      # both gold paragraphs found in context
        "n_gold_found_1": 0,      # only 1 found
        "n_gold_found_0": 0,      # none found (should be 0)
        # Hop ordering
        "n_hop_order_swapped": 0, # cases where SF order was reversed
        "n_hop_order_issues": 0,  # ambiguous (overlap tie, non-bridge type)
        # Context size distribution
        "context_sizes": [],      # will be summarised in stats
        "n_context_lt_2": 0,      # context with < 2 paragraphs
    }

    out_f = open(args.out_evidence, "w", encoding="utf-8")
    n_written = 0

    for ex_idx, ex in enumerate(data):
        qid      = str(ex.get("_id", ex.get("id", "")))   # framolfese repack uses "id"
        question = ex["question"]
        answer   = ex["answer"]
        qtype    = ex.get("type", "bridge").lower().strip()

        # ── Count question types ──
        if qtype == "bridge":
            stats["n_bridge"] += 1
        elif qtype == "comparison":
            stats["n_comparison"] += 1
        elif qtype == "inference":
            stats["n_inference"] += 1
        elif qtype == "compositional":
            stats["n_compositional"] += 1
        else:
            stats["n_other_type"] += 1
            # Normalise unknown types to "bridge" for downstream compatibility
            qtype = "bridge"

        # ── Build paragraph lookup from context ──
        # Two formats exist:
        #   Original:    [[title, [sent0, sent1, ...]], ...]   — list of pairs
        #   Framolfese:  {"title": [t1,t2,...], "sentences": [[s0,s1],[s2,s3],...]}
        context = ex.get("context", {})

        para_map   = {}   # title → full_text
        para_order = []   # titles in original order (for all_paragraphs)

        if isinstance(context, dict):
            # ── Framolfese dict format ──
            titles_list    = context.get("title", [])
            sentences_list = context.get("sentences", [])
            for title, sents in zip(titles_list, sentences_list):
                title = str(title)
                if isinstance(sents, list):
                    full_text = " ".join(str(s) for s in sents)
                else:
                    full_text = str(sents)
                if title not in para_map:
                    para_map[title]  = full_text
                    para_order.append(title)

        elif isinstance(context, list):
            # ── Original list-of-pairs / list-of-dicts format ──
            for entry in context:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    title = str(entry[0])
                    sents = entry[1]
                    if isinstance(sents, list):
                        full_text = " ".join(str(s) for s in sents)
                    else:
                        full_text = str(sents)
                elif isinstance(entry, dict):
                    title     = str(entry.get("title", f"Para_{ex_idx}"))
                    sents     = entry.get("sentences", entry.get("text", ""))
                    if isinstance(sents, list):
                        full_text = " ".join(str(s) for s in sents)
                    else:
                        full_text = str(sents)
                else:
                    continue
                if title not in para_map:
                    para_map[title]  = full_text
                    para_order.append(title)

        stats["context_sizes"].append(len(para_order))
        if len(para_order) < 2:
            stats["n_context_lt_2"] += 1

        # ── Identify gold paragraph titles from supporting_facts ──
        sup_facts  = ex.get("supporting_facts", [])
        sf_pairs   = parse_supporting_facts(sup_facts)
        gold_titles_all = get_unique_gold_titles(sf_pairs)  # deduplicated, ordered

        # Restrict to titles that actually appear in context
        gold_titles_found = [t for t in gold_titles_all if t in para_map]

        n_found = len(gold_titles_found)
        if n_found >= 2:
            stats["n_gold_found_2"] += 1
        elif n_found == 1:
            stats["n_gold_found_1"] += 1
            # Duplicate the single title so downstream tools always get 2 hops
            gold_titles_found = gold_titles_found * 2
        else:
            stats["n_gold_found_0"] += 1
            # Fallback: use first two context paragraphs
            gold_titles_found = para_order[:2] if len(para_order) >= 2 else (para_order * 2)[:2]

        # ── Order hops ──
        hop1_title, hop2_title, swapped = order_hops(
            gold_titles_found[:2], question, qtype
        )
        if swapped:
            stats["n_hop_order_swapped"] += 1

        hop1_text = para_map.get(hop1_title, "")
        hop2_text = para_map.get(hop2_title, "")

        # ── Build chain record (IDENTICAL schema to distractor_prepare_evidence.py) ──
        chain = {
            "chain_id": 0,
            "hops": [
                {"hop": 1, "title": hop1_title, "text": hop1_text},
                {"hop": 2, "title": hop2_title, "text": hop2_text},
            ],
        }

        # all_paragraphs: all context paragraphs in original order
        all_paragraphs = [
            {"title": t, "text": para_map[t]}
            for t in para_order
        ]

        flat_evidence = format_flat_evidence(all_paragraphs, args.max_evidence_chars)

        # ── Build output record — EXACT same schema as distractor_prepare_evidence.py ──
        record = {
            "qid":          qid,
            "question":     question,
            "gold":         {"answer": answer},  # dict format — exp1_compute_oracle.py reads j["gold"]["answer"]
            "type":         qtype,           # bridge|comparison|inference|compositional
            "chains":       [chain],
            "flat_evidence": flat_evidence,
            "all_paragraphs": all_paragraphs,
            "gold_titles":  [hop1_title, hop2_title],
            "flags": {
                "doc_recall_at_k":  True,   # gold always present in 2Wiki dev (same as distractor)
                "wiki2_setting":    True,
                "distractor_setting": False,
                "n_context_paras":  len(para_order),
            },
        }

        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        n_written += 1

        if (n_written % 1000) == 0:
            print(f"  ... {n_written:,} / {len(data):,}")

    out_f.close()

    # ── Summarise context size distribution ──
    sizes = stats.pop("context_sizes")
    if sizes:
        stats["context_size_min"]    = min(sizes)
        stats["context_size_max"]    = max(sizes)
        stats["context_size_mean"]   = round(sum(sizes) / len(sizes), 2)
        stats["context_size_median"] = sorted(sizes)[len(sizes) // 2]
    else:
        stats["context_size_min"] = stats["context_size_max"] = 0
        stats["context_size_mean"] = stats["context_size_median"] = 0

    stats["n_written"] = n_written

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"  2WikiMultiHopQA evidence preparation complete")
    print(f"{'='*60}")
    print(f"  Written:         {n_written:,} questions")
    print(f"  bridge:          {stats['n_bridge']:,}")
    print(f"  comparison:      {stats['n_comparison']:,}")
    print(f"  inference:       {stats['n_inference']:,}")
    print(f"  compositional:   {stats['n_compositional']:,}")
    print(f"  other type:      {stats['n_other_type']:,}")
    print(f"  Gold found (2):  {stats['n_gold_found_2']:,}")
    print(f"  Gold found (1):  {stats['n_gold_found_1']:,}")
    print(f"  Gold found (0):  {stats['n_gold_found_0']:,}  ← should be 0")
    print(f"  Hop order swapped: {stats['n_hop_order_swapped']:,}")
    print(f"  Context size:    min={stats['context_size_min']} "
          f"max={stats['context_size_max']} "
          f"mean={stats['context_size_mean']} "
          f"median={stats['context_size_median']}")
    print(f"  Context < 2 para: {stats['n_context_lt_2']:,}  ← should be 0")

    if stats["n_gold_found_0"] > 0:
        print(f"\n  ⚠  WARNING: {stats['n_gold_found_0']} questions have 0 gold paragraphs!")
        print(f"     Check supporting_facts parsing logic.")
    if stats["n_context_lt_2"] > 0:
        print(f"\n  ⚠  WARNING: {stats['n_context_lt_2']} questions have < 2 context paragraphs!")
    if n_written != len(data):
        print(f"\n  ⚠  WARNING: wrote {n_written} but expected {len(data)}")

    with open(args.out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats → {args.out_stats}")
    print(f"  Evidence → {args.out_evidence}")
    print(f"\n  ✓ Ready for hop validation: python3 tools/wiki2_hop_validation.py")
    print(f"    --evidence {args.out_evidence}")


if __name__ == "__main__":
    main()