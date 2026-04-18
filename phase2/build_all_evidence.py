#!/usr/bin/env python3
"""
phase2/build_all_evidence.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unified Evidence Builder — All 3 setups × Both Datasets = 6 files
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS REPLACES THE PREVIOUS SCRIPTS
    The previous build_attributed_evidence.py and build_ablation_evidence.py
    both contained a silent bug: they selected from a pool of exactly 1 chain
    (always the gold chain from chains[0]) because the evidence files store
    only the gold chain. The "lexical overlap selection" was trivially always
    picking gold, making llm_attributed.jsonl == gold.jsonl.

    This script fixes that by building a real chain pool from all_paragraphs
    (the full paragraph context stored in the evidence file) for BOTH datasets.

THE THREE SETUPS
━━━━━━━━━━━━━━━
  Primary — llm_attributed.jsonl   (CANDIDATE-SPECIFIC selection)
    For each candidate answer independently:
    1. Build chain pool = all ordered paragraph pairs from all_paragraphs
    2. Score each pair: token_jaccard(answer, hop1_text + hop2_text)
    3. Select the highest-scoring pair as that candidate's evidence
    → Different candidates for the same question may use different chains
    → The gold pair will be selected when the answer is grounded in gold
    → Non-gold pairs selected for distractor-grounded or wrong answers

  Ablation 1 — gold.jsonl   (QUESTION-LEVEL, always gold)
    Use the explicitly labeled gold chain (chains[0] from the evidence file).
    Same chain for all candidates of a question. Gold guaranteed.
    This is the idealized upper bound.

  Ablation 2 — top_chain.jsonl   (QUESTION-LEVEL, answer-blind)
    For each question, compute the best-scoring pair using the QUESTION text
    (not any candidate answer) as the query signal.
    → Same chain for all candidates of a question (no candidate signal used)
    → Simulates a retriever that picks the chain most relevant to the question
      but doesn't know which answer the candidate produced
    → Will differ from gold when distractors are topically close to the question

THE KEY DISTINCTION: Primary vs Top-chain
    llm_attributed: driven by the CANDIDATE ANSWER  → per-candidate chain
    top_chain:      driven by the QUESTION TEXT     → per-question chain
    gold:           driven by gold labels           → per-question, always correct

    This gives a clean 3-way decomposition:
      gold vs top_chain:     what does knowing gold give you?
      top_chain vs primary:  what does candidate-specific selection give you?

DATASET-SPECIFIC NOTES
    HotpotQA distractor setting:
      - all_paragraphs: 10 paragraphs (2 gold + 8 distractors) per question
      - Chain pool: 10×9 = 90 ordered pairs
      - Gold guaranteed (Bucket A = 0%)

    2WikiMultiHopQA:
      - all_paragraphs: variable (~10 paragraphs) per question
      - Chain pool: N×(N-1) ordered pairs
      - Gold guaranteed (Bucket A = 0%, same as distractor)
      - Both datasets are NOW SYMMETRIC in pool construction

INPUT FILES
    HotpotQA:
      candidates: exp_distractor/candidates/dev_M5_sampling.jsonl
      evidence:   exp_distractor/evidence/dev_distractor_chains.jsonl
    2Wiki:
      candidates: exp_wiki2/candidates/dev_M5_sampling.jsonl
      evidence:   exp_wiki2/evidence/dev_wiki2_chains.jsonl

OUTPUT (replaces all previous phase2 files)
    phase2/
    ├── hotpotqa/
    │   ├── llm_attributed.jsonl   ← PRIMARY   (candidate-specific, answer-driven)
    │   ├── gold.jsonl             ← ABLATION 1 (gold pair, question-level)
    │   └── top_chain.jsonl        ← ABLATION 2 (question-driven, answer-blind)
    └── 2wiki/
        ├── llm_attributed.jsonl   ← PRIMARY
        ├── gold.jsonl             ← ABLATION 1
        └── top_chain.jsonl        ← ABLATION 2

OUTPUT SCHEMA (identical across all 6 files)
    {
        "question_id"         : str,
        "candidate_id"        : str,        # "{qid}_c{answer_id}"
        "answer_text"         : str,
        "used_chains"         : [int],      # [chain_id of selected pair]
        "hop1_title"          : str,
        "hop1_text"           : str,
        "hop2_title"          : str,
        "hop2_text"           : str,
        "attribution_score"   : float|null, # jaccard score; null for gold
        "attribution_method"  : str,        # see METHODS below
        "query_text"          : str,        # what drove the selection
        "dataset"             : str,
        "n_chains_available"  : int,        # size of pair pool
        "gold_chain_selected" : bool        # did we land on the gold pair?
    }

ATTRIBUTION METHODS
    "lexical_overlap_answer"    → Primary: driven by candidate answer
    "lexical_overlap_question"  → Top-chain: driven by question text
    "gold"                      → Gold: explicitly labeled gold pair
    "fallback_non_gold_pair"    → Fallback for zero-overlap (primary only)
    "fallback_first_pair"       → Fallback when pool is empty

USAGE
    # Full run (both datasets, all 3 setups):
    python3 phase2/build_all_evidence.py \
        --proj_root /var/tmp/u24sf51014/sro/work/sro-proof-hotpot \
        --out_dir   phase2

    # Test 100 questions:
    python3 phase2/build_all_evidence.py ... --max_q 100

    # Dry run (no files written):
    python3 phase2/build_all_evidence.py ... --dry_run
"""

import argparse
import json
import os
import re
import string
import time
from typing import Dict, Iterator, List, Optional, Set, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 1 — TEXT UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def token_jaccard(text_a: str, text_b: str) -> float:
    na = set(normalize(text_a).split())
    nb = set(normalize(text_b).split())
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 2 — I/O UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, records: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"    ✓ Written {len(records):,} records → {path}")


def load_candidates(path: str) -> Dict[str, List[dict]]:
    result = {}
    for rec in iter_jsonl(path):
        qid = str(rec["qid"])
        raw = rec.get("candidates", [])
        normed = []
        for i, c in enumerate(raw):
            if isinstance(c, dict):
                normed.append({
                    "answer_id":   int(c.get("answer_id", i)),
                    "answer_text": str(c.get("answer_text", c.get("answer", ""))).strip(),
                })
            else:
                normed.append({"answer_id": i, "answer_text": str(c).strip()})
        result[qid] = normed
    return result


def load_evidence(path: str) -> Dict[str, dict]:
    result = {}
    for rec in iter_jsonl(path):
        result[str(rec["qid"])] = rec
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 3 — CHAIN POOL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_pair_pool(
    all_paragraphs: List[dict],
    gold_titles:    List[str],
) -> List[dict]:
    """
    Build all ordered paragraph pairs as candidate 2-hop chains.
    Chain ID = i * N + j (unique, deterministic).
    Each pair stores whether it is the gold pair.
    """
    N = len(all_paragraphs)
    gold_set = {normalize(t) for t in gold_titles}
    pool = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            pi = all_paragraphs[i]
            pj = all_paragraphs[j]
            h1t = str(pi.get("title", "")).strip()
            h1x = str(pi.get("text",  "")).strip()
            h2t = str(pj.get("title", "")).strip()
            h2x = str(pj.get("text",  "")).strip()

            is_gold = (normalize(h1t) in gold_set and
                       normalize(h2t) in gold_set and
                       len(gold_set) >= 2)

            pool.append({
                "chain_id":      i * N + j,
                "hop1_title":    h1t,
                "hop1_text":     h1x,
                "hop2_title":    h2t,
                "hop2_text":     h2x,
                "is_gold":       is_gold,
                "combined_text": h1x + " " + h2x,
                "i": i, "j": j,
            })
    return pool


def select_best(
    pool:       List[dict],
    query:      str,
    prefer_non_gold_on_tie: bool = False,
) -> Tuple[dict, float]:
    """
    Select best chain from pool by token_jaccard(query, combined_text).
    Tie-breaking: (score desc, i asc, j asc) — deterministic.
    Returns (best_chain, best_score).
    """
    if not pool:
        return {}, 0.0

    scored = sorted(
        pool,
        key=lambda c: (-token_jaccard(query, c["combined_text"]), c["i"], c["j"])
    )
    best = scored[0]
    score = token_jaccard(query, best["combined_text"])
    return best, score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 4 — RECORD BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_record(
    qid:        str,
    cand:       dict,
    chain:      dict,
    score:      Optional[float],
    method:     str,
    query_text: str,
    dataset:    str,
    n_pool:     int,
) -> dict:
    return {
        "question_id":         qid,
        "candidate_id":        f"{qid}_c{cand['answer_id']}",
        "answer_text":         cand["answer_text"],
        "used_chains":         [chain.get("chain_id", 0)],
        "hop1_title":          chain.get("hop1_title", ""),
        "hop1_text":           chain.get("hop1_text", ""),
        "hop2_title":          chain.get("hop2_title", ""),
        "hop2_text":           chain.get("hop2_text", ""),
        "attribution_score":   round(score, 6) if score is not None else None,
        "attribution_method":  method,
        "query_text":          query_text,  # what drove the selection
        "dataset":             dataset,
        "n_chains_available":  n_pool,
        "gold_chain_selected": chain.get("is_gold", False),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 5 — DATASET PROCESSOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_dataset(
    dataset:    str,
    cand_path:  str,
    ev_path:    str,
    out_primary: str,
    out_gold:    str,
    out_topchain: str,
    max_q:      Optional[int],
    dry_run:    bool,
) -> dict:

    print(f"\n{'━'*70}")
    print(f"  Dataset : {dataset.upper()}")
    print(f"  Candidates : {cand_path}")
    print(f"  Evidence   : {ev_path}")
    print(f"{'━'*70}")

    for p, lbl in [(cand_path, "candidates"), (ev_path, "evidence")]:
        if not os.path.exists(p):
            print(f"  ✗ {lbl} not found: {p}")
            return {"dataset": dataset, "status": "skipped"}

    cand_map = load_candidates(cand_path)
    ev_map   = load_evidence(ev_path)

    valid_qids = sorted(set(cand_map.keys()) & set(ev_map.keys()))
    if max_q:
        valid_qids = valid_qids[:max_q]

    # ── Schema check ─────────────────────────────────────────────────
    first_ev = ev_map[valid_qids[0]]
    n_paras  = len(first_ev.get("all_paragraphs", []))
    gold_t   = first_ev.get("gold_titles", [])
    print(f"\n  Schema check (first question):")
    print(f"    all_paragraphs : {n_paras} paragraphs")
    print(f"    gold_titles    : {gold_t}")
    print(f"    pool size      : {n_paras*(n_paras-1)} ordered pairs")
    print(f"    chains[0].hop1 : '{first_ev.get('chains',[{}])[0].get('hops',[{}])[0].get('title','')}'")

    if n_paras < 2:
        print(f"  ✗ all_paragraphs has < 2 paragraphs — cannot build pair pool")
        return {"dataset": dataset, "status": "skipped_empty_paragraphs"}

    print(f"\n  Processing {len(valid_qids):,} questions ...")
    t0 = time.time()

    primary_records  = []
    gold_records     = []
    topchain_records = []

    # per-question gold chain (from chains[0] — always gold by construction)
    def get_gold_chain(ev_rec: dict) -> dict:
        chains = ev_rec.get("chains") or ev_rec.get("evidence", {}).get("chains", [])
        if not chains:
            return {}
        c = chains[0]
        hops = c.get("hops", [])
        h1t = h1x = h2t = h2x = ""
        if len(hops) >= 1: h1t, h1x = hops[0].get("title",""), hops[0].get("text","")
        if len(hops) >= 2: h2t, h2x = hops[1].get("title",""), hops[1].get("text","")
        return {
            "chain_id": 0, "hop1_title": h1t, "hop1_text": h1x,
            "hop2_title": h2t, "hop2_text": h2x, "is_gold": True,
            "combined_text": h1x + " " + h2x, "i": -1, "j": -1,
        }

    # ── Diagnostic counters ───────────────────────────────────────────
    primary_gold_selected    = 0
    topchain_gold_selected   = 0
    primary_fallback         = 0
    topchain_lexical         = 0
    pool_sizes               = []

    for idx, qid in enumerate(valid_qids):
        candidates  = cand_map[qid]
        ev_rec      = ev_map[qid]
        all_paras   = ev_rec.get("all_paragraphs", [])
        gold_titles = ev_rec.get("gold_titles", [])
        question    = ev_rec.get("question", "")

        pool = build_pair_pool(all_paras, gold_titles)
        pool_sizes.append(len(pool))

        gold_chain_rec = get_gold_chain(ev_rec)

        # ── Top-chain: question-driven, same for all candidates ───────
        # Select best pair using QUESTION text as query (answer-blind)
        top_chain, top_score = select_best(pool, question)
        if not top_chain:
            top_chain  = gold_chain_rec
            top_method = "fallback_first_pair"
        else:
            top_method = "lexical_overlap_question"
            topchain_lexical += len(candidates)
        if top_chain.get("is_gold"):
            topchain_gold_selected += len(candidates)

        for cand in candidates:
            answer = cand["answer_text"]

            # ── PRIMARY: candidate-specific, answer-driven ────────────
            best, best_score = select_best(pool, answer)
            if not best or best_score <= 0.0:
                # Fallback: first non-gold pair (ensures ≠ gold for garbage answers)
                non_gold = sorted(
                    [c for c in pool if not c["is_gold"]],
                    key=lambda c: (c["i"], c["j"])
                )
                if non_gold:
                    best, best_score = non_gold[0], 0.0
                    p_method = "fallback_non_gold_pair"
                else:
                    best, best_score = pool[0] if pool else gold_chain_rec, 0.0
                    p_method = "fallback_first_pair"
                primary_fallback += 1
            else:
                p_method = "lexical_overlap_answer"

            if best.get("is_gold"):
                primary_gold_selected += 1

            primary_records.append(make_record(
                qid, cand, best, best_score, p_method,
                query_text=answer, dataset=dataset, n_pool=len(pool)
            ))

            # ── GOLD: always gold chain, same for all candidates ──────
            gold_records.append(make_record(
                qid, cand, gold_chain_rec, None, "gold",
                query_text="[gold_labels]", dataset=dataset, n_pool=1
            ))

            # ── TOP-CHAIN: question-driven, same for all candidates ───
            topchain_records.append(make_record(
                qid, cand, top_chain, top_score, top_method,
                query_text=question, dataset=dataset, n_pool=len(pool)
            ))

        if (idx + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"    {idx+1:,}/{len(valid_qids):,}  ({rate:.0f} q/s  "
                  f"ETA {(len(valid_qids)-idx-1)/rate:.0f}s)")

    elapsed = time.time() - t0
    total   = len(primary_records)
    mean_pool = sum(pool_sizes) / len(pool_sizes)

    print(f"\n  Done: {total:,} candidate records in {elapsed:.1f}s")
    print(f"  Mean pool size: {mean_pool:.1f} ordered pairs per question")
    print(f"\n  PRIMARY (llm_attributed):")
    print(f"    lexical_overlap_answer  : {total-primary_fallback:,} ({100*(total-primary_fallback)/total:.1f}%)")
    print(f"    fallback (zero overlap) : {primary_fallback:,} ({100*primary_fallback/total:.1f}%)")
    print(f"    gold selected           : {primary_gold_selected:,} ({100*primary_gold_selected/total:.1f}%)")
    print(f"    NON-gold selected       : {total-primary_gold_selected:,} ({100*(total-primary_gold_selected)/total:.1f}%)")
    print(f"\n  TOP-CHAIN (question-driven):")
    print(f"    lexical_overlap_question: {topchain_lexical:,} ({100*topchain_lexical/total:.1f}%)")
    print(f"    gold selected           : {topchain_gold_selected:,} ({100*topchain_gold_selected/total:.1f}%)")
    print(f"    NON-gold selected       : {total-topchain_gold_selected:,} ({100*(total-topchain_gold_selected)/total:.1f}%)")
    print(f"\n  GOLD: 100% gold selected (by construction)")

    # ── Show examples ─────────────────────────────────────────────────
    print(f"\n  Example PRIMARY record (first non-gold selection found):")
    for r in primary_records:
        if not r["gold_chain_selected"]:
            print(json.dumps(r, indent=4, ensure_ascii=False))
            break

    print(f"\n  Example GOLD record:")
    print(json.dumps(gold_records[0], indent=4, ensure_ascii=False))

    print(f"\n  Example TOP-CHAIN record (first non-gold selection found):")
    for r in topchain_records:
        if not r["gold_chain_selected"]:
            print(json.dumps(r, indent=4, ensure_ascii=False))
            break

    # ── Write ─────────────────────────────────────────────────────────
    if dry_run:
        print(f"\n  [DRY RUN] Would write:")
        print(f"    {len(primary_records):,} records → {out_primary}")
        print(f"    {len(gold_records):,} records → {out_gold}")
        print(f"    {len(topchain_records):,} records → {out_topchain}")
    else:
        write_jsonl(out_primary,   primary_records)
        write_jsonl(out_gold,      gold_records)
        write_jsonl(out_topchain,  topchain_records)

    return {
        "dataset":             dataset,
        "status":              "ok",
        "n_records":           total,
        "mean_pool_size":      round(mean_pool, 1),
        "primary_gold_pct":    round(100 * primary_gold_selected / total, 1),
        "topchain_gold_pct":   round(100 * topchain_gold_selected / total, 1),
        "elapsed_s":           round(elapsed, 1),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 6 — MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--proj_root", default="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot")
    ap.add_argument("--out_dir",   default="phase2")
    ap.add_argument("--dry_run",   action="store_true")
    ap.add_argument("--max_q",     type=int, default=None)
    args = ap.parse_args()

    R  = args.proj_root
    OD = args.out_dir

    print("\n" + "━"*70)
    print("  PHASE 2 — Unified Evidence Builder (all 3 setups × 2 datasets)")
    print("  REPLACES: build_attributed_evidence.py + build_ablation_evidence.py")
    print("  FIX: Primary analysis now uses all_paragraphs pairs, not 1-chain pool")
    print("━"*70)
    if args.dry_run:
        print("  ⚠ DRY RUN — no files will be written\n")

    DATASETS = [
        {
            "dataset":    "hotpotqa",
            "cand_path":  f"{R}/exp_distractor/candidates/dev_M5_sampling.jsonl",
            "ev_path":    f"{R}/exp_distractor/evidence/dev_distractor_chains.jsonl",
            "out_primary":    f"{OD}/hotpotqa/llm_attributed.jsonl",
            "out_gold":       f"{OD}/hotpotqa/gold.jsonl",
            "out_topchain":   f"{OD}/hotpotqa/top_chain.jsonl",
        },
        {
            "dataset":    "wiki2",
            "cand_path":  f"{R}/exp_wiki2/candidates/dev_M5_sampling.jsonl",
            "ev_path":    f"{R}/exp_wiki2/evidence/dev_wiki2_chains.jsonl",
            "out_primary":    f"{OD}/2wiki/llm_attributed.jsonl",
            "out_gold":       f"{OD}/2wiki/gold.jsonl",
            "out_topchain":   f"{OD}/2wiki/top_chain.jsonl",
        },
    ]

    all_stats = []
    for cfg in DATASETS:
        stats = process_dataset(
            dataset      = cfg["dataset"],
            cand_path    = cfg["cand_path"],
            ev_path      = cfg["ev_path"],
            out_primary  = cfg["out_primary"],
            out_gold     = cfg["out_gold"],
            out_topchain = cfg["out_topchain"],
            max_q        = args.max_q,
            dry_run      = args.dry_run,
        )
        all_stats.append(stats)

    print("\n" + "━"*70)
    print("  FINAL SUMMARY")
    print("━"*70)
    print(f"\n  {'Dataset':12s}  {'Records':>8s}  {'Pool':>6s}  "
          f"{'Primary→gold%':>14s}  {'TopChain→gold%':>15s}  {'Time':>6s}")
    print(f"  {'-'*68}")
    for s in all_stats:
        if s["status"] != "ok":
            print(f"  {s['dataset']:12s}  SKIPPED")
            continue
        print(f"  {s['dataset']:12s}  {s['n_records']:>8,}  "
              f"{s['mean_pool_size']:>6.1f}  "
              f"{s['primary_gold_pct']:>13.1f}%  "
              f"{s['topchain_gold_pct']:>14.1f}%  "
              f"{s['elapsed_s']:>5.1f}s")

    print(f"""
  Interpretation guide:
  ┌────────────────────┬────────────────────────────────────────────────┐
  │ Primary→gold%      │ How often the answer points to gold evidence   │
  │                    │ High = correct answers dominate the dataset    │
  │                    │ Low  = many wrong/distractor answers           │
  ├────────────────────┼────────────────────────────────────────────────┤
  │ TopChain→gold%     │ How often the question text points to gold     │
  │                    │ Measures "question-evidence relevance" signal  │
  │                    │ Should differ from Primary→gold% for wrong ans │
  └────────────────────┴────────────────────────────────────────────────┘

  All 6 files share the same schema.
  gold.jsonl:         attribution_score=null, query_text="[gold_labels]"
  llm_attributed:     query_text=answer_text, method=lexical_overlap_answer
  top_chain.jsonl:    query_text=question,    method=lexical_overlap_question
""")


if __name__ == "__main__":
    main()