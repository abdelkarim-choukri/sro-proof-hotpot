#!/usr/bin/env python3
"""
phase2/build_all_evidence_v2.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Complete Phase2 Evidence Builder — 3 datasets × 3 setups = 9 files
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATASETS
    1. HotpotQA MDR        exp0c/           ← THE MAIN PIPELINE
       Pool type: K=200 real MDR-retrieved chains
       Gold NOT guaranteed. Bucket A = 12.9%. This is the paper's main result.

    2. HotpotQA Distractor exp_distractor/  ← controlled upper-bound setting
       Pool type: synthetic pairs from all_paragraphs (10 paragraphs)
       Gold guaranteed. Bucket A = 0%.

    3. 2WikiMultiHopQA     exp_wiki2/       ← cross-dataset generalization
       Pool type: synthetic pairs from all_paragraphs (~N paragraphs)
       Gold guaranteed. Bucket A = 0%. No MDR equivalent exists.

THREE SETUPS PER DATASET
    llm_attributed  PRIMARY     answer drives chain selection (per-candidate)
    gold            ABLATION 1  gold chain always (per-question)
    top_chain       ABLATION 2  question drives chain selection (per-question)
                                For MDR: chains[0] = MDR top-ranked (real)
                                For distractor/2Wiki: question-overlap selection

CHAIN POOLS
    MDR pool:       K=200 real chains from exp0c/evidence/dev_K200_chains.jsonl
                    Each chain has 2 hops already. No pair construction needed.
                    chains[0] is MDR's top-ranked chain.
                    Gold chain = whichever chain contains both gold_titles.
                    Gold may be ABSENT (Bucket A questions).

    Synthetic pool: All ordered paragraph pairs built from all_paragraphs.
                    10 paragraphs → 90 pairs. N paragraphs → N*(N-1) pairs.
                    chains[0] in evidence file IS the gold pair.

MDR EVIDENCE SCHEMA (nested — different from distractor/2Wiki)
    {qid, question, evidence: {chains: [{hops: [{title, text}]}]},
     gold: {gold_titles: [...]}}

DISTRACTOR/WIKI2 SCHEMA (flat)
    {qid, question, chains: [{hops: [{title, text}]}],
     all_paragraphs: [{title, text}], gold_titles: [...]}

OUTPUT
    phase2/
    ├── hotpotqa_mdr/
    │   ├── llm_attributed.jsonl
    │   ├── gold.jsonl
    │   └── top_chain.jsonl
    ├── hotpotqa_distractor/
    │   ├── llm_attributed.jsonl
    │   ├── gold.jsonl
    │   └── top_chain.jsonl
    └── 2wiki/
        ├── llm_attributed.jsonl
        ├── gold.jsonl
        └── top_chain.jsonl

OUTPUT SCHEMA (same across all 9 files)
    {
        "question_id"         : str,
        "candidate_id"        : str,
        "answer_text"         : str,
        "used_chains"         : [int],
        "hop1_title"          : str,
        "hop1_text"           : str,
        "hop2_title"          : str,
        "hop2_text"           : str,
        "attribution_score"   : float | null,
        "attribution_method"  : str,
        "query_text"          : str,
        "dataset"             : str,
        "pool_type"           : str,    # "mdr_k200" | "synthetic_pairs"
        "n_chains_available"  : int,
        "gold_chain_selected" : bool,
        "gold_absent"         : bool    # True only for MDR Bucket A questions
    }

USAGE
    python3 phase2/build_all_evidence_v2.py \
        --proj_root /var/tmp/u24sf51014/sro/work/sro-proof-hotpot \
        --out_dir   phase2

    python3 phase2/build_all_evidence_v2.py ... --max_q 100
    python3 phase2/build_all_evidence_v2.py ... --dry_run
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

def token_jaccard(a: str, b: str) -> float:
    na = set(normalize(a).split())
    nb = set(normalize(b).split())
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
#  SECTION 3 — CHAIN EXTRACTION (handles both MDR and distractor schemas)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_hop_texts(chain: dict) -> Tuple[str, str, str, str]:
    """Extract (hop1_title, hop1_text, hop2_title, hop2_text) from a chain."""
    hops = chain.get("hops", [])
    h1t = h1x = h2t = h2x = ""
    if len(hops) >= 1:
        h1t = str(hops[0].get("title", "")).strip()
        h1x = str(hops[0].get("text",  "")).strip()
    if len(hops) >= 2:
        h2t = str(hops[1].get("title", "")).strip()
        h2x = str(hops[1].get("text",  "")).strip()
    return h1t, h1x, h2t, h2x


def get_mdr_chains(ev_rec: dict) -> List[dict]:
    """
    Extract chains from MDR evidence record.
    MDR schema: {evidence: {chains: [...]}}
    Falls back to top-level chains key.
    """
    chains = ev_rec.get("evidence", {}).get("chains")
    if not chains:
        chains = ev_rec.get("chains", [])
    return chains or []


def get_mdr_gold_titles(ev_rec: dict) -> List[str]:
    """Extract gold titles from MDR evidence record."""
    gold = ev_rec.get("gold", {})
    if isinstance(gold, dict):
        return gold.get("gold_titles", [])
    return []


def chain_contains_gold(chain: dict, gold_set: Set[str]) -> bool:
    """Return True if both hop titles appear in gold_set."""
    h1t, _, h2t, _ = extract_hop_texts(chain)
    return (normalize(h1t) in gold_set and normalize(h2t) in gold_set
            and len(gold_set) >= 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 4 — SYNTHETIC PAIR POOL (for distractor + 2Wiki)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_pair_pool(all_paragraphs: List[dict], gold_titles: List[str]) -> List[dict]:
    """
    Build all ordered paragraph pairs as candidate chains.
    N paragraphs → N*(N-1) ordered pairs.
    chain_id = i * N + j, unique and deterministic.
    """
    N = len(all_paragraphs)
    gold_set = {normalize(t) for t in gold_titles}
    pool = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            pi, pj = all_paragraphs[i], all_paragraphs[j]
            h1t = str(pi.get("title", "")).strip()
            h1x = str(pi.get("text",  "")).strip()
            h2t = str(pj.get("title", "")).strip()
            h2x = str(pj.get("text",  "")).strip()
            is_gold = (normalize(h1t) in gold_set and
                       normalize(h2t) in gold_set and len(gold_set) >= 2)
            pool.append({
                "chain_id": i * N + j,
                "hop1_title": h1t, "hop1_text": h1x,
                "hop2_title": h2t, "hop2_text": h2x,
                "is_gold": is_gold,
                "combined_text": h1x + " " + h2x,
                "i": i, "j": j,
            })
    return pool


def select_best_from_pool(pool: List[dict], query: str) -> Tuple[dict, float]:
    """Select chain with highest jaccard(query, combined_text). Tie: (i asc, j asc)."""
    if not pool:
        return {}, 0.0
    scored = sorted(pool, key=lambda c: (-token_jaccard(query, c["combined_text"]),
                                         c.get("i", 0), c.get("j", 0)))
    best = scored[0]
    return best, token_jaccard(query, best["combined_text"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 5 — RECORD BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_record(qid, cand, chain, score, method, query_text,
                dataset, pool_type, n_pool, gold_absent=False) -> dict:
    return {
        "question_id":         qid,
        "candidate_id":        f"{qid}_c{cand['answer_id']}",
        "answer_text":         cand["answer_text"],
        "used_chains":         [chain.get("chain_id", 0)],
        "hop1_title":          chain.get("hop1_title", ""),
        "hop1_text":           chain.get("hop1_text",  ""),
        "hop2_title":          chain.get("hop2_title", ""),
        "hop2_text":           chain.get("hop2_text",  ""),
        "attribution_score":   round(score, 6) if score is not None else None,
        "attribution_method":  method,
        "query_text":          query_text,
        "dataset":             dataset,
        "pool_type":           pool_type,
        "n_chains_available":  n_pool,
        "gold_chain_selected": bool(chain.get("is_gold", False)),
        "gold_absent":         gold_absent,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 6 — MDR DATASET PROCESSOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_mdr(
    cand_path: str, ev_path: str,
    out_primary: str, out_gold: str, out_topchain: str,
    max_q: Optional[int], dry_run: bool,
) -> dict:
    """
    Process HotpotQA MDR setting.

    Pool = K=200 real MDR chains (already structured as 2-hop chains).
    top_chain = chains[0] (MDR's top-ranked chain, same for all candidates).
    gold = whichever chain in K=200 contains both gold_titles.
           If no such chain exists → Bucket A, gold_absent=True.
    primary = lexical overlap over K=200 chains, per-candidate.
    """
    label = "HotpotQA MDR (exp0c — MAIN PIPELINE)"
    pool_type = "mdr_k200"
    dataset = "hotpotqa_mdr"

    print(f"\n{'━'*70}")
    print(f"  {label}")
    print(f"  Candidates : {cand_path}")
    print(f"  Evidence   : {ev_path}")
    print(f"{'━'*70}")

    for p, l in [(cand_path, "candidates"), (ev_path, "evidence")]:
        if not os.path.exists(p):
            print(f"  ✗ {l} not found: {p}")
            return {"dataset": dataset, "status": "skipped"}

    cand_map = load_candidates(cand_path)
    ev_map   = load_evidence(ev_path)

    valid_qids = sorted(set(cand_map.keys()) & set(ev_map.keys()))
    if max_q:
        valid_qids = valid_qids[:max_q]

    # Schema check
    first_ev = ev_map[valid_qids[0]]
    mdr_chains = get_mdr_chains(first_ev)
    gold_titles = get_mdr_gold_titles(first_ev)
    print(f"\n  Schema check (first question):")
    print(f"    MDR chains available : {len(mdr_chains)}")
    print(f"    gold_titles          : {gold_titles}")
    if mdr_chains:
        h1t, h1x, h2t, h2x = extract_hop_texts(mdr_chains[0])
        print(f"    chains[0].hop1       : '{h1t}' ({len(h1x)} chars)")
        print(f"    chains[0].hop2       : '{h2t}' ({len(h2x)} chars)")

    print(f"\n  Processing {len(valid_qids):,} questions ...")
    t0 = time.time()

    primary_recs = []
    gold_recs    = []
    topchain_recs= []

    n_bucket_a = 0
    n_primary_gold = 0
    n_topchain_gold = 0
    n_primary_fallback = 0
    pool_sizes = []

    for idx, qid in enumerate(valid_qids):
        candidates  = cand_map[qid]
        ev_rec      = ev_map[qid]
        question    = ev_rec.get("question", "")
        mdr_chains  = get_mdr_chains(ev_rec)
        gold_titles = get_mdr_gold_titles(ev_rec)
        gold_set    = {normalize(t) for t in gold_titles}
        n_chains    = len(mdr_chains)
        pool_sizes.append(n_chains)

        # ── Build annotated pool from MDR chains ──────────────────────
        pool = []
        for ci, chain in enumerate(mdr_chains):
            h1t, h1x, h2t, h2x = extract_hop_texts(chain)
            is_gold = chain_contains_gold(chain, gold_set)
            pool.append({
                "chain_id": ci,
                "hop1_title": h1t, "hop1_text": h1x,
                "hop2_title": h2t, "hop2_text": h2x,
                "is_gold": is_gold,
                "combined_text": h1x + " " + h2x,
                "i": ci, "j": 0,  # for stable sort
            })

        # ── gold chain: whichever MDR chain has both gold titles ──────
        gold_chains = [c for c in pool if c["is_gold"]]
        gold_absent = len(gold_chains) == 0
        if gold_absent:
            n_bucket_a += 1
            # Bucket A: gold not retrieved — use chains[0] as fallback
            gold_chain = pool[0] if pool else {}
            gold_method = "fallback_top_chain_bucket_a"
        else:
            gold_chain  = gold_chains[0]  # first gold chain found
            gold_method = "gold"

        # ── top_chain: chains[0] — MDR's top-ranked chain (answer-blind)
        top_chain  = pool[0] if pool else {}
        top_method = "mdr_top_ranked"
        if top_chain.get("is_gold"):
            n_topchain_gold += len(candidates)

        for cand in candidates:
            answer = cand["answer_text"]

            # ── PRIMARY: answer-driven lexical overlap over MDR pool ──
            best, best_score = select_best_from_pool(pool, answer)
            if not best or best_score <= 0.0:
                # Fallback: first non-gold chain (or chains[0] if all gold)
                non_gold = [c for c in pool if not c["is_gold"]]
                best = (non_gold[0] if non_gold else pool[0]) if pool else {}
                p_method = "fallback_non_gold_mdr"
                n_primary_fallback += 1
            else:
                p_method = "lexical_overlap_answer_mdr"
            if best.get("is_gold"):
                n_primary_gold += 1

            primary_recs.append(make_record(
                qid, cand, best, best_score, p_method,
                answer, dataset, pool_type, n_chains, gold_absent
            ))

            gold_recs.append(make_record(
                qid, cand, gold_chain, None, gold_method,
                "[gold_titles]", dataset, pool_type, n_chains, gold_absent
            ))

            topchain_recs.append(make_record(
                qid, cand, top_chain, None, top_method,
                question, dataset, pool_type, n_chains, gold_absent
            ))

        if (idx + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"    {idx+1:,}/{len(valid_qids):,}  "
                  f"({rate:.0f} q/s  ETA {(len(valid_qids)-idx-1)/rate:.0f}s)")

    elapsed = time.time() - t0
    total = len(primary_recs)
    mean_pool = sum(pool_sizes) / len(pool_sizes)

    print(f"\n  Done: {total:,} records in {elapsed:.1f}s")
    print(f"\n  Pool statistics:")
    print(f"    mean MDR chains/question : {mean_pool:.1f}")
    print(f"    Bucket A (gold absent)   : {n_bucket_a:,} questions ({100*n_bucket_a/len(valid_qids):.1f}%)")
    print(f"\n  PRIMARY:   gold selected  = {n_primary_gold:,} ({100*n_primary_gold/total:.1f}%)")
    print(f"             fallback       = {n_primary_fallback:,} ({100*n_primary_fallback/total:.1f}%)")
    print(f"  TOP-CHAIN: gold selected  = {n_topchain_gold:,} ({100*n_topchain_gold/total:.1f}%)")
    print(f"  GOLD:      gold selected  = {total - n_bucket_a*len(candidates):,} "
          f"(gold_absent for {n_bucket_a:,} Bucket A questions)")

    if not dry_run:
        write_jsonl(out_primary,  primary_recs)
        write_jsonl(out_gold,     gold_recs)
        write_jsonl(out_topchain, topchain_recs)
    else:
        print(f"\n  [DRY RUN] Would write {total:,} records to each of 3 files")

    return {
        "dataset": dataset, "status": "ok",
        "n_records": total, "mean_pool": round(mean_pool, 1),
        "bucket_a_pct": round(100*n_bucket_a/len(valid_qids), 1),
        "primary_gold_pct": round(100*n_primary_gold/total, 1),
        "topchain_gold_pct": round(100*n_topchain_gold/total, 1),
        "elapsed_s": round(elapsed, 1),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 7 — SYNTHETIC-POOL DATASET PROCESSOR (distractor + 2Wiki)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_synthetic(
    dataset: str, cand_path: str, ev_path: str,
    out_primary: str, out_gold: str, out_topchain: str,
    max_q: Optional[int], dry_run: bool,
) -> dict:
    """
    Process HotpotQA distractor or 2Wiki.
    Pool = all ordered paragraph pairs built from all_paragraphs.
    top_chain = question-driven lexical overlap (same chain per question).
    gold = chains[0] (always gold by construction, same per question).
    primary = answer-driven lexical overlap (per-candidate).
    """
    pool_type = "synthetic_pairs"

    print(f"\n{'━'*70}")
    print(f"  Dataset : {dataset.upper()}")
    print(f"  Candidates : {cand_path}")
    print(f"  Evidence   : {ev_path}")
    print(f"{'━'*70}")

    for p, l in [(cand_path, "candidates"), (ev_path, "evidence")]:
        if not os.path.exists(p):
            print(f"  ✗ {l} not found: {p}")
            return {"dataset": dataset, "status": "skipped"}

    cand_map = load_candidates(cand_path)
    ev_map   = load_evidence(ev_path)

    valid_qids = sorted(set(cand_map.keys()) & set(ev_map.keys()))
    if max_q:
        valid_qids = valid_qids[:max_q]

    first_ev   = ev_map[valid_qids[0]]
    n_paras    = len(first_ev.get("all_paragraphs", []))
    gold_titles_ex = first_ev.get("gold_titles", [])
    print(f"\n  Schema check: {n_paras} all_paragraphs, "
          f"gold_titles={gold_titles_ex}, pool size={n_paras*(n_paras-1)}")

    if n_paras < 2:
        print(f"  ✗ all_paragraphs < 2 — cannot build pair pool")
        return {"dataset": dataset, "status": "skipped"}

    print(f"\n  Processing {len(valid_qids):,} questions ...")
    t0 = time.time()

    primary_recs = []
    gold_recs    = []
    topchain_recs= []

    n_primary_gold   = 0
    n_topchain_gold  = 0
    n_primary_fallback = 0
    pool_sizes = []

    def get_gold_chain_from_chains0(ev_rec):
        chains = ev_rec.get("chains") or ev_rec.get("evidence", {}).get("chains", [])
        if not chains:
            return {}
        c = chains[0]
        h1t, h1x, h2t, h2x = extract_hop_texts(c)
        return {
            "chain_id": 0, "hop1_title": h1t, "hop1_text": h1x,
            "hop2_title": h2t, "hop2_text": h2x, "is_gold": True,
            "combined_text": h1x + " " + h2x, "i": -1, "j": -1,
        }

    for idx, qid in enumerate(valid_qids):
        candidates  = cand_map[qid]
        ev_rec      = ev_map[qid]
        all_paras   = ev_rec.get("all_paragraphs", [])
        gold_titles = ev_rec.get("gold_titles", [])
        question    = ev_rec.get("question", "")

        pool = build_pair_pool(all_paras, gold_titles)
        pool_sizes.append(len(pool))

        gold_chain = get_gold_chain_from_chains0(ev_rec)

        # top_chain: question-driven, same for all candidates
        top_chain, top_score = select_best_from_pool(pool, question)
        top_method = "lexical_overlap_question"
        if not top_chain:
            top_chain, top_score = gold_chain, 0.0
            top_method = "fallback_gold"
        if top_chain.get("is_gold"):
            n_topchain_gold += len(candidates)

        for cand in candidates:
            answer = cand["answer_text"]

            # PRIMARY: answer-driven, per-candidate
            best, best_score = select_best_from_pool(pool, answer)
            if not best or best_score <= 0.0:
                non_gold = sorted([c for c in pool if not c["is_gold"]],
                                  key=lambda c: (c["i"], c["j"]))
                best = non_gold[0] if non_gold else (pool[0] if pool else gold_chain)
                p_method = "fallback_non_gold_pair"
                n_primary_fallback += 1
            else:
                p_method = "lexical_overlap_answer"
            if best.get("is_gold"):
                n_primary_gold += 1

            primary_recs.append(make_record(
                qid, cand, best, best_score, p_method,
                answer, dataset, pool_type, len(pool)
            ))
            gold_recs.append(make_record(
                qid, cand, gold_chain, None, "gold",
                "[gold_labels]", dataset, pool_type, 1
            ))
            topchain_recs.append(make_record(
                qid, cand, top_chain, top_score, top_method,
                question, dataset, pool_type, len(pool)
            ))

        if (idx + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"    {idx+1:,}/{len(valid_qids):,}  "
                  f"({rate:.0f} q/s  ETA {(len(valid_qids)-idx-1)/rate:.0f}s)")

    elapsed = time.time() - t0
    total = len(primary_recs)
    mean_pool = sum(pool_sizes) / len(pool_sizes)

    print(f"\n  Done: {total:,} records in {elapsed:.1f}s")
    print(f"  Mean pool size: {mean_pool:.1f} pairs per question")
    print(f"  PRIMARY:   gold={n_primary_gold:,} ({100*n_primary_gold/total:.1f}%)  "
          f"fallback={n_primary_fallback:,} ({100*n_primary_fallback/total:.1f}%)")
    print(f"  TOP-CHAIN: gold={n_topchain_gold:,} ({100*n_topchain_gold/total:.1f}%)")
    print(f"  GOLD:      100% gold (by construction)")

    if not dry_run:
        write_jsonl(out_primary,  primary_recs)
        write_jsonl(out_gold,     gold_recs)
        write_jsonl(out_topchain, topchain_recs)
    else:
        print(f"\n  [DRY RUN] Would write {total:,} records to each of 3 files")

    return {
        "dataset": dataset, "status": "ok",
        "n_records": total, "mean_pool": round(mean_pool, 1),
        "bucket_a_pct": 0.0,
        "primary_gold_pct": round(100*n_primary_gold/total, 1),
        "topchain_gold_pct": round(100*n_topchain_gold/total, 1),
        "elapsed_s": round(elapsed, 1),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 8 — MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--proj_root", default="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot")
    ap.add_argument("--out_dir",   default="phase2")
    ap.add_argument("--dry_run",   action="store_true")
    ap.add_argument("--max_q",     type=int, default=None)
    args = ap.parse_args()
    R, OD = args.proj_root, args.out_dir

    print("\n" + "━"*70)
    print("  PHASE 2 v2 — All 3 datasets × 3 setups = 9 output files")
    print("  NEW: HotpotQA MDR setting (K=200 real chains) included")
    print("━"*70)
    if args.dry_run:
        print("  ⚠ DRY RUN\n")

    all_stats = []

    # ── 1. HotpotQA MDR (main pipeline) ──────────────────────────────
    s = process_mdr(
        cand_path    = f"{R}/exp0c/candidates/dev_M5_7b_K200.jsonl",
        ev_path      = f"{R}/exp0c/evidence/dev_K200_chains.jsonl",
        out_primary  = f"{OD}/hotpotqa_mdr/llm_attributed.jsonl",
        out_gold     = f"{OD}/hotpotqa_mdr/gold.jsonl",
        out_topchain = f"{OD}/hotpotqa_mdr/top_chain.jsonl",
        max_q=args.max_q, dry_run=args.dry_run,
    )
    all_stats.append(s)

    # ── 2. HotpotQA Distractor (controlled) ──────────────────────────
    s = process_synthetic(
        dataset      = "hotpotqa_distractor",
        cand_path    = f"{R}/exp_distractor/candidates/dev_M5_sampling.jsonl",
        ev_path      = f"{R}/exp_distractor/evidence/dev_distractor_chains.jsonl",
        out_primary  = f"{OD}/hotpotqa_distractor/llm_attributed.jsonl",
        out_gold     = f"{OD}/hotpotqa_distractor/gold.jsonl",
        out_topchain = f"{OD}/hotpotqa_distractor/top_chain.jsonl",
        max_q=args.max_q, dry_run=args.dry_run,
    )
    all_stats.append(s)

    # ── 3. 2WikiMultiHopQA ────────────────────────────────────────────
    s = process_synthetic(
        dataset      = "wiki2",
        cand_path    = f"{R}/exp_wiki2/candidates/dev_M5_sampling.jsonl",
        ev_path      = f"{R}/exp_wiki2/evidence/dev_wiki2_chains.jsonl",
        out_primary  = f"{OD}/2wiki/llm_attributed.jsonl",
        out_gold     = f"{OD}/2wiki/gold.jsonl",
        out_topchain = f"{OD}/2wiki/top_chain.jsonl",
        max_q=args.max_q, dry_run=args.dry_run,
    )
    all_stats.append(s)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "━"*70)
    print("  FINAL SUMMARY")
    print("━"*70)
    print(f"\n  {'Dataset':26s}  {'Records':>8s}  {'Pool':>6s}  "
          f"{'BucketA':>8s}  {'Primary→gold':>13s}  {'Top→gold':>9s}  {'Time':>5s}")
    print(f"  {'─'*80}")
    for s in all_stats:
        if s["status"] != "ok":
            print(f"  {s['dataset']:26s}  SKIPPED")
            continue
        print(f"  {s['dataset']:26s}  {s['n_records']:>8,}  "
              f"{s['mean_pool']:>6.1f}  "
              f"{s['bucket_a_pct']:>7.1f}%  "
              f"{s['primary_gold_pct']:>12.1f}%  "
              f"{s['topchain_gold_pct']:>8.1f}%  "
              f"{s['elapsed_s']:>4.1f}s")

    print(f"""
  Output folder:
  phase2/
  ├── hotpotqa_mdr/          ← MAIN PIPELINE  (K=200 real MDR chains)
  │   ├── llm_attributed.jsonl   primary: answer-grounded, per-candidate
  │   ├── gold.jsonl             gold chain (may be absent in Bucket A)
  │   └── top_chain.jsonl        chains[0] = MDR top-ranked (answer-blind)
  ├── hotpotqa_distractor/   ← controlled setting (synthetic pairs)
  │   ├── llm_attributed.jsonl
  │   ├── gold.jsonl
  │   └── top_chain.jsonl
  └── 2wiki/                 ← cross-dataset (synthetic pairs, no MDR)
      ├── llm_attributed.jsonl
      ├── gold.jsonl
      └── top_chain.jsonl
""")


if __name__ == "__main__":
    main()