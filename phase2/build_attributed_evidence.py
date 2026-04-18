#!/usr/bin/env python3
"""
phase2/build_attributed_evidence.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIMARY ANALYSIS — LLM-Attributed Evidence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
    Post-hoc attribution of evidence chains to candidate answers.
    For each candidate answer, find the single retrieved chain whose
    hop texts are most lexically grounded in that answer. Store those
    two hop passages as hop1_text / hop2_text.

    This is Option B — no vLLM re-run required. Uses existing candidates
    and evidence files that are already on disk.

WHY THIS APPROACH IS VALID
    True LLM chain attribution (used_chains) does not exist in the pipeline.
    The LLM never logs which passages it cited. Asking it to self-report via
    prompt engineering degrades oracle performance by −2.6pp to −3.1pp
    (confirmed in Exp3/Exp5a — see llm_verifier_pilot_report.docx).

    Lexical overlap between the candidate answer and each chain's hop texts
    is the best available post-hoc signal. It is honest about what it is
    and costs nothing. Each candidate is evaluated INDEPENDENTLY — no
    cross-candidate features, no voting signals, no answer_freq.

ATTRIBUTION METHOD
    For each candidate answer `a` and each retrieved chain `c`:
        score(a, c) = token_jaccard(a, hop1_text(c) + hop2_text(c))

    where token_jaccard = |tokens(a) ∩ tokens(c)| / |tokens(a) ∪ tokens(c)|
    after lowercasing and stripping punctuation/articles.

    The chain with the highest score is selected as `used_chains[0]`.
    Its hops are stored as hop1_text and hop2_text.

    If all scores are zero (answer tokens appear in no chain), the
    top-ranked chain (chains[0]) is used as fallback, and
    attribution_method is set to "fallback_top_chain".

INPUT FILES
    HotpotQA:
        candidates : exp_distractor/candidates/dev_M5_sampling.jsonl
        evidence   : exp_distractor/evidence/dev_distractor_chains.jsonl

    2WikiMultiHopQA:
        candidates : exp_wiki2/candidates/dev_M5_sampling.jsonl
        evidence   : exp_wiki2/evidence/dev_wiki2_chains.jsonl

    Candidate schema   : {qid, candidates: [{answer_id, answer_text}]}
    Evidence schema    : {qid, question, chains: [{chain_id, hops: [{hop, title, text}]}]}

OUTPUT FILES
    phase2/hotpotqa/llm_attributed.jsonl
    phase2/2wiki/llm_attributed.jsonl

OUTPUT SCHEMA (one record per candidate)
    {
        "question_id"        : str,      # original qid
        "candidate_id"       : str,      # "{qid}_c{answer_id}"
        "answer_text"        : str,      # the candidate answer
        "used_chains"        : [int],    # [chain_id of attributed chain]
        "hop1_title"         : str,      # title of hop 1
        "hop1_text"          : str,      # text of hop 1
        "hop2_title"         : str,      # title of hop 2
        "hop2_text"          : str,      # text of hop 2
        "attribution_score"  : float,    # jaccard score (0.0 = fallback used)
        "attribution_method" : str,      # "lexical_overlap" | "fallback_top_chain"
        "dataset"            : str,      # "hotpotqa" | "wiki2"
        "n_chains_available" : int       # total chains in evidence for this question
    }

USAGE
    python3 phase2/build_attributed_evidence.py \
        --proj_root /var/tmp/u24sf51014/sro/work/sro-proof-hotpot \
        --out_dir   phase2

    # Dry run — schema check and 3-example preview only:
    python3 phase2/build_attributed_evidence.py ... --dry_run

    # Limit to N questions per dataset (for testing):
    python3 phase2/build_attributed_evidence.py ... --max_q 100

REPRODUCIBILITY
    No randomness. Attribution is deterministic: ties broken by
    lowest chain_id (stable sort). Seed: N/A.
"""

import argparse
import collections
import json
import os
import re
import string
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 1 — TEXT UTILITIES
#  Identical normalisation to every other tool in this repo.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def normalize(s: str) -> str:
    """Lowercase, strip articles/punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def token_jaccard(text_a: str, text_b: str) -> float:
    """
    Token-level Jaccard similarity between two strings.
    Returns 0.0 if either string normalises to empty.

    Used as the attribution score: how much do the tokens of the
    candidate answer overlap with the combined tokens of the chain's
    hop texts?
    """
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
    """
    Load candidates file.
    Handles both schemas observed in the pipeline:
      {qid, candidates: [{answer_id, answer_text}]}   ← exp_wiki2, exp_distractor
      {qid, candidates: [{answer_text}]}               ← older format

    Returns: {qid: [{answer_id, answer_text}, ...]}
    """
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
    """
    Load evidence file.
    Handles both evidence schema variants:
      top-level chains key              ← exp_wiki2, exp_distractor
      nested evidence.chains key        ← older MDR format

    Returns: {qid: evidence_record}
    """
    result = {}
    for rec in iter_jsonl(path):
        qid = str(rec["qid"])
        result[qid] = rec
    return result


def get_chains(ev_rec: dict) -> List[dict]:
    """Retrieve chain list, handling both schema variants."""
    chains = ev_rec.get("chains")
    if not chains:
        chains = ev_rec.get("evidence", {}).get("chains", [])
    return chains or []


def extract_hops(chain: dict) -> Tuple[str, str, str, str]:
    """
    Extract (hop1_title, hop1_text, hop2_title, hop2_text) from a chain.
    Returns empty strings for missing hops.
    """
    hops = chain.get("hops", [])
    h1t = h1x = h2t = h2x = ""
    if len(hops) >= 1:
        h1t = str(hops[0].get("title", "")).strip()
        h1x = str(hops[0].get("text",  "")).strip()
    if len(hops) >= 2:
        h2t = str(hops[1].get("title", "")).strip()
        h2x = str(hops[1].get("text",  "")).strip()
    return h1t, h1x, h2t, h2x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 3 — CORE ATTRIBUTION LOGIC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def attribute_candidate(
    qid:      str,
    cand:     dict,
    chains:   List[dict],
    dataset:  str,
) -> dict:
    """
    Attribute one candidate answer to its best-matching chain.

    Each candidate is treated as completely independent — no knowledge
    of other candidates, no voting, no frequency features.

    Returns one output record in the phase2 schema.
    """
    answer      = cand["answer_text"]
    answer_id   = cand["answer_id"]
    candidate_id = f"{qid}_c{answer_id}"

    # ── No chains available ───────────────────────────────────────────
    if not chains:
        return {
            "question_id":        qid,
            "candidate_id":       candidate_id,
            "answer_text":        answer,
            "used_chains":        [],
            "hop1_title":         "",
            "hop1_text":          "",
            "hop2_title":         "",
            "hop2_text":          "",
            "attribution_score":  0.0,
            "attribution_method": "no_chains",
            "dataset":            dataset,
            "n_chains_available": 0,
        }

    # ── Score each chain ──────────────────────────────────────────────
    # Score = Jaccard overlap between answer tokens and
    #         combined (hop1_text + " " + hop2_text) tokens.
    # Ties broken by chain position (lower index wins) — stable sort.
    best_score    = -1.0
    best_chain_id = 0
    best_h1t = best_h1x = best_h2t = best_h2x = ""

    for chain in chains:
        cid = int(chain.get("chain_id", 0))
        h1t, h1x, h2t, h2x = extract_hops(chain)
        combined = h1x + " " + h2x
        score = token_jaccard(answer, combined)

        if score > best_score:
            best_score    = score
            best_chain_id = cid
            best_h1t, best_h1x = h1t, h1x
            best_h2t, best_h2x = h2t, h2x

    # ── Fallback: if all scores are zero, use top-ranked chain ────────
    if best_score <= 0.0:
        top = chains[0]
        best_chain_id = int(top.get("chain_id", 0))
        best_h1t, best_h1x, best_h2t, best_h2x = extract_hops(top)
        method = "fallback_top_chain"
    else:
        method = "lexical_overlap"

    return {
        "question_id":        qid,
        "candidate_id":       candidate_id,
        "answer_text":        answer,
        "used_chains":        [best_chain_id],
        "hop1_title":         best_h1t,
        "hop1_text":          best_h1x,
        "hop2_title":         best_h2t,
        "hop2_text":          best_h2x,
        "attribution_score":  round(best_score, 6),
        "attribution_method": method,
        "dataset":            dataset,
        "n_chains_available": len(chains),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 4 — DATASET PROCESSOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_dataset(
    dataset:       str,
    cand_path:     str,
    ev_path:       str,
    out_path:      str,
    max_q:         Optional[int],
    dry_run:       bool,
    n_examples:    int = 2,
) -> dict:
    """
    Process one dataset end-to-end.
    Returns a stats dict for the final summary.
    """
    print(f"\n{'━'*70}")
    print(f"  Dataset : {dataset.upper()}")
    print(f"  Candidates : {cand_path}")
    print(f"  Evidence   : {ev_path}")
    print(f"  Output     : {out_path}")
    print(f"{'━'*70}")

    # ── Schema check ─────────────────────────────────────────────────
    for p, label in [(cand_path, "candidates"), (ev_path, "evidence")]:
        if not os.path.exists(p):
            print(f"\n  ✗ {label} file not found: {p}")
            print(f"    Skipping {dataset}.")
            return {"dataset": dataset, "status": "skipped_missing_file"}

    print("\n  [1/3] Schema check ...")
    first_cand = next(iter_jsonl(cand_path))
    first_ev   = next(iter_jsonl(ev_path))
    chains_0   = get_chains(first_ev)
    print(f"    candidates keys  : {list(first_cand.keys())}")
    print(f"    candidates[0]    : {first_cand['candidates'][0] if first_cand.get('candidates') else 'N/A'}")
    print(f"    evidence keys    : {list(first_ev.keys())}")
    print(f"    chains available : {len(chains_0)} for first question")
    if chains_0:
        h1t, h1x, h2t, h2x = extract_hops(chains_0[0])
        print(f"    chains[0].hop1   : '{h1t}' ({len(h1x)} chars)")
        print(f"    chains[0].hop2   : '{h2t}' ({len(h2x)} chars)")

    # ── Load data ─────────────────────────────────────────────────────
    print("\n  [2/3] Loading data ...")
    cand_map = load_candidates(cand_path)
    ev_map   = load_evidence(ev_path)

    valid_qids = sorted(set(cand_map.keys()) & set(ev_map.keys()))
    if max_q:
        valid_qids = valid_qids[:max_q]

    print(f"    candidates     : {len(cand_map):,} questions")
    print(f"    evidence       : {len(ev_map):,} questions")
    print(f"    valid overlap  : {len(valid_qids):,} questions")

    # ── Attribution ───────────────────────────────────────────────────
    print(f"\n  [3/3] Attributing evidence (treating each candidate independently) ...")
    t0 = time.time()

    records   = []
    stats = collections.Counter()

    for i, qid in enumerate(valid_qids):
        candidates = cand_map[qid]
        ev_rec     = ev_map[qid]
        chains     = get_chains(ev_rec)

        for cand in candidates:
            rec = attribute_candidate(qid, cand, chains, dataset)
            records.append(rec)
            stats[rec["attribution_method"]] += 1

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(valid_qids) - i - 1) / rate
            print(f"    {i+1:,}/{len(valid_qids):,} questions  "
                  f"({rate:.0f} q/s  ETA {remaining:.0f}s)")

    elapsed = time.time() - t0
    print(f"    Done: {len(records):,} candidate records in {elapsed:.1f}s")

    # ── Attribution quality summary ───────────────────────────────────
    total = len(records)
    lexical   = stats.get("lexical_overlap", 0)
    fallback  = stats.get("fallback_top_chain", 0)
    no_chains = stats.get("no_chains", 0)

    scores = [r["attribution_score"] for r in records if r["attribution_method"] == "lexical_overlap"]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\n  Attribution quality:")
    print(f"    lexical_overlap   : {lexical:,} ({100*lexical/total:.1f}%) — answer found in chain")
    print(f"    fallback_top_chain: {fallback:,} ({100*fallback/total:.1f}%) — answer tokens absent, used chains[0]")
    print(f"    no_chains         : {no_chains:,} ({100*no_chains/total:.1f}%) — evidence missing")
    print(f"    mean jaccard (lexical only): {mean_score:.4f}")

    # ── Show examples ─────────────────────────────────────────────────
    print(f"\n  Example records:")
    for rec in records[:n_examples]:
        print(f"\n  {'─'*60}")
        print(json.dumps(rec, indent=4, ensure_ascii=False))

    # ── Write output ─────────────────────────────────────────────────
    if dry_run:
        print(f"\n  [DRY RUN] Would write {len(records):,} records to {out_path}")
    else:
        write_jsonl(out_path, records)

    return {
        "dataset":          dataset,
        "status":           "ok",
        "n_questions":      len(valid_qids),
        "n_records":        len(records),
        "lexical_pct":      round(100 * lexical / total, 1),
        "fallback_pct":     round(100 * fallback / total, 1),
        "mean_jaccard":     round(mean_score, 4),
        "elapsed_s":        round(elapsed, 1),
        "out_path":         out_path,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 5 — MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument(
        "--proj_root",
        default="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot",
        help="Project root directory (default: pipeline project root)",
    )
    ap.add_argument(
        "--out_dir",
        default="phase2",
        help="Output directory root (default: phase2)",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Schema check + examples only. No output files written.",
    )
    ap.add_argument(
        "--max_q",
        type=int,
        default=None,
        help="Limit to first N questions per dataset (for testing).",
    )
    ap.add_argument(
        "--n_examples",
        type=int,
        default=2,
        help="Number of example records to print per dataset (default: 2).",
    )
    args = ap.parse_args()

    R = args.proj_root

    # ── Dataset configurations ────────────────────────────────────────
    DATASETS = [
        {
            "dataset":    "hotpotqa",
            "cand_path":  f"{R}/exp_distractor/candidates/dev_M5_sampling.jsonl",
            "ev_path":    f"{R}/exp_distractor/evidence/dev_distractor_chains.jsonl",
            "out_path":   os.path.join(args.out_dir, "hotpotqa", "llm_attributed.jsonl"),
        },
        {
            "dataset":    "wiki2",
            "cand_path":  f"{R}/exp_wiki2/candidates/dev_M5_sampling.jsonl",
            "ev_path":    f"{R}/exp_wiki2/evidence/dev_wiki2_chains.jsonl",
            "out_path":   os.path.join(args.out_dir, "2wiki", "llm_attributed.jsonl"),
        },
    ]

    print("\n" + "━"*70)
    print("  PHASE 2 — PRIMARY ANALYSIS: LLM-Attributed Evidence")
    print("  Method: post-hoc lexical overlap (Option B)")
    print("  Each candidate treated independently (M=1 philosophy)")
    print("━"*70)

    if args.dry_run:
        print("\n  ⚠ DRY RUN — no files will be written\n")

    all_stats = []
    for cfg in DATASETS:
        stats = process_dataset(
            dataset    = cfg["dataset"],
            cand_path  = cfg["cand_path"],
            ev_path    = cfg["ev_path"],
            out_path   = cfg["out_path"],
            max_q      = args.max_q,
            dry_run    = args.dry_run,
            n_examples = args.n_examples,
        )
        all_stats.append(stats)

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "━"*70)
    print("  SUMMARY")
    print("━"*70)
    for s in all_stats:
        if s["status"] == "skipped_missing_file":
            print(f"  {s['dataset']:12s}  SKIPPED (missing file)")
            continue
        print(f"  {s['dataset']:12s}  "
              f"{s['n_records']:>7,} records  "
              f"({s['n_questions']:,} questions)  "
              f"lexical={s['lexical_pct']}%  "
              f"fallback={s['fallback_pct']}%  "
              f"mean_jaccard={s['mean_jaccard']:.4f}  "
              f"{s['elapsed_s']}s")

    if not args.dry_run:
        print("\n  Output files:")
        for s in all_stats:
            if s["status"] == "ok":
                print(f"    {s['out_path']}")

    print("\n  Schema reminder for downstream verification:")
    print("  Each record contains:")
    print("    question_id, candidate_id, answer_text  — identity")
    print("    used_chains                              — [chain_id] attributed")
    print("    hop1_title, hop1_text                   — first hop passage")
    print("    hop2_title, hop2_text                   — second hop passage")
    print("    attribution_score, attribution_method   — provenance")
    print("    dataset, n_chains_available             — metadata")
    print()


if __name__ == "__main__":
    main()