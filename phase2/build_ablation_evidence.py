#!/usr/bin/env python3
"""
phase2/build_ablation_evidence.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABLATION SETUPS — Gold and Top-Chain Evidence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
    Builds the two ablation evidence datasets to sit alongside the primary
    analysis (llm_attributed.jsonl) in the phase2 folder.

    Ablation 1 — gold.jsonl
        Uses the known-correct 2 gold paragraphs (hop1 = bridge, hop2 = answer).
        Idealized upper bound: gold is always present, Bucket A = 0%.
        Every candidate for the same question sees the same gold chain —
        candidate independence is still preserved because each candidate
        is SCORED independently against those hops.

    Ablation 2 — top_chain.jsonl
        Uses chains[0] from the MDR evidence file (top-ranked retrieved chain).
        Realistic noisy setting — mirrors exactly what the existing NLI/QA/lex
        scorers do. Gold is NOT guaranteed for HotpotQA (Bucket A = 12.9%).
        For 2Wiki, this is the same as gold (no MDR retrieval — chains[0] IS gold).

INPUT FILES
    HotpotQA gold      : exp_distractor/evidence/dev_distractor_chains.jsonl
    HotpotQA top_chain : exp0c/evidence/dev_K200_chains.jsonl
    2Wiki gold         : exp_wiki2/evidence/dev_wiki2_chains.jsonl
    2Wiki top_chain    : exp_wiki2/evidence/dev_wiki2_chains.jsonl  ← SAME FILE
                         (2Wiki has no MDR — chains[0] is always gold)

    Candidates (shared):
    HotpotQA           : exp_distractor/candidates/dev_M5_sampling.jsonl
    2Wiki              : exp_wiki2/candidates/dev_M5_sampling.jsonl

OUTPUT
    phase2/hotpotqa/gold.jsonl
    phase2/hotpotqa/top_chain.jsonl
    phase2/2wiki/gold.jsonl
    phase2/2wiki/top_chain.jsonl

OUTPUT SCHEMA (identical to llm_attributed.jsonl for easy comparison)
    {
        "question_id"        : str,
        "candidate_id"       : str,      # "{qid}_c{answer_id}"
        "answer_text"        : str,
        "used_chains"        : [int],    # always [0] — both setups use one chain
        "hop1_title"         : str,
        "hop1_text"          : str,
        "hop2_title"         : str,
        "hop2_text"          : str,
        "attribution_score"  : null,     # N/A for ablations — evidence is fixed
        "attribution_method" : str,      # "gold" | "top_chain_mdr" | "top_chain_same_as_gold"
        "dataset"            : str,
        "n_chains_available" : int
    }

FINAL PHASE2 FOLDER STRUCTURE
    phase2/
    ├── hotpotqa/
    │   ├── llm_attributed.jsonl   ← PRIMARY  (already built)
    │   ├── gold.jsonl             ← ABLATION 1  (this script)
    │   └── top_chain.jsonl        ← ABLATION 2  (this script)
    └── 2wiki/
        ├── llm_attributed.jsonl   ← PRIMARY  (already built)
        ├── gold.jsonl             ← ABLATION 1  (this script)
        └── top_chain.jsonl        ← ABLATION 2  (this script)

USAGE
    python3 phase2/build_ablation_evidence.py \
        --proj_root /var/tmp/u24sf51014/sro/work/sro-proof-hotpot \
        --out_dir   phase2

    # Test on 100 questions:
    python3 phase2/build_ablation_evidence.py ... --max_q 100

    # Dry run:
    python3 phase2/build_ablation_evidence.py ... --dry_run
"""

import argparse
import json
import os
import re
import string
import time
from typing import Dict, Iterator, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 1 — I/O UTILITIES
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
    """Returns {qid: [{answer_id, answer_text}, ...]}"""
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
    """Returns {qid: evidence_record}"""
    result = {}
    for rec in iter_jsonl(path):
        result[str(rec["qid"])] = rec
    return result


def get_chains(ev_rec: dict) -> List[dict]:
    chains = ev_rec.get("chains")
    if not chains:
        chains = ev_rec.get("evidence", {}).get("chains", [])
    return chains or []


def extract_hops(chain: dict) -> Tuple[str, str, str, str]:
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
#  SECTION 2 — RECORD BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_record(
    qid:     str,
    cand:    dict,
    chain:   dict,
    method:  str,   # "gold" | "top_chain_mdr" | "top_chain_same_as_gold"
    dataset: str,
    n_chains: int,
) -> dict:
    """
    Build one output record.
    Same schema as llm_attributed.jsonl — attribution_score is None
    because evidence is fixed (not answer-dependent) for ablations.
    """
    h1t, h1x, h2t, h2x = extract_hops(chain)
    chain_id = int(chain.get("chain_id", 0))

    return {
        "question_id":        qid,
        "candidate_id":       f"{qid}_c{cand['answer_id']}",
        "answer_text":        cand["answer_text"],
        "used_chains":        [chain_id],
        "hop1_title":         h1t,
        "hop1_text":          h1x,
        "hop2_title":         h2t,
        "hop2_text":          h2x,
        # attribution_score is None for ablations — chain is fixed, not answer-dependent
        "attribution_score":  None,
        "attribution_method": method,
        "dataset":            dataset,
        "n_chains_available": n_chains,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 3 — DATASET PROCESSOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_ablation(
    label:        str,          # e.g. "HotpotQA — gold"
    dataset:      str,          # "hotpotqa" | "wiki2"
    cand_path:    str,
    ev_path:      str,
    out_path:     str,
    method:       str,          # attribution_method value
    note:         str,          # printed as context note
    max_q:        Optional[int],
    dry_run:      bool,
    cand_map:     Optional[Dict] = None,  # pass pre-loaded map to avoid reload
) -> dict:
    """Process one ablation setup for one dataset."""

    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  Evidence   : {ev_path}")
    print(f"  Output     : {out_path}")
    print(f"  Note       : {note}")

    if not os.path.exists(ev_path):
        print(f"  ✗ Evidence file not found — skipping.")
        return {"label": label, "status": "skipped_missing_file"}

    # Load evidence
    ev_map = load_evidence(ev_path)

    # Load candidates only if not pre-loaded
    if cand_map is None:
        if not os.path.exists(cand_path):
            print(f"  ✗ Candidates file not found — skipping.")
            return {"label": label, "status": "skipped_missing_cands"}
        cand_map = load_candidates(cand_path)

    valid_qids = sorted(set(cand_map.keys()) & set(ev_map.keys()))
    if max_q:
        valid_qids = valid_qids[:max_q]

    print(f"  Questions  : {len(valid_qids):,}")

    t0 = time.time()
    records = []
    missing_chain = 0

    for qid in valid_qids:
        candidates = cand_map[qid]
        ev_rec     = ev_map.get(qid, {})
        chains     = get_chains(ev_rec)
        n_chains   = len(chains)

        if not chains:
            # No chain available — emit empty evidence, flag it
            missing_chain += len(candidates)
            for cand in candidates:
                records.append({
                    "question_id":        qid,
                    "candidate_id":       f"{qid}_c{cand['answer_id']}",
                    "answer_text":        cand["answer_text"],
                    "used_chains":        [],
                    "hop1_title":         "",
                    "hop1_text":          "",
                    "hop2_title":         "",
                    "hop2_text":          "",
                    "attribution_score":  None,
                    "attribution_method": f"{method}_no_chain",
                    "dataset":            dataset,
                    "n_chains_available": 0,
                })
            continue

        # Use chains[0] — for gold setting this IS the gold chain;
        # for top_chain setting this IS the MDR top-ranked chain
        top_chain = chains[0]

        for cand in candidates:
            rec = build_record(qid, cand, top_chain, method, dataset, n_chains)
            records.append(rec)

    elapsed = time.time() - t0
    print(f"  Records    : {len(records):,}  ({elapsed:.1f}s)")
    if missing_chain > 0:
        print(f"  ⚠ {missing_chain} candidates had no chain — empty evidence emitted")

    # Show one example
    if records:
        print(f"\n  Example:")
        print(json.dumps(records[0], indent=4, ensure_ascii=False))

    if dry_run:
        print(f"\n  [DRY RUN] Would write {len(records):,} records to {out_path}")
    else:
        write_jsonl(out_path, records)

    return {
        "label":     label,
        "status":    "ok",
        "n_records": len(records),
        "elapsed_s": round(elapsed, 1),
        "out_path":  out_path,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 4 — MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument(
        "--proj_root",
        default="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot",
    )
    ap.add_argument("--out_dir",  default="phase2")
    ap.add_argument("--dry_run",  action="store_true")
    ap.add_argument("--max_q",    type=int, default=None)
    args = ap.parse_args()

    R  = args.proj_root
    OD = args.out_dir

    print("\n" + "━"*70)
    print("  PHASE 2 — ABLATION EVIDENCE: Gold + Top-Chain")
    print("  Output schema identical to llm_attributed.jsonl")
    print("━"*70)
    if args.dry_run:
        print("\n  ⚠ DRY RUN — no files will be written\n")

    # ── Pre-load candidate maps once per dataset (avoid double load) ──
    hotpot_cand_path = f"{R}/exp_distractor/candidates/dev_M5_sampling.jsonl"
    wiki2_cand_path  = f"{R}/exp_wiki2/candidates/dev_M5_sampling.jsonl"

    print("\n  Loading candidate maps ...")

    hotpot_cands = None
    if os.path.exists(hotpot_cand_path):
        hotpot_cands = load_candidates(hotpot_cand_path)
        print(f"    HotpotQA: {len(hotpot_cands):,} questions")
    else:
        print(f"    ✗ HotpotQA candidates not found: {hotpot_cand_path}")

    wiki2_cands = None
    if os.path.exists(wiki2_cand_path):
        wiki2_cands = load_candidates(wiki2_cand_path)
        print(f"    2Wiki   : {len(wiki2_cands):,} questions")
    else:
        print(f"    ✗ 2Wiki candidates not found: {wiki2_cand_path}")

    # ── Define all four ablation jobs ─────────────────────────────────
    #
    #  HotpotQA gold      → exp_distractor/evidence/dev_distractor_chains.jsonl
    #                        chains[0] = 2 gold paragraphs (bridge + answer)
    #                        Bucket A = 0%, gold always present
    #
    #  HotpotQA top_chain → exp0c/evidence/dev_K200_chains.jsonl
    #                        chains[0] = MDR top-ranked retrieved chain
    #                        Bucket A = 12.9% (gold NOT guaranteed)
    #
    #  2Wiki gold         → exp_wiki2/evidence/dev_wiki2_chains.jsonl
    #                        chains[0] = gold chain (always present in 2Wiki dev)
    #
    #  2Wiki top_chain    → exp_wiki2/evidence/dev_wiki2_chains.jsonl  ← SAME FILE
    #                        2Wiki has no MDR retrieval — chains[0] IS the gold chain
    #                        top_chain == gold for this dataset
    #
    JOBS = [
        # ── HotpotQA ──────────────────────────────────────────────────
        dict(
            label    = "HotpotQA — Ablation 1: Gold",
            dataset  = "hotpotqa",
            cand_map = hotpot_cands,
            cand_path= hotpot_cand_path,
            ev_path  = f"{R}/exp_distractor/evidence/dev_distractor_chains.jsonl",
            out_path = f"{OD}/hotpotqa/gold.jsonl",
            method   = "gold",
            note     = "Gold paragraphs guaranteed. Bucket A = 0%. Idealized upper bound.",
        ),
        dict(
            label    = "HotpotQA — Ablation 2: Top-Chain (MDR K=200)",
            dataset  = "hotpotqa",
            cand_map = hotpot_cands,
            cand_path= hotpot_cand_path,
            ev_path  = f"{R}/exp0c/evidence/dev_K200_chains.jsonl",
            out_path = f"{OD}/hotpotqa/top_chain.jsonl",
            method   = "top_chain_mdr",
            note     = "MDR top-ranked chain. Gold NOT guaranteed. Bucket A = 12.9%.",
        ),
        # ── 2WikiMultiHopQA ───────────────────────────────────────────
        dict(
            label    = "2Wiki — Ablation 1: Gold",
            dataset  = "wiki2",
            cand_map = wiki2_cands,
            cand_path= wiki2_cand_path,
            ev_path  = f"{R}/exp_wiki2/evidence/dev_wiki2_chains.jsonl",
            out_path = f"{OD}/2wiki/gold.jsonl",
            method   = "gold",
            note     = "Gold paragraphs always present in 2Wiki dev. Bucket A = 0%.",
        ),
        dict(
            label    = "2Wiki — Ablation 2: Top-Chain (same as gold)",
            dataset  = "wiki2",
            cand_map = wiki2_cands,
            cand_path= wiki2_cand_path,
            ev_path  = f"{R}/exp_wiki2/evidence/dev_wiki2_chains.jsonl",
            out_path = f"{OD}/2wiki/top_chain.jsonl",
            method   = "top_chain_same_as_gold",
            note     = "Same file as gold — 2Wiki has no MDR. chains[0] IS the gold chain.",
        ),
    ]

    # ── Run all jobs ──────────────────────────────────────────────────
    all_stats = []
    for job in JOBS:
        stats = process_ablation(
            label    = job["label"],
            dataset  = job["dataset"],
            cand_map = job["cand_map"],
            cand_path= job["cand_path"],
            ev_path  = job["ev_path"],
            out_path = job["out_path"],
            method   = job["method"],
            note     = job["note"],
            max_q    = args.max_q,
            dry_run  = args.dry_run,
        )
        all_stats.append(stats)

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "━"*70)
    print("  SUMMARY")
    print("━"*70)
    for s in all_stats:
        status = s.get("status", "?")
        if status != "ok":
            print(f"  {s['label']:50s}  {status.upper()}")
        else:
            print(f"  {s['label']:50s}  "
                  f"{s['n_records']:>7,} records  {s['elapsed_s']}s")

    print("\n" + "━"*70)
    print("  COMPLETE PHASE2 FOLDER STRUCTURE")
    print("━"*70)
    print(f"""
  phase2/
  ├── hotpotqa/
  │   ├── llm_attributed.jsonl  ← PRIMARY   (answer-grounded chain selection)
  │   ├── gold.jsonl            ← ABLATION 1 (2 known-correct gold paragraphs)
  │   └── top_chain.jsonl       ← ABLATION 2 (MDR top-ranked chain, K=200)
  └── 2wiki/
      ├── llm_attributed.jsonl  ← PRIMARY   (answer-grounded chain selection)
      ├── gold.jsonl            ← ABLATION 1 (gold paragraphs always present)
      └── top_chain.jsonl       ← ABLATION 2 (same as gold — no MDR for 2Wiki)

  All files share the same schema:
    question_id, candidate_id, answer_text
    used_chains, hop1_title, hop1_text, hop2_title, hop2_text
    attribution_score, attribution_method, dataset, n_chains_available

  Differences between setups (what the verifier sees per candidate):
  ┌─────────────────────┬──────────────────────────────────────────────────┐
  │ llm_attributed      │ Chain most lexically grounded in THIS answer     │
  │                     │ Different candidates → potentially different chain│
  ├─────────────────────┼──────────────────────────────────────────────────┤
  │ gold                │ The 2 known-correct gold paragraphs               │
  │                     │ Same chain for all candidates of this question    │
  ├─────────────────────┼──────────────────────────────────────────────────┤
  │ top_chain           │ MDR's top-ranked retrieved chain (HotpotQA only)  │
  │                     │ Same chain for all candidates of this question    │
  │                     │ 2Wiki: identical to gold (noted in method field)  │
  └─────────────────────┴──────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
    