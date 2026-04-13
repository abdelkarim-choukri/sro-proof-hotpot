#!/usr/bin/env python3
"""
wiki2_hop_validation.py — Mandatory Hop Mapping Validation (Day 2 Gate)

PURPOSE:
  Before running ANY pipeline steps (generation, scoring, ablations),
  manually verify that the hop1/hop2 mapping produced by
  wiki2_prepare_evidence.py is semantically correct.

  This is the MANDATORY DAY 2 CHECK from the publication plan:
    "Sample 50 examples. Print question, hop1 text, hop2 text, gold answer.
     Manually inspect: does hop1 represent reasoning step 1?
     Does hop2 represent reasoning step 2?"

DECISION GATE:
  ≥ 40/50 clean (80%)  →  proceed with full pipeline
  30–39/50 clean       →  proceed with explicit noise caveat in paper;
                          expect attenuated chain gains (Case B framing)
  < 30/50 clean        →  STOP — fix the mapping in wiki2_prepare_evidence.py
                          before proceeding

  "Clean" means: hop1 and hop2 represent two distinct, meaningful
  reasoning steps, and the question is decomposable using those steps.

SAMPLING STRATEGY:
  50 questions stratified by type (proportional to dev set composition).
  Seed = 42 for reproducibility. Running this script twice gives the
  same 50 questions.

OUTPUT:
  exp_wiki2/evidence/hop_validation.json
    {
      "n_clean": int,           # your Y count
      "n_total": 50,
      "pct_clean": float,
      "decision": "proceed" | "proceed_with_caveat" | "stop_fix_mapping",
      "type_breakdown": {...},  # clean counts per question type
      "examples": [...]         # full annotation records
    }

INTERACTIVE vs AUTO MODE:
  Default: interactive (Y/N prompt for each of 50 examples)
  --auto:  non-interactive, prints all 50 and exits 0
           (use for inspection without decision gate; run without --auto
            to get the actual annotation)

Usage:
  # Interactive (mandatory before running pipeline):
  python3 tools/wiki2_hop_validation.py \\
      --evidence exp_wiki2/evidence/dev_wiki2_chains.jsonl \\
      --out_json exp_wiki2/evidence/hop_validation.json

  # Non-interactive dry-run (inspect examples without annotating):
  python3 tools/wiki2_hop_validation.py \\
      --evidence exp_wiki2/evidence/dev_wiki2_chains.jsonl \\
      --out_json exp_wiki2/evidence/hop_validation.json \\
      --auto
"""

import argparse
import json
import os
import random
import sys
import textwrap


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1: CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

N_SAMPLE    = 50
SEED        = 42
HOP_PREVIEW = 350   # chars shown per hop (enough to judge reasoning step)

KNOWN_TYPES = {"bridge", "comparison", "inference", "compositional"}

# Decision thresholds
GATE_PROCEED        = 40   # ≥ 40/50 → proceed
GATE_PROCEED_CAVEAT = 30   # 30–39/50 → proceed with caveat


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2: DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════

def truncate(text: str, n: int) -> str:
    """Truncate text to n chars, appending '...' if cut."""
    text = text.strip()
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0] + " ..."


def divider(char: str = "─", width: int = 72) -> str:
    return char * width


def print_example(idx: int, total: int, rec: dict, interactive: bool, hop_preview: int = HOP_PREVIEW):
    """Print a single evidence record for manual inspection."""
    qtype  = rec.get("type", "?")
    qid    = rec["qid"]
    q      = rec["question"]
    gold   = rec["gold"]
    hops   = rec["chains"][0]["hops"]
    hop1   = hops[0]
    hop2   = hops[1]
    n_para = rec["flags"].get("n_context_paras", len(rec.get("all_paragraphs", [])))

    print()
    print(divider("═"))
    print(f"  Example {idx}/{total}   [type={qtype}]   [qid={qid}]   [context_paras={n_para}]")
    print(divider("─"))
    print(f"  QUESTION:    {q}")
    print(f"  GOLD ANSWER: {gold}")
    print(divider("·"))
    print(f"  HOP 1 — {hop1['title']}")
    for line in textwrap.wrap(truncate(hop1["text"], hop_preview), width=68):
        print(f"    {line}")
    print(divider("·"))
    print(f"  HOP 2 — {hop2['title']}")
    for line in textwrap.wrap(truncate(hop2["text"], hop_preview), width=68):
        print(f"    {line}")
    print(divider("─"))
    print()
    if interactive:
        print("  Is hop1 → hop2 a clean two-step reasoning chain?")
        print("  (hop1 leads to an intermediate fact; hop2 leads to the answer)")


def ask_yn() -> bool:
    """Prompt until user enters Y or N. Returns True for Y."""
    while True:
        try:
            ans = input("  [Y=clean / N=not clean / S=skip/unsure]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted. Exiting.")
            sys.exit(1)
        if ans in ("Y", "YES"):
            return True
        if ans in ("N", "NO"):
            return False
        if ans in ("S", "SKIP"):
            return None     # type: ignore
        print("  Please enter Y, N, or S.")


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3: STRATIFIED SAMPLING
# ═══════════════════════════════════════════════════════════════════════

def stratified_sample(records: list, n: int, seed: int) -> list:
    """
    Stratified sample of n records proportional to question type.
    If a type is missing, the quota is redistributed to 'bridge'.
    Preserves reproducibility via fixed seed.
    """
    # Group by type
    by_type = {}
    for rec in records:
        t = rec.get("type", "bridge")
        by_type.setdefault(t, []).append(rec)

    total = len(records)
    print(f"  Type distribution in full dev set:")
    for t, group in sorted(by_type.items()):
        print(f"    {t:20s}: {len(group):,}  ({100*len(group)/total:.1f}%)")
    print()

    # Compute quota per type
    quotas = {}
    for t, group in by_type.items():
        frac = len(group) / total
        quotas[t] = max(1, round(frac * n))

    # Adjust so sum = n
    diff = n - sum(quotas.values())
    if diff != 0:
        # Add/remove from the largest group
        largest = max(by_type.keys(), key=lambda t: len(by_type[t]))
        quotas[largest] += diff

    # Sample within each type
    rng = random.Random(seed)
    sampled = []
    for t, quota in sorted(quotas.items()):
        group = by_type.get(t, [])
        k = min(quota, len(group))
        sampled.extend(rng.sample(group, k))
        print(f"    Sampling {k}/{len(group)} from type={t}")

    # Shuffle the combined sample so types are interleaved
    rng.shuffle(sampled)
    print()
    return sampled[:n]


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 4: DECISION GATE
# ═══════════════════════════════════════════════════════════════════════

def compute_decision(n_clean: int, n_total: int) -> str:
    pct = 100.0 * n_clean / n_total if n_total > 0 else 0.0
    if n_clean >= GATE_PROCEED:
        return "proceed"
    elif n_clean >= GATE_PROCEED_CAVEAT:
        return "proceed_with_caveat"
    else:
        return "stop_fix_mapping"


def print_verdict(n_clean: int, n_total: int, n_skipped: int, decision: str):
    pct = 100.0 * n_clean / n_total if n_total > 0 else 0.0
    print()
    print(divider("═"))
    print(f"  VALIDATION RESULT: {n_clean}/{n_total} clean  ({pct:.1f}%)")
    if n_skipped > 0:
        print(f"  (Skipped/unsure: {n_skipped})")
    print(divider("─"))

    if decision == "proceed":
        print(f"  ✓ DECISION: PROCEED")
        print(f"    Hop mapping is clean. Run the full pipeline.")
    elif decision == "proceed_with_caveat":
        print(f"  ⚠  DECISION: PROCEED WITH CAVEAT")
        print(f"    Hop mapping is noisy ({pct:.0f}% clean < {GATE_PROCEED}%).")
        print(f"    Add this to the paper:")
        print(f"    'Per-hop decomposition is noisier on 2WikiMultiHopQA")
        print(f"     than HotpotQA; hop clarity correlates with chain gain'")
        print(f"    Use Case B framing from the publication plan.")
        print(f"    Proceed but run hop clarity analysis after ablations.")
    else:
        print(f"  ✗ DECISION: STOP — FIX MAPPING")
        print(f"    Only {n_clean}/{n_total} ({pct:.0f}%) are clean.")
        print(f"    Do NOT run the pipeline with this evidence.")
        print(f"    Fix wiki2_prepare_evidence.py hop ordering logic first.")
        print(f"    Inspect the N=0 examples above for the failure pattern.")

    print(divider("═"))
    print()


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 5: MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Mandatory hop-mapping validation for 2WikiMultiHopQA evidence"
    )
    ap.add_argument("--evidence", required=True,
        help="Output of wiki2_prepare_evidence.py "
             "(exp_wiki2/evidence/dev_wiki2_chains.jsonl)")
    ap.add_argument("--out_json", required=True,
        help="Where to save the validation results "
             "(exp_wiki2/evidence/hop_validation.json)")
    ap.add_argument("--n_sample", type=int, default=N_SAMPLE,
        help=f"Number of examples to validate (default: {N_SAMPLE})")
    ap.add_argument("--seed", type=int, default=SEED,
        help=f"Random seed for sampling (default: {SEED}; keep at {SEED} for paper)")
    ap.add_argument("--auto", action="store_true",
        help="Non-interactive mode: print all examples, don't prompt for Y/N. "
             "Exits 0 without writing a decision. Use for inspection only.")
    ap.add_argument("--hop_preview", type=int, default=HOP_PREVIEW,
        help=f"Characters shown per hop (default: {HOP_PREVIEW})")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)

    hop_preview = args.hop_preview

    # ── Load evidence ──
    print(f"Loading evidence from {args.evidence} ...")
    records = []
    with open(args.evidence, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  {len(records):,} records loaded")
    if not records:
        print("ERROR: evidence file is empty. Run wiki2_prepare_evidence.py first.")
        sys.exit(1)

    # ── Stratified sample ──
    print(f"\nStratified sampling {args.n_sample} examples (seed={args.seed}) ...")
    sample = stratified_sample(records, args.n_sample, args.seed)
    print(f"  Sample size: {len(sample)}")

    # ── Display and (optionally) annotate ──
    annotations = []
    n_clean   = 0
    n_unclear = 0
    n_dirty   = 0
    n_skipped = 0
    type_clean = {}

    if args.auto:
        print("\n" + divider("═"))
        print("  AUTO MODE — Displaying all examples (no Y/N prompts)")
        print(divider("═"))

    for i, rec in enumerate(sample, 1):
        print_example(i, len(sample), rec, interactive=not args.auto,
                      hop_preview=hop_preview)

        if args.auto:
            label = None
            is_clean_bool = None
        else:
            label_bool = ask_yn()
            if label_bool is True:
                label = "clean"
                n_clean += 1
                t = rec.get("type", "unknown")
                type_clean[t] = type_clean.get(t, 0) + 1
                is_clean_bool = True
            elif label_bool is False:
                label = "not_clean"
                n_dirty += 1
                is_clean_bool = False
            else:
                label = "skipped"
                n_skipped += 1
                is_clean_bool = None

        annotations.append({
            "qid":      rec["qid"],
            "type":     rec.get("type", "?"),
            "question": rec["question"],
            "gold":     rec["gold"],
            "hop1_title": rec["chains"][0]["hops"][0]["title"],
            "hop2_title": rec["chains"][0]["hops"][1]["title"],
            "n_context_paras": rec["flags"].get("n_context_paras", "?"),
            "annotation": label,
            "is_clean":  is_clean_bool,
        })

    if args.auto:
        print()
        print(divider("═"))
        print(f"  AUTO MODE complete — {len(sample)} examples shown.")
        print(f"  Re-run without --auto to annotate and get the decision gate.")
        print(divider("═"))
        sys.exit(0)

    # ── Compute decision and print verdict ──
    n_total   = n_clean + n_dirty + n_skipped
    decision  = compute_decision(n_clean, n_total)
    print_verdict(n_clean, n_total, n_skipped, decision)

    # ── Write results ──
    result = {
        "n_clean":    n_clean,
        "n_dirty":    n_dirty,
        "n_skipped":  n_skipped,
        "n_total":    n_total,
        "pct_clean":  round(100.0 * n_clean / n_total, 1) if n_total else 0.0,
        "decision":   decision,
        "seed":       args.seed,
        "n_sample":   args.n_sample,
        "gate_proceed":        GATE_PROCEED,
        "gate_proceed_caveat": GATE_PROCEED_CAVEAT,
        "type_breakdown_clean": type_clean,
        "examples":   annotations,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {args.out_json}")

    # ── Exit code encodes the decision ──
    if decision == "proceed":
        sys.exit(0)
    elif decision == "proceed_with_caveat":
        sys.exit(2)   # non-zero but distinct from fatal error
    else:
        # stop_fix_mapping
        sys.exit(1)


if __name__ == "__main__":
    main()