#!/usr/bin/env python3
"""
sfav_generate_diverse.py — Diverse candidate generation for the SFAV experiment.

Generates M=5 candidates per question using 5 structurally different prompts,
each targeting a different reasoning strategy.  Diversity comes from prompt
variation, not sampling noise (T=0.3 throughout; fallback to T=0.0 on bad output).

Input:  exp_distractor/evidence/dev_distractor_chains.jsonl
Output: exp_distractor/candidates/dev_M5_diverse.jsonl

Output schema per line:
  {
    "qid": str,
    "split": str,
    "candidates": [
      {"answer_id": 0, "answer_text": str, "prompt_id": "direct"},
      {"answer_id": 1, "answer_text": str, "prompt_id": "decomposed"},
      {"answer_id": 2, "answer_text": str, "prompt_id": "entity"},
      {"answer_id": 3, "answer_text": str, "prompt_id": "quote"},
      {"answer_id": 4, "answer_text": str, "prompt_id": "skeptical"},
    ],
    "stats": {"unique_count": int, "bad_count": int, "fallback_count": int}
  }

Usage:
  # Step 1 — start vLLM in a separate terminal (or tmux pane):
  CUDA_VISIBLE_DEVICES=0,1 \\
    /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 \\
      -m vllm.entrypoints.openai.api_server \\
      --model /var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct \\
      --port 8000 --dtype auto --tensor-parallel-size 2 --max-model-len 4096

  # Step 2 — run this script (resume-safe):
  cd /var/tmp/u24sf51014/sro/work/sro-proof-hotpot
  /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/sfav_generate_diverse.py \\
      --evidence  exp_distractor/evidence/dev_distractor_chains.jsonl \\
      --out_jsonl exp_distractor/candidates/dev_M5_diverse.jsonl \\
      --port 8000

  # Step 3 — validate diversity (run after generation completes):
  /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/sfav_generate_diverse.py \\
      --validate_only \\
      --out_jsonl exp_distractor/candidates/dev_M5_diverse.jsonl

Estimated runtime: ~3-4 hours on 2x A100 with TP=2, max_workers=6.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: PROMPT TEMPLATES
# Five structurally different prompts targeting different reasoning strategies.
# Each produces one candidate; together they form the M=5 diverse pool.
# ═══════════════════════════════════════════════════════════════════════════

PROMPTS: dict[str, str] = {

    # Prompt 1 — Direct extraction.
    # Baseline prompt. Targets bridge questions where the answer is a single
    # entity that can be directly extracted from one hop.
    "direct": (
        "Read the passages carefully and answer the question.\n"
        "Give the shortest accurate answer (a name, number, or short phrase).\n"
        "Do not explain.\n\n"
        "Question: {q}\n"
        "Passage 1: {hop1}\n"
        "Passage 2: {hop2}\n\n"
        "Answer:"
    ),

    # Prompt 2 — Decomposed reasoning.
    # Targets compositional and bridge questions where the answer requires
    # chaining facts. Forces the model to commit to an intermediate entity,
    # making errors at intermediate steps surface as different final answers.
    "decomposed": (
        "Break the question into intermediate steps. "
        "Answer each step from the passages, then give the final answer.\n\n"
        "Question: {q}\n"
        "Passage 1: {hop1}\n"
        "Passage 2: {hop2}\n\n"
        "Step 1 (what does the question need first?):\n"
        "Step 2 (answer to Step 1, from the passages):\n"
        "Step 3 (what does the question need given Step 2?):\n"
        "Final answer (short):"
    ),

    # Prompt 3 — Entity-grounded.
    # Targets comparison questions. Forces the model to identify both compared
    # entities and their attributes before committing to an answer.
    "entity": (
        "If this is a comparison question, identify both entities being compared "
        "and the attribute being compared.\n"
        "Otherwise, identify the key entities in the question and find their "
        "relevant attributes in the passages.\n\n"
        "Question: {q}\n"
        "Passage 1: {hop1}\n"
        "Passage 2: {hop2}\n\n"
        "Entities and attributes:\n"
        "Final answer (short):"
    ),

    # Prompt 4 — Quote-grounded.
    # Forces the model to commit to specific evidence sentences before answering.
    # Produces different candidates when multiple sentences are plausibly relevant.
    "quote": (
        "Find the most relevant sentence in each passage. "
        "Write those sentences, then answer the question based on them.\n\n"
        "Question: {q}\n"
        "Passage 1: {hop1}\n"
        "Passage 2: {hop2}\n\n"
        "Most relevant sentence from Passage 1:\n"
        "Most relevant sentence from Passage 2:\n"
        "Answer (short):"
    ),

    # Prompt 5 — Skeptical alternative.
    # Specifically designed to produce non-modal candidates. Even when Prompts
    # 1–4 collapse to the same modal answer, Prompt 5 has structural pressure
    # to produce a different answer.
    "skeptical": (
        "Read the passages and consider the question carefully.\n"
        "What is the most plausible answer? Now consider: is there a less obvious "
        "but better-supported answer?\n"
        "Provide whichever answer is more defensible based on the passages.\n\n"
        "Question: {q}\n"
        "Passage 1: {hop1}\n"
        "Passage 2: {hop2}\n\n"
        "Most defensible answer (short):"
    ),
}

PROMPT_ORDER = ["direct", "decomposed", "entity", "quote", "skeptical"]

# Stop tokens shared with existing pipeline
STOP_TOKENS = ["</final>", "\n\n", "<|im_end|>", "<|endoftext|>"]

# Bad-answer detection (consistent with exp3_generate_candidates.py logic)
_BAD_PATTERNS = [
    r"(?i)^(i\s+(don'?t|do not)\s+know)",
    r"(?i)^(i\s+cannot)",
    r"(?i)^unknown$",
    r"(?i)^n/a$",
    r"(?i)^not\s+(mentioned|stated|given|found|specified|provided)",
    r"(?i)^cannot\s+(determine|find|answer|tell)",
    r"(?i)^(no\s+answer|unanswerable)",
    r"(?i)^the\s+passag",
    r"(?i)^based\s+on\s+(the\s+)?(passag|text|context|information)",
    r"(?i)^(both|neither)\s+of\s+them",
]
_BAD_RE = re.compile("|".join(_BAD_PATTERNS))
_MAX_ANS_WORDS = 20


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: ANSWER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def is_bad_answer(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if len(t.split()) > _MAX_ANS_WORDS:
        return True
    return bool(_BAD_RE.match(t))


def extract_answer(raw: str, prompt_id: str) -> str:
    """Extract short answer from raw model output.

    Tries several heuristics in order:
      1. Last occurrence of an explicit 'Answer:' / 'Final answer:' marker
      2. Last non-empty line
    Strips prefill artifacts.
    """
    text = raw.strip()
    text = re.sub(r"^<final>\s*", "", text)
    text = re.sub(r"</final>.*", "", text, flags=re.DOTALL)
    text = text.strip()

    # Ordered list of answer marker patterns (most specific first)
    markers = [
        r"(?im)most\s+defensible\s+answer[^:\n]*:\s*(.+)",
        r"(?im)final\s+answer[^:\n]*:\s*(.+)",
        r"(?im)^answer[^:\n]*:\s*(.+)",
    ]
    for pat in markers:
        m = re.search(pat, text)
        if m:
            ans = m.group(1).strip().split("\n")[0].strip()
            if ans:
                return ans

    # Fallback: last non-empty line
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def normalize_answer(s: str) -> str:
    """Lowercase, remove articles, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: EVIDENCE LOADING
# Handles both MDR-style chains schema and wiki2-style direct hop texts.
# ═══════════════════════════════════════════════════════════════════════════

def load_evidence(path: str) -> dict[str, dict]:
    """Load evidence file → {qid: {question, hop1, hop2, split}}."""
    evidence: dict[str, dict] = {}
    n_skip = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # Resolve qid
            qid: str = rec.get("_id") or rec.get("qid", "")
            if not qid:
                n_skip += 1
                continue

            question: str = rec.get("question", "")

            # Extract hop texts — try schemas in priority order
            hop1 = hop2 = ""

            # Schema A: direct hop1_text / hop2_text fields
            if "hop1_text" in rec:
                hop1 = rec["hop1_text"]
                hop2 = rec.get("hop2_text", "")

            # Schema B: chains[0].hops[0] / chains[0].hops[1]
            elif "chains" in rec and rec["chains"]:
                hops = rec["chains"][0].get("hops", [])
                if hops:
                    def _hop_text(h: dict) -> str:
                        if "text" in h:
                            return h["text"]
                        if "sentences" in h:
                            return " ".join(h["sentences"])
                        return ""
                    hop1 = _hop_text(hops[0])
                    hop2 = _hop_text(hops[1]) if len(hops) > 1 else ""

            if not hop1:
                n_skip += 1
                continue

            evidence[qid] = {
                "qid": qid,
                "question": question,
                "hop1": hop1.strip(),
                "hop2": hop2.strip(),
                "split": rec.get("split", "dev"),
            }

    if n_skip:
        print(f"[load_evidence] Skipped {n_skip} records (missing qid or hops)")
    return evidence


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def call_model(
    client: OpenAI,
    model_id: str,
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 80,
) -> str:
    """Single chat completion call. Returns raw text."""
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stop=STOP_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_for_question(
    client: OpenAI,
    model_id: str,
    ev: dict,
    temperature: float = 0.3,
) -> dict:
    """Generate one answer per prompt for a single question.

    For each of the 5 prompts:
      1. Call the model at T=0.3 (main attempt)
      2. If the answer is bad, retry with the direct prompt at T=0.0 (fallback)
    """
    q    = ev["question"]
    hop1 = ev["hop1"]
    hop2 = ev["hop2"]
    qid  = ev["qid"]

    candidates = []
    bad_count     = 0
    fallback_count = 0

    for pid in PROMPT_ORDER:
        prompt = PROMPTS[pid].format(q=q, hop1=hop1, hop2=hop2)

        # Main attempt
        raw    = call_model(client, model_id, prompt, temperature=temperature)
        answer = extract_answer(raw, pid)

        # Fallback: bad output → retry direct at T=0
        if is_bad_answer(answer):
            bad_count += 1
            raw2   = call_model(
                client, model_id,
                PROMPTS["direct"].format(q=q, hop1=hop1, hop2=hop2),
                temperature=0.0,
            )
            answer2 = extract_answer(raw2, "direct")
            if not is_bad_answer(answer2):
                answer = answer2
                fallback_count += 1
            # If still bad, keep the bad answer (it will be filtered by Stage 1)

        candidates.append({
            "answer_id": len(candidates),
            "answer_text": answer,
            "prompt_id": pid,
        })

    unique_count = len(set(
        normalize_answer(c["answer_text"])
        for c in candidates
        if not is_bad_answer(c["answer_text"])
    ))

    return {
        "qid": qid,
        "split": ev.get("split", "dev"),
        "candidates": candidates,
        "stats": {
            "unique_count": unique_count,
            "bad_count": bad_count,
            "fallback_count": fallback_count,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: RESUME + IO
# ═══════════════════════════════════════════════════════════════════════════

def load_done_qids(out_path: str) -> set[str]:
    """Return set of qids already written to the output file."""
    done: set[str] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["qid"])
            except Exception:
                pass
    return done


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: DIVERSITY DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════

def print_diversity_report(out_path: str) -> None:
    """Print diversity stats for the generated candidate file."""
    recs = [json.loads(l) for l in open(out_path) if l.strip()]
    total = len(recs)
    if total == 0:
        print("[diversity] No records found.")
        return

    unique_dist = Counter(r["stats"]["unique_count"] for r in recs)
    bad_total   = sum(r["stats"]["bad_count"]      for r in recs)
    fb_total    = sum(r["stats"]["fallback_count"]  for r in recs)

    print(f"\n{'='*60}")
    print(f"  DIVERSITY REPORT  —  {out_path}")
    print(f"  {total} questions")
    print(f"{'='*60}")
    for k in sorted(unique_dist):
        pct = 100 * unique_dist[k] / total
        bar = "█" * int(pct / 2)
        print(f"  unique={k}: {unique_dist[k]:5d}  ({pct:5.1f}%)  {bar}")
    pct_3plus = 100 * sum(v for k, v in unique_dist.items() if k >= 3) / total
    pct_2plus = 100 * sum(v for k, v in unique_dist.items() if k >= 2) / total
    print(f"{'─'*60}")
    print(f"  2+ unique : {pct_2plus:.1f}%")
    print(f"  3+ unique : {pct_3plus:.1f}%  (target ≥40%)")
    status = "PASS" if pct_3plus >= 40 else ("BELOW TARGET" if pct_3plus >= 25 else "FAIL")
    print(f"  Status    : [{status}]")
    print(f"  Bad answers (prefallback): {bad_total} ({100*bad_total/(total*5):.1f}% of all slots)")
    print(f"  Fallback used            : {fb_total}")

    # Per-prompt answer distribution
    all_cands: list[dict] = []
    for r in recs:
        all_cands.extend(r["candidates"])
    print(f"\n  Per-prompt bad-answer rate:")
    for pid in PROMPT_ORDER:
        pc = [c for c in all_cands if c["prompt_id"] == pid]
        n_bad = sum(1 for c in pc if is_bad_answer(c["answer_text"]))
        print(f"    {pid:12s}: {n_bad:4d}/{len(pc)} bad  ({100*n_bad/max(len(pc),1):.1f}%)")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate M=5 diverse candidates for SFAV experiment."
    )
    ap.add_argument("--evidence", default="exp_distractor/evidence/dev_distractor_chains.jsonl",
                    help="Evidence JSONL file (hop1/hop2 texts per question)")
    ap.add_argument("--out_jsonl", default="exp_distractor/candidates/dev_M5_diverse.jsonl",
                    help="Output JSONL path")
    ap.add_argument("--port", type=int, default=8000,
                    help="vLLM server port")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max_workers", type=int, default=6,
                    help="Concurrent requests to vLLM. 6 works well for 7B on 2x A100.")
    ap.add_argument("--max_q", type=int, default=None,
                    help="Limit questions processed (for debugging)")
    ap.add_argument("--validate_only", action="store_true",
                    help="Skip generation; just print diversity report on existing output")
    args = ap.parse_args()

    if args.validate_only:
        if not os.path.exists(args.out_jsonl):
            print(f"ERROR: {args.out_jsonl} does not exist", file=sys.stderr)
            sys.exit(1)
        print_diversity_report(args.out_jsonl)
        return

    # ── Connect to vLLM ──────────────────────────────────────────────────
    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY")
    try:
        models = client.models.list()
        model_id = models.data[0].id
    except Exception as e:
        print(f"ERROR: Cannot connect to vLLM at port {args.port}: {e}", file=sys.stderr)
        print("  → Is vLLM running? Start it with:", file=sys.stderr)
        print("    CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \\", file=sys.stderr)
        print("        --model /var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct \\", file=sys.stderr)
        print("        --port 8000 --dtype auto --tensor-parallel-size 2 --max-model-len 4096", file=sys.stderr)
        sys.exit(1)
    print(f"[sfav_generate] Connected to vLLM. Model: {model_id}")

    # ── Load evidence ────────────────────────────────────────────────────
    print(f"[sfav_generate] Loading evidence: {args.evidence}")
    evidence = load_evidence(args.evidence)
    all_qids = sorted(evidence.keys())
    if args.max_q:
        all_qids = all_qids[: args.max_q]
    print(f"[sfav_generate] {len(all_qids)} questions total")

    # ── Resume ───────────────────────────────────────────────────────────
    done_qids = load_done_qids(args.out_jsonl)
    todo = [q for q in all_qids if q not in done_qids]
    print(f"[sfav_generate] {len(done_qids)} already done, {len(todo)} remaining")

    if not todo:
        print("[sfav_generate] Nothing to do.")
        print_diversity_report(args.out_jsonl)
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)

    # ── Generate ─────────────────────────────────────────────────────────
    n_written = 0
    t0 = time.time()
    n_already = len(done_qids)

    with open(args.out_jsonl, "a") as fout:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    generate_for_question, client, model_id,
                    evidence[qid], args.temperature
                ): qid
                for qid in todo
            }

            for fut in as_completed(futures):
                qid = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    print(f"[sfav_generate] ERROR qid={qid}: {e}", file=sys.stderr)
                    continue

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_written += 1

                if n_written % 50 == 0:
                    elapsed = time.time() - t0
                    rate = n_written / elapsed * 60  # q/min
                    remaining = len(todo) - n_written
                    eta_h = remaining / max(rate / 60, 1e-9) / 3600
                    print(
                        f"[sfav_generate] {n_already + n_written}/{len(all_qids)} "
                        f"| {rate:.0f} q/min | ETA {eta_h:.2f}h"
                    )

    print(f"\n[sfav_generate] Generation complete. {n_written} new records written.")
    print_diversity_report(args.out_jsonl)


if __name__ == "__main__":
    main()