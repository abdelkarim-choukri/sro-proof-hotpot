#!/usr/bin/env python3
"""
sfav_generate_diverse_10para.py — Diverse generation with ALL 10 paragraphs.

Same 5 prompts as sfav_generate_diverse.py but instead of oracle hop1/hop2,
the generator receives all 10 paragraphs (2 gold + 8 distractors).
This is the realistic deployment setting.

Input:  exp_distractor/evidence/dev_distractor_chains.jsonl  (has all_paragraphs field)
Output: exp_distractor/candidates/dev_M5_diverse_10para.jsonl

Usage:
  # Step 1: start vLLM
  CUDA_VISIBLE_DEVICES=0,1 \
    /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 \
    -m vllm.entrypoints.openai.api_server \
    --model /var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct \
    --port 8000 --dtype auto --tensor-parallel-size 2 --max-model-len 4096

  # Step 2: generate
  /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/sfav_generate_diverse_10para.py

  # Step 3: validate diversity
  /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/sfav_generate_diverse_10para.py --validate_only
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

from openai import OpenAI


# ═══════════════════════════════════════════════════════════════════════
# PROMPTS — same 5 strategies, adapted for 10-paragraph input
# ═══════════════════════════════════════════════════════════════════════

PROMPTS = {
    "direct": (
        "You are given a question and 10 passages. Most passages are irrelevant distractors.\n"
        "Find the relevant passages, read them carefully, and answer the question.\n"
        "Give the shortest accurate answer (a name, number, or short phrase).\n"
        "Do not explain.\n\n"
        "Question: {q}\n\n"
        "{passages}\n\n"
        "Answer:"
    ),
    "decomposed": (
        "You are given a question and 10 passages. Most passages are irrelevant.\n"
        "Break the question into intermediate steps. Answer each step from the passages, "
        "then give the final answer.\n\n"
        "Question: {q}\n\n"
        "{passages}\n\n"
        "Step 1 (what does the question need first?):\n"
        "Step 2 (answer to Step 1, from the passages):\n"
        "Step 3 (what does the question need given Step 2?):\n"
        "Final answer (short):"
    ),
    "entity": (
        "You are given a question and 10 passages. Most are distractors.\n"
        "If this is a comparison question, identify both entities being compared "
        "and the attribute being compared.\n"
        "Otherwise, identify the key entities in the question and find their "
        "relevant attributes in the passages.\n\n"
        "Question: {q}\n\n"
        "{passages}\n\n"
        "Entities and attributes:\n"
        "Final answer (short):"
    ),
    "quote": (
        "You are given a question and 10 passages. Most are distractors.\n"
        "First, identify which passages are relevant. "
        "Then find the most relevant sentence in each relevant passage. "
        "Write those sentences, then answer the question based on them.\n\n"
        "Question: {q}\n\n"
        "{passages}\n\n"
        "Relevant passages and key sentences:\n"
        "Answer (short):"
    ),
    "skeptical": (
        "You are given a question and 10 passages. Most are distractors — "
        "be careful not to be misled by irrelevant information.\n"
        "What is the most plausible answer? Now consider: is there a less obvious "
        "but better-supported answer?\n"
        "Provide whichever answer is more defensible based on the passages.\n\n"
        "Question: {q}\n\n"
        "{passages}\n\n"
        "Most defensible answer (short):"
    ),
}

PROMPT_ORDER = ["direct", "decomposed", "entity", "quote", "skeptical"]
STOP_TOKENS = ["</final>", "\n\n", "<|im_end|>", "<|endoftext|>"]

_BAD_RE = re.compile("|".join([
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
]))
_MAX_ANS_WORDS = 20


def is_bad_answer(text):
    t = text.strip()
    if not t or len(t.split()) > _MAX_ANS_WORDS:
        return True
    return bool(_BAD_RE.match(t))


def extract_answer(raw, prompt_id):
    text = raw.strip()
    text = re.sub(r"^<final>\s*", "", text)
    text = re.sub(r"</final>.*", "", text, flags=re.DOTALL).strip()
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
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def format_passages(paragraphs):
    """Format 10 paragraphs into numbered passage text."""
    parts = []
    for i, para in enumerate(paragraphs):
        title = para.get("title", "")
        text = para.get("text", "")
        parts.append(f"Passage {i+1} ({title}): {text}")
    return "\n\n".join(parts)


def load_evidence_10para(path):
    """Load evidence with all 10 paragraphs."""
    evidence = {}
    n_skip = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("qid") or rec.get("_id", "")
            if not qid:
                n_skip += 1
                continue
            paras = rec.get("all_paragraphs", [])
            if not paras:
                n_skip += 1
                continue
            evidence[qid] = {
                "qid": qid,
                "question": rec.get("question", ""),
                "passages_text": format_passages(paras),
                "n_paragraphs": len(paras),
                "split": rec.get("split", "dev"),
            }
    if n_skip:
        print(f"[load] Skipped {n_skip} records")
    return evidence


def call_model(client, model_id, prompt, temperature=0.3, max_tokens=80):
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stop=STOP_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_for_question(client, model_id, ev, temperature=0.3):
    q = ev["question"]
    passages = ev["passages_text"]
    qid = ev["qid"]

    candidates = []
    bad_count = fallback_count = 0

    for pid in PROMPT_ORDER:
        prompt = PROMPTS[pid].format(q=q, passages=passages)
        raw = call_model(client, model_id, prompt, temperature=temperature)
        answer = extract_answer(raw, pid)

        if is_bad_answer(answer):
            bad_count += 1
            raw2 = call_model(
                client, model_id,
                PROMPTS["direct"].format(q=q, passages=passages),
                temperature=0.0,
            )
            answer2 = extract_answer(raw2, "direct")
            if not is_bad_answer(answer2):
                answer = answer2
                fallback_count += 1

        candidates.append({
            "answer_id": len(candidates),
            "answer_text": answer,
            "prompt_id": pid,
        })

    unique_count = len(set(
        normalize_answer(c["answer_text"])
        for c in candidates if not is_bad_answer(c["answer_text"])
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


def load_done_qids(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            if line.strip():
                try:
                    done.add(json.loads(line)["qid"])
                except:
                    pass
    return done


def print_diversity_report(out_path):
    recs = [json.loads(l) for l in open(out_path) if l.strip()]
    total = len(recs)
    if total == 0:
        print("[diversity] No records.")
        return

    unique_dist = Counter(r["stats"]["unique_count"] for r in recs)
    bad_total = sum(r["stats"]["bad_count"] for r in recs)
    fb_total = sum(r["stats"]["fallback_count"] for r in recs)

    print(f"\n{'='*60}")
    print(f"  DIVERSITY REPORT (10-paragraph setting)")
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
    print(f"  3+ unique : {pct_3plus:.1f}%")
    print(f"  Bad answers (pre-fallback): {bad_total} ({100*bad_total/(total*5):.1f}%)")
    print(f"  Fallback used: {fb_total}")

    # Per-prompt bad rate
    all_cands = []
    for r in recs:
        all_cands.extend(r["candidates"])
    print(f"\n  Per-prompt bad-answer rate:")
    for pid in PROMPT_ORDER:
        pc = [c for c in all_cands if c["prompt_id"] == pid]
        n_bad = sum(1 for c in pc if is_bad_answer(c["answer_text"]))
        print(f"    {pid:12s}: {n_bad:4d}/{len(pc)} bad ({100*n_bad/max(len(pc),1):.1f}%)")

    # Compute EM against gold
    try:
        gold_data = json.load(open("data/hotpotqa/raw/hotpot_dev_distractor_v1.json"))
        gold_map = {str(ex["_id"]): ex["answer"] for ex in gold_data}

        # Majority vote EM
        correct_mv = 0
        correct_oracle = 0
        for r in recs:
            qid = r["qid"]
            gold = gold_map.get(qid, "")
            # Majority vote
            answers = [normalize_answer(c["answer_text"]) for c in r["candidates"]
                       if not is_bad_answer(c["answer_text"])]
            if answers:
                majority = Counter(answers).most_common(1)[0][0]
                if majority == normalize_answer(gold):
                    correct_mv += 1
            # Oracle
            if any(normalize_answer(c["answer_text"]) == normalize_answer(gold)
                   for c in r["candidates"]):
                correct_oracle += 1

        print(f"\n  Majority vote EM: {correct_mv}/{total} = {100*correct_mv/total:.2f}%")
        print(f"  Oracle@5 EM:      {correct_oracle}/{total} = {100*correct_oracle/total:.2f}%")
    except Exception as e:
        print(f"\n  [Could not compute EM: {e}]")

    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence",
                    default="exp_distractor/evidence/dev_distractor_chains.jsonl")
    ap.add_argument("--out_jsonl",
                    default="exp_distractor/candidates/dev_M5_diverse_10para.jsonl")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max_workers", type=int, default=6)
    ap.add_argument("--max_q", type=int, default=None)
    ap.add_argument("--validate_only", action="store_true")
    args = ap.parse_args()

    if args.validate_only:
        if not os.path.exists(args.out_jsonl):
            print(f"ERROR: {args.out_jsonl} not found")
            sys.exit(1)
        print_diversity_report(args.out_jsonl)
        return

    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY")
    try:
        models = client.models.list()
        model_id = models.data[0].id
    except Exception as e:
        print(f"ERROR: Cannot connect to vLLM: {e}")
        sys.exit(1)
    print(f"[10para] Connected. Model: {model_id}")

    print(f"[10para] Loading evidence (all 10 paragraphs)...")
    evidence = load_evidence_10para(args.evidence)
    all_qids = sorted(evidence.keys())
    if args.max_q:
        all_qids = all_qids[:args.max_q]
    print(f"[10para] {len(all_qids)} questions")

    done_qids = load_done_qids(args.out_jsonl)
    todo = [q for q in all_qids if q not in done_qids]
    print(f"[10para] {len(done_qids)} done, {len(todo)} remaining")

    if not todo:
        print("[10para] Nothing to do.")
        print_diversity_report(args.out_jsonl)
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)

    n_written = 0
    t0 = time.time()

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
                    print(f"[10para] ERROR qid={qid}: {e}", file=sys.stderr)
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_written += 1
                if n_written % 50 == 0:
                    elapsed = time.time() - t0
                    rate = n_written / elapsed * 60
                    remaining = len(todo) - n_written
                    eta_h = remaining / max(rate / 60, 1e-9) / 3600
                    print(f"[10para] {len(done_qids)+n_written}/{len(all_qids)} "
                          f"| {rate:.0f} q/min | ETA {eta_h:.2f}h")

    print(f"\n[10para] Done. {n_written} new records.")
    print_diversity_report(args.out_jsonl)


if __name__ == "__main__":
    main()