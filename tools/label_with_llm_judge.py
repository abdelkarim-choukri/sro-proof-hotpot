#!/usr/bin/env python3
"""
label_with_llm_judge.py — Label candidates using LLM-as-judge.

For each (question, candidate_answer, gold_answer), asks Qwen-7B:
  "Is the candidate answer equivalent to the correct answer?"

This replaces EM matching with semantic equivalence judging, correctly
handling paraphrases ("Yoruba people" ≈ "The Yoruba"), verbose-but-correct
answers ("Peter Dinklage is the actor" ≈ "Peter Dinklage"), and rejecting
garbage that merely mentions the correct entity.

Processes candidates from BOTH generation runs:
  1. dev_M5_diverse.jsonl        (2-gold paragraphs, clean answers)
  2. dev_M5_diverse_10para.jsonl (10 paragraphs, noisy answers)

Deduplicates by normalized answer per question across both sources.
Attaches all 10 paragraphs from the chains file.

Output: one JSONL file where each line is a training instance.

Resume-safe: saves progress to a checkpoint file. Re-run to continue.

Usage:
  # Start vLLM first:
  CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
      --model /var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct \
      --port 8000 --dtype auto --tensor-parallel-size 2 --max-model-len 4096

  # Then run:
  cd /var/tmp/u24sf51014/sro/work/sro-proof-hotpot
  /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/label_with_llm_judge.py

  # ~3-4 hours for 74K candidates
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def normalize(s):
    s = (s or "").lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


JUDGE_PROMPT = (
    "Given this question and the correct answer, is the candidate answer "
    "providing the same answer, even if it includes extra explanation?\n\n"
    "Question: {question}\n"
    "Correct answer: {gold}\n"
    "Candidate: {candidate}\n\n"
    "Reply ONLY Yes or No."
)


def call_judge(
    client: OpenAI,
    model_id: str,
    question: str,
    gold: str,
    candidate: str,
) -> int:
    """Ask LLM if candidate is equivalent to gold. Returns 1 (yes) or 0 (no)."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold=gold,
        candidate=candidate[:300],  # truncate very long candidates
    )
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        verdict = (resp.choices[0].message.content or "").strip().lower()
        if verdict.startswith("yes"):
            return 1
        return 0
    except Exception as e:
        print(f"  [judge ERROR] {e}", file=sys.stderr)
        # Fallback to EM
        return int(normalize(candidate) == normalize(gold))


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_gold(path):
    """Load gold answers, questions, types, and context from HotpotQA."""
    data = json.load(open(path))
    out = {}
    for ex in data:
        qid = str(ex["_id"])
        sf = ex.get("supporting_facts", [])
        seen = set()
        gold_titles = [s[0] for s in sf if s[0] not in seen and not seen.add(s[0])]
        out[qid] = {
            "answer": ex["answer"],
            "question": ex["question"],
            "type": ex.get("type", "bridge"),
            "gold_titles": gold_titles,
        }
    return out


def load_paragraphs(chains_path):
    """Load all 10 paragraphs per question from chains file."""
    out = {}
    for line in open(chains_path):
        rec = json.loads(line.strip())
        qid = str(rec.get("qid", ""))
        paras = rec.get("all_paragraphs", [])
        if qid and paras:
            out[qid] = paras
    return out


def load_candidates(path, source_tag):
    """Load candidates from a generation file. Returns {qid: [(answer, prompt_id, source), ...]}."""
    out = defaultdict(list)
    for line in open(path):
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = str(rec["qid"])
        for c in rec.get("candidates", []):
            ans = c.get("answer_text", "").strip()
            pid = c.get("prompt_id", "unknown")
            if ans:
                out[qid].append((ans, pid, source_tag))
    return out


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════

def load_checkpoint(ckpt_path):
    """Load judged labels from checkpoint. Returns {(qid, norm_answer): (label, raw_answer, prompt_id, source)}."""
    judged = {}
    if not os.path.exists(ckpt_path):
        return judged
    with open(ckpt_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec["qid"], normalize(rec["answer"]))
            judged[key] = {
                "label": rec["label"],
                "answer": rec["answer"],
                "prompt_id": rec["prompt_id"],
                "source": rec["source"],
            }
    return judged


def save_judgment(ckpt_f, qid, answer, label, prompt_id, source):
    """Append one judgment to checkpoint file."""
    ckpt_f.write(json.dumps({
        "qid": qid,
        "answer": answer,
        "label": label,
        "prompt_id": prompt_id,
        "source": source,
    }, ensure_ascii=False) + "\n")
    ckpt_f.flush()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands_2gold",
                    default="exp_distractor/candidates/dev_M5_diverse.jsonl")
    ap.add_argument("--cands_10para",
                    default="exp_distractor/candidates/dev_M5_diverse_10para.jsonl")
    ap.add_argument("--gold",
                    default="data/hotpotqa/raw/hotpot_dev_distractor_v1.json")
    ap.add_argument("--chains",
                    default="exp_distractor/evidence/dev_distractor_chains.jsonl")
    ap.add_argument("--out",
                    default="exp_10para/instances/dev_llm_judged.jsonl")
    ap.add_argument("--checkpoint",
                    default="exp_10para/instances/judge_checkpoint.jsonl")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--max_q", type=int, default=None,
                    help="Limit questions (for testing)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # ── Connect to vLLM ──────────────────────────────────────────────
    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY")
    try:
        models = client.models.list()
        model_id = models.data[0].id
    except Exception as e:
        print(f"ERROR: Cannot connect to vLLM: {e}")
        print("Start vLLM first.")
        sys.exit(1)
    print(f"[judge] Connected. Model: {model_id}")

    # ── Load data ────────────────────────────────────────────────────
    print("[judge] Loading gold answers...")
    gold_map = load_gold(args.gold)
    print(f"  {len(gold_map)} questions")

    print("[judge] Loading paragraphs...")
    para_map = load_paragraphs(args.chains)
    print(f"  {len(para_map)} questions with paragraphs")

    print("[judge] Loading 2-gold candidates...")
    cands_2g = load_candidates(args.cands_2gold, "2gold")
    print(f"  {len(cands_2g)} questions")

    print("[judge] Loading 10-para candidates...")
    cands_10p = load_candidates(args.cands_10para, "10para")
    print(f"  {len(cands_10p)} questions")

    # ── Load checkpoint ──────────────────────────────────────────────
    print("[judge] Loading checkpoint...")
    judged = load_checkpoint(args.checkpoint)
    print(f"  {len(judged)} judgments already done")

    # ── Build work queue ─────────────────────────────────────────────
    # For each question, collect unique candidates across both sources
    all_qids = sorted(set(gold_map.keys()) & set(para_map.keys()))
    if args.max_q:
        all_qids = all_qids[:args.max_q]

    work_queue = []  # list of (qid, answer, prompt_id, source)
    n_skip = 0

    for qid in all_qids:
        seen_norm = set()
        # Merge candidates from both sources
        all_cands = cands_2g.get(qid, []) + cands_10p.get(qid, [])
        for ans, pid, source in all_cands:
            n = normalize(ans)
            if not n:
                continue
            if n in seen_norm:
                continue
            seen_norm.add(n)
            key = (qid, n)
            if key in judged:
                n_skip += 1
                continue
            work_queue.append((qid, ans, pid, source))

    print(f"[judge] Total unique candidates to judge: {len(work_queue)}")
    print(f"[judge] Already judged (skip): {n_skip}")

    if not work_queue:
        print("[judge] Nothing to judge. Building output...")
    else:
        # ── Judge ────────────────────────────────────────────────────
        print(f"[judge] Judging {len(work_queue)} candidates with {args.max_workers} workers...")
        t0 = time.time()
        n_done = 0
        n_yes = 0

        with open(args.checkpoint, "a") as ckpt_f:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {}
                for qid, ans, pid, source in work_queue:
                    question = gold_map[qid]["question"]
                    gold_ans = gold_map[qid]["answer"]
                    fut = executor.submit(
                        call_judge, client, model_id, question, gold_ans, ans
                    )
                    futures[fut] = (qid, ans, pid, source)

                for fut in as_completed(futures):
                    qid, ans, pid, source = futures[fut]
                    try:
                        label = fut.result()
                    except Exception as e:
                        print(f"  [ERROR] qid={qid}: {e}", file=sys.stderr)
                        label = int(normalize(ans) == normalize(gold_map[qid]["answer"]))

                    # Save to checkpoint
                    save_judgment(ckpt_f, qid, ans, label, pid, source)
                    key = (qid, normalize(ans))
                    judged[key] = {
                        "label": label, "answer": ans,
                        "prompt_id": pid, "source": source,
                    }

                    n_done += 1
                    if label == 1:
                        n_yes += 1

                    if n_done % 200 == 0:
                        elapsed = time.time() - t0
                        rate = n_done / elapsed * 60
                        remaining = len(work_queue) - n_done
                        eta_h = remaining / max(rate, 1) / 60
                        pct_yes = 100 * n_yes / n_done
                        print(f"  [{n_done}/{len(work_queue)}] "
                              f"{rate:.0f}/min  ETA {eta_h:.1f}h  "
                              f"yes={pct_yes:.1f}%")

        elapsed = (time.time() - t0) / 60
        print(f"\n[judge] Judging complete. {n_done} in {elapsed:.1f} min.")
        print(f"  Yes: {n_yes} ({100*n_yes/n_done:.1f}%)  No: {n_done-n_yes}")

    # ── Build final output ───────────────────────────────────────────
    print("\n[judge] Building final output...")

    n_instances = 0
    n_pos = 0
    n_questions = 0
    n_q_with_pos = 0
    source_counts = Counter()
    unique_per_q = []

    with open(args.out, "w") as fout:
        for qid in all_qids:
            if qid not in para_map:
                continue

            g = gold_map[qid]
            paragraphs = para_map[qid]
            n_questions += 1

            # Collect all judged candidates for this question
            seen_norm = set()
            instances = []
            all_cands = cands_2g.get(qid, []) + cands_10p.get(qid, [])

            for ans, pid, source in all_cands:
                n = normalize(ans)
                if not n or n in seen_norm:
                    continue
                seen_norm.add(n)
                key = (qid, n)
                if key not in judged:
                    continue
                j = judged[key]
                instances.append({
                    "answer": j["answer"],
                    "label": j["label"],
                    "prompt_id": j["prompt_id"],
                    "source": j["source"],
                })

            has_pos = False
            for idx, inst in enumerate(instances):
                if inst["label"] == 1:
                    has_pos = True
                    n_pos += 1
                source_counts[inst["source"]] += 1

                fout.write(json.dumps({
                    "qid": qid,
                    "instance_id": f"{qid}_{idx}",
                    "question": g["question"],
                    "answer": inst["answer"],
                    "label": inst["label"],
                    "qtype": g["type"],
                    "source": inst["source"],
                    "prompt_id": inst["prompt_id"],
                    "paragraphs": paragraphs,
                    "gold_titles": g["gold_titles"],
                    "n_paragraphs": len(paragraphs),
                }, ensure_ascii=False) + "\n")
                n_instances += 1

            if has_pos:
                n_q_with_pos += 1
            unique_per_q.append(len(instances))

    # ── Report ───────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  FINAL DATASET")
    print("=" * 65)
    print(f"  Questions:                 {n_questions}")
    print(f"  Total instances:           {n_instances}")
    print(f"  Avg per question:          {sum(unique_per_q)/len(unique_per_q):.1f}")
    print()
    print(f"  Positive (correct):        {n_pos} ({100*n_pos/n_instances:.1f}%)")
    print(f"  Negative (wrong):          {n_instances - n_pos}")
    print()
    print(f"  Questions with ≥1 correct: {n_q_with_pos} ({100*n_q_with_pos/n_questions:.1f}%)")
    print(f"  Questions all wrong:       {n_questions - n_q_with_pos}")
    print()
    print(f"  Source breakdown:")
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src}: {cnt}")
    print()
    dist = Counter(unique_per_q)
    print(f"  Instances per question:")
    for k in sorted(dist):
        pct = 100 * dist[k] / len(unique_per_q)
        print(f"    {k}: {dist[k]:>5} ({pct:.1f}%)")
    print()
    pct2 = 100 * sum(v for k, v in dist.items() if k >= 2) / len(unique_per_q)
    pct3 = 100 * sum(v for k, v in dist.items() if k >= 3) / len(unique_per_q)
    print(f"  2+ instances: {pct2:.1f}%")
    print(f"  3+ instances: {pct3:.1f}%")
    print()
    print(f"  Output: {args.out}")
    print(f"  Size: {os.path.getsize(args.out) / 1024 / 1024:.1f} MB")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 65)

    # ── Sample ───────────────────────────────────────────────────────
    print()
    print("  Samples:")
    shown = set()
    with open(args.out) as f:
        for line in f:
            inst = json.loads(line)
            if len(shown) >= 3 and inst["qid"] not in shown:
                break
            shown.add(inst["qid"])
            tag = "✓" if inst["label"] == 1 else "✗"
            print(f"    {tag} [{inst['source']:6s}|{inst['prompt_id']:12s}] "
                  f"{inst['answer'][:50]}")


if __name__ == "__main__":
    main()