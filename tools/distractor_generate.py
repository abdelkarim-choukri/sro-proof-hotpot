#!/usr/bin/env python3
"""
distractor_generate.py — Generate candidate answers in the distractor setting

PURPOSE:
  Generate M=1 (greedy) and M=5 (sampling) candidate answers using
  Qwen2.5-7B-Instruct with all 10 distractor paragraphs as context.

  This is a simplified fork of exp3_generate_candidates.py, adapted for
  the distractor setting where evidence comes from the distractor file
  (all_paragraphs field) rather than MDR retrieval chains.

  KEY DIFFERENCE FROM MDR PIPELINE:
  - Evidence = all 10 distractor paragraphs (2 gold + 8 distractors)
  - Gold is ALWAYS present (Bucket A = 0%)
  - We generate M=1 (T=0) for the extractive-SOTA comparison baseline
  - We generate M=5 (T=0.7) for the verification experiments

PROMPT:
  Uses the same prompt_v2.txt template as the main pipeline:
    "Answer the QUESTION using ONLY the EVIDENCE below..."
  with {evidence} replaced by all 10 paragraphs.

OUTPUT FORMAT:
  Same schema as existing candidate files — compatible with oracle,
  NLI scoring, QA scoring, lex scoring, and the verifier.

Usage:
  # M=1 greedy (for extractive SOTA comparison):
  python3 tools/distractor_generate.py \
      --evidence    exp_distractor/evidence/dev_distractor_chains.jsonl \
      --gold        data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
      --prompt_file exp1/inputs/prompt_v2.txt \
      --out_jsonl   exp_distractor/candidates/dev_M1_greedy.jsonl \
      --llm_base_url http://127.0.0.1:8000/v1 \
      --llm_model_id /var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct \
      --m 1 --temperature 0.0 --seed 12345

  # M=5 sampling (for verification experiments):
  python3 tools/distractor_generate.py \
      --evidence    exp_distractor/evidence/dev_distractor_chains.jsonl \
      --gold        data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
      --prompt_file exp1/inputs/prompt_v2.txt \
      --out_jsonl   exp_distractor/candidates/dev_M5_sampling.jsonl \
      --llm_base_url http://127.0.0.1:8000/v1 \
      --llm_model_id /var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct \
      --m 5 --temperature 0.7 --seed 12345
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional
from urllib import request, error


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1: CONSTANTS AND SCHEMA
# ═══════════════════════════════════════════════════════════════════════

SCHEMA_VERSION = "sro-proof.distractor.candidates.v1"

FINAL_RE = re.compile(r"<final>\s*(.*?)\s*(?:</final>|$)", re.DOTALL)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2: TEXT / ANSWER UTILITIES
#  Same as your existing pipeline — ensures compatibility.
# ═══════════════════════════════════════════════════════════════════════

def extract_final(raw: str) -> str:
    """Extract answer from model output. Handles both prefill and non-prefill cases."""
    if not raw:
        return ""
    # Strip thinking block if present
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    # Case 1: full <final>...</final> tags
    matches = FINAL_RE.findall(raw)
    if matches:
        ans = (matches[-1] or "").strip()
        lines = [l.strip() for l in ans.splitlines() if l.strip()]
        return " ".join(lines).strip() if lines else ""
    # Case 2: prefill path — raw IS the answer
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def is_bad(ans: str) -> bool:
    """Garbage detector — same as existing pipeline."""
    a = ans.strip()
    if not a:
        return True
    low = a.lower()
    if low.startswith("[chain"):
        return True
    if "if the evidence does not contain" in low:
        return True
    if low.startswith("the evidence provided"):
        return True
    if low.startswith(("okay,", "alright,", "so,")):
        return True
    if low in {"unknown", "unk"}:
        return True
    if len(a) > 120:
        return True
    return False


def _norm(s: str) -> str:
    """Normalize for EM comparison — same as pipeline's normalize()."""
    import string as _string
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in _string.punctuation)
    return ' '.join(s.split())


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3: EVIDENCE FORMATTING
#  Formats the 10 distractor paragraphs into the prompt's {evidence} slot.
#  Uses the same labeling style as the flat evidence format.
# ═══════════════════════════════════════════════════════════════════════

def format_distractor_evidence(all_paragraphs: list, max_chars: int = 12000) -> str:
    """
    Format all 10 distractor paragraphs for the generator prompt.

    Uses the same style as the MDR flat format:
      [paragraph 1] Title: text
      [paragraph 2] Title: text
      ...

    Truncates to max_chars to stay within context window limits.
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
#  SECTION 4: vLLM CLIENT
#  Same API as your existing pipeline — calls chat/completions with
#  the <final> prefill trick for Qwen.
# ═══════════════════════════════════════════════════════════════════════

class GenerationError(RuntimeError):
    pass


def call_vllm(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    n: int,
    stop: Optional[List[str]] = None,
    timeout_s: int = 120,
) -> List[str]:
    """Call vLLM chat/completions endpoint with <final> prefill."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
            # Prefill: Qwen outputs answer directly after <final>
            {"role": "assistant", "content": "<final>"},
        ],
        "n": int(n),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "seed": int(seed),
    }
    if stop:
        payload["stop"] = stop

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data,
                          headers={"Content-Type": "application/json"})

    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise GenerationError(f"HTTPError {e.code}: {body[:300]}")
    except Exception as e:
        raise GenerationError(f"Network error: {e}")

    choices = out.get("choices", [])
    return [(c.get("message", {}).get("content", "") or "") for c in choices]


def retry_call(fn, retries=3, sleep_s=2.0):
    """Retry with exponential backoff."""
    for t in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            if t == retries:
                raise
            time.sleep(sleep_s * (2 ** t))


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 5: MAIN GENERATION LOOP
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Distractor-setting generation")
    ap.add_argument("--evidence", required=True,
        help="Distractor evidence JSONL from distractor_prepare_evidence.py")
    ap.add_argument("--gold", required=True,
        help="HotpotQA distractor dev JSON (for gold answers)")
    ap.add_argument("--prompt_file", required=True,
        help="Prompt template (exp1/inputs/prompt_v2.txt)")
    ap.add_argument("--out_jsonl", required=True,
        help="Output candidates JSONL")

    ap.add_argument("--llm_base_url", required=True,
        help="vLLM base URL, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--llm_model_id", required=True,
        help="Model ID for vLLM, e.g. /path/to/qwen2.5-7b-instruct")

    ap.add_argument("--m", type=int, required=True,
        help="Number of candidates per question (1 for greedy, 5 for sampling)")
    ap.add_argument("--temperature", type=float, required=True,
        help="Sampling temperature (0.0 for greedy, 0.7 for sampling)")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_evidence_chars", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--resume", action="store_true",
        help="Resume from existing output file")
    ap.add_argument("--manifest", default=None)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)

    # ── Load prompt template ──
    prompt_template = open(args.prompt_file).read().strip()
    print(f"Prompt template loaded from {args.prompt_file}")
    print(f"  Template has {{evidence}}: {'evidence' in prompt_template}")
    print(f"  Template has {{question}}: {'question' in prompt_template}")

    # ── Load gold answers ──
    gold_data = json.load(open(args.gold))
    gold_map = {str(ex["_id"]): ex["answer"] for ex in gold_data}
    print(f"Gold answers: {len(gold_map)}")

    # ── Handle resume ──
    done = set()
    if args.resume and os.path.isfile(args.out_jsonl):
        with open(args.out_jsonl) as f:
            for line in f:
                rec = json.loads(line.strip())
                done.add(str(rec["qid"]))
        print(f"Resuming: {len(done)} questions already done")

    # ── Configuration ──
    M = args.m
    STOP_TOKENS = ["</final>"]
    mode = "greedy" if args.temperature == 0.0 else f"sampling_T{args.temperature}"
    print(f"\nGeneration config:")
    print(f"  M={M}, temperature={args.temperature}, top_p={args.top_p}")
    print(f"  Mode: {mode}")
    print(f"  max_new_tokens={args.max_new_tokens}")
    print(f"  Stop tokens: {STOP_TOKENS}")

    # ── Write manifest ──
    if args.manifest:
        os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
        manifest = {
            "distractor_generation": {
                "schema_version": SCHEMA_VERSION,
                "mode": mode,
                "m": M,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
                "model": args.llm_model_id,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        }
        with open(args.manifest, "w") as f:
            json.dump(manifest, f, indent=2)

    # ── Main generation loop ──
    wrote = skipped = errors = 0
    n5_ok = n5_fallback = 0
    t_start = time.time()

    out_f = open(args.out_jsonl, "a" if args.resume else "w")

    # Count total for progress
    n_total = sum(1 for _ in open(args.evidence))

    with open(args.evidence) as ev_f:
        for line_idx, line in enumerate(ev_f):
            evd = json.loads(line.strip())
            qid = str(evd["qid"])

            if qid in done:
                skipped += 1
                continue

            question = evd["question"]
            all_paragraphs = evd.get("all_paragraphs", [])

            # Format evidence for the prompt
            evidence_text = format_distractor_evidence(
                all_paragraphs, max_chars=args.max_evidence_chars
            )

            # Build the full prompt
            prompt = prompt_template.replace("{evidence}", evidence_text)
            prompt = prompt.replace("{question}", question)

            # ── Generate candidates ──
            # Strategy: try n=M first, fall back to M individual calls if needed
            # (same fallback strategy as existing pipeline)
            answers = []
            try:
                raw_outputs = retry_call(lambda: call_vllm(
                    base_url=args.llm_base_url,
                    model=args.llm_model_id,
                    prompt=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_new_tokens,
                    seed=args.seed,
                    n=M,
                    stop=STOP_TOKENS,
                ))

                answers = [extract_final(r) for r in raw_outputs]

                if len(answers) == M:
                    n5_ok += 1
                else:
                    # Fallback: individual calls with seed+i
                    n5_fallback += 1
                    answers = []
                    for i in range(M):
                        raw = retry_call(lambda i=i: call_vllm(
                            base_url=args.llm_base_url,
                            model=args.llm_model_id,
                            prompt=prompt,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_tokens=args.max_new_tokens,
                            seed=args.seed + i,
                            n=1,
                            stop=STOP_TOKENS,
                        ))
                        answers.append(extract_final(raw[0] if raw else ""))

            except Exception as e:
                errors += 1
                print(f"  ✗ qid={qid}: {e}")
                answers = [""] * M

            # ── Build candidate records ──
            candidates = []
            for ci, ans in enumerate(answers):
                candidates.append({
                    "answer_id": ci,
                    "answer_text": ans,
                })

            # ── Compute quick stats ──
            gold = gold_map.get(qid, "")

            n_bad = sum(1 for c in candidates if is_bad(c["answer_text"]))
            has_correct = any(_norm(c["answer_text"]) == _norm(gold) for c in candidates)

            record = {
                "schema_version": SCHEMA_VERSION,
                "split": "dev",
                "qid": qid,
                "candidates": candidates,
                "generation_context": {
                    "mode": mode,
                    "m": M,
                    "temperature": args.temperature,
                    "n_paragraphs": len(all_paragraphs),
                    "distractor_setting": True,
                },
                "stats": {
                    "n_bad": n_bad,
                    "has_correct": has_correct,
                },
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            wrote += 1

            # Progress
            if wrote % 200 == 0 or wrote <= 5:
                elapsed = time.time() - t_start
                rate = wrote / elapsed if elapsed > 0 else 0
                eta_min = (n_total - skipped - wrote) / rate / 60 if rate > 0 else 0
                print(f"  [{wrote}/{n_total - len(done)}] "
                      f"{rate:.1f} q/s  ETA {eta_min:.0f}min  "
                      f"correct_rate={sum(1 for _ in [] if has_correct):.0%}  "
                      f"errors={errors}")

    out_f.close()
    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  Generation complete: {mode}")
    print(f"  Written: {wrote}  Skipped: {skipped}  Errors: {errors}")
    print(f"  n=M OK: {n5_ok}  Fallback: {n5_fallback}")
    print(f"  Time: {elapsed/60:.1f} min ({elapsed/max(wrote,1):.2f}s per question)")
    print(f"  Output: {args.out_jsonl}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()