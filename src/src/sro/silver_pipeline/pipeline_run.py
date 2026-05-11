"""
Silver Pipeline — Orchestration Script

Runs the judge on a set of instances and produces:
  - silver labels (one JSON object per instance)
  - malformed_outputs.jsonl (Tier 3 discards with failure classification)
  - bakeoff_report.json (parsing + quote failure rates for the bake-off)

Two modes:
  --mode bakeoff   Run on the 384-instance audit sample (Phase 4).
                   Computes parsing failure rate and hallucinated-quote rate
                   against the 2% circuit-breaker thresholds.

  --mode full      Run on ALL instances from the repaired candidate file.
                   Activates the 2% circuit breaker for runtime monitoring.

Both modes are resume-safe: completed instances are checkpointed to a JSONL
file and skipped on restart.

Usage (bake-off):
  python3 -m sro.silver_pipeline.pipeline_run --mode bakeoff

Usage (full generation, after bake-off passes):
  python3 -m sro.silver_pipeline.pipeline_run --mode full

Requires: vLLM serving the judge model (Qwen 2.5 72B Instruct or Llama-3-70B).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .judge_runner import judge_one_instance, JudgeResult
from .judge_client import JudgeClient
from .prompt_v1 import PROMPT_VERSION, PROMPT_HASH


# ────────────────────────────────────────────────────────────────────────────
# Defaults
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_AUDIT_SAMPLE = "data/silver/audit_sample_384.jsonl"
DEFAULT_REPAIRED = "data/silver/dev_M5_repaired.jsonl"
DEFAULT_GOLD = "data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
DEFAULT_OUT_DIR = "data/silver"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_SEED = 12345


# ────────────────────────────────────────────────────────────────────────────
# Circuit breaker (§9.4) — 2% cumulative discard rate
# ────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Monitors the cumulative discard rate during generation.
    Trips (raises) if it exceeds the threshold at any point.
    """
    def __init__(self, threshold_pct: float = 2.0):
        self.threshold = threshold_pct / 100.0
        self.n_total = 0
        self.n_discards = 0

    def record(self, is_discard: bool):
        self.n_total += 1
        if is_discard:
            self.n_discards += 1

    @property
    def discard_rate(self) -> float:
        return self.n_discards / self.n_total if self.n_total else 0.0

    @property
    def is_tripped(self) -> bool:
        # Only start checking after 50 instances (avoid noisy early trips)
        return self.n_total >= 50 and self.discard_rate > self.threshold

    def status_str(self) -> str:
        rate = 100 * self.discard_rate
        return (
            f"CircuitBreaker: {self.n_discards}/{self.n_total} discards "
            f"({rate:.1f}%, threshold={self.threshold * 100:.0f}%)"
        )


# ────────────────────────────────────────────────────────────────────────────
# Instance loading
# ────────────────────────────────────────────────────────────────────────────

def _load_instances_from_audit_sample(path: str) -> list[dict]:
    """Load the 384 enriched instances from draw_audit_sample.py output."""
    instances = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(json.loads(line))
    return instances


def _load_instances_from_repaired(
    repaired_path: str,
    gold_path: str,
) -> list[dict]:
    """Load ALL instances from the repaired candidate file + gold data."""
    # Load gold for paragraphs + supporting_facts
    with open(gold_path, encoding="utf-8") as f:
        gold_data = json.load(f)
    gold_by_qid = {}
    for rec in gold_data:
        qid = str(rec.get("_id") or rec.get("id"))
        gold_by_qid[qid] = rec

    instances = []
    with open(repaired_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec.get("qid", ""))
            gold_rec = gold_by_qid.get(qid, {})

            for cidx, cand in enumerate(rec.get("candidates", [])):
                if isinstance(cand, dict):
                    cand_text = str(cand.get("answer_text", "")).strip()
                else:
                    cand_text = str(cand).strip()
                if not cand_text:
                    continue

                instances.append({
                    "qid": qid,
                    "candidate_idx": cidx,
                    "candidate_text": cand_text,
                    "qtype": (gold_rec.get("type") or gold_rec.get("qtype", "bridge")),
                    "question": gold_rec.get("question", ""),
                    "gold_answer": gold_rec.get("answer", ""),
                    "paragraphs": gold_rec.get("context", []),
                    "supporting_facts": gold_rec.get("supporting_facts", []),
                })

    return instances


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint management (resume-safe)
# ────────────────────────────────────────────────────────────────────────────

def _load_completed(checkpoint_path: str) -> set[str]:
    """Return set of instance keys already completed."""
    done = set()
    if not os.path.exists(checkpoint_path):
        return done
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = f"{rec['qid']}_{rec['candidate_idx']}"
            done.add(key)
    return done


# ────────────────────────────────────────────────────────────────────────────
# Bake-off report (§4.1)
# ────────────────────────────────────────────────────────────────────────────

def _compute_bakeoff_report(results: list[JudgeResult]) -> dict:
    """
    Compute the two bake-off gate metrics:
      1. Parsing failure rate = (Tier 2 retries + Tier 3 discards) / total
      2. Hallucinated-quote rate = fraction of outputs where quote failed fuzzy match

    Both must be < 2% for the model to pass the bake-off.
    """
    n_total = len(results)
    n_tier2_retries = sum(r.n_retries_total for r in results)
    n_tier3_discards = sum(1 for r in results if not r.is_valid)
    n_quote_failures = 0

    for r in results:
        for a in r.attempts:
            if a.parsed.status.value == "quote_failed":
                n_quote_failures += 1

    # Gate 1: parsing failure rate
    parsing_failure_rate = (n_tier2_retries + n_tier3_discards) / (n_total * 3) if n_total else 0
    # Gate 2: hallucinated-quote rate (among valid outputs)
    n_valid_outputs = sum(r.n_valid for r in results)
    quote_failure_rate = n_quote_failures / (n_total * 3) if n_total else 0

    gate1_pass = parsing_failure_rate < 0.02
    gate2_pass = quote_failure_rate < 0.02

    report = {
        "n_instances": n_total,
        "n_total_generations": n_total * 3,
        "n_tier2_retries": n_tier2_retries,
        "n_tier3_discards": n_tier3_discards,
        "parsing_failure_rate": round(parsing_failure_rate, 4),
        "parsing_gate_pass": gate1_pass,
        "n_quote_failures": n_quote_failures,
        "hallucinated_quote_rate": round(quote_failure_rate, 4),
        "quote_gate_pass": gate2_pass,
        "overall_pass": gate1_pass and gate2_pass,
        "n_valid_instances": sum(1 for r in results if r.is_valid),
        "label_distribution": dict(Counter(r.final_label for r in results)),
        "agreement_rate": round(
            sum(1 for r in results if r.supporting_paragraphs_agree and r.is_valid) /
            max(sum(1 for r in results if r.is_valid), 1), 4
        ),
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": PROMPT_HASH,
    }
    return report


# ────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    mode: str = "bakeoff",
    base_url: str = DEFAULT_BASE_URL,
    model_id: str = "",
    audit_sample_path: str = DEFAULT_AUDIT_SAMPLE,
    repaired_path: str = DEFAULT_REPAIRED,
    gold_path: str = DEFAULT_GOLD,
    out_dir: str = DEFAULT_OUT_DIR,
    seed: int = DEFAULT_SEED,
    K: int = 3,
    base_temperature: float = 0.3,
    retry_temperature: float = 0.1,
    max_retries_per_gen: int = 2,
    circuit_breaker_pct: float = 2.0,
    max_workers: int = 1,     # sequential for now; can be parallelized later
    verify_quotes: bool = True,
):
    created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Silver Pipeline — {mode.upper()} mode")
    print(f"  {created_utc}")
    print("=" * 70)

    # ── Load instances ────────────────────────────────────────────────────
    if mode == "bakeoff":
        if not os.path.exists(audit_sample_path):
            print(f"ERROR: audit sample not found at {audit_sample_path}", file=sys.stderr)
            print(f"  Run: python3 -m sro.silver_pipeline.draw_audit_sample", file=sys.stderr)
            sys.exit(1)
        instances = _load_instances_from_audit_sample(audit_sample_path)
        out_label = "bakeoff"
    elif mode == "full":
        instances = _load_instances_from_repaired(repaired_path, gold_path)
        out_label = "full"
    else:
        print(f"ERROR: unknown mode {mode!r}. Use 'bakeoff' or 'full'.", file=sys.stderr)
        sys.exit(1)

    print(f"  Instances to judge: {len(instances)}")
    print(f"  K (self-consistency): {K}")
    print(f"  Temperatures: base={base_temperature}, retry={retry_temperature}")

    # ── Connect to judge ──────────────────────────────────────────────────
    client = JudgeClient(base_url=base_url, model_id=model_id)
    detected_model = client.connect()
    print(f"  Judge model: {detected_model}")
    print(f"  Prompt: {PROMPT_VERSION} (hash={PROMPT_HASH[:16]}...)")
    print()

    # ── Resume logic ──────────────────────────────────────────────────────
    silver_path = out / f"silver_{out_label}.jsonl"
    malformed_path = out / f"malformed_{out_label}.jsonl"
    checkpoint_path = out / f"checkpoint_{out_label}.jsonl"

    completed = _load_completed(str(checkpoint_path))
    todo = [
        inst for inst in instances
        if f"{inst['qid']}_{inst['candidate_idx']}" not in completed
    ]
    print(f"  Already completed: {len(completed)}")
    print(f"  Remaining: {len(todo)}")
    print()

    if not todo:
        print("  Nothing to do — all instances already judged.")
        if mode == "bakeoff":
            # Still compute the bake-off report from checkpoint
            _finalize_bakeoff(checkpoint_path, out, detected_model, created_utc)
        return

    # ── Run ───────────────────────────────────────────────────────────────
    cb = CircuitBreaker(threshold_pct=circuit_breaker_pct) if mode == "full" else None
    results: list[JudgeResult] = []
    t0 = time.time()

    with open(str(checkpoint_path), "a", encoding="utf-8") as ckpt_f, \
         open(str(malformed_path), "a", encoding="utf-8") as mal_f:

        for i, inst in enumerate(todo):
            qid = inst["qid"]
            cidx = inst["candidate_idx"]
            paragraphs = inst.get("paragraphs", [])
            # Convert paragraphs to (title, [sentences]) format if needed
            para_tuples = []
            for p in paragraphs:
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    para_tuples.append((p[0], p[1]))
                elif isinstance(p, dict):
                    para_tuples.append((p.get("title", ""), p.get("sentences", [])))
                else:
                    para_tuples.append((str(p), []))

            # Judge this instance
            result = judge_one_instance(
                client=client,
                question=inst.get("question", ""),
                paragraphs=para_tuples,
                candidate=inst["candidate_text"],
                qid=qid,
                candidate_idx=cidx,
                K=K,
                base_temperature=base_temperature,
                retry_temperature=retry_temperature,
                max_retries_per_gen=max_retries_per_gen,
                global_seed=seed,
                verify_quotes=verify_quotes,
            )

            results.append(result)

            # Write to checkpoint
            out_rec = result.to_dict()
            out_rec["judge_model"] = detected_model
            out_rec["qtype"] = inst.get("qtype", "")
            out_rec["question"] = inst.get("question", "")
            out_rec["gold_answer"] = inst.get("gold_answer", "")
            ckpt_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            ckpt_f.flush()

            # Log malformed outputs
            if not result.is_valid:
                mal_rec = {
                    "qid": qid,
                    "candidate_idx": cidx,
                    "candidate_text": inst["candidate_text"],
                    "qtype": inst.get("qtype", ""),
                    "n_valid": result.n_valid,
                    "n_discards": result.n_discards,
                    "attempts": [a.to_dict() for a in result.attempts],
                    "raw_outputs": [a.raw_output[:500] for a in result.attempts],
                }
                mal_f.write(json.dumps(mal_rec, ensure_ascii=False) + "\n")
                mal_f.flush()

            # Circuit breaker (full mode only)
            if cb:
                cb.record(is_discard=not result.is_valid)
                if cb.is_tripped:
                    print(f"\n  🔴 CIRCUIT BREAKER TRIPPED at instance {i + 1}")
                    print(f"  {cb.status_str()}")
                    print(f"  HALTING. This signals a prompt-alignment problem, "
                          f"not a parsing problem (§9.4).")
                    print(f"  Most likely culprit: nested quotes in the reasoning field.")
                    sys.exit(2)

            # Progress logging
            if (i + 1) % 10 == 0 or (i + 1) == len(todo):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                label_str = f"label={result.final_label}" if result.is_valid else "DISCARDED"
                n_valid = sum(1 for r in results if r.is_valid)
                print(
                    f"  [{i + 1:4d}/{len(todo)}] "
                    f"qid={qid[:12]}... cand={cidx} {label_str} | "
                    f"valid={n_valid}/{len(results)} | "
                    f"{rate:.0f} inst/min | "
                    f"elapsed={elapsed / 60:.1f}m"
                )

    elapsed_total = time.time() - t0

    print()
    print(f"  Completed {len(results)} instances in {elapsed_total / 60:.1f} min")

    # ── Mode-specific finalization ────────────────────────────────────────
    if mode == "bakeoff":
        _finalize_bakeoff(checkpoint_path, out, detected_model, created_utc)
    else:
        _finalize_full(results, checkpoint_path, silver_path, out,
                       detected_model, created_utc)


def _finalize_bakeoff(
    checkpoint_path: Path,
    out: Path,
    model_id: str,
    created_utc: str,
):
    """Compute and display the bake-off report."""
    # Reload all results from checkpoint
    results = []
    with open(str(checkpoint_path), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Reconstruct minimal JudgeResult for reporting
            r = JudgeResult(
                qid=rec["qid"],
                candidate_idx=rec["candidate_idx"],
                candidate_text=rec.get("candidate_text", ""),
                final_label=rec["final_label"],
                is_valid=rec["is_valid"],
                votes=rec["votes"],
                n_valid=rec["n_valid"],
                n_retries_total=rec["n_retries_total"],
                n_discards=rec["n_discards"],
                supporting_paragraphs_agree=rec["supporting_paragraphs_agree"],
                supporting_paragraphs_sets=[],
                attempts=[],  # not needed for report
            )
            results.append(r)

    report = _compute_bakeoff_report(results)
    report["judge_model"] = model_id
    report["created_utc"] = created_utc

    report_path = out / "bakeoff_report.json"
    with open(str(report_path), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Display
    print("=" * 70)
    print("  BAKE-OFF REPORT")
    print("=" * 70)
    print(f"  Judge model:              {model_id}")
    print(f"  Instances:                {report['n_instances']}")
    print(f"  Total generations (K=3):  {report['n_total_generations']}")
    print()
    gate1 = "✅ PASS" if report['parsing_gate_pass'] else "❌ FAIL"
    gate2 = "✅ PASS" if report['quote_gate_pass'] else "❌ FAIL"
    overall = "✅ PASS" if report['overall_pass'] else "❌ FAIL"
    print(f"  Gate 1 — Parsing failure rate:     {report['parsing_failure_rate']:.2%}  {gate1}")
    print(f"    (Tier 2 retries: {report['n_tier2_retries']}, "
          f"Tier 3 discards: {report['n_tier3_discards']})")
    print(f"  Gate 2 — Hallucinated-quote rate:  {report['hallucinated_quote_rate']:.2%}  {gate2}")
    print(f"    (Quote failures: {report['n_quote_failures']})")
    print()
    print(f"  OVERALL:  {overall}")
    print()
    print(f"  Valid instances:    {report['n_valid_instances']}/{report['n_instances']}")
    print(f"  Label distribution: {report['label_distribution']}")
    print(f"  SP agreement rate:  {report['agreement_rate']:.1%}")
    print()

    if report['overall_pass']:
        print(f"  🟢 {model_id} passes both gates.")
        print(f"     → Lock this model + prompt for full generation.")
        print(f"     → Proceed to manual audit of all {report['n_instances']} instances.")
    else:
        if not report['parsing_gate_pass']:
            print(f"  🔴 Parsing gate FAILED. Check malformed_bakeoff.jsonl for patterns.")
        if not report['quote_gate_pass']:
            print(f"  🔴 Quote gate FAILED. Judge is hallucinating quotes.")
        print(f"     → If this is Qwen 2.5 72B Instruct, swap to Llama-3-70B fallback")
        print(f"       and re-run on the SAME 384 instances (§4.1 step 5b).")

    print(f"\n  Report → {report_path}")
    print("=" * 70)


def _finalize_full(
    results: list[JudgeResult],
    checkpoint_path: Path,
    silver_path: Path,
    out: Path,
    model_id: str,
    created_utc: str,
):
    """Build the final silver_dataset.jsonl from the checkpoint."""
    # Convert checkpoint to silver format
    n_written = 0
    with open(str(checkpoint_path), encoding="utf-8") as f_in, \
         open(str(silver_path), "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("is_valid", False):
                continue
            silver_rec = {
                "question_id": rec["qid"],
                "question": rec.get("question", ""),
                "qtype": rec.get("qtype", ""),
                "candidate": rec["candidate_text"],
                "candidate_idx": rec["candidate_idx"],
                "prompt_id": rec.get("prompt_version", PROMPT_VERSION),
                "judge_model": model_id,
                "judge_label": rec["final_label"],
                "self_consistency_votes": rec["votes"],
                "supporting_paragraphs_agreement": rec["supporting_paragraphs_agree"],
            }
            f_out.write(json.dumps(silver_rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"  Silver dataset: {n_written} valid instances → {silver_path}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def _cli():
    ap = argparse.ArgumentParser(
        description="Run the silver labeling pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", choices=["bakeoff", "full"], default="bakeoff",
                    help="'bakeoff' = 384 audit sample; 'full' = all instances")
    ap.add_argument("--base_url", default=DEFAULT_BASE_URL,
                    help=f"vLLM endpoint (default: {DEFAULT_BASE_URL})")
    ap.add_argument("--model_id", default="",
                    help="Model ID (auto-detected if empty)")
    ap.add_argument("--audit_sample", default=DEFAULT_AUDIT_SAMPLE)
    ap.add_argument("--candidates", default=DEFAULT_REPAIRED)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--base_temp", type=float, default=0.3)
    ap.add_argument("--retry_temp", type=float, default=0.1)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--circuit_breaker_pct", type=float, default=2.0)
    ap.add_argument("--no_quote_verify", action="store_true",
                    help="Skip fuzzy quote verification (faster, less safe)")
    args = ap.parse_args()

    run_pipeline(
        mode=args.mode,
        base_url=args.base_url,
        model_id=args.model_id,
        audit_sample_path=args.audit_sample,
        repaired_path=args.candidates,
        gold_path=args.gold,
        out_dir=args.out_dir,
        seed=args.seed,
        K=args.K,
        base_temperature=args.base_temp,
        retry_temperature=args.retry_temp,
        max_retries_per_gen=args.max_retries,
        circuit_breaker_pct=args.circuit_breaker_pct,
        verify_quotes=not args.no_quote_verify,
    )


if __name__ == "__main__":
    _cli()