"""
Silver Pipeline — Draw 384-Instance Audit Sample

Implements data_pipeline_reference.md §7 (manual audit protocol).

Draws a stratified (by qtype) sample of 384 (question, candidate) instances
from the repaired candidate file. This sample serves a dual purpose:
  1. Manual audit of label correctness (§7.1–7.3)
  2. Qwen-vs-Llama bake-off for judge model selection (§4.1)

Sample size justification (§7.1):
  n = Z² × p(1-p) / MoE² = 1.96² × 0.5 × 0.5 / 0.05² ≈ 384
  → 95% confidence that the estimated error rate is within ±5%.

The sample is frozen once drawn (manifest records seed + sha256).

Usage:
  python3 -m sro.silver_pipeline.draw_audit_sample
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REPAIRED = "data/silver/dev_M5_repaired.jsonl"
DEFAULT_GOLD = "data/hotpotqa/raw/hotpot_dev_distractor_v1.json"
DEFAULT_OUT = "data/silver/audit_sample_384.jsonl"
DEFAULT_MANIFEST = "data/silver/audit_sample_manifest.json"
DEFAULT_SEED = 12345
DEFAULT_N = 384


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def draw_audit_sample(
    repaired_path: str = DEFAULT_REPAIRED,
    gold_path: str = DEFAULT_GOLD,
    out_path: str = DEFAULT_OUT,
    manifest_path: str = DEFAULT_MANIFEST,
    n_sample: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Draw a stratified sample of n_sample (question, candidate) instances.

    Stratification: preserves the bridge/comparison ratio from the full
    candidate pool. Each sampled instance is a single (qid, candidate_idx)
    pair — the bake-off and audit evaluate individual judge calls, not
    question-level aggregates.

    Returns the manifest dict.
    """
    created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"Drawing {n_sample}-instance audit sample...")
    print(f"  Candidates: {repaired_path}")
    print(f"  Gold:       {gold_path}")
    print(f"  Seed:       {seed}")

    # ── Load gold for qtype lookup ────────────────────────────────────────
    with open(gold_path, encoding="utf-8") as f:
        gold_data = json.load(f)
    qid_to_qtype = {}
    qid_to_gold = {}
    for rec in gold_data:
        qid = str(rec.get("_id") or rec.get("id"))
        qid_to_qtype[qid] = rec.get("type") or rec.get("qtype", "bridge")
        qid_to_gold[qid] = rec

    # ── Load all (qid, candidate_idx) instances from repaired file ────────
    instances_by_qtype: dict[str, list[dict]] = defaultdict(list)
    with open(repaired_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec.get("qid", ""))
            qtype = qid_to_qtype.get(qid, "bridge")

            for cidx, cand in enumerate(rec.get("candidates", [])):
                if isinstance(cand, dict):
                    cand_text = str(cand.get("answer_text", "")).strip()
                else:
                    cand_text = str(cand).strip()
                if not cand_text:
                    continue

                instances_by_qtype[qtype].append({
                    "qid": qid,
                    "candidate_idx": cidx,
                    "candidate_text": cand_text,
                    "qtype": qtype,
                })

    total_instances = sum(len(v) for v in instances_by_qtype.values())
    print(f"  Total instances: {total_instances}")
    for qt, insts in sorted(instances_by_qtype.items()):
        print(f"    {qt}: {len(insts)}")

    # ── Stratified sampling ───────────────────────────────────────────────
    rng = random.Random(seed)
    sampled: list[dict] = []

    for qtype, pool in instances_by_qtype.items():
        share = len(pool) / total_instances
        n_this_type = round(n_sample * share)
        rng.shuffle(pool)
        sampled.extend(pool[:n_this_type])

    # Adjust if rounding gave us slightly more or fewer than n_sample
    rng.shuffle(sampled)
    if len(sampled) > n_sample:
        sampled = sampled[:n_sample]
    while len(sampled) < n_sample:
        # Add from the largest qtype pool
        largest_qt = max(instances_by_qtype, key=lambda qt: len(instances_by_qtype[qt]))
        remaining = [i for i in instances_by_qtype[largest_qt] if i not in sampled]
        if remaining:
            sampled.append(rng.choice(remaining))
        else:
            break

    # ── Enrich with gold data (paragraphs, supporting_facts, answer) ──────
    enriched: list[dict] = []
    for inst in sampled:
        qid = inst["qid"]
        gold_rec = qid_to_gold.get(qid, {})
        enriched.append({
            "qid": qid,
            "candidate_idx": inst["candidate_idx"],
            "candidate_text": inst["candidate_text"],
            "qtype": inst["qtype"],
            "question": gold_rec.get("question", ""),
            "gold_answer": gold_rec.get("answer", ""),
            "paragraphs": gold_rec.get("context", []),
            "supporting_facts": gold_rec.get("supporting_facts", []),
        })

    # ── Write output ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for inst in enriched:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    # ── Manifest ──────────────────────────────────────────────────────────
    qtype_counts = defaultdict(int)
    for inst in enriched:
        qtype_counts[inst["qtype"]] += 1

    manifest = {
        "schema_version": "sro-proof.audit_sample.v1",
        "created_utc": created_utc,
        "n_requested": n_sample,
        "n_drawn": len(enriched),
        "seed": seed,
        "qtype_counts": dict(qtype_counts),
        "source_candidates": repaired_path,
        "source_candidates_sha256": _sha256_file(repaired_path),
        "source_gold": gold_path,
        "out_path": out_path,
        "out_sha256": _sha256_file(out_path),
        "note": (
            "This sample is FROZEN. The same 384 instances are used for both "
            "the Qwen-vs-Llama bake-off (§4.1) and the manual audit (§7). "
            "Do NOT regenerate unless the candidate file changes."
        ),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Audit sample: {len(enriched)} instances → {out_path}")
    print(f"  Qtype split:  {dict(qtype_counts)}")
    print(f"  Manifest:     {manifest_path}")

    return manifest


def _cli():
    ap = argparse.ArgumentParser(description="Draw 384-instance audit sample.")
    ap.add_argument("--candidates", default=DEFAULT_REPAIRED)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    draw_audit_sample(
        repaired_path=args.candidates,
        gold_path=args.gold,
        out_path=args.out,
        manifest_path=args.manifest,
        n_sample=args.n,
        seed=args.seed,
    )


if __name__ == "__main__":
    _cli()