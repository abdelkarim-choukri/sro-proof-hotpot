"""
Silver Pipeline — Judge Runner (K=3 Self-Consistency)

Implements data_pipeline_reference.md §5 (self-consistency protocol).

For each (question, candidate) instance:
  1. Run K=3 independent judge generations, each with a different
     paragraph ordering (position randomization, §5.3).
  2. Each generation is parsed via recovery.py (Tier 1 structural repair).
  3. If a generation fails parsing, retry with reduced temperature
     (Tier 2: 0.3 → 0.1, max 2 retries per generation, §5.4).
  4. Require exactly K=3 valid outputs for majority vote — never fall
     back to majority-of-2 (§5.4: "This breaks the statistical power
     of the self-consistency baseline").
  5. Majority vote on `label` → final silver label.
  6. Track supporting_paragraphs agreement across the K generations.

The caller (pipeline_run.py) handles Tier 3 (discard with accountability)
when judge_one_instance returns a result with n_valid < K.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .prompt_v1 import format_judge_messages, make_paragraph_order, PROMPT_VERSION, PROMPT_HASH
from .recovery import ParsedJudgeOutput, ParseStatus, parse_judge_output
from .judge_client import JudgeClient


# ────────────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationAttempt:
    """Record of one generation attempt (including retries)."""
    generation_idx: int          # 0, 1, 2 for K=3
    attempt: int                 # 0 = first try, 1-2 = retries
    temperature: float
    paragraph_order: list[int]
    parsed: ParsedJudgeOutput
    raw_output: str = ""

    def to_dict(self) -> dict:
        return {
            "generation_idx": self.generation_idx,
            "attempt": self.attempt,
            "temperature": self.temperature,
            "paragraph_order": self.paragraph_order,
            "status": self.parsed.status.value,
            "label": self.parsed.label,
            "quote_verified": self.parsed.quote_verified,
            "quote_score": round(self.parsed.quote_score, 4),
            "failure_mode": self.parsed.failure_mode.value if self.parsed.failure_mode else None,
        }


@dataclass
class JudgeResult:
    """Result of judging one (question, candidate) instance with K-way self-consistency."""
    qid: str
    candidate_idx: int
    candidate_text: str

    # Final verdict
    final_label: int              # majority vote, or -1 if insufficient valid outputs
    is_valid: bool                # True iff we got exactly K valid generations

    # Vote details
    votes: list[int]              # label from each valid generation
    n_valid: int                  # how many of K generations succeeded
    n_retries_total: int          # total Tier 2 retries across all generations
    n_discards: int               # generations that failed after all retries

    # Agreement diagnostics (§5.2)
    supporting_paragraphs_agree: bool   # all K generations cite the same paragraphs
    supporting_paragraphs_sets: list[frozenset[str]]  # one set per valid generation

    # Full generation log (for audit)
    attempts: list[GenerationAttempt] = field(default_factory=list)

    # Metadata
    prompt_version: str = PROMPT_VERSION
    prompt_hash: str = PROMPT_HASH

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "candidate_idx": self.candidate_idx,
            "candidate_text": self.candidate_text,
            "final_label": self.final_label,
            "is_valid": self.is_valid,
            "votes": self.votes,
            "n_valid": self.n_valid,
            "n_retries_total": self.n_retries_total,
            "n_discards": self.n_discards,
            "supporting_paragraphs_agree": self.supporting_paragraphs_agree,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "attempts": [a.to_dict() for a in self.attempts],
        }


# ────────────────────────────────────────────────────────────────────────────
# Seed generation (deterministic, per-instance)
# ────────────────────────────────────────────────────────────────────────────

def _instance_seed(qid: str, candidate_idx: int, generation_idx: int,
                   global_seed: int) -> int:
    """Deterministic seed for paragraph order and reproducibility."""
    raw = f"{global_seed}_{qid}_{candidate_idx}_{generation_idx}"
    h = hashlib.md5(raw.encode()).hexdigest()
    return int(h[:8], 16)


# ────────────────────────────────────────────────────────────────────────────
# Core: judge one instance
# ────────────────────────────────────────────────────────────────────────────

def judge_one_instance(
    client: JudgeClient,
    question: str,
    paragraphs: list[tuple[str, list[str]]],
    candidate: str,
    qid: str,
    candidate_idx: int,
    K: int = 3,
    base_temperature: float = 0.3,
    retry_temperature: float = 0.1,
    max_retries_per_gen: int = 2,
    global_seed: int = 12345,
    verify_quotes: bool = True,
    quote_threshold: float = 0.9,
) -> JudgeResult:
    """
    Run K self-consistency judge calls on one (question, candidate) pair.

    Each generation uses a different paragraph order (position randomization).
    If a generation fails parsing, it is retried with reduced temperature.
    The retry is bound to the individual generation, not the overall instance.

    Returns a JudgeResult. If n_valid < K, the result is marked invalid and
    the caller should discard it (Tier 3) and log to malformed_outputs.jsonl.
    """
    valid_outputs: list[ParsedJudgeOutput] = []
    all_attempts: list[GenerationAttempt] = []
    total_retries = 0
    n_discards = 0

    for gen_idx in range(K):
        # Generate a unique paragraph order for this generation (§5.3)
        seed = _instance_seed(qid, candidate_idx, gen_idx, global_seed)
        para_order = make_paragraph_order(len(paragraphs), seed=seed)

        # Format the prompt
        messages = format_judge_messages(
            question=question,
            paragraphs=paragraphs,
            candidate=candidate,
            paragraph_order=para_order,
        )

        # Try the generation (with Tier 2 retries on parse failure)
        gen_succeeded = False
        for attempt in range(1 + max_retries_per_gen):
            temp = base_temperature if attempt == 0 else retry_temperature

            # Call the judge
            raw_output = client.generate(messages, temperature=temp)

            # Parse the output
            parsed = parse_judge_output(
                raw_output,
                paragraphs=paragraphs if verify_quotes else None,
                verify_quote=verify_quotes,
                quote_threshold=quote_threshold,
            )

            # Log the attempt
            all_attempts.append(GenerationAttempt(
                generation_idx=gen_idx,
                attempt=attempt,
                temperature=temp,
                paragraph_order=para_order,
                parsed=parsed,
                raw_output=raw_output,
            ))

            if parsed.is_valid:
                valid_outputs.append(parsed)
                gen_succeeded = True
                break
            else:
                if attempt < max_retries_per_gen:
                    total_retries += 1
                # If QUOTE_FAILED, still count as a retry candidate
                # (the caller can decide whether to accept quote-failed outputs)

        if not gen_succeeded:
            n_discards += 1

    # ── Majority vote ──────────────────────────────────────────────────────
    votes = [p.label for p in valid_outputs]
    n_valid = len(valid_outputs)

    if n_valid == K:
        # Standard majority vote
        vote_counts = Counter(votes)
        final_label = vote_counts.most_common(1)[0][0]
        is_valid = True
    else:
        # Insufficient valid outputs — DO NOT fall back to majority-of-(K-1)
        final_label = -1
        is_valid = False

    # ── Supporting paragraphs agreement (§5.2) ─────────────────────────────
    sp_sets = [frozenset(p.supporting_paragraphs) for p in valid_outputs]
    sp_agree = len(set(sp_sets)) <= 1 if sp_sets else True

    return JudgeResult(
        qid=qid,
        candidate_idx=candidate_idx,
        candidate_text=candidate,
        final_label=final_label,
        is_valid=is_valid,
        votes=votes,
        n_valid=n_valid,
        n_retries_total=total_retries,
        n_discards=n_discards,
        supporting_paragraphs_agree=sp_agree,
        supporting_paragraphs_sets=sp_sets,
        attempts=all_attempts,
    )