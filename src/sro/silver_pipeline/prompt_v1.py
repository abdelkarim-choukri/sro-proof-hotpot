"""
Silver Pipeline — Judge Prompt V1

Implements the structured judge prompt from data_pipeline_reference.md §4.

Design principles (all from the locked reference doc):
  - Fields are ordered: exact_quote → supporting_paragraphs → reasoning → label.
    This forces the autoregressive model to assemble evidence BEFORE committing
    to a conclusion. Not cosmetic — mechanically prevents decide-then-justify.
  - Three explicit rules block known shortcuts:
    1. Exact-quote requirement (blocks lexical-overlap sycophancy)
    2. Multi-paragraph requirement (blocks single-hop collapse)
    3. Lexical-match-without-answering guard (blocks surface matching)
  - Paragraphs are numbered and titled for unambiguous reference.

Versioning:
  PROMPT_VERSION = "judge_v1"
  PROMPT_HASH is computed from the template text at import time.
  Every silver instance carries both fields for full reproducibility.
"""

from __future__ import annotations

import hashlib
import random
from typing import Optional


PROMPT_VERSION = "judge_v1"


# ────────────────────────────────────────────────────────────────────────────
# System prompt — instructions for the judge
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert multi-hop question answering evaluator. Your task is to \
determine whether a candidate answer to a question is correct, based ONLY \
on the provided paragraphs.

RULES — you must follow all three:
1. The candidate answer must be supported by an exact quote that you can \
extract verbatim from the provided paragraphs. If you cannot find such a \
quote, set label to 0.
2. For multi-hop questions, the supporting evidence must come from at least \
two different paragraphs. If only one paragraph supports the answer, set \
label to 0.
3. If the candidate happens to match text in a paragraph but that text does \
not actually answer the question in context, set label to 0.

OUTPUT FORMAT — you must output ONLY a single JSON object with these four \
fields in this exact order, and nothing else:
{
  "exact_quote": "<a verbatim quote from the paragraphs that supports the answer>",
  "supporting_paragraphs": ["<Title A>", "<Title B>"],
  "reasoning": "<a brief explanation of why the evidence supports or refutes the candidate>",
  "label": 0 or 1
}

IMPORTANT:
- Output raw JSON only. Do NOT wrap it in markdown code fences.
- The "exact_quote" field must contain text copied verbatim from the paragraphs.
- The "supporting_paragraphs" field must list the exact titles of the paragraphs \
you used as evidence.
- The "label" field must be the integer 0 or 1, not a string.
- Do not include any text before or after the JSON object.\
"""


# ────────────────────────────────────────────────────────────────────────────
# User message template
# ────────────────────────────────────────────────────────────────────────────

_USER_TEMPLATE = """\
Question: {question}

Candidate answer: {candidate}

Paragraphs:
{formatted_paragraphs}

Based on the paragraphs above, evaluate whether the candidate answer is correct. \
Output your evaluation as the specified JSON object.\
"""


# ────────────────────────────────────────────────────────────────────────────
# Paragraph formatting
# ────────────────────────────────────────────────────────────────────────────

def _format_paragraphs(
    paragraphs: list[tuple[str, list[str]]],
    order: Optional[list[int]] = None,
) -> str:
    """
    Format 10 paragraphs for the judge prompt.

    Args:
        paragraphs: list of (title, [sentence_0, sentence_1, ...])
                    — the HotpotQA `context` field.
        order: optional permutation of paragraph indices. If provided,
               paragraphs are presented in this order (for position
               randomization across self-consistency calls).

    Returns:
        Formatted string with numbered paragraphs.
    """
    if order is None:
        order = list(range(len(paragraphs)))

    parts: list[str] = []
    for display_idx, real_idx in enumerate(order, start=1):
        title, sentences = paragraphs[real_idx]
        text = " ".join(sentences) if sentences else ""
        parts.append(f"[Paragraph {display_idx}] {title}:\n{text}")

    return "\n\n".join(parts)


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def format_judge_messages(
    question: str,
    paragraphs: list[tuple[str, list[str]]],
    candidate: str,
    paragraph_order: Optional[list[int]] = None,
) -> list[dict[str, str]]:
    """
    Build the (system, user) message list for a single judge call.

    Args:
        question:        the HotpotQA question text.
        paragraphs:      list of (title, [sentences]) from context.
        candidate:       the candidate answer text to evaluate.
        paragraph_order: optional permutation of paragraph indices for
                         position randomization (§5.3).

    Returns:
        List of {"role": ..., "content": ...} dicts ready for the
        OpenAI chat/completions API.
    """
    formatted = _format_paragraphs(paragraphs, order=paragraph_order)
    user_content = _USER_TEMPLATE.format(
        question=question,
        candidate=candidate,
        formatted_paragraphs=formatted,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def make_paragraph_order(
    n_paragraphs: int = 10,
    seed: Optional[int] = None,
) -> list[int]:
    """
    Generate a random permutation of paragraph indices.

    For self-consistency (§5.3), each of the K=3 calls uses a different
    random order. The seed should be derived from (qid, generation_index)
    so the permutation is reproducible.

    Args:
        n_paragraphs: number of paragraphs (default 10 for HotpotQA).
        seed: deterministic seed for the permutation.

    Returns:
        A permuted list of indices [0, n_paragraphs).
    """
    order = list(range(n_paragraphs))
    rng = random.Random(seed)
    rng.shuffle(order)
    return order


# ────────────────────────────────────────────────────────────────────────────
# Prompt hash for versioning
# ────────────────────────────────────────────────────────────────────────────

# Hash the template text so every silver instance can record which exact
# prompt was used. If the prompt is iterated, the hash changes and the
# strict-separation rule from §8.2 prevents mixing versions.
PROMPT_HASH = hashlib.sha256(
    (SYSTEM_PROMPT + _USER_TEMPLATE).encode("utf-8")
).hexdigest()


# ────────────────────────────────────────────────────────────────────────────
# Convenience: dump prompt to file for auditing
# ────────────────────────────────────────────────────────────────────────────

def save_prompt_artifact(path: str = "src/sro/silver_pipeline/prompts/judge_v1.txt"):
    """Write the prompt text + hash to a file for paper trail."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# PROMPT_VERSION: {PROMPT_VERSION}\n")
        f.write(f"# PROMPT_HASH: {PROMPT_HASH}\n")
        f.write(f"# Fields ordered: exact_quote → supporting_paragraphs → reasoning → label\n")
        f.write("\n")
        f.write("=== SYSTEM MESSAGE ===\n\n")
        f.write(SYSTEM_PROMPT)
        f.write("\n\n=== USER MESSAGE TEMPLATE ===\n\n")
        f.write(_USER_TEMPLATE)
        f.write("\n")
    print(f"Prompt artifact saved to {path}")
    print(f"  Version: {PROMPT_VERSION}")
    print(f"  Hash:    {PROMPT_HASH}")


if __name__ == "__main__":
    save_prompt_artifact()