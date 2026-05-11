"""
Silver Pipeline — 3-Tier JSON Recovery + Quote Verification

Implements data_pipeline_reference.md §9 (recovery) and §4.4 (quote matching).

Tier 1 — Structural repair via `json_repair` library.
         Handles: markdown fences, trailing commas, unclosed braces,
         missing quotes, unicode escaping. This alone recovers the
         majority of structurally invalid outputs.

Tier 2 — Retry with reduced temperature (handled by the caller;
         this module signals when Tier 2 is needed via the return value).

Tier 3 — Discard with full accountability (caller logs to
         malformed_outputs.jsonl with the failure classification from
         this module).

Quote verification:
  LLMs at the 70B-72B scale occasionally drop a comma, alter casing,
  or "fix" a typo when outputting exact_quote. Strict equality would
  reject valid outputs. We use fuzzy matching:
    - ROUGE-L ≥ 0.9  (primary)
    - Levenshtein ratio ≥ 0.85  (backup if rouge_score unavailable)

Design principle from §9.5:
  "Never silently discard." Every failure is classified and returned
  to the caller for logging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ────────────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────────────

class ParseStatus(Enum):
    OK = "ok"                           # Parsed and validated successfully
    REPAIRED = "repaired"               # Tier 1 structural repair succeeded
    SCHEMA_INVALID = "schema_invalid"   # JSON parsed but missing/wrong fields
    QUOTE_FAILED = "quote_failed"       # exact_quote doesn't match any paragraph
    PARSE_FAILED = "parse_failed"       # JSON could not be recovered at all


class FailureMode(Enum):
    """Classification of why a judge output was malformed (§9.3)."""
    MARKDOWN_FENCE = "markdown_fence"
    TRAILING_COMMA = "trailing_comma"
    UNCLOSED_BRACE = "unclosed_brace"
    MISSING_FIELD = "missing_field"
    WRONG_TYPE = "wrong_type"
    NESTED_QUOTE = "nested_quote_in_reasoning"   # §9.4 identified culprit
    MULTI_JSON = "multiple_json_objects"
    EMPTY_OUTPUT = "empty_output"
    NON_JSON_TEXT = "non_json_text"
    UNKNOWN = "unknown"


@dataclass
class ParsedJudgeOutput:
    """Result of parsing a single judge generation."""
    status: ParseStatus
    exact_quote: str = ""
    supporting_paragraphs: list[str] = field(default_factory=list)
    reasoning: str = ""
    label: int = -1                     # -1 = no valid label extracted
    quote_verified: bool = False        # True if fuzzy match passed
    quote_score: float = 0.0            # ROUGE-L or Levenshtein score
    failure_mode: Optional[FailureMode] = None
    raw_output: str = ""                # original text for debugging

    @property
    def is_valid(self) -> bool:
        return self.status in (ParseStatus.OK, ParseStatus.REPAIRED)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "exact_quote": self.exact_quote,
            "supporting_paragraphs": self.supporting_paragraphs,
            "reasoning": self.reasoning,
            "label": self.label,
            "quote_verified": self.quote_verified,
            "quote_score": round(self.quote_score, 4),
            "failure_mode": self.failure_mode.value if self.failure_mode else None,
        }


# ────────────────────────────────────────────────────────────────────────────
# Tier 1 — Structural repair
# ────────────────────────────────────────────────────────────────────────────

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers that LLMs love to add."""
    text = text.strip()
    # Strip leading ```json or ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    # Strip trailing ```
    text = re.sub(r"\n?\s*```\s*$", "", text)
    return text.strip()


def _try_parse_json(text: str) -> Optional[dict]:
    """Attempt JSON parsing with json_repair, falling back to stdlib."""
    text = _strip_markdown_fences(text)
    if not text:
        return None

    # Try json_repair first (handles trailing commas, unclosed braces, etc.)
    try:
        import json_repair
        result = json_repair.loads(text)
        if isinstance(result, dict):
            return result
        # json_repair sometimes returns a list if there are multiple objects
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0]
    except Exception:
        pass

    # Fallback: stdlib json
    import json
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Last resort: try to extract the first {...} block
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            import json_repair
            result = json_repair.loads(match.group())
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    return None


# ────────────────────────────────────────────────────────────────────────────
# Schema validation
# ────────────────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS = {"exact_quote", "supporting_paragraphs", "reasoning", "label"}


def _validate_schema(parsed: dict) -> tuple[bool, Optional[FailureMode]]:
    """
    Check that the parsed dict has all required fields with correct types.

    Returns (is_valid, failure_mode_if_invalid).
    """
    # Check required fields
    missing = _REQUIRED_FIELDS - set(parsed.keys())
    if missing:
        return False, FailureMode.MISSING_FIELD

    # Type checks
    if not isinstance(parsed.get("exact_quote"), str):
        return False, FailureMode.WRONG_TYPE
    if not isinstance(parsed.get("supporting_paragraphs"), list):
        return False, FailureMode.WRONG_TYPE
    if not isinstance(parsed.get("reasoning"), str):
        return False, FailureMode.WRONG_TYPE

    # Label must be 0 or 1 (accept int, float-that-is-int, or string "0"/"1")
    label_raw = parsed.get("label")
    if isinstance(label_raw, (int, float)):
        if int(label_raw) not in (0, 1):
            return False, FailureMode.WRONG_TYPE
    elif isinstance(label_raw, str):
        if label_raw.strip() not in ("0", "1"):
            return False, FailureMode.WRONG_TYPE
    else:
        return False, FailureMode.WRONG_TYPE

    return True, None


def _coerce_label(raw) -> int:
    """Convert label to int 0 or 1."""
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        return int(raw.strip())
    return -1


# ────────────────────────────────────────────────────────────────────────────
# Quote verification — fuzzy matching
# ────────────────────────────────────────────────────────────────────────────

def _rouge_l_score(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 between hypothesis (the judge's exact_quote) and
    reference (the paragraph text). Returns score in [0, 1].
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(reference, hypothesis)
        return scores["rougeL"].fmeasure
    except ImportError:
        # Fallback to Levenshtein ratio
        return _levenshtein_ratio(hypothesis, reference)


def _levenshtein_ratio(a: str, b: str) -> float:
    """
    Levenshtein similarity ratio. Falls back to simple character-level
    comparison if python-Levenshtein is not installed.
    """
    try:
        import Levenshtein
        return Levenshtein.ratio(a, b)
    except ImportError:
        # Ultra-simple fallback: token overlap ratio
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return 2 * len(ta & tb) / (len(ta) + len(tb))


def fuzzy_match_quote(
    quote: str,
    paragraphs: list[tuple[str, list[str]]],
    threshold: float = 0.9,
) -> tuple[bool, float]:
    """
    Check whether `quote` appears (fuzzy) in any of the provided paragraphs.

    For each paragraph, we compute ROUGE-L between the quote and the full
    paragraph text. If any score ≥ threshold, the quote is considered verified.

    We also try matching against individual sentences for short quotes,
    since ROUGE-L against a long paragraph dilutes the score.

    Args:
        quote: the judge's exact_quote field.
        paragraphs: list of (title, [sentences]).
        threshold: ROUGE-L score threshold (default 0.9 per §4.4).

    Returns:
        (is_verified, best_score).
    """
    if not quote or not quote.strip():
        return False, 0.0

    quote_clean = quote.strip()
    best_score = 0.0

    for _title, sentences in paragraphs:
        # Try full paragraph text
        full_text = " ".join(sentences)
        if not full_text:
            continue

        # Quick exact substring check (fast path)
        if quote_clean in full_text:
            return True, 1.0

        # ROUGE-L against full paragraph
        score = _rouge_l_score(quote_clean, full_text)
        best_score = max(best_score, score)
        if score >= threshold:
            return True, score

        # Try individual sentences (for short quotes)
        for sent in sentences:
            if not sent:
                continue
            if quote_clean in sent:
                return True, 1.0
            sent_score = _rouge_l_score(quote_clean, sent)
            best_score = max(best_score, sent_score)
            if sent_score >= threshold:
                return True, sent_score

    return best_score >= threshold, best_score


# ────────────────────────────────────────────────────────────────────────────
# Failure classification
# ────────────────────────────────────────────────────────────────────────────

def classify_failure(raw_output: str) -> FailureMode:
    """
    Classify the failure mode of a malformed judge output.
    Used for the malformed_outputs.jsonl breakdown in §9.3.
    """
    text = raw_output.strip()

    if not text:
        return FailureMode.EMPTY_OUTPUT
    if text.startswith("```"):
        return FailureMode.MARKDOWN_FENCE
    if '"""' in text or "'''" in text:
        return FailureMode.NESTED_QUOTE
    # Count unmatched braces
    opens = text.count("{")
    closes = text.count("}")
    if opens > closes:
        return FailureMode.UNCLOSED_BRACE
    if text.count("{") > 1:
        return FailureMode.MULTI_JSON
    if not text.startswith("{"):
        return FailureMode.NON_JSON_TEXT
    # Check for trailing commas in common positions
    if re.search(r",\s*}", text):
        return FailureMode.TRAILING_COMMA
    return FailureMode.UNKNOWN


# ────────────────────────────────────────────────────────────────────────────
# Main parse function
# ────────────────────────────────────────────────────────────────────────────

def parse_judge_output(
    raw_output: str,
    paragraphs: Optional[list[tuple[str, list[str]]]] = None,
    verify_quote: bool = True,
    quote_threshold: float = 0.9,
) -> ParsedJudgeOutput:
    """
    Parse a single judge generation output through the 3-tier recovery pipeline.

    Tier 1: Structural repair (json_repair + markdown stripping).
    Schema validation: required fields + correct types.
    Quote verification: fuzzy matching against paragraphs.

    Tier 2 (retry) and Tier 3 (discard) are handled by the CALLER based
    on the returned ParsedJudgeOutput.status:
      - OK / REPAIRED → use this output
      - SCHEMA_INVALID / PARSE_FAILED → caller should retry (Tier 2)
      - QUOTE_FAILED → output is structurally valid but quote is hallucinated;
        caller decides whether to retry or accept with a flag

    Args:
        raw_output:      the raw text from the LLM generation.
        paragraphs:      list of (title, [sentences]) for quote verification.
                         If None, quote verification is skipped.
        verify_quote:    whether to run fuzzy quote matching.
        quote_threshold: ROUGE-L threshold for quote acceptance (default 0.9).

    Returns:
        ParsedJudgeOutput with status, extracted fields, and diagnostics.
    """
    raw_output = raw_output or ""

    # ── Tier 1: Structural repair ──────────────────────────────────────────
    parsed = _try_parse_json(raw_output)

    if parsed is None:
        return ParsedJudgeOutput(
            status=ParseStatus.PARSE_FAILED,
            failure_mode=classify_failure(raw_output),
            raw_output=raw_output,
        )

    # Determine if repair was needed (did we need json_repair or just stdlib?)
    import json
    try:
        stripped = _strip_markdown_fences(raw_output)
        json.loads(stripped)
        was_repaired = False
    except (json.JSONDecodeError, ValueError):
        was_repaired = True

    # ── Schema validation ──────────────────────────────────────────────────
    is_valid, failure = _validate_schema(parsed)
    if not is_valid:
        return ParsedJudgeOutput(
            status=ParseStatus.SCHEMA_INVALID,
            failure_mode=failure,
            raw_output=raw_output,
        )

    # Extract fields
    exact_quote = str(parsed["exact_quote"]).strip()
    supporting_paragraphs = [str(t).strip() for t in parsed["supporting_paragraphs"]]
    reasoning = str(parsed["reasoning"]).strip()
    label = _coerce_label(parsed["label"])

    # ── Quote verification ─────────────────────────────────────────────────
    quote_verified = False
    quote_score = 0.0

    if verify_quote and paragraphs is not None and exact_quote:
        quote_verified, quote_score = fuzzy_match_quote(
            exact_quote, paragraphs, threshold=quote_threshold,
        )
        if not quote_verified:
            return ParsedJudgeOutput(
                status=ParseStatus.QUOTE_FAILED,
                exact_quote=exact_quote,
                supporting_paragraphs=supporting_paragraphs,
                reasoning=reasoning,
                label=label,
                quote_verified=False,
                quote_score=quote_score,
                raw_output=raw_output,
            )
    elif exact_quote:
        # No paragraphs provided — skip verification, mark as unverified
        quote_verified = False
        quote_score = -1.0

    # ── Success ────────────────────────────────────────────────────────────
    status = ParseStatus.REPAIRED if was_repaired else ParseStatus.OK

    return ParsedJudgeOutput(
        status=status,
        exact_quote=exact_quote,
        supporting_paragraphs=supporting_paragraphs,
        reasoning=reasoning,
        label=label,
        quote_verified=quote_verified,
        quote_score=quote_score,
        raw_output=raw_output,
    )