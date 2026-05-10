"""
SRO Evaluation Harness (v2.1)

Computes all metrics needed for the HotpotQA multi-hop QA benchmark:

  1. Answer Exact Match (EM) & F1
     - Given M candidates per question, the model selects the one with
       the highest verifier confidence. Compare to gold answer.

  2. Paragraph Selection Accuracy
     - Are the 2 selected paragraphs the gold paragraphs?
     - Reported as: pair accuracy, individual recall, individual precision.

  3. Supporting Fact EM & F1
     - Do the [SENT] token predictions align with gold supporting_facts?

  4. Sufficiency
     - Given ONLY the selected paragraphs, does the gold answer appear
       in their text? If not, the model's selection is insufficient
       regardless of verifier output.

  5. Comprehensiveness
     - Does the model actually use BOTH selected paragraphs?
     - Measured by checking if supporting_fact predictions span both
       selected paragraphs (not collapsed to one).

  6. Joint Metrics (HotpotQA official)
     - Joint EM: answer EM AND supporting fact EM both correct.
     - Joint F1: product of answer F1 and supporting fact F1.

Dependencies:
  - sro_model.py (SROModel, SROBatch, SROOutput)
  - PyTorch
"""

import re
import string
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Text Normalization (HotpotQA standard)
# ════════════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    """
    Standard HotpotQA answer normalization.
    Lowercase, strip articles/punctuation/whitespace.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ════════════════════════════════════════════════════════════════════════
# Token-Level Metrics
# ════════════════════════════════════════════════════════════════════════

def compute_em(prediction: str, gold: str) -> float:
    """Exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def compute_f1(prediction: str, gold: str) -> float:
    """Token-level F1 after normalization."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if len(gold_tokens) == 0:
        return float(len(pred_tokens) == 0)
    if len(pred_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ════════════════════════════════════════════════════════════════════════
# Supporting Fact Metrics
# ════════════════════════════════════════════════════════════════════════

def compute_sp_em(
    predicted_sps: list[tuple[str, int]],
    gold_sps: list[tuple[str, int]],
) -> float:
    """
    Supporting fact Exact Match.
    Both are lists of (title, sentence_idx) tuples.
    EM = 1 iff predicted set == gold set.
    """
    return float(set(predicted_sps) == set(gold_sps))


def compute_sp_f1(
    predicted_sps: list[tuple[str, int]],
    gold_sps: list[tuple[str, int]],
) -> tuple[float, float, float]:
    """
    Supporting fact F1.
    Returns: (precision, recall, f1)
    """
    pred_set = set(predicted_sps)
    gold_set = set(gold_sps)

    if len(gold_set) == 0:
        return (float(len(pred_set) == 0),) * 3
    if len(pred_set) == 0:
        return (0.0, 0.0, 0.0)

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)

    if precision + recall == 0:
        return (0.0, 0.0, 0.0)

    f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


# ════════════════════════════════════════════════════════════════════════
# Paragraph Selection Metrics
# ════════════════════════════════════════════════════════════════════════

def compute_paragraph_metrics(
    selected_indices: tuple[int, int],
    gold_indices: set[int],
) -> dict:
    """
    Evaluate paragraph pair selection against gold.

    Args:
        selected_indices: (anchor_idx, answer_idx) from argmax of q_i, q_j
        gold_indices: set of gold paragraph indices (typically 2)

    Returns dict with:
        pair_em: 1.0 iff both selected paragraphs are gold
        recall: fraction of gold paragraphs that were selected
        precision: fraction of selected paragraphs that are gold
    """
    selected_set = set(selected_indices)

    tp = len(selected_set & gold_indices)
    precision = tp / len(selected_set) if selected_set else 0.0
    recall = tp / len(gold_indices) if gold_indices else 0.0
    pair_em = float(selected_set == gold_indices)

    return {
        "pair_em": pair_em,
        "precision": precision,
        "recall": recall,
    }


# ════════════════════════════════════════════════════════════════════════
# Sufficiency & Comprehensiveness
# ════════════════════════════════════════════════════════════════════════

def compute_sufficiency(
    selected_paragraph_texts: list[str],
    gold_answer: str,
) -> float:
    """
    Sufficiency: does the gold answer appear in the selected paragraphs?

    Uses word-boundary regex to prevent partial-word false positives.
    E.g., gold="18" must not match "1800"; gold="ton" must not match "Washington".

    REVIEWER FIX: Previously used bare substring matching (answer_normalized
    in combined_text), which was dangerously permissive for short answers.
    """
    combined_text = normalize_answer(" ".join(selected_paragraph_texts))
    answer_normalized = normalize_answer(gold_answer)

    if not answer_normalized:
        return 1.0

    # Enforce word boundaries to prevent partial-word matches
    pattern = r'\b' + re.escape(answer_normalized) + r'\b'
    return float(bool(re.search(pattern, combined_text)))


def compute_comprehensiveness(
    predicted_sps_per_paragraph: dict[int, list[int]],
) -> float:
    """
    Comprehensiveness: does the model use BOTH selected paragraphs?

    Args:
        predicted_sps_per_paragraph: {paragraph_idx: [sentence_indices]}
            Only includes paragraphs that have at least one predicted
            supporting sentence.

    Returns 1.0 if supporting sentences span 2+ paragraphs, 0.0 otherwise.

    A model that collapses to single-paragraph reasoning will have
    comprehensiveness near 0 — it finds evidence in only one paragraph
    even when two are selected.
    """
    paragraphs_with_evidence = sum(
        1 for sents in predicted_sps_per_paragraph.values() if len(sents) > 0
    )
    return float(paragraphs_with_evidence >= 2)


# ════════════════════════════════════════════════════════════════════════
# Evaluation Results Container
# ════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResults:
    """Aggregated evaluation results across the full eval set."""

    # Answer metrics
    answer_em: float = 0.0
    answer_f1: float = 0.0

    # Supporting fact metrics
    sp_em: float = 0.0
    sp_precision: float = 0.0
    sp_recall: float = 0.0
    sp_f1: float = 0.0

    # Joint metrics (HotpotQA official)
    joint_em: float = 0.0
    joint_f1: float = 0.0

    # Paragraph selection
    pair_em: float = 0.0
    para_precision: float = 0.0
    para_recall: float = 0.0

    # Sufficiency & comprehensiveness
    sufficiency: float = 0.0
    comprehensiveness: float = 0.0

    # Breakdown by question type
    bridge_answer_em: float = 0.0
    bridge_answer_f1: float = 0.0
    comparison_answer_em: float = 0.0
    comparison_answer_f1: float = 0.0

    # Counts
    total_questions: int = 0
    bridge_count: int = 0
    comparison_count: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "EVALUATION RESULTS",
            "=" * 60,
            f"Total questions: {self.total_questions} "
            f"(bridge: {self.bridge_count}, comparison: {self.comparison_count})",
            "",
            "Answer Metrics:",
            f"  EM:  {self.answer_em:.4f}",
            f"  F1:  {self.answer_f1:.4f}",
            "",
            "Supporting Fact Metrics:",
            f"  EM:        {self.sp_em:.4f}",
            f"  Precision: {self.sp_precision:.4f}",
            f"  Recall:    {self.sp_recall:.4f}",
            f"  F1:        {self.sp_f1:.4f}",
            "",
            "Joint Metrics (HotpotQA official):",
            f"  Joint EM:  {self.joint_em:.4f}",
            f"  Joint F1:  {self.joint_f1:.4f}",
            "",
            "Paragraph Selection:",
            f"  Pair EM:   {self.pair_em:.4f}",
            f"  Precision: {self.para_precision:.4f}",
            f"  Recall:    {self.para_recall:.4f}",
            "",
            "Sufficiency & Comprehensiveness:",
            f"  Sufficiency:       {self.sufficiency:.4f}",
            f"  Comprehensiveness: {self.comprehensiveness:.4f}",
            "",
            "By Question Type:",
            f"  Bridge     EM: {self.bridge_answer_em:.4f}  F1: {self.bridge_answer_f1:.4f}",
            f"  Comparison EM: {self.comparison_answer_em:.4f}  F1: {self.comparison_answer_f1:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# Main Evaluation Function
# ════════════════════════════════════════════════════════════════════════

@dataclass
class EvalInstance:
    """
    A single evaluation instance with all metadata needed for metrics.
    Created by the dataset class alongside SROBatch.
    """
    question_id: str
    question_text: str
    qtype: str                          # "bridge" or "comparison"
    candidates: list[str]               # M candidate answers
    gold_answer: str
    gold_paragraph_indices: set[int]    # indices of gold paragraphs in the 10
    gold_supporting_facts: list[tuple[str, int]]  # (title, sentence_idx)
    paragraph_titles: list[str]         # titles of all 10 paragraphs
    paragraph_texts: list[str]          # full text of all 10 paragraphs


def evaluate(
    model,
    eval_loader: DataLoader,
    eval_instances: list[EvalInstance],
    device: torch.device,
    sent_threshold: float = 0.5,
) -> EvalResults:
    """
    Full evaluation over the eval set.

    For each question:
      1. Run the model on all M candidates.
      2. Select the candidate with the highest verifier confidence.
      3. Extract paragraph selections (argmax of q_i, q_j).
      4. Extract supporting fact predictions from [SENT] logits.
      5. Compute all metrics against gold labels.

    Args:
        model: SROModel in eval mode (deterministic argmax, no Gumbel noise)
        eval_loader: yields SROBatch instances (one batch per question,
                     with M candidates as the batch dimension)
        eval_instances: parallel list of EvalInstance metadata
        device: torch device
        sent_threshold: sigmoid threshold for [SENT] classification
    """
    model.eval()

    # Accumulators
    all_answer_em = []
    all_answer_f1 = []
    all_sp_em = []
    all_sp_prec = []
    all_sp_rec = []
    all_sp_f1 = []
    all_joint_em = []
    all_joint_f1 = []
    all_pair_em = []
    all_para_prec = []
    all_para_rec = []
    all_sufficiency = []
    all_comprehensiveness = []

    bridge_em, bridge_f1 = [], []
    comparison_em, comparison_f1 = [], []

    with torch.no_grad():
        for batch_idx, (batch, instance) in enumerate(zip(eval_loader, eval_instances)):
            batch = _move_batch_to_device(batch, device)

            # ── Step 1: Forward pass on all M candidates ───────────
            # batch dimension is M (number of candidates for this question)
            output = model(batch, tau=0.1)  # tau irrelevant in eval (deterministic argmax)

            # ── Step 2: Select best candidate by verifier confidence ─
            if output.verifier_logits is not None:
                confidences = torch.sigmoid(output.verifier_logits)  # (M,)
                best_idx = confidences.argmax().item()
            else:
                best_idx = 0

            predicted_answer = instance.candidates[best_idx]

            # ── Step 3: Extract paragraph selections ───────────────
            # q_i, q_j are one-hot in eval mode (deterministic argmax)
            anchor_idx = output.q_i[best_idx].argmax().item()
            answer_idx = output.q_j[best_idx].argmax().item()
            selected_indices = (anchor_idx, answer_idx)

            # ── Step 4: Extract supporting fact predictions ────────
            # sent_logits from the model output (not in SROOutput currently,
            # so we re-extract from the forward pass internals)
            predicted_sps = _extract_supporting_facts(
                model, batch, best_idx, instance, sent_threshold
            )

            # Supporting facts broken down by paragraph
            sps_per_paragraph = {}
            for title, sent_idx in predicted_sps:
                para_idx = instance.paragraph_titles.index(title) if title in instance.paragraph_titles else -1
                if para_idx >= 0:
                    sps_per_paragraph.setdefault(para_idx, []).append(sent_idx)

            # ── Step 5: Compute all metrics ────────────────────────

            # Answer EM & F1
            a_em = compute_em(predicted_answer, instance.gold_answer)
            a_f1 = compute_f1(predicted_answer, instance.gold_answer)
            all_answer_em.append(a_em)
            all_answer_f1.append(a_f1)

            # Supporting fact EM & F1
            s_em = compute_sp_em(predicted_sps, instance.gold_supporting_facts)
            s_prec, s_rec, s_f1 = compute_sp_f1(predicted_sps, instance.gold_supporting_facts)
            all_sp_em.append(s_em)
            all_sp_prec.append(s_prec)
            all_sp_rec.append(s_rec)
            all_sp_f1.append(s_f1)

            # Joint metrics
            all_joint_em.append(float(a_em == 1.0 and s_em == 1.0))
            all_joint_f1.append(a_f1 * s_f1)

            # Paragraph selection
            para_metrics = compute_paragraph_metrics(
                selected_indices, instance.gold_paragraph_indices
            )
            all_pair_em.append(para_metrics["pair_em"])
            all_para_prec.append(para_metrics["precision"])
            all_para_rec.append(para_metrics["recall"])

            # Sufficiency
            selected_texts = [
                instance.paragraph_texts[anchor_idx],
                instance.paragraph_texts[answer_idx],
            ]
            all_sufficiency.append(
                compute_sufficiency(selected_texts, instance.gold_answer)
            )

            # Comprehensiveness
            # Only count paragraphs that are in the selected pair
            selected_sps = {
                k: v for k, v in sps_per_paragraph.items()
                if k in selected_indices
            }
            all_comprehensiveness.append(
                compute_comprehensiveness(selected_sps)
            )

            # By question type
            if instance.qtype == "bridge":
                bridge_em.append(a_em)
                bridge_f1.append(a_f1)
            else:
                comparison_em.append(a_em)
                comparison_f1.append(a_f1)

    # ── Aggregate ──────────────────────────────────────────────────

    n = len(all_answer_em)

    results = EvalResults(
        answer_em=_mean(all_answer_em),
        answer_f1=_mean(all_answer_f1),
        sp_em=_mean(all_sp_em),
        sp_precision=_mean(all_sp_prec),
        sp_recall=_mean(all_sp_rec),
        sp_f1=_mean(all_sp_f1),
        joint_em=_mean(all_joint_em),
        joint_f1=_mean(all_joint_f1),
        pair_em=_mean(all_pair_em),
        para_precision=_mean(all_para_prec),
        para_recall=_mean(all_para_rec),
        sufficiency=_mean(all_sufficiency),
        comprehensiveness=_mean(all_comprehensiveness),
        bridge_answer_em=_mean(bridge_em),
        bridge_answer_f1=_mean(bridge_f1),
        comparison_answer_em=_mean(comparison_em),
        comparison_answer_f1=_mean(comparison_f1),
        total_questions=n,
        bridge_count=len(bridge_em),
        comparison_count=len(comparison_em),
    )

    return results


# ════════════════════════════════════════════════════════════════════════
# Supporting Fact Extraction
# ════════════════════════════════════════════════════════════════════════

def _extract_supporting_facts(
    model,
    batch,
    candidate_idx: int,
    instance: EvalInstance,
    threshold: float,
) -> list[tuple[str, int]]:
    """
    Extract predicted supporting facts from [SENT] token logits.

    Runs a targeted forward pass through the encoder to get sent_logits,
    then thresholds at the given value.

    Returns list of (title, sentence_idx) tuples.
    """
    # Get sent_logits from encoder output
    # In a full implementation, these would be cached from the main forward pass.
    # For now, we extract them from the model's paragraph encoder.

    predicted_sps = []

    # The model's forward pass already computed sent_logits internally.
    # In a production implementation, add sent_logits to SROOutput.
    # For this evaluation, we re-run the encoder for the selected candidate.

    # Simplified extraction using the model's [SENT] token positions:
    # For each paragraph, check which [SENT] positions have logits > threshold
    with torch.no_grad():
        p_bridge = model.qtype_gate(
            model.paragraph_encoder.encoder(
                input_ids=batch.question_input_ids[candidate_idx:candidate_idx+1],
                attention_mask=batch.question_attention_mask[candidate_idx:candidate_idx+1],
            ).last_hidden_state[:, 0, :]
        )

        enc_out = model.paragraph_encoder(
            paragraph_input_ids=batch.paragraph_input_ids[candidate_idx:candidate_idx+1],
            paragraph_attention_mask=batch.paragraph_attention_mask[candidate_idx:candidate_idx+1],
            question_input_ids=batch.question_input_ids[candidate_idx:candidate_idx+1],
            question_attention_mask=batch.question_attention_mask[candidate_idx:candidate_idx+1],
            p_bridge=p_bridge,
        )

        sent_logits = enc_out["sent_logits"]  # (1, N, S_max)
        sent_probs = torch.sigmoid(sent_logits[0])  # (N, S_max)

        N = sent_probs.size(0)
        for para_idx in range(N):
            title = instance.paragraph_titles[para_idx]
            for sent_idx in range(sent_probs.size(1)):
                if sent_probs[para_idx, sent_idx] > threshold:
                    predicted_sps.append((title, sent_idx))

    return predicted_sps


# ════════════════════════════════════════════════════════════════════════
# Held-Out Test Evaluation (1000-question contractual set)
# ════════════════════════════════════════════════════════════════════════

def evaluate_held_out(
    model,
    test_loader: DataLoader,
    test_instances: list[EvalInstance],
    device: torch.device,
    output_path: str = "./eval_results.txt",
) -> EvalResults:
    """
    Final evaluation on the 1000-question held-out test set.

    This function is called ONCE at the end of the project.
    The held-out set has never been touched during training or
    hyperparameter tuning.

    Results are saved to a file for reporting.
    """
    logger.info("=" * 60)
    logger.info("HELD-OUT TEST SET EVALUATION")
    logger.info("This set has never been used during training or tuning.")
    logger.info("=" * 60)

    results = evaluate(model, test_loader, test_instances, device)

    # Save results
    summary = results.summary()
    logger.info(summary)

    with open(output_path, "w") as f:
        f.write(summary)
    logger.info(f"Results saved to {output_path}")

    return results


# ════════════════════════════════════════════════════════════════════════
# Diagnostic: Per-Instance Error Analysis
# ════════════════════════════════════════════════════════════════════════

def error_analysis(
    model,
    eval_loader: DataLoader,
    eval_instances: list[EvalInstance],
    device: torch.device,
    output_path: str = "./error_analysis.jsonl",
) -> dict:
    """
    Detailed per-instance error analysis for debugging.

    Categorizes every prediction into one of five types:

      - strict_correct:     right answer AND right paragraphs (genuine multi-hop)
      - spurious_shortcut:  right answer but WRONG paragraphs (distractor leakage)
      - selection_error:    wrong paragraphs, wrong answer
      - verification_error: right paragraphs, answer in text, but wrong candidate
      - insufficiency_error: right paragraphs, but answer not in their text

    REVIEWER FIX: Previously checked EM first without verifying paragraph
    selection, which counted distractor-leakage shortcuts as "correct" —
    directly hiding the shortcut reasoning this paper exists to study.
    The fix enforces joint structural correctness: a prediction is only
    "strict_correct" if BOTH the answer and the paragraph selection are right.

    Returns distribution of error types.
    """
    import json

    model.eval()
    error_counts = {
        "strict_correct": 0,
        "spurious_shortcut": 0,
        "selection_error": 0,
        "verification_error": 0,
        "insufficiency_error": 0,
    }

    with open(output_path, "w") as f:
        with torch.no_grad():
            for batch, instance in zip(eval_loader, eval_instances):
                batch = _move_batch_to_device(batch, device)
                output = model(batch, tau=0.1)

                if output.verifier_logits is not None:
                    best_idx = torch.sigmoid(output.verifier_logits).argmax().item()
                else:
                    best_idx = 0

                predicted_answer = instance.candidates[best_idx]
                anchor_idx = output.q_i[best_idx].argmax().item()
                answer_idx = output.q_j[best_idx].argmax().item()
                selected = {anchor_idx, answer_idx}

                em = compute_em(predicted_answer, instance.gold_answer)
                is_selected_correct = (selected == instance.gold_paragraph_indices)

                # ── Joint structural correctness ───────────────────
                if em == 1.0 and is_selected_correct:
                    error_type = "strict_correct"
                elif em == 1.0 and not is_selected_correct:
                    # Right answer, wrong paragraphs = distractor leakage
                    error_type = "spurious_shortcut"
                elif not is_selected_correct:
                    error_type = "selection_error"
                else:
                    # Right paragraphs, wrong answer — why?
                    selected_texts = [
                        instance.paragraph_texts[anchor_idx],
                        instance.paragraph_texts[answer_idx],
                    ]
                    suff = compute_sufficiency(selected_texts, instance.gold_answer)
                    if suff == 0.0:
                        error_type = "insufficiency_error"
                    else:
                        error_type = "verification_error"

                error_counts[error_type] += 1

                # Write per-instance record
                record = {
                    "question_id": instance.question_id,
                    "qtype": instance.qtype,
                    "error_type": error_type,
                    "predicted_answer": predicted_answer,
                    "gold_answer": instance.gold_answer,
                    "selected_paragraphs": list(selected),
                    "gold_paragraphs": list(instance.gold_paragraph_indices),
                    "paragraph_selection_correct": is_selected_correct,
                    "em": em,
                }
                f.write(json.dumps(record) + "\n")

    total = sum(error_counts.values())
    logger.info("Error Analysis:")
    for error_type, count in error_counts.items():
        pct = count / total * 100 if total > 0 else 0
        logger.info(f"  {error_type}: {count} ({pct:.1f}%)")

    return error_counts


# ════════════════════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════════════════════

def _mean(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def _move_batch_to_device(batch, device):
    for field_name in batch.__dataclass_fields__:
        val = getattr(batch, field_name)
        if isinstance(val, torch.Tensor):
            setattr(batch, field_name, val.to(device))
    return batch
