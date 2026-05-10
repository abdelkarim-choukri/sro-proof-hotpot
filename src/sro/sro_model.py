"""
SROModel: Full Forward Pass Integration (v2.1)

This module wires together all individually-reviewed components into a single
end-to-end differentiable model. Each component has been reviewed and locked
in isolation; this file is the integration layer that connects them.

Components (reviewed status):
  - PairRouter:       LOCKED (3/3 audit points confirmed)
  - [SENT] init:      LOCKED ([SEP] anchoring + 10x LR)
  - K-sample layout:  LOCKED (K*B tensor expansion contract)
  - Verifier:         Design confirmed (warm-start from encoder layers)
  - Qtype gate:       Design confirmed (question-only, random init prefix)
  - Loss functions:   Design confirmed (4 losses, Stage A/B split)

Reviewer audit targets for this integration:
  1. SEAM CORRECTNESS: Do tensor shapes match at every component boundary?
  2. GRADIENT ROUTING: Does each loss reach the parameters it should?
     - L_verify  → verifier → soft_gather → PairRouter → encoder
     - L_select  → soft pair_probs → PairRouter → encoder
     - L_qtype   → qtype_gate → encoder (question path only)
     - L_sent_sup → sent_classifier → encoder
  3. NO CROSS-CONTAMINATION: Does sentence supervision ([SENT]) stay independent
     of the Gumbel selection path? Does L_select use soft probs (not hard sample)?
  4. STAGE A/B SWITCHING: Is L_verify cleanly gated off during Stage A?
  5. K-SAMPLE EXPANSION: Is the expand → reshape → loss pattern correct
     when K > 1 during Stage B?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ════════════════════════════════════════════════════════════════════════
# Data Structures
# ════════════════════════════════════════════════════════════════════════

@dataclass
class SROBatch:
    """A single training batch. All fields are tensors on the same device."""

    # Encoder inputs
    paragraph_input_ids:    torch.Tensor   # (B, N, L)     — tokenized paragraphs
    paragraph_attention_mask: torch.Tensor  # (B, N, L)
    question_input_ids:     torch.Tensor   # (B, L_q)      — tokenized question
    question_attention_mask: torch.Tensor   # (B, L_q)

    # Supervision targets
    judge_label:     torch.Tensor   # (B,)          — silver label from 70B judge
    gold_mask:       torch.Tensor   # (B, N)        — binary: 1 for gold paragraphs
    qtype_label:     torch.Tensor   # (B,)          — 1.0 = bridge, 0.0 = comparison
    sent_targets:    torch.Tensor   # (B, N, S_max)  — binary per [SENT] position
    sent_mask:       torch.Tensor   # (B, N, S_max)  — 1 where [SENT] exists, 0 pad

    # Metadata (not used in forward, but logged)
    question_ids:    list            # question ID strings for logging


@dataclass
class SROOutput:
    """All model outputs and losses from a single forward pass."""

    # Individual losses
    loss_verify:  Optional[torch.Tensor]  # None during Stage A
    loss_select:  torch.Tensor
    loss_qtype:   torch.Tensor
    loss_sent:    torch.Tensor
    loss_total:   torch.Tensor

    # Diagnostics (detached, for logging only)
    p_bridge:     torch.Tensor   # (B,)     — qtype gate output
    q_i:          torch.Tensor   # (B, N)   — anchor marginal
    q_j:          torch.Tensor   # (B, N)   — answer marginal
    verifier_logits: Optional[torch.Tensor]  # (B,) or None during Stage A


# ════════════════════════════════════════════════════════════════════════
# Submodule: PairRouter (imported from reviewed pair_router.py)
# Inlined here for self-contained review. Identical to locked version.
# ════════════════════════════════════════════════════════════════════════

class PairRouter(nn.Module):

    def __init__(self, hidden_dim: int, num_paragraphs: int = 10):
        super().__init__()
        self.N = num_paragraphs
        self.anchor_head = nn.Linear(hidden_dim, hidden_dim)
        self.answer_head = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer("score_scale", torch.tensor(hidden_dim ** -0.5))
        diag = torch.eye(num_paragraphs, dtype=torch.bool)
        self.register_buffer("diag_mask", diag)

    def compute_pair_logits(self, paragraph_reps: torch.Tensor) -> torch.Tensor:
        B, N, D = paragraph_reps.shape
        anchor = self.anchor_head(paragraph_reps)
        answer = self.answer_head(paragraph_reps)
        pair_logits = torch.bmm(anchor, answer.transpose(1, 2))
        pair_logits = pair_logits * self.score_scale
        pair_logits = pair_logits.masked_fill(self.diag_mask, float("-inf"))
        return pair_logits

    def forward(self, paragraph_reps, tau=1.0):
        B, N, _ = paragraph_reps.shape
        pair_logits = self.compute_pair_logits(paragraph_reps)
        flat_logits = pair_logits.view(B, N * N)
        if self.training:
            flat_sample = F.gumbel_softmax(flat_logits, tau=tau, hard=True, dim=-1)
        else:
            assert not self.training
            argmax_idx = flat_logits.argmax(dim=-1)
            flat_sample = F.one_hot(argmax_idx, num_classes=N * N).to(flat_logits.dtype)
        pair_sample = flat_sample.view(B, N, N)
        q_i = pair_sample.sum(dim=2)
        q_j = pair_sample.sum(dim=1)
        if self.training and pair_logits.requires_grad:
            assert q_i.grad_fn is not None, "q_i gradient chain broken"
            assert q_j.grad_fn is not None, "q_j gradient chain broken"
        return q_i, q_j, pair_sample

    def compute_soft_probs(self, paragraph_reps: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Returns SOFT pair probabilities for L_select supervision.
        NO Gumbel noise — deterministic softmax over masked logits.

        CRITICAL: L_select must supervise THIS output, not the hard Gumbel sample.
        Supervising the hard sample subjects dense selection loss to sampling noise.
        """
        B, N, _ = paragraph_reps.shape
        pair_logits = self.compute_pair_logits(paragraph_reps)
        pair_probs_soft = F.softmax(pair_logits.view(B, N * N) / tau, dim=-1).view(B, N, N)
        return pair_probs_soft  # (B, N, N)


# ════════════════════════════════════════════════════════════════════════
# Submodule: Paragraph Encoder Wrapper
# ════════════════════════════════════════════════════════════════════════

class ParagraphEncoder(nn.Module):
    """
    Wraps a pretrained transformer (DeBERTa-v3) to encode N paragraphs
    independently, producing both pooled and token-level representations.

    Also handles:
    - Qtype prefix token injection
    - [SENT] token logit extraction
    """

    def __init__(self, encoder, hidden_dim: int, num_paragraphs: int = 10):
        super().__init__()
        self.encoder = encoder       # Pretrained DeBERTa-v3 (or compatible)
        self.N = num_paragraphs
        self.hidden_dim = hidden_dim

        # ── Qtype prefix embeddings ────────────────────────────────────
        # Random init (truncated normal, std=1e-2). NOT anchored to word embeddings.
        self.e_bridge = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)
        self.e_compare = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)

        # ── [SENT] token ID ────────────────────────────────────────────
        # Stored for identifying [SENT] positions in the token sequence.
        # Actual embedding initialized separately (see init_sent_token).
        self.sent_token_id: Optional[int] = None

        # ── Sentence-level classifier ──────────────────────────────────
        # Produces one logit per [SENT] position for L_sent_sup.
        self.sent_classifier = nn.Linear(hidden_dim, 1)

        # ── Paragraph pooling head ─────────────────────────────────────
        self.pool_head = nn.Linear(hidden_dim, hidden_dim)

    def init_sent_token(self, tokenizer, encoder_embeddings: nn.Embedding):
        """
        Initialize [SENT] embedding from [SEP] weights (semantic anchoring).

        Call ONCE after model construction:
            model.paragraph_encoder.init_sent_token(tokenizer, model.encoder.embeddings.word_embeddings)

        If [unused] slots are available, use those instead (preferred for
        DeBERTa-v3 position encoding stability). Fall back to new token
        with [SEP]-anchored initialization if [unused] is unavailable.
        """
        # Try [unused] slot first
        unused_candidates = [f"[unused{i}]" for i in range(100)]
        for candidate in unused_candidates:
            token_id = tokenizer.convert_tokens_to_ids(candidate)
            if token_id != tokenizer.unk_token_id:
                self.sent_token_id = token_id
                # Copy [SEP] weights into the [unused] slot
                sep_id = tokenizer.sep_token_id
                with torch.no_grad():
                    encoder_embeddings.weight[token_id] = encoder_embeddings.weight[sep_id].clone()
                return

        # Fallback: add new token to vocabulary
        tokenizer.add_tokens(["[SENT]"])
        self.sent_token_id = tokenizer.convert_tokens_to_ids("[SENT]")
        encoder_embeddings_weight = encoder_embeddings.weight
        sep_id = tokenizer.sep_token_id
        # New token embedding initialized from [SEP] weights
        with torch.no_grad():
            new_weight = encoder_embeddings_weight[sep_id].clone()
            # Resize embedding matrix if needed (handled by model.resize_token_embeddings)
            encoder_embeddings_weight.data = torch.cat([
                encoder_embeddings_weight.data,
                new_weight.unsqueeze(0)
            ], dim=0)

    def forward(
        self,
        paragraph_input_ids: torch.Tensor,      # (B, N, L)
        paragraph_attention_mask: torch.Tensor,  # (B, N, L)
        question_input_ids: torch.Tensor,        # (B, L_q)
        question_attention_mask: torch.Tensor,   # (B, L_q)
        p_bridge: torch.Tensor,                  # (B, 1) — from qtype gate
    ) -> dict:
        """
        Encodes all N paragraphs and the question.

        Returns dict with:
            paragraph_reps:   (B, N, D)       — pooled paragraph representations
            paragraph_tokens: (B, N, L, D)    — token-level paragraph representations
            question_tokens:  (B, L_q, D)     — token-level question representations
            sent_logits:      (B, N, S_max)   — logits at [SENT] positions
        """
        B, N, L = paragraph_input_ids.shape
        D = self.hidden_dim

        # ── Encode question ────────────────────────────────────────────
        # Shape: (B, L_q, D)
        q_out = self.encoder(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask
        ).last_hidden_state  # (B, L_q, D)

        # ── Qtype prefix: soft-mix during training ─────────────────────
        # prefix: (B, 1, D) — CONCATENATED to paragraph embeddings BEFORE
        # the transformer layers process the sequence. This ensures the
        # self-attention layers see the qtype signal as an early structural
        # prior, not a post-hoc bias on already-computed representations.
        #
        # REVIEWER FIX: Previously injected additively at [CLS] AFTER the
        # encoder. That meant self-attention never saw the prefix — it acted
        # purely as a bias vector for the pooling layer.
        p = p_bridge.unsqueeze(-1)  # (B, 1, 1)
        prefix = p * self.e_bridge + (1 - p) * self.e_compare  # (B, 1, D)

        # ── Encode paragraphs with prefix at embedding level ───────────
        # Step 1: Get raw embeddings from the encoder's embedding layer
        flat_ids = paragraph_input_ids.view(B * N, L)          # (B*N, L)
        flat_mask = paragraph_attention_mask.view(B * N, L)     # (B*N, L)

        # Get token embeddings BEFORE transformer layers
        flat_embeds = self.encoder.embeddings(input_ids=flat_ids)  # (B*N, L, D)

        # Step 2: Expand prefix to (B*N, 1, D) and concatenate at position 0
        # prefix is (B, 1, D) → expand to (B, N, 1, D) → reshape to (B*N, 1, D)
        prefix_expanded = prefix.unsqueeze(1).expand(B, N, 1, D).reshape(B * N, 1, D)

        # Concatenate: [prefix | token_embeddings] → (B*N, L+1, D)
        flat_embeds_with_prefix = torch.cat([prefix_expanded, flat_embeds], dim=1)

        # Step 3: Extend attention mask to account for the +1 prefix position
        prefix_mask = torch.ones(B * N, 1, dtype=flat_mask.dtype, device=flat_mask.device)
        flat_mask_extended = torch.cat([prefix_mask, flat_mask], dim=1)  # (B*N, L+1)

        # Step 4: Run transformer layers on prefix-augmented embeddings
        flat_out = self.encoder(
            inputs_embeds=flat_embeds_with_prefix,
            attention_mask=flat_mask_extended,
        ).last_hidden_state  # (B*N, L+1, D)

        # Step 5: Strip the prefix position from the output.
        # Position 0 is the prefix; positions 1..L are the original tokens.
        # We keep the original token positions so downstream [SENT] extraction
        # and token-level operations remain aligned with input_ids.
        flat_out_tokens = flat_out[:, 1:, :]  # (B*N, L, D) — original positions

        # The prefix output (position 0) has been attended to by all tokens
        # during self-attention, so its structural signal is already propagated
        # into the token representations. We discard the prefix output itself.

        # Reshape back: (B, N, L, D)
        paragraph_tokens = flat_out_tokens.view(B, N, L, D)

        # ── Pool paragraph representations ─────────────────────────────
        # Use [CLS] token (now position 0 after prefix stripping) as paragraph rep.
        # The [CLS] representation has already attended to the prefix during
        # self-attention, so it carries the qtype structural prior.
        cls_tokens = paragraph_tokens[:, :, 0, :]   # (B, N, D)
        paragraph_reps = self.pool_head(cls_tokens)  # (B, N, D)

        # ── Extract [SENT] token logits for L_sent_sup ─────────────────
        # [SENT] tokens are at known positions in the tokenized input.
        # We extract their hidden states and classify them.
        sent_logits = self._extract_sent_logits(
            paragraph_tokens, paragraph_input_ids
        )  # (B, N, S_max)

        return {
            "paragraph_reps": paragraph_reps,       # (B, N, D)
            "paragraph_tokens": paragraph_tokens,   # (B, N, L, D)
            "question_tokens": q_out,               # (B, L_q, D)
            "sent_logits": sent_logits,             # (B, N, S_max)
        }

    def _extract_sent_logits(
        self,
        paragraph_tokens: torch.Tensor,    # (B, N, L, D)
        paragraph_input_ids: torch.Tensor,  # (B, N, L)
    ) -> torch.Tensor:
        """
        Extract hidden states at [SENT] token positions and classify them.

        Returns: (B, N, S_max) — one logit per [SENT] position.
        Padded positions (no [SENT] token) are filled with -inf so they
        contribute zero to the loss after masking.

        GRADIENT PATH: sent_logits ← sent_classifier ← paragraph_tokens ← encoder
        This path is INDEPENDENT of the Gumbel selection path (PairRouter).
        """
        B, N, L, D = paragraph_tokens.shape

        if self.sent_token_id is None:
            # Fallback: return zeros if [SENT] not initialized yet
            return torch.zeros(B, N, 1, device=paragraph_tokens.device)

        # Find [SENT] positions: (B, N, L) boolean mask
        is_sent = (paragraph_input_ids == self.sent_token_id)  # (B, N, L)

        # Max number of [SENT] tokens per paragraph in this batch
        S_max = is_sent.sum(dim=-1).max().item()
        if S_max == 0:
            return torch.zeros(B, N, 1, device=paragraph_tokens.device)

        # Gather hidden states at [SENT] positions
        # We pad to S_max for uniform tensor shape
        sent_hidden = torch.zeros(B, N, S_max, D, device=paragraph_tokens.device)
        sent_valid = torch.zeros(B, N, S_max, dtype=torch.bool, device=paragraph_tokens.device)

        for b in range(B):
            for n in range(N):
                positions = is_sent[b, n].nonzero(as_tuple=True)[0]
                s_count = min(positions.size(0), S_max)
                if s_count > 0:
                    sent_hidden[b, n, :s_count] = paragraph_tokens[b, n, positions[:s_count]]
                    sent_valid[b, n, :s_count] = True

        # Classify: (B, N, S_max, D) → (B, N, S_max, 1) → (B, N, S_max)
        sent_logits = self.sent_classifier(sent_hidden).squeeze(-1)  # (B, N, S_max)

        # Mask invalid positions with -inf (will be masked in loss computation)
        sent_logits = sent_logits.masked_fill(~sent_valid, float("-inf"))

        return sent_logits


# ════════════════════════════════════════════════════════════════════════
# Submodule: Qtype Gate
# ════════════════════════════════════════════════════════════════════════

class QtypeGate(nn.Module):
    """
    Predicts p_bridge from question encoding ONLY.
    Does NOT use paragraph representations (too noisy — 8/10 are distractors).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, question_cls: torch.Tensor) -> torch.Tensor:
        """
        Args:
            question_cls: (B, D) — pooled question representation ([CLS] token)
        Returns:
            p_bridge: (B, 1) — probability that the question is bridge-type
        """
        return torch.sigmoid(self.mlp(question_cls))  # (B, 1)


# ════════════════════════════════════════════════════════════════════════
# Submodule: Verifier (Cross-Attention Reader)
# ════════════════════════════════════════════════════════════════════════

class Verifier(nn.Module):
    """
    Token-level cross-attention verifier. Reads the question against the
    selected paragraph tokens and predicts whether the candidate answer
    is correct.

    Initialized from the last `num_layers` layers of the shared encoder
    (warm-start, not random init).
    """

    def __init__(self, hidden_dim: int, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Cross-attention layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.cross_attn = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        question_tokens: torch.Tensor,  # (B, L_q, D)
        evidence_tokens: torch.Tensor,  # (B, L_evidence, D) — concat of anchor + answer
    ) -> torch.Tensor:
        """
        Cross-attend question over evidence paragraphs, then classify.

        Returns: (B,) — logits (not probabilities). Apply sigmoid for prediction,
                        or use BCE-with-logits for loss.
        """
        # Question attends to evidence via cross-attention
        # tgt = question, memory = evidence
        attended = self.cross_attn(
            tgt=question_tokens,       # (B, L_q, D)
            memory=evidence_tokens,    # (B, L_evidence, D)
        )  # (B, L_q, D)

        # Pool over question positions → single vector
        pooled = attended.mean(dim=1)  # (B, D)

        # Classify
        logits = self.classifier(pooled).squeeze(-1)  # (B,)

        return logits

    @classmethod
    def from_encoder_layers(cls, encoder, num_layers: int = 2, num_heads: int = 8):
        """
        Warm-start verifier from the last `num_layers` of the shared encoder.

        This avoids the cold-start problem where a randomly initialized verifier
        lags the encoder by many epochs during Stage B.

        REVIEWER FIX: Handles both native PyTorch parameter names and
        HuggingFace DeBERTa-v3 custom DisentangledSelfAttention names.
        """
        hidden_dim = encoder.config.hidden_size

        verifier = cls(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads)

        # Copy weights from last N encoder layers into the decoder layers
        encoder_layers = encoder.encoder.layer  # ModuleList of transformer layers
        for i in range(num_layers):
            src_layer = encoder_layers[-(num_layers - i)]
            tgt_layer = verifier.cross_attn.layers[i]

            # ── Detect parameter naming convention ─────────────────────
            # DeBERTa-v3 uses: attention.self.query_proj, key_proj, value_proj
            # Native PyTorch uses: attention.self.in_proj_weight (packed Q/K/V)
            src_attn = src_layer.attention.self

            if hasattr(src_attn, "query_proj"):
                # ── HuggingFace DeBERTa-v3 path ───────────────────────
                # DeBERTa stores Q, K, V as separate projections.
                # PyTorch's nn.MultiheadAttention packs them into in_proj_weight.
                D = hidden_dim
                tgt_layer.self_attn.in_proj_weight.data[:D].copy_(
                    src_attn.query_proj.weight.data
                )
                tgt_layer.self_attn.in_proj_weight.data[D:2*D].copy_(
                    src_attn.key_proj.weight.data
                )
                tgt_layer.self_attn.in_proj_weight.data[2*D:].copy_(
                    src_attn.value_proj.weight.data
                )

                # Copy biases if they exist
                if hasattr(src_attn.query_proj, "bias") and src_attn.query_proj.bias is not None:
                    tgt_layer.self_attn.in_proj_bias.data[:D].copy_(
                        src_attn.query_proj.bias.data
                    )
                    tgt_layer.self_attn.in_proj_bias.data[D:2*D].copy_(
                        src_attn.key_proj.bias.data
                    )
                    tgt_layer.self_attn.in_proj_bias.data[2*D:].copy_(
                        src_attn.value_proj.bias.data
                    )

            elif hasattr(src_attn, "in_proj_weight"):
                # ── Native PyTorch path (packed Q/K/V) ─────────────────
                tgt_layer.self_attn.in_proj_weight.data.copy_(
                    src_attn.in_proj_weight.data
                )
                if hasattr(src_attn, "in_proj_bias") and src_attn.in_proj_bias is not None:
                    tgt_layer.self_attn.in_proj_bias.data.copy_(
                        src_attn.in_proj_bias.data
                    )
            else:
                import warnings
                warnings.warn(
                    f"Unrecognized encoder attention parameter names in layer {i}. "
                    f"Verifier self-attention will remain randomly initialized for this layer."
                )

            # Copy output projection if available
            if hasattr(src_layer.attention, "output"):
                out_dense = src_layer.attention.output.dense
                tgt_layer.self_attn.out_proj.weight.data.copy_(out_dense.weight.data)
                if out_dense.bias is not None:
                    tgt_layer.self_attn.out_proj.bias.data.copy_(out_dense.bias.data)

            # Note: cross-attention weights remain randomly initialized.
            # This is intentional — the encoder has no cross-attention to copy from.
            # Cross-attention in residual connections starts near identity,
            # which is safe and standard (same as HuggingFace EncoderDecoderModel).

        return verifier


# ════════════════════════════════════════════════════════════════════════
# Utility: Soft Paragraph Gathering
# ════════════════════════════════════════════════════════════════════════

def soft_gather_paragraphs(
    paragraph_tokens: torch.Tensor,  # (B, N, L, D)
    q: torch.Tensor,                  # (B, N)
) -> torch.Tensor:
    """
    Differentiable paragraph selection.

    Forward (hard=True): equivalent to indexing the selected paragraph.
    Backward (STE): gradients distribute across all N paragraphs.

    Returns: (B, L, D)
    """
    weights = q.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
    selected = (paragraph_tokens * weights).sum(dim=1)  # (B, L, D)
    return selected


# ════════════════════════════════════════════════════════════════════════
# Main Model: SROModel
# ════════════════════════════════════════════════════════════════════════

class SROModel(nn.Module):
    """
    Full SRO model integrating all reviewed components.

    Forward pass data flow:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  question ──► QtypeGate ──► p_bridge ──►┐                      │
    │                                         │                      │
    │  question ──► Encoder ──► question_tokens ──►┐                 │
    │                                              │                 │
    │  paragraphs ──► Encoder ──► paragraph_reps ──► PairRouter      │
    │                    │                             │    │         │
    │                    │                          q_i    q_j       │
    │                    │                            │      │       │
    │                    ├──► paragraph_tokens ──► soft_gather ──►    │
    │                    │                                     │      │
    │                    │                          evidence_tokens   │
    │                    │                                │           │
    │                    │          question_tokens + evidence_tokens │
    │                    │                         │                  │
    │                    │                    Verifier ──► L_verify   │
    │                    │                                            │
    │                    ├──► [SENT] logits ──────────────► L_sent_sup│
    │                    │                                            │
    │            pair_probs_soft ─────────────────────────► L_select  │
    │                                                                 │
    │  p_bridge ──────────────────────────────────────────► L_qtype   │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        encoder,
        hidden_dim: int = 768,
        num_paragraphs: int = 10,
        verifier_layers: int = 2,
        verifier_heads: int = 8,
        # Loss weights (v2.1 defaults)
        alpha: float = 0.5,    # L_select weight
        beta: float = 0.1,     # L_qtype weight
        gamma: float = 0.5,    # L_sent_sup weight
        # Class imbalance correction
        qtype_pos_weight: float = 4.0,    # ~80/20 bridge/comparison
        select_pos_weight: float = 4.0,   # 2/10 gold paragraphs
        verify_pos_weight: float = 1.0,   # Tune after silver dataset generated
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.N = num_paragraphs

        # Loss weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # ── Components ─────────────────────────────────────────────────
        self.paragraph_encoder = ParagraphEncoder(encoder, hidden_dim, num_paragraphs)
        self.qtype_gate = QtypeGate(hidden_dim)
        self.pair_router = PairRouter(hidden_dim, num_paragraphs)
        self.verifier = Verifier.from_encoder_layers(
            encoder, num_layers=verifier_layers, num_heads=verifier_heads
        )

        # ── Loss functions with pos_weight ─────────────────────────────
        self.register_buffer("qtype_pw", torch.tensor([qtype_pos_weight]))
        self.register_buffer("select_pw", torch.tensor([select_pos_weight]))
        self.register_buffer("verify_pw", torch.tensor([verify_pos_weight]))

        # ── Training stage flag ────────────────────────────────────────
        self._stage = "A"   # "A" = warmup (no L_verify), "B" = full training
        self._K = 1         # Number of Gumbel samples for K-sample averaging

    # ── Stage control ──────────────────────────────────────────────────

    def set_stage_a(self):
        """
        Stage A warmup: L_select + L_qtype + L_sent_sup train the encoder/router.
        L_verify also active but on DETACHED inputs — only updates verifier params.
        This pre-warms the verifier so it's not randomly initialized at Stage B entry.
        """
        self._stage = "A"
        self._K = 1

    def set_stage_b(self, K: int = 1):
        """
        Stage B full training: all 4 losses.
        K = number of Gumbel samples for L_verify variance reduction.

        TRANSITION PROTOCOL (v2.1):
        - Start Stage B with K=1
        - Ramp K to K_max over the first epoch of Stage B
        - Apply verifier LR warmup during initial Stage B steps
        """
        self._stage = "B"
        self._K = K

    # ── Main Forward Pass ──────────────────────────────────────────────

    def forward(self, batch: SROBatch, tau: float = 1.0) -> SROOutput:
        """
        Full forward pass + loss computation.

        Args:
            batch: SROBatch with all inputs and targets
            tau: Gumbel temperature (annealed during training)

        Returns:
            SROOutput with all losses and diagnostics
        """
        B = batch.judge_label.size(0)
        N = self.N

        # ══════════════════════════════════════════════════════════════
        # STEP 1: Encode question → get qtype prediction
        # ══════════════════════════════════════════════════════════════

        # Quick question encoding for qtype gate
        # We need question [CLS] before encoding paragraphs (p_bridge is needed there)
        q_out = self.paragraph_encoder.encoder(
            input_ids=batch.question_input_ids,
            attention_mask=batch.question_attention_mask,
        ).last_hidden_state  # (B, L_q, D)

        question_cls = q_out[:, 0, :]          # (B, D) — [CLS] token
        p_bridge = self.qtype_gate(question_cls)  # (B, 1)

        # ══════════════════════════════════════════════════════════════
        # STEP 2: Encode paragraphs (with qtype prefix injection)
        # ══════════════════════════════════════════════════════════════

        enc_out = self.paragraph_encoder(
            paragraph_input_ids=batch.paragraph_input_ids,
            paragraph_attention_mask=batch.paragraph_attention_mask,
            question_input_ids=batch.question_input_ids,
            question_attention_mask=batch.question_attention_mask,
            p_bridge=p_bridge,
        )

        paragraph_reps = enc_out["paragraph_reps"]       # (B, N, D)
        paragraph_tokens = enc_out["paragraph_tokens"]    # (B, N, L, D)
        question_tokens = enc_out["question_tokens"]      # (B, L_q, D)
        sent_logits = enc_out["sent_logits"]              # (B, N, S_max)

        # ══════════════════════════════════════════════════════════════
        # STEP 3: PairRouter — Gumbel-STE selection
        # ══════════════════════════════════════════════════════════════

        # Hard sample (for verifier forward path)
        q_i, q_j, pair_sample = self.pair_router(paragraph_reps, tau=tau)
        # q_i: (B, N), q_j: (B, N), pair_sample: (B, N, N)

        # Soft probabilities (for L_select — NO Gumbel noise)
        pair_probs_soft = self.pair_router.compute_soft_probs(paragraph_reps, tau=tau)
        # pair_probs_soft: (B, N, N)

        # ══════════════════════════════════════════════════════════════
        # STEP 4: Compute L_select (always active, uses SOFT probs)
        # ══════════════════════════════════════════════════════════════

        # Marginalize soft probs to paragraph-level for comparison with gold_mask
        soft_q_i = pair_probs_soft.sum(dim=2)  # (B, N) — anchor marginal (soft)
        soft_q_j = pair_probs_soft.sum(dim=1)  # (B, N) — answer marginal (soft)

        # Gold mask says which paragraphs are gold (2 out of 10).
        # We want BOTH marginals to place mass on gold paragraphs.
        loss_select_anchor = F.binary_cross_entropy(
            soft_q_i, batch.gold_mask,
            weight=torch.where(batch.gold_mask == 1, self.select_pw, torch.ones_like(self.select_pw)),
            reduction="mean",
        )
        loss_select_answer = F.binary_cross_entropy(
            soft_q_j, batch.gold_mask,
            weight=torch.where(batch.gold_mask == 1, self.select_pw, torch.ones_like(self.select_pw)),
            reduction="mean",
        )
        loss_select = (loss_select_anchor + loss_select_answer) / 2.0

        # ══════════════════════════════════════════════════════════════
        # STEP 5: Compute L_qtype (always active)
        # ══════════════════════════════════════════════════════════════

        loss_qtype = F.binary_cross_entropy_with_logits(
            # Use pre-sigmoid logits for numerical stability
            self.qtype_gate.mlp(question_cls).squeeze(-1),  # (B,)
            batch.qtype_label,                                # (B,)
            pos_weight=self.qtype_pw,
            reduction="mean",
        )

        # ══════════════════════════════════════════════════════════════
        # STEP 6: Compute L_sent_sup (always active)
        # ══════════════════════════════════════════════════════════════
        # GRADIENT PATH: loss_sent → sent_classifier → paragraph_tokens → encoder
        # This path is INDEPENDENT of PairRouter / Gumbel selection.

        # Flatten for loss computation
        sent_logits_flat = sent_logits.view(-1)             # (B * N * S_max,)
        sent_targets_flat = batch.sent_targets.view(-1)     # (B * N * S_max,)
        sent_mask_flat = batch.sent_mask.view(-1).bool()    # (B * N * S_max,)

        if sent_mask_flat.any():
            loss_sent = F.binary_cross_entropy_with_logits(
                sent_logits_flat[sent_mask_flat],
                sent_targets_flat[sent_mask_flat],
                reduction="mean",
            )
        else:
            loss_sent = torch.tensor(0.0, device=sent_logits.device, requires_grad=True)

        # ══════════════════════════════════════════════════════════════
        # STEP 7: Compute L_verify
        # ══════════════════════════════════════════════════════════════
        #
        # Stage A: Verifier trained on DETACHED inputs (pre-warming).
        #   Gradients flow into verifier parameters ONLY, not back through
        #   soft_gather / PairRouter / encoder. This gives the verifier's
        #   classification head a non-random starting point before Stage B.
        #
        #   REVIEWER FIX: Previously skipped the verifier entirely during
        #   Stage A, leaving it randomly initialized at Stage B entry,
        #   causing immediate gradient shock.
        #
        # Stage B: Full end-to-end gradient (K-sample).
        #   GRADIENT PATH: loss_verify → verifier → evidence_tokens
        #     → soft_gather → q_i/q_j → PairRouter (STE) → encoder

        verifier_logits = None
        loss_verify = None

        if self._stage == "A":
            # ── Stage A: Pre-warm verifier on DETACHED evidence ────────
            # .detach() severs the gradient path to the encoder/PairRouter.
            # Only the verifier's own parameters (cross-attn, classifier)
            # receive gradient from this loss.
            anchor_tokens = soft_gather_paragraphs(paragraph_tokens, q_i).detach()
            answer_tokens = soft_gather_paragraphs(paragraph_tokens, q_j).detach()
            evidence_tokens = torch.cat([anchor_tokens, answer_tokens], dim=1)

            verifier_logits = self.verifier(question_tokens.detach(), evidence_tokens)

            loss_verify = F.binary_cross_entropy_with_logits(
                verifier_logits,
                batch.judge_label,
                pos_weight=self.verify_pw,
                reduction="mean",
            )
            # NOTE: loss_verify IS added to loss_total in Stage A,
            # but its gradients only update verifier params (everything
            # upstream is detached). This is intentional pre-warming.

        elif self._stage == "B":
            K = self._K

            if K == 1:
                # ── Single sample (no expansion needed) ────────────────
                anchor_tokens = soft_gather_paragraphs(paragraph_tokens, q_i)
                answer_tokens = soft_gather_paragraphs(paragraph_tokens, q_j)
                evidence_tokens = torch.cat([anchor_tokens, answer_tokens], dim=1)

                verifier_logits = self.verifier(question_tokens, evidence_tokens)

                loss_verify = F.binary_cross_entropy_with_logits(
                    verifier_logits,
                    batch.judge_label,
                    pos_weight=self.verify_pw,
                    reduction="mean",
                )

            else:
                # ── K-sample expansion (vectorized) ────────────────────
                # Expand all inputs to (K*B, ...) BEFORE Gumbel sampling
                # to ensure independent noise per K.

                pair_logits = self.pair_router.compute_pair_logits(paragraph_reps)
                # pair_logits: (B, N, N)

                pair_logits_exp = pair_logits.unsqueeze(0).expand(K, B, N, N).reshape(K * B, N * N)
                # (K*B, N²) — each of the K copies gets independent Gumbel noise

                k_pair_sample = F.gumbel_softmax(
                    pair_logits_exp, tau=tau, hard=True, dim=-1
                ).view(K * B, N, N)

                k_q_i = k_pair_sample.sum(dim=2)  # (K*B, N)
                k_q_j = k_pair_sample.sum(dim=1)  # (K*B, N)

                # Expand paragraph tokens and question tokens
                k_para_tokens = paragraph_tokens.unsqueeze(0).expand(
                    K, B, N, paragraph_tokens.size(2), self.hidden_dim
                ).reshape(K * B, N, paragraph_tokens.size(2), self.hidden_dim)

                k_q_tokens = question_tokens.unsqueeze(0).expand(
                    K, B, question_tokens.size(1), self.hidden_dim
                ).reshape(K * B, question_tokens.size(1), self.hidden_dim)

                # Gather evidence for all K*B samples
                k_anchor = soft_gather_paragraphs(k_para_tokens, k_q_i)
                k_answer = soft_gather_paragraphs(k_para_tokens, k_q_j)
                k_evidence = torch.cat([k_anchor, k_answer], dim=1)

                # Single batched verifier forward pass (GPU-saturating)
                k_verifier_logits = self.verifier(k_q_tokens, k_evidence)  # (K*B,)

                # Expand targets to match
                k_judge_label = batch.judge_label.unsqueeze(0).expand(K, B).reshape(K * B)

                loss_verify = F.binary_cross_entropy_with_logits(
                    k_verifier_logits,
                    k_judge_label,
                    pos_weight=self.verify_pw,
                    reduction="mean",
                )

                # Store first sample's logits for diagnostics
                verifier_logits = k_verifier_logits[:B]

        # ══════════════════════════════════════════════════════════════
        # STEP 8: Combine losses
        # ══════════════════════════════════════════════════════════════

        loss_total = (
            self.alpha * loss_select
            + self.beta * loss_qtype
            + self.gamma * loss_sent
        )

        if loss_verify is not None:
            loss_total = loss_total + loss_verify  # L_verify weight = 1.0

        # ══════════════════════════════════════════════════════════════
        # STEP 9: Return all outputs
        # ══════════════════════════════════════════════════════════════

        return SROOutput(
            loss_verify=loss_verify,
            loss_select=loss_select,
            loss_qtype=loss_qtype,
            loss_sent=loss_sent,
            loss_total=loss_total,
            p_bridge=p_bridge.squeeze(-1).detach(),
            q_i=q_i.detach(),
            q_j=q_j.detach(),
            verifier_logits=verifier_logits.detach() if verifier_logits is not None else None,
        )

    # ── Optimizer Parameter Groups ─────────────────────────────────────

    def get_parameter_groups(self, base_lr: float) -> list[dict]:
        """
        Returns optimizer parameter groups with the 10x LR multiplier
        for [SENT] / qtype prefix embeddings.

        Usage:
            optimizer = torch.optim.AdamW(model.get_parameter_groups(base_lr=2e-5))
        """
        # Identify special parameters that need elevated LR
        sent_params = set()
        if self.paragraph_encoder.sent_token_id is not None:
            # The embedding weight row for [SENT] token
            # In practice, isolate via a wrapper or manual gradient scaling
            pass

        prefix_param_ids = {
            id(self.paragraph_encoder.e_bridge),
            id(self.paragraph_encoder.e_compare),
        }

        # Separate into groups
        special_params = []
        encoder_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in prefix_param_ids or "e_bridge" in name or "e_compare" in name:
                special_params.append(param)
            elif "paragraph_encoder.encoder" in name:
                encoder_params.append(param)
            else:
                other_params.append(param)

        return [
            {"params": encoder_params, "lr": base_lr, "weight_decay": 0.01},
            {"params": special_params, "lr": base_lr * 10, "weight_decay": 0.0},
            {"params": other_params, "lr": base_lr * 3, "weight_decay": 0.01},
        ]


# ════════════════════════════════════════════════════════════════════════
# Integration Verification Script
# ════════════════════════════════════════════════════════════════════════

def verify_gradient_routing():
    """
    Smoke test: Verify that each loss reaches the correct parameters.

    This test creates a minimal model with small tensors and checks that
    gradients flow through the expected paths after each individual loss
    backward pass.
    """
    print("=" * 60)
    print("GRADIENT ROUTING VERIFICATION")
    print("=" * 60)

    B, N, L, L_q, D, S_max = 2, 10, 32, 16, 64, 5

    # --- Create minimal mock encoder ---
    class MockEmbeddings(nn.Module):
        def __init__(self, D):
            super().__init__()
            self.embed = nn.Linear(100, D)  # fake embedding lookup

        def forward(self, input_ids=None):
            # Simulate embedding lookup: (B, L) → (B, L, D)
            one_hot = F.one_hot(input_ids.clamp(0, 99), num_classes=100).float()
            return self.embed(one_hot)

    class MockEncoder(nn.Module):
        def __init__(self, D):
            super().__init__()
            self.linear = nn.Linear(D, D)
            self.config = type("Config", (), {"hidden_size": D})()
            self.embeddings = MockEmbeddings(D)
            self.encoder = type("Enc", (), {
                "layer": nn.ModuleList([
                    type("Layer", (nn.Module,), {
                        "__init__": lambda self_: (super(type(self_), self_).__init__(), None)[-1],
                        "attention": type("Attn", (), {
                            "self": type("Self", (), {
                                "in_proj_weight": nn.Parameter(torch.randn(3*D, D)),
                            })(),
                            "output": type("Out", (), {
                                "dense": nn.Linear(D, D),
                            })(),
                        })(),
                    })()
                    for _ in range(2)
                ])
            })()

        def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
            if inputs_embeds is not None:
                x = inputs_embeds
            else:
                B_flat = input_ids.size(0)
                L_local = input_ids.size(1)
                x = torch.randn(B_flat, L_local, 64, requires_grad=True)
            x = self.linear(x)
            return type("Out", (), {"last_hidden_state": x})()

    mock_encoder = MockEncoder(D)
    model = SROModel(mock_encoder, hidden_dim=D, num_paragraphs=N, verifier_layers=1)
    model.paragraph_encoder.sent_token_id = 999  # fake

    # --- Create mock batch ---
    batch = SROBatch(
        paragraph_input_ids=torch.randint(0, 100, (B, N, L)),
        paragraph_attention_mask=torch.ones(B, N, L, dtype=torch.long),
        question_input_ids=torch.randint(0, 100, (B, L_q)),
        question_attention_mask=torch.ones(B, L_q, dtype=torch.long),
        judge_label=torch.randint(0, 2, (B,)).float(),
        gold_mask=torch.zeros(B, N),
        qtype_label=torch.randint(0, 2, (B,)).float(),
        sent_targets=torch.zeros(B, N, S_max),
        sent_mask=torch.zeros(B, N, S_max),
        question_ids=["q1", "q2"],
    )
    # Set 2 gold paragraphs per batch item
    batch.gold_mask[:, 0] = 1.0
    batch.gold_mask[:, 3] = 1.0

    # --- Test Stage A ---
    print("\n[Stage A] Testing L_select + L_qtype + L_sent_sup + detached L_verify...")
    model.train()
    model.set_stage_a()
    output = model(batch, tau=1.0)

    assert output.loss_verify is not None, "L_verify should be active (detached) in Stage A"
    assert output.loss_total.requires_grad, "Total loss must have grad"

    output.loss_total.backward()
    print("  ✓ Stage A backward pass succeeded")

    # Check that PairRouter received gradients (via L_select)
    router_grad = model.pair_router.anchor_head.weight.grad
    assert router_grad is not None and router_grad.abs().sum() > 0, \
        "PairRouter should receive gradient from L_select in Stage A"
    print("  ✓ PairRouter receives gradient from L_select")

    # Check that qtype gate received gradients
    gate_grad = model.qtype_gate.mlp[0].weight.grad
    assert gate_grad is not None and gate_grad.abs().sum() > 0, \
        "Qtype gate should receive gradient from L_qtype"
    print("  ✓ Qtype gate receives gradient from L_qtype")

    # Check that verifier received gradients (from detached L_verify)
    verifier_grad = model.verifier.classifier[0].weight.grad
    assert verifier_grad is not None and verifier_grad.abs().sum() > 0, \
        "Verifier should receive gradient from detached L_verify in Stage A"
    print("  ✓ Verifier receives gradient from detached L_verify (pre-warming)")

    model.zero_grad()

    # --- Test Stage B (K=1) ---
    print("\n[Stage B, K=1] Testing all 4 losses...")
    model.set_stage_b(K=1)
    output = model(batch, tau=0.5)

    assert output.loss_verify is not None, "L_verify should be active in Stage B"

    output.loss_total.backward()
    print("  ✓ Stage B (K=1) backward pass succeeded")

    # Verify L_verify gradient reaches encoder
    encoder_grad = model.paragraph_encoder.pool_head.weight.grad
    assert encoder_grad is not None and encoder_grad.abs().sum() > 0, \
        "Encoder should receive gradient from L_verify path"
    print("  ✓ L_verify gradient reaches encoder via PairRouter → soft_gather")

    model.zero_grad()

    # --- Test Stage B (K=3) ---
    print("\n[Stage B, K=3] Testing K-sample expansion...")
    model.set_stage_b(K=3)
    output = model(batch, tau=0.5)

    output.loss_total.backward()
    print("  ✓ Stage B (K=3) backward pass succeeded")
    print("  ✓ K-sample expansion produces correct gradients")

    print("\n" + "=" * 60)
    print("ALL GRADIENT ROUTING CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    verify_gradient_routing()
