#!/usr/bin/env python3
"""
sfav_model.py — Supporting-Fact-Aware Verifier (SFAV).

Extends Architecture A (CrossHopVerifier, per-hop separate) with a second
token-level head that predicts, for each evidence token, whether it belongs
to a gold supporting sentence.

Architecture:
  Encoder  : DeBERTa-v3-base (shared weights, two forward passes)
  Head 1   : Verifier — concat[Z1, Z2, |Z1-Z2|, min(Z1,Z2)] → MLP → score
  Head 2   : SupportingFact — linear(768→1) per token → sigmoid

Loss (training only):
  L_total = L_main + λ · L_sup
  L_main  : BCE with label smoothing 0.05 on EM label (verifier head)
  L_sup   : class-weighted BCE on token-level supporting-fact labels

At inference, only Head 1's score is used for candidate selection.
Head 2's only function is to constrain the encoder's representations during
training, preventing the hop1 ≈ hop2 collapse observed under EM-only training.

Why this prevents collapse
--------------------------
Under EM-only training, the encoder can reduce L_main by mapping hop1 and
hop2 to the same representation — the verifier head works equally well with
Z1 ≈ Z2. SFAV penalises this: the supporting-fact head for hop1 must identify
*hop1-specific* supporting sentences, and likewise for hop2. Collapsing
Z1 ≈ Z2 forces at least one hop's head to predict the wrong sentences, raising
L_sup. The auxiliary loss makes per-hop differentiation rewarding.

Usage:
  from sfav_model import SFAVVerifier

  model = SFAVVerifier(encoder_name="microsoft/deberta-v3-base", lam=0.3)

  # Training step
  logits, l_sup = model(
      h1_ids, h1_mask, h2_ids, h2_mask,
      h1_sup_labels=h1_labels,   # (B, max_len), float, -1 = ignore
      h2_sup_labels=h2_labels,
  )
  # l_sup is already λ-weighted; loss = main_loss + l_sup

  # Inference step
  logits = model(h1_ids, h1_mask, h2_ids, h2_mask)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: SUPPORTING-FACT HEAD
# ═══════════════════════════════════════════════════════════════════════

class SupportingFactHead(nn.Module):
    """Per-token binary classifier: does this token belong to a supporting sentence?

    Input : sequence output from DeBERTa, shape (B, seq_len, hidden_dim)
    Output: token logits,                  shape (B, seq_len)

    Loss  : class-weighted BCE.
    Positive-class weight is set at construction to handle class imbalance
    (most tokens are not supporting), typically ~5-10× depending on passage length.

    The evidence_mask parameter confines the loss to evidence tokens only —
    [CLS], question tokens, answer tokens, and [SEP] are excluded from the
    supporting-fact supervision signal.
    """

    def __init__(self, hidden_dim: int = 768, pos_weight: float = 8.0):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)
        # pos_weight balances the class imbalance: in a ~100-token evidence segment,
        # typically ~10-20 tokens are from supporting sentences → ratio ~5-10×.
        # Default of 8.0 is a reasonable starting point; ablated in Week 4.
        self.register_buffer(
            "pos_weight", torch.tensor([pos_weight], dtype=torch.float32)
        )

    def forward(
        self,
        sequence_output: torch.Tensor,          # (B, seq_len, D)
        sup_labels: Optional[torch.Tensor],     # (B, seq_len), float, -1 = ignore
        evidence_mask: Optional[torch.Tensor],  # (B, seq_len), 1 = evidence token
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, loss).

        logits : (B, seq_len) — token-level supporting-fact scores
        loss   : scalar — class-weighted BCE, or 0.0 at inference
        """
        logits = self.linear(sequence_output).squeeze(-1)  # (B, seq_len)

        if sup_labels is None:
            return logits, logits.new_tensor(0.0)

        # Mask: only compute loss on evidence tokens with valid labels
        valid_mask = (sup_labels >= 0)
        if evidence_mask is not None:
            valid_mask = valid_mask & evidence_mask.bool()

        n_valid = valid_mask.sum()
        if n_valid == 0:
            return logits, logits.new_tensor(0.0)

        flat_logits = logits[valid_mask]          # (N,)
        flat_labels = sup_labels[valid_mask]       # (N,)

        loss = F.binary_cross_entropy_with_logits(
            flat_logits,
            flat_labels,
            pos_weight=self.pos_weight.to(flat_logits.device),
        )
        return logits, loss


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: VERIFIER HEAD (identical to Architecture A)
# ═══════════════════════════════════════════════════════════════════════

class VerifierHead(nn.Module):
    """Two-layer MLP on concat[Z1, Z2, |Z1-Z2|, min(Z1,Z2)].

    Identical to the head in crosshop_model.py — kept here so sfav_model.py
    is self-contained without needing to import crosshop_model.py.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """z1, z2: (B, D) → logits (B,)."""
        diff = torch.abs(z1 - z2)
        mn   = torch.minimum(z1, z2)
        feat = torch.cat([z1, z2, diff, mn], dim=-1)   # (B, 4D)
        return self.net(feat).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: SFAV VERIFIER
# ═══════════════════════════════════════════════════════════════════════

class SFAVVerifier(nn.Module):
    """Supporting-Fact-Aware Verifier.

    Encoder + two heads:
      Head 1 (verifier)       — Architecture A's CLS-based MLP
      Head 2 (supporting-fact)— token-level BCE classifier

    λ (lam) weights L_sup relative to L_main.
    """

    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-base",
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        lam: float = 0.3,
        pos_weight: float = 8.0,
        encoder_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.lam = lam
        enc_kwargs = encoder_kwargs or {}
        self.encoder = AutoModel.from_pretrained(encoder_name, **enc_kwargs)
        hidden_dim = self.encoder.config.hidden_size  # 768 for base

        self.verifier_head = VerifierHead(
            hidden_dim=hidden_dim,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
        )
        self.sup_head = SupportingFactHead(
            hidden_dim=hidden_dim,
            pos_weight=pos_weight,
        )

    # ── Internal: one hop pass ──────────────────────────────────────────

    def _encode_hop(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode one hop.

        Returns:
          cls_vec  : (B, D) — [CLS] token representation
          seq_out  : (B, seq_len, D) — full token representations
        """
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        seq_out = out.last_hidden_state          # (B, seq_len, D)
        cls_vec = seq_out[:, 0, :]               # (B, D)
        return cls_vec, seq_out

    # ── Forward ────────────────────────────────────────────────────────

    def forward(
        self,
        h1_input_ids: torch.Tensor,
        h1_attention_mask: torch.Tensor,
        h2_input_ids: torch.Tensor,
        h2_attention_mask: torch.Tensor,
        # Supporting-fact supervision (training only — pass None at inference)
        h1_sup_labels: Optional[torch.Tensor] = None,   # (B, seq_len)
        h2_sup_labels: Optional[torch.Tensor] = None,   # (B, seq_len)
        h1_evidence_mask: Optional[torch.Tensor] = None,  # (B, seq_len)
        h2_evidence_mask: Optional[torch.Tensor] = None,  # (B, seq_len)
        # Diagnostic: return pre/post interaction representations
        return_representations: bool = False,
    ):
        """
        Returns:
          Training (sup_labels provided):
            (logits, l_sup_weighted)
            logits         : (B,) — verifier logits (not sigmoid'd)
            l_sup_weighted : scalar — λ * (L_sup_h1 + L_sup_h2) / 2
          Inference (no sup_labels):
            logits : (B,)
          With return_representations=True (diagnostic):
            Adds (z1_pre, z2_pre, z1_post, z2_post) — all (B, D)
            [pre/post are the same for Architecture A; z_pre = z_post = cls_vec]
        """
        z1, seq1 = self._encode_hop(h1_input_ids, h1_attention_mask)
        z2, seq2 = self._encode_hop(h2_input_ids, h2_attention_mask)

        # Verifier head (Architecture A — no cross-hop interaction)
        logits = self.verifier_head(z1, z2)

        # Supporting-fact head
        training_mode = (h1_sup_labels is not None)
        if training_mode:
            _, l_sup_h1 = self.sup_head(seq1, h1_sup_labels, h1_evidence_mask)
            _, l_sup_h2 = self.sup_head(seq2, h2_sup_labels, h2_evidence_mask)
            l_sup_mean = (l_sup_h1 + l_sup_h2) / 2.0
            l_sup_weighted = self.lam * l_sup_mean

            if return_representations:
                return logits, l_sup_weighted, (z1, z2, z1, z2)
            return logits, l_sup_weighted

        if return_representations:
            return logits, (z1, z2, z1, z2)
        return logits

    # ── Parameter groups for optimiser ─────────────────────────────────

    def parameter_groups(
        self,
        encoder_lr: float = 2e-5,
        head_lr: float = 1e-4,
    ) -> list:
        """Return two param groups: encoder (slow LR) and heads (fast LR)."""
        enc_ids = {id(p) for p in self.encoder.parameters()}
        enc_params  = list(self.encoder.parameters())
        head_params = [p for p in self.parameters() if id(p) not in enc_ids]
        return [
            {"params": enc_params,  "lr": encoder_lr},
            {"params": head_params, "lr": head_lr},
        ]


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: SANITY CHECK
# ═══════════════════════════════════════════════════════════════════════

def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("SFAV model sanity check ...")
    B, L, D = 2, 64, 768

    # Mock encoder to avoid downloading weights
    class _MockEncoder(nn.Module):
        class _Cfg:
            hidden_size = D
        config = _Cfg()
        def forward(self, input_ids, attention_mask):
            B_, S = input_ids.shape
            seq = torch.randn(B_, S, D)
            class _Out:
                last_hidden_state = seq
            return _Out()

    model = SFAVVerifier.__new__(SFAVVerifier)
    nn.Module.__init__(model)
    model.lam = 0.3
    model.encoder = _MockEncoder()
    model.verifier_head = VerifierHead(hidden_dim=D)
    model.sup_head = SupportingFactHead(hidden_dim=D)

    ids   = torch.randint(0, 100, (B, L))
    mask  = torch.ones(B, L, dtype=torch.long)
    # sup_labels: 0/1 for evidence tokens, -1 for question/answer tokens
    slabs = torch.full((B, L), -1.0)
    slabs[:, 30:] = torch.randint(0, 2, (B, L - 30)).float()
    ev_mask = torch.zeros(B, L, dtype=torch.long)
    ev_mask[:, 30:] = 1

    # Training forward
    logits, l_sup = model(ids, mask, ids, mask,
                          h1_sup_labels=slabs, h2_sup_labels=slabs,
                          h1_evidence_mask=ev_mask, h2_evidence_mask=ev_mask)
    print(f"  logits shape : {logits.shape}   (expect ({B},))")
    print(f"  l_sup        : {l_sup.item():.4f}")

    # Inference forward
    logits_inf = model(ids, mask, ids, mask)
    print(f"  inference logits shape: {logits_inf.shape}")

    # With representations
    logits_r, (z1, z2, z1p, z2p) = model(ids, mask, ids, mask,
                                          return_representations=True)
    print(f"  z1 shape: {z1.shape}  (expect ({B}, {D}))")

    n_ver = _count_params(model.verifier_head)
    n_sup = _count_params(model.sup_head)
    n_enc = _count_params(model.encoder)
    print(f"\n  Encoder params    : {n_enc:,}  (mock, would be 184M for DeBERTa-base)")
    print(f"  Verifier head     : {n_ver:,}")
    print(f"  Sup-fact head     : {n_sup:,}")
    print(f"  λ                 : {model.lam}")
    print("\nSanity check PASSED.")