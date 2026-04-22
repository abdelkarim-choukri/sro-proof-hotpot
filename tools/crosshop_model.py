#!/usr/bin/env python3
"""
crosshop_model.py — Model definitions for the cross-hop attention experiment.

Three architectures, all sharing a DeBERTa-v3-base encoder with shared weights
across two forward passes (one per hop). They differ only in what happens
AFTER the encoder produces z_1 and z_2.

  Arch A — Per-Hop Separate (baseline)
    No interaction between z_1 and z_2.
    Combine via concat[z_1, z_2, |z_1-z_2|, min(z_1, z_2)] → MLP → score

  Arch B — Lightweight Cross-Hop
    1-head 64-dim cross-attention, no FFN, residual:
      z_1' = z_1 + Attn(Q=z_1, K=z_2, V=z_2)
      z_2' = z_2 + Attn(Q=z_2, K=z_1, V=z_1)
    Combine via the same concat → same MLP → score

  Arch C — Standard Cross-Hop (transformer block)
    4-head 128-dim cross-attention + FFN (standard transformer block)
    Pre-LayerNorm:
      z_1' = z_1 + Attn_MH(LN(z_1), LN(z_2), LN(z_2))
      z_1'' = z_1' + FFN(LN(z_1'))
      symmetric for z_2
    Combine via the same concat → same MLP → score

The final MLP and input format are IDENTICAL across architectures so that the
only architectural variable is the cross-hop interaction.

Input format (per hop pass):
  [CLS] Question: {q} [SEP] Answer: {a} [SEP] Evidence: {h_k} [SEP]
  Tokenized with DeBERTa-v3-base tokenizer, max_length=512.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: CROSS-HOP INTERACTION MODULES
# ═══════════════════════════════════════════════════════════════════════

class NoInteraction(nn.Module):
    """Arch A: pass z_1, z_2 through unchanged."""

    def __init__(self):
        super().__init__()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        # z1, z2: (B, D)
        return z1, z2


class LightweightCrossAttn(nn.Module):
    """Arch B: 1-head 64-dim cross-attention, no FFN, residual connection.

    Each hop queries the other hop. Residual ensures the model can learn
    the identity (i.e., recover Arch A behavior) if the attention isn't helpful.
    """

    def __init__(self, hidden_dim: int = 768, attn_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        # Single head — one projection each for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.o_proj = nn.Linear(attn_dim, hidden_dim, bias=False)

        # Init so that initial attention output is near-zero
        # (helps the residual path start as near-identity)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.o_proj.weight)

    def _attend(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        # q_in, kv_in: (B, D) — single "token" each
        q = self.q_proj(q_in).unsqueeze(1)   # (B, 1, attn_dim)
        k = self.k_proj(kv_in).unsqueeze(1)  # (B, 1, attn_dim)
        v = self.v_proj(kv_in).unsqueeze(1)  # (B, 1, attn_dim)
        scale = 1.0 / math.sqrt(self.attn_dim)
        # Attention weights: (B, 1, 1) — trivially softmax to 1 for a single KV
        # but we keep the math explicit so extension to multi-token is clean.
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).squeeze(1)  # (B, attn_dim)
        return self.o_proj(out)                 # (B, D)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        z1_out = z1 + self._attend(z1, z2)
        z2_out = z2 + self._attend(z2, z1)
        return z1_out, z2_out


class StandardCrossAttnBlock(nn.Module):
    """Arch C: standard transformer cross-attention block.

    4 heads, 128-dim per head (total 512) — but projected back to hidden_dim.
    Includes FFN (4× hidden expansion) and pre-LayerNorm. O-projection
    zero-init for residual stability at start of training.
    """

    def __init__(self, hidden_dim: int = 768, n_heads: int = 4,
                 head_dim: int = 128, ffn_mult: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim   # 512

        # Cross-attention
        self.ln_q = nn.LayerNorm(hidden_dim)
        self.ln_kv = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, inner, bias=False)
        self.k_proj = nn.Linear(hidden_dim, inner, bias=False)
        self.v_proj = nn.Linear(hidden_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, hidden_dim, bias=False)

        # FFN
        self.ln_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_mult * hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_mult * hidden_dim, hidden_dim),
        )

        # Init — attention output zero so residual starts as identity
        for m in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.o_proj.weight)
        # FFN output zero-init too
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def _cross_attend(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        # q_in, kv_in: (B, D). Single-token cross attention (B, 1, D) internally.
        B = q_in.size(0)
        q = self.q_proj(q_in).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        # shapes: (B, n_heads, 1, head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, H, 1, 1)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                               # (B, H, 1, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, self.n_heads * self.head_dim)
        return self.o_proj(out)                                   # (B, D)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        # Cross-attend (pre-LN)
        z1_attn = self._cross_attend(self.ln_q(z1), self.ln_kv(z2))
        z2_attn = self._cross_attend(self.ln_q(z2), self.ln_kv(z1))
        z1a = z1 + z1_attn
        z2a = z2 + z2_attn
        # FFN (pre-LN)
        z1b = z1a + self.ffn(self.ln_ffn(z1a))
        z2b = z2a + self.ffn(self.ln_ffn(z2a))
        return z1b, z2b


def make_interaction(arch: str) -> nn.Module:
    if arch == "A":
        return NoInteraction()
    elif arch == "B":
        return LightweightCrossAttn(hidden_dim=768, attn_dim=64)
    elif arch == "C":
        return StandardCrossAttnBlock(hidden_dim=768, n_heads=4, head_dim=128)
    else:
        raise ValueError(f"Unknown arch: {arch}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: VERIFIER SCORING HEAD
# ═══════════════════════════════════════════════════════════════════════

class VerifierHead(nn.Module):
    """2-layer MLP that maps [z_1; z_2; |z_1-z_2|; min(z_1,z_2)] → scalar score.

    Identical across all three architectures. Only the INPUTS (z_1, z_2 or
    z_1', z_2') differ depending on the interaction layer upstream.
    """

    def __init__(self, hidden_dim: int = 768, mlp_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        in_dim = hidden_dim * 4  # z1 | z2 | |z1-z2| | min
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # z1, z2: (B, D). Return logits (B,).
        diff = torch.abs(z1 - z2)
        mn = torch.minimum(z1, z2)
        feat = torch.cat([z1, z2, diff, mn], dim=-1)  # (B, 4D)
        return self.net(feat).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: FULL VERIFIER MODEL
# ═══════════════════════════════════════════════════════════════════════

class CrossHopVerifier(nn.Module):
    """Complete verifier: encoder + interaction + head.

    Two forward passes per candidate, one per hop. Shared encoder weights.
    The interaction module is what varies across A/B/C.
    """

    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-base",
        arch: str = "A",
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        encoder_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.arch = arch
        enc_kwargs = encoder_kwargs or {}
        self.encoder = AutoModel.from_pretrained(encoder_name, **enc_kwargs)
        hidden_dim = self.encoder.config.hidden_size  # 768 for base

        self.interaction = make_interaction(arch)
        self.head = VerifierHead(hidden_dim=hidden_dim,
                                  mlp_hidden=mlp_hidden,
                                  dropout=dropout)

    def encode_hop(self, input_ids: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        """Run one hop through the encoder. Returns [CLS] vector (B, D)."""
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask)
        # DeBERTa doesn't return pooler_output by default; take CLS.
        return out.last_hidden_state[:, 0, :]

    def forward(
        self,
        h1_input_ids: torch.Tensor,
        h1_attention_mask: torch.Tensor,
        h2_input_ids: torch.Tensor,
        h2_attention_mask: torch.Tensor,
        return_representations: bool = False,
    ):
        """Full forward: two hop passes → interaction → head.

        If return_representations=True, also returns (z1_pre, z2_pre, z1_post, z2_post)
        for diagnostic purposes (CKA, Pearson, etc.).
        """
        z1_pre = self.encode_hop(h1_input_ids, h1_attention_mask)
        z2_pre = self.encode_hop(h2_input_ids, h2_attention_mask)

        z1_post, z2_post = self.interaction(z1_pre, z2_pre)

        logits = self.head(z1_post, z2_post)

        if return_representations:
            return logits, (z1_pre, z2_pre, z1_post, z2_post)
        return logits


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: QUICK SANITY
# ═══════════════════════════════════════════════════════════════════════

def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity that all three architectures instantiate and produce shapes.
    print("Instantiating models (this downloads DeBERTa-v3-base if not cached)...")
    for arch in ["A", "B", "C"]:
        m = CrossHopVerifier(arch=arch)
        B, L = 2, 32
        h1_ids = torch.randint(0, 1000, (B, L))
        h1_mask = torch.ones(B, L, dtype=torch.long)
        h2_ids = torch.randint(0, 1000, (B, L))
        h2_mask = torch.ones(B, L, dtype=torch.long)
        logits = m(h1_ids, h1_mask, h2_ids, h2_mask)
        print(f"  Arch {arch}:  params={_count_params(m):>10,}   "
              f"logits.shape={tuple(logits.shape)}")