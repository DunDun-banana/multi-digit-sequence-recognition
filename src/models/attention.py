"""Additive (Bahdanau-style) attention mechanism."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BahdanauAttention(nn.Module):
    """Computes attention weights and context vector over encoder spatial features."""

    def __init__(self, encoder_dim: int, decoder_hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        self.key_proj = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.energy = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute attention context vector and normalized weights."""
        query = self.query_proj(decoder_hidden).unsqueeze(1)  # [B, 1, A]
        keys = self.key_proj(encoder_outputs)  # [B, S, A]
        scores = self.energy(torch.tanh(query + keys)).squeeze(-1)  # [B, S]
        attn_weights = torch.softmax(scores, dim=-1)  # [B, S]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, E]
        return context, attn_weights
