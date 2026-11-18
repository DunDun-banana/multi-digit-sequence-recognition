"""Autoregressive LSTM Decoder with Bahdanau Attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.attention import BahdanauAttention


class LSTMDecoder(nn.Module):
    """Step-by-step LSTM decoder attending over CNN spatial feature maps."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        encoder_dim: int,
        decoder_hidden_dim: int,
        attention_dim: int,
        pad_id: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(
            encoder_dim=encoder_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            attention_dim=attention_dim,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim + encoder_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.init_h = nn.Linear(encoder_dim, decoder_hidden_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc_out = nn.Linear(decoder_hidden_dim + encoder_dim, vocab_size)

    def init_hidden(self, encoder_outputs: Tensor) -> tuple[Tensor, Tensor]:
        """Initialize LSTM hidden and cell states from mean-pooled encoder features."""
        mean_enc = encoder_outputs.mean(dim=1)  # [B, E]
        h0 = torch.tanh(self.init_h(mean_enc))  # [B, H]
        c0 = torch.tanh(self.init_c(mean_enc))  # [B, H]
        h = h0.unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
        c = c0.unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
        return h, c

    def forward_step(
        self,
        input_token: Tensor,
        hidden: Tensor,
        cell: Tensor,
        encoder_outputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run a single decoding step."""
        embedded = self.dropout(self.embedding(input_token))  # [B, embed_dim]
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)  # [B, E], [B, S]

        lstm_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)  # [B, 1, embed + E]
        lstm_output, (next_hidden, next_cell) = self.lstm(lstm_input, (hidden, cell))
        lstm_output = lstm_output.squeeze(1)  # [B, H]

        combined = self.dropout(torch.cat([lstm_output, context], dim=-1))  # [B, H + E]
        logits = self.fc_out(combined)  # [B, vocab_size]
        return logits, next_hidden, next_cell, attn_weights
