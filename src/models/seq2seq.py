"""Full Attention-based CNN Encoder - LSTM Decoder sequence-to-sequence model."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

# Allow direct execution: python src/models/seq2seq.py
_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch
from torch import Tensor, nn

from data.labels import EOS_ID, PAD_ID, SOS_ID, DigitVocabulary
from models.decoder import LSTMDecoder
from models.encoder import CNNEncoder


class CNNSeq2Seq(nn.Module):
    """Image-to-sequence model combining a CNN Encoder, Bahdanau Attention, and LSTM Decoder."""

    def __init__(
        self,
        vocab_size: int = 13,
        encoder_dim: int = 256,
        embed_dim: int = 64,
        decoder_hidden_dim: int = 256,
        attention_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_id: int = PAD_ID,
        sos_id: int = SOS_ID,
        eos_id: int = EOS_ID,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.encoder = CNNEncoder(in_channels=1, encoder_dim=encoder_dim, dropout=dropout)
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            encoder_dim=encoder_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            attention_dim=attention_dim,
            pad_id=pad_id,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        images: Tensor,
        decoder_inputs: Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tensor:
        """Forward pass for training or teacher-forced validation."""
        _, seq_len = decoder_inputs.shape
        encoder_outputs = self.encoder(images)  # [B, S, E]
        hidden, cell = self.decoder.init_hidden(encoder_outputs)

        logits_steps: list[Tensor] = []
        input_token = decoder_inputs[:, 0]

        for t in range(seq_len):
            step_logits, hidden, cell, _ = self.decoder.forward_step(
                input_token=input_token,
                hidden=hidden,
                cell=cell,
                encoder_outputs=encoder_outputs,
            )
            logits_steps.append(step_logits)

            if t + 1 < seq_len:
                use_teacher = self.training and (random.random() < teacher_forcing_ratio)
                if use_teacher:
                    input_token = decoder_inputs[:, t + 1]
                else:
                    input_token = step_logits.argmax(dim=-1)

        return torch.stack(logits_steps, dim=1)  # [B, T, vocab_size]

    @torch.inference_mode()
    def predict(
        self,
        images: Tensor,
        max_len: int = 6,
    ) -> tuple[Tensor, Tensor]:
        """Autoregressive greedy decoding until EOS or max_len."""
        was_training = self.training
        self.eval()

        batch_size = images.size(0)
        device = images.device

        encoder_outputs = self.encoder(images)
        hidden, cell = self.decoder.init_hidden(encoder_outputs)

        input_token = torch.full((batch_size,), fill_value=self.sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        pred_steps: list[Tensor] = []
        attn_steps: list[Tensor] = []

        for _ in range(max_len):
            step_logits, hidden, cell, attn_weights = self.decoder.forward_step(
                input_token=input_token,
                hidden=hidden,
                cell=cell,
                encoder_outputs=encoder_outputs,
            )
            next_token = step_logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, self.pad_id), next_token)
            pred_steps.append(next_token)
            attn_steps.append(attn_weights)

            finished = finished | (next_token == self.eos_id)
            input_token = next_token

            if bool(finished.all()):
                break

        if was_training:
            self.train()

        predicted_ids = torch.stack(pred_steps, dim=1)
        attention_weights = torch.stack(attn_steps, dim=1)
        return predicted_ids, attention_weights


def build_model_from_config(config: dict[str, Any], vocab: DigitVocabulary) -> CNNSeq2Seq:
    """Instantiate CNNSeq2Seq using model hyperparameters from YAML config."""
    model_cfg = config.get("model", {})
    return CNNSeq2Seq(
        vocab_size=vocab.vocab_size,
        encoder_dim=int(model_cfg.get("encoder_dim", 256)),
        embed_dim=int(model_cfg.get("embed_dim", 64)),
        decoder_hidden_dim=int(model_cfg.get("decoder_hidden_dim", 256)),
        attention_dim=int(model_cfg.get("attention_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pad_id=vocab.pad_id,
        sos_id=vocab.sos_id,
        eos_id=vocab.eos_id,
    )


if __name__ == "__main__":
    dummy_images = torch.randn(4, 1, 32, 160)
    dummy_inputs = torch.full((4, 6), fill_value=SOS_ID, dtype=torch.long)
    model = CNNSeq2Seq()
    enc_out = model.encoder(dummy_images)
    logits = model(dummy_images, dummy_inputs)
    preds, attn = model.predict(dummy_images, max_len=6)
    params = sum(p.numel() for p in model.parameters())
    print(f"Input shape   : {tuple(dummy_images.shape)}")
    print(f"Encoder output: {tuple(enc_out.shape)}")
    print(f"Train logits  : {tuple(logits.shape)}")
    print(f"Predict IDs   : {tuple(preds.shape)}")
    print(f"Attention map : {tuple(attn.shape)}")
    print(f"Total params  : {params:,}")
