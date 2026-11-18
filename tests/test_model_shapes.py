"""Unit tests for CNN Encoder, Bahdanau Attention, LSTM Decoder, Seq2Seq, and end-to-end smoke run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn

from data.dataset import DigitSequenceDataset, create_dataloader
from data.labels import EOS_ID, PAD_ID, SOS_ID, DigitVocabulary
from evaluate import evaluate_model
from models import BahdanauAttention, CNNEncoder, CNNSeq2Seq, LSTMDecoder
from predict import Predictor
from train import Trainer


def test_encoder_attention_decoder_shapes() -> None:
    batch_size = 4
    images = torch.randn(batch_size, 1, 32, 160)

    encoder = CNNEncoder(in_channels=1, encoder_dim=128)
    enc_out = encoder(images)
    assert enc_out.ndim == 3
    assert enc_out.shape[0] == batch_size
    assert enc_out.shape[2] == 128
    seq_spatial = enc_out.shape[1]

    attention = BahdanauAttention(encoder_dim=128, decoder_hidden_dim=96, attention_dim=64)
    dec_hidden = torch.randn(batch_size, 96)
    context, weights = attention(dec_hidden, enc_out)
    assert context.shape == (batch_size, 128)
    assert weights.shape == (batch_size, seq_spatial)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    decoder = LSTMDecoder(
        vocab_size=13,
        embed_dim=32,
        encoder_dim=128,
        decoder_hidden_dim=96,
        attention_dim=64,
        pad_id=PAD_ID,
    )
    h, c = decoder.init_hidden(enc_out)
    assert h.shape == (1, batch_size, 96)
    assert c.shape == (1, batch_size, 96)

    step_tokens = torch.full((batch_size,), SOS_ID, dtype=torch.long)
    logits, next_h, next_c, step_attn = decoder.forward_step(step_tokens, h, c, enc_out)
    assert logits.shape == (batch_size, 13)
    assert next_h.shape == (1, batch_size, 96)
    assert next_c.shape == (1, batch_size, 96)
    assert step_attn.shape == (batch_size, seq_spatial)


def test_seq2seq_forward_loss_backward_and_predict() -> None:
    model = CNNSeq2Seq(
        vocab_size=13,
        encoder_dim=64,
        embed_dim=32,
        decoder_hidden_dim=64,
        attention_dim=32,
        pad_id=PAD_ID,
        sos_id=SOS_ID,
        eos_id=EOS_ID,
    )
    images = torch.randn(3, 1, 32, 160)
    dec_inputs = torch.tensor(
        [
            [SOS_ID, 1, 2, EOS_ID, PAD_ID],
            [SOS_ID, 3, 4, 5, EOS_ID],
            [SOS_ID, 9, EOS_ID, PAD_ID, PAD_ID],
        ],
        dtype=torch.long,
    )
    dec_targets = torch.tensor(
        [
            [1, 2, EOS_ID, PAD_ID, PAD_ID],
            [3, 4, 5, EOS_ID, PAD_ID],
            [9, EOS_ID, PAD_ID, PAD_ID, PAD_ID],
        ],
        dtype=torch.long,
    )

    logits = model(images, dec_inputs, teacher_forcing_ratio=0.5)
    assert logits.shape == (3, 5, 13)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    loss = criterion(logits.reshape(-1, 13), dec_targets.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()

    enc_grad = next(model.encoder.parameters()).grad
    dec_grad = next(model.decoder.parameters()).grad
    assert enc_grad is not None and enc_grad.abs().sum() > 0
    assert dec_grad is not None and dec_grad.abs().sum() > 0

    pred_ids, attn = model.predict(images, max_len=6)
    assert pred_ids.shape[0] == 3
    assert 1 <= pred_ids.shape[1] <= 6
    assert attn.shape[:2] == pred_ids.shape


def test_end_to_end_synthetic_training_eval_and_predict_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    rows = []
    labels = ["12", "345", "07", "8901", "5", "678", "24680", "91"]
    for idx, label in enumerate(labels):
        img = Image.new("L", (32 * len(label), 32), color=0)
        ImageDraw.Draw(img).text((4, 8), label, fill=255)
        img_path = data_dir / f"{idx:03d}_{label}.png"
        img.save(img_path)
        rows.append(
            {
                "image_path": img_path.relative_to(tmp_path).as_posix(),
                "label": label,
                "sequence_length": len(label),
            }
        )

    df = pd.DataFrame(rows)
    vocab = DigitVocabulary()
    train_ds = DigitSequenceDataset(df.iloc[:4], vocab=vocab, base_dir=tmp_path)
    val_ds = DigitSequenceDataset(df.iloc[4:6], vocab=vocab, base_dir=tmp_path)
    test_ds = DigitSequenceDataset(df.iloc[6:], vocab=vocab, base_dir=tmp_path)

    train_loader = create_dataloader(train_ds, batch_size=2, shuffle=True, pad_id=vocab.pad_id)
    val_loader = create_dataloader(val_ds, batch_size=2, shuffle=False, pad_id=vocab.pad_id)
    test_loader = create_dataloader(test_ds, batch_size=2, shuffle=False, pad_id=vocab.pad_id)

    config = {
        "project": {"name": "smoke", "seed": 42, "device": "cpu"},
        "data": {"max_sequence_length": 5},
        "image": {"height": 32, "width": 160, "pad_mode": "right", "pad_value": 0, "normalize_mean": 0.0, "normalize_std": 1.0},
        "model": {"encoder_dim": 32, "embed_dim": 16, "decoder_hidden_dim": 32, "attention_dim": 16, "num_layers": 1, "dropout": 0.0},
        "training": {
            "batch_size": 2,
            "num_workers": 0,
            "epochs": 2,
            "learning_rate": 0.005,
            "weight_decay": 0.0,
            "teacher_forcing_ratio": 0.5,
            "grad_clip_norm": 5.0,
            "early_stopping_patience": 5,
            "checkpoint_dir": "checkpoints",
            "best_checkpoint_name": "best_model.pt",
            "final_checkpoint_name": "final_model.pt",
            "output_dir": "outputs",
        },
        "inference": {"max_decode_length": 6},
    }

    trainer = Trainer(config=config, vocab=vocab, base_dir=tmp_path)
    history_df = trainer.fit(train_loader, val_loader, epochs=2, show_progress=False)
    assert len(history_df) == 2

    best_ckpt = tmp_path / "checkpoints" / "best_model.pt"
    final_ckpt = tmp_path / "checkpoints" / "final_model.pt"
    assert best_ckpt.exists()
    assert final_ckpt.exists()
    assert (tmp_path / "outputs" / "epoch_metrics.csv").exists()
    assert (tmp_path / "outputs" / "epoch_metrics.json").exists()

    trainer2 = Trainer(config=config, vocab=vocab, base_dir=tmp_path)
    history_resumed = trainer2.fit(train_loader, val_loader, epochs=3, resume_from=final_ckpt, show_progress=False)
    assert len(history_resumed) == 3

    predictor = Predictor.from_checkpoint(best_ckpt)
    metrics = evaluate_model(
        model=predictor.model,
        dataloader=test_loader,
        vocab=vocab,
        device=predictor.device,
        max_decode_length=6,
        criterion=trainer.criterion,
        output_dir=tmp_path / "outputs",
        split_name="test",
    )
    assert "recognition" in metrics
    assert "numerical_equal_length" in metrics
    assert (tmp_path / "outputs" / "test_metrics.json").exists()

    sample_img = tmp_path / df.iloc[6]["image_path"]
    pred = predictor.predict_image(sample_img, ground_truth=df.iloc[6]["label"])
    assert isinstance(pred["predicted_sequence"], str)
    assert pred["predicted_length"] == len(pred["predicted_sequence"])
    assert pred["ground_truth"] == df.iloc[6]["label"]


def test_trainer_scheduled_teacher_forcing_and_lr_scheduler(tmp_path: Path) -> None:
    import pytest

    config = {
        "project": {"name": "sched_test", "seed": 42, "device": "cpu"},
        "data": {"max_sequence_length": 5},
        "model": {"encoder_dim": 16, "embed_dim": 8, "decoder_hidden_dim": 16, "attention_dim": 8, "num_layers": 1, "dropout": 0.0},
        "training": {
            "learning_rate": 0.001,
            "epochs": 6,
            "teacher_forcing_start": 0.8,
            "teacher_forcing_end": 0.3,
            "lr_scheduler": {"enabled": True, "factor": 0.5, "patience": 2, "min_lr": 1e-6},
            "checkpoint_dir": "checkpoints",
            "output_dir": "outputs",
        },
        "inference": {"max_decode_length": 6},
    }
    trainer = Trainer(config=config, vocab=DigitVocabulary(), base_dir=tmp_path)
    assert trainer.scheduler is not None
    assert trainer.get_teacher_forcing_ratio(1, 6) == pytest.approx(0.8)
    assert trainer.get_teacher_forcing_ratio(6, 6) == pytest.approx(0.3)
    assert trainer.get_teacher_forcing_ratio(3, 5) == pytest.approx(0.55)
