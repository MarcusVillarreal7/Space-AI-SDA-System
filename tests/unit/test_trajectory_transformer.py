"""
Tests for TrajectoryTransformer model.

Covers: TransformerConfig, PositionalEncoding, MultiHeadAttention,
        Encoder/Decoder layers, ParallelPredictionHead, predict(),
        and config round-trip serialization.
"""

import pytest
import torch
import torch.nn as nn

from src.ml.models.trajectory_transformer import (
    TransformerConfig,
    PositionalEncoding,
    MultiHeadAttention,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    ParallelPredictionHead,
    TrajectoryTransformer,
    create_causal_mask,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def default_config():
    return TransformerConfig()


@pytest.fixture
def parallel_config():
    return TransformerConfig(use_parallel_decoder=True, pred_horizon=30)


# ──────────────────────────────────────────────
# TransformerConfig
# ──────────────────────────────────────────────
class TestTransformerConfig:
    def test_default_values(self):
        cfg = TransformerConfig()
        assert cfg.d_model == 64
        assert cfg.n_heads == 4
        assert cfg.input_dim == 24
        assert cfg.output_dim == 6
        assert cfg.pred_horizon == 30
        assert cfg.use_parallel_decoder is False

    def test_custom_config(self):
        cfg = TransformerConfig(d_model=128, n_heads=8, input_dim=32)
        assert cfg.d_model == 128
        assert cfg.n_heads == 8
        assert cfg.input_dim == 32

    def test_feature_dim_defaults(self):
        cfg = TransformerConfig()
        assert cfg.d_ff == 256
        assert cfg.dropout == 0.1
        assert cfg.max_seq_len == 100


# ──────────────────────────────────────────────
# PositionalEncoding
# ──────────────────────────────────────────────
class TestPositionalEncoding:
    def test_output_shape(self, device):
        pe = PositionalEncoding(d_model=64, max_len=100).to(device)
        x = torch.zeros(4, 20, 64, device=device)
        out = pe(x)
        assert out.shape == (4, 20, 64)

    def test_adds_nonzero_values(self, device):
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0).to(device)
        x = torch.zeros(2, 10, 64, device=device)
        out = pe(x)
        assert out.abs().sum() > 0, "PE should add non-zero positional information"

    def test_different_seq_lengths(self, device):
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0).to(device)
        for seq_len in [1, 10, 50, 99]:
            x = torch.zeros(2, seq_len, 64, device=device)
            out = pe(x)
            assert out.shape == (2, seq_len, 64)


# ──────────────────────────────────────────────
# MultiHeadAttention
# ──────────────────────────────────────────────
class TestMultiHeadAttention:
    def test_output_shape(self, device):
        mha = MultiHeadAttention(d_model=64, n_heads=4).to(device)
        q = k = v = torch.randn(2, 10, 64, device=device)
        out = mha(q, k, v)
        assert out.shape == (2, 10, 64)

    def test_with_mask(self, device):
        mha = MultiHeadAttention(d_model=64, n_heads=4).to(device)
        q = k = v = torch.randn(2, 10, 64, device=device)
        # Mask shape must be broadcastable to (batch, n_heads, seq, seq)
        mask = torch.ones(2, 1, 10, 10, device=device)
        out = mha(q, k, v, mask=mask)
        assert out.shape == (2, 10, 64)

    def test_cross_attention(self, device):
        mha = MultiHeadAttention(d_model=64, n_heads=4).to(device)
        q = torch.randn(2, 5, 64, device=device)
        k = v = torch.randn(2, 20, 64, device=device)
        out = mha(q, k, v)
        assert out.shape == (2, 5, 64)

    def test_d_model_divisibility_assertion(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=65, n_heads=4)


# ──────────────────────────────────────────────
# Encoder / Decoder layers
# ──────────────────────────────────────────────
class TestEncoderDecoder:
    def test_encoder_output_shape(self, device):
        layer = TransformerEncoderLayer(d_model=64, n_heads=4, d_ff=256).to(device)
        x = torch.randn(2, 20, 64, device=device)
        out = layer(x)
        assert out.shape == (2, 20, 64)

    def test_decoder_output_shape(self, device):
        layer = TransformerDecoderLayer(d_model=64, n_heads=4, d_ff=256).to(device)
        x = torch.randn(2, 30, 64, device=device)
        enc_out = torch.randn(2, 20, 64, device=device)
        out = layer(x, enc_out)
        assert out.shape == (2, 30, 64)

    def test_full_forward_pass(self, device):
        cfg = TransformerConfig()
        model = TrajectoryTransformer(cfg).to(device)
        src = torch.randn(4, 20, 24, device=device)
        tgt = torch.randn(4, 30, 24, device=device)
        out = model(src, tgt)
        assert out.shape == (4, 30, 6)

    def test_causal_mask(self, device):
        mask = create_causal_mask(10, device)
        assert mask.shape == (1, 10, 10)
        # Lower-triangular + diagonal should be ones (value 0 in the mask
        # means "allow"; the mask is constructed so upper triangle is 0)
        # Verify upper triangle is zero (blocked)
        upper = mask[0].triu(diagonal=1)
        assert upper.sum() == 0


# ──────────────────────────────────────────────
# ParallelPredictionHead
# ──────────────────────────────────────────────
class TestParallelHead:
    def test_forward_shape(self, device, parallel_config):
        model = TrajectoryTransformer(parallel_config).to(device)
        src = torch.randn(4, 20, 24, device=device)
        out = model.forward_parallel(src)
        assert out.shape == (4, 30, 6)

    def test_raises_without_head(self, device, default_config):
        model = TrajectoryTransformer(default_config).to(device)
        src = torch.randn(4, 20, 24, device=device)
        with pytest.raises(RuntimeError, match="Parallel prediction head not enabled"):
            model.forward_parallel(src)

    def test_predict_dispatches_to_parallel(self, device, parallel_config):
        model = TrajectoryTransformer(parallel_config).to(device)
        model.eval()
        src = torch.randn(2, 20, 24, device=device)
        out = model.predict(src, pred_horizon=30)
        assert out.shape == (2, 30, 6)

    def test_query_token_expansion(self, device, parallel_config):
        head = ParallelPredictionHead(parallel_config).to(device)
        assert head.query_tokens.shape == (1, 30, 64)
        enc_out = torch.randn(8, 20, 64, device=device)
        out = head(enc_out)
        assert out.shape == (8, 30, 6)


# ──────────────────────────────────────────────
# Predict (AR fallback + parallel)
# ──────────────────────────────────────────────
class TestPredict:
    def test_ar_fallback_shape(self, device, default_config):
        model = TrajectoryTransformer(default_config).to(device)
        model.eval()
        src = torch.randn(2, 20, 24, device=device)
        out = model.predict(src, pred_horizon=10)
        assert out.shape == (2, 10, 6)

    def test_ar_padding_path(self, device):
        """output_dim < input_dim triggers the padding branch."""
        cfg = TransformerConfig(input_dim=24, output_dim=6)
        model = TrajectoryTransformer(cfg).to(device)
        model.eval()
        src = torch.randn(2, 20, 24, device=device)
        out = model.predict(src, pred_horizon=5)
        assert out.shape == (2, 5, 6)

    def test_parallel_predict_shape(self, device, parallel_config):
        model = TrajectoryTransformer(parallel_config).to(device)
        model.eval()
        src = torch.randn(3, 20, 24, device=device)
        out = model.predict(src, pred_horizon=30)
        assert out.shape == (3, 30, 6)

    def test_eval_deterministic(self, device, parallel_config):
        model = TrajectoryTransformer(parallel_config).to(device)
        model.eval()
        src = torch.randn(2, 20, 24, device=device)
        with torch.no_grad():
            out1 = model.predict(src, pred_horizon=30)
            out2 = model.predict(src, pred_horizon=30)
        torch.testing.assert_close(out1, out2)


# ──────────────────────────────────────────────
# Config round-trip
# ──────────────────────────────────────────────
class TestConfigRoundTrip:
    def test_get_config_from_config(self):
        cfg = TransformerConfig(d_model=128, n_heads=8)
        model = TrajectoryTransformer(cfg)
        config_dict = model.get_config()
        model2 = TrajectoryTransformer.from_config(config_dict)
        assert model2.config.d_model == 128
        assert model2.config.n_heads == 8

    def test_from_config_creates_correct_arch(self):
        config_dict = {'d_model': 32, 'n_heads': 2, 'n_encoder_layers': 1,
                       'n_decoder_layers': 1, 'd_ff': 64, 'dropout': 0.0,
                       'input_dim': 24, 'output_dim': 6, 'pred_horizon': 10,
                       'use_parallel_decoder': False}
        model = TrajectoryTransformer.from_config(config_dict)
        assert len(model.encoder_layers) == 1
        assert len(model.decoder_layers) == 1

    def test_parallel_flag_preserved(self):
        cfg = TransformerConfig(use_parallel_decoder=True)
        model = TrajectoryTransformer(cfg)
        config_dict = model.get_config()
        assert config_dict['use_parallel_decoder'] is True
        model2 = TrajectoryTransformer.from_config(config_dict)
        assert model2.parallel_head is not None
