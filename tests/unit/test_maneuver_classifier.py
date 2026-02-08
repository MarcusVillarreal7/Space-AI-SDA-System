"""
Tests for CNNLSTMManeuverClassifier model.

Covers: CNN1D, LSTMEncoder, AttentionPooling, CNNLSTMManeuverClassifier,
        config/factory methods, and class name lookup.
"""

import pytest
import torch
import torch.nn as nn

from src.ml.models.maneuver_classifier import (
    CNN1D,
    LSTMEncoder,
    AttentionPooling,
    CNNLSTMManeuverClassifier,
    CNNLSTMClassifierConfig,
    ManeuverClassifier,
    get_class_name,
    CLASS_NAMES,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def default_config():
    return CNNLSTMClassifierConfig()


# ──────────────────────────────────────────────
# CNN1D
# ──────────────────────────────────────────────
class TestCNN1D:
    def test_output_shape(self, device):
        cnn = CNN1D(input_dim=24, channels=[32, 64, 128]).to(device)
        x = torch.randn(4, 20, 24, device=device)
        out = cnn(x)
        assert out.shape == (4, 20, 128)

    def test_transpose_correctness(self, device):
        """Output should be (batch, seq_len, channels[-1]), not transposed."""
        cnn = CNN1D(input_dim=24, channels=[16]).to(device)
        x = torch.randn(2, 15, 24, device=device)
        out = cnn(x)
        assert out.shape[1] == 15  # seq_len preserved
        assert out.shape[2] == 16  # last channel dim

    def test_custom_channel_progression(self, device):
        cnn = CNN1D(input_dim=10, channels=[8, 16, 32, 64]).to(device)
        x = torch.randn(2, 10, 10, device=device)
        out = cnn(x)
        assert out.shape == (2, 10, 64)


# ──────────────────────────────────────────────
# LSTMEncoder
# ──────────────────────────────────────────────
class TestLSTMEncoder:
    def test_output_shape(self, device):
        lstm = LSTMEncoder(input_dim=24, hidden_dim=128, bidirectional=True).to(device)
        x = torch.randn(4, 20, 24, device=device)
        out, (h_n, c_n) = lstm(x)
        assert out.shape == (4, 20, 256)  # bidirectional doubles output

    def test_hidden_state_shapes(self, device):
        lstm = LSTMEncoder(input_dim=24, hidden_dim=128, num_layers=2, bidirectional=True).to(device)
        x = torch.randn(4, 20, 24, device=device)
        _, (h_n, c_n) = lstm(x)
        # num_layers * num_directions, batch, hidden_dim
        assert h_n.shape == (4, 4, 128)
        assert c_n.shape == (4, 4, 128)

    def test_unidirectional(self, device):
        lstm = LSTMEncoder(input_dim=24, hidden_dim=64, bidirectional=False).to(device)
        x = torch.randn(2, 10, 24, device=device)
        out, _ = lstm(x)
        assert out.shape == (2, 10, 64)  # no doubling


# ──────────────────────────────────────────────
# AttentionPooling
# ──────────────────────────────────────────────
class TestAttentionPooling:
    def test_output_shape(self, device):
        pool = AttentionPooling(hidden_dim=256).to(device)
        x = torch.randn(4, 20, 256, device=device)
        out = pool(x)
        assert out.shape == (4, 256)

    def test_attention_weights_sum(self, device):
        pool = AttentionPooling(hidden_dim=64).to(device)
        x = torch.randn(2, 10, 64, device=device)
        # Manually compute attention weights to verify sum
        scores = torch.matmul(x, pool.weight.t()) + pool.bias
        weights = torch.softmax(scores, dim=1)
        sums = weights.sum(dim=1).squeeze(-1)
        torch.testing.assert_close(sums, torch.ones(2, device=device), atol=1e-5, rtol=1e-5)

    def test_different_seq_lengths(self, device):
        pool = AttentionPooling(hidden_dim=128).to(device)
        for seq_len in [5, 20, 50]:
            x = torch.randn(2, seq_len, 128, device=device)
            out = pool(x)
            assert out.shape == (2, 128)


# ──────────────────────────────────────────────
# CNNLSTMManeuverClassifier
# ──────────────────────────────────────────────
class TestCNNLSTMClassifier:
    def test_forward_shape(self, device, default_config):
        model = CNNLSTMManeuverClassifier(default_config).to(device)
        x = torch.randn(4, 20, 24, device=device)
        logits = model(x)
        assert logits.shape == (4, 6)

    def test_predict_returns_indices(self, device, default_config):
        model = CNNLSTMManeuverClassifier(default_config).to(device)
        model.eval()
        x = torch.randn(4, 20, 24, device=device)
        preds = model.predict(x)
        assert preds.shape == (4,)
        assert preds.min() >= 0
        assert preds.max() <= 5

    def test_predict_proba_sums_to_one(self, device, default_config):
        model = CNNLSTMManeuverClassifier(default_config).to(device)
        model.eval()
        x = torch.randn(4, 20, 24, device=device)
        probs = model.predict_proba(x)
        assert probs.shape == (4, 6)
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(4, device=device), atol=1e-5, rtol=1e-5)

    def test_param_count(self, default_config):
        model = CNNLSTMManeuverClassifier(default_config)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be approximately 719K
        assert 600_000 < n_params < 900_000, f"Unexpected param count: {n_params}"

    def test_eval_deterministic(self, device, default_config):
        model = CNNLSTMManeuverClassifier(default_config).to(device)
        model.eval()
        x = torch.randn(2, 20, 24, device=device)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)


# ──────────────────────────────────────────────
# Config & Factory
# ──────────────────────────────────────────────
class TestConfigAndFactory:
    def test_get_config_from_config(self):
        cfg = CNNLSTMClassifierConfig(lstm_hidden_dim=64)
        model = CNNLSTMManeuverClassifier(cfg)
        config_dict = model.get_config()
        model2 = CNNLSTMManeuverClassifier.from_config(config_dict)
        assert model2.config.lstm_hidden_dim == 64

    def test_post_init_fills_defaults(self):
        cfg = CNNLSTMClassifierConfig()
        assert cfg.cnn_channels == [32, 64, 128]
        assert cfg.classifier_dims == [256, 128]


# ──────────────────────────────────────────────
# Class names
# ──────────────────────────────────────────────
class TestClassNames:
    def test_all_six_classes(self):
        expected = ["Normal", "Drift/Decay", "Station-keeping",
                    "Minor Maneuver", "Major Maneuver", "Deorbit"]
        for idx, name in enumerate(expected):
            assert get_class_name(idx) == name

    def test_unknown_class(self):
        assert get_class_name(99) == "Unknown"
        assert get_class_name(-1) == "Unknown"
