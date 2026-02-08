"""Tests for TemporalEEGNet model and sequence utilities."""

import numpy as np
import pytest
import torch

from thoughtlink.models.temporal import TemporalEEGNet, create_sequences


class TestCreateSequences:
    def test_output_shapes(self):
        features = np.random.randn(50, 66)
        labels = np.random.randint(0, 5, 50)
        seqs, seq_labels = create_sequences(features, labels, seq_len=10)
        assert seqs.shape == (41, 10, 66)
        assert seq_labels.shape == (41,)

    def test_label_is_last_window(self):
        features = np.random.randn(20, 66)
        labels = np.arange(20)
        seqs, seq_labels = create_sequences(features, labels, seq_len=5)
        # Each label should be the last element in the sequence
        for i, lbl in enumerate(seq_labels):
            assert lbl == i + 4

    def test_too_few_windows_raises(self):
        features = np.random.randn(3, 66)
        labels = np.random.randint(0, 5, 3)
        with pytest.raises(ValueError, match="Not enough windows"):
            create_sequences(features, labels, seq_len=10)

    def test_seq_len_equals_n_windows(self):
        features = np.random.randn(10, 66)
        labels = np.random.randint(0, 5, 10)
        seqs, seq_labels = create_sequences(features, labels, seq_len=10)
        assert seqs.shape == (1, 10, 66)


class TestTemporalEEGNet:
    def test_output_shape(self):
        model = TemporalEEGNet(n_features=66, n_classes=5)
        x = torch.randn(4, 10, 66)
        out = model(x)
        assert out.shape == (4, 5)

    def test_single_sequence(self):
        model = TemporalEEGNet(n_features=66, n_classes=5)
        x = torch.randn(1, 10, 66)
        out = model(x)
        assert out.shape == (1, 5)

    def test_different_n_classes(self):
        for n_classes in [2, 3, 5]:
            model = TemporalEEGNet(n_classes=n_classes)
            x = torch.randn(2, 10, 66)
            out = model(x)
            assert out.shape == (2, n_classes)

    def test_different_seq_lengths(self):
        model = TemporalEEGNet(n_features=66, n_classes=5)
        for seq_len in [5, 10, 20]:
            x = torch.randn(2, seq_len, 66)
            out = model(x)
            assert out.shape == (2, 5)

    def test_predict_proba_numpy(self):
        model = TemporalEEGNet(n_features=66, n_classes=5)
        sequences = np.random.randn(3, 10, 66).astype(np.float32)
        probs = model.predict_proba_numpy(sequences)
        assert probs.shape == (3, 5)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0)

    def test_predict_proba_single_sequence(self):
        model = TemporalEEGNet(n_features=66, n_classes=5)
        seq = np.random.randn(10, 66).astype(np.float32)
        probs = model.predict_proba_numpy(seq)
        assert probs.shape == (1, 5)

    def test_eval_mode_deterministic(self):
        model = TemporalEEGNet(n_features=66, n_classes=5)
        model.eval()
        x = torch.randn(2, 10, 66)
        with torch.no_grad():
            out1 = model(x).clone()
            out2 = model(x).clone()
        torch.testing.assert_close(out1, out2)

    def test_unidirectional(self):
        model = TemporalEEGNet(n_features=66, n_classes=5, bidirectional=False)
        x = torch.randn(2, 10, 66)
        out = model(x)
        assert out.shape == (2, 5)
