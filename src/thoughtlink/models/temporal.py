"""Temporal models (GRU) for sequential EEG feature decoding.

Processes sequences of extracted feature vectors to capture temporal
dynamics across consecutive EEG windows.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Group consecutive feature vectors into sequences.

    Args:
        features: Shape (n_windows, n_features).
        labels: Shape (n_windows,).
        seq_len: Number of consecutive windows per sequence.

    Returns:
        sequences: Shape (n_sequences, seq_len, n_features).
        seq_labels: Shape (n_sequences,), label of the last window.
    """
    n = len(features)
    if n < seq_len:
        raise ValueError(f"Not enough windows ({n}) for seq_len={seq_len}")

    sequences = []
    seq_labels = []
    for i in range(n - seq_len + 1):
        sequences.append(features[i : i + seq_len])
        seq_labels.append(labels[i + seq_len - 1])

    return np.array(sequences), np.array(seq_labels)


class TemporalEEGNet(nn.Module):
    """GRU-based temporal model for EEG feature sequences.

    Processes sequences of pre-extracted features (e.g. band power + Hjorth)
    to capture temporal dynamics across consecutive windows.

    Input shape: (batch, seq_len, n_features)
    Output shape: (batch, n_classes) logits
    """

    def __init__(
        self,
        n_features: int = 66,
        n_classes: int = 5,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, n_features).

        Returns:
            logits: (batch, n_classes).
        """
        rnn_out, _ = self.gru(x)
        last_hidden = rnn_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out)

    def predict_proba_numpy(self, sequences: np.ndarray) -> np.ndarray:
        """Inference from numpy arrays.

        Args:
            sequences: (n_sequences, seq_len, n_features) or
                       (seq_len, n_features) for a single sequence.

        Returns:
            Probability matrix (n_sequences, n_classes).
        """
        self.eval()
        with torch.no_grad():
            if sequences.ndim == 2:
                sequences = sequences[np.newaxis]
            x = torch.from_numpy(sequences).float()
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1).numpy()
        return probs
