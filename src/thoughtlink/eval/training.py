"""Training-as-a-function adapters used by the ablation harness.

The original `scripts/train_*.py` are CLI wrappers that load data, preprocess,
extract features, train and persist artefacts. For factorial ablation we need
to invoke training in-process and many times (once per seed × model variant)
without re-running the full data pipeline. This module exposes the training
loop as a pure function that takes already-prepared arrays and returns
`(model, metrics)`.

The CLI scripts in `scripts/train_*.py` delegate to these functions to stay
backward-compatible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

from thoughtlink.eval._torch_device import select_device
from thoughtlink.models.baseline import build_baselines, evaluate_model
from thoughtlink.models.cnn import EEGNet
from thoughtlink.models.hierarchical import RELAX_IDX, HierarchicalClassifier
from thoughtlink.preprocessing.alignment import euclidean_align
from thoughtlink.preprocessing.augmentation import EEGAugmentationDataset


@dataclass
class TrainConfig:
    """Hyperparameters for the in-process trainers."""

    seed: int = 42

    # CNN-specific
    n_epochs: int = 100
    patience: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_alignment: bool = True
    cnn_f1: int = 16
    cnn_f2: int = 32
    cnn_d: int = 2
    cnn_dropout: float = 0.3

    # Selection rule for the baseline family. One of {"random_forest",
    # "svm_rbf", "svm_linear", "logreg"}.
    baseline_model: str = "random_forest"


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainConfig,
) -> tuple:
    """Train the canonical baseline model on engineered features.

    Args:
        X_train: Feature matrix (n, n_features).
        y_train: Integer labels (n,).
        config:  TrainConfig (only `baseline_model` is relevant here).

    Returns:
        (fitted_pipeline, metrics_dict).
    """
    pipelines = build_baselines()
    if config.baseline_model not in pipelines:
        raise ValueError(
            f"Unknown baseline_model {config.baseline_model!r}; "
            f"choose from {list(pipelines)}"
        )
    model = pipelines[config.baseline_model]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    metrics = {
        "model_kind": config.baseline_model,
        "train_accuracy": float(accuracy_score(y_train, y_pred)),
        "train_kappa": float(cohen_kappa_score(y_train, y_pred)),
    }
    return model, metrics


def train_hierarchical(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainConfig,
) -> tuple:
    """Train the 2-stage hierarchical classifier on engineered features."""
    model = HierarchicalClassifier(stage1_threshold=0.5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    y_binary = (y_train != RELAX_IDX).astype(int)
    stage1_proba = model.stage1_model.predict_proba(X_train)
    stage1_pred = (stage1_proba[:, 1] > 0.5).astype(int)
    metrics = {
        "train_accuracy": float(accuracy_score(y_train, y_pred)),
        "train_kappa": float(cohen_kappa_score(y_train, y_pred)),
        "stage1_train_accuracy": float((stage1_pred == y_binary).mean()),
    }
    return model, metrics


def _prepare_cnn_loader(
    X_windows: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    augment: bool,
) -> DataLoader:
    """(n, samples, channels) -> DataLoader with shape (n, 1, channels, samples)."""
    X_t = np.transpose(X_windows, (0, 2, 1)).astype(np.float32)
    np.nan_to_num(X_t, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if augment:
        dataset = EEGAugmentationDataset(X_t, y, augment=True)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    X_tensor = torch.from_numpy(X_t).unsqueeze(1)
    y_tensor = torch.from_numpy(y).long()
    return DataLoader(
        TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=False
    )


def train_cnn(
    X_train_windows: np.ndarray,
    y_train: np.ndarray,
    train_subj_ids: np.ndarray,
    config: TrainConfig,
    *,
    n_classes: int = 5,
    n_channels: int = 6,
    n_samples: int = 500,
) -> tuple:
    """Train EEGNet on raw windowed EEG.

    Note: this trainer does not consume a held-out validation set. It runs
    the full epoch budget without early stopping, since the ablation harness
    holds out subjects for calibration / test downstream and re-using them
    here would leak into the post-hoc layer's evaluation.

    Args:
        X_train_windows: (n_windows, n_samples_per_window, n_channels) raw EEG.
        y_train: integer labels (n_windows,).
        train_subj_ids: per-window subject ids (used by Euclidean alignment).
        config: TrainConfig.

    Returns:
        (trained_eegnet, metrics_dict). The model is on CPU.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.use_alignment:
        X_train_windows = euclidean_align(X_train_windows, train_subj_ids)

    device = select_device()
    print(f"[CNN] device={device.type}")
    train_loader = _prepare_cnn_loader(
        X_train_windows, y_train, batch_size=config.batch_size, augment=True
    )
    eval_loader = _prepare_cnn_loader(
        X_train_windows, y_train, batch_size=config.batch_size, augment=False
    )

    model = EEGNet(
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        f1=config.cnn_f1,
        f2=config.cnn_f2,
        d=config.cnn_d,
        dropout=config.cnn_dropout,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    counts = Counter(y_train.tolist())
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)]
    ).float()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)

    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[5]
    )

    best_acc = 0.0
    epochs_without_improvement = 0
    best_state = None
    for epoch in range(config.n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Use the train set itself to track progress (no peeking at calib/test).
        model.eval()
        all_pred = []
        with torch.no_grad():
            for X_batch, _ in eval_loader:
                X_batch = X_batch.to(device)
                all_pred.append(model(X_batch).argmax(dim=1).cpu().numpy())
        acc = accuracy_score(y_train, np.concatenate(all_pred))
        if acc > best_acc:
            best_acc = acc
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu()
    n_params = sum(p.numel() for p in model.parameters())
    metrics = {
        "n_params": int(n_params),
        "best_train_accuracy": float(best_acc),
        "epochs_run": int(epoch + 1),
    }
    return model, metrics
