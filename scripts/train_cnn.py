"""Train EEGNet CNN on windowed EEG data and export to ONNX."""

import sys
import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thoughtlink.data.loader import load_all, CLASS_NAMES, get_class_distribution
from thoughtlink.data.splitter import split_by_subject
from thoughtlink.preprocessing.eeg import preprocess_all
from thoughtlink.preprocessing.windowing import windows_from_samples
from thoughtlink.preprocessing.alignment import euclidean_align
from thoughtlink.preprocessing.augmentation import EEGAugmentationDataset
from thoughtlink.models.cnn import EEGNet


def prepare_loader(
    X_windows: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Convert windowed EEG (n, 500, 6) to PyTorch DataLoader (n, 1, 6, 500)."""
    X_t = np.transpose(X_windows, (0, 2, 1)).astype(np.float32)
    np.nan_to_num(X_t, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    X_tensor = torch.from_numpy(X_t).unsqueeze(1)
    y_tensor = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def prepare_augmented_loader(
    X_windows: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    augment: bool = True,
) -> DataLoader:
    """Create augmented DataLoader for training."""
    X_t = np.transpose(X_windows, (0, 2, 1)).astype(np.float32)
    np.nan_to_num(X_t, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    dataset = EEGAugmentationDataset(X_t, y, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


def evaluate_cnn(model, loader, device):
    """Evaluate CNN. Returns (y_pred, y_proba)."""
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_probs)


def export_onnx(model, output_path: str, device: torch.device):
    """Export EEGNet to ONNX and verify with onnxruntime."""
    model.eval()
    model_cpu = model.cpu()
    dummy = torch.randn(1, 1, 6, 500)

    try:
        torch.onnx.export(
            model_cpu,
            dummy,
            output_path,
            input_names=["eeg_input"],
            output_names=["class_logits"],
            dynamic_axes={"eeg_input": {0: "batch"}, "class_logits": {0: "batch"}},
            opset_version=17,
        )
        print(f"  ONNX model saved to {output_path}")

        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_path)
            onnx_out = session.run(None, {"eeg_input": dummy.numpy()})
            with torch.no_grad():
                torch_out = model_cpu(dummy).numpy()
            diff = np.abs(onnx_out[0] - torch_out).max()
            print(f"  ONNX verification: max abs diff = {diff:.6f}")
        except ImportError:
            print("  onnxruntime not installed — skipping ONNX verification")
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        print("  Skipping ONNX export (install onnxscript if needed)")

    model.to(device)


def main():
    print("=" * 60)
    print("ThoughtLink - CNN (EEGNet) Training (Improved)")
    print("=" * 60)

    n_epochs = 100
    patience = 15
    batch_size = 64
    lr = 1e-3
    weight_decay = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # 1. Load data
    print("\n[1/6] Loading dataset...")
    samples = load_all()
    print(f"Loaded {len(samples)} samples")
    print(f"Class distribution: {get_class_distribution(samples)}")

    # 2. Split by subject
    print("\n[2/6] Splitting by subject...")
    train_samples, test_samples = split_by_subject(samples, test_size=0.2)
    print(f"Train subjects: {len(set(s['subject_id'] for s in train_samples))}, "
          f"Test subjects: {len(set(s['subject_id'] for s in test_samples))}")

    # 3. Preprocess
    print("\n[3/6] Preprocessing EEG...")
    preprocess_all(train_samples)
    preprocess_all(test_samples)

    # 4. Extract windows + Euclidean alignment
    print("\n[4/6] Extracting windows + alignment...")
    X_train_windows, y_train, train_subj_ids = windows_from_samples(train_samples)
    X_test_windows, y_test, test_subj_ids = windows_from_samples(test_samples)

    # Apply Euclidean alignment
    print("  Applying Euclidean alignment...")
    X_train_windows = euclidean_align(X_train_windows, train_subj_ids)
    X_test_windows = euclidean_align(X_test_windows, test_subj_ids)
    print(f"Train: {X_train_windows.shape}, Test: {X_test_windows.shape}")

    # Augmented training loader, standard test loader
    train_loader = prepare_augmented_loader(X_train_windows, y_train, batch_size=batch_size, augment=True)
    test_loader = prepare_loader(X_test_windows, y_test, batch_size=batch_size, shuffle=False)

    # 5. Train
    print("\n[5/6] Training EEGNet (improved)...")
    print("-" * 40)
    model = EEGNet(n_classes=5, n_channels=6, n_samples=500, f1=16, f2=32, d=2, dropout=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Class weights for balanced loss
    counts = Counter(y_train.tolist())
    total = sum(counts.values())
    weights = torch.tensor([total / (len(counts) * counts.get(i, 1)) for i in range(5)]).float()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)

    # LR scheduler: warmup + cosine
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[5])

    best_acc = 0.0
    epochs_without_improvement = 0
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        y_pred, _ = evaluate_cnn(model, test_loader, device)
        acc = accuracy_score(y_test, y_pred)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "cnn_best.pt")
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={train_loss:.4f}  acc={acc:.3f}  best={best_acc:.3f}")

        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # 6. Final evaluation with best model
    print("\n[6/6] Final evaluation...")
    print("-" * 40)
    model.load_state_dict(torch.load(output_dir / "cnn_best.pt", weights_only=True))
    model.to(device)
    y_pred, y_proba = evaluate_cnn(model, test_loader, device)

    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

    print(f"\nCNN (EEGNet Improved) Results:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  Kappa:    {kappa:.3f}")
    print(f"  Params:   {n_params:,}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES)}")

    # Save results
    results = {
        "accuracy": float(acc),
        "kappa": float(kappa),
        "n_params": n_params,
        "n_epochs": n_epochs,
        "best_epoch_acc": float(best_acc),
        "report": report,
    }
    with open(output_dir / "cnn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), output_dir / "cnn_model.pt")

    # ONNX export
    print("\nExporting to ONNX...")
    export_onnx(model, str(output_dir / "cnn_model.onnx"), device)

    print(f"\nAll outputs saved to {output_dir}/")
    print(f"  cnn_best.pt, cnn_model.pt, cnn_model.onnx, cnn_results.json")


if __name__ == "__main__":
    main()
