"""Train and evaluate the hierarchical 2-stage classifier."""

import sys
import json
import pickle
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thoughtlink.data.loader import load_all, CLASS_NAMES, get_class_distribution
from thoughtlink.data.splitter import split_by_subject
from thoughtlink.preprocessing.eeg import preprocess_all
from thoughtlink.preprocessing.windowing import windows_from_samples
from thoughtlink.features.eeg_features import extract_features_from_windows
from thoughtlink.models.hierarchical import HierarchicalClassifier, RELAX_IDX
from thoughtlink.models.baseline import evaluate_model


def main():
    print("=" * 60)
    print("ThoughtLink - Hierarchical Model Training")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading dataset...")
    samples = load_all()
    print(f"Class distribution: {get_class_distribution(samples)}")

    # 2. Split by subject
    print("\n[2/5] Splitting by subject...")
    train_samples, test_samples = split_by_subject(samples, test_size=0.2)

    # 3. Preprocess
    print("\n[3/5] Preprocessing EEG...")
    preprocess_all(train_samples)
    preprocess_all(test_samples)

    # 4. Extract windows and features
    print("\n[4/5] Extracting windows and features...")
    X_train_windows, y_train, _ = windows_from_samples(train_samples)
    X_test_windows, y_test, _ = windows_from_samples(test_samples)

    X_train = extract_features_from_windows(X_train_windows)
    X_test = extract_features_from_windows(X_test_windows)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 5. Train hierarchical model
    print("\n[5/5] Training hierarchical classifier...")
    print("-" * 40)

    model = HierarchicalClassifier(stage1_threshold=0.5)
    model.fit(X_train, y_train)

    # Evaluate
    results = evaluate_model(model, X_test, y_test, class_names=CLASS_NAMES)

    print(f"\nHierarchical Model Results:")
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  Kappa: {results['kappa']:.3f}")
    print(f"\nConfusion Matrix:")
    print(results["confusion_matrix"])

    # Stage 1 accuracy (binary: relax vs active)
    y_binary_test = (y_test != RELAX_IDX).astype(int)
    stage1_proba = model.stage1_model.predict_proba(X_test)
    stage1_pred = (stage1_proba[:, 1] > 0.5).astype(int)
    stage1_acc = (stage1_pred == y_binary_test).mean()
    print(f"\nStage 1 (Relax vs Active) Accuracy: {stage1_acc:.3f}")

    # False trigger rate: non-Relax predictions during Relax periods
    relax_mask = y_test == RELAX_IDX
    if relax_mask.sum() > 0:
        y_pred = model.predict(X_test)
        false_triggers = (y_pred[relax_mask] != RELAX_IDX).sum()
        ftr = false_triggers / relax_mask.sum()
        print(f"False Trigger Rate (during Relax): {ftr:.3f} ({false_triggers}/{relax_mask.sum()})")

    # Save model
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "hierarchical_model.pkl", "wb") as f:
        pickle.dump(model, f)

    results_summary = {
        "accuracy": results["accuracy"],
        "kappa": results["kappa"],
        "stage1_accuracy": float(stage1_acc),
        "false_trigger_rate": float(ftr) if relax_mask.sum() > 0 else None,
    }
    with open(output_dir / "hierarchical_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nModel and results saved to {output_dir}/")


if __name__ == "__main__":
    main()
