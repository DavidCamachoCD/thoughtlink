"""Train and evaluate baseline models on the ThoughtLink dataset."""

import sys
import os
import json
import pickle
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thoughtlink.data.loader import load_all, CLASS_NAMES, get_class_distribution
from thoughtlink.data.splitter import split_by_subject
from thoughtlink.preprocessing.eeg import preprocess_all
from thoughtlink.preprocessing.windowing import windows_from_samples
from thoughtlink.features.eeg_features import extract_features_from_windows
from thoughtlink.models.baseline import build_baselines, train_and_evaluate


def main():
    print("=" * 60)
    print("ThoughtLink - Baseline Training")
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
    print(f"Train windows: {X_train_windows.shape}, Test windows: {X_test_windows.shape}")

    X_train = extract_features_from_windows(X_train_windows, include_time_domain=True)
    X_test = extract_features_from_windows(X_test_windows, include_time_domain=True)
    print(f"Train features: {X_train.shape}, Test features: {X_test.shape}")

    # 5. Train and evaluate
    print("\n[5/5] Training baselines...")
    print("-" * 40)

    # Binary: Left Fist (1) vs Right Fist (0)
    left_idx = CLASS_NAMES.index("Left Fist")
    right_idx = CLASS_NAMES.index("Right Fist")

    binary_mask_train = np.isin(y_train, [left_idx, right_idx])
    binary_mask_test = np.isin(y_test, [left_idx, right_idx])

    if binary_mask_train.sum() > 0 and binary_mask_test.sum() > 0:
        X_bin_train = X_train[binary_mask_train]
        y_bin_train = (y_train[binary_mask_train] == left_idx).astype(int)
        X_bin_test = X_test[binary_mask_test]
        y_bin_test = (y_test[binary_mask_test] == left_idx).astype(int)

        print(f"\n--- Binary Classification (Left vs Right Fist) ---")
        print(f"Train: {len(y_bin_train)}, Test: {len(y_bin_test)}")
        binary_models = build_baselines()
        binary_results = train_and_evaluate(
            binary_models, X_bin_train, y_bin_train, X_bin_test, y_bin_test,
            class_names=["Right Fist", "Left Fist"],
        )
    else:
        print("Warning: Not enough Left/Right Fist samples for binary classification.")
        binary_results = {}

    # Multi-class: All 5 classes
    print(f"\n--- Multi-class Classification (5 classes) ---")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    multi_models = build_baselines()
    multi_results = train_and_evaluate(
        multi_models, X_train, y_train, X_test, y_test,
        class_names=CLASS_NAMES,
    )

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    results_summary = {
        "binary": {k: {"accuracy": v["accuracy"], "kappa": v["kappa"]}
                   for k, v in binary_results.items()},
        "multiclass": {k: {"accuracy": v["accuracy"], "kappa": v["kappa"]}
                       for k, v in multi_results.items()},
    }

    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    # Save best multi-class model
    best_name = max(multi_results, key=lambda k: multi_results[k]["accuracy"])
    best_model = multi_models[best_name]
    with open(output_dir / "best_baseline.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"\nBest model: {best_name} (accuracy={multi_results[best_name]['accuracy']:.3f})")
    print(f"Results saved to {output_dir}/")

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<20} {'Binary Acc':<15} {'Multi Acc':<15} {'Multi Kappa':<15}")
    print("-" * 65)
    for name in multi_results:
        bin_acc = binary_results.get(name, {}).get("accuracy", 0)
        multi_acc = multi_results[name]["accuracy"]
        multi_kappa = multi_results[name]["kappa"]
        print(f"{name:<20} {bin_acc:<15.3f} {multi_acc:<15.3f} {multi_kappa:<15.3f}")


if __name__ == "__main__":
    main()
