"""Train and evaluate models with DWT wavelet features on ThoughtLink dataset.

This is a separate training pipeline for testing wavelet-based feature extraction
alongside the baseline pipeline. Uses db4 wavelet with configurable levels.
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thoughtlink.data.loader import load_all, CLASS_NAMES, get_class_distribution
from thoughtlink.data.splitter import split_by_subject
from thoughtlink.preprocessing.eeg import preprocess_all
from thoughtlink.preprocessing.windowing import windows_from_samples
from thoughtlink.features.eeg_features import extract_features_from_windows
from thoughtlink.models.baseline import build_baselines, evaluate_model


def main():
    print("=" * 60)
    print("ThoughtLink - Wavelet Feature Training (Ari's Branch)")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading dataset...")
    samples = load_all()
    print(f"Class distribution: {get_class_distribution(samples)}")

    # 2. Split by subject
    print("\n[2/6] Splitting by subject...")
    train_samples, test_samples = split_by_subject(samples, test_size=0.2, random_state=42)

    # 3. Preprocess
    print("\n[3/6] Preprocessing EEG...")
    preprocess_all(train_samples)
    preprocess_all(test_samples)

    # 4. Load wavelet configuration
    print("\n[4/6] Loading wavelet configuration...")
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    feat_cfg = config.get("features", {})
    wavelet_cfg = feat_cfg.get("wavelet", {})

    print(f"Wavelet family: {wavelet_cfg.get('family', 'db4')}")
    print(f"Decomposition levels: {wavelet_cfg.get('levels', [3, 4, 5])}")
    print(f"Statistics: {wavelet_cfg.get('stats', ['energy', 'entropy', 'std'])}")

    # 5. Extract windows and features WITH wavelets
    print("\n[5/6] Extracting windows and wavelet features...")
    X_train_windows, y_train, _ = windows_from_samples(train_samples)
    X_test_windows, y_test, _ = windows_from_samples(test_samples)
    print(f"Train windows: {X_train_windows.shape}, Test windows: {X_test_windows.shape}")

    # Extract features with wavelets enabled
    X_train = extract_features_from_windows(
        X_train_windows,
        include_wavelet=True,
        wavelet_config=wavelet_cfg,
    )
    X_test = extract_features_from_windows(
        X_test_windows,
        include_wavelet=True,
        wavelet_config=wavelet_cfg,
    )
    print(f"Train features: {X_train.shape} (42 baseline + 72 wavelet = 114 total)")
    print(f"Test features: {X_test.shape}")

    # 6. Train and evaluate models
    print("\n[6/6] Training models with wavelet features...")
    print("-" * 60)

    models = build_baselines()
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, CLASS_NAMES)
        results[name] = metrics

        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.3f}")

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    results_path = output_dir / "wavelet_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")

    # Save best model (SVM RBF typically best for wavelets)
    best_model = models["svm_rbf"]
    model_path = output_dir / "wavelet_svm_rbf.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"✓ Best model saved to: {model_path}")

    print("\n" + "=" * 60)
    print("Wavelet training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
