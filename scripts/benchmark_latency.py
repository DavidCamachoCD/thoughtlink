"""Benchmark inference latency for all pipeline components."""

import sys
import time
import pickle
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thoughtlink.preprocessing.eeg import preprocess_eeg
from thoughtlink.features.eeg_features import extract_window_features
from thoughtlink.inference.confidence import StabilityPipeline
from thoughtlink.data.loader import CLASS_NAMES


def benchmark(func, *args, n_runs: int = 100, **kwargs) -> tuple[float, float]:
    """Benchmark a function over n_runs. Returns (mean_ms, std_ms)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


def main():
    print("=" * 60)
    print("ThoughtLink - Latency Benchmark")
    print("=" * 60)

    n_runs = 100
    n_channels = 6
    n_samples = 500  # 1-second window at 500 Hz

    # Generate synthetic data for benchmarking
    window = np.random.randn(n_samples, n_channels) * 10  # ~10 uV

    # 1. Preprocessing (bandpass filter + CAR)
    print(f"\nBenchmarking over {n_runs} runs...")
    print("-" * 50)

    # Full 15s preprocessing
    full_eeg = np.random.randn(7499, n_channels) * 10
    mean_ms, std_ms = benchmark(preprocess_eeg, full_eeg, n_runs=n_runs)
    print(f"{'EEG Preprocessing (15s):':<35} {mean_ms:>8.2f} +/- {std_ms:.2f} ms")

    # 2. Feature extraction
    mean_ms, std_ms = benchmark(extract_window_features, window, n_runs=n_runs)
    print(f"{'Feature Extraction (1s window):':<35} {mean_ms:>8.2f} +/- {std_ms:.2f} ms")

    # 3. Model inference (if trained model exists)
    model_path = Path("results/best_baseline.pkl")
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        features = extract_window_features(window)
        mean_ms, std_ms = benchmark(
            model.predict_proba, features.reshape(1, -1), n_runs=n_runs,
        )
        print(f"{'Model Inference (baseline):':<35} {mean_ms:>8.2f} +/- {std_ms:.2f} ms")

    hierarchical_path = Path("results/hierarchical_model.pkl")
    if hierarchical_path.exists():
        with open(hierarchical_path, "rb") as f:
            h_model = pickle.load(f)

        features = extract_window_features(window)
        mean_ms, std_ms = benchmark(
            h_model.predict_proba, features.reshape(1, -1), n_runs=n_runs,
        )
        print(f"{'Model Inference (hierarchical):':<35} {mean_ms:>8.2f} +/- {std_ms:.2f} ms")

    # 4. Stability pipeline
    pipeline = StabilityPipeline()
    probs = np.random.dirichlet(np.ones(5))
    mean_ms, std_ms = benchmark(pipeline.process, probs, CLASS_NAMES, n_runs=n_runs)
    print(f"{'Stability Pipeline:':<35} {mean_ms:>8.2f} +/- {std_ms:.2f} ms")

    # 5. End-to-end (feature extraction + model + stability)
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        def end_to_end(window):
            features = extract_window_features(window)
            probs = model.predict_proba(features.reshape(1, -1))[0]
            pipeline.process(probs, CLASS_NAMES)

        pipeline.reset()
        mean_ms, std_ms = benchmark(end_to_end, window, n_runs=n_runs)
        print(f"{'End-to-End (feat+model+stability):':<35} {mean_ms:>8.2f} +/- {std_ms:.2f} ms")

    print("\n" + "=" * 60)
    print("TARGET: < 50ms end-to-end for real-time BCI control")
    print("=" * 60)


if __name__ == "__main__":
    main()
