"""End-to-end integration tests for the full BCI pipeline."""

import numpy as np
import pytest

from thoughtlink.data.loader import CLASS_NAMES
from thoughtlink.preprocessing.eeg import preprocess_eeg
from thoughtlink.preprocessing.windowing import extract_eeg_windows
from thoughtlink.features.eeg_features import extract_features_from_windows
from thoughtlink.models.baseline import build_baselines
from thoughtlink.bridge.intent_to_action import intent_to_action_name
from thoughtlink.inference.confidence import IntentConfidenceFilter


class TestEndToEndPipeline:
    """Test the complete pipeline from raw EEG to robot action."""

    @pytest.fixture
    def synthetic_eeg_chunk(self):
        """Create a synthetic 15s EEG chunk (7499, 6)."""
        rng = np.random.RandomState(42)
        return rng.randn(7499, 6) * 10  # microvolts

    @pytest.fixture
    def trained_model(self):
        """Train a simple model on synthetic data."""
        rng = np.random.RandomState(42)
        n_samples = 200
        n_features = 42
        X_train = rng.randn(n_samples, n_features)
        y_train = rng.choice(len(CLASS_NAMES), size=n_samples)

        models = build_baselines()
        models["svm_rbf"].fit(X_train, y_train)
        return models["svm_rbf"]

    def test_full_pipeline_eeg_to_action(self, synthetic_eeg_chunk, trained_model):
        """Test EEG → preprocessing → windowing → features → model → action."""
        # Step 1: Preprocess
        preprocessed = preprocess_eeg(synthetic_eeg_chunk, sfreq=500.0)
        assert preprocessed.shape == (7499, 6)
        assert np.all(np.isfinite(preprocessed))

        # Step 2: Window (15s chunk, 8s stimulus duration assumed)
        windows = extract_eeg_windows(
            preprocessed,
            duration=8.0,
            sfreq=500.0,
            window_duration_s=1.0,
            window_stride_s=0.5
        )
        assert windows.shape[1] == 500  # 1s window
        assert windows.shape[2] == 6    # 6 channels

        # Step 3: Extract features
        features = extract_features_from_windows(windows)
        assert features.shape[0] == windows.shape[0]  # One feature vector per window
        assert features.shape[1] == 42  # Default features (band power + Hjorth)
        assert np.all(np.isfinite(features))

        # Step 4: Model prediction
        predictions = trained_model.predict(features)
        assert predictions.shape == (features.shape[0],)
        assert all(p in range(len(CLASS_NAMES)) for p in predictions)

        probabilities = trained_model.predict_proba(features)
        assert probabilities.shape == (features.shape[0], len(CLASS_NAMES))
        assert np.all(probabilities >= 0.0)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

        # Step 5: Intent to action mapping
        for pred in predictions:
            intent = CLASS_NAMES[pred]
            action_name = intent_to_action_name(intent)
            assert action_name in {"RIGHT", "LEFT", "FORWARD", "STOP"}

    def test_pipeline_with_wavelets(self, synthetic_eeg_chunk, trained_model):
        """Test pipeline with wavelet features enabled."""
        # Retrain model with wavelet features (114 features)
        rng = np.random.RandomState(42)
        n_samples = 200
        n_features = 114  # With wavelets
        X_train = rng.randn(n_samples, n_features)
        y_train = rng.choice(len(CLASS_NAMES), size=n_samples)

        models = build_baselines()
        models["svm_rbf"].fit(X_train, y_train)

        # Preprocess and window
        preprocessed = preprocess_eeg(synthetic_eeg_chunk, sfreq=500.0)
        windows = extract_eeg_windows(
            preprocessed,
            duration=8.0,
            sfreq=500.0,
            window_duration_s=1.0,
            window_stride_s=0.5
        )

        # Extract features with wavelets
        wavelet_config = {
            "family": "db4",
            "levels": [3, 4, 5],
            "include_approx": True,
            "stats": ["energy", "entropy", "std"]
        }
        features = extract_features_from_windows(
            windows,
            include_wavelet=True,
            wavelet_config=wavelet_config
        )

        assert features.shape[1] == 114, "Should have 42 + 72 = 114 features with wavelets"
        assert np.all(np.isfinite(features))

        # Predict
        predictions = models["svm_rbf"].predict(features)
        assert predictions.shape == (features.shape[0],)

    def test_pipeline_with_stability_filter(self, synthetic_eeg_chunk, trained_model):
        """Test pipeline with stability/confidence filtering."""
        # Preprocess → window → features
        preprocessed = preprocess_eeg(synthetic_eeg_chunk, sfreq=500.0)
        windows = extract_eeg_windows(
            preprocessed,
            duration=8.0,
            sfreq=500.0,
            window_duration_s=1.0,
            window_stride_s=0.5
        )
        features = extract_features_from_windows(windows)

        # Predict probabilities
        probabilities = trained_model.predict_proba(features)

        # Apply stability filter
        stability = IntentConfidenceFilter(
            confidence_threshold=0.6,
            debounce_count=2,
            hysteresis_margin=0.1
        )

        stable_intents = []
        for probs in probabilities:
            filtered_intent = stability.update(probs, CLASS_NAMES)
            stable_intents.append(filtered_intent)

        # Verify all filtered intents are valid
        assert all(intent in CLASS_NAMES for intent in stable_intents)

        # Stability should reduce switches (compared to raw predictions)
        raw_switches = np.sum(np.diff([np.argmax(p) for p in probabilities]) != 0)
        stable_switches = np.sum([a != b for a, b in zip(stable_intents[:-1], stable_intents[1:])])
        assert stable_switches <= raw_switches, \
            "Stability filter should reduce or maintain the number of intent switches"

    def test_pipeline_preserves_data_shape_consistency(self, synthetic_eeg_chunk):
        """Verify shapes are consistent throughout the pipeline."""
        # Raw chunk
        assert synthetic_eeg_chunk.shape == (7499, 6)

        # After preprocessing
        preprocessed = preprocess_eeg(synthetic_eeg_chunk)
        assert preprocessed.shape == (7499, 6)

        # After windowing (1s windows, 50% overlap)
        windows = extract_eeg_windows(
            preprocessed,
            duration=8.0,
            window_duration_s=1.0,
            window_stride_s=0.5
        )
        n_windows = windows.shape[0]
        assert windows.shape == (n_windows, 500, 6)

        # After feature extraction
        features = extract_features_from_windows(windows)
        assert features.shape == (n_windows, 42)

    def test_pipeline_deterministic_with_same_input(self, synthetic_eeg_chunk):
        """Pipeline should produce identical results for the same input."""
        # Run pipeline twice
        def run_pipeline(eeg):
            preprocessed = preprocess_eeg(eeg)
            windows = extract_eeg_windows(
                preprocessed,
                duration=8.0,
                window_duration_s=1.0,
                window_stride_s=0.5
            )
            features = extract_features_from_windows(windows)
            return features

        features1 = run_pipeline(synthetic_eeg_chunk.copy())
        features2 = run_pipeline(synthetic_eeg_chunk.copy())

        np.testing.assert_allclose(features1, features2, rtol=1e-10, atol=1e-12,
            err_msg="Pipeline should be deterministic")
