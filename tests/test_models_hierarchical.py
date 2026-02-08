"""Tests for hierarchical two-stage classifier."""

import numpy as np
import pytest

from thoughtlink.models.hierarchical import (
    HierarchicalClassifier,
    RELAX_IDX,
    ACTIVE_CLASSES,
    ACTIVE_CLASS_NAMES,
)
from thoughtlink.data.loader import CLASS_NAMES


class TestHierarchicalConstants:
    def test_relax_idx(self):
        assert CLASS_NAMES[RELAX_IDX] == "Relax"

    def test_active_classes_excludes_relax(self):
        assert len(ACTIVE_CLASSES) == 4
        assert RELAX_IDX not in ACTIVE_CLASSES

    def test_active_class_names(self):
        assert len(ACTIVE_CLASS_NAMES) == 4
        assert "Relax" not in ACTIVE_CLASS_NAMES


class TestHierarchicalClassifier:
    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.RandomState(42)
        n_samples = 200
        n_features = 42
        X = rng.randn(n_samples, n_features)
        y = rng.choice(len(CLASS_NAMES), size=n_samples)
        return X, y

    def test_fit_returns_self(self, synthetic_data):
        X, y = synthetic_data
        clf = HierarchicalClassifier()
        result = clf.fit(X, y)
        assert result is clf

    def test_predict_shape(self, synthetic_data):
        X, y = synthetic_data
        clf = HierarchicalClassifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_valid_classes(self, synthetic_data):
        X, y = synthetic_data
        clf = HierarchicalClassifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert all(p in range(len(CLASS_NAMES)) for p in preds)

    def test_predict_proba_shape(self, synthetic_data):
        X, y = synthetic_data
        clf = HierarchicalClassifier()
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        assert probs.shape == (len(y), len(CLASS_NAMES))

    def test_predict_proba_sums_to_one(self, synthetic_data):
        X, y = synthetic_data
        clf = HierarchicalClassifier()
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_non_negative(self, synthetic_data):
        X, y = synthetic_data
        clf = HierarchicalClassifier()
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        assert np.all(probs >= 0.0)

    def test_classes_attribute(self, synthetic_data):
        X, y = synthetic_data
        clf = HierarchicalClassifier()
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.classes_, np.arange(len(CLASS_NAMES)))

    def test_relax_predictions_possible(self, synthetic_data):
        X, y = synthetic_data
        clf = HierarchicalClassifier()
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        assert probs[:, RELAX_IDX].sum() > 0

    def test_make_binary_labels(self, synthetic_data):
        X, y = synthetic_data
        clf = HierarchicalClassifier()
        binary = clf._make_binary_labels(y)
        assert set(binary).issubset({0, 1})
        assert (binary == 0).sum() == (y == RELAX_IDX).sum()


class TestHierarchicalStageBehavior:
    """Test stage-level gating and probability decomposition."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a trained hierarchical classifier with synthetic data."""
        rng = np.random.RandomState(42)
        n_samples = 300
        n_features = 42

        # Create separable data: Relax vs Active
        X_relax = rng.randn(60, n_features) - 2.0  # Shifted left
        y_relax = np.full(60, RELAX_IDX)

        X_active = rng.randn(240, n_features) + 2.0  # Shifted right
        y_active = rng.choice(ACTIVE_CLASSES, size=240)

        X = np.vstack([X_relax, X_active])
        y = np.hstack([y_relax, y_active])

        # Shuffle
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]

        clf = HierarchicalClassifier()
        clf.fit(X, y)
        return clf, X, y

    def test_stage1_gates_relax_predictions(self, trained_classifier):
        """When Stage 1 predicts Relax, final prediction should be Relax."""
        clf, X, y = trained_classifier

        # Get stage 1 probabilities
        stage1_proba = clf.stage1_model.predict_proba(X)
        p_relax_stage1 = stage1_proba[:, 0]

        # Get final predictions
        final_preds = clf.predict(X)

        # Samples where Stage 1 strongly predicts Relax should get Relax final prediction
        strong_relax_mask = p_relax_stage1 > 0.8
        if strong_relax_mask.sum() > 0:
            assert np.all(final_preds[strong_relax_mask] == RELAX_IDX), \
                "Strong Stage 1 Relax predictions should result in final Relax predictions"

    def test_stage1_gates_active_predictions(self, trained_classifier):
        """When Stage 1 predicts Active, final prediction should be one of the active classes."""
        clf, X, y = trained_classifier

        # Get stage 1 probabilities
        stage1_proba = clf.stage1_model.predict_proba(X)
        p_active_stage1 = stage1_proba[:, 1]

        # Get final predictions
        final_preds = clf.predict(X)

        # Samples where Stage 1 strongly predicts Active should get active class predictions
        strong_active_mask = p_active_stage1 > 0.8
        if strong_active_mask.sum() > 0:
            assert np.all(np.isin(final_preds[strong_active_mask], ACTIVE_CLASSES)), \
                "Strong Stage 1 Active predictions should result in final active class predictions"

    def test_probability_decomposition_is_correct(self, trained_classifier):
        """Verify P(class) = P(active) * P(class|active) for active classes."""
        clf, X, y = trained_classifier

        # Get probabilities from both stages
        stage1_proba = clf.stage1_model.predict_proba(X)
        p_active = stage1_proba[:, 1]
        stage2_proba = clf.stage2_model.predict_proba(X)

        # Get final probabilities
        final_proba = clf.predict_proba(X)

        # For each active class, verify: P(class) = P(active) * P(class|active)
        for new_idx, orig_idx in clf._active_label_inverse.items():
            expected = p_active * stage2_proba[:, new_idx]
            actual = final_proba[:, orig_idx]
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8,
                err_msg=f"Probability decomposition failed for class {CLASS_NAMES[orig_idx]}")

    def test_relax_probability_from_stage1(self, trained_classifier):
        """Verify P(Relax) comes directly from Stage 1."""
        clf, X, y = trained_classifier

        stage1_proba = clf.stage1_model.predict_proba(X)
        p_relax_stage1 = stage1_proba[:, 0]

        final_proba = clf.predict_proba(X)
        p_relax_final = final_proba[:, RELAX_IDX]

        np.testing.assert_allclose(p_relax_final, p_relax_stage1, rtol=1e-5, atol=1e-8,
            err_msg="P(Relax) should come directly from Stage 1")

    def test_all_probabilities_sum_to_one(self, trained_classifier):
        """Verify probability decomposition preserves normalization."""
        clf, X, y = trained_classifier

        final_proba = clf.predict_proba(X)
        sums = final_proba.sum(axis=1)

        np.testing.assert_allclose(sums, 1.0, rtol=1e-5, atol=1e-8,
            err_msg="All samples should have probabilities summing to 1.0")

    def test_stage2_only_trained_on_active_samples(self, trained_classifier):
        """Verify Stage 2 was trained only on active (non-Relax) samples."""
        clf, X, y = trained_classifier

        # Stage 2 should have 4 classes (the active classes)
        assert hasattr(clf.stage2_model, 'classes_'), "Stage 2 should be fitted"
        assert len(clf.stage2_model.classes_) == 4, "Stage 2 should have 4 classes"

        # Active label map should have 4 entries
        assert len(clf._active_label_map) == 4
        assert len(clf._active_label_inverse) == 4
