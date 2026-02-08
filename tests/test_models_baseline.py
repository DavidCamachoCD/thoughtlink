"""Tests for baseline classification models."""

import numpy as np
import pytest

from thoughtlink.models.baseline import (
    build_baselines,
    evaluate_model,
    train_and_evaluate,
)


class TestBuildBaselines:
    def test_returns_four_models(self):
        models = build_baselines()
        assert len(models) == 4
        assert set(models.keys()) == {"logreg", "svm_linear", "svm_rbf", "random_forest"}

    def test_models_have_scaler(self):
        models = build_baselines()
        for name, model in models.items():
            assert "scaler" in model.named_steps


class TestEvaluateModel:
    def test_evaluate_returns_expected_keys(self):
        models = build_baselines()
        model = models["logreg"]

        X_train = np.random.randn(100, 42)
        y_train = np.random.randint(0, 5, size=100)
        X_test = np.random.randn(30, 42)
        y_test = np.random.randint(0, 5, size=30)

        model.fit(X_train, y_train)
        results = evaluate_model(model, X_test, y_test)

        assert "accuracy" in results
        assert "kappa" in results
        assert "confusion_matrix" in results
        assert "report" in results

    def test_accuracy_range(self):
        models = build_baselines()
        model = models["logreg"]

        X_train = np.random.randn(200, 42)
        y_train = np.random.randint(0, 5, size=200)
        X_test = np.random.randn(50, 42)
        y_test = np.random.randint(0, 5, size=50)

        model.fit(X_train, y_train)
        results = evaluate_model(model, X_test, y_test)

        assert 0.0 <= results["accuracy"] <= 1.0


class TestTrainAndEvaluate:
    def test_trains_all_models(self):
        models = build_baselines()

        X_train = np.random.randn(200, 42)
        y_train = np.random.randint(0, 5, size=200)
        X_test = np.random.randn(50, 42)
        y_test = np.random.randint(0, 5, size=50)

        results = train_and_evaluate(models, X_train, y_train, X_test, y_test)

        assert set(results.keys()) == set(models.keys())
        for name, res in results.items():
            assert "accuracy" in res

    def test_models_fitted_after_train(self):
        models = build_baselines()

        X_train = np.random.randn(200, 42)
        y_train = np.random.randint(0, 5, size=200)
        X_test = np.random.randn(50, 42)
        y_test = np.random.randint(0, 5, size=50)

        train_and_evaluate(models, X_train, y_train, X_test, y_test)

        for name, model in models.items():
            pred = model.predict(X_test)
            assert len(pred) == len(y_test)
