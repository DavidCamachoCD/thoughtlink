"""Baseline classification models using sklearn pipelines."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
import numpy as np


def build_baselines() -> dict[str, Pipeline]:
    """Build sklearn pipeline baselines.

    Returns:
        Dict mapping model name to sklearn Pipeline with predict_proba support.
    """
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]),
        "svm_linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1.0, probability=True, random_state=42)),
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42,
            )),
        ]),
    }


def evaluate_model(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Evaluate a trained model on test data.

    Args:
        model: Trained sklearn Pipeline.
        X_test: Test features.
        y_test: Test labels (integer).
        class_names: Optional class name mapping.

    Returns:
        Dict with accuracy, kappa, confusion_matrix, and classification_report.
    """
    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "kappa": cohen_kappa_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=True,
        ),
    }
    return results


def train_and_evaluate(
    models: dict[str, Pipeline],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, dict]:
    """Train and evaluate all baseline models.

    Returns:
        Dict mapping model name to evaluation results.
    """
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        res = evaluate_model(model, X_test, y_test, class_names)
        print(f"  {name}: accuracy={res['accuracy']:.3f}, kappa={res['kappa']:.3f}")
        results[name] = res

    return results
