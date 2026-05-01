"""Fit calibration + conformal predictors on top of trained ThoughtLink models.

Reads trained artifacts from `results/` and fits, for each, the appropriate
calibrator (isotonic for sklearn pipelines, hierarchical-aware for the 2-stage
classifier, temperature scaling for the CNN) and an APS conformal predictor.
Calibration data is drawn from a held-out subject via
`split_by_subject_3way` (default 13/1/3 of 17 subjects).

Outputs (under `results/`):
    *_calibrated.pkl    -- fitted calibrator (drop-in for `predict_proba`)
    *_conformal.pkl     -- fitted APS predictor (provides `.predict_set`)
    calibration_report.json  -- ECE / Brier / coverage before vs after

References:
- Naeini et al. 2015 (ECE), Guo et al. 2017 (temperature scaling),
  Romano et al. 2020 (APS), Angelopoulos & Bates 2023 (conformal).
  See `docs/references.md`.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thoughtlink.data.loader import CLASS_NAMES, load_all
from thoughtlink.data.splitter import split_by_subject_3way
from thoughtlink.features.eeg_features import extract_features_from_windows
from thoughtlink.inference.calibration import (
    HierarchicalCalibrator,
    SklearnCalibrator,
    TemperatureScaler,
)
from thoughtlink.inference.conformal import (
    APSConformalPredictor,
    WeightedAPSConformalPredictor,
)
from thoughtlink.inference.domain_weights import (
    diagnose_weights,
    estimate_likelihood_ratio,
)
from thoughtlink.inference.diagnostics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
)
from thoughtlink.models.hierarchical import HierarchicalClassifier
from thoughtlink.preprocessing.eeg import preprocess_all
from thoughtlink.preprocessing.windowing import windows_from_samples


RESULTS_DIR = Path("results")
CONFIG_PATH = Path("configs/default.yaml")


def _diagnostics(name: str, y: np.ndarray, probs: np.ndarray) -> dict:
    return {
        "name": name,
        "ece": expected_calibration_error(y, probs),
        "mce": maximum_calibration_error(y, probs),
        "brier": brier_score(y, probs),
    }


def _print_diag(label: str, diag: dict) -> None:
    print(
        f"  {label:<24} ECE={diag['ece']:.4f}  MCE={diag['mce']:.4f}  "
        f"Brier={diag['brier']:.4f}"
    )


def _fit_conformal_and_report(
    name: str,
    probs_calib: np.ndarray,
    y_calib: np.ndarray,
    probs_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    *,
    X_calib: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    fit_weighted: bool = False,
) -> tuple[APSConformalPredictor, WeightedAPSConformalPredictor | None, dict]:
    cp = APSConformalPredictor(alpha=alpha).fit(probs_calib, y_calib)
    coverage = cp.empirical_coverage(probs_test, y_test)
    avg_size = cp.average_set_size(probs_test)
    print(
        f"  Conformal (alpha={alpha:.2f})           coverage={coverage:.3f}  "
        f"avg_set_size={avg_size:.2f}  q_hat={cp.q_hat:.4f}"
    )
    out = {
        "alpha": alpha,
        "q_hat": cp.q_hat,
        "empirical_coverage": coverage,
        "avg_set_size": avg_size,
    }

    weighted_cp: WeightedAPSConformalPredictor | None = None
    if fit_weighted:
        if X_calib is None or X_test is None:
            raise ValueError("fit_weighted=True requires X_calib and X_test")
        weights = estimate_likelihood_ratio(X_calib, X_test)
        wdiag = diagnose_weights(weights)
        weighted_cp = WeightedAPSConformalPredictor(alpha=alpha).fit(
            probs_calib, y_calib, weights
        )
        wcov = weighted_cp.empirical_coverage(probs_test, y_test)
        wavg = weighted_cp.average_set_size(probs_test)
        print(
            f"  Weighted conformal (alpha={alpha:.2f})  coverage={wcov:.3f}  "
            f"avg_set_size={wavg:.2f}  q_hat={weighted_cp.q_hat:.4f}  "
            f"ESS={wdiag['ess_ratio']:.2f}"
        )
        out["weighted"] = {
            "q_hat": weighted_cp.q_hat,
            "empirical_coverage": wcov,
            "avg_set_size": wavg,
            "weight_stats": wdiag,
        }
    return cp, weighted_cp, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration + conformal pipeline.")
    parser.add_argument(
        "--method",
        choices=["isotonic", "sigmoid"],
        default=None,
        help="Sklearn calibration method (overrides configs/default.yaml).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Conformal coverage level (overrides configs/default.yaml).",
    )
    parser.add_argument(
        "--n-calib-subjects",
        type=int,
        default=1,
        help="Number of held-out calibration subjects (default 1; try 2 if the "
             "calibration set is too small for isotonic to be stable).",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Also fit a weighted-conformal predictor (Tibshirani et al. 2019) "
             "to recover marginal coverage under cross-subject covariate shift.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ThoughtLink - Calibration + Conformal Pipeline")
    print("=" * 70)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    cal_cfg = config.get("calibration", {})
    conf_cfg = config.get("conformal", {})
    split_cfg = config["data"].get("split_3way", {})
    sklearn_method = args.method or cal_cfg.get("sklearn_method", "isotonic")
    alpha = args.alpha if args.alpha is not None else conf_cfg.get("alpha", 0.1)
    print(f"Method: {sklearn_method}    alpha={alpha}    "
          f"calibration subjects={args.n_calib_subjects}")

    # 1. Load + 3-way subject-aware split.
    print("\n[1/4] Loading dataset...")
    samples = load_all()

    print("\n[2/4] Splitting (subject-aware) train / calib / test...")
    n_subjects = len({s["subject_id"] for s in samples})
    train_samples, calib_samples, test_samples = split_by_subject_3way(
        samples,
        calib_size=args.n_calib_subjects / n_subjects,
        test_size=split_cfg.get("test_size", 3 / 17),
    )

    print("\n[3/4] Preprocessing + feature extraction...")
    preprocess_all(calib_samples)
    preprocess_all(test_samples)

    X_calib_windows, y_calib, _ = windows_from_samples(calib_samples)
    X_test_windows, y_test, _ = windows_from_samples(test_samples)

    X_calib = extract_features_from_windows(X_calib_windows, include_time_domain=True)
    X_test = extract_features_from_windows(X_test_windows, include_time_domain=True)
    print(f"  calib features: {X_calib.shape}, test features: {X_test.shape}")

    report: dict = {
        "split": {
            "calib_size": int(len(y_calib)),
            "test_size": int(len(y_test)),
            "n_classes": len(CLASS_NAMES),
        },
        "calibration_method": sklearn_method,
        "conformal_alpha": alpha,
        "models": {},
    }

    print("\n[4/4] Fitting calibrators per model...")

    # ----------------------------------------------------------------------
    # Baseline sklearn classifier (best_baseline.pkl)
    # ----------------------------------------------------------------------
    baseline_path = RESULTS_DIR / "best_baseline.pkl"
    if baseline_path.exists():
        print("\n--- best_baseline ---")
        with open(baseline_path, "rb") as f:
            baseline = pickle.load(f)

        probs_test_pre = baseline.predict_proba(X_test)
        diag_pre = _diagnostics("best_baseline_pre", y_test, probs_test_pre)
        _print_diag("Pre  (test)", diag_pre)

        cal = SklearnCalibrator(method=sklearn_method).fit(baseline, X_calib, y_calib)
        probs_test_post = cal.predict_proba(X_test)
        diag_post = _diagnostics("best_baseline_post", y_test, probs_test_post)
        _print_diag("Post (test)", diag_post)

        probs_calib_post = cal.predict_proba(X_calib)
        cp, weighted_cp, conf_report = _fit_conformal_and_report(
            "best_baseline",
            probs_calib_post, y_calib, probs_test_post, y_test, alpha,
            X_calib=X_calib, X_test=X_test, fit_weighted=args.weighted,
        )

        with open(RESULTS_DIR / "best_baseline_calibrated.pkl", "wb") as f:
            pickle.dump(cal, f)
        with open(RESULTS_DIR / "best_baseline_conformal.pkl", "wb") as f:
            pickle.dump(cp, f)
        if weighted_cp is not None:
            with open(RESULTS_DIR / "best_baseline_conformal_weighted.pkl", "wb") as f:
                pickle.dump(weighted_cp, f)

        report["models"]["best_baseline"] = {
            "pre": diag_pre,
            "post": diag_post,
            "conformal": conf_report,
        }
    else:
        print(f"\n[skip] {baseline_path} not found")

    # ----------------------------------------------------------------------
    # Hierarchical classifier (hierarchical_model.pkl)
    # ----------------------------------------------------------------------
    hier_path = RESULTS_DIR / "hierarchical_model.pkl"
    if hier_path.exists():
        print("\n--- hierarchical ---")
        with open(hier_path, "rb") as f:
            hier: HierarchicalClassifier = pickle.load(f)

        probs_test_pre = hier.predict_proba(X_test)
        diag_pre = _diagnostics("hierarchical_pre", y_test, probs_test_pre)
        _print_diag("Pre  (test)", diag_pre)

        cal = HierarchicalCalibrator(method=sklearn_method).fit(hier, X_calib, y_calib)
        probs_test_post = cal.predict_proba(X_test)
        diag_post = _diagnostics("hierarchical_post", y_test, probs_test_post)
        _print_diag("Post (test)", diag_post)

        probs_calib_post = cal.predict_proba(X_calib)
        cp, weighted_cp, conf_report = _fit_conformal_and_report(
            "hierarchical",
            probs_calib_post, y_calib, probs_test_post, y_test, alpha,
            X_calib=X_calib, X_test=X_test, fit_weighted=args.weighted,
        )

        with open(RESULTS_DIR / "hierarchical_calibrated.pkl", "wb") as f:
            pickle.dump(cal, f)
        with open(RESULTS_DIR / "hierarchical_conformal.pkl", "wb") as f:
            pickle.dump(cp, f)
        if weighted_cp is not None:
            with open(RESULTS_DIR / "hierarchical_conformal_weighted.pkl", "wb") as f:
                pickle.dump(weighted_cp, f)

        report["models"]["hierarchical"] = {
            "pre": diag_pre,
            "post": diag_post,
            "conformal": conf_report,
        }
    else:
        print(f"\n[skip] {hier_path} not found")

    # ----------------------------------------------------------------------
    # CNN (cnn_best.pt / cnn_model.pt) -- temperature scaling
    # ----------------------------------------------------------------------
    cnn_path = next(
        (p for p in [RESULTS_DIR / "cnn_best.pt", RESULTS_DIR / "cnn_model.pt"] if p.exists()),
        None,
    )
    if cnn_path is not None:
        try:
            import torch

            from thoughtlink.models.cnn import EEGNet

            print(f"\n--- CNN ({cnn_path.name}) ---")
            # Match `scripts/train_cnn.py` -- it uses f1=16, f2=32, d=2.
            cnn = EEGNet(
                n_classes=len(CLASS_NAMES),
                n_channels=X_calib_windows.shape[2],
                n_samples=X_calib_windows.shape[1],
                f1=16,
                f2=32,
                d=2,
                dropout=0.3,
            )
            cnn.load_state_dict(torch.load(cnn_path, map_location="cpu", weights_only=True))
            cnn.eval()

            logits_calib = cnn.predict_proba_numpy(X_calib_windows, return_logits=True)
            probs_test_pre = cnn.predict_proba_numpy(X_test_windows)
            diag_pre = _diagnostics("cnn_pre", y_test, probs_test_pre)
            _print_diag("Pre  (test)", diag_pre)

            scaler = TemperatureScaler().fit(logits_calib, y_calib)
            print(f"  Learned T = {scaler.T:.3f}")

            logits_test = cnn.predict_proba_numpy(X_test_windows, return_logits=True)
            probs_test_post = scaler.predict_proba(logits_test)
            diag_post = _diagnostics("cnn_post", y_test, probs_test_post)
            _print_diag("Post (test)", diag_post)

            probs_calib_post = scaler.predict_proba(logits_calib)
            # The likelihood ratio P_test(x) / P_calib(x) is invariant to the
            # choice of representation (Jacobian cancels), so we can reuse the
            # 66-dim features for the CNN's weights -- they carry the same
            # subject-discriminative information as the raw windows.
            cp, weighted_cp, conf_report = _fit_conformal_and_report(
                "cnn",
                probs_calib_post, y_calib, probs_test_post, y_test, alpha,
                X_calib=X_calib, X_test=X_test, fit_weighted=args.weighted,
            )

            with open(RESULTS_DIR / "cnn_calibrated.pkl", "wb") as f:
                pickle.dump({"temperature": scaler.T, "scaler": scaler}, f)
            with open(RESULTS_DIR / "cnn_conformal.pkl", "wb") as f:
                pickle.dump(cp, f)
            if weighted_cp is not None:
                with open(RESULTS_DIR / "cnn_conformal_weighted.pkl", "wb") as f:
                    pickle.dump(weighted_cp, f)

            report["models"]["cnn"] = {
                "temperature": scaler.T,
                "pre": diag_pre,
                "post": diag_post,
                "conformal": conf_report,
            }
        except Exception as exc:
            print(f"[skip] CNN calibration failed: {exc}")
    else:
        print("\n[skip] no CNN checkpoint found in results/")

    out = RESULTS_DIR / "calibration_report.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
