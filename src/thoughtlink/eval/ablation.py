"""Factorial ablation harness for the calibration + conformal layer.

Declares three registries -- model trainers, calibrators, and conformal
predictors -- plus a `run_factorial` runner that walks the cross-product over
seeds and writes per-cell metrics to a versioned `results/ablations/<run-id>/`
folder. The folder layout is:

    config.json   -- variants requested, git hash, timestamp, library versions
    cells.jsonl   -- one row per (model, calibration, conformal, seed) cell
    summary.csv   -- the same data as a tidy DataFrame
    summary.md    -- markdown table aggregated across seeds (mean, std)

`cells.jsonl` is appended one cell at a time so partial runs are resumable
via `--resume <run-id>`. Each cell carries its own `error` field so a
failure does not abort the whole sweep.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from thoughtlink.data.loader import CLASS_NAMES
from thoughtlink.data.splitter import split_by_subject_3way
from thoughtlink.eval.diagnostics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
)
from thoughtlink.eval.training import (
    TrainConfig,
    train_baseline,
    train_cnn,
    train_hierarchical,
)
from thoughtlink.features.eeg_features import extract_features_from_windows
from thoughtlink.inference.calibration import (
    HierarchicalCalibrator,
    SklearnCalibrator,
    TemperatureScaler,
)
from thoughtlink.inference.conformal import (
    APSConformalPredictor,
    NaiveConformalPredictor,
    WeightedAPSConformalPredictor,
)
from thoughtlink.inference.domain_weights import (
    diagnose_weights,
    estimate_likelihood_ratio,
)
from thoughtlink.models.hierarchical import HierarchicalClassifier
from thoughtlink.preprocessing.eeg import preprocess_all
from thoughtlink.preprocessing.windowing import windows_from_samples


# ---------------------------------------------------------------------------
# Cell schema
# ---------------------------------------------------------------------------


@dataclass
class AblationCell:
    """Result of running one (model, calibration, conformal, seed) combination."""

    seed: int
    model: str
    calibration: str
    conformal: str
    metrics: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    error: str | None = None

    def cell_id(self) -> str:
        return f"{self.seed}/{self.model}/{self.calibration}/{self.conformal}"


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


# Model trainers receive (X_train_features, X_train_windows, y_train,
# train_subj_ids, config) and return (model, train_metrics_dict). The richer
# context lets each trainer pick the inputs it actually needs.
def _train_baseline(X_feat, _X_win, y, _subj, cfg):
    return train_baseline(X_feat, y, cfg)


def _train_hierarchical(X_feat, _X_win, y, _subj, cfg):
    return train_hierarchical(X_feat, y, cfg)


def _train_cnn(_X_feat, X_win, y, subj, cfg):
    return train_cnn(X_win, y, subj, cfg)


MODEL_TRAINERS: dict[str, Callable] = {
    "baseline": _train_baseline,
    "hierarchical": _train_hierarchical,
    "cnn": _train_cnn,
}


# Compatibility: which (model, calibration) pairs are valid.
COMPATIBILITY: dict[str, set[str]] = {
    "baseline": {"raw", "isotonic", "sigmoid"},
    "hierarchical": {"raw", "isotonic", "sigmoid"},
    "cnn": {"raw", "temperature"},
}


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------


def _model_calibrate_and_predict(
    *,
    model,
    model_name: str,
    calibration: str,
    X_calib_feat: np.ndarray,
    X_test_feat: np.ndarray,
    X_calib_windows: np.ndarray,
    X_test_windows: np.ndarray,
    y_calib: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the chosen calibration variant and return (probs_calib, probs_test, info).

    `info` carries calibration-specific metadata (e.g., learned T for
    temperature scaling) that goes into the cell metrics.
    """
    info: dict[str, Any] = {}

    if calibration == "raw":
        if model_name == "cnn":
            probs_calib = model.predict_proba_numpy(X_calib_windows)
            probs_test = model.predict_proba_numpy(X_test_windows)
        else:
            probs_calib = model.predict_proba(X_calib_feat)
            probs_test = model.predict_proba(X_test_feat)
        return probs_calib, probs_test, info

    if calibration == "temperature":
        if model_name != "cnn":
            raise ValueError("temperature scaling is CNN-only in this harness")
        logits_calib = model.predict_proba_numpy(X_calib_windows, return_logits=True)
        logits_test = model.predict_proba_numpy(X_test_windows, return_logits=True)
        scaler = TemperatureScaler().fit(logits_calib, y_calib)
        info["temperature"] = float(scaler.T)
        return scaler.predict_proba(logits_calib), scaler.predict_proba(logits_test), info

    if calibration in ("isotonic", "sigmoid"):
        if isinstance(model, HierarchicalClassifier):
            cal = HierarchicalCalibrator(method=calibration).fit(
                model, X_calib_feat, y_calib
            )
        else:
            cal = SklearnCalibrator(method=calibration).fit(
                model, X_calib_feat, y_calib
            )
        return cal.predict_proba(X_calib_feat), cal.predict_proba(X_test_feat), info

    raise ValueError(f"Unknown calibration variant: {calibration!r}")


def _fit_conformal(
    *,
    conformal: str,
    probs_calib: np.ndarray,
    y_calib: np.ndarray,
    X_calib_feat: np.ndarray,
    X_test_feat: np.ndarray,
    alpha: float,
):
    """Return a fitted conformal predictor or None for `conformal == 'none'`."""
    if conformal == "none":
        return None
    if conformal == "aps":
        return APSConformalPredictor(alpha=alpha).fit(probs_calib, y_calib)
    if conformal == "naive":
        return NaiveConformalPredictor(alpha=alpha).fit(probs_calib, y_calib)
    if conformal == "weighted_aps":
        weights = estimate_likelihood_ratio(X_calib_feat, X_test_feat)
        return WeightedAPSConformalPredictor(alpha=alpha).fit(
            probs_calib, y_calib, weights
        )
    raise ValueError(f"Unknown conformal variant: {conformal!r}")


def _eval_metrics(
    *,
    probs_test: np.ndarray,
    y_test: np.ndarray,
    cp,
    X_calib_feat: np.ndarray,
    X_test_feat: np.ndarray,
    cal_info: dict,
) -> dict:
    """Compute the metrics persisted per cell."""
    preds = probs_test.argmax(axis=1)
    metrics: dict[str, float | int] = {
        "accuracy": float((preds == y_test).mean()),
        "ece": expected_calibration_error(y_test, probs_test),
        "mce": maximum_calibration_error(y_test, probs_test),
        "brier": brier_score(y_test, probs_test),
    }
    metrics.update(cal_info)
    if cp is not None:
        metrics["q_hat"] = float(getattr(cp, "q_hat", float("nan")))
        metrics["coverage"] = cp.empirical_coverage(probs_test, y_test)
        metrics["avg_set_size"] = cp.average_set_size(probs_test)
        sets = cp.predict_set(probs_test)
        metrics["fraction_singleton"] = float(np.mean([len(s) == 1 for s in sets]))
        if isinstance(cp, WeightedAPSConformalPredictor):
            weights = estimate_likelihood_ratio(X_calib_feat, X_test_feat)
            wd = diagnose_weights(weights)
            metrics["ess_ratio"] = wd["ess_ratio"]
            metrics["weight_max"] = wd["max"]
    return metrics


# ---------------------------------------------------------------------------
# Run-id, persistence, resumability
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _make_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    sha = _git_sha() or "nogit"
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{sha}"


def _read_existing_cells(cells_path: Path) -> set[str]:
    if not cells_path.exists():
        return set()
    seen = set()
    with cells_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seen.add(
                f"{obj['seed']}/{obj['model']}/{obj['calibration']}/{obj['conformal']}"
            )
    return seen


def _append_cell(cells_path: Path, cell: AblationCell) -> None:
    with cells_path.open("a") as f:
        f.write(json.dumps(asdict(cell), default=float))
        f.write("\n")


# ---------------------------------------------------------------------------
# Per-seed pipeline + factorial driver
# ---------------------------------------------------------------------------


def _prepare_seed_data(samples, seed: int, n_calib_subjects: int, n_test_subjects: int):
    """Run a fresh subject-aware split + preprocessing + feature extraction."""
    n_subjects = len({s["subject_id"] for s in samples})
    train, calib, test = split_by_subject_3way(
        samples,
        calib_size=n_calib_subjects / n_subjects,
        test_size=n_test_subjects / n_subjects,
        random_state=seed,
    )
    preprocess_all(train)
    preprocess_all(calib)
    preprocess_all(test)

    X_train_w, y_train, train_subj = windows_from_samples(train)
    X_calib_w, y_calib, _ = windows_from_samples(calib)
    X_test_w, y_test, _ = windows_from_samples(test)

    X_train = extract_features_from_windows(X_train_w, include_time_domain=True)
    X_calib = extract_features_from_windows(X_calib_w, include_time_domain=True)
    X_test = extract_features_from_windows(X_test_w, include_time_domain=True)

    return {
        "train": (X_train_w, X_train, y_train, train_subj),
        "calib": (X_calib_w, X_calib, y_calib),
        "test": (X_test_w, X_test, y_test),
    }


def _evaluate_cell(
    *,
    seed: int,
    model_name: str,
    calibration: str,
    conformal: str,
    model,
    data: dict,
    alpha: float,
) -> AblationCell:
    cell = AblationCell(
        seed=seed, model=model_name, calibration=calibration, conformal=conformal
    )
    try:
        X_calib_w, X_calib, y_calib = data["calib"]
        X_test_w, X_test, y_test = data["test"]

        t0 = time.perf_counter()
        probs_calib, probs_test, cal_info = _model_calibrate_and_predict(
            model=model,
            model_name=model_name,
            calibration=calibration,
            X_calib_feat=X_calib,
            X_test_feat=X_test,
            X_calib_windows=X_calib_w,
            X_test_windows=X_test_w,
            y_calib=y_calib,
        )
        t_cal = time.perf_counter() - t0

        t0 = time.perf_counter()
        cp = _fit_conformal(
            conformal=conformal,
            probs_calib=probs_calib,
            y_calib=y_calib,
            X_calib_feat=X_calib,
            X_test_feat=X_test,
            alpha=alpha,
        )
        t_conf = time.perf_counter() - t0

        cell.metrics = _eval_metrics(
            probs_test=probs_test,
            y_test=y_test,
            cp=cp,
            X_calib_feat=X_calib,
            X_test_feat=X_test,
            cal_info=cal_info,
        )
        cell.timing = {"calibrate_s": t_cal, "conformal_s": t_conf}
    except Exception as exc:  # noqa: BLE001  -- intentional: per-cell isolation
        cell.error = f"{type(exc).__name__}: {exc}"
    return cell


def run_factorial(
    *,
    samples: list[dict],
    models: list[str],
    calibrators: list[str],
    conformals: list[str],
    seeds: list[int],
    out_dir: Path,
    alpha: float = 0.1,
    n_calib_subjects: int = 2,
    n_test_subjects: int = 3,
    train_config: TrainConfig | None = None,
    on_cell: Callable[[AblationCell], None] | None = None,
) -> Path:
    """Run the factorial ablation. Returns the path to `summary.csv`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_path = out_dir / "cells.jsonl"
    config_path = out_dir / "config.json"
    base_train_cfg = train_config or TrainConfig()

    if not config_path.exists():
        config = {
            "models": list(models),
            "calibrators": list(calibrators),
            "conformals": list(conformals),
            "seeds": list(seeds),
            "alpha": alpha,
            "n_calib_subjects": n_calib_subjects,
            "n_test_subjects": n_test_subjects,
            "train_config": asdict(base_train_cfg),
            "git_sha": _git_sha(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        config_path.write_text(json.dumps(config, indent=2))

    seen_ids = _read_existing_cells(cells_path)

    for seed in seeds:
        seed_data = None  # lazy: only prepare when at least one cell remains
        per_seed_models: dict[str, Any] = {}

        for model_name in models:
            applicable_cals = [
                c for c in calibrators if c in COMPATIBILITY[model_name]
            ]
            if not applicable_cals:
                continue
            cells_pending = [
                (cal, conf)
                for cal in applicable_cals
                for conf in conformals
                if f"{seed}/{model_name}/{cal}/{conf}" not in seen_ids
            ]
            if not cells_pending:
                continue

            if seed_data is None:
                seed_data = _prepare_seed_data(
                    samples, seed, n_calib_subjects, n_test_subjects
                )

            X_train_w, X_train, y_train, train_subj = seed_data["train"]

            if model_name not in per_seed_models:
                cfg = TrainConfig(**{**asdict(base_train_cfg), "seed": seed})
                t0 = time.perf_counter()
                model, train_metrics = MODEL_TRAINERS[model_name](
                    X_train, X_train_w, y_train, train_subj, cfg
                )
                per_seed_models[model_name] = {
                    "model": model,
                    "train_metrics": train_metrics,
                    "train_seconds": time.perf_counter() - t0,
                }

            entry = per_seed_models[model_name]
            for cal_name, conf_name in cells_pending:
                cell = _evaluate_cell(
                    seed=seed,
                    model_name=model_name,
                    calibration=cal_name,
                    conformal=conf_name,
                    model=entry["model"],
                    data=seed_data,
                    alpha=alpha,
                )
                # Attach training metrics + timing once per cell so each row is self-contained.
                cell.metrics = {**entry["train_metrics"], **cell.metrics}
                cell.timing = {"train_s": entry["train_seconds"], **cell.timing}
                _append_cell(cells_path, cell)
                seen_ids.add(cell.cell_id())
                if on_cell is not None:
                    on_cell(cell)

    return _write_summary(cells_path, out_dir)


def _write_summary(cells_path: Path, out_dir: Path) -> Path:
    """Read cells.jsonl, write summary.csv (raw) + summary.md (aggregated)."""
    import pandas as pd

    rows: list[dict] = []
    if cells_path.exists():
        with cells_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                row = {
                    "seed": obj["seed"],
                    "model": obj["model"],
                    "calibration": obj["calibration"],
                    "conformal": obj["conformal"],
                    "error": obj.get("error"),
                }
                row.update({f"metric_{k}": v for k, v in obj.get("metrics", {}).items()})
                row.update({f"time_{k}": v for k, v in obj.get("timing", {}).items()})
                rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)

    md_path = out_dir / "summary.md"
    if df.empty:
        md_path.write_text("# Ablation summary\n\n(no cells)\n")
        return csv_path

    df_ok = df[df["error"].isna()].copy()
    if df_ok.empty:
        md_path.write_text("# Ablation summary\n\n(all cells errored)\n")
        return csv_path

    metric_cols = [
        c
        for c in df_ok.columns
        if c.startswith("metric_") and c not in {"metric_model_kind"}
    ]
    md_lines = [
        "# Ablation summary",
        "",
        f"`{len(df_ok)}` non-error cells across {df_ok['seed'].nunique()} seeds.",
        "",
        "| model | calibration | conformal | n |"
        + "".join(f" {c[len('metric_'):]} (mean ± std) |" for c in metric_cols),
        "|---|---|---|---|" + "---|" * len(metric_cols),
    ]
    grouped = df_ok.groupby(["model", "calibration", "conformal"], dropna=False)
    for (model, cal, conf), sub in grouped:
        n = len(sub)
        cells = [f"| {model} | {cal} | {conf} | {n} |"]
        for col in metric_cols:
            mean = sub[col].mean()
            std = sub[col].std() if n > 1 else 0.0
            cells.append(f" {mean:.3f} ± {std:.3f} |")
        md_lines.append("".join(cells))
    md_path.write_text("\n".join(md_lines) + "\n")
    return csv_path


# ---------------------------------------------------------------------------
# Public surface for tests / external callers
# ---------------------------------------------------------------------------


__all__ = [
    "AblationCell",
    "COMPATIBILITY",
    "MODEL_TRAINERS",
    "run_factorial",
]
