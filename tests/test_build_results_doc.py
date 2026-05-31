"""Tests for scripts/build_results_doc.py: idempotency, marker preservation,
graceful handling of missing artifacts, and orphan-marker detection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/ is not a package; add it to sys.path so we can import the module.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import build_results_doc as brd  # type: ignore  # noqa: E402


def _write_minimal_results(results_dir: Path) -> None:
    """Materialise a tiny but valid results/ tree the renderers can consume."""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "analysis_summary.json").write_text(json.dumps({
        "dataset": {
            "total_samples": 10, "train_samples": 8, "test_samples": 2,
            "train_windows": 80, "test_windows": 20, "n_features": 4,
            "train_subjects": ["a", "b"], "test_subjects": ["c"],
        },
        "models": {
            "logreg": {"accuracy": 0.5, "kappa": 0.1, "f1_macro": 0.4, "latency_ms": 0.1},
        },
        "hierarchical_details": {"missed_active_rate": 0.3},
    }))
    (results_dir / "baseline_results.json").write_text(json.dumps({
        "binary": {"logreg": {"accuracy": 0.6, "kappa": 0.2}},
        "multiclass": {"logreg": {"accuracy": 0.3, "kappa": 0.05}},
    }))
    (results_dir / "hierarchical_results.json").write_text(json.dumps({
        "accuracy": 0.4, "kappa": 0.1,
        "stage1_accuracy": 0.7, "false_trigger_rate": 0.3,
    }))


def _make_doc(*sections: str) -> str:
    """Build a minimal markdown doc with the requested marker sections."""
    parts = ["# Heading\n\nBefore prose.\n"]
    for name in sections:
        parts.append(f"\n<!-- BEGIN: {name} -->\n<!-- END: {name} -->\n")
    parts.append("\nAfter prose.\n")
    return "".join(parts)


def test_regenerate_is_idempotent(tmp_path: Path):
    results = tmp_path / "results"
    _write_minimal_results(results)
    ctx = brd.Context(results_dir=results, ablation_dir=None)

    doc = _make_doc("dataset_summary", "hierarchical_table")
    once = brd.regenerate(doc, ctx)
    twice = brd.regenerate(once, ctx)
    assert once == twice, "Second regeneration must be a no-op."
    # The body should now contain real values (not just empty markers).
    assert "Total samples" in once
    assert "Stage 1 accuracy" in once


def test_preserves_unmarked_prose(tmp_path: Path):
    results = tmp_path / "results"
    _write_minimal_results(results)
    ctx = brd.Context(results_dir=results, ablation_dir=None)

    doc = _make_doc("dataset_summary")
    new = brd.regenerate(doc, ctx)
    # Surrounding prose untouched.
    assert "# Heading" in new
    assert "Before prose." in new
    assert "After prose." in new


def test_handles_missing_artifact_gracefully(tmp_path: Path):
    """An empty results/ dir should yield 'missing' notices, not raise."""
    results = tmp_path / "results"
    results.mkdir()
    ctx = brd.Context(results_dir=results, ablation_dir=None)

    doc = _make_doc("dataset_summary", "calibration_table", "ablation_summary")
    new = brd.regenerate(doc, ctx)
    assert "artifact missing" in new
    # All three markers remain intact.
    assert new.count("<!-- BEGIN: ") == 3
    assert new.count("<!-- END: ") == 3


def test_unknown_marker_is_left_alone(tmp_path: Path):
    """A marker name not in the renderer registry must not be touched."""
    results = tmp_path / "results"
    _write_minimal_results(results)
    ctx = brd.Context(results_dir=results, ablation_dir=None)

    doc = (
        "<!-- BEGIN: not_a_real_section -->\n"
        "user-written content\n"
        "<!-- END: not_a_real_section -->\n"
    )
    out = brd.regenerate(doc, ctx)
    assert out == doc


def test_orphan_marker_raises(tmp_path: Path):
    results = tmp_path / "results"
    _write_minimal_results(results)
    ctx = brd.Context(results_dir=results, ablation_dir=None)

    doc = "<!-- BEGIN: dataset_summary -->\nno end marker\n"
    with pytest.raises(ValueError, match="Marker mismatch"):
        brd.regenerate(doc, ctx)


def test_check_mode_returns_nonzero_when_stale(tmp_path: Path):
    """`--check` should exit 1 when the doc would change."""
    results = tmp_path / "results"
    _write_minimal_results(results)
    doc_path = tmp_path / "results.md"
    doc_path.write_text(_make_doc("dataset_summary"))

    rc = brd.main([
        "--doc", str(doc_path),
        "--results-dir", str(results),
        "--check",
    ])
    assert rc == 1, "Stale doc must trigger non-zero exit in --check mode."


def test_check_mode_returns_zero_when_in_sync(tmp_path: Path):
    results = tmp_path / "results"
    _write_minimal_results(results)
    doc_path = tmp_path / "results.md"
    doc_path.write_text(_make_doc("dataset_summary"))

    # Populate, then verify --check is happy on the second pass.
    rc = brd.main(["--doc", str(doc_path), "--results-dir", str(results)])
    assert rc == 0
    rc = brd.main([
        "--doc", str(doc_path),
        "--results-dir", str(results),
        "--check",
    ])
    assert rc == 0
