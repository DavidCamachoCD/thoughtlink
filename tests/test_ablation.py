"""Tests for the factorial ablation harness (eval/ablation.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thoughtlink.eval.ablation import (
    COMPATIBILITY,
    MODEL_TRAINERS,
    AblationCell,
    _append_cell,
    _make_run_id,
    _read_existing_cells,
    _write_summary,
)


class TestRegistries:
    def test_model_trainers_keys(self):
        assert set(MODEL_TRAINERS) == {"baseline", "hierarchical", "cnn"}

    def test_compatibility_matrix(self):
        assert COMPATIBILITY["baseline"] == {"raw", "isotonic", "sigmoid"}
        assert COMPATIBILITY["hierarchical"] == {"raw", "isotonic", "sigmoid"}
        assert COMPATIBILITY["cnn"] == {"raw", "temperature"}

    def test_compatibility_keys_match_trainers(self):
        assert set(COMPATIBILITY) == set(MODEL_TRAINERS)


class TestAblationCell:
    def test_cell_id_format(self):
        cell = AblationCell(
            seed=42, model="hierarchical", calibration="sigmoid", conformal="aps"
        )
        assert cell.cell_id() == "42/hierarchical/sigmoid/aps"

    def test_default_metrics_and_timing_are_empty_dicts(self):
        cell = AblationCell(seed=0, model="baseline", calibration="raw", conformal="none")
        assert cell.metrics == {}
        assert cell.timing == {}
        assert cell.error is None

    def test_error_field_set(self):
        cell = AblationCell(
            seed=0,
            model="cnn",
            calibration="isotonic",
            conformal="aps",
            error="ValueError: bad combo",
        )
        assert "ValueError" in cell.error


class TestPersistence:
    def test_run_id_format(self):
        run_id = _make_run_id()
        # YYYYMMDD-HHMMSS-<sha-or-nogit>
        parts = run_id.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 8 and parts[0].isdigit()
        assert len(parts[1]) == 6 and parts[1].isdigit()
        assert len(parts[2]) >= 4

    def test_append_and_read_roundtrip(self, tmp_path: Path):
        cells_path = tmp_path / "cells.jsonl"
        cells = [
            AblationCell(seed=0, model="baseline", calibration="raw", conformal="none"),
            AblationCell(seed=0, model="baseline", calibration="sigmoid", conformal="aps"),
        ]
        for c in cells:
            _append_cell(cells_path, c)
        seen = _read_existing_cells(cells_path)
        assert seen == {c.cell_id() for c in cells}

    def test_read_existing_on_missing_file_is_empty(self, tmp_path: Path):
        assert _read_existing_cells(tmp_path / "missing.jsonl") == set()


class TestSummary:
    def _make_jsonl(self, tmp_path: Path) -> Path:
        cells_path = tmp_path / "cells.jsonl"
        rows = [
            AblationCell(
                seed=0, model="baseline", calibration="raw", conformal="none",
                metrics={"accuracy": 0.30, "ece": 0.05}, timing={"train_s": 1.0},
            ),
            AblationCell(
                seed=0, model="baseline", calibration="sigmoid", conformal="aps",
                metrics={"accuracy": 0.30, "ece": 0.20, "coverage": 0.71},
                timing={"train_s": 1.0, "calibrate_s": 0.05},
            ),
            AblationCell(
                seed=0, model="hierarchical", calibration="raw", conformal="none",
                error="RuntimeError: training failed",
            ),
        ]
        for c in rows:
            _append_cell(cells_path, c)
        return cells_path

    def test_writes_csv_and_md(self, tmp_path: Path):
        cells_path = self._make_jsonl(tmp_path)
        summary_csv = _write_summary(cells_path, tmp_path)
        assert summary_csv == tmp_path / "summary.csv"
        assert summary_csv.exists()
        assert (tmp_path / "summary.md").exists()

    def test_csv_has_one_row_per_cell(self, tmp_path: Path):
        import pandas as pd

        cells_path = self._make_jsonl(tmp_path)
        _write_summary(cells_path, tmp_path)
        df = pd.read_csv(tmp_path / "summary.csv")
        assert len(df) == 3
        assert set(df["model"].unique()) == {"baseline", "hierarchical"}
        # Errored cells preserve the error string.
        err_row = df[df["error"].notna()].iloc[0]
        assert "RuntimeError" in err_row["error"]

    def test_md_aggregates_only_non_error_cells(self, tmp_path: Path):
        cells_path = self._make_jsonl(tmp_path)
        _write_summary(cells_path, tmp_path)
        md = (tmp_path / "summary.md").read_text()
        assert "Ablation summary" in md
        # 2 non-error cells are expected.
        assert "`2` non-error cells" in md

    def test_empty_cells_jsonl_produces_empty_summary(self, tmp_path: Path):
        cells_path = tmp_path / "cells.jsonl"
        cells_path.touch()
        _write_summary(cells_path, tmp_path)
        assert "(no cells)" in (tmp_path / "summary.md").read_text()
