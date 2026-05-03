"""Deprecated location for diagnostics. Re-exports from `thoughtlink.eval.diagnostics`.

The canonical home is now `thoughtlink.eval.diagnostics`. This shim exists so
external imports keep working while we migrate. Will be removed in a future
release.
"""

from __future__ import annotations

import warnings

from thoughtlink.eval.diagnostics import (  # noqa: F401
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_curve,
)

warnings.warn(
    "thoughtlink.inference.diagnostics is deprecated; "
    "import from thoughtlink.eval.diagnostics instead.",
    DeprecationWarning,
    stacklevel=2,
)
