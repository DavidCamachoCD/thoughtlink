"""Run a factorial ablation over (model, calibration, conformal, seed).

Default = full factorial:
    models       = baseline, hierarchical, cnn
    calibrators  = raw, isotonic, sigmoid, temperature   (filtered per model)
    conformals   = none, aps, naive, weighted_aps
    seeds        = 42, 123, 456

Usage:
    uv run python scripts/run_ablation.py                        # full factorial
    uv run python scripts/run_ablation.py --models baseline      # just baseline
    uv run python scripts/run_ablation.py --resume 20260501-...  # continue a run

Outputs live in `results/ablations/<run-id>/`:
    config.json  cells.jsonl  summary.csv  summary.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thoughtlink.data.loader import load_all  # noqa: E402
from thoughtlink.eval.ablation import (  # noqa: E402
    AblationCell,
    _make_run_id,
    run_factorial,
)


DEFAULT_MODELS = ["baseline", "hierarchical", "cnn"]
DEFAULT_CALIBRATORS = ["raw", "isotonic", "sigmoid", "temperature"]
DEFAULT_CONFORMALS = ["none", "aps", "naive", "weighted_aps"]
DEFAULT_SEEDS = [42, 123, 456]


def _print_cell(cell: AblationCell) -> None:
    if cell.error is not None:
        print(
            f"[seed={cell.seed} {cell.model}/{cell.calibration}/{cell.conformal}] "
            f"ERROR: {cell.error}"
        )
        return
    m = cell.metrics
    parts = [
        f"acc={m.get('accuracy', float('nan')):.3f}",
        f"ECE={m.get('ece', float('nan')):.3f}",
    ]
    if "coverage" in m:
        parts.append(f"cov={m['coverage']:.3f}")
        parts.append(f"|set|={m['avg_set_size']:.2f}")
    if "temperature" in m:
        parts.append(f"T={m['temperature']:.2f}")
    print(
        f"[seed={cell.seed} {cell.model}/{cell.calibration}/{cell.conformal}] "
        + "  ".join(parts)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Factorial ablation runner for the calibration + conformal layer."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--calibrators", nargs="+", default=DEFAULT_CALIBRATORS)
    parser.add_argument("--conformals", nargs="+", default=DEFAULT_CONFORMALS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n-calib-subjects", type=int, default=2)
    parser.add_argument("--n-test-subjects", type=int, default=3)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to results/ablations/<auto-run-id>/.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Run id (folder name under results/ablations/) to resume.",
    )
    args = parser.parse_args()

    if args.resume is not None:
        out_dir = Path("results/ablations") / args.resume
        if not out_dir.exists():
            raise SystemExit(f"--resume target not found: {out_dir}")
    elif args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = Path("results/ablations") / _make_run_id()

    print("=" * 70)
    print("ThoughtLink - Factorial Ablation")
    print("=" * 70)
    print(f"Run id:   {out_dir.name}")
    print(f"Out dir:  {out_dir}")
    print(f"Models:   {args.models}")
    print(f"Calib:    {args.calibrators}")
    print(f"Conf:     {args.conformals}")
    print(f"Seeds:    {args.seeds}")
    print(f"alpha:    {args.alpha}")
    print(f"calib subj / test subj: {args.n_calib_subjects} / {args.n_test_subjects}")
    print("-" * 70)

    print("Loading dataset...")
    samples = load_all()
    print(f"Loaded {len(samples)} samples")
    print("-" * 70)

    summary_csv = run_factorial(
        samples=samples,
        models=args.models,
        calibrators=args.calibrators,
        conformals=args.conformals,
        seeds=args.seeds,
        out_dir=out_dir,
        alpha=args.alpha,
        n_calib_subjects=args.n_calib_subjects,
        n_test_subjects=args.n_test_subjects,
        on_cell=_print_cell,
    )
    print("-" * 70)
    print(f"Wrote {summary_csv}")
    print(f"      {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
