"""Regenerate marked sections of docs/results.md from canonical artifacts.

Reads the JSON files in `results/` and the latest `results/ablations/<run>/summary.csv`
(or an explicit override), then rewrites every

    <!-- BEGIN: <name> -->
    ...
    <!-- END: <name> -->

block in the target Markdown file with freshly-rendered tables. Sections whose
source artifact is missing are replaced with a short notice instead of crashing,
so the doc stays renderable even if an upstream training step is pending.

Usage:
    uv run python scripts/build_results_doc.py
    uv run python scripts/build_results_doc.py --doc PATH --ablation-run NAME
    uv run python scripts/build_results_doc.py --check        # exit 1 if changed
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_DOC = REPO_ROOT / "docs" / "results.md"

MARKER_RE = re.compile(
    r"(<!-- BEGIN: (\w+) -->)(.*?)(<!-- END: \2 -->)",
    re.DOTALL,
)
BEGIN_RE = re.compile(r"<!-- BEGIN: (\w+) -->")
END_RE = re.compile(r"<!-- END: (\w+) -->")


@dataclass
class Context:
    results_dir: Path
    ablation_dir: Path | None


def _fmt(x, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "—"
        return f"{x:.{digits}f}"
    return str(x)


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _missing(source: str) -> str:
    return f"\n_(artifact missing: `{source}` not found.)_\n"


def _wrap(lines: list[str]) -> str:
    return "\n" + "\n".join(lines) + "\n"


def _find_latest_ablation(results_dir: Path) -> Path | None:
    abl = results_dir / "ablations"
    if not abl.exists():
        return None
    dirs = sorted(d for d in abl.iterdir() if d.is_dir())
    return dirs[-1] if dirs else None


# --- Renderers --------------------------------------------------------------


def render_dataset_summary(ctx: Context) -> str:
    data = _load_json(ctx.results_dir / "analysis_summary.json")
    if data is None:
        return _missing("results/analysis_summary.json")
    ds = data["dataset"]
    rows = [
        ("Total samples (.npz files)", ds.get("total_samples")),
        ("Train samples", ds.get("train_samples")),
        ("Test samples", ds.get("test_samples")),
        ("Train windows", ds.get("train_windows")),
        ("Test windows", ds.get("test_windows")),
        ("# features (standard set)", ds.get("n_features")),
        ("# train subjects", len(ds.get("train_subjects", []))),
        ("# test subjects", len(ds.get("test_subjects", []))),
    ]
    out = ["", "| Property | Value |", "|---|---:|"]
    out += [f"| {k} | {_fmt(v, 0)} |" for k, v in rows]
    return _wrap(out)


def render_baseline_table(ctx: Context) -> str:
    summary = _load_json(ctx.results_dir / "analysis_summary.json")
    binary = _load_json(ctx.results_dir / "baseline_results.json")
    if summary is None or binary is None:
        return _missing("results/analysis_summary.json or baseline_results.json")
    out = [
        "",
        "Multiclass test-set metrics (5 classes, hold-out subject):",
        "",
        "| Model | Accuracy | Kappa | F1 macro | Latency (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["logreg", "svm_linear", "svm_rbf", "random_forest"]:
        m = summary["models"].get(name, {})
        out.append(
            f"| {name} | {_fmt(m.get('accuracy'))} | {_fmt(m.get('kappa'))} "
            f"| {_fmt(m.get('f1_macro'))} | {_fmt(m.get('latency_ms'))} |"
        )
    out += [
        "",
        "Binary rest-vs-active accuracy (sanity check):",
        "",
        "| Model | Accuracy | Kappa |",
        "|---|---:|---:|",
    ]
    for name, m in binary["binary"].items():
        out.append(f"| {name} | {_fmt(m['accuracy'])} | {_fmt(m['kappa'])} |")
    return _wrap(out)


def render_hierarchical_table(ctx: Context) -> str:
    h = _load_json(ctx.results_dir / "hierarchical_results.json")
    summary = _load_json(ctx.results_dir / "analysis_summary.json")
    if h is None:
        return _missing("results/hierarchical_results.json")
    out = ["", "| Metric | Value |", "|---|---:|"]
    out.append(f"| Stage 1 accuracy (rest vs active) | {_fmt(h.get('stage1_accuracy'))} |")
    out.append(f"| Full pipeline accuracy (5 classes) | {_fmt(h.get('accuracy'))} |")
    out.append(f"| Cohen's kappa | {_fmt(h.get('kappa'))} |")
    out.append(f"| False-trigger rate | {_fmt(h.get('false_trigger_rate'))} |")
    if summary and "hierarchical_details" in summary:
        d = summary["hierarchical_details"]
        out.append(
            f"| Missed-active rate (analysis split) | {_fmt(d.get('missed_active_rate'))} |"
        )
    return _wrap(out)


def render_cnn_table(ctx: Context) -> str:
    c = _load_json(ctx.results_dir / "cnn_results.json")
    if c is None:
        return _missing("results/cnn_results.json")
    out = ["", "**Headline metrics**", "", "| Metric | Value |", "|---|---:|"]
    out.append(f"| Accuracy | {_fmt(c.get('accuracy'))} |")
    out.append(f"| Kappa | {_fmt(c.get('kappa'))} |")
    out.append(f"| # parameters | {_fmt(c.get('n_params'), 0)} |")
    out.append(f"| Epochs trained | {_fmt(c.get('n_epochs'), 0)} |")
    out.append(f"| Best epoch accuracy | {_fmt(c.get('best_epoch_acc'))} |")
    rep = c.get("report", {})
    classes = ["Right Fist", "Left Fist", "Both Fists", "Tongue Tapping", "Relax"]
    out += [
        "",
        "**Per-class** (test set)",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls in classes:
        m = rep.get(cls, {})
        out.append(
            f"| {cls} | {_fmt(m.get('precision'))} | {_fmt(m.get('recall'))} "
            f"| {_fmt(m.get('f1-score'))} | {_fmt(m.get('support'), 0)} |"
        )
    return _wrap(out)


def render_wavelet_table(ctx: Context) -> str:
    cmp = _load_json(ctx.results_dir / "wavelet_vs_standard_comparison.json")
    if cmp is None:
        return _missing("results/wavelet_vs_standard_comparison.json")
    out = [
        "",
        f"Standard set: {cmp['n_features']['standard']} features. "
        f"Wavelet set: {cmp['n_features']['wavelet']} features (DWT-augmented).",
        "",
        "| Model | Standard acc | Wavelet acc | Δ acc | Standard F1 | Wavelet F1 | Δ F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["logreg", "svm_linear", "svm_rbf", "random_forest", "hierarchical"]:
        s = cmp["standard"].get(name, {})
        w = cmp["wavelet"].get(name, {})
        if not s or not w:
            continue
        d_acc = w["acc"] - s["acc"]
        d_f1 = w["f1"] - s["f1"]
        out.append(
            f"| {name} | {_fmt(s['acc'])} | {_fmt(w['acc'])} | {_fmt(d_acc)} "
            f"| {_fmt(s['f1'])} | {_fmt(w['f1'])} | {_fmt(d_f1)} |"
        )
    return _wrap(out)


def render_calibration_table(ctx: Context) -> str:
    rep = _load_json(ctx.results_dir / "calibration_report.json")
    if rep is None:
        return _missing("results/calibration_report.json")
    method = rep.get("calibration_method", "?")
    alpha = rep.get("conformal_alpha", "?")
    out = [
        "",
        f"Single-split snapshot (calibrator = `{method}`, conformal α = {alpha}). "
        "For the full factorial across seeds + methods, see the ablation table below.",
        "",
        "| Model | ECE pre | ECE post | Brier pre | Brier post | "
        "Coverage (APS) | Avg set size (APS) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in rep["models"].items():
        pre = m["pre"]
        post = m["post"]
        cf = m["conformal"]
        out.append(
            f"| {name} | {_fmt(pre['ece'])} | {_fmt(post['ece'])} "
            f"| {_fmt(pre['brier'])} | {_fmt(post['brier'])} "
            f"| {_fmt(cf['empirical_coverage'])} | {_fmt(cf['avg_set_size'])} |"
        )
    return _wrap(out)


def _agg_csv_rows(ablation_dir: Path) -> list[dict]:
    csv_path = ablation_dir / "summary.csv"
    if not csv_path.exists():
        return []
    rows = list(csv.DictReader(csv_path.open()))
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["model"], r["calibration"], r["conformal"])
        groups[key].append(r)
    out = []
    for (model, cal, conf), seed_rows in sorted(groups.items()):
        def col(name):
            vals = []
            for r in seed_rows:
                v = r.get(name, "")
                if v:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
            if not vals:
                return None, None
            if len(vals) == 1:
                return vals[0], 0.0
            return mean(vals), pstdev(vals)
        acc_m, acc_s = col("metric_accuracy")
        ece_m, ece_s = col("metric_ece")
        cov_m, cov_s = col("metric_coverage")
        sz_m, sz_s = col("metric_avg_set_size")
        out.append({
            "model": model, "calibration": cal, "conformal": conf,
            "n_seeds": len(seed_rows),
            "acc_m": acc_m, "acc_s": acc_s,
            "ece_m": ece_m, "ece_s": ece_s,
            "cov_m": cov_m, "cov_s": cov_s,
            "sz_m": sz_m, "sz_s": sz_s,
        })
    return out


def _ms(m, s, digits=3):
    if m is None:
        return "—"
    return f"{_fmt(m, digits)} ± {_fmt(s, digits)}"


def render_ablation_summary(ctx: Context) -> str:
    if ctx.ablation_dir is None:
        return _missing("results/ablations/<run>/summary.csv")
    cells = _agg_csv_rows(ctx.ablation_dir)
    if not cells:
        return _missing(f"results/ablations/{ctx.ablation_dir.name}/summary.csv")
    out = [
        "",
        f"Source: `results/ablations/{ctx.ablation_dir.name}/summary.csv`. "
        f"Aggregated as mean ± std across seeds. Target coverage = 1 − α.",
        "",
        "| Model | Calibration | Conformal | n seeds | Accuracy | ECE | "
        "Coverage | Avg set size |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        out.append(
            f"| {c['model']} | {c['calibration']} | {c['conformal']} | "
            f"{c['n_seeds']} | {_ms(c['acc_m'], c['acc_s'])} "
            f"| {_ms(c['ece_m'], c['ece_s'])} "
            f"| {_ms(c['cov_m'], c['cov_s'])} "
            f"| {_ms(c['sz_m'], c['sz_s'], digits=2)} |"
        )
    return _wrap(out)


def render_ablation_runs(ctx: Context) -> str:
    abl = ctx.results_dir / "ablations"
    if not abl.exists():
        return _missing("results/ablations/")
    dirs = sorted(d for d in abl.iterdir() if d.is_dir())
    if not dirs:
        return _missing("results/ablations/ (empty)")
    out = [
        "",
        "| Run ID | Timestamp (UTC) | Git SHA | Models × Cal × Conf × Seeds "
        "| Cells run | Summary |",
        "|---|---|---|---|---:|:---:|",
    ]
    for d in dirs:
        cfg = _load_json(d / "config.json") or {}
        n_models = len(cfg.get("models", []))
        n_cal = len(cfg.get("calibrators", []))
        n_conf = len(cfg.get("conformals", []))
        n_seed = len(cfg.get("seeds", []))
        sha = cfg.get("git_sha", d.name.split("-")[-1])
        ts = cfg.get("timestamp", "?")
        summary = d / "summary.csv"
        if summary.exists():
            with summary.open() as f:
                cells_run = sum(1 for _ in f) - 1  # minus header
            cells_run_s = str(max(cells_run, 0))
            has_summary = "✓"
        else:
            cells_run_s = "—"
            has_summary = "—"
        out.append(
            f"| `{d.name}` | {ts} | `{sha}` | "
            f"{n_models}×{n_cal}×{n_conf}×{n_seed} | {cells_run_s} | {has_summary} |"
        )
    return _wrap(out)


RENDERERS: dict[str, Callable[[Context], str]] = {
    "dataset_summary": render_dataset_summary,
    "baseline_table": render_baseline_table,
    "hierarchical_table": render_hierarchical_table,
    "cnn_table": render_cnn_table,
    "wavelet_table": render_wavelet_table,
    "calibration_table": render_calibration_table,
    "ablation_summary": render_ablation_summary,
    "ablation_runs": render_ablation_runs,
}


# --- Top-level regenerate ---------------------------------------------------


def _check_marker_balance(content: str) -> None:
    begins = sorted(BEGIN_RE.findall(content))
    ends = sorted(END_RE.findall(content))
    if begins != ends:
        raise ValueError(
            f"Marker mismatch in document: BEGIN tags = {begins}, END tags = {ends}. "
            "Each <!-- BEGIN: name --> must be paired with a <!-- END: name -->."
        )


def regenerate(content: str, ctx: Context) -> str:
    _check_marker_balance(content)

    def repl(match: re.Match) -> str:
        begin, name, _body, end = match.group(1), match.group(2), match.group(3), match.group(4)
        renderer = RENDERERS.get(name)
        if renderer is None:
            return match.group(0)
        new_body = renderer(ctx)
        return begin + new_body + end

    return MARKER_RE.sub(repl, content)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--doc", type=Path, default=DEFAULT_DOC,
                   help="Path to the markdown file to update (default: docs/results.md).")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                   help="Path to the results/ directory (default: <repo>/results).")
    p.add_argument("--ablation-run", type=str, default=None,
                   help="Specific ablation run dir name to use (default: lexicographically latest).")
    p.add_argument("--check", action="store_true",
                   help="Exit 1 if regeneration would change the file (CI guard).")
    args = p.parse_args(argv)

    if args.ablation_run:
        ablation_dir = args.results_dir / "ablations" / args.ablation_run
        if not ablation_dir.exists():
            print(f"error: ablation run '{args.ablation_run}' not found", file=sys.stderr)
            return 2
    else:
        ablation_dir = _find_latest_ablation(args.results_dir)

    ctx = Context(results_dir=args.results_dir, ablation_dir=ablation_dir)

    original = args.doc.read_text()
    new = regenerate(original, ctx)

    if args.check:
        if original != new:
            print(f"{args.doc} is stale — re-run scripts/build_results_doc.py", file=sys.stderr)
            return 1
        print(f"{args.doc} is up to date.")
        return 0

    if original == new:
        print(f"{args.doc} unchanged.")
    else:
        args.doc.write_text(new)
        print(f"{args.doc} regenerated ({len(RENDERERS)} sections checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
