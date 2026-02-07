"""End-to-end ThoughtLink demo: replay .npz files through the full pipeline.

Usage:
    # Run on downloaded dataset (loads model from results/)
    uv run python scripts/run_demo.py

    # Run on specific files
    uv run python scripts/run_demo.py --files data/raw/file1.npz data/raw/file2.npz

    # Use hierarchical model instead of baseline
    uv run python scripts/run_demo.py --model results/hierarchical_model.pkl

    # Limit number of files
    uv run python scripts/run_demo.py --max-files 5
"""

import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thoughtlink.data.loader import CLASS_NAMES, load_sample
from thoughtlink.bridge.brain_policy import BrainPolicy, StepResult, load_config


ACTION_SYMBOLS = {
    "RIGHT": "-->",
    "LEFT": "<--",
    "FORWARD": " ^ ",
    "STOP": " # ",
}

ACTION_COLORS = {
    "RIGHT": "\033[94m",   # blue
    "LEFT": "\033[93m",    # yellow
    "FORWARD": "\033[92m", # green
    "STOP": "\033[91m",    # red
}
RESET = "\033[0m"


def format_step(step: StepResult, file_label: str = "") -> str:
    """Format a single step result for terminal output."""
    sym = ACTION_SYMBOLS.get(step.action, " ? ")
    color = ACTION_COLORS.get(step.action, "")

    probs_str = " ".join(
        f"{CLASS_NAMES[i][:2]}:{step.probs[i]:.2f}" for i in range(len(CLASS_NAMES))
    )

    return (
        f"  {step.timestamp_s:6.1f}s  "
        f"{color}{sym}{RESET}  "
        f"{step.action:<8s}  "
        f"(raw: {step.raw_intent:<16s} "
        f"conf: {step.confidence:.2f}  "
        f"lat: {step.latency_ms:5.1f}ms)  "
        f"[{probs_str}]"
    )


def print_step_live(step: StepResult) -> None:
    """Callback for real-time output during streaming."""
    sym = ACTION_SYMBOLS.get(step.action, " ? ")
    color = ACTION_COLORS.get(step.action, "")
    print(
        f"  {step.timestamp_s:6.1f}s  "
        f"{color}{sym}{RESET}  "
        f"{step.action:<8s}  "
        f"conf: {step.confidence:.2f}  "
        f"lat: {step.latency_ms:5.1f}ms",
        flush=True,
    )


def find_model(model_path: str | None) -> Path:
    """Find the best available trained model."""
    if model_path:
        p = Path(model_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Model not found: {p}")

    # Try hierarchical first, then baseline
    candidates = [
        Path("results/hierarchical_model.pkl"),
        Path("results/best_baseline.pkl"),
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "No trained model found in results/. "
        "Run 'uv run python scripts/train_baseline.py' or "
        "'uv run python scripts/train_hierarchical.py' first."
    )


def find_npz_files(
    file_paths: list[str] | None,
    data_dir: str,
    max_files: int,
) -> list[Path]:
    """Resolve .npz files for the demo."""
    if file_paths:
        files = [Path(f) for f in file_paths]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Files not found: {missing}")
        return files[:max_files]

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_path}. "
            "Run 'uv run python scripts/train_baseline.py' to download the dataset, "
            "or provide --files explicitly."
        )

    files = sorted(data_path.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_path}")

    return files[:max_files]


def summarize_results(
    all_results: list[tuple[str, str, list[StepResult]]],
) -> None:
    """Print summary statistics across all files."""
    total_steps = 0
    correct_actions = 0
    action_counts: dict[str, int] = {}
    latencies: list[float] = []

    for label, _, results in all_results:
        expected_action = {
            "Right Fist": "RIGHT",
            "Left Fist": "LEFT",
            "Both Fists": "FORWARD",
            "Tongue Tapping": "STOP",
            "Relax": "STOP",
        }.get(label, "STOP")

        for step in results:
            total_steps += 1
            latencies.append(step.latency_ms)
            action_counts[step.action] = action_counts.get(step.action, 0) + 1
            if step.action == expected_action:
                correct_actions += 1

    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    print(f"Files processed:  {len(all_results)}")
    print(f"Total steps:      {total_steps}")

    if total_steps > 0:
        acc = correct_actions / total_steps
        print(f"Action accuracy:  {acc:.1%} ({correct_actions}/{total_steps})")
        print(f"Avg latency:      {np.mean(latencies):.1f} ms")
        print(f"Max latency:      {np.max(latencies):.1f} ms")
        print(f"P95 latency:      {np.percentile(latencies, 95):.1f} ms")

    print(f"\nAction distribution:")
    for action, count in sorted(action_counts.items()):
        bar = "#" * min(count, 40)
        print(f"  {action:<8s}: {count:4d}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="ThoughtLink end-to-end demo")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model .pkl")
    parser.add_argument("--files", nargs="+", type=str, default=None, help=".npz files to replay")
    parser.add_argument("--data-dir", type=str, default="./data/raw", help="Dataset directory")
    parser.add_argument("--max-files", type=int, default=10, help="Max files to process")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config path")
    parser.add_argument("--live", action="store_true", help="Print steps in real-time")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between steps (seconds), for live visualization")
    args = parser.parse_args()

    print("=" * 60)
    print("ThoughtLink - End-to-End Demo")
    print("=" * 60)

    # Load config
    config = load_config(args.config)
    print(f"Config: {args.config}")

    # Load model
    model_path = find_model(args.model)
    print(f"Model:  {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Find files
    npz_files = find_npz_files(args.files, args.data_dir, args.max_files)
    print(f"Files:  {len(npz_files)} .npz files")

    # Set up callback
    on_step = None
    if args.live:
        def on_step_with_delay(step: StepResult) -> None:
            print_step_live(step)
            if args.delay > 0:
                time.sleep(args.delay)
        on_step = on_step_with_delay

    # Create policy
    policy = BrainPolicy(model=model, config=config, on_step=on_step)

    # Run demo
    print("\n" + "-" * 60)
    all_results: list[tuple[str, str, list[StepResult]]] = []

    for idx, npz_path in enumerate(npz_files):
        sample = load_sample(npz_path)
        label = sample["label"]
        subject = sample["subject_id"]

        print(f"\n[{idx + 1}/{len(npz_files)}] {npz_path.name}")
        print(f"  Label: {label}  |  Subject: {subject}")

        policy.reset()

        t0 = time.perf_counter()
        results = policy.run_on_file(npz_path)
        elapsed = (time.perf_counter() - t0) * 1000

        if not args.live:
            for step in results:
                print(format_step(step))

        print(f"  -> {len(results)} steps in {elapsed:.0f}ms")

        all_results.append((label, subject, results))

    # Summary
    summarize_results(all_results)


if __name__ == "__main__":
    main()
