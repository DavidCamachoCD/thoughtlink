"""ThoughtLink MuJoCo demo: replay .npz files and drive a humanoid robot.

Connects the full brain decoding pipeline to a Unitree G1 humanoid
walking in MuJoCo simulation via the bri package.

Usage (macOS — requires mjpython for MuJoCo viewer):
    uv run mjpython scripts/run_mujoco_demo.py
    uv run mjpython scripts/run_mujoco_demo.py --files data/raw/file1.npz
    uv run mjpython scripts/run_mujoco_demo.py --model results/hierarchical_model.pkl

Usage (Linux):
    uv run python scripts/run_mujoco_demo.py
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
from thoughtlink.bridge.mujoco_controller import MuJoCoController


ACTION_SYMBOLS = {
    "RIGHT": "-->",
    "LEFT": "<--",
    "FORWARD": " ^ ",
    "STOP": " # ",
}

ACTION_COLORS = {
    "RIGHT": "\033[94m",
    "LEFT": "\033[93m",
    "FORWARD": "\033[92m",
    "STOP": "\033[91m",
}
RESET = "\033[0m"


def find_model(model_path: str | None) -> Path:
    """Find the best available trained model."""
    if model_path:
        p = Path(model_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Model not found: {p}")

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


def main():
    parser = argparse.ArgumentParser(
        description="ThoughtLink MuJoCo demo: brain signals -> humanoid robot"
    )
    parser.add_argument("--model", type=str, default=None, help="Path to trained model .pkl")
    parser.add_argument("--files", nargs="+", type=str, default=None, help=".npz files to replay")
    parser.add_argument("--data-dir", type=str, default="./data/raw", help="Dataset directory")
    parser.add_argument("--max-files", type=int, default=5, help="Max files to process")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config path")
    parser.add_argument("--hold-s", type=float, default=None, help="Action hold time (seconds)")
    parser.add_argument("--forward-speed", type=float, default=None, help="Forward velocity (m/s)")
    parser.add_argument("--yaw-rate", type=float, default=None, help="Yaw rate (rad/s)")
    parser.add_argument("--smooth-alpha", type=float, default=None, help="Smoothing factor (0-1)")
    args = parser.parse_args()

    print("=" * 60)
    print("ThoughtLink - MuJoCo Demo")
    print("Brain signals -> Humanoid robot in simulation")
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

    # Controller parameters from config, overridable by CLI
    bridge_cfg = config.get("bridge", {}).get("controller", {})
    ctrl_kwargs = {
        "hold_s": args.hold_s or bridge_cfg.get("hold_s", 0.3),
        "forward_speed": args.forward_speed or bridge_cfg.get("forward_speed", 0.6),
        "yaw_rate": args.yaw_rate or bridge_cfg.get("yaw_rate", 0.8),
        "smooth_alpha": args.smooth_alpha or bridge_cfg.get("smooth_alpha", 0.3),
    }

    # Start MuJoCo controller
    print("\nStarting MuJoCo simulation...")
    ctrl = MuJoCoController(robot_id="g1_thoughtlink", **ctrl_kwargs)
    ctrl.start()
    print("MuJoCo viewer ready.\n")

    # Callback: dispatch each prediction to the robot
    step_count = 0
    latencies: list[float] = []

    def on_step(step: StepResult) -> None:
        nonlocal step_count
        step_count += 1
        latencies.append(step.latency_ms)

        ctrl.execute(step.action)

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

    # Create policy
    policy = BrainPolicy(model=model, config=config, on_step=on_step)

    # Run demo
    print("-" * 60)
    try:
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

            print(f"  -> {len(results)} steps in {elapsed:.0f}ms")

            # Brief pause between files so the robot visibly transitions
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        ctrl.stop()

    # Summary
    print("\n" + "=" * 60)
    print("MUJOCO DEMO SUMMARY")
    print("=" * 60)
    print(f"Files processed:  {len(npz_files)}")
    print(f"Total steps:      {step_count}")
    if latencies:
        print(f"Avg latency:      {np.mean(latencies):.1f} ms")
        print(f"P95 latency:      {np.percentile(latencies, 95):.1f} ms")


if __name__ == "__main__":
    main()
