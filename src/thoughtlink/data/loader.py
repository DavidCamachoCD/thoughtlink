"""Data loading utilities for KernelCo/robot_control dataset."""

from pathlib import Path
from typing import Optional

import numpy as np
from huggingface_hub import snapshot_download


DATASET_ID = "KernelCo/robot_control"

CLASS_NAMES = ["Right Fist", "Left Fist", "Both Fists", "Tongue Tapping", "Relax"]

EEG_CHANNELS = ["AFF6", "AFp2", "AFp1", "AFF5", "FCz", "CPz"]
EEG_SFREQ = 500.0
NIRS_SFREQ = 4.76
CHUNK_DURATION_S = 15.0
STIMULUS_ONSET_S = 3.0


def download_dataset(cache_dir: str = "./data/raw") -> Path:
    """Download dataset from HuggingFace Hub.

    Returns the local path to the downloaded dataset.
    """
    return Path(snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=cache_dir,
    ))


def load_sample(path: Path) -> dict:
    """Load a single .npz file and return structured dict.

    Returns:
        dict with keys: eeg, nirs, label, subject_id, session_id, duration
    """
    arr = np.load(path, allow_pickle=True)
    label_info = arr["label"].item()

    return {
        "eeg": arr["feature_eeg"],            # (7499, 6) float - microvolts
        "nirs": arr["feature_moments"],        # (72, 40, 3, 2, 3) float
        "label": label_info["label"],          # str: one of CLASS_NAMES
        "subject_id": label_info["subject_id"],
        "session_id": label_info["session_id"],
        "duration": label_info["duration"],    # stimulus duration in seconds
        "file_path": str(path),
    }


def load_all(data_dir: Optional[str] = None, cache_dir: str = "./data/raw") -> list[dict]:
    """Load all samples from the dataset.

    If data_dir is not provided, downloads the dataset first.

    Returns:
        List of sample dicts sorted by file path.
    """
    if data_dir is None:
        data_dir = download_dataset(cache_dir)
    else:
        data_dir = Path(data_dir)

    npz_files = sorted(data_dir.rglob("*.npz"))

    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files found in {data_dir}. "
            "Check that the dataset was downloaded correctly."
        )

    samples = []
    for f in npz_files:
        try:
            sample = load_sample(f)
            samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    print(f"Loaded {len(samples)} samples from {data_dir}")
    print(f"Classes: {sorted(set(s['label'] for s in samples))}")
    print(f"Subjects: {len(set(s['subject_id'] for s in samples))}")

    return samples


def get_class_distribution(samples: list[dict]) -> dict[str, int]:
    """Return count of samples per class."""
    from collections import Counter
    return dict(Counter(s["label"] for s in samples))


def get_subject_distribution(samples: list[dict]) -> dict[str, int]:
    """Return count of samples per subject."""
    from collections import Counter
    return dict(Counter(s["subject_id"] for s in samples))
