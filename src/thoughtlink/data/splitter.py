"""Subject-aware data splitting to avoid data leakage."""

from collections import defaultdict

import numpy as np


def split_by_subject(
    samples: list[dict],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split samples into train/test sets by subject_id.

    All samples from a given subject go entirely into train or test,
    never split across both. This prevents data leakage from
    subject-specific brain patterns.

    Args:
        samples: List of sample dicts with 'subject_id' key.
        test_size: Fraction of subjects to use for test.
        random_state: Random seed for reproducibility.

    Returns:
        (train_samples, test_samples) tuple.
    """
    rng = np.random.RandomState(random_state)

    # Group samples by subject
    subject_samples = defaultdict(list)
    for s in samples:
        subject_samples[s["subject_id"]].append(s)

    subjects = sorted(subject_samples.keys())
    n_test = max(1, int(len(subjects) * test_size))

    # Shuffle and split subjects
    rng.shuffle(subjects)
    test_subjects = set(subjects[:n_test])
    train_subjects = set(subjects[n_test:])

    train = [s for s in samples if s["subject_id"] in train_subjects]
    test = [s for s in samples if s["subject_id"] in test_subjects]

    print(f"Split: {len(train_subjects)} train subjects ({len(train)} samples), "
          f"{len(test_subjects)} test subjects ({len(test)} samples)")

    return train, test


def split_by_subject_3way(
    samples: list[dict],
    calib_size: float = 1 / 17,
    test_size: float = 3 / 17,
    random_state: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/calibration/test by subject_id.

    All samples from a given subject go entirely into one of the three sets,
    never split across them. The calibration set is held out for post-hoc
    confidence calibration and conformal prediction (Romano et al. 2020),
    where exchangeability between the calibration and test sets is required
    for coverage guarantees — subject-level holdout preserves it under the
    "new subject" deployment assumption.

    With 17 subjects and the defaults, the split is exactly 13/1/3.

    Args:
        samples: List of sample dicts with 'subject_id' key.
        calib_size: Fraction of subjects for calibration.
        test_size: Fraction of subjects for test.
        random_state: Random seed for reproducibility.

    Returns:
        (train_samples, calib_samples, test_samples) tuple.
    """
    if calib_size + test_size >= 1.0:
        raise ValueError("calib_size + test_size must be < 1.0")

    rng = np.random.RandomState(random_state)

    subject_samples = defaultdict(list)
    for s in samples:
        subject_samples[s["subject_id"]].append(s)

    subjects = sorted(subject_samples.keys())
    n_total = len(subjects)
    n_test = max(1, int(round(n_total * test_size)))
    n_calib = max(1, int(round(n_total * calib_size)))
    if n_calib + n_test >= n_total:
        raise ValueError(
            f"Not enough subjects ({n_total}) for the requested split "
            f"(calib={n_calib}, test={n_test})."
        )

    rng.shuffle(subjects)
    test_subjects = set(subjects[:n_test])
    calib_subjects = set(subjects[n_test : n_test + n_calib])
    train_subjects = set(subjects[n_test + n_calib :])

    train = [s for s in samples if s["subject_id"] in train_subjects]
    calib = [s for s in samples if s["subject_id"] in calib_subjects]
    test = [s for s in samples if s["subject_id"] in test_subjects]

    print(
        f"3-way split: "
        f"{len(train_subjects)} train ({len(train)} samples), "
        f"{len(calib_subjects)} calib ({len(calib)} samples), "
        f"{len(test_subjects)} test ({len(test)} samples)"
    )

    return train, calib, test


def get_subject_folds(
    samples: list[dict],
    n_folds: int = 5,
    random_state: int = 42,
) -> list[tuple[list[dict], list[dict]]]:
    """Create leave-N-subjects-out cross-validation folds.

    Each fold holds out a group of subjects for validation.

    Args:
        samples: List of sample dicts.
        n_folds: Number of folds.
        random_state: Random seed.

    Returns:
        List of (train, val) tuples.
    """
    rng = np.random.RandomState(random_state)

    subject_samples = defaultdict(list)
    for s in samples:
        subject_samples[s["subject_id"]].append(s)

    subjects = sorted(subject_samples.keys())
    rng.shuffle(subjects)

    # Distribute subjects across folds
    fold_subjects = [[] for _ in range(n_folds)]
    for i, subj in enumerate(subjects):
        fold_subjects[i % n_folds].append(subj)

    folds = []
    for fold_idx in range(n_folds):
        val_subjects = set(fold_subjects[fold_idx])
        train = [s for s in samples if s["subject_id"] not in val_subjects]
        val = [s for s in samples if s["subject_id"] in val_subjects]
        folds.append((train, val))

    return folds
