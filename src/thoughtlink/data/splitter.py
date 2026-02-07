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
