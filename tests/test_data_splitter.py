"""Tests for subject-aware data splitting."""

import numpy as np
import pytest

from thoughtlink.data.splitter import (
    get_subject_folds,
    split_by_subject,
    split_by_subject_3way,
)


def _make_samples(n_subjects: int = 10, samples_per_subject: int = 5) -> list[dict]:
    """Create synthetic sample dicts for splitting tests."""
    samples = []
    for s in range(n_subjects):
        for i in range(samples_per_subject):
            samples.append({
                "subject_id": f"s{s:03d}",
                "label": "Right Fist",
                "eeg": np.random.randn(100, 6),  # small for speed
                "duration": 9.0,
            })
    return samples


class TestSplitBySubject:
    def test_no_subject_overlap(self):
        samples = _make_samples(10, 5)
        train, test = split_by_subject(samples, test_size=0.2)

        train_subjects = {s["subject_id"] for s in train}
        test_subjects = {s["subject_id"] for s in test}
        assert train_subjects.isdisjoint(test_subjects)

    def test_all_samples_preserved(self):
        samples = _make_samples(10, 5)
        train, test = split_by_subject(samples, test_size=0.2)
        assert len(train) + len(test) == len(samples)

    def test_approximate_test_ratio(self):
        samples = _make_samples(10, 5)
        train, test = split_by_subject(samples, test_size=0.2)
        n_test_subjects = len({s["subject_id"] for s in test})
        assert n_test_subjects == 2

    def test_reproducible_with_same_seed(self):
        samples = _make_samples(10, 5)
        train1, test1 = split_by_subject(samples, random_state=42)
        train2, test2 = split_by_subject(samples, random_state=42)
        assert {s["subject_id"] for s in test1} == {s["subject_id"] for s in test2}

    def test_different_seed_different_split(self):
        samples = _make_samples(20, 5)
        _, test1 = split_by_subject(samples, random_state=42)
        _, test2 = split_by_subject(samples, random_state=99)
        assert {s["subject_id"] for s in test1} != {s["subject_id"] for s in test2}


class TestSplitBySubject3Way:
    def test_disjoint_subjects(self):
        samples = _make_samples(17, 5)
        train, calib, test = split_by_subject_3way(samples)
        train_s = {s["subject_id"] for s in train}
        calib_s = {s["subject_id"] for s in calib}
        test_s = {s["subject_id"] for s in test}
        assert train_s.isdisjoint(calib_s)
        assert train_s.isdisjoint(test_s)
        assert calib_s.isdisjoint(test_s)

    def test_all_samples_preserved(self):
        samples = _make_samples(17, 5)
        train, calib, test = split_by_subject_3way(samples)
        assert len(train) + len(calib) + len(test) == len(samples)

    def test_default_split_for_17_subjects_is_13_1_3(self):
        samples = _make_samples(17, 5)
        train, calib, test = split_by_subject_3way(samples)
        assert len({s["subject_id"] for s in train}) == 13
        assert len({s["subject_id"] for s in calib}) == 1
        assert len({s["subject_id"] for s in test}) == 3

    def test_invalid_proportions_rejected(self):
        samples = _make_samples(17, 5)
        with pytest.raises(ValueError):
            split_by_subject_3way(samples, calib_size=0.5, test_size=0.6)


class TestGetSubjectFolds:
    def test_returns_correct_number_of_folds(self):
        samples = _make_samples(10, 5)
        folds = get_subject_folds(samples, n_folds=5)
        assert len(folds) == 5

    def test_no_overlap_within_fold(self):
        samples = _make_samples(10, 5)
        folds = get_subject_folds(samples, n_folds=5)
        for train, val in folds:
            train_subj = {s["subject_id"] for s in train}
            val_subj = {s["subject_id"] for s in val}
            assert train_subj.isdisjoint(val_subj)

    def test_all_samples_in_each_fold(self):
        samples = _make_samples(10, 5)
        folds = get_subject_folds(samples, n_folds=5)
        for train, val in folds:
            assert len(train) + len(val) == len(samples)

    def test_each_subject_in_val_at_least_once(self):
        samples = _make_samples(10, 5)
        folds = get_subject_folds(samples, n_folds=5)
        val_subjects_all = set()
        for _, val in folds:
            val_subjects_all.update(s["subject_id"] for s in val)
        all_subjects = {s["subject_id"] for s in samples}
        assert val_subjects_all == all_subjects
