"""Tests for data loader utilities (no real dataset downloads)."""

import numpy as np
import pytest
from pathlib import Path

from thoughtlink.data.loader import (
    CLASS_NAMES,
    EEG_CHANNELS,
    EEG_SFREQ,
    NIRS_SFREQ,
    load_sample,
    get_class_distribution,
    get_subject_distribution,
)


class TestConstants:
    def test_class_names_count(self):
        assert len(CLASS_NAMES) == 5

    def test_class_names_content(self):
        expected = {"Right Fist", "Left Fist", "Both Fists", "Tongue Tapping", "Relax"}
        assert set(CLASS_NAMES) == expected

    def test_eeg_channels_count(self):
        assert len(EEG_CHANNELS) == 6

    def test_eeg_sfreq(self):
        assert EEG_SFREQ == 500.0

    def test_nirs_sfreq(self):
        assert NIRS_SFREQ == 4.76


class TestGetClassDistribution:
    def test_counts_correct(self):
        samples = [
            {"label": "Right Fist"},
            {"label": "Right Fist"},
            {"label": "Left Fist"},
            {"label": "Relax"},
        ]
        dist = get_class_distribution(samples)
        assert dist["Right Fist"] == 2
        assert dist["Left Fist"] == 1
        assert dist["Relax"] == 1

    def test_empty_input(self):
        dist = get_class_distribution([])
        assert dist == {}


class TestGetSubjectDistribution:
    def test_counts_correct(self):
        samples = [
            {"subject_id": "s1"},
            {"subject_id": "s1"},
            {"subject_id": "s2"},
        ]
        dist = get_subject_distribution(samples)
        assert dist["s1"] == 2
        assert dist["s2"] == 1


class TestLoadSample:
    def test_load_synthetic_npz(self, tmp_path):
        """Create a synthetic .npz matching real format and verify round-trip."""
        eeg = np.random.randn(7499, 6).astype(np.float32)
        nirs = np.random.randn(72, 40, 3, 2, 3).astype(np.float32)
        label_dict = {
            "label": "Right Fist",
            "subject_id": "s001",
            "session_id": "sess_01",
            "duration": 9.0,
        }

        path = tmp_path / "sample.npz"
        np.savez(
            path,
            feature_eeg=eeg,
            feature_moments=nirs,
            label=label_dict,
        )

        sample = load_sample(path)
        assert sample["eeg"].shape == (7499, 6)
        assert sample["nirs"].shape == (72, 40, 3, 2, 3)
        assert sample["label"] == "Right Fist"
        assert sample["subject_id"] == "s001"
        assert sample["session_id"] == "sess_01"
        assert sample["duration"] == 9.0
