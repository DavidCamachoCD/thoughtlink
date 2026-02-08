"""BrainPolicy: main loop from brain signals to robot actions.

Orchestrates the full pipeline:
    .npz stream -> preprocess -> decoder -> stability -> intent -> action
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from thoughtlink.data.loader import CLASS_NAMES, load_sample
from thoughtlink.preprocessing.eeg import preprocess_eeg
from thoughtlink.features.eeg_features import extract_window_features
from thoughtlink.inference.decoder import RealtimeDecoder
from thoughtlink.inference.confidence import StabilityPipeline
from thoughtlink.bridge.intent_to_action import intent_to_action_name


@dataclass
class StepResult:
    """Result of a single prediction step."""

    timestamp_s: float
    raw_intent: str
    stable_intent: str
    action: str
    confidence: float
    probs: np.ndarray
    latency_ms: float


class BrainPolicy:
    """Orchestrates the full brain-to-robot pipeline.

    In demo mode, replays .npz files simulating real-time streaming.
    The architecture is identical to a live BCI system — only the
    data source changes.
    """

    def __init__(
        self,
        model,
        config: dict,
        on_step: Callable[[StepResult], None] | None = None,
    ):
        """
        Args:
            model: Trained classifier with predict_proba(X).
            config: Full config dict (from configs/default.yaml).
            on_step: Optional callback invoked on each prediction step.
        """
        eeg_cfg = config["preprocessing"]["eeg"]
        inf_cfg = config["inference"]
        feat_cfg = config.get("features", {})
        wavelet_cfg = feat_cfg.get("wavelet", {})
        use_wavelet = wavelet_cfg.get("enabled", False)

        self.sfreq = eeg_cfg["sfreq"]
        self.prediction_hz = inf_cfg["prediction_hz"]
        self.config = config

        self.decoder = RealtimeDecoder(
            model=model,
            feature_extractor=lambda w: extract_window_features(
                w, sfreq=self.sfreq,
                include_wavelet=use_wavelet,
                wavelet_config=wavelet_cfg,
            ),
            window_size_s=config["preprocessing"]["window_duration_s"],
            sfreq=self.sfreq,
        )

        self.stability = StabilityPipeline(
            confidence_threshold=inf_cfg["confidence_threshold"],
            hysteresis_margin=inf_cfg["hysteresis_margin"],
            debounce_count=inf_cfg["debounce_count"],
            smoother_window=inf_cfg["smoother_window"],
        )

        self.on_step = on_step

    def run_on_file(self, npz_path: str | Path) -> list[StepResult]:
        """Simulate real-time streaming from a single .npz file.

        Loads the file, preprocesses the full 15s chunk, then feeds
        chunks into the decoder at the configured prediction rate.

        Args:
            npz_path: Path to a .npz file from the dataset.

        Returns:
            List of StepResult for each prediction tick.
        """
        sample = load_sample(Path(npz_path))
        eeg_clean = preprocess_eeg(sample["eeg"], sfreq=self.sfreq)

        results = self._stream_eeg(eeg_clean)
        return results

    def run_on_array(self, eeg_data: np.ndarray) -> list[StepResult]:
        """Run on a pre-preprocessed EEG array.

        Args:
            eeg_data: Shape (n_samples, n_channels), already preprocessed.

        Returns:
            List of StepResult for each prediction tick.
        """
        return self._stream_eeg(eeg_data)

    def step(self, probs: np.ndarray) -> StepResult:
        """Process a single probability vector through stability + mapping.

        Useful when the caller handles decoding externally.

        Args:
            probs: Class probability vector, shape (n_classes,).

        Returns:
            StepResult with stable intent and action.
        """
        raw_idx = int(np.argmax(probs))
        raw_intent = CLASS_NAMES[raw_idx]

        stable_intent = self.stability.process(probs, CLASS_NAMES)
        action = intent_to_action_name(stable_intent)

        return StepResult(
            timestamp_s=0.0,
            raw_intent=raw_intent,
            stable_intent=stable_intent,
            action=action,
            confidence=float(probs[raw_idx]),
            probs=probs,
            latency_ms=0.0,
        )

    def reset(self) -> None:
        """Reset decoder buffer and stability pipeline state."""
        self.decoder.clear()
        self.stability.reset()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stream_eeg(self, eeg_clean: np.ndarray) -> list[StepResult]:
        """Feed preprocessed EEG through the pipeline in simulated real-time.

        Args:
            eeg_clean: Preprocessed EEG, shape (n_samples, n_channels).

        Returns:
            List of StepResult, one per prediction tick.
        """
        samples_per_step = int(self.sfreq / self.prediction_hz)
        results: list[StepResult] = []

        for i in range(0, eeg_clean.shape[0], samples_per_step):
            chunk = eeg_clean[i : i + samples_per_step]
            self.decoder.feed_samples(chunk)

            probs, latency_ms = self.decoder.predict()
            if probs is None:
                continue

            raw_idx = int(np.argmax(probs))
            raw_intent = CLASS_NAMES[raw_idx]

            stable_intent = self.stability.process(probs, CLASS_NAMES)
            action = intent_to_action_name(stable_intent)

            result = StepResult(
                timestamp_s=i / self.sfreq,
                raw_intent=raw_intent,
                stable_intent=stable_intent,
                action=action,
                confidence=float(probs[raw_idx]),
                probs=probs,
                latency_ms=latency_ms,
            )
            results.append(result)

            if self.on_step is not None:
                self.on_step(result)

        return results


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)
