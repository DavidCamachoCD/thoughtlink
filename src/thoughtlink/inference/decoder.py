"""Real-time windowed decoder for brain signal inference."""

import time
from collections import deque

import numpy as np


class RealtimeDecoder:
    """Simulates real-time decoding by processing EEG windows from a rolling buffer.

    In a live system, new samples would be fed continuously from the headset.
    For the hackathon demo, we replay .npz files as a simulated stream.
    """

    def __init__(
        self,
        model,
        feature_extractor,
        preprocessor=None,
        window_size_s: float = 1.0,
        sfreq: float = 500.0,
        buffer_duration_s: float = 5.0,
    ):
        """
        Args:
            model: Trained classifier with predict_proba(X) method.
            feature_extractor: Function (window) -> feature_vector.
            preprocessor: Optional function to preprocess a window.
            window_size_s: Window duration in seconds.
            sfreq: EEG sampling frequency.
            buffer_duration_s: Rolling buffer duration.
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        self.window_size = int(window_size_s * sfreq)
        self.sfreq = sfreq
        self.buffer = deque(maxlen=int(buffer_duration_s * sfreq))

    def feed_samples(self, samples: np.ndarray) -> None:
        """Feed new EEG samples into the rolling buffer.

        Args:
            samples: Shape (n_new_samples, n_channels).
        """
        for s in samples:
            self.buffer.append(s)

    def predict(self) -> tuple[np.ndarray | None, float]:
        """Run prediction on the latest window in the buffer.

        Returns:
            (class_probabilities, latency_ms) or (None, 0.0) if buffer too short.
        """
        if len(self.buffer) < self.window_size:
            return None, 0.0

        t0 = time.perf_counter()

        # Extract latest window
        window = np.array(list(self.buffer))[-self.window_size:]

        # Optional preprocessing
        if self.preprocessor is not None:
            window = self.preprocessor(window)

        # Feature extraction
        features = self.feature_extractor(window)

        # Model prediction
        probs = self.model.predict_proba(features.reshape(1, -1))[0]

        latency_ms = (time.perf_counter() - t0) * 1000
        return probs, latency_ms

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
