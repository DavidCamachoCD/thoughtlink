"""Confidence filtering with hysteresis and debouncing for temporal stability."""

import numpy as np
from collections import deque, Counter


class IntentConfidenceFilter:
    """Implements three stability mechanisms:

    1. Confidence threshold: only act if P(class) > threshold
    2. Hysteresis: require higher confidence to START a new action
       than to CONTINUE the current one (prevents oscillation)
    3. Debouncing: require N consecutive agreeing predictions before
       committing to a new action (prevents flicker)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        hysteresis_margin: float = 0.1,
        debounce_count: int = 3,
        default_action: str = "Relax",
    ):
        self.confidence_threshold = confidence_threshold
        self.hysteresis_margin = hysteresis_margin
        self.debounce_count = debounce_count
        self.default_action = default_action

        self.current_action = default_action
        self.pending_action: str | None = None
        self.pending_count = 0

    def update(self, probs: np.ndarray, class_names: list[str]) -> str:
        """Process a new probability vector and return the stable action.

        Args:
            probs: Array of class probabilities, shape (n_classes,).
            class_names: List of class names matching probs indices.

        Returns:
            Confirmed action string.
        """
        best_idx = np.argmax(probs)
        best_class = class_names[best_idx]
        best_prob = probs[best_idx]

        # Hysteresis: switching to a new action requires higher confidence
        if best_class == self.current_action:
            threshold = self.confidence_threshold - self.hysteresis_margin
        else:
            threshold = self.confidence_threshold + self.hysteresis_margin

        # Below threshold: keep current action, reset pending
        if best_prob < threshold:
            self.pending_action = None
            self.pending_count = 0
            return self.current_action

        # Debouncing: require consecutive agreement
        if best_class == self.pending_action:
            self.pending_count += 1
        else:
            self.pending_action = best_class
            self.pending_count = 1

        # Commit when debounce count reached
        if self.pending_count >= self.debounce_count:
            self.current_action = best_class
            self.pending_action = None
            self.pending_count = 0

        return self.current_action

    def reset(self):
        """Reset filter state."""
        self.current_action = self.default_action
        self.pending_action = None
        self.pending_count = 0


class MajorityVotingSmoother:
    """Smooth action output using majority voting over a sliding window."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.window: deque = deque(maxlen=window_size)

    def smooth(self, action: str) -> str:
        """Add an action and return the majority vote.

        Args:
            action: Latest action string.

        Returns:
            Most common action in the sliding window.
        """
        self.window.append(action)
        counts = Counter(self.window)
        return counts.most_common(1)[0][0]

    def reset(self):
        """Clear the window."""
        self.window.clear()


class StabilityPipeline:
    """Combined confidence filter + majority voting."""

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        hysteresis_margin: float = 0.1,
        debounce_count: int = 3,
        smoother_window: int = 5,
    ):
        self.confidence_filter = IntentConfidenceFilter(
            confidence_threshold=confidence_threshold,
            hysteresis_margin=hysteresis_margin,
            debounce_count=debounce_count,
        )
        self.smoother = MajorityVotingSmoother(window_size=smoother_window)

    def process(self, probs: np.ndarray, class_names: list[str]) -> str:
        """Process probability vector through full stability pipeline.

        Returns:
            Stable, debounced, smoothed action string.
        """
        filtered = self.confidence_filter.update(probs, class_names)
        smoothed = self.smoother.smooth(filtered)
        return smoothed

    def reset(self):
        """Reset all state."""
        self.confidence_filter.reset()
        self.smoother.reset()
