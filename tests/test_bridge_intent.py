"""Tests for intent-to-action mapping."""

import pytest

from thoughtlink.bridge.intent_to_action import (
    INTENT_TO_ACTION_NAME,
    intent_to_action_name,
)


class TestIntentToActionMapping:
    def test_right_fist(self):
        assert intent_to_action_name("Right Fist") == "RIGHT"

    def test_left_fist(self):
        assert intent_to_action_name("Left Fist") == "LEFT"

    def test_both_fists(self):
        assert intent_to_action_name("Both Fists") == "FORWARD"

    def test_tongue_tapping(self):
        assert intent_to_action_name("Tongue Tapping") == "STOP"

    def test_relax(self):
        assert intent_to_action_name("Relax") == "STOP"

    def test_unknown_intent_defaults_to_stop(self):
        assert intent_to_action_name("Unknown") == "STOP"
        assert intent_to_action_name("") == "STOP"

    def test_all_five_classes_mapped(self):
        expected = {"Right Fist", "Left Fist", "Both Fists", "Tongue Tapping", "Relax"}
        assert set(INTENT_TO_ACTION_NAME.keys()) == expected

    def test_action_values_valid(self):
        valid_actions = {"RIGHT", "LEFT", "FORWARD", "STOP"}
        for action in INTENT_TO_ACTION_NAME.values():
            assert action in valid_actions
