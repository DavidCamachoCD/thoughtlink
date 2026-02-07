"""Mapping from decoded brain intent classes to robot actions."""

# Intent-to-Action mapping
# Both Fists = bilateral motor imagery = go forward
# Tongue Tapping = distinct, intentional halt signal
# Relax = no intent = stop

INTENT_TO_ACTION_NAME = {
    "Right Fist": "RIGHT",
    "Left Fist": "LEFT",
    "Both Fists": "FORWARD",
    "Tongue Tapping": "STOP",
    "Relax": "STOP",
}


def intent_to_action_name(intent: str) -> str:
    """Convert intent class name to action name string.

    Args:
        intent: One of the 5 class names.

    Returns:
        Action name: RIGHT, LEFT, FORWARD, or STOP.
    """
    return INTENT_TO_ACTION_NAME.get(intent, "STOP")
