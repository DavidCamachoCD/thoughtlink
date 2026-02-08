"""Multi-robot orchestrator: dispatch decoded intent to N robot controllers.

Demonstrates scalability — one human brain decodes intent once, and the
orchestrator fans out the action to multiple robots simultaneously.

Architecture:
    BrainPolicy (1) --> Orchestrator --> Controller_1
                                    --> Controller_2
                                    --> ...
                                    --> Controller_N

Dispatch is O(N) but each controller runs independently, so the system
scales linearly. With async dispatch, latency stays constant regardless
of fleet size.
"""

import time
import logging
from dataclasses import dataclass
from typing import Protocol

from thoughtlink.bridge.brain_policy import StepResult

logger = logging.getLogger(__name__)


class RobotController(Protocol):
    """Protocol for any robot backend (MuJoCo, ROS, sim, or mock)."""

    @property
    def robot_id(self) -> str: ...

    def execute(self, action: str) -> bool:
        """Execute a high-level action. Returns True on success."""
        ...

    def stop(self) -> None:
        """Emergency stop."""
        ...


@dataclass
class DispatchResult:
    """Result of dispatching an action to the fleet."""

    action: str
    n_robots: int
    n_success: int
    n_failed: int
    dispatch_ms: float
    per_robot_ms: float
    failed_ids: list[str]


class SimulatedController:
    """Lightweight simulated robot for testing orchestration without MuJoCo.

    Each instance represents one robot in the fleet.
    """

    def __init__(self, robot_id: str, fail_rate: float = 0.0):
        """
        Args:
            robot_id: Unique identifier for this robot.
            fail_rate: Probability of simulated execution failure (0.0-1.0).
        """
        self._robot_id = robot_id
        self.fail_rate = fail_rate
        self.last_action: str | None = None
        self.action_count = 0

    @property
    def robot_id(self) -> str:
        return self._robot_id

    def execute(self, action: str) -> bool:
        import random
        self.last_action = action
        self.action_count += 1
        if self.fail_rate > 0 and random.random() < self.fail_rate:
            return False
        return True

    def stop(self) -> None:
        self.last_action = "STOP"
        self.action_count += 1


class Orchestrator:
    """Dispatches decoded brain intent to a fleet of robot controllers.

    Supports:
    - Synchronous fan-out to N robots
    - Per-robot success/failure tracking
    - Fleet-wide emergency stop
    - Action filtering (only dispatch on action change)
    """

    def __init__(
        self,
        controllers: list[RobotController] | None = None,
        deduplicate: bool = True,
    ):
        """
        Args:
            controllers: List of robot controllers. Can be added later.
            deduplicate: If True, only dispatch when action changes.
        """
        self.controllers: list[RobotController] = controllers or []
        self.deduplicate = deduplicate
        self._last_action: str | None = None
        self._dispatch_count = 0
        self._total_failures = 0

    @property
    def fleet_size(self) -> int:
        return len(self.controllers)

    def add_controller(self, controller: RobotController) -> None:
        """Add a robot to the fleet."""
        self.controllers.append(controller)

    def dispatch(self, step: StepResult) -> DispatchResult | None:
        """Dispatch a decoded action to all robots in the fleet.

        Args:
            step: StepResult from BrainPolicy containing the action.

        Returns:
            DispatchResult with success/failure counts, or None if
            deduplicated (same action as last dispatch).
        """
        action = step.action

        if self.deduplicate and action == self._last_action:
            return None

        t0 = time.perf_counter()

        successes = 0
        failures = 0
        failed_ids: list[str] = []

        for controller in self.controllers:
            try:
                ok = controller.execute(action)
                if ok:
                    successes += 1
                else:
                    failures += 1
                    failed_ids.append(controller.robot_id)
            except Exception as e:
                logger.warning(
                    "Robot %s failed to execute %s: %s",
                    controller.robot_id, action, e,
                )
                failures += 1
                failed_ids.append(controller.robot_id)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        n = len(self.controllers)
        per_robot = elapsed_ms / n if n > 0 else 0.0

        self._last_action = action
        self._dispatch_count += 1
        self._total_failures += failures

        result = DispatchResult(
            action=action,
            n_robots=n,
            n_success=successes,
            n_failed=failures,
            dispatch_ms=elapsed_ms,
            per_robot_ms=per_robot,
            failed_ids=failed_ids,
        )

        if failures > 0:
            logger.warning(
                "Dispatch %s: %d/%d failed [%s]",
                action, failures, n, ", ".join(failed_ids),
            )

        return result

    def emergency_stop(self) -> None:
        """Send stop command to all robots immediately."""
        for controller in self.controllers:
            try:
                controller.stop()
            except Exception as e:
                logger.error(
                    "Emergency stop failed for %s: %s",
                    controller.robot_id, e,
                )
        self._last_action = "STOP"

    def get_stats(self) -> dict:
        """Return fleet dispatch statistics."""
        return {
            "fleet_size": self.fleet_size,
            "total_dispatches": self._dispatch_count,
            "total_failures": self._total_failures,
            "failure_rate": (
                self._total_failures / (self._dispatch_count * self.fleet_size)
                if self._dispatch_count > 0 and self.fleet_size > 0
                else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset dispatch state."""
        self._last_action = None
        self._dispatch_count = 0
        self._total_failures = 0


def create_simulated_fleet(n_robots: int, fail_rate: float = 0.0) -> Orchestrator:
    """Create an orchestrator with N simulated robots.

    Args:
        n_robots: Number of robots in the fleet.
        fail_rate: Simulated failure probability per robot.

    Returns:
        Orchestrator ready to dispatch.
    """
    controllers = [
        SimulatedController(robot_id=f"robot_{i:03d}", fail_rate=fail_rate)
        for i in range(n_robots)
    ]
    return Orchestrator(controllers=controllers)
