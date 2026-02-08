"""MuJoCo robot controller using the bri package.

Wraps bri.Controller to satisfy the RobotController protocol defined
in orchestrator.py, connecting ThoughtLink's decoded brain intents to
a Unitree G1 humanoid walking in MuJoCo simulation.

Requires:
    uv sync --extra sim
    uv pip install --no-deps /path/to/brain-robot-interface
"""

import logging

try:
    from bri import Action, Controller
except ImportError:
    Action = None
    Controller = None

logger = logging.getLogger(__name__)


class MuJoCoController:
    """RobotController implementation backed by bri + MuJoCo.

    Satisfies the RobotController protocol (orchestrator.py):
        - robot_id: str property
        - execute(action: str) -> bool
        - stop() -> None
    """

    def __init__(
        self,
        robot_id: str = "g1_mujoco",
        hold_s: float = 0.3,
        forward_speed: float = 0.6,
        yaw_rate: float = 0.8,
        smooth_alpha: float = 0.3,
    ):
        """
        Args:
            robot_id: Unique identifier for this controller.
            hold_s: Seconds an action is held before auto-stopping.
            forward_speed: Linear velocity (m/s) for FORWARD.
            yaw_rate: Rotational velocity (rad/s) for LEFT/RIGHT.
            smooth_alpha: Exponential smoothing factor (0-1).
        """
        if Controller is None:
            raise ImportError(
                "bri package not installed. Run: uv sync --extra sim"
            )
        self._robot_id = robot_id
        self._ctrl = Controller(
            backend="sim",
            hold_s=hold_s,
            forward_speed=forward_speed,
            yaw_rate=yaw_rate,
            smooth_alpha=smooth_alpha,
        )
        self._started = False

    @property
    def robot_id(self) -> str:
        return self._robot_id

    def start(self) -> None:
        """Start the MuJoCo simulation backend and control loop."""
        if self._started:
            return
        self._ctrl.start()
        self._started = True
        logger.info("MuJoCo controller %s started", self._robot_id)

    def execute(self, action: str) -> bool:
        """Execute a high-level action on the simulated robot.

        Args:
            action: One of RIGHT, LEFT, FORWARD, STOP.

        Returns:
            True on success.
        """
        if not self._started:
            self.start()
        try:
            bri_action = Action.from_str(action)
            self._ctrl.set_action(bri_action)
            return True
        except (ValueError, KeyError) as e:
            logger.warning("Invalid action '%s': %s", action, e)
            return False

    def stop(self) -> None:
        """Stop the robot and shut down the MuJoCo backend."""
        if self._started:
            self._ctrl.set_action(Action.STOP)
            self._ctrl.stop()
            self._started = False
            logger.info("MuJoCo controller %s stopped", self._robot_id)
