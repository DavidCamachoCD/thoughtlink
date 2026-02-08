"""Tests for bridge module: BrainPolicy, Orchestrator, and MuJoCoController."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from thoughtlink.bridge.brain_policy import BrainPolicy, StepResult, load_config
from thoughtlink.bridge.intent_to_action import intent_to_action_name
from thoughtlink.bridge.orchestrator import (
    Orchestrator,
    SimulatedController,
    create_simulated_fleet,
)
from thoughtlink.data.loader import CLASS_NAMES


# ── Fixtures ─────────────────────────────────────────────────


class MockModel:
    """Minimal model that always predicts the same class."""

    def __init__(self, predicted_class: int = 0):
        self.predicted_class = predicted_class

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        probs = np.full((n_samples, len(CLASS_NAMES)), 0.05)
        probs[:, self.predicted_class] = 0.80
        return probs


@pytest.fixture
def config():
    return load_config("configs/default.yaml")


@pytest.fixture
def mock_model():
    return MockModel(predicted_class=0)  # Right Fist


# ── BrainPolicy Tests ────────────────────────────────────────


class TestBrainPolicy:
    def test_init(self, mock_model, config):
        policy = BrainPolicy(model=mock_model, config=config)
        assert policy.sfreq == 500.0
        assert policy.prediction_hz == 2.0

    def test_step_returns_step_result(self, mock_model, config):
        policy = BrainPolicy(model=mock_model, config=config)
        probs = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
        result = policy.step(probs)

        assert isinstance(result, StepResult)
        assert result.raw_intent == "Right Fist"
        assert result.action in ("RIGHT", "LEFT", "FORWARD", "STOP")
        assert result.confidence == pytest.approx(0.8)

    def test_run_on_array(self, mock_model, config):
        policy = BrainPolicy(model=mock_model, config=config)
        # 3 seconds of synthetic EEG (6 channels)
        eeg = np.random.randn(1500, 6) * 10
        results = policy.run_on_array(eeg)

        assert isinstance(results, list)
        # With 3s of data at 2Hz prediction rate, after buffer fills (1s),
        # we should get some results
        assert len(results) > 0
        assert all(isinstance(r, StepResult) for r in results)

    def test_on_step_callback(self, mock_model, config):
        captured: list[StepResult] = []
        policy = BrainPolicy(
            model=mock_model,
            config=config,
            on_step=lambda r: captured.append(r),
        )
        eeg = np.random.randn(1500, 6) * 10
        results = policy.run_on_array(eeg)

        assert len(captured) == len(results)

    def test_reset(self, mock_model, config):
        policy = BrainPolicy(model=mock_model, config=config)
        eeg = np.random.randn(1500, 6) * 10
        policy.run_on_array(eeg)

        policy.reset()
        # After reset, buffer should be empty (no immediate predictions)
        assert len(policy.decoder.buffer) == 0


# ── Intent to Action Tests ───────────────────────────────────


class TestIntentToAction:
    def test_all_classes_mapped(self):
        expected = {
            "Right Fist": "RIGHT",
            "Left Fist": "LEFT",
            "Both Fists": "FORWARD",
            "Tongue Tapping": "STOP",
            "Relax": "STOP",
        }
        for intent, action in expected.items():
            assert intent_to_action_name(intent) == action

    def test_unknown_defaults_to_stop(self):
        assert intent_to_action_name("Unknown") == "STOP"


# ── Orchestrator Tests ───────────────────────────────────────


class TestSimulatedController:
    def test_execute(self):
        ctrl = SimulatedController("robot_001")
        assert ctrl.execute("RIGHT") is True
        assert ctrl.last_action == "RIGHT"
        assert ctrl.action_count == 1

    def test_stop(self):
        ctrl = SimulatedController("robot_001")
        ctrl.stop()
        assert ctrl.last_action == "STOP"

    def test_robot_id(self):
        ctrl = SimulatedController("test_bot")
        assert ctrl.robot_id == "test_bot"


class TestOrchestrator:
    def _make_step(self, action: str) -> StepResult:
        return StepResult(
            timestamp_s=0.0,
            raw_intent="Right Fist",
            stable_intent="Right Fist",
            action=action,
            confidence=0.9,
            probs=np.array([0.9, 0.025, 0.025, 0.025, 0.025]),
            latency_ms=1.0,
        )

    def test_dispatch_to_fleet(self):
        orch = create_simulated_fleet(n_robots=10)
        step = self._make_step("RIGHT")
        result = orch.dispatch(step)

        assert result is not None
        assert result.n_robots == 10
        assert result.n_success == 10
        assert result.n_failed == 0
        assert result.action == "RIGHT"
        assert result.dispatch_ms >= 0

    def test_deduplication(self):
        orch = create_simulated_fleet(n_robots=5)
        step = self._make_step("RIGHT")

        result1 = orch.dispatch(step)
        result2 = orch.dispatch(step)  # same action

        assert result1 is not None
        assert result2 is None  # deduplicated

    def test_deduplication_disabled(self):
        orch = create_simulated_fleet(n_robots=5)
        orch.deduplicate = False
        step = self._make_step("RIGHT")

        result1 = orch.dispatch(step)
        result2 = orch.dispatch(step)

        assert result1 is not None
        assert result2 is not None

    def test_action_change_dispatches(self):
        orch = create_simulated_fleet(n_robots=3)

        r1 = orch.dispatch(self._make_step("RIGHT"))
        r2 = orch.dispatch(self._make_step("LEFT"))
        r3 = orch.dispatch(self._make_step("LEFT"))  # duplicate

        assert r1 is not None
        assert r2 is not None
        assert r3 is None

    def test_failure_tracking(self):
        orch = create_simulated_fleet(n_robots=5, fail_rate=1.0)
        result = orch.dispatch(self._make_step("FORWARD"))

        assert result is not None
        assert result.n_failed == 5
        assert result.n_success == 0
        assert len(result.failed_ids) == 5

    def test_emergency_stop(self):
        orch = create_simulated_fleet(n_robots=3)
        orch.emergency_stop()

        for ctrl in orch.controllers:
            assert ctrl.last_action == "STOP"

    def test_fleet_stats(self):
        orch = create_simulated_fleet(n_robots=4)
        orch.dispatch(self._make_step("RIGHT"))
        orch.dispatch(self._make_step("LEFT"))

        stats = orch.get_stats()
        assert stats["fleet_size"] == 4
        assert stats["total_dispatches"] == 2
        assert stats["failure_rate"] == 0.0

    def test_add_controller(self):
        orch = Orchestrator()
        assert orch.fleet_size == 0

        orch.add_controller(SimulatedController("new_bot"))
        assert orch.fleet_size == 1

    def test_reset(self):
        orch = create_simulated_fleet(n_robots=2)
        orch.dispatch(self._make_step("RIGHT"))
        orch.reset()

        stats = orch.get_stats()
        assert stats["total_dispatches"] == 0

    def test_scalability_100_robots(self):
        """Verify dispatch to 100 robots completes in reasonable time."""
        orch = create_simulated_fleet(n_robots=100)
        result = orch.dispatch(self._make_step("FORWARD"))

        assert result is not None
        assert result.n_robots == 100
        assert result.n_success == 100
        # Dispatch to 100 simulated robots should be well under 10ms
        assert result.dispatch_ms < 10.0


# ── MuJoCoController Tests ──────────────────────────────────


class TestMuJoCoController:
    """Tests for MuJoCoController with mocked bri backend."""

    @patch("thoughtlink.bridge.mujoco_controller.Controller")
    def test_satisfies_protocol(self, MockCtrl):
        """MuJoCoController has robot_id, execute, stop."""
        from thoughtlink.bridge.mujoco_controller import MuJoCoController

        ctrl = MuJoCoController(robot_id="test_g1")
        assert hasattr(ctrl, "robot_id")
        assert hasattr(ctrl, "execute")
        assert hasattr(ctrl, "stop")

    @patch("thoughtlink.bridge.mujoco_controller.Controller")
    def test_robot_id(self, MockCtrl):
        from thoughtlink.bridge.mujoco_controller import MuJoCoController

        ctrl = MuJoCoController(robot_id="my_robot")
        assert ctrl.robot_id == "my_robot"

    @patch("thoughtlink.bridge.mujoco_controller.Controller")
    def test_start(self, MockCtrl):
        from thoughtlink.bridge.mujoco_controller import MuJoCoController

        mock_instance = MockCtrl.return_value
        ctrl = MuJoCoController()
        ctrl.start()

        mock_instance.start.assert_called_once()

    @patch("thoughtlink.bridge.mujoco_controller.Controller")
    def test_start_idempotent(self, MockCtrl):
        from thoughtlink.bridge.mujoco_controller import MuJoCoController

        mock_instance = MockCtrl.return_value
        ctrl = MuJoCoController()
        ctrl.start()
        ctrl.start()  # second call should be no-op

        mock_instance.start.assert_called_once()

    @patch("thoughtlink.bridge.mujoco_controller.Controller")
    @patch("thoughtlink.bridge.mujoco_controller.Action")
    def test_execute_valid_actions(self, MockAction, MockCtrl):
        from thoughtlink.bridge.mujoco_controller import MuJoCoController

        mock_instance = MockCtrl.return_value
        ctrl = MuJoCoController()
        ctrl.start()

        for action_str in ("RIGHT", "LEFT", "FORWARD", "STOP"):
            result = ctrl.execute(action_str)
            assert result is True

        assert mock_instance.set_action.call_count == 4

    @patch("thoughtlink.bridge.mujoco_controller.Controller")
    def test_execute_auto_starts(self, MockCtrl):
        from thoughtlink.bridge.mujoco_controller import MuJoCoController

        mock_instance = MockCtrl.return_value
        ctrl = MuJoCoController()
        # Don't call start() explicitly
        ctrl.execute("FORWARD")

        mock_instance.start.assert_called_once()

    @patch("thoughtlink.bridge.mujoco_controller.Controller")
    @patch("thoughtlink.bridge.mujoco_controller.Action")
    def test_stop(self, MockAction, MockCtrl):
        from thoughtlink.bridge.mujoco_controller import MuJoCoController

        mock_instance = MockCtrl.return_value
        ctrl = MuJoCoController()
        ctrl.start()
        ctrl.stop()

        mock_instance.set_action.assert_called()
        mock_instance.stop.assert_called_once()

    @patch("thoughtlink.bridge.mujoco_controller.Controller")
    def test_stop_when_not_started(self, MockCtrl):
        from thoughtlink.bridge.mujoco_controller import MuJoCoController

        mock_instance = MockCtrl.return_value
        ctrl = MuJoCoController()
        ctrl.stop()  # should not raise

        mock_instance.stop.assert_not_called()

    @patch("thoughtlink.bridge.mujoco_controller.Controller")
    def test_works_with_orchestrator(self, MockCtrl):
        """MuJoCoController works as a drop-in for Orchestrator."""
        from thoughtlink.bridge.mujoco_controller import MuJoCoController

        ctrl = MuJoCoController(robot_id="g1_test")
        ctrl.start()

        orch = Orchestrator(controllers=[ctrl])
        step = StepResult(
            timestamp_s=0.0,
            raw_intent="Right Fist",
            stable_intent="Right Fist",
            action="RIGHT",
            confidence=0.9,
            probs=np.array([0.9, 0.025, 0.025, 0.025, 0.025]),
            latency_ms=1.0,
        )
        result = orch.dispatch(step)

        assert result is not None
        assert result.n_robots == 1
        assert result.n_success == 1
