import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from openwam_recovery.hierarchical_wam import (
    EventWindow,
    HierarchicalWAMController,
    RuleBasedMacroPlanner,
    TemporalObservation,
    TemporalRepairPolicy,
)


class _WAM:
    def __init__(self):
        self.refreshes = []

    def refresh(self, subtask, observation=None, reason=""):
        self.refreshes.append((subtask.subtask_id, reason))


def _obs(now=0.0, *, world=0.0, phase=0.0, residual=0.02):
    return TemporalObservation(
        now_s=now,
        residual_m=np.asarray([residual, 0.0, 0.0]),
        phase_error_s=phase,
        events=(EventWindow("contact", 1.0), EventWindow("grasp_close", 2.0)),
        world_change_probability=world,
        mechanical_fault_probability=1.0,
    )


def test_macro_planner_creates_short_horizons():
    subtasks = RuleBasedMacroPlanner().plan("pick up the cup then place it on the tray")
    assert [item.instruction for item in subtasks] == [
        "pick up the cup", "place it on the tray"
    ]


def test_world_change_has_priority_over_local_repair():
    policy = TemporalRepairPolicy()
    decision = policy.decide(_obs(world=0.9))
    assert decision.action == "wam_replan"
    assert decision.reason == "task_relevant_world_change"


def test_repair_is_rejected_after_deadline():
    policy = TemporalRepairPolicy()
    decision = policy.decide(_obs(now=0.65))
    assert decision.action == "wam_replan"
    assert decision.reason == "repair_window_closed"


def test_controller_routes_replan_to_middle_wam():
    wam = _WAM()
    controller = HierarchicalWAMController(RuleBasedMacroPlanner(), wam)
    controller.start("pick up the cup then place it on the tray")
    command = controller.step(_obs(world=0.9))
    assert command.kind == "wam_replan"
    assert wam.refreshes == [("subtask_1", "task_relevant_world_change")]


def test_temporal_repair_returns_bounded_timing_adjustment():
    policy = TemporalRepairPolicy()
    decision = policy.decide(_obs(phase=0.18))
    assert decision.action == "local_repair"
    assert 0.80 <= decision.time_scale < 1.0
    assert 0.0 < decision.gripper_time_delta_s <= 0.12

