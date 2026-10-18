"""Hierarchical WAM orchestration with a time-aware local repair gate.

This module defines the control contract for a three-level policy:

* a long-horizon planner decomposes a language task into short subtasks;
* an existing WAM policy produces action chunks for the active subtask;
* a small repair policy decides whether a bounded correction is still
  temporally feasible, or whether WAM should be refreshed.

The planner and WAM are protocols on purpose.  The repository does not ship
large model weights, so a real long-horizon model can be connected without
changing the repair or evaluation code.  ``RuleBasedMacroPlanner`` is only a
deterministic fallback for tests and smoke runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

import numpy as np


EVENTS = ("contact", "grasp_close", "release")


@dataclass(frozen=True)
class EventWindow:
    """A planned event in the active WAM subtask timeline."""

    name: str
    expected_time_s: float
    tolerance_s: float = 0.12

    def __post_init__(self) -> None:
        if self.name not in EVENTS:
            raise ValueError(f"unknown event {self.name!r}")
        if self.expected_time_s < 0.0 or self.tolerance_s < 0.0:
            raise ValueError("event times and tolerances must be non-negative")


@dataclass(frozen=True)
class SubtaskSpec:
    """A short horizon that the middle WAM policy can execute."""

    subtask_id: str
    instruction: str
    events: tuple[EventWindow, ...] = ()
    estimated_duration_s: float = 0.0
    success_condition: str = ""

    def __post_init__(self) -> None:
        if not self.subtask_id:
            raise ValueError("subtask_id must be non-empty")
        if self.estimated_duration_s < 0.0:
            raise ValueError("estimated_duration_s must be non-negative")


class LongHorizonPlanner(Protocol):
    """Adapter for the large world-model/LLM-like planner."""

    def plan(self, task: str, observation: Any | None = None,
             memory: Sequence[dict[str, Any]] = ()) -> list[SubtaskSpec]:
        ...


class MediumWAM(Protocol):
    """Adapter for the existing short-horizon OpenWAM policy."""

    def refresh(self, subtask: SubtaskSpec, observation: Any | None = None,
                reason: str = "") -> None:
        ...


class RuleBasedMacroPlanner:
    """Small deterministic fallback; replace with a large model in deployment."""

    _SEPARATORS = (" then ", "然后", ";", "；")

    def plan(self, task: str, observation: Any | None = None,
             memory: Sequence[dict[str, Any]] = ()) -> list[SubtaskSpec]:
        text = " ".join(str(task).strip().split())
        if not text:
            raise ValueError("task must be non-empty")
        parts = [text]
        for separator in self._SEPARATORS:
            pieces: list[str] = []
            for part in parts:
                pieces.extend(part.split(separator))
            parts = pieces
        parts = [part.strip(" ,，") for part in parts if part.strip(" ,，")]
        return [SubtaskSpec(f"subtask_{i + 1}", part)
                for i, part in enumerate(parts)]


@dataclass
class TemporalObservation:
    """Low-dimensional state consumed by the fast time-aware repair head."""

    now_s: float
    residual_m: np.ndarray
    velocity_error_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    phase_error_s: float = 0.0
    events: tuple[EventWindow, ...] = ()
    world_change_probability: float = 0.0
    mechanical_fault_probability: float = 1.0

    def __post_init__(self) -> None:
        self.residual_m = np.asarray(self.residual_m, dtype=np.float64)[:3]
        self.velocity_error_mps = np.asarray(
            self.velocity_error_mps, dtype=np.float64)[:3]
        if self.residual_m.shape != (3,) or self.velocity_error_mps.shape != (3,):
            raise ValueError("residual and velocity error must have shape (3,)")
        if self.now_s < 0.0:
            raise ValueError("now_s must be non-negative")
        self.world_change_probability = float(np.clip(
            self.world_change_probability, 0.0, 1.0))
        self.mechanical_fault_probability = float(np.clip(
            self.mechanical_fault_probability, 0.0, 1.0))


@dataclass(frozen=True)
class RepairDecision:
    """A safe action envelope returned by the fast repair policy."""

    action: str
    reason: str
    time_to_event_s: float | None
    time_required_s: float
    predicted_reduction_m: float
    delta_xyz_m: np.ndarray
    time_scale: float = 1.0
    gripper_time_delta_s: float = 0.0

    def __post_init__(self) -> None:
        if self.action not in ("continue", "local_repair", "wam_replan"):
            raise ValueError(f"unknown repair action {self.action!r}")
        delta = np.asarray(self.delta_xyz_m, dtype=np.float64)
        if delta.shape != (3,):
            raise ValueError("delta_xyz_m must have shape (3,)")
        object.__setattr__(self, "delta_xyz_m", delta)


class TemporalRepairPolicy:
    """Time-window gate around the existing geometric repair controller.

    The policy does not replace WAM or invent a task plan.  It first checks
    whether a correction can finish and be validated before the next critical
    event.  A target/world change has priority over local repair.
    """

    def __init__(self, *, command_latency_s: float = 0.05,
                 settle_s: float = 0.10, verify_s: float = 0.10,
                 safety_margin_s: float = 0.15, residual_threshold_m: float = 0.006,
                 minimum_reduction_m: float = 0.001, world_threshold: float = 0.55,
                 max_phase_error_s: float = 0.30, phase_deadband_s: float = 0.04,
                 kp: float = 0.35, timing_gain: float = 0.65,
                 max_gripper_shift_s: float = 0.12,
                 max_delta_m: float = 0.010) -> None:
        self.command_latency_s = float(command_latency_s)
        self.settle_s = float(settle_s)
        self.verify_s = float(verify_s)
        self.safety_margin_s = float(safety_margin_s)
        self.residual_threshold_m = float(residual_threshold_m)
        self.minimum_reduction_m = float(minimum_reduction_m)
        self.world_threshold = float(world_threshold)
        self.max_phase_error_s = float(max_phase_error_s)
        self.phase_deadband_s = float(phase_deadband_s)
        self.kp = float(kp)
        self.timing_gain = float(timing_gain)
        self.max_gripper_shift_s = float(max_gripper_shift_s)
        self.max_delta_m = float(max_delta_m)
        self.time_required_s = (self.command_latency_s + self.settle_s
                                + self.verify_s + self.safety_margin_s)

    def _next_event(self, obs: TemporalObservation) -> EventWindow | None:
        future = [event for event in obs.events
                  if event.expected_time_s > obs.now_s]
        return min(future, key=lambda event: event.expected_time_s, default=None)

    def decide(self, obs: TemporalObservation) -> RepairDecision:
        zero = np.zeros(3, dtype=np.float64)
        if obs.world_change_probability >= self.world_threshold:
            return RepairDecision("wam_replan", "task_relevant_world_change",
                                  None, self.time_required_s, 0.0, zero)

        event = self._next_event(obs)
        remaining = (event.expected_time_s - obs.now_s) if event else None
        if remaining is None:
            return RepairDecision("wam_replan", "no_known_event_window",
                                  None, self.time_required_s, 0.0, zero)
        if remaining <= self.time_required_s:
            return RepairDecision("wam_replan", "repair_window_closed", remaining,
                                  self.time_required_s, 0.0, zero)
        if abs(float(obs.phase_error_s)) > self.max_phase_error_s:
            return RepairDecision("wam_replan", "phase_error_too_large", remaining,
                                  self.time_required_s, 0.0, zero)

        residual_norm = float(np.linalg.norm(obs.residual_m))
        spatial_fault = residual_norm > self.residual_threshold_m
        temporal_fault = abs(float(obs.phase_error_s)) > self.phase_deadband_s
        attributable = obs.mechanical_fault_probability >= 0.35
        if not attributable or not (spatial_fault or temporal_fault):
            return RepairDecision("continue", "within_tolerance", remaining,
                                  self.time_required_s, 0.0, zero)

        delta = -self.kp * obs.residual_m if spatial_fault else zero.copy()
        norm = float(np.linalg.norm(delta))
        if norm > self.max_delta_m:
            delta *= self.max_delta_m / max(norm, 1e-12)
        predicted = min(residual_norm, float(np.linalg.norm(delta)))
        if spatial_fault and predicted < self.minimum_reduction_m and not temporal_fault:
            return RepairDecision("wam_replan", "insufficient_predicted_reduction",
                                  remaining, self.time_required_s, predicted, zero)
        # Positive phase error means the executed motion is behind WAM's event
        # schedule. Compress the remaining chunk slightly and shift gripper
        # closure by the estimated phase error; both remain tightly bounded.
        time_scale = 1.0
        gripper_delta = 0.0
        if temporal_fault:
            time_scale = float(np.clip(
                1.0 - self.timing_gain * obs.phase_error_s
                / max(remaining, self.time_required_s), 0.80, 1.20))
            gripper_delta = float(np.clip(
                obs.phase_error_s, -self.max_gripper_shift_s,
                self.max_gripper_shift_s))
        return RepairDecision("local_repair", "repairable_before_event", remaining,
                              self.time_required_s, predicted, delta,
                              time_scale, gripper_delta)


@dataclass(frozen=True)
class ControllerCommand:
    """Command sent to the middle WAM/executor."""

    kind: str
    subtask: SubtaskSpec
    payload: dict[str, Any] = field(default_factory=dict)


class HierarchicalWAMController:
    """Three-level coordinator with explicit replan priority."""

    def __init__(self, planner: LongHorizonPlanner, wam: MediumWAM,
                 repair: TemporalRepairPolicy | None = None) -> None:
        self.planner = planner
        self.wam = wam
        self.repair = repair or TemporalRepairPolicy()
        self.subtasks: list[SubtaskSpec] = []
        self.active_index = 0
        self.memory: list[dict[str, Any]] = []

    @property
    def active_subtask(self) -> SubtaskSpec | None:
        if 0 <= self.active_index < len(self.subtasks):
            return self.subtasks[self.active_index]
        return None

    def start(self, task: str, observation: Any | None = None) -> list[SubtaskSpec]:
        self.subtasks = list(self.planner.plan(task, observation, tuple(self.memory)))
        if not self.subtasks:
            raise ValueError("long-horizon planner returned no subtasks")
        self.active_index = 0
        self.memory.append({"event": "plan", "task": task,
                            "subtasks": [item.instruction for item in self.subtasks]})
        return list(self.subtasks)

    def step(self, observation: TemporalObservation,
             *, subtask_complete: bool = False) -> ControllerCommand:
        subtask = self.active_subtask
        if subtask is None:
            raise RuntimeError("controller has not been started or is complete")
        if subtask_complete:
            self.memory.append({"event": "subtask_complete",
                                "subtask_id": subtask.subtask_id})
            self.active_index += 1
            subtask = self.active_subtask
            if subtask is None:
                raise RuntimeError("all subtasks complete")
        decision = self.repair.decide(observation)
        self.memory.append({"event": decision.action, "reason": decision.reason,
                            "subtask_id": subtask.subtask_id,
                            "time_to_event_s": decision.time_to_event_s})
        if decision.action == "wam_replan":
            self.wam.refresh(subtask, observation, decision.reason)
            return ControllerCommand("wam_replan", subtask,
                                     {"reason": decision.reason})
        if decision.action == "local_repair":
            return ControllerCommand("local_repair", subtask,
                                     {"delta_xyz_m": decision.delta_xyz_m.copy(),
                                      "time_scale": decision.time_scale,
                                      "gripper_time_delta_s": decision.gripper_time_delta_s,
                                      "time_to_event_s": decision.time_to_event_s,
                                      "verify_after_s": self.repair.verify_s})
        return ControllerCommand("wam_chunk", subtask,
                                 {"time_to_event_s": decision.time_to_event_s})

