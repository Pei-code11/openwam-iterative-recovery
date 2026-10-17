"""Fast execution-drift estimation and bounded local repair.

The fast path deliberately does not plan a task.  It estimates the short-term
mapping between a nominal Cartesian command and the achieved EEF motion, then
uses a small geometric feedback correction.  A caller supplies the task-stage
and gripper guard so that the module cannot rewrite gripper timing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clip_norm(value: np.ndarray, limit: float) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).copy()
    norm = float(np.linalg.norm(out))
    if norm > float(limit):
        out *= float(limit) / max(norm, 1e-12)
    return out


@dataclass
class ErrorEstimate:
    """Current online plant estimate in physical units."""

    gain: np.ndarray
    bias_m: np.ndarray
    residual_m: np.ndarray
    innovation_m: np.ndarray
    innovation_norm: float
    innovation_baseline: float
    innovation_change: float
    velocity_error_mps: np.ndarray
    confidence: float
    fault_score: float
    innovation_score: float
    gain_drift: float
    bias_drift: float
    drift_score: float
    stable: bool


class FastErrorEstimator:
    """Tiny exponentially weighted system-identification head.

    ``update`` only uses proprioceptive motion and the command issued in the
    previous control tick.  It estimates ``observed_delta ~= gain * command
    + bias`` independently for XYZ.  The confidence term suppresses repair on
    clean transient motion and before enough excitation has been observed.
    """

    def __init__(self, scale_m: np.ndarray, *, alpha: float = 0.22,
                 min_excitation_m: float = 0.0008) -> None:
        self.scale_m = np.maximum(np.asarray(scale_m, dtype=np.float64)[:3], 1e-5)
        self.alpha = float(alpha)
        self.min_excitation_m = float(min_excitation_m)
        self.gain = np.ones(3, dtype=np.float64)
        self.bias_m = np.zeros(3, dtype=np.float64)
        self.residual_m = np.zeros(3, dtype=np.float64)
        self.innovation_m = np.zeros(3, dtype=np.float64)
        self._innovation_fast = 0.0
        self._innovation_baseline = 0.0
        self._innovation_change = 0.0
        self.velocity_error_mps = np.zeros(3, dtype=np.float64)
        self._last_residual = np.zeros(3, dtype=np.float64)
        self._excitation = np.zeros(3, dtype=np.float64)
        self._count = 0
        self._confidence = 0.0
        self._fault_score = 0.0
        self._innovation_score = 0.0
        self._slow_gain = np.ones(3, dtype=np.float64)
        self._slow_bias = np.zeros(3, dtype=np.float64)
        self._reference_gain = np.ones(3, dtype=np.float64)
        self._reference_bias = np.zeros(3, dtype=np.float64)
        self._reference_ready = False
        self._drift_score = 0.0
        self._gain_drift = 0.0
        self._bias_drift = 0.0

    def reset(self) -> None:
        self.gain.fill(1.0)
        self.bias_m.fill(0.0)
        self.residual_m.fill(0.0)
        self.innovation_m.fill(0.0)
        self._innovation_fast = 0.0
        self._innovation_baseline = 0.0
        self._innovation_change = 0.0
        self.velocity_error_mps.fill(0.0)
        self._last_residual.fill(0.0)
        self._excitation.fill(0.0)
        self._count = 0
        self._confidence = 0.0
        self._fault_score = 0.0
        self._innovation_score = 0.0
        self._slow_gain.fill(1.0)
        self._slow_bias.fill(0.0)
        self._reference_gain.fill(1.0)
        self._reference_bias.fill(0.0)
        self._reference_ready = False
        self._drift_score = 0.0
        self._gain_drift = 0.0
        self._bias_drift = 0.0

    def update(self, command_xyz: np.ndarray, observed_delta_m: np.ndarray,
               expected_delta_m: np.ndarray, dt_s: float) -> ErrorEstimate:
        command = np.asarray(command_xyz, dtype=np.float64)[:3]
        observed = np.asarray(observed_delta_m, dtype=np.float64)[:3]
        expected = np.asarray(expected_delta_m, dtype=np.float64)[:3]
        # Do not let a nearly stationary tick dominate the gain estimate.
        excitation = np.abs(expected)
        active = excitation > self.min_excitation_m
        self._excitation = 0.96 * self._excitation + 0.04 * excitation
        self._count += 1
        previous_gain = self.gain.copy()
        previous_bias = self.bias_m.copy()
        if np.any(active):
            # A bounded EWMA is intentionally lighter than a learned action
            # head and remains stable when WAM changes direction.
            ratio = np.divide(observed, expected,
                              out=np.ones(3, dtype=np.float64), where=active)
            residual = expected - observed
            innovation = (observed - expected) / np.maximum(self.scale_m, 1e-5)
            innovation[~active] = 0.0
            self.innovation_m = ((1.0 - self.alpha) * self.innovation_m
                                 + self.alpha * innovation)
            self.gain[active] = np.clip(
                (1.0 - self.alpha) * self.gain[active] + self.alpha * ratio[active],
                0.45, 1.35)
            predicted = self.gain * expected
            bias_sample = observed - predicted
            self.bias_m[active] = ((1.0 - self.alpha) * self.bias_m[active]
                                   + self.alpha * bias_sample[active])
            self.residual_m = ((1.0 - self.alpha) * self.residual_m
                               + self.alpha * residual)
            self.velocity_error_mps = ((1.0 - self.alpha) * self.velocity_error_mps
                                       + self.alpha * (residual - self._last_residual)
                                       / max(float(dt_s), 1e-4))
            self._last_residual = residual
        gain_jump = np.abs(self.gain - previous_gain) if np.any(active) else np.zeros(3)
        bias_jump = (np.abs(self.bias_m - previous_bias)
                     / np.maximum(self.scale_m, 1e-5)) if np.any(active) else np.zeros(3)
        shock = float(np.max(np.maximum(gain_jump, bias_jump)))
        self._innovation_score = 0.80 * self._innovation_score + 0.20 * shock
        innovation_norm = float(np.linalg.norm(self.innovation_m))
        self._innovation_fast = 0.35 * innovation_norm + 0.65 * self._innovation_fast
        if self._count <= 8:
            self._innovation_baseline = (0.75 * self._innovation_baseline
                                          + 0.25 * self._innovation_fast)
        elif self._count < 32:
            # Adapt to the ordinary plant/controller transient before
            # allowing a repair trigger.
            self._innovation_baseline = (0.97 * self._innovation_baseline
                                          + 0.03 * self._innovation_fast)
        self._innovation_change = max(0.0, self._innovation_fast
                                      - self._innovation_baseline)
        # Normal WAM motion also produces a nonzero gain/bias fit error.  A
        # fault is the persistent change from the locally learned motion
        # regime, not the absolute value of that fit.  The slow head follows
        # ordinary dynamics and moves only a little when a gain/bias fault
        # appears, giving a cheap change detector without a task model.
        if self._count <= 8:
            slow_alpha = 0.35
        else:
            # Track ordinary phase and chunk changes quickly enough that a
            # clean rollout does not look like a permanent actuator fault.
            # A genuine gain/bias step still separates the fast and slow
            # estimates for several control ticks.
            slow_alpha = 0.12
        self._slow_gain = ((1.0 - slow_alpha) * self._slow_gain
                           + slow_alpha * self.gain)
        self._slow_bias = ((1.0 - slow_alpha) * self._slow_bias
                           + slow_alpha * self.bias_m)
        if self._count <= 8:
            self._reference_gain = ((1.0 - 0.25) * self._reference_gain
                                    + 0.25 * self.gain)
            self._reference_bias = ((1.0 - 0.25) * self._reference_bias
                                    + 0.25 * self.bias_m)
        elif not self._reference_ready:
            self._reference_ready = True
        self._gain_drift = float(np.max(np.abs(self.gain - self._reference_gain)))
        self._bias_drift = float(np.max(
            np.abs(self.bias_m - self._reference_bias) / np.maximum(self.scale_m, 1e-5)))
        self._drift_score = (0.85 * self._drift_score
                             + 0.15 * max(self._gain_drift, self._bias_drift))
        signal = np.clip(self._excitation / max(self.min_excitation_m, 1e-6), 0.0, 1.0)
        excited_axes = self._excitation > self.min_excitation_m
        signal_level = float(np.mean(signal[excited_axes])) if np.any(excited_axes) else 0.0
        count_gate = np.clip(self._count / 8.0, 0.0, 1.0)
        self._confidence = float(signal_level * count_gate)
        # Require a persistent change relative to the slow regime.  The
        # innovation term catches an abrupt onset; drift catches a sustained
        # loose-joint/calibration error after the onset.
        self._fault_score = float(max(self._drift_score,
                                      self._innovation_change,
                                      0.50 * self._innovation_score))
        stable = bool(self._reference_ready and self._count >= 8 and self._confidence >= 0.35
                      and self._fault_score >= 0.16
                      and (self._gain_drift >= 0.100 or self._bias_drift >= 0.090))
        return self.snapshot(stable=stable)

    def snapshot(self, *, stable: bool | None = None) -> ErrorEstimate:
        if stable is None:
            stable = bool(self._reference_ready and self._count >= 8 and self._confidence >= 0.35
                          and self._fault_score >= 0.16
                          and (self._gain_drift >= 0.100 or self._bias_drift >= 0.090))
        return ErrorEstimate(
            gain=self.gain.copy(), bias_m=self.bias_m.copy(),
            residual_m=self.residual_m.copy(),
            innovation_m=self.innovation_m.copy(),
            innovation_norm=float(np.linalg.norm(self.innovation_m)),
            innovation_baseline=float(self._innovation_baseline),
            innovation_change=float(self._innovation_change),
            velocity_error_mps=self.velocity_error_mps.copy(),
            confidence=float(self._confidence), fault_score=float(self._fault_score),
            innovation_score=float(self._innovation_score),
            gain_drift=float(self._gain_drift), bias_drift=float(self._bias_drift),
            drift_score=float(self._drift_score),
            stable=bool(stable))

    @property
    def confidence(self) -> float:
        return float(getattr(self, "_confidence", 0.0))

    @property
    def fault_score(self) -> float:
        return float(getattr(self, "_fault_score", 0.0))


class GeometricCorrector:
    """Turn a pose error and plant estimate into a bounded XYZ increment."""

    def __init__(self, scale_m: np.ndarray, *, kp: float = 0.35,
                 kd: float = 0.04, max_norm: float = 0.12,
                 feedforward_weight: float = 0.70) -> None:
        self.scale_m = np.maximum(np.asarray(scale_m, dtype=np.float64)[:3], 1e-5)
        self.kp = float(kp)
        self.kd = float(kd)
        self.max_norm = float(max_norm)
        self.feedforward_weight = float(feedforward_weight)

    def action_delta(self, nominal_action_xyz: np.ndarray,
                     tracking_error_m: np.ndarray,
                     estimate: ErrorEstimate,
                     *, allow_feedforward: bool = True) -> np.ndarray:
        nominal = np.asarray(nominal_action_xyz, dtype=np.float64)[:3]
        error = np.asarray(tracking_error_m, dtype=np.float64)[:3]
        # If observed = gain * command + bias, this is the command increment
        # that brings the achieved motion back to the nominal WAM motion.
        gain = np.clip(estimate.gain, 0.55, 1.25)
        bias_norm = estimate.bias_m / self.scale_m
        feedforward = ((1.0 / gain) - 1.0) * nominal - bias_norm / gain
        feedback = self.kp * error / self.scale_m
        feedback -= self.kd * estimate.velocity_error_mps / self.scale_m
        if allow_feedforward and estimate.confidence >= 0.35:
            delta = self.feedforward_weight * feedforward + (1.0 - self.feedforward_weight) * feedback
        else:
            delta = feedback
        return _clip_norm(delta, self.max_norm)


class RepairValidator:
    """Stateful verify-then-continue gate for a short repair segment."""

    def __init__(self, *, min_reduction_m: float = 0.0010,
                 max_failed_checks: int = 2) -> None:
        self.min_reduction_m = float(min_reduction_m)
        self.max_failed_checks = int(max_failed_checks)
        self.failed_checks = 0
        self.attempts = 0

    def start(self) -> None:
        self.failed_checks = 0
        self.attempts += 1

    def observe(self, before_m: float, after_m: float) -> tuple[bool, bool]:
        improved = float(after_m) <= float(before_m) - self.min_reduction_m
        if improved:
            self.failed_checks = 0
        else:
            self.failed_checks += 1
        stop = self.failed_checks >= self.max_failed_checks
        return bool(improved), bool(stop)

