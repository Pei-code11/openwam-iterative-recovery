"""Long-horizon timing stress test for the three-level WAM design.

This is a deterministic, lightweight simulator for architecture ablations. It
does not load OpenWAM weights and must not be described as an official LIBERO
or real-robot result. Every method receives the same generated long-horizon
episodes, including the same target changes and execution faults.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openwam_recovery.hierarchical_wam import (
    EventWindow,
    TemporalObservation,
    TemporalRepairPolicy,
)


CONDITIONS = (
    "clean",
    "arm_offset",
    "arm_speed_loss",
    "arm_speed_offset",
    "target_shift",
    "distractor_shift",
    "wam_plan_failure",
)
METHODS = (
    "wam_only",
    "local_no_time",
    "temporal_repair",
    "fixed_target_replan",
    "three_level_hybrid",
)


@dataclass(frozen=True)
class Fault:
    condition: str
    subtask_index: int | None
    bias_m: np.ndarray
    speed_gain: float
    timing_offset_s: float
    inject_s: float


@dataclass
class Episode:
    episode_id: int
    n_subtasks: int
    fault: Fault


@dataclass
class Rollout:
    success: bool
    subtask_successes: int
    repairs: int = 0
    validated_repairs: int = 0
    repair_window_hits: int = 0
    repair_window_misses: int = 0
    late_repairs: int = 0
    repair_harms: int = 0
    scene_replans: int = 0
    macro_replans: int = 0
    gripper_timing_error_s: float = 0.0
    first_repair_delay_s: float | None = None
    target_detection_delay_s: float | None = None


def _wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.96
    rate = successes / total
    den = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / den
    radius = z * math.sqrt(rate * (1.0 - rate) / total
                           + z * z / (4.0 * total * total)) / den
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _make_fault(condition: str, subtask_index: int | None,
                rng: np.random.Generator) -> Fault:
    if condition == "arm_offset":
        bias = rng.normal(0.0, 0.018, 3)
        return Fault(condition, subtask_index, bias, 1.0,
                     float(rng.uniform(0.0, 0.08)), float(rng.uniform(0.45, 0.85)))
    if condition == "arm_speed_loss":
        return Fault(condition, subtask_index, np.zeros(3),
                     float(rng.uniform(0.62, 0.78)),
                     float(rng.uniform(0.18, 0.42)), float(rng.uniform(0.40, 0.75)))
    if condition == "arm_speed_offset":
        return Fault(condition, subtask_index, rng.normal(0.0, 0.014, 3),
                     float(rng.uniform(0.62, 0.78)),
                     float(rng.uniform(0.22, 0.48)), float(rng.uniform(0.40, 0.75)))
    if condition in ("target_shift", "distractor_shift"):
        return Fault(condition, subtask_index, np.zeros(3), 1.0,
                     0.0, float(rng.uniform(0.45, 0.90)))
    if condition == "wam_plan_failure":
        return Fault(condition, subtask_index, np.zeros(3), 1.0, 0.0, 0.0)
    return Fault("clean", None, np.zeros(3), 1.0, 0.0, 0.0)


def build_episodes(episodes_per_condition: int, seed: int) -> list[Episode]:
    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []
    episode_id = 0
    for condition in CONDITIONS:
        for _ in range(int(episodes_per_condition)):
            n_subtasks = int(rng.integers(3, 7))
            subtask_index = (None if condition == "clean"
                             else int(rng.integers(0, n_subtasks)))
            episodes.append(Episode(
                episode_id=episode_id,
                n_subtasks=n_subtasks,
                fault=_make_fault(condition, subtask_index, rng)))
            episode_id += 1
    return episodes


def _fault_active(ep: Episode, subtask_index: int) -> bool:
    return ep.fault.subtask_index == subtask_index


def _rollout(ep: Episode, method: str, seed: int) -> Rollout:
    rng = np.random.default_rng(seed)
    # Draw all episode randomness before method-specific branching so every
    # method sees exactly the same latent episode and paired comparisons are
    # meaningful.
    base_residuals = rng.normal(0.0, 0.0035, (ep.n_subtasks, 3))
    base_phases = rng.normal(0.0, 0.025, ep.n_subtasks)
    base_failures = rng.random(ep.n_subtasks) < 0.028
    false_positives = rng.random(ep.n_subtasks) < 0.12
    detection_delays = rng.uniform(0.08, 0.22, ep.n_subtasks)
    gripper_noise = rng.normal(0.0, 0.012, ep.n_subtasks)
    policy = TemporalRepairPolicy(
        command_latency_s=0.05, settle_s=0.10, verify_s=0.10,
        safety_margin_s=0.15, residual_threshold_m=0.006,
        minimum_reduction_m=0.001, max_phase_error_s=0.30,
        max_delta_m=0.012,
    )
    out = Rollout(success=True, subtask_successes=0)
    contact_t, close_t, release_t = 1.20, 1.65, 2.70
    for subtask_index in range(ep.n_subtasks):
        active_fault = _fault_active(ep, subtask_index)
        condition = ep.fault.condition if active_fault else "clean"
        # Small nominal tracking noise is independent of the fault injection.
        residual = base_residuals[subtask_index].copy()
        phase_error = float(base_phases[subtask_index])
        target_stale = False
        base_failure = bool(base_failures[subtask_index])
        late_harm = False
        local_attempted = False
        validated = False
        target_detection_delay = None

        if active_fault:
            residual += ep.fault.bias_m
            phase_error += ep.fault.timing_offset_s
            if condition in ("target_shift", "distractor_shift"):
                target_stale = condition == "target_shift"
            if condition == "wam_plan_failure":
                base_failure = True

        # A plan failure can only be repaired by the long-horizon controller.
        if base_failure and condition == "wam_plan_failure":
            if method == "three_level_hybrid" and rng.random() < 0.86:
                out.macro_replans += 1
                base_failure = False
            else:
                out.subtask_successes += 0
                out.success = False
                continue

        detection_delay = float(detection_delays[subtask_index])
        detect_t = ep.fault.inject_s + detection_delay if active_fault else 0.0
        if active_fault and condition == "target_shift":
            target_detection_delay = detection_delay

        if active_fault and condition in ("arm_offset", "arm_speed_loss", "arm_speed_offset"):
            # The fast estimator is reliable on true execution faults, but a
            # small false-positive probability remains on clean motion.
            mechanical_prob = 0.88
            obs_world = 0.0
        elif active_fault and condition == "target_shift":
            mechanical_prob = 0.30
            obs_world = 0.92 if method == "three_level_hybrid" else 0.0
        elif active_fault and condition == "distractor_shift":
            mechanical_prob = 0.24
            obs_world = 0.0
        else:
            false_positive = bool(false_positives[subtask_index])
            mechanical_prob = 0.62 if false_positive else 0.18
            obs_world = 0.0

        if method in ("temporal_repair", "three_level_hybrid"):
            events = (EventWindow("contact", contact_t),
                      EventWindow("grasp_close", close_t),
                      EventWindow("release", release_t))
            obs = TemporalObservation(
                now_s=detect_t,
                residual_m=residual,
                velocity_error_mps=np.asarray([phase_error, 0.0, 0.0]),
                phase_error_s=phase_error,
                events=events,
                world_change_probability=obs_world,
                mechanical_fault_probability=mechanical_prob,
            )
            decision = policy.decide(obs)
            if decision.action == "wam_replan":
                if condition == "target_shift" and method == "three_level_hybrid":
                    out.scene_replans += 1
                    target_stale = False
                    residual *= 0.18
                    phase_error *= 0.55
                    target_detection_delay = detection_delay
                elif decision.reason == "repair_window_closed":
                    out.repair_window_misses += 1
            elif decision.action == "local_repair":
                local_attempted = True
                out.repairs += 1
                if out.first_repair_delay_s is None:
                    out.first_repair_delay_s = detect_t
                out.repair_window_hits += 1
                # The time-aware gate guarantees enough time before contact.
                residual_after = residual + decision.delta_xyz_m
                if condition in ("arm_offset", "arm_speed_offset"):
                    residual_after *= 0.38
                elif condition == "arm_speed_loss":
                    residual_after *= 0.48
                    phase_error *= 0.62
                else:
                    # False repairs or distractor changes have no reliable
                    # plant error to correct and can disturb the trajectory.
                    residual_after *= 1.05
                    phase_error += 0.025
                # A temporal repair also applies its bounded action-rate and
                # gripper-time adjustment. This is the part missing from the
                # old spatial-only small repair.
                if method in ("temporal_repair", "three_level_hybrid"):
                    phase_error = phase_error * decision.time_scale - decision.gripper_time_delta_s
                improved = (np.linalg.norm(residual_after)
                            <= np.linalg.norm(residual) - 0.001)
                if improved:
                    validated = True
                    out.validated_repairs += 1
                    residual = residual_after
                else:
                    out.repair_harms += 1
                    residual = residual_after

        elif method == "local_no_time":
            # Old policy: it reacts to residual magnitude and ignores the
            # contact/close deadline.
            trigger = (np.linalg.norm(residual) > 0.006
                       or (active_fault and condition.startswith("arm_")))
            if trigger:
                local_attempted = True
                out.repairs += 1
                out.first_repair_delay_s = detect_t
                if detect_t + 0.40 <= contact_t:
                    out.repair_window_hits += 1
                    residual *= 0.45 if condition.startswith("arm_") else 1.04
                    if condition == "arm_speed_loss":
                        phase_error *= 0.78
                    validated = condition.startswith("arm_")
                    out.validated_repairs += int(validated)
                else:
                    out.repair_window_misses += 1
                    out.late_repairs += 1
                    out.repair_harms += 1
                    late_harm = True
                    residual *= 1.22
                    phase_error += 0.16

        elif method == "fixed_target_replan":
            if active_fault and condition == "target_shift":
                out.scene_replans += 1
                target_stale = False
                residual *= 0.18
                phase_error *= 0.55
                target_detection_delay = detection_delay

        if target_detection_delay is not None:
            out.target_detection_delay_s = target_detection_delay

        # A late correction or a stale target invalidates the subtask. The
        # close event uses phase error as a proxy for gripper timing error.
        gripper_error = abs(phase_error) + float(gripper_noise[subtask_index])
        out.gripper_timing_error_s += abs(gripper_error)
        subtask_ok = (
            not base_failure
            and not target_stale
            and not late_harm
            and float(np.linalg.norm(residual)) <= 0.030
            and abs(phase_error) <= 0.18
            and abs(gripper_error) <= 0.20
        )
        if subtask_ok:
            out.subtask_successes += 1
        else:
            out.success = False

    out.gripper_timing_error_s /= max(ep.n_subtasks, 1)
    out.target_detection_delay_s = (None if out.target_detection_delay_s is None
                                    else float(out.target_detection_delay_s))
    return out


def run(episodes_per_condition: int, seed: int) -> dict[str, object]:
    episodes = build_episodes(episodes_per_condition, seed)
    rows: list[dict[str, object]] = []
    rollouts: dict[tuple[int, str], Rollout] = {}
    for ep in episodes:
        for method_index, method in enumerate(METHODS):
            rollout = _rollout(ep, method, seed + ep.episode_id * 101)
            rollouts[(ep.episode_id, method)] = rollout
            rows.append({"episode_id": ep.episode_id, "condition": ep.fault.condition,
                         "n_subtasks": ep.n_subtasks, "method": method,
                         "success": rollout.success,
                         "subtask_successes": rollout.subtask_successes,
                         "repairs": rollout.repairs,
                         "validated_repairs": rollout.validated_repairs,
                         "repair_window_hits": rollout.repair_window_hits,
                         "repair_window_misses": rollout.repair_window_misses,
                         "late_repairs": rollout.late_repairs,
                         "repair_harms": rollout.repair_harms,
                         "scene_replans": rollout.scene_replans,
                         "macro_replans": rollout.macro_replans,
                         "gripper_timing_error_s": rollout.gripper_timing_error_s,
                         "first_repair_delay_s": rollout.first_repair_delay_s,
                         "target_detection_delay_s": rollout.target_detection_delay_s})

    aggregate: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for method in METHODS:
            group = [row for row in rows
                     if row["condition"] == condition and row["method"] == method]
            n = len(group)
            successes = sum(bool(row["success"]) for row in group)
            aggregate.append({
                "condition": condition, "method": method, "n": n,
                "successes": successes, "success_rate": successes / n,
                "wilson_95": _wilson(successes, n),
                "subtask_success_rate": sum(float(row["subtask_successes"])
                                             for row in group)
                / sum(int(row["n_subtasks"]) for row in group),
                "repair_attempt_rate": sum(int(row["repairs"]) > 0 for row in group) / n,
                "mean_repairs": float(np.mean([row["repairs"] for row in group])),
                "validated_repair_rate": (
                    sum(int(row["validated_repairs"]) for row in group)
                    / max(sum(int(row["repairs"]) for row in group), 1)),
                "repair_window_hit_rate": (
                    sum(int(row["repair_window_hits"]) for row in group)
                    / max(sum(int(row["repairs"]) + int(row["repair_window_misses"])
                              for row in group), 1)),
                "late_repair_rate": sum(int(row["late_repairs"]) > 0 for row in group) / n,
                "repair_harm_rate": sum(int(row["repair_harms"]) for row in group)
                / n,
                "mean_scene_replans": float(np.mean([row["scene_replans"] for row in group])),
                "mean_macro_replans": float(np.mean([row["macro_replans"] for row in group])),
                "mean_gripper_timing_error_s": float(np.mean([
                    row["gripper_timing_error_s"] for row in group])),
                "median_first_repair_delay_s": (
                    float(np.median([row["first_repair_delay_s"] for row in group
                                     if row["first_repair_delay_s"] is not None]))
                    if any(row["first_repair_delay_s"] is not None for row in group)
                    else None),
                "median_target_detection_delay_s": (
                    float(np.median([row["target_detection_delay_s"] for row in group
                                     if row["target_detection_delay_s"] is not None]))
                    if any(row["target_detection_delay_s"] is not None for row in group)
                    else None),
            })

    paired: list[dict[str, object]] = []
    for condition in CONDITIONS:
        base = {ep.episode_id: rollouts[(ep.episode_id, "wam_only")]
                for ep in episodes if ep.fault.condition == condition}
        for method in METHODS[1:]:
            rescues = harms = 0
            for ep in episodes:
                if ep.fault.condition != condition:
                    continue
                b = base[ep.episode_id].success
                m = rollouts[(ep.episode_id, method)].success
                rescues += int((not b) and m)
                harms += int(b and (not m))
            paired.append({"condition": condition, "method": method,
                           "baseline": "wam_only", "pairs": len(base),
                           "rescues": rescues, "harms": harms,
                           "net": rescues - harms})
    return {"protocol": {"episodes_per_condition": episodes_per_condition,
                          "conditions": list(CONDITIONS), "methods": list(METHODS),
                          "seed": seed,
                          "description": "custom long-horizon timing stress simulation; not official LIBERO or real robot"},
            "aggregate": aggregate, "paired": paired, "rows": rows}


def write_report(result: dict[str, object], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Hierarchical WAM timing stress benchmark", "",
             "Custom deterministic long-horizon simulation; not official LIBERO or a real-robot result.", "",
             "| Condition | Method | Success | Subtask SR | Repair valid | Window hit | Late repair | Replan | Gripper timing error (s) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["aggregate"]:
        lines.append(f"| {row['condition']} | {row['method']} | {row['successes']}/{row['n']} ({row['success_rate']:.1%}) | "
                     f"{row['subtask_success_rate']:.1%} | {row['validated_repair_rate']:.1%} | "
                     f"{row['repair_window_hit_rate']:.1%} | {row['late_repair_rate']:.1%} | "
                     f"{row['mean_scene_replans'] + row['mean_macro_replans']:.2f} | "
                     f"{row['mean_gripper_timing_error_s']:.3f} |")
    lines += ["", "## Paired rescue/harm versus wam_only", "",
              "| Condition | Method | Rescue | Harm | Net |", "|---|---|---:|---:|---:|"]
    for row in result["paired"]:
        lines.append(f"| {row['condition']} | {row['method']} | {row['rescues']} | {row['harms']} | {row['net']} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-condition", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/hierarchical_timing_benchmark"))
    args = parser.parse_args()
    result = run(args.episodes_per_condition, args.seed)
    write_report(result, args.output)
    for row in result["aggregate"]:
        print(f"{row['condition']:18s} {row['method']:20s} "
              f"SR={row['success_rate']:.3f} sub={row['subtask_success_rate']:.3f} "
              f"valid={row['validated_repair_rate']:.3f} late={row['late_repair_rate']:.3f}")


if __name__ == "__main__":
    main()
