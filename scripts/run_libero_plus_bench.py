"""Paired LIBERO test of execution-chain drift repair and task-aware replanning.

Execution faults are a downstream plant model (Cartesian command gain and
millimeter-per-step bias), leaving WAM's nominal action and the scene unchanged.
Scene-change detection uses simulator object poses as an oracle.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from openwam_recovery.fast_repair import (FastErrorEstimator,
                                           GeometricCorrector,
                                           RepairValidator)


class _LocalTask:
    def __init__(self, name, language, problem_folder="libero_10"):
        self.name = name
        self.language = language
        self.problem = "Libero"
        self.problem_folder = problem_folder
        self.bddl_file = f"{name}.bddl"
        self.init_states_file = f"{name}.pruned_init"


class _LocalSuite:
    """Small LIBERO suite adapter that avoids metadata-time EGL startup."""
    def __init__(self, task_ids):
        from libero.libero import get_libero_path
        root = Path(get_libero_path("bddl_files"))
        map_file = Path(os.environ["LIBERO_PATH"]) / "libero" / "libero" / "benchmark" / "libero_suite_task_map.py"
        source = map_file.read_text(encoding="utf-8")
        task_map = ast.literal_eval(source.split("libero_task_map = ", 1)[1])["libero_10"]
        prompt_map = {}
        prompt_file = os.environ.get("WAM_PROMPTS_JSON")
        if prompt_file and Path(prompt_file).is_file():
            records = json.loads(Path(prompt_file).read_text())
            for task_id, record in zip((287, 677, 1099, 1932, 2092), records):
                prompt_map[task_id] = record["task_language"]
        self.tasks = {}
        for task_id in task_ids:
            name = task_map[int(task_id)]
            self.tasks[int(task_id)] = _LocalTask(name, prompt_map.get(int(task_id), name.replace("_", " ")))
        self._root = root

    def get_task(self, task_id):
        return self.tasks[int(task_id)]

    def get_task_init_states(self, task_id):
        import torch
        task = self.get_task(task_id)
        path = self._root.parent / "init_files" / task.problem_folder
        base = task.name
        for marker in ("_view_", "_noise_", "_initstate_", "_level", "_add_", "_table_", "_tb_"):
            if marker in base:
                base = base.split(marker, 1)[0]
        return torch.load(path / f"{base}.pruned_init", weights_only=False)


CONDITIONS = (
    "clean", "arm_speed_loss", "arm_offset", "arm_speed_offset",
    "target_shift", "distractor_shift",
)
PAIRED_METHODS = {
    "clean": ("wam", "hybrid"),
    "arm_speed_loss": ("fault_only", "hybrid"),
    "arm_offset": ("fault_only", "hybrid"),
    "arm_speed_offset": ("fault_only", "hybrid"),
    "target_shift": ("stale_plan", "local_only", "hybrid", "fixed_refresh"),
    "distractor_shift": ("stale_plan", "hybrid", "fixed_refresh"),
}
FAULT_STEPS = (8, 80, 160)
# Scene changes are injected while the arm is still in the approach phase so
# that the benchmark measures task-aware replanning, not recovery from an
# object already grasped by the robot.
SCENE_STEPS = (15, 15, 15)
# Loose-joint / calibration drift persists for the remainder of the task.
FAULT_DURATION = 10_000
SCENE_SHIFT_MAGNITUDES_M = (0.04, 0.08)
DISTRACTOR_SHIFT_MAGNITUDES_M = (0.08, 0.12)
SCENE_CHANGE_THRESHOLD_M = 0.03
CONTROL_PERIOD_S = 0.05
MAX_CORRECTION_NORM = 0.20
CORRECTION_GAIN = 0.65
VALIDATE_EVERY = 4
MAX_NO_IMPROVEMENT = 2
HARD_GAIN = 0.72
HARD_BIAS_M = 0.065
HARD_TARGET_SHIFT_M = 0.095
HARD_DISTRACTOR_SHIFT_M = 0.12


def _sha(value) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _write_libero_config() -> None:
    import yaml

    repo = Path(os.environ["LIBERO_PATH"]).expanduser().resolve()
    candidates = (repo, repo / "libero", repo / "libero" / "libero")
    package_root = next((p for p in candidates if (p / "bddl_files").is_dir()
                         and (p / "init_files").is_dir()), None)
    if package_root is None:
        raise FileNotFoundError(f"LIBERO assets not found below {repo}")
    config_root = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero-hybrid"))
    config_root.mkdir(parents=True, exist_ok=True)
    config_file = config_root / "config.yaml"
    config = {"benchmark_root": str(package_root), "bddl_files": str(package_root / "bddl_files"),
              "init_states": str(package_root / "init_files"), "datasets": str(repo / "datasets"),
              "assets": str(package_root / "assets")}
    if config_file.exists():
        import yaml
        previous = yaml.safe_load(config_file.read_text()) or {}
        # The public asset merge intentionally uses a separate runtime asset
        # root while keeping the package's BDDL and init-state paths. Reuse
        # that validated config rather than replacing its asset entry.
        for key in ("benchmark_root", "bddl_files", "init_states"):
            if previous.get(key) != config[key]:
                raise RuntimeError(f"refusing to overwrite incompatible LIBERO config: {config_file}")
    else:
        config_file.write_text(yaml.safe_dump(config, sort_keys=True))


def _bddl_objects(bddl: Path) -> tuple[list[str], list[str]]:
    """Read movable object symbols from this task's BDDL, without a task map."""
    lines = bddl.read_text(encoding="utf-8").splitlines()
    objects: list[str] = []
    interests: list[str] = []
    section = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("(:objects"):
            section = "objects"
            continue
        if stripped.startswith("(:obj_of_interest"):
            section = "interest"
            continue
        if section and stripped == ")":
            section = None
            continue
        if section == "objects":
            match = re.match(r"([A-Za-z0-9_]+)\s+-\s+", stripped)
            if match:
                objects.append(match.group(1))
        elif section == "interest" and stripped and not stripped.startswith(";"):
            interests.append(stripped.split()[0])
    return objects, interests


def _resolve_bddl_path(root: Path, task) -> Path:
    """Resolve Plus perturbation variants to their shared base BDDL file."""
    exact = root / task.problem_folder / task.bddl_file
    if exact.is_file():
        return exact
    name = task.bddl_file.removesuffix(".bddl")
    candidates = [name]
    # Plus encodes camera/init/noise/layout variants in task names while the
    # scene/problem BDDL is shared with the base task.
    for marker in ("_view_", "_noise_", "_initstate_", "_level", "_add_", "_table_", "_tb_"):
        if marker in name:
            candidates.append(name.split(marker, 1)[0])
    for candidate in candidates:
        path = root / task.problem_folder / f"{candidate}.bddl"
        if path.is_file():
            return path
    raise FileNotFoundError(f"no base BDDL found for {task.bddl_file}")


def _infer_task_objects(env, bddl: Path) -> dict[str, str]:
    """Choose the task object and a movable distractor from the live model."""
    objects, interests = _bddl_objects(bddl)
    joint_names = set(getattr(env.env.sim.model, "joint_names", ()))

    def free_joint(symbol: str) -> str | None:
        for candidate in (f"{symbol}_joint0", symbol):
            if candidate in joint_names:
                try:
                    q0, q1 = env.env.sim.model.get_joint_qpos_addr(candidate)
                    v0, v1 = env.env.sim.model.get_joint_qvel_addr(candidate)
                    if q1 - q0 == 7 and v1 - v0 == 6:
                        return candidate
                except (KeyError, ValueError):
                    pass
        return None

    target = next((joint for symbol in interests if (joint := free_joint(symbol))), None)
    if target is None:
        target = next((joint for symbol in objects if (joint := free_joint(symbol))), None)
    if target is None:
        raise RuntimeError(f"could not identify a movable task object in {bddl}")
    target_symbol = target.removesuffix("_joint0")
    distractor = next((joint for symbol in objects if symbol != target_symbol
                       and (joint := free_joint(symbol))), None)
    if distractor is None:
        # A negative-control distractor is optional for single-object tasks.
        distractor = target
    return {"target": target, "distractor": distractor,
            "target_symbol": target_symbol, "objects": objects}


def _joint_pos(env, joint_name: str) -> np.ndarray:
    q0, q1 = env.env.sim.model.get_joint_qpos_addr(joint_name)
    if q1 - q0 != 7:
        raise ValueError(f"{joint_name} is not a free object joint")
    return np.asarray(env.env.sim.data.qpos[int(q0):int(q0) + 3], dtype=np.float64).copy()


def _shift_object(env, joint_name: str, delta: np.ndarray) -> None:
    sim = env.env.sim
    q0, q1 = sim.model.get_joint_qpos_addr(joint_name)
    v0, v1 = sim.model.get_joint_qvel_addr(joint_name)
    if q1 - q0 != 7 or v1 - v0 != 6:
        raise ValueError(f"{joint_name} is not a free object joint")
    sim.data.qpos[int(q0):int(q0) + 3] += np.asarray(delta, dtype=np.float64)
    sim.data.qvel[int(v0):int(v1)] = 0
    sim.forward()
    env._update_observables(force=True)


def _xy_vector(task_id: int, trial: int, condition: str, magnitude: float) -> np.ndarray:
    # Fault direction is deterministically paired, including arm-bias runs
    # whose label is intentionally absent from the scene-condition table.
    condition_index = next((i for i, name in enumerate(CONDITIONS) if condition.startswith(name)), 0)
    seed = 20261013 + task_id * 100000 + trial * 100 + condition_index
    angle = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))
    return np.asarray([magnitude * np.cos(angle), magnitude * np.sin(angle), 0.0], np.float64)


def _scenario_seed(task_id: int, trial: int, condition: str) -> int:
    condition_index = next((i for i, name in enumerate(CONDITIONS) if condition.startswith(name)), 0)
    return 20261015 + task_id * 100000 + trial * 100 + condition_index


def _scenario_parameters(task_id: int, trial: int, condition: str,
                         trial_slot: int | None = None,
                         difficulty: str = "standard") -> dict:
    rng = np.random.default_rng(_scenario_seed(task_id, trial, condition))
    phase_index = trial % len(FAULT_STEPS)
    injection_steps = SCENE_STEPS if condition in ("target_shift", "distractor_shift") else FAULT_STEPS
    injection_step = int(injection_steps[phase_index])
    hard = difficulty == "hard"
    target_m = (HARD_TARGET_SHIFT_M if hard else
                (0.05 if (trial_slot if trial_slot is not None else trial) < 10 else 0.075))
    distractor_m = HARD_DISTRACTOR_SHIFT_M if hard else 0.06
    bias_magnitude = (HARD_BIAS_M if hard else 0.04) if condition in ("arm_offset", "arm_speed_offset") else 0.0
    gain = HARD_GAIN if hard and condition in ("arm_speed_loss", "arm_speed_offset") else (
        0.85 if condition in ("arm_speed_loss", "arm_speed_offset") else 1.0)
    scene_event = condition in ("target_shift", "distractor_shift")
    return {"injection_step": injection_step,
            "injection_phase": "early_approach" if scene_event else ("early", "middle", "late")[phase_index],
            "target_shift_magnitude": target_m,
            "distractor_shift_magnitude": distractor_m,
            "bias_action_magnitude": bias_magnitude,
            "plant_gain": gain, "difficulty": difficulty}


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _norm_limit(value: np.ndarray, limit: float) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).copy()
    norm = float(np.linalg.norm(out))
    return out if norm <= limit else out * (limit / max(norm, 1e-12))


def run_rollout(args, suite, task_id: int, trial: int, condition: str, method: str,
                threshold_m: float, trial_slot: int | None = None) -> dict:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openwam2libero_interface import OpenWAMLiberoPolicy

    task = suite.get_task(task_id)
    bddl = _resolve_bddl_path(Path(get_libero_path("bddl_files")), task)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    policy = None
    try:
        env.seed(42 + task_id)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(task_id)[trial])
        for _ in range(30):
            obs, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, -1], np.float32))

        controller = env.env.robots[0].controller
        scale_m = np.asarray(controller.output_max, dtype=np.float64)[:3].copy()
        scale_source = "controller.output_max"
        if getattr(args, "scale_overrides", None) and str(task_id) in args.scale_overrides:
            scale_m = np.asarray(args.scale_overrides[str(task_id)], dtype=np.float64).reshape(-1)[:3]
            scale_source = "frozen_clean_calibration_ols"
        if np.any(scale_m <= 0) or not np.isfinite(scale_m).all():
            raise RuntimeError(f"invalid EEF translation scales: {scale_m}")
        task_objects = _infer_task_objects(env, bddl)
        target_name, distractor_name = task_objects["target"], task_objects["distractor"]
        # Fail early if an object mapping is invalid for this benchmark task.
        _joint_pos(env, target_name)
        _joint_pos(env, distractor_name)

        policy = OpenWAMLiberoPolicy(host=args.wam_host, port=args.wam_port, request_timeout=300,
                                     action_mode="eef", image_transform="rotate_180", send_state=True,
                                     state_dim=10, action_dim=7)
        policy.reset()
        response_holder = {}
        original_predict = policy._client.predict
        world_replan_pending_at = None

        def capture(payload):
            result = original_predict(payload)
            response_holder.clear()
            response_holder.update(result)
            return result

        policy._client.predict = capture
        initial_target = _joint_pos(env, target_name)
        target_reference = initial_target.copy()
        scenario = _scenario_parameters(task_id, trial, condition, trial_slot,
                                         getattr(args, "difficulty", "standard"))
        mapped = getattr(args, "scene_scenarios", {}).get(str(task_id), {}).get(str(trial), {})
        scenario.update(mapped)
        injection_step = scenario["injection_step"]
        target_shift = np.asarray([scenario["target_shift_magnitude"], 0.0, 0.0], dtype=np.float64)
        distractor_shift = np.asarray([scenario["distractor_shift_magnitude"], 0.0, 0.0], dtype=np.float64)
        bias = _xy_vector(task_id, trial, condition + "_bias", scenario["bias_action_magnitude"])
        # Use one randomly selected table-plane axis for a transparent action-space fault.
        axis_hint = bias if np.linalg.norm(bias) > 0 else _xy_vector(task_id, trial, condition + "_axis", 1.0)
        bias_axis = 0 if abs(axis_hint[0]) >= abs(axis_hint[1]) else 1
        bias = np.eye(3, dtype=np.float64)[bias_axis] * np.sign(bias[bias_axis]) * np.linalg.norm(bias[:2])
        scene_injected = False
        scene_injected_at = None
        plant_fault_active = False
        tracking_reference = None
        last_generation = None
        repair_active = False
        repair_start_error = None
        repair_start_step = None
        no_improvement = 0
        correction_segments = []
        active_segment = None
        local_corrections = 0
        trigger_history = []
        scene_replans = 0
        fixed_refreshes = 0
        irrelevant_scene_changes = 0
        target_change_detected = False
        irrelevant_change_ignored = False
        scene_gate_at_injection = None
        target_change_ms = None
        first_replan_action_ms = None
        correction_fallback_replans = 0
        first_local_command_ms = None
        first_fault_time = None
        fault_steps_applied = 0
        intervention_events = []
        residual_trace = []
        plant_trace = []
        raw_actions = []
        applied_actions = []
        eef_positions = []
        generation_events = []
        success = False
        started = time.perf_counter()
        step = -1
        estimator = FastErrorEstimator(scale_m)
        corrector = GeometricCorrector(
            scale_m, kp=float(getattr(args, "feedback_gain", 0.30)),
            kd=float(getattr(args, "feedback_derivative", 0.035)),
            max_norm=float(getattr(args, "max_correction_norm", 0.12)),
            feedforward_weight=float(getattr(args, "feedforward_weight", 0.70)))
        validator = RepairValidator(
            min_reduction_m=float(getattr(args, "min_validation_reduction_m", 0.0010)),
            max_failed_checks=MAX_NO_IMPROVEMENT)
        latest_estimate = estimator.snapshot(stable=False)
        estimator_trace = []
        scene_recovery_lock_until = -1

        for step in range(args.max_steps):
            tick_start = time.perf_counter()
            if method == "fixed_refresh" and step == 16:
                policy.reset()
                tracking_reference = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
                fixed_refreshes += 1
                intervention_events.append({"step": step, "kind": "fixed_refresh",
                                            "action": "wam_replan"})
            if step == injection_step and condition == "target_shift":
                _shift_object(env, target_name, target_shift)
                obs = env.env._get_observations()
                scene_injected = True
                scene_injected_at = time.perf_counter()
            if step == injection_step and condition == "distractor_shift":
                _shift_object(env, distractor_name, distractor_shift)
                obs = env.env._get_observations()
                scene_injected = True
                scene_injected_at = time.perf_counter()
            fault_active = injection_step <= step < injection_step + FAULT_DURATION
            if fault_active and condition in ("arm_offset", "arm_speed_offset", "arm_speed_loss"):
                fault_steps_applied += 1
                if not plant_fault_active:
                    first_fault_time = time.perf_counter()
                plant_fault_active = True
            else:
                plant_fault_active = False

            current_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
            current_target = _joint_pos(env, target_name)
            # This controlled test detects an abrupt target move only while the
            # gripper is open and before contact; normal carrying is not a scene change.
            gripper_qpos = np.asarray(obs.get("robot0_gripper_qpos", [0.0]), dtype=np.float64).reshape(-1)
            # Panda finger joints move with opposite signs, so their sum can
            # stay near zero even when the gripper is fully open. Use the
            # joint-vector magnitude instead of the signed sum.
            gripper_open = float(np.linalg.norm(gripper_qpos)) >= 0.025
            precontact = np.linalg.norm(current_eef - current_target) > 0.12
            if scene_injected and step == injection_step and condition in (
                    "target_shift", "distractor_shift"):
                scene_gate_at_injection = {
                    "gripper_open": bool(gripper_open), "precontact": bool(precontact),
                    "gripper_qpos": gripper_qpos.tolist(),
                    "gripper_qpos_norm": float(np.linalg.norm(gripper_qpos)),
                    "target_delta_from_reference_m": float(np.linalg.norm(current_target - target_reference)),
                    "eef_to_target_distance_m": float(np.linalg.norm(current_eef - current_target)),
                }
            if (method == "hybrid" and step % VALIDATE_EVERY == 0
                    and condition == "target_shift"
                    and not target_change_detected and gripper_open and precontact):
                change_m = float(np.linalg.norm(current_target - target_reference))
                if scene_injected and change_m >= SCENE_CHANGE_THRESHOLD_M:
                    target_change_detected = True
                    target_change_ms = (time.perf_counter() - scene_injected_at) * 1000 if scene_injected_at else None
                    world_replan_pending_at = time.perf_counter()
                    policy.reset()
                    estimator.reset()
                    latest_estimate = estimator.snapshot(stable=False)
                    scene_recovery_lock_until = step + 64
                    tracking_reference = current_eef.copy()
                    repair_active = False
                    active_segment = None
                    no_improvement = 0
                    scene_replans += 1
                    intervention_events.append({"step": step, "kind": "task_target_change",
                                                "target_delta_m": change_m, "action": "wam_replan"})
                elif not scene_injected or change_m < SCENE_CHANGE_THRESHOLD_M:
                    # Follow reliable pre-contact target motion to avoid
                    # confusing ordinary object motion with the injected jump.
                    target_reference = current_target.copy()
            elif (condition == "distractor_shift" and scene_injected and not target_change_detected
                  and step % VALIDATE_EVERY == 0):
                # Record the negative control's task-relevant target state.
                target_change_m = float(np.linalg.norm(current_target - target_reference))
                if target_change_m >= SCENE_CHANGE_THRESHOLD_M:
                    target_change_detected = True
                    # This branch should be impossible for a distractor-only
                    # intervention. Keep it visible if the task object itself
                    # moved through contact or a simulator side effect.
                elif precontact:
                    target_reference = current_target.copy()
                    if not irrelevant_change_ignored:
                        irrelevant_scene_changes += 1
                        irrelevant_change_ignored = True
                        if method == "hybrid":
                            # A scene monitor has just explained the visual
                            # change as nuisance.  Give the new observation a
                            # short settling window before allowing an arm
                            # repair gate to fire.
                            estimator.reset()
                            latest_estimate = estimator.snapshot(stable=False)
                            scene_recovery_lock_until = step + 32

            action = np.asarray(policy.act(obs, task.language), dtype=np.float32).copy()
            if world_replan_pending_at is not None and first_replan_action_ms is None:
                first_replan_action_ms = (time.perf_counter() - world_replan_pending_at) * 1000
                world_replan_pending_at = None
            generation_id = response_holder.get("generation_id")
            if generation_id is None and response_holder.get("step") is not None:
                # The standard action response does not expose a generation
                # id. The LIBERO OpenWAM executor returns one action from its
                # 32-action buffer per request; its server step counter lets
                # us identify the chunk boundary without enabling latent
                # export on every observation.
                generation_id = (int(response_holder["step"]) - 1) // 32
            if generation_id != last_generation:
                # WAM conditions its next action chunk on achieved
                # proprioception. Re-anchor at that genuine replanning point;
                # never rebase inside a chunk while measuring drift.
                tracking_reference = current_eef.copy()
                trigger_history.clear()
                if repair_active and last_generation is not None:
                    repair_active = False
                    active_segment = None
                last_generation = generation_id
                generation_events.append({"step": step, "generation_id": generation_id})
            if tracking_reference is None:
                tracking_reference = current_eef.copy()

            error = tracking_reference - current_eef
            residual_m = float(np.linalg.norm(error))
            residual_trace.append({"step": step, "residual_m": residual_m,
                                   "world_target_residual_m": float(np.linalg.norm(current_target - target_reference)),
                                   "correction_active": repair_active})

            local_allowed = (method in ("hybrid", "local_only")
                             and not (method == "hybrid" and step < scene_recovery_lock_until))
            # The fast head distinguishes a persistent plant mismatch from a
            # one-off visual/proprioceptive residual.  It is intentionally
            # blind to the condition label; only command/achieved motion is
            # used here.  Scene changes are handled by the WAM branch above.
            # Deliberately blind to the injected-fault label: the same residual
            # gate runs in clean and perturbed episodes. Clean episodes measure
            # false repairs; scene-change episodes verify the task-level
            # replanner takes precedence over local execution repair.
            active_threshold_m = threshold_m + float(
                getattr(args, "threshold_adjustment_m", 0.0))
            trigger_now = False
            stage_distance = float(np.linalg.norm(current_eef - current_target))
            contact_guard = (not gripper_open) and stage_distance < 0.10
            if step % VALIDATE_EVERY == 0 and local_allowed and not repair_active:
                if (residual_m > active_threshold_m and latest_estimate.stable
                        and not contact_guard):
                    trigger_history.append({"step": step, "error": error.copy()})
                    keep = max(1, int(getattr(args, "persistence_checks", 1)))
                    trigger_history = trigger_history[-keep:]
                    trigger_now = len(trigger_history) >= keep
                    if trigger_now and keep > 1:
                        dirs = []
                        for sample in trigger_history:
                            norm = float(np.linalg.norm(sample["error"]))
                            dirs.append(sample["error"] / max(norm, 1e-9))
                        min_cos = min(float(np.dot(a, b)) for a, b in zip(dirs, dirs[1:]))
                        trigger_now = min_cos >= float(getattr(args, "coherence_cos", 0.2))
                        if not trigger_now:
                            trigger_history = trigger_history[-1:]
                else:
                    trigger_history.clear()
            if trigger_now:
                repair_active = True
                repair_start_error = residual_m
                repair_start_step = step
                no_improvement = 0
                active_segment = {"step": step, "before_residual_m": residual_m,
                                  "correction_steps": 0, "verified": False}
                correction_segments.append(active_segment)
                local_corrections += 1
                validator.start()
                if first_local_command_ms is None and first_fault_time is not None:
                    first_local_command_ms = (time.perf_counter() - first_fault_time) * 1000
                intervention_events.append({"step": step, "kind": "eef_tracking_error",
                                            "residual_m": residual_m,
                                            "persistence_checks": int(getattr(args, "persistence_checks", 1)),
                                            "action": "geometric_correction"})

            # Verify every four actions. A repair must reduce the same EEF
            # tracking residual; otherwise retry once, then hand back to WAM.
            if (repair_active and active_segment is not None and step > repair_start_step
                    and step % VALIDATE_EVERY == 0):
                current_segment_residual = residual_m
                improved, stop_after_failures = validator.observe(
                    active_segment["before_residual_m"], current_segment_residual)
                active_segment.update(after_residual_m=current_segment_residual, verified=True,
                                      improved=bool(improved), validation_step=step)
                if current_segment_residual <= threshold_m:
                    repair_active = False
                    active_segment = None
                elif stop_after_failures:
                    policy.reset()
                    estimator.reset()
                    latest_estimate = estimator.snapshot(stable=False)
                    tracking_reference = current_eef.copy()
                    repair_active = False
                    active_segment["fallback"] = "wam_replan_after_no_improvement"
                    active_segment = None
                    correction_fallback_replans += 1
                    intervention_events.append({"step": step, "kind": "repair_failed_verification",
                                                "residual_m": current_segment_residual,
                                                "action": "wam_replan"})
                else:
                    repair_start_error = current_segment_residual
                    repair_start_step = step
                    active_segment = {"step": step, "before_residual_m": current_segment_residual,
                                      "correction_steps": 0, "verified": False}
                    correction_segments.append(active_segment)
                trigger_history.clear()

            correction = np.zeros(3, dtype=np.float64)
            if repair_active:
                correction = corrector.action_delta(
                    action[:3], error, latest_estimate,
                    allow_feedforward=latest_estimate.confidence >= 0.35)
                correction *= float(getattr(args, "correction_gain", CORRECTION_GAIN)) / max(CORRECTION_GAIN, 1e-6)
                correction = _norm_limit(correction,
                                         float(getattr(args, "max_correction_norm", 0.12)))
                active_segment["correction_steps"] += 1
            applied = action.copy()
            applied[:3] = np.clip(applied[:3] + correction, -1.0, 1.0)
            nominal_plant_input = applied[:3].astype(np.float64).copy()
            repair_was_active = bool(repair_active)
            # Downstream execution-chain model: WAM and its nominal plan remain
            # untouched; gain/bias act only between the command and MuJoCo OSC.
            if plant_fault_active:
                applied[:3] = np.clip(
                    (scenario["plant_gain"] * nominal_plant_input + bias).astype(np.float32),
                    -1.0, 1.0)

            raw_actions.append(action.copy())
            applied_actions.append(applied.copy())
            eef_positions.append(current_eef.copy())
            # The reference follows the original WAM command. Local corrections
            # and injected actuator faults therefore appear as measurable error.
            expected_before = tracking_reference.copy()
            tracking_reference = tracking_reference + action[:3].astype(np.float64) * scale_m
            obs, _, done, _ = env.step(applied)
            next_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
            # Update the estimator from the pre-plant command and achieved
            # motion. Freeze identification during an active repair so the
            # feedback action cannot explain away the original drift.
            if not repair_was_active:
                latest_estimate = estimator.update(
                    nominal_plant_input, next_eef - current_eef,
                    action[:3].astype(np.float64) * scale_m,
                    CONTROL_PERIOD_S)
            estimator_trace.append({
                "step": step,
                "gain": latest_estimate.gain.astype(float).tolist(),
                "bias_m": latest_estimate.bias_m.astype(float).tolist(),
                "innovation_m": latest_estimate.innovation_m.astype(float).tolist(),
                "innovation_norm": float(latest_estimate.innovation_norm),
                "innovation_baseline": float(latest_estimate.innovation_baseline),
                "innovation_change": float(latest_estimate.innovation_change),
                "confidence": float(latest_estimate.confidence),
                "fault_score": float(latest_estimate.fault_score),
                "innovation_score": float(latest_estimate.innovation_score),
                "gain_drift": float(latest_estimate.gain_drift),
                "bias_drift": float(latest_estimate.bias_drift),
                "drift_score": float(latest_estimate.drift_score),
                "stable": bool(latest_estimate.stable),
            })
            plant_trace.append({
                "step": step, "fault_active": bool(plant_fault_active),
                "nominal_command_xyz": action[:3].astype(float).tolist(),
                "local_correction_xyz": correction.astype(float).tolist(),
                "plant_command_xyz": applied[:3].astype(float).tolist(),
                "expected_eef_before_m": expected_before.astype(float).tolist(),
                "actual_eef_before_m": current_eef.astype(float).tolist(),
                "actual_eef_after_m": next_eef.astype(float).tolist(),
                "expected_eef_after_m": tracking_reference.astype(float).tolist(),
                "actual_step_delta_m": (next_eef - current_eef).astype(float).tolist(),
                "expected_step_delta_m": (action[:3].astype(np.float64) * scale_m).astype(float).tolist(),
                "tracking_error_after_m": float(np.linalg.norm(tracking_reference - next_eef)),
            })
            elapsed = time.perf_counter() - tick_start
            if args.pace:
                time.sleep(max(0.0, CONTROL_PERIOD_S - elapsed))
            if done:
                success = True
                break

        wall_ms = (time.perf_counter() - started) * 1000
        final_target = _joint_pos(env, target_name)
        result = {
            "task_id": task_id, "trial": trial, "condition": condition, "method": method,
            "task_language": task.language, "success": bool(success), "steps": step + 1,
            "task_objects": task_objects,
            "wall_ms": wall_ms, "paced": bool(args.pace), "control_period_ms": 50,
            "scene_injected": scene_injected, "scene_replans": scene_replans,
            "fixed_refreshes": fixed_refreshes,
            "correction_fallback_replans": correction_fallback_replans,
            "irrelevant_scene_changes_detected": irrelevant_scene_changes,
            "irrelevant_scene_change_ignored": irrelevant_change_ignored,
            "target_change_detected": target_change_detected,
            "scene_gate_at_injection": scene_gate_at_injection,
            "target_change_detection_ms": target_change_ms,
            "replans_after_target_change": 1 if target_change_detected and scene_replans else 0,
            "local_corrections": local_corrections, "correction_segments": correction_segments,
            "first_local_command_ms": first_local_command_ms,
            "fault_to_first_replan_action_ms": first_replan_action_ms,
            "tracking_threshold_m": threshold_m + float(
                getattr(args, "threshold_adjustment_m", 0.0)),
            "local_trigger_gate": "persistent_eef_residual_plus_online_innovation_and_gain_bias_drift; injected_fault_label_not_used",
            "residual_p50_m": float(np.percentile([r["residual_m"] for r in residual_trace], 50)),
            "residual_p95_m": float(np.percentile([r["residual_m"] for r in residual_trace], 95)),
            "residual_max_m": float(max(r["residual_m"] for r in residual_trace)),
            "trajectory_rmse_m": float(np.sqrt(np.mean([r["tracking_error_after_m"] ** 2 for r in plant_trace]))),
            "trajectory_p95_m": float(np.percentile([r["tracking_error_after_m"] for r in plant_trace], 95)),
            "fault_trajectory_rmse_m": float(np.sqrt(np.mean([
                r["tracking_error_after_m"] ** 2 for r in plant_trace if r["fault_active"]
            ]))) if any(r["fault_active"] for r in plant_trace) else None,
            "fault_trajectory_p95_m": float(np.percentile([
                r["tracking_error_after_m"] for r in plant_trace if r["fault_active"]
            ], 95)) if any(r["fault_active"] for r in plant_trace) else None,
            "fault_bias_action_norm": float(np.linalg.norm(bias)),
            "fault_gain": float(scenario["plant_gain"]),
            "correction_gain": float(getattr(args, "correction_gain", CORRECTION_GAIN)),
            "persistence_checks": int(getattr(args, "persistence_checks", 1)),
            "coherence_cos": float(getattr(args, "coherence_cos", 0.2)),
            "fault_window_steps": step + 1 - injection_step if condition.startswith("arm_") and fault_steps_applied else 0,
            "fault_persistent": bool(condition.startswith("arm_")),
            "fault_steps_applied": fault_steps_applied,
            "injection_step": injection_step if condition != "clean" else None,
            "injection_phase": scenario["injection_phase"],
            "target_shift_m": scenario["target_shift_magnitude"] if condition == "target_shift" else 0.0,
            "distractor_shift_m": scenario["distractor_shift_magnitude"] if condition == "distractor_shift" else 0.0,
            "target_shift_vector_m": target_shift.tolist(),
            "distractor_shift_vector_m": distractor_shift.tolist(),
            "fault_bias_vector": bias.tolist(), "fault_axis": bias_axis,
            "difficulty": scenario.get("difficulty", "standard"),
            "fast_estimator": "ewma_gain_bias_plus_geometric_feedback",
            "fast_estimator_final": estimator_trace[-1] if estimator_trace else None,
            "fast_estimator_trace": estimator_trace,
            "repair_validator": "residual_reduction_then_wam_refresh",
            "initial_wam_action_sha256": _sha(raw_actions[:min(8, len(raw_actions))]),
            "first_generation_id": generation_events[0]["generation_id"] if generation_events else None,
            "generation_events": generation_events,
            "interventions": intervention_events, "residual_trace": residual_trace,
            "plant_trace": plant_trace,
            "physical_gpu_for_simulator": args.sim_gpu,
            "eef_action_scale_m": scale_m.tolist(),
            "eef_action_scale_source": scale_source,
        }
        if getattr(args, "collect_scale_trace", False):
            result["raw_actions"] = [np.asarray(a, dtype=np.float32).tolist() for a in raw_actions]
            result["eef_positions"] = [np.asarray(p, dtype=np.float32).tolist() for p in eef_positions]
        return result
    finally:
        if policy is not None:
            try:
                policy._client.predict = original_predict
            except Exception:
                pass
            policy.close()
        env.close()


def run_calibration(args, suite, output: Path) -> dict:
    from copy import copy

    # Fit the action-to-EEF scale on one clean calibration state. Freeze it,
    # then estimate the clean residual threshold on a disjoint state.
    scale_rows = []
    scales = {}
    for task_id in args.tasks:
        trial = args.calibration_states[0]
        scale_args = copy(args)
        scale_args.collect_scale_trace = True
        row = run_rollout(scale_args, suite, task_id, trial, "clean", "wam", threshold_m=1e6)
        actions = np.asarray(row["raw_actions"], dtype=np.float64)
        positions = np.asarray(row["eef_positions"], dtype=np.float64)
        estimates = []
        for axis in range(3):
            if len(actions) < 2:
                estimates.append(float(row["eef_action_scale_m"][axis]))
                continue
            command = actions[:-1, axis]
            movement = np.diff(positions[:, axis])
            denominator = float(np.dot(command, command))
            estimate = float(np.dot(command, movement) / denominator) if denominator > 1e-8 else 0.0
            fallback = float(row["eef_action_scale_m"][axis])
            # Reject near-zero / sign-flipped fits; those are not useful plant
            # scale estimates and generally indicate an unexcited action axis.
            if not np.isfinite(estimate) or estimate * fallback <= 0 or abs(estimate) < fallback * 0.1:
                estimate = fallback
            estimates.append(estimate)
        scales[str(task_id)] = estimates
        scale_rows.append({"task_id": task_id, "trial": trial,
                           "nominal_scale_m": row["eef_action_scale_m"],
                           "fitted_scale_m": estimates,
                           "actions": len(actions), "success": row["success"]})

    values = {task: [] for task in args.tasks}
    rows = []
    calibration_args = copy(args)
    calibration_args.max_steps = args.max_steps
    calibration_args.scale_overrides = scales
    for task_id in args.tasks:
        for trial in args.calibration_states[1:]:
            row = run_rollout(calibration_args, suite, task_id, trial, "clean", "wam", threshold_m=1e6)
            values[task_id].extend(r["residual_m"] for r in row["residual_trace"])
            rows.append({"task_id": task_id, "trial": trial, "success": row["success"],
                         "steps": row["steps"], "residual_prefix_m": values[task_id][-32:]})
            _atomic_json(output, {"stage": "calibration", "rows": rows})
            print(f"[calibration] task={task_id} state={trial} n={len(values[task_id])}", flush=True)
    thresholds = {}
    calibration_stats = {}
    for task_id, observations in values.items():
        array = np.asarray(observations, dtype=np.float64)
        q95 = float(np.quantile(array, 0.95)) if len(array) else 0.0
        thresholds[str(task_id)] = max(0.010, q95 + 0.005)
        calibration_stats[str(task_id)] = {"n": int(len(array)), "p50_m": float(np.median(array)) if len(array) else None,
                                           "p95_m": q95, "max_m": float(np.max(array)) if len(array) else None,
                                           "frozen_threshold_m": thresholds[str(task_id)]}
    report = {"stage": "calibration_complete", "thresholds_m": thresholds,
              "frozen_action_scales_m": scales, "action_scale_fit": scale_rows,
              "clean_tracking_stats": calibration_stats,
              "calibration_tasks": args.tasks, "calibration_states": args.calibration_states,
              "threshold_rule": "max(10 mm, clean full-rollout P95 + 5 mm)"}
    _atomic_json(output.parent / "calibration.json", report)
    return report


def load_calibration(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if "frozen_action_scales_m" not in payload or "thresholds_m" not in payload:
        raise ValueError("frozen calibration must include action scales and residual thresholds")
    return payload


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _exact_p(rescues: int, harms: int) -> float:
    discordant = rescues + harms
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(rescues, harms) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def summarize(rows: list[dict]) -> dict:
    groups = {}
    for row in rows:
        groups.setdefault((row["condition"], row["method"]), []).append(row)
    aggregate = []
    for (condition, method), group in sorted(groups.items()):
        successes = sum(bool(row["success"]) for row in group)
        reductions = [(segment["before_residual_m"] - segment["after_residual_m"])
                      / max(segment["before_residual_m"], 1e-9)
                      for row in group for segment in row["correction_segments"]
                      if segment.get("verified")]
        aggregate.append({
            "condition": condition, "method": method, "n": len(group),
            "successes": successes, "success_rate": successes / len(group),
            "success_rate_wilson_95": _wilson(successes, len(group)),
            "mean_steps": float(np.mean([row["steps"] for row in group])),
            "mean_wall_s": float(np.mean([row["wall_ms"] for row in group]) / 1000.0),
            "episodes_with_local_correction": sum(row["local_corrections"] > 0 for row in group),
            "mean_local_corrections": float(np.mean([row["local_corrections"] for row in group])),
            "mean_scene_replans": float(np.mean([row["scene_replans"] for row in group])),
            "mean_fixed_refreshes": float(np.mean([row.get("fixed_refreshes", 0) for row in group])),
            "mean_residual_p95_m": float(np.mean([row["residual_p95_m"] for row in group])),
            "mean_trajectory_rmse_m": float(np.mean([row["trajectory_rmse_m"] for row in group])),
            "mean_trajectory_p95_m": float(np.mean([row["trajectory_p95_m"] for row in group])),
            "median_verified_residual_reduction_fraction": float(np.median(reductions)) if reductions else None,
            "median_local_command_delay_ms": float(np.median(
                [row["first_local_command_ms"] for row in group if row["first_local_command_ms"] is not None]))
                if any(row["first_local_command_ms"] is not None for row in group) else None,
            "median_target_detection_delay_ms": float(np.median(
                [row["target_change_detection_ms"] for row in group if row["target_change_detection_ms"] is not None]))
                if any(row["target_change_detection_ms"] is not None for row in group) else None,
            "median_replan_action_ready_ms": float(np.median(
                [row["fault_to_first_replan_action_ms"] for row in group
                 if row["fault_to_first_replan_action_ms"] is not None]))
                if any(row["fault_to_first_replan_action_ms"] is not None for row in group) else None,
            "mean_fast_estimator_confidence": float(np.mean([
                (row.get("fast_estimator_final") or {}).get("confidence", 0.0)
                for row in group])),
            "mean_fast_estimator_fault_score": float(np.mean([
                (row.get("fast_estimator_final") or {}).get("fault_score", 0.0)
                for row in group])),
        })

    baseline_method = {"clean": "wam", "arm_speed_loss": "fault_only",
                       "arm_offset": "fault_only", "arm_speed_offset": "fault_only",
                       "target_shift": "stale_plan", "distractor_shift": "stale_plan"}
    index = {(row["task_id"], row["trial"], row["condition"], row["method"]): row for row in rows}
    paired = []
    for (condition, method), group in sorted(groups.items()):
        baseline = baseline_method[condition]
        if method == baseline:
            continue
        rescues = harms = shared = 0
        step_diffs = []
        for row in group:
            base = index[(row["task_id"], row["trial"], condition, baseline)]
            if not base["success"] and row["success"]:
                rescues += 1
            elif base["success"] and not row["success"]:
                harms += 1
            if base["success"] and row["success"]:
                shared += 1
                step_diffs.append(row["steps"] - base["steps"])
        paired.append({"condition": condition, "method": method, "baseline": baseline,
                       "pairs": len(group), "shared_success": shared, "rescues": rescues,
                       "harms": harms, "net_rescues": rescues - harms,
                       "two_sided_exact_p": _exact_p(rescues, harms),
                       "mean_step_difference_on_shared_success":
                           float(np.mean(step_diffs)) if step_diffs else None})

    def event_rate(condition: str, field: str) -> dict:
        group = [row for row in rows if row["condition"] == condition and row["method"] == "hybrid"]
        return {"episodes": len(group), "episodes_with_event": sum(bool(row[field]) for row in group),
                "rate": (sum(bool(row[field]) for row in group) / len(group)) if group else None}

    return {"n_rows": len(rows), "aggregate": aggregate, "paired_comparisons": paired,
            "negative_controls": {
                "clean_local_correction": event_rate("clean", "local_corrections"),
                "clean_scene_replan": event_rate("clean", "scene_replans"),
                "distractor_scene_replan": event_rate("distractor_shift", "scene_replans"),
                "distractor_local_correction": event_rate("distractor_shift", "local_corrections"),
                "distractor_scene_change_ignored": event_rate("distractor_shift", "irrelevant_scene_change_ignored"),
                "target_shift_replan": event_rate("target_shift", "scene_replans"),
                "fixed_refresh": event_rate("target_shift", "fixed_refreshes"),
            }}


def write_summary(output: Path, rows: list[dict]) -> dict:
    report = summarize(rows)
    _atomic_json(output.parent / "summary.json", report)
    difficulty = sorted({row.get("difficulty", "standard") for row in rows})
    lines = ["# Hybrid execution repair benchmark", "",
             f"LIBERO-plus simulation with OpenWAM (difficulty: {', '.join(difficulty)}). Execution drift is injected downstream of nominal WAM commands; target-scene change detection uses simulator object poses as an oracle.", "",
             "## Success and control outcomes", "",
             "| Condition | Method | Success | Local repair episodes | Mean repairs | Mean WAM replans | Tracking RMSE (m) | Tracking P95 (m) | Fast confidence | Fault score |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["aggregate"]:
        lines.append(f"| {row['condition']} | {row['method']} | {row['successes']}/{row['n']} ({row['success_rate']:.1%}) | "
                     f"{row['episodes_with_local_correction']}/{row['n']} | {row['mean_local_corrections']:.2f} | "
                     f"{row['mean_scene_replans']:.2f} | {row['mean_trajectory_rmse_m']:.4f} | "
                     f"{row['mean_trajectory_p95_m']:.4f} | {row['mean_fast_estimator_confidence']:.3f} | "
                     f"{row['mean_fast_estimator_fault_score']:.3f} |")
    lines.extend(["", "## Paired outcome changes", "",
                  "| Condition | Method vs baseline | Rescues | Harms | Net | Exact p |",
                  "|---|---|---:|---:|---:|---:|"])
    for row in report["paired_comparisons"]:
        lines.append(f"| {row['condition']} | {row['method']} vs {row['baseline']} | {row['rescues']} | "
                     f"{row['harms']} | {row['net_rescues']} | {row['two_sided_exact_p']:.4f} |")
    lines.extend(["", "## Negative controls", "",
                  f"Clean local correction: {report['negative_controls']['clean_local_correction']['episodes_with_event']}/"
                  f"{report['negative_controls']['clean_local_correction']['episodes']} episodes.",
                  f"Clean WAM replan: {report['negative_controls']['clean_scene_replan']['episodes_with_event']}/"
                  f"{report['negative_controls']['clean_scene_replan']['episodes']} episodes.",
                  f"Distractor-only WAM replan: {report['negative_controls']['distractor_scene_replan']['episodes_with_event']}/"
                  f"{report['negative_controls']['distractor_scene_replan']['episodes']} episodes.", ""])
    (output.parent / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 7])
    parser.add_argument("--states", type=int, nargs="+", required=True,
                        help="held-out evaluation states; kept separate from calibration")
    parser.add_argument("--calibration-states", type=int, nargs="+", required=True)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--wam-host", default="127.0.0.1")
    parser.add_argument("--wam-port", type=int, default=8940)
    parser.add_argument("--sim-gpu", type=int, default=1)
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--state-map", type=Path,
                        help="optional JSON mapping task ids to per-task evaluation state ids")
    parser.add_argument("--scene-map", type=Path,
                        help="optional frozen per-task/state world perturbation manifest")
    parser.add_argument("--job-manifest", type=Path,
                        help="optional explicit paired evaluation jobs")
    parser.add_argument("--correction-gain", type=float, default=CORRECTION_GAIN)
    parser.add_argument("--difficulty", choices=("standard", "hard"), default="standard",
                        help="public LIBERO-plus perturbation severity; hard freezes a stronger combination")
    parser.add_argument("--feedback-gain", type=float, default=0.30)
    parser.add_argument("--feedback-derivative", type=float, default=0.035)
    parser.add_argument("--feedforward-weight", type=float, default=0.70)
    parser.add_argument("--max-correction-norm", type=float, default=0.12)
    parser.add_argument("--min-validation-reduction-m", type=float, default=0.0010)
    parser.add_argument("--threshold-adjustment-m", type=float, default=0.0,
                        help="fixed offset to the frozen clean-calibration residual threshold")
    parser.add_argument("--persistence-checks", type=int, default=1,
                        help="consecutive four-action residual checks required before repair")
    parser.add_argument("--coherence-cos", type=float, default=0.2,
                        help="minimum direction cosine across persistent residual checks")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    args = parser.parse_args()

    _write_libero_config()
    suite = _LocalSuite(args.tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "evaluate" and set(args.states) & set(args.calibration_states):
        raise ValueError("calibration states and held-out evaluation states must be disjoint")
    if args.stage == "calibrate":
        result = run_calibration(args, suite, args.output)
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.calibration is None or not args.calibration.is_file():
        raise FileNotFoundError("evaluation requires --calibration with frozen scales and thresholds")
    frozen_calibration = load_calibration(args.calibration)
    thresholds = frozen_calibration["thresholds_m"]
    args.scale_overrides = frozen_calibration["frozen_action_scales_m"]
    args.states_by_task = (json.loads(args.state_map.read_text()) if args.state_map else {})
    args.scene_scenarios = (json.loads(args.scene_map.read_text()) if args.scene_map else {})
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {args.output}")
    if args.job_manifest:
        jobs = json.loads(args.job_manifest.read_text())["jobs"]
    else:
        jobs = [{"task_id": task_id, "trial": trial, "condition": condition,
                 "methods": list(PAIRED_METHODS[condition])}
                for task_id in args.tasks
                for trial in args.states_by_task.get(str(task_id), args.states)
                for condition in args.conditions]
    total = sum(len(job["methods"]) for job in jobs)
    rows = []
    completed = set()
    _atomic_json(args.output, {"stage": "running", "expected_rows": total, "rows": rows})
    row_store = args.output.with_suffix(".jsonl")
    if row_store.exists():
        raise FileExistsError(f"refusing to overwrite row log: {row_store}")
    with row_store.open("w", encoding="utf-8") as row_handle:
      for job_index, job in enumerate(jobs):
        task_id, trial, condition = int(job["task_id"]), int(job["trial"]), job["condition"]
        if str(task_id) not in thresholds:
            raise KeyError(f"no frozen calibration threshold for task {task_id}")
        methods = list(job["methods"])
        seed = 20261014 + task_id * 10000 + trial * 100 + CONDITIONS.index(condition)
        order = list(np.random.default_rng(seed).permutation(methods))
        for method in order:
            key = (task_id, trial, condition, method)
            if key in completed:
                continue
            row = run_rollout(args, suite, task_id, trial, condition, method,
                              float(thresholds[str(task_id)]), job.get("trial_slot", job_index))
            rows.append(row)
            completed.add(key)
            row_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            row_handle.flush()
            _atomic_json(args.output, {"stage": "running", "expected_rows": total,
                                       "completed_rows": len(rows), "row_log": str(row_store)})
            print(f"[evaluation] {key} success={row['success']} steps={row['steps']} "
                  f"local={row['local_corrections']} replan={row['scene_replans']}", flush=True)
    _atomic_json(args.output, {"stage": "complete", "expected_rows": total,
                               "completed_rows": len(rows), "rows": rows})
    report = write_summary(args.output, rows)
    print(json.dumps({"summary": str(args.output.parent / "summary.json"),
                      "report": str(args.output.parent / "REPORT.md"),
                      "n_rows": report["n_rows"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
