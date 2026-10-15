"""Train and evaluate a lightweight WAM failure discriminator.

This is a self-contained synthetic stress benchmark for the routing idea.  It
does not claim to be an official LIBERO score: the plant and visual signals
are deliberately abstracted so that the contribution of the router can be
tested without an OpenWAM server.  Each state is generated from one episode,
and the labels are counterfactual success probabilities for continue, local
repair, WAM replanning, contact recovery, and safe stop.  Episodes are split
before training to avoid adjacent-frame leakage.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openwam_recovery.failure_router import ACTIONS, SOURCES, FailureRouter


FEATURE_NAMES = (
    "eef_residual", "residual_trend", "direction_agreement",
    "target_roi_change", "distractor_roi_change", "target_motion",
    "plan_confidence", "gripper_closed", "contact_force", "collision_risk",
    "action_magnitude", "remaining_fraction", "observability",
    "phase_fraction", "command_gain_estimate",
)


@dataclass(frozen=True)
class Scenario:
    kind: str
    onset: int
    severity: float
    visual_noise: float
    contact_phase: int
    seed: int


@dataclass
class SimState:
    scenario: Scenario
    step: int = 0
    residual: float = 0.006
    previous_residual: float = 0.006
    target_delta: float = 0.0
    target_delta_previous: float = 0.0
    repaired: bool = False
    replanned: bool = False
    contact_recovered: bool = False
    stopped: bool = False
    failed: bool = False
    resolved: bool = False


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def _scenario(rng: np.random.Generator, episode_id: int, difficulty: str) -> Scenario:
    kinds = ("clean", "nuisance", "execution_drift", "world_change",
             "wam_failure", "contact_failure", "mixed", "unsafe")
    weights = np.asarray([0.14, 0.11, 0.24, 0.20, 0.11, 0.07, 0.09, 0.04])
    kind = str(rng.choice(kinds, p=weights / weights.sum()))
    onset = int(rng.integers(2, 7))
    if difficulty == "heldout":
        severity = float(rng.uniform(0.85, 1.30))
        noise = float(rng.uniform(0.80, 1.35))
    elif difficulty == "development":
        severity = float(rng.uniform(0.90, 1.15))
        noise = float(rng.uniform(0.75, 1.15))
    else:
        severity = float(rng.uniform(0.90, 1.10))
        noise = float(rng.uniform(0.70, 1.05))
    return Scenario(kind=kind, onset=onset, severity=severity,
                    visual_noise=noise, contact_phase=7,
                    seed=episode_id * 7919 + 20261015)


def _event_active(state: SimState) -> bool:
    return state.step >= state.scenario.onset


def _requires_intervention(state: SimState) -> bool:
    return state.scenario.kind not in ("clean", "nuisance") and _event_active(state)


def _advance_latent(state: SimState) -> None:
    """Advance an un-intervened plant by one decision period."""
    state.previous_residual = state.residual
    state.target_delta_previous = state.target_delta
    kind = state.scenario.kind
    if _event_active(state):
        if kind in ("execution_drift", "mixed") and not state.repaired:
            state.residual = _clip(state.residual + 0.018 * state.scenario.severity,
                                   0.0, 0.34)
        if kind == "world_change" or kind == "mixed":
            state.target_delta = 0.075 * state.scenario.severity
        if kind == "nuisance":
            state.target_delta = 0.0
    state.step += 1


def _features(state: SimState, rng: np.random.Generator) -> np.ndarray:
    """Return only deployable observations; hidden scenario fields stay out."""
    kind = state.scenario.kind
    active = _event_active(state)
    # These are two separate visual streams.  A target-region change and a
    # distractor/background change overlap under noise, so relevance is not an
    # oracle label even though the synthetic renderer exposes both regions.
    target_roi = state.target_delta + rng.normal(0.0, 0.010 * state.scenario.visual_noise)
    distractor_roi = (0.075 * state.scenario.severity if kind == "nuisance" and active
                      else (0.018 if kind == "world_change" and active else 0.0))
    distractor_roi += rng.normal(0.0, 0.012 * state.scenario.visual_noise)
    target_motion = (state.target_delta - state.target_delta_previous)
    target_motion += rng.normal(0.0, 0.006 * state.scenario.visual_noise)
    if kind == "wam_failure" and active:
        plan_confidence = 0.26 + rng.normal(0.0, 0.04 * state.scenario.visual_noise)
    else:
        plan_confidence = 0.92 + rng.normal(0.0, 0.035 * state.scenario.visual_noise)
    if kind == "unsafe" and active:
        collision_risk = 0.80 + rng.normal(0.0, 0.06 * state.scenario.visual_noise)
    else:
        collision_risk = 0.05 + rng.normal(0.0, 0.025 * state.scenario.visual_noise)
    if kind == "contact_failure" and state.step >= state.scenario.contact_phase:
        gripper_closed, contact_force = 1.0, 0.08 + rng.normal(0.0, 0.03)
    else:
        gripper_closed = float(state.step >= state.scenario.contact_phase)
        contact_force = 0.52 + rng.normal(0.0, 0.07) if gripper_closed else 0.05
    gain_estimate = (0.70 + rng.normal(0.0, 0.04) if kind in ("execution_drift", "mixed") and active
                     else 0.98 + rng.normal(0.0, 0.025))
    direction = (0.85 if kind in ("execution_drift", "mixed") and active else
                 0.25 if kind in ("world_change", "wam_failure") and active else 0.05)
    action_magnitude = 0.36 + 0.10 * math.sin(0.8 * state.step)
    values = np.asarray([
        state.residual + rng.normal(0.0, 0.004 * state.scenario.visual_noise),
        state.residual - state.previous_residual + rng.normal(0.0, 0.003 * state.scenario.visual_noise),
        direction + rng.normal(0.0, 0.08 * state.scenario.visual_noise),
        target_roi, distractor_roi, target_motion, plan_confidence,
        gripper_closed, contact_force, collision_risk, action_magnitude,
        1.0 - state.step / 12.0, 0.92 + rng.normal(0.0, 0.025),
        state.step / 12.0, gain_estimate,
    ], dtype=np.float64)
    return values


def counterfactual_success(state: SimState, action: str) -> float:
    """Expected final success for one action at this observed state.

    This is the training target, not an input to the deployed router.  It is
    deliberately asymmetric: a plant drift is recoverable by local repair,
    while a task-relevant world change or bad WAM plan needs replanning.
    """
    kind = state.scenario.kind
    active = _event_active(state)
    remaining = max(0.0, (11.0 - state.step) / 8.0)
    r = state.residual
    if kind == "unsafe" and active:
        return {"safe_stop": 0.97, "continue": 0.03, "local_repair": 0.04,
                "wam_replan": 0.08, "contact_recovery": 0.06}.get(action, 0.02)
    if not active or kind in ("clean", "nuisance"):
        if action == "continue":
            return _clip(0.985 - 0.06 * r, 0.80, 0.99)
        if action == "local_repair":
            # False repairs are usually harmless, but they can disturb a clean
            # grasp.  The nuisance stream makes this penalty larger.
            penalty = 0.15 if kind == "clean" else 0.27
            return _clip(0.94 - penalty - 1.2 * r, 0.30, 0.90)
        if action == "wam_replan":
            return 0.87 if kind == "clean" else 0.78
        if action == "contact_recovery":
            return 0.15
        return 0.05
    if kind in ("execution_drift", "mixed"):
        repairability = _clip(1.0 - r / 0.25, 0.0, 1.0)
        if action == "continue":
            return _clip(0.95 - 3.5 * r - 0.24 * (1.0 - remaining), 0.03, 0.94)
        if action == "local_repair":
            base = 0.18 + 0.77 * repairability
            if kind == "mixed":
                base -= 0.18
            return _clip(base + 0.08 * remaining, 0.05, 0.97)
        if action == "wam_replan":
            return _clip(0.63 + 0.22 * remaining, 0.45, 0.90)
        if action == "contact_recovery":
            return 0.18
        return 0.04
    if kind == "world_change":
        if action == "continue":
            return _clip(0.12 - 0.10 * (1.0 - remaining), 0.03, 0.15)
        if action == "local_repair":
            return 0.14
        if action == "wam_replan":
            return _clip(0.79 + 0.15 * remaining, 0.65, 0.96)
        if action == "contact_recovery":
            return 0.09
        return 0.04
    if kind == "wam_failure":
        if action == "continue":
            return 0.10
        if action == "local_repair":
            return 0.20
        if action == "wam_replan":
            return _clip(0.84 + 0.10 * remaining, 0.70, 0.96)
        if action == "contact_recovery":
            return 0.08
        return 0.04
    if kind == "contact_failure":
        if action == "continue":
            return 0.08
        if action == "local_repair":
            return 0.17
        if action == "wam_replan":
            return _clip(0.48 + 0.25 * remaining, 0.35, 0.78)
        if action == "contact_recovery":
            return _clip(0.79 + 0.13 * remaining, 0.65, 0.95)
        return 0.04
    raise ValueError(f"unknown scenario kind: {kind}")


def _source_label(kind: str) -> np.ndarray:
    source = {
        "clean": "nominal_or_nuisance", "nuisance": "nominal_or_nuisance",
        "execution_drift": "execution_drift", "mixed": "execution_drift",
        "world_change": "task_relevant_world_change", "wam_failure": "wam_plan_failure",
        "contact_failure": "contact_failure", "unsafe": "unsafe_state",
    }[kind]
    out = np.zeros(len(SOURCES), dtype=np.float64)
    out[SOURCES.index(source)] = 1.0
    return out


def _labels_and_states(episode_id: int, difficulty: str, *, n_steps: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(episode_id * 1009 + (3 if difficulty == "heldout" else 1))
    scenario = _scenario(rng, episode_id, difficulty)
    state = SimState(scenario=scenario)
    rows, labels, source_labels = [], [], []
    for _ in range(n_steps):
        rows.append(_features(state, rng))
        labels.append([counterfactual_success(state, action) for action in ACTIONS])
        source_labels.append(_source_label(scenario.kind))
        _advance_latent(state)
    return np.asarray(rows), np.asarray(labels), np.asarray(source_labels)


def build_dataset(ids: range, difficulty: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, source_labels = zip(*[_labels_and_states(i, difficulty) for i in ids])
    return (np.concatenate(features, axis=0), np.concatenate(labels, axis=0),
            np.concatenate(source_labels, axis=0))


def _new_env(episode_id: int, difficulty: str) -> tuple[SimState, np.random.Generator]:
    rng = np.random.default_rng(episode_id * 1009 + 17)
    return SimState(_scenario(rng, episode_id, difficulty)), rng


def _apply_action(state: SimState, action: str, rng: np.random.Generator) -> bool:
    if action == "safe_stop":
        state.stopped = True
        state.resolved = state.scenario.kind == "unsafe" and _event_active(state)
        state.failed = not state.resolved
        return state.resolved
    p = counterfactual_success(state, action)
    success = bool(rng.random() < p)
    if success and action == "local_repair":
        state.repaired = True
        state.residual *= 0.22
        state.resolved = state.scenario.kind in ("execution_drift", "mixed", "clean", "nuisance")
    elif success and action == "wam_replan":
        state.replanned = True
        state.target_delta = 0.0
        state.residual *= 0.45
        state.resolved = state.scenario.kind in ("world_change", "wam_failure", "execution_drift",
                                                 "mixed", "clean", "nuisance", "contact_failure")
    elif success and action == "contact_recovery":
        state.contact_recovered = True
        state.resolved = state.scenario.kind == "contact_failure"
    return success


def choose_rule(features: np.ndarray) -> str:
    residual, trend, direction, target_roi, distractor_roi, target_motion = features[:6]
    plan_confidence, gripper, contact, collision = features[6:10]
    if collision > 0.50:
        return "safe_stop"
    if gripper > 0.8 and contact < 0.25:
        return "contact_recovery"
    if target_roi > 0.048 or target_motion > 0.035 or plan_confidence < 0.55:
        return "wam_replan"
    if residual > 0.030 and trend > 0.008 and direction > 0.45:
        return "local_repair"
    return "continue"


def choose_always_local(features: np.ndarray) -> str:
    if features[0] > 0.018 or features[1] > 0.007:
        return "local_repair"
    return "continue"


def choose_always_replan(features: np.ndarray) -> str:
    if features[0] > 0.018 or abs(features[3]) > 0.025 or features[6] < 0.70:
        return "wam_replan"
    return "continue"


def rollout(episode_id: int, difficulty: str, method: str, router: FailureRouter | None,
            costs: dict[str, float], uncertainty_margin: float) -> dict:
    state, rng = _new_env(episode_id, difficulty)
    actions, probabilities = [], []
    intervention_step = None
    false_repair = False
    false_replan = False
    safe_stop = False
    source_predictions = []
    source_prediction_at_event = None
    decision_latencies_ms = []
    for _ in range(12):
        obs = _features(state, rng)
        if method == "stale_wam":
            action = "continue"
        elif method == "always_local":
            action = choose_always_local(obs)
        elif method == "always_replan":
            action = choose_always_replan(obs)
        elif method == "rule_hybrid":
            action = choose_rule(obs)
        elif method == "learned_router":
            assert router is not None
            decision_started = time.perf_counter()
            decision = router.decide(obs, costs=costs, uncertainty_margin=uncertainty_margin)
            decision_latencies_ms.append((time.perf_counter() - decision_started) * 1000.0)
            action = decision.action
            probabilities.append(decision.probabilities)
            source_predictions.append(max(decision.source_probabilities,
                                          key=decision.source_probabilities.get))
            if source_prediction_at_event is None and state.step >= state.scenario.onset:
                source_prediction_at_event = source_predictions[-1]
        elif method == "oracle_router":
            action = ACTIONS[int(np.argmax([counterfactual_success(state, a) - costs.get(a, 0.0)
                                            for a in ACTIONS]))]
        else:
            raise ValueError(method)
        actions.append(action)
        if action in ("local_repair", "wam_replan", "contact_recovery", "safe_stop"):
            if intervention_step is None:
                intervention_step = state.step
            if action == "local_repair" and state.scenario.kind in ("clean", "nuisance"):
                false_repair = True
            if action == "wam_replan" and state.scenario.kind in ("clean", "nuisance", "execution_drift"):
                false_replan = True
            if action == "safe_stop":
                safe_stop = True
            _apply_action(state, action, rng)
            if state.resolved or state.failed:
                break
        _advance_latent(state)
        # An unresolved failure that is ignored too late cannot be rescued by
        # continuing; later intervention can still succeed while time remains.
        if state.step >= 11 and _requires_intervention(state) and not state.resolved:
            state.failed = True
            break
    if not state.resolved and not state.failed:
        state.resolved = state.scenario.kind in ("clean", "nuisance")
    success = bool(state.resolved and not state.failed)
    return {
        "episode": episode_id, "kind": state.scenario.kind, "success": success,
        "actions": actions, "first_intervention_step": intervention_step,
        "false_repair": false_repair, "false_replan": false_replan,
        "safe_stop": safe_stop, "probabilities": probabilities,
        "source_predictions": source_predictions,
        "source_prediction_at_event": source_prediction_at_event,
        "decision_latencies_ms": decision_latencies_ms,
    }


def _evaluate(ids: range, difficulty: str, method: str, router: FailureRouter | None,
              costs: dict[str, float], uncertainty_margin: float) -> list[dict]:
    return [rollout(i, difficulty, method, router, costs, uncertainty_margin) for i in ids]


def summarize(rows: list[dict], baseline: list[dict] | None = None) -> dict:
    success = np.asarray([row["success"] for row in rows], dtype=bool)
    summary = {
        "episodes": len(rows),
        "success_rate": float(np.mean(success)) if len(rows) else None,
        "false_repair_episodes": int(sum(row["false_repair"] for row in rows)),
        "false_replan_episodes": int(sum(row["false_replan"] for row in rows)),
        "safe_stop_episodes": int(sum(row["safe_stop"] for row in rows)),
        "intervention_rate": float(np.mean([bool(row["actions"] and any(a != "continue" for a in row["actions"]))
                                             for row in rows])) if rows else None,
        "mean_first_intervention_step": float(np.mean([row["first_intervention_step"] for row in rows
                                                        if row["first_intervention_step"] is not None]))
        if any(row["first_intervention_step"] is not None for row in rows) else None,
    }
    if any(row["source_prediction_at_event"] is not None for row in rows):
        source_correct = []
        confusion = {truth: {pred: 0 for pred in SOURCES} for truth in SOURCES}
        for row in rows:
            if row["source_prediction_at_event"] is not None:
                truth = _source_label(row["kind"])
                expected = SOURCES[int(np.argmax(truth))]
                predicted = row["source_prediction_at_event"]
                source_correct.append(predicted == expected)
                confusion[expected][predicted] += 1
        summary["source_accuracy_at_first_observation"] = float(np.mean(source_correct))
        summary["source_confusion"] = confusion
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)
    summary["by_failure_source"] = {
        kind: {"episodes": len(group), "success_rate": float(np.mean([r["success"] for r in group])),
               "false_repair_episodes": int(sum(r["false_repair"] for r in group)),
               "false_replan_episodes": int(sum(r["false_replan"] for r in group))}
        for kind, group in sorted(by_kind.items())
    }
    latencies = [value for row in rows for value in row["decision_latencies_ms"]]
    if latencies:
        summary["router_latency_ms"] = {"p50": float(np.percentile(latencies, 50)),
                                         "p95": float(np.percentile(latencies, 95)),
                                         "mean": float(np.mean(latencies))}
    if baseline is not None:
        b = {row["episode"]: row for row in baseline}
        paired_rescue = sum((not b[row["episode"]]["success"]) and row["success"] for row in rows)
        paired_harm = sum(b[row["episode"]]["success"] and (not row["success"]) for row in rows)
        summary.update({"paired_rescue": int(paired_rescue), "paired_harm": int(paired_harm)})
    return summary


def _choose_costs(router: FailureRouter, dev_ids: range) -> tuple[dict[str, float], float, dict]:
    candidates = []
    for local_cost in (0.00, 0.02, 0.05, 0.08, 0.12):
        for replan_cost in (0.00, 0.04, 0.08, 0.12, 0.16):
            costs = {"local_repair": local_cost, "wam_replan": replan_cost,
                     "contact_recovery": 0.02, "safe_stop": 0.03}
            for margin in (0.00, 0.03, 0.06, 0.10):
                learned = _evaluate(dev_ids, "development", "learned_router", router, costs, margin)
                baseline = _evaluate(dev_ids, "development", "stale_wam", None, costs, margin)
                s = summarize(learned, baseline)
                # Development selection explicitly trades rescue against false
                # interventions.  The held-out split is never inspected here.
                objective = (s["success_rate"] - 0.015 * s["false_repair_episodes"] / len(dev_ids)
                             - 0.008 * s["false_replan_episodes"] / len(dev_ids)
                             - 0.0002 * s["mean_first_intervention_step"]
                             if s["mean_first_intervention_step"] is not None else -1.0)
                candidates.append((objective, costs, margin, s))
    candidates.sort(key=lambda item: item[0], reverse=True)
    objective, costs, margin, summary = candidates[0]
    return costs, margin, {"objective": float(objective), "dev_summary": summary,
                           "candidates": len(candidates)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=600)
    parser.add_argument("--dev-episodes", type=int, default=120)
    parser.add_argument("--heldout-episodes", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "failure_router_benchmark")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train_ids = range(0, args.train_episodes)
    dev_ids = range(args.train_episodes, args.train_episodes + args.dev_episodes)
    heldout_start = args.train_episodes + args.dev_episodes
    heldout_ids = range(heldout_start, heldout_start + args.heldout_episodes)
    started = time.perf_counter()
    x_train, y_train, source_train = build_dataset(train_ids, "train")
    router = FailureRouter(hidden=64, seed=7)
    fit = router.fit(x_train, y_train, source_train, epochs=args.epochs, verbose=False)
    costs, margin, selection = _choose_costs(router, dev_ids)
    methods = ("stale_wam", "always_local", "always_replan", "rule_hybrid",
               "learned_router", "oracle_router")
    reports = {}
    raw = {}
    for method in methods:
        rows = _evaluate(heldout_ids, "heldout", method, router, costs, margin)
        baseline = raw.get("stale_wam")
        raw[method] = rows
        reports[method] = summarize(rows, baseline if method != "stale_wam" else None)
    # Report the learned router's calibration on development after the frozen
    # choice, but do not use held-out outcomes to change it.
    dev_rows = _evaluate(dev_ids, "development", "learned_router", router, costs, margin)
    dev_base = _evaluate(dev_ids, "development", "stale_wam", None, costs, margin)
    result = {
        "protocol": {
            "benchmark": "custom synthetic OpenWAM recovery stress test",
            "not_official_libero": True,
            "not_real_robot": True,
            "feature_names": list(FEATURE_NAMES),
            "actions": list(ACTIONS),
            "split": {"train_episodes": args.train_episodes, "development_episodes": args.dev_episodes,
                       "heldout_episodes": args.heldout_episodes},
            "counterfactual_labels": "soft final-success probabilities from simulator branches",
            "controller": "fixed bounded geometric local repair; router is the only learned component",
            "uncertainty": "mean Bernoulli entropy; a decision margin proxy, not calibrated confidence",
        },
        "model": {"type": "two_hidden_layer_numpy_mlp", "hidden": 64,
                  "parameters": router.parameter_count, **fit},
        "selection": {"costs": costs, "uncertainty_margin": margin, **selection,
                       "post_selection_dev": summarize(dev_rows, dev_base)},
        "heldout": reports,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Failure router benchmark results", "",
             "This is a custom synthetic OpenWAM recovery stress test, not an official LIBERO score or real-arm result.", "",
             f"Model: 2x64 NumPy MLP, {router.parameter_count} parameters; held-out episodes: {args.heldout_episodes}.",
             f"Frozen development costs: local={costs['local_repair']:.2f}, replan={costs['wam_replan']:.2f}, uncertainty margin={margin:.2f}.", "",
             "| Method | Success | Rescue | Harm | False repair | False replan | Intervention rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        s = reports[method]
        lines.append(f"| {method} | {s['success_rate']:.1%} | {s.get('paired_rescue', '-') } | {s.get('paired_harm', '-') } | "
                     f"{s['false_repair_episodes']} | {s['false_replan_episodes']} | {s['intervention_rate']:.1%} |")
    lines += ["", "The learned router is evaluated with the same fixed local-repair action as the rule baseline.",
              "Train/development/held-out splits are episode-disjoint; development selection is frozen before held-out evaluation.",
              "Source classification and action success are reported separately. The uncertainty score is an entropy proxy, not a calibrated confidence bound."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
