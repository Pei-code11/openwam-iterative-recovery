"""Counterfactual hard benchmark for a lightweight WAM recovery router.

Every episode creates one latent execution state and forks it into the same
five candidate actions.  The labels used for training and evaluation are the
actual terminal outcomes of those deterministic branches, rather than a
hand-written probability target.  The simulator is deliberately small and
transparent: it models Cartesian execution drift, a task target shift, a bad
nominal plan, contact loss, nuisance motion, and unsafe contact risk.

This remains a custom simulator, not a LIBERO score or a real-arm result. Its
purpose is to test the routing question with paired counterfactual rollouts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openwam_recovery.failure_router import ACTIONS, FailureRouter


KINDS = ("clean", "distractor", "execution_drift", "world_change",
         "wam_failure", "contact_failure", "mixed", "unsafe")
SOURCE_TO_ACTION = {
    "clean": "continue", "distractor": "continue",
    "execution_drift": "local_repair", "world_change": "wam_replan",
    "wam_failure": "wam_replan", "contact_failure": "contact_recovery",
    "mixed": "wam_replan", "unsafe": "safe_stop",
}


@dataclass(frozen=True)
class Episode:
    kind: str
    error: np.ndarray
    target_shift: np.ndarray
    plan_error: float
    contact: float
    collision_risk: float
    remaining: float
    action_size: float
    noise: float
    seed: int


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


def make_episode(rng: np.random.Generator, episode_id: int, split: str) -> Episode:
    # Keep the difficult test distribution slightly broader than training.
    p = np.asarray([.12, .10, .23, .18, .11, .09, .12, .05])
    kind = str(rng.choice(KINDS, p=p / p.sum()))
    if split == "train":
        scale = rng.uniform(.85, 1.10)
        noise = rng.uniform(.65, 1.05)
    elif split == "dev":
        scale = rng.uniform(.85, 1.20)
        noise = rng.uniform(.75, 1.25)
    else:
        scale = rng.uniform(.70, 1.40)
        noise = rng.uniform(.85, 1.45)
    drift = rng.normal(0.0, 0.015 * scale, 3)
    shift = rng.normal(0.0, 0.012 * scale, 3)
    if kind in ("execution_drift", "mixed"):
        drift += rng.normal(0.0, 0.075 * scale, 3)
    if kind in ("world_change", "mixed"):
        shift += rng.normal(0.0, 0.085 * scale, 3)
    plan_error = rng.uniform(.55, .95) if kind == "wam_failure" else rng.uniform(.0, .12)
    if kind == "mixed":
        plan_error = rng.uniform(.35, .75)
    contact = rng.uniform(.05, .20) if kind == "contact_failure" else rng.uniform(.70, .98)
    collision = rng.uniform(.62, .98) if kind == "unsafe" else rng.uniform(.01, .16)
    if kind == "mixed":
        collision = rng.uniform(.10, .38)
    return Episode(kind, drift, shift, plan_error, contact, collision,
                   rng.uniform(.30, .95), rng.uniform(.05, .22), noise,
                   episode_id * 7919 + 20261015)


def observe(ep: Episode, rng: np.random.Generator) -> np.ndarray:
    """Noisy deployable features; no scenario kind is exposed."""
    residual = float(np.linalg.norm(ep.error))
    target = float(np.linalg.norm(ep.target_shift))
    # The sensor sees target and distractor motion through overlapping regions.
    target_roi = target + rng.normal(0, .030 * ep.noise)
    distractor = (rng.uniform(.01, .12) if ep.kind == "distractor"
                  else rng.uniform(.0, .035)) + rng.normal(0, .025 * ep.noise)
    direction = float(np.dot(ep.error, ep.target_shift) /
                      (np.linalg.norm(ep.error) * np.linalg.norm(ep.target_shift) + 1e-6))
    gain = _clip(1.0 - residual * 1.8 + rng.normal(0, .03 * ep.noise), .25, 1.05)
    plan_conf = _clip(1.0 - ep.plan_error + rng.normal(0, .045 * ep.noise), 0.0, 1.0)
    features = np.asarray([
        residual + rng.normal(0, .010 * ep.noise),
        target_roi, distractor,
        float(np.linalg.norm(ep.target_shift - ep.error)),
        direction + rng.normal(0, .10 * ep.noise),
        plan_conf, ep.contact + rng.normal(0, .035 * ep.noise),
        ep.collision_risk + rng.normal(0, .035 * ep.noise),
        ep.remaining, ep.action_size,
        gain, float(ep.error[2]) + rng.normal(0, .012 * ep.noise),
        float(ep.target_shift[2]) + rng.normal(0, .012 * ep.noise),
        rng.uniform(0.0, 1.0),
        rng.uniform(0.0, 1.0),
    ], dtype=np.float64)
    return features


def branch_success(ep: Episode, action: str) -> bool:
    """Return whether the episode is resolved safely, including safe stop."""
    task_success, safe_stop = branch_outcome(ep, action)
    return task_success or safe_stop


def branch_outcome(ep: Episode, action: str) -> tuple[bool, bool]:
    """Return (task_completed, safely_stopped) for one deterministic branch."""
    residual = float(np.linalg.norm(ep.error))
    target = float(np.linalg.norm(ep.target_shift))
    if ep.kind == "unsafe":
        return False, action == "safe_stop"
    if action == "safe_stop":
        return False, False
    if action == "continue":
        final_error = residual + .55 * target + .50 * ep.plan_error
        contact = ep.contact - .35 * max(final_error - .08, 0.0)
    elif action == "local_repair":
        # Bounded geometric correction fixes plant drift, but chasing a moved
        # target or a bad WAM plan leaves a task-level error.
        corrected = max(0.0, residual - min(.095, residual * .82))
        final_error = corrected + .82 * target + .55 * ep.plan_error
        contact = ep.contact - .20 * max(target + ep.plan_error - .15, 0.0)
        if ep.kind in ("world_change", "wam_failure", "mixed"):
            final_error += .05
    elif action == "wam_replan":
        # Replanning uses world knowledge, but the extra handoff is late and
        # does not remove hardware tracking error.
        final_error = .42 * residual + .10 * target + .08 * ep.plan_error
        contact = ep.contact - .12 * max(residual - .10, 0.0)
        if ep.kind in ("clean", "distractor", "execution_drift"):
            final_error += .035  # unnecessary replan latency/perturbation
    elif action == "contact_recovery":
        final_error = residual + .25 * target + .35 * ep.plan_error
        contact = max(ep.contact, .82) if ep.contact < .35 else ep.contact - .08
    else:
        return False, False
    # Remaining time makes severe late interventions fail more often.
    final_error += .035 * (1.0 - ep.remaining)
    task_success = bool(final_error < .125 and contact > .48 and ep.collision_risk < .55)
    return task_success, False


def rule_action(x: np.ndarray) -> str:
    residual, target, distractor, _, _, plan_conf, contact, risk = x[:8]
    if risk > .58:
        return "safe_stop"
    if contact < .35:
        return "contact_recovery"
    if target > .12 and target > distractor * 1.35:
        return "wam_replan"
    if plan_conf < .52:
        return "wam_replan"
    if residual > .105:
        return "local_repair"
    return "continue"


def source_action(ep: Episode) -> str:
    return SOURCE_TO_ACTION[ep.kind]


def build_dataset(ids: range, split: str, seed: int):
    xs, ys, episodes = [], [], []
    for eid in ids:
        ep = make_episode(np.random.default_rng(seed + eid * 13), eid, split)
        obs_rng = np.random.default_rng(seed + eid * 17 + 3)
        xs.append(observe(ep, obs_rng))
        ys.append([float(branch_success(ep, action)) for action in ACTIONS])
        episodes.append(ep)
    return np.asarray(xs), np.asarray(ys), episodes


def pick(router: FailureRouter, x: np.ndarray, costs: dict[str, float], margin: float) -> str:
    return router.decide(x, costs=costs, uncertainty_margin=margin).action


def evaluate(episodes, xs, method, router=None, costs=None, margin=0.0):
    rows = []
    latencies = []
    for ep, x in zip(episodes, xs):
        started = time.perf_counter_ns()
        if method == "stale_wam":
            action = "continue"
        elif method == "always_local":
            action = "local_repair"
        elif method == "always_replan":
            action = "wam_replan"
        elif method == "rule_hybrid":
            action = rule_action(x)
        elif method == "source_oracle":
            action = source_action(ep)
        elif method == "counterfactual_oracle":
            # This oracle is only an upper-bound reference: it sees all five
            # branch outcomes for the current episode and picks a successful
            # action when one exists.  The deployed router never sees these
            # labels.
            action = next((candidate for candidate in ACTIONS
                           if branch_success(ep, candidate)), "continue")
        else:
            assert router is not None
            action = pick(router, x, costs or {}, margin)
        if method == "learned":
            latencies.append((time.perf_counter_ns() - started) / 1e6)
        task_success, safe_stop = branch_outcome(ep, action)
        best_success = max(branch_success(ep, candidate) for candidate in ACTIONS)
        rows.append({"id": ep.seed, "kind": ep.kind, "action": action,
                     "success": task_success or safe_stop,
                     "task_success": task_success,
                     "safe_stop": safe_stop,
                     "oracle_action": source_action(ep),
                     "oracle_success": branch_success(ep, source_action(ep)),
                     "source_route_match": action == source_action(ep),
                     "best_action_match": branch_success(ep, action) == best_success})
    return rows, latencies


def summarize(rows, baseline=None):
    n = len(rows)
    success = sum(r["success"] for r in rows)
    false_repair = sum(r["action"] == "local_repair" and r["kind"] in ("clean", "distractor") for r in rows)
    false_replan = sum(r["action"] == "wam_replan" and r["kind"] in ("clean", "distractor", "execution_drift") for r in rows)
    task_success = sum(r["task_success"] for r in rows)
    out = {"episodes": n, "resolved_rate": success / n,
           "task_success_rate": task_success / n,
           "safe_stop_episodes": sum(r["safe_stop"] for r in rows),
           "false_repair": false_repair, "false_replan": false_replan,
           "intervention_rate": sum(r["action"] != "continue" for r in rows) / n,
           "source_route_accuracy": sum(r["source_route_match"] for r in rows) / n,
           "best_action_accuracy": sum(r["best_action_match"] for r in rows) / n}
    if baseline is not None:
        b = {r["id"]: r for r in baseline}
        out["paired_rescue"] = sum((not b[r["id"]]["success"]) and r["success"] for r in rows)
        out["paired_harm"] = sum(b[r["id"]]["success"] and (not r["success"]) for r in rows)
    out["by_kind"] = {kind: {"n": sum(r["kind"] == kind for r in rows),
                              "success_rate": sum(r["success"] for r in rows if r["kind"] == kind) /
                              max(1, sum(r["kind"] == kind for r in rows))}
                     for kind in KINDS}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=2400)
    p.add_argument("--dev", type=int, default=400)
    p.add_argument("--test", type=int, default=800)
    p.add_argument("--epochs", type=int, default=260)
    p.add_argument("--seed", type=int, default=8121,
                   help="base seed for the disjoint train/dev/test episodes")
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "failure_router_hard")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    xtr, ytr, _ = build_dataset(range(args.train), "train", args.seed)
    router = FailureRouter(hidden=64, seed=11)
    source = np.zeros((len(ytr), 6), dtype=np.float64)
    # The source head is auxiliary; labels are generated from the episode and
    # never exposed as input features.
    _, _, train_eps = build_dataset(range(args.train), "train", args.seed)
    source_names = ("clean", "execution_drift", "world_change", "wam_failure", "contact_failure", "unsafe")
    for i, ep in enumerate(train_eps):
        label = {"distractor": "clean", "mixed": "world_change"}.get(ep.kind, ep.kind)
        source[i, source_names.index(label)] = 1.0
    router.fit(xtr, ytr, source, epochs=args.epochs)
    # Freeze intervention costs on the development split.
    dev_x, _, dev_eps = build_dataset(range(args.train, args.train + args.dev), "dev", args.seed)
    best = None
    for margin in (0.0, .02, .04, .06, .08):
        for local, replan in ((0.0, .00), (.01, .02), (.02, .04), (.04, .06), (.06, .08)):
            rows, _ = evaluate(dev_eps, dev_x, "learned", router, {"local_repair": local, "wam_replan": replan}, margin)
            s = summarize(rows)
            score = s["resolved_rate"] - .04 * s["false_repair"] / len(rows) - .015 * s["false_replan"] / len(rows)
            if best is None or score > best[0]:
                best = (score, {"local_repair": local, "wam_replan": replan}, margin)
    assert best is not None
    costs, margin = best[1], best[2]
    test_x, _, test_eps = build_dataset(range(args.train + args.dev, args.train + args.dev + args.test), "test", args.seed)
    methods = ("stale_wam", "always_local", "always_replan", "rule_hybrid",
               "learned", "source_oracle", "counterfactual_oracle")
    raw, reports = {}, {}
    latencies = []
    for method in methods:
        raw[method], method_latencies = evaluate(test_eps, test_x, method, router, costs, margin)
        if method == "learned":
            latencies = method_latencies
        reports[method] = summarize(raw[method], raw["stale_wam"] if method != "stale_wam" else None)
    for method in ("always_replan", "rule_hybrid"):
        reports["learned"][f"paired_vs_{method}"] = {
            "resolved_rescue": sum(not base["success"] and learned["success"]
                                    for base, learned in zip(raw[method], raw["learned"])),
            "resolved_harm": sum(base["success"] and not learned["success"]
                                 for base, learned in zip(raw[method], raw["learned"])),
            "task_rescue": sum(not base["task_success"] and learned["task_success"]
                                for base, learned in zip(raw[method], raw["learned"])),
            "task_harm": sum(base["task_success"] and not learned["task_success"]
                             for base, learned in zip(raw[method], raw["learned"])),
        }
    reports["learned"]["router_latency_ms"] = {
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "mean": float(np.mean(latencies)),
    }
    result = {"protocol": {"benchmark": "custom paired counterfactual hard recovery simulator",
                            "official_libero": False, "real_robot": False,
                            "features": 15, "actions": list(ACTIONS),
                            "splits": {"train": args.train, "dev": args.dev, "test": args.test},
                            "seed": args.seed,
                            "labels": "deterministic terminal outcome of each branch from the same latent state",
                            "hard_cases": list(KINDS)},
             "model": {"type": "two_hidden_layer_numpy_mlp", "hidden": 64,
                       "parameters": router.parameter_count},
             "selection": {"costs": costs, "uncertainty_margin": margin,
                           "development_score": best[0]},
             "test": reports, "runtime_seconds": time.perf_counter() - start}
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Hard counterfactual failure-router benchmark", "",
             "Custom paired simulator; not an official LIBERO or real-robot result.", "",
             f"Test episodes: {args.test}; model parameters: {router.parameter_count}; frozen costs: {costs}; margin: {margin}.", "",
             "| Method | Task complete | Resolved incl. safe stop | Rescue | Harm | False repair | False replan |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        s = reports[method]
        lines.append(f"| {method} | {s['task_success_rate']:.1%} | {s['resolved_rate']:.1%} | {s.get('paired_rescue', '-') } | {s.get('paired_harm', '-') } | {s['false_repair']} | {s['false_replan']} |")
    learned = reports["learned"]
    lines += ["", f"Learned router latency (Python CPU, feature vector already computed): p50 {learned['router_latency_ms']['p50']:.3f} ms, p95 {learned['router_latency_ms']['p95']:.3f} ms.",
              f"Pairwise learned vs rule hybrid: {learned['paired_vs_rule_hybrid']['resolved_rescue']} resolved rescues / {learned['paired_vs_rule_hybrid']['resolved_harm']} harms; task completion {learned['paired_vs_rule_hybrid']['task_rescue']} rescues / {learned['paired_vs_rule_hybrid']['task_harm']} harms.",
              "Labels come from the transparent deterministic branch model, which computes terminal outcomes from the same latent episode state; it is not a physics simulator and does not execute OpenWAM or LIBERO.", "source_oracle uses the known simulated failure source; counterfactual_oracle sees all branch outcomes and is an upper bound.", "The learned router is a routing prototype; the simulator does not establish real-arm or official LIBERO performance."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
