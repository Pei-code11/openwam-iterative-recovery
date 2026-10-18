# Hierarchical timing benchmark

`scripts/run_hierarchical_timing_benchmark.py` is a deterministic architecture
stress test for long tasks. It is intentionally lightweight and does not load
OpenWAM weights. The result is a simulation prediction, not an official LIBERO
score or a real-robot measurement.

## Protocol

The default run uses 400 paired episodes for each of seven conditions. Each
episode contains 3--6 sequential subtasks and all methods receive the same
latent episode, including the same base failures and perturbation timing.

Conditions:

* clean;
* arm position offset;
* arm speed loss;
* combined speed and position error;
* target-object movement;
* distractor-only movement;
* WAM plan failure.

Methods:

* `wam_only`: no recovery;
* `local_no_time`: spatial repair without a deadline check;
* `temporal_repair`: bounded repair with a contact-window deadline;
* `fixed_target_replan`: diagnostic immediate refresh for target movement;
* `three_level_hybrid`: temporal repair plus target replan and macro recovery.

The time-aware policy checks whether command, settling, validation, and a safety
margin fit before the next contact/close/release event. It also returns a
bounded action-rate scale and gripper-time shift. A world change has priority
over local repair.

## Reproduction

```powershell
python scripts/run_hierarchical_timing_benchmark.py `
  --episodes-per-condition 400 `
  --seed 20261018 `
  --output outputs/hierarchical_timing_benchmark
```

The output contains `results.json` and `REPORT.md`. The paired comparison is
computed against `wam_only` using the same episode identifiers.

## Interpretation

The benchmark is designed to test the architecture's timing hypothesis:
spatial residual reduction alone is insufficient when the correction misses a
contact or gripper window. Before claiming a real-system result, connect the
same interfaces to an OpenWAM RPC, obtain event times from the actual action
chunk, and repeat the evaluation with simulator or hardware observations.

