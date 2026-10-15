# Failure router and the Jev analogy

The linked Zhihu article presents Jev as a small fast decision model placed in
front of a slower model. Its public description uses a compact decision
envelope with a selected `choice`, a `score`, and a flag indicating that the
fast model is unsure and should defer. The article was not accessible for
independent full-text verification in this run, and it is not a peer-reviewed
robotics benchmark or a reproducible public implementation. This repository
therefore uses Jev only as a systems analogy and does not claim to reproduce it.

## Robotics mapping

The proposed OpenWAM layer has the same division of labor:

* the fast router reads low-dimensional online state features;
* `choice` selects `continue`, `local_repair`, `wam_replan`,
  `contact_recovery`, or `safe_stop`;
* `score` reports decision decisiveness on a 0--10 scale;
* `noul` asks for another observation or a slower WAM check when the two best
  routes are too close;
* WAM keeps the world model and generates a new task-level plan when the
  target, task relation, or original plan has changed.

The fast JSON envelope is available through
`RouteDecision.as_fast_json()`. The score is explicitly a decisiveness score,
not a calibrated probability. The action head separately estimates
`P(success | observation, action)`, and the source head estimates the likely
failure source. This makes the claim testable without pretending that the
small model contains WAM's world knowledge.

## What the hard benchmark tests

`run_failure_router_hard_benchmark.py` creates one latent state per episode and
executes five counterfactual branches from that same state. The test includes
hardware-like Cartesian drift, task-relevant target movement, WAM plan failure,
contact failure, mixed drift plus world change, unsafe contact risk, and two
negative controls. The train, development, and test episodes are disjoint.

The frozen 800-episode run gives the following prototype result. “Task
complete” excludes safe-stop episodes; “resolved” includes a safe stop as a
safe terminal response.

| Method | Task complete | Resolved incl. safe stop | Paired rescue | Paired harm | False repair | False replan |
|---|---:|---:|---:|---:|---:|---:|
| Stale WAM | 30.8% | 30.8% | - | - | 0 | 0 |
| Always local repair | 38.8% | 38.8% | 100 | 36 | 169 | 0 |
| Always replan | 75.0% | 75.0% | 354 | 0 | 0 | 365 |
| Rule hybrid | 73.1% | 78.4% | 381 | 0 | 0 | 1 |
| Learned fast router | **83.2%** | **88.6%** | 463 | 0 | 3 | 79 |
| Source oracle | 82.0% | 87.4% | 453 | 0 | 0 | 0 |
| Counterfactual oracle | 83.6% | 89.0% | 466 | 0 | 3 | 10 |

The learned router is 8.2 task-completion points above always replanning and
10.1 points above the hand-written hybrid in this custom stress simulator. It
also has a high unnecessary-replan count, so the result is evidence for a
useful routing prototype, not proof that a 5,899-parameter model is ready for
deployment. The source oracle is not an upper bound because the source label
does not always identify the best branch under late or mixed perturbations; the
counterfactual oracle is the upper bound for this simulator. The next required
step is to replace the synthetic features and branch dynamics with actual
OpenWAM/LIBERO state forks and camera-derived features, then compare the same
baselines under a frozen protocol.

## Related literature direction

The closest established directions are robot execution monitoring and failure
detection, VLA failure detection, recovery policy learning, and task-aware
replanning. The router's distinct experimental question is narrower: given an
observed deviation, does a bounded local correction have higher expected task
success than continuing or invoking the expensive planner? A paper should
compare against failure monitors such as SAFE/VLA-FAIL/RAFAIL-style detectors
when their implementations and protocols are available, and should report
paired rescue/harm, false interventions, latency, and model size.
