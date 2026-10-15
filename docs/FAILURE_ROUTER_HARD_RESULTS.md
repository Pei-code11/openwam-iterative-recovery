# Hard failure-router benchmark

## Purpose

This experiment tests the part of the proposal that a small model should
decide **what to do after a suspicious observation**. It does not try to
replace WAM's world model or generate a complete task plan. The fast model
estimates the terminal success of five routes and chooses the route with the
largest success estimate after a fixed intervention cost:

\[
U(a\mid x)=\widehat P(\mathrm{success}\mid x,a)-c_a.
\]

The routes are `continue`, `local_repair`, `wam_replan`,
`contact_recovery`, and `safe_stop`. The local repair branch is a bounded
geometric correction. In a real implementation, its input would be camera,
robot-state, and contact features; this benchmark uses 15 low-dimensional
features so the routing logic is reproducible without OpenWAM or LIBERO.

## Protocol

`scripts/run_failure_router_hard_benchmark.py` creates one latent state for
each episode and forks that state into all five candidate routes. The labels
are the deterministic terminal outcomes of those branches. The router never
sees those labels at deployment time. Episodes, rather than individual
frames, are split into train/dev/test. Intervention costs and the uncertainty
margin are selected on the development split and then frozen.

The held-out run contains 800 episodes. Its test distribution is broader and
noisier than training and includes:

* clean and distractor-only changes as negative controls;
* Cartesian execution drift, modelling a small plant/control mismatch;
* task-relevant target/world change;
* a bad WAM nominal plan;
* contact failure and unsafe contact risk;
* mixed execution drift plus world change.

This is a custom paired counterfactual simulator. It is not an official
LIBERO score, a reproduction of the linked Jev system, or evidence from a
physical arm. The positive-control state is intentionally difficult so that a
one-size-fits-all intervention exposes its failure mode.

## Held-out result

The frozen router has two 64-unit hidden layers and 5,899 trainable
parameters. The following table is from `outputs/failure_router_hard/results.json`.

| Method | Task complete | Resolved (incl. safe stop) | Rescue | Harm | False repair | False replan | Best-action match |
|---|---:|---:|---:|---:|---:|---:|---:|
| stale WAM (`continue`) | 30.8% | 30.8% | — | — | 0 | 0 | 41.8% |
| always local repair | 38.8% | 38.8% | 100 | 36 | 169 | 0 | 49.8% |
| always WAM replan | 75.0% | 75.0% | 354 | 0 | 0 | 365 | 86.0% |
| rule hybrid | 73.1% | 78.4% | 381 | 0 | 0 | 1 | 89.4% |
| learned fast router | **83.2%** | **88.6%** | **463** | **0** | **3** | 79 | **99.6%** |
| source oracle | 82.0% | 87.4% | 453 | 0 | 0 | 0 | 98.4% |
| counterfactual oracle | 83.6% | 89.0% | 466 | 0 | 3 | 10 | 100.0% |

Relative to stale WAM, the learned router gains 52.4 percentage points in
task completion and 58.0 points in resolved episodes. It gains 8.2 points in
task completion over always replanning and 10.1 points over the hand-written
rule hybrid. The three false repairs occur on clean or distractor controls.
The 79 false replans are the main cost: the router is aggressive about asking
WAM to look again when the visual evidence is noisy. There is no paired harm
against stale WAM in this run because the learned route did not turn a
successful stale-WAM episode into a failed task.

The source oracle is **not** the upper bound: knowing only the nominal source
does not resolve mixed and late cases. The counterfactual oracle sees all
branch outcomes and is the simulator's upper bound. The learned router is
within 0.4 percentage points of that upper bound, while using only the noisy
deployable features.

## Breakdown by failure source

| Source | n | Stale WAM resolved | Learned router resolved | Counterfactual oracle resolved |
|---|---:|---:|---:|---:|
| clean | 91 | 97.8% | 100.0% | 100.0% |
| distractor | 78 | 98.7% | 100.0% | 100.0% |
| execution drift | 196 | 20.4% | 74.5% | 75.0% |
| target/world change | 146 | 27.4% | 99.3% | 100.0% |
| WAM plan failure | 93 | 0.0% | 100.0% | 100.0% |
| contact failure | 66 | 0.0% | 100.0% | 100.0% |
| mixed drift + world change | 87 | 0.0% | 54.0% | 55.2% |
| unsafe | 43 | 0.0% | 100.0% | 100.0% |

The per-source percentages above use the resolved definition, so `unsafe`
includes a correctly selected safe stop. For task-completion rates, the
learned router is 74.5% on execution drift, 99.3% on target/world change,
100% on WAM-plan and contact cases, and 54.0% on mixed cases. The remaining
gap is concentrated in mixed cases. This is useful for the paper: it shows why
“large residual means local repair” is insufficient and why the router must
estimate the value of a route, not only detect an anomaly.

## Model size and interpretation

The router is a NumPy MLP with 15 inputs, two 64-unit ReLU layers, five
action-success outputs, and six auxiliary source outputs. It has 5,899
parameters (about 23 KB in float32). The default command takes about 10.2
seconds including synthetic data generation, training, and evaluation; this
is **not** a per-frame control latency measurement. With features already
computed, router decision time is 0.144 ms at p50 and 0.205 ms at p95 in this
run. These figures exclude camera preprocessing, feature extraction, WAM
handoff, and simulator/robot stepping.

Against always-replan, the learned route has 68 task-completion rescues and 2
harms (111 resolved rescues and 2 resolved harms). Against the rule hybrid it
has 84 task-completion rescues and 3 harms. Paired rescue/harm is reported
alongside, rather than instead of, task completion.

## What this does and does not establish

The result supports a concrete systems claim: a lightweight, task-conditioned
failure-cause router can improve recovery on a paired difficult distribution
when it chooses among continuation, bounded repair, and expensive replanning.
It does not establish generalization to unseen physical hardware, official
LIBERO performance, or superiority over every failure detector. The next
credible experiment is to replace the synthetic features with camera/state
features from OpenWAM/LIBERO, retain the frozen split and paired rescue/harm
metrics, and compare against SAFE-, VLA-FAIL-, and RAFAIL-style monitors.
