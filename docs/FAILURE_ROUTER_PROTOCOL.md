# Lightweight failure-cause router protocol

## Research question

Can a small online discriminator decide whether a WAM rollout should continue,
apply a bounded local repair, ask WAM to replan, run contact recovery, or stop
safely? The intended contribution is **failure-cause-aware recovery routing**.
The discriminator is a control layer after WAM; it does not replace WAM's
world knowledge and it does not generate the full action sequence.

## Interface

At each observation window the fast path receives low-dimensional deployable
signals:

* end-effector residual magnitude, trend, and direction agreement;
* target-region and distractor-region visual change estimates;
* target motion estimate, plan confidence, and observability;
* gripper/contact state and collision risk;
* current action magnitude, phase, remaining time, and command-gain estimate.

The public implementation uses a two-hidden-layer NumPy MLP with 64 units per
layer (5,899 parameters). It has two heads. The action head estimates
`P(success | observation, action)` for `continue`, `local_repair`,
`wam_replan`, `contact_recovery`, and `safe_stop`. The source head estimates
the posterior over nominal/nuisance change, execution drift, task-relevant
world change, WAM plan failure, contact failure, and unsafe state.

For a candidate action `a`, the route utility is:

\[
 U(a|x) = \hat P(S=1|x,a) - c_a.
\]

The router chooses the largest utility. A small top-two margin is treated as
“not sure”; the fast path continues for another observation unless the safety
head is dominant. `repair_benefit` is reported as
`P(success|local_repair) - P(success|continue)`, rather than using residual
magnitude as a proxy for benefit.

## Counterfactual labels and splits

For each decision state, the simulator branches into the five actions. The
original smoke benchmark records a soft final-success target. The hard
benchmark records the deterministic terminal outcome of every branch from the
same latent state. The branch outcome is never provided to the deployed
router. Episodes, rather than frames, are split into train, development, and
held-out partitions. Development chooses intervention costs and the
uncertainty margin. Held-out outcomes are not used to retune them.

The included benchmark is a self-contained stress test so that the routing
logic can run without a LIBERO or OpenWAM installation. It contains clean and
distractor-only controls, execution drift, target/world changes, WAM plan
failures, contact failures, unsafe states, and mixed drift plus world change.
The same bounded geometric local-repair controller is used for rule and learned
routes; only the route decision changes.

## Required reporting

Report task success with paired rescue/harm, clean false-repair episodes,
distractor-only false-repair/replan episodes, source confusion, first
intervention delay, route latency, WAM calls, local repair count, and model
parameters. A trigger is not a rescue: a rescue requires the paired baseline to
fail and the routed rollout to succeed. A harm requires the reverse. A
safe-stop counts as resolved for the safety metric, but not as task success.

## Scope

This benchmark is custom synthetic OpenWAM recovery evidence. It is not an
official LIBERO leaderboard result and is not a real-arm result. It does not
claim to reproduce the private Jev implementation. Jev is used only as a
high-level analogy for a fast state-evaluation/router layer. A future claim
about visual failure detection should replace the synthetic ROI signals with
camera-derived features and compare against SAFE, VLA-FAIL, and RAFAIL-style
detectors.
