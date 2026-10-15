# OpenWAM recovery benchmark retry05: persistent residual gate

## Research question

Can a small residual controller repair execution-chain drift while leaving
valid WAM actions alone, and can a separate task-relevant scene monitor route
meaningful world changes back to WAM for replanning?

## Disturbance split

The paired LIBERO-10 simulation separates two causes that call for different
responses.

* **Execution-chain drift, same scene and same valid WAM plan.** At steps 8,
  80, or 160, apply a persistent 15% Cartesian command-gain loss, a 4%
  normalized-command bias on a seeded table-plane axis, or both. The local
  repair gate receives only the end-effector tracking residual. It never reads
  the disturbance label or onset flag. Clean WAM episodes use the same gate
  and measure false repairs. Repair is a bounded XYZ compensation to WAM's
  current command. Residual is checked every four actions; two failed checks
  hand control back to WAM.
* **Task-relevant scene change.** Move the goal object by 5 cm or 7.5 cm, with
  no plant fault. The local repair is suspended and WAM is asked to replan.
  Compare against continuing the stale plan and refreshing WAM at a fixed
  step. A 6 cm distractor-only move is the irrelevant-change control. The
  current scene monitor reads simulator object pose and identity; this tests
  routing separately from perception and is not a learned visual detector.

## Local repair gate

Calibration from clean task/state rollouts fixes per-task XYZ action scales
and a residual threshold at `max(10 mm, clean chunk-residual P95 + 5 mm)`.
Observations remain at a fixed four-action interval. Retry05 tests the
following development-only settings:

| Candidate | Threshold | Consecutive checks | Direction agreement |
|---|---:|---:|---:|
| p2 q95 | clean P95 | 2 | cosine >= 0.2 |
| p2 q95+5 | P95 + 5 mm | 2 | cosine >= 0.2 |
| p2 q95+10 | P95 + 10 mm | 2 | cosine >= 0.2 |
| p2 q95+15 | P95 + 15 mm | 2 | cosine >= 0.2 |
| p3 q95 | clean P95 | 3 | cosine >= 0.2 |

Residual directions are reset at WAM action-chunk boundaries and after a
repair attempt. The threshold and persistence settings are selected using
development episodes only. Selection first minimizes clean episodes with a
false repair; among ties it maximizes mean paired fault-window EEF tracking
RMSE reduction. Correction gains 0.30, 0.50, 0.65, and 0.80 are then compared
with the selected gate held fixed. The held-out set is opened only if the
selected development configuration has zero clean false repairs and positive
paired fault-window RMSE reduction.

## Splits and outcomes

Calibration uses task states 27 and 28. Development uses task/state pairs
already listed in the frozen manifest (tasks 0, 1, and 4; states 20--23).
Held-out drift evaluation uses 100 paired task/state clusters over tasks 0,
1, 4, 6, and 7. The world-change set reuses 100 target-shift and 50
distractor-only cases from the frozen CVPR manifest. The held-out task states,
scene cases, onset phases, and manifest hashes cannot be used for selection.

Primary drift metrics are fault-window trajectory RMSE and P95, with paired
rescue/harm counts, task success, clean false-repair rate, repair delay,
per-check residual change, and WAM fallback rate as additional outcomes.
World-change outcomes include success against stale-plan and fixed-refresh
baselines, detection and first-refreshed-action delay, and irrelevant-change
false refresh/repair rates. Report task-wise results and paired uncertainty;
do not tune on held-out outcomes.

## Limits

This is a custom OpenWAM/LIBERO simulation benchmark, not an official LIBERO
leaderboard result. Cartesian gain and command bias approximate execution
calibration drift; they do not simulate joint-level backlash, friction, or a
loose screw. EEF and object poses are simulator state, and target-change
routing uses object-pose oracle access. Results do not demonstrate camera-only
drift estimation, learned scene detection, arbitrary-object generalization,
or real-arm performance.

## Retry04 audit trail

Retry04's three threshold batches completed, but its selector violated this
protocol when no candidate had zero clean false repairs: it selected q95 with
4/6 false-repair episodes despite q95+10 mm having 1/6. Its partial gain
batch was stopped before any held-out run. Retry04 remains preserved as an
exploratory, invalid-selection run and is excluded from retry05 evidence.
