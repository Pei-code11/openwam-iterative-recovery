# Hierarchical WAM timing prediction report

This report records the reproducible architecture stress simulation run with
seed `20261018`, 400 paired episodes per condition, and 3--6 sequential
subtasks per episode. It is a prediction for the proposed controller. It is
not an official LIBERO benchmark, an OpenWAM checkpoint result, or a real-arm
measurement.

## Main results

| Condition | WAM only | No-time local repair | Time-aware repair | Three-level hybrid |
|---|---:|---:|---:|---:|
| clean | 89.2% | 89.2% | 89.2% | 89.2% |
| arm position offset | 53.2% | 46.5% | 72.8% | 72.8% |
| arm speed loss | 4.0% | 17.0% | 34.5% | 34.5% |
| arm speed + position | 0.0% | 0.2% | 19.5% | 19.5% |
| target movement | 0.0% | 0.0% | 0.0% | 88.8% |
| distractor movement | 85.8% | 64.0% | 85.8% | 85.8% |
| WAM plan failure | 0.0% | 0.0% | 0.0% | 78.8% |

The time-aware policy returned a valid bounded repair on 94--95% of attempted
true arm-fault corrections in the position and speed conditions, while the
no-time policy produced late repairs in 26--50% of episodes. The hybrid
replanned target movement in every target-shift episode and recovered 355/400
paired failures. It recovered 315/400 simulated WAM plan failures through the
macro layer.

## Paired rescue/harm

Relative to `wam_only`, the time-aware repair path produced:

* position offset: 78 rescues / 0 harms;
* speed loss: 122 / 0;
* combined speed and position: 78 / 0;
* distractor movement: 0 / 0.

The no-time path produced 105 harms on position offset and 87 harms on
distractor movement, mostly from late or irrelevant corrections.

## What the simulation supports

The result supports the design hypothesis that a small repair should estimate
the remaining contact window, adjust motion timing and gripper timing, and
validate the correction before continuing. A target or plan change should be
handled by the middle/large planner rather than by a spatial repair head.

## Limits

The simulator uses a simplified Cartesian plant, synthetic phase and gripper
errors, and an oracle target-change label for the hybrid route. It does not
measure a real OpenWAM policy or visual long-horizon planner. The next required
step is an OpenWAM/LIBERO run with event times extracted from actual action
chunks and the same paired conditions.

