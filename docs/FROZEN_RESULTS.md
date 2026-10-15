# Frozen Retry06 evaluation summary

The numbers below are from a held-out custom OpenWAM/LIBERO simulation evaluation. They are included to document the code's provenance; they are not official LIBERO leaderboard results and do not represent a real robot.

| Condition | Baseline | Hybrid | Rescue / harm |
|---|---:|---:|---:|
| Cartesian arm offset | 95% | 98% | 4 / 1 |
| Arm speed loss | 95% | 94% | 0 / 1 |
| Speed plus offset | 81% | 88% | 11 / 4 |
| Clean control | 99% | 97% | 0 / 2 |
| Target-object shift | 73% | 90% | 20 / 3 |
| Distractor-only shift | 96% | 96% | 1 / 1 |

The held-out execution-drift run contained 100 paired episodes per condition. The world-change run contained 100 target-shift and 50 distractor-only paired episodes. Target-change routing used simulator pose oracle access; the public code does not claim a learned visual detector from these numbers.

The selected development configuration was correction gain `0.65`, a threshold adjustment of `-0.005 m`, two consecutive four-step checks, and direction cosine `0.2`. This configuration was frozen before opening the held-out evaluation. The held-out results must not be used to retune these values while retaining an independent-validation claim.

The results support separating plant drift from task-relevant world change. They also show the remaining limitation: local correction can be over-sensitive on clean or nuisance-only episodes, and pure speed loss was not rescued reliably.
