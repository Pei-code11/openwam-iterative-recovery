# Multi-seed hard benchmark check

The primary result uses base seed `8121`, with 2,400 train, 400 development,
and 800 held-out episodes. To check whether the routing gain is an artifact of
one episode draw, the same frozen protocol and model were run with two further
base seeds. The model was retrained on each training split and intervention
costs were selected on that run's development split; no held-out outcomes were
used for tuning.

| Base seed | Stale WAM task success | Always replan | Rule hybrid | Learned router | Learned resolved | Rescue | Harm | False repair | False replan |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8121 | 30.75% | 75.00% | 73.12% | **83.25%** | 88.62% | 463 | 0 | 3 | 79 |
| 9121 | 33.25% | 73.88% | 74.25% | **84.12%** | 89.12% | 447 | 0 | 2 | 286 |
| 10121 | 27.38% | 75.62% | 73.12% | **83.88%** | 90.25% | 503 | 0 | 3 | 263 |

Across these three runs, the learned router averages **83.75% task success**
and improves over the rule hybrid by 10.25 percentage points on average. It
improves over always replanning by 8.92 points on average. The paired harm
count is zero in all three runs. The main cost is false replanning under noisy
visual evidence: it varies from 79 to 286 episodes, while false repair stays
at 2--3 episodes. This is a limitation of the current synthetic router, not a
claim that the model is calibrated for physical deployment.

Reproduce one run with:

```powershell
python scripts/run_failure_router_hard_benchmark.py --seed 9121 `
  --output outputs/failure_router_hard_seed_9121
```

These results are still from the custom paired counterfactual simulator. They
are not official LIBERO scores, a Jev reproduction, or real-arm evidence.
