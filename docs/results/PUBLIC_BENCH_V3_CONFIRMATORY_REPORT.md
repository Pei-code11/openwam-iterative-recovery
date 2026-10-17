# Hybrid execution repair benchmark

LIBERO-plus simulation with OpenWAM (difficulty: hard). Execution drift is injected downstream of nominal WAM commands; target-scene change detection uses simulator object poses as an oracle.

## Success and control outcomes

| Condition | Method | Success | Local repair episodes | Mean repairs | Mean WAM replans | Tracking RMSE (m) | Tracking P95 (m) | Fast confidence | Fault score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arm_offset | fault_only | 8/10 (80.0%) | 0/10 | 0.00 | 0.00 | 0.0369 | 0.0724 | 1.000 | 0.440 |
| arm_offset | hybrid | 10/10 (100.0%) | 10/10 | 4.80 | 0.00 | 0.0287 | 0.0651 | 1.000 | 0.296 |
| arm_speed_loss | fault_only | 9/10 (90.0%) | 0/10 | 0.00 | 0.00 | 0.0299 | 0.0607 | 1.000 | 0.427 |
| arm_speed_loss | hybrid | 10/10 (100.0%) | 10/10 | 6.60 | 0.00 | 0.0270 | 0.0563 | 0.963 | 0.231 |
| arm_speed_offset | fault_only | 5/10 (50.0%) | 0/10 | 0.00 | 0.00 | 0.0439 | 0.0857 | 1.000 | 0.438 |
| arm_speed_offset | hybrid | 5/10 (50.0%) | 10/10 | 11.20 | 0.00 | 0.0337 | 0.0664 | 0.850 | 0.199 |
| clean | hybrid | 10/10 (100.0%) | 10/10 | 3.00 | 0.00 | 0.0191 | 0.0405 | 0.938 | 0.261 |
| clean | wam | 10/10 (100.0%) | 0/10 | 0.00 | 0.00 | 0.0209 | 0.0418 | 1.000 | 0.374 |
| distractor_shift | fixed_refresh | 10/10 (100.0%) | 0/10 | 0.00 | 0.00 | 0.0222 | 0.0542 | 1.000 | 0.431 |
| distractor_shift | hybrid | 8/10 (80.0%) | 10/10 | 6.00 | 0.00 | 0.0320 | 0.0656 | 0.900 | 0.276 |
| distractor_shift | stale_plan | 8/10 (80.0%) | 0/10 | 0.00 | 0.00 | 0.0343 | 0.0719 | 1.000 | 0.393 |
| target_shift | fixed_refresh | 7/10 (70.0%) | 0/10 | 0.00 | 0.00 | 0.0366 | 0.0813 | 1.000 | 0.344 |
| target_shift | hybrid | 8/10 (80.0%) | 10/10 | 5.10 | 1.00 | 0.0370 | 0.0906 | 1.000 | 0.281 |
| target_shift | local_only | 8/10 (80.0%) | 10/10 | 8.00 | 0.00 | 0.0411 | 0.0947 | 0.950 | 0.332 |
| target_shift | stale_plan | 5/10 (50.0%) | 0/10 | 0.00 | 0.00 | 0.0409 | 0.0907 | 1.000 | 0.362 |

## Paired outcome changes

| Condition | Method vs baseline | Rescues | Harms | Net | Exact p |
|---|---|---:|---:|---:|---:|
| arm_offset | hybrid vs fault_only | 2 | 0 | 2 | 0.5000 |
| arm_speed_loss | hybrid vs fault_only | 1 | 0 | 1 | 1.0000 |
| arm_speed_offset | hybrid vs fault_only | 0 | 0 | 0 | 1.0000 |
| clean | hybrid vs wam | 0 | 0 | 0 | 1.0000 |
| distractor_shift | fixed_refresh vs stale_plan | 2 | 0 | 2 | 0.5000 |
| distractor_shift | hybrid vs stale_plan | 0 | 0 | 0 | 1.0000 |
| target_shift | fixed_refresh vs stale_plan | 2 | 0 | 2 | 0.5000 |
| target_shift | hybrid vs stale_plan | 3 | 0 | 3 | 0.2500 |
| target_shift | local_only vs stale_plan | 3 | 0 | 3 | 0.2500 |

## Negative controls

Clean local correction: 10/10 episodes.
Clean WAM replan: 0/10 episodes.
Distractor-only WAM replan: 0/10 episodes.
