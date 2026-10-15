# Reframed recovery benchmark v1

## Question and benchmark split

This custom OpenWAM/LIBERO-10 benchmark separates two failure causes that require different responses.

1. **Execution-chain drift:** the scene and WAM policy remain valid. After the policy outputs a nominal Cartesian command, a plant wrapper modifies only the command delivered to MuJoCo. Conditions are a 15% Cartesian gain reduction, a persistent 4% normalized-command bias on one table-plane axis, and both together. The gain/bias fault persists from its seeded onset to episode end and starts at steps 8, 80, or 160 (early/middle/late). The local geometric controller observes the achieved EEF state, estimates its error against the nominal WAM trajectory, applies bounded XYZ compensation, and validates whether tracking error shrinks. It must not move the target or change WAM's requested action.
2. **World change:** no plant fault. At step 15, the task object moves 5 cm (first ten held-out states per task) or 7.5 cm (remaining ten); a 6 cm irrelevant-object shift is the negative control. Compare stale action chunk, task-target-triggered WAM refresh, and a single fixed refresh at step 16.

The execution-drift primary evaluation uses tasks 0, 1, 4, 6, 7; development uses task 0 at states 20/21, task 1 at 22/23, and task 4 at 20/21, with four preregistered controller gains (0.3, 0.5, 0.65, 0.8); calibration states are [27,28]; held-out execution tests use the exact 20 per-task states in the source CVPR manifest, selected by its unseen-state rule. All are paired by task, state, condition, and seeded bias direction. World-change evaluation reuses all 100 target-and-distractor plus 50 distractor-only cases from the frozen CVPR manifest, with exact source states, magnitudes, and injection steps. Existing CVPR results are retained as historical evidence, with new code run logged under this benchmark.

## Methods and primary outcomes

For plant drift: fault-only, local geometric correction with four-step verification, and clean nominal OpenWAM control. Primary outcome is achieved trajectory tracking RMSE/P95; task success, paired rescue/harm, clean false-correction rate, correction delay, and the fraction of validation windows with lower residual are secondary outcomes. WAM action chunks are anchored at each newly generated chunk because the policy conditions on achieved proprioception; accumulated error inside a chunk is never silently rebased. A correction is held to a maximum normalized XYZ norm of 0.20, and after two failed four-step verifications the controller yields to a newly generated WAM chunk.

For scene changes: stale plan, target-object oracle monitor with 4-step polling, and one fixed refresh at step 16. Primary outcome is paired LIBERO success. Report rescue/harm, exact McNemar tests, task-wise rates, trigger/replan latency, and distractor-only false refresh. The oracle measures routing efficacy, not a learned visual detector.

## Freeze and interpretation

The development batch may be used once to reject an invalid plant implementation or choose one controller gain from the frozen candidates [0.3, 0.5, 0.65, 0.8]. Freeze that choice before held-out evaluation. Held-out outcomes will not be used to retune thresholds, select states, remove failures, or restart a favorable subset. A positive execution result requires lower median and P95 within-chunk tracking error than fault-only, with no clean-condition correction; task-success benefit is claimed only if paired rescue exceeds harm. A positive replanning result requires a positive paired benefit over stale WAM; fixed 16-step refresh remains a budget baseline.

## Limits

The arm plant is represented by a downstream Cartesian command gain/bias wrapper in the MuJoCo simulator. It approximates control gain mismatch and persistent calibration offset; it is not a joint-level motor/friction/backlash model and is not a real-arm validation. Scene-change sensing uses simulator target identity/pose as an oracle. Results are custom perturbation benchmarks, not official LIBERO leaderboard scores. Five tasks and a finite set of perturbations do not establish arbitrary task or object generalization.
