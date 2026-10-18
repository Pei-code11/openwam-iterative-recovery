# Frozen evaluation protocol

The primary evaluation contains five fixed public LIBERO-Plus scenarios, ten
held-out states per scenario (states 2 through 11), six predeclared conditions, and the
paired methods in `manifest.json` (600 rows). States 0 and 1 are calibration
only. Every method is run on the same task/state/condition and deterministic
perturbation seed.

Execution conditions use persistent hard gain/bias faults. Scene conditions
inject a hard target or distractor displacement during early approach. The
hybrid route continuously observes, uses the fast plant estimator plus bounded
geometric XYZ repair for execution drift, validates residual decrease, and
returns to WAM after failed validation. Target changes are sent to WAM for a
fresh chunk; distractor motion is a negative control.

Primary outcomes are paired task success and rescue/harm against the stale or
fault-only baseline. Secondary outcomes are clean and distractor false
interventions, tracking RMSE/P95, correction and replan latency, repair
verification rate, action deviation, and task-wise Wilson intervals. No
held-out result may be used to select thresholds, gains, states, or conditions.
