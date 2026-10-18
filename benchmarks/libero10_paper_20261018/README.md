# LIBERO-10 paper-scale paired benchmark

This manifest expands the pilot to ten held-out initial states for each of
five public LIBERO-Plus scenarios (50 matched scenes) and the same six
conditions and paired methods as the public hard benchmark, giving 600 paired
rollout rows. Calibration uses states 0 and 1 only; evaluation uses states
2--11. The task list, perturbation amplitudes, timing, threshold rule, and
metrics are frozen before evaluation.

The benchmark is a custom OpenWAM/LIBERO simulation study. Execution drift is
injected as a downstream Cartesian gain/bias plant wrapper. World-change
routing uses simulator target pose and identity as an oracle. Results are not
official LIBERO leaderboard scores and do not establish real-robot performance.

The run must reserve simulator GPU1 and leave WAM on GPU2. GPU0 and GPU3 are
outside this experiment.
