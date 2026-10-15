# OpenWAM Iterative Recovery Benchmark

Code for a custom paired OpenWAM/LIBERO simulation study of two recovery paths:

1. Bounded end-effector XYZ correction when the scene is unchanged but execution drifts.
2. WAM replanning when the task target changes, with distractor-only changes as a negative control.

The repository also contains a prototype lightweight failure-cause router that
chooses between continuing, local repair, WAM replanning, contact recovery, and
safe stop. Its first benchmark is a self-contained synthetic stress test; it is
not evidence of a real-arm or official LIBERO improvement.

This is a research prototype, not an official LIBERO benchmark or a real-robot controller. Scene-change routing in the reported experiment uses simulator object poses as an oracle. The execution perturbation is a Cartesian gain/bias wrapper, not a joint-level model of backlash, friction, or loose hardware. See [the frozen protocol](docs/RETRY06_PROTOCOL.md) and [benchmark protocol](docs/BENCHMARK_PROTOCOL.md).

## Contents

- `scripts/run_reframed_bench.py`: paired rollout and analysis entry point.
- `scripts/run_pipeline.py`: experiment orchestration template, configurable through environment variables.
- `openwam_recovery/failure_router.py`: small NumPy action-success and failure-source heads.
- `scripts/run_failure_router_benchmark.py`: episode-split counterfactual stress benchmark.
- `scripts/run_failure_router_hard_benchmark.py`: paired hard-case benchmark in which all candidate actions are executed from the same latent state.
- `docs/`: frozen protocol and benchmark scope.

Model weights, LIBERO assets, generated trajectories, logs, and server credentials are not included. The pipeline expects an existing OpenWAM checkout and checkpoint, a LIBERO checkout, and separately prepared calibration and job-manifest files.

## Setup

Use the Python environment supported by your OpenWAM and LIBERO versions. Install OpenWAM and LIBERO according to their upstream instructions, then make the benchmark modules importable. The rollout script expects the LIBERO package, `openwam2libero_interface`, MuJoCo/EGL dependencies, NumPy, and PyYAML.

For the orchestration script, configure paths and GPU indices before running:

```powershell
$env:WAM_BENCH_ROOT = "D:\robotics"
$env:OPENWAM_REPO = "D:\robotics\openwam"
$env:LIBERO_PATH = "D:\robotics\LIBERO"
$env:SIM_PYTHON = "D:\robotics\.venv\Scripts\python.exe"
$env:WAM_RUN_ROOT = "D:\robotics\outputs\recovery-benchmark"
$env:SIM_GPU = "0"
$env:WAM_GPU = "1"
$env:WAM_SERVER_PORT = "8962"
python scripts/run_pipeline.py
```

The selected GPUs must be available to the user. The launcher checks GPU memory, utilization, and compute-process conflicts before starting. EGL/library configuration can be passed through `WAM_EXTRA_LD_LIBRARY_PATH` and `EGL_VENDOR_LIBRARY_FILENAMES` when needed by the local machine.

The rollout entry point can also be run directly with `--stage evaluate`; use `python scripts/run_reframed_bench.py --help` for arguments. It requires a frozen calibration JSON and a disjoint evaluation state/job manifest. The orchestration template additionally expects its development job manifest and frozen scenario manifest under `WAM_RUN_ROOT`, as described in the protocol.

## Reproducibility and interpretation

The reported evaluation separates arm execution drift from task-scene changes and pairs methods by task, initial state, and perturbation. Thresholds and gains are selected on development episodes before held-out evaluation. Do not retune on the held-out outcomes and continue to describe them as independent validation.

Reported task success should be read alongside paired rescues/harms, false repairs, trajectory error, intervention count, and latency. A trigger or lower tracking residual alone does not establish that a task was rescued.

## Failure router prototype

Run the self-contained stress benchmark with:

```powershell
python scripts/run_failure_router_benchmark.py --output outputs/failure_router_benchmark
```

The protocol and result are documented in [docs/FAILURE_ROUTER_PROTOCOL.md](docs/FAILURE_ROUTER_PROTOCOL.md) and [docs/FAILURE_ROUTER_RESULTS.md](docs/FAILURE_ROUTER_RESULTS.md). The ordinary held-out smoke test is synthetic and deliberately difficult; its rule hybrid reaches 100%, so that test does not establish a learned-router advantage. The harder paired benchmark below is the relevant stress test: its learned router reaches 83.25% task success versus 73.13% for the rule hybrid, with 79 false replans and 3 false repairs.

The harder paired prototype can be run with:

```powershell
python scripts/run_failure_router_hard_benchmark.py --output outputs/failure_router_hard
```

The frozen held-out result and per-source analysis are in
[docs/FAILURE_ROUTER_HARD_RESULTS.md](docs/FAILURE_ROUTER_HARD_RESULTS.md).
The three-seed robustness check is in
[docs/FAILURE_ROUTER_SEEDS.md](docs/FAILURE_ROUTER_SEEDS.md).
This is the most relevant stress test for the proposed fast discriminator:
it measures route choice and recovery value, rather than anomaly detection
alone.

It reports clean and distractor controls, execution drift, target movement,
WAM plan failure, contact failure, mixed failure, and unsafe states. Each
episode is cloned into `continue`, bounded local repair, WAM replan, contact
recovery, and safe-stop branches; the branch terminal result supplies the
counterfactual label. The current run is still a custom simulator and must
not be described as an official LIBERO or real-robot result.

## License

No license is granted by this repository at this time.
