#!/usr/bin/env bash
set -euo pipefail
T=/mnt/workspace/openwam_async_handoff_20261009
B=$T/public_bench_libero_plus
R=$B/results_paper_scale_v1
P=$B/LIBERO-plus-main
PY=$T/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
export PYTHONPATH="$R/code:$B/extra_packages:$P:$T/source/openwam:$T/source/openwam/benchmarks/libero"
export LIBERO_PATH="$P" LIBERO_CONFIG_PATH="$B/results_v1/libero-config"
export WAM_PROMPTS_JSON="$R/prompts.json" WAM_CPU_TEXT=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LD_LIBRARY_PATH="$B/private_magick/root/usr/lib/x86_64-linux-gnu:$T/experiment_results/reframed_recovery_bench_v1_20261013_retry06_positive_control/private_driver/NVIDIA-Linux-x86_64-570.207:$T/cuda570:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$T/experiment_results/reframed_recovery_bench_v1_20261013_retry06_positive_control/private_driver/egl_vendor.d/10_nvidia.json"
mkdir -p "$R/logs"

if [[ ! -f "$R/calibration.json" ]]; then
  "$PY" -u "$R/code/run_libero_plus_bench.py" --stage calibrate \
    --output "$R/calibration_status.json" --tasks 287 677 1099 1932 2092 \
    --states 0 1 --calibration-states 0 1 --max-steps 700 --wam-port 8977 \
    --sim-gpu 1 --difficulty hard 2>&1 | tee "$R/logs/calibration.log"
fi

if [[ ! -f "$R/evaluation_status_v2.json" ]]; then
  "$PY" -u "$R/code/run_libero_plus_bench.py" --stage evaluate \
    --output "$R/evaluation_status_v2.json" --tasks 287 677 1099 1932 2092 \
    --states 2 3 4 5 6 7 8 9 10 11 --calibration-states 0 1 --calibration "$R/calibration.json" \
    --job-manifest "$R/manifest.json" --max-steps 700 --wam-port 8977 \
    --sim-gpu 1 --difficulty hard 2>&1 | tee "$R/logs/evaluation.log"
fi
