"""Owned remote runner for the frozen reframed recovery benchmark."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


TASK_ROOT = Path(os.environ.get("WAM_BENCH_ROOT", Path.cwd())).expanduser().resolve()
BASE = Path(os.environ.get("OPENWAM_REPO", TASK_ROOT / "openwam")).expanduser().resolve()
LIBERO = Path(os.environ.get("LIBERO_PATH", TASK_ROOT / "LIBERO")).expanduser().resolve()
UPSTREAM = Path(os.environ.get(
    "OPENWAM_LIBERO_BENCHMARKS", BASE / "benchmarks/libero/latent_alignment"
)).expanduser().resolve()
RUN_ROOT = Path(os.environ.get(
    "WAM_RUN_ROOT", TASK_ROOT / "outputs/reframed_recovery_bench"
)).expanduser().resolve()
RUN_CODE = RUN_ROOT / "code"
SIM_PY = Path(os.environ.get("SIM_PYTHON", sys.executable)).expanduser().resolve()
SIM_GPU = int(os.environ.get("SIM_GPU", "0"))
WAM_GPU = int(os.environ.get("WAM_GPU", "1"))
PORT = int(os.environ.get("WAM_SERVER_PORT", "8962"))
TASKS = [0, 1, 4, 6, 7]
CALIBRATION_STATES = [27, 28]
EVAL_SCRIPT = Path(__file__).resolve().with_name("run_reframed_bench.py")
OWNED: list[subprocess.Popen] = []


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def process_command(pid: int) -> str:
    path = Path(f"/proc/{pid}/cmdline")
    return path.read_bytes().replace(b"\0", b" ").decode(errors="replace") if path.exists() else ""


def gpu_snapshot(*, require_idle: bool = False) -> dict:
    raw = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    records = {}
    for line in raw.splitlines():
        idx, uuid, free, util = [part.strip() for part in line.split(",")]
        records[int(idx)] = {"index": int(idx), "uuid": uuid,
                             "free_mib": int(free), "utilization_pct": int(util)}
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
         "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout
    active = [line.strip() for line in apps.splitlines() if line.strip()]
    allowed = {proc.pid for proc in OWNED}
    for idx in (SIM_GPU, WAM_GPU):
        row = records.get(idx)
        if row is None or (require_idle and (row["free_mib"] < 30000 or row["utilization_pct"] > 10)):
            raise RuntimeError(f"GPU {idx} failed availability gate: {row}")
        conflicts = [line for line in active if line.split(",", 1)[0].strip() == row["uuid"]
                     and int(line.split(",", 2)[1].strip()) not in allowed]
        if conflicts:
            raise RuntimeError(f"GPU {idx} has unrelated compute process(es): {conflicts}")
    return {"simulator": records[SIM_GPU], "wam": records[WAM_GPU],
            "other_gpu_compute_processes": active}


def child_env(gpu: int, *, renderer: bool = False) -> dict:
    pythonpath = [str(RUN_CODE), str(BASE), str(LIBERO),
                  str(BASE / "benchmarks/libero"), str(UPSTREAM)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    env = dict(os.environ, ASYNC_BASE=str(BASE), OPENWAM_REPO=str(BASE),
               LIBERO_PATH=str(LIBERO), LIBERO_CONFIG_PATH=str(RUN_ROOT / "libero-config"),
               PYTHONPATH=os.pathsep.join(pythonpath), CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    extra_library_path = os.environ.get("WAM_EXTRA_LD_LIBRARY_PATH")
    if extra_library_path:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [extra_library_path, os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    egl_vendor_file = os.environ.get("EGL_VENDOR_LIBRARY_FILENAMES")
    if egl_vendor_file:
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = egl_vendor_file
    if renderer:
        # EGL selects the physical simulator GPU; Python/Torch remains CPU-only.
        env.update(CUDA_VISIBLE_DEVICES="", MUJOCO_EGL_DEVICE_ID=str(SIM_GPU),
                   MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    return env


def launch(command: list, env: dict, log_path: Path) -> subprocess.Popen:
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen([str(x) for x in command], cwd=BASE, env=env,
                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    OWNED.append(proc)
    return proc


def poll_stage(proc: subprocess.Popen, stage: str, log_path: Path,
               progress_path: Path | None = None) -> None:
    last_update = 0.0
    while proc.poll() is None:
        now = time.time()
        if now - last_update >= 20:
            status = {"stage": stage, "running": True, "pipeline_pid": os.getpid(),
                      "child_pid": proc.pid, "gpu": gpu_snapshot(), "unix_time": now}
            if progress_path and progress_path.exists():
                try:
                    snapshot = json.loads(progress_path.read_text())
                    status["completed_rows"] = snapshot.get(
                        "completed_rows", len(snapshot.get("rows", [])))
                except (OSError, json.JSONDecodeError):
                    pass
            atomic_json(RUN_ROOT / "status.json", status)
            last_update = now
        time.sleep(2)
    if proc.returncode:
        tail = log_path.read_text(errors="replace")[-14000:] if log_path.exists() else ""
        raise RuntimeError(f"{stage} exited {proc.returncode}:\n{tail}")


def run_driver(args: list[str], stage: str, progress: Path | None = None) -> None:
    log_path = RUN_ROOT / "logs" / f"{stage}.log"
    proc = launch([SIM_PY, EVAL_SCRIPT, *args], child_env(SIM_GPU, renderer=True), log_path)
    poll_stage(proc, stage, log_path, progress)


def summarize_development_candidate(path: Path, *, gain: float,
                                    threshold_adjustment_m: float,
                                    persistence_checks: int) -> dict:
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    index = {(int(r["task_id"]), int(r["trial"]), r["condition"], r["method"]): r
             for r in rows}
    reductions, paired_success = [], {"rescues": 0, "harms": 0, "pairs": 0}
    for condition in ("arm_speed_loss", "arm_offset", "arm_speed_offset"):
        cases = sorted({(int(r["task_id"]), int(r["trial"])) for r in rows
                        if r["condition"] == condition})
        for task, trial in cases:
            base = index.get((task, trial, condition, "fault_only"))
            hybrid = index.get((task, trial, condition, "hybrid"))
            if base is None or hybrid is None:
                continue
            paired_success["pairs"] += 1
            paired_success["rescues"] += int(not base["success"] and hybrid["success"])
            paired_success["harms"] += int(base["success"] and not hybrid["success"])
            a, b = base.get("fault_trajectory_rmse_m"), hybrid.get("fault_trajectory_rmse_m")
            if a is not None and b is not None:
                reductions.append(float(a) - float(b))
    clean = [r for r in rows if r["condition"] == "clean" and r["method"] == "hybrid"]
    clean_false = sum(int(r.get("local_corrections", 0) > 0) for r in clean)
    fault_hybrid = [float(r["fault_trajectory_rmse_m"]) for r in rows
                    if r["condition"].startswith("arm_") and r["method"] == "hybrid"
                    and r.get("fault_trajectory_rmse_m") is not None]
    fault_base = [float(r["fault_trajectory_rmse_m"]) for r in rows
                  if r["condition"].startswith("arm_") and r["method"] == "fault_only"
                  and r.get("fault_trajectory_rmse_m") is not None]
    return {"gain": float(gain), "threshold_adjustment_m": float(threshold_adjustment_m),
            "persistence_checks": int(persistence_checks), "coherence_cos": 0.2,
            "completed_rows": len(rows), "clean_episodes": len(clean),
            "clean_false_corrections": clean_false,
            "fault_window_pairs_with_metrics": len(reductions),
            "mean_paired_fault_rmse_reduction_m":
                sum(reductions) / len(reductions) if reductions else None,
            "median_paired_fault_rmse_reduction_m":
                float(__import__("statistics").median(reductions)) if reductions else None,
            "hybrid_fault_rmse_mean_m": sum(fault_hybrid) / len(fault_hybrid)
                if fault_hybrid else None,
            "fault_only_fault_rmse_mean_m": sum(fault_base) / len(fault_base)
                if fault_base else None,
            "paired_task_success": paired_success}


def choose_development_candidate(choices: list[dict]) -> dict:
    if not choices:
        raise RuntimeError("no complete development candidates")
    min_false = min(x["clean_false_corrections"] for x in choices)
    pool = [x for x in choices if x["clean_false_corrections"] == min_false]
    return max(pool, key=lambda x: (x["mean_paired_fault_rmse_reduction_m"]
                                   if x["mean_paired_fault_rmse_reduction_m"] is not None
                                   else float("-inf")))


def make_world_jobs() -> None:
    source = json.loads((RUN_ROOT / "source_cvpr_manifest.json").read_text())
    jobs, scenes = [], {}
    for case in source["cases"]:
        task, trial = int(case["task_id"]), int(case["trial"])
        if case["fault_kind"] == "target_and_distractor":
            condition = "target_shift"
            methods = ["stale_plan", "hybrid", "fixed_refresh"]
        elif case["fault_kind"] == "distractor_only":
            condition = "distractor_shift"
            methods = ["stale_plan", "hybrid", "fixed_refresh"]
        else:
            continue
        jobs.append({"task_id": task, "trial": trial, "condition": condition,
                     "methods": methods, "trial_slot": sum(
                         1 for j in jobs if j["task_id"] == task and j["condition"] == "target_shift")})
        scenes.setdefault(str(task), {})[str(trial)] = {
            "injection_step": int(case["fault_step"]),
            "target_shift_magnitude": float(case["fault_dx_m"]),
            "distractor_shift_magnitude": float(case["distractor_dx_m"]),
        }
    atomic_json(RUN_ROOT / "world_change_jobs.json", {"jobs": jobs})
    atomic_json(RUN_ROOT / "world_change_scene_map.json", scenes)


def make_drift_jobs() -> None:
    source = json.loads((RUN_ROOT / "source_cvpr_manifest.json").read_text())
    states = {str(task): [int(c["trial"]) for c in source["cases"]
                          if c["task_id"] == task and c["fault_kind"] == "target_and_distractor"]
              for task in TASKS}
    atomic_json(RUN_ROOT / "evaluation_state_map.json", states)
    jobs = []
    for task in TASKS:
        for slot, trial in enumerate(states[str(task)]):
            jobs.append({"task_id": task, "trial": trial, "trial_slot": slot,
                         "condition": "clean", "methods": ["wam", "hybrid"]})
            for condition in ("arm_speed_loss", "arm_offset", "arm_speed_offset"):
                jobs.append({"task_id": task, "trial": trial, "trial_slot": slot,
                             "condition": condition, "methods": ["fault_only", "hybrid"]})
    atomic_json(RUN_ROOT / "execution_drift_jobs.json", {"jobs": jobs})


def setup_wam_code() -> None:
    RUN_CODE.mkdir(parents=True, exist_ok=True)
    original = UPSTREAM / "memory_efficient_repair_wam_server.py"
    source = original.read_text()
    old = "PROMPTS=Path(os.environ.get('ASYNC_BASE','/home/pjh/research/openwam'))/'outputs/wam_long_horizon_v2/outcomes/results.json'"
    new = "PROMPTS=Path(os.environ['WAM_PROMPTS_JSON'])"
    if old not in source:
        raise RuntimeError("upstream prompt-cache line changed; refusing an unverified server patch")
    source = source.replace(old, new)
    (RUN_CODE / "memory_efficient_repair_wam_server.py").write_text(source)
    repair = (UPSTREAM / "repair_wam_server.py").read_text()
    needle = "    def predict(self, obs):\n        export_plan = bool(obs.get(\"include_action_plan\"))\n"
    replacement = ("    def predict(self, obs):\n"
                   "        import torch\n"
                   "        torch.manual_seed(20261013)\n"
                   "        if torch.cuda.is_available():\n"
                   "            torch.cuda.manual_seed_all(20261013)\n"
                   "        export_plan = bool(obs.get(\"include_action_plan\"))\n")
    if needle not in repair:
        raise RuntimeError("upstream instrumented predict method changed; refusing seed patch")
    (RUN_CODE / "repair_wam_server.py").write_text(repair.replace(needle, replacement))
    prompt_script = RUN_CODE / "make_prompts.py"
    prompt_script.write_text(
        "import json,sys\n"
        "from run_reframed_bench import _write_libero_config\n"
        "_write_libero_config()\n"
        "from libero.libero import benchmark\n"
        "suite=benchmark.get_benchmark_dict()['libero_10']()\n"
        "ids={0,1,4,6,7}\n"
        "json.dump([{'task_language':suite.get_task(i).language} for i in sorted(ids)],"
        "open(sys.argv[1],'w'))\n")
    env = child_env(WAM_GPU)
    log = RUN_ROOT / "logs" / "make_prompts.log"
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run([str(SIM_PY), str(prompt_script),
                                 str(RUN_ROOT / "prompts.json")], cwd=BASE, env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"LIBERO prompt extraction failed: {log.read_text(errors='replace')[-6000:]}")
    prompts = json.loads((RUN_ROOT / "prompts.json").read_text())
    if len(prompts) != len(TASKS) or any(not row.get("task_language") for row in prompts):
        raise RuntimeError(f"unexpected task prompt list: {prompts}")


def stop_owned(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    command = process_command(proc.pid)
    if "reframed_recovery_bench_v1_20261013" not in command and +            "memory_efficient_repair_wam_server.py" not in command:
        raise RuntimeError(f"refusing to stop unexpected child PID {proc.pid}: {command}")
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        command = process_command(proc.pid)
        if "memory_efficient_repair_wam_server.py" not in command:
            raise RuntimeError(f"child identity changed; refusing SIGKILL: {proc.pid} {command}")
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=15)


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "logs").mkdir(exist_ok=True)
    status_path = RUN_ROOT / "status.json"
    if status_path.exists():
        old = json.loads(status_path.read_text())
        pid = int(old.get("pipeline_pid", -1))
        if old.get("running") and pid > 0 and Path(f"/proc/{pid}").exists():
            raise RuntimeError(f"another reframed benchmark is running as PID {pid}")
        raise FileExistsError(f"refusing to rerun or overwrite benchmark directory: {RUN_ROOT}")
    port_probe = socket.socket()
    try:
        port_probe.bind(("127.0.0.1", PORT))
    finally:
        port_probe.close()
    gpu = gpu_snapshot(require_idle=True)
    atomic_json(RUN_ROOT / "preflight.json", {"gpu": gpu, "wam_port": PORT,
                                               "simulator_gpu": SIM_GPU, "wam_gpu": WAM_GPU})
    try:
        setup_wam_code()
        make_world_jobs()
        make_drift_jobs()
        server_env = child_env(WAM_GPU)
        server_env["WAM_PROMPTS_JSON"] = str(RUN_ROOT / "prompts.json")
        checkpoint = BASE / "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO"
        server_log = RUN_ROOT / "logs" / "wam_server.log"
        server = launch([SIM_PY, RUN_CODE / "memory_efficient_repair_wam_server.py",
                         "--ckpt-dir", checkpoint, "--host", "127.0.0.1", "--port", PORT,
                         "--denoise-steps", 1, "--compile-enabled", "false",
                         "optimization.dit_cache.enabled=false"], server_env, server_log)

        deadline = time.monotonic() + 5400
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"WAM startup failed: {server_log.read_text(errors='replace')[-12000:]}")
            if "WebSocket server started" in server_log.read_text(errors="replace"):
                break
            atomic_json(status_path, {"stage": "starting_wam", "running": True,
                                      "pipeline_pid": os.getpid(), "wam_pid": server.pid,
                                      "gpu": gpu_snapshot(), "unix_time": time.time()})
            time.sleep(5)
        else:
            raise TimeoutError("WAM did not become ready")
        atomic_json(RUN_ROOT / "services.json", {"pipeline_pid": os.getpid(),
                                                  "wam_pid": server.pid, "wam_port": PORT,
                                                  "gpu": gpu})
        common = ["--tasks", *map(str, TASKS), "--states", "20", "21",
                  "--calibration-states", *map(str, CALIBRATION_STATES),
                  "--max-steps", "700", "--sim-gpu", str(SIM_GPU),
                  "--wam-port", str(PORT), "--pace"]
        calibration = RUN_ROOT / "calibration" / "calibration.json"
        if not calibration.is_file():
            raise RuntimeError("missing separately frozen clean calibration")

        # Stage A tests persistence and threshold sensitivity. The online gate
        # uses only residual magnitude and direction; fault labels remain hidden.
        threshold_candidates = [
            ("development_gate_p2_q95", -0.005, 2),
            ("development_gate_p2_q95_plus5mm", 0.0, 2),
            ("development_gate_p2_q95_plus10mm", 0.005, 2),
            ("development_gate_p2_q95_plus15mm", 0.010, 2),
            ("development_gate_p3_q95", -0.005, 3),
        ]
        threshold_choices = []
        for name, adjustment, persistence_checks in threshold_candidates:
            args = ["--stage", "evaluate", "--output",
                    str(RUN_ROOT / "development" / name / "results.json"), *common,
                    "--calibration", str(calibration),
                    "--job-manifest", str(RUN_ROOT / "code" / "development_jobs.json"),
                    "--correction-gain", "0.65",
                    "--threshold-adjustment-m", str(adjustment),
                    "--persistence-checks", str(persistence_checks),
                    "--coherence-cos", "0.2"]
            run_driver(args, name, RUN_ROOT / "development" / name / "results.json")
            threshold_choices.append(summarize_development_candidate(
                RUN_ROOT / "development" / name / "results.json", gain=0.65,
                threshold_adjustment_m=adjustment,
                persistence_checks=persistence_checks) | {"name": name})
        selected_threshold = choose_development_candidate(threshold_choices)
        atomic_json(RUN_ROOT / "development" / "threshold_selection.json", {
            "selection_rule": "minimize clean false-repair episodes; among ties maximize mean paired fault-window RMSE reduction",
            "candidates": threshold_choices, "selected": selected_threshold})

        # Stage B holds the selected, development-only threshold fixed and
        # selects correction gain on the same disjoint development split.
        gain_choices = []
        for gain in (0.3, 0.5, 0.65, 0.8):
            name = f"development_gain_{gain:.2f}"
            args = ["--stage", "evaluate", "--output",
                    str(RUN_ROOT / "development" / name / "results.json"), *common,
                    "--calibration", str(calibration),
                    "--job-manifest", str(RUN_ROOT / "code" / "development_jobs.json"),
                    "--correction-gain", str(gain),
                    "--threshold-adjustment-m", str(selected_threshold["threshold_adjustment_m"]),
                    "--persistence-checks", str(selected_threshold["persistence_checks"]),
                    "--coherence-cos", "0.2"]
            run_driver(args, name, RUN_ROOT / "development" / name / "results.json")
            gain_choices.append(summarize_development_candidate(
                RUN_ROOT / "development" / name / "results.json", gain=gain,
                threshold_adjustment_m=selected_threshold["threshold_adjustment_m"],
                persistence_checks=selected_threshold["persistence_checks"]))
        selected = choose_development_candidate(gain_choices)
        selected["selection_rule"] = (
            "minimize clean false-repair episodes; among ties maximize mean paired "
            "fault-window RMSE reduction")
        selected["selected_threshold_adjustment_m"] = selected_threshold["threshold_adjustment_m"]
        selected["selected_persistence_checks"] = selected_threshold["persistence_checks"]
        selected["threshold_selection"] = selected_threshold
        selected["development_positive"] = (
            selected.get("mean_paired_fault_rmse_reduction_m") is not None and
            selected["mean_paired_fault_rmse_reduction_m"] > 0 and
            selected["clean_false_corrections"] == 0)
        atomic_json(RUN_ROOT / "development" / "gain_selection.json",
                    {"candidates": gain_choices, "selected": selected})
        if not selected["development_positive"]:
            raise RuntimeError("development gate did not achieve zero clean false repairs plus positive paired residual improvement; held-out evaluation remains unopened")

        eval_args = ["--stage", "evaluate", "--output", str(RUN_ROOT / "evaluation" / "execution_drift" / "results.json"),
                     *common, "--calibration", str(calibration),
                     "--state-map", str(RUN_ROOT / "evaluation_state_map.json"),
                     "--job-manifest", str(RUN_ROOT / "execution_drift_jobs.json"),
                     "--correction-gain", str(selected["gain"]),
                     "--threshold-adjustment-m", str(selected["selected_threshold_adjustment_m"]),
                     "--persistence-checks", str(selected["selected_persistence_checks"]),
                     "--coherence-cos", "0.2"]
        run_driver(eval_args, "execution_drift", RUN_ROOT / "evaluation" / "execution_drift" / "results.json")
        world_args = ["--stage", "evaluate",
                      "--output", str(RUN_ROOT / "evaluation" / "world_change" / "results.json"),
                      *common, "--calibration", str(calibration),
                      "--scene-map", str(RUN_ROOT / "world_change_scene_map.json"),
                      "--job-manifest", str(RUN_ROOT / "world_change_jobs.json"),
                      "--correction-gain", str(selected["gain"]),
                      "--threshold-adjustment-m", str(selected["selected_threshold_adjustment_m"]),
                      "--persistence-checks", str(selected["selected_persistence_checks"]),
                      "--coherence-cos", "0.2"]
        run_driver(world_args, "world_change", RUN_ROOT / "evaluation" / "world_change" / "results.json")
        final = {"stage": "complete", "running": False, "pipeline_pid": os.getpid(),
                 "gpu": gpu, "selected_gain": selected["gain"],
                 "selected_threshold_adjustment_m": selected["selected_threshold_adjustment_m"],
                 "selected_persistence_checks": selected["selected_persistence_checks"],
                 "development_positive": selected["development_positive"],
                 "drift_summary": str(RUN_ROOT / "evaluation/execution_drift/summary.json"),
                 "world_summary": str(RUN_ROOT / "evaluation/world_change/summary.json"),
                 "unix_time": time.time()}
        atomic_json(status_path, final)
        return 0
    except Exception as exc:
        atomic_json(status_path, {"stage": "failed", "running": False,
                                  "pipeline_pid": os.getpid(), "error": repr(exc),
                                  "unix_time": time.time()})
        raise
    finally:
        for child in reversed(OWNED):
            stop_owned(child)


if __name__ == "__main__":
    sys.exit(main())
