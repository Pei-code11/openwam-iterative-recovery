"""Create the frozen ten-family, four-state paired job manifest."""
from __future__ import annotations
import json
from pathlib import Path

TASKS = [287, 677, 1099, 1932, 2092]
TASK_NAMES = {
    287: "pick up the book and place it in the back compartment of the caddy tb 17",
    677: "put the yellow and white mug in the microwave and close it view 0 0 100 0 0 initstate 460",
    1099: "put the yellow and white mug in the microwave and close it view 358 15 100 0 0 initstate 0",
    1932: "put the yellow and white mug in the microwave and close it view 0 0 100 0 0 initstate 0 noise 50",
    2092: "put both the alphabet soup and the tomato sauce in the basket level5 sample4",
}
CONDITIONS = {
    "clean": ["wam", "hybrid"],
    "arm_speed_loss": ["fault_only", "hybrid"],
    "arm_offset": ["fault_only", "hybrid"],
    "arm_speed_offset": ["fault_only", "hybrid"],
    "target_shift": ["stale_plan", "local_only", "hybrid", "fixed_refresh"],
    "distractor_shift": ["stale_plan", "hybrid", "fixed_refresh"],
}

def main() -> None:
    root = Path(__file__).parent
    prompts = {str(k): {"task_language": v} for k, v in TASK_NAMES.items()}
    (root / "prompts.json").write_text(json.dumps(prompts, indent=2) + "\n")
    jobs = []
    for task_id in TASKS:
        for trial in range(2, 12):
            for condition, methods in CONDITIONS.items():
                jobs.append({"task_id": task_id, "trial": trial,
                             "condition": condition, "methods": methods})
    manifest = {
        "name": "libero10_paper_scale_v1",
        "tasks": TASKS, "calibration_states": [0, 1],
        "evaluation_states": list(range(2, 12)),
        "difficulty": "hard", "sim_gpu": 1, "wam_gpu": 2,
        "conditions": list(CONDITIONS), "jobs": jobs,
        "expected_rows": sum(len(j["methods"]) for j in jobs),
        "seed_rule": "runner deterministic task/trial/condition seed",
        "limits": ["custom OpenWAM/LIBERO simulator", "simulator target-pose oracle",
                    "Cartesian downstream gain/bias plant", "not official LIBERO or real robot"],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"tasks": len(TASKS), "scenes": 50,
                      "expected_rows": manifest["expected_rows"]}, indent=2))

if __name__ == "__main__":
    main()
