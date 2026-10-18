"""Create the frozen ten-family, four-state paired job manifest."""
from __future__ import annotations
import json
from pathlib import Path

TASKS = [0, 33, 78, 106, 137, 145, 158, 193, 237, 276]
TASK_NAMES = {
    0: "turn on the stove and put the moka pot on it",
    33: "put the black bowl in the bottom drawer of the cabinet and close it",
    78: "put the yellow and white mug in the microwave and close it",
    106: "put both moka pots on the stove",
    137: "put both the alphabet soup and the cream cheese box in the basket",
    145: "put both the alphabet soup and the tomato sauce in the basket",
    158: "put both the cream cheese box and the butter in the basket",
    193: "put the white mug on the left plate and the yellow and white mug on the right plate",
    237: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    276: "pick up the book and place it in the back compartment of the caddy",
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
        for trial in (2, 3, 4, 5):
            for condition, methods in CONDITIONS.items():
                jobs.append({"task_id": task_id, "trial": trial,
                             "condition": condition, "methods": methods})
    manifest = {
        "name": "libero10_paper_scale_v1",
        "tasks": TASKS, "calibration_states": [0, 1],
        "evaluation_states": [2, 3, 4, 5],
        "difficulty": "hard", "sim_gpu": 1, "wam_gpu": 2,
        "conditions": list(CONDITIONS), "jobs": jobs,
        "expected_rows": sum(len(j["methods"]) for j in jobs),
        "seed_rule": "runner deterministic task/trial/condition seed",
        "limits": ["custom OpenWAM/LIBERO simulator", "simulator target-pose oracle",
                    "Cartesian downstream gain/bias plant", "not official LIBERO or real robot"],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"tasks": len(TASKS), "scenes": 40,
                      "expected_rows": manifest["expected_rows"]}, indent=2))

if __name__ == "__main__":
    main()
