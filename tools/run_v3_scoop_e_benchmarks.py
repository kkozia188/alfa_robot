#!/usr/bin/env python3
import json, os, shlex, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SUITE = json.loads((ROOT / "tools/v3_scoop_golden_20260921/E_BENCHMARK_SUITE.json").read_text())
OUTPUT = Path(os.environ.get("OUTPUT_DIR", "/home/astesia/v3-scoop-golden-generalization-E/benchmarks"))
SCRIPT = ROOT / "ros2_ws/src/alfa_robot_moveit_config/scripts/v3_scoop_5x5_grasp_sequence_rerun.py"
OUTPUT.mkdir(parents=True, exist_ok=True)
results = []
for scenario in SUITE["scenarios"]:
    directory = OUTPUT / scenario["name"]
    directory.mkdir(parents=True, exist_ok=True)
    active = scenario["args"][scenario["args"].index("--active-box-ids") + 1]
    limit = len(active.split(","))
    recording = directory / f"{scenario['name']}.rrd"
    command = ["/usr/bin/python3", str(SCRIPT), "--save", str(recording),
               "--ompl-seed", str(scenario["seed"]), "--limit-boxes", str(limit),
               "--no-spawn", "--no-resume", "--planning-timeout", "15", "--rrt-retries", "2",
               *scenario["args"]]
    (directory / "command.txt").write_text(shlex.join(command) + "\n")
    with (directory / "run.log").open("w") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    results.append({"name": scenario["name"], "level": scenario["level"],
                    "seed": scenario["seed"], "returncode": completed.returncode})
(OUTPUT / "status.json").write_text(json.dumps(results, indent=2) + "\n")
raise SystemExit(0 if all(item["returncode"] == 0 for item in results) else 1)
