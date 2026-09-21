#!/usr/bin/env python3
"""Try later pregrasp IK candidates for rounds whose 35 cm extraction failed."""

import argparse
import json
import time
from pathlib import Path

from v3_pregrasp_15_task_probe import candidate_stream, one_candidate, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pregrasp-summary", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--srdf", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", default=[4, 8, 11])
    parser.add_argument("--max-attempts", type=int, default=64)
    parser.add_argument("--task-budget-s", type=float, default=120.0)
    parser.add_argument("--candidate-planning-s", type=float, default=2.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog = {task["round"]: task for task in json.loads(
        args.pregrasp_summary.read_text())["tasks"]}
    results = []
    for round_number in args.rounds:
        task = catalog[round_number]
        started = time.monotonic()
        scene = args.scene_root / f"round_{round_number:02}_scene.json"
        candidates = json.loads((args.scene_root /
            f"round_{round_number:02}_side_candidates.json").read_text())
        success = None
        failure_counts = {}
        attempts = 0
        for home, height, left, right, cost in candidate_stream(candidates):
            if attempts >= args.max_attempts or time.monotonic() - started >= args.task_budget_s:
                break
            attempts += 1
            selected = one_candidate(candidates, home, height, left, right, cost)
            selected_path = args.output_dir / f"round_{round_number:02}_attempt_{attempts:03}_candidate.json"
            plan_path = args.output_dir / f"round_{round_number:02}_attempt_{attempts:03}_plan.json"
            approach_path = args.output_dir / f"round_{round_number:02}_attempt_{attempts:03}_approach.json"
            extract_path = args.output_dir / f"round_{round_number:02}_attempt_{attempts:03}_extract.json"
            selected_path.write_text(json.dumps(selected, separators=(",", ":")) + "\n")
            run(["python3", str(args.runner), "--scene", str(scene), "--srdf", str(args.srdf),
                 "--backend", str(args.backend), "--dual-pregrasp-candidates", str(selected_path),
                 "--time-s", str(args.candidate_planning_s), "--output", str(plan_path)])
            plan = json.loads(plan_path.read_text())
            if not plan["success"]:
                key = "pregrasp:" + plan.get("failure_stage", "unknown")
                failure_counts[key] = failure_counts.get(key, 0) + 1
                continue
            run(["python3", str(args.runner), "--scene", str(scene), "--srdf", str(args.srdf),
                 "--backend", str(args.backend), "--dual-cartesian-candidates", str(selected_path),
                 "--dual-cartesian-pregrasp-plan", str(plan_path), "--output", str(approach_path)])
            approach = json.loads(approach_path.read_text())
            if not approach["success"]:
                key = "approach:" + approach.get("failure_stage", "unknown")
                failure_counts[key] = failure_counts.get(key, 0) + 1
                continue
            run(["python3", str(args.runner), "--scene", str(scene), "--srdf", str(args.srdf),
                 "--backend", str(args.backend), "--dual-cartesian-candidates", str(selected_path),
                 "--dual-cartesian-pregrasp-plan", str(plan_path),
                 "--dual-cartesian-approach", str(approach_path), "--stop-after-extraction",
                 "--output", str(extract_path)])
            extraction = json.loads(extract_path.read_text())
            if extraction["success"]:
                success = {"attempt": attempts, "home": home, "height_m": height,
                           "cost": cost, "candidate": str(selected_path), "plan": str(plan_path),
                           "approach": str(approach_path), "extraction": str(extract_path)}
                break
            key = "extract:" + extraction.get("failure_reason",
                                                extraction.get("failure_stage", "unknown"))
            failure_counts[key] = failure_counts.get(key, 0) + 1
        result = {"round": round_number, "left": task["left"], "right": task["right"],
                  "success": success is not None, "attempts": attempts,
                  "elapsed_ms": (time.monotonic() - started) * 1000.0,
                  "failure_counts": failure_counts, "selected": success}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {"kind": "v3_failed_35cm_extract_later_candidate_probe", "tasks": results,
               "success_count": sum(task["success"] for task in results)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
