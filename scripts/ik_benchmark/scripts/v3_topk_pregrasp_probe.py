#!/usr/bin/env python3
"""Validate the lowest-joint-cost pregrasp candidates for each V3 wall task."""

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
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--candidate-planning-s", type=float, default=2.0)
    parser.add_argument("--rounds", type=int, nargs="+")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.pregrasp_summary.read_text())
    requested = set(args.rounds or [task["round"] for task in source["tasks"] if task["success"]])
    results = []
    for task in source["tasks"]:
        if not task["success"] or task["round"] not in requested:
            continue
        round_number = task["round"]
        started = time.monotonic()
        scene = args.scene_root / f"round_{round_number:02}_scene.json"
        candidates = json.loads((args.scene_root /
            f"round_{round_number:02}_side_candidates.json").read_text())
        attempts = []
        for rank, (home, height, left, right, cost) in enumerate(candidate_stream(candidates), 1):
            if rank > args.top_k:
                break
            selected = one_candidate(candidates, home, height, left, right, cost)
            selected_path = args.output_dir / f"round_{round_number:02}_rank_{rank:02}_candidate.json"
            plan_path = args.output_dir / f"round_{round_number:02}_rank_{rank:02}_plan.json"
            selected_path.write_text(json.dumps(selected, separators=(",", ":")) + "\n")
            run(["python3", str(args.runner), "--scene", str(scene), "--srdf", str(args.srdf),
                 "--backend", str(args.backend), "--dual-pregrasp-candidates", str(selected_path),
                 "--time-s", str(args.candidate_planning_s), "--output", str(plan_path)])
            plan = json.loads(plan_path.read_text())
            attempts.append({
                "rank": rank, "home": home, "height_m": height, "joint_cost": cost,
                "success": bool(plan["success"]),
                "shortcut_valid": bool(plan.get("shortcut_valid", False)),
                "rrt_exact": bool(plan.get("rrt_exact", False)),
                "failure_stage": plan.get("failure_stage"),
                "collision_checks": plan.get("collision_checks", 0),
                "planning_ms": plan.get("wall_ms", 0.0),
                "plan": str(plan_path),
            })
        failure_counts = {}
        for attempt in attempts:
            if attempt["success"]:
                continue
            key = attempt["failure_stage"] or "unknown"
            failure_counts[key] = failure_counts.get(key, 0) + 1
        result = {
            "round": round_number, "left": task["left"], "right": task["right"],
            "tested": len(attempts), "success_count": sum(item["success"] for item in attempts),
            "shortcut_success_count": sum(item["success"] and item["shortcut_valid"] for item in attempts),
            "rrt_success_count": sum(item["success"] and not item["shortcut_valid"] for item in attempts),
            "failure_counts": failure_counts,
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
            "attempts": attempts,
        }
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "attempts"},
                         ensure_ascii=False), flush=True)
    total_tested = sum(task["tested"] for task in results)
    total_success = sum(task["success_count"] for task in results)
    summary = {
        "kind": "v3_top_joint_cost_pregrasp_validation",
        "top_k": args.top_k,
        "tasks": results,
        "total_tested": total_tested,
        "total_success": total_success,
        "success_rate": total_success / total_tested if total_tested else 0.0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
