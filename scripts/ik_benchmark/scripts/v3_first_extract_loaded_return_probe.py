#!/usr/bin/env python3
"""Return the first 35 cm extraction-success candidate to its named initial pose."""

import argparse
import json
import time
from pathlib import Path

from v3_pregrasp_15_task_probe import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pregrasp-summary", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--alternative-root", type=Path, required=True)
    parser.add_argument("--srdf", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.pregrasp_summary.read_text())
    combined_alternatives = {}
    if (args.alternative_root.joinpath("summary.json").exists()):
        combined_alternatives = {item["round"]: item for item in json.loads(
            args.alternative_root.joinpath("summary.json").read_text())["tasks"]}
    results = []
    for task in source["tasks"]:
        if not task["success"]:
            continue
        round_number = task["round"]
        started = time.monotonic()
        scene = args.scene_root / f"round_{round_number:02}_scene.json"
        alternative_summary = args.alternative_root / f"round_{round_number:02}" / "summary.json"
        if round_number in combined_alternatives or alternative_summary.exists():
            selected = (combined_alternatives.get(round_number) or
                        json.loads(alternative_summary.read_text())["tasks"][0])["selected"]
            candidate_rank = selected["attempt"]
            candidates = Path(selected["candidate"])
            plan = Path(selected["plan"])
            approach = Path(selected["approach"])
        else:
            candidate_rank = 1
            candidates = args.scene_root / f"round_{round_number:02}_selected.json"
            plan = args.scene_root / f"round_{round_number:02}_plan.json"
            approach = args.output_dir / f"round_{round_number:02}_approach.json"
            run(["python3", str(args.runner), "--scene", str(scene), "--srdf", str(args.srdf),
                 "--backend", str(args.backend), "--dual-cartesian-candidates", str(candidates),
                 "--dual-cartesian-pregrasp-plan", str(plan), "--output", str(approach)])
        output = args.output_dir / f"round_{round_number:02}_loaded_return.json"
        run(["python3", str(args.runner), "--scene", str(scene), "--srdf", str(args.srdf),
             "--backend", str(args.backend), "--dual-cartesian-candidates", str(candidates),
             "--dual-cartesian-pregrasp-plan", str(plan),
             "--dual-cartesian-approach", str(approach), "--loaded-joint-return",
             "--output", str(output)])
        result_data = json.loads(output.read_text())
        loaded = result_data.get("loaded_return", {})
        result = {
            "round": round_number, "left": task["left"], "right": task["right"],
            "candidate_rank": candidate_rank,
            "extraction_success": bool(result_data.get("success", False)),
            "loaded_return_success": bool(loaded.get("success", False)),
            "shortcut_valid": bool(loaded.get("shortcut_valid", False)),
            "rrt_exact": bool(loaded.get("rrt_exact", False)),
            "failure_stage": loaded.get("failure_stage") or result_data.get("failure_stage"),
            "failure_reason": loaded.get("failure_reason") or result_data.get("failure_reason"),
            "planning_ms": loaded.get("wall_ms", 0.0),
            "collision_checks": loaded.get("collision_checks", 0),
            "frames": len(loaded.get("frames", [])),
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
            "result": str(output),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {
        "kind": "v3_first_extract_success_loaded_joint_return",
        "tasks": results,
        "tested": len(results),
        "success_count": sum(task["loaded_return_success"] for task in results),
        "shortcut_count": sum(task["loaded_return_success"] and task["shortcut_valid"]
                              for task in results),
        "rrt_count": sum(task["loaded_return_success"] and not task["shortcut_valid"]
                         for task in results),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
