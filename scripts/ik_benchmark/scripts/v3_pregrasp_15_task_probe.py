#!/usr/bin/env python3
"""Try full multi-branch pregrasp candidates for all fifteen V3 wall rounds."""

import argparse
import heapq
import json
import subprocess
import shutil
import time
from pathlib import Path


def run(command):
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)


def candidate_stream(data):
    contexts = []
    for initial in data["initials"]:
        for shared in initial["shared_heights"]:
            left = shared.get("left_options", [])
            right = shared.get("right_options", [])
            if left and (right or data["right_pose_in_base_link"] is None):
                contexts.append((initial["home"], shared["height_m"], left, right))
    heap = []
    visited = [set() for _ in contexts]
    for context_index, (_, _, left, right) in enumerate(contexts):
        cost = left[0]["cost"] + (right[0]["cost"] if right else 0.0)
        heapq.heappush(heap, (cost, context_index, 0, 0))
        visited[context_index].add((0, 0))
    while heap:
        cost, context_index, left_index, right_index = heapq.heappop(heap)
        home, height, left, right = contexts[context_index]
        yield home, height, left[left_index], right[right_index] if right else None, cost
        neighbors = [(left_index + 1, right_index)]
        if right:
            neighbors.append((left_index, right_index + 1))
        for next_left, next_right in neighbors:
            if next_left >= len(left) or (right and next_right >= len(right)):
                continue
            if (next_left, next_right) in visited[context_index]:
                continue
            visited[context_index].add((next_left, next_right))
            next_cost = left[next_left]["cost"] + (right[next_right]["cost"] if right else 0.0)
            heapq.heappush(heap, (next_cost, context_index, next_left, next_right))


def one_candidate(data, home, height, left, right, cost):
    pair = {"home": home, "height_m": height, "left": left, "cost": cost}
    if right is not None:
        pair["right"] = right
    variant = {"home": home, "shared_heights": [pair]}
    output = dict(data)
    output["initials"] = [variant]
    output["success"] = True
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--srdf", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--task-budget-s", type=float, default=60.0)
    parser.add_argument("--candidate-planning-s", type=float, default=2.0)
    parser.add_argument("--only-round", type=int)
    args = parser.parse_args()
    catalog = json.loads(args.requests.read_text())
    source_scene = json.loads(args.scene.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.scene.parent / "robot.urdf", args.output_dir / "robot.urdf")
    shutil.copy2(args.srdf, args.output_dir / "robot.srdf")
    results = []
    removed = set()
    for task in catalog["rounds"]:
        if args.only_round is not None and task["round"] != args.only_round:
            continue
        started = time.monotonic()
        task_scene = dict(source_scene)
        task_scene["wall_boxes"] = [box for box in source_scene["wall_boxes"]
                                      if box["box_id"] not in removed]
        scene_path = args.output_dir / f"round_{task['round']:02}_scene.json"
        scene_path.write_text(json.dumps(task_scene, indent=2) + "\n")
        request = json.loads(Path(task["request"]).read_text())
        mode = "side"
        request["grasp_mode"] = mode
        request_path = args.output_dir / f"round_{task['round']:02}_{mode}_request.json"
        request_path.write_text(json.dumps(request, indent=2) + "\n")
        candidates_path = args.output_dir / f"round_{task['round']:02}_{mode}_candidates.json"
        run(["python3", str(args.runner), "--scene", str(scene_path), "--srdf", str(args.srdf),
             "--backend", str(args.backend), "--dual-face-ik-request", str(request_path),
             "--output", str(candidates_path)])
        candidates = json.loads(candidates_path.read_text())
        if not candidates["success"]:
            mode = "top"
            request["grasp_mode"] = mode
            request_path = args.output_dir / f"round_{task['round']:02}_{mode}_request.json"
            request_path.write_text(json.dumps(request, indent=2) + "\n")
            candidates_path = args.output_dir / f"round_{task['round']:02}_{mode}_candidates.json"
            run(["python3", str(args.runner), "--scene", str(scene_path), "--srdf", str(args.srdf),
                 "--backend", str(args.backend), "--dual-face-ik-request", str(request_path),
                 "--output", str(candidates_path)])
            candidates = json.loads(candidates_path.read_text())
        success = None
        attempts = 0
        failures = {}
        if candidates["success"]:
            for home, height, left, right, cost in candidate_stream(candidates):
                if attempts >= args.max_attempts or time.monotonic() - started >= args.task_budget_s:
                    break
                attempts += 1
                selected = one_candidate(candidates, home, height, left, right, cost)
                selected_path = args.output_dir / f"round_{task['round']:02}_selected.json"
                selected_path.write_text(json.dumps(selected, separators=(",", ":")) + "\n")
                plan_path = args.output_dir / f"round_{task['round']:02}_plan.json"
                run(["python3", str(args.runner), "--scene", str(scene_path), "--srdf", str(args.srdf),
                     "--backend", str(args.backend), "--dual-pregrasp-candidates", str(selected_path),
                     "--time-s", str(args.candidate_planning_s), "--output", str(plan_path)])
                plan = json.loads(plan_path.read_text())
                if plan["success"]:
                    success = {"home": home, "height_m": height, "cost": cost,
                               "plan": str(plan_path), "frames": len(plan["frames"])}
                    break
                stage = plan.get("failure_stage", "unknown")
                failures[stage] = failures.get(stage, 0) + 1
        result = {"round": task["round"], "left": task["left"], "right": task["right"],
                  "grasp_mode": mode, "success": success is not None, "attempts": attempts,
                  "elapsed_ms": (time.monotonic() - started) * 1000.0,
                  "candidate_failure_counts": failures, "selected": success}
        results.append(result)
        if result["success"]:
            removed.add(task["left"])
            if task["right"] is not None:
                removed.add(task["right"])
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {"kind": "v3_15_task_multi_branch_pregrasp_probe", "tasks": results,
               "success_count": sum(item["success"] for item in results),
               "scope": "pregrasp only; no contact, attachment, extraction, placement or execution"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
