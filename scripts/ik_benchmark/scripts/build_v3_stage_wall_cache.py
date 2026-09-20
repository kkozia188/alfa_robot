#!/usr/bin/env python3
"""Build deterministic dual-arm stage trajectories for the V3 5x5 wall fixture."""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path


DUAL_ROUNDS = [pair for row in range(4, -1, -1) for pair in (
    (row * 5 + 4, row * 5), (row * 5 + 3, row * 5 + 1))]


def run(command):
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)


def contiguous(parts):
    frames = []
    for part in parts:
        for frame in part:
            if frames and frame["joints"] == frames[-1]["joints"]:
                continue
            frames.append(frame)
    return frames


def densify(frames):
    if not frames:
        return []
    output = [frames[0]]
    for target in frames[1:]:
        source = output[-1]
        first = source["joints"]
        last = target["joints"]
        steps = max(
            1,
            math.ceil(max(abs(last[index] - first[index]) for index in range(14)) /
                      math.radians(0.5)),
            math.ceil(abs(last[14] - first[14]) / 0.005),
        )
        for step in range(1, steps + 1):
            fraction = step / steps
            item = dict(target)
            item["joints"] = [a * (1.0 - fraction) + b * fraction
                              for a, b in zip(first, last)]
            output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--srdf", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--home-transition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.scene.read_text())
    transition = json.loads(args.home_transition.read_text())
    if not transition.get("success") or not transition.get("frames"):
        raise ValueError("home transition must be post-validated")
    request_catalog = json.loads(args.requests.read_text())
    by_pair = {(item["left"], item["right"]): item for item in request_catalog["rounds"]
               if item["right"] is not None}
    working = args.output.parent / "cache_build"
    working.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.scene.parent / "robot.urdf", working / "robot.urdf")
    shutil.copy2(args.srdf, working / "robot.srdf")
    removed = set()
    entries = []
    current_initial = "home"
    for round_index, (left, right) in enumerate(DUAL_ROUNDS):
        record = by_pair[(left, right)]
        scene = dict(source)
        scene["wall_boxes"] = [box for box in source["wall_boxes"]
                               if box["box_id"] not in removed]
        scene_file = working / f"round_{round_index:02}_scene.json"
        scene_file.write_text(json.dumps(scene, indent=2) + "\n")
        candidates_file = working / f"round_{round_index:02}_candidates.json"
        run(["python3", str(args.runner), "--scene", str(scene_file), "--srdf", str(args.srdf),
             "--backend", str(args.backend), "--dual-face-ik-request", record["request"],
             "--output", str(candidates_file)])
        candidates = json.loads(candidates_file.read_text())
        candidate_initial = "second_home"
        candidates["initials"] = [item for item in candidates["initials"]
                                  if item["home"] == candidate_initial]
        candidates["success"] = any(item["shared_heights"] for item in candidates["initials"])
        if not candidates["success"]:
            entries.append({"round": record["round"], "left": left, "right": right,
                            "success": False, "failure": "no_shared_height_ik"})
            continue
        options = sorted(candidates["initials"][0]["shared_heights"], key=lambda item: item["cost"])
        success = None
        last_failure = "candidate_exhausted"
        for option_index, option in enumerate(options):
            attempt = dict(candidates)
            attempt["initials"] = [dict(candidates["initials"][0])]
            attempt["initials"][0]["shared_heights"] = [option]
            candidates_file.write_text(json.dumps(attempt, indent=2) + "\n")
            prefix = f"round_{round_index:02}_candidate_{option_index:02}"
            pregrasp = working / f"{prefix}_pregrasp.json"
            approach = working / f"{prefix}_approach.json"
            retreat = working / f"{prefix}_retreat.json"
            run(["python3", str(args.runner), "--scene", str(scene_file), "--srdf", str(args.srdf),
                 "--backend", str(args.backend), "--dual-pregrasp-candidates", str(candidates_file),
                 "--time-s", "2", "--output", str(pregrasp)])
            pre = json.loads(pregrasp.read_text())
            if not pre["success"]:
                last_failure = pre.get("failure_stage", "pregrasp")
                continue
            run(["python3", str(args.runner), "--scene", str(scene_file), "--srdf", str(args.srdf),
                 "--backend", str(args.backend), "--dual-cartesian-candidates", str(candidates_file),
                 "--dual-cartesian-pregrasp-plan", str(pregrasp), "--output", str(approach)])
            app = json.loads(approach.read_text())
            if not app["success"]:
                last_failure = app.get("failure_stage", "approach")
                continue
            run(["python3", str(args.runner), "--scene", str(scene_file), "--srdf", str(args.srdf),
                 "--backend", str(args.backend), "--dual-cartesian-candidates", str(candidates_file),
                 "--dual-cartesian-pregrasp-plan", str(pregrasp),
                 "--dual-cartesian-approach", str(approach), "--output", str(retreat)])
            loaded = json.loads(retreat.read_text())
            placement = loaded.get("named_placement_shortcut", {})
            if not (loaded["success"] and loaded["loaded_return"]["success"] and
                    loaded["placement_entry_bridge"]["success"] and placement.get("success")):
                last_failure = "loaded_transfer_or_placement"
                continue
            success = (attempt, pre, app, loaded, placement, option_index)
            break
        if success is None:
            entries.append({"round": record["round"], "left": left, "right": right,
                            "success": False, "failure": last_failure,
                            "attempted_candidates": len(options)})
            continue
        attempt, pre, app, loaded, placement, option_index = success
        pregrasp_frames = pre["frames"]
        if current_initial == "home":
            pregrasp_frames = contiguous([transition["frames"], pregrasp_frames])
        place = contiguous([
            loaded["frames"], loaded["loaded_return"]["frames"],
            loaded["placement_entry_bridge"]["frames"], placement["frames"],
        ])
        home = [{"stage": "home", "joints": frame["joints"]}
                for frame in reversed(placement["frames"])]
        entry = {
            "round": record["round"], "left": left, "right": right, "success": True,
            "initial_pose": current_initial, "height_m": pre["height_m"],
            "selected_candidate": option_index, "attempted_candidates": option_index + 1,
            "pregrasp": densify(pregrasp_frames), "approach": densify(app["frames"]),
            "place": densify(place), "home": densify(home),
            "planning_ms": pre["wall_ms"] + candidates["wall_ms"] +
                           loaded["loaded_return"]["wall_ms"],
        }
        entries.append(entry)
        removed.update((left, right))
        current_initial = "second_home"
        print(json.dumps({key: entry[key] for key in
                          ("round", "left", "right", "success", "initial_pose", "planning_ms")}),
              flush=True)
    cache = {
        "schema_version": 1,
        "kind": "v3_fixed_wall_motion_stage_cache",
        "model": "current_alfa_v3_suction_description",
        "policy": "dual side suction; placed boxes disappear after release; failed rounds do not move",
        "joint_names": source["joint_names"],
        "entries": entries,
    }
    args.output.write_text(json.dumps(cache, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
