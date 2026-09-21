#!/usr/bin/env python3
"""Build the verified eleven-round V3 Motion stage cache and continuous replay."""

import argparse
import json
import math
from pathlib import Path


ROTARY_STEP = math.radians(0.5)
UPDOWN_STEP = 0.005


def joints(frame):
    return list(frame["joints"])


def same(first, second, tolerance=1e-7):
    return len(first) == len(second) and max(abs(a - b) for a, b in zip(first, second)) <= tolerance


def join(*segments):
    output = []
    for segment in segments:
        if not segment:
            continue
        if output:
            if not same(joints(output[-1]), joints(segment[0])):
                raise ValueError("trajectory boundary mismatch")
            output.extend(segment[1:])
        else:
            output.extend(segment)
    return output


def resample(frames):
    if not frames:
        return []
    output = [dict(frames[0])]
    for target in frames[1:]:
        first = joints(output[-1])
        last = joints(target)
        required = abs(last[14] - first[14]) / UPDOWN_STEP
        required = max(required, max(abs(last[index] - first[index])
                                     for index in range(14)) / ROTARY_STEP)
        count = max(1, math.ceil(required))
        for step in range(1, count + 1):
            item = dict(target)
            fraction = step / count
            item["joints"] = [a * (1.0 - fraction) + b * fraction
                              for a, b in zip(first, last)]
            output.append(item)
    return output


def tagged(frames, stage, round_number, box_ids, attached, visible):
    return [{"stage": stage, "round": round_number, "joints": joints(frame),
             "box_ids": box_ids, "attached": attached, "visible": visible}
            for frame in frames]


def cache_frames(frames):
    return [{"joints": joints(frame)} for frame in frames]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pregrasp-summary", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--alternative-root", type=Path, required=True)
    parser.add_argument("--loaded-root", type=Path, required=True)
    parser.add_argument("--transition", type=Path, required=True)
    parser.add_argument("--first-placement", type=Path, required=True)
    parser.add_argument("--second-placement", type=Path, required=True)
    parser.add_argument("--joint-names", type=Path, required=True)
    parser.add_argument("--cache-output", type=Path, required=True)
    parser.add_argument("--sequence-output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.pregrasp_summary.read_text())
    alternatives = {item["round"]: item for item in json.loads(
        args.alternative_root.joinpath("summary.json").read_text())["tasks"]}
    loaded_summary = {item["round"]: item for item in json.loads(
        args.loaded_root.joinpath("summary.json").read_text())["tasks"]}
    transition = json.loads(args.transition.read_text())
    first_placement = json.loads(args.first_placement.read_text())
    second_placement = json.loads(args.second_placement.read_text())
    joint_names = json.loads(args.joint_names.read_text())["joint_names"]
    if not transition.get("success") or not first_placement.get("success") or not second_placement.get("success"):
        raise ValueError("transition or placement trajectory is not verified")
    transition_forward = resample(transition["frames"])
    transition_reverse = [dict(frame) for frame in reversed(transition_forward)]
    placements = {"home": resample(first_placement["frames"]),
                  "second_home": resample(second_placement["frames"])}
    unloading_names = {"home": "unloading", "second_home": "second_unloading"}
    current_home = "home"
    current_joints = joints(transition_forward[0])
    removed = set()
    entries = []
    sequence = []
    boundaries = []
    for task in source["tasks"]:
        if not task["success"] or task["round"] > 11:
            continue
        round_number = task["round"]
        if round_number in alternatives:
            selected = alternatives[round_number]["selected"]
            candidate_rank = selected["attempt"]
            plan_path = Path(selected["plan"])
            approach_path = Path(selected["approach"])
        else:
            candidate_rank = 1
            plan_path = args.scene_root / f"round_{round_number:02}_plan.json"
            approach_path = args.loaded_root / f"round_{round_number:02}_approach.json"
        loaded_path = Path(loaded_summary[round_number]["result"])
        plan = json.loads(plan_path.read_text())
        approach = json.loads(approach_path.read_text())
        loaded = json.loads(loaded_path.read_text())
        desired_home = plan["initial_pose"]
        if not plan.get("success") or not approach.get("success") or not loaded.get("success") or \
                not loaded.get("loaded_return", {}).get("success"):
            raise ValueError(f"round {round_number} contains an unsuccessful component")
        if current_home != desired_home:
            switch = transition_forward if current_home == "home" else transition_reverse
            if not same(current_joints, joints(switch[0])):
                raise ValueError(f"round {round_number} home-switch start mismatch")
            switch = resample(switch)
        else:
            switch = [{"joints": current_joints}]
        pregrasp = resample(join(switch, plan["frames"]))
        approach_frames = resample(join([pregrasp[-1]], approach["frames"]))
        extraction = resample(loaded["frames"])
        loaded_return = resample(loaded["loaded_return"]["frames"])
        placement = placements[desired_home]
        place = resample(join([approach_frames[-1]], extraction, loaded_return, placement))
        release = dict(place[-1])
        home_return = resample([release] + [dict(frame) for frame in reversed(placement[:-1])])
        box_ids = [task["left"]] + ([] if task["right"] is None else [task["right"]])
        entry = {
            "round": round_number,
            "left": task["left"],
            "right": -1 if task["right"] is None else task["right"],
            "success": True,
            "initial_pose": desired_home,
            "unloading_pose": unloading_names[desired_home],
            "selected_candidate": candidate_rank,
            "planning_ms": loaded_summary[round_number]["planning_ms"],
            "pregrasp": cache_frames(pregrasp),
            "approach": cache_frames(approach_frames),
            "place": cache_frames(place),
            "home": cache_frames(home_return),
        }
        entries.append(entry)
        begin = len(sequence)
        sequence.extend(tagged(pregrasp, "PREGRASP", round_number, box_ids, False, True))
        sequence.extend(tagged(approach_frames[1:], "APPROACH", round_number, box_ids, False, True))
        sequence.append({"stage": "ATTACH", "round": round_number,
                         "joints": joints(approach_frames[-1]), "box_ids": box_ids,
                         "attached": True, "visible": True})
        sequence.extend(tagged(place[1:], "PLACE", round_number, box_ids, True, True))
        sequence.append({"stage": "RELEASE", "round": round_number,
                         "joints": joints(place[-1]), "box_ids": box_ids,
                         "attached": False, "visible": False})
        removed.update(box_ids)
        sequence.extend(tagged(home_return[1:], "HOME", round_number, box_ids, False, False))
        current_home = desired_home
        current_joints = joints(home_return[-1])
        boundaries.append({"round": round_number, "begin": begin, "end": len(sequence),
                           "home": desired_home, "candidate_rank": candidate_rank,
                           "removed_after": sorted(removed)})
    if len(entries) != 11:
        raise ValueError(f"expected 11 cache entries, got {len(entries)}")
    if current_home != "home" or not same(current_joints, joints(transition_forward[0])):
        raise ValueError("eleven-round sequence does not finish at first home")
    maximum_rotary_step = 0.0
    maximum_updown_step = 0.0
    stationary_boundaries = 0
    for index in range(1, len(sequence)):
        first = sequence[index - 1]["joints"]
        last = sequence[index]["joints"]
        maximum_rotary_step = max(maximum_rotary_step,
                                  max(abs(last[joint] - first[joint]) for joint in range(14)))
        maximum_updown_step = max(maximum_updown_step, abs(last[14] - first[14]))
        if same(first, last):
            stationary_boundaries += 1
    if maximum_rotary_step > ROTARY_STEP + 1e-9 or maximum_updown_step > UPDOWN_STEP + 1e-9:
        raise ValueError("resampled sequence exceeds continuity limits")
    cache = {"kind": "v3_fixed_wall_motion_stage_cache", "schema_version": 1,
             "joint_names": joint_names, "wall_distance_m": .90,
             "verified_rounds": 11, "entries": entries}
    replay = {"kind": "v3_complete_11_round_side_suction_sequence",
              "joint_names": joint_names, "success": True, "frames": sequence,
              "boundaries": boundaries, "initial_pose": "home", "final_pose": "home",
              "removed_box_ids": sorted(removed),
              "maximum_rotary_step_deg": math.degrees(maximum_rotary_step),
              "maximum_updown_step_m": maximum_updown_step,
              "stationary_boundaries": stationary_boundaries,
              "stage6_validation": "strict reverse of collision-verified loaded placement after payload removal"}
    args.cache_output.parent.mkdir(parents=True, exist_ok=True)
    args.sequence_output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_output.write_text(json.dumps(cache, separators=(",", ":")) + "\n")
    args.sequence_output.write_text(json.dumps(replay, separators=(",", ":")) + "\n")
    print(json.dumps({"cache": str(args.cache_output), "sequence": str(args.sequence_output),
                      "entries": len(entries), "frames": len(sequence),
                      "maximum_rotary_step_deg": replay["maximum_rotary_step_deg"],
                      "maximum_updown_step_m": maximum_updown_step,
                      "final_pose": current_home}, indent=2))


if __name__ == "__main__":
    main()
