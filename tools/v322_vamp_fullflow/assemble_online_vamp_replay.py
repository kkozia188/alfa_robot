#!/usr/bin/python3
"""Assemble current-run VAMP tree paths into the 15-cycle production replay."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


GROUPS = [
    (1, 5), (2, 4), (3,), (6, 10), (7, 9), (8,),
    (11, 15), (12, 14), (13,), (16, 20), (17, 19), (18,),
    (21, 25), (22, 24), (23,),
]
STAGES = ("rrt_to_precontact", "rrt_to_contact", "rrt_loaded_retreat", "rrt_to_place")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def select_payload(box_id: int, roots: list[Path]):
    for root in roots:
        path = root / f"box-{box_id:02d}.json"
        if path.is_file():
            value = read_json(path)
            if value.get("success"):
                return value, path
    raise RuntimeError(f"missing successful payload {box_id}")


def interpolate(path: list[list[float]], count: int) -> list[list[float]]:
    if len(path) == count:
        return path
    if len(path) == 1:
        return [list(path[0]) for _ in range(count)]
    output = []
    for sample in range(count):
        coordinate = sample * (len(path) - 1) / (count - 1)
        left = int(math.floor(coordinate))
        right = min(left + 1, len(path) - 1)
        ratio = coordinate - left
        output.append([
            a + (b - a) * ratio for a, b in zip(path[left], path[right])
        ])
    return output


def combine_paths(
    payloads: list[dict[str, Any]], names: list[str], stage: str,
    left_attached: bool, right_attached: bool, base_pose: list[float],
) -> list[dict[str, Any]]:
    paths = [
        [[float(v) for v in frame["joints"]]
         for frame in payload["frames"] if frame["stage"] == stage]
        for payload in payloads
    ]
    if any(not path for path in paths):
        raise RuntimeError(f"missing {stage}")
    count = max(len(path) for path in paths)
    paths = [interpolate(path, count) for path in paths]
    output = []
    for samples in zip(*paths):
        combined = list(samples[0])
        combined[names.index("updown")] = max(
            sample[names.index("updown")] for sample in samples
        )
        for payload, sample in zip(payloads, samples):
            side = str(payload["side"])
            for joint in range(1, 8):
                index = names.index(f"{side}_joint{joint}")
                combined[index] = sample[index]
        output.append({
            "stage": stage, "joints": combined,
            "left_attached": left_attached,
            "right_attached": right_attached,
            "base_pose_map": list(base_pose),
        })
    return output


def base_operation(base_replay: dict[str, Any], label: str, phase: str):
    return next(
        item for item in base_replay["operations"]
        if f"{label} / {phase}" in item["label"] and item["phase"] == phase
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", type=Path, action="append", required=True)
    parser.add_argument("--base-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = [root.resolve() for root in args.payload_dir]
    payloads = {box: select_payload(box, roots) for box in range(1, 26)}
    base_replay = read_json(args.base_replay.resolve())
    names = [str(value) for value in payloads[1][0]["joint_names"]]
    operations, box_planning, manifest = [], [], {}
    for cycle, group in enumerate(GROUPS, 1):
        selected = [payloads[box][0] for box in group]
        label = " + ".join(f"Box {box}" for box in group)
        delivery = base_operation(base_replay, label, "loaded_delivery")
        dual_stow = base_operation(base_replay, label, "dual_stow")
        empty_return = base_operation(base_replay, label, "empty_return")
        pickup_pose = list(delivery["frames"][0]["base_pose_map"])
        safe_pose = list(delivery["frames"][-1]["base_pose_map"])
        held_left = any(item["side"] == "left" for item in selected)
        held_right = any(item["side"] == "right" for item in selected)
        frames = []
        frames.extend(combine_paths(selected, names, "rrt_to_precontact", False, False, pickup_pose))
        frames.extend(combine_paths(selected, names, "rrt_to_contact", False, False, pickup_pose)[1:])
        contact = dict(frames[-1], stage="attach_box",
                       left_attached=held_left, right_attached=held_right)
        frames.append(contact)
        frames.extend(combine_paths(selected, names, "rrt_loaded_retreat", held_left, held_right, pickup_pose)[1:])
        frames.extend(combine_paths(selected, names, "rrt_to_place", held_left, held_right, pickup_pose)[1:])
        frames[-1]["stage"] = "chest_carry_ready"
        delivery_frames = [dict(frame) for frame in delivery["frames"]]
        for index, frame in enumerate(delivery_frames):
            frame["stage"] = (
                "base_backoff_after_chest_carry" if index == 0 else
                "base_turn_180_loaded_at_safe_exit" if index + 1 == len(delivery_frames)
                else "vamp_loaded_delivery"
            )
        # Preserve the searched delivery start as an explicit contract event.
        # It duplicates the chest-ready state geometrically, but marks that all
        # subsequent base motion belongs to the loaded backoff tree.
        frames.extend(delivery_frames)
        released = dict(frames[-1], stage="release_after_turn_at_safe_exit",
                        left_attached=False, right_attached=False)
        frames.append(released)
        stow_frames = [dict(frame) for frame in dual_stow["frames"]]
        for frame in stow_frames:
            frame["stage"] = "vamp_rrt_stow_at_safe_exit"
        frames.extend(stow_frames[1:])
        return_frames = [dict(frame) for frame in empty_return["frames"]]
        for frame in return_frames:
            frame["stage"] = "vamp_empty_return_to_pickup"
        frames.extend(return_frames[1:])
        cycle_ms = sum(float(item["total_ms"]) for item in selected)
        cycle_ms += sum(float(item["phase_search"]["planning_core_ms"])
                        for item in (delivery, dual_stow, empty_return))
        operation = {
            "kind": "dual_safe_turn_unload_cycle" if len(group) == 2 else "single_safe_turn_unload_cycle",
            "label": label, "row": (group[0] - 1) // 5 + 1,
            "left_box_id": next((box for box, item in zip(group, selected) if item["side"] == "left"), 0),
            "right_box_id": next((box for box, item in zip(group, selected) if item["side"] == "right"), 0),
            "base_pose_map": pickup_pose, "pickup_mode": "simultaneous" if len(group) == 2 else "single",
            "left_attachment": delivery["left_attachment"],
            "right_attachment": delivery["right_attachment"],
            "obstacles": delivery["obstacles"], "allow_ground_model_base": True,
            "frames": frames,
            "box_planning": [],
            "online_vamp_contract": {
                "payload_paths": [str(payloads[box][1]) for box in group],
                "loaded_delivery": delivery["phase_search"],
                "dual_stow": dual_stow["phase_search"],
                "empty_return": empty_return["phase_search"],
                "fallback_used": False,
            },
        }
        for box, payload in zip(group, selected):
            entry = {
                "box_id": box, "row": (box - 1) // 5 + 1,
                "column": (box - 1) % 5 + 1, "side": payload["side"],
                "mode": payload["grasp_mode"], "source": "online_pose_vamp_full_flow",
                "selected_task_core_ms": cycle_ms,
                "accumulated_task_search_ms": cycle_ms,
                "selected_transition_core_ms": 0.0,
                "accumulated_transition_search_ms": 0.0,
            }
            operation["box_planning"].append(entry)
            box_planning.append(entry)
            manifest[str(box)] = str(payloads[box][1])
        operations.append(operation)
    write_json(args.output.resolve(), {
        "schema": "alfa.v3_scoop_5x5_dual_replay.v1",
        "model_revision": "robot_v3.2.2-suction",
        "tool0_offset_local_z_m": 0.151,
        "joint_names": names, "group_order": [list(group) for group in GROUPS],
        "source_replay": None, "cached_joint_endpoint_used": False,
        "planning_time_definition": {
            "selected_task_core_ms": "complete online Pose IK + VAMP trees + stow + base trees"
        },
        "box_planning": box_planning, "operations": operations,
        "payload_manifest": manifest,
    })
    print(f"RESULT cycles={len(operations)} boxes={len(box_planning)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
