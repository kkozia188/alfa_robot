#!/usr/bin/python3
"""Summarize chest geometry, stage ordering, and base-speed improvement."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
RERUN_DIR = WORKSPACE / "ros2_ws/src/alfa_robot_rerun"
if str(RERUN_DIR) not in sys.path:
    sys.path.insert(0, str(RERUN_DIR))

from alfa_robot_rerun.visualize_rerun import UrdfRobot


VEHICLE_FRONT_X_M = 0.500000002779484


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def box_pose(
    robot: UrdfRobot,
    names: list[str],
    frame: dict[str, Any],
    attachment: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joints = {
        name: float(value) for name, value in zip(names, frame["joints"])
    }
    tool = robot.fk(joints)[str(attachment["tool_link"])]
    offset = np.asarray(attachment["center_in_tool"], dtype=float)
    rotation = tool[:3, :3] @ np.asarray(
        attachment["rotation_in_tool"], dtype=float
    )
    center = tool[:3, 3] + tool[:3, :3] @ offset
    size = np.asarray(attachment["size"], dtype=float)
    return center, rotation, size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model-urdf", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    replay = read_json(args.replay.resolve())
    validation = read_json(args.validation.resolve())
    rerun_summary = read_json(args.summary.resolve())
    robot = UrdfRobot(args.model_urdf.resolve().read_text(encoding="utf-8"))
    names = [str(value) for value in replay["joint_names"]]
    cases: list[dict[str, Any]] = []
    minimum_front_clearance = math.inf
    minimum_pair_gap = math.inf
    maximum_center_error = 0.0
    ordering_failures: list[str] = []

    for operation in replay["operations"]:
        if operation["kind"] == "base_transition":
            continue
        frames = list(operation["frames"])
        chest_index = next(
            index for index, frame in enumerate(frames)
            if frame["stage"] == "chest_carry_ready"
        )
        base_index = next(
            index for index, frame in enumerate(frames)
            if frame["stage"] == "base_backoff_after_chest_carry"
        )
        if chest_index >= base_index:
            ordering_failures.append(str(operation["label"]))
        chest = frames[chest_index]
        centers: dict[str, np.ndarray] = {}
        extents_y: dict[str, float] = {}
        box_cases: list[dict[str, Any]] = []
        for side in ("left", "right"):
            box_id = int(operation.get(f"{side}_box_id", 0))
            if box_id <= 0:
                continue
            attachment = operation[f"{side}_attachment"]
            center, rotation, size = box_pose(robot, names, chest, attachment)
            centers[side] = center
            half = 0.5 * size
            extent_x = float(np.abs(rotation[0, :]) @ half)
            extent_y = float(np.abs(rotation[1, :]) @ half)
            extents_y[side] = extent_y
            front_clearance = float(center[0] - extent_x - VEHICLE_FRONT_X_M)
            mode = next(
                item["mode"] for item in operation["box_planning"]
                if int(item["box_id"]) == box_id
            )
            target = np.asarray(
                [0.90, 0.40 if side == "left" else -0.40, 0.85]
                if mode == "top_suction" else
                [0.75, 0.32 if side == "left" else -0.32, 0.95],
                dtype=float,
            )
            center_error = float(np.linalg.norm(center - target))
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rotation[2, 2])))))
            minimum_front_clearance = min(minimum_front_clearance, front_clearance)
            maximum_center_error = max(maximum_center_error, center_error)
            box_cases.append({
                "box_id": box_id,
                "side": side,
                "mode": mode,
                "center_base_m": center.tolist(),
                "target_center_base_m": target.tolist(),
                "center_error_m": center_error,
                "front_clearance_m": front_clearance,
                "tilt_deg": tilt,
            })
        pair_gap = None
        if set(centers) == {"left", "right"}:
            pair_gap = float(
                abs(centers["left"][1] - centers["right"][1])
                - extents_y["left"] - extents_y["right"]
            )
            minimum_pair_gap = min(minimum_pair_gap, pair_gap)
        cases.append({
            "label": operation["label"],
            "chest_frame_index": chest_index,
            "first_base_motion_frame_index": base_index,
            "base_speed_m_s": float(operation["base_speed_m_s"]),
            "pair_gap_m": pair_gap,
            "boxes": box_cases,
        })

    old_speed = 0.10
    new_speed = float(replay["chest_carry"]["base_speed_m_s"])
    total_travel = float(rerun_summary["base_travel_m"])
    report = {
        "schema": "alfa.v322_chest_carry_fast_base_report.v1",
        "success": (
            bool(validation.get("success"))
            and not ordering_failures
            and maximum_center_error <= 1e-4
            and minimum_front_clearance > 0.0
            and minimum_pair_gap > 0.0
        ),
        "model_revision": replay["model_revision"],
        "completed_boxes": int(rerun_summary["completed_boxes"]),
        "group_count": int(rerun_summary["group_count"]),
        "checked_frames": int(validation["checked_frames"]),
        "checked_edge_samples": int(validation["checked_edge_samples"]),
        "maximum_joint_step_deg": float(rerun_summary["maximum_joint_step_deg"]),
        "joint_flip_events": int(rerun_summary["joint_flip_events"]),
        "maximum_left_tilt_deg": float(validation["maximum_left_tilt_deg"]),
        "maximum_right_tilt_deg": float(validation["maximum_right_tilt_deg"]),
        "maximum_chest_center_error_m": maximum_center_error,
        "minimum_vehicle_front_to_box_clearance_m": minimum_front_clearance,
        "minimum_dual_box_gap_m": minimum_pair_gap,
        "ordering_failures": ordering_failures,
        "base": {
            "travel_m": total_travel,
            "old_speed_m_s": old_speed,
            "new_speed_m_s": new_speed,
            "speed_multiplier": new_speed / old_speed,
            "old_ideal_motion_time_s": total_travel / old_speed,
            "new_ideal_motion_time_s": total_travel / new_speed,
            "ideal_motion_time_reduction_percent": (
                1.0 - old_speed / new_speed
            ) * 100.0,
        },
        "cases": cases,
    }
    write_json(args.output_json.resolve(), report)
    lines = [
        "# V3.2.2 Chest Carry + Fast Base Result",
        "",
        f"- Result: **{'PASS' if report['success'] else 'FAIL'}**",
        f"- Boxes/cycles: **{report['completed_boxes']}/25**, **{report['group_count']}/15**",
        f"- MoveIt/FCL: **{report['checked_frames']:,} frames + {report['checked_edge_samples']:,} edge samples**",
        f"- Maximum joint step / flips: **{report['maximum_joint_step_deg']:.6f} deg / {report['joint_flip_events']}**",
        f"- Maximum carried-box tilt L/R: **{report['maximum_left_tilt_deg']:.6f} / {report['maximum_right_tilt_deg']:.6f} deg**",
        f"- Maximum chest-center error: **{1000.0 * maximum_center_error:.6f} mm**",
        f"- Minimum vehicle-front clearance: **{minimum_front_clearance:.3f} m**",
        f"- Minimum dual-box gap: **{minimum_pair_gap:.3f} m**",
        f"- Base speed: **{old_speed:.2f} -> {new_speed:.2f} m/s ({new_speed / old_speed:.1f}x)**",
        f"- Ideal base-motion time reduction: **{report['base']['ideal_motion_time_reduction_percent']:.1f}%**",
        "- Ordering: **chest carry ready before every base backoff**",
    ]
    args.output_md.resolve().write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"RESULT success={report['success']} clearance={minimum_front_clearance:.3f}m "
        f"pair_gap={minimum_pair_gap:.3f}m center_error={maximum_center_error:.9f}m"
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
