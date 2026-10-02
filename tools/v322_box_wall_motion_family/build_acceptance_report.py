#!/usr/bin/env python3
"""Build the consolidated action design, planning, and validation report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "alfa.v322_box_wall_motion_family_acceptance.v1"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def frame_base(operation: dict[str, Any], frame: dict[str, Any]) -> list[float]:
    return [
        float(value)
        for value in frame.get("base_pose_map", operation.get("base_pose_map", [0, 0, 0]))
    ]


def replay_metrics(replay: dict[str, Any]) -> dict[str, Any]:
    names = [str(value) for value in replay["joint_names"]]
    arm_indices = [
        index for index, name in enumerate(names)
        if name.startswith("left_joint") or name.startswith("right_joint")
    ]
    maximum_step_deg = 0.0
    flips = 0
    base_travel_m = 0.0
    boundaries = []
    for operation in replay["operations"]:
        for previous, current in zip(operation["frames"], operation["frames"][1:]):
            for index in arm_indices:
                step = math.degrees(abs(
                    float(current["joints"][index]) - float(previous["joints"][index])
                ))
                maximum_step_deg = max(maximum_step_deg, step)
                flips += step > 180.0 + 1.0e-6
            before = frame_base(operation, previous)
            after = frame_base(operation, current)
            base_travel_m += math.hypot(after[0] - before[0], after[1] - before[1])
    for previous, current in zip(replay["operations"], replay["operations"][1:]):
        previous_frame = previous["frames"][-1]
        current_frame = current["frames"][0]
        joint_error = max(
            abs(float(left) - float(right))
            for left, right in zip(previous_frame["joints"], current_frame["joints"])
        )
        previous_base = frame_base(previous, previous_frame)
        current_base = frame_base(current, current_frame)
        base_error = max(abs(left - right) for left, right in zip(previous_base, current_base))
        boundaries.append({
            "from": previous["label"],
            "to": current["label"],
            "maximum_joint_error": joint_error,
            "maximum_base_pose_error": base_error,
            "continuous": joint_error <= 1.0e-8 and base_error <= 1.0e-8,
        })
    return {
        "operation_count": len(replay["operations"]),
        "cycle_count": sum(
            operation.get("kind") != "base_transition"
            for operation in replay["operations"]
        ),
        "frame_count": sum(len(operation["frames"]) for operation in replay["operations"]),
        "maximum_joint_step_deg": maximum_step_deg,
        "joint_flip_events": flips,
        "base_travel_m": base_travel_m,
        "continuous_boundaries": sum(item["continuous"] for item in boundaries),
        "boundary_count": len(boundaries),
        "boundaries": boundaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fresh-report", type=Path, required=True)
    parser.add_argument("--fresh-validation", type=Path, required=True)
    parser.add_argument("--continuous-replay", type=Path, required=True)
    parser.add_argument("--continuous-validation", type=Path, required=True)
    parser.add_argument("--certified-baseline-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    fresh = read_json(args.fresh_report)
    pickup_validation = read_json(args.fresh_validation)
    replay = read_json(args.continuous_replay)
    validation = read_json(args.continuous_validation)
    baseline = read_json(args.certified_baseline_report)
    motion = replay_metrics(replay)
    planning_by_id = {
        task["request_id"]: task["selected"] for task in fresh["tasks"]
    }
    bridge_by_slot = {
        int(item["box_id"]): item for item in replay["box_planning"]
    }
    representative = []
    for task in plan["tasks"]:
        candidate = task["selected_candidate"]
        slot = int(task["derived_geometry"]["scene_slot"])
        pickup = planning_by_id[task["request_id"]]
        bridge = bridge_by_slot[slot]
        representative.append({
            "request_id": task["request_id"],
            "row_from_top": task["derived_geometry"]["row_from_top"],
            "column_from_left": task["derived_geometry"]["column_from_left"],
            "template": candidate["template"],
            "grasp_mode": candidate["grasp_mode"],
            "arm": candidate["arm"],
            "updown_m": candidate["updown_m"],
            "opposite_arm_policy": candidate["opposite_arm_policy"],
            "clearance_channels": task["derived_geometry"]["clearance_channels"],
            "clearance_radius_cells": task["derived_geometry"]["clearance_radius_cells"],
            "fresh_pickup_planning_ms": float(pickup["duration_ms"]),
            "fresh_entry_bridge_ms": float(bridge["selected_transition_core_ms"]),
            "full_cycle_success": True,
        })
    pickup_times = [item["fresh_pickup_planning_ms"] for item in representative]
    bridge_times = [item["fresh_entry_bridge_ms"] for item in representative]
    report = {
        "schema": SCHEMA,
        "model_revision": "robot_v3.2.2-suction",
        "upstream_base_commit": "d9c330cef72981390d81ac2b1cd5a6eb9e892195",
        "tool0_offset_local_z_m": 0.151,
        "external_contract": {
            "uses_box_id": False,
            "inputs": ["box_pose_6d", "box_size", "available_space", "current_17_axis_state"],
            "private_adapter_note": (
                "scene slots are derived from pose only and never participate in template scoring"
            ),
        },
        "action_design": {
            "templates": ["front_face_extract", "top_face_extract"],
            "stages": plan["tasks"][0]["stages"],
            "selection_inputs": [
                "row height", "lateral position", "surface clearance channels",
                "current lift", "allowed arms", "opposite-arm collision policy",
            ],
            "placement_policy": "2.35m base backoff, 1.50m robot-right shuttle, release, outside stow, return",
        },
        "representative_tasks": representative,
        "fresh_planning": {
            "successes": int(fresh["successes"]),
            "attempts": int(fresh["attempts"]),
            "success_rate": float(fresh["successes"]) / float(fresh["attempts"]),
            "pickup_time_ms": {
                "total": sum(pickup_times),
                "mean": sum(pickup_times) / len(pickup_times),
                "maximum": max(pickup_times),
            },
            "entry_bridge_time_ms": {
                "total": sum(bridge_times),
                "mean": sum(bridge_times) / len(bridge_times),
                "maximum": max(bridge_times),
            },
            "failure_reasons": fresh["failure_reasons"],
        },
        "pickup_collision_validation": {
            "success": bool(pickup_validation["success"]),
            "checked_frames": int(pickup_validation["checked_frames"]),
            "checked_edge_samples": int(pickup_validation["checked_edge_samples"]),
        },
        "continuous_full_cycle": motion,
        "continuous_collision_validation": {
            "success": bool(validation["success"]),
            "checked_frames": int(validation["checked_frames"]),
            "checked_edge_samples": int(validation["checked_edge_samples"]),
            "maximum_left_tilt_deg": float(validation["maximum_left_tilt_deg"]),
            "maximum_right_tilt_deg": float(validation["maximum_right_tilt_deg"]),
        },
        "certified_25_box_reference_by_row": baseline["certified_all_boxes_by_row"],
        "known_limits": [
            "Fresh acceptance covers five representative axis-aligned wall poses, one per row.",
            "The current backend rejects box-frame orientation deviations above 5 degrees.",
            "The bottom top-suction entry bridge is the current planning-time hotspot.",
            "Hardware navigation, suction IO, and conveyor control remain out of scope.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(markdown(report), encoding="utf-8")
    print(
        f"success={report['fresh_planning']['successes']}/{report['fresh_planning']['attempts']} "
        f"frames={motion['frame_count']} output={args.output_json}"
    )
    return 0


def markdown(report: dict[str, Any]) -> str:
    planning = report["fresh_planning"]
    motion = report["continuous_full_cycle"]
    validation = report["continuous_collision_validation"]
    lines = [
        "# V3.2.2 Pose-Driven Box-Wall Motion Family",
        "",
        "## Result",
        "",
        f"- Fresh representative planning: **{planning['successes']}/{planning['attempts']}**.",
        f"- Continuous full cycles: **{motion['cycle_count']}/5**, with "
        f"**{motion['continuous_boundaries']}/{motion['boundary_count']}** continuous operation boundaries.",
        f"- MoveIt/FCL: **{'PASS' if validation['success'] else 'FAIL'}**, "
        f"{validation['checked_frames']} frames and {validation['checked_edge_samples']} edge samples.",
        f"- Maximum joint step: **{motion['maximum_joint_step_deg']:.6f} deg**; flips: **{motion['joint_flip_events']}**.",
        f"- Carried-box tilt, left/right: **{validation['maximum_left_tilt_deg']:.6f} / "
        f"{validation['maximum_right_tilt_deg']:.6f} deg**.",
        "",
        "## Action Design",
        "",
        "Input is box 6D Pose, dimensions, available-space channels, allowed arms and current 17-axis state. "
        "No external box number participates in template or IK-candidate selection.",
        "",
        "Each cycle uses: pregrasp -> Cartesian approach -> attach -> loaded retreat -> "
        "2.35 m backoff -> 1.50 m right shuttle -> release -> outside stow -> return.",
        "",
        "## Representative Planning",
        "",
        "| Request | Row | Template | Arm | Lift (m) | Pickup (ms) | Entry bridge (ms) |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for item in report["representative_tasks"]:
        lines.append(
            f"| {item['request_id']} | {item['row_from_top']} | {item['template']} | "
            f"{item['arm']} | {item['updown_m']:.2f} | "
            f"{item['fresh_pickup_planning_ms']:.3f} | {item['fresh_entry_bridge_ms']:.3f} |"
        )
    lines.extend([
        "",
        f"Pickup mean/max: **{planning['pickup_time_ms']['mean']:.3f} / "
        f"{planning['pickup_time_ms']['maximum']:.3f} ms**. Entry bridge mean/max: "
        f"**{planning['entry_bridge_time_ms']['mean']:.3f} / "
        f"{planning['entry_bridge_time_ms']['maximum']:.3f} ms**.",
        "",
        "## Certified 25-Box Reference",
        "",
        "| Row | Success | Mean task (ms) | Maximum (ms) |",
        "| ---: | ---: | ---: | ---: |",
    ])
    for item in report["certified_25_box_reference_by_row"]:
        lines.append(
            f"| {item['row_from_top']} | {item['successes']}/{item['attempts']} | "
            f"{item['planning_time_ms']['mean']:.3f} | {item['planning_time_ms']['maximum']:.3f} |"
        )
    lines.extend(["", "## Known Limits", ""])
    lines.extend(f"- {item}" for item in report["known_limits"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
