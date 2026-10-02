#!/usr/bin/env python3
"""Aggregate the eleven complete yaw certificates into JSON, CSV, and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def operation_metrics(
    operation: dict[str, Any], joint_names: list[str]
) -> dict[str, float | int]:
    frames = operation["frames"]
    arm_indices = [
        index for index, name in enumerate(joint_names)
        if name.startswith("left_joint") or name.startswith("right_joint")
    ]
    maximum_step_deg = 0.0
    joint_flip_events = 0
    base_travel_m = 0.0
    default_pose = [float(value) for value in operation.get("base_pose_map", [0, 0, 0])]
    for previous, current in zip(frames, frames[1:]):
        for index in arm_indices:
            degrees = math.degrees(abs(
                float(current["joints"][index]) - float(previous["joints"][index])
            ))
            maximum_step_deg = max(maximum_step_deg, degrees)
            joint_flip_events += degrees > 180.0 + 1e-6
        previous_pose = [float(value) for value in previous.get("base_pose_map", default_pose)]
        current_pose = [float(value) for value in current.get("base_pose_map", default_pose)]
        base_travel_m += math.hypot(
            current_pose[0] - previous_pose[0], current_pose[1] - previous_pose[1]
        )
    return {
        "maximum_joint_step_deg": maximum_step_deg,
        "joint_flip_events": joint_flip_events,
        "base_travel_m": base_travel_m,
    }


def yaw_slug(yaw: int) -> str:
    return f"yaw-{'p' if yaw >= 0 else 'm'}{abs(yaw):02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_root = args.cases_root.resolve()
    output_dir = args.output_dir.resolve()
    baseline = read_json(args.baseline_cache.resolve())
    baseline_tasks = {int(task["box_id"]): task for task in baseline["tasks"]}
    rows: list[dict[str, Any]] = []
    total_frames = 0
    total_edges = 0
    maximum_joint_step = 0.0
    joint_flip_events = 0
    maximum_left_tilt = 0.0
    maximum_right_tilt = 0.0
    maximum_task_core_ms = 0.0
    all_task_core_ms: list[float] = []
    base_travel_values = set()
    adaptations: list[dict[str, Any]] = []

    for yaw in range(-5, 6):
        case_dir = cases_root / yaw_slug(yaw)
        case = read_json(case_dir / "summary.json")
        replay = read_json(case_dir / "v322-yaw-conveyor-replay.json")
        validation = read_json(case_dir / "v322-yaw-conveyor-validation.json")
        cache = read_json(case_dir / "v322-yaw-plan-cache.json")
        pickup_summary = read_json(case_dir / "pickup-summary.json")
        if not validation.get("success") or case.get("completed_boxes") != 25:
            raise RuntimeError(f"yaw {yaw:+d} is not a complete successful case")
        if case.get("cycle_count") != 15 or case.get("dual_cycle_count") != 10:
            raise RuntimeError(f"yaw {yaw:+d} violates the cycle contract")
        if case.get("one_sided_attachment_frames") != 0:
            raise RuntimeError(f"yaw {yaw:+d} has one-sided dual attachment frames")

        observed_yaws = {
            round(math.degrees(float(task["payload"]["base_pose_map"][2])))
            for task in cache["tasks"]
        }
        if observed_yaws != {yaw}:
            raise RuntimeError(f"yaw {yaw:+d} cache root mismatch: {observed_yaws}")
        metrics = [
            operation_metrics(operation, list(replay["joint_names"]))
            for operation in replay["operations"]
        ]
        frames = int(validation["checked_frames"])
        edges = int(validation["checked_edge_samples"])
        case_max_step = max(float(item["maximum_joint_step_deg"]) for item in metrics)
        case_flips = sum(int(item["joint_flip_events"]) for item in metrics)
        case_base_travel = sum(float(item["base_travel_m"]) for item in metrics)
        task_times = [
            float(item["selected_task_core_ms"]) for item in replay["box_planning"]
        ]
        total_frames += frames
        total_edges += edges
        maximum_joint_step = max(maximum_joint_step, case_max_step)
        joint_flip_events += case_flips
        maximum_left_tilt = max(maximum_left_tilt, float(validation["maximum_left_tilt_deg"]))
        maximum_right_tilt = max(maximum_right_tilt, float(validation["maximum_right_tilt_deg"]))
        maximum_task_core_ms = max(maximum_task_core_ms, max(task_times))
        all_task_core_ms.extend(task_times)
        base_travel_values.add(round(case_base_travel, 6))

        for group in pickup_summary:
            box_id = int(group["group"][0])
            baseline_height = float(baseline_tasks[box_id]["updown"])
            selected_height = float(group["height"])
            if not math.isclose(baseline_height, selected_height, abs_tol=1e-9):
                adaptations.append({
                    "yaw_deg": yaw,
                    "group": group["group"],
                    "baseline_updown_m": baseline_height,
                    "selected_updown_m": selected_height,
                    "mode": group["mode"],
                })

        rows.append({
            "yaw_deg": yaw,
            "completed_boxes": 25,
            "cycles": 15,
            "dual_cycles": 10,
            "one_sided_attachment_frames": 0,
            "checked_frames": frames,
            "checked_edge_samples": edges,
            "maximum_joint_step_deg": case_max_step,
            "joint_flip_events": case_flips,
            "maximum_left_tilt_deg": float(validation["maximum_left_tilt_deg"]),
            "maximum_right_tilt_deg": float(validation["maximum_right_tilt_deg"]),
            "task_core_total_ms": sum(task_times),
            "task_core_max_ms": max(task_times),
            "base_travel_m": case_base_travel,
        })

    if len(base_travel_values) != 1:
        raise RuntimeError(f"base travel differs across yaw cases: {sorted(base_travel_values)}")
    aggregate = {
        "schema": "alfa.v322_yaw_robustness_certificate.v1",
        "model_revision": "robot_v3.2.2-suction",
        "tool0_offset_local_z_m": 0.151,
        "upstream_base_commit": "d9c330cef72981390d81ac2b1cd5a6eb9e892195",
        "yaw_range_deg": [-5, 5],
        "yaw_step_deg": 1,
        "yaw_cases": len(rows),
        "successful_yaw_cases": len(rows),
        "completed_box_tasks": 25 * len(rows),
        "completed_cycles": 15 * len(rows),
        "completed_dual_cycles": 10 * len(rows),
        "one_sided_attachment_frames": 0,
        "checked_frames": total_frames,
        "checked_edge_samples": total_edges,
        "maximum_joint_step_deg": maximum_joint_step,
        "joint_flip_events": joint_flip_events,
        "maximum_left_tilt_deg": maximum_left_tilt,
        "maximum_right_tilt_deg": maximum_right_tilt,
        "task_core_total_ms": sum(all_task_core_ms),
        "task_core_average_ms": sum(all_task_core_ms) / len(all_task_core_ms),
        "task_core_max_ms": maximum_task_core_ms,
        "task_core_below_3s": sum(value < 3000.0 for value in all_task_core_ms),
        "base_travel_per_case_m": base_travel_values.pop(),
        "cache_loading_counted": False,
        "algorithm": {
            "ik_seed": "yaw-zero or adjacent certified-yaw precontact joints; target pose always re-solved",
            "candidate_selection": "common group updown and suction-mode search",
            "entry_planning": "fresh yaw-specific whole-body bridge with validated prefix/suffix fallback",
            "validation": "complete MoveIt/FCL frames plus 1deg/1cm edge samples",
        },
        "candidate_adaptations": adaptations,
        "cases": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "yaw-certificate.json", aggregate)
    with (output_dir / "yaw-results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# V3.2.2 Dual-Arm 5x5 Yaw Robustness",
        "",
        "## Result",
        "",
        "- Yaw range: `-5 deg .. +5 deg`, integer `1 deg` steps",
        f"- Successful yaw cases: `{len(rows)}/{len(rows)}`",
        f"- Completed box tasks: `{25 * len(rows)}/{25 * len(rows)}`",
        f"- Completed cycles: `{15 * len(rows)}/{15 * len(rows)}`",
        f"- Simultaneous dual cycles: `{10 * len(rows)}/{10 * len(rows)}`",
        "- One-sided attachment frames: `0`",
        f"- MoveIt/FCL frames: `{total_frames:,}`",
        f"- Additional 1 deg / 1 cm edge samples: `{total_edges:,}`",
        f"- Maximum joint step: `{maximum_joint_step:.6f} deg`",
        f"- Joint flip events: `{joint_flip_events}`",
        f"- Maximum carried tilt, left/right: `{maximum_left_tilt:.6f} / {maximum_right_tilt:.6f} deg`",
        f"- Task core average/max: `{aggregate['task_core_average_ms'] / 1000.0:.3f} / {maximum_task_core_ms / 1000.0:.3f} s`",
        f"- Task cores below 3 s: `{aggregate['task_core_below_3s']}/{len(all_task_core_ms)}`",
        f"- Base travel per case: `{aggregate['base_travel_per_case_m']:.2f} m`",
        "- Cache loading counted as planning time: `false`",
        "",
        "## Per-yaw validation",
        "",
        "| Yaw | Frames | Edge samples | Max joint step | Left/right tilt | Task max |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['yaw_deg']:+d} deg | {row['checked_frames']:,} | "
            f"{row['checked_edge_samples']:,} | {row['maximum_joint_step_deg']:.6f} deg | "
            f"{row['maximum_left_tilt_deg']:.4f}/{row['maximum_right_tilt_deg']:.4f} deg | "
            f"{row['task_core_max_ms'] / 1000.0:.3f} s |"
        )
    lines.extend([
        "",
        "## Algorithm change",
        "",
        "The original yaw-zero task fixed one candidate branch and one shared lift per group. "
        "Yaw-specific planning now re-solves each current 6D target, searches a collision-valid "
        "common lift for paired boxes, and rebuilds every whole-body entry bridge. The yaw-zero "
        "or adjacent-yaw joints are numerical IK seeds only; no trajectory is reused as the result.",
        "",
        "At `-5 deg`, group `2+4` fails retreat at the original `0.00 m` lift and succeeds at "
        "`-0.25 m`. At `+5 deg`, top-suction box 23 fails at the original `-0.75 m` lift and "
        "succeeds at `-1.00 m`. These are the boundary candidate-selection improvements.",
        "",
        "## Boundary",
        "",
        "This certificate covers planning and collision validation for the configured robot, box "
        "wall, station x/y, and conveyor replay. It does not certify navigation, localization, "
        "suction hardware, or execution control.",
        "",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
