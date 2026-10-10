#!/usr/bin/python3
"""Plan whole-body home-to-precontact bridges for the chest-carry task."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


TOOL_DIR = Path(__file__).resolve().parent
WORKSPACE = TOOL_DIR.parents[1]
SCRIPT_DIR = WORKSPACE / "ros2_ws/src/alfa_robot_moveit_config/scripts"
for path in (TOOL_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_chest_payloads as chest
import scan_chest_carry_box_wall as scan


DEFAULT_BASELINE = chest.DEFAULT_BASELINE
GROUPS: list[tuple[int, ...]] = [
    (1, 5), (2, 4), (3,),
    (6, 10), (7, 9), (8,),
    (11, 15), (12, 14), (13,),
    (16, 20), (17, 19), (18,),
    (21, 25), (22, 24), (23,),
]
HOME = [
    -0.3,
    0.0,
    *[math.radians(value) for value in (155, -105, 20, 90, -90, -40, 0)],
    *[math.radians(value) for value in (-155, 105, -20, -90, 90, 40, 0)],
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def precontact(payload: dict[str, Any]) -> list[float]:
    return [float(value) for value in next(
        frame["joints"] for frame in payload["frames"]
        if frame["stage"] == "precontact"
    )]


def active_indices(names: list[str], side: str) -> list[int]:
    return [names.index(f"{side}_joint{index}") for index in range(1, 8)]


def group_goal(payloads: list[dict[str, Any]]) -> list[float]:
    if len(payloads) == 1:
        return precontact(payloads[0])
    names = [str(value) for value in payloads[0]["joint_names"]]
    goal = list(HOME)
    updowns = {round(float(payload["initial_updown"]), 9) for payload in payloads}
    if len(updowns) != 1:
        raise ValueError("paired payloads do not share one pickup updown")
    goal[0] = updowns.pop()
    for payload in payloads:
        values = precontact(payload)
        for index in active_indices(names, str(payload["side"])):
            goal[index] = values[index]
    return goal


def same_state(left: list[float], right: list[float], tolerance: float = 1e-7) -> bool:
    return len(left) == len(right) and max(
        abs(a - b) for a, b in zip(left, right)
    ) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cache", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--payload-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--only-group",
        action="append",
        default=[],
        help="plan only the exact dash-separated group label, for example 23 or 22-24",
    )
    args = parser.parse_args()

    cache = read_json(args.baseline_cache.resolve())
    tasks = {int(task["box_id"]): task for task in cache["tasks"]}
    payload_dir = args.payload_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    removed: set[int] = set()
    summary: list[dict[str, Any]] = []

    selected_labels = set(args.only_group)
    selected_groups = [
        group for group in GROUPS
        if not selected_labels
        or "-".join(str(box_id) for box_id in group) in selected_labels
    ]
    if not selected_groups:
        raise ValueError(f"no groups match --only-group={args.only_group}")

    for group in GROUPS:
        label = "-".join(str(box_id) for box_id in group)
        if selected_labels and label not in selected_labels:
            removed.update(group)
            continue
        payloads = [read_json(payload_dir / f"box-{box_id:02d}.json") for box_id in group]
        goal = group_goal(payloads)
        file_label = "-".join(f"{box_id:02d}" for box_id in group)
        path = output_dir / f"group-{file_label}.json"
        bridge = read_json(path) if path.is_file() else {}
        resumed = (
            bridge.get("success") is True
            and bridge.get("frames")
            and same_state([float(value) for value in bridge["frames"][0]["joints"]], HOME)
            and same_state([float(value) for value in bridge["frames"][-1]["joints"]], goal)
        )
        if not resumed:
            source = payloads[0]
            planner = chest.planner_args(
                tasks[group[0]], source, [0.0] * 7, None, None
            )
            planner.timeout = 360.0
            planner.rrt_planning_time = 30.0
            planner.rrt_planning_attempts = 16
            attempt = scan.run_attempt(
                planner,
                group[0],
                str(source["grasp_mode"]),
                removed | set(group),
                str(source["side"]),
                float(source["initial_updown"]),
                chest.box_center(group[0]),
                path,
                transition_from_joints=HOME,
                transition_to_joints=goal,
            )
            bridge = read_json(path) if path.is_file() else {}
            if not attempt.get("success") or not bridge.get("success"):
                raise RuntimeError(
                    f"group {group} bridge failed: "
                    f"{attempt.get('failure_stage')} {attempt.get('failure_reason')}"
                )
        summary.append({
            "group": list(group),
            "success": True,
            "resumed": resumed,
            "frame_count": len(bridge["frames"]),
            "total_ms": float(bridge.get("total_ms", 0.0)),
            "path": str(path),
        })
        chest.write_json(output_dir / "summary.json", summary)
        removed.update(group)
        print(
            f"ENTRY_BRIDGE group={group} resumed={resumed} "
            f"frames={len(bridge['frames'])} "
            f"time_ms={float(bridge.get('total_ms', 0.0)):.3f}",
            flush=True,
        )

    print(f"RESULT success={len(summary)}/{len(selected_groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
