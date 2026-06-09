#!/usr/bin/env python3
"""Visualize dual_arm_planner MoveIt trajectory JSONL with Rerun."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr


def load_rerun_helpers():
    candidates = [
        Path(__file__).resolve().parents[4] / "scripts" / "ik_benchmark" / "scripts" / "visualize_rerun.py",
        Path.cwd() / "scripts" / "ik_benchmark" / "scripts" / "visualize_rerun.py",
        Path.cwd().parent / "scripts" / "ik_benchmark" / "scripts" / "visualize_rerun.py",
    ]
    for helper_path in candidates:
        if helper_path.exists():
            spec = importlib.util.spec_from_file_location("alfa_visualize_rerun_helpers", helper_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("cannot find scripts/ik_benchmark/scripts/visualize_rerun.py")


def read_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    header: dict[str, Any] = {}
    stages: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    with path.open() as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_type = row.get("type")
            if row_type == "header":
                header = row
            elif row_type == "stage":
                stages.append(row)
            elif row_type == "summary":
                summary = row
    return header, stages, summary


def all_boxes(box_x: float) -> dict[int, tuple[float, float, float]]:
    rows = [
        [(1, 0.6), (3, 0.2), (2, -0.2), (4, -0.6)],
        [(5, 0.6), (7, 0.2), (6, -0.2), (8, -0.6)],
        [(9, 0.6), (11, 0.2), (10, -0.2), (12, -0.6)],
        [(13, 0.6), (15, 0.2), (14, -0.2), (16, -0.6)],
        [(17, 0.6), (19, 0.2), (18, -0.2), (20, -0.6)],
    ]
    out: dict[int, tuple[float, float, float]] = {}
    for row_i, row in enumerate(rows):
        z = 0.2 + 0.4 * (len(rows) - 1 - row_i)
        for box_id, y in row:
            out[box_id] = (box_x, y, z)
    return out


def log_box_stack(box_x: float) -> None:
    centers = []
    half_sizes = []
    colors = []
    labels = []
    for box_id, (x, y, z) in sorted(all_boxes(box_x).items()):
        centers.append([x + 0.15, y, z])
        half_sizes.append([0.15, 0.2, 0.2])
        colors.append([255, 180, 60, 90])
        labels.append(str(box_id))
    rr.log("scene/boxes", rr.Boxes3D(centers=centers, half_sizes=half_sizes, colors=colors, labels=labels), static=True)


def transform_point(transform: np.ndarray, point: list[float]) -> list[float]:
    homogeneous = transform @ np.array([point[0], point[1], point[2], 1.0], dtype=float)
    return [float(homogeneous[0]), float(homogeneous[1]), float(homogeneous[2])]


def joint_dict_from_point(stage: dict[str, Any], point: dict[str, Any]) -> dict[str, float]:
    state_map = stage.get("start_state", {}).get("joint_map", {})
    out = {str(name): float(value) for name, value in state_map.items()}
    names = stage.get("trajectory", {}).get("joint_names", [])
    positions = point.get("positions", [])
    for name, value in zip(names, positions):
        out[str(name)] = float(value)
    return out


def log_target_pose(path: str, pose: dict[str, Any], color: list[int], label: str, base_to_world: np.ndarray) -> None:
    position = pose.get("position", [])
    if len(position) != 3:
        return
    world_position = transform_point(base_to_world, [float(v) for v in position])
    rr.log(path, rr.Points3D([world_position], colors=[color], radii=[0.035], labels=[label]))


def log_stage_text(stage: dict[str, Any], point_index: int, point_count: int) -> None:
    extra = stage.get("extra", {})
    ik = extra.get("ik", {})
    selected = ik.get("selected", {})
    lines = [
        f"stage: {stage.get('stage')}",
        f"point: {point_index + 1}/{point_count}",
        f"kind: {extra.get('stage_kind', 'joint_target')}",
        f"ik_strategy: {ik.get('strategy', '')}",
        f"selected_h: {selected.get('h', '')} score: {selected.get('score', '')}",
        f"h_index/seed_index: {selected.get('h_index', '')}/{selected.get('seed_index', '')}",
        f"legal/trials: {ik.get('legal_count', '')}/{ik.get('trial_count', '')} wall_ms={ik.get('wall_ms', '')}",
    ]
    rr.log("info/stage", rr.TextLog("\n".join(lines)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MoveIt box-stack flow JSONL")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--stride", type=int, default=1, help="Log every Nth trajectory point")
    parser.add_argument("--no-meshes", action="store_true")
    parser.add_argument("--robot-path", default="robot")
    args = parser.parse_args()

    helpers = load_rerun_helpers()
    header, stages, summary = read_jsonl(args.jsonl)
    if not stages:
        raise SystemExit(f"no stage records in {args.jsonl}")

    if args.save:
        rr.init("moveit_box_stack_flow", recording_id=f"moveit_flow_{args.jsonl.stem}")
        rr.save(str(args.save))
    elif args.connect:
        rr.init("moveit_box_stack_flow", recording_id=f"moveit_flow_{args.jsonl.stem}")
        rr.connect()
    else:
        rr.init("moveit_box_stack_flow", recording_id=f"moveit_flow_{args.jsonl.stem}")
        rr.spawn()

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    robot = helpers.UrdfRobot(helpers.render_current_urdf())
    helpers.log_robot_static_model(robot, args.robot_path, log_meshes=not args.no_meshes)
    base_to_world = robot.fk({}).get("base_link", np.eye(4))
    log_box_stack(float(header.get("box_front_x", 0.625)))

    sample = 0
    stride = max(1, args.stride)
    for stage in stages:
        extra = stage.get("extra", {})
        log_target_pose("targets/left", extra.get("left_target", {}), [0, 220, 255], "left target", base_to_world)
        log_target_pose("targets/right", extra.get("right_target", {}), [255, 120, 0], "right target", base_to_world)
        points = stage.get("trajectory", {}).get("points", [])
        if not points:
            continue
        selected_indices = list(range(0, len(points), stride))
        if selected_indices[-1] != len(points) - 1:
            selected_indices.append(len(points) - 1)
        for point_index in selected_indices:
            helpers.set_sample_time(sample)
            point = points[point_index]
            helpers.log_robot_state(robot, joint_dict_from_point(stage, point), args.robot_path)
            log_stage_text(stage, point_index, len(points))
            sample += 1

    rr.log("info/summary", rr.TextLog(json.dumps(summary, ensure_ascii=False, indent=2)))
    print(f"Loaded {len(stages)} stages, logged {sample} samples from {args.jsonl}")
    if args.save:
        print(f"saved: {args.save}")


if __name__ == "__main__":
    main()
