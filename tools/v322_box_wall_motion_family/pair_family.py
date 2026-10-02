#!/usr/bin/env python3
"""Compile two box IDs into synchronized dual-arm action candidates."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from motion_family import ContractError, Pose, Template, Wall, compile_task


PAIR_SCHEMA = "alfa.v322_box_wall_dual_id_request.v1"
PAIR_PLAN_SCHEMA = "alfa.v322_box_wall_dual_id_plan.v1"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain one JSON object")
    return value


def pose_for_box_id(box_id: int, wall: Wall, center_x_m: float) -> Pose:
    if box_id < 1 or box_id > wall.rows * wall.columns:
        raise ContractError(
            f"box ID {box_id} is outside [1, {wall.rows * wall.columns}]"
        )
    row_from_top = (box_id - 1) // wall.columns + 1
    column_from_left = (box_id - 1) % wall.columns + 1
    row_from_bottom = wall.rows - row_from_top
    y = wall.center_y_m + (
        0.5 * (wall.columns - 1) - (column_from_left - 1)
    ) * (wall.width_m + wall.gap_y_m)
    z = wall.bottom_z_m + (
        row_from_bottom + 0.5
    ) * (wall.height_m + wall.gap_z_m)
    return Pose((center_x_m, y, z), (0.0, 0.0, 0.0, 1.0))


def compile_pair_contract(
    contract: dict[str, Any], family: dict[str, Any]
) -> dict[str, Any]:
    if contract.get("schema") != PAIR_SCHEMA:
        raise ContractError(f"pair schema must be {PAIR_SCHEMA}")
    wall_value = contract.get("wall", {})
    wall = Wall.from_json(wall_value)
    center_x_m = float(wall_value.get("center_x_m", 0.9))
    if not math.isfinite(center_x_m) or not 0.0 < center_x_m < 10.0:
        raise ContractError("wall.center_x_m must be finite and positive")
    current_state = contract.get("current_state", {})
    joint_positions = current_state.get("joint_positions")
    if not isinstance(joint_positions, list) or len(joint_positions) != 17:
        raise ContractError("current_state.joint_positions must contain 17 values")
    requests = contract.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ContractError("requests must be a non-empty array")

    expanded_family = copy.deepcopy(family)
    expanded_family["candidate_limit"] = 100
    updown_grid = tuple(
        sorted({float(value) for value in family["updown_grid_m"]}, reverse=True)
    )
    templates = tuple(
        Template.from_json(value)
        for value in family["templates"]
    )
    compiled = []
    failures = []
    for index, request in enumerate(requests, start=1):
        request_id = str(request.get("request_id", f"pair-{index}"))
        try:
            compiled.append(compile_pair_request(
                request,
                request_id,
                wall,
                center_x_m,
                tuple(float(value) for value in joint_positions),
                templates,
                updown_grid,
                expanded_family,
            ))
        except (ContractError, TypeError, ValueError) as error:
            failures.append({"request_id": request_id, "reason": str(error)})
    return {
        "schema": PAIR_PLAN_SCHEMA,
        "model_revision": "robot_v3.2.2-suction",
        "tool0_offset_local_z_m": 0.151,
        "source_schema": PAIR_SCHEMA,
        "wall": {
            **wall_value,
            "center_x_m": center_x_m,
        },
        "current_state": {"joint_positions": list(map(float, joint_positions))},
        "requests": compiled,
        "compile_failures": failures,
        "summary": {
            "requested": len(requests),
            "compiled": len(compiled),
            "failed": len(failures),
        },
        "contract_note": (
            "box IDs are geometry lookup keys only; no cached per-ID trajectory "
            "or per-ID action rule participates in candidate scoring"
        ),
    }


def compile_pair_request(
    request: dict[str, Any],
    request_id: str,
    wall: Wall,
    center_x_m: float,
    current_state: tuple[float, ...],
    templates: tuple[Any, ...],
    updown_grid: tuple[float, ...],
    family: dict[str, Any],
) -> dict[str, Any]:
    box_ids = request.get("box_ids")
    if (
        not isinstance(box_ids, list)
        or len(box_ids) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in box_ids)
    ):
        raise ContractError("box_ids must contain exactly two integer IDs")
    if box_ids[0] == box_ids[1]:
        raise ContractError("box_ids must be distinct")
    removed = request.get("removed_box_ids", [])
    if not isinstance(removed, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in removed
    ):
        raise ContractError("removed_box_ids must be an integer array")
    if any(value in box_ids for value in removed):
        raise ContractError("requested boxes cannot already be removed")

    tasks = []
    for box_id in box_ids:
        pose = pose_for_box_id(box_id, wall, center_x_m)
        task = compile_task(
            {
                "box_pose": pose.to_json(),
                "available_space": {
                    "front_clearance_m": 1.0,
                    "top_clearance_m": 1.0,
                    "left_clearance_m": 1.0,
                    "right_clearance_m": 1.0,
                },
                "allowed_arms": ["left", "right"],
            },
            f"{request_id}:box-{box_id}",
            wall,
            templates,
            updown_grid,
            current_state,
            family,
        )
        task["box_id"] = box_id
        tasks.append(task)

    paired = []
    for first in tasks[0]["candidates"]:
        for second in tasks[1]["candidates"]:
            if first["arm"] == second["arm"]:
                continue
            if first["grasp_mode"] != second["grasp_mode"]:
                continue
            if abs(float(first["updown_m"]) - float(second["updown_m"])) > 1.0e-9:
                continue
            left_box_id = box_ids[0] if first["arm"] == "left" else box_ids[1]
            right_box_id = box_ids[0] if first["arm"] == "right" else box_ids[1]
            left_candidate = first if first["arm"] == "left" else second
            right_candidate = first if first["arm"] == "right" else second
            row_max = max(
                int(task["derived_geometry"]["row_from_top"]) for task in tasks
            )
            base_candidates = [-0.60, -0.35] if row_max <= 3 else [-0.35, -0.60]
            retreat_distances = (
                [0.35, 0.30, 0.25, 0.20]
                if first["grasp_mode"] == "front" else [0.20]
            )
            for base_rank, base_x in enumerate(base_candidates):
                for retreat_distance in retreat_distances:
                    score = (
                        float(first["score"])
                        + float(second["score"])
                        + 0.25 * base_rank
                        + 2.0 * (retreat_distances[0] - retreat_distance)
                    )
                    paired.append({
                        "left_box_id": left_box_id,
                        "right_box_id": right_box_id,
                        "grasp_mode": first["grasp_mode"],
                        "common_updown_m": first["updown_m"],
                        "retreat_distance_m": retreat_distance,
                        "base_pose_map": [base_x, 0.0, 0.0],
                        "score": round(score, 6),
                        "left_contact_pose": left_candidate["contact_pose"],
                        "right_contact_pose": right_candidate["contact_pose"],
                        "source_candidate_ranks": [first["rank"], second["rank"]],
                    })
    paired.sort(key=lambda value: (
        float(value["score"]),
        abs(float(value["base_pose_map"][0])),
        int(value["left_box_id"]),
    ))
    if not paired:
        raise ContractError("no synchronized two-arm candidate shares one mode and lift")
    limit = int(request.get("candidate_limit", 64))
    if limit <= 0:
        raise ContractError("candidate_limit must be positive")
    for rank, candidate in enumerate(paired[:limit], start=1):
        candidate["rank"] = rank
    return {
        "request_id": request_id,
        "box_ids": box_ids,
        "removed_box_ids": sorted(set(removed)),
        "targets": [
            {
                "box_id": task["box_id"],
                "box_pose": task["box_pose"],
                "row_from_top": task["derived_geometry"]["row_from_top"],
                "column_from_left": task["derived_geometry"]["column_from_left"],
            }
            for task in tasks
        ],
        "different_rows": (
            tasks[0]["derived_geometry"]["row_from_top"]
            != tasks[1]["derived_geometry"]["row_from_top"]
        ),
        "selected_candidate": paired[0],
        "candidates": paired[:limit],
    }
