#!/usr/bin/env python3
"""Compile pose-driven box-wall requests into reusable motion candidates."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "alfa.v322_box_wall_motion_family.v1"
PLAN_SCHEMA = "alfa.v322_box_wall_motion_family_plan.v1"
JOINT_COUNT = 17
EPSILON = 1.0e-9
CONTROL_JOINT_NAMES = (
    "updown",
    *(f"left_joint{index}" for index in range(1, 8)),
    *(f"right_joint{index}" for index in range(1, 8)),
    "head_joint",
    "head_pitch_joint",
)


class ContractError(ValueError):
    """Raised when a task contract cannot be interpreted safely."""


@dataclasses.dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    @classmethod
    def from_json(cls, value: dict[str, Any], field: str) -> "Pose":
        try:
            position = _finite_tuple(value["position_m"], 3, f"{field}.position_m")
            quaternion = _finite_tuple(
                value["orientation_xyzw"], 4, f"{field}.orientation_xyzw"
            )
        except (KeyError, TypeError) as error:
            raise ContractError(f"{field} must contain position_m and orientation_xyzw") from error
        norm = math.sqrt(sum(component * component for component in quaternion))
        if norm < EPSILON:
            raise ContractError(f"{field}.orientation_xyzw must be non-zero")
        return cls(position, tuple(component / norm for component in quaternion))

    def to_json(self) -> dict[str, list[float]]:
        return {
            "position_m": list(self.position),
            "orientation_xyzw": list(self.orientation_xyzw),
        }


@dataclasses.dataclass(frozen=True)
class Wall:
    frame: str
    rows: int
    columns: int
    bottom_z_m: float
    center_y_m: float
    box_size_m: tuple[float, float, float]
    gap_y_m: float
    gap_z_m: float
    orientation_tolerance_deg: float

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Wall":
        rows = _positive_int(value.get("rows"), "wall.rows")
        columns = _positive_int(value.get("columns"), "wall.columns")
        size = _finite_tuple(value.get("box_size_m"), 3, "wall.box_size_m")
        if min(size) <= 0.0:
            raise ContractError("wall.box_size_m values must be positive")
        gap_y = _finite_number(value.get("gap_y_m", 0.0), "wall.gap_y_m")
        gap_z = _finite_number(value.get("gap_z_m", 0.0), "wall.gap_z_m")
        tolerance = _finite_number(
            value.get("orientation_tolerance_deg", 5.0),
            "wall.orientation_tolerance_deg",
        )
        if min(gap_y, gap_z, tolerance) < 0.0:
            raise ContractError("wall gaps and orientation tolerance must be non-negative")
        return cls(
            frame=str(value.get("frame", "map")),
            rows=rows,
            columns=columns,
            bottom_z_m=_finite_number(value.get("bottom_z_m", 0.0), "wall.bottom_z_m"),
            center_y_m=_finite_number(value.get("center_y_m", 0.0), "wall.center_y_m"),
            box_size_m=size,
            gap_y_m=gap_y,
            gap_z_m=gap_z,
            orientation_tolerance_deg=tolerance,
        )

    @property
    def depth_m(self) -> float:
        return self.box_size_m[0]

    @property
    def width_m(self) -> float:
        return self.box_size_m[1]

    @property
    def height_m(self) -> float:
        return self.box_size_m[2]

    def locate(self, pose: Pose) -> tuple[int, int, int]:
        """Return row-from-top, column-from-left and a private scene slot."""
        _, y, z = pose.position
        column_value = (
            (self.center_y_m - y) / (self.width_m + self.gap_y_m)
            + 0.5 * (self.columns - 1)
        )
        row_from_bottom_value = (
            z - self.bottom_z_m - 0.5 * self.height_m
        ) / (self.height_m + self.gap_z_m)
        column = int(round(column_value))
        row_from_bottom = int(round(row_from_bottom_value))
        expected_y = self.center_y_m + (
            0.5 * (self.columns - 1) - column
        ) * (self.width_m + self.gap_y_m)
        expected_z = self.bottom_z_m + (
            row_from_bottom + 0.5
        ) * (self.height_m + self.gap_z_m)
        position_tolerance = max(self.gap_y_m, self.gap_z_m, 0.03)
        if not 0 <= column < self.columns or not 0 <= row_from_bottom < self.rows:
            raise ContractError("target pose lies outside the configured box wall")
        if abs(y - expected_y) > position_tolerance or abs(z - expected_z) > position_tolerance:
            raise ContractError(
                "target pose does not resolve to a wall cell within "
                f"{position_tolerance:.3f}m"
            )
        row_from_top = self.rows - row_from_bottom
        scene_slot = (row_from_top - 1) * self.columns + column + 1
        return row_from_top, column + 1, scene_slot

    def clearance_slots(
        self,
        row_from_top: int,
        column_from_left: int,
        channels: Iterable[str],
        radius_cells: int,
    ) -> list[int]:
        """Resolve geometric clearance channels to private scene adapter slots."""
        slots: set[int] = set()
        for channel in channels:
            if channel == "top":
                rows = range(1, row_from_top)
                columns = range(
                    max(1, column_from_left - radius_cells),
                    min(self.columns, column_from_left + radius_cells) + 1,
                )
            elif channel == "bottom":
                rows = range(row_from_top + 1, self.rows + 1)
                columns = range(
                    max(1, column_from_left - radius_cells),
                    min(self.columns, column_from_left + radius_cells) + 1,
                )
            elif channel == "left":
                rows = (row_from_top,)
                columns = range(max(1, column_from_left - radius_cells), column_from_left)
            elif channel == "right":
                rows = (row_from_top,)
                columns = range(
                    column_from_left + 1,
                    min(self.columns, column_from_left + radius_cells) + 1,
                )
            else:
                raise ContractError(
                    "clearance_channels values must be top, bottom, left, or right"
                )
            for row in rows:
                for column in columns:
                    slots.add((row - 1) * self.columns + column)
        return sorted(slots)


@dataclasses.dataclass(frozen=True)
class Space:
    front_clearance_m: float
    top_clearance_m: float
    left_clearance_m: float
    right_clearance_m: float

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Space":
        result = cls(*(
            _finite_number(value.get(field, 0.0), f"available_space.{field}")
            for field in (
                "front_clearance_m",
                "top_clearance_m",
                "left_clearance_m",
                "right_clearance_m",
            )
        ))
        if min(dataclasses.astuple(result)) < 0.0:
            raise ContractError("available-space clearances must be non-negative")
        return result


@dataclasses.dataclass(frozen=True)
class Template:
    name: str
    grasp_mode: str
    minimum_front_clearance_m: float
    minimum_top_clearance_m: float
    approach_distance_m: float
    retreat_distance_m: float
    preferred_rows_from_top: tuple[int, ...]
    place_policy: str

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Template":
        name = str(value.get("name", "")).strip()
        mode = str(value.get("grasp_mode", "")).strip()
        if not name:
            raise ContractError("template name must not be empty")
        if mode not in {"front", "top_suction"}:
            raise ContractError(f"template {name} has unsupported grasp_mode {mode!r}")
        rows = tuple(int(item) for item in value.get("preferred_rows_from_top", []))
        return cls(
            name=name,
            grasp_mode=mode,
            minimum_front_clearance_m=_finite_number(
                value.get("minimum_front_clearance_m", 0.0),
                f"template {name} minimum_front_clearance_m",
            ),
            minimum_top_clearance_m=_finite_number(
                value.get("minimum_top_clearance_m", 0.0),
                f"template {name} minimum_top_clearance_m",
            ),
            approach_distance_m=_positive_number(
                value.get("approach_distance_m"), f"template {name} approach_distance_m"
            ),
            retreat_distance_m=_positive_number(
                value.get("retreat_distance_m"), f"template {name} retreat_distance_m"
            ),
            preferred_rows_from_top=rows,
            place_policy=str(value.get("place_policy", "mobile_base_conveyor")),
        )


@dataclasses.dataclass(frozen=True)
class Candidate:
    template: Template
    arm: str
    updown_m: float
    score: float
    contact_pose: Pose
    reasons: tuple[str, ...]

    def to_json(self, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "template": self.template.name,
            "grasp_mode": self.template.grasp_mode,
            "arm": self.arm,
            "updown_m": self.updown_m,
            "score": round(self.score, 6),
            "contact_pose": self.contact_pose.to_json(),
            "approach_distance_m": self.template.approach_distance_m,
            "retreat_distance_m": self.template.retreat_distance_m,
            "place_policy": self.template.place_policy,
            "selection_reasons": list(self.reasons),
        }


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain one JSON object")
    return value


def compile_contract(contract: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ContractError(f"task schema must be {SCHEMA}")
    wall = Wall.from_json(contract.get("wall", {}))
    templates = tuple(Template.from_json(value) for value in family.get("templates", []))
    if not templates:
        raise ContractError("action-family config must define at least one template")
    updown_grid = tuple(
        sorted({_finite_number(value, "updown_grid_m") for value in family.get("updown_grid_m", [])}, reverse=True)
    )
    if not updown_grid or min(updown_grid) < -1.0 or max(updown_grid) > 0.0:
        raise ContractError("updown_grid_m must contain values in [-1, 0]")
    current_state = _current_state(contract.get("current_state", {}))
    tasks = contract.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ContractError("tasks must be a non-empty array")

    compiled: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, task in enumerate(tasks):
        request_id = str(task.get("request_id", f"request-{index + 1}"))
        try:
            item = compile_task(
                task, request_id, wall, templates, updown_grid, current_state, family
            )
            compiled.append(item)
            current_state = _estimated_terminal_state(current_state, item)
        except ContractError as error:
            failures.append({"request_id": request_id, "reason": str(error)})

    return {
        "schema": PLAN_SCHEMA,
        "model_revision": "robot_v3.2.2-suction",
        "tool0_offset_local_z_m": 0.151,
        "current_state_joint_order": list(CONTROL_JOINT_NAMES),
        "source_task_schema": SCHEMA,
        "wall": dataclasses.asdict(wall),
        "tasks": compiled,
        "compile_failures": failures,
        "summary": summarize_compilation(compiled, failures),
        "scope": (
            "pose-driven template and IK-candidate ordering; scene_slot is a private "
            "adapter key and is not an action-selection input"
        ),
    }


def compile_task(
    task: dict[str, Any],
    request_id: str,
    wall: Wall,
    templates: tuple[Template, ...],
    updown_grid: tuple[float, ...],
    current_state: tuple[float, ...],
    family: dict[str, Any],
) -> dict[str, Any]:
    pose = Pose.from_json(task.get("box_pose", {}), f"task {request_id}.box_pose")
    _require_axis_aligned_box(pose, wall.orientation_tolerance_deg)
    space = Space.from_json(task.get("available_space", {}))
    row, column, scene_slot = wall.locate(pose)
    channels_value = task.get("clearance_channels", [])
    if not isinstance(channels_value, list) or any(
        not isinstance(value, str) for value in channels_value
    ):
        raise ContractError("clearance_channels must be an array of direction strings")
    clearance_channels = tuple(dict.fromkeys(channels_value))
    clearance_radius_cells = task.get("clearance_radius_cells", 1)
    if (
        isinstance(clearance_radius_cells, bool)
        or not isinstance(clearance_radius_cells, int)
        or clearance_radius_cells <= 0
    ):
        raise ContractError("clearance_radius_cells must be a positive integer")
    clearance_slots = wall.clearance_slots(
        row, column, clearance_channels, clearance_radius_cells
    )
    allowed_arms = tuple(task.get("allowed_arms", ("left", "right")))
    if not allowed_arms or any(arm not in {"left", "right"} for arm in allowed_arms):
        raise ContractError("allowed_arms must contain left and/or right")

    candidates: list[Candidate] = []
    rejected: list[dict[str, str]] = []
    for template in templates:
        rejection = _template_rejection(template, space)
        if rejection:
            rejected.append({"template": template.name, "reason": rejection})
            continue
        for arm in allowed_arms:
            if arm == "left" and space.left_clearance_m <= 0.0:
                rejected.append({"template": template.name, "reason": "left side unavailable"})
                continue
            if arm == "right" and space.right_clearance_m <= 0.0:
                rejected.append({"template": template.name, "reason": "right side unavailable"})
                continue
            for updown in _rank_updown_grid(pose.position[2], current_state[0], updown_grid):
                contact = _contact_pose(pose, wall, template.grasp_mode, arm, family)
                score, reasons = _candidate_score(
                    template, arm, updown, row, pose.position[1], pose.position[2],
                    current_state, family
                )
                candidates.append(Candidate(template, arm, updown, score, contact, reasons))

    candidates.sort(key=lambda candidate: (candidate.score, candidate.template.name, candidate.arm))
    if not candidates:
        reasons = "; ".join(sorted({item["reason"] for item in rejected}))
        raise ContractError(f"no feasible action template: {reasons or 'no candidates'}")
    limit = _positive_int(family.get("candidate_limit", 12), "candidate_limit")
    selected = candidates[:limit]
    candidate_payloads = [
        candidate.to_json(rank) for rank, candidate in enumerate(selected, 1)
    ]
    for candidate_payload in candidate_payloads:
        candidate_payload["opposite_arm_policy"] = (
            "auto_safe"
            if abs(pose.position[1] - wall.center_y_m) <= EPSILON
            or float(candidate_payload["updown_m"]) <= -0.5 + EPSILON
            else "hold_current"
        )
    return {
        "request_id": request_id,
        "box_pose": pose.to_json(),
        "available_space": dataclasses.asdict(space),
        "derived_geometry": {
            "row_from_top": row,
            "column_from_left": column,
            "scene_slot": scene_slot,
            "scene_slot_role": "private collision-scene adapter only",
            "clearance_channels": list(clearance_channels),
            "clearance_radius_cells": clearance_radius_cells,
            "clearance_scene_slots": clearance_slots,
        },
        "current_state": {"joint_positions": list(current_state)},
        "selected_candidate": candidate_payloads[0],
        "candidates": candidate_payloads,
        "rejected_templates": rejected,
        "stages": [
            "pregrasp",
            "cartesian_approach",
            "attach",
            "cartesian_retreat",
            "loaded_return",
            "place",
            "release",
            "stow",
        ],
    }


def summarize_compilation(
    tasks: list[dict[str, Any]], failures: list[dict[str, str]]
) -> dict[str, Any]:
    rows: dict[int, dict[str, Any]] = {}
    for task in tasks:
        row = int(task["derived_geometry"]["row_from_top"])
        item = rows.setdefault(row, {"row_from_top": row, "compiled": 0, "templates": {}})
        item["compiled"] += 1
        name = str(task["selected_candidate"]["template"])
        item["templates"][name] = item["templates"].get(name, 0) + 1
    return {
        "requested": len(tasks) + len(failures),
        "compiled": len(tasks),
        "failed": len(failures),
        "rows": [rows[row] for row in sorted(rows)],
        "failure_reasons": _count_values(item["reason"] for item in failures),
    }


def _contact_pose(
    box_pose: Pose,
    wall: Wall,
    mode: str,
    arm: str,
    family: dict[str, Any],
) -> Pose:
    box_rotation = _quaternion_matrix(box_pose.orientation_xyzw)
    if mode == "front":
        lateral_offset = 0.0
        vertical_offset = 0.0
        if abs(box_pose.position[1] - wall.center_y_m) <= EPSILON:
            magnitude = _finite_number(
                family.get("center_front_y_offset_m", 0.08),
                "center_front_y_offset_m",
            )
            lateral_offset = magnitude if arm == "left" else -magnitude
            vertical_offset = _finite_number(
                family.get("center_front_z_offset_m", -0.05),
                "center_front_z_offset_m",
            )
        bottom_center_z = wall.bottom_z_m + 0.5 * wall.height_m
        if abs(box_pose.position[2] - bottom_center_z) <= max(wall.gap_z_m, 0.03):
            vertical_offset = _finite_number(
                family.get("bottom_front_z_offset_m", 0.12),
                "bottom_front_z_offset_m",
            )
        center_offset = (-0.5 * wall.depth_m, lateral_offset, vertical_offset)
        tool_rotation = _matrix_multiply(box_rotation, _rotation_y(math.pi / 2.0))
    else:
        top_x = _finite_number(family.get("top_suction_x_offset_m", -0.10), "top_suction_x_offset_m")
        center_offset = (top_x, 0.0, 0.5 * wall.height_m)
        roll = math.pi / 2.0 if arm == "left" else -math.pi / 2.0
        tool_rotation = _matrix_multiply(
            box_rotation,
            _matrix_multiply(_rotation_z(roll), _rotation_x(math.pi)),
        )
    rotated_offset = _matrix_vector(box_rotation, center_offset)
    position = tuple(a + b for a, b in zip(box_pose.position, rotated_offset))
    return Pose(position, _matrix_quaternion(tool_rotation))


def _candidate_score(
    template: Template,
    arm: str,
    updown: float,
    row: int,
    y: float,
    z: float,
    current_state: tuple[float, ...],
    family: dict[str, Any],
) -> tuple[float, tuple[str, ...]]:
    preferred_arm = "left" if y > EPSILON else "right" if y < -EPSILON else ""
    if not preferred_arm:
        center_preferences = family.get("center_arm_preference", {})
        preferred_arm = str(center_preferences.get(template.grasp_mode, ""))
        if preferred_arm not in {"", "left", "right"}:
            raise ContractError(
                f"center_arm_preference for {template.grasp_mode} must be left or right"
            )
    arm_penalty = 0.0 if not preferred_arm or arm == preferred_arm else 1.0
    row_penalty = 0.0 if row in template.preferred_rows_from_top else 0.75
    lift_weight = _finite_number(family.get("lift_change_weight", 2.0), "lift_change_weight")
    height_weight = _finite_number(
        family.get("target_height_weight", 4.0), "target_height_weight"
    )
    ideal_updown = min(0.0, max(-1.0, (z - 1.4) * 0.625))
    if abs(y) <= EPSILON:
        center_bias = family.get("center_lift_bias_m", {})
        ideal_updown = min(0.0, max(-1.0, ideal_updown + _finite_number(
            center_bias.get(template.grasp_mode, 0.0),
            f"center_lift_bias_m.{template.grasp_mode}",
        )))
    height_error = abs(updown - ideal_updown)
    score = (
        arm_penalty
        + row_penalty
        + lift_weight * abs(updown - current_state[0])
        + height_weight * height_error
    )
    reasons = [
        "arm follows target lateral side" if arm_penalty == 0.0 else "cross-body fallback arm",
        "template preferred for row" if row_penalty == 0.0 else "template row fallback",
        f"target-height error={height_error:.3f}m",
        f"lift change={abs(updown - current_state[0]):.3f}m",
    ]
    return score, tuple(reasons)


def _rank_updown_grid(
    target_z: float, current_updown: float, grid: tuple[float, ...]
) -> tuple[float, ...]:
    # Calibrated from the five V3.2.2 wall-height bands, then quantized to the
    # configured lift grid. The target height remains the only selector input.
    ideal = min(0.0, max(-1.0, (target_z - 1.4) * 0.625))
    return tuple(sorted(grid, key=lambda value: (abs(value - ideal), abs(value - current_updown))))


def _template_rejection(template: Template, space: Space) -> str:
    if space.front_clearance_m + EPSILON < template.minimum_front_clearance_m:
        return (
            f"front clearance {space.front_clearance_m:.3f}m below "
            f"{template.minimum_front_clearance_m:.3f}m"
        )
    if space.top_clearance_m + EPSILON < template.minimum_top_clearance_m:
        return (
            f"top clearance {space.top_clearance_m:.3f}m below "
            f"{template.minimum_top_clearance_m:.3f}m"
        )
    return ""


def _require_axis_aligned_box(pose: Pose, tolerance_deg: float) -> None:
    rotation = _quaternion_matrix(pose.orientation_xyzw)
    trace = max(-1.0, min(3.0, rotation[0][0] + rotation[1][1] + rotation[2][2]))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, 0.5 * (trace - 1.0)))))
    if angle > tolerance_deg + EPSILON:
        raise ContractError(
            f"box orientation differs from the certified wall frame by {angle:.3f}deg; "
            f"current backend limit is {tolerance_deg:.3f}deg"
        )


def _current_state(value: dict[str, Any]) -> tuple[float, ...]:
    joints = _finite_tuple(value.get("joint_positions"), JOINT_COUNT, "current_state.joint_positions")
    if not -1.0 - EPSILON <= joints[0] <= EPSILON:
        raise ContractError("current updown joint must be in [-1, 0]m")
    return joints


def _estimated_terminal_state(
    current: tuple[float, ...], task: dict[str, Any]
) -> tuple[float, ...]:
    # The mobile-base placement policy reverses the unladen entry path after
    # release, so every complete cycle returns to the caller's start posture.
    return current


def _finite_tuple(value: Any, size: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ContractError(f"{field} must contain {size} numbers")
    result = tuple(_finite_number(item, field) for item in value)
    return result


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{field} must be a finite number") from error
    if not math.isfinite(result):
        raise ContractError(f"{field} must be a finite number")
    return result


def _positive_number(value: Any, field: str) -> float:
    result = _finite_number(value, field)
    if result <= 0.0:
        raise ContractError(f"{field} must be positive")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field} must be a positive integer")
    return value


def _count_values(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _quaternion_matrix(q: tuple[float, float, float, float]) -> tuple[tuple[float, ...], ...]:
    x, y, z, w = q
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _matrix_quaternion(matrix: tuple[tuple[float, ...], ...]) -> tuple[float, float, float, float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
            0.25 * scale,
        )
    else:
        index = max(range(3), key=lambda item: matrix[item][item])
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
            quaternion = (
                0.25 * scale,
                (matrix[0][1] + matrix[1][0]) / scale,
                (matrix[0][2] + matrix[2][0]) / scale,
                (matrix[2][1] - matrix[1][2]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
            quaternion = (
                (matrix[0][1] + matrix[1][0]) / scale,
                0.25 * scale,
                (matrix[1][2] + matrix[2][1]) / scale,
                (matrix[0][2] - matrix[2][0]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
            quaternion = (
                (matrix[0][2] + matrix[2][0]) / scale,
                (matrix[1][2] + matrix[2][1]) / scale,
                0.25 * scale,
                (matrix[1][0] - matrix[0][1]) / scale,
            )
    norm = math.sqrt(sum(value * value for value in quaternion))
    return tuple(value / norm for value in quaternion)


def _matrix_multiply(a: tuple[tuple[float, ...], ...], b: tuple[tuple[float, ...], ...]):
    return tuple(
        tuple(sum(a[row][k] * b[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )


def _matrix_vector(matrix: tuple[tuple[float, ...], ...], vector: tuple[float, float, float]):
    return tuple(sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3))


def _rotation_x(angle: float):
    cosine, sine = math.cos(angle), math.sin(angle)
    return ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))


def _rotation_y(angle: float):
    cosine, sine = math.cos(angle), math.sin(angle)
    return ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))


def _rotation_z(angle: float):
    cosine, sine = math.cos(angle), math.sin(angle)
    return ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
