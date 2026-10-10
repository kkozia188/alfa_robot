#!/usr/bin/env python3
"""Shared contracts and failure classification for docking-error experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


SCHEMA = "alfa.v322_docking_error_matrix.v1"
MODEL_REVISION = "robot_v3.2.2-suction"
TOOL0_OFFSET_LOCAL_Z_M = 0.151
UPSTREAM_BASE_COMMIT = "d9c330cef72981390d81ac2b1cd5a6eb9e892195"
NOMINAL_UPPER_BASE_X_M = -0.60
NOMINAL_LOWER_BASE_X_M = -0.35
NOMINAL_BASE_Y_M = 0.0
CONTACT_X_M = 0.75
VEHICLE_FRONT_X_IN_BASE_M = 0.500000002779484
DEFAULT_UPPER_FRONT_CLEARANCE_M = 0.85
DEFAULT_LOWER_FRONT_CLEARANCE_M = 0.60
SENTINEL_GROUP_INDICES = (1, 2, 8, 9, 10, 13, 14, 15)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_slug(value: float, scale: int, width: int) -> str:
    scaled = int(round(abs(value) * scale))
    sign = "p" if value >= 0.0 else "m"
    return f"{sign}{scaled:0{width}d}"


def case_slug(dx_m: float, dy_m: float, yaw_deg: float) -> str:
    return (
        f"dx-{signed_slug(dx_m, 1000, 3)}mm_"
        f"dy-{signed_slug(dy_m, 1000, 3)}mm_"
        f"yaw-{signed_slug(yaw_deg, 10, 3)}d10"
    )


def clearance_slug(upper_m: float, lower_m: float) -> str:
    return (
        f"front-u{int(round(upper_m * 1000)):04d}mm_"
        f"l{int(round(lower_m * 1000)):04d}mm"
    )


def quantize_front_clearance(clearance_m: float) -> float:
    """Round a terminal clearance input onto the certified 1 cm grid."""
    return float(
        Decimal(str(clearance_m)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )


def base_x_for_front_clearance(clearance_m: float) -> float:
    return CONTACT_X_M - VEHICLE_FRONT_X_IN_BASE_M - clearance_m


def clearance_pose_contract(upper_m: float, lower_m: float) -> dict[str, Any]:
    return {
        "front_clearance": {
            "reference": "vehicle_front_contact_plane_to_box_front_contact_plane",
            "upper_rows_1_to_3_m": upper_m,
            "lower_rows_4_to_5_m": lower_m,
        },
        "upper_base_pose_map": [base_x_for_front_clearance(upper_m), 0.0, 0.0],
        "lower_base_pose_map": [base_x_for_front_clearance(lower_m), 0.0, 0.0],
    }


def x_clearance_variable(upper_m: float, lower_m: float) -> str:
    upper_changed = not math.isclose(
        upper_m, DEFAULT_UPPER_FRONT_CLEARANCE_M, abs_tol=1e-12
    )
    lower_changed = not math.isclose(
        lower_m, DEFAULT_LOWER_FRONT_CLEARANCE_M, abs_tol=1e-12
    )
    if upper_changed and lower_changed:
        raise ValueError("X experiment may vary only one row-family clearance")
    return "upper_rows_1_to_3" if upper_changed else (
        "lower_rows_4_to_5" if lower_changed else "nominal"
    )


def pose_contract(dx_m: float, dy_m: float, yaw_deg: float) -> dict[str, Any]:
    return {
        "error": {"dx_m": dx_m, "dy_m": dy_m, "yaw_deg": yaw_deg},
        "upper_base_pose_map": [
            NOMINAL_UPPER_BASE_X_M + dx_m,
            NOMINAL_BASE_Y_M + dy_m,
            math.radians(yaw_deg),
        ],
        "lower_base_pose_map": [
            NOMINAL_LOWER_BASE_X_M + dx_m,
            NOMINAL_BASE_Y_M + dy_m,
            math.radians(yaw_deg),
        ],
    }


def is_single_axis_error(dx_m: float, dy_m: float, yaw_deg: float) -> bool:
    return sum(abs(value) > 1e-12 for value in (dx_m, dy_m, yaw_deg)) <= 1


def local_shuttle_poses(
    pickup_pose: list[float], backoff_m: float, right_m: float
) -> tuple[list[float], list[float]]:
    if len(pickup_pose) != 3 or not all(math.isfinite(value) for value in pickup_pose):
        raise ValueError("pickup pose must contain three finite values")
    cosine = math.cos(pickup_pose[2])
    sine = math.sin(pickup_pose[2])
    backed = [
        pickup_pose[0] - backoff_m * cosine,
        pickup_pose[1] - backoff_m * sine,
        pickup_pose[2],
    ]
    conveyor = [
        backed[0] + right_m * sine,
        backed[1] - right_m * cosine,
        backed[2],
    ]
    return backed, conveyor


def classify_failure(stage: str, reason: str) -> str:
    stage_text = stage.lower()
    text = f"{stage} {reason}".lower()
    if "timeout" in text or "budget" in text:
        return "planning_timeout"
    if "tilt" in text or "upside" in text or "flip" in text:
        return "carried_box_tilt"
    if "ik" in stage_text or "precontact" in stage_text or "unreachable" in text:
        return "ik_or_reachability"
    collision_count = re.search(r"collision=(\d+)", text)
    if (
        "self_collision" in text
        or "collision:" in text
        or "collision" in stage_text
        or (collision_count and int(collision_count.group(1)) > 0)
    ):
        return "collision"
    if "contact" in text or "suction" in text:
        return "contact_or_suction"
    if "approach" in text:
        return "approach"
    if "retreat" in text or "extract" in text:
        return "retreat"
    if "transition" in text or "bridge" in text:
        return "whole_body_transition"
    if "validation" in text or "edge" in text:
        return "trajectory_validation"
    if "process" in text or "exit_" in text:
        return "planner_process"
    return "unknown"


def first_failed_attempt(output_dir: Path) -> dict[str, Any]:
    for name in ("pickup-attempts.json", "bridge-attempts.json"):
        path = output_dir / name
        if not path.is_file():
            continue
        value = read_json(path)
        attempts = value if isinstance(value, list) else value.get("attempts", [])
        failed = [item for item in attempts if not item.get("success")]
        if failed:
            item = dict(failed[-1])
            item["source"] = name
            return item
    return {}


def evaluate_quality(
    *,
    task_core_ms: list[float],
    maximum_bridge_ms: float,
    maximum_tilt_deg: float,
    maximum_joint_step_deg: float,
    joint_flip_events: int,
    thresholds: dict[str, float],
) -> tuple[str, list[str]]:
    findings: list[str] = []
    if task_core_ms and max(task_core_ms) >= thresholds["task_core_fail_ms"]:
        findings.append("task_core_timeout")
    elif task_core_ms and max(task_core_ms) >= thresholds["task_core_warn_ms"]:
        findings.append("task_core_over_target")
    if maximum_bridge_ms >= thresholds["bridge_fail_ms"]:
        findings.append("bridge_timeout")
    elif maximum_bridge_ms >= thresholds["bridge_warn_ms"]:
        findings.append("bridge_over_target")
    if maximum_tilt_deg >= thresholds["tilt_fail_deg"]:
        findings.append("carried_box_tilt_failure")
    elif maximum_tilt_deg >= thresholds["tilt_warn_deg"]:
        findings.append("carried_box_tilt_warning")
    if joint_flip_events:
        findings.append("joint_flip")
    if maximum_joint_step_deg > thresholds["maximum_joint_step_deg"] + 1e-9:
        findings.append("joint_step_exceeded")
    failures = {
        "task_core_timeout",
        "bridge_timeout",
        "carried_box_tilt_failure",
        "joint_flip",
        "joint_step_exceeded",
    }
    if failures.intersection(findings):
        return "failed_quality_gate", findings
    return ("degraded" if findings else "passed"), findings
