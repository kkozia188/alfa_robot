#!/usr/bin/env python3
"""Pure data helpers for the V3 docking-error benchmark."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STAGES = ("PREGRASP", "APPROACH", "PLACE", "HOME")
PHASES = ("IK", "PREGRASP", "EXTRACT", "RETURN")

# MOTION-261 numbering is top-to-bottom and +Y to -Y. The planner catalog is
# bottom-to-top and -Y to +Y, hence catalog_id = 25 - motion_box_id.
MOTION_GROUPS = (
    (1, (1, 5), "side_suction"),
    (2, (2, 4), "side_suction"),
    (3, (3,), "side_suction"),
    (4, (6, 10), "side_suction"),
    (5, (7, 9), "side_suction"),
    (6, (8,), "side_suction"),
    (7, (11, 15), "side_suction"),
    (8, (12, 14), "side_suction"),
    (9, (13,), "side_suction"),
    (10, (16, 20), "side_suction"),
    (11, (17, 19), "side_suction"),
    (12, (18,), "side_suction"),
    (13, (21, 25), "side_suction"),
    (14, (22, 24), "top_suction"),
    (15, (23,), "top_suction"),
)

LEFT_MOTION_BOXES = {1, 2, 6, 7, 11, 12, 16, 17, 21, 22, 23}
RIGHT_MOTION_BOXES = set(range(1, 26)) - LEFT_MOTION_BOXES


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def cycle_catalog() -> list[dict[str, Any]]:
    output = []
    for cycle_id, motion_boxes, grasp_mode in MOTION_GROUPS:
        left = [box_id for box_id in motion_boxes if box_id in LEFT_MOTION_BOXES]
        right = [box_id for box_id in motion_boxes if box_id in RIGHT_MOTION_BOXES]
        if len(left) > 1 or len(right) > 1:
            raise RuntimeError(f"invalid arm assignment for cycle {cycle_id}")
        output.append(
            {
                "cycle_id": cycle_id,
                "motion_box_ids": list(motion_boxes),
                "left_motion_box_id": left[0] if left else None,
                "right_motion_box_id": right[0] if right else None,
                "left_catalog_box_id": 25 - left[0] if left else None,
                "right_catalog_box_id": 25 - right[0] if right else None,
                "grasp_mode": grasp_mode,
                "row_from_top": (motion_boxes[0] - 1) // 5 + 1,
            }
        )
    return output


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "alfa.v3_docking_error_benchmark_config.v1":
        raise ValueError("unsupported benchmark config schema")
    matrix = config.get("error_matrix")
    if not isinstance(matrix, dict):
        raise ValueError("error_matrix must be an object")
    for key in ("x_m", "y_m", "yaw_deg"):
        values = matrix.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"error_matrix.{key} must be a non-empty list")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError(f"error_matrix.{key} values must be finite numbers")
        if len(set(float(value) for value in values)) != len(values):
            raise ValueError(f"error_matrix.{key} contains duplicates")
    cycle_ids = config.get("cycle_ids", list(range(1, 16)))
    if not isinstance(cycle_ids, list) or not cycle_ids:
        raise ValueError("cycle_ids must be a non-empty list")
    if len(set(cycle_ids)) != len(cycle_ids) or any(
        not isinstance(value, int) or not 1 <= value <= 15 for value in cycle_ids
    ):
        raise ValueError("cycle_ids must be unique integers in [1, 15]")
    poses = config.get("nominal_base_pose_map")
    if not isinstance(poses, dict):
        raise ValueError("nominal_base_pose_map must be an object")
    for key in ("upper_rows", "lower_rows"):
        pose = poses.get(key)
        if not isinstance(pose, list) or len(pose) != 3 or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) for value in pose
        ):
            raise ValueError(f"nominal_base_pose_map.{key} must be finite [x,y,yaw]")


def sample_id(x_m: float, y_m: float, yaw_deg: float) -> str:
    def token(value: float, scale: float, width: int) -> str:
        rounded = int(round(abs(value) * scale))
        return ("p" if value >= 0.0 else "m") + f"{rounded:0{width}d}"

    return (
        f"dx_{token(x_m, 1000.0, 3)}_"
        f"dy_{token(y_m, 1000.0, 3)}_"
        f"yaw_{token(yaw_deg, 1000.0, 4)}"
    )


def matrix_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_config(config)
    matrix = config["error_matrix"]
    output = []
    for x_m, y_m, yaw_deg in itertools.product(
        matrix["x_m"], matrix["y_m"], matrix["yaw_deg"]
    ):
        output.append(
            {
                "sample_id": sample_id(float(x_m), float(y_m), float(yaw_deg)),
                "x_m": float(x_m),
                "y_m": float(y_m),
                "yaw_deg": float(yaw_deg),
            }
        )
    return output


def selected_cycles(config: dict[str, Any]) -> list[dict[str, Any]]:
    requested = set(config.get("cycle_ids", range(1, 16)))
    return [cycle for cycle in cycle_catalog() if cycle["cycle_id"] in requested]


def design_fingerprint(config: dict[str, Any]) -> str:
    design = {
        "model_revision": config.get("model_revision"),
        "tool0_offset_local_z_m": config.get("tool0_offset_local_z_m"),
        "upstream_base_commit": config.get("upstream_base_commit"),
        "error_matrix": config.get("error_matrix"),
        "cycle_ids": config.get("cycle_ids", list(range(1, 16))),
        "nominal_base_pose_map": config.get("nominal_base_pose_map"),
        "wall_pose_map": config.get("wall_pose_map"),
    }
    encoded = json.dumps(design, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def failure_text(stage: dict[str, Any]) -> str:
    return " ".join(
        str(stage.get(key, ""))
        for key in ("failure_stage", "error_message", "error_detail", "diagnostic")
    ).lower()


def planning_failure_text(stage: dict[str, Any]) -> str:
    return " ".join(
        str(stage.get(key, ""))
        for key in ("failure_stage", "error_detail", "diagnostic")
    ).lower()


def classify_failure(stage: dict[str, Any]) -> str:
    if stage.get("ok"):
        return "none"
    text = failure_text(stage)
    if stage.get("timed_out") or "timeout" in text or "timed out" in text:
        return "timeout"
    if any(token in text for token in ("tf unavailable", "older than 500ms", "base pose tf", "planar_root")):
        return "tf_or_base_pose"
    if any(token in text for token in ("invalid goal", "invalid_request", "quaternion", "non-positive", "outside the fixed", "does not match", "already removed")):
        return "invalid_input_or_pose"
    if any(token in text for token in ("precontact_ik", "shared_height_ik", "candidates=0", "ik failed", "no ik")):
        return "ik_unreachable"
    if any(token in text for token in ("joint_bounds", "bounds=", "joint limit", "satisfiesbounds")):
        return "joint_limit"
    if "collision" in text:
        return "collision"
    if any(token in text for token in ("rrt", "cartesian", "shortcut", "planning failed", "no complete", "loaded_transfer", "dual_independent_plan")):
        return "path_planning"
    if any(token in text for token in ("execution failed", "controller", "fjt", "settling", "goal rejected")):
        return "execution_or_protocol"
    if any(token in text for token in ("process", "internal error", "exception", "unavailable")):
        return "infrastructure"
    return "other"


def infer_planning_phases(cycle: dict[str, Any]) -> dict[str, bool | None]:
    phases: dict[str, bool | None] = {name: None for name in PHASES}
    stages = cycle.get("stages", [])
    pregrasp = next((stage for stage in stages if stage.get("stage") == "PREGRASP"), None)
    if pregrasp is None:
        return phases
    if pregrasp.get("ok"):
        for name in PHASES:
            phases[name] = True
        return phases

    text = planning_failure_text(pregrasp)
    if any(token in text for token in ("precontact_ik", "shared_height_ik", "candidates=0", "no ik")):
        phases["IK"] = False
        return phases

    phases["IK"] = True
    if any(token in text for token in ("rrt_to_precontact", "pregrasp", "transition")):
        phases["PREGRASP"] = False
        return phases

    phases["PREGRASP"] = True
    if any(token in text for token in (
        "cartesian_approach", "attach_box", "cartesian_lift", "cartesian_retreat",
        "loaded_transfer", "rear_placement", "dual_combined", "dual_independent_plan",
    )):
        phases["EXTRACT"] = False
        return phases

    phases["EXTRACT"] = True
    if any(token in text for token in ("rrt_return", "home_return", "home_updown", "loaded_home")):
        phases["RETURN"] = False
    return phases


def _rate(success: int, denominator: int) -> float | None:
    return success / denominator if denominator else None


def build_summary(config: dict[str, Any], records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    expected_cycles = len(selected_cycles(config))
    stage_counts = {stage: Counter() for stage in STAGES}
    phase_counts = {phase: Counter() for phase in PHASES}
    failures: Counter[str] = Counter()
    axis_data: dict[str, dict[str, Counter[str]]] = {
        axis: defaultdict(Counter) for axis in ("x_m", "y_m", "yaw_deg")
    }
    sample_summaries = []
    flat_cycles = []

    for record in rows:
        sample_successes = 0
        cycles = record.get("cycles", [])
        error = record["error"]
        for cycle in cycles:
            phase_outcomes = cycle.get("planning_phases") or infer_planning_phases(cycle)
            cycle["planning_phases"] = phase_outcomes
            stages_by_name = {stage["stage"]: stage for stage in cycle.get("stages", [])}
            for name in STAGES:
                entry = stages_by_name.get(name)
                if entry is None:
                    stage_counts[name]["not_attempted"] += 1
                else:
                    stage_counts[name]["attempted"] += 1
                    stage_counts[name]["success" if entry.get("ok") else "failed"] += 1
            for name, outcome in phase_outcomes.items():
                phase_counts[name][
                    "not_reached" if outcome is None else ("success" if outcome else "failed")
                ] += 1
            failed_stage = next((stage for stage in cycle.get("stages", []) if not stage.get("ok")), None)
            category = classify_failure(failed_stage) if failed_stage else "none"
            if category != "none":
                failures[category] += 1
            cycle_success = bool(cycle.get("ok"))
            sample_successes += int(cycle_success)
            for axis in axis_data:
                key = f"{float(error[axis]):+.6f}"
                axis_data[axis][key]["total"] += 1
                axis_data[axis][key]["success"] += int(cycle_success)
            flat_cycles.append(
                {
                    "sample_id": record["sample_id"],
                    "x_m": error["x_m"],
                    "y_m": error["y_m"],
                    "yaw_deg": error["yaw_deg"],
                    "cycle_id": cycle["cycle_id"],
                    "motion_box_ids": "+".join(map(str, cycle["motion_box_ids"])),
                    "grasp_mode": cycle["grasp_mode"],
                    "cycle_success": cycle_success,
                    **{f"phase_{name.lower()}": phase_outcomes[name] for name in PHASES},
                    "failure_category": category,
                    "failure_stage": failed_stage.get("failure_stage", "") if failed_stage else "",
                    "failure_detail": failed_stage.get("error_detail", "") if failed_stage else "",
                }
            )
        missing_cycles = expected_cycles - len(cycles)
        if missing_cycles > 0:
            for name in STAGES:
                stage_counts[name]["not_attempted"] += missing_cycles
            for name in PHASES:
                phase_counts[name]["not_reached"] += missing_cycles
            if record.get("infrastructure_error"):
                failures["infrastructure"] += missing_cycles
        sample_summaries.append(
            {
                "sample_id": record["sample_id"],
                "error": error,
                "successful_cycles": sample_successes,
                "expected_cycles": expected_cycles,
                "cycle_success_rate": _rate(sample_successes, expected_cycles),
                "complete": len(cycles) == expected_cycles,
            }
        )

    total_expected = len(rows) * expected_cycles
    stage_summary = {}
    for name, counts in stage_counts.items():
        attempted = counts["attempted"]
        stage_summary[name] = {
            **counts,
            "conditional_success_rate": _rate(counts["success"], attempted),
            "all_cycle_success_rate": _rate(counts["success"], total_expected),
        }
    phase_summary = {}
    for name, counts in phase_counts.items():
        reached = counts["success"] + counts["failed"]
        phase_summary[name] = {
            **counts,
            "conditional_success_rate": _rate(counts["success"], reached),
            "all_cycle_success_rate": _rate(counts["success"], total_expected),
        }
    axis_slices = {}
    for axis, values in axis_data.items():
        axis_slices[axis] = {
            key: {
                **counts,
                "success_rate": _rate(counts["success"], counts["total"]),
            }
            for key, counts in sorted(values.items(), key=lambda item: float(item[0]))
        }
    successful_cycles = sum(int(row["cycle_success"]) for row in flat_cycles)
    return {
        "schema": "alfa.v3_docking_error_benchmark_summary.v1",
        "experiment_label": config.get("experiment_label", ""),
        "strategy_label": config.get("strategy_label", ""),
        "design_fingerprint": design_fingerprint(config),
        "model_revision": config.get("model_revision"),
        "tool0_offset_local_z_m": config.get("tool0_offset_local_z_m"),
        "upstream_base_commit": config.get("upstream_base_commit"),
        "cache_loading_counted": False,
        "sample_count": len(rows),
        "expected_cycle_count": total_expected,
        "recorded_cycle_count": len(flat_cycles),
        "successful_cycle_count": successful_cycles,
        "cycle_success_rate": _rate(successful_cycles, total_expected),
        "stage_success": stage_summary,
        "planning_phase_success": phase_summary,
        "failure_categories": dict(failures.most_common()),
        "axis_slices": axis_slices,
        "samples": sample_summaries,
        "cycle_rows": flat_cycles,
    }


def write_summary_bundle(output_dir: Path, summary: dict[str, Any]) -> None:
    write_json(output_dir / "summary.json", summary)
    rows = summary["cycle_rows"]
    with (output_dir / "cycle-results.csv").open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# V3 Docking Error Planning Report",
        "",
        f"- Experiment: `{summary['experiment_label']}`",
        f"- Strategy: `{summary['strategy_label']}`",
        f"- Model: `{summary['model_revision']}`",
        f"- Design fingerprint: `{summary['design_fingerprint']}`",
        f"- Samples: `{summary['sample_count']}`",
        f"- Successful cycles: `{summary['successful_cycle_count']}/{summary['expected_cycle_count']}`",
        f"- Overall cycle success rate: `{(summary['cycle_success_rate'] or 0.0):.2%}`",
        "- Cache loading counted as planning time: `false`",
        "",
        "## Planning phase success",
        "",
        "| Phase | Success | Failed | Not reached | Conditional rate | All-cycle rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase in PHASES:
        item = summary["planning_phase_success"][phase]
        conditional = item["conditional_success_rate"]
        all_cycle = item["all_cycle_success_rate"]
        lines.append(
            f"| {phase} | {item.get('success', 0)} | {item.get('failed', 0)} | "
            f"{item.get('not_reached', 0)} | "
            f"{conditional:.2%} | {all_cycle:.2%} |"
            if conditional is not None and all_cycle is not None
            else f"| {phase} | 0 | 0 | {item.get('not_reached', 0)} | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            "## Failure categories",
            "",
            "| Category | Count |",
            "| --- | ---: |",
        ]
    )
    if summary["failure_categories"]:
        lines.extend(
            f"| {name} | {count} |"
            for name, count in summary["failure_categories"].items()
        )
    else:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Rates apply only to the configured matrix, task cycles, model revision, and fixed planning seed.",
            "A stage's conditional rate uses attempted stages as its denominator; the all-cycle rate keeps every configured cycle in the denominator.",
            "Navigation, localization quality, suction hardware, and execution control are outside this benchmark.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
