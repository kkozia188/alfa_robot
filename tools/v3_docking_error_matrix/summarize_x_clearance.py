#!/usr/bin/env python3
"""Merge certified seed data and 1 cm X-clearance extension results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from matrix_common import (
    DEFAULT_LOWER_FRONT_CLEARANCE_M,
    DEFAULT_UPPER_FRONT_CLEARANCE_M,
    clearance_slug,
    read_json,
    write_json,
)


FAMILY_FIELD = {
    "upper": "upper_rows_1_to_3_m",
    "lower": "lower_rows_4_to_5_m",
}
SEED_CENTIMETERS = {
    "upper": range(80, 91),
    "lower": range(55, 66),
}
NOMINAL = {
    "upper": DEFAULT_UPPER_FRONT_CLEARANCE_M,
    "lower": DEFAULT_LOWER_FRONT_CLEARANCE_M,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper-root", type=Path, required=True)
    parser.add_argument("--lower-root", type=Path, required=True)
    parser.add_argument("--seed-certificate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def imported_seed_rows(certificate_path: Path) -> list[dict[str, Any]]:
    certificate = read_json(certificate_path)
    pairs = {
        (
            round(float(item["upper_front_clearance_m"]), 2),
            round(float(item["lower_front_clearance_m"]), 2),
        ): item
        for item in certificate["pairs"]
    }
    rows: list[dict[str, Any]] = []
    for family in ("upper", "lower"):
        for centimeters in SEED_CENTIMETERS[family]:
            clearance = centimeters / 100.0
            upper = clearance if family == "upper" else NOMINAL["upper"]
            lower = clearance if family == "lower" else NOMINAL["lower"]
            pair = pairs[(upper, lower)]
            rows.append({
                "family": family,
                "clearance_m": clearance,
                "fixed_clearance_m": lower if family == "upper" else upper,
                "case_id": clearance_slug(upper, lower),
                "source": "MOTION-261_v322_121_pair_certificate",
                "scope": "certified_seed",
                "status": "passed",
                "functional_success": True,
                "completed_boxes": 25,
                "failure_class": "",
                "failure_stage": "",
                "failure_reason": "",
                "task_core_max_ms": "",
                "task_core_below_3s": True,
                "bridge_core_max_ms": "",
                "maximum_tilt_deg": max(
                    float(pair["maximum_left_tilt_deg"]),
                    float(pair["maximum_right_tilt_deg"]),
                ),
                "posture_warning": False,
                "checked_frames": int(pair["checked_frames"]),
                "checked_edge_samples": int(pair["checked_edge_samples"]),
                "joint_flip_events": 0,
                "quality_findings": "",
                "evidence": str(certificate_path),
            })
    return rows


def local_rows(family: str, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/case-result.json")):
        result = read_json(path)
        clearance = result["front_clearance"]
        value = float(clearance[FAMILY_FIELD[family]])
        fixed = float(clearance[FAMILY_FIELD["lower" if family == "upper" else "upper"]])
        planning = result.get("planning", {})
        validation = result.get("validation", {})
        trajectory = result.get("trajectory", {})
        findings = list(result.get("quality_findings", []))
        maximum_tilt = max(
            float(validation.get("maximum_left_tilt_deg", 0.0)),
            float(validation.get("maximum_right_tilt_deg", 0.0)),
        )
        functional_success = (
            result.get("status") in {"passed", "degraded"}
            and int(result.get("completed_boxes", 0)) == 25
            and bool(validation.get("success"))
        )
        task_max = planning.get("task_core_max_ms", "")
        rows.append({
            "family": family,
            "clearance_m": value,
            "fixed_clearance_m": fixed,
            "case_id": result["case_id"],
            "source": "MOTION-257_1cm_extension",
            "scope": "full_extension",
            "status": result.get("status", "unknown"),
            "functional_success": functional_success,
            "completed_boxes": int(result.get("completed_boxes", 0)),
            "failure_class": result.get("failure_class", ""),
            "failure_stage": result.get("failure_stage", ""),
            "failure_reason": result.get("failure_reason", ""),
            "task_core_max_ms": task_max,
            "task_core_below_3s": bool(task_max != "" and float(task_max) < 3000.0),
            "bridge_core_max_ms": planning.get("bridge_core_max_ms", ""),
            "maximum_tilt_deg": maximum_tilt,
            "posture_warning": maximum_tilt >= 5.0,
            "checked_frames": int(validation.get("checked_frames", 0)),
            "checked_edge_samples": int(validation.get("checked_edge_samples", 0)),
            "joint_flip_events": int(trajectory.get("joint_flip_events", 0)),
            "quality_findings": ";".join(findings),
            "evidence": str(path),
        })
    return rows


def contiguous_range(rows: list[dict[str, Any]], predicate: Any) -> list[float]:
    by_cm = {int(round(float(row["clearance_m"]) * 100)): row for row in rows}
    nominal_cm = int(round(NOMINAL[str(rows[0]["family"])] * 100))
    if nominal_cm not in by_cm or not predicate(by_cm[nominal_cm]):
        return []
    low = high = nominal_cm
    while low - 1 in by_cm and predicate(by_cm[low - 1]):
        low -= 1
    while high + 1 in by_cm and predicate(by_cm[high + 1]):
        high += 1
    return [low / 100.0, high / 100.0]


def family_summary(family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: float(row["clearance_m"]))
    functional = contiguous_range(rows, lambda row: bool(row["functional_success"]))
    posture = contiguous_range(
        rows,
        lambda row: bool(row["functional_success"])
        and not bool(row["posture_warning"])
        and int(row["joint_flip_events"]) == 0,
    )
    posture_and_task = contiguous_range(
        rows,
        lambda row: bool(row["functional_success"])
        and not bool(row["posture_warning"])
        and bool(row["task_core_below_3s"])
        and int(row["joint_flip_events"]) == 0,
    )
    by_cm = {int(round(float(row["clearance_m"]) * 100)): row for row in rows}
    low_cm = int(round(functional[0] * 100))
    high_cm = int(round(functional[1] * 100))
    adjacent = [by_cm.get(low_cm - 1), by_cm.get(high_cm + 1)]
    if any(row is None or row["functional_success"] for row in adjacent):
        raise RuntimeError(f"{family} functional range lacks adjacent failures")
    seed = [centimeters / 100.0 for centimeters in SEED_CENTIMETERS[family]]
    success_rows = [row for row in rows if row["functional_success"]]
    return {
        "varied_rows": "rows_1_to_3" if family == "upper" else "rows_4_to_5",
        "fixed_other_clearance_m": (
            DEFAULT_LOWER_FRONT_CLEARANCE_M
            if family == "upper" else DEFAULT_UPPER_FRONT_CLEARANCE_M
        ),
        "certified_seed_range_m": [min(seed), max(seed)],
        "complete_functional_range_m": functional,
        "posture_qualified_range_m": posture,
        "posture_and_task_core_below_3s_range_m": posture_and_task,
        "nearest_failed_m": [
            float(adjacent[0]["clearance_m"]),
            float(adjacent[1]["clearance_m"]),
        ],
        "nearest_failure_classes": [
            adjacent[0]["failure_class"], adjacent[1]["failure_class"]
        ],
        "nearest_failure_stages": [
            adjacent[0]["failure_stage"], adjacent[1]["failure_stage"]
        ],
        "nearest_failure_reasons": [
            adjacent[0]["failure_reason"], adjacent[1]["failure_reason"]
        ],
        "sample_count": len(rows),
        "functional_success_count": len(success_rows),
        "failed_count": len(rows) - len(success_rows),
        "checked_frames": sum(int(row["checked_frames"]) for row in rows),
        "checked_edge_samples": sum(
            int(row["checked_edge_samples"]) for row in rows
        ),
        "maximum_success_tilt_deg": max(
            float(row["maximum_tilt_deg"]) for row in success_rows
        ),
        "joint_flip_events": sum(int(row["joint_flip_events"]) for row in rows),
    }


def main() -> int:
    args = parse_args()
    certificate_path = args.seed_certificate.resolve()
    rows = imported_seed_rows(certificate_path)
    rows.extend(local_rows("upper", args.upper_root.resolve()))
    rows.extend(local_rows("lower", args.lower_root.resolve()))
    rows.sort(key=lambda row: (row["family"], float(row["clearance_m"])))
    for family in ("upper", "lower"):
        values = [
            round(float(row["clearance_m"]), 2)
            for row in rows if row["family"] == family
        ]
        if len(values) != len(set(values)):
            raise RuntimeError(f"duplicate {family} clearance evidence")
    summaries = {
        family: family_summary(
            family, [row for row in rows if row["family"] == family]
        )
        for family in ("upper", "lower")
    }
    certificate = read_json(certificate_path)
    payload = {
        "schema": "alfa.v322_x_front_clearance_range.v1",
        "date": "2026-10-09",
        "reference": "vehicle front contact plane to box front contact plane",
        "input_resolution_m": 0.01,
        "rounding": "ROUND_HALF_UP to the nearest 0.01 m",
        "experiment_contract": {
            "upper": "vary rows 1-3 clearance; fix rows 4-5 at 0.60 m; Y=Yaw=0",
            "lower": "vary rows 4-5 clearance; fix rows 1-3 at 0.85 m; Y=Yaw=0",
            "functional_success": "25/25 boxes and complete MoveIt/FCL validation",
            "combined_clearance_changes_included": False,
        },
        "strongest_cross_grid_certificate": {
            "upper_range_m": [0.80, 0.90],
            "lower_range_m": [0.55, 0.65],
            "certified_pairs": int(certificate["certified_pair_count"]),
            "all_selected_task_cores_below_3s": (
                int(certificate["profile_task_core"]["tasks_below_3s"])
                == int(certificate["profile_task_core"]["task_count"])
            ),
            "maximum_selected_task_core_ms": (
                float(certificate["profile_task_core"]["maximum_task_core_s"])
                * 1000.0
            ),
            "evidence": str(certificate_path),
        },
        "ranges": summaries,
        "coverage": {
            "result_rows": len(rows),
            "functional_success_rows": sum(
                bool(row["functional_success"]) for row in rows
            ),
            "failed_rows": sum(not bool(row["functional_success"]) for row in rows),
            "checked_frames": sum(int(row["checked_frames"]) for row in rows),
            "checked_edge_samples": sum(
                int(row["checked_edge_samples"]) for row in rows
            ),
        },
        "rows": rows,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "X_CLEARANCE_RANGE.json", payload)
    with (output / "X_CLEARANCE_RESULTS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    upper = summaries["upper"]
    lower = summaries["lower"]
    lines = [
        "# X Front-Clearance Range",
        "",
        "X is the distance from the vehicle-front contact plane to the box-front contact plane. ",
        "Inputs are rounded half-up to the nearest 1 cm before planning.",
        "",
        "| Sweep | Strong certificate | Independent functional range | Adjacent failures |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Rows 1-3; rows 4-5 fixed at 0.60 m | 0.80-0.90 m | "
            f"{upper['complete_functional_range_m'][0]:.2f}-{upper['complete_functional_range_m'][1]:.2f} m | "
            f"{upper['nearest_failed_m'][0]:.2f}/{upper['nearest_failed_m'][1]:.2f} m |"
        ),
        (
            f"| Rows 4-5; rows 1-3 fixed at 0.85 m | 0.55-0.65 m | "
            f"{lower['complete_functional_range_m'][0]:.2f}-{lower['complete_functional_range_m'][1]:.2f} m | "
            f"{lower['nearest_failed_m'][0]:.2f}/{lower['nearest_failed_m'][1]:.2f} m |"
        ),
        "",
        "## Quality Layers",
        "",
        (
            f"- Rows 1-3 posture-qualified range: `"
            f"{upper['posture_qualified_range_m'][0]:.2f}-{upper['posture_qualified_range_m'][1]:.2f} m`; "
            "0.74-0.76 m completes but exceeds the 5 degree tilt warning."
        ),
        (
            f"- Rows 1-3 posture plus selected-task-core-under-3s range: `"
            f"{upper['posture_and_task_core_below_3s_range_m'][0]:.2f}-"
            f"{upper['posture_and_task_core_below_3s_range_m'][1]:.2f} m`."
        ),
        (
            f"- Rows 4-5 posture-qualified range: `"
            f"{lower['posture_qualified_range_m'][0]:.2f}-{lower['posture_qualified_range_m'][1]:.2f} m`."
        ),
        (
            f"- Rows 4-5 posture plus selected-task-core-under-3s contiguous range around nominal: `"
            f"{lower['posture_and_task_core_below_3s_range_m'][0]:.2f}-"
            f"{lower['posture_and_task_core_below_3s_range_m'][1]:.2f} m`; "
            "0.51 and 0.52 m exceeded 3 seconds."
        ),
        "- Fresh whole-body entry bridges exceeded 3 seconds in extension runs, so the expanded functional ranges are not an all-planning-under-3s certificate.",
        "",
        "## Boundary Failures",
        "",
        "- Rows 1-3 at 0.73 m: `left_link7 <-> warehouse_right_wall` during post-release stow.",
        "- Rows 1-3 at 1.03 m: box 13 has no precontact IK.",
        "- Rows 4-5 at 0.35 m: both link7 bodies collide with the warehouse rear wall at the 17+19 transition endpoint.",
        "- Rows 4-5 at 0.80 m: box 21 top-suction precontact has no IK.",
        "",
        "The expanded ranges are empirical, one-centimeter single-variable ranges. The 121-pair cross-grid remains the strongest guarantee when both row-family inputs may vary together.",
        "",
    ]
    (output / "X_CLEARANCE_RANGE.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"X_RANGE rows={len(rows)} success={payload['coverage']['functional_success_rows']} "
        f"upper={upper['complete_functional_range_m']} "
        f"lower={lower['complete_functional_range_m']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
