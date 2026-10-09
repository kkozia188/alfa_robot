#!/usr/bin/env python3
"""Aggregate probe and complete-case results into JSON, CSV, and Markdown."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from matrix_common import is_single_axis_error, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, action="append", required=True,
        help="repeat for probe and full result roots",
    )
    parser.add_argument("--imported-baselines", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    row_map: dict[tuple[str, str], dict[str, Any]] = {}
    for root_arg in args.results_root:
        root = root_arg.resolve()
        for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            full = case_dir / "case-result.json"
            probe = case_dir / "probe-result.json"
            path = full if full.is_file() else probe
            if not path.is_file():
                continue
            result = read_json(path)
            error = result.get("error", {})
            planning = result.get("planning", {})
            validation = result.get("validation", {})
            trajectory = result.get("trajectory", {})
            findings = list(result.get("quality_findings", []))
            row = {
                "case_id": result["case_id"],
                "scope": "full" if path == full else "screen",
                "dx_m": float(error.get("dx_m", 0.0)),
                "dy_m": float(error.get("dy_m", 0.0)),
                "yaw_deg": float(error.get("yaw_deg", 0.0)),
                "status": str(result.get("status", "unknown")),
                "failure_class": str(
                    result.get("failure_class", "")
                    or (findings[0] if findings else "")
                ),
                "failure_stage": str(result.get("failure_stage", "")),
                "failure_reason": str(result.get("failure_reason", "")),
                "completed_boxes": int(result.get("completed_boxes", 0)),
                "maximum_task_core_ms": float(
                    planning.get(
                        "task_core_max_ms", result.get("maximum_task_core_ms", 0.0)
                    )
                ),
                "maximum_bridge_ms": float(planning.get("bridge_core_max_ms", 0.0)),
                "checked_frames": int(validation.get("checked_frames", 0)),
                "checked_edge_samples": int(validation.get("checked_edge_samples", 0)),
                "maximum_tilt_deg": max(
                    float(validation.get("maximum_left_tilt_deg", 0.0)),
                    float(validation.get("maximum_right_tilt_deg", 0.0)),
                ),
                "maximum_joint_step_deg": float(
                    trajectory.get("maximum_joint_step_deg", 0.0)
                ),
                "joint_flip_events": int(trajectory.get("joint_flip_events", 0)),
            }
            if is_single_axis_error(row["dx_m"], row["dy_m"], row["yaw_deg"]):
                row_map[(row["case_id"], row["scope"])] = row
    if args.imported_baselines:
        imported = read_json(args.imported_baselines.resolve())
        for item in imported["rows"]:
            row = {
                "case_id": item["case_id"],
                "scope": "imported_" + str(item["source"]),
                "dx_m": float(item["dx_m"]),
                "dy_m": float(item["dy_m"]),
                "yaw_deg": float(item["yaw_deg"]),
                "status": str(item["status"]),
                "failure_class": "planning_time_over_target" if item["status"] == "degraded" else "",
                "failure_stage": "task_core" if item["status"] == "degraded" else "",
                "failure_reason": ";".join(item["quality_findings"]),
                "completed_boxes": int(item["completed_boxes"]),
                "maximum_task_core_ms": float(item["task_core_max_ms"] or 0.0),
                "maximum_bridge_ms": 0.0,
                "checked_frames": int(item["checked_frames"]),
                "checked_edge_samples": int(item["checked_edge_samples"]),
                "maximum_tilt_deg": max(
                    float(item["maximum_left_tilt_deg"]),
                    float(item["maximum_right_tilt_deg"]),
                ),
                "maximum_joint_step_deg": 0.0,
                "joint_flip_events": 0,
            }
            if is_single_axis_error(row["dx_m"], row["dy_m"], row["yaw_deg"]):
                row_map[(row["case_id"], row["scope"])] = row
    rows = sorted(
        row_map.values(), key=lambda row: (
            row["scope"], row["yaw_deg"], row["dy_m"], row["dx_m"]
        )
    )
    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in sorted({row["status"] for row in rows})
    }
    failure_counts = {
        failure: sum(row["failure_class"] == failure for row in rows)
        for failure in sorted({row["failure_class"] for row in rows if row["failure_class"]})
    }
    aggregate = {
        "schema": "alfa.v322_docking_error_matrix_summary.v1",
        "result_count": len(rows),
        "screen_result_count": sum(row["scope"] == "screen" for row in rows),
        "full_result_count": sum(row["scope"] == "full" for row in rows),
        "imported_result_count": sum(row["scope"].startswith("imported_") for row in rows),
        "status_counts": status_counts,
        "failure_class_counts": failure_counts,
        "checked_frames": sum(row["checked_frames"] for row in rows),
        "checked_edge_samples": sum(row["checked_edge_samples"] for row in rows),
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "matrix-summary.json", aggregate)
    if rows:
        with (output / "matrix-results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# V3.2.2 Docking Error Matrix",
        "",
        "## Coverage",
        "",
        f"- Recorded results: `{len(rows)}`",
        f"- Screen / full: `{aggregate['screen_result_count']} / {aggregate['full_result_count']}`",
        f"- Status counts: `{status_counts}`",
        f"- Failure classes: `{failure_counts}`",
        f"- MoveIt/FCL frames / edge samples: `{aggregate['checked_frames']:,} / {aggregate['checked_edge_samples']:,}`",
        f"- Carried-box tilt failures observed: `{failure_counts.get('carried_box_tilt', 0)}`",
        "",
        "## Cases",
        "",
        "| Case | Scope | dx | dy | yaw | Status | Failure | Task max | Tilt |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | {row['scope']} | {row['dx_m']:+.3f} m | "
            f"{row['dy_m']:+.3f} m | {row['yaw_deg']:+.1f} deg | "
            f"{row['status']} | {row['failure_class'] or '-'} | "
            f"{row['maximum_task_core_ms'] / 1000.0:.3f} s | "
            f"{row['maximum_tilt_deg']:.3f} deg |"
        )
    lines.extend([
        "",
        "Screen results cover selected high-risk pickup groups only. A screen pass is not a "
        "complete-task certificate; only full cases include all 25 boxes and MoveIt/FCL replay validation.",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"SUMMARY results={len(rows)} statuses={status_counts} failures={failure_counts}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
