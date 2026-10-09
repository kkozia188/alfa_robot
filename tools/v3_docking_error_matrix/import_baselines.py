#!/usr/bin/env python3
"""Import certified MOTION-261 X and MOTION-257 Yaw baselines."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from matrix_common import case_slug, clearance_slug, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-certificate", type=Path, required=True)
    parser.add_argument("--yaw-certificate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    x_certificate = read_json(args.x_certificate.resolve())
    yaw_certificate = read_json(args.yaw_certificate.resolve())
    rows: list[dict[str, Any]] = []
    pairs = {
        (
            round(float(item["upper_front_clearance_m"]), 2),
            round(float(item["lower_front_clearance_m"]), 2),
        ): item
        for item in x_certificate["pairs"]
    }
    for family, centimeters in (
        ("upper_rows_1_to_3", range(80, 91)),
        ("lower_rows_4_to_5", range(55, 66)),
    ):
        for centimeter in centimeters:
            clearance_m = centimeter / 100.0
            upper = clearance_m if family == "upper_rows_1_to_3" else 0.85
            lower = clearance_m if family == "lower_rows_4_to_5" else 0.60
            pair = pairs[(upper, lower)]
            rows.append({
                "case_id": clearance_slug(upper, lower),
                "source": f"MOTION-261_X_{family}",
                "x_family": family,
                "front_clearance_m": clearance_m,
                "upper_front_clearance_m": upper,
                "lower_front_clearance_m": lower,
                "dx_m": 0.0,
                "dy_m": 0.0,
                "yaw_deg": 0.0,
                "status": "passed",
                "completed_boxes": 25,
                "checked_frames": int(pair["checked_frames"]),
                "checked_edge_samples": int(pair["checked_edge_samples"]),
                "maximum_left_tilt_deg": float(pair["maximum_left_tilt_deg"]),
                "maximum_right_tilt_deg": float(pair["maximum_right_tilt_deg"]),
                "task_core_max_ms": None,
                "quality_findings": [],
                "evidence": str(args.x_certificate.resolve()),
            })
    for item in yaw_certificate["cases"]:
        yaw_deg = float(item["yaw_deg"])
        task_max = float(item["task_core_max_ms"])
        findings = ["task_core_over_target"] if task_max >= 3000.0 else []
        rows.append({
            "case_id": case_slug(0.0, 0.0, yaw_deg),
            "source": "MOTION-257_Yaw",
            "x_family": "",
            "front_clearance_m": "",
            "upper_front_clearance_m": "",
            "lower_front_clearance_m": "",
            "dx_m": 0.0,
            "dy_m": 0.0,
            "yaw_deg": yaw_deg,
            "status": "degraded" if findings else "passed",
            "completed_boxes": int(item["completed_boxes"]),
            "checked_frames": int(item["checked_frames"]),
            "checked_edge_samples": int(item["checked_edge_samples"]),
            "maximum_left_tilt_deg": float(item["maximum_left_tilt_deg"]),
            "maximum_right_tilt_deg": float(item["maximum_right_tilt_deg"]),
            "task_core_max_ms": task_max,
            "quality_findings": findings,
            "evidence": str(args.yaw_certificate.resolve()),
        })
    # Nominal appears in both X families and Yaw; family identity remains explicit.
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "alfa.v322_docking_error_imported_baselines.v1",
        "x_case_count": 22,
        "yaw_case_count": 11,
        "successful_rows": sum(row["status"] in {"passed", "degraded"} for row in rows),
        "degraded_rows": sum(row["status"] == "degraded" for row in rows),
        "rows": rows,
    }
    write_json(output / "imported-baselines.json", payload)
    with (output / "imported-baselines.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "quality_findings": ";".join(row["quality_findings"]),
            })
    if not math.isclose(
        max(float(row["yaw_deg"]) for row in rows), 5.0, abs_tol=1e-9
    ):
        raise RuntimeError("Yaw baseline does not include +5 deg")
    print(
        f"IMPORTED x=22 yaw=11 degraded={payload['degraded_rows']} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
