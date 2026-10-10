#!/usr/bin/env python3
"""Generate physical X-clearance and independent Y/Yaw experiment matrices."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from matrix_common import SCHEMA, case_slug, clearance_slug, read_json, write_json


def add_case(
    cases: dict[tuple[float, float, float], dict[str, Any]],
    dx_m: float,
    dy_m: float,
    yaw_deg: float,
    phase: str,
    mode: str,
    evidence: str = "",
) -> None:
    key = (round(dx_m, 6), round(dy_m, 6), round(yaw_deg, 6))
    case = cases.setdefault(key, {
        "case_id": case_slug(*key),
        "dx_m": key[0],
        "dy_m": key[1],
        "yaw_deg": key[2],
        "phases": [],
        "recommended_mode": mode,
        "prior_evidence": [],
    })
    if phase not in case["phases"]:
        case["phases"].append(phase)
    precedence = {"import": 0, "screen": 1, "screen_then_full": 2, "full": 3}
    if precedence[mode] > precedence[case["recommended_mode"]]:
        case["recommended_mode"] = mode
    if evidence and evidence not in case["prior_evidence"]:
        case["prior_evidence"].append(evidence)


def build_matrix(spec: dict[str, Any]) -> list[dict[str, Any]]:
    cases: dict[tuple[float, float, float], dict[str, Any]] = {}
    y = spec["y_baseline"]
    for dy_m in y["dy_values_m"]:
        add_case(
            cases, y["dx_m"], dy_m, y["yaw_deg"],
            "y_baseline", "screen_then_full",
        )
    yaw = spec["yaw_baseline"]
    for yaw_deg in yaw["yaw_values_deg"]:
        add_case(
            cases, yaw["dx_m"], yaw["dy_m"], yaw_deg,
            "yaw_baseline", "import", yaw["source"],
        )
    boundary = spec["boundary_probe"]
    for dy_m in boundary["y_values_m"]:
        add_case(cases, 0.0, dy_m, 0.0, "boundary_probe_y", "screen")
    for yaw_deg in boundary["yaw_values_deg"]:
        add_case(cases, 0.0, 0.0, yaw_deg, "boundary_probe_yaw", "screen")
    return sorted(cases.values(), key=lambda item: (
        float(item["yaw_deg"]), float(item["dy_m"]), float(item["dx_m"])
    ))


def build_x_clearance_matrix(spec: dict[str, Any]) -> list[dict[str, Any]]:
    config = spec["x_clearance_sweeps"]
    cases: list[dict[str, Any]] = []
    for family in ("upper_rows_1_to_3", "lower_rows_4_to_5"):
        sweep = config[family]
        values = [
            (value, "certified_seed", "import")
            for value in sweep["certified_values_m"]
        ] + [
            (value, "one_centimeter_extension", "full")
            for value in sweep["extension_values_m"]
        ]
        for value, phase, mode in values:
            upper = (
                float(value) if family == "upper_rows_1_to_3"
                else float(sweep["fixed_upper_clearance_m"])
            )
            lower = (
                float(sweep["fixed_lower_clearance_m"])
                if family == "upper_rows_1_to_3" else float(value)
            )
            cases.append({
                "case_id": clearance_slug(upper, lower),
                "family": family,
                "upper_front_clearance_m": upper,
                "lower_front_clearance_m": lower,
                "phase": phase,
                "recommended_mode": mode,
                "runner": "run_x_clearance_sweep.py",
                "prior_evidence": config["source"] if mode == "import" else "",
            })
    return sorted(cases, key=lambda item: (
        item["family"], float(item["upper_front_clearance_m"]),
        float(item["lower_front_clearance_m"]),
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path,
        default=Path(__file__).resolve().parent / "default_matrix.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = read_json(args.spec.resolve())
    if spec.get("schema") != "alfa.v322_docking_error_matrix_spec.v1":
        raise SystemExit("unexpected matrix specification schema")
    cases = build_matrix(spec)
    x_cases = build_x_clearance_matrix(spec)
    output_dir = args.output_dir.resolve()
    payload = {
        "schema": SCHEMA,
        "specification": str(args.spec.resolve()),
        "thresholds": spec["thresholds"],
        "boundary_probe_group_indices": spec["boundary_probe"]["group_indices"],
        "case_count": len(cases),
        "x_clearance_case_count": len(x_cases),
        "counts_by_mode": {
            mode: sum(case["recommended_mode"] == mode for case in cases)
            for mode in ("import", "screen", "screen_then_full", "full")
        },
        "cases": cases,
        "x_clearance_cases": x_cases,
    }
    write_json(output_dir / "matrix-cases.json", payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "matrix-cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "case_id", "dx_m", "dy_m", "yaw_deg", "phases",
            "recommended_mode", "prior_evidence",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({
                **case,
                "phases": ";".join(case["phases"]),
                "prior_evidence": ";".join(case["prior_evidence"]),
            })
    write_json(output_dir / "x-clearance-cases.json", {
        "schema": "alfa.v322_x_clearance_matrix.v1",
        "reference": spec["x_clearance_sweeps"]["reference"],
        "input_resolution_m": spec["x_clearance_sweeps"]["input_resolution_m"],
        "case_count": len(x_cases),
        "cases": x_cases,
    })
    with (output_dir / "x-clearance-cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "case_id", "family", "upper_front_clearance_m",
            "lower_front_clearance_m", "phase", "recommended_mode",
            "runner", "prior_evidence",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(x_cases)
    print(
        f"MATRIX y_yaw_cases={len(cases)} x_clearance_cases={len(x_cases)} "
        f"modes={payload['counts_by_mode']} "
        f"output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
