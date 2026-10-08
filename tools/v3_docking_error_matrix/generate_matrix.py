#!/usr/bin/env python3
"""Generate the staged X/Y/Yaw experiment matrix as JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Any

from matrix_common import SCHEMA, case_slug, read_json, write_json


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
    x = spec["certified_x_baseline"]
    for dx_m in x["dx_values_m"]:
        add_case(
            cases, dx_m, x["dy_m"], x["yaw_deg"],
            "certified_x_baseline", "import", x["source"],
        )
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
    joint = spec["full_joint_matrix"]
    for dx_m, dy_m, yaw_deg in itertools.product(
        joint["dx_values_m"], joint["dy_values_m"], joint["yaw_values_deg"]
    ):
        add_case(cases, dx_m, dy_m, yaw_deg, "full_joint_matrix", "full")
    boundary = spec["boundary_probe"]
    for dx_m in boundary["x_values_m"]:
        add_case(cases, dx_m, 0.0, 0.0, "boundary_probe_x", "screen")
    for dy_m in boundary["y_values_m"]:
        add_case(cases, 0.0, dy_m, 0.0, "boundary_probe_y", "screen")
    for yaw_deg in boundary["yaw_values_deg"]:
        add_case(cases, 0.0, 0.0, yaw_deg, "boundary_probe_yaw", "screen")
    return sorted(cases.values(), key=lambda item: (
        float(item["yaw_deg"]), float(item["dy_m"]), float(item["dx_m"])
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
    output_dir = args.output_dir.resolve()
    payload = {
        "schema": SCHEMA,
        "specification": str(args.spec.resolve()),
        "thresholds": spec["thresholds"],
        "boundary_probe_group_indices": spec["boundary_probe"]["group_indices"],
        "case_count": len(cases),
        "counts_by_mode": {
            mode: sum(case["recommended_mode"] == mode for case in cases)
            for mode in ("import", "screen", "screen_then_full", "full")
        },
        "cases": cases,
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
    print(
        f"MATRIX cases={len(cases)} modes={payload['counts_by_mode']} "
        f"output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
