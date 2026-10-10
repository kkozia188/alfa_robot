#!/usr/bin/env python3
"""Find pass/fail brackets on single-axis probe results and propose bisection cases."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from matrix_common import case_slug, read_json, write_json


def axis_value(result: dict[str, Any], axis: str) -> float:
    key = {"x": "dx_m", "y": "dy_m", "yaw": "yaw_deg"}[axis]
    return float(result["error"][key])


def on_axis(result: dict[str, Any], axis: str) -> bool:
    error = result["error"]
    if axis == "x":
        return abs(float(error["dy_m"])) < 1e-12 and abs(float(error["yaw_deg"])) < 1e-12
    if axis == "y":
        return abs(float(error["dx_m"])) < 1e-12 and abs(float(error["yaw_deg"])) < 1e-12
    return abs(float(error["dx_m"])) < 1e-12 and abs(float(error["dy_m"])) < 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = [
        read_json(path) for path in args.results_root.resolve().glob("*/probe-result.json")
    ]
    output: dict[str, Any] = {
        "schema": "alfa.v322_docking_error_boundaries.v1",
        "axes": {},
        "recommended_cases": [],
    }
    for axis in ("x", "y", "yaw"):
        values = sorted(
            (axis_value(item, axis), item) for item in results if on_axis(item, axis)
        )
        brackets = []
        for (left_value, left), (right_value, right) in zip(values, values[1:]):
            left_pass = left["status"] in {"screen_passed", "screen_degraded"}
            right_pass = right["status"] in {"screen_passed", "screen_degraded"}
            if left_pass == right_pass:
                continue
            midpoint = 0.5 * (left_value + right_value)
            case = {"dx_m": 0.0, "dy_m": 0.0, "yaw_deg": 0.0}
            case[{"x": "dx_m", "y": "dy_m", "yaw": "yaw_deg"}[axis]] = midpoint
            case["case_id"] = case_slug(
                case["dx_m"], case["dy_m"], case["yaw_deg"]
            )
            brackets.append({
                "pass_value": left_value if left_pass else right_value,
                "fail_value": right_value if left_pass else left_value,
                "midpoint": midpoint,
                "failure_class": (
                    right.get("failure_class") if left_pass else left.get("failure_class")
                ),
            })
            output["recommended_cases"].append(case)
        output["axes"][axis] = {
            "observed_values": [value for value, _ in values],
            "brackets": brackets,
        }
    write_json(args.output.resolve(), output)
    print(
        f"BOUNDARIES brackets={sum(len(v['brackets']) for v in output['axes'].values())} "
        f"recommended={len(output['recommended_cases'])} output={args.output.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
