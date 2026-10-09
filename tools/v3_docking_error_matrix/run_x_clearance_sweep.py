#!/usr/bin/env python3
"""Run one X row-family clearance sweep while holding the other at default."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import subprocess
from pathlib import Path
from typing import Any

from matrix_common import (
    DEFAULT_LOWER_FRONT_CLEARANCE_M,
    DEFAULT_UPPER_FRONT_CLEARANCE_M,
    clearance_slug,
    quantize_front_clearance,
    read_json,
    write_json,
)


ROOT = Path(__file__).resolve().parent
CASE_RUNNER = ROOT / "run_pose_case.py"


def parse_values(raw: str) -> list[float]:
    try:
        values = [
            quantize_front_clearance(float(value))
            for value in raw.split(",") if value.strip()
        ]
    except ValueError as error:
        raise argparse.ArgumentTypeError("values must be comma-separated meters") from error
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            "values must remain non-empty and unique after 1 cm rounding"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("upper", "lower"), required=True)
    parser.add_argument("--values", type=parse_values, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ros-domain-base", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case-timeout-s", type=float, default=2400.0)
    return parser.parse_args()


def run_case(
    family: str,
    value: float,
    args: argparse.Namespace,
    domain_id: int,
) -> dict[str, Any]:
    upper = value if family == "upper" else DEFAULT_UPPER_FRONT_CLEARANCE_M
    lower = value if family == "lower" else DEFAULT_LOWER_FRONT_CLEARANCE_M
    case_id = clearance_slug(upper, lower)
    result_path = args.output_root.resolve() / case_id / "case-result.json"
    if result_path.is_file() and args.resume:
        return read_json(result_path)
    command = [
        "/usr/bin/python3", str(CASE_RUNNER),
        "--base-dx-m", "0", "--base-dy-m", "0", "--yaw-deg", "0",
        "--upper-front-clearance-m", f"{upper:.6f}",
        "--lower-front-clearance-m", f"{lower:.6f}",
        "--baseline-cache", str(args.baseline_cache.resolve()),
        "--output-root", str(args.output_root.resolve()),
        "--pickup-attempt-timeout-s", "60",
        "--bridge-attempt-timeout-s", "180",
        "--stage-wall-timeout-s", f"{args.case_timeout_s:.1f}",
    ]
    if args.resume:
        command.append("--resume")
    environment = dict(os.environ)
    environment["ROS_DOMAIN_ID"] = str(domain_id)
    completed = subprocess.run(command, check=False, env=environment)
    if result_path.is_file():
        return read_json(result_path)
    return {
        "case_id": case_id,
        "front_clearance": {
            "upper_rows_1_to_3_m": upper,
            "lower_rows_4_to_5_m": lower,
        },
        "status": "orchestrator_failure",
        "failure_class": "missing_result",
        "return_code": completed.returncode,
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be in [1, 4]")
    if args.ros_domain_base < 0 or args.ros_domain_base + args.workers - 1 > 232:
        raise SystemExit("ROS domain lanes must remain in [0, 232]")
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)

    def run_lane(lane: int) -> list[dict[str, Any]]:
        return [
            run_case(args.family, value, args, args.ros_domain_base + lane)
            for value in args.values[lane::args.workers]
        ]

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_lane, lane) for lane in range(args.workers)]
        for future in concurrent.futures.as_completed(futures):
            for result in future.result():
                results.append(result)
                print(
                    f"X_CLEARANCE case={result['case_id']} status={result.get('status')} "
                    f"failure={result.get('failure_class', '') or 'none'}",
                    flush=True,
                )
    results.sort(key=lambda result: float(
        result["front_clearance"][
            "upper_rows_1_to_3_m" if args.family == "upper"
            else "lower_rows_4_to_5_m"
        ]
    ))
    payload = {
        "schema": "alfa.v322_x_clearance_sweep.v1",
        "family": args.family,
        "fixed_clearance_m": (
            DEFAULT_LOWER_FRONT_CLEARANCE_M if args.family == "upper"
            else DEFAULT_UPPER_FRONT_CLEARANCE_M
        ),
        "case_count": len(results),
        "status_counts": {
            status: sum(result.get("status") == status for result in results)
            for status in sorted({str(result.get("status")) for result in results})
        },
        "results": results,
    }
    prefix = f"x-{args.family}-clearance-sweep"
    write_json(args.output_root.resolve() / f"{prefix}.json", payload)
    rows = []
    for result in results:
        clearance = result["front_clearance"]
        planning = result.get("planning", {})
        rows.append({
            "case_id": result["case_id"],
            "upper_clearance_m": clearance["upper_rows_1_to_3_m"],
            "lower_clearance_m": clearance["lower_rows_4_to_5_m"],
            "status": result.get("status", ""),
            "failure_class": result.get("failure_class", ""),
            "failure_stage": result.get("failure_stage", ""),
            "failure_reason": result.get("failure_reason", ""),
            "task_core_max_ms": planning.get("task_core_max_ms", ""),
            "bridge_core_max_ms": planning.get("bridge_core_max_ms", ""),
        })
    with (args.output_root.resolve() / f"{prefix}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"X_CLEARANCE_DONE family={args.family} cases={len(results)} "
        f"statuses={payload['status_counts']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
