#!/usr/bin/env python3
"""Execute selected independent Y/Yaw matrix cases without stopping on failures."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from matrix_common import is_single_axis_error, read_json, write_json


ROOT = Path(__file__).resolve().parent
PROBE = ROOT / "probe_pose_case.py"
FULL = ROOT / "run_pose_case.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--execution", choices=("screen", "full"), required=True)
    parser.add_argument(
        "--phases",
        default="",
        help="comma-separated phase names; empty selects every non-import case",
    )
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ros-domain-base", type=int, default=210)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def select_cases(matrix: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    phases = {value for value in args.phases.split(",") if value}
    ids = {value for value in args.case_ids.split(",") if value}
    selected = []
    for case in matrix["cases"]:
        if abs(float(case["dx_m"])) > 1e-12:
            raise ValueError(
                "official Y/Yaw matrix contains common base dx; use X clearance matrix"
            )
        if not is_single_axis_error(
            float(case["dx_m"]), float(case["dy_m"]), float(case["yaw_deg"])
        ):
            raise ValueError(f"official matrix contains a combined-error case: {case}")
        if ids and case["case_id"] not in ids:
            continue
        if phases and not phases.intersection(case["phases"]):
            continue
        if not ids and not phases and case["recommended_mode"] == "import":
            continue
        selected.append(case)
    return selected


def run_case(
    case: dict[str, Any], args: argparse.Namespace, domain_id: int
) -> dict[str, Any]:
    case_dir = args.output_root.resolve() / case["case_id"]
    result_name = "probe-result.json" if args.execution == "screen" else "case-result.json"
    result_path = case_dir / result_name
    if result_path.is_file() and not args.force:
        result = read_json(result_path)
        return {"case_id": case["case_id"], "resumed": True, **result}
    runner = PROBE if args.execution == "screen" else FULL
    command = [
        "/usr/bin/python3", str(runner),
        "--base-dx-m", str(case["dx_m"]),
        "--base-dy-m", str(case["dy_m"]),
        "--yaw-deg", str(case["yaw_deg"]),
        "--baseline-cache", str(args.baseline_cache.resolve()),
        "--output-root", str(args.output_root.resolve()),
    ]
    if args.resume:
        command.append("--resume")
    environment = dict(os.environ)
    environment["ROS_DOMAIN_ID"] = str(domain_id)
    started = time.monotonic()
    completed = subprocess.run(command, check=False, env=environment)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if result_path.is_file():
        result = read_json(result_path)
    else:
        result = {
            "status": "orchestrator_failure",
            "failure_class": "missing_result",
            "return_code": completed.returncode,
        }
    return {
        "case_id": case["case_id"],
        "dx_m": case["dx_m"],
        "dy_m": case["dy_m"],
        "yaw_deg": case["yaw_deg"],
        "execution": args.execution,
        "orchestrator_wall_ms": elapsed_ms,
        "resumed": False,
        **result,
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("--workers must be in [1, 8]")
    if args.ros_domain_base < 0 or args.ros_domain_base + args.workers - 1 > 232:
        raise SystemExit("ROS domain lanes must remain in [0, 232]")
    matrix = read_json(args.matrix.resolve())
    cases = select_cases(matrix, args)
    if not cases:
        raise SystemExit("selection contains no cases")
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    def run_lane(lane: int) -> list[dict[str, Any]]:
        return [
            run_case(case, args, args.ros_domain_base + lane)
            for case in cases[lane::args.workers]
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_lane, lane) for lane in range(args.workers)]
        for future in concurrent.futures.as_completed(futures):
            for result in future.result():
                results.append(result)
                print(
                    f"MATRIX_RESULT case={result['case_id']} "
                    f"status={result.get('status')} "
                    f"failure={result.get('failure_class', '') or 'none'}",
                    flush=True,
                )
    results.sort(key=lambda item: item["case_id"])
    summary = {
        "schema": "alfa.v322_docking_error_matrix_run.v1",
        "execution": args.execution,
        "matrix": str(args.matrix.resolve()),
        "case_count": len(results),
        "status_counts": {
            status: sum(item.get("status") == status for item in results)
            for status in sorted({str(item.get("status")) for item in results})
        },
        "failure_class_counts": {
            name: sum(item.get("failure_class") == name for item in results)
            for name in sorted({
                str(item.get("failure_class")) for item in results
                if item.get("failure_class")
            })
        },
        "results": results,
    }
    prefix = f"{args.execution}-matrix-run"
    write_json(args.output_root.resolve() / f"{prefix}.json", summary)
    with (args.output_root.resolve() / f"{prefix}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "case_id", "dx_m", "dy_m", "yaw_deg", "status",
            "failure_class", "failure_stage", "failure_reason",
            "maximum_task_core_ms", "groups_completed", "orchestrator_wall_ms",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(
        f"MATRIX_DONE execution={args.execution} cases={len(results)} "
        f"statuses={summary['status_counts']} failures={summary['failure_class_counts']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
