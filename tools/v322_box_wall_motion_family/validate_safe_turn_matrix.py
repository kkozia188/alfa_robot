#!/usr/bin/env python3
"""Rebuild and MoveIt/FCL-validate the safe-turn route for all supported pairs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from build_safe_turn_unload_replay import read_json, transform_replay, write_json
from open_pair_rerun import ros_environment, selected_paths


def validation_failure(validation: dict[str, Any]) -> str:
    operations = validation.get("operations", [])
    if operations:
        operation = operations[-1]
        reason = str(operation.get("failure_reason", "unknown"))
        frame = operation.get("failure_frame")
        return reason + (f" at frame {frame}" if frame is not None else "")
    return str(validation.get("failure_reason", "validator produced no operation result"))


def validate_partition(
    worker_index: int,
    entries: list[dict[str, Any]],
    matrix_root: Path,
    output_root: Path,
    base_environment: dict[str, str],
    base_speed_m_s: float,
    rebuild: bool,
) -> list[dict[str, Any]]:
    environment = dict(base_environment)
    environment["ROS_DOMAIN_ID"] = str(180 + worker_index)
    results: list[dict[str, Any]] = []
    for entry in entries:
        left, right = [int(value) for value in entry["pair"]]
        pair_name = f"pair-{left:02d}-{right:02d}"
        destination = output_root / pair_name
        destination.mkdir(parents=True, exist_ok=True)
        replay_path = destination / "replay.json"
        validation_path = destination / "validation.json"
        if rebuild:
            validation_path.unlink(missing_ok=True)
        if rebuild or not replay_path.is_file():
            source_replay, _ = selected_paths(matrix_root, entry)
            replay = transform_replay(
                read_json(source_replay),
                turn_direction="auto",
                base_speed_m_s=base_speed_m_s,
                maximum_yaw_step_deg=1.0,
            )
            write_json(replay_path, replay)

        started = time.monotonic()
        output = ""
        returncode = 0
        if not validation_path.is_file():
            completed = subprocess.run(
                [
                    "ros2",
                    "launch",
                    "alfa_robot_moveit_config",
                    "v3_dual_arm_5x5_replay_validator.launch.py",
                    f"replay_json_path:={replay_path}",
                    f"result_json_path:={validation_path}",
                    "edge_joint_step_deg:=1.0",
                    "edge_updown_step_m:=0.01",
                ],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=90.0,
                check=False,
            )
            output = completed.stdout
            returncode = completed.returncode
        elapsed_s = time.monotonic() - started
        if validation_path.is_file():
            validation = read_json(validation_path)
            success = bool(validation.get("success"))
            failure = "" if success else validation_failure(validation)
        else:
            validation = {}
            success = False
            tail = " | ".join(output.splitlines()[-3:])
            failure = f"validator_exit_{returncode}: {tail}"
        result = {
            "pair": [left, right],
            "success": success,
            "source_candidate_rank": int(entry["selected"]["candidate_rank"]),
            "checked_frames": int(validation.get("checked_frames", 0)),
            "checked_edge_samples": int(validation.get("checked_edge_samples", 0)),
            "validation_elapsed_s": elapsed_s,
            "replay": str(replay_path),
            "validation": str(validation_path),
            "failure": failure,
        }
        results.append(result)
        print(
            f"PAIR {left:02d}+{right:02d} "
            f"{'PASS' if success else 'FAIL'} frames={result['checked_frames']} "
            f"edges={result['checked_edge_samples']} elapsed={elapsed_s:.3f}s",
            flush=True,
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-speed-m-s", type=float, default=0.25)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    if args.base_speed_m_s <= 0.0:
        parser.error("base-speed-m-s must be positive")

    matrix = read_json(args.summary)
    supported = [entry for entry in matrix["results"] if entry.get("success")]
    unsupported = [entry for entry in matrix["results"] if not entry.get("success")]
    partitions = [supported[index :: args.workers] for index in range(args.workers)]
    environment = ros_environment()
    started = time.monotonic()
    all_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                validate_partition,
                worker_index,
                partition,
                args.matrix_root,
                args.output_root,
                environment,
                args.base_speed_m_s,
                args.rebuild,
            )
            for worker_index, partition in enumerate(partitions)
            if partition
        ]
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())

    all_results.sort(key=lambda item: item["pair"])
    failures = [item for item in all_results if not item["success"]]
    report = {
        "schema": "alfa.v322_safe_turn_pair_matrix.v1",
        "model_revision": "robot_v3.2.2-suction",
        "tool0_offset_local_z_m": 0.151,
        "source_matrix": str(args.summary),
        "route_contract": {
            "backoff_before_turn_m": 2.35,
            "turn_angle_deg": 180.0,
            "turn_direction_strategy": "clockwise_for_positive_initial_yaw_else_counterclockwise",
            "lateral_shuttle_m": 0.0,
            "base_speed_m_s": args.base_speed_m_s,
            "maximum_yaw_step_deg": 1.0,
        },
        "total_combinations": len(matrix["results"]),
        "supported_input_pairs": len(supported),
        "unsupported_input_pairs": len(unsupported),
        "validated_success_pairs": len(all_results) - len(failures),
        "validated_failure_pairs": len(failures),
        "all_supported_pairs_passed": not failures and len(all_results) == len(supported),
        "total_checked_frames": sum(item["checked_frames"] for item in all_results),
        "total_checked_edge_samples": sum(
            item["checked_edge_samples"] for item in all_results
        ),
        "wall_elapsed_s": time.monotonic() - started,
        "unsupported_pairs": [entry["pair"] for entry in unsupported],
        "failures": failures,
        "results": all_results,
    }
    write_json(args.output_summary, report)
    print(
        f"RESULT pass={report['validated_success_pairs']}/{len(supported)} "
        f"unsupported={len(unsupported)} frames={report['total_checked_frames']} "
        f"edges={report['total_checked_edge_samples']} "
        f"elapsed={report['wall_elapsed_s']:.3f}s summary={args.output_summary}",
        flush=True,
    )
    return 0 if report["all_supported_pairs_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
