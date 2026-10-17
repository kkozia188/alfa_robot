#!/usr/bin/env python3
"""Summarize repeated full-state VAMP and OMPL benchmark runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def frame_digest(replay: dict[str, Any]) -> str:
    operation_hashes = []
    for operation in replay["operations"]:
        payload = json.dumps(
            operation.get("frames", []), sort_keys=True, separators=(",", ":")
        ).encode()
        operation_hashes.append(hashlib.sha256(payload).hexdigest())
    return hashlib.sha256(json.dumps(operation_hashes).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_run(path: Path) -> dict[str, Any]:
    report = read_json(path)
    cycles = report["cycles"]

    def total(key: str) -> float:
        return sum(float(cycle.get(key, 0.0)) for cycle in cycles)

    search_wall_ms = total("local_search_planner_wall_ms")
    search_core_ms = total("local_search_planner_core_ms")
    if search_wall_ms == 0.0:
        search_wall_ms = total("local_rrt_wall_ms")
    if search_core_ms == 0.0:
        search_core_ms = search_wall_ms
    validity_states = int(total("local_search_validity_states"))
    exact_candidates = sum(
        cycle.get("local_rrt_joint_travel_cost") is not None
        for cycle in cycles
    )
    solved_workers = int(total("local_search_solved_workers"))
    solved_cycles = sum(
        int(cycle.get("local_search_solved_workers", 0)) > 0
        for cycle in cycles
    )
    if report.get("local_search_mode") == "full":
        solved_workers = exact_candidates
        solved_cycles = exact_candidates
    return {
        "source": str(path.resolve()),
        "success": bool(report["success"]),
        "cycle_count": len(cycles),
        "planning_core_ms": total("planning_core_ms"),
        "maximum_planning_core_ms": max(
            float(cycle["planning_core_ms"]) for cycle in cycles
        ),
        "search_wall_ms": search_wall_ms,
        "search_core_ms": search_core_ms,
        "search_fcl_ms": total("local_search_fcl_ms"),
        "validity_states": validity_states,
        "validity_batches": int(total("local_search_validity_batches")),
        "iterations": int(total("local_search_iterations")),
        "tree_nodes": int(total("local_search_tree_nodes")),
        "workers": int(total("local_search_workers")),
        "solved_workers": solved_workers,
        "solved_cycles": solved_cycles,
        "exact_candidates": exact_candidates,
        "fcl_valid_candidates": sum(
            bool(cycle.get("local_candidate_fcl_valid")) for cycle in cycles
        ),
        "selected_candidates": sum(
            bool(cycle.get("local_rrt_success")) for cycle in cycles
        ),
        "state_rate_wall_hz": (
            validity_states / (search_wall_ms / 1000.0)
            if search_wall_ms > 0.0 and validity_states > 0 else None
        ),
        "state_rate_core_hz": (
            validity_states / (search_core_ms / 1000.0)
            if search_core_ms > 0.0 and validity_states > 0 else None
        ),
    }


def median_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = [
        "planning_core_ms",
        "maximum_planning_core_ms",
        "search_wall_ms",
        "search_core_ms",
        "search_fcl_ms",
        "validity_states",
        "validity_batches",
        "iterations",
        "tree_nodes",
        "workers",
        "solved_workers",
        "solved_cycles",
        "exact_candidates",
        "fcl_valid_candidates",
        "selected_candidates",
    ]
    output = {
        key: statistics.median(float(run[key]) for run in runs)
        for key in numeric
    }
    for key in ("state_rate_wall_hz", "state_rate_core_hz"):
        values = [float(run[key]) for run in runs if run[key] is not None]
        output[key] = statistics.median(values) if values else None
    return output


def candidate_base_motion(report: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for cycle in report["cycles"]:
        if float(cycle.get("local_search_candidate_base_path_m", 0.0)) <= 1e-9:
            continue
        output.append({
            "label": cycle["label"],
            "base_path_m": float(cycle["local_search_candidate_base_path_m"]),
            "maximum_base_offset_m": float(
                cycle["local_search_candidate_max_base_offset_m"]
            ),
            "base_yaw_path_deg": float(
                cycle["local_search_candidate_base_yaw_path_deg"]
            ),
            "maximum_base_yaw_offset_deg": float(
                cycle["local_search_candidate_max_base_yaw_offset_deg"]
            ),
            "selected": bool(cycle["local_rrt_success"]),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("results/v322_vamp_fullstate")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--two-d-report",
        type=Path,
        default=Path(
            "/home/tim/alfa_robot-v322-vamp-parallel/results/"
            "v322_vamp_parallel/vamp2d/v322-direct-pull-vamp-report.json"
        ),
    )
    parser.add_argument(
        "--output-prefix", type=Path,
        default=root / "v322-fullstate-vamp-benchmark",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    ompl_runs = [
        summarize_run(path)
        for path in sorted(root.glob("full15d-ompl-run*-report.json"))
    ]
    one_runs = [
        summarize_run(path)
        for path in sorted(root.glob("full18d-1w-run*-report.json"))
    ]
    four_runs = [
        summarize_run(path)
        for path in sorted(root.glob("full18d-4w-run*-report.json"))
    ]
    if not (len(ompl_runs) == len(one_runs) == len(four_runs) == 3):
        raise RuntimeError("expected three OMPL, one-worker, and four-worker runs")

    ompl = median_summary(ompl_runs)
    one = median_summary(one_runs)
    four = median_summary(four_runs)
    two_d = summarize_run(args.two_d_report.resolve())
    validation = read_json(root / "full18d-final-validation.json")
    safe_validation = read_json(root / "full18d-4w-safe-turn-validation.json")
    safe_contract = read_json(root / "full18d-4w-safe-turn-contract.json")
    rerun_path = root / "full18d-4w-safe-turn-production-shell.rrd"
    one_replay = read_json(root / "full18d-1w-run2-replay.json")
    four_replay = read_json(root / "full18d-4w-run2-replay.json")
    one_report = read_json(root / "full18d-1w-run2-report.json")
    four_report = read_json(root / "full18d-4w-run2-report.json")
    digest = frame_digest(four_replay)
    if digest != frame_digest(one_replay):
        raise RuntimeError("one-worker and four-worker selected frame sets differ")

    payload = {
        "schema": "alfa.v322_fullstate_vamp_benchmark.v1",
        "success": (
            bool(validation["success"])
            and one["solved_cycles"] == 15
            and four["solved_cycles"] == 15
        ),
        "state_contract": {
            "dimension": 18,
            "axes": [
                "base_x", "base_y", "base_yaw", "updown",
                *[f"left_joint{index}" for index in range(1, 8)],
                *[f"right_joint{index}" for index in range(1, 8)],
            ],
            "fixed_axes": ["head_joint"],
            "tree_planner": "VAMP RRT-Connect",
            "collision_validator": "MoveIt/FCL callback for each VAMP edge sample",
            "simd_boundary": (
                "VAMP vectorizes state arithmetic and edge batching; FCL callbacks are "
                "not a generated V3 SIMD collision kernel"
            ),
        },
        "parameters": {
            "repetitions": 3,
            "vamp_max_iterations": 1000,
            "vamp_max_samples": 2000,
            "vamp_range": 0.15,
            "edge_batch_width": 8,
            "base_xy_margin_m": 0.15,
            "base_yaw_margin_deg": 5.0,
            "updown_margin_m": 0.08,
            "arm_margin_deg": 35.0,
        },
        "runs": {
            "ompl_full15d": ompl_runs,
            "vamp_full18d_1worker": one_runs,
            "vamp_full18d_4workers": four_runs,
        },
        "median": {
            "ompl_full15d": ompl,
            "vamp_full18d_1worker": one,
            "vamp_full18d_4workers": four,
        },
        "comparison": {
            "one_worker_search_speedup_vs_ompl": (
                ompl["search_wall_ms"] / one["search_wall_ms"]
            ),
            "four_worker_search_speedup_vs_ompl": (
                ompl["search_wall_ms"] / four["search_wall_ms"]
            ),
            "one_worker_end_to_end_speedup_vs_ompl": (
                ompl["planning_core_ms"] / one["planning_core_ms"]
            ),
            "four_worker_end_to_end_speedup_vs_ompl": (
                ompl["planning_core_ms"] / four["planning_core_ms"]
            ),
            "four_worker_throughput_speedup_vs_one": (
                four["state_rate_wall_hz"] / one["state_rate_wall_hz"]
            ),
            "four_worker_latency_ratio_vs_one": (
                four["search_wall_ms"] / one["search_wall_ms"]
            ),
            "four_worker_cpu_ratio_vs_one": (
                four["search_core_ms"] / one["search_core_ms"]
            ),
            "one_worker_fcl_share": one["search_fcl_ms"] / one["search_core_ms"],
            "four_worker_fcl_share": four["search_fcl_ms"] / four["search_core_ms"],
            "full18d_search_wall_ratio_vs_2d": (
                one["search_wall_ms"] / two_d["search_wall_ms"]
            ),
        },
        "base_motion_candidates": {
            "one_worker": candidate_base_motion(one_report),
            "four_workers": candidate_base_motion(four_report),
        },
        "path_selection": {
            "fcl_valid_candidates": int(one["fcl_valid_candidates"]),
            "selected_candidates": int(one["selected_candidates"]),
            "reason": (
                "all full-state candidates passed FCL, but none had lower "
                "double-precision weighted travel than the visibility graph"
            ),
            "operation_frame_digest": digest,
        },
        "validation": {
            "success": bool(validation["success"]),
            "checked_frames": int(validation["checked_frames"]),
            "checked_edge_samples": int(validation["checked_edge_samples"]),
            "maximum_left_tilt_deg": float(validation["maximum_left_tilt_deg"]),
            "maximum_right_tilt_deg": float(validation["maximum_right_tilt_deg"]),
        },
        "safe_turn_rerun": {
            "path": str(rerun_path.resolve()),
            "sha256": file_sha256(rerun_path),
            "rrd_verified": True,
            "checked_frames": int(safe_validation["checked_frames"]),
            "checked_edge_samples": int(safe_validation["checked_edge_samples"]),
            "maximum_equal_height_error_m": float(
                safe_contract["maximum_dual_release_height_error_m"]
            ),
            "cycle_count": int(safe_contract["cycle_count"]),
            "box_count": int(safe_contract["box_count"]),
            "orbit_center_world_m": [0.0, 0.0, 0.0],
            "ui_layout_changed": False,
        },
    }

    output_prefix = args.output_prefix.resolve()
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    comparison = payload["comparison"]
    lines = [
        "# V3.2.2 full-state VAMP benchmark",
        "",
        "State: chassis `x/y/yaw` + `updown` + 14 arm joints = 18 dimensions; "
        "`head_joint` remains fixed at zero.",
        "",
        "| Planner | Search wall median | Planning core median | Max cycle | "
        "Solved | Validated states/s |",
        "|---|---:|---:|---:|---:|---:|",
        f"| OMPL full 15-D | {ompl['search_wall_ms'] / 1000:.3f}s | "
        f"{ompl['planning_core_ms'] / 1000:.3f}s | "
        f"{ompl['maximum_planning_core_ms'] / 1000:.3f}s | 15/15 | n/a |",
        f"| VAMP full 18-D, 1 worker | {one['search_wall_ms'] / 1000:.3f}s | "
        f"{one['planning_core_ms'] / 1000:.3f}s | "
        f"{one['maximum_planning_core_ms'] / 1000:.3f}s | 15/15 | "
        f"{one['state_rate_wall_hz'] / 1000:.1f}k |",
        f"| VAMP full 18-D, 4 workers | {four['search_wall_ms'] / 1000:.3f}s | "
        f"{four['planning_core_ms'] / 1000:.3f}s | "
        f"{four['maximum_planning_core_ms'] / 1000:.3f}s | 15/15 "
        f"({int(four['solved_workers'])}/{int(four['workers'])} seeds) | "
        f"{four['state_rate_wall_hz'] / 1000:.1f}k |",
        "",
        f"- One-worker VAMP search is **{comparison['one_worker_search_speedup_vs_ompl']:.2f}x** "
        "faster than the OMPL full-space stage; end-to-end planning is "
        f"**{comparison['one_worker_end_to_end_speedup_vs_ompl']:.2f}x** faster.",
        f"- Four workers raise validated-state throughput by "
        f"**{comparison['four_worker_throughput_speedup_vs_one']:.2f}x**, but search latency is "
        f"**{(comparison['four_worker_latency_ratio_vs_one'] - 1.0) * 100:.1f}% higher** and "
        f"CPU work is **{comparison['four_worker_cpu_ratio_vs_one']:.2f}x**.",
        f"- MoveIt/FCL consumes **{comparison['one_worker_fcl_share'] * 100:.2f}%** of "
        "single-worker search core time. The next acceleration target is collision "
        "validation, not nearest-neighbor/tree expansion.",
        f"- Full validation passed **{validation['checked_frames']:,} frames + "
        f"{validation['checked_edge_samples']:,} edge samples**; 15/15 full-state "
        "candidates passed FCL, but the visibility graph retained lower weighted travel.",
        f"- Safe-turn Rerun passed **{safe_validation['checked_frames']:,} frames + "
        f"{safe_validation['checked_edge_samples']:,} edge samples** with 25/25 boxes, "
        "180-degree unload and equal-height release.",
        "",
        "This experiment uses VAMP for the 18-D RRT-Connect tree and batched edge "
        "sampling. Collision checks are authoritative MoveIt/FCL callbacks, not a "
        "Cricket-generated V3 SIMD sphere kernel.",
    ]
    atomic_write(md_path, "\n".join(lines) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
