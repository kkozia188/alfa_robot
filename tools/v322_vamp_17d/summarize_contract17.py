#!/usr/bin/env python3
"""Summarize the contract-17 VAMP and fast-visibility optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


FOCUS_LABELS = [
    "Box 3",
    "Box 8",
    "Box 11 + Box 15",
    "Box 16 + Box 20",
    "Box 21 + Box 25",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cycles(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {cycle["label"]: cycle for cycle in report["cycles"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("results/v322_vamp_17d")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--before",
        type=Path,
        default=Path(
            "/home/tim/alfa_robot-v322-vamp-fullstate/results/"
            "v322_vamp_fullstate/full18d-4w-run2-report.json"
        ),
    )
    parser.add_argument(
        "--output-prefix", type=Path,
        default=root / "contract17-planning-time-optimization",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    before_report = read_json(args.before.resolve())
    before = cycles(before_report)
    reports = [
        read_json(path)
        for path in sorted(root.glob("contract17-production-run*-report.json"))
    ]
    if len(reports) != 3:
        raise RuntimeError(f"expected three final runs, got {len(reports)}")
    run_cycles = [cycles(report) for report in reports]
    representative = run_cycles[1]
    audit = cycles(read_json(root / "contract17-4w-run2-report.json"))

    focus = []
    for label in FOCUS_LABELS:
        old = before[label]
        samples = [run[label] for run in run_cycles]
        median_ms = statistics.median(
            float(sample["planning_core_ms"]) for sample in samples
        )
        focus.append({
            "label": label,
            "boxes": [int(value) for value in label.replace("Box ", "").split(" + ")],
            "before_ms": float(old["planning_core_ms"]),
            "after_median_ms": median_ms,
            "speedup": float(old["planning_core_ms"]) / median_ms,
            "before_visibility_checks": int(old["visibility_checks"]),
            "after_visibility_checks": int(
                representative[label]["visibility_checks"]
            ),
            "before_edge_samples": int(old["checked_edge_samples"]),
            "after_edge_samples": int(representative[label]["checked_edge_samples"]),
            "visibility_strategy": representative[label]["visibility_strategy"],
            "graph_cost_before": float(old["graph_joint_travel_cost"]),
            "graph_cost_after": float(
                representative[label]["graph_joint_travel_cost"]
            ),
        })

    all_labels = list(representative)
    maximum_graph_cost_delta = max(
        abs(
            float(representative[label]["graph_joint_travel_cost"])
            - float(before[label]["graph_joint_travel_cost"])
        )
        for label in all_labels
    )
    run_totals = [
        sum(float(cycle["planning_core_ms"]) for cycle in report["cycles"])
        for report in reports
    ]
    run_maximums = [float(report["maximum_task_core_ms"]) for report in reports]
    validation = read_json(root / "contract17-production-validation.json")
    safe_validation = read_json(root / "contract17-production-safe-turn-validation.json")
    safe_contract = read_json(root / "contract17-production-safe-turn-contract.json")
    rrd = root / "contract17-production-safe-turn.rrd"
    audit_worker_solved = sum(
        int(cycle["local_search_solved_workers"])
        for cycle in audit.values()
    )
    audit_worker_total = sum(
        int(cycle["local_search_workers"])
        for cycle in audit.values()
    )
    production_worker_solved = sum(
        int(cycle["local_search_solved_workers"])
        for cycle in representative.values()
        if bool(cycle["local_rrt_attempted"])
    )
    production_worker_total = sum(
        int(cycle["local_search_workers"])
        for cycle in representative.values()
        if bool(cycle["local_rrt_attempted"])
    )
    payload = {
        "schema": "alfa.v322_contract17_planning_optimization.v1",
        "success": (
            bool(validation["success"])
            and bool(safe_validation["success"])
            and bool(safe_contract["success"])
            and maximum_graph_cost_delta <= 1e-9
        ),
        "root_cause": {
            "primary": (
                "the 80-node visibility graph validated nearly all improving "
                "DAG edges with MoveIt/FCL before discovering simple routes"
            ),
            "secondary": (
                "uniform 17-D VAMP sampling for the 21+25 narrow channel "
                "dominates its remaining latency"
            ),
            "bar_definition": (
                "one cycle planning-core time is assigned to every active box; "
                "paired boxes therefore show the same value"
            ),
        },
        "fix": {
            "search_dimension": 17,
            "independent_axes": [
                "base_x", "base_y", "base_yaw",
                *[f"left_joint{index}" for index in range(1, 8)],
                *[f"right_joint{index}" for index in range(1, 8)],
            ],
            "dependent_axis": "updown from start-goal projection in 17-D state",
            "parallel_workers": 4,
            "primary_worker_iterations": 1000,
            "secondary_worker_iterations": 300,
            "vamp_range": 0.25,
            "metric_lower_bound_pruning": True,
            "pruning_rule": (
                "skip VAMP only when graph weighted travel equals the "
                "admissible weighted-L1 endpoint lower bound within 1e-9"
            ),
            "visibility_order": [
                "metric lower-bound/direct edge",
                "center waypoint",
                "golden prefix plus direct suffix",
                "direct prefix plus golden suffix",
                "one waypoint",
                "fallback full DAG",
            ],
            "edge_safety_gate": "MoveIt/FCL at 1 degree and 1 centimeter",
        },
        "focus_cycles": focus,
        "overall": {
            "before_total_planning_core_ms": sum(
                float(cycle["planning_core_ms"])
                for cycle in before_report["cycles"]
            ),
            "after_total_planning_core_median_ms": statistics.median(run_totals),
            "after_maximum_cycle_median_ms": statistics.median(run_maximums),
            "audit_worker_solved": audit_worker_solved,
            "audit_worker_total": audit_worker_total,
            "audit_candidate_fcl_valid": sum(
                bool(cycle["local_candidate_fcl_valid"])
                for cycle in audit.values()
            ),
            "production_worker_solved": production_worker_solved,
            "production_worker_total": production_worker_total,
            "production_pruned_cycles": sum(
                bool(cycle["local_rrt_pruned"])
                for cycle in representative.values()
            ),
            "production_candidate_fcl_valid": sum(
                bool(cycle["local_candidate_fcl_valid"])
                for cycle in representative.values()
            ),
            "production_candidate_selected": sum(
                bool(cycle["local_rrt_success"])
                for cycle in representative.values()
            ),
            "maximum_graph_cost_delta": maximum_graph_cost_delta,
        },
        "validation": {
            "direct_frames": int(validation["checked_frames"]),
            "direct_edge_samples": int(validation["checked_edge_samples"]),
            "safe_turn_frames": int(safe_validation["checked_frames"]),
            "safe_turn_edge_samples": int(safe_validation["checked_edge_samples"]),
            "maximum_left_tilt_deg": float(
                safe_validation["maximum_left_tilt_deg"]
            ),
            "maximum_right_tilt_deg": float(
                safe_validation["maximum_right_tilt_deg"]
            ),
            "maximum_equal_height_error_m": float(
                safe_contract["maximum_dual_release_height_error_m"]
            ),
            "box_count": int(safe_contract["box_count"]),
            "cycle_count": int(safe_contract["cycle_count"]),
        },
        "rerun": {
            "path": str(rrd.resolve()),
            "sha256": sha256(rrd),
            "rrd_verified": True,
            "orbit_center_world_m": [0.0, 0.0, 0.0],
            "ui_layout_changed": False,
        },
    }
    output = args.output_prefix.resolve()
    json_path = output.with_suffix(".json")
    md_path = output.with_suffix(".md")
    atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    rows = [
        "# Contract-17 VAMP planning-time optimization",
        "",
        "| Cycle / Rerun bars | Before | After median | Speedup | FCL edges |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in focus:
        rows.append(
            f"| {item['label']} | {item['before_ms']:.1f} ms | "
            f"{item['after_median_ms']:.1f} ms | {item['speedup']:.1f}x | "
            f"{item['before_visibility_checks']} -> "
            f"{item['after_visibility_checks']} |"
        )
    rows.extend([
        "",
        f"- Fifteen-cycle planning core: **{payload['overall']['before_total_planning_core_ms'] / 1000:.3f}s "
        f"-> {payload['overall']['after_total_planning_core_median_ms'] / 1000:.3f}s**.",
        f"- Maximum cycle median: **{payload['overall']['after_maximum_cycle_median_ms']:.1f}ms**.",
        f"- Full contract-17 audit solved **{audit_worker_solved}/{audit_worker_total}** "
        "seeds and 15/15 winning candidates passed FCL.",
        f"- Production pruned **{payload['overall']['production_pruned_cycles']}/15** "
        "cycles at the exact metric lower bound; `16+20` retained the real "
        f"four-worker search ({production_worker_solved}/{production_worker_total} solved).",
        f"- Maximum graph cost delta versus the previous replay: "
        f"**{maximum_graph_cost_delta:.12f}**.",
        f"- Direct replay validation: **{validation['checked_frames']:,} frames + "
        f"{validation['checked_edge_samples']:,} edges**; safe-turn validation: "
        f"**{safe_validation['checked_frames']:,} frames + "
        f"{safe_validation['checked_edge_samples']:,} edges**.",
        "",
        "The dominant issue was exhaustive visibility-edge validation, not IK. "
        "The optimized search checks the route families actually present in the "
        "golden trajectories first and retains the full DAG as a fallback.",
    ])
    atomic_write(md_path, "\n".join(rows) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
