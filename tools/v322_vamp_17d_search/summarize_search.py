#!/usr/bin/python3
"""Summarize mandatory endpoint-only contract-17 VAMP searches."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
SEEDS = (169343505, 271828183, 314159265)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=WORKSPACE / "results/v322_vamp_17d_search",
    )
    parser.add_argument(
        "--output", type=Path,
        default=WORKSPACE / (
            "results/v322_vamp_17d_search/"
            "contract17-unbiased-search-benchmark"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    reports = [
        read_json(root / f"seed-{seed}-report.json") for seed in SEEDS
    ]
    direct_validation = read_json(root / "seed-314159265-validation.json")
    safe_validation = read_json(root / "search-production-safe-turn-validation.json")
    safe_contract = read_json(root / "search-production-safe-turn-contract.json")
    safe_summary = read_json(root / "search-production-safe-turn-summary.json")
    rrd = root / "search-production-safe-turn.rrd"
    fullstate = read_json(
        Path(
            "/home/tim/alfa_robot-v322-vamp-fullstate/results/"
            "v322_vamp_fullstate/v322-fullstate-vamp-benchmark.json"
        )
    )
    prior_fast = read_json(
        Path(
            "/home/tim/alfa_robot-v322-vamp-17d-parallel/results/"
            "v322_vamp_17d/contract17-planning-time-optimization.json"
        )
    )

    cycle_labels = [cycle["label"] for cycle in reports[0]["cycles"]]
    failures: list[str] = []
    for run_index, report in enumerate(reports, 1):
        if not report.get("success"):
            failures.append(f"run{run_index} did not complete")
        if report.get("schema") != "alfa.v322_contract17_vamp_search.v1":
            failures.append(f"run{run_index} has the wrong schema")
        if len(report.get("cycles", [])) != 15:
            failures.append(f"run{run_index} does not contain 15 cycles")
        for cycle in report.get("cycles", []):
            required = (
                cycle.get("strategy") == "mandatory_contract17_vamp_search"
                and cycle.get("path_source") == "vamp_rrtc_search_tree"
                and cycle.get("search_executed") is True
                and cycle.get("search_result_required") is True
                and cycle.get("fallback_used") is False
                and cycle.get("golden_intermediate_frames_used_for_search") is False
                and cycle.get("search_input_frame_count") == 2
                and cycle.get("local_rrt_state_dimension") == 17
                and cycle.get("local_rrt_pruned") is False
                and cycle.get("local_rrt_success") is True
                and cycle.get("local_candidate_fcl_valid") is True
                and cycle.get("visibility_checks") == 0
                and len(cycle.get("local_search_bounds", [])) == 17
                and len(cycle.get("local_search_worker_trials", [])) == 4
            )
            if not required:
                failures.append(
                    f"run{run_index} {cycle.get('label')} violates search provenance"
                )
    if not direct_validation.get("success"):
        failures.append("direct replay MoveIt/FCL validation failed")
    if not safe_validation.get("success"):
        failures.append("safe-turn MoveIt/FCL validation failed")
    if not safe_contract.get("success"):
        failures.append("safe-turn/equal-height contract failed")
    if not rrd.is_file():
        failures.append("Rerun recording is missing")
    else:
        verified = subprocess.run(
            ["rerun", "rrd", "verify", str(rrd)],
            check=False, capture_output=True, text=True,
        )
        if verified.returncode != 0:
            failures.append("Rerun verification failed")

    cycle_rows: list[dict[str, Any]] = []
    for label in cycle_labels:
        samples = [
            next(cycle for cycle in report["cycles"] if cycle["label"] == label)
            for report in reports
        ]
        cycle_rows.append({
            "label": label,
            "solve_kind": "tree_expansion" if any(
                int(item["local_search_iterations"]) > 0 for item in samples
            ) else "direct_connection",
            "planning_core_median_ms": median([
                float(item["planning_core_ms"]) for item in samples
            ]),
            "search_wall_median_ms": median([
                float(item["local_search_planner_wall_ms"]) for item in samples
            ]),
            "iterations_median": median([
                float(item["local_search_iterations"]) for item in samples
            ]),
            "sparse_states_median": median([
                float(item["local_search_candidate_sparse_states"])
                for item in samples
            ]),
            "worker_solved": sum(
                int(item["local_search_solved_workers"]) for item in samples
            ),
            "worker_total": sum(
                int(item["local_search_workers"]) for item in samples
            ),
            "searched_to_reference_cost_ratio_median": median([
                float(item["optimized_joint_travel_cost"])
                / float(item["source_joint_travel_cost"])
                for item in samples
            ]),
        })

    per_run = []
    for report in reports:
        cycles = report["cycles"]
        per_run.append({
            "seed": int(report["vamp_seed"]),
            "cycle_count": len(cycles),
            "planning_core_total_ms": sum(
                float(item["planning_core_ms"]) for item in cycles
            ),
            "maximum_cycle_ms": max(
                float(item["planning_core_ms"]) for item in cycles
            ),
            "search_wall_total_ms": sum(
                float(item["local_search_planner_wall_ms"]) for item in cycles
            ),
            "direct_connection_cycles": sum(
                int(item["local_search_iterations"]) == 0 for item in cycles
            ),
            "tree_expansion_cycles": sum(
                int(item["local_search_iterations"]) > 0 for item in cycles
            ),
            "worker_solved": sum(
                int(item["local_search_solved_workers"]) for item in cycles
            ),
            "worker_total": sum(
                int(item["local_search_workers"]) for item in cycles
            ),
            "worker_full_fcl_valid": sum(
                bool(worker["full_fcl_valid"])
                for item in cycles
                for worker in item["local_search_worker_trials"]
            ),
            "worker_full_fcl_rejected": sum(
                bool(worker["solved"]) and not bool(worker["full_fcl_valid"])
                for item in cycles
                for worker in item["local_search_worker_trials"]
            ),
        })

    report = {
        "schema": "alfa.v322_contract17_unbiased_search_summary.v1",
        "success": not failures,
        "failures": failures,
        "search_contract": {
            "dimension": 17,
            "input": "start state, goal state, explicit margins, limits and scene",
            "intermediate_golden_frames_used": False,
            "visibility_graph_used": False,
            "search_mandatory": True,
            "fallback_allowed": False,
            "postprocess": "edge densification only",
            "authoritative_collision_backend": "MoveIt/FCL",
        },
        "three_seed_acceptance": {
            "runs": len(reports),
            "successful_runs": sum(bool(item["success"]) for item in reports),
            "searched_cycles": sum(item["cycle_count"] for item in per_run),
            "searched_boxes": 25 * sum(bool(item["success"]) for item in reports),
            "fallback_cycles": 0,
            "golden_guided_cycles": 0,
            "planning_core_total_median_ms": median([
                item["planning_core_total_ms"] for item in per_run
            ]),
            "maximum_cycle_median_ms": median([
                item["maximum_cycle_ms"] for item in per_run
            ]),
            "maximum_cycle_observed_ms": max(
                item["maximum_cycle_ms"] for item in per_run
            ),
            "search_wall_total_median_ms": median([
                item["search_wall_total_ms"] for item in per_run
            ]),
            "worker_solved": sum(item["worker_solved"] for item in per_run),
            "worker_total": sum(item["worker_total"] for item in per_run),
            "worker_full_fcl_valid": sum(
                item["worker_full_fcl_valid"] for item in per_run
            ),
            "worker_full_fcl_rejected": sum(
                item["worker_full_fcl_rejected"] for item in per_run
            ),
        },
        "per_run": per_run,
        "cycles": cycle_rows,
        "validation": {
            "direct_frames": int(direct_validation["checked_frames"]),
            "direct_edge_samples": int(direct_validation["checked_edge_samples"]),
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
        },
        "rerun": {
            "path": str(rrd),
            "sha256": sha256(rrd) if rrd.is_file() else None,
            "verified": not any(
                failure == "Rerun verification failed" for failure in failures
            ),
            "model": safe_summary["rerun_model"],
        },
        "reference_comparison": {
            "warning": (
                "The historical OMPL baseline is 15-D while this planner is "
                "contract-17; ratios are indicative, not a same-dimensional proof."
            ),
            "historical_ompl_15d_search_wall_ms": fullstate["median"]
                ["ompl_full15d"]["search_wall_ms"],
            "historical_ompl_15d_planning_core_ms": fullstate["median"]
                ["ompl_full15d"]["planning_core_ms"],
            "indicative_search_speedup": fullstate["median"]["ompl_full15d"]
                ["search_wall_ms"] / median([
                    item["search_wall_total_ms"] for item in per_run
                ]),
            "indicative_end_to_end_speedup": fullstate["median"]["ompl_full15d"]
                ["planning_core_ms"] / median([
                    item["planning_core_total_ms"] for item in per_run
                ]),
            "prior_fast_path_planning_core_ms": prior_fast["overall"]
                ["after_total_planning_core_median_ms"],
            "prior_fast_path_pruned_cycles": prior_fast["overall"]
                ["production_pruned_cycles"],
            "prior_fast_path_worker_total": prior_fast["overall"]
                ["production_worker_total"],
        },
        "comparison_note": (
            "The earlier 0.132s production total is a prior-guided fast-path "
            "result that prunes all VAMP calls. It is not a searched-path baseline."
        ),
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    acceptance = report["three_seed_acceptance"]
    validation = report["validation"]
    lines = [
        "# V3.2.2 mandatory contract-17 VAMP search",
        "",
        "This result requires every cycle to use a path returned by the VAMP "
        "RRT-Connect search tree. Only the two task endpoints, explicit bounds, "
        "robot limits and the collision scene are passed to the search. Golden "
        "intermediate frames, visibility families and fallback paths are disabled.",
        "",
        "## Acceptance",
        "",
        f"- Three independent seeds: **{acceptance['successful_runs']}/3 passed**.",
        f"- Mandatory searches: **{acceptance['searched_cycles']}/45 cycles**, "
        f"**{acceptance['searched_boxes']}/75 boxes**.",
        "- Golden-guided/fallback cycles: **0/45**.",
        f"- Median 15-cycle planning core: "
        f"**{acceptance['planning_core_total_median_ms'] / 1000.0:.3f}s**.",
        f"- Median/observed maximum cycle: "
        f"**{acceptance['maximum_cycle_median_ms']:.1f}ms / "
        f"{acceptance['maximum_cycle_observed_ms']:.1f}ms**.",
        f"- VAMP workers solved: **{acceptance['worker_solved']}/"
        f"{acceptance['worker_total']}**.",
        f"- Dense-FCL worker candidates: **"
        f"{acceptance['worker_full_fcl_valid']} passed**, **"
        f"{acceptance['worker_full_fcl_rejected']} rejected**.",
        f"- Direct replay: **{validation['direct_frames']:,} frames + "
        f"{validation['direct_edge_samples']:,} edges** passed MoveIt/FCL.",
        f"- Safe-turn replay: **{validation['safe_turn_frames']:,} frames + "
        f"{validation['safe_turn_edge_samples']:,} edges** passed MoveIt/FCL.",
        "",
        "## Per-cycle medians",
        "",
        "| Cycle | Solve | Planning | Search | Iterations | Sparse states | Workers | Cost/reference |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cycle_rows:
        lines.append(
            f"| {row['label']} | {row['solve_kind']} | "
            f"{row['planning_core_median_ms']:.1f}ms | "
            f"{row['search_wall_median_ms']:.1f}ms | "
            f"{row['iterations_median']:.0f} | "
            f"{row['sparse_states_median']:.0f} | "
            f"{row['worker_solved']}/{row['worker_total']} | "
            f"{row['searched_to_reference_cost_ratio_median']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Twelve cycles are collision-free direct start-to-goal connections, so "
        "RRT-Connect correctly returns a two-state path without random expansion. "
        "Boxes 3, 16+20 and 21+25 require actual tree expansion in every run. "
        "A zero-iteration direct connection is still a planner result: the planner "
        "validates the complete edge rather than loading a stored route.",
        "",
        "The earlier 0.132s total must remain labeled as the prior-guided fast-path "
        "version. The mandatory-search median is slower but has the required path "
        "provenance and no hidden fallback.",
        "",
        "The historical OMPL result is a 15-D single-worker reference, so the "
        f"{report['reference_comparison']['indicative_search_speedup']:.2f}x search "
        "and "
        f"{report['reference_comparison']['indicative_end_to_end_speedup']:.2f}x "
        "end-to-end ratios are indicative rather than a same-dimensional proof.",
    ])
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_files = [
        *(root / f"seed-{seed}-report.json" for seed in SEEDS),
        *(root / f"seed-{seed}-validation.json" for seed in SEEDS),
        root / "search-production-safe-turn-replay.json",
        root / "search-production-safe-turn-validation.json",
        root / "search-production-safe-turn-contract.json",
        root / "search-production-safe-turn-summary.json",
        root / "search-production-safe-turn-metrics.csv",
        root / "search-production-safe-turn-box-planning.csv",
        rrd,
        output.with_suffix(".json"),
        output.with_suffix(".md"),
    ]
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in manifest_files),
        encoding="utf-8",
    )
    print(
        f"RESULT success={report['success']} searched="
        f"{acceptance['searched_cycles']}/45 max_observed="
        f"{acceptance['maximum_cycle_observed_ms']:.3f}ms"
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
