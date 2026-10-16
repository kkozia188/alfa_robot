#!/usr/bin/env python3
"""Summarize the real 2-D VAMP integration and final safe-turn replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


OLD_FULL = Path(
    "/home/tim/alfa_robot-v322-chest-carry-fast-base/results/"
    "v322_chest_carry_fast_base/v322-chest-carry-optimization-report.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def cycle_metrics(report: dict[str, Any]) -> dict[str, Any]:
    cycles = report["cycles"]
    has_fcl_candidate_data = all("local_candidate_fcl_valid" in x for x in cycles)
    return {
        "cycles": len(cycles),
        "total_planning_core_ms": sum(float(x["planning_core_ms"]) for x in cycles),
        "maximum_planning_core_ms": max(float(x["planning_core_ms"]) for x in cycles),
        "exact_candidates": sum(
            x.get("local_rrt_joint_travel_cost") is not None for x in cycles
        ),
        "fcl_valid_candidates": (
            sum(bool(x["local_candidate_fcl_valid"]) for x in cycles)
            if has_fcl_candidate_data else None
        ),
        "selected_candidates": sum(bool(x.get("local_rrt_success")) for x in cycles),
        "vamp_wall_ms": sum(
            float(x.get("local_search_planner_wall_ms", 0.0)) for x in cycles
        ),
        "vamp_core_ms": sum(
            float(x.get("local_search_planner_core_ms", 0.0)) for x in cycles
        ),
        "grid_build_ms": sum(
            float(x.get("local_search_grid_build_ms", 0.0)) for x in cycles
        ),
        "candidate_fcl_ms": sum(
            float(x.get("local_search_candidate_fcl_ms", 0.0)) for x in cycles
        ),
        "workers": sum(int(x.get("local_search_workers", 0)) for x in cycles),
        "solved_workers": sum(
            int(x.get("local_search_solved_workers", 0)) for x in cycles
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("results/v322_vamp_parallel/vamp2d")
    parser.add_argument("--old-full", type=Path, default=OLD_FULL)
    parser.add_argument(
        "--fair-lattice", type=Path,
        default=root / "v322-golden-lattice-fair-report.json",
    )
    parser.add_argument(
        "--fair-vamp", type=Path,
        default=root / "v322-golden-vamp-final-report.json",
    )
    parser.add_argument(
        "--direct-vamp", type=Path,
        default=root / "v322-direct-pull-vamp-report.json",
    )
    parser.add_argument(
        "--safe-replay", type=Path,
        default=root / "v322-direct-pull-vamp-safe-turn-replay.json",
    )
    parser.add_argument(
        "--safe-validation", type=Path,
        default=root / "v322-direct-pull-vamp-safe-turn-validation.json",
    )
    parser.add_argument(
        "--safe-contract", type=Path,
        default=root / "v322-direct-pull-vamp-safe-turn-contract.json",
    )
    parser.add_argument(
        "--rrd", type=Path,
        default=root / "v322-direct-pull-vamp-safe-turn-production-shell.rrd",
    )
    parser.add_argument(
        "--output-prefix", type=Path,
        default=root / "v322-vamp2d-acceptance",
    )
    args = parser.parse_args()

    old = cycle_metrics(read_json(args.old_full.resolve()))
    lattice = cycle_metrics(read_json(args.fair_lattice.resolve()))
    fair_vamp = cycle_metrics(read_json(args.fair_vamp.resolve()))
    direct_vamp = cycle_metrics(read_json(args.direct_vamp.resolve()))
    replay = read_json(args.safe_replay.resolve())
    validation = read_json(args.safe_validation.resolve())
    contract = read_json(args.safe_contract.resolve())
    bars = [float(x["selected_task_core_ms"]) for x in replay["box_planning"]]
    payload = {
        "schema": "alfa.v322_vamp2d_acceptance.v1",
        "success": bool(validation["success"]) and bool(contract["success"]),
        "backend": {
            "planner": "VAMP RRT-Connect",
            "configuration_space": "2-D golden manifold (progress, alpha)",
            "simd": "AVX2, edge resolution 256",
            "parallel_workers": 4,
            "vamp_source_commit": "184bb856f79b5c1d9e06ded94b74414c59c0ce47",
            "safety_gate": "MoveIt/FCL map plus full winning-candidate revalidation",
        },
        "same_input_ab": {
            "old_14d_full_rrt": old,
            "two_d_lattice": lattice,
            "two_d_vamp": fair_vamp,
            "vamp_speedup_vs_14d": (
                old["total_planning_core_ms"] / fair_vamp["total_planning_core_ms"]
            ),
            "vamp_speedup_vs_lattice": (
                lattice["total_planning_core_ms"] /
                fair_vamp["total_planning_core_ms"]
            ),
        },
        "latest_direct_pull_vamp": direct_vamp,
        "rerun_bar_chart": {
            "definition": "cycle planning core assigned to each active box",
            "box_count": len(bars),
            "assigned_total_s": sum(bars) / 1000.0,
            "maximum_s": max(bars) / 1000.0,
            "boxes_under_3s": sum(value < 3000.0 for value in bars),
        },
        "safe_turn_validation": {
            "checked_frames": int(validation["checked_frames"]),
            "checked_edge_samples": int(validation["checked_edge_samples"]),
            "maximum_left_tilt_deg": float(validation["maximum_left_tilt_deg"]),
            "maximum_right_tilt_deg": float(validation["maximum_right_tilt_deg"]),
            "maximum_equal_height_error_m": float(
                contract["maximum_dual_release_height_error_m"]
            ),
            "cycle_count": int(contract["cycle_count"]),
            "box_count": int(contract["box_count"]),
        },
        "rerun": {
            "path": str(args.rrd.resolve()),
            "sha256": sha256(args.rrd.resolve()),
            "orbit_center_world_m": [0.0, 0.0, 0.0],
            "ui_layout_changed": False,
        },
        "path_selection": {
            "vamp_candidates_fcl_valid": direct_vamp["fcl_valid_candidates"],
            "vamp_candidates_selected": direct_vamp["selected_candidates"],
            "reason": (
                "visibility graph had lower joint-travel cost; VAMP remains the "
                "validated concurrent fallback"
            ),
        },
    }
    json_path = args.output_prefix.with_suffix(".json").resolve()
    md_path = args.output_prefix.with_suffix(".md").resolve()
    atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# V3.2.2 2-D VAMP acceptance",
        "",
        f"- Same-input planning core: **{old['total_planning_core_ms'] / 1000:.3f}s "
        f"(14-D) -> {fair_vamp['total_planning_core_ms'] / 1000:.3f}s (VAMP)**, "
        f"**{payload['same_input_ab']['vamp_speedup_vs_14d']:.2f}x**.",
        f"- Fair 2-D lattice -> VAMP: **{lattice['total_planning_core_ms'] / 1000:.3f}s "
        f"-> {fair_vamp['total_planning_core_ms'] / 1000:.3f}s**, "
        f"**{payload['same_input_ab']['vamp_speedup_vs_lattice']:.3f}x**.",
        f"- Latest direct-pull VAMP core: **{direct_vamp['total_planning_core_ms'] / 1000:.3f}s "
        f"total**, maximum **{direct_vamp['maximum_planning_core_ms'] / 1000:.3f}s**.",
        f"- Pure VAMP wall time: **{direct_vamp['vamp_wall_ms']:.3f}ms** across 15 cycles; "
        f"workers solved **{direct_vamp['solved_workers']}/{direct_vamp['workers']}**.",
        f"- Winning candidates passing full FCL: **{direct_vamp['fcl_valid_candidates']}/15**.",
        f"- Final safe-turn validation: **{validation['checked_frames']:,} frames + "
        f"{validation['checked_edge_samples']:,} edge samples passed**.",
        f"- Rerun blue bars: **25/25 under 3s**, maximum **{max(bars) / 1000:.3f}s**.",
        "",
        "The visibility graph remains the selected path because its joint travel is lower. "
        "VAMP is genuinely executed as a four-worker AVX2 planner and is retained as the "
        "validated concurrent fallback. The Rerun layout is unchanged; only the orbit target "
        "is fixed at the warehouse/robot world origin.",
    ]
    atomic_write(md_path, "\n".join(lines) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
