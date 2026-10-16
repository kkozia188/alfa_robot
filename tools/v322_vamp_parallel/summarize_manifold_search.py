#!/usr/bin/env python3
"""Compare the old full-space local RRT with the 2-D golden manifold search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OLD_REPORT = Path(
    "/home/tim/alfa_robot-v322-chest-carry-fast-base/results/"
    "v322_chest_carry_fast_base/v322-chest-carry-optimization-report.json"
)
DEFAULT_OLD_REPLAY = Path(
    "/home/tim/alfa_robot-v322-chest-carry-fast-base/results/"
    "v322_chest_carry_fast_base/v322-chest-carry-fast-base-optimized-replay.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def frame_hash(operation: dict[str, Any]) -> str:
    encoded = json.dumps(
        operation.get("frames", []), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate(report: dict[str, Any]) -> dict[str, Any]:
    cycles = report["cycles"]
    local_search = [
        float(item["local_rrt_wall_ms"])
        for item in cycles
        if "local_rrt_wall_ms" in item
    ]
    return {
        "cycle_count": len(cycles),
        "boxes_within_3s": int(report["boxes_within_3s"]),
        "total_planning_core_ms": sum(float(item["planning_core_ms"]) for item in cycles),
        "maximum_planning_core_ms": max(
            float(item["planning_core_ms"]) for item in cycles
        ),
        "exact_candidate_cycles": sum(
            item.get("local_rrt_joint_travel_cost") is not None for item in cycles
        ),
        "selected_candidate_cycles": sum(
            bool(item.get("local_rrt_success")) for item in cycles
        ),
        "total_local_search_ms": (
            sum(local_search) if len(local_search) == len(cycles) else None
        ),
        "maximum_local_search_ms": (
            max(local_search) if len(local_search) == len(cycles) else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-report", type=Path, default=DEFAULT_OLD_REPORT)
    parser.add_argument("--old-replay", type=Path, default=DEFAULT_OLD_REPLAY)
    parser.add_argument(
        "--new-report",
        type=Path,
        default=Path(
            "results/v322_vamp_parallel/manifold/"
            "v322-golden-lattice-32x9-report.json"
        ),
    )
    parser.add_argument(
        "--new-replay",
        type=Path,
        default=Path(
            "results/v322_vamp_parallel/manifold/"
            "v322-golden-lattice-32x9-replay.json"
        ),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path(
            "results/v322_vamp_parallel/manifold/"
            "v322-golden-lattice-32x9-validation.json"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "results/v322_vamp_parallel/manifold/"
            "v322-golden-manifold-comparison"
        ),
    )
    args = parser.parse_args()

    old_report = read_json(args.old_report.resolve())
    new_report = read_json(args.new_report.resolve())
    old_replay = read_json(args.old_replay.resolve())
    new_replay = read_json(args.new_replay.resolve())
    validation = read_json(args.validation.resolve())
    old = aggregate(old_report)
    new = aggregate(new_report)
    old_hashes = [frame_hash(item) for item in old_replay["operations"]]
    new_hashes = [frame_hash(item) for item in new_replay["operations"]]
    matching_hashes = sum(left == right for left, right in zip(old_hashes, new_hashes))
    payload = {
        "schema": "alfa.v322_golden_manifold_comparison.v1",
        "success": (
            bool(new_report["success"])
            and bool(validation["success"])
            and matching_hashes == len(old_hashes) == len(new_hashes)
        ),
        "old_full_space_rrt": {
            **old,
            "state_dimension": 14,
            "mode": "full dual-arm local RRT",
        },
        "new_golden_lattice": {
            **new,
            "state_dimension": 2,
            "mode": "32 x 9 progress/alpha lattice",
        },
        "improvement": {
            "total_planning_core_speedup": (
                old["total_planning_core_ms"] / new["total_planning_core_ms"]
            ),
            "maximum_planning_core_speedup": (
                old["maximum_planning_core_ms"] / new["maximum_planning_core_ms"]
            ),
            "exact_candidate_gain": (
                new["exact_candidate_cycles"] - old["exact_candidate_cycles"]
            ),
        },
        "trajectory_identity": {
            "matching_operation_frame_hashes": matching_hashes,
            "operation_count": len(old_hashes),
        },
        "moveit_fcl_validation": {
            "success": bool(validation["success"]),
            "checked_frames": int(validation["checked_frames"]),
            "checked_edge_samples": int(validation["checked_edge_samples"]),
            "maximum_left_tilt_deg": float(validation["maximum_left_tilt_deg"]),
            "maximum_right_tilt_deg": float(validation["maximum_right_tilt_deg"]),
        },
        "decision": {
            "selected_path": "80-node visibility graph",
            "fallback": "2-D golden lattice",
            "vamp_backend_enabled": False,
            "reason": (
                "The lattice makes the search batchable and deterministic, but stock VAMP "
                "still lacks a validated V3.2.2 sphere collision kernel."
            ),
        },
    }
    output_json = args.output_prefix.with_suffix(".json").resolve()
    output_md = args.output_prefix.with_suffix(".md").resolve()
    atomic_write(output_json, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Golden-manifold search comparison",
        "",
        f"- State dimension: **14 -> 2**.",
        f"- Exact candidates: **{old['exact_candidate_cycles']}/15 -> "
        f"{new['exact_candidate_cycles']}/15**.",
        f"- Total planning core: **{old['total_planning_core_ms'] / 1000.0:.3f}s -> "
        f"{new['total_planning_core_ms'] / 1000.0:.3f}s** "
        f"(**{payload['improvement']['total_planning_core_speedup']:.2f}x**).",
        f"- Maximum cycle core: **{old['maximum_planning_core_ms'] / 1000.0:.3f}s -> "
        f"{new['maximum_planning_core_ms'] / 1000.0:.3f}s**.",
        f"- Final operation frame hashes: **{matching_hashes}/{len(old_hashes)} identical**.",
        f"- MoveIt/FCL: **{validation['checked_frames']:,} frames + "
        f"{validation['checked_edge_samples']:,} edge samples passed**.",
        "",
        "The visibility graph remains the selected path because all lattice candidates "
        "have slightly greater joint travel. The two-dimensional lattice is retained as "
        "the deterministic fallback and as the future VAMP batch-validation boundary. "
        "No VAMP speedup is claimed until the V3 sphere kernel is validated.",
    ]
    atomic_write(output_md, "\n".join(lines) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
