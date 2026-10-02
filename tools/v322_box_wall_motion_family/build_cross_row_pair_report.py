#!/usr/bin/env python3
"""Build the acceptance report for cross-row synchronized two-box requests."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from build_acceptance_report import replay_metrics


SCHEMA = "alfa.v322_cross_row_pair_acceptance.v1"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-plan", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    pair_plan = read_json(args.pair_plan)
    benchmark = read_json(args.benchmark)
    replay = read_json(args.replay)
    validation = read_json(args.validation)
    geometry = {item["request_id"]: item for item in pair_plan["requests"]}
    failures: collections.Counter[str] = collections.Counter()
    results = []
    for item in benchmark["results"]:
        selected = item.get("selected")
        for attempt in item["attempts"]:
            for field in (
                "left_failure", "right_failure", "bridge_failure", "composition_failure"
            ):
                value = attempt.get(field)
                if value:
                    failures[str(value)] += 1
        targets = geometry[item["request_id"]]["targets"]
        results.append({
            "request_id": item["request_id"],
            "box_ids": item["box_ids"],
            "rows_from_top": [target["row_from_top"] for target in targets],
            "columns_from_left": [target["column_from_left"] for target in targets],
            "different_rows": item["different_rows"],
            "success": item["success"],
            "candidate_attempts": len(item["attempts"]),
            "selected": selected,
        })
    motion = replay_metrics(replay)
    report = {
        "schema": SCHEMA,
        "model_revision": "robot_v3.2.2-suction",
        "tool0_offset_local_z_m": 0.151,
        "request_contract": {
            "input": "two box IDs plus current wall occupancy",
            "id_role": "geometry lookup only",
            "selection": (
                "ID -> 6D Pose -> arm assignment, common lift, common grasp mode, "
                "retreat distance, base station, IK candidates"
            ),
        },
        "successes": int(benchmark["successes"]),
        "requests": int(benchmark["requests"]),
        "different_row_successes": int(benchmark["different_row_successes"]),
        "completed_boxes": 2 * int(benchmark["successes"]),
        "one_sided_attachment_frames": int(benchmark["one_sided_attachment_frames"]),
        "results": results,
        "candidate_failure_reasons": dict(failures.most_common()),
        "continuous_motion": motion,
        "collision_validation": {
            "success": bool(validation["success"]),
            "checked_frames": int(validation["checked_frames"]),
            "checked_edge_samples": int(validation["checked_edge_samples"]),
            "maximum_left_tilt_deg": float(validation["maximum_left_tilt_deg"]),
            "maximum_right_tilt_deg": float(validation["maximum_right_tilt_deg"]),
        },
        "acceptance_boundary": (
            "Three representative cross-row pairs are certified. Arbitrary 25 choose 2 "
            "coverage is not yet certified; infeasible candidates return explicit reasons."
        ),
    }
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(markdown(report), encoding="utf-8")
    print(
        f"success={report['successes']}/{report['requests']} "
        f"frames={motion['frame_count']} output={args.output_json}"
    )
    return 0


def markdown(report: dict[str, Any]) -> str:
    motion = report["continuous_motion"]
    validation = report["collision_validation"]
    lines = [
        "# Cross-Row Two-Box Synchronized Pickup",
        "",
        "## Result",
        "",
        f"- Cross-row requests: **{report['successes']}/{report['requests']}**.",
        f"- Completed boxes: **{report['completed_boxes']}/{2 * report['requests']}**.",
        f"- One-sided attachment frames: **{report['one_sided_attachment_frames']}**.",
        f"- Continuous operation boundaries: **{motion['continuous_boundaries']}/{motion['boundary_count']}**.",
        f"- MoveIt/FCL: **{'PASS' if validation['success'] else 'FAIL'}**, "
        f"{validation['checked_frames']} frames and {validation['checked_edge_samples']} edge samples.",
        f"- Maximum joint step: **{motion['maximum_joint_step_deg']:.6f} deg**; flips: **{motion['joint_flip_events']}**.",
        f"- Carried-box tilt, left/right: **{validation['maximum_left_tilt_deg']:.6f} / "
        f"{validation['maximum_right_tilt_deg']:.6f} deg**.",
        "",
        "## Selected Pair Plans",
        "",
        "| Request | Rows | Left / right ID | Lift (m) | Retreat (m) | Base X (m) | Attempts | Task planning L/R (ms) | Bridge (ms) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for item in report["results"]:
        selected = item["selected"]
        if not selected:
            lines.append(
                f"| {item['request_id']} | {'/'.join(map(str, item['rows_from_top']))} | "
                f"- | - | - | - | {item['candidate_attempts']} | - | - |"
            )
            continue
        lines.append(
            f"| {item['request_id']} | {'/'.join(map(str, item['rows_from_top']))} | "
            f"{selected['left_box_id']} / {selected['right_box_id']} | "
            f"{selected['common_updown_m']:.2f} | {selected['retreat_distance_m']:.2f} | "
            f"{selected['base_pose_map'][0]:.2f} | {item['candidate_attempts']} | "
            f"{selected['left_planning_ms']:.3f} / {selected['right_planning_ms']:.3f} | "
            f"{selected['bridge_planning_ms']:.3f} |"
        )
    lines.extend([
        "",
        "## Candidate Failures",
        "",
    ])
    if report["candidate_failure_reasons"]:
        lines.extend(
            f"- `{reason}`: {count}"
            for reason, count in report["candidate_failure_reasons"].items()
        )
    else:
        lines.append("- None")
    lines.extend(["", "## Boundary", "", report["acceptance_boundary"], ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
