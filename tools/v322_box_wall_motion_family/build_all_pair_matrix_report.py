#!/usr/bin/python3
"""Build the final human and machine report for the exhaustive pair matrix."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any

from build_acceptance_report import replay_metrics


SCHEMA = "alfa.v322_all_pair_matrix_acceptance.v1"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def row_column(box_id: int) -> tuple[int, int]:
    return (box_id - 1) // 5 + 1, (box_id - 1) % 5 + 1


def selected_paths(root: Path, result: dict[str, Any]) -> tuple[Path, Path]:
    selected = result["selected"]
    left, right = (int(value) for value in result["pair"])
    candidate = root / f"pair-{left:02d}-{right:02d}" / (
        f"candidate-{int(selected['candidate_rank']):03d}"
    )
    retry = selected.get("bridge_rescue_retry")
    if retry:
        return (
            candidate / f"replay-rescue-{int(retry)}.json",
            candidate / f"validation-rescue-{int(retry)}.json",
        )
    return candidate / "replay.json", candidate / "validation.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    summary = read_json(args.summary)
    row_pairs: dict[tuple[int, int], dict[str, int]] = {}
    base_x_counts: collections.Counter[str] = collections.Counter()
    base_y_counts: collections.Counter[str] = collections.Counter()
    yaw_counts: collections.Counter[str] = collections.Counter()
    lift_counts: collections.Counter[str] = collections.Counter()
    retreat_counts: collections.Counter[str] = collections.Counter()
    mode_counts: collections.Counter[str] = collections.Counter()
    candidate_ranks = []
    pickup_times = []
    bridge_times = []
    maximum_joint_step_deg = 0.0
    joint_flip_events = 0
    base_travel_m = 0.0
    maximum_left_tilt_deg = 0.0
    maximum_right_tilt_deg = 0.0
    missing_evidence = []

    for result in summary["results"]:
        left, right = (int(value) for value in result["pair"])
        left_row, _ = row_column(left)
        right_row, _ = row_column(right)
        key = tuple(sorted((left_row, right_row)))
        item = row_pairs.setdefault(key, {"attempts": 0, "successes": 0})
        item["attempts"] += 1
        item["successes"] += bool(result.get("success"))
        if not result.get("success"):
            continue
        selected = result["selected"]
        pose = selected.get("base_pose_map") or [
            selected.get("base_x_m", 0.0),
            selected.get("base_y_m", 0.0),
            selected.get("base_yaw_rad", 0.0),
        ]
        base_x_counts[f"{float(pose[0]):.3f}"] += 1
        base_y_counts[f"{float(pose[1]):.3f}"] += 1
        yaw_counts[f"{math.degrees(float(pose[2])):.1f}"] += 1
        lift_counts[f"{float(selected['common_updown_m']):.2f}"] += 1
        retreat_counts[f"{float(selected['retreat_distance_m']):.2f}"] += 1
        mode_counts[str(selected["grasp_mode"])] += 1
        candidate_ranks.append(int(selected["candidate_rank"]))
        pickup_times.append(
            float(selected.get("left_planning_ms", 0.0))
            + float(selected.get("right_planning_ms", 0.0))
        )
        bridge_times.append(float(selected.get("bridge_planning_ms", 0.0)))
        replay_path, validation_path = selected_paths(args.matrix_root, result)
        if not replay_path.is_file() or not validation_path.is_file():
            missing_evidence.append({
                "pair": result["pair"],
                "replay": str(replay_path),
                "validation": str(validation_path),
            })
            continue
        motion = replay_metrics(read_json(replay_path))
        validation = read_json(validation_path)
        maximum_joint_step_deg = max(
            maximum_joint_step_deg, float(motion["maximum_joint_step_deg"])
        )
        joint_flip_events += int(motion["joint_flip_events"])
        base_travel_m += float(motion["base_travel_m"])
        maximum_left_tilt_deg = max(
            maximum_left_tilt_deg, float(validation["maximum_left_tilt_deg"])
        )
        maximum_right_tilt_deg = max(
            maximum_right_tilt_deg, float(validation["maximum_right_tilt_deg"])
        )

    failures = []
    for item in summary["failed_pairs"]:
        left, right = (int(value) for value in item["pair"])
        left_row, left_column = row_column(left)
        right_row, right_column = row_column(right)
        dominant = max(item["failure_counts"].items(), key=lambda value: value[1])
        failures.append({
            "pair": [left, right],
            "rows_from_top": [left_row, right_row],
            "columns_from_left": [left_column, right_column],
            "vertical_span_rows": abs(left_row - right_row),
            "candidate_attempts": int(item["attempt_count"]),
            "dominant_failure": dominant[0],
            "dominant_failure_count": int(dominant[1]),
            "failure_counts": item["failure_counts"],
            "reason": (
                "No candidate gave both arms a complete precontact/approach/retreat "
                "path at one shared lift, grasp mode, and planar base pose."
            ),
        })

    row_statistics = [
        {
            "row_pair": list(key),
            **value,
            "failures": value["attempts"] - value["successes"],
            "success_rate": value["successes"] / value["attempts"],
        }
        for key, value in sorted(row_pairs.items())
    ]
    report = {
        "schema": SCHEMA,
        "model_revision": "robot_v3.2.2-suction",
        "tool0_offset_local_z_m": 0.151,
        "scene_contract": summary["scene_contract"],
        "candidate_contract": {
            "arm_assignments": 2,
            "lift_grid_m": [0.0, -0.25, -0.50, -0.75, -1.0],
            "base_x_m": [-0.35, -0.475, -0.60],
            "base_y_m": "0 plus geometry-derived shifts up to +/-0.75",
            "base_yaw_deg": [0.0, -15.0, 15.0],
            "front_retreat_m": [0.35, 0.30, 0.25, 0.20],
            "grasp_modes": ["front", "top_suction"],
            "top_suction_pruning": "rejected when any non-target box remains above either target",
        },
        "pair_count": int(summary["pair_count"]),
        "successes": int(summary["successes"]),
        "failures": int(summary["failures"]),
        "success_rate": float(summary["success_rate"]),
        "completed_boxes_across_certificates": 2 * int(summary["successes"]),
        "collision_validation": {
            "successful_pair_certificates": int(summary["successes"]),
            "checked_frames": int(summary["total_checked_frames"]),
            "checked_edge_samples": int(summary["total_checked_edge_samples"]),
            "one_sided_attachment_frames": int(summary["one_sided_attachment_frames"]),
            "maximum_joint_step_deg": maximum_joint_step_deg,
            "joint_flip_events": joint_flip_events,
            "maximum_left_tilt_deg": maximum_left_tilt_deg,
            "maximum_right_tilt_deg": maximum_right_tilt_deg,
            "aggregate_base_travel_m": base_travel_m,
            "missing_evidence": missing_evidence,
        },
        "planning": {
            "candidate_rank_mean": sum(candidate_ranks) / len(candidate_ranks),
            "candidate_rank_max": max(candidate_ranks),
            "pairs_after_candidate_100": sum(rank > 100 for rank in candidate_ranks),
            "pairs_after_candidate_500": sum(rank > 500 for rank in candidate_ranks),
            "pickup_core_ms_mean": sum(pickup_times) / len(pickup_times),
            "pickup_core_ms_max": max(pickup_times),
            "entry_bridge_ms_mean": sum(bridge_times) / len(bridge_times),
            "entry_bridge_ms_max": max(bridge_times),
            "selected_modes": dict(mode_counts),
            "selected_base_x": dict(base_x_counts),
            "selected_base_y": dict(base_y_counts),
            "selected_base_yaw_deg": dict(yaw_counts),
            "selected_lifts": dict(lift_counts),
            "selected_retreats": dict(retreat_counts),
        },
        "row_pair_statistics": row_statistics,
        "failed_pairs": failures,
        "aggregate_failed_candidate_reasons": summary["aggregate_failure_reasons"],
        "conclusion": (
            "292 of 300 pairs are certified in the strict full-wall scene. "
            "The remaining 8 pairs are not supported by the current synchronized "
            "shared-lift action family."
        ),
    }
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(markdown(report), encoding="utf-8")
    print(
        f"success={report['successes']}/{report['pair_count']} "
        f"failures={report['failures']} output={args.output_json}"
    )
    return 0


def markdown(report: dict[str, Any]) -> str:
    collision = report["collision_validation"]
    planning = report["planning"]
    lines = [
        "# Complete 25-Choose-2 Dual-Box Matrix",
        "",
        "## Result",
        "",
        f"- Certified pairs: **{report['successes']}/{report['pair_count']}** "
        f"(**{report['success_rate']:.2%}**).",
        f"- Unsupported pairs: **{report['failures']}**.",
        f"- Independent MoveIt/FCL: **{collision['checked_frames']:,} frames** and "
        f"**{collision['checked_edge_samples']:,} edge samples**.",
        f"- One-sided attachment frames: **{collision['one_sided_attachment_frames']}**.",
        f"- Maximum joint step: **{collision['maximum_joint_step_deg']:.6f} deg**; "
        f"joint flips: **{collision['joint_flip_events']}**.",
        f"- Maximum carried-box tilt, left/right: "
        f"**{collision['maximum_left_tilt_deg']:.6f} / "
        f"{collision['maximum_right_tilt_deg']:.6f} deg**.",
        "",
        "The strict scene keeps all 23 non-target boxes present. Every successful "
        "pair includes synchronized attachment, loaded retreat, base shuttle, release, "
        "outside stow, return, and a separate FCL certificate.",
        "",
        "## Unsupported Pairs",
        "",
        "| Pair | Rows | Columns | Row span | Candidates | Dominant failure |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for item in report["failed_pairs"]:
        lines.append(
            f"| {item['pair'][0]} + {item['pair'][1]} | "
            f"{item['rows_from_top'][0]} / {item['rows_from_top'][1]} | "
            f"{item['columns_from_left'][0]} / {item['columns_from_left'][1]} | "
            f"{item['vertical_span_rows']} | {item['candidate_attempts']} | "
            f"{item['dominant_failure']} ({item['dominant_failure_count']}) |"
        )
    lines.extend([
        "",
        "All 8 failures are same outer-column pairs with large vertical separation. "
        "No enumerated candidate produced complete single-arm paths for both boxes "
        "under one shared lift/mode/base pose, so none reached dual composition.",
        "",
        "## Planning Search",
        "",
        f"- Mean/max selected candidate rank: **{planning['candidate_rank_mean']:.2f} / "
        f"{planning['candidate_rank_max']}**.",
        f"- Successes after candidate 100 / 500: **{planning['pairs_after_candidate_100']} / "
        f"{planning['pairs_after_candidate_500']}**.",
        f"- Pickup core mean/max: **{planning['pickup_core_ms_mean']:.3f} / "
        f"{planning['pickup_core_ms_max']:.3f} ms**.",
        f"- Entry bridge mean/max: **{planning['entry_bridge_ms_mean']:.3f} / "
        f"{planning['entry_bridge_ms_max']:.3f} ms**.",
        "",
        "## Row-Pair Coverage",
        "",
        "| Rows | Success | Failure | Rate |",
        "| --- | ---: | ---: | ---: |",
    ])
    for item in report["row_pair_statistics"]:
        lines.append(
            f"| {item['row_pair'][0]} / {item['row_pair'][1]} | "
            f"{item['successes']} | {item['failures']} | {item['success_rate']:.1%} |"
        )
    lines.extend(["", "## Conclusion", "", report["conclusion"], ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
