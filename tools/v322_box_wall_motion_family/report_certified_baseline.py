#!/usr/bin/env python3
"""Evaluate a compiled family plan against the certified V3.2.2 baseline."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "alfa.v322_box_wall_motion_family_report.v1"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--runtime-times", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    plan = read_json(args.plan)
    cache = read_json(args.cache)
    validation = read_json(args.validation)
    runtime = read_json(args.runtime_times)
    cached = {int(task["box_id"]): task for task in cache["tasks"]}

    representative: list[dict[str, Any]] = []
    failure_counts: collections.Counter[str] = collections.Counter()
    for task in plan["tasks"]:
        candidate = task["selected_candidate"]
        slot = int(task["derived_geometry"]["scene_slot"])
        baseline = cached.get(slot)
        reason = ""
        if baseline is None:
            reason = "no certified geometry match"
        elif not baseline.get("payload", {}).get("success", False):
            reason = str(baseline.get("payload", {}).get("failure_reason", "baseline failed"))
        elif str(baseline["mode"]) != str(candidate["grasp_mode"]):
            reason = "certified_template_mismatch"
        elif str(baseline["side"]) != str(candidate["arm"]):
            reason = "certified_arm_mismatch"
        elif abs(float(baseline["updown"]) - float(candidate["updown_m"])) > 1.0e-8:
            reason = "certified_lift_mismatch"
        success = not reason
        if reason:
            failure_counts[reason] += 1
        representative.append({
            "request_id": task["request_id"],
            "row_from_top": task["derived_geometry"]["row_from_top"],
            "column_from_left": task["derived_geometry"]["column_from_left"],
            "geometry_match_key": slot,
            "selected_template": candidate["template"],
            "selected_grasp_mode": candidate["grasp_mode"],
            "selected_arm": candidate["arm"],
            "selected_updown_m": candidate["updown_m"],
            "success": success,
            "planning_time_ms": (
                float(baseline["task_accumulated_search_ms"]) if baseline else None
            ),
            "failure_reason": reason,
            "evidence": "certified V3.2.2 pose-matched trajectory",
        })

    row_stats: list[dict[str, Any]] = []
    for row in range(1, 6):
        tasks = [task for task in cache["tasks"] if int(task["row"]) == row]
        times = [float(task["task_accumulated_search_ms"]) for task in tasks]
        successes = sum(bool(task.get("payload", {}).get("success", False)) for task in tasks)
        row_stats.append({
            "row_from_top": row,
            "successes": successes,
            "attempts": len(tasks),
            "success_rate": successes / len(tasks) if tasks else 0.0,
            "planning_time_ms": {
                "mean": sum(times) / len(times) if times else None,
                "maximum": max(times) if times else None,
                "minimum": min(times) if times else None,
            },
            "grasp_modes": sorted({str(task["mode"]) for task in tasks}),
        })

    report = {
        "schema": REPORT_SCHEMA,
        "model_revision": cache.get("model_revision"),
        "tool0_offset_local_z_m": 0.151,
        "evidence_level": "certified_baseline_adapter",
        "evidence_limit": (
            "This proves pose-driven selection reproduces certified representatives; "
            "off-grid generalization requires fresh MoveIt/FCL runs."
        ),
        "representative_tasks": representative,
        "representative_summary": {
            "successes": sum(item["success"] for item in representative),
            "attempts": len(representative),
            "failure_reasons": dict(sorted(failure_counts.items())),
        },
        "certified_all_boxes_by_row": row_stats,
        "collision_validation": {
            "success": bool(validation["success"]),
            "checked_frames": int(validation["checked_frames"]),
            "checked_edge_samples": int(validation["checked_edge_samples"]),
            "maximum_left_tilt_deg": float(validation["maximum_left_tilt_deg"]),
            "maximum_right_tilt_deg": float(validation["maximum_right_tilt_deg"]),
        },
        "runtime_validation": {
            "operations": len(runtime["operations"]),
            "maximum_ms_including_process_startup": max(
                float(item["validation_ms_including_process_startup"])
                for item in runtime["operations"]
            ),
            "all_below_3s": all(
                float(item["validation_ms_including_process_startup"]) < 3000.0
                for item in runtime["operations"]
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(markdown(report), encoding="utf-8")
    summary = report["representative_summary"]
    print(
        f"representatives={summary['successes']}/{summary['attempts']} "
        f"collision={report['collision_validation']['success']} output={args.output_json}"
    )
    return 0 if summary["successes"] == summary["attempts"] else 1


def markdown(report: dict[str, Any]) -> str:
    representative = report["representative_summary"]
    collision = report["collision_validation"]
    lines = [
        "# V3.2.2 Box-Wall Motion-Family Baseline Report",
        "",
        f"Evidence level: `{report['evidence_level']}`.",
        "",
        report["evidence_limit"],
        "",
        "## Representative Tasks",
        "",
        "| Request | Row | Template | Arm | Lift (m) | Planning (ms) | Result |",
        "| --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for item in report["representative_tasks"]:
        result = "PASS" if item["success"] else f"FAIL: {item['failure_reason']}"
        lines.append(
            f"| {item['request_id']} | {item['row_from_top']} | "
            f"{item['selected_template']} | {item['selected_arm']} | "
            f"{item['selected_updown_m']:.2f} | {item['planning_time_ms']:.3f} | {result} |"
        )
    lines.extend([
        "",
        f"Representative result: **{representative['successes']}/{representative['attempts']}**.",
        "",
        "## Certified Row Statistics",
        "",
        "| Row | Success | Rate | Mean planning (ms) | Maximum (ms) | Modes |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for item in report["certified_all_boxes_by_row"]:
        lines.append(
            f"| {item['row_from_top']} | {item['successes']}/{item['attempts']} | "
            f"{item['success_rate']:.0%} | {item['planning_time_ms']['mean']:.3f} | "
            f"{item['planning_time_ms']['maximum']:.3f} | {', '.join(item['grasp_modes'])} |"
        )
    lines.extend([
        "",
        "## Collision Evidence",
        "",
        f"MoveIt/FCL: **{'PASS' if collision['success'] else 'FAIL'}**; "
        f"{collision['checked_frames']} frames and {collision['checked_edge_samples']} edge samples.",
        "",
        f"Runtime operation checks: {report['runtime_validation']['operations']} operations; "
        f"maximum {report['runtime_validation']['maximum_ms_including_process_startup']:.3f} ms.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
