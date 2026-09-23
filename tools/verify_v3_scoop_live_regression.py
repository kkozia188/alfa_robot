#!/usr/bin/env python3
"""Verify one cold live MOTION-224 regression run."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-core-planning-s", type=float, default=5.0)
    args = parser.parse_args()
    recording = args.recording.resolve()
    stem = recording.with_suffix("")
    paths = {
        "recording": recording,
        "summary": stem.with_name(stem.name + "-summary.json"),
        "plan_cache": stem.with_name(stem.name + "-plan-cache.json"),
        "metrics": stem.with_name(stem.name + "-metrics.csv"),
    }
    assert all(path.is_file() for path in paths.values()), paths
    summary = json.loads(paths["summary"].read_text())
    cache = json.loads(paths["plan_cache"].read_text())
    rows = list(csv.DictReader(paths["metrics"].open()))
    assert summary["completed_boxes"] == summary["validated_transitions"] == 25
    assert len(summary["boxes"]) == len(cache["tasks"]) == len(rows) == 25
    assert sorted(int(row["box_id"]) for row in rows) == list(range(1, 26))

    times = [float(row["selected_planner_s"]) for row in rows]
    assert all(value <= args.max_core_planning_s for value in times), [
        (row["box_id"], row["selected_planner_s"])
        for row in rows if float(row["selected_planner_s"]) > args.max_core_planning_s
    ]
    flips = sum(int(row["joint_flip_events"]) for row in rows)
    inverted = sum(
        int(box["carried_box_orientation"]["inverted_frames"])
        for box in summary["boxes"]
    )
    assert flips == inverted == 0

    segments = []
    for box in summary["boxes"]:
        segments.extend(box.get("segment_search", []))
        for leg in box.get("transition_motion", {}).get("search_legs", []):
            segments.extend(leg.get("segments", []))
    successful = [segment for segment in segments if segment.get("success")]
    strategies = {segment.get("strategy") for segment in successful}
    repair_dimensions = {
        len(segment.get("repaired_joints", []))
        for segment in segments if segment.get("repaired_joints")
    }
    aggregate = summary["planning_search"]
    assert "joint_shortcut" in strategies
    assert "shortcut_repair_rrt" in strategies and {1, 2} <= repair_dimensions
    assert "joint_rrt_fallback" in strategies
    assert aggregate["task"]["tcp_shortcut_attempts"] + aggregate["transition"]["tcp_shortcut_attempts"] > 0

    ordered = sorted(times)
    report = {
        "schema": "alfa.v3_scoop_golden_live_regression.v1",
        "completed_boxes": 25,
        "validated_transitions": 25,
        "core_planning_s": {
            "average": statistics.fmean(times),
            "median": statistics.median(times),
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "maximum": max(times),
            "at_or_below_limit": len(times),
            "limit": args.max_core_planning_s,
        },
        "joint_flip_events": flips,
        "carried_box_inverted_frames": inverted,
        "strategy_evidence": {
            "successful": sorted(strategy for strategy in strategies if strategy),
            "shortcut_repair_dimensions": sorted(repair_dimensions),
            "tcp_shortcut_attempts": (
                aggregate["task"]["tcp_shortcut_attempts"]
                + aggregate["transition"]["tcp_shortcut_attempts"]
            ),
        },
        "artifacts": {
            name: {"name": path.name, "sha256": sha256(path), "size": path.stat().st_size}
            for name, path in paths.items()
        },
        "current_l3_claimed": False,
        "current_l4_claimed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
