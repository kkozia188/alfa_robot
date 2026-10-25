#!/usr/bin/env python3
"""Summarize shared-IK paired full-flow trials."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    coordinate = fraction * (len(ordered) - 1)
    lower = int(coordinate)
    upper = min(lower + 1, len(ordered) - 1)
    ratio = coordinate - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    trials_doc = json.loads((root / "trials.json").read_text(encoding="utf-8"))
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    round_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for trial in trials_doc["trials"]:
        box_id = int(trial["box_id"])
        success = bool(trial["success"])
        payload = None
        winning_ompl = winning_vamp = 0.0
        if Path(trial["result_json"]).is_file():
            payload = json.loads(Path(trial["result_json"]).read_text(encoding="utf-8"))
        if success and payload:
            for segment in payload["paper_benchmark"]["selected_segments"]:
                ompl_winner = float(segment["ompl"]["attempts"][-1]["planner_ms"])
                vamp_winner = float(segment["vamp"]["attempts"][-1]["planner_ms"])
                winning_ompl += ompl_winner
                winning_vamp += vamp_winner
                stage_rows.append({
                    "round": int(trial["round"]) + 1,
                    "box_id": box_id,
                    "stage": segment["stage"],
                    "side": trial["side"], "mode": trial["mode"],
                    "base_x": trial["base_x"],
                    "ompl_winning_search_ms": ompl_winner,
                    "vamp_winning_search_ms": vamp_winner,
                    "winning_search_speedup": ompl_winner / vamp_winner if vamp_winner > 0 else None,
                    "ompl_cumulative_ms": segment["ompl"]["accumulated_planner_ms"],
                    "vamp_cumulative_ms": segment["vamp"]["accumulated_planner_ms"],
                    "ompl_attempts": len(segment["ompl"]["attempts"]),
                    "vamp_attempts": len(segment["vamp"]["attempts"]),
                })
        ompl_ms = float(trial.get("ompl_ms") or 0.0)
        vamp_ms = float(trial.get("vamp_ms") or 0.0)
        row = {
            "round": int(trial["round"]) + 1,
            "box_id": box_id,
            "success": success,
            "failure_stage": trial.get("failure_stage", ""),
            "side": trial["side"], "mode": trial["mode"], "base_x": trial["base_x"],
            "ompl_winning_search_ms": winning_ompl if success else None,
            "vamp_winning_search_ms": winning_vamp if success else None,
            "winning_search_speedup": winning_ompl / winning_vamp if winning_vamp > 0 else None,
            "ompl_cumulative_to_solution_ms": ompl_ms,
            "vamp_cumulative_to_solution_ms": vamp_ms,
            "cumulative_speedup": ompl_ms / vamp_ms if vamp_ms > 0 else None,
            "time_reduction_percent": 100.0 * (1.0 - vamp_ms / ompl_ms) if ompl_ms > 0 else None,
        }
        grouped[box_id].append(row)
        round_rows.append(row)

    box_rows: list[dict[str, Any]] = []
    for box_id in range(1, 26):
        rows = grouped.get(box_id, [])
        successes = [row for row in rows if row["success"]]
        ow = [float(row["ompl_winning_search_ms"]) for row in successes]
        vw = [float(row["vamp_winning_search_ms"]) for row in successes]
        oc = [float(row["ompl_cumulative_to_solution_ms"]) for row in successes]
        vc = [float(row["vamp_cumulative_to_solution_ms"]) for row in successes]
        wr = [float(row["winning_search_speedup"]) for row in successes]
        cr = [float(row["cumulative_speedup"]) for row in successes]
        exemplar = successes[0] if successes else (rows[0] if rows else {})
        box_rows.append({
            "box_id": box_id, "rounds": len(rows), "successes": len(successes),
            "ompl_winning_median_ms": median(ow), "vamp_winning_median_ms": median(vw),
            "winning_median_speedup": median(wr),
            "ompl_median_ms": median(oc), "vamp_median_ms": median(vc),
            "cumulative_median_speedup": median(cr),
            "cumulative_aggregate_speedup": sum(oc) / sum(vc) if vc and sum(vc) > 0 else None,
            "ompl_p95_ms": percentile(oc, 0.95), "vamp_p95_ms": percentile(vc, 0.95),
            "median_reduction_percent": 100.0 * (1.0 - 1.0 / median(cr)) if cr and median(cr) else None,
            "selected_base_x": str(exemplar.get("base_x", "")),
            "selected_side": str(exemplar.get("side", "")),
            "selected_mode": str(exemplar.get("mode", "")),
        })

    successful = [row for row in round_rows if row["success"]]
    ow = [float(row["ompl_winning_search_ms"]) for row in successful]
    vw = [float(row["vamp_winning_search_ms"]) for row in successful]
    oc = [float(row["ompl_cumulative_to_solution_ms"]) for row in successful]
    vc = [float(row["vamp_cumulative_to_solution_ms"]) for row in successful]
    wr = [float(row["winning_search_speedup"]) for row in successful]
    cr = [float(row["cumulative_speedup"]) for row in successful]
    summary = {
        "schema": "alfa.v322.shared_ik_paired_summary.v1",
        "contract": trials_doc["contract"],
        "coverage": {
            "boxes": len(grouped),
            "rounds": max((row["round"] for row in round_rows), default=0),
            "successful_trials": len(successful), "total_trials": len(round_rows),
            "all_25_boxes_have_success": all(
                any(row["success"] for row in grouped.get(box_id, []))
                for box_id in range(1, 26)
            ),
        },
        "search_metric": {
            "winning_search": {
                "ompl_total_ms": sum(ow), "vamp_total_ms": sum(vw),
                "aggregate_speedup": sum(ow) / sum(vw) if vw and sum(vw) > 0 else None,
                "paired_speedup_median": median(wr),
                "ompl_median_ms": median(ow), "vamp_median_ms": median(vw),
            },
            "cumulative_to_solution": {
                "ompl_total_ms": sum(oc), "vamp_total_ms": sum(vc),
                "aggregate_speedup": sum(oc) / sum(vc) if vc and sum(vc) > 0 else None,
                "paired_speedup_median": median(cr),
                "ompl_median_ms": median(oc), "vamp_median_ms": median(vc),
                "ompl_p95_ms": percentile(oc, 0.95), "vamp_p95_ms": percentile(vc, 0.95),
            },
        },
        "per_box": box_rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(root / "per-box.csv", box_rows)
    write_csv(root / "per-round.csv", round_rows)
    write_csv(root / "per-stage.csv", stage_rows)
    print(json.dumps(summary["coverage"], indent=2))
    print(json.dumps(summary["search_metric"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
