#!/usr/bin/env python3
"""Summarize the 25-box matched adaptive VAMP paper experiment."""

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
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    ratio = index - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    trials_doc = json.loads((root / "trials.json").read_text(encoding="utf-8"))
    trials = trials_doc["trials"]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    stage_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    for trial in trials:
        box_id = int(trial["box_id"])
        selected = trial.get("selected")
        ompl_total = sum(float(item.get("ompl_ms") or 0.0) for item in trial["config_attempts"])
        vamp_total = sum(float(item.get("vamp_ms") or 0.0) for item in trial["config_attempts"])
        winning_ompl = 0.0
        winning_vamp = 0.0
        payload: dict[str, Any] | None = None
        if selected:
            payload = json.loads(Path(selected["result_json"]).read_text(encoding="utf-8"))
            for segment in payload["paper_benchmark"]["selected_segments"]:
                winning_ompl += float(segment["ompl"]["attempts"][-1]["planner_ms"])
                winning_vamp += float(segment["vamp"]["attempts"][-1]["planner_ms"])
        row = {
            "round": int(trial["round"]) + 1,
            "box_id": box_id,
            "success": bool(trial["success"]),
            "config_attempts": len(trial["config_attempts"]),
            "base_x": selected.get("base_x") if selected else None,
            "side": selected.get("side") if selected else None,
            "mode": selected.get("mode") if selected else None,
            "ompl_winning_search_ms": winning_ompl if selected else None,
            "vamp_winning_search_ms": winning_vamp if selected else None,
            "winning_search_speedup": (
                winning_ompl / winning_vamp if winning_vamp > 0.0 else None
            ),
            "ompl_cumulative_to_solution_ms": ompl_total,
            "vamp_cumulative_to_solution_ms": vamp_total,
            "cumulative_speedup": ompl_total / vamp_total if vamp_total > 0.0 else None,
            "time_reduction_percent": (
                100.0 * (1.0 - vamp_total / ompl_total) if ompl_total > 0.0 else None
            ),
        }
        grouped[box_id].append(row)
        round_rows.append(row)
        if not selected or payload is None:
            continue
        for segment in payload["paper_benchmark"]["selected_segments"]:
            ompl_ms = float(segment["ompl"]["accumulated_planner_ms"])
            vamp_ms = float(segment["vamp"]["accumulated_planner_ms"])
            ompl_winner = float(segment["ompl"]["attempts"][-1]["planner_ms"])
            vamp_winner = float(segment["vamp"]["attempts"][-1]["planner_ms"])
            stage_rows.append({
                "round": row["round"], "box_id": box_id,
                "stage": segment["stage"], "base_x": row["base_x"],
                "side": row["side"], "mode": row["mode"],
                "ompl_winning_search_ms": ompl_winner,
                "vamp_winning_search_ms": vamp_winner,
                "winning_search_speedup": ompl_winner / vamp_winner if vamp_winner > 0.0 else None,
                "ompl_cumulative_ms": ompl_ms, "vamp_cumulative_ms": vamp_ms,
                "cumulative_speedup": ompl_ms / vamp_ms if vamp_ms > 0.0 else None,
                "ompl_attempts": len(segment["ompl"]["attempts"]),
                "vamp_attempts": len(segment["vamp"]["attempts"]),
            })

    box_rows: list[dict[str, Any]] = []
    for box_id in range(1, 26):
        rows = grouped.get(box_id, [])
        ompl = [float(row["ompl_cumulative_to_solution_ms"]) for row in rows if row["success"]]
        vamp = [float(row["vamp_cumulative_to_solution_ms"]) for row in rows if row["success"]]
        ompl_winning = [float(row["ompl_winning_search_ms"]) for row in rows if row["success"]]
        vamp_winning = [float(row["vamp_winning_search_ms"]) for row in rows if row["success"]]
        ratios = [float(row["cumulative_speedup"]) for row in rows if row["cumulative_speedup"] is not None]
        winning_ratios = [float(row["winning_search_speedup"]) for row in rows if row["winning_search_speedup"] is not None]
        box_rows.append({
            "box_id": box_id,
            "rounds": len(rows),
            "successes": sum(bool(row["success"]) for row in rows),
            "ompl_median_ms": median(ompl),
            "vamp_median_ms": median(vamp),
            "ompl_winning_median_ms": median(ompl_winning),
            "vamp_winning_median_ms": median(vamp_winning),
            "winning_median_speedup": median(winning_ratios),
            "cumulative_median_speedup": median(ratios),
            "cumulative_aggregate_speedup": sum(ompl) / sum(vamp) if vamp and sum(vamp) > 0.0 else None,
            "ompl_p95_ms": percentile(ompl, 0.95),
            "vamp_p95_ms": percentile(vamp, 0.95),
            "median_reduction_percent": (
                100.0 * (1.0 - 1.0 / median(ratios)) if ratios and median(ratios) else None
            ),
            "selected_base_x": ",".join(str(row["base_x"]) for row in rows if row["success"]),
            "selected_side": ",".join(str(row["side"]) for row in rows if row["success"]),
            "selected_mode": ",".join(str(row["mode"]) for row in rows if row["success"]),
        })

    successful = [row for row in round_rows if row["success"]]
    all_ompl = [float(row["ompl_cumulative_to_solution_ms"]) for row in successful]
    all_vamp = [float(row["vamp_cumulative_to_solution_ms"]) for row in successful]
    all_ompl_winning = [float(row["ompl_winning_search_ms"]) for row in successful]
    all_vamp_winning = [float(row["vamp_winning_search_ms"]) for row in successful]
    all_ratios = [float(row["cumulative_speedup"]) for row in successful if row["cumulative_speedup"] is not None]
    all_winning_ratios = [float(row["winning_search_speedup"]) for row in successful if row["winning_search_speedup"] is not None]
    summary = {
        "schema": "alfa.v322.vamp_paper_adaptive_summary.v1",
        "contract": trials_doc["contract"],
        "coverage": {
            "boxes": len(grouped),
            "rounds": max((int(row["round"]) for row in round_rows), default=0),
            "successful_trials": len(successful),
            "total_trials": len(round_rows),
            "all_25_boxes_have_success": all(
                any(row["success"] for row in grouped.get(box_id, []))
                for box_id in range(1, 26)
            ),
        },
        "search_metric": {
            "winning_search": {
                "ompl_total_ms": sum(all_ompl_winning),
                "vamp_total_ms": sum(all_vamp_winning),
                "aggregate_speedup": sum(all_ompl_winning) / sum(all_vamp_winning) if all_vamp_winning and sum(all_vamp_winning) > 0.0 else None,
                "paired_speedup_median": median(all_winning_ratios),
                "ompl_median_ms": median(all_ompl_winning),
                "vamp_median_ms": median(all_vamp_winning),
            },
            "cumulative_to_solution": {
                "ompl_total_ms": sum(all_ompl),
                "vamp_total_ms": sum(all_vamp),
                "aggregate_speedup": sum(all_ompl) / sum(all_vamp) if all_vamp and sum(all_vamp) > 0.0 else None,
                "paired_speedup_median": median(all_ratios),
                "ompl_median_ms": median(all_ompl),
                "vamp_median_ms": median(all_vamp),
                "ompl_p95_ms": percentile(all_ompl, 0.95),
                "vamp_p95_ms": percentile(all_vamp, 0.95),
            },
        },
        "per_box": box_rows,
    }
    write_json(root / "summary.json", summary)
    write_csv(root / "per-box.csv", box_rows)
    write_csv(root / "per-round.csv", round_rows)
    write_csv(root / "per-stage.csv", stage_rows)
    print(json.dumps(summary["coverage"], indent=2))
    print(json.dumps(summary["search_metric"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
