#!/usr/bin/env python3
"""Compare two docking-error summaries that share one experiment design."""

from __future__ import annotations

import argparse
from pathlib import Path

from docking_error_common import PHASES, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    before = read_json(args.before.resolve())
    after = read_json(args.after.resolve())
    if before.get("design_fingerprint") != after.get("design_fingerprint"):
        raise SystemExit("summaries do not use the same matrix/task design")
    comparison = {
        "schema": "alfa.v3_docking_error_benchmark_comparison.v1",
        "design_fingerprint": before["design_fingerprint"],
        "before": before.get("strategy_label", ""),
        "after": after.get("strategy_label", ""),
        "cycle_success_rate_before": before["cycle_success_rate"],
        "cycle_success_rate_after": after["cycle_success_rate"],
        "cycle_success_rate_delta": after["cycle_success_rate"] - before["cycle_success_rate"],
        "planning_phase_deltas": {},
        "failure_categories_before": before["failure_categories"],
        "failure_categories_after": after["failure_categories"],
    }
    for phase in PHASES:
        old = before["planning_phase_success"][phase]["all_cycle_success_rate"]
        new = after["planning_phase_success"][phase]["all_cycle_success_rate"]
        comparison["planning_phase_deltas"][phase] = {
            "before": old,
            "after": new,
            "delta": new - old,
        }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "comparison.json", comparison)
    lines = [
        "# Docking Error Planning Comparison",
        "",
        f"- Design fingerprint: `{comparison['design_fingerprint']}`",
        f"- Before: `{comparison['before']}`",
        f"- After: `{comparison['after']}`",
        f"- Cycle success: `{comparison['cycle_success_rate_before']:.2%}` -> "
        f"`{comparison['cycle_success_rate_after']:.2%}` "
        f"(`{comparison['cycle_success_rate_delta']:+.2%}`)",
        "",
        "| Phase | Before | After | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for phase, values in comparison["planning_phase_deltas"].items():
        lines.append(
            f"| {phase} | {values['before']:.2%} | {values['after']:.2%} | "
            f"{values['delta']:+.2%} |"
        )
    (output_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
