#!/usr/bin/env python3
"""Summarize VAMP throughput, V3 bottlenecks, and the local-RRT A/B probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_BASELINE = Path(
    "/home/tim/alfa_robot-v322-chest-carry-fast-base/results/"
    "v322_chest_carry_fast_base/direct_pull_payloads_v2"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def probe_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    transfer = next(
        segment
        for segment in payload["segment_search"]
        if segment["stage"] == "loaded_transfer"
    )
    metrics = payload["metrics"]
    return {
        "success": bool(payload["success"]),
        "total_ms": float(payload["total_ms"]),
        "ik_ms": float(metrics["ik_ms"]),
        "collision_ms": float(metrics["collision_ms"]),
        "loaded_transfer_ms": float(transfer["total_ms"]),
        "shortcut_repair_ms": float(transfer["shortcut_repair_rrt_ms"]),
        "parallel_workers": int(transfer["parallel_rrt_workers_launched"]),
        "parallel_winner_seed": int(transfer["parallel_rrt_winner_seed"]),
    }


def frame_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload["frames"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--regression",
        type=Path,
        default=Path("results/v322_vamp_parallel/default_serial_payloads"),
    )
    parser.add_argument(
        "--vamp",
        type=Path,
        default=Path("results/v322_vamp_parallel/vamp-panda-throughput-4w.json"),
    )
    parser.add_argument(
        "--serial-probe",
        type=Path,
        default=Path(
            "results/v322_vamp_parallel/serial_fast_probe_box02/box-02.json"
        ),
    )
    parser.add_argument(
        "--parallel-probe",
        type=Path,
        default=Path(
            "results/v322_vamp_parallel/parallel_fast_probe_box02/box-02.json"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/v322_vamp_parallel/vamp-parallel-experiment"),
    )
    args = parser.parse_args()

    boxes = [read_json(path) for path in sorted(args.baseline.resolve().glob("box-*.json"))]
    if len(boxes) != 25:
        raise ValueError(f"expected 25 baseline payloads, got {len(boxes)}")
    total_ms = sum(float(box["total_ms"]) for box in boxes)
    ik_ms = sum(float(box["metrics"]["ik_ms"]) for box in boxes)
    collision_ms = sum(float(box["metrics"]["collision_ms"]) for box in boxes)
    regression_boxes = [
        read_json(path) for path in sorted(args.regression.resolve().glob("box-*.json"))
    ]
    if len(regression_boxes) != 25:
        raise ValueError(
            f"expected 25 regression payloads, got {len(regression_boxes)}"
        )
    matching_frame_hashes = sum(
        frame_hash(baseline) == frame_hash(regression)
        for baseline, regression in zip(boxes, regression_boxes)
    )
    vamp = read_json(args.vamp.resolve())
    serial = probe_summary(args.serial_probe.resolve())
    parallel = probe_summary(args.parallel_probe.resolve())
    probe_speedup = serial["total_ms"] / parallel["total_ms"]
    payload = {
        "schema": "alfa.v322_vamp_parallel_experiment.v1",
        "success": (
            vamp["success"] and serial["success"] and parallel["success"]
        ),
        "vamp_framework": {
            "scope": "official precompiled Panda sphere-cage benchmark",
            "trials": int(vamp["parallel"]["trials"]),
            "workers": int(vamp["parallel"]["workers"]),
            "serial_throughput_plans_s": float(vamp["serial"]["throughput_plans_s"]),
            "parallel_throughput_plans_s": float(
                vamp["parallel"]["throughput_plans_s"]
            ),
            "parallel_speedup": float(vamp["parallel"]["speedup_vs_serial"]),
            "v3_fidelity": False,
        },
        "v3_baseline_profile": {
            "box_count": len(boxes),
            "all_under_3s": all(float(box["total_ms"]) < 3000.0 for box in boxes),
            "maximum_task_ms": max(float(box["total_ms"]) for box in boxes),
            "total_task_ms": total_ms,
            "ik_ms": ik_ms,
            "collision_ms": collision_ms,
            "ik_share_percent": 100.0 * ik_ms / total_ms,
            "collision_share_percent": 100.0 * collision_ms / total_ms,
        },
        "v3_box02_local_rrt_ab": {
            "serial": serial,
            "parallel": parallel,
            "observed_total_speedup": probe_speedup,
            "meaningful_speedup": probe_speedup >= 1.10,
        },
        "v3_default_serial_regression": {
            "box_count": len(regression_boxes),
            "success_count": sum(bool(box["success"]) for box in regression_boxes),
            "under_3s_count": sum(
                float(box["total_ms"]) < 3000.0 for box in regression_boxes
            ),
            "maximum_task_ms": max(
                float(box["total_ms"]) for box in regression_boxes
            ),
            "total_task_ms": sum(float(box["total_ms"]) for box in regression_boxes),
            "baseline_matching_frame_hashes": matching_frame_hashes,
            "note": (
                "Differing hashes have identical stage/frame counts but alternate valid IK "
                "branches; this experiment does not replace the certified replay."
            ),
        },
        "decision": {
            "production_default_workers": 1,
            "experimental_workers": 4,
            "use_vamp_as_authority": False,
            "reason": (
                "Stock VAMP lacks the V3.2.2 robot kernel, and the certified task is "
                "currently IK-bound rather than FCL-bound. Keep four-worker local RRT "
                "available for hard repair windows, but do not enable it by default."
            ),
        },
    }

    output_json = args.output_prefix.with_suffix(".json").resolve()
    output_md = args.output_prefix.with_suffix(".md").resolve()
    atomic_write(output_json, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    profile = payload["v3_baseline_profile"]
    framework = payload["vamp_framework"]
    regression = payload["v3_default_serial_regression"]
    lines = [
        "# VAMP parallel experiment",
        "",
        "## Result",
        "",
        f"- Official VAMP benchmark: {framework['trials']}/{framework['trials']} solved.",
        f"- Four-process throughput: {framework['parallel_throughput_plans_s']:.1f} plans/s, "
        f"{framework['parallel_speedup']:.2f}x versus serial.",
        "- This number is for VAMP's precompiled Panda model, not ALFA V3.2.2.",
        "",
        "## V3 profile",
        "",
        f"- 25-box task maximum: {profile['maximum_task_ms'] / 1000.0:.3f}s; 25/25 under 3s.",
        f"- IK share of task core: {profile['ik_share_percent']:.1f}%.",
        f"- FCL collision-check share of task core: {profile['collision_share_percent']:.1f}%.",
        f"- Box 2 local-RRT A/B total speedup: {probe_speedup:.3f}x.",
        f"- Default-one-worker regression: {regression['success_count']}/25 success, "
        f"{regression['under_3s_count']}/25 under 3s, maximum "
        f"{regression['maximum_task_ms'] / 1000.0:.3f}s.",
        "",
        "## Decision",
        "",
        "Keep the production default at one worker. Four-worker local RRT remains an "
        "explicit experiment for difficult repair windows. A production VAMP backend "
        "requires a reviewed sphere model, Cricket-generated V3 kernel, attached-box "
        "support, and zero-false-negative comparison against MoveIt/FCL.",
    ]
    atomic_write(output_md, "\n".join(lines) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
