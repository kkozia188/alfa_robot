#!/usr/bin/python3
"""Audit full online Pose-to-VAMP results and generate the acceptance report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_payload(box_id: int, roots: list[Path]):
    for root in roots:
        path = root / f"box-{box_id:02d}.json"
        if path.is_file():
            payload = read_json(path)
            if payload.get("success"):
                return payload, path
    raise RuntimeError(f"missing successful box {box_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", type=Path, action="append", required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = [root.resolve() for root in args.payload_dir]
    failures, rows = [], []
    total_ms = ik_ms = tree_ms = 0.0
    worker_trials = worker_solved = 0
    for box_id in range(1, 26):
        payload, path = select_payload(box_id, roots)
        segments = list(payload["segment_search"])
        forbidden = [
            item for item in segments
            if item.get("backend_audit", {}).get("backend") != "vamp_rrt_connect_15d"
            or (item.get("success") and
                item.get("strategy") != "pure_vamp_rrt_connect_dual_arm_updown")
            or item.get("backend_audit", {}).get("fallback_used") is not False
            or item.get("backend_audit", {}).get("shortcut_used") is not False
            or item.get("backend_audit", {}).get("endpoint_source") != "online_pose_ik"
        ]
        if forbidden:
            failures.append(f"box {box_id} has non-VAMP or fallback segments")
        trials = [
            worker
            for item in segments
            for worker in item.get("backend_audit", {}).get("workers", [])
        ]
        worker_trials += len(trials)
        worker_solved += sum(bool(worker["solved"]) for worker in trials)
        task_ms = float(payload["total_ms"])
        task_ik_ms = float(payload["metrics"]["ik_ms"])
        task_tree_ms = sum(float(item["joint_rrt_ms"]) for item in segments)
        total_ms += task_ms
        ik_ms += task_ik_ms
        tree_ms += task_tree_ms
        rows.append({
            "box_id": box_id, "payload": str(path),
            "total_ms": task_ms, "ik_ms": task_ik_ms,
            "tree_ms": task_tree_ms, "segments": len(segments),
            "worker_trials": len(trials),
            "worker_solved": sum(bool(worker["solved"]) for worker in trials),
        })
    base_report = read_json(args.base_report.resolve())
    base_cycles = list(base_report.get("cycles", []))
    invalid_base = [
        item for item in base_cycles
        if not item.get("search_executed")
        or item.get("path_source") != "vamp_rrtc_search_tree"
        or item.get("fallback_used") is not False
        or item.get("shortcut_edge_count") != 0
        or item.get("golden_intermediate_frames_used_for_search") is not False
    ]
    if len(base_cycles) != 45 or invalid_base:
        failures.append(
            f"base/dual-stow search audit invalid: {len(base_cycles)} phases, "
            f"{len(invalid_base)} forbidden"
        )
    base_tree_ms = sum(float(item["planning_core_ms"]) for item in base_cycles)
    base_worker_trials = sum(len(item.get("local_search_worker_trials", [])) for item in base_cycles)
    base_worker_solved = sum(int(item.get("local_search_solved_workers", 0)) for item in base_cycles)
    total_ms += base_tree_ms
    tree_ms += base_tree_ms
    worker_trials += base_worker_trials
    worker_solved += base_worker_solved
    validation = read_json(args.validation.resolve())
    if not base_report.get("success"):
        failures.append("base VAMP phase search failed")
    if not validation.get("success"):
        failures.append("final MoveIt/FCL replay validation failed")
    rrd = args.rerun.resolve()
    if not rrd.is_file():
        failures.append("Rerun file is missing")
    report = {
        "schema": "alfa.v322_online_pose_vamp_fullflow.v1",
        "success": not failures,
        "failures": failures,
        "contract": {
            "input": "current RobotState + target box/TCP Pose + scene",
            "online_ik": True,
            "cached_joint_endpoint_used": False,
            "cached_path_or_waypoint_used": False,
            "shortcut_used": False,
            "fallback_used": False,
            "arm_search_state": "15D updown + dual-arm 14 axes",
            "base_search_state": "17D base x/y/yaw + dual-arm 14 axes; updown projected",
            "tree_backend": "VAMP RRT-Connect",
        },
        "boxes": rows,
        "totals": {
            "boxes": len(rows), "online_total_ms": total_ms,
            "online_ik_ms": ik_ms, "vamp_tree_ms": tree_ms,
            "searched_base_and_dual_stow_phases": len(base_cycles),
            "worker_trials": worker_trials, "worker_solved": worker_solved,
        },
        "validation": validation,
        "rerun": {"path": str(rrd), "sha256": sha256(rrd) if rrd.is_file() else None},
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(
        "# V3.2.2 full online Pose-to-VAMP result\n\n"
        f"- Success: **{report['success']}**\n"
        f"- Boxes: **{len(rows)}/25**\n"
        f"- Accumulated online planning: **{total_ms / 1000.0:.3f}s**\n"
        f"- Online IK: **{ik_ms / 1000.0:.3f}s**\n"
        f"- VAMP tree stages: **{tree_ms / 1000.0:.3f}s**\n"
        f"- Workers solved: **{worker_solved}/{worker_trials}**\n"
        f"- Validation: **{validation.get('checked_frames', 0)} frames + "
        f"{validation.get('checked_edge_samples', 0)} edge samples**\n"
        f"- Rerun SHA-256: `{report['rerun']['sha256']}`\n",
        encoding="utf-8")
    print(f"RESULT success={report['success']} boxes={len(rows)}/25")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
