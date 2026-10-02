#!/usr/bin/env python3
"""Replan all 25 V3.2.2 pickup/retreat payloads for one base yaw."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve().parent / "baseline" / "scan_v3_single_arm_box_wall.py"
GROUPS = [
    (1, 5), (2, 4), (3,), (6, 10), (7, 9), (8,), (11, 15),
    (12, 14), (13,), (16, 20), (17, 19), (18,), (21, 25),
    (22, 24), (23,),
]
HEIGHTS = {
    1: (0.0, -0.25, -0.50),
    2: (0.0, -0.25, -0.50),
    3: (-0.25, 0.0, -0.50),
    4: (-0.25, -0.50, 0.0, -0.75),
    5: (-0.75, -0.50, -0.25, -1.0),
}
LEFT_HOME = "155,-105,20,90,-90,-40,0"
RIGHT_HOME = "-155,105,-20,-90,90,40,0"
CENTER_BOXES = {3, 8, 13, 18, 23}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def row_for(box_id: int) -> int:
    return (box_id - 1) // 5 + 1


def unique(values: list[Any]) -> list[Any]:
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def arm_seed_degrees(payload: dict[str, Any], side: str) -> str:
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError("seed payload has no frames")
    joints = [float(value) for value in frames[0]["joints"]]
    active = joints[2:9] if side == "left" else joints[9:16]
    if len(active) != 7:
        raise ValueError("seed payload does not contain the 16-axis research contract")
    return ",".join(f"{math.degrees(value):.10f}" for value in active)


def task_map(cache: dict[str, Any]) -> dict[int, dict[str, Any]]:
    tasks = {int(task["box_id"]): task for task in cache.get("tasks", [])}
    if set(tasks) != set(range(1, 26)):
        raise ValueError("seed cache must contain boxes 1..25")
    return tasks


def run_candidate(
    *,
    box_id: int,
    side: str,
    height: float,
    mode: str,
    removed: set[int],
    yaw_rad: float,
    ik_seed: str,
    candidate_dir: Path,
    timeout_s: float,
) -> dict[str, Any]:
    result = candidate_dir / f"box-{box_id:02d}.json"
    scan = candidate_dir / f"box-{box_id:02d}.csv"
    command = [
        "/usr/bin/python3", str(SCRIPT),
        "--box-ids", str(box_id),
        "--contact-x", "0.75",
        "--base-x", "-0.60" if box_id <= 15 else "-0.35",
        "--base-y", "0.0",
        "--base-yaw", f"{yaw_rad:.12f}",
        "--end-effector", "scoop",
        "--force-side", side,
        "--force-updown-m", f"{height:.2f}",
        "--continuous-sequence",
        "--analytic-path-only",
        "--no-ignore-opposite-arm",
        "--initial-left-arm-joints-deg", LEFT_HOME,
        f"--initial-right-arm-joints-deg={RIGHT_HOME}",
        f"--ik-seed-arm-joints-deg={ik_seed}",
        "--timeout", f"{timeout_s:.1f}",
        "--maximum-cartesian-joint-step-deg", "180",
        "--natural-max-proximal-step-deg", "180",
        "--natural-max-wrist-step-deg", "180",
        "--front-retreat-distance-m", "0.30" if box_id > 20 else "0.35",
        "--top-retreat-distance-m", "0.20",
        "--selected-result-json", str(result),
        "--output", str(scan),
    ]
    if removed:
        command.extend(("--initial-removed-box-ids", ",".join(map(str, sorted(removed)))))
    if box_id in CENTER_BOXES:
        command.append("--auto-safe-opposite-arm")
    if mode == "front":
        command.extend(("--grasp-strategy", "front"))
    elif mode == "top_suction":
        command.extend(("--grasp-strategy", "mixed", "--top-suction-box-ids", str(box_id)))
    else:
        raise ValueError(f"unsupported grasp mode {mode}")
    completed = subprocess.run(command, check=False)
    if not result.is_file():
        return {
            "success": False,
            "failure_stage": f"exit_{completed.returncode}",
            "failure_reason": "planner did not write result_json_path",
        }
    return read_json(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaw-deg", type=float, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--continuation-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.yaw_deg) or not -5.0 <= args.yaw_deg <= 5.0:
        raise SystemExit("--yaw-deg must be finite and in [-5, 5]")
    if args.timeout_s <= 0.0:
        raise SystemExit("--timeout-s must be positive")

    baseline_cache = read_json(args.baseline_cache.resolve())
    baseline_tasks = task_map(baseline_cache)
    continuation_tasks = (
        task_map(read_json(args.continuation_cache.resolve()))
        if args.continuation_cache else {}
    )
    output_dir = args.output_dir.resolve()
    accepted_dir = output_dir / "pickups"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    yaw_rad = math.radians(args.yaw_deg)
    removed: set[int] = set()
    summary: list[dict[str, Any]] = []

    for group_index, group in enumerate(GROUPS, start=1):
        existing = [accepted_dir / f"box-{box_id:02d}.json" for box_id in group]
        if args.resume and all(
            path.is_file() and read_json(path).get("success") for path in existing
        ):
            payloads = [read_json(path) for path in existing]
            heights = {round(float(payload["initial_updown"]), 8) for payload in payloads}
            modes = {str(payload["grasp_mode"]) for payload in payloads}
            if len(heights) != 1 or len(modes) != 1:
                raise RuntimeError(f"resumed group {group} has incompatible payloads")
            removed.update(group)
            summary.append({
                "group": list(group), "height": heights.pop(), "mode": modes.pop(),
                "resumed": True, "payloads": [str(path) for path in existing],
            })
            print(f"RESUME yaw={args.yaw_deg:+.0f} group={group}", flush=True)
            continue

        baseline_group = [baseline_tasks[box_id] for box_id in group]
        baseline_modes = unique([str(task["mode"]) for task in baseline_group])
        if len(baseline_modes) != 1:
            raise RuntimeError(f"baseline group {group} has mixed grasp modes")
        row = row_for(group[0])
        modes = unique([baseline_modes[0], *("front", "top_suction")])
        if row <= 3:
            modes = [mode for mode in modes if mode == "front"]
        baseline_height = float(baseline_group[0]["updown"])
        heights = unique([baseline_height, *HEIGHTS[row]])
        seed_sources = unique([
            "continuation" if continuation_tasks else "baseline",
            "baseline",
        ])
        accepted: tuple[float, str, str, list[Path]] | None = None
        attempts: list[dict[str, Any]] = []

        for seed_source in seed_sources:
            seed_tasks = continuation_tasks if seed_source == "continuation" else baseline_tasks
            for mode in modes:
                for height in heights:
                    candidate_dir = output_dir / "candidates" / (
                        f"group-{group_index:02d}-{seed_source}-{mode}-"
                        f"u{abs(int(round(height * 100))):03d}"
                    )
                    candidate_dir.mkdir(parents=True, exist_ok=True)
                    candidate_paths: list[Path] = []
                    successful = True
                    for box_id in group:
                        task = baseline_tasks[box_id]
                        side = str(task["side"])
                        seed = arm_seed_degrees(seed_tasks[box_id]["payload"], side)
                        payload = run_candidate(
                            box_id=box_id,
                            side=side,
                            height=height,
                            mode=mode,
                            removed=removed,
                            yaw_rad=yaw_rad,
                            ik_seed=seed,
                            candidate_dir=candidate_dir,
                            timeout_s=args.timeout_s,
                        )
                        path = candidate_dir / f"box-{box_id:02d}.json"
                        candidate_paths.append(path)
                        attempts.append({
                            "box_id": box_id,
                            "seed_source": seed_source,
                            "mode": mode,
                            "height": height,
                            "success": bool(payload.get("success")),
                            "failure_stage": payload.get("failure_stage", ""),
                            "failure_reason": payload.get("failure_reason", ""),
                        })
                        if not payload.get("success"):
                            print(
                                f"REJECT yaw={args.yaw_deg:+.0f} group={group} "
                                f"seed={seed_source} mode={mode} updown={height:.2f} "
                                f"box={box_id} stage={payload.get('failure_stage')}",
                                flush=True,
                            )
                            successful = False
                            break
                    if successful:
                        accepted = (height, mode, seed_source, candidate_paths)
                        break
                if accepted:
                    break
            if accepted:
                break
        if not accepted:
            write_json(output_dir / "pickup-attempts.json", attempts)
            raise RuntimeError(
                f"no V3.2.2 pickup solution at yaw={args.yaw_deg:+.0f} for group {group}"
            )

        height, mode, seed_source, paths = accepted
        final_paths = []
        for box_id, path in zip(group, paths):
            final = accepted_dir / f"box-{box_id:02d}.json"
            shutil.copyfile(path, final)
            final_paths.append(str(final))
        removed.update(group)
        summary.append({
            "group": list(group), "height": height, "mode": mode,
            "seed_source": seed_source, "removed_after": sorted(removed),
            "payloads": final_paths,
        })
        write_json(output_dir / "pickup-summary.json", summary)
        print(
            f"ACCEPT yaw={args.yaw_deg:+.0f} group={group} mode={mode} "
            f"updown={height:.2f} completed={len(removed)}/25",
            flush=True,
        )

    print(
        f"RESULT yaw={args.yaw_deg:+.0f} success=25/25 "
        f"summary={output_dir / 'pickup-summary.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
