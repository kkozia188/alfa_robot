#!/usr/bin/python3
"""Plan fresh release-to-home VAMP RRT paths for current-run payloads."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
TOOL_DIR = WORKSPACE / "tools/v322_chest_carry_fast_base"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import plan_chest_payloads as chest
import scan_chest_carry_box_wall as scan


HOME = [
    -0.3, 0.0,
    *[math.radians(value) for value in (155, -105, 20, 90, -90, -40, 0)],
    *[math.radians(value) for value in (-155, 105, -20, -90, 90, 40, 0)],
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    payload = read_json(args.payload.resolve())
    box_id = int(payload["target_box_id"])
    release = next(
        frame for frame in payload["frames"] if frame["stage"] == "release_at_place"
    )
    planner_args = chest.planner_args(
        {"box_id": box_id, "transition_frames": [1]},
        payload, payload["place_tcp_pose"], None, None,
        False, 1, True, "vamp", 4, 100000, 200000, 0.08, args.seed,
    )
    result = scan.run_attempt(
        planner_args, box_id, str(payload["grasp_mode"]),
        set(int(value) for value in payload["removed_box_ids"]),
        str(payload["side"]), float(payload["initial_updown"]),
        tuple(float(value) for value in payload["box_center"]),
        args.output.resolve(),
        transition_from_joints=[float(value) for value in release["joints"]],
        transition_to_joints=HOME,
    )
    print(
        f"STOW box={box_id:02d} success={result['success']} "
        f"duration_ms={result.get('duration_ms')} "
        f"stage={result.get('failure_stage', '')}"
    )
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
