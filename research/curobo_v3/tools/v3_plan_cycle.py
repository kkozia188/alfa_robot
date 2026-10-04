"""Headless planning demo. Produces untimed joint frames, never hardware commands."""

import argparse
import json
from pathlib import Path
import sys

from curobo_core.contracts import PlanRequest, PlannerAssets
from curobo_core.planner import FullCyclePlanner


def main():
    defaults = PlannerAssets()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("robot_config", "mobile_robot_config", "urdf", "named_poses", "box_fit", "target_poses"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=getattr(defaults, name))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--left-box", type=int, default=24)
    parser.add_argument("--right-box", type=int, default=20)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    planner = FullCyclePlanner(args)
    request = PlanRequest.from_dict(json.loads(args.request.read_text())) if args.request else planner.demo_request(
        {"left": args.left_box, "right": args.right_box})
    results = []
    for _ in range(args.runs):
        result = planner.plan_request(request, lambda message: print(message, file=sys.stderr, flush=True))
        results.append(result)
        if not result["success"]:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"request": request.to_dict(), "results": results},
                                      ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{key: result.get(key) for key in (
        "success", "total_ms", "selected_candidate", "final_home_error", "error", "scene_id")}
        for result in results], ensure_ascii=False), flush=True)
    if not results[-1]["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
