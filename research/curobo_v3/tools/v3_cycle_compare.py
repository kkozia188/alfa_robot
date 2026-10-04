import argparse
import json
import time

from v3_interactive_60_tasks import (
    DEFAULT_ROBOT_CONFIG, DEFAULT_URDF, DEFAULT_NAMED_POSES,
    DEFAULT_BOX_FIT, DEFAULT_MOBILE_ROBOT_CONFIG,
)
from v3_full_cycle_planner import FullCyclePlanner, CycleBlocked


def main():
    args = argparse.Namespace(robot_config=DEFAULT_ROBOT_CONFIG, urdf=DEFAULT_URDF,
                              named_poses=DEFAULT_NAMED_POSES, box_fit=DEFAULT_BOX_FIT,
                              mobile_robot_config=DEFAULT_MOBILE_ROBOT_CONFIG)
    planner = FullCyclePlanner(args)
    results = {}
    for kind in ("informed_rrt", "rrt", "rrtconnect", "prm"):
        planner.planner_kind = kind
        started = time.perf_counter()
        try:
            result = planner.full_cycle(planner.task_options[0][1],
                                        lambda text: print(kind, text, flush=True))
        except CycleBlocked as error:
            result = error.partial or planner.last_partial
            result["blocked_stage"] = error.stage
            result["blocked_details"] = error.details
        except Exception as error:
            result = planner.last_partial
            result["blocked_stage"] = "程序异常"
            result["blocked_details"] = f"{type(error).__name__}: {error}"
        result["measured_wall_ms"] = (time.perf_counter() - started) * 1000
        results[kind] = result
        if kind == "informed_rrt" and result.get("selected_candidate"):
            planner.comparison_contact = result["contact_candidates"][result["selected_candidate"] - 1]
        print("RESULT", kind, result["success"], result.get("blocked_stage"),
              result["measured_wall_ms"], flush=True)
        path = DEFAULT_ROBOT_CONFIG.parent / "full_cycle_four_planners.json"
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        if not result["success"] and kind == "informed_rrt":
            break


if __name__ == "__main__":
    main()
