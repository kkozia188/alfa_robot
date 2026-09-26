#!/usr/bin/env python3

import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE / "config" / "v3_scoop_5x5_station.json"
SOURCE = PACKAGE / "src" / "v3_single_arm_box_extract_demo.cpp"


def between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_scoop_hot_start_is_identity_independent_and_planner_order_is_locked():
    planning = json.loads(CONFIG.read_text())["planning"]
    assert "7" not in planning["box_overrides"]
    assert "20" not in planning["box_overrides"]
    assert planning["loaded_transfer_waypoint_start_deg"]
    assert planning["loaded_transfer_joint_waypoints_deg"]

    source = SOURCE.read_text()
    loaded_transfer = between(source, "RrtPlanResult planLoadedTransfer(", "void appendStates(")
    profiles = json.loads((PACKAGE / "config/v3_scoop_validated_waypoint_profiles.json").read_text())
    assert "box_id" not in json.dumps(profiles)
    assert "start_error(profile.start_deg)" in loaded_transfer
    assert "if (!waypoint_degrees)" in loaded_transfer
    assert "planRrt(scene, start_state, goal_state, direct_only, metrics)" in loaded_transfer
    for validation in (
        "next->satisfiesBounds(planning_group_)",
        "carriedBoxUpright(*next)",
        "collisionReason(scene, *next, metrics)",
        "edgeClear(",
        "std::make_shared<moveit::core::RobotState>(goal_state)",
    ):
        assert validation in loaded_transfer

    plan_rrt = between(source, "RrtPlanResult planRrt(", "RrtPlanResult planLoadedTransfer(")
    repair = between(source, "bool searchShortcutRepairRrt(", "RrtPlanResult planRrt(")
    one_joint = repair.index("for (size_t joint = 0U; joint < start_joints.size(); ++joint)")
    two_joint = repair.index("for (size_t second = first + 1U; second < start_joints.size(); ++second)")
    repair_call = plan_rrt.index("const bool repair_success = searchShortcutRepairRrt(")
    tcp = plan_rrt.index("const bool shortcut_success = traceToolSpaceSegment(")
    full_rrt = plan_rrt.index('request.planner_id = "RRTConnectkConfigDefault"')
    assert one_joint < two_joint
    assert repair_call < tcp < full_rrt


if __name__ == "__main__":
    test_scoop_hot_start_is_identity_independent_and_planner_order_is_locked()
    print("v3 scoop identity-independence gate: PASS")
