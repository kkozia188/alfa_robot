#!/usr/bin/env python3
"""Contract checks for the isolated V3.2.2 yaw-robust 5x5 planner."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def main() -> None:
    source = (ROOT / "src" / "v3_yaw_single_arm_box_extract_demo.cpp").read_text(
        encoding="utf-8"
    )
    launch = (
        ROOT / "launch" / "v3_yaw_single_arm_box_extract_demo.launch.py"
    ).read_text(encoding="utf-8")
    stage_source = (ROOT / "src" / "v3_single_arm_box_extract_demo.cpp").read_text(
        encoding="utf-8"
    )

    assert "V3RedundantArmModel::V322Left" in source
    assert "V3RedundantArmModel::V322Right" in source
    assert 'getParameter<std::string>("ik_seed_arm_joints_deg", "")' in source
    assert "use_configured_ik_seed" in source
    assert "request.seed = solver_seed" in source
    assert 'DeclareLaunchArgument("base_yaw"' in launch
    assert 'DeclareLaunchArgument("ik_seed_arm_joints_deg"' in launch
    assert 'executable="vehicle_pose_source_node"' in launch
    assert 'executable="v3_yaw_single_arm_box_extract_demo"' in launch

    # The new research entry must not replace the existing Stage Action server.
    assert "robot_motion_interfaces/action/execute_motion_stage.hpp" in stage_source
    assert '"/motion/execute_stage"' in stage_source
    assert "ik_seed_arm_joints_deg" not in stage_source


if __name__ == "__main__":
    main()
