"""V3 fixed-wall Motion Action server for simulation integration."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    planner_launch = Path(get_package_share_directory("alfa_robot_moveit_config")) / "launch" / "v3_box_wall_grasp_demo.launch.py"
    return LaunchDescription(
        [
            DeclareLaunchArgument("x", default_value="0.90"),
            DeclareLaunchArgument("execution_backend", default_value="replay", choices=["replay", "fjt"]),
            DeclareLaunchArgument(
                "follow_joint_trajectory_action",
                default_value="/whole_body_jtc/follow_joint_trajectory",
            ),
            DeclareLaunchArgument("start_rviz", default_value="false"),
            DeclareLaunchArgument("start_rerun", default_value="true"),
            DeclareLaunchArgument("spawn_viewer", default_value="true"),
            DeclareLaunchArgument("rerun_recording_path", default_value=""),
            DeclareLaunchArgument(
                "trajectory_cache_file",
                default_value=str(
                    Path(get_package_share_directory("alfa_robot_moveit_config"))
                    / "config" / "v3_stage_wall_cache.json"
                ),
            ),
            DeclareLaunchArgument("display_rate_hz", default_value="200.0"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(planner_launch)),
                launch_arguments={
                    "x": LaunchConfiguration("x"),
                    "box_id": "20",
                    "arm": "auto",
                    "auto_run_once": "false",
                    "sequence_mode": "false",
                    "initial_pose": "home",
                    "rear_placement_strategy": "named_unloading",
                    "height_strategy": "comfort_radius",
                    "comfort_ratio_min": "1.10",
                    "comfort_ratio_preferred": "1.15",
                    "comfort_ratio_max": "1.15",
                    "top_shoulder_above_wrist": "0.10",
                    "wall_bottom_z": "0.000001",
                    "planning_seed": "104729",
                    "enable_stage_action": "true",
                    "playback_enabled": "false",
                    "execution_backend": LaunchConfiguration("execution_backend"),
                    "follow_joint_trajectory_action": LaunchConfiguration(
                        "follow_joint_trajectory_action"
                    ),
                    "trajectory_cache_file": LaunchConfiguration("trajectory_cache_file"),
                    "start_rviz": LaunchConfiguration("start_rviz"),
                    "start_rerun": LaunchConfiguration("start_rerun"),
                    "spawn_viewer": LaunchConfiguration("spawn_viewer"),
                    "rerun_recording_path": LaunchConfiguration(
                        "rerun_recording_path"
                    ),
                    "display_rate_hz": LaunchConfiguration("display_rate_hz"),
                }.items(),
            ),
        ]
    )
