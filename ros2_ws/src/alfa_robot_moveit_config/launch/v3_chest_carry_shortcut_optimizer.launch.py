from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "alfa_robot", package_name="alfa_robot_moveit_config"
    ).to_moveit_configs()
    return LaunchDescription(
        [
            DeclareLaunchArgument("replay_json_path"),
            DeclareLaunchArgument("output_replay_json_path"),
            DeclareLaunchArgument("optimization_report_json_path"),
            DeclareLaunchArgument("edge_joint_step_deg", default_value="1.0"),
            DeclareLaunchArgument("edge_updown_step_m", default_value="0.01"),
            DeclareLaunchArgument("collision_inset", default_value="0.002"),
            DeclareLaunchArgument(
                "maximum_carried_box_tilt_deg", default_value="8.0"
            ),
            DeclareLaunchArgument("local_rrt_mode", default_value="full"),
            DeclareLaunchArgument("local_rrt_time_s", default_value="0.95"),
            DeclareLaunchArgument("manifold_alpha_max", default_value="1.25"),
            DeclareLaunchArgument("manifold_levels", default_value="64"),
            DeclareLaunchArgument("manifold_alpha_samples", default_value="11"),
            DeclareLaunchArgument(
                "manifold_alpha_neighbor_steps", default_value="1"
            ),
            Node(
                package="alfa_robot_moveit_config",
                executable="v3_chest_carry_shortcut_optimizer",
                output="screen",
                parameters=[
                    moveit_config.to_dict(),
                    {
                        "replay_json_path": LaunchConfiguration(
                            "replay_json_path"
                        ),
                        "output_replay_json_path": LaunchConfiguration(
                            "output_replay_json_path"
                        ),
                        "optimization_report_json_path": LaunchConfiguration(
                            "optimization_report_json_path"
                        ),
                        "edge_joint_step_deg": ParameterValue(
                            LaunchConfiguration("edge_joint_step_deg"),
                            value_type=float,
                        ),
                        "edge_updown_step_m": ParameterValue(
                            LaunchConfiguration("edge_updown_step_m"),
                            value_type=float,
                        ),
                        "collision_inset": ParameterValue(
                            LaunchConfiguration("collision_inset"),
                            value_type=float,
                        ),
                        "maximum_carried_box_tilt_deg": ParameterValue(
                            LaunchConfiguration("maximum_carried_box_tilt_deg"),
                            value_type=float,
                        ),
                        "local_rrt_mode": LaunchConfiguration("local_rrt_mode"),
                        "local_rrt_time_s": ParameterValue(
                            LaunchConfiguration("local_rrt_time_s"),
                            value_type=float,
                        ),
                        "manifold_alpha_max": ParameterValue(
                            LaunchConfiguration("manifold_alpha_max"),
                            value_type=float,
                        ),
                        "manifold_levels": ParameterValue(
                            LaunchConfiguration("manifold_levels"),
                            value_type=int,
                        ),
                        "manifold_alpha_samples": ParameterValue(
                            LaunchConfiguration("manifold_alpha_samples"),
                            value_type=int,
                        ),
                        "manifold_alpha_neighbor_steps": ParameterValue(
                            LaunchConfiguration("manifold_alpha_neighbor_steps"),
                            value_type=int,
                        ),
                    },
                ],
            ),
        ]
    )
