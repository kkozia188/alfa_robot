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
            DeclareLaunchArgument("visibility_graph_mode", default_value="full"),
            DeclareLaunchArgument("local_rrt_time_s", default_value="0.95"),
            DeclareLaunchArgument("manifold_alpha_max", default_value="1.25"),
            DeclareLaunchArgument("manifold_levels", default_value="64"),
            DeclareLaunchArgument("manifold_alpha_samples", default_value="11"),
            DeclareLaunchArgument(
                "manifold_alpha_neighbor_steps", default_value="1"
            ),
            DeclareLaunchArgument("vamp_workers", default_value="4"),
            DeclareLaunchArgument("vamp_max_iterations", default_value="1000"),
            DeclareLaunchArgument("vamp_max_samples", default_value="2000"),
            DeclareLaunchArgument(
                "vamp_secondary_max_iterations", default_value="300"
            ),
            DeclareLaunchArgument(
                "vamp_prune_at_metric_lower_bound", default_value="true"
            ),
            DeclareLaunchArgument(
                "vamp_max_unsearched_improvement_ratio", default_value="0.0"
            ),
            DeclareLaunchArgument("vamp_range", default_value="0.08"),
            DeclareLaunchArgument("vamp_max_refinements", default_value="8"),
            DeclareLaunchArgument(
                "full_vamp_base_xy_margin_m", default_value="0.15"
            ),
            DeclareLaunchArgument(
                "full_vamp_base_yaw_margin_deg", default_value="5.0"
            ),
            DeclareLaunchArgument(
                "full_vamp_arm_margin_deg", default_value="35.0"
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
                        "visibility_graph_mode": LaunchConfiguration(
                            "visibility_graph_mode"
                        ),
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
                        "vamp_workers": ParameterValue(
                            LaunchConfiguration("vamp_workers"),
                            value_type=int,
                        ),
                        "vamp_max_iterations": ParameterValue(
                            LaunchConfiguration("vamp_max_iterations"),
                            value_type=int,
                        ),
                        "vamp_max_samples": ParameterValue(
                            LaunchConfiguration("vamp_max_samples"),
                            value_type=int,
                        ),
                        "vamp_secondary_max_iterations": ParameterValue(
                            LaunchConfiguration("vamp_secondary_max_iterations"),
                            value_type=int,
                        ),
                        "vamp_prune_at_metric_lower_bound": ParameterValue(
                            LaunchConfiguration("vamp_prune_at_metric_lower_bound"),
                            value_type=bool,
                        ),
                        "vamp_max_unsearched_improvement_ratio": ParameterValue(
                            LaunchConfiguration(
                                "vamp_max_unsearched_improvement_ratio"
                            ),
                            value_type=float,
                        ),
                        "vamp_range": ParameterValue(
                            LaunchConfiguration("vamp_range"),
                            value_type=float,
                        ),
                        "vamp_max_refinements": ParameterValue(
                            LaunchConfiguration("vamp_max_refinements"),
                            value_type=int,
                        ),
                        "full_vamp_base_xy_margin_m": ParameterValue(
                            LaunchConfiguration("full_vamp_base_xy_margin_m"),
                            value_type=float,
                        ),
                        "full_vamp_base_yaw_margin_deg": ParameterValue(
                            LaunchConfiguration("full_vamp_base_yaw_margin_deg"),
                            value_type=float,
                        ),
                        "full_vamp_arm_margin_deg": ParameterValue(
                            LaunchConfiguration("full_vamp_arm_margin_deg"),
                            value_type=float,
                        ),
                    },
                ],
            ),
        ]
    )
