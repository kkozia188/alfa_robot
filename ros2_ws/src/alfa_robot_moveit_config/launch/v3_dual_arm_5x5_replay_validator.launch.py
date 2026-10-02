from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    replay_json_path = LaunchConfiguration("replay_json_path")
    result_json_path = LaunchConfiguration("result_json_path")
    moveit_config = MoveItConfigsBuilder(
        "alfa_robot", package_name="alfa_robot_moveit_config"
    ).to_moveit_configs()
    return LaunchDescription(
        [
            DeclareLaunchArgument("replay_json_path"),
            DeclareLaunchArgument("result_json_path"),
            DeclareLaunchArgument("edge_joint_step_deg", default_value="1.0"),
            DeclareLaunchArgument("edge_updown_step_m", default_value="0.01"),
            DeclareLaunchArgument("collision_inset", default_value="0.002"),
            DeclareLaunchArgument("maximum_carried_box_tilt_deg", default_value="95.0"),
            Node(
                package="alfa_robot_moveit_config",
                executable="v3_dual_arm_5x5_replay_validator",
                output="screen",
                parameters=[
                    moveit_config.to_dict(),
                    {
                        "replay_json_path": replay_json_path,
                        "result_json_path": result_json_path,
                        "edge_joint_step_deg": ParameterValue(
                            LaunchConfiguration("edge_joint_step_deg"), value_type=float
                        ),
                        "edge_updown_step_m": ParameterValue(
                            LaunchConfiguration("edge_updown_step_m"), value_type=float
                        ),
                        "collision_inset": ParameterValue(
                            LaunchConfiguration("collision_inset"), value_type=float
                        ),
                        "maximum_carried_box_tilt_deg": ParameterValue(
                            LaunchConfiguration("maximum_carried_box_tilt_deg"),
                            value_type=float,
                        ),
                    },
                ],
            ),
        ]
    )
