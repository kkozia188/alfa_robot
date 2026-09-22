from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    end_effector = LaunchConfiguration("end_effector")
    moveit_config = (
        MoveItConfigsBuilder("alfa_robot", package_name="alfa_robot_moveit_config")
        .robot_description(mappings={"end_effector": end_effector})
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    start_rviz = LaunchConfiguration("start_rviz")
    start_rerun = LaunchConfiguration("start_rerun")
    rviz_config = LaunchConfiguration("rviz_config")
    recording_path = LaunchConfiguration("rerun_recording_path")
    auto_run_once = LaunchConfiguration("auto_run_once")
    initial_box_x = LaunchConfiguration("initial_box_x")
    initial_box_y = LaunchConfiguration("initial_box_y")
    initial_box_z = LaunchConfiguration("initial_box_z")
    initial_updown = LaunchConfiguration("initial_updown")
    side = LaunchConfiguration("side")

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_rviz", default_value="true"),
            DeclareLaunchArgument("start_rerun", default_value="true"),
            DeclareLaunchArgument("end_effector", default_value="suction"),
            DeclareLaunchArgument("auto_run_once", default_value="false"),
            DeclareLaunchArgument("task_mode", default_value="full_extract"),
            DeclareLaunchArgument("ignore_opposite_arm", default_value="false"),
            DeclareLaunchArgument("continuous_sequence", default_value="false"),
            DeclareLaunchArgument("continuous_plan_approach", default_value="false"),
            DeclareLaunchArgument("maximum_carried_box_tilt_deg", default_value="180.0"),
            DeclareLaunchArgument("result_json_path", default_value=""),
            DeclareLaunchArgument("place_tcp_pose", default_value=""),
            DeclareLaunchArgument("place_arm_joints_deg", default_value=""),
            DeclareLaunchArgument("loaded_transfer_joint_waypoints_deg", default_value=""),
            DeclareLaunchArgument("loaded_transfer_waypoint_start_deg", default_value=""),
            DeclareLaunchArgument("transition_from_joints", default_value=""),
            DeclareLaunchArgument("transition_to_joints", default_value=""),
            DeclareLaunchArgument("initial_box_x", default_value="0.88"),
            DeclareLaunchArgument("initial_box_y", default_value="-0.20"),
            DeclareLaunchArgument("initial_box_z", default_value="0.55"),
            DeclareLaunchArgument("initial_updown", default_value="-0.3"),
            DeclareLaunchArgument("initial_arm_pose", default_value="v3_home"),
            DeclareLaunchArgument("initial_left_arm_joints_deg", default_value=""),
            DeclareLaunchArgument("initial_right_arm_joints_deg", default_value=""),
            DeclareLaunchArgument("side", default_value="left"),
            DeclareLaunchArgument("full_box_wall_scene", default_value="false"),
            DeclareLaunchArgument("box_grid_columns", default_value="5"),
            DeclareLaunchArgument("box_grid_rows", default_value="5"),
            DeclareLaunchArgument("box_grid_center_y", default_value="0.0"),
            DeclareLaunchArgument("box_grid_bottom_z", default_value="0.0"),
            DeclareLaunchArgument("target_box_id", default_value="0"),
            DeclareLaunchArgument("removed_box_ids", default_value=""),
            DeclareLaunchArgument("grasp_mode", default_value="front"),
            DeclareLaunchArgument("front_suction_y_offset", default_value="0.0"),
            DeclareLaunchArgument("front_suction_z_offset", default_value="0.0"),
            DeclareLaunchArgument("top_suction_x_offset", default_value="0.0"),
            DeclareLaunchArgument("contact_tool_roll_deg", default_value="0.0"),
            DeclareLaunchArgument("retreat_distance", default_value="0.35"),
            DeclareLaunchArgument("ground_enabled", default_value="true"),
            DeclareLaunchArgument("ground_surface_z", default_value="0.0"),
            DeclareLaunchArgument("ground_clearance", default_value="0.005"),
            DeclareLaunchArgument("ground_size_x", default_value="6.0"),
            DeclareLaunchArgument("ground_size_y", default_value="6.0"),
            DeclareLaunchArgument("ground_thickness", default_value="0.10"),
            DeclareLaunchArgument("warehouse_enabled", default_value="false"),
            DeclareLaunchArgument("warehouse_opening_x", default_value="-1.18"),
            DeclareLaunchArgument("warehouse_center_y", default_value="0.0"),
            DeclareLaunchArgument("warehouse_floor_z", default_value="0.0"),
            DeclareLaunchArgument("warehouse_length", default_value="2.38"),
            DeclareLaunchArgument("warehouse_width", default_value="2.38"),
            DeclareLaunchArgument("warehouse_height", default_value="2.35"),
            DeclareLaunchArgument("warehouse_wall_thickness", default_value="0.05"),
            DeclareLaunchArgument("psi_step_deg", default_value="5.0"),
            DeclareLaunchArgument("maximum_cartesian_joint_step_deg", default_value="15.0"),
            DeclareLaunchArgument("precontact_candidate_limit", default_value="8"),
            DeclareLaunchArgument("rrt_planning_time", default_value="1.0"),
            DeclareLaunchArgument("rrt_planning_attempts", default_value="1"),
            DeclareLaunchArgument("natural_motion_enabled", default_value="true"),
            DeclareLaunchArgument("natural_swivel_weight", default_value="0.02"),
            DeclareLaunchArgument("natural_wrist_singularity_weight", default_value="0.03"),
            DeclareLaunchArgument("natural_wrist_neutral_weight", default_value="0.02"),
            DeclareLaunchArgument("natural_joint_limit_weight", default_value="0.05"),
            DeclareLaunchArgument("natural_joint_wrap_weight", default_value="0.0"),
            DeclareLaunchArgument("natural_seed_swivel_sampling", default_value="false"),
            DeclareLaunchArgument("natural_seed_swivel_step_deg", default_value="1.0"),
            DeclareLaunchArgument("natural_seed_swivel_neighbor_steps", default_value="2"),
            DeclareLaunchArgument("natural_joint_acceleration_weight", default_value="0.0"),
            DeclareLaunchArgument("natural_place_return_weight", default_value="0.0"),
            DeclareLaunchArgument("natural_cartesian_replay_step_deg", default_value="0.0"),
            DeclareLaunchArgument("natural_rrt_shortcut_enabled", default_value="false"),
            DeclareLaunchArgument("natural_rrt_shortcut_max_nodes", default_value="0"),
            DeclareLaunchArgument("cartesian_transfer_search_enabled", default_value="false"),
            DeclareLaunchArgument("cartesian_transfer_translation_step", default_value="0.02"),
            DeclareLaunchArgument("cartesian_transfer_rotation_step_deg", default_value="2.0"),
            DeclareLaunchArgument("cartesian_transfer_max_search_attempts", default_value="0"),
            DeclareLaunchArgument("shortcut_repair_rrt_enabled", default_value="false"),
            DeclareLaunchArgument("shortcut_repair_rrt_budget_ms", default_value="1800.0"),
            DeclareLaunchArgument("shortcut_repair_rrt_max_samples", default_value="240"),
            DeclareLaunchArgument("shortcut_repair_max_joint_offset_deg", default_value="35.0"),
            DeclareLaunchArgument("place_updown_enabled", default_value="false"),
            DeclareLaunchArgument("place_updown", default_value="0.0"),
            DeclareLaunchArgument("top_loaded_transfer_direct_only", default_value="false"),
            DeclareLaunchArgument("natural_max_proximal_step_deg", default_value="12.0"),
            DeclareLaunchArgument("natural_max_wrist_step_deg", default_value="8.0"),
            DeclareLaunchArgument("natural_rrt_replay_step_deg", default_value="3.0"),
            DeclareLaunchArgument("analytic_path_only", default_value="false"),
            DeclareLaunchArgument("rerun_recording_path", default_value=""),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=str(
                    moveit_config.package_path
                    / "config"
                    / "v3_single_arm_box_extract_demo.rviz"
                ),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[moveit_config.robot_description],
                remappings=[
                    (
                        "/joint_states",
                        "/v3_single_arm_box_extract_demo/joint_states",
                    )
                ],
            ),
            Node(
                package="alfa_robot_moveit_config",
                executable="v3_single_arm_box_extract_demo",
                output="screen",
                parameters=[
                    moveit_config.to_dict(),
                    {
                        "auto_run_once": auto_run_once,
                        "task_mode": LaunchConfiguration("task_mode"),
                        "ignore_opposite_arm": LaunchConfiguration("ignore_opposite_arm"),
                        "continuous_sequence": ParameterValue(
                            LaunchConfiguration("continuous_sequence"), value_type=bool
                        ),
                        "continuous_plan_approach": ParameterValue(
                            LaunchConfiguration("continuous_plan_approach"), value_type=bool
                        ),
                        "maximum_carried_box_tilt_deg": ParameterValue(
                            LaunchConfiguration("maximum_carried_box_tilt_deg"), value_type=float
                        ),
                        "result_json_path": LaunchConfiguration("result_json_path"),
                        "place_tcp_pose": ParameterValue(
                            LaunchConfiguration("place_tcp_pose"), value_type=str
                        ),
                        "place_arm_joints_deg": ParameterValue(
                            LaunchConfiguration("place_arm_joints_deg"), value_type=str
                        ),
                        "loaded_transfer_joint_waypoints_deg": ParameterValue(
                            LaunchConfiguration("loaded_transfer_joint_waypoints_deg"),
                            value_type=str,
                        ),
                        "loaded_transfer_waypoint_start_deg": ParameterValue(
                            LaunchConfiguration("loaded_transfer_waypoint_start_deg"),
                            value_type=str,
                        ),
                        "transition_from_joints": ParameterValue(
                            LaunchConfiguration("transition_from_joints"), value_type=str
                        ),
                        "transition_to_joints": ParameterValue(
                            LaunchConfiguration("transition_to_joints"), value_type=str
                        ),
                        "side": side,
                        "initial_box_x": ParameterValue(initial_box_x, value_type=float),
                        "initial_box_y": ParameterValue(initial_box_y, value_type=float),
                        "initial_box_z": ParameterValue(initial_box_z, value_type=float),
                        "initial_updown": ParameterValue(initial_updown, value_type=float),
                        "initial_arm_pose": LaunchConfiguration("initial_arm_pose"),
                        "initial_left_arm_joints_deg": ParameterValue(
                            LaunchConfiguration("initial_left_arm_joints_deg"), value_type=str
                        ),
                        "initial_right_arm_joints_deg": ParameterValue(
                            LaunchConfiguration("initial_right_arm_joints_deg"), value_type=str
                        ),
                        "full_box_wall_scene": LaunchConfiguration("full_box_wall_scene"),
                        "box_grid_columns": ParameterValue(
                            LaunchConfiguration("box_grid_columns"), value_type=int
                        ),
                        "box_grid_rows": ParameterValue(
                            LaunchConfiguration("box_grid_rows"), value_type=int
                        ),
                        "box_grid_center_y": ParameterValue(
                            LaunchConfiguration("box_grid_center_y"), value_type=float
                        ),
                        "box_grid_bottom_z": ParameterValue(
                            LaunchConfiguration("box_grid_bottom_z"), value_type=float
                        ),
                        "target_box_id": ParameterValue(
                            LaunchConfiguration("target_box_id"), value_type=int
                        ),
                        "removed_box_ids": ParameterValue(
                            LaunchConfiguration("removed_box_ids"), value_type=str
                        ),
                        "grasp_mode": LaunchConfiguration("grasp_mode"),
                        "front_suction_y_offset": ParameterValue(
                            LaunchConfiguration("front_suction_y_offset"), value_type=float
                        ),
                        "front_suction_z_offset": ParameterValue(
                            LaunchConfiguration("front_suction_z_offset"), value_type=float
                        ),
                        "top_suction_x_offset": ParameterValue(
                            LaunchConfiguration("top_suction_x_offset"), value_type=float
                        ),
                        "contact_tool_roll_deg": ParameterValue(
                            LaunchConfiguration("contact_tool_roll_deg"), value_type=float
                        ),
                        "retreat_distance": ParameterValue(
                            LaunchConfiguration("retreat_distance"), value_type=float
                        ),
                        "ground_enabled": LaunchConfiguration("ground_enabled"),
                        "ground_surface_z": ParameterValue(
                            LaunchConfiguration("ground_surface_z"), value_type=float
                        ),
                        "ground_clearance": ParameterValue(
                            LaunchConfiguration("ground_clearance"), value_type=float
                        ),
                        "ground_size_x": ParameterValue(
                            LaunchConfiguration("ground_size_x"), value_type=float
                        ),
                        "ground_size_y": ParameterValue(
                            LaunchConfiguration("ground_size_y"), value_type=float
                        ),
                        "ground_thickness": ParameterValue(
                            LaunchConfiguration("ground_thickness"), value_type=float
                        ),
                        "warehouse_enabled": LaunchConfiguration("warehouse_enabled"),
                        "warehouse_opening_x": ParameterValue(
                            LaunchConfiguration("warehouse_opening_x"), value_type=float
                        ),
                        "warehouse_center_y": ParameterValue(
                            LaunchConfiguration("warehouse_center_y"), value_type=float
                        ),
                        "warehouse_floor_z": ParameterValue(
                            LaunchConfiguration("warehouse_floor_z"), value_type=float
                        ),
                        "warehouse_length": ParameterValue(
                            LaunchConfiguration("warehouse_length"), value_type=float
                        ),
                        "warehouse_width": ParameterValue(
                            LaunchConfiguration("warehouse_width"), value_type=float
                        ),
                        "warehouse_height": ParameterValue(
                            LaunchConfiguration("warehouse_height"), value_type=float
                        ),
                        "warehouse_wall_thickness": ParameterValue(
                            LaunchConfiguration("warehouse_wall_thickness"), value_type=float
                        ),
                        "psi_step_deg": ParameterValue(
                            LaunchConfiguration("psi_step_deg"), value_type=float
                        ),
                        "maximum_cartesian_joint_step_deg": ParameterValue(
                            LaunchConfiguration("maximum_cartesian_joint_step_deg"), value_type=float
                        ),
                        "precontact_candidate_limit": ParameterValue(
                            LaunchConfiguration("precontact_candidate_limit"), value_type=int
                        ),
                        "rrt_planning_time": ParameterValue(
                            LaunchConfiguration("rrt_planning_time"), value_type=float
                        ),
                        "rrt_planning_attempts": ParameterValue(
                            LaunchConfiguration("rrt_planning_attempts"), value_type=int
                        ),
                        "natural_motion_enabled": LaunchConfiguration("natural_motion_enabled"),
                        "natural_swivel_weight": ParameterValue(
                            LaunchConfiguration("natural_swivel_weight"), value_type=float
                        ),
                        "natural_wrist_singularity_weight": ParameterValue(
                            LaunchConfiguration("natural_wrist_singularity_weight"), value_type=float
                        ),
                        "natural_wrist_neutral_weight": ParameterValue(
                            LaunchConfiguration("natural_wrist_neutral_weight"), value_type=float
                        ),
                        "natural_joint_limit_weight": ParameterValue(
                            LaunchConfiguration("natural_joint_limit_weight"), value_type=float
                        ),
                        "natural_joint_wrap_weight": ParameterValue(
                            LaunchConfiguration("natural_joint_wrap_weight"), value_type=float
                        ),
                        "natural_seed_swivel_sampling": LaunchConfiguration(
                            "natural_seed_swivel_sampling"
                        ),
                        "natural_seed_swivel_step_deg": ParameterValue(
                            LaunchConfiguration("natural_seed_swivel_step_deg"), value_type=float
                        ),
                        "natural_seed_swivel_neighbor_steps": ParameterValue(
                            LaunchConfiguration("natural_seed_swivel_neighbor_steps"), value_type=int
                        ),
                        "natural_joint_acceleration_weight": ParameterValue(
                            LaunchConfiguration("natural_joint_acceleration_weight"), value_type=float
                        ),
                        "natural_place_return_weight": ParameterValue(
                            LaunchConfiguration("natural_place_return_weight"), value_type=float
                        ),
                        "natural_cartesian_replay_step_deg": ParameterValue(
                            LaunchConfiguration("natural_cartesian_replay_step_deg"), value_type=float
                        ),
                        "natural_rrt_shortcut_enabled": LaunchConfiguration(
                            "natural_rrt_shortcut_enabled"
                        ),
                        "natural_rrt_shortcut_max_nodes": ParameterValue(
                            LaunchConfiguration("natural_rrt_shortcut_max_nodes"), value_type=int
                        ),
                        "cartesian_transfer_search_enabled": LaunchConfiguration(
                            "cartesian_transfer_search_enabled"
                        ),
                        "cartesian_transfer_translation_step": ParameterValue(
                            LaunchConfiguration("cartesian_transfer_translation_step"),
                            value_type=float,
                        ),
                        "cartesian_transfer_rotation_step_deg": ParameterValue(
                            LaunchConfiguration("cartesian_transfer_rotation_step_deg"),
                            value_type=float,
                        ),
                        "cartesian_transfer_max_search_attempts": ParameterValue(
                            LaunchConfiguration("cartesian_transfer_max_search_attempts"),
                            value_type=int,
                        ),
                        "shortcut_repair_rrt_enabled": LaunchConfiguration(
                            "shortcut_repair_rrt_enabled"
                        ),
                        "shortcut_repair_rrt_budget_ms": ParameterValue(
                            LaunchConfiguration("shortcut_repair_rrt_budget_ms"), value_type=float
                        ),
                        "shortcut_repair_rrt_max_samples": ParameterValue(
                            LaunchConfiguration("shortcut_repair_rrt_max_samples"), value_type=int
                        ),
                        "shortcut_repair_max_joint_offset_deg": ParameterValue(
                            LaunchConfiguration("shortcut_repair_max_joint_offset_deg"),
                            value_type=float,
                        ),
                        "place_updown_enabled": LaunchConfiguration("place_updown_enabled"),
                        "place_updown": ParameterValue(
                            LaunchConfiguration("place_updown"), value_type=float
                        ),
                        "top_loaded_transfer_direct_only": LaunchConfiguration(
                            "top_loaded_transfer_direct_only"
                        ),
                        "natural_max_proximal_step_deg": ParameterValue(
                            LaunchConfiguration("natural_max_proximal_step_deg"), value_type=float
                        ),
                        "natural_max_wrist_step_deg": ParameterValue(
                            LaunchConfiguration("natural_max_wrist_step_deg"), value_type=float
                        ),
                        "natural_rrt_replay_step_deg": ParameterValue(
                            LaunchConfiguration("natural_rrt_replay_step_deg"), value_type=float
                        ),
                        "analytic_path_only": LaunchConfiguration("analytic_path_only"),
                    },
                ],
            ),
            Node(
                package="alfa_robot_rerun",
                executable="v3_single_arm_box_extract_viewer",
                output="screen",
                parameters=[
                    {
                        "recording_path": recording_path,
                        "spawn_viewer": True,
                    }
                ],
                condition=IfCondition(start_rerun),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(start_rviz),
            ),
        ]
    )
