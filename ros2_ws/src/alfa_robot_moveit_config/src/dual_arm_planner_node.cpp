/**
 * dual_arm_planner_node.cpp
 *
 * MoveIt-backed dual-arm box-stack flow reproducer.
 *
 * Grasp IK uses the project benchmark solver:
 *   fixed discrete h candidates × multiple BioIK seeds × cost scoring.
 * MoveIt is only used for joint-space trajectory planning/execution.
 */

#include "ik_benchmark/parallel_updown_aware_ik_solver.h"

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_state/robot_state.h>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct BoxSpec
{
  int id = 0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct PickPair
{
  int round = 0;
  int left_box = 0;
  int right_box = 0;
  bool top_suction = false;
};

struct ContainerPanel
{
  std::string id;
  std::array<double, 3> center;
  std::array<double, 3> size;
};

struct StaticBoxObstacle
{
  std::string id;
  std::array<double, 3> center;
  std::array<double, 3> size;
};

struct AttachedBoxSpec
{
  std::string id;
  std::string link_name;
  std::array<double, 3> center_in_link;
  std::array<double, 3> size;
};

struct AxisAlignedBox
{
  std::array<double, 3> center;
  std::array<double, 3> size;
};


struct ExtractCandidate
{
  size_t step_index = 0;
  size_t candidate_index = 0;
  double retreat_x = 0.0;
  double retreat_delta_x = 0.0;
  double lift_z = 0.0;
  double lift_delta_z = 0.0;
  double pitch_up_rad = 0.0;
  double pitch_delta_rad = 0.0;
  geometry_msgs::msg::Pose target_pose;
  bool ik_success = false;
  bool state_valid = false;
  bool carried_clear = false;
  bool detached_from_neighbors = false;
  std::string rejection_reason;
  moveit::core::RobotStatePtr state;
};

struct DualExtractStepCandidate
{
  ExtractCandidate left;
  ExtractCandidate right;
  moveit::core::RobotStatePtr state;
  bool state_valid = false;
  bool left_detached = false;
  bool right_detached = false;
  double score = std::numeric_limits<double>::infinity();
  std::string rejection_reason;
};

struct ArmExtractPath
{
  bool success = false;
  std::string failure_reason;
  std::vector<moveit::core::RobotStatePtr> states;
  std::vector<ExtractCandidate> selected_candidates;
  double final_retreat_x = 0.0;
  double final_lift_z = 0.0;
  double final_pitch_deg = 0.0;
  size_t accepted_steps = 0;
  size_t failed_steps = 0;
};

struct ExtractMotionDelta
{
  double retreat_ratio = 1.0;
  double lift_ratio = 0.0;
  double pitch_delta_deg = 0.0;
};

std::vector<double> deg_to_rad(const std::vector<double>& degrees)
{
  std::vector<double> radians;
  radians.reserve(degrees.size());
  for (double degree : degrees) {
    radians.push_back(degree * M_PI / 180.0);
  }
  return radians;
}

std::string trim_copy(std::string value)
{
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<double> parse_degrees_list(std::string value)
{
  value = trim_copy(value);
  if (!value.empty() && value.front() == '[') value.erase(value.begin());
  if (!value.empty() && value.back() == ']') value.pop_back();
  std::vector<double> result;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    token = trim_copy(token);
    if (!token.empty()) {
      result.push_back(std::stod(token));
    }
  }
  return result;
}

std::vector<std::vector<double>> parse_pose_family_degrees(const std::string& value)
{
  std::vector<std::vector<double>> family;
  std::stringstream stream(value);
  std::string segment;
  while (std::getline(stream, segment, ';')) {
    auto pose_deg = parse_degrees_list(segment);
    if (pose_deg.size() != 6) {
      continue;
    }
    family.push_back(deg_to_rad(pose_deg));
  }
  return family;
}

nlohmann::json pose_family_degrees_json(const std::vector<std::vector<double>>& family)
{
  nlohmann::json out = nlohmann::json::array();
  for (const auto& pose : family) {
    nlohmann::json pose_json = nlohmann::json::array();
    for (double value : pose) {
      pose_json.push_back(value * 180.0 / M_PI);
    }
    out.push_back(pose_json);
  }
  return out;
}

nlohmann::json pose_degrees_json(const std::vector<double>& pose)
{
  nlohmann::json out = nlohmann::json::array();
  for (double value : pose) {
    out.push_back(value * 180.0 / M_PI);
  }
  return out;
}

double shortest_angular_distance(double a, double b)
{
  return std::abs(std::atan2(std::sin(a - b), std::cos(a - b)));
}

std::string format_degrees(const std::vector<std::string>& names, const std::vector<double>& values)
{
  std::ostringstream oss;
  const size_t count = std::min(names.size(), values.size());
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) oss << ", ";
    oss << names[i] << "=" << values[i] * 180.0 / M_PI;
  }
  return oss.str();
}

Eigen::Quaterniond forward_x_orientation()
{
  // tool +Z -> world/base +X, same convention as the previous box-stack benchmark.
  return Eigen::Quaterniond(0.70710678, 0.0, 0.70710678, 0.0);
}

Eigen::Quaterniond top_suction_orientation()
{
  // tool +Z -> world/base -Z.
  return Eigen::Quaterniond(0.0, 1.0, 0.0, 0.0);
}

Eigen::Quaterniond pitch_up_orientation(double pitch_up_rad)
{
  // Relax the extraction grasp by pitching tool +Z upward in world/base frame.
  // Eigen's positive Y rotation maps +X toward -Z, which is a downward press for
  // our front grasp. Use the opposite sign so a positive parameter means "lift
  // the tool normal toward +Z".
  Eigen::Quaterniond q = Eigen::AngleAxisd(-pitch_up_rad, Eigen::Vector3d::UnitY()) * forward_x_orientation();
  q.normalize();
  return q;
}

geometry_msgs::msg::Pose make_pose(double x, double y, double z, const Eigen::Quaterniond& quat)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  Eigen::Quaterniond q = quat;
  q.normalize();
  pose.orientation.w = q.w();
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  return pose;
}

geometry_msgs::msg::Pose make_identity_pose(double x, double y, double z)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  pose.orientation.w = 1.0;
  return pose;
}

Eigen::Isometry3d pose_to_eigen(const geometry_msgs::msg::Pose& pose)
{
  Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  q.normalize();
  Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
  tf.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  tf.linear() = q.toRotationMatrix();
  return tf;
}

std::map<int, BoxSpec> make_boxes(double front_x)
{
  const std::vector<std::vector<std::pair<int, double>>> rows_top_to_bottom = {
    {{1, 0.8}, {2, 0.4}, {3, 0.0}, {4, -0.4}, {5, -0.8}},
    {{6, 0.8}, {7, 0.4}, {8, 0.0}, {9, -0.4}, {10, -0.8}},
    {{11, 0.8}, {12, 0.4}, {13, 0.0}, {14, -0.4}, {15, -0.8}},
    {{16, 0.8}, {17, 0.4}, {18, 0.0}, {19, -0.4}, {20, -0.8}},
    {{21, 0.8}, {22, 0.4}, {23, 0.0}, {24, -0.4}, {25, -0.8}},
  };

  std::map<int, BoxSpec> boxes;
  for (size_t row = 0; row < rows_top_to_bottom.size(); ++row) {
    const double z = 0.2 + 0.4 * static_cast<double>(rows_top_to_bottom.size() - 1 - row);
    for (const auto& [id, y] : rows_top_to_bottom[row]) {
      boxes[id] = BoxSpec{id, front_x, y, z};
    }
  }
  return boxes;
}

std::vector<PickPair> make_pick_pairs(bool include_top_suction)
{
  std::vector<PickPair> pairs = {
    {1, 2, 4, false},
    {2, 7, 9, false},
    {3, 12, 14, false},
    {4, 17, 19, false},
  };
  if (include_top_suction) {
    pairs.push_back({5, 22, 24, true});
  } else {
    pairs.push_back({5, 22, 24, false});
  }
  return pairs;
}

nlohmann::json pose_json(const geometry_msgs::msg::Pose& pose)
{
  return {
    {"position", {pose.position.x, pose.position.y, pose.position.z}},
    {"orientation_xyzw", {pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w}},
  };
}

nlohmann::json vector_json(const std::vector<double>& values)
{
  nlohmann::json out = nlohmann::json::array();
  for (double value : values) out.push_back(value);
  return out;
}

nlohmann::json names_values_json(const std::vector<std::string>& names, const std::vector<double>& values)
{
  nlohmann::json out = nlohmann::json::object();
  const size_t count = std::min(names.size(), values.size());
  for (size_t i = 0; i < count; ++i) out[names[i]] = values[i];
  return out;
}

double pose_position_error(const Eigen::Isometry3d& target, const Eigen::Isometry3d& actual)
{
  return (target.translation() - actual.translation()).norm();
}

double pose_orientation_error(const Eigen::Isometry3d& target, const Eigen::Isometry3d& actual)
{
  Eigen::AngleAxisd aa(target.linear().transpose() * actual.linear());
  return aa.angle();
}

}  // namespace

class DualArmPlannerNode : public rclcpp::Node
{
public:
  explicit DualArmPlannerNode(const rclcpp::NodeOptions& options)
  : Node("dual_arm_planner", options)
  {}

  void init()
  {
    planning_group_ = get_or_declare_parameter<std::string>("planning_group", "dual_v5_arm_with_base");
    left_tip_ = get_or_declare_parameter<std::string>("left_tip", "left_v5_tool0");
    right_tip_ = get_or_declare_parameter<std::string>("right_tip", "right_v5_tool0");
    execute_ = get_or_declare_parameter<bool>("execute", true);
    reject_ik_collisions_ = get_or_declare_parameter<bool>("reject_ik_collisions", false);
    check_goal_collision_ = get_or_declare_parameter<bool>("check_goal_collision", false);
    prefer_commanded_state_ = get_or_declare_parameter<bool>("prefer_commanded_state", true);
    fixed_updown_ = get_or_declare_parameter<double>("fixed_updown", 0.45);
    box_front_x_ = get_or_declare_parameter<double>("box_front_x", 0.625);
    world_to_base_z_ = get_or_declare_parameter<double>("world_to_base_z", 0.202094);
    top_suction_x_offset_ = get_or_declare_parameter<double>("top_suction_x_offset", 0.15);
    top_suction_z_offset_ = get_or_declare_parameter<double>("top_suction_z_offset", 0.2);
    max_rounds_ = get_or_declare_parameter<int>("max_rounds", 10);
    include_top_suction_ = get_or_declare_parameter<bool>("include_top_suction", true);
    ik_timeout_ = get_or_declare_parameter<double>("ik_timeout", 2.0);
    planning_time_ = get_or_declare_parameter<double>("planning_time", 8.0);
    planning_attempts_ = get_or_declare_parameter<int>("planning_attempts", 20);
    velocity_scale_ = get_or_declare_parameter<double>("velocity_scale", 1.0);
    acceleration_scale_ = get_or_declare_parameter<double>("acceleration_scale", 1.0);
    joint_goal_tolerance_rad_ = get_or_declare_parameter<double>("joint_goal_tolerance_rad", 0.02);
    state_wait_timeout_s_ = get_or_declare_parameter<double>("state_wait_timeout_s", 2.0);
    record_jsonl_path_ = get_or_declare_parameter<std::string>(
      "record_jsonl_path", "/mnt/mydisk/ALFA/alfa_robot/data/ik_benchmark/moveit_box_stack_flow/moveit_box_stack_flow.jsonl");
    record_trajectories_ = get_or_declare_parameter<bool>("record_trajectories", true);

    ik_config_.fixed_group = get_or_declare_parameter<std::string>("ik_fixed_group", "dual_v5_arm");
    ik_config_.free_group = get_or_declare_parameter<std::string>("ik_free_group", "dual_v5_arm_with_base");
    ik_config_.solver_plugin = get_or_declare_parameter<std::string>("ik_solver_plugin", "bio_ik/BioIKKinematicsPlugin");
    ik_config_.base_frame = get_or_declare_parameter<std::string>("ik_base_frame", "base_link");
    ik_config_.left_tip = left_tip_;
    ik_config_.right_tip = right_tip_;
    ik_config_.tool0_offset = get_or_declare_parameter<double>("ik_tool0_offset", 0.0);
    ik_config_.gripper_z_reach_lower = get_or_declare_parameter<double>("front_z_reach_lower", 0.45) - world_to_base_z_;
    ik_config_.gripper_z_reach_upper = get_or_declare_parameter<double>("front_z_reach_upper", 1.25) - world_to_base_z_;
    ik_config_.top_suction_z_reach_lower = get_or_declare_parameter<double>("top_z_reach_lower", 0.3) - world_to_base_z_;
    ik_config_.top_suction_z_reach_upper = get_or_declare_parameter<double>("top_z_reach_upper", 0.45) - world_to_base_z_;
    ik_config_.h_lower = get_or_declare_parameter<double>("ik_h_lower", 0.0);
    ik_config_.h_upper = get_or_declare_parameter<double>("ik_h_upper", 0.99);
    ik_config_.h_search_mode = ik_benchmark::UpdownAwareIkConfig::HSearchMode::FixedDiscrete;
    ik_config_.h_search_margin = get_or_declare_parameter<double>("ik_h_search_margin", 0.2);
    ik_config_.h_step = get_or_declare_parameter<double>("ik_h_step", 0.1);
    ik_config_.h_candidate_count = static_cast<size_t>(std::max(1, get_or_declare_parameter<int>("ik_h_candidate_count", 16)));
    ik_config_.seed_count = static_cast<size_t>(std::max(1, get_or_declare_parameter<int>("ik_seed_count", 32)));
    ik_config_.cost_loaded_family_distance =
      get_or_declare_parameter<double>("ik_loaded_family_distance_weight", 0.2);
    ik_config_.cost_loaded_preferred_distance =
      get_or_declare_parameter<double>("ik_loaded_preferred_distance_weight", 0.1);
    ik_config_.workers = static_cast<size_t>(std::max(1, get_or_declare_parameter<int>("ik_workers", 16)));
    ik_config_.timeout = get_or_declare_parameter<double>("ik_candidate_timeout", 0.01);
    ik_config_.try_target_orders = get_or_declare_parameter<bool>("ik_try_target_orders", false);
    ik_config_.use_reversed_target_order = get_or_declare_parameter<bool>("ik_use_reversed_target_order", true);
    ik_config_.check_tip_error = true;
    ik_config_.position_tolerance = get_or_declare_parameter<double>("ik_position_tolerance", 0.02);
    ik_config_.top_suction_position_tolerance = get_or_declare_parameter<double>("ik_top_position_tolerance", 0.04);
    ik_config_.orientation_tolerance = get_or_declare_parameter<double>("ik_orientation_tolerance", 0.05);
    ik_config_.top_suction_orientation_tolerance =
      get_or_declare_parameter<double>("ik_top_orientation_tolerance_deg", 7.0) * M_PI / 180.0;
    ik_config_.check_collision = get_or_declare_parameter<bool>("optimized_ik_check_collision", reject_ik_collisions_);
    ik_config_.enforce_arm_base_collisions = get_or_declare_parameter<bool>("ik_enforce_arm_base_collisions", true);
    ik_config_.reject_swapped_tips = get_or_declare_parameter<bool>("ik_reject_swapped_tips", true);
    ik_config_.fallback_enabled = get_or_declare_parameter<bool>("ik_fallback_enabled", false);
    ik_config_.fallback_seed_count = static_cast<size_t>(std::max(1, get_or_declare_parameter<int>("ik_fallback_seed_count", 64)));
    ik_config_.fallback_rounds = static_cast<size_t>(std::max(1, get_or_declare_parameter<int>("ik_fallback_rounds", 1)));
    ik_config_.fallback_timeout = get_or_declare_parameter<double>("ik_fallback_timeout", ik_config_.timeout);

    enable_container_obstacle_ = get_or_declare_parameter<bool>("enable_container_obstacle", true);
    container_frame_ = get_or_declare_parameter<std::string>("container_frame", "world");
    container_length_ = get_or_declare_parameter<double>("container_length", 4.0);
    container_width_ = get_or_declare_parameter<double>("container_width", 2.2);
    container_height_ = get_or_declare_parameter<double>("container_height", 2.4);
    container_center_x_ = get_or_declare_parameter<double>("container_center_x", 0.8);
    container_center_y_ = get_or_declare_parameter<double>("container_center_y", 0.0);
    container_floor_z_ = get_or_declare_parameter<double>("container_floor_z", 0.0);
    container_wall_thickness_ = get_or_declare_parameter<double>("container_wall_thickness", 0.02);
    enable_attached_box_collision_ = get_or_declare_parameter<bool>("enable_attached_box_collision", true);
    carried_box_depth_ = get_or_declare_parameter<double>("carried_box_depth", 0.3);
    carried_box_width_ = get_or_declare_parameter<double>("carried_box_width", 0.4);
    carried_box_height_ = get_or_declare_parameter<double>("carried_box_height", 0.4);
    enable_static_box_obstacles_ = get_or_declare_parameter<bool>("enable_static_box_obstacles", true);
    static_box_obstacle_inset_ = get_or_declare_parameter<double>("static_box_obstacle_inset", 0.002);

    extract_demo_left_box_id_ = get_or_declare_parameter<int>("extract_demo_left_box_id", 2);
    extract_demo_right_box_id_ = get_or_declare_parameter<int>("extract_demo_right_box_id", 4);
    extract_demo_all_rows_ = get_or_declare_parameter<bool>("extract_demo_all_rows", false);
    extract_step_x_ = get_or_declare_parameter<double>("extract_step_x", 0.03);
    extract_max_x_ = get_or_declare_parameter<double>("extract_max_x", 0.36);
    extract_lift_candidates_ = get_or_declare_parameter<std::vector<double>>("extract_lift_candidates", std::vector<double>{0.0, 0.02, 0.05, 0.08});
    extract_pitch_candidates_deg_ = get_or_declare_parameter<std::vector<double>>("extract_pitch_candidates_deg", std::vector<double>{0.0, 5.0, 10.0, 15.0});
    extract_neighbor_margin_ = get_or_declare_parameter<double>("extract_neighbor_margin", 0.02);
    extract_fail_fast_ = get_or_declare_parameter<bool>("extract_fail_fast", false);
    extract_success_extra_steps_ = static_cast<size_t>(
      std::max(0, get_or_declare_parameter<int>("extract_success_extra_steps", 3)));
    extract_kdl_timeout_ = get_or_declare_parameter<double>("extract_kdl_timeout", 0.01);
    extract_position_tolerance_ = get_or_declare_parameter<double>("extract_position_tolerance", 0.01);
    extract_orientation_tolerance_ = get_or_declare_parameter<double>("extract_orientation_tolerance", 0.05);
    extract_max_tip_z_drop_ = get_or_declare_parameter<double>("extract_max_tip_z_drop", 0.002);
    extract_min_tool_normal_z_ = get_or_declare_parameter<double>("extract_min_tool_normal_z", -1e-4);
    extract_score_lift_weight_ = get_or_declare_parameter<double>("extract_score_lift_weight", 10.0);
    extract_score_pitch_weight_ = get_or_declare_parameter<double>("extract_score_pitch_weight", 0.02);
    extract_score_retreat_continuity_weight_ = get_or_declare_parameter<double>("extract_score_retreat_continuity_weight", 0.2);
    extract_score_joint_delta_weight_ = get_or_declare_parameter<double>("extract_score_joint_delta_weight", 0.6);
    extract_score_tip_position_delta_weight_ = get_or_declare_parameter<double>("extract_score_tip_position_delta_weight", 2.0);
    extract_score_tip_orientation_delta_weight_ = get_or_declare_parameter<double>("extract_score_tip_orientation_delta_weight", 0.05);
    extract_max_joint_delta_ = get_or_declare_parameter<double>("extract_max_joint_delta", 0.0);
    extract_demo_direct_grasp_start_ = get_or_declare_parameter<bool>("extract_demo_direct_grasp_start", false);
    extract_grasp_ik_home_updown_ = get_or_declare_parameter<double>("extract_grasp_ik_home_updown", 0.3);
    extract_benchmark_all_legal_ik_ = get_or_declare_parameter<bool>("extract_benchmark_all_legal_ik", false);
    extract_benchmark_dual_arm_ = get_or_declare_parameter<bool>("extract_benchmark_dual_arm", false);
    extract_benchmark_dual_async_ = get_or_declare_parameter<bool>("extract_benchmark_dual_async", false);
    extract_benchmark_csv_path_ = get_or_declare_parameter<std::string>(
      "extract_benchmark_csv_path",
      "/mnt/mydisk/ALFA/alfa_robot/data/ik_benchmark/motion51_extract_replay/extract_all_legal_ik_timing.csv");
    extract_benchmark_record_rollouts_ = get_or_declare_parameter<bool>("extract_benchmark_record_rollouts", false);
    extract_benchmark_candidate_limit_ = static_cast<size_t>(
      std::max(0, get_or_declare_parameter<int>("extract_benchmark_candidate_limit", 0)));
    extract_benchmark_extract_workers_ = static_cast<size_t>(
      std::max(1, get_or_declare_parameter<int>("extract_benchmark_extract_workers", 1)));
    extract_ik_dedup_enabled_ = get_or_declare_parameter<bool>("extract_ik_dedup_enabled", false);
    extract_ik_dedup_joint_threshold_ =
      get_or_declare_parameter<double>("extract_ik_dedup_joint_threshold_deg", 1.0) * M_PI / 180.0;
    extract_ik_dedup_h_threshold_ = get_or_declare_parameter<double>("extract_ik_dedup_h_threshold", 0.005);
    extract_benchmark_plan_loaded_after_success_ =
      get_or_declare_parameter<bool>("extract_benchmark_plan_loaded_after_success", false);
    extract_loaded_planning_group_ =
      get_or_declare_parameter<std::string>("extract_loaded_planning_group", "dual_v5_arm_with_base");
    extract_loaded_planning_time_ = get_or_declare_parameter<double>("extract_loaded_planning_time", 1.0);
    extract_loaded_planning_attempts_ =
      std::max(1, get_or_declare_parameter<int>("extract_loaded_planning_attempts", 4));
    extract_loaded_candidate_limit_ = static_cast<size_t>(
      std::max(0, get_or_declare_parameter<int>("extract_loaded_candidate_limit", 0)));
    extract_loaded_sort_by_pose_distance_ =
      get_or_declare_parameter<bool>("extract_loaded_sort_by_pose_distance", false);
    extract_loaded_stop_on_first_success_ =
      get_or_declare_parameter<bool>("extract_loaded_stop_on_first_success", false);
    extract_loaded_target_updown_ = get_or_declare_parameter<double>("extract_loaded_target_updown", 0.3);
    record_tip_error_ik_candidates_ = get_or_declare_parameter<bool>("record_tip_error_ik_candidates", false);
    record_tip_error_ik_candidate_limit_ = static_cast<size_t>(
      std::max(0, get_or_declare_parameter<int>("record_tip_error_ik_candidate_limit", 80)));

    left_loaded_pose_family_ = parse_pose_family_degrees(get_or_declare_parameter<std::string>(
      "loaded_left_pose_family_deg",
      "[-0.0,59.04,-135.16,0.0,-76.13,0.0];[0.0,-75.0,135.0,0.0,60.0,0.0];[33.87,75.82,-135.08,0.0,-59.25,-33.87]"));
    right_loaded_pose_family_ = parse_pose_family_degrees(get_or_declare_parameter<std::string>(
      "loaded_right_pose_family_deg",
      "[0.0,58.88,-134.84,0.0,-75.96,0.0];[0.0,-75.0,135.0,0.0,60.0,0.0];[-30.93,74.17,-134.92,0.0,-60.74,30.93]"));
    if (left_loaded_pose_family_.empty()) {
      left_loaded_pose_family_.push_back(deg_to_rad({0, -75, 135, 0, 60, 0}));
    }
    if (right_loaded_pose_family_.empty()) {
      right_loaded_pose_family_.push_back(deg_to_rad({0, -75, 135, 0, 60, 0}));
    }
    const size_t loaded_preferred_index = static_cast<size_t>(
      std::max(0, get_or_declare_parameter<int>("loaded_preferred_pose_index", 1)));
    left_preferred_loaded_pose_index_ = std::min(loaded_preferred_index, left_loaded_pose_family_.size() - 1);
    right_preferred_loaded_pose_index_ = std::min(loaded_preferred_index, right_loaded_pose_family_.size() - 1);
    ik_config_.left_loaded_pose_family = left_loaded_pose_family_;
    ik_config_.right_loaded_pose_family = right_loaded_pose_family_;
    ik_config_.left_preferred_loaded_pose_index = left_preferred_loaded_pose_index_;
    ik_config_.right_preferred_loaded_pose_index = right_preferred_loaded_pose_index_;

    left_pregrasp_arm_ = deg_to_rad({0, -90, 135, -45, 0, 0});
    right_pregrasp_arm_ = deg_to_rad({0, -90, 135, 45, 0, 0});
    left_loaded_arm_ = left_loaded_pose_family_[left_preferred_loaded_pose_index_];
    right_loaded_arm_ = right_loaded_pose_family_[right_preferred_loaded_pose_index_];

    joint_state_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions joint_state_sub_options;
    joint_state_sub_options.callback_group = joint_state_callback_group_;
    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::JointState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        latest_joint_state_ = msg;
      },
      joint_state_sub_options);

    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), planning_group_);
    move_group_->setPlanningTime(planning_time_);
    move_group_->setNumPlanningAttempts(planning_attempts_);
    move_group_->setMaxVelocityScalingFactor(velocity_scale_);
    move_group_->setMaxAccelerationScalingFactor(acceleration_scale_);
    move_group_->setGoalJointTolerance(joint_goal_tolerance_rad_);

    loaded_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), extract_loaded_planning_group_);
    loaded_move_group_->setPlanningTime(extract_loaded_planning_time_);
    loaded_move_group_->setNumPlanningAttempts(extract_loaded_planning_attempts_);
    loaded_move_group_->setMaxVelocityScalingFactor(velocity_scale_);
    loaded_move_group_->setMaxAccelerationScalingFactor(acceleration_scale_);
    loaded_move_group_->setGoalJointTolerance(joint_goal_tolerance_rad_);

    robot_model_ = move_group_->getRobotModel();
    joint_group_ = robot_model_->getJointModelGroup(planning_group_);
    if (!joint_group_) {
      throw std::runtime_error("No JointModelGroup named " + planning_group_);
    }
    left_arm_group_ = robot_model_->getJointModelGroup("left_v5_arm");
    right_arm_group_ = robot_model_->getJointModelGroup("right_v5_arm");
    if (!left_arm_group_ || !right_arm_group_) {
      throw std::runtime_error("Missing single-arm JointModelGroup left_v5_arm/right_v5_arm");
    }

    optimized_ik_solver_ = std::make_unique<ik_benchmark::ParallelUpdownAwareIkSolver>(ik_config_);

    planning_scene_monitor_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
      shared_from_this(), "robot_description");
    if (!planning_scene_monitor_->getPlanningScene()) {
      RCLCPP_WARN(get_logger(), "PlanningSceneMonitor init failed; collision checks disabled");
    } else {
      planning_scene_monitor_->startSceneMonitor();
      planning_scene_monitor_->startWorldGeometryMonitor();
      planning_scene_monitor_->startStateMonitor("/joint_states");
      planning_scene_monitor_->requestPlanningSceneState();
    }

    planning_scene_interface_ = std::make_unique<moveit::planning_interface::PlanningSceneInterface>();
    apply_container_obstacles();
    set_static_box_wall_opening(extract_demo_left_box_id_, extract_demo_right_box_id_, "initial");

    demo_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/plan_and_execute",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        const bool ok = run_one_pair_flow(5, 6, false, 1);
        response->success = ok;
        response->message = ok ? "one-pair MoveIt flow finished" : last_error_;
      });

    box_stack_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/run_box_stack_flow",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        RCLCPP_INFO(get_logger(), "Received /%s/run_box_stack_flow request", get_name());
        const bool ok = run_box_stack_flow();
        response->success = ok;
        response->message = ok ? "box-stack MoveIt flow finished" : last_error_;
      });

    extract_demo_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/run_left_extract_demo",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        RCLCPP_INFO(get_logger(), "Received /%s/run_left_extract_demo request", get_name());
        const bool ok = run_left_extract_demo();
        response->success = ok;
        response->message = ok ? "left extract primitive finished" : last_error_;
      });

    RCLCPP_INFO(get_logger(), "DualArmPlannerNode ready");
    RCLCPP_INFO(get_logger(), "  group=%s execute=%s box_front_x=%.3f max_rounds=%d include_top=%s",
                planning_group_.c_str(), execute_ ? "true" : "false", box_front_x_, max_rounds_,
                include_top_suction_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  Services: /%s/plan_and_execute, /%s/run_box_stack_flow, /%s/run_left_extract_demo",
                get_name(), get_name(), get_name());
    RCLCPP_INFO(get_logger(),
                "  IK strategy=fixed_discrete h=%zu seed=%zu workers=%zu timeout=%.3fs collision=%s",
                ik_config_.h_candidate_count, ik_config_.seed_count, ik_config_.workers, ik_config_.timeout,
                ik_config_.check_collision ? "true" : "false");
    RCLCPP_INFO(get_logger(),
                "  loaded pose prior left=%zu right=%zu preferred=(%zu,%zu) weights=(family %.3f, preferred %.3f)",
                left_loaded_pose_family_.size(), right_loaded_pose_family_.size(),
                left_preferred_loaded_pose_index_, right_preferred_loaded_pose_index_,
                ik_config_.cost_loaded_family_distance, ik_config_.cost_loaded_preferred_distance);
    RCLCPP_INFO(get_logger(),
                "  Extract primitive IK=left_v5_arm/KDL fixed-updown timeout=%.3fs pos_tol=%.3fm ori_tol=%.3frad",
                extract_kdl_timeout_, extract_position_tolerance_, extract_orientation_tolerance_);
    RCLCPP_INFO(get_logger(), "  speed scale velocity=%.2f acceleration=%.2f",
                velocity_scale_, acceleration_scale_);
    RCLCPP_INFO(get_logger(), "  attached carried-box collision=%s size=(%.2f, %.2f, %.2f)",
                enable_attached_box_collision_ ? "true" : "false",
                carried_box_depth_, carried_box_width_, carried_box_height_);

    open_record_file();
  }

private:
  template<typename T>
  T get_or_declare_parameter(const std::string& name, const T& default_value)
  {
    if (!has_parameter(name)) {
      declare_parameter<T>(name, default_value);
    }
    T value = default_value;
    if (!get_parameter(name, value)) {
      return default_value;
    }
    return value;
  }

  moveit::core::RobotStatePtr get_current_robot_state()
  {
    if (prefer_commanded_state_ && last_commanded_state_) {
      return std::make_shared<moveit::core::RobotState>(*last_commanded_state_);
    }

    if (execute_ && move_group_) {
      auto current_state = move_group_->getCurrentState(1.0);
      if (current_state) {
        current_state->update();
        return current_state;
      }
    }

    sensor_msgs::msg::JointState::SharedPtr joint_state_msg;
    for (int retry = 0; retry < 30; ++retry) {
      {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        joint_state_msg = latest_joint_state_;
      }
      if (joint_state_msg) break;
      rclcpp::sleep_for(std::chrono::milliseconds(100));
    }

    auto state = std::make_shared<moveit::core::RobotState>(robot_model_);
    state->setToDefaultValues();

    if (!joint_state_msg) {
      RCLCPP_WARN(get_logger(), "No /joint_states received; using model default state");
      state->update();
      return state;
    }

    for (size_t i = 0; i < joint_state_msg->name.size() && i < joint_state_msg->position.size(); ++i) {
      if (robot_model_->hasJointModel(joint_state_msg->name[i])) {
        state->setJointPositions(joint_state_msg->name[i], &joint_state_msg->position[i]);
      }
    }
    state->update();
    return state;
  }

  bool is_state_valid(const moveit::core::RobotState& state, bool check_collision) const
  {
    if (!state.satisfiesBounds(joint_group_)) return false;
    if (!check_collision) return true;
    if (!planning_scene_monitor_ || !planning_scene_monitor_->getPlanningScene()) return true;
    planning_scene_monitor::LockedPlanningSceneRO scene(planning_scene_monitor_);
    return !scene->isStateColliding(state, joint_group_->getName());
  }

  std::vector<ContainerPanel> container_panels() const
  {
    const double half_width = container_width_ * 0.5;
    const double half_thickness = container_wall_thickness_ * 0.5;
    const double z_center = container_floor_z_ + container_height_ * 0.5;
    return {
      {
        "container_left_wall",
        {container_center_x_, container_center_y_ + half_width + half_thickness, z_center},
        {container_length_, container_wall_thickness_, container_height_},
      },
      {
        "container_right_wall",
        {container_center_x_, container_center_y_ - half_width - half_thickness, z_center},
        {container_length_, container_wall_thickness_, container_height_},
      },
      {
        "container_ceiling",
        {container_center_x_, container_center_y_, container_floor_z_ + container_height_ + half_thickness},
        {container_length_, container_width_ + 2.0 * container_wall_thickness_, container_wall_thickness_},
      },
    };
  }

  void apply_container_obstacles()
  {
    if (!enable_container_obstacle_) {
      RCLCPP_INFO(get_logger(), "Container obstacle disabled");
      return;
    }
    if (!planning_scene_interface_) return;

    std::vector<moveit_msgs::msg::CollisionObject> objects;
    for (const auto& panel : container_panels()) {
      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
      primitive.dimensions = {panel.size[0], panel.size[1], panel.size[2]};

      moveit_msgs::msg::CollisionObject object;
      object.header.frame_id = container_frame_;
      object.id = panel.id;
      object.primitives.push_back(primitive);
      object.primitive_poses.push_back(make_identity_pose(panel.center[0], panel.center[1], panel.center[2]));
      object.operation = moveit_msgs::msg::CollisionObject::ADD;
      objects.push_back(object);
    }

    if (planning_scene_interface_->applyCollisionObjects(objects)) {
      RCLCPP_INFO(get_logger(),
                  "Applied container obstacle: frame=%s length=%.2f width=%.2f height=%.2f panels=%zu",
                  container_frame_.c_str(), container_length_, container_width_, container_height_, objects.size());
    } else {
      RCLCPP_WARN(get_logger(), "Failed to apply container obstacle collision objects");
    }
  }

  void add_static_wall_piece(
    std::vector<StaticBoxObstacle>& obstacles,
    const std::string& id,
    double x_min,
    double x_max,
    double y_min,
    double y_max,
    double z_min,
    double z_max) const
  {
    if (x_max < x_min) std::swap(x_min, x_max);
    if (y_max < y_min) std::swap(y_min, y_max);
    if (z_max < z_min) std::swap(z_min, z_max);
    if ((x_max - x_min) < 1e-4 || (y_max - y_min) < 1e-4 || (z_max - z_min) < 1e-4) {
      return;
    }

    obstacles.push_back({
      id,
      {0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.5 * (z_min + z_max)},
      {x_max - x_min, y_max - y_min, z_max - z_min},
    });
  }

  std::vector<StaticBoxObstacle> make_box_wall_obstacles_for_opening(int left_box_id, int right_box_id) const
  {
    std::vector<StaticBoxObstacle> obstacles;
    if (!enable_static_box_obstacles_) return obstacles;

    const auto boxes = make_boxes(box_front_x_);
    const auto left_it = boxes.find(left_box_id);
    const auto right_it = boxes.find(right_box_id);
    if (left_it == boxes.end() || right_it == boxes.end()) return obstacles;

    const BoxSpec& first = left_it->second;
    const BoxSpec& second = right_it->second;
    const BoxSpec& positive_y_box = first.y >= second.y ? first : second;
    const BoxSpec& negative_y_box = first.y >= second.y ? second : first;

    const double half_width = carried_box_width_ * 0.5;
    const double half_height = carried_box_height_ * 0.5;
    const double inset = std::max(0.0, static_box_obstacle_inset_);
    const double x_min = std::min(first.x, second.x);
    const double x_max = std::max(first.x, second.x) + carried_box_depth_;
    const double inner_y_min = container_center_y_ - container_width_ * 0.5;
    const double inner_y_max = container_center_y_ + container_width_ * 0.5;
    const double z_min = std::min(first.z, second.z) - half_height;
    const double z_max = std::max(first.z, second.z) + half_height;

    const double positive_hole_y_min = positive_y_box.y - half_width;
    const double positive_hole_y_max = positive_y_box.y + half_width;
    const double negative_hole_y_min = negative_y_box.y - half_width;
    const double negative_hole_y_max = negative_y_box.y + half_width;
    const std::string prefix = "box_wall_L" + std::to_string(left_box_id) +
                               "_R" + std::to_string(right_box_id);

    add_static_wall_piece(
      obstacles, prefix + "_left_side",
      x_min, x_max,
      positive_hole_y_max + inset, inner_y_max,
      z_min, z_max);
    add_static_wall_piece(
      obstacles, prefix + "_between",
      x_min, x_max,
      negative_hole_y_max + inset, positive_hole_y_min - inset,
      z_min, z_max);
    add_static_wall_piece(
      obstacles, prefix + "_right_side",
      x_min, x_max,
      inner_y_min, negative_hole_y_min - inset,
      z_min, z_max);
    add_static_wall_piece(
      obstacles, prefix + "_below",
      x_min, x_max,
      inner_y_min, inner_y_max,
      container_floor_z_, z_min - inset);

    return obstacles;
  }

  std::vector<StaticBoxObstacle> static_box_obstacles() const
  {
    if (!enable_static_box_obstacles_) return {};
    return current_static_box_obstacles_;
  }

  void clear_applied_static_box_obstacles()
  {
    if (!planning_scene_interface_ || applied_static_box_obstacle_ids_.empty()) return;

    std::vector<moveit_msgs::msg::CollisionObject> remove_objects;
    remove_objects.reserve(applied_static_box_obstacle_ids_.size());
    for (const auto& id : applied_static_box_obstacle_ids_) {
      moveit_msgs::msg::CollisionObject object;
      object.header.frame_id = container_frame_;
      object.id = id;
      object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
      remove_objects.push_back(object);
    }
    planning_scene_interface_->applyCollisionObjects(remove_objects);
    applied_static_box_obstacle_ids_.clear();
  }

  void apply_static_box_obstacles()
  {
    clear_applied_static_box_obstacles();
    if (!enable_static_box_obstacles_) {
      current_static_box_obstacles_.clear();
      active_static_left_box_id_ = 0;
      active_static_right_box_id_ = 0;
      RCLCPP_INFO(get_logger(), "Static box-wall obstacles disabled");
      return;
    }
    if (!planning_scene_interface_) return;

    std::vector<moveit_msgs::msg::CollisionObject> objects;
    for (const auto& box : static_box_obstacles()) {
      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
      primitive.dimensions = {box.size[0], box.size[1], box.size[2]};

      moveit_msgs::msg::CollisionObject object;
      object.header.frame_id = container_frame_;
      object.id = box.id;
      object.primitives.push_back(primitive);
      object.primitive_poses.push_back(make_identity_pose(box.center[0], box.center[1], box.center[2]));
      object.operation = moveit_msgs::msg::CollisionObject::ADD;
      objects.push_back(object);
    }

    if (objects.empty()) {
      RCLCPP_WARN(get_logger(), "Static box-wall obstacles requested but no wall pieces were generated");
      return;
    }

    if (planning_scene_interface_->applyCollisionObjects(objects)) {
      applied_static_box_obstacle_ids_.clear();
      for (const auto& object : objects) {
        applied_static_box_obstacle_ids_.push_back(object.id);
      }
      RCLCPP_INFO(get_logger(),
                  "Applied dynamic box-wall obstacles: opening=(L%d,R%d) pieces=%zu inset=%.4fm",
                  active_static_left_box_id_, active_static_right_box_id_, objects.size(), static_box_obstacle_inset_);
    } else {
      RCLCPP_WARN(get_logger(), "Failed to apply dynamic box-wall obstacles");
    }
  }

  bool set_static_box_wall_opening(int left_box_id, int right_box_id, const std::string& reason)
  {
    if (!enable_static_box_obstacles_) {
      current_static_box_obstacles_.clear();
      active_static_left_box_id_ = 0;
      active_static_right_box_id_ = 0;
      apply_static_box_obstacles();
      return true;
    }

    active_static_left_box_id_ = left_box_id;
    active_static_right_box_id_ = right_box_id;
    current_static_box_obstacles_ = make_box_wall_obstacles_for_opening(left_box_id, right_box_id);
    apply_static_box_obstacles();
    if (current_static_box_obstacles_.empty()) {
      RCLCPP_WARN(get_logger(), "No dynamic box wall for opening L%d/R%d (%s)",
                  left_box_id, right_box_id, reason.c_str());
      return false;
    }
    RCLCPP_INFO(get_logger(), "Using dynamic box wall opening L%d/R%d for %s",
                left_box_id, right_box_id, reason.c_str());
    return true;
  }

  AttachedBoxSpec make_attached_box_spec(const std::string& side, int box_id, bool top_suction) const
  {
    AttachedBoxSpec spec;
    spec.id = "carried_" + side + "_box_" + std::to_string(box_id);
    spec.link_name = side + "_v5_tool0";
    if (top_suction) {
      spec.center_in_link = {0.0, 0.0, carried_box_height_ * 0.5};
      spec.size = {carried_box_depth_, carried_box_width_, carried_box_height_};
    } else {
      spec.center_in_link = {0.0, 0.0, carried_box_depth_ * 0.5};
      spec.size = {carried_box_width_, carried_box_height_, carried_box_depth_};
    }
    return spec;
  }

  moveit_msgs::msg::AttachedCollisionObject make_attached_collision_object(
    const AttachedBoxSpec& spec, int operation) const
  {
    moveit_msgs::msg::AttachedCollisionObject attached;
    attached.link_name = spec.link_name;
    attached.touch_links = {spec.link_name};
    if (spec.link_name.rfind("left_", 0) == 0) {
      attached.touch_links.push_back("left_v5_link6");
      attached.touch_links.push_back("left_v5_link5");
    } else if (spec.link_name.rfind("right_", 0) == 0) {
      attached.touch_links.push_back("right_v5_link6");
      attached.touch_links.push_back("right_v5_link5");
    }

    attached.object.header.frame_id = spec.link_name;
    attached.object.id = spec.id;
    attached.object.operation = operation;
    if (operation == moveit_msgs::msg::CollisionObject::ADD) {
      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
      primitive.dimensions = {spec.size[0], spec.size[1], spec.size[2]};
      attached.object.primitives.push_back(primitive);
      attached.object.primitive_poses.push_back(
        make_identity_pose(spec.center_in_link[0], spec.center_in_link[1], spec.center_in_link[2]));
    }
    return attached;
  }

  bool apply_attached_box_state(
    const std::vector<AttachedBoxSpec>& specs, int operation, const std::string& action_name)
  {
    if (!enable_attached_box_collision_) return true;
    if (!planning_scene_interface_) return true;
    if (specs.empty()) return true;

    std::vector<moveit_msgs::msg::AttachedCollisionObject> objects;
    objects.reserve(specs.size());
    for (const auto& spec : specs) {
      objects.push_back(make_attached_collision_object(spec, operation));
    }
    if (!planning_scene_interface_->applyAttachedCollisionObjects(objects)) {
      return fail(action_name + ": failed to update attached carried boxes");
    }
    rclcpp::sleep_for(std::chrono::milliseconds(100));
    return true;
  }

  Eigen::Vector3d transform_point(const Eigen::Isometry3d& transform, const std::array<double, 3>& point) const
  {
    return transform * Eigen::Vector3d(point[0], point[1], point[2]);
  }

  AxisAlignedBox attached_box_world_aabb(
    const moveit::core::RobotState& state,
    const AttachedBoxSpec& box) const
  {
    const Eigen::Isometry3d& link_tf = state.getGlobalLinkTransform(box.link_name);
    Eigen::Vector3d min_corner(
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity());
    Eigen::Vector3d max_corner(
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity());

    for (double sx : {-0.5, 0.5}) {
      for (double sy : {-0.5, 0.5}) {
        for (double sz : {-0.5, 0.5}) {
          const std::array<double, 3> local_corner = {
            box.center_in_link[0] + sx * box.size[0],
            box.center_in_link[1] + sy * box.size[1],
            box.center_in_link[2] + sz * box.size[2],
          };
          const Eigen::Vector3d world_corner = transform_point(link_tf, local_corner);
          min_corner = min_corner.cwiseMin(world_corner);
          max_corner = max_corner.cwiseMax(world_corner);
        }
      }
    }

    const Eigen::Vector3d center = 0.5 * (min_corner + max_corner);
    const Eigen::Vector3d size = max_corner - min_corner;
    return {{
      center.x(), center.y(), center.z()
    }, {
      size.x(), size.y(), size.z()
    }};
  }

  bool aabb_overlaps(const AxisAlignedBox& lhs, const AxisAlignedBox& rhs) const
  {
    for (size_t i = 0; i < 3; ++i) {
      if (std::abs(lhs.center[i] - rhs.center[i]) > 0.5 * (lhs.size[i] + rhs.size[i])) {
        return false;
      }
    }
    return true;
  }

  bool carried_boxes_clear_static_obstacles(
    const moveit::core::RobotState& state,
    std::string* reason) const
  {
    if (!enable_attached_box_collision_ || active_attached_boxes_.empty()) return true;

    for (const auto& carried_box : active_attached_boxes_) {
      if (!carried_box_clear_scene_obstacles(state, carried_box, reason)) {
        return false;
      }
    }
    return true;
  }

  AxisAlignedBox expanded_aabb(const AxisAlignedBox& box, double margin) const
  {
    return {box.center, {
      box.size[0] + 2.0 * margin,
      box.size[1] + 2.0 * margin,
      box.size[2] + 2.0 * margin,
    }};
  }

  bool carried_box_detached_from_neighbors(
    const moveit::core::RobotState& state,
    const AttachedBoxSpec& carried_box,
    int box_id,
    std::string* reason) const
  {
    const int column = (box_id - 1) % 5 + 1;
    const int row = (box_id - 1) / 5;
    std::vector<int> neighbor_ids;
    if (column > 1) neighbor_ids.push_back(row * 5 + column - 1);
    if (column < 5) neighbor_ids.push_back(row * 5 + column + 1);

    const auto boxes = make_boxes(box_front_x_);
    const AxisAlignedBox carried = expanded_aabb(attached_box_world_aabb(state, carried_box), extract_neighbor_margin_);
    const double carried_min_x = carried.center[0] - 0.5 * carried.size[0];
    const double carried_max_x = carried.center[0] + 0.5 * carried.size[0];
    const double carried_min_z = carried.center[2] - 0.5 * carried.size[2];
    const double carried_max_z = carried.center[2] + 0.5 * carried.size[2];
    for (int neighbor_id : neighbor_ids) {
      const auto it = boxes.find(neighbor_id);
      if (it == boxes.end()) continue;
      const AxisAlignedBox neighbor = expanded_aabb({{
        it->second.x + carried_box_depth_ * 0.5,
        it->second.y,
        it->second.z,
      }, {
        carried_box_depth_,
        carried_box_width_,
        carried_box_height_,
      }}, extract_neighbor_margin_);
      const double neighbor_min_x = neighbor.center[0] - 0.5 * neighbor.size[0];
      const double neighbor_max_x = neighbor.center[0] + 0.5 * neighbor.size[0];
      const double neighbor_min_z = neighbor.center[2] - 0.5 * neighbor.size[2];
      const double neighbor_max_z = neighbor.center[2] + 0.5 * neighbor.size[2];
      const bool side_face_projection_overlaps =
        carried_min_x <= neighbor_max_x &&
        carried_max_x >= neighbor_min_x &&
        carried_min_z <= neighbor_max_z &&
        carried_max_z >= neighbor_min_z;
      if (side_face_projection_overlaps) {
        if (reason) {
          *reason = carried_box.id + " side face still overlaps neighbor box " + std::to_string(neighbor_id);
        }
        return false;
      }
    }
    return true;
  }

  bool left_carried_box_detached_from_neighbors(
    const moveit::core::RobotState& state,
    const AttachedBoxSpec& carried_box,
    int left_box_id,
    std::string* reason) const
  {
    return carried_box_detached_from_neighbors(state, carried_box, left_box_id, reason);
  }

  bool carried_box_clear_scene_obstacles(
    const moveit::core::RobotState& state,
    const AttachedBoxSpec& carried_box,
    std::string* reason) const
  {
    if (!enable_attached_box_collision_) return true;
    const auto carried_aabb = attached_box_world_aabb(state, carried_box);

    if (enable_static_box_obstacles_) {
      for (const auto& obstacle : static_box_obstacles()) {
        const AxisAlignedBox obstacle_aabb{obstacle.center, obstacle.size};
        if (aabb_overlaps(carried_aabb, obstacle_aabb)) {
          if (reason) *reason = carried_box.id + " overlaps " + obstacle.id;
          return false;
        }
      }
    }

    if (enable_container_obstacle_) {
      for (const auto& panel : container_panels()) {
        const AxisAlignedBox panel_aabb{panel.center, panel.size};
        if (aabb_overlaps(carried_aabb, panel_aabb)) {
          if (reason) *reason = carried_box.id + " overlaps " + panel.id;
          return false;
        }
      }
    }

    return true;
  }

  bool state_clear_for_extract(
    const moveit::core::RobotState& state,
    const AttachedBoxSpec& left_carried_box,
    int left_box_id,
    bool* detached,
    std::string* reason) const
  {
    if (!is_state_valid(state, true)) {
      if (reason) *reason = "robot state colliding or out of bounds";
      return false;
    }

    std::string carried_reason;
    if (!carried_box_clear_scene_obstacles(state, left_carried_box, &carried_reason)) {
      if (reason) *reason = carried_reason;
      return false;
    }

    std::string detached_reason;
    const bool detached_now = left_carried_box_detached_from_neighbors(state, left_carried_box, left_box_id, &detached_reason);
    if (detached) *detached = detached_now;
    if (!detached_now && reason) *reason = detached_reason;
    return true;
  }

  bool state_clear_for_single_extract(
    const moveit::core::RobotState& state,
    const AttachedBoxSpec& carried_box,
    int box_id,
    bool* detached,
    std::string* reason) const
  {
    if (!is_state_valid(state, true)) {
      if (reason) *reason = "robot state colliding or out of bounds";
      return false;
    }

    std::string carried_reason;
    if (!carried_box_clear_scene_obstacles(state, carried_box, &carried_reason)) {
      if (reason) *reason = carried_reason;
      return false;
    }

    std::string detached_reason;
    const bool detached_now = carried_box_detached_from_neighbors(state, carried_box, box_id, &detached_reason);
    if (detached) *detached = detached_now;
    if (!detached_now && reason) *reason = detached_reason;
    return true;
  }

  bool state_clear_for_dual_extract(
    const moveit::core::RobotState& state,
    const AttachedBoxSpec& left_box,
    int left_box_id,
    const AttachedBoxSpec& right_box,
    int right_box_id,
    bool* left_detached,
    bool* right_detached,
    std::string* reason) const
  {
    if (!is_state_valid(state, true)) {
      if (reason) *reason = "robot state colliding or out of bounds";
      return false;
    }

    for (const auto& box : {left_box, right_box}) {
      std::string carried_reason;
      if (!carried_box_clear_scene_obstacles(state, box, &carried_reason)) {
        if (reason) *reason = carried_reason;
        return false;
      }
    }

    std::string left_reason;
    std::string right_reason;
    const bool left_ok = carried_box_detached_from_neighbors(state, left_box, left_box_id, &left_reason);
    const bool right_ok = carried_box_detached_from_neighbors(state, right_box, right_box_id, &right_reason);
    if (left_detached) *left_detached = left_ok;
    if (right_detached) *right_detached = right_ok;
    if ((!left_ok || !right_ok) && reason) {
      *reason = !left_ok ? left_reason : right_reason;
    }
    return true;
  }

  bool planned_carried_boxes_clear_static_obstacles(
    const moveit::planning_interface::MoveGroupInterface::Plan& plan,
    const moveit::core::RobotState& start_state,
    std::string* reason) const
  {
    if (!enable_attached_box_collision_ || active_attached_boxes_.empty()) return true;

    const auto& trajectory = plan.trajectory_.joint_trajectory;
    for (size_t point_index = 0; point_index < trajectory.points.size(); ++point_index) {
      moveit::core::RobotState state(start_state);
      const auto& point = trajectory.points[point_index];
      for (size_t i = 0; i < trajectory.joint_names.size() && i < point.positions.size(); ++i) {
        if (is_robot_variable(trajectory.joint_names[i])) {
          state.setVariablePosition(trajectory.joint_names[i], point.positions[i]);
        }
      }
      state.update();
      std::string point_reason;
      if (!carried_boxes_clear_static_obstacles(state, &point_reason)) {
        if (reason) {
          *reason = "trajectory point " + std::to_string(point_index) + ": " + point_reason;
        }
        return false;
      }
    }
    return true;
  }

  bool remove_carried_box_ids(const std::vector<std::string>& ids, const std::string& action_name)
  {
    if (!enable_attached_box_collision_) return true;
    if (!planning_scene_interface_) return true;
    if (ids.empty()) return true;

    std::vector<moveit_msgs::msg::AttachedCollisionObject> attached_removes;
    attached_removes.reserve(ids.size());
    for (const auto& id : ids) {
      moveit_msgs::msg::AttachedCollisionObject attached;
      attached.link_name = id.find("_right_") != std::string::npos ? right_tip_ : left_tip_;
      attached.object.header.frame_id = attached.link_name;
      attached.object.id = id;
      attached.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
      attached_removes.push_back(attached);
    }

    std::vector<moveit_msgs::msg::CollisionObject> world_removes;
    world_removes.reserve(ids.size());
    for (const auto& id : ids) {
      moveit_msgs::msg::CollisionObject object;
      object.header.frame_id = container_frame_;
      object.id = id;
      object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
      world_removes.push_back(object);
    }

    const bool attached_ok = planning_scene_interface_->applyAttachedCollisionObjects(attached_removes);
    planning_scene_interface_->applyCollisionObjects(world_removes);
    rclcpp::sleep_for(std::chrono::milliseconds(100));
    if (!attached_ok) {
      return fail(action_name + ": failed to remove carried boxes from planning scene");
    }
    return true;
  }

  bool clear_carried_boxes_from_scene()
  {
    if (!enable_attached_box_collision_) return true;
    if (active_attached_boxes_.empty()) return true;
    return detach_carried_boxes();
  }

  bool attach_carried_boxes(int left_box_id, int right_box_id, bool top_suction)
  {
    active_attached_boxes_ = {
      make_attached_box_spec("left", left_box_id, top_suction),
      make_attached_box_spec("right", right_box_id, top_suction),
    };
    if (!apply_attached_box_state(active_attached_boxes_, moveit_msgs::msg::CollisionObject::ADD, "attach_carried_boxes")) {
      active_attached_boxes_.clear();
      return false;
    }
    RCLCPP_INFO(get_logger(), "Attached carried boxes: left=%d right=%d mode=%s",
                left_box_id, right_box_id, top_suction ? "top_suction" : "front");
    return true;
  }

  bool detach_carried_boxes()
  {
    if (active_attached_boxes_.empty()) return true;
    std::vector<std::string> ids;
    ids.reserve(active_attached_boxes_.size());
    for (const auto& box : active_attached_boxes_) {
      ids.push_back(box.id);
    }
    if (!remove_carried_box_ids(ids, "detach_carried_boxes")) {
      return false;
    }
    active_attached_boxes_.clear();
    RCLCPP_INFO(get_logger(), "Detached carried boxes");
    return true;
  }

  sensor_msgs::msg::JointState make_dual_arm_joint_target(
    double updown, const std::vector<double>& left_arm, const std::vector<double>& right_arm) const
  {
    sensor_msgs::msg::JointState target;
    target.name = {
      "updown",
      "left_v5_joint1", "left_v5_joint2", "left_v5_joint3",
      "left_v5_joint4", "left_v5_joint5", "left_v5_joint6",
      "right_v5_joint1", "right_v5_joint2", "right_v5_joint3",
      "right_v5_joint4", "right_v5_joint5", "right_v5_joint6",
    };
    target.position.reserve(target.name.size());
    target.position.push_back(updown);
    target.position.insert(target.position.end(), left_arm.begin(), left_arm.end());
    target.position.insert(target.position.end(), right_arm.begin(), right_arm.end());
    return target;
  }

  bool plan_to_joint_target(const std::string& stage_name, const sensor_msgs::msg::JointState& target)
  {
    auto start_state = get_current_robot_state();
    if (!start_state) {
      return fail(stage_name + ": cannot get start state");
    }

    moveit::core::RobotState goal_state(*start_state);
    for (size_t i = 0; i < target.name.size() && i < target.position.size(); ++i) {
      if (is_robot_variable(target.name[i])) {
        goal_state.setVariablePosition(target.name[i], target.position[i]);
      }
    }
    goal_state.enforceBounds(joint_group_);
    goal_state.update();

    if (!is_state_valid(goal_state, check_goal_collision_)) {
      return fail(stage_name + ": joint target out of bounds or colliding");
    }

    RCLCPP_INFO(get_logger(), "[%s] planning joint target", stage_name.c_str());
    return plan_to_goal_state(stage_name, *start_state, goal_state, target.name);
  }

  bool plan_dual_tip_ik(
    const std::string& stage_name,
    const geometry_msgs::msg::Pose& left_pose,
    const geometry_msgs::msg::Pose& right_pose,
    bool top_suction)
  {
    auto start_state = get_current_robot_state();
    if (!start_state) {
      return fail(stage_name + ": cannot get start state");
    }
    if (!optimized_ik_solver_) {
      return fail(stage_name + ": optimized IK solver is not initialized");
    }

    ik_benchmark::UpdownAwareIkRequest request;
    request.left_target = pose_to_eigen(left_pose);
    request.right_target = pose_to_eigen(right_pose);
    request.current_h = current_updown(*start_state);
    request.grasp_mode = top_suction
      ? ik_benchmark::UpdownAwareIkRequest::GraspMode::TopSuction
      : ik_benchmark::UpdownAwareIkRequest::GraspMode::Front;
    request.current_arm_joints = state_values(*start_state, optimized_ik_solver_->fixedVariableNames());
    request.current_full_joints = state_values(*start_state, optimized_ik_solver_->freeVariableNames());

    RCLCPP_INFO(get_logger(), "[%s] optimized IK L=(%.3f, %.3f, %.3f) R=(%.3f, %.3f, %.3f) current_h=%.3f",
                stage_name.c_str(),
                left_pose.position.x, left_pose.position.y, left_pose.position.z,
                right_pose.position.x, right_pose.position.y, right_pose.position.z, request.current_h);

    const auto result = optimized_ik_solver_->solve(request);
    if (!result.success) {
      std::ostringstream oss;
      oss << stage_name << ": optimized IK failed reason=" << result.failure_reason
          << " trials=" << result.trial_count << " legal=" << result.legal_count
          << " wall_ms=" << result.wall_ms;
      return fail(oss.str());
    }

    moveit::core::RobotState goal_state(*start_state);
    for (size_t i = 0; i < result.selected.full_joint_names.size() && i < result.selected.full_joint_values.size(); ++i) {
      const auto& name = result.selected.full_joint_names[i];
      if (is_robot_variable(name)) {
        goal_state.setVariablePosition(name, result.selected.full_joint_values[i]);
      }
    }
    goal_state.enforceBounds(joint_group_);
    goal_state.update();

    if (!is_state_valid(goal_state, check_goal_collision_)) {
      return fail(stage_name + ": selected IK state out of bounds or colliding");
    }

    RCLCPP_INFO(get_logger(),
                "[%s] IK selected h=%.3f score=%.3f path=%s h_index=%zu seed_index=%zu trials=%zu legal=%zu wall=%.1fms",
                stage_name.c_str(), result.selected.h, result.selected.score, result.selected.solver_path.c_str(),
                result.selected.h_index, result.selected.seed_index, result.trial_count, result.legal_count,
                result.wall_ms);

    nlohmann::json extra = {
      {"stage_kind", "optimized_dual_tip_ik"},
      {"grasp_mode", top_suction ? "top_suction" : "front"},
      {"left_target", pose_json(left_pose)},
      {"right_target", pose_json(right_pose)},
      {"ik", {
        {"strategy", "fixed_discrete_h_multi_seed_cost_scorer"},
        {"success", result.success},
        {"fallback_used", result.fallback_used},
        {"failure_reason", result.failure_reason},
        {"trial_count", result.trial_count},
        {"legal_count", result.legal_count},
        {"timeout_like_count", result.timeout_like_count},
        {"wall_ms", result.wall_ms},
        {"sum_solve_ms", result.sum_solve_ms},
        {"h_interval", {{"lower", result.h_interval_lower}, {"upper", result.h_interval_upper}, {"center", result.h_center}}},
        {"h_candidates", vector_json(result.h_candidates)},
        {"candidate_rejection_counts", ik_candidate_rejection_counts_json(result)},
        {"selected", {
          {"h", result.selected.h},
          {"h_index", result.selected.h_index},
          {"seed_index", result.selected.seed_index},
          {"score", result.selected.score},
          {"solver_path", result.selected.solver_path},
          {"target_order", result.selected.target_order},
          {"direct_pos_error", result.selected.direct_pos_error},
          {"direct_ori_error", result.selected.direct_ori_error},
          {"updown_delta", result.selected.updown_delta},
          {"joint_delta", result.selected.joint_delta},
          {"collision_free", result.selected.collision_free},
          {"collision_pairs", result.selected.collision_pairs},
          {"joint_names", result.selected.full_joint_names},
          {"joint_values", result.selected.full_joint_values}
        }}
      }}
    };

    return plan_to_goal_state(stage_name, *start_state, goal_state, result.selected.full_joint_names, extra);
  }

  bool solve_dual_tip_ik_state(
    const std::string& stage_name,
    const geometry_msgs::msg::Pose& left_pose,
    const geometry_msgs::msg::Pose& right_pose,
    bool top_suction,
    const moveit::core::RobotState& seed_state,
    moveit::core::RobotState* goal_state,
    nlohmann::json* extra_out,
    ik_benchmark::UpdownAwareIkResult* result_out = nullptr)
  {
    if (!optimized_ik_solver_) {
      return fail(stage_name + ": optimized IK solver is not initialized");
    }
    if (!goal_state) {
      return fail(stage_name + ": output goal_state is null");
    }

    ik_benchmark::UpdownAwareIkRequest request;
    request.left_target = pose_to_eigen(left_pose);
    request.right_target = pose_to_eigen(right_pose);
    request.current_h = current_updown(seed_state);
    request.grasp_mode = top_suction
      ? ik_benchmark::UpdownAwareIkRequest::GraspMode::TopSuction
      : ik_benchmark::UpdownAwareIkRequest::GraspMode::Front;
    request.current_arm_joints = state_values(seed_state, optimized_ik_solver_->fixedVariableNames());
    request.current_full_joints = state_values(seed_state, optimized_ik_solver_->freeVariableNames());

    RCLCPP_INFO(get_logger(), "[%s] direct optimized IK L=(%.3f, %.3f, %.3f) R=(%.3f, %.3f, %.3f) current_h=%.3f",
                stage_name.c_str(),
                left_pose.position.x, left_pose.position.y, left_pose.position.z,
                right_pose.position.x, right_pose.position.y, right_pose.position.z, request.current_h);

    const auto result = optimized_ik_solver_->solve(request);
    if (result_out) {
      *result_out = result;
    }
    if (!result.success) {
      std::ostringstream oss;
      oss << stage_name << ": optimized IK failed reason=" << result.failure_reason
          << " trials=" << result.trial_count << " legal=" << result.legal_count
          << " wall_ms=" << result.wall_ms;
      return fail(oss.str());
    }

    *goal_state = seed_state;
    for (size_t i = 0; i < result.selected.full_joint_names.size() && i < result.selected.full_joint_values.size(); ++i) {
      const auto& name = result.selected.full_joint_names[i];
      if (is_robot_variable(name)) {
        goal_state->setVariablePosition(name, result.selected.full_joint_values[i]);
      }
    }
    goal_state->enforceBounds(joint_group_);
    goal_state->update();

    if (!is_state_valid(*goal_state, check_goal_collision_)) {
      return fail(stage_name + ": selected IK state out of bounds or colliding");
    }

    RCLCPP_INFO(get_logger(),
                "[%s] direct IK selected h=%.3f score=%.3f path=%s h_index=%zu seed_index=%zu trials=%zu legal=%zu wall=%.1fms",
                stage_name.c_str(), result.selected.h, result.selected.score, result.selected.solver_path.c_str(),
                result.selected.h_index, result.selected.seed_index, result.trial_count, result.legal_count,
                result.wall_ms);

    if (extra_out) {
      *extra_out = {
        {"stage_kind", "optimized_dual_tip_ik_direct_seed"},
        {"grasp_mode", top_suction ? "top_suction" : "front"},
        {"left_target", pose_json(left_pose)},
        {"right_target", pose_json(right_pose)},
        {"ik", {
          {"strategy", "fixed_discrete_h_multi_seed_cost_scorer"},
          {"success", result.success},
          {"fallback_used", result.fallback_used},
          {"failure_reason", result.failure_reason},
          {"trial_count", result.trial_count},
          {"legal_count", result.legal_count},
          {"timeout_like_count", result.timeout_like_count},
          {"wall_ms", result.wall_ms},
          {"sum_solve_ms", result.sum_solve_ms},
          {"h_interval", {{"lower", result.h_interval_lower}, {"upper", result.h_interval_upper}, {"center", result.h_center}}},
          {"h_candidates", vector_json(result.h_candidates)},
          {"candidate_rejection_counts", ik_candidate_rejection_counts_json(result)},
          {"selected", {
            {"h", result.selected.h},
            {"h_index", result.selected.h_index},
            {"seed_index", result.selected.seed_index},
            {"score", result.selected.score},
            {"solver_path", result.selected.solver_path},
            {"target_order", result.selected.target_order},
            {"direct_pos_error", result.selected.direct_pos_error},
            {"direct_ori_error", result.selected.direct_ori_error},
            {"updown_delta", result.selected.updown_delta},
            {"joint_delta", result.selected.joint_delta},
            {"collision_free", result.selected.collision_free},
            {"collision_pairs", result.selected.collision_pairs},
            {"joint_names", result.selected.full_joint_names},
            {"joint_values", result.selected.full_joint_values}
          }}
        }}
      };
    }
    return true;
  }

  geometry_msgs::msg::Pose link_pose(const moveit::core::RobotState& state, const std::string& link_name) const
  {
    const Eigen::Isometry3d& tf = state.getGlobalLinkTransform(link_name);
    Eigen::Quaterniond q(tf.linear());
    q.normalize();
    return make_pose(tf.translation().x(), tf.translation().y(), tf.translation().z(), q);
  }

  bool solve_extract_candidate(
    const moveit::core::RobotState& current_state,
    const geometry_msgs::msg::Pose& left_pose,
    const geometry_msgs::msg::Pose& right_hold_pose,
    size_t step_index,
    size_t candidate_index,
    double retreat_x,
    double lift_z,
    double pitch_up_rad,
    ExtractCandidate* out) const
  {
    if (!out || !optimized_ik_solver_) return false;
    out->step_index = step_index;
    out->candidate_index = candidate_index;
    out->retreat_x = retreat_x;
    out->lift_z = lift_z;
    out->pitch_up_rad = pitch_up_rad;
    out->target_pose = left_pose;

    ik_benchmark::UpdownAwareIkRequest request;
    request.left_target = pose_to_eigen(left_pose);
    request.right_target = pose_to_eigen(right_hold_pose);
    request.current_h = current_updown(current_state);
    request.grasp_mode = ik_benchmark::UpdownAwareIkRequest::GraspMode::Front;
    request.current_arm_joints = state_values(current_state, optimized_ik_solver_->fixedVariableNames());
    request.current_full_joints = state_values(current_state, optimized_ik_solver_->freeVariableNames());

    auto result = optimized_ik_solver_->solve(request);
    if (!result.success) {
      out->rejection_reason = "ik_failed:" + result.failure_reason;
      return false;
    }
    out->ik_success = true;

    out->state = std::make_shared<moveit::core::RobotState>(current_state);
    for (size_t i = 0; i < result.selected.full_joint_names.size() && i < result.selected.full_joint_values.size(); ++i) {
      const auto& name = result.selected.full_joint_names[i];
      if (!is_robot_variable(name)) continue;
      if (name.rfind("right_v5_", 0) == 0) continue;
      if (name == "updown") continue;
      out->state->setVariablePosition(name, result.selected.full_joint_values[i]);
    }
    out->state->enforceBounds(joint_group_);
    out->state->update();
    return true;
  }

  bool solve_left_extract_candidate_kdl(
    const moveit::core::RobotState& current_state,
    const geometry_msgs::msg::Pose& left_pose,
    size_t step_index,
    size_t candidate_index,
    double retreat_x,
    double retreat_delta_x,
    double lift_z,
    double lift_delta_z,
    double pitch_up_rad,
    double pitch_delta_rad,
    double min_allowed_tip_z,
    const AttachedBoxSpec& left_box,
    int left_box_id,
    ExtractCandidate* out) const
  {
    if (!out || !left_arm_group_) return false;
    out->step_index = step_index;
    out->candidate_index = candidate_index;
    out->retreat_x = retreat_x;
    out->retreat_delta_x = retreat_delta_x;
    out->lift_z = lift_z;
    out->lift_delta_z = lift_delta_z;
    out->pitch_up_rad = pitch_up_rad;
    out->pitch_delta_rad = pitch_delta_rad;
    out->target_pose = left_pose;

    auto state = std::make_shared<moveit::core::RobotState>(current_state);
    state->setVariablePosition("updown", current_updown(current_state));
    state->update();

    const Eigen::Isometry3d target = pose_to_eigen(left_pose);
    const bool ik_ok = state->setFromIK(left_arm_group_, target, left_tip_, extract_kdl_timeout_);
    if (!ik_ok) {
      out->rejection_reason = "left_kdl_no_solution";
      return false;
    }

    state->setVariablePosition("updown", current_updown(current_state));
    state->enforceBounds(joint_group_);
    state->update();

    const Eigen::Isometry3d& actual = state->getGlobalLinkTransform(left_tip_);
    const double pos_error = pose_position_error(target, actual);
    const double ori_error = pose_orientation_error(target, actual);
    if (pos_error > extract_position_tolerance_ || ori_error > extract_orientation_tolerance_) {
      std::ostringstream oss;
      oss << "left_kdl_tip_error pos=" << pos_error << " ori=" << ori_error;
      out->rejection_reason = oss.str();
      return false;
    }

    const Eigen::Vector3d tool_normal = actual.linear() * Eigen::Vector3d::UnitZ();
    if (tool_normal.z() < extract_min_tool_normal_z_) {
      std::ostringstream oss;
      oss << "tool_normal_down z=" << tool_normal.z();
      out->rejection_reason = oss.str();
      return false;
    }

    if (actual.translation().z() + extract_max_tip_z_drop_ < min_allowed_tip_z) {
      std::ostringstream oss;
      oss << "tip_z_dropped z=" << actual.translation().z() << " min=" << min_allowed_tip_z;
      out->rejection_reason = oss.str();
      return false;
    }

    if (extract_max_joint_delta_ > 0.0) {
      const double joint_delta = extract_left_arm_joint_delta(current_state, *state);
      if (joint_delta > extract_max_joint_delta_) {
        std::ostringstream oss;
        oss << "joint_delta_too_large delta=" << joint_delta << " limit=" << extract_max_joint_delta_;
        out->rejection_reason = oss.str();
        return false;
      }
    }

    bool detached = false;
    std::string reason;
    if (!state_clear_for_extract(*state, left_box, left_box_id, &detached, &reason)) {
      out->rejection_reason = reason;
      return false;
    }

    out->ik_success = true;
    out->state_valid = true;
    out->carried_clear = true;
    out->detached_from_neighbors = detached;
    out->state = state;
    return true;
  }

  bool solve_extract_candidate_kdl_for_side(
    const std::string& side,
    const moveit::core::RobotState& current_state,
    const geometry_msgs::msg::Pose& target_pose,
    size_t step_index,
    size_t candidate_index,
    double retreat_x,
    double retreat_delta_x,
    double lift_z,
    double lift_delta_z,
    double pitch_up_rad,
    double pitch_delta_rad,
    double min_allowed_tip_z,
    const AttachedBoxSpec& carried_box,
    int box_id,
    ExtractCandidate* out) const
  {
    const auto* arm_group = side == "left" ? left_arm_group_ : right_arm_group_;
    const std::string& tip = side == "left" ? left_tip_ : right_tip_;
    if (!out || !arm_group) return false;
    out->step_index = step_index;
    out->candidate_index = candidate_index;
    out->retreat_x = retreat_x;
    out->retreat_delta_x = retreat_delta_x;
    out->lift_z = lift_z;
    out->lift_delta_z = lift_delta_z;
    out->pitch_up_rad = pitch_up_rad;
    out->pitch_delta_rad = pitch_delta_rad;
    out->target_pose = target_pose;

    auto state = std::make_shared<moveit::core::RobotState>(current_state);
    state->setVariablePosition("updown", current_updown(current_state));
    state->update();

    const Eigen::Isometry3d target = pose_to_eigen(target_pose);
    const bool ik_ok = state->setFromIK(arm_group, target, tip, extract_kdl_timeout_);
    if (!ik_ok) {
      out->rejection_reason = side + "_kdl_no_solution";
      return false;
    }

    state->setVariablePosition("updown", current_updown(current_state));
    state->enforceBounds(joint_group_);
    state->update();

    const Eigen::Isometry3d& actual = state->getGlobalLinkTransform(tip);
    const double pos_error = pose_position_error(target, actual);
    const double ori_error = pose_orientation_error(target, actual);
    if (pos_error > extract_position_tolerance_ || ori_error > extract_orientation_tolerance_) {
      std::ostringstream oss;
      oss << side << "_kdl_tip_error pos=" << pos_error << " ori=" << ori_error;
      out->rejection_reason = oss.str();
      return false;
    }

    const Eigen::Vector3d tool_normal = actual.linear() * Eigen::Vector3d::UnitZ();
    if (tool_normal.z() < extract_min_tool_normal_z_) {
      std::ostringstream oss;
      oss << side << "_tool_normal_down z=" << tool_normal.z();
      out->rejection_reason = oss.str();
      return false;
    }

    if (actual.translation().z() + extract_max_tip_z_drop_ < min_allowed_tip_z) {
      std::ostringstream oss;
      oss << side << "_tip_z_dropped z=" << actual.translation().z() << " min=" << min_allowed_tip_z;
      out->rejection_reason = oss.str();
      return false;
    }

    if (extract_max_joint_delta_ > 0.0) {
      const double joint_delta = extract_arm_joint_delta(side, current_state, *state);
      if (joint_delta > extract_max_joint_delta_) {
        std::ostringstream oss;
        oss << side << "_joint_delta_too_large delta=" << joint_delta << " limit=" << extract_max_joint_delta_;
        out->rejection_reason = oss.str();
        return false;
      }
    }

    bool detached = false;
    std::string reason;
    if (!state_clear_for_single_extract(*state, carried_box, box_id, &detached, &reason)) {
      out->rejection_reason = reason;
      return false;
    }

    out->ik_success = true;
    out->state_valid = true;
    out->carried_clear = true;
    out->detached_from_neighbors = detached;
    out->state = state;
    return true;
  }

  double extract_left_arm_joint_delta(
    const moveit::core::RobotState& from,
    const moveit::core::RobotState& to) const
  {
    if (!left_arm_group_) return 0.0;
    double sum = 0.0;
    const auto& names = left_arm_group_->getVariableNames();
    for (const auto& name : names) {
      if (!is_robot_variable(name)) continue;
      const double delta = to.getVariablePosition(name) - from.getVariablePosition(name);
      sum += delta * delta;
    }
    return std::sqrt(sum);
  }

  double extract_arm_joint_delta(
    const std::string& side,
    const moveit::core::RobotState& from,
    const moveit::core::RobotState& to) const
  {
    const auto* group = side == "left" ? left_arm_group_ : right_arm_group_;
    if (!group) return 0.0;
    double sum = 0.0;
    const auto& names = group->getVariableNames();
    for (const auto& name : names) {
      if (!is_robot_variable(name)) continue;
      const double delta = to.getVariablePosition(name) - from.getVariablePosition(name);
      sum += delta * delta;
    }
    return std::sqrt(sum);
  }

  double extract_tip_position_delta(
    const moveit::core::RobotState& from,
    const moveit::core::RobotState& to) const
  {
    const auto& from_tf = from.getGlobalLinkTransform(left_tip_);
    const auto& to_tf = to.getGlobalLinkTransform(left_tip_);
    return (to_tf.translation() - from_tf.translation()).norm();
  }

  double extract_tip_position_delta_for_side(
    const std::string& side,
    const moveit::core::RobotState& from,
    const moveit::core::RobotState& to) const
  {
    const std::string& tip = side == "left" ? left_tip_ : right_tip_;
    const auto& from_tf = from.getGlobalLinkTransform(tip);
    const auto& to_tf = to.getGlobalLinkTransform(tip);
    return (to_tf.translation() - from_tf.translation()).norm();
  }

  double extract_tip_orientation_delta(
    const moveit::core::RobotState& from,
    const moveit::core::RobotState& to) const
  {
    const auto& from_tf = from.getGlobalLinkTransform(left_tip_);
    const auto& to_tf = to.getGlobalLinkTransform(left_tip_);
    return pose_orientation_error(from_tf, to_tf);
  }

  double extract_tip_orientation_delta_for_side(
    const std::string& side,
    const moveit::core::RobotState& from,
    const moveit::core::RobotState& to) const
  {
    const std::string& tip = side == "left" ? left_tip_ : right_tip_;
    const auto& from_tf = from.getGlobalLinkTransform(tip);
    const auto& to_tf = to.getGlobalLinkTransform(tip);
    return pose_orientation_error(from_tf, to_tf);
  }

  double extract_candidate_score(
    const ExtractCandidate& candidate,
    const moveit::core::RobotState& current_state,
    double last_retreat_x) const
  {
    if (!candidate.state_valid || !candidate.state) {
      return std::numeric_limits<double>::infinity();
    }

    const double retreat_continuity =
      std::abs((candidate.retreat_x - last_retreat_x) - extract_step_x_);
    const double joint_delta = extract_left_arm_joint_delta(current_state, *candidate.state);
    const double tip_position_delta = extract_tip_position_delta(current_state, *candidate.state);
    const double tip_orientation_delta = extract_tip_orientation_delta(current_state, *candidate.state);

    return extract_score_lift_weight_ * candidate.lift_z +
           extract_score_pitch_weight_ * std::abs(candidate.pitch_up_rad) +
           extract_score_retreat_continuity_weight_ * retreat_continuity +
           extract_score_joint_delta_weight_ * joint_delta +
           extract_score_tip_position_delta_weight_ * tip_position_delta +
           extract_score_tip_orientation_delta_weight_ * tip_orientation_delta;
  }

  double extract_candidate_score_for_side(
    const std::string& side,
    const ExtractCandidate& candidate,
    const moveit::core::RobotState& current_state,
    double last_retreat_x) const
  {
    if (!candidate.state_valid || !candidate.state) {
      return std::numeric_limits<double>::infinity();
    }

    const double retreat_continuity =
      std::abs((candidate.retreat_x - last_retreat_x) - extract_step_x_);
    const double joint_delta = extract_arm_joint_delta(side, current_state, *candidate.state);
    const double tip_position_delta = extract_tip_position_delta_for_side(side, current_state, *candidate.state);
    const double tip_orientation_delta = extract_tip_orientation_delta_for_side(side, current_state, *candidate.state);

    return extract_score_lift_weight_ * candidate.lift_z +
           extract_score_pitch_weight_ * std::abs(candidate.pitch_up_rad) +
           extract_score_retreat_continuity_weight_ * retreat_continuity +
           extract_score_joint_delta_weight_ * joint_delta +
           extract_score_tip_position_delta_weight_ * tip_position_delta +
           extract_score_tip_orientation_delta_weight_ * tip_orientation_delta;
  }

  std::vector<ExtractMotionDelta> extract_motion_deltas() const
  {
    return {
      {1.0, 0.0, 0.0},
      {0.8, 0.6, 0.0},
      {0.6, 0.8, 0.0},
      {0.0, 1.0, 0.0},
    };
  }

  std::vector<double> extract_pitch_delta_degrees(double current_pitch_rad) const
  {
    const double current_pitch_deg = current_pitch_rad * 180.0 / M_PI;
    std::vector<double> deltas{0.0, 1.0};
    if (current_pitch_deg >= 1.0) {
      deltas.push_back(-1.0);
    }
    deltas.push_back(3.0);
    if (current_pitch_deg >= 3.0) {
      deltas.push_back(-3.0);
    }
    return deltas;
  }

  double extract_current_pitch_up_rad(const moveit::core::RobotState& state) const
  {
    const Eigen::Vector3d tool_normal =
      state.getGlobalLinkTransform(left_tip_).linear() * Eigen::Vector3d::UnitZ();
    const double x = std::max(0.0, tool_normal.x());
    const double z = tool_normal.z();
    return std::max(0.0, std::atan2(z, x));
  }

  double extract_current_pitch_up_rad_for_side(const std::string& side, const moveit::core::RobotState& state) const
  {
    const std::string& tip = side == "left" ? left_tip_ : right_tip_;
    const Eigen::Vector3d tool_normal =
      state.getGlobalLinkTransform(tip).linear() * Eigen::Vector3d::UnitZ();
    const double x = std::max(0.0, tool_normal.x());
    const double z = tool_normal.z();
    return std::max(0.0, std::atan2(z, x));
  }

  std::vector<ExtractCandidate> make_left_extract_candidates(
    const moveit::core::RobotState& current_state,
    const BoxSpec& source_box,
    const AttachedBoxSpec& left_box,
    int left_box_id,
    size_t step,
    double last_retreat_x,
    double current_lift_z,
    double min_allowed_tip_z) const
  {
    std::vector<ExtractCandidate> candidates;
    size_t candidate_index = 0;
    const double current_pitch = extract_current_pitch_up_rad(current_state);

    for (double pitch_delta_deg : extract_pitch_delta_degrees(current_pitch)) {
      std::vector<ExtractCandidate> layer_candidates;
      const double pitch_delta = pitch_delta_deg * M_PI / 180.0;
      const double pitch_rad = std::max(0.0, current_pitch + pitch_delta);
      for (const auto& delta : extract_motion_deltas()) {
        const double retreat_delta = std::max(0.0, delta.retreat_ratio * extract_step_x_);
        const double retreat_x = std::min(extract_max_x_, last_retreat_x + retreat_delta);
        if (retreat_x <= last_retreat_x + 1e-6 && last_retreat_x >= extract_max_x_ - 1e-6) {
          continue;
        }

        const double lift_delta = std::max(0.0, delta.lift_ratio * extract_step_x_);
        const double lift_z = current_lift_z + lift_delta;
        const BoxSpec shifted_box{left_box_id, source_box.x - retreat_x, source_box.y, source_box.z + lift_z};
        const auto left_pose = make_pose(shifted_box.x, shifted_box.y, shifted_box.z, pitch_up_orientation(pitch_rad));

        ExtractCandidate candidate;
        solve_left_extract_candidate_kdl(current_state, left_pose, step, candidate_index,
                                         retreat_x, retreat_x - last_retreat_x,
                                         lift_z, lift_delta, pitch_rad, pitch_delta,
                                         min_allowed_tip_z, left_box, left_box_id, &candidate);
        layer_candidates.push_back(candidate);
        ++candidate_index;
      }

      const bool layer_has_valid = std::any_of(
        layer_candidates.begin(), layer_candidates.end(),
        [](const ExtractCandidate& candidate) { return candidate.state_valid; });
      candidates.insert(candidates.end(), layer_candidates.begin(), layer_candidates.end());
      if (layer_has_valid) {
        break;
      }
    }

    return candidates;
  }

  std::vector<ExtractCandidate> make_extract_candidates_for_side(
    const std::string& side,
    const moveit::core::RobotState& current_state,
    const BoxSpec& source_box,
    const AttachedBoxSpec& carried_box,
    int box_id,
    size_t step,
    double last_retreat_x,
    double current_lift_z,
    double min_allowed_tip_z) const
  {
    std::vector<ExtractCandidate> candidates;
    size_t candidate_index = 0;
    const double current_pitch = extract_current_pitch_up_rad_for_side(side, current_state);

    for (double pitch_delta_deg : extract_pitch_delta_degrees(current_pitch)) {
      std::vector<ExtractCandidate> layer_candidates;
      const double pitch_delta = pitch_delta_deg * M_PI / 180.0;
      const double pitch_rad = std::max(0.0, current_pitch + pitch_delta);
      for (const auto& delta : extract_motion_deltas()) {
        const double retreat_delta = std::max(0.0, delta.retreat_ratio * extract_step_x_);
        const double retreat_x = std::min(extract_max_x_, last_retreat_x + retreat_delta);
        if (retreat_x <= last_retreat_x + 1e-6 && last_retreat_x >= extract_max_x_ - 1e-6) {
          continue;
        }

        const double lift_delta = std::max(0.0, delta.lift_ratio * extract_step_x_);
        const double lift_z = current_lift_z + lift_delta;
        const BoxSpec shifted_box{box_id, source_box.x - retreat_x, source_box.y, source_box.z + lift_z};
        const auto target_pose = make_pose(shifted_box.x, shifted_box.y, shifted_box.z, pitch_up_orientation(pitch_rad));

        ExtractCandidate candidate;
        solve_extract_candidate_kdl_for_side(side, current_state, target_pose, step, candidate_index,
                                             retreat_x, retreat_x - last_retreat_x,
                                             lift_z, lift_delta, pitch_rad, pitch_delta,
                                             min_allowed_tip_z, carried_box, box_id, &candidate);
        layer_candidates.push_back(candidate);
        ++candidate_index;
      }

      const bool layer_has_valid = std::any_of(
        layer_candidates.begin(), layer_candidates.end(),
        [](const ExtractCandidate& candidate) { return candidate.state_valid; });
      candidates.insert(candidates.end(), layer_candidates.begin(), layer_candidates.end());
      if (layer_has_valid) {
        break;
      }
    }

    return candidates;
  }

  void copy_arm_state(
    const std::string& side,
    const moveit::core::RobotState& from,
    moveit::core::RobotState& to) const
  {
    const auto* group = side == "left" ? left_arm_group_ : right_arm_group_;
    if (!group) return;
    for (const auto& name : group->getVariableNames()) {
      if (is_robot_variable(name)) {
        to.setVariablePosition(name, from.getVariablePosition(name));
      }
    }
  }

  std::vector<ExtractCandidate> top_valid_extract_candidates(
    const std::string& side,
    const std::vector<ExtractCandidate>& candidates,
    const moveit::core::RobotState& current_state,
    double last_retreat_x,
    size_t limit = 6) const
  {
    std::vector<ExtractCandidate> valid;
    for (const auto& candidate : candidates) {
      if (candidate.state_valid && candidate.state) {
        valid.push_back(candidate);
      }
    }
    std::sort(valid.begin(), valid.end(),
              [&](const ExtractCandidate& a, const ExtractCandidate& b) {
                return extract_candidate_score_for_side(side, a, current_state, last_retreat_x) <
                       extract_candidate_score_for_side(side, b, current_state, last_retreat_x);
              });
    if (valid.size() > limit) {
      valid.resize(limit);
    }
    return valid;
  }

  ArmExtractPath rollout_arm_extract_path(
    const std::string& side,
    const moveit::core::RobotState& start_state,
    const AttachedBoxSpec& carried_box,
    int box_id) const
  {
    ArmExtractPath path;
    const auto boxes = make_boxes(box_front_x_);
    const auto source_it = boxes.find(box_id);
    if (source_it == boxes.end()) {
      path.failure_reason = "unknown_box_id";
      return path;
    }

    moveit::core::RobotState current_state(start_state);
    double last_retreat_x = 0.0;
    double current_lift_z = 0.0;
    double min_allowed_tip_z =
      current_state.getGlobalLinkTransform(side == "left" ? left_tip_ : right_tip_).translation().z();
    const size_t max_steps = static_cast<size_t>(
      std::ceil(std::max(0.0, extract_max_x_) / std::max(1e-6, 0.6 * extract_step_x_))) + 2;

    path.states.push_back(std::make_shared<moveit::core::RobotState>(current_state));
    bool detached_seen = false;
    size_t extra_steps_after_detached = 0;
    for (size_t step = 1; step <= max_steps; ++step) {
      const auto candidates =
        make_extract_candidates_for_side(side, current_state, source_it->second, carried_box, box_id,
                                         step, last_retreat_x, current_lift_z, min_allowed_tip_z);
      auto best_it = std::min_element(
        candidates.begin(), candidates.end(),
        [&](const ExtractCandidate& a, const ExtractCandidate& b) {
          return extract_candidate_score_for_side(side, a, current_state, last_retreat_x) <
                 extract_candidate_score_for_side(side, b, current_state, last_retreat_x);
        });
      if (best_it == candidates.end() || !best_it->state_valid || !best_it->state) {
        ++path.failed_steps;
        std::map<std::string, size_t> rejection_counts;
        for (const auto& candidate : candidates) {
          const std::string key = candidate.rejection_reason.empty() ? "unknown" : candidate.rejection_reason;
          rejection_counts[key]++;
        }
        if (!rejection_counts.empty()) {
          size_t best_count = 0;
          for (const auto& [reason, count] : rejection_counts) {
            if (count > best_count) {
              best_count = count;
              path.failure_reason = side + ":" + reason;
            }
          }
        } else {
          path.failure_reason = side + ":no_valid_candidate";
        }
        if (detached_seen && path.success) {
          path.failure_reason.clear();
        }
        break;
      }

      current_state = *best_it->state;
      min_allowed_tip_z =
        std::max(min_allowed_tip_z, current_state.getGlobalLinkTransform(side == "left" ? left_tip_ : right_tip_).translation().z());
      last_retreat_x = best_it->retreat_x;
      current_lift_z = best_it->lift_z;
      path.final_retreat_x = best_it->retreat_x;
      path.final_lift_z = best_it->lift_z;
      path.final_pitch_deg = best_it->pitch_up_rad * 180.0 / M_PI;
      path.selected_candidates.push_back(*best_it);
      path.states.push_back(std::make_shared<moveit::core::RobotState>(current_state));
      ++path.accepted_steps;

      if (best_it->detached_from_neighbors) {
        detached_seen = true;
        path.success = true;
        path.failure_reason.clear();
      }
      if (detached_seen) {
        ++extra_steps_after_detached;
      }
      if (detached_seen && extra_steps_after_detached > extract_success_extra_steps_) {
        path.success = true;
        path.failure_reason.clear();
        break;
      }
    }

    if (!path.success && path.failure_reason.empty()) {
      path.failure_reason = side + ":reached_max_retreat_without_neighbor_detachment";
    }
    return path;
  }

  moveit::core::RobotState combine_async_arm_states(
    const moveit::core::RobotState& base_state,
    const ArmExtractPath& left_path,
    const ArmExtractPath& right_path,
    size_t left_index,
    size_t right_index) const
  {
    moveit::core::RobotState state(base_state);
    const auto& left_state = *left_path.states[std::min(left_index, left_path.states.size() - 1)];
    const auto& right_state = *right_path.states[std::min(right_index, right_path.states.size() - 1)];
    copy_arm_state("left", left_state, state);
    copy_arm_state("right", right_state, state);
    state.setVariablePosition("updown", current_updown(base_state));
    state.enforceBounds(joint_group_);
    state.update();
    return state;
  }

  bool validate_async_extract_path(
    const moveit::core::RobotState& start_state,
    const ArmExtractPath& left_path,
    const ArmExtractPath& right_path,
    const AttachedBoxSpec& left_box,
    int left_box_id,
    const AttachedBoxSpec& right_box,
    int right_box_id,
    std::vector<moveit::core::RobotStatePtr>* combined_states,
    std::string* reason) const
  {
    if (!left_path.success) {
      if (reason) *reason = left_path.failure_reason.empty() ? "left_async_extract_failed" : left_path.failure_reason;
      return false;
    }
    if (!right_path.success) {
      if (reason) *reason = right_path.failure_reason.empty() ? "right_async_extract_failed" : right_path.failure_reason;
      return false;
    }
    if (left_path.states.empty() || right_path.states.empty()) {
      if (reason) *reason = "empty_async_extract_path";
      return false;
    }

    const size_t step_count = std::max(left_path.states.size(), right_path.states.size());
    if (combined_states) {
      combined_states->clear();
      combined_states->reserve(step_count);
    }
    for (size_t step = 0; step < step_count; ++step) {
      auto state = std::make_shared<moveit::core::RobotState>(
        combine_async_arm_states(start_state, left_path, right_path, step, step));
      bool left_detached = false;
      bool right_detached = false;
      std::string state_reason;
      if (!state_clear_for_dual_extract(*state, left_box, left_box_id, right_box, right_box_id,
                                        &left_detached, &right_detached, &state_reason)) {
        if (reason) *reason = "async_combined_step_" + std::to_string(step) + ":" + state_reason;
        return false;
      }
      if (combined_states) {
        combined_states->push_back(state);
      }
    }
    return true;
  }

  DualExtractStepCandidate select_dual_extract_step_candidate(
    const moveit::core::RobotState& current_state,
    const std::vector<ExtractCandidate>& left_candidates,
    const std::vector<ExtractCandidate>& right_candidates,
    double left_last_retreat_x,
    double right_last_retreat_x,
    const AttachedBoxSpec& left_box,
    int left_box_id,
    const AttachedBoxSpec& right_box,
    int right_box_id) const
  {
    DualExtractStepCandidate best;
    const auto left_valid = top_valid_extract_candidates("left", left_candidates, current_state, left_last_retreat_x);
    const auto right_valid = top_valid_extract_candidates("right", right_candidates, current_state, right_last_retreat_x);
    if (left_valid.empty() || right_valid.empty()) {
      best.rejection_reason = left_valid.empty() ? "no_valid_left_extract_candidate" : "no_valid_right_extract_candidate";
      return best;
    }

    for (const auto& left_candidate : left_valid) {
      for (const auto& right_candidate : right_valid) {
        auto state = std::make_shared<moveit::core::RobotState>(current_state);
        copy_arm_state("left", *left_candidate.state, *state);
        copy_arm_state("right", *right_candidate.state, *state);
        state->setVariablePosition("updown", current_updown(current_state));
        state->enforceBounds(joint_group_);
        state->update();

        bool left_detached = false;
        bool right_detached = false;
        std::string reason;
        if (!state_clear_for_dual_extract(*state, left_box, left_box_id, right_box, right_box_id,
                                          &left_detached, &right_detached, &reason)) {
          if (best.rejection_reason.empty()) {
            best.rejection_reason = reason;
          }
          continue;
        }

        const double score =
          extract_candidate_score_for_side("left", left_candidate, current_state, left_last_retreat_x) +
          extract_candidate_score_for_side("right", right_candidate, current_state, right_last_retreat_x) +
          0.2 * std::abs(left_candidate.retreat_x - right_candidate.retreat_x) +
          0.2 * std::abs(left_candidate.lift_z - right_candidate.lift_z);
        if (!best.state_valid || score < best.score) {
          best.left = left_candidate;
          best.right = right_candidate;
          best.state = state;
          best.state_valid = true;
          best.left_detached = left_detached;
          best.right_detached = right_detached;
          best.score = score;
          best.rejection_reason.clear();
        }
      }
    }
    if (!best.state_valid && best.rejection_reason.empty()) {
      best.rejection_reason = "no_collision_free_dual_extract_candidate";
    }
    return best;
  }

  nlohmann::json ik_candidate_rejection_counts_json(const ik_benchmark::UpdownAwareIkResult& result) const
  {
    std::map<std::string, size_t> counts;
    for (const auto& candidate : result.candidates) {
      if (candidate.legal) {
        counts["legal"]++;
      } else if (!candidate.rejection_reason.empty()) {
        counts[candidate.rejection_reason]++;
      } else {
        counts["unknown"]++;
      }
    }
    nlohmann::json out = nlohmann::json::object();
    for (const auto& [reason, count] : counts) {
      out[reason] = count;
    }
    return out;
  }

  moveit::core::RobotState state_from_ik_candidate(
    const moveit::core::RobotState& seed_state,
    const ik_benchmark::UpdownAwareIkCandidate& candidate) const
  {
    moveit::core::RobotState state(seed_state);
    for (size_t i = 0; i < candidate.full_joint_names.size() && i < candidate.full_joint_values.size(); ++i) {
      const auto& name = candidate.full_joint_names[i];
      if (is_robot_variable(name)) {
        state.setVariablePosition(name, candidate.full_joint_values[i]);
      }
    }
    state.enforceBounds(joint_group_);
    state.update();
    return state;
  }

  struct IkDedupStats
  {
    bool enabled = false;
    size_t input_count = 0;
    size_t unique_count = 0;
    size_t selected_count = 0;
    size_t removed_count = 0;
    double elapsed_ms = 0.0;
  };

  static double wrapped_angle_delta(double lhs, double rhs)
  {
    double delta = std::fmod(lhs - rhs + M_PI, 2.0 * M_PI);
    if (delta < 0.0) {
      delta += 2.0 * M_PI;
    }
    return std::abs(delta - M_PI);
  }

  std::optional<double> candidate_joint_value(
    const ik_benchmark::UpdownAwareIkCandidate& candidate,
    const std::string& joint_name) const
  {
    for (size_t i = 0; i < candidate.full_joint_names.size() && i < candidate.full_joint_values.size(); ++i) {
      if (candidate.full_joint_names[i] == joint_name) {
        return candidate.full_joint_values[i];
      }
    }
    return std::nullopt;
  }

  bool ik_candidates_similar(
    const ik_benchmark::UpdownAwareIkCandidate& candidate,
    const ik_benchmark::UpdownAwareIkCandidate& kept) const
  {
    if (extract_ik_dedup_h_threshold_ >= 0.0 &&
        std::abs(candidate.h - kept.h) > extract_ik_dedup_h_threshold_) {
      return false;
    }
    if (extract_ik_dedup_joint_threshold_ <= 0.0) {
      return false;
    }

    static const std::array<const char*, 12> arm_joints = {
      "left_v5_joint1", "left_v5_joint2", "left_v5_joint3",
      "left_v5_joint4", "left_v5_joint5", "left_v5_joint6",
      "right_v5_joint1", "right_v5_joint2", "right_v5_joint3",
      "right_v5_joint4", "right_v5_joint5", "right_v5_joint6",
    };

    size_t compared = 0;
    for (const auto* joint_name : arm_joints) {
      const auto lhs = candidate_joint_value(candidate, joint_name);
      const auto rhs = candidate_joint_value(kept, joint_name);
      if (!lhs || !rhs) {
        continue;
      }
      ++compared;
      if (wrapped_angle_delta(*lhs, *rhs) > extract_ik_dedup_joint_threshold_) {
        return false;
      }
    }
    return compared > 0;
  }

  std::vector<ik_benchmark::UpdownAwareIkCandidate> select_benchmark_ik_candidates(
    const std::vector<ik_benchmark::UpdownAwareIkCandidate>& sorted_legal_candidates,
    IkDedupStats* stats) const
  {
    IkDedupStats local_stats;
    local_stats.enabled = extract_ik_dedup_enabled_ && extract_ik_dedup_joint_threshold_ > 0.0;
    local_stats.input_count = sorted_legal_candidates.size();

    std::vector<ik_benchmark::UpdownAwareIkCandidate> selected;
    if (local_stats.enabled) {
      const auto start = std::chrono::steady_clock::now();
      selected.reserve(sorted_legal_candidates.size());
      for (const auto& candidate : sorted_legal_candidates) {
        bool duplicate = false;
        for (const auto& kept : selected) {
          if (ik_candidates_similar(candidate, kept)) {
            duplicate = true;
            break;
          }
        }
        if (!duplicate) {
          selected.push_back(candidate);
        }
      }
      local_stats.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    } else {
      selected = sorted_legal_candidates;
    }

    local_stats.unique_count = selected.size();
    local_stats.removed_count = local_stats.input_count > local_stats.unique_count
      ? local_stats.input_count - local_stats.unique_count
      : 0;
    if (extract_benchmark_candidate_limit_ > 0 && selected.size() > extract_benchmark_candidate_limit_) {
      selected.resize(extract_benchmark_candidate_limit_);
    }
    local_stats.selected_count = selected.size();
    if (stats) {
      *stats = local_stats;
    }
    return selected;
  }

  struct ExtractRolloutTiming
  {
    size_t candidate_order = 0;
    size_t h_index = 0;
    size_t seed_index = 0;
    double h = 0.0;
    double ik_score = 0.0;
    double ik_solve_ms = 0.0;
    double rollout_ms = 0.0;
    double interval_ms = 0.0;
    bool success = false;
    bool loaded_plan_attempted = false;
    bool loaded_plan_success = false;
    double loaded_plan_ms = 0.0;
    size_t loaded_plan_points = 0;
    size_t selected_left_loaded_pose_index = 0;
    size_t selected_right_loaded_pose_index = 0;
    double selected_left_loaded_pose_distance = 0.0;
    double selected_right_loaded_pose_distance = 0.0;
    double loaded_pose_distance_sum = 0.0;
    double loaded_pose_distance_l2 = 0.0;
    double loaded_pose_max_joint_delta = 0.0;
    size_t loaded_plan_rank = 0;
    size_t accepted_steps = 0;
    size_t failed_steps = 0;
    double final_retreat_x = 0.0;
    double final_lift_z = 0.0;
    double final_pitch_deg = 0.0;
    double right_final_retreat_x = 0.0;
    double right_final_lift_z = 0.0;
    double right_final_pitch_deg = 0.0;
    std::string failure_reason;
    std::string loaded_plan_failure_reason;
    moveit::core::RobotStatePtr final_state;
  };

  std::vector<std::string> arm_joint_target_names() const
  {
    return {
      "updown",
      "left_v5_joint1", "left_v5_joint2", "left_v5_joint3",
      "left_v5_joint4", "left_v5_joint5", "left_v5_joint6",
      "right_v5_joint1", "right_v5_joint2", "right_v5_joint3",
      "right_v5_joint4", "right_v5_joint5", "right_v5_joint6",
    };
  }

  double arm_pose_distance(
    const moveit::core::RobotState& state,
    const std::string& prefix,
    const std::vector<double>& pose) const
  {
    if (pose.size() < 6) return std::numeric_limits<double>::infinity();
    double squared_sum = 0.0;
    for (size_t i = 0; i < 6; ++i) {
      const std::string joint_name = prefix + "_v5_joint" + std::to_string(i + 1);
      if (!is_robot_variable(joint_name)) {
        return std::numeric_limits<double>::infinity();
      }
      const double diff = shortest_angular_distance(state.getVariablePosition(joint_name), pose[i]);
      squared_sum += diff * diff;
    }
    return std::sqrt(squared_sum);
  }

  std::array<double, 3> loaded_pose_distance_metrics(
    const moveit::core::RobotState& state,
    size_t left_index,
    size_t right_index) const
  {
    if (left_index >= left_loaded_pose_family_.size() || right_index >= right_loaded_pose_family_.size()) {
      return {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()
      };
    }
    const auto& left_pose = left_loaded_pose_family_[left_index];
    const auto& right_pose = right_loaded_pose_family_[right_index];
    double abs_sum = 0.0;
    double squared_sum = 0.0;
    double max_delta = 0.0;
    for (size_t i = 0; i < 6; ++i) {
      const std::array<std::pair<std::string, const std::vector<double>*>, 2> arms = {{
        {std::string("left_v5_joint") + std::to_string(i + 1), &left_pose},
        {std::string("right_v5_joint") + std::to_string(i + 1), &right_pose},
      }};
      for (const auto& [joint_name, pose] : arms) {
        if (!is_robot_variable(joint_name) || pose->size() <= i) {
          return {
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()
          };
        }
        const double delta = std::abs(shortest_angular_distance(state.getVariablePosition(joint_name), (*pose)[i]));
        abs_sum += delta;
        squared_sum += delta * delta;
        max_delta = std::max(max_delta, delta);
      }
    }
    return {abs_sum, std::sqrt(squared_sum), max_delta};
  }

  void fill_loaded_pose_distance_metrics(ExtractRolloutTiming& timing) const
  {
    if (!timing.final_state) {
      return;
    }
    timing.selected_left_loaded_pose_index = nearest_loaded_pose_index(
      *timing.final_state, "left", left_loaded_pose_family_, &timing.selected_left_loaded_pose_distance);
    timing.selected_right_loaded_pose_index = nearest_loaded_pose_index(
      *timing.final_state, "right", right_loaded_pose_family_, &timing.selected_right_loaded_pose_distance);
    const auto metrics = loaded_pose_distance_metrics(
      *timing.final_state,
      timing.selected_left_loaded_pose_index,
      timing.selected_right_loaded_pose_index);
    timing.loaded_pose_distance_sum = metrics[0];
    timing.loaded_pose_distance_l2 = metrics[1];
    timing.loaded_pose_max_joint_delta = metrics[2];
  }

  size_t nearest_loaded_pose_index(
    const moveit::core::RobotState& state,
    const std::string& prefix,
    const std::vector<std::vector<double>>& family,
    double* distance = nullptr) const
  {
    size_t best_index = 0;
    double best_distance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < family.size(); ++i) {
      const double candidate_distance = arm_pose_distance(state, prefix, family[i]);
      if (candidate_distance < best_distance) {
        best_distance = candidate_distance;
        best_index = i;
      }
    }
    if (distance) {
      *distance = best_distance;
    }
    return best_index;
  }

  moveit::core::RobotState loaded_goal_from_extract_state(
    const moveit::core::RobotState& extract_state,
    size_t* selected_left_index = nullptr,
    size_t* selected_right_index = nullptr,
    double* selected_left_distance = nullptr,
    double* selected_right_distance = nullptr) const
  {
    moveit::core::RobotState goal_state(extract_state);
    const size_t left_index = nearest_loaded_pose_index(
      extract_state, "left", left_loaded_pose_family_, selected_left_distance);
    const size_t right_index = nearest_loaded_pose_index(
      extract_state, "right", right_loaded_pose_family_, selected_right_distance);
    if (selected_left_index) *selected_left_index = left_index;
    if (selected_right_index) *selected_right_index = right_index;
    const auto& left_pose = left_loaded_pose_family_[left_index];
    const auto& right_pose = right_loaded_pose_family_[right_index];
    if (is_robot_variable("updown")) {
      goal_state.setVariablePosition("updown", extract_loaded_target_updown_);
    }
    for (size_t i = 0; i < left_pose.size(); ++i) {
      goal_state.setVariablePosition("left_v5_joint" + std::to_string(i + 1), left_pose[i]);
    }
    for (size_t i = 0; i < right_pose.size(); ++i) {
      goal_state.setVariablePosition("right_v5_joint" + std::to_string(i + 1), right_pose[i]);
    }
    goal_state.enforceBounds(joint_group_);
    goal_state.update();
    return goal_state;
  }

  bool plan_loaded_from_extract_state(
    const std::string& stage_name,
    const moveit::core::RobotState& extract_state,
    const std::vector<AttachedBoxSpec>& carried_boxes,
    ExtractRolloutTiming* timing)
  {
    if (!loaded_move_group_) {
      if (timing) timing->loaded_plan_failure_reason = "loaded_move_group_not_initialized";
      return false;
    }

    const auto saved_boxes = active_attached_boxes_;
    active_attached_boxes_ = carried_boxes;

    auto restore_boxes = [&]() {
      std::vector<std::string> ids;
      ids.reserve(carried_boxes.size());
      for (const auto& box : carried_boxes) ids.push_back(box.id);
      remove_carried_box_ids(ids, "extract_loaded_plan_detach_box");
      active_attached_boxes_ = saved_boxes;
    };

    if (!apply_attached_box_state(active_attached_boxes_, moveit_msgs::msg::CollisionObject::ADD,
                                  "extract_loaded_plan_attach_box")) {
      if (timing) timing->loaded_plan_failure_reason = "failed_to_attach_box_for_loaded_plan";
      restore_boxes();
      return false;
    }

    size_t selected_left_loaded_pose_index = 0;
    size_t selected_right_loaded_pose_index = 0;
    double selected_left_loaded_pose_distance = 0.0;
    double selected_right_loaded_pose_distance = 0.0;
    moveit::core::RobotState goal_state = loaded_goal_from_extract_state(
      extract_state,
      &selected_left_loaded_pose_index,
      &selected_right_loaded_pose_index,
      &selected_left_loaded_pose_distance,
      &selected_right_loaded_pose_distance);
    const auto loaded_pose_metrics = loaded_pose_distance_metrics(
      extract_state,
      selected_left_loaded_pose_index,
      selected_right_loaded_pose_index);
    const auto target_names = arm_joint_target_names();

    loaded_move_group_->setStartState(extract_state);
    loaded_move_group_->setJointValueTarget(goal_state);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const auto t0 = std::chrono::steady_clock::now();
    const auto plan_result = loaded_move_group_->plan(plan);
    const auto t1 = std::chrono::steady_clock::now();
    const double plan_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (timing) {
      timing->loaded_plan_attempted = true;
      timing->loaded_plan_ms = plan_ms;
      timing->loaded_plan_points = plan.trajectory_.joint_trajectory.points.size();
      timing->selected_left_loaded_pose_index = selected_left_loaded_pose_index;
      timing->selected_right_loaded_pose_index = selected_right_loaded_pose_index;
      timing->selected_left_loaded_pose_distance = selected_left_loaded_pose_distance;
      timing->selected_right_loaded_pose_distance = selected_right_loaded_pose_distance;
      timing->loaded_pose_distance_sum = loaded_pose_metrics[0];
      timing->loaded_pose_distance_l2 = loaded_pose_metrics[1];
      timing->loaded_pose_max_joint_delta = loaded_pose_metrics[2];
    }

    if (plan_result != moveit::core::MoveItErrorCode::SUCCESS) {
      if (timing) {
        timing->loaded_plan_failure_reason = "moveit_planning_failed_code_" + std::to_string(plan_result.val);
      }
      restore_boxes();
      return false;
    }

    std::string carried_collision_reason;
    const bool carried_clear = planned_carried_boxes_clear_static_obstacles(plan, extract_state, &carried_collision_reason);
    if (!carried_clear) {
      if (timing) {
        timing->loaded_plan_failure_reason = carried_collision_reason;
      }
    }

    if (record_stream_) {
      nlohmann::json extra = {
        {"stage_kind", "post_extract_loaded_plan"},
        {"valid", carried_clear},
        {"loaded_plan_ms", plan_ms},
        {"loaded_plan_points", plan.trajectory_.joint_trajectory.points.size()},
        {"start_updown", current_updown(extract_state)},
        {"target_updown", current_updown(goal_state)},
        {"carried_box_count", carried_boxes.size()},
        {"loaded_plan_rank", timing ? timing->loaded_plan_rank : 0},
        {"loaded_pose_distance_sum", loaded_pose_metrics[0]},
        {"loaded_pose_distance_l2", loaded_pose_metrics[1]},
        {"loaded_pose_max_joint_delta", loaded_pose_metrics[2]},
        {"loaded_pose_max_joint_delta_deg", loaded_pose_metrics[2] * 180.0 / M_PI},
        {"selected_left_loaded_pose_index", selected_left_loaded_pose_index},
        {"selected_right_loaded_pose_index", selected_right_loaded_pose_index},
        {"selected_left_loaded_pose_distance", selected_left_loaded_pose_distance},
        {"selected_right_loaded_pose_distance", selected_right_loaded_pose_distance},
        {"selected_left_loaded_pose_deg", pose_degrees_json(left_loaded_pose_family_[selected_left_loaded_pose_index])},
        {"selected_right_loaded_pose_deg", pose_degrees_json(right_loaded_pose_family_[selected_right_loaded_pose_index])},
        {"failure_reason", carried_clear ? "" : carried_collision_reason}
      };
      record_stage(stage_name, plan, extract_state, goal_state, target_names, extra);
    }

    if (!carried_clear) {
      restore_boxes();
      return false;
    }

    if (timing) {
      timing->loaded_plan_success = true;
      timing->loaded_plan_failure_reason.clear();
    }
    restore_boxes();
    return true;
  }

  bool plan_loaded_from_extract_state(
    const std::string& stage_name,
    const moveit::core::RobotState& extract_state,
    const AttachedBoxSpec& left_box,
    ExtractRolloutTiming* timing)
  {
    return plan_loaded_from_extract_state(stage_name, extract_state, std::vector<AttachedBoxSpec>{left_box}, timing);
  }

  ExtractRolloutTiming rollout_left_extract_from_state(
    const moveit::core::RobotState& start_state,
    const AttachedBoxSpec& left_box,
    int left_box_id,
    size_t candidate_order,
    const ik_benchmark::UpdownAwareIkCandidate& ik_candidate,
    const std::function<void(size_t, const moveit::core::RobotState&, const nlohmann::json&)>& record_step = {}) const
  {
    ExtractRolloutTiming timing;
    timing.candidate_order = candidate_order;
    timing.h_index = ik_candidate.h_index;
    timing.seed_index = ik_candidate.seed_index;
    timing.h = ik_candidate.h;
    timing.ik_score = ik_candidate.score;
    timing.ik_solve_ms = ik_candidate.solve_ms;

    const auto boxes = make_boxes(box_front_x_);
    const auto left_it = boxes.find(left_box_id);
    if (left_it == boxes.end()) {
      timing.failure_reason = "unknown_left_box_id";
      return timing;
    }

    const auto t0 = std::chrono::steady_clock::now();
    moveit::core::RobotState current_state(start_state);
    double last_retreat_x = 0.0;
    double current_lift_z = 0.0;
    double min_allowed_tip_z = current_state.getGlobalLinkTransform(left_tip_).translation().z();
    const size_t max_steps = static_cast<size_t>(
      std::ceil(std::max(0.0, extract_max_x_) / std::max(1e-6, 0.6 * extract_step_x_))) + 2;

    if (record_step) {
      nlohmann::json extra = {
        {"stage_kind", "left_extract_all_legal_ik_start"},
        {"candidate_order", candidate_order},
        {"h_index", ik_candidate.h_index},
        {"seed_index", ik_candidate.seed_index},
        {"h", ik_candidate.h},
        {"ik_score", ik_candidate.score},
        {"ik_solve_ms", ik_candidate.solve_ms},
        {"accepted", true},
        {"step", 0},
        {"retreat_x", 0.0},
        {"lift_z", 0.0},
        {"pitch_up_deg", 0.0},
        {"left_tip_z", current_state.getGlobalLinkTransform(left_tip_).translation().z()},
        {"tool_normal_z", (current_state.getGlobalLinkTransform(left_tip_).linear() * Eigen::Vector3d::UnitZ()).z()},
        {"detached_from_neighbors", false}
      };
      record_step(0, current_state, extra);
    }

    for (size_t step = 1; step <= max_steps; ++step) {
      std::vector<ExtractCandidate> candidates =
        make_left_extract_candidates(current_state, left_it->second, left_box, left_box_id,
                                     step, last_retreat_x, current_lift_z, min_allowed_tip_z);

      auto best_it = std::min_element(candidates.begin(), candidates.end(), [&](const ExtractCandidate& a, const ExtractCandidate& b) {
        return extract_candidate_score(a, current_state, last_retreat_x) <
               extract_candidate_score(b, current_state, last_retreat_x);
      });

      if (best_it == candidates.end() || !best_it->state_valid) {
        ++timing.failed_steps;
        std::map<std::string, size_t> rejection_counts;
        for (const auto& candidate : candidates) {
          const std::string key = candidate.rejection_reason.empty() ? "unknown" : candidate.rejection_reason;
          rejection_counts[key]++;
        }
        if (!rejection_counts.empty()) {
          timing.failure_reason = rejection_counts.begin()->first;
          size_t best_count = 0;
          for (const auto& [reason, count] : rejection_counts) {
            if (count > best_count) {
              best_count = count;
              timing.failure_reason = reason;
            }
          }
        } else {
          timing.failure_reason = "no_valid_candidate";
        }
        if (record_step) {
          nlohmann::json rejection_json = nlohmann::json::object();
          for (const auto& [reason, count] : rejection_counts) {
            rejection_json[reason] = count;
          }
          nlohmann::json extra = {
            {"stage_kind", "left_extract_all_legal_ik_failed_step"},
            {"candidate_order", candidate_order},
            {"h_index", ik_candidate.h_index},
            {"seed_index", ik_candidate.seed_index},
            {"h", ik_candidate.h},
            {"ik_score", ik_candidate.score},
            {"ik_solve_ms", ik_candidate.solve_ms},
            {"step", step},
            {"retreat_x", last_retreat_x},
            {"accepted", false},
            {"candidate_count", candidates.size()},
            {"rejection_counts", rejection_json},
            {"failure_reason", timing.failure_reason}
          };
          record_step(step, current_state, extra);
        }
        if (extract_fail_fast_) break;
        continue;
      }

      const double selected_score = extract_candidate_score(*best_it, current_state, last_retreat_x);
      const double selected_joint_delta = extract_left_arm_joint_delta(current_state, *best_it->state);
      const double selected_tip_position_delta = extract_tip_position_delta(current_state, *best_it->state);
      const double selected_tip_orientation_delta = extract_tip_orientation_delta(current_state, *best_it->state);

      current_state = *best_it->state;
      min_allowed_tip_z = std::max(min_allowed_tip_z, current_state.getGlobalLinkTransform(left_tip_).translation().z());
      last_retreat_x = best_it->retreat_x;
      current_lift_z = best_it->lift_z;
      timing.final_retreat_x = best_it->retreat_x;
      timing.final_lift_z = best_it->lift_z;
      timing.final_pitch_deg = best_it->pitch_up_rad * 180.0 / M_PI;
      ++timing.accepted_steps;

      if (record_step) {
        nlohmann::json extra = {
          {"stage_kind", "left_extract_all_legal_ik_step"},
          {"candidate_order", candidate_order},
          {"h_index", ik_candidate.h_index},
          {"seed_index", ik_candidate.seed_index},
          {"h", ik_candidate.h},
          {"ik_score", ik_candidate.score},
          {"ik_solve_ms", ik_candidate.solve_ms},
          {"step", step},
          {"candidate_index", best_it->candidate_index},
          {"retreat_x", best_it->retreat_x},
          {"retreat_delta_x", best_it->retreat_delta_x},
          {"lift_z", best_it->lift_z},
          {"lift_delta_z", best_it->lift_delta_z},
          {"pitch_up_deg", best_it->pitch_up_rad * 180.0 / M_PI},
          {"pitch_delta_deg", best_it->pitch_delta_rad * 180.0 / M_PI},
          {"left_tip_z", current_state.getGlobalLinkTransform(left_tip_).translation().z()},
          {"tool_normal_z", (current_state.getGlobalLinkTransform(left_tip_).linear() * Eigen::Vector3d::UnitZ()).z()},
          {"detached_from_neighbors", best_it->detached_from_neighbors},
          {"candidate_count", candidates.size()},
          {"score", selected_score},
          {"joint_delta", selected_joint_delta},
          {"tip_position_delta", selected_tip_position_delta},
          {"tip_orientation_delta", selected_tip_orientation_delta},
          {"accepted", true}
        };
        record_step(step, current_state, extra);
      }

      if (best_it->detached_from_neighbors) {
        timing.success = true;
        timing.failure_reason.clear();
        timing.final_state = std::make_shared<moveit::core::RobotState>(current_state);
        break;
      }
    }

    if (!timing.success && timing.failure_reason.empty()) {
      timing.failure_reason = "reached_max_retreat_without_neighbor_detachment";
    }
    const auto t1 = std::chrono::steady_clock::now();
    timing.rollout_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return timing;
  }

  ExtractRolloutTiming rollout_dual_extract_from_state(
    const moveit::core::RobotState& start_state,
    const AttachedBoxSpec& left_box,
    int left_box_id,
    const AttachedBoxSpec& right_box,
    int right_box_id,
    size_t candidate_order,
    const ik_benchmark::UpdownAwareIkCandidate& ik_candidate,
    const std::function<void(size_t, const moveit::core::RobotState&, const nlohmann::json&)>& record_step = {}) const
  {
    ExtractRolloutTiming timing;
    timing.candidate_order = candidate_order;
    timing.h_index = ik_candidate.h_index;
    timing.seed_index = ik_candidate.seed_index;
    timing.h = ik_candidate.h;
    timing.ik_score = ik_candidate.score;
    timing.ik_solve_ms = ik_candidate.solve_ms;

    const auto boxes = make_boxes(box_front_x_);
    const auto left_it = boxes.find(left_box_id);
    const auto right_it = boxes.find(right_box_id);
    if (left_it == boxes.end() || right_it == boxes.end()) {
      timing.failure_reason = "unknown_box_id";
      return timing;
    }

    const auto t0 = std::chrono::steady_clock::now();
    moveit::core::RobotState current_state(start_state);
    if (extract_benchmark_dual_async_) {
      const auto left_path = rollout_arm_extract_path("left", start_state, left_box, left_box_id);
      const auto right_path = rollout_arm_extract_path("right", start_state, right_box, right_box_id);
      timing.accepted_steps = left_path.accepted_steps + right_path.accepted_steps;
      timing.failed_steps = left_path.failed_steps + right_path.failed_steps;
      timing.final_retreat_x = left_path.final_retreat_x;
      timing.final_lift_z = left_path.final_lift_z;
      timing.final_pitch_deg = left_path.final_pitch_deg;
      timing.right_final_retreat_x = right_path.final_retreat_x;
      timing.right_final_lift_z = right_path.final_lift_z;
      timing.right_final_pitch_deg = right_path.final_pitch_deg;

      std::vector<moveit::core::RobotStatePtr> combined_states;
      std::string async_reason;
      const bool async_valid = validate_async_extract_path(
        start_state, left_path, right_path, left_box, left_box_id, right_box, right_box_id,
        &combined_states, &async_reason);
      if (!async_valid) {
        timing.failure_reason = async_reason;
      } else {
        timing.success = true;
        timing.failure_reason.clear();
        timing.final_state = combined_states.empty() ? std::make_shared<moveit::core::RobotState>(start_state) : combined_states.back();
      }

      if (record_step) {
        nlohmann::json start_extra = {
          {"stage_kind", "dual_extract_async_start"},
          {"candidate_order", candidate_order},
          {"h_index", ik_candidate.h_index},
          {"seed_index", ik_candidate.seed_index},
          {"h", ik_candidate.h},
          {"ik_score", ik_candidate.score},
          {"ik_solve_ms", ik_candidate.solve_ms},
          {"accepted", true},
          {"step", 0},
          {"left_path_success", left_path.success},
          {"right_path_success", right_path.success},
          {"left_path_steps", left_path.accepted_steps},
          {"right_path_steps", right_path.accepted_steps},
          {"async_valid", async_valid},
          {"failure_reason", timing.failure_reason}
        };
        record_step(0, start_state, start_extra);
        const size_t recorded_step_count = std::max(
          combined_states.size(),
          std::max(left_path.states.empty() ? 0 : left_path.states.size() - 1,
                   right_path.states.empty() ? 0 : right_path.states.size() - 1));
        for (size_t step = 1; step <= recorded_step_count; ++step) {
          const auto left_index = std::min(step, left_path.selected_candidates.size());
          const auto right_index = std::min(step, right_path.selected_candidates.size());
          auto display_state = combined_states.empty() || step > combined_states.size()
            ? std::make_shared<moveit::core::RobotState>(
                combine_async_arm_states(start_state, left_path, right_path, left_index, right_index))
            : combined_states[step - 1];
          nlohmann::json extra = {
            {"stage_kind", "dual_extract_async_step"},
            {"candidate_order", candidate_order},
            {"h_index", ik_candidate.h_index},
            {"seed_index", ik_candidate.seed_index},
            {"h", ik_candidate.h},
            {"ik_score", ik_candidate.score},
            {"ik_solve_ms", ik_candidate.solve_ms},
            {"step", step},
            {"accepted", true},
            {"left_path_success", left_path.success},
            {"right_path_success", right_path.success},
            {"left_path_steps", left_path.accepted_steps},
            {"right_path_steps", right_path.accepted_steps},
            {"async_valid", async_valid},
            {"failure_reason", timing.failure_reason}
          };
          if (left_index > 0 && left_index <= left_path.selected_candidates.size()) {
            const auto& left = left_path.selected_candidates[left_index - 1];
            extra["left_retreat_x"] = left.retreat_x;
            extra["left_lift_z"] = left.lift_z;
            extra["left_pitch_up_deg"] = left.pitch_up_rad * 180.0 / M_PI;
            extra["left_detached_from_neighbors"] = left.detached_from_neighbors;
          }
          if (right_index > 0 && right_index <= right_path.selected_candidates.size()) {
            const auto& right = right_path.selected_candidates[right_index - 1];
            extra["right_retreat_x"] = right.retreat_x;
            extra["right_lift_z"] = right.lift_z;
            extra["right_pitch_up_deg"] = right.pitch_up_rad * 180.0 / M_PI;
            extra["right_detached_from_neighbors"] = right.detached_from_neighbors;
          }
          record_step(step, *display_state, extra);
        }
      }

      const auto t1 = std::chrono::steady_clock::now();
      timing.rollout_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      return timing;
    }

    double left_last_retreat_x = 0.0;
    double right_last_retreat_x = 0.0;
    double left_lift_z = 0.0;
    double right_lift_z = 0.0;
    double left_min_tip_z = current_state.getGlobalLinkTransform(left_tip_).translation().z();
    double right_min_tip_z = current_state.getGlobalLinkTransform(right_tip_).translation().z();
    const size_t max_steps = static_cast<size_t>(
      std::ceil(std::max(0.0, extract_max_x_) / std::max(1e-6, 0.6 * extract_step_x_))) + 2;

    if (record_step) {
      nlohmann::json extra = {
        {"stage_kind", "dual_extract_all_legal_ik_start"},
        {"candidate_order", candidate_order},
        {"h_index", ik_candidate.h_index},
        {"seed_index", ik_candidate.seed_index},
        {"h", ik_candidate.h},
        {"ik_score", ik_candidate.score},
        {"ik_solve_ms", ik_candidate.solve_ms},
        {"accepted", true},
        {"step", 0},
        {"left_retreat_x", 0.0},
        {"right_retreat_x", 0.0},
        {"left_lift_z", 0.0},
        {"right_lift_z", 0.0},
        {"left_detached_from_neighbors", false},
        {"right_detached_from_neighbors", false}
      };
      record_step(0, current_state, extra);
    }

    for (size_t step = 1; step <= max_steps; ++step) {
      const auto left_candidates =
        make_extract_candidates_for_side("left", current_state, left_it->second, left_box, left_box_id,
                                         step, left_last_retreat_x, left_lift_z, left_min_tip_z);
      const auto right_candidates =
        make_extract_candidates_for_side("right", current_state, right_it->second, right_box, right_box_id,
                                         step, right_last_retreat_x, right_lift_z, right_min_tip_z);
      const auto best = select_dual_extract_step_candidate(
        current_state,
        left_candidates,
        right_candidates,
        left_last_retreat_x,
        right_last_retreat_x,
        left_box,
        left_box_id,
        right_box,
        right_box_id);

      if (!best.state_valid || !best.state) {
        ++timing.failed_steps;
        timing.failure_reason = best.rejection_reason.empty() ? "no_valid_dual_extract_candidate" : best.rejection_reason;
        if (record_step) {
          nlohmann::json extra = {
            {"stage_kind", "dual_extract_all_legal_ik_failed_step"},
            {"candidate_order", candidate_order},
            {"h_index", ik_candidate.h_index},
            {"seed_index", ik_candidate.seed_index},
            {"h", ik_candidate.h},
            {"ik_score", ik_candidate.score},
            {"ik_solve_ms", ik_candidate.solve_ms},
            {"step", step},
            {"accepted", false},
            {"left_candidate_count", left_candidates.size()},
            {"right_candidate_count", right_candidates.size()},
            {"failure_reason", timing.failure_reason}
          };
          record_step(step, current_state, extra);
        }
        if (extract_fail_fast_) break;
        continue;
      }

      current_state = *best.state;
      left_min_tip_z = std::max(left_min_tip_z, current_state.getGlobalLinkTransform(left_tip_).translation().z());
      right_min_tip_z = std::max(right_min_tip_z, current_state.getGlobalLinkTransform(right_tip_).translation().z());
      left_last_retreat_x = best.left.retreat_x;
      right_last_retreat_x = best.right.retreat_x;
      left_lift_z = best.left.lift_z;
      right_lift_z = best.right.lift_z;
      timing.final_retreat_x = best.left.retreat_x;
      timing.final_lift_z = best.left.lift_z;
      timing.final_pitch_deg = best.left.pitch_up_rad * 180.0 / M_PI;
      timing.right_final_retreat_x = best.right.retreat_x;
      timing.right_final_lift_z = best.right.lift_z;
      timing.right_final_pitch_deg = best.right.pitch_up_rad * 180.0 / M_PI;
      ++timing.accepted_steps;

      if (record_step) {
        nlohmann::json extra = {
          {"stage_kind", "dual_extract_all_legal_ik_step"},
          {"candidate_order", candidate_order},
          {"h_index", ik_candidate.h_index},
          {"seed_index", ik_candidate.seed_index},
          {"h", ik_candidate.h},
          {"ik_score", ik_candidate.score},
          {"ik_solve_ms", ik_candidate.solve_ms},
          {"step", step},
          {"left_candidate_index", best.left.candidate_index},
          {"right_candidate_index", best.right.candidate_index},
          {"left_retreat_x", best.left.retreat_x},
          {"right_retreat_x", best.right.retreat_x},
          {"left_retreat_delta_x", best.left.retreat_delta_x},
          {"right_retreat_delta_x", best.right.retreat_delta_x},
          {"left_lift_z", best.left.lift_z},
          {"right_lift_z", best.right.lift_z},
          {"left_pitch_up_deg", best.left.pitch_up_rad * 180.0 / M_PI},
          {"right_pitch_up_deg", best.right.pitch_up_rad * 180.0 / M_PI},
          {"left_detached_from_neighbors", best.left_detached},
          {"right_detached_from_neighbors", best.right_detached},
          {"left_candidate_count", left_candidates.size()},
          {"right_candidate_count", right_candidates.size()},
          {"score", best.score},
          {"accepted", true}
        };
        record_step(step, current_state, extra);
      }

      if (best.left_detached && best.right_detached) {
        timing.success = true;
        timing.failure_reason.clear();
        timing.final_state = std::make_shared<moveit::core::RobotState>(current_state);
        break;
      }
    }

    if (!timing.success && timing.failure_reason.empty()) {
      timing.failure_reason = "reached_max_retreat_without_dual_neighbor_detachment";
    }
    const auto t1 = std::chrono::steady_clock::now();
    timing.rollout_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return timing;
  }

  bool write_extract_benchmark_csv(const std::vector<ExtractRolloutTiming>& timings) const
  {
    if (extract_benchmark_csv_path_.empty()) return true;
    const std::filesystem::path path(extract_benchmark_csv_path_);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
      RCLCPP_WARN(get_logger(), "Failed to open extract benchmark CSV: %s", extract_benchmark_csv_path_.c_str());
      return false;
    }
    out << "candidate_order,h_index,seed_index,h,ik_score,ik_solve_ms,rollout_ms,interval_ms,success,"
           "loaded_plan_attempted,loaded_plan_success,loaded_plan_ms,loaded_plan_points,"
           "loaded_plan_rank,loaded_pose_distance_sum,loaded_pose_distance_l2,loaded_pose_max_joint_delta,"
           "selected_left_loaded_pose_index,selected_right_loaded_pose_index,"
           "selected_left_loaded_pose_distance,selected_right_loaded_pose_distance,"
           "accepted_steps,failed_steps,final_retreat_x,final_lift_z,final_pitch_deg,"
           "right_final_retreat_x,right_final_lift_z,right_final_pitch_deg,"
           "failure_reason,loaded_plan_failure_reason\n";
    out << std::setprecision(12);
    for (const auto& timing : timings) {
      out << timing.candidate_order << ','
          << timing.h_index << ','
          << timing.seed_index << ','
          << timing.h << ','
          << timing.ik_score << ','
          << timing.ik_solve_ms << ','
          << timing.rollout_ms << ','
          << timing.interval_ms << ','
          << (timing.success ? 1 : 0) << ','
          << (timing.loaded_plan_attempted ? 1 : 0) << ','
          << (timing.loaded_plan_success ? 1 : 0) << ','
          << timing.loaded_plan_ms << ','
          << timing.loaded_plan_points << ','
          << timing.loaded_plan_rank << ','
          << timing.loaded_pose_distance_sum << ','
          << timing.loaded_pose_distance_l2 << ','
          << timing.loaded_pose_max_joint_delta << ','
          << timing.selected_left_loaded_pose_index << ','
          << timing.selected_right_loaded_pose_index << ','
          << timing.selected_left_loaded_pose_distance << ','
          << timing.selected_right_loaded_pose_distance << ','
          << timing.accepted_steps << ','
          << timing.failed_steps << ','
          << timing.final_retreat_x << ','
          << timing.final_lift_z << ','
          << timing.final_pitch_deg << ','
          << timing.right_final_retreat_x << ','
          << timing.right_final_lift_z << ','
          << timing.right_final_pitch_deg << ','
          << '"' << timing.failure_reason << '"' << ','
          << '"' << timing.loaded_plan_failure_reason << '"' << '\n';
    }
    RCLCPP_INFO(get_logger(), "Wrote extract all-legal-IK timing CSV: %s rows=%zu",
                extract_benchmark_csv_path_.c_str(), timings.size());
    return true;
  }

  ExtractRolloutTiming run_dual_extract_candidate(
    const std::string& prefix,
    const moveit::core::RobotState& seed_state,
    const std::vector<ik_benchmark::UpdownAwareIkCandidate>& legal_candidates,
    const AttachedBoxSpec& left_box,
    int left_box_id,
    const AttachedBoxSpec& right_box,
    int right_box_id,
    size_t index,
    bool record_rollout)
  {
    auto state = state_from_ik_candidate(seed_state, legal_candidates[index]);
    std::function<void(size_t, const moveit::core::RobotState&, const nlohmann::json&)> record_step;
    if (record_rollout) {
      const std::vector<AttachedBoxSpec> boxes{left_box, right_box};
      record_step = [&, boxes, index](size_t step_index, const moveit::core::RobotState& rollout_state, const nlohmann::json& extra) {
        record_extract_keyframe(
          prefix + "/candidate_" + std::to_string(index) + "/step_" + std::to_string(step_index),
          rollout_state,
          boxes,
          extra);
      };
    }
    auto timing = rollout_dual_extract_from_state(
      state, left_box, left_box_id, right_box, right_box_id, index, legal_candidates[index], record_step);
    if (timing.success && timing.final_state) {
      fill_loaded_pose_distance_metrics(timing);
    }
    return timing;
  }

  bool benchmark_all_legal_ik_extract(
    const std::string& prefix,
    const moveit::core::RobotState& seed_state,
    const ik_benchmark::UpdownAwareIkResult& ik_result,
    const AttachedBoxSpec& left_box,
    int left_box_id)
  {
    std::vector<ik_benchmark::UpdownAwareIkCandidate> legal_candidates;
    for (const auto& candidate : ik_result.candidates) {
      if (candidate.legal) {
        legal_candidates.push_back(candidate);
      }
    }
    std::sort(legal_candidates.begin(), legal_candidates.end(),
              [](const auto& a, const auto& b) { return a.score < b.score; });
    if (legal_candidates.empty()) {
      return fail(prefix + "/extract_benchmark: no legal IK candidates");
    }
    const size_t original_legal_count = legal_candidates.size();
    IkDedupStats dedup_stats;
    legal_candidates = select_benchmark_ik_candidates(legal_candidates, &dedup_stats);

    if (record_tip_error_ik_candidates_) {
      record_tip_error_ik_candidates(prefix, seed_state, ik_result, left_box);
    }

    std::vector<ExtractRolloutTiming> timings;
    timings.reserve(legal_candidates.size());
    auto previous_start = std::chrono::steady_clock::now();
    bool any_success = false;
    for (size_t i = 0; i < legal_candidates.size(); ++i) {
      const auto start = std::chrono::steady_clock::now();
      auto state = state_from_ik_candidate(seed_state, legal_candidates[i]);
      std::function<void(size_t, const moveit::core::RobotState&, const nlohmann::json&)> record_step;
      if (extract_benchmark_record_rollouts_) {
        record_step = [&](size_t step_index, const moveit::core::RobotState& rollout_state, const nlohmann::json& extra) {
          record_extract_keyframe(
            prefix + "/candidate_" + std::to_string(i) + "/step_" + std::to_string(step_index),
            rollout_state,
            left_box,
            extra);
        };
      }
      auto timing = rollout_left_extract_from_state(state, left_box, left_box_id, i, legal_candidates[i], record_step);
      if (timing.success && timing.final_state) {
        fill_loaded_pose_distance_metrics(timing);
      }
      timing.interval_ms = std::chrono::duration<double, std::milli>(start - previous_start).count();
      previous_start = start;
      any_success = any_success || timing.success;
      timings.push_back(std::move(timing));
    }
    if (!timings.empty()) {
      timings.front().interval_ms = 0.0;
    }

    size_t loaded_plan_requests = 0;
    std::vector<size_t> loaded_plan_indices;
    if (extract_benchmark_plan_loaded_after_success_) {
      for (size_t i = 0; i < timings.size(); ++i) {
        if (timings[i].success && timings[i].final_state) {
          loaded_plan_indices.push_back(i);
        }
      }
      if (extract_loaded_sort_by_pose_distance_) {
        std::sort(loaded_plan_indices.begin(), loaded_plan_indices.end(),
                  [&](const size_t a, const size_t b) {
                    const auto& lhs = timings[a];
                    const auto& rhs = timings[b];
                    if (lhs.loaded_pose_distance_sum != rhs.loaded_pose_distance_sum) {
                      return lhs.loaded_pose_distance_sum < rhs.loaded_pose_distance_sum;
                    }
                    if (lhs.loaded_pose_distance_l2 != rhs.loaded_pose_distance_l2) {
                      return lhs.loaded_pose_distance_l2 < rhs.loaded_pose_distance_l2;
                    }
                    if (lhs.loaded_pose_max_joint_delta != rhs.loaded_pose_max_joint_delta) {
                      return lhs.loaded_pose_max_joint_delta < rhs.loaded_pose_max_joint_delta;
                    }
                    return lhs.ik_score < rhs.ik_score;
                  });
      }
      const size_t loaded_limit = extract_loaded_candidate_limit_ > 0
        ? std::min(extract_loaded_candidate_limit_, loaded_plan_indices.size())
        : loaded_plan_indices.size();
      for (size_t rank = 0; rank < loaded_plan_indices.size(); ++rank) {
        auto& timing = timings[loaded_plan_indices[rank]];
        timing.loaded_plan_rank = rank + 1;
        if (rank < loaded_limit) {
          ++loaded_plan_requests;
          plan_loaded_from_extract_state(
            prefix + "/candidate_" + std::to_string(timing.candidate_order) + "/post_extract_loaded",
            *timing.final_state,
            left_box,
            &timing);
          timing.loaded_plan_rank = rank + 1;
        } else {
          timing.loaded_plan_failure_reason = "loaded_plan_skipped_by_limit";
        }
      }
    }

    write_extract_benchmark_csv(timings);

    double total_interval_ms = 0.0;
    double total_rollout_ms = 0.0;
    double total_loaded_plan_ms = 0.0;
    size_t loaded_plan_attempted_count = 0;
    size_t loaded_plan_success_count = 0;
    for (const auto& timing : timings) {
      total_interval_ms += timing.interval_ms;
      total_rollout_ms += timing.rollout_ms;
      if (timing.loaded_plan_attempted) {
        ++loaded_plan_attempted_count;
        total_loaded_plan_ms += timing.loaded_plan_ms;
      }
      if (timing.loaded_plan_success) {
        ++loaded_plan_success_count;
      }
    }
    const double mean_interval_ms = timings.size() > 1
      ? total_interval_ms / static_cast<double>(timings.size() - 1)
      : 0.0;
    const double mean_rollout_ms = total_rollout_ms / static_cast<double>(timings.size());
    const double mean_loaded_plan_ms = loaded_plan_attempted_count > 0
      ? total_loaded_plan_ms / static_cast<double>(loaded_plan_attempted_count)
      : 0.0;
    RCLCPP_INFO(get_logger(),
                "[%s/extract_benchmark] legal_ik=%zu selected=%zu dedup=%s unique=%zu removed=%zu dedup_ms=%.3f success_any=%s loaded_plan=%zu/%zu mean_interval=%.3fms mean_rollout=%.3fms mean_loaded_plan=%.3fms",
                prefix.c_str(), original_legal_count, timings.size(),
                dedup_stats.enabled ? "true" : "false", dedup_stats.unique_count,
                dedup_stats.removed_count, dedup_stats.elapsed_ms,
                any_success ? "true" : "false",
                loaded_plan_success_count, loaded_plan_attempted_count,
                mean_interval_ms, mean_rollout_ms, mean_loaded_plan_ms);

    if (record_stream_) {
      record_stream_ << nlohmann::json({
        {"type", "extract_benchmark_summary"},
        {"legal_ik_count", original_legal_count},
        {"tested_ik_count", legal_candidates.size()},
        {"candidate_limit", extract_benchmark_candidate_limit_},
        {"ik_dedup_enabled", dedup_stats.enabled},
        {"ik_dedup_joint_threshold_deg", extract_ik_dedup_joint_threshold_ * 180.0 / M_PI},
        {"ik_dedup_h_threshold", extract_ik_dedup_h_threshold_},
        {"ik_dedup_input_count", dedup_stats.input_count},
        {"ik_dedup_unique_count", dedup_stats.unique_count},
        {"ik_dedup_removed_count", dedup_stats.removed_count},
        {"ik_dedup_selected_count", dedup_stats.selected_count},
        {"ik_dedup_ms", dedup_stats.elapsed_ms},
        {"success_any", any_success},
        {"mean_interval_ms", mean_interval_ms},
        {"mean_rollout_ms", mean_rollout_ms},
        {"loaded_plan_after_success", extract_benchmark_plan_loaded_after_success_},
        {"loaded_plan_candidate_limit", extract_loaded_candidate_limit_},
        {"loaded_plan_sort_by_pose_distance", extract_loaded_sort_by_pose_distance_},
        {"loaded_plan_sorted_success_candidate_count", loaded_plan_indices.size()},
        {"loaded_plan_attempted_count", loaded_plan_attempted_count},
        {"loaded_plan_success_count", loaded_plan_success_count},
        {"mean_loaded_plan_ms", mean_loaded_plan_ms},
        {"csv_path", extract_benchmark_csv_path_},
        {"rollouts_recorded", extract_benchmark_record_rollouts_},
        {"ik_candidate_rejection_counts", ik_candidate_rejection_counts_json(ik_result)}
      }).dump() << '\n';
      record_stream_.flush();
    }
    return any_success;
  }

  bool benchmark_all_legal_ik_dual_extract(
    const std::string& prefix,
    const moveit::core::RobotState& seed_state,
    const ik_benchmark::UpdownAwareIkResult& ik_result,
    const AttachedBoxSpec& left_box,
    int left_box_id,
    const AttachedBoxSpec& right_box,
    int right_box_id)
  {
    std::vector<ik_benchmark::UpdownAwareIkCandidate> legal_candidates;
    for (const auto& candidate : ik_result.candidates) {
      if (candidate.legal) {
        legal_candidates.push_back(candidate);
      }
    }
    std::sort(legal_candidates.begin(), legal_candidates.end(),
              [](const auto& a, const auto& b) { return a.score < b.score; });
    if (legal_candidates.empty()) {
      return fail(prefix + "/dual_extract_benchmark: no legal IK candidates");
    }
    const size_t original_legal_count = legal_candidates.size();
    IkDedupStats dedup_stats;
    legal_candidates = select_benchmark_ik_candidates(legal_candidates, &dedup_stats);

    std::vector<ExtractRolloutTiming> timings(legal_candidates.size());
    bool any_success = false;
    const size_t requested_extract_workers = std::max<size_t>(1, extract_benchmark_extract_workers_);
    const size_t used_extract_workers = extract_benchmark_record_rollouts_
      ? 1
      : std::max<size_t>(1, std::min(requested_extract_workers, legal_candidates.size()));
    const auto extract_wall_start = std::chrono::steady_clock::now();
    if (used_extract_workers <= 1) {
      auto previous_start = extract_wall_start;
      for (size_t i = 0; i < legal_candidates.size(); ++i) {
        const auto start = std::chrono::steady_clock::now();
        auto timing = run_dual_extract_candidate(
          prefix, seed_state, legal_candidates, left_box, left_box_id, right_box, right_box_id,
          i, extract_benchmark_record_rollouts_);
        timing.interval_ms = std::chrono::duration<double, std::milli>(start - previous_start).count();
        previous_start = start;
        timings[i] = std::move(timing);
      }
      if (!timings.empty()) {
        timings.front().interval_ms = 0.0;
      }
    } else {
      std::atomic<size_t> next_index{0};
      std::vector<std::thread> workers;
      workers.reserve(used_extract_workers);
      for (size_t worker_index = 0; worker_index < used_extract_workers; ++worker_index) {
        workers.emplace_back([&, worker_index]() {
          (void)worker_index;
          while (true) {
            const size_t i = next_index.fetch_add(1);
            if (i >= legal_candidates.size()) {
              break;
            }
            timings[i] = run_dual_extract_candidate(
              prefix, seed_state, legal_candidates, left_box, left_box_id, right_box, right_box_id,
              i, false);
          }
        });
      }
      for (auto& worker : workers) {
        if (worker.joinable()) {
          worker.join();
        }
      }
      for (auto& timing : timings) {
        timing.interval_ms = 0.0;
      }
    }
    const auto extract_wall_end = std::chrono::steady_clock::now();
    const double extract_wall_ms =
      std::chrono::duration<double, std::milli>(extract_wall_end - extract_wall_start).count();
    for (const auto& timing : timings) {
      any_success = any_success || timing.success;
    }

    size_t loaded_plan_requests = 0;
    double loaded_plan_wall_ms = 0.0;
    std::vector<size_t> loaded_plan_indices;
    if (extract_benchmark_plan_loaded_after_success_) {
      for (size_t i = 0; i < timings.size(); ++i) {
        if (timings[i].success && timings[i].final_state) {
          loaded_plan_indices.push_back(i);
        }
      }
      if (extract_loaded_sort_by_pose_distance_) {
        std::sort(loaded_plan_indices.begin(), loaded_plan_indices.end(),
                  [&](const size_t a, const size_t b) {
                    const auto& lhs = timings[a];
                    const auto& rhs = timings[b];
                    if (lhs.loaded_pose_distance_sum != rhs.loaded_pose_distance_sum) {
                      return lhs.loaded_pose_distance_sum < rhs.loaded_pose_distance_sum;
                    }
                    if (lhs.loaded_pose_distance_l2 != rhs.loaded_pose_distance_l2) {
                      return lhs.loaded_pose_distance_l2 < rhs.loaded_pose_distance_l2;
                    }
                    if (lhs.loaded_pose_max_joint_delta != rhs.loaded_pose_max_joint_delta) {
                      return lhs.loaded_pose_max_joint_delta < rhs.loaded_pose_max_joint_delta;
                    }
                    return lhs.ik_score < rhs.ik_score;
                  });
      }
      const size_t loaded_limit = extract_loaded_candidate_limit_ > 0
        ? std::min(extract_loaded_candidate_limit_, loaded_plan_indices.size())
        : loaded_plan_indices.size();
      const auto loaded_plan_wall_start = std::chrono::steady_clock::now();
      for (size_t rank = 0; rank < loaded_plan_indices.size(); ++rank) {
        auto& timing = timings[loaded_plan_indices[rank]];
        timing.loaded_plan_rank = rank + 1;
        if (rank < loaded_limit) {
          ++loaded_plan_requests;
          plan_loaded_from_extract_state(
            prefix + "/candidate_" + std::to_string(timing.candidate_order) + "/post_extract_loaded",
            *timing.final_state,
            std::vector<AttachedBoxSpec>{left_box, right_box},
            &timing);
          timing.loaded_plan_rank = rank + 1;
          if (extract_loaded_stop_on_first_success_ && timing.loaded_plan_success) {
            for (size_t rest_rank = rank + 1; rest_rank < loaded_plan_indices.size(); ++rest_rank) {
              auto& skipped = timings[loaded_plan_indices[rest_rank]];
              skipped.loaded_plan_rank = rest_rank + 1;
              skipped.loaded_plan_failure_reason = "loaded_plan_skipped_after_first_success";
            }
            break;
          }
        } else {
          timing.loaded_plan_failure_reason = "loaded_plan_skipped_by_limit";
        }
      }
      const auto loaded_plan_wall_end = std::chrono::steady_clock::now();
      loaded_plan_wall_ms =
        std::chrono::duration<double, std::milli>(loaded_plan_wall_end - loaded_plan_wall_start).count();
    }

    write_extract_benchmark_csv(timings);

    double total_interval_ms = 0.0;
    double total_rollout_ms = 0.0;
    double total_loaded_plan_ms = 0.0;
    size_t loaded_plan_attempted_count = 0;
    size_t loaded_plan_success_count = 0;
    for (const auto& timing : timings) {
      total_interval_ms += timing.interval_ms;
      total_rollout_ms += timing.rollout_ms;
      if (timing.loaded_plan_attempted) {
        ++loaded_plan_attempted_count;
        total_loaded_plan_ms += timing.loaded_plan_ms;
      }
      if (timing.loaded_plan_success) {
        ++loaded_plan_success_count;
      }
    }
    const double mean_interval_ms = timings.size() > 1
      ? total_interval_ms / static_cast<double>(timings.size() - 1)
      : 0.0;
    const double mean_rollout_ms = total_rollout_ms / static_cast<double>(timings.size());
    const double mean_loaded_plan_ms = loaded_plan_attempted_count > 0
      ? total_loaded_plan_ms / static_cast<double>(loaded_plan_attempted_count)
      : 0.0;
    const double task_wall_ms =
      ik_result.wall_ms + dedup_stats.elapsed_ms + extract_wall_ms + loaded_plan_wall_ms;
    RCLCPP_INFO(get_logger(),
                "[%s/dual_extract_benchmark] legal_ik=%zu selected=%zu dedup=%s unique=%zu removed=%zu dedup_ms=%.3f extract=%zu workers wall=%.3fms success_any=%s loaded_plan=%zu/%zu wall=%.3fms task_wall=%.3fms mean_interval=%.3fms mean_rollout=%.3fms mean_loaded_plan=%.3fms",
                prefix.c_str(), original_legal_count, timings.size(),
                dedup_stats.enabled ? "true" : "false", dedup_stats.unique_count,
                dedup_stats.removed_count, dedup_stats.elapsed_ms,
                used_extract_workers, extract_wall_ms,
                any_success ? "true" : "false",
                loaded_plan_success_count, loaded_plan_attempted_count, loaded_plan_wall_ms, task_wall_ms,
                mean_interval_ms, mean_rollout_ms, mean_loaded_plan_ms);

    if (record_stream_) {
      record_stream_ << nlohmann::json({
        {"type", "extract_benchmark_summary"},
        {"mode", "dual_extract"},
        {"dual_async", extract_benchmark_dual_async_},
        {"legal_ik_count", original_legal_count},
        {"tested_ik_count", legal_candidates.size()},
        {"candidate_limit", extract_benchmark_candidate_limit_},
        {"ik_dedup_enabled", dedup_stats.enabled},
        {"ik_dedup_joint_threshold_deg", extract_ik_dedup_joint_threshold_ * 180.0 / M_PI},
        {"ik_dedup_h_threshold", extract_ik_dedup_h_threshold_},
        {"ik_dedup_input_count", dedup_stats.input_count},
        {"ik_dedup_unique_count", dedup_stats.unique_count},
        {"ik_dedup_removed_count", dedup_stats.removed_count},
        {"ik_dedup_selected_count", dedup_stats.selected_count},
        {"ik_dedup_ms", dedup_stats.elapsed_ms},
        {"ik_wall_ms", ik_result.wall_ms},
        {"extract_parallel_requested_workers", requested_extract_workers},
        {"extract_parallel_used_workers", used_extract_workers},
        {"extract_parallel_enabled", used_extract_workers > 1},
        {"extract_wall_ms", extract_wall_ms},
        {"extract_sum_rollout_ms", total_rollout_ms},
        {"success_any", any_success},
        {"mean_interval_ms", mean_interval_ms},
        {"mean_rollout_ms", mean_rollout_ms},
        {"loaded_plan_after_success", extract_benchmark_plan_loaded_after_success_},
        {"loaded_plan_candidate_limit", extract_loaded_candidate_limit_},
        {"loaded_plan_sort_by_pose_distance", extract_loaded_sort_by_pose_distance_},
        {"loaded_plan_stop_on_first_success", extract_loaded_stop_on_first_success_},
        {"loaded_plan_sorted_success_candidate_count", loaded_plan_indices.size()},
        {"loaded_plan_attempted_count", loaded_plan_attempted_count},
        {"loaded_plan_success_count", loaded_plan_success_count},
        {"loaded_plan_wall_ms", loaded_plan_wall_ms},
        {"loaded_plan_sum_ms", total_loaded_plan_ms},
        {"mean_loaded_plan_ms", mean_loaded_plan_ms},
        {"task_wall_ms", task_wall_ms},
        {"csv_path", extract_benchmark_csv_path_},
        {"rollouts_recorded", extract_benchmark_record_rollouts_},
        {"ik_candidate_rejection_counts", ik_candidate_rejection_counts_json(ik_result)}
      }).dump() << '\n';
      record_stream_.flush();
    }
    return any_success;
  }

  bool record_extract_keyframe(
    const std::string& stage_name,
    const moveit::core::RobotState& state,
    const std::vector<AttachedBoxSpec>& boxes,
    const nlohmann::json& extra)
  {
    if (!record_stream_) return true;
    const auto saved_boxes = active_attached_boxes_;
    active_attached_boxes_ = boxes;

    trajectory_msgs::msg::JointTrajectory traj;
    const auto names = optimized_ik_solver_ ? optimized_ik_solver_->freeVariableNames() : robot_model_->getVariableNames();
    traj.joint_names = names;
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.time_from_start = rclcpp::Duration::from_seconds(0.0);
    point.positions.reserve(names.size());
    for (const auto& name : names) {
      point.positions.push_back(is_robot_variable(name) ? state.getVariablePosition(name) : 0.0);
    }
    traj.points.push_back(point);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_.joint_trajectory = traj;
    record_stage(stage_name, plan, state, state, names, extra);

    active_attached_boxes_ = saved_boxes;
    return true;
  }

  bool record_extract_keyframe(
    const std::string& stage_name,
    const moveit::core::RobotState& state,
    const AttachedBoxSpec& left_box,
    const nlohmann::json& extra)
  {
    return record_extract_keyframe(stage_name, state, std::vector<AttachedBoxSpec>{left_box}, extra);
  }

  void record_tip_error_ik_candidates(
    const std::string& prefix,
    const moveit::core::RobotState& seed_state,
    const ik_benchmark::UpdownAwareIkResult& ik_result,
    const AttachedBoxSpec& left_box)
  {
    if (!record_stream_ || record_tip_error_ik_candidate_limit_ == 0) return;
    size_t recorded = 0;
    for (size_t i = 0; i < ik_result.candidates.size(); ++i) {
      const auto& candidate = ik_result.candidates[i];
      if (candidate.rejection_reason != "tip_error_too_large" || candidate.full_joint_values.empty()) {
        continue;
      }
      auto state = std::make_shared<moveit::core::RobotState>(seed_state);
      for (size_t j = 0; j < candidate.full_joint_names.size() && j < candidate.full_joint_values.size(); ++j) {
        const auto& name = candidate.full_joint_names[j];
        if (is_robot_variable(name)) {
          state->setVariablePosition(name, candidate.full_joint_values[j]);
        }
      }
      state->enforceBounds(joint_group_);
      state->update();
      nlohmann::json extra = {
        {"stage_kind", "tip_error_ik_candidate"},
        {"candidate_index", i},
        {"h_index", candidate.h_index},
        {"seed_index", candidate.seed_index},
        {"h", candidate.h},
        {"score", candidate.score},
        {"solve_ms", candidate.solve_ms},
        {"direct_pos_error", candidate.direct_pos_error},
        {"direct_ori_error", candidate.direct_ori_error},
        {"swapped_pos_error", candidate.swapped_pos_error},
        {"target_order", candidate.target_order},
        {"rejection_reason", candidate.rejection_reason},
        {"collision_free", candidate.collision_free},
        {"collision_pairs", candidate.collision_pairs}
      };
      record_extract_keyframe(
        prefix + "/tip_error_candidate_" + std::to_string(recorded),
        *state,
        left_box,
        extra);
      ++recorded;
      if (recorded >= record_tip_error_ik_candidate_limit_) break;
    }
  }

  bool plan_left_extract_primitive(int left_box_id, const std::string& prefix)
  {
    auto start_state = last_commanded_state_
      ? std::make_shared<moveit::core::RobotState>(*last_commanded_state_)
      : get_current_robot_state();
    if (!start_state) return fail(prefix + "/extract: cannot get start state");

    const AttachedBoxSpec left_box = make_attached_box_spec("left", left_box_id, false);
    const auto boxes = make_boxes(box_front_x_);
    const auto left_it = boxes.find(left_box_id);
    if (left_it == boxes.end()) return fail(prefix + "/extract: unknown left box id");

    const size_t max_steps = static_cast<size_t>(
      std::ceil(std::max(0.0, extract_max_x_) / std::max(1e-6, 0.6 * extract_step_x_))) + 2;
    std::vector<ExtractCandidate> accepted_path;
    moveit::core::RobotState current_state(*start_state);
    double last_retreat_x = 0.0;
    double current_lift_z = 0.0;
    double min_allowed_tip_z = current_state.getGlobalLinkTransform(left_tip_).translation().z();

    for (size_t step = 1; step <= max_steps; ++step) {
      std::vector<ExtractCandidate> candidates =
        make_left_extract_candidates(current_state, left_it->second, left_box, left_box_id,
                                     step, last_retreat_x, current_lift_z, min_allowed_tip_z);

      auto best_it = std::min_element(candidates.begin(), candidates.end(), [&](const ExtractCandidate& a, const ExtractCandidate& b) {
        return extract_candidate_score(a, current_state, last_retreat_x) <
               extract_candidate_score(b, current_state, last_retreat_x);
      });

      if (best_it == candidates.end() || !best_it->state_valid) {
        std::map<std::string, size_t> rejection_counts;
        for (const auto& candidate : candidates) {
          const std::string key = candidate.rejection_reason.empty() ? "unknown" : candidate.rejection_reason;
          rejection_counts[key]++;
        }
        nlohmann::json rejection_json = nlohmann::json::object();
        for (const auto& [reason, count] : rejection_counts) {
          rejection_json[reason] = count;
        }
        nlohmann::json extra = {
          {"stage_kind", "left_extract_primitive_candidates"},
          {"step", step},
          {"retreat_x", last_retreat_x},
          {"accepted", false},
          {"candidate_count", candidates.size()},
          {"rejection_counts", rejection_json},
        };
        record_extract_keyframe(prefix + "/extract_failed_step_" + std::to_string(step), current_state, left_box, extra);
        if (extract_fail_fast_) return fail(prefix + "/extract: no valid candidate at step " + std::to_string(step));
        continue;
      }

      const double selected_score = extract_candidate_score(*best_it, current_state, last_retreat_x);
      const double selected_joint_delta = extract_left_arm_joint_delta(current_state, *best_it->state);
      const double selected_tip_position_delta = extract_tip_position_delta(current_state, *best_it->state);
      const double selected_tip_orientation_delta = extract_tip_orientation_delta(current_state, *best_it->state);

      current_state = *best_it->state;
      min_allowed_tip_z = std::max(min_allowed_tip_z, current_state.getGlobalLinkTransform(left_tip_).translation().z());
      last_retreat_x = best_it->retreat_x;
      current_lift_z = best_it->lift_z;
      accepted_path.push_back(*best_it);
      nlohmann::json extra = {
        {"stage_kind", "left_extract_primitive"},
        {"extract_ik", "left_v5_arm_kdl_fixed_updown"},
        {"left_box_id", left_box_id},
        {"step", step},
        {"candidate_index", best_it->candidate_index},
        {"retreat_x", best_it->retreat_x},
        {"retreat_delta_x", best_it->retreat_delta_x},
        {"lift_z", best_it->lift_z},
        {"lift_delta_z", best_it->lift_delta_z},
        {"pitch_up_deg", best_it->pitch_up_rad * 180.0 / M_PI},
        {"pitch_delta_deg", best_it->pitch_delta_rad * 180.0 / M_PI},
        {"left_tip_z", current_state.getGlobalLinkTransform(left_tip_).translation().z()},
        {"tool_normal_z", (current_state.getGlobalLinkTransform(left_tip_).linear() * Eigen::Vector3d::UnitZ()).z()},
        {"detached_from_neighbors", best_it->detached_from_neighbors},
        {"candidate_count", candidates.size()},
        {"score", selected_score},
        {"joint_delta", selected_joint_delta},
        {"tip_position_delta", selected_tip_position_delta},
        {"tip_orientation_delta", selected_tip_orientation_delta},
      };
      record_extract_keyframe(prefix + "/extract_step_" + std::to_string(step), current_state, left_box, extra);

      if (best_it->detached_from_neighbors) {
        last_commanded_state_ = std::make_shared<moveit::core::RobotState>(current_state);
        RCLCPP_INFO(get_logger(), "[%s/extract] detached at step=%zu retreat=%.3f lift=%.3f pitch=%.1fdeg",
                    prefix.c_str(), step, best_it->retreat_x, best_it->lift_z, best_it->pitch_up_rad * 180.0 / M_PI);
        return true;
      }
    }

    if (!accepted_path.empty()) {
      last_commanded_state_ = std::make_shared<moveit::core::RobotState>(current_state);
    }
    return fail(prefix + "/extract: reached max retreat without neighbor detachment");
  }

  bool plan_to_goal_state(
    const std::string& stage_name,
    const moveit::core::RobotState& start_state,
    const moveit::core::RobotState& goal_state,
    const std::vector<std::string>& target_names,
    const nlohmann::json& extra = nlohmann::json::object())
  {
    move_group_->setStartState(start_state);
    move_group_->setJointValueTarget(goal_state);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const auto plan_result = move_group_->plan(plan);
    if (plan_result != moveit::core::MoveItErrorCode::SUCCESS) {
      return fail(stage_name + ": MoveIt planning failed, code=" + std::to_string(plan_result.val));
    }

    const auto& trajectory = plan.trajectory_.joint_trajectory;
    RCLCPP_INFO(get_logger(), "[%s] planned points=%zu execute=%s", stage_name.c_str(),
                trajectory.points.size(), execute_ ? "true" : "false");

    std::string carried_collision_reason;
    if (!planned_carried_boxes_clear_static_obstacles(plan, start_state, &carried_collision_reason)) {
      return fail(stage_name + ": carried box collides with static box obstacle (" + carried_collision_reason + ")");
    }

    record_stage(stage_name, plan, start_state, goal_state, target_names, extra);

    if (execute_) {
      const auto exec_result = move_group_->execute(plan);
      if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
        return fail(stage_name + ": MoveIt execute failed, code=" + std::to_string(exec_result.val));
      }
    }

    last_commanded_state_ = std::make_shared<moveit::core::RobotState>(goal_state);
    wait_for_joint_state_near(goal_state, target_names);
    return true;
  }

  bool wait_for_joint_state_near(
    const moveit::core::RobotState& goal_state,
    const std::vector<std::string>& target_names)
  {
    if (!execute_ || state_wait_timeout_s_ <= 0.0) return true;

    const auto deadline = now() + rclcpp::Duration::from_seconds(state_wait_timeout_s_);
    while (rclcpp::ok() && now() < deadline) {
      if (move_group_) {
        auto current_state = move_group_->getCurrentState(0.1);
        if (current_state && robot_state_matches(goal_state, *current_state, target_names)) {
          return true;
        }
      }

      sensor_msgs::msg::JointState::SharedPtr msg;
      {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        msg = latest_joint_state_;
      }
      if (msg && joint_state_matches(goal_state, *msg, target_names)) {
        return true;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(50));
    }

    RCLCPP_WARN(get_logger(),
                "Executed trajectory but /joint_states did not reach target within %.2fs. "
                "Continuing with last commanded state as planning seed. If RViz snaps back, check duplicate /joint_states publishers.",
                state_wait_timeout_s_);
    return false;
  }

  bool robot_state_matches(
    const moveit::core::RobotState& goal_state,
    const moveit::core::RobotState& current_state,
    const std::vector<std::string>& target_names) const
  {
    for (const auto& name : target_names) {
      if (!is_robot_variable(name)) continue;
      const double error = std::abs(current_state.getVariablePosition(name) - goal_state.getVariablePosition(name));
      if (error > joint_goal_tolerance_rad_) {
        return false;
      }
    }
    return true;
  }

  bool joint_state_matches(
    const moveit::core::RobotState& goal_state,
    const sensor_msgs::msg::JointState& msg,
    const std::vector<std::string>& target_names) const
  {
    for (const auto& name : target_names) {
      const auto it = std::find(msg.name.begin(), msg.name.end(), name);
      if (it == msg.name.end()) continue;
      const size_t index = static_cast<size_t>(std::distance(msg.name.begin(), it));
      if (index >= msg.position.size()) continue;
      if (!is_robot_variable(name)) continue;
      const double error = std::abs(msg.position[index] - goal_state.getVariablePosition(name));
      if (error > joint_goal_tolerance_rad_) {
        return false;
      }
    }
    return true;
  }

  bool is_robot_variable(const std::string& name) const
  {
    const auto& variable_names = robot_model_->getVariableNames();
    return std::find(variable_names.begin(), variable_names.end(), name) != variable_names.end();
  }

  std::vector<double> state_values(
    const moveit::core::RobotState& state, const std::vector<std::string>& names) const
  {
    std::vector<double> values;
    values.reserve(names.size());
    for (const auto& name : names) {
      values.push_back(is_robot_variable(name) ? state.getVariablePosition(name) : 0.0);
    }
    return values;
  }

  double current_updown(const moveit::core::RobotState& state) const
  {
    return is_robot_variable("updown") ? state.getVariablePosition("updown") : fixed_updown_;
  }

  void open_record_file()
  {
    if (!record_trajectories_ || record_jsonl_path_.empty()) return;
    const std::filesystem::path path(record_jsonl_path_);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    record_stream_.open(path, std::ios::out | std::ios::trunc);
    if (!record_stream_) {
      RCLCPP_WARN(get_logger(), "Failed to open trajectory record JSONL: %s", record_jsonl_path_.c_str());
      return;
    }

    nlohmann::json header = {
      {"type", "header"},
      {"schema", "moveit_box_stack_flow_v1"},
      {"ik_strategy", "fixed_discrete_h_multi_seed_cost_scorer"},
      {"planning_group", planning_group_},
      {"box_front_x", box_front_x_},
      {"world_to_base_z", world_to_base_z_},
      {"fixed_updown", fixed_updown_},
      {"velocity_scale", velocity_scale_},
      {"acceleration_scale", acceleration_scale_},
      {"max_rounds", max_rounds_},
      {"include_top_suction", include_top_suction_},
      {"execute", execute_},
      {"container_obstacle", container_obstacle_json()},
      {"static_box_obstacles", static_box_obstacles_json()},
      {"attached_box_collision", attached_box_config_json()},
      {"loaded_pose_family", {
        {"left_candidates_deg", pose_family_degrees_json(left_loaded_pose_family_)},
        {"right_candidates_deg", pose_family_degrees_json(right_loaded_pose_family_)},
        {"left_preferred_index", left_preferred_loaded_pose_index_},
        {"right_preferred_index", right_preferred_loaded_pose_index_},
        {"family_distance_weight", ik_config_.cost_loaded_family_distance},
        {"preferred_distance_weight", ik_config_.cost_loaded_preferred_distance}
      }},
      {"ik_config", {
        {"fixed_group", ik_config_.fixed_group},
        {"free_group", ik_config_.free_group},
        {"solver_plugin", ik_config_.solver_plugin},
        {"h_search_mode", "fixed_discrete"},
        {"h_candidate_count", ik_config_.h_candidate_count},
        {"seed_count", ik_config_.seed_count},
        {"workers", ik_config_.workers},
        {"timeout", ik_config_.timeout},
        {"front_z_reach_window", {ik_config_.gripper_z_reach_lower, ik_config_.gripper_z_reach_upper}},
        {"top_z_reach_window", {ik_config_.top_suction_z_reach_lower, ik_config_.top_suction_z_reach_upper}},
        {"h_limits", {ik_config_.h_lower, ik_config_.h_upper}},
        {"check_collision", ik_config_.check_collision},
        {"cost_loaded_family_distance", ik_config_.cost_loaded_family_distance},
        {"cost_loaded_preferred_distance", ik_config_.cost_loaded_preferred_distance}
      }}
    };
    record_stream_ << header.dump() << '\n';
    RCLCPP_INFO(get_logger(), "Recording MoveIt flow JSONL: %s", record_jsonl_path_.c_str());
  }

  nlohmann::json container_obstacle_json() const
  {
    nlohmann::json panels = nlohmann::json::array();
    for (const auto& panel : container_panels()) {
      panels.push_back({
        {"id", panel.id},
        {"center", {panel.center[0], panel.center[1], panel.center[2]}},
        {"size", {panel.size[0], panel.size[1], panel.size[2]}},
      });
    }
    return {
      {"enabled", enable_container_obstacle_},
      {"frame", container_frame_},
      {"length", container_length_},
      {"width", container_width_},
      {"height", container_height_},
      {"center_x", container_center_x_},
      {"center_y", container_center_y_},
      {"floor_z", container_floor_z_},
      {"wall_thickness", container_wall_thickness_},
      {"panels", panels},
    };
  }

  nlohmann::json attached_box_config_json() const
  {
    return {
      {"enabled", enable_attached_box_collision_},
      {"depth", carried_box_depth_},
      {"width", carried_box_width_},
      {"height", carried_box_height_},
    };
  }

  nlohmann::json static_box_obstacles_json() const
  {
    nlohmann::json boxes = nlohmann::json::array();
    for (const auto& box : static_box_obstacles()) {
      boxes.push_back({
        {"id", box.id},
        {"center", {box.center[0], box.center[1], box.center[2]}},
        {"size", {box.size[0], box.size[1], box.size[2]}},
      });
    }
    return {
      {"enabled", enable_static_box_obstacles_},
      {"mode", "dynamic_box_wall_with_pair_opening"},
      {"opening_left_box_id", active_static_left_box_id_},
      {"opening_right_box_id", active_static_right_box_id_},
      {"inset", static_box_obstacle_inset_},
      {"boxes", boxes},
    };
  }

  nlohmann::json active_attached_boxes_json() const
  {
    nlohmann::json boxes = nlohmann::json::array();
    for (const auto& box : active_attached_boxes_) {
      boxes.push_back({
        {"id", box.id},
        {"link_name", box.link_name},
        {"center_in_link", {box.center_in_link[0], box.center_in_link[1], box.center_in_link[2]}},
        {"size", {box.size[0], box.size[1], box.size[2]}},
      });
    }
    return boxes;
  }

  nlohmann::json robot_state_json(const moveit::core::RobotState& state) const
  {
    const auto& names = robot_model_->getVariableNames();
    std::vector<double> values;
    values.reserve(names.size());
    for (const auto& name : names) values.push_back(state.getVariablePosition(name));
    return {{"joint_names", names}, {"joint_values", values}, {"joint_map", names_values_json(names, values)}};
  }

  void record_stage(
    const std::string& stage_name,
    const moveit::planning_interface::MoveGroupInterface::Plan& plan,
    const moveit::core::RobotState& start_state,
    const moveit::core::RobotState& goal_state,
    const std::vector<std::string>& target_names,
    const nlohmann::json& extra)
  {
    if (!record_stream_) return;
    const auto& traj = plan.trajectory_.joint_trajectory;
    nlohmann::json points = nlohmann::json::array();
    for (const auto& point : traj.points) {
      points.push_back({
        {"time_from_start_sec", rclcpp::Duration(point.time_from_start).seconds()},
        {"positions", point.positions},
        {"velocities", point.velocities}
      });
    }

    nlohmann::json record = {
      {"type", "stage"},
      {"stage", stage_name},
      {"stage_index", record_stage_index_++},
      {"trajectory", {
        {"joint_names", traj.joint_names},
        {"point_count", traj.points.size()},
        {"points", points}
      }},
      {"target_names", target_names},
      {"start_state", robot_state_json(start_state)},
      {"goal_state", robot_state_json(goal_state)},
      {"attached_boxes", active_attached_boxes_json()},
      {"static_box_obstacles", static_box_obstacles_json()},
      {"extra", extra}
    };
    record_stream_ << record.dump() << '\n';
    record_stream_.flush();
  }

  geometry_msgs::msg::Pose front_grasp_pose(const BoxSpec& box) const
  {
    return make_pose(box.x, box.y, box.z - world_to_base_z_, forward_x_orientation());
  }

  geometry_msgs::msg::Pose top_suction_pose(const BoxSpec& box) const
  {
    return make_pose(
      box.x + top_suction_x_offset_, box.y, box.z + top_suction_z_offset_ - world_to_base_z_, top_suction_orientation());
  }

  bool run_left_extract_demo()
  {
    clear_carried_boxes_from_scene();
    last_error_.clear();
    last_commanded_state_.reset();

    if (extract_demo_all_rows_) {
      const std::vector<std::pair<int, int>> pairs{{2, 4}, {7, 9}, {12, 14}, {17, 19}};
      bool all_ok = true;
      std::string first_error;
      for (const auto& [left_box_id, right_box_id] : pairs) {
        const bool ok = run_left_extract_pair(left_box_id, right_box_id);
        all_ok = all_ok && ok;
        if (!ok && first_error.empty()) {
          first_error = last_error_;
        }
        clear_carried_boxes_from_scene();
        last_commanded_state_.reset();
      }
      if (record_stream_) {
        record_stream_ << nlohmann::json({
          {"type", "summary"},
          {"success", all_ok},
          {"error", all_ok ? "" : first_error},
          {"stages", record_stage_index_},
          {"pairs", nlohmann::json::array({{2, 4}, {7, 9}, {12, 14}, {17, 19}})}
        }).dump() << '\n';
        record_stream_.flush();
      }
      if (!all_ok && !first_error.empty()) {
        last_error_ = first_error;
      }
      return all_ok;
    }

    return run_left_extract_pair(extract_demo_left_box_id_, extract_demo_right_box_id_);
  }

  bool run_left_extract_pair(int left_box_id, int right_box_id)
  {
    clear_carried_boxes_from_scene();
    last_error_.clear();
    last_commanded_state_.reset();

    const auto boxes = make_boxes(box_front_x_);
    const auto left_it = boxes.find(left_box_id);
    const auto right_it = boxes.find(right_box_id);
    if (left_it == boxes.end() || right_it == boxes.end()) {
      return fail("left extract demo: unknown box id");
    }
    set_static_box_wall_opening(left_box_id, right_box_id, "left_extract_pair");

    const std::string prefix = "left_extract_demo_L" + std::to_string(left_box_id) +
                               "_R" + std::to_string(right_box_id);

    const auto left_pose = front_grasp_pose(left_it->second);
    const auto right_pose = front_grasp_pose(right_it->second);

    if (extract_demo_direct_grasp_start_) {
      auto seed_state = std::make_shared<moveit::core::RobotState>(robot_model_);
      seed_state->setToDefaultValues();
      for (size_t i = 0; i < left_pregrasp_arm_.size(); ++i) {
        seed_state->setVariablePosition("left_v5_joint" + std::to_string(i + 1), left_pregrasp_arm_[i]);
      }
      for (size_t i = 0; i < right_pregrasp_arm_.size(); ++i) {
        seed_state->setVariablePosition("right_v5_joint" + std::to_string(i + 1), right_pregrasp_arm_[i]);
      }
      seed_state->setVariablePosition("updown", extract_grasp_ik_home_updown_);
      seed_state->enforceBounds(joint_group_);
      seed_state->update();

      moveit::core::RobotState grasp_state(*seed_state);
      nlohmann::json grasp_extra;
      ik_benchmark::UpdownAwareIkResult direct_ik_result;
      if (!solve_dual_tip_ik_state(prefix + "/grasp_ik_direct_start", left_pose, right_pose, false,
                                   *seed_state, &grasp_state, &grasp_extra, &direct_ik_result)) {
        return false;
      }
      if (extract_benchmark_all_legal_ik_) {
        const AttachedBoxSpec left_box = make_attached_box_spec("left", left_box_id, false);
        const AttachedBoxSpec right_box = make_attached_box_spec("right", right_box_id, false);
        const std::string original_csv_path = extract_benchmark_csv_path_;
        if (extract_demo_all_rows_ && !original_csv_path.empty()) {
          const std::filesystem::path csv_path(original_csv_path);
          const std::string stem = csv_path.stem().string() + "_L" + std::to_string(left_box_id) +
                                   "_R" + std::to_string(right_box_id);
          extract_benchmark_csv_path_ = (csv_path.parent_path() / (stem + csv_path.extension().string())).string();
        }
        const bool ok = extract_benchmark_dual_arm_
          ? benchmark_all_legal_ik_dual_extract(prefix, *seed_state, direct_ik_result,
                                                left_box, left_box_id, right_box, right_box_id)
          : benchmark_all_legal_ik_extract(prefix, *seed_state, direct_ik_result,
                                           left_box, left_box_id);
        extract_benchmark_csv_path_ = original_csv_path;
        if (record_stream_ && !extract_demo_all_rows_) {
          record_stream_ << nlohmann::json({
            {"type", "summary"},
            {"success", ok},
            {"error", ok ? "" : last_error_},
            {"stages", record_stage_index_}
          }).dump() << '\n';
          record_stream_.flush();
        }
        return ok;
      }
      last_commanded_state_ = std::make_shared<moveit::core::RobotState>(grasp_state);
      record_extract_keyframe(prefix + "/grasp_ik_direct_start", grasp_state,
                              make_attached_box_spec("left", left_box_id, false), grasp_extra);
    } else {
      if (!plan_to_joint_target(prefix + "/pregrasp",
                                make_dual_arm_joint_target(fixed_updown_, left_pregrasp_arm_, right_pregrasp_arm_))) {
        return false;
      }

      if (!plan_dual_tip_ik(prefix + "/grasp_ik", left_pose, right_pose, false)) {
        return false;
      }
    }

    active_attached_boxes_ = {make_attached_box_spec("left", left_box_id, false)};
    if (!apply_attached_box_state(active_attached_boxes_, moveit_msgs::msg::CollisionObject::ADD,
                                  prefix + "/attach_left_carried_box")) {
      active_attached_boxes_.clear();
      return false;
    }

    if (last_commanded_state_) {
      record_extract_keyframe(
        prefix + "/attach_hold",
        *last_commanded_state_,
        active_attached_boxes_.front(),
        {
          {"stage_kind", "attach_hold"},
          {"left_box_id", left_box_id},
          {"note", "attached box added without changing robot joint state"}
        });
    }

    const bool ok = plan_left_extract_primitive(left_box_id, prefix);
    detach_carried_boxes();
    if (record_stream_ && !extract_demo_all_rows_) {
      record_stream_ << nlohmann::json({{"type", "summary"}, {"success", ok}, {"error", ok ? "" : last_error_}, {"stages", record_stage_index_}}).dump() << '\n';
      record_stream_.flush();
    }
    return ok;
  }

  bool run_one_pair_flow(int left_box_id, int right_box_id, bool top_suction, int round)
  {
    clear_carried_boxes_from_scene();
    const auto boxes = make_boxes(box_front_x_);
    const auto left_it = boxes.find(left_box_id);
    const auto right_it = boxes.find(right_box_id);
    if (left_it == boxes.end() || right_it == boxes.end()) {
      return fail("unknown box id in one-pair flow");
    }
    set_static_box_wall_opening(left_box_id, right_box_id, "one_pair_flow");

    const std::string prefix = "round_" + std::to_string(round) + "_L" + std::to_string(left_box_id) +
                               "_R" + std::to_string(right_box_id);
    if (!plan_to_joint_target(prefix + "/pregrasp",
                              make_dual_arm_joint_target(fixed_updown_, left_pregrasp_arm_, right_pregrasp_arm_))) {
      return false;
    }

    const auto left_pose = top_suction ? top_suction_pose(left_it->second) : front_grasp_pose(left_it->second);
    const auto right_pose = top_suction ? top_suction_pose(right_it->second) : front_grasp_pose(right_it->second);
    if (!plan_dual_tip_ik(prefix + "/grasp_ik", left_pose, right_pose, top_suction)) {
      return false;
    }
    if (!attach_carried_boxes(left_box_id, right_box_id, top_suction)) {
      return false;
    }

    if (auto attached_state = get_current_robot_state()) {
      std::string carried_collision_reason;
      if (!carried_boxes_clear_static_obstacles(*attached_state, &carried_collision_reason)) {
        detach_carried_boxes();
        return fail(prefix + "/attach: carried box collides with static box obstacle (" +
                    carried_collision_reason + ")");
      }
    }

    if (!plan_to_joint_target(prefix + "/loaded",
                              make_dual_arm_joint_target(fixed_updown_, left_loaded_arm_, right_loaded_arm_))) {
      detach_carried_boxes();
      return false;
    }

    if (!detach_carried_boxes()) {
      return false;
    }

    if (!plan_to_joint_target(prefix + "/return_pregrasp",
                              make_dual_arm_joint_target(fixed_updown_, left_pregrasp_arm_, right_pregrasp_arm_))) {
      return false;
    }
    return true;
  }

  bool run_box_stack_flow()
  {
    clear_carried_boxes_from_scene();
    last_error_.clear();
    last_commanded_state_.reset();

    const auto pairs = make_pick_pairs(include_top_suction_);
    const int rounds_to_run = std::min<int>(std::max(1, max_rounds_), static_cast<int>(pairs.size()));
    RCLCPP_INFO(get_logger(), "Starting box-stack flow: rounds=%d/%zu execute=%s", rounds_to_run, pairs.size(),
                execute_ ? "true" : "false");

    for (int i = 0; i < rounds_to_run; ++i) {
      const auto& pair = pairs[static_cast<size_t>(i)];
      RCLCPP_INFO(get_logger(), "=== round %d: left box %d, right box %d, mode=%s ===",
                  pair.round, pair.left_box, pair.right_box, pair.top_suction ? "top_suction" : "front");
      if (!run_one_pair_flow(pair.left_box, pair.right_box, pair.top_suction, pair.round)) {
        return false;
      }
    }
    RCLCPP_INFO(get_logger(), "Box-stack flow finished");
    if (record_stream_) {
      record_stream_ << nlohmann::json({{"type", "summary"}, {"success", true}, {"stages", record_stage_index_}}).dump() << '\n';
      record_stream_.flush();
    }
    return true;
  }

  bool fail(const std::string& message)
  {
    last_error_ = message;
    RCLCPP_ERROR(get_logger(), "%s", message.c_str());
    if (record_stream_) {
      record_stream_ << nlohmann::json({{"type", "summary"}, {"success", false}, {"error", message}, {"stages", record_stage_index_}}).dump() << '\n';
      record_stream_.flush();
    }
    return false;
  }

  std::string planning_group_;
  std::string left_tip_;
  std::string right_tip_;
  bool execute_ = true;
  bool reject_ik_collisions_ = false;
  bool check_goal_collision_ = false;
  bool prefer_commanded_state_ = true;
  bool include_top_suction_ = true;
  double fixed_updown_ = 0.45;
  double box_front_x_ = 0.625;
  double world_to_base_z_ = 0.202094;
  double top_suction_x_offset_ = 0.15;
  double top_suction_z_offset_ = 0.2;
  double ik_timeout_ = 2.0;
  double planning_time_ = 8.0;
  double velocity_scale_ = 1.0;
  double acceleration_scale_ = 1.0;
  double joint_goal_tolerance_rad_ = 0.02;
  double state_wait_timeout_s_ = 2.0;
  bool enable_container_obstacle_ = true;
  std::string container_frame_ = "world";
  double container_length_ = 4.0;
  double container_width_ = 2.2;
  double container_height_ = 2.4;
  double container_center_x_ = 0.8;
  double container_center_y_ = 0.0;
  double container_floor_z_ = 0.0;
  double container_wall_thickness_ = 0.02;
  bool enable_attached_box_collision_ = true;
  double carried_box_depth_ = 0.3;
  double carried_box_width_ = 0.4;
  double carried_box_height_ = 0.4;
  bool enable_static_box_obstacles_ = true;
  double static_box_obstacle_inset_ = 0.002;
  int extract_demo_left_box_id_ = 2;
  int extract_demo_right_box_id_ = 4;
  bool extract_demo_all_rows_ = false;
  double extract_step_x_ = 0.03;
  double extract_max_x_ = 0.36;
  std::vector<double> extract_lift_candidates_;
  std::vector<double> extract_pitch_candidates_deg_;
  double extract_neighbor_margin_ = 0.02;
  bool extract_fail_fast_ = false;
  size_t extract_success_extra_steps_ = 3;
  double extract_kdl_timeout_ = 0.01;
  double extract_position_tolerance_ = 0.01;
  double extract_orientation_tolerance_ = 0.05;
  double extract_max_tip_z_drop_ = 0.002;
  double extract_min_tool_normal_z_ = -1e-4;
  double extract_score_lift_weight_ = 10.0;
  double extract_score_pitch_weight_ = 0.02;
  double extract_score_retreat_continuity_weight_ = 0.2;
  double extract_score_joint_delta_weight_ = 0.6;
  double extract_score_tip_position_delta_weight_ = 2.0;
  double extract_score_tip_orientation_delta_weight_ = 0.05;
  double extract_max_joint_delta_ = 0.0;
  bool extract_demo_direct_grasp_start_ = false;
  double extract_grasp_ik_home_updown_ = 0.3;
  bool extract_benchmark_all_legal_ik_ = false;
  bool extract_benchmark_dual_arm_ = false;
  bool extract_benchmark_dual_async_ = false;
  std::string extract_benchmark_csv_path_;
  bool extract_benchmark_record_rollouts_ = false;
  size_t extract_benchmark_candidate_limit_ = 0;
  size_t extract_benchmark_extract_workers_ = 1;
  bool extract_ik_dedup_enabled_ = false;
  double extract_ik_dedup_joint_threshold_ = 1.0 * M_PI / 180.0;
  double extract_ik_dedup_h_threshold_ = 0.005;
  bool extract_benchmark_plan_loaded_after_success_ = false;
  std::string extract_loaded_planning_group_ = "dual_v5_arm_with_base";
  double extract_loaded_planning_time_ = 1.0;
  int extract_loaded_planning_attempts_ = 4;
  size_t extract_loaded_candidate_limit_ = 0;
  bool extract_loaded_sort_by_pose_distance_ = false;
  bool extract_loaded_stop_on_first_success_ = false;
  double extract_loaded_target_updown_ = 0.3;
  bool record_tip_error_ik_candidates_ = false;
  size_t record_tip_error_ik_candidate_limit_ = 80;
  std::string record_jsonl_path_;
  bool record_trajectories_ = true;
  int max_rounds_ = 10;
  int planning_attempts_ = 20;
  size_t record_stage_index_ = 0;
  std::vector<double> left_pregrasp_arm_;
  std::vector<double> right_pregrasp_arm_;
  std::vector<double> left_loaded_arm_;
  std::vector<double> right_loaded_arm_;
  std::vector<std::vector<double>> left_loaded_pose_family_;
  std::vector<std::vector<double>> right_loaded_pose_family_;
  size_t left_preferred_loaded_pose_index_ = 0;
  size_t right_preferred_loaded_pose_index_ = 0;
  std::vector<AttachedBoxSpec> active_attached_boxes_;
  std::vector<StaticBoxObstacle> current_static_box_obstacles_;
  std::vector<std::string> applied_static_box_obstacle_ids_;
  int active_static_left_box_id_ = 0;
  int active_static_right_box_id_ = 0;
  std::string last_error_;
  ik_benchmark::UpdownAwareIkConfig ik_config_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> loaded_move_group_;
  std::unique_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;
  moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::JointModelGroup* joint_group_ = nullptr;
  const moveit::core::JointModelGroup* left_arm_group_ = nullptr;
  const moveit::core::JointModelGroup* right_arm_group_ = nullptr;
  moveit::core::RobotStatePtr last_commanded_state_;
  std::unique_ptr<ik_benchmark::ParallelUpdownAwareIkSolver> optimized_ik_solver_;
  std::ofstream record_stream_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr demo_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr box_stack_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr extract_demo_srv_;
  rclcpp::CallbackGroup::SharedPtr joint_state_callback_group_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  sensor_msgs::msg::JointState::SharedPtr latest_joint_state_;
  std::mutex joint_state_mutex_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto node = std::make_shared<DualArmPlannerNode>(options);

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() { executor.spin(); });

  try {
    node->init();
    spin_thread.join();
  } catch (const std::exception& error) {
    RCLCPP_FATAL(node->get_logger(), "dual_arm_planner init failed: %s", error.what());
    executor.cancel();
    if (spin_thread.joinable()) spin_thread.join();
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
