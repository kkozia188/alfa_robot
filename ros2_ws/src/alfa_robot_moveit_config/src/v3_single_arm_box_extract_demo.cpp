#include "alfa_robot_moveit_config/demo_failure_markers.hpp"
#include <alfa_robot_analytic_ik/v3_redundant_analytic_ik.hpp>
#include <alfa_robot_moveit_config/planning_diagnostics.hpp>
#include <alfa_robot_moveit_config/natural_joint_motion.hpp>
#include <alfa_robot_moveit_config/comfort_height.hpp>
#include <alfa_robot_moveit_config/wall_sequence.hpp>
#include <alfa_robot_moveit_config/shortcut_local_repair.hpp>
#include <ompl/util/RandomNumbers.h>
#include <alfa_robot_moveit_config/srv/plan_wall_box_demo.hpp>

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <builtin_interfaces/msg/duration.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometric_shapes/shapes.h>
#include <interactive_markers/interactive_marker_server.hpp>
#include <interactive_markers/menu_handler.hpp>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <robot_motion_interfaces/action/execute_motion_stage.hpp>
#include <robot_interfaces_qos/profiles.hpp>
#include <robot_system_interfaces/msg/domain_readiness.hpp>
#include <robot_system_interfaces/msg/error_code.hpp>
#include <robot_system_interfaces/msg/error_info.hpp>
#include <robot_rt_control_interfaces/msg/safety_state.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <visualization_msgs/msg/interactive_marker.hpp>
#include <visualization_msgs/msg/interactive_marker_control.hpp>
#include <visualization_msgs/msg/interactive_marker_feedback.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using alfa_robot::analytic_ik::V3RedundantArmAnalyticIk;
using alfa_robot::analytic_ik::V3RedundantArmModel;
using alfa_robot::analytic_ik::V3RedundantIkRequest;
using alfa_robot::analytic_ik::V3RedundantIkSolution;
using WallRequest = alfa_robot_moveit_config::srv::PlanWallBoxDemo;
using StageAction = robot_motion_interfaces::action::ExecuteMotionStage;
using StageGoalHandle = rclcpp_action::ServerGoalHandle<StageAction>;
using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
using FollowJointTrajectoryGoalHandle = rclcpp_action::ClientGoalHandle<FollowJointTrajectory>;
using Feedback = visualization_msgs::msg::InteractiveMarkerFeedback;
using InteractiveMarker = visualization_msgs::msg::InteractiveMarker;
using InteractiveMarkerControl = visualization_msgs::msg::InteractiveMarkerControl;
using Marker = visualization_msgs::msg::Marker;

constexpr double kPi = 3.14159265358979323846;
constexpr char kMarkerName[] = "extract_box";
constexpr char kCarriedBoxId[] = "carried_target_box";
constexpr char kCarriedBoxLeftId[] = "carried_target_box_left";
constexpr char kCarriedBoxRightId[] = "carried_target_box_right";

double degToRad(double value)
{
  return value * kPi / 180.0;
}

double radToDeg(double value)
{
  return value * 180.0 / kPi;
}

double normalizedAngle(double value)
{
  return std::atan2(std::sin(value), std::cos(value));
}

double maximumJointDelta(
  const std::array<double, 7>& from,
  const std::array<double, 7>& to, bool bounded = false)
{
  double maximum = 0.0;
  for (size_t index = 0; index < from.size(); ++index) {
    const double delta = to[index] - from[index];
    maximum = std::max(maximum, std::abs(bounded ? delta : normalizedAngle(delta)));
  }
  return maximum;
}


std_msgs::msg::ColorRGBA color(float red, float green, float blue, float alpha = 1.0F)
{
  std_msgs::msg::ColorRGBA output;
  output.r = red;
  output.g = green;
  output.b = blue;
  output.a = alpha;
  return output;
}

InteractiveMarkerControl axisControl(
  const std::string& name,
  double x,
  double y,
  double z)
{
  InteractiveMarkerControl control;
  control.name = name;
  control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
  control.orientation.w = 1.0;
  control.orientation.x = x;
  control.orientation.y = y;
  control.orientation.z = z;
  return control;
}

geometry_msgs::msg::Pose eigenToPose(const Eigen::Isometry3d& transform)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = transform.translation().x();
  pose.position.y = transform.translation().y();
  pose.position.z = transform.translation().z();
  const Eigen::Quaterniond quaternion(transform.linear());
  pose.orientation.x = quaternion.x();
  pose.orientation.y = quaternion.y();
  pose.orientation.z = quaternion.z();
  pose.orientation.w = quaternion.w();
  return pose;
}

struct ReplayFrame
{
  std::string stage;
  std::vector<double> joints;
  bool box_attached = false;
  bool box_visible = true;
  size_t scene_index = 0;
  nlohmann::json carried_boxes = nlohmann::json::array();
};

struct AnalyticCandidate
{
  moveit::core::RobotStatePtr state;
  V3RedundantIkSolution solution;
  double score = std::numeric_limits<double>::infinity();
};

struct RrtPlanResult
{
  bool success = false;
  double wall_ms = 0.0;
  double planner_ms = 0.0;
  std::string reason;
  std::vector<moveit::core::RobotStatePtr> states;
  moveit::core::RobotStatePtr rejected_state;
};

struct PlanningMetrics
{
  uint64_t ik_calls = 0;
  double ik_ms = 0.0;
  uint64_t collision_checks = 0;
  double collision_ms = 0.0;
  std::vector<double> collision_sample_ms;
  double analytic_path_ms = 0.0;
  double rrt_approach_ms = 0.0;
  double rrt_return_ms = 0.0;
  uint64_t shortcut_connections = 0;
  uint64_t shortcut_direct_successes = 0;
  uint64_t shortcut_blocked_edges = 0;
  uint64_t local_rrt_calls = 0;
  uint64_t local_rrt_failures = 0;
  double local_rrt_wall_ms = 0.0;

  void add(const PlanningMetrics& other)
  {
    ik_calls += other.ik_calls;
    ik_ms += other.ik_ms;
    collision_checks += other.collision_checks;
    collision_ms += other.collision_ms;
    collision_sample_ms.insert(collision_sample_ms.end(),
      other.collision_sample_ms.begin(), other.collision_sample_ms.end());
    analytic_path_ms += other.analytic_path_ms;
    rrt_approach_ms += other.rrt_approach_ms;
    rrt_return_ms += other.rrt_return_ms;
    shortcut_connections += other.shortcut_connections;
    shortcut_direct_successes += other.shortcut_direct_successes;
    shortcut_blocked_edges += other.shortcut_blocked_edges;
    local_rrt_calls += other.local_rrt_calls;
    local_rrt_failures += other.local_rrt_failures;
    local_rrt_wall_ms += other.local_rrt_wall_ms;
  }
};

struct TaskResult
{
  bool success = false;
  std::string failure_stage;
  std::string failure_reason;
  double total_ms = 0.0;
  PlanningMetrics metrics;
  std::vector<ReplayFrame> frames;
  std::vector<ReplayFrame> diagnostic_frames;
  moveit::core::RobotStatePtr rejected_state;
  nlohmann::json diagnostic = nlohmann::json::object();
};

enum class PublicFlowState
{
  Idle,
  PregraspComplete,
  ApproachComplete,
  PlaceComplete,
};

struct ResolvedWallTarget
{
  bool active = false;
  bool top = false;
  std::string side;
  int box_id = -1;
  double wall_distance = 0.0;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
};

struct CachedPublicFlow
{
  uint64_t generation = 0;
  TaskResult plan;
  std::vector<ReplayFrame> pregrasp;
  std::vector<ReplayFrame> approach;
  std::vector<ReplayFrame> place;
  std::vector<ReplayFrame> home;
  std::vector<int> box_ids;
  std::string single_side;
  Eigen::Vector3d display_center = Eigen::Vector3d::Zero();
  bool dual = false;
  bool top = false;
  double wall_distance = 0.0;
};

class V3SingleArmBoxExtractDemo : public rclcpp::Node
{
public:
  explicit V3SingleArmBoxExtractDemo(const rclcpp::NodeOptions& options)
  : Node("v3_single_arm_box_extract_demo", options)
  {}

  void init()
  {
    distance_demo_ = getParameter<bool>("distance_demo", false);
    direct_attach_ = getParameter<bool>("direct_attach", false);
    if (direct_attach_ && !distance_demo_)
      throw std::invalid_argument("direct_attach requires distance_demo");
    direct_placement_pose_ = getParameter<std::string>("direct_placement_pose", "unloading");
    if (direct_placement_pose_ != "unloading" && direct_placement_pose_ != "second_unloading")
      throw std::invalid_argument("direct_placement_pose must be unloading or second_unloading");
    post_extract_policy_ = getParameter<std::string>(
      "post_extract_policy", "rear_release");
    if (post_extract_policy_ != "rear_release" &&
        post_extract_policy_ != "loaded_home") {
      throw std::invalid_argument(
              "post_extract_policy must be rear_release or loaded_home");
    }
    rear_placement_strategy_ = getParameter<std::string>(
      "rear_placement_strategy", "geometric");
    if (rear_placement_strategy_ != "geometric" &&
        rear_placement_strategy_ != "named_unloading") {
      throw std::invalid_argument(
              "rear_placement_strategy must be geometric or named_unloading");
    }
    initial_pose_ = getParameter<std::string>("initial_pose", "home");
    if (initial_pose_ != "home" && initial_pose_ != "second_home" &&
        (initial_pose_ != "arms_down" || !distance_demo_))
      throw std::invalid_argument(
              "initial_pose must be home, second_home, or arms_down for wall simulation");
    side_ = getParameter<std::string>("side", "left");
    world_frame_ = getParameter<std::string>("world_frame", "world");
    arm_base_link_ = getParameter<std::string>("arm_base_link", "arm_carriage");
    planning_group_name_ = getParameter<std::string>(
      "planning_group", side_ == "left" ? "left_arm" : "right_arm");
    tool_link_ = getParameter<std::string>(
      "tool_link", side_ == "left" ? "left_tool0" : "right_tool0");
    const auto initial_box = getParameter<std::vector<double>>(
      "initial_box_center", {0.88, -0.20, 0.55});
    if (initial_box.size() != 3) {
      throw std::invalid_argument("initial_box_center must contain x/y/z");
    }
    box_center_ = Eigen::Vector3d(initial_box[0], initial_box[1], initial_box[2]);
    wall_context_ = getParameter<std::string>("wall_context", "full");
    if (wall_context_ != "full" && wall_context_ != "sequence_prefix" &&
        wall_context_ != "target_only") {
      throw std::invalid_argument(
              "wall_context must be full, sequence_prefix, or target_only");
    }
    scene_layout_ = getParameter<std::string>("scene_layout", "cross");
    if (distance_demo_) scene_layout_ = "wall_5x5";
    wall_target_row_ = getParameter<int>("wall_target_row", 0);
    wall_target_column_ = getParameter<int>("wall_target_column", 2);
    wall_gap_ = getParameter<double>("wall_gap", 0.01);
    if (scene_layout_ != "cross" && scene_layout_ != "wall_5x5") {
      throw std::invalid_argument("scene_layout must be cross or wall_5x5");
    }
    if (wall_target_row_ < 0 || wall_target_row_ >= 5 ||
        wall_target_column_ < 0 || wall_target_column_ >= 5 ||
        !std::isfinite(wall_gap_) || wall_gap_ < 0.0) {
      throw std::invalid_argument("wall target indices must be 0..4 and wall_gap finite/nonnegative");
    }
    box_depth_ = getParameter<double>("box_depth", 0.30);
    box_width_ = getParameter<double>("box_width", 0.40);
    box_height_ = getParameter<double>("box_height", 0.40);
    if (scene_layout_ == "wall_5x5" && !distance_demo_) {
      const auto origin = getParameter<std::vector<double>>("wall_origin", {});
      if (origin.size() != 3) {
        throw std::invalid_argument("wall_origin must contain the bottom-row column-0 box center x/y/z");
      }
      box_center_ = Eigen::Vector3d(origin[0], origin[1], origin[2]) + Eigen::Vector3d(
        0.0, wall_target_column_ * (box_width_ + wall_gap_),
        wall_target_row_ * (box_height_ + wall_gap_));
    }
    initial_box_center_ = box_center_;
    control_handle_clearance_ = std::max(
      0.10, getParameter<double>("control_handle_clearance", 0.25));
    control_handle_lateral_offset_ = std::max(
      0.50, getParameter<double>("control_handle_lateral_offset", 0.90));
    approach_distance_ = getParameter<double>("approach_distance", 0.05);
    retreat_distance_ = getParameter<double>("retreat_distance", 0.35);
    cartesian_step_ = getParameter<double>("cartesian_step", 0.01);
    collision_inset_ = getParameter<double>(
      "collision_inset", scene_layout_ == "wall_5x5" ? 0.0 : 0.002);
    psi_step_ = degToRad(getParameter<double>("psi_step_deg", 5.0));
    maximum_cartesian_joint_step_ = degToRad(
      getParameter<double>("maximum_cartesian_joint_step_deg", 15.0));
    edge_joint_resolution_ = degToRad(
      getParameter<double>("edge_joint_resolution_deg", distance_demo_ ? 0.25 : 2.5));
    precontact_candidate_limit_ = static_cast<size_t>(std::max(
      1, getParameter<int>("precontact_candidate_limit", 8)));
    rrt_planning_time_ = std::max(0.05, getParameter<double>("rrt_planning_time", 1.0));
    rrt_planning_attempts_ = std::max(1, getParameter<int>("rrt_planning_attempts", 1));
    local_rrt_planning_time_ = getParameter<double>("local_rrt_planning_time", 8.0);
    if (!std::isfinite(local_rrt_planning_time_) || local_rrt_planning_time_ <= 0.0)
      throw std::invalid_argument("local_rrt_planning_time must be finite and positive");
    connection_planner_ = getParameter<std::string>("connection_planner",
      "rrt_connect");
    if (connection_planner_ != "shortcut_local_rrt" && connection_planner_ != "rrt_connect")
      throw std::invalid_argument("connection_planner must be shortcut_local_rrt or rrt_connect");
    shortcut_padding_points_ = static_cast<size_t>(std::max(0,
      getParameter<int>("shortcut_padding_points", 5)));
    shortcut_step_ = degToRad(getParameter<double>("shortcut_step_deg", 5.0));
    shortcut_updown_step_ = getParameter<double>("shortcut_updown_step_m", 0.01);
    if (!std::isfinite(shortcut_step_) || shortcut_step_ <= 0.0 ||
        !std::isfinite(shortcut_updown_step_) || shortcut_updown_step_ <= 0.0)
      throw std::invalid_argument("shortcut sampling steps must be finite and positive");
    auto_run_once_ = getParameter<bool>("auto_run_once", false);
    sequence_mode_ = getParameter<bool>("sequence_mode", false);
    playback_enabled_ = getParameter<bool>("playback_enabled", true);
    enable_stage_action_ = getParameter<bool>("enable_stage_action", false);
    execution_backend_ = getParameter<std::string>("execution_backend", "replay");
    follow_joint_trajectory_action_ = getParameter<std::string>(
      "follow_joint_trajectory_action", "/whole_body_jtc/follow_joint_trajectory");
    trajectory_cache_file_ = getParameter<std::string>("trajectory_cache_file", "");
    target_match_tolerance_ = getParameter<double>("target_match_tolerance", 0.06);
    target_orientation_tolerance_ = getParameter<double>(
      "target_orientation_tolerance", degToRad(5.0));
    maximum_rotary_velocity_ = getParameter<double>("maximum_rotary_velocity", degToRad(20.0));
    maximum_updown_velocity_ = getParameter<double>("maximum_updown_velocity", 0.15);
    minimum_trajectory_step_s_ = getParameter<double>("minimum_trajectory_step_s", 0.05);
    display_rate_hz_ = getParameter<double>("display_rate_hz", 20.0);
    if (sequence_mode_ && !distance_demo_) throw std::invalid_argument("sequence_mode requires distance_demo");
    if (enable_stage_action_ && !distance_demo_)
      throw std::invalid_argument("enable_stage_action requires distance_demo");
    if (enable_stage_action_ && (auto_run_once_ || sequence_mode_))
      throw std::invalid_argument("stage Action mode cannot run the legacy automatic sequence");
    if (execution_backend_ != "replay" && execution_backend_ != "fjt")
      throw std::invalid_argument("execution_backend must be replay or fjt");
    if (direct_attach_ && (sequence_mode_ || enable_stage_action_ ||
        execution_backend_ != "replay" || wall_context_ != "target_only" ||
        connection_planner_ != "shortcut_local_rrt"))
      throw std::invalid_argument(
        "direct_attach is a target_only replay experiment, not a wall sequence or hardware command");
    for (const double value : {target_match_tolerance_, target_orientation_tolerance_,
         maximum_rotary_velocity_,
         maximum_updown_velocity_, minimum_trajectory_step_s_, display_rate_hz_})
      if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument("stage action tolerances and trajectory limits must be finite/positive");

    if (side_ != "left" && side_ != "right") {
      throw std::invalid_argument("side must be left or right");
    }
    if (planning_group_name_ != side_ + "_arm" || tool_link_ != side_ + "_tool0" ||
        arm_base_link_ != "arm_carriage") {
      throw std::invalid_argument("planning_group/tool_link/arm_base_link must match the V3.1.1 side");
    }
    for (const double value : {box_depth_, box_width_, box_height_, approach_distance_,
         retreat_distance_, cartesian_step_, psi_step_, maximum_cartesian_joint_step_,
         edge_joint_resolution_}) {
      if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument("box dimensions, Cartesian distances and angular steps must be finite/positive");
      }
    }
    if (!box_center_.allFinite() || !std::isfinite(collision_inset_) ||
        collision_inset_ < 0.0 ||
        2.0 * collision_inset_ >= std::min({box_depth_, box_width_, box_height_})) {
      throw std::invalid_argument("box center must be finite and collision_inset must preserve positive box dimensions");
    }

    planning_seed_ = getParameter<int>("planning_seed", 0);
    if (planning_seed_ < 0) throw std::invalid_argument("planning_seed must be nonnegative");
    if (planning_seed_ > 0) ompl::RNG::setSeed(static_cast<unsigned int>(planning_seed_));
    robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(
      shared_from_this(), "robot_description");
    robot_model_ = robot_model_loader_->getModel();
    if (!robot_model_) {
      throw std::runtime_error("failed to load robot model");
    }
    if (world_frame_ != robot_model_->getModelFrame()) {
      throw std::invalid_argument("world_frame must match the robot model frame");
    }
    planning_group_ = robot_model_->getJointModelGroup(planning_group_name_);
    if (!planning_group_ || planning_group_->getVariableCount() != 7U) {
      throw std::runtime_error("planning group must be a seven-axis arm: " + planning_group_name_);
    }
    if (!robot_model_->hasLinkModel(tool_link_) || !robot_model_->hasLinkModel(arm_base_link_)) {
      throw std::runtime_error("missing tool or arm base link");
    }

    all_joint_names_.reserve(17);
    for (const std::string arm_side : {std::string("left"), std::string("right")}) {
      for (int index = 1; index <= 7; ++index) {
        all_joint_names_.push_back(arm_side + "_joint" + std::to_string(index));
      }
    }

    // Keep the original 14 arm entries in order; publish the shared axes for complete TF.
    all_joint_names_.push_back("updown");
    all_joint_names_.push_back("head_joint");
    all_joint_names_.push_back("head_pitch_joint");

    initial_state_ = std::make_shared<moveit::core::RobotState>(robot_model_);
    initial_state_->setToDefaultValues();
    for (const auto& name : all_joint_names_) {
      if (robot_model_->hasJointModel(name)) {
        initial_state_->setVariablePosition(name, 0.0);
      }
    }
    const std::string named_initial_pose = initial_pose_ == "second_home" ? "second_home" : "home";
    if (distance_demo_ && !initial_state_->setToDefaultValues(
        robot_model_->getJointModelGroup("whole_body"), named_initial_pose)) {
      throw std::runtime_error("distance demo requires SRDF whole_body/" + named_initial_pose);
    }
    initial_state_->update(true);
    home_state_ = std::make_shared<moveit::core::RobotState>(*initial_state_);
    resetInitialState();
    display_state_ = std::make_shared<moveit::core::RobotState>(*initial_state_);
    if (!trajectory_cache_file_.empty()) {
      std::ifstream cache_file(trajectory_cache_file_);
      if (!cache_file) throw std::runtime_error("cannot read trajectory_cache_file");
      trajectory_cache_ = nlohmann::json::parse(cache_file);
      if (trajectory_cache_.value("kind", "") != "v3_fixed_wall_motion_stage_cache" ||
          trajectory_cache_.value("schema_version", 0) != 1 ||
          trajectory_cache_.at("joint_names").get<std::vector<std::string>>() != all_joint_names_)
        throw std::runtime_error("trajectory cache does not match V3 stage contract");
    }
    if (distance_demo_) {
      // Keep rounded STL faces strictly behind the box plane without relaxing collisions.
      // This is a simulation numerical gap, not calibrated suction compliance.
      contact_numerical_gap_ = getParameter<double>("contact_numerical_gap", 1e-6);
      if (!std::isfinite(contact_numerical_gap_) || contact_numerical_gap_ < 0.0 ||
          contact_numerical_gap_ > 1e-4) {
        throw std::invalid_argument("contact_numerical_gap must be finite and in [0, 0.0001] metres");
      }
      align_height_ = getParameter<bool>("align_height", true);
      height_strategy_ = getParameter<std::string>("height_strategy", "fixed_offset");
      top_shoulder_above_wrist_ = getParameter<double>("top_shoulder_above_wrist", 0.10);
      if (!std::isfinite(top_shoulder_above_wrist_) || top_shoulder_above_wrist_ < 0.0)
        throw std::invalid_argument("top_shoulder_above_wrist must be finite and nonnegative metres");
      comfort_branch_ = getParameter<std::string>("comfort_branch", "auto");
      comfort_min_ = getParameter<double>("comfort_ratio_min", 0.8);
      comfort_preferred_ = getParameter<double>("comfort_ratio_preferred", 0.8);
      comfort_max_ = getParameter<double>("comfort_ratio_max", 0.8);
      if (height_strategy_ != "fixed_offset" && height_strategy_ != "comfort_radius")
        throw std::invalid_argument("height_strategy must be fixed_offset or comfort_radius");
      // Validate even when disabled; do not silently accept a broken experiment config.
      alfa_robot::motion::chooseComfortHeight(1, 0, 0, 1, 0, -1, 0,
        comfort_min_, comfort_preferred_, comfort_max_, comfort_branch_);
      shoulder_box_offset_ = getParameter<double>("shoulder_box_offset", 0.25);
      if (!std::isfinite(shoulder_box_offset_) || shoulder_box_offset_ < 0.0) {
        throw std::invalid_argument("shoulder_box_offset must be finite and nonnegative metres");
      }
      const Eigen::Vector3d shoulder_midpoint = 0.5 * (
        V3RedundantArmAnalyticIk(V3RedundantArmModel::V311Left).modelShoulderCenterInArmBase() +
        V3RedundantArmAnalyticIk(V3RedundantArmModel::V311Right).modelShoulderCenterInArmBase());
      initial_shoulder_z_ = (initial_state_->getGlobalLinkTransform(arm_base_link_) *
        shoulder_midpoint).z();
      chassis_front_x_ = getParameter<double>("chassis_front_x", modelChassisFrontX());
      chassis_rear_x_ = modelChassisFrontX(true);
      rear_clearance_ = getParameter<double>("rear_clearance", 0.02);
      if (!std::isfinite(rear_clearance_) || rear_clearance_ < 0.01)
        throw std::invalid_argument("rear_clearance must be finite and at least 0.01m");
      wall_center_y_ = getParameter<double>("wall_center_y", 0.0);
      wall_bottom_z_ = getParameter<double>("wall_bottom_z", 0.0);
      if (!std::isfinite(chassis_front_x_) || !std::isfinite(wall_center_y_) ||
          !std::isfinite(wall_bottom_z_) || collision_inset_ != 0.0) {
        throw std::invalid_argument("distance demo requires finite placement and zero collision_inset");
      }
      updateWallTarget(getParameter<double>("x", -1.0), getParameter<int>("box_id", 0));
      initial_box_center_ = box_center_;
      requested_suction_mode_ = getParameter<std::string>("suction_mode", "auto");
      if (requested_suction_mode_ != "auto" && requested_suction_mode_ != "top")
        throw std::invalid_argument("suction_mode must be auto or top");
      top_suction_ = requested_suction_mode_ == "top" || alfa_robot::motion::isBottomBox(
        box_center_.z(), box_height_, wall_bottom_z_);
      requested_arm_ = getParameter<std::string>("arm", "auto");
      if (requested_arm_ != "auto" && requested_arm_ != "left" && requested_arm_ != "right")
        throw std::invalid_argument("arm must be left/right/auto");
    }

    solver_ = std::make_unique<V3RedundantArmAnalyticIk>(
      side_ == "left" ? V3RedundantArmModel::V311Left : V3RedundantArmModel::V311Right);

    std::vector<std::string> request_adapters = {
      "default_planner_request_adapters/AddTimeOptimalParameterization",
      "default_planner_request_adapters/ResolveConstraintFrames",
      "default_planner_request_adapters/FixWorkspaceBounds",
      "default_planner_request_adapters/FixStartStateBounds",
      "default_planner_request_adapters/FixStartStateCollision",
      "default_planner_request_adapters/FixStartStatePathConstraints",
    };
    // This standalone demo returns geometric paths, not time-parameterized commands.
    // Avoid TOTG resampling/deforming the path; validate every returned edge below.
    if (distance_demo_) request_adapters.erase(request_adapters.begin());
    planning_pipeline_ = std::make_shared<planning_pipeline::PlanningPipeline>(
      robot_model_, shared_from_this(), "ompl", "ompl_interface/OMPLPlanner", request_adapters);
    planning_pipeline_->displayComputedMotionPlans(false);
    planning_pipeline_->publishReceivedRequests(false);
    planning_pipeline_->checkSolutionPaths(true);

    task_publisher_ = create_publisher<std_msgs::msg::String>(
      "~/task_json", rclcpp::QoS(1).reliable().transient_local());
    // At most 25 checked box segments per task. The separate depth-1 topic remains
    // the authoritative full snapshot for late subscribers/reconnects.
    segment_publisher_ = create_publisher<std_msgs::msg::String>(
      "~/task_json_segments", rclcpp::QoS(32).reliable().transient_local());
    wall_target_catalog_publisher_ = create_publisher<std_msgs::msg::String>(
      "~/wall_target_catalog", rclcpp::QoS(1).reliable().transient_local());
    if (execution_backend_ == "replay") {
      joint_state_publisher_ = create_publisher<sensor_msgs::msg::JointState>("~/joint_states", 10);
    }
    scene_marker_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "~/scene_markers", rclcpp::QoS(1).reliable().transient_local());
    status_marker_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "~/status_markers", rclcpp::QoS(1).reliable().transient_local());
    if (enable_stage_action_) {
      readiness_publisher_ = create_publisher<robot_system_interfaces::msg::DomainReadiness>(
        "/motion/readiness", robot_interfaces_qos::latched());
      stage_action_server_ = rclcpp_action::create_server<StageAction>(
        shared_from_this(), "/motion/execute_stage",
        std::bind(&V3SingleArmBoxExtractDemo::handleStageGoal, this,
          std::placeholders::_1, std::placeholders::_2),
        std::bind(&V3SingleArmBoxExtractDemo::handleStageCancel, this, std::placeholders::_1),
        std::bind(&V3SingleArmBoxExtractDemo::handleStageAccepted, this, std::placeholders::_1));
      if (execution_backend_ == "fjt") {
        fjt_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
          shared_from_this(), follow_joint_trajectory_action_);
        feedback_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
          "/joint_states", robot_interfaces_qos::fast_state(),
          [this](const sensor_msgs::msg::JointState::SharedPtr message) {
            if (message->name.size() != message->position.size()) return;
            std::map<std::string, double> sample;
            for (size_t index = 0; index < message->name.size(); ++index) {
              if (!std::isfinite(message->position[index]) ||
                  !sample.emplace(message->name[index], message->position[index]).second) return;
            }
            std::lock_guard<std::mutex> lock(feedback_mutex_);
            latest_feedback_ = std::move(sample);
            feedback_received_at_ = std::chrono::steady_clock::now();
          });
        safety_subscription_ = create_subscription<robot_rt_control_interfaces::msg::SafetyState>(
          "/control/safety_state", robot_interfaces_qos::state(),
          [this](const robot_rt_control_interfaces::msg::SafetyState::SharedPtr message) {
            std::lock_guard<std::mutex> lock(feedback_mutex_);
            latest_safety_ready_ = message->safe_to_start_motion;
            safety_received_at_ = std::chrono::steady_clock::now();
          });
      }
      readiness_timer_ = create_wall_timer(
        std::chrono::seconds(1), [this]() {publishReadiness();});
      publishReadiness();
    }
    if (!enable_stage_action_) run_service_ = create_service<std_srvs::srv::Trigger>(
      "~/run_current_box",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        response->success = requestPlanning();
        response->message = response->success ?
          "planning request accepted" : "planner is already running";
      });

    if (distance_demo_ && !enable_stage_action_ && !direct_attach_) {
      sequence_service_ = create_service<std_srvs::srv::Trigger>("~/plan_wall_sequence",
        [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
               std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
          if (busy()) {
            response->message = "planner or sequence playback is busy";
            return;
          }
          sequence_requested_ = true;
          response->success = requestPlanning();
          response->message = response->success ? "sequence accepted; result on ~/task_json" : "busy";
        });
      wall_service_ = create_service<WallRequest>("~/plan_wall_box",
        [this](const std::shared_ptr<WallRequest::Request> request,
               std::shared_ptr<WallRequest::Response> response) {
          if (!std::isfinite(request->x) || request->x <= 0.0 ||
              request->box_id < 0 || request->box_id >= 25 ||
              (request->arm != "left" && request->arm != "right" && request->arm != "auto")) {
            response->failure_stage = "invalid_request";
            response->failure_reason = "x must be finite/positive, box_id 0..24, arm left/right/auto";
            return;
          }
          if (busy()) {
            response->failure_stage = "busy";
            response->failure_reason = "a planning request is pending";
            return;
          }
          try {
            updateWallTarget(request->x, request->box_id);
          } catch (const std::exception& error) {
            response->failure_stage = "invalid_request";
            response->failure_reason = error.what();
            return;
          }
          requested_arm_ = request->arm;
          sequence_requested_ = false;
          requestPlanning();
          onWorkerTimer();
          response->success = last_result_.at("success").get<bool>();
          response->generation = generation_;
          response->selected_arm = response->success ? side_ : "";
          response->failure_stage = last_result_.at("failure_stage").get<std::string>();
          response->failure_reason = last_result_.at("failure_reason").get<std::string>();
          response->result_json = last_result_.dump();
        });
    }

    marker_server_ = std::make_unique<interactive_markers::InteractiveMarkerServer>(
      "v3_single_arm_box_extract_demo_marker",
      get_node_base_interface(),
      get_node_clock_interface(),
      get_node_logging_interface(),
      get_node_topics_interface(),
      get_node_services_interface());
    if (!distance_demo_) createBoxMarker();
    confirm_menu_entry_ = menu_handler_.insert(
      "确认并计算当前箱位",
      [this](const Feedback::ConstSharedPtr&) {requestPlanning();});
    reset_menu_entry_ = menu_handler_.insert(
      "恢复默认箱位",
      [this](const Feedback::ConstSharedPtr&) {resetBoxPose();});
    (void)confirm_menu_entry_;
    (void)reset_menu_entry_;
    if (!distance_demo_) menu_handler_.apply(*marker_server_, kMarkerName);
    marker_server_->applyChanges();

    worker_timer_ = create_wall_timer(
      std::chrono::milliseconds(25), [this]() {onWorkerTimer();});
    const auto display_period = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::duration<double>(1.0 / display_rate_hz_));
    display_timer_ = create_wall_timer(display_period, [this]() {publishDisplayState();});
    if (distance_demo_) selectArm(requested_arm_ == "auto" ? "left" : requested_arm_);
    publishPreview(direct_attach_ ? "第二初始姿态直接吸附双箱，Shortcut+局部RRT到命名放置位" :
      distance_demo_ ? "用 plan_wall_box 服务选择距离和箱号" :
      "拖动箱体XYZ；右键箱体并选择“确认并计算当前箱位”");
    if (distance_demo_ && !direct_attach_) publishWallTargetCatalog();
    publishSceneMarkers();
    publishStatus("READY", true);

    if (auto_run_once_) {
      auto_run_timer_ = create_wall_timer(
        std::chrono::milliseconds(500), [this]() {
          if (auto_run_timer_) {
            auto_run_timer_->cancel();
          }
          sequence_requested_ = sequence_mode_;
          requestPlanning();
        });
    }

    RCLCPP_INFO(
      get_logger(),
      "V3 single-arm box extract demo ready: side=%s box=(%.2f,%.2f,%.2f) "
      "approach=%.2fm retreat=%.2fm psi_step=%.1fdeg RRT=%.2fs",
      side_.c_str(), box_depth_, box_width_, box_height_, approach_distance_,
      retreat_distance_, radToDeg(psi_step_), rrt_planning_time_);
  }

private:
  template<typename T>
  T getParameter(const std::string& name, const T& default_value)
  {
    if (!has_parameter(name)) {
      declare_parameter<T>(name, default_value);
    }
    return get_parameter(name).get_value<T>();
  }

  static const char* publicStageName(uint8_t stage)
  {
    switch (stage) {
      case StageAction::Goal::EXECUTION_STAGE_PREGRASP: return "PREGRASP";
      case StageAction::Goal::EXECUTION_STAGE_APPROACH: return "APPROACH";
      case StageAction::Goal::EXECUTION_STAGE_PLACE: return "PLACE";
      case StageAction::Goal::EXECUTION_STAGE_HOME: return "HOME";
      case StageAction::Goal::EXECUTION_STAGE_CAMERA_VIEW: return "CAMERA_VIEW";
      case StageAction::Goal::EXECUTION_STAGE_TURN: return "TURN";
      case StageAction::Goal::EXECUTION_STAGE_NAMED_JOINT_POSE: return "NAMED_JOINT_POSE";
      default: return "UNSPECIFIED";
    }
  }

  bool publicStageAllowed(uint8_t stage) const
  {
    switch (stage) {
      case StageAction::Goal::EXECUTION_STAGE_PREGRASP:
        return public_flow_state_.load() == PublicFlowState::Idle;
      case StageAction::Goal::EXECUTION_STAGE_APPROACH:
        return public_flow_state_.load() == PublicFlowState::PregraspComplete;
      case StageAction::Goal::EXECUTION_STAGE_PLACE:
        return public_flow_state_.load() == PublicFlowState::ApproachComplete;
      case StageAction::Goal::EXECUTION_STAGE_HOME:
        return public_flow_state_.load() == PublicFlowState::PlaceComplete;
      default:
        return false;
    }
  }

  rclcpp_action::GoalResponse handleStageGoal(
    const rclcpp_action::GoalUUID&,
    std::shared_ptr<const StageAction::Goal> goal)
  {
    std::lock_guard<std::mutex> lock(public_flow_mutex_);
    if (!enable_stage_action_ || !goal || public_goal_active_ ||
        planning_active_ || planning_requested_ || sequence_running_) {
      return rclcpp_action::GoalResponse::REJECT;
    }
    if (goal->execution_stage != StageAction::Goal::EXECUTION_STAGE_PREGRASP &&
        goal->execution_stage != StageAction::Goal::EXECUTION_STAGE_APPROACH &&
        goal->execution_stage != StageAction::Goal::EXECUTION_STAGE_PLACE &&
        goal->execution_stage != StageAction::Goal::EXECUTION_STAGE_HOME)
      return rclcpp_action::GoalResponse::REJECT;
    public_goal_active_ = true;
    public_cancel_requested_ = false;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handleStageCancel(
    const std::shared_ptr<StageGoalHandle>)
  {
    public_cancel_requested_ = true;
    std::lock_guard<std::mutex> lock(fjt_goal_mutex_);
    if (fjt_goal_handle_ && fjt_client_) {
      fjt_client_->async_cancel_goal(fjt_goal_handle_);
    }
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handleStageAccepted(const std::shared_ptr<StageGoalHandle> goal_handle)
  {
    auto self = std::static_pointer_cast<V3SingleArmBoxExtractDemo>(shared_from_this());
    std::thread([self, goal_handle]() {self->executeStageGoal(goal_handle);}).detach();
  }

  robot_system_interfaces::msg::ErrorInfo makeError(
    uint32_t code, const std::string& message, const std::string& detail = "") const
  {
    robot_system_interfaces::msg::ErrorInfo error;
    error.code = code;
    error.message = message;
    error.retryable = code >= 3100U && code < 3200U;
    error.severity = code == robot_system_interfaces::msg::ErrorCode::SUCCESS ?
      robot_system_interfaces::msg::ErrorInfo::OK : robot_system_interfaces::msg::ErrorInfo::FAULT;
    error.source = get_name();
    error.detail = detail;
    return error;
  }

  static bool finitePose(const geometry_msgs::msg::Pose& pose)
  {
    const std::array<double, 7> values = {pose.position.x, pose.position.y, pose.position.z,
      pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w};
    if (!std::all_of(values.begin(), values.end(), [](double value) {return std::isfinite(value);}))
      return false;
    const double norm = std::sqrt(
      pose.orientation.x * pose.orientation.x + pose.orientation.y * pose.orientation.y +
      pose.orientation.z * pose.orientation.z + pose.orientation.w * pose.orientation.w);
    return norm > 1e-6;
  }

  static Eigen::Isometry3d poseToEigen(const geometry_msgs::msg::Pose& pose)
  {
    Eigen::Quaterniond quaternion(
      pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    quaternion.normalize();
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.linear() = quaternion.toRotationMatrix();
    transform.translation() = Eigen::Vector3d(
      pose.position.x, pose.position.y, pose.position.z);
    return transform;
  }

  static nlohmann::json poseJson(const Eigen::Isometry3d& pose)
  {
    const Eigen::Quaterniond quaternion(pose.linear());
    return {{"position", {pose.translation().x(), pose.translation().y(), pose.translation().z()}},
      {"orientation", {quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()}}};
  }

  std::optional<ResolvedWallTarget> resolveWallTarget(
    const std::string& side, uint8_t grasp_mode, const geometry_msgs::msg::Pose& pose,
    std::string* reason) const
  {
    ResolvedWallTarget output;
    output.side = side;
    if (grasp_mode == robot_motion_interfaces::msg::DualArmPoseTargets::GRASP_MODE_NO_MOVE) {
      return output;
    }
    if (grasp_mode != robot_motion_interfaces::msg::DualArmPoseTargets::GRASP_MODE_SIDE_SUCTION &&
        grasp_mode != robot_motion_interfaces::msg::DualArmPoseTargets::GRASP_MODE_TOP_SUCTION) {
      if (reason) *reason = side + " grasp_mode must be SIDE_SUCTION, TOP_SUCTION, or NO_MOVE";
      return std::nullopt;
    }
    if (!finitePose(pose)) {
      if (reason) *reason = side + " pose contains invalid position or quaternion";
      return std::nullopt;
    }
    const auto* base_link = robot_model_->getLinkModel("base_link");
    if (!base_link) {
      if (reason) *reason = "robot model has no base_link";
      return std::nullopt;
    }
    const Eigen::Isometry3d world_pose =
      initial_state_->getGlobalLinkTransform(base_link) * poseToEigen(pose);
    output.active = true;
    output.top = grasp_mode ==
      robot_motion_interfaces::msg::DualArmPoseTargets::GRASP_MODE_TOP_SUCTION;
    output.center = world_pose.translation();
    if (output.top) {
      output.center.z() -= box_height_ / 2.0 + contact_numerical_gap_;
      output.wall_distance = output.center.x() - chassis_front_x_ - box_depth_ / 2.0;
    } else {
      output.center.x() += box_depth_ / 2.0 + contact_numerical_gap_;
      output.wall_distance = world_pose.translation().x() - chassis_front_x_;
    }
    if (!std::isfinite(output.wall_distance) || output.wall_distance <= 0.0) {
      if (reason) *reason = side + " target resolves to a non-positive chassis-to-wall distance";
      return std::nullopt;
    }
    const double column_value =
      (output.center.y() - wall_center_y_) / (box_width_ + wall_gap_) + 2.0;
    const double row_value =
      (output.center.z() - wall_bottom_z_ - box_height_ / 2.0) /
      (box_height_ + wall_gap_);
    const int column = static_cast<int>(std::llround(column_value));
    const int row = static_cast<int>(std::llround(row_value));
    if (column < 0 || column >= 5 || row < 0 || row >= 5) {
      if (reason) *reason = side + " target is outside the fixed 5x5 wall fixture";
      return std::nullopt;
    }
    const int box_id = row * 5 + column;
    const Eigen::Vector3d expected = wallBoxCenter(output.wall_distance, box_id);
    if (std::abs(expected.y() - output.center.y()) > target_match_tolerance_ ||
        std::abs(expected.z() - output.center.z()) > target_match_tolerance_) {
      if (reason) {
        *reason = side + " target does not match a wall cell within " +
          std::to_string(target_match_tolerance_) + "m";
      }
      return std::nullopt;
    }
    const Eigen::Isometry3d expected_pose = contactPose(expected, side, output.top);
    const double orientation_error = Eigen::AngleAxisd(
      expected_pose.linear().transpose() * world_pose.linear()).angle();
    if (orientation_error > target_orientation_tolerance_) {
      if (reason) {
        *reason = side + " target orientation differs from the fixed-wall grasp by " +
          std::to_string(radToDeg(orientation_error)) + "deg";
      }
      return std::nullopt;
    }
    if (removed_boxes_.count(box_id)) {
      if (reason) *reason = side + " target box " + std::to_string(box_id) + " was already removed";
      return std::nullopt;
    }
    output.box_id = box_id;
    output.center = expected;
    return output;
  }

  static int publicFramePhase(const ReplayFrame& frame)
  {
    constexpr char prefix[] = "dual_";
    if (frame.stage.rfind(prefix, 0) == 0) {
      try {
        return std::stoi(frame.stage.substr(sizeof(prefix) - 1));
      } catch (const std::exception&) {
        return -1;
      }
    }
    if (frame.stage == "rrt_to_precontact") return 1;
    return transferPhase(frame.stage);
  }

  static void prependBoundary(
    std::vector<ReplayFrame>* segment, const std::vector<ReplayFrame>& previous,
    const std::string& stage)
  {
    if (!segment || previous.empty()) return;
    ReplayFrame boundary = previous.back();
    boundary.stage = stage + "_boundary";
    segment->insert(segment->begin(), std::move(boundary));
  }

  bool partitionPublicFlow(CachedPublicFlow* flow, std::string* reason) const
  {
    if (!flow) return false;
    std::optional<ReplayFrame> released;
    for (const auto& frame : flow->plan.frames) {
      const int phase = publicFramePhase(frame);
      if (phase <= 1) flow->pregrasp.push_back(frame);
      else if (phase == 2) flow->approach.push_back(frame);
      else if (phase >= 4 && phase <= 6) flow->place.push_back(frame);
      else if (phase == 7) released = frame;
      else if (phase >= 8) flow->home.push_back(frame);
    }
    prependBoundary(&flow->approach, flow->pregrasp, "APPROACH");
    prependBoundary(&flow->place, flow->approach, "PLACE");
    if (released && !flow->place.empty() && released->joints == flow->place.back().joints &&
        !released->box_attached && !released->box_visible) {
      released->stage = "HOME_release_boundary";
      if (!flow->dual && released->carried_boxes.empty()) {
        released->carried_boxes.push_back(carriedBoxJson(
          flow->box_ids.front(), flow->display_center, flow->single_side,
          flow->top, false, false));
      }
      flow->home.insert(flow->home.begin(), *released);
    }
    if (flow->pregrasp.empty() || flow->approach.size() < 2U ||
        flow->place.size() < 2U || flow->home.size() < 2U) {
      if (reason) {
        *reason = "planned flow cannot be partitioned into PREGRASP/APPROACH/PLACE/HOME";
      }
      return false;
    }
    return true;
  }

  bool loadCachedPublicFlow(CachedPublicFlow* flow, std::string* reason) const
  {
    if (!flow || trajectory_cache_.is_null()) return false;
    const nlohmann::json* selected = nullptr;
    const int expected_right = flow->dual ? flow->box_ids.at(1) : -1;
    for (const auto& entry : trajectory_cache_.at("entries")) {
      if (entry.value("success", false) &&
          entry.at("left").get<int>() == flow->box_ids.at(0) &&
          entry.value("right", -1) == expected_right) {
        selected = &entry;
        break;
      }
    }
    if (!selected) {
      if (reason) *reason = "no complete cached trajectory for this target";
      return false;
    }
    const auto convert = [&](const nlohmann::json& input, const std::string& stage,
      bool attached, bool visible) {
      std::vector<ReplayFrame> output;
      for (const auto& item : input) {
        ReplayFrame frame;
        frame.stage = stage;
        frame.joints = item.at("joints").get<std::vector<double>>();
        if (frame.joints.size() != all_joint_names_.size())
          throw std::runtime_error("cached frame does not contain 17 joints");
        frame.box_attached = attached;
        frame.box_visible = visible;
        if (attached) {
          if (flow->dual) {
            frame.carried_boxes.push_back(carriedBoxJson(
              flow->box_ids[0], wallBoxCenter(flow->wall_distance, flow->box_ids[0]),
              "left", false, true, true));
            frame.carried_boxes.push_back(carriedBoxJson(
              flow->box_ids[1], wallBoxCenter(flow->wall_distance, flow->box_ids[1]),
              "right", false, true, true));
          } else {
            frame.carried_boxes.push_back(carriedBoxJson(
              flow->box_ids.front(), wallBoxCenter(flow->wall_distance, flow->box_ids.front()),
              flow->single_side, false, true, true));
          }
        }
        output.push_back(std::move(frame));
      }
      return output;
    };
    flow->pregrasp = convert(selected->at("pregrasp"), "PREGRASP", false, true);
    flow->approach = convert(selected->at("approach"), "APPROACH", false, true);
    flow->place = convert(selected->at("place"), "PLACE", true, true);
    flow->home = convert(selected->at("home"), "HOME", false, false);
    if (flow->pregrasp.empty() || flow->approach.empty() ||
        flow->place.empty() || flow->home.empty()) {
      if (reason) *reason = "cached trajectory has an empty stage";
      return false;
    }
    const auto current = allJoints(*initial_state_);
    for (size_t index = 0; index < current.size(); ++index) {
      const double tolerance = all_joint_names_[index] == "updown" ? 0.015 : degToRad(0.75);
      if (std::abs(current[index] - flow->pregrasp.front().joints[index]) > tolerance) {
        if (reason) *reason = "cached trajectory start differs from current state at " +
          all_joint_names_[index];
        return false;
      }
    }
    flow->plan.success = true;
    flow->plan.total_ms = selected->value("planning_ms", 0.0);
    for (const auto* segment : {&flow->pregrasp, &flow->approach, &flow->place, &flow->home})
      flow->plan.frames.insert(flow->plan.frames.end(), segment->begin(), segment->end());
    return true;
  }

  bool planPublicFlow(
    const robot_motion_interfaces::msg::DualArmPoseTargets& targets,
    CachedPublicFlow* flow, std::string* reason)
  {
    if (!flow) return false;
    auto left = resolveWallTarget("left", targets.left_grasp_mode, targets.left_pose, reason);
    if (!left) return false;
    auto right = resolveWallTarget("right", targets.right_grasp_mode, targets.right_pose, reason);
    if (!right) return false;
    if (!left->active && !right->active) {
      if (reason) *reason = "both arms are NO_MOVE";
      return false;
    }
    if (left->active && right->active && left->top != right->top) {
      if (reason) *reason = "first simulation demo requires equal dual-arm grasp modes";
      return false;
    }
    if (left->active && right->active &&
        std::abs(left->wall_distance - right->wall_distance) > target_match_tolerance_) {
      if (reason) *reason = "left/right targets resolve to different wall distances";
      return false;
    }

    flow->generation = ++generation_;
    flow->dual = left->active && right->active;
    flow->top = left->active ? left->top : right->top;
    flow->wall_distance = flow->dual ?
      0.5 * (left->wall_distance + right->wall_distance) :
      (left->active ? left->wall_distance : right->wall_distance);
    flow->display_center = flow->dual ? left->center :
      (left->active ? left->center : right->center);
    attempts_ = nlohmann::json::array();
    requested_suction_mode_ = flow->top ? "top" : "auto";

    if (flow->dual) {
      flow->box_ids = {left->box_id, right->box_id};
      updateWallTarget(flow->wall_distance, left->box_id);
      selectArm("left");
      top_suction_ = flow->top;
      publishPlanningStarted(flow->generation, flow->display_center);
      if (!flow->top && !trajectory_cache_.is_null())
        return loadCachedPublicFlow(flow, reason);
      flow->plan = planDualPair(left->box_id, right->box_id, flow->top, flow->wall_distance);
    } else {
      const auto& target = left->active ? *left : *right;
      flow->box_ids = {target.box_id};
      flow->single_side = target.side;
      if (!trajectory_cache_.is_null())
        return loadCachedPublicFlow(flow, reason);
      requested_arm_ = target.side;
      selectArm(target.side);
      top_suction_ = target.top;
      updateWallTarget(target.wall_distance, target.box_id);
      publishPlanningStarted(flow->generation, target.center);
      flow->plan = planTask(target.center);
    }
    if (!flow->plan.success) {
      ensureFailurePlayback(flow->plan, flow->display_center);
      if (reason) *reason = flow->plan.failure_stage + ": " + flow->plan.failure_reason;
      return false;
    }
    return partitionPublicFlow(flow, reason);
  }

  static builtin_interfaces::msg::Duration durationFromSeconds(double seconds)
  {
    builtin_interfaces::msg::Duration duration;
    duration.sec = static_cast<int32_t>(std::floor(seconds));
    duration.nanosec = static_cast<uint32_t>(std::llround(
      (seconds - static_cast<double>(duration.sec)) * 1e9));
    if (duration.nanosec >= 1000000000U) {
      ++duration.sec;
      duration.nanosec -= 1000000000U;
    }
    return duration;
  }

  bool readFjtState(std::map<std::string, double>* positions, std::string* reason) const
  {
    std::lock_guard<std::mutex> lock(feedback_mutex_);
    if (feedback_received_at_ == std::chrono::steady_clock::time_point{} ||
        std::chrono::steady_clock::now() - feedback_received_at_ > std::chrono::milliseconds(500)) {
      if (reason) *reason = "authoritative /joint_states is missing or older than 500ms";
      return false;
    }
    for (const auto& name : robot_model_->getVariableNames()) {
      if (!latest_feedback_.count(name)) {
        if (reason) *reason = "authoritative /joint_states is missing model axis " + name;
        return false;
      }
    }
    if (positions) *positions = latest_feedback_;
    return true;
  }

  bool readSafetyState(std::string* reason) const
  {
    std::lock_guard<std::mutex> lock(feedback_mutex_);
    if (safety_received_at_ == std::chrono::steady_clock::time_point{} ||
        std::chrono::steady_clock::now() - safety_received_at_ > std::chrono::milliseconds(500)) {
      if (reason) *reason = "safety state is missing or older than 500ms";
      return false;
    }
    if (!latest_safety_ready_) {
      if (reason) *reason = "safety state does not allow motion";
      return false;
    }
    return true;
  }

  bool feedbackMatches(
    const std::map<std::string, double>& feedback, const std::vector<double>& expected,
    std::string* reason) const
  {
    if (expected.size() != all_joint_names_.size()) {
      if (reason) *reason = "trajectory frame does not contain all 17 controlled joints";
      return false;
    }
    for (size_t index = 0; index < all_joint_names_.size(); ++index) {
      const auto& name = all_joint_names_[index];
      const double tolerance = name == "updown" ? 0.015 : degToRad(0.75);
      if (std::abs(feedback.at(name) - expected[index]) > tolerance) {
        if (reason) *reason = name + " differs from planned boundary by " +
          std::to_string(std::abs(feedback.at(name) - expected[index]));
        return false;
      }
    }
    return true;
  }

  FollowJointTrajectory::Goal trajectoryGoal(const std::vector<ReplayFrame>& frames) const
  {
    FollowJointTrajectory::Goal goal;
    goal.trajectory.joint_names = all_joint_names_;
    if (frames.empty()) return goal;
    std::vector<double> times(frames.size(), 0.001);
    for (size_t index = 1; index < frames.size(); ++index) {
      double required = minimum_trajectory_step_s_;
      for (size_t joint = 0; joint < all_joint_names_.size(); ++joint) {
        const double velocity = all_joint_names_[joint] == "updown" ?
          maximum_updown_velocity_ : maximum_rotary_velocity_;
        required = std::max(required,
          std::abs(frames[index].joints.at(joint) - frames[index - 1].joints.at(joint)) /
          velocity);
      }
      times[index] = times[index - 1] + required;
    }
    goal.trajectory.points.resize(frames.size());
    for (size_t index = 0; index < frames.size(); ++index) {
      auto& point = goal.trajectory.points[index];
      point.positions = frames[index].joints;
      point.velocities.assign(all_joint_names_.size(), 0.0);
      point.accelerations.assign(all_joint_names_.size(), 0.0);
      point.time_from_start = durationFromSeconds(times[index]);
    }
    for (size_t index = 1; index + 1 < frames.size(); ++index) {
      const double dt = times[index + 1] - times[index - 1];
      for (size_t joint = 0; joint < all_joint_names_.size(); ++joint) {
        goal.trajectory.points[index].velocities[joint] =
          (frames[index + 1].joints[joint] - frames[index - 1].joints[joint]) / dt;
      }
    }
    return goal;
  }

  bool executeReplay(
    const std::vector<ReplayFrame>& frames, bool* canceled, std::string* reason)
  {
    if (frames.empty()) {
      if (reason) *reason = "stage contains no trajectory frames";
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(display_mutex_);
      playback_frames_ = frames;
      playback_index_ = 0;
      sequence_playback_ = true;
      display_diagnostic_ = nlohmann::json::object();
      display_failure_frozen_ = false;
    }
    while (rclcpp::ok()) {
      if (public_cancel_requested_) {
        if (canceled) *canceled = true;
        std::lock_guard<std::mutex> lock(display_mutex_);
        sequence_playback_ = false;
        return false;
      }
      {
        std::lock_guard<std::mutex> lock(display_mutex_);
        if (!sequence_playback_) return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (reason) *reason = "ROS shutdown during replay";
    return false;
  }

  bool executeFjt(
    const std::vector<ReplayFrame>& frames, bool* canceled, std::string* reason)
  {
    if (frames.empty()) {
      if (reason) *reason = "stage contains no trajectory frames";
      return false;
    }
    std::map<std::string, double> feedback;
    if (!readFjtState(&feedback, reason) ||
        !feedbackMatches(feedback, frames.front().joints, reason)) return false;
    if (public_cancel_requested_) {
      if (canceled) *canceled = true;
      return false;
    }
    if (!fjt_client_ || !fjt_client_->wait_for_action_server(std::chrono::seconds(3))) {
      if (reason) *reason = "FollowJointTrajectory server unavailable: " + follow_joint_trajectory_action_;
      return false;
    }
    auto goal_future = fjt_client_->async_send_goal(trajectoryGoal(frames));
    if (goal_future.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
      if (reason) *reason = "FollowJointTrajectory acceptance timed out; goal state is unknown";
      fjt_outcome_unknown_ = true;
      return false;
    }
    auto goal_handle = goal_future.get();
    if (!goal_handle) {
      if (reason) *reason = "FollowJointTrajectory goal rejected";
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(fjt_goal_mutex_);
      fjt_goal_handle_ = goal_handle;
    }
    auto result_future = fjt_client_->async_get_result(goal_handle);
    const auto planned_time = trajectoryGoal(frames).trajectory.points.back().time_from_start;
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::seconds(20 + planned_time.sec);
    bool cancel_sent = false;
    while (result_future.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready) {
      if (public_cancel_requested_ && !cancel_sent) {
        fjt_client_->async_cancel_goal(goal_handle);
        cancel_sent = true;
      }
      if (!rclcpp::ok() || std::chrono::steady_clock::now() > deadline) {
        if (!cancel_sent) fjt_client_->async_cancel_goal(goal_handle);
        if (reason) *reason = "FollowJointTrajectory result not confirmed; controller state is unknown";
        fjt_outcome_unknown_ = true;
        return false;
      }
    }
    const auto wrapped = result_future.get();
    {
      std::lock_guard<std::mutex> lock(fjt_goal_mutex_);
      fjt_goal_handle_.reset();
    }
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED || !wrapped.result ||
        wrapped.result->error_code != FollowJointTrajectory::Result::SUCCESSFUL) {
      fjt_outcome_unknown_ = true;
      if (cancel_sent && wrapped.code == rclcpp_action::ResultCode::CANCELED) {
        if (canceled) *canceled = true;
        return false;
      }
      if (reason) {
        *reason = "FollowJointTrajectory failed code=" +
          std::to_string(wrapped.result ? wrapped.result->error_code : -1);
      }
      return false;
    }
    if (cancel_sent) {
      if (reason) *reason = "controller reported success after cancel; inspect actual state";
      fjt_outcome_unknown_ = true;
      return false;
    }
    const auto settle_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    do {
      if (readFjtState(&feedback, reason) &&
          feedbackMatches(feedback, frames.back().joints, nullptr)) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(25));
    } while (rclcpp::ok() && std::chrono::steady_clock::now() < settle_deadline);
    if (reason) *reason = "FJT reported success but authoritative joint state did not reach the final point";
    fjt_outcome_unknown_ = true;
    return false;
  }

  bool executePublicFrames(
    const std::vector<ReplayFrame>& frames, bool* canceled, std::string* reason)
  {
    return execution_backend_ == "fjt" ?
      executeFjt(frames, canceled, reason) : executeReplay(frames, canceled, reason);
  }

  void publishPublicStageSnapshot(
    const CachedPublicFlow& flow, uint8_t stage, const std::vector<ReplayFrame>& frames,
    bool success, const std::string& failure_reason = "")
  {
    TaskResult stage_result;
    stage_result.success = success;
    stage_result.failure_stage = success ? "" : publicStageName(stage);
    stage_result.failure_reason = failure_reason;
    stage_result.total_ms = stage == StageAction::Goal::EXECUTION_STAGE_PREGRASP ?
      flow.plan.total_ms : 0.0;
    stage_result.metrics = flow.plan.metrics;
    if (success) {
      stage_result.frames = frames;
    } else {
      stage_result.diagnostic_frames = frames;
      stage_result.diagnostic = {
        {"diagnostic_only", true}, {"freeze_at_end", true},
        {"stage", publicStageName(stage)}, {"reason", failure_reason}};
    }
    if (!flow.box_ids.empty()) {
      updateWallTarget(flow.wall_distance, flow.box_ids.front());
      selectArm(flow.dual ? "left" : flow.single_side);
      top_suction_ = flow.top;
    }
    publishTaskResult(flow.generation, flow.display_center, stage_result, false);
    if (flow.dual) {
      const Eigen::Vector3d right_center = wallBoxCenter(flow.wall_distance, flow.box_ids[1]);
      auto& neighbors = last_result_["neighbor_centers"];
      neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(),
        [&right_center](const nlohmann::json& center) {
          return std::abs(center[0].get<double>() - right_center.x()) < 1e-8 &&
            std::abs(center[1].get<double>() - right_center.y()) < 1e-8 &&
            std::abs(center[2].get<double>() - right_center.z()) < 1e-8;
        }), neighbors.end());
    }
    last_result_["public_action"] = "/motion/execute_stage";
    last_result_["public_stage"] = publicStageName(stage);
    last_result_["box_ids"] = flow.box_ids;
    last_result_["dual"] = flow.dual;
    last_result_["execution_backend"] = execution_backend_;
    publishJson(last_result_);
  }

  void finishPublicGoal()
  {
    std::lock_guard<std::mutex> lock(public_flow_mutex_);
    public_goal_active_ = false;
    public_cancel_requested_ = false;
  }

  void executeStageGoal(const std::shared_ptr<StageGoalHandle>& goal_handle)
  {
    const auto goal = goal_handle->get_goal();
    auto result = std::make_shared<StageAction::Result>();
    auto publish_feedback = [&goal_handle](uint8_t state) {
      auto feedback = std::make_shared<StageAction::Feedback>();
      feedback->motion_state = state;
      goal_handle->publish_feedback(feedback);
    };
    auto abort = [&](uint32_t code, const std::string& message, const std::string& detail) {
      result->ok = false;
      result->error = makeError(code, message, detail);
      result->diagnostic = detail;
      goal_handle->abort(result);
      std::lock_guard<std::mutex> lock(public_flow_mutex_);
      if (goal->execution_stage == StageAction::Goal::EXECUTION_STAGE_PREGRASP)
        public_flow_state_ = PublicFlowState::Idle;
      public_goal_active_ = false;
      public_cancel_requested_ = false;
    };

    try {
      if (session_requires_reset_) {
        abort(robot_system_interfaces::msg::ErrorCode::MOTION_STATE_UNAVAILABLE,
          "motion stage session requires reset", "a previous execution did not complete");
        return;
      }
      if (!publicStageAllowed(goal->execution_stage)) {
        abort(robot_system_interfaces::msg::ErrorCode::MOTION_STAGE_SEQUENCE_INVALID,
          "motion stage sequence is invalid", publicStageName(goal->execution_stage));
        return;
      }
      if (goal->execution_stage == StageAction::Goal::EXECUTION_STAGE_PREGRASP) {
        publish_feedback(StageAction::Feedback::MOTION_STATE_PLANNING);
        if (execution_backend_ == "fjt") {
          std::map<std::string, double> feedback;
          std::string state_reason;
          if (fjt_outcome_unknown_ || !readFjtState(&feedback, &state_reason) ||
              !readSafetyState(&state_reason)) {
            abort(robot_system_interfaces::msg::ErrorCode::MOTION_STATE_UNAVAILABLE,
              "authoritative robot state unavailable", fjt_outcome_unknown_ ?
                "previous FJT goal ended without a confirmed controller state" : state_reason);
            return;
          }
          for (const auto& name : robot_model_->getVariableNames())
            initial_state_->setVariablePosition(name, feedback.at(name));
          initial_state_->update(true);
        }
        CachedPublicFlow planned;
        std::string planning_reason;
        if (!planPublicFlow(goal->targets, &planned, &planning_reason)) {
          if (public_cancel_requested_) {
            result->ok = false;
            result->error = makeError(robot_system_interfaces::msg::ErrorCode::CANCELED,
              "PREGRASP canceled during planning");
            goal_handle->canceled(result);
            finishPublicGoal();
            return;
          }
          if (!planned.plan.frames.empty() || !planned.plan.diagnostic_frames.empty()) {
            publishPublicStageSnapshot(
              planned, goal->execution_stage, planned.plan.diagnostic_frames, false, planning_reason);
          }
          abort(planned.generation == 0 ?
            robot_system_interfaces::msg::ErrorCode::INVALID_GOAL :
            robot_system_interfaces::msg::ErrorCode::MOTION_PLANNING_FAILED,
            "PREGRASP planning failed", planning_reason);
          return;
        }
        if (public_cancel_requested_) {
          result->ok = false;
          result->error = makeError(robot_system_interfaces::msg::ErrorCode::CANCELED,
            "PREGRASP canceled after planning");
          goal_handle->canceled(result);
          finishPublicGoal();
          return;
        }
        {
          std::lock_guard<std::mutex> lock(public_flow_mutex_);
          cached_public_flow_ = std::move(planned);
        }
      }

      std::vector<ReplayFrame> frames;
      CachedPublicFlow snapshot;
      bool missing_flow = false;
      {
        std::lock_guard<std::mutex> lock(public_flow_mutex_);
        if (!cached_public_flow_) {
          missing_flow = true;
        } else {
          snapshot = *cached_public_flow_;
          switch (goal->execution_stage) {
            case StageAction::Goal::EXECUTION_STAGE_PREGRASP: frames = snapshot.pregrasp; break;
            case StageAction::Goal::EXECUTION_STAGE_APPROACH: frames = snapshot.approach; break;
            case StageAction::Goal::EXECUTION_STAGE_PLACE: frames = snapshot.place; break;
            case StageAction::Goal::EXECUTION_STAGE_HOME: frames = snapshot.home; break;
            default: break;
          }
        }
      }
      if (missing_flow) {
        abort(robot_system_interfaces::msg::ErrorCode::MOTION_STAGE_SEQUENCE_INVALID,
          "no active planned flow", publicStageName(goal->execution_stage));
        return;
      }

      publish_feedback(StageAction::Feedback::MOTION_STATE_EXECUTING);
      bool canceled = false;
      std::string execution_reason;
      if (!executePublicFrames(frames, &canceled, &execution_reason)) {
        result->ok = false;
        result->error = makeError(
          canceled ? robot_system_interfaces::msg::ErrorCode::CANCELED :
          robot_system_interfaces::msg::ErrorCode::MOTION_EXECUTION_FAILED,
          canceled ? "motion stage canceled" : "motion stage execution failed",
          execution_reason);
        result->diagnostic = execution_reason;
        if (canceled) goal_handle->canceled(result);
        else goal_handle->abort(result);
        std::lock_guard<std::mutex> lock(public_flow_mutex_);
        session_requires_reset_ = true;
        cached_public_flow_.reset();
        public_flow_state_ = PublicFlowState::Idle;
        public_goal_active_ = false;
        public_cancel_requested_ = false;
        return;
      }

      publish_feedback(StageAction::Feedback::MOTION_STATE_SETTLING);
      publishPublicStageSnapshot(snapshot, goal->execution_stage, frames, true);
      {
        std::lock_guard<std::mutex> lock(public_flow_mutex_);
        switch (goal->execution_stage) {
          case StageAction::Goal::EXECUTION_STAGE_PREGRASP:
            public_flow_state_ = PublicFlowState::PregraspComplete;
            break;
          case StageAction::Goal::EXECUTION_STAGE_APPROACH:
            public_flow_state_ = PublicFlowState::ApproachComplete;
            break;
          case StageAction::Goal::EXECUTION_STAGE_PLACE:
            public_flow_state_ = PublicFlowState::PlaceComplete;
            break;
          case StageAction::Goal::EXECUTION_STAGE_HOME:
            if (!frames.empty()) {
              moveit::core::RobotState release_state(*initial_state_);
              for (size_t index = 0; index < all_joint_names_.size(); ++index)
                release_state.setVariablePosition(
                  all_joint_names_[index], snapshot.place.back().joints.at(index));
              release_state.update(true);
              if (trajectory_cache_.is_null() && snapshot.dual) {
                placed_boxes_[snapshot.box_ids[0]] =
                  release_state.getGlobalLinkTransform("left_tool0") * toolToBox("left", snapshot.top);
                placed_boxes_[snapshot.box_ids[1]] =
                  release_state.getGlobalLinkTransform("right_tool0") * toolToBox("right", snapshot.top);
              } else if (trajectory_cache_.is_null()) {
                placed_boxes_[snapshot.box_ids.front()] =
                  release_state.getGlobalLinkTransform(snapshot.single_side + "_tool0") *
                  toolToBox(snapshot.single_side, snapshot.top);
              }
              if (execution_backend_ == "fjt") {
                std::map<std::string, double> feedback;
                std::string state_reason;
                if (!readFjtState(&feedback, &state_reason))
                  throw std::runtime_error(state_reason);
                for (const auto& name : robot_model_->getVariableNames())
                  initial_state_->setVariablePosition(name, feedback.at(name));
              } else {
                for (size_t index = 0; index < all_joint_names_.size(); ++index)
                  initial_state_->setVariablePosition(all_joint_names_[index], frames.back().joints.at(index));
              }
              initial_state_->clearAttachedBodies();
              initial_state_->update(true);
              removed_boxes_.insert(snapshot.box_ids.begin(), snapshot.box_ids.end());
            }
            cached_public_flow_.reset();
            public_flow_state_ = PublicFlowState::Idle;
            publishWallTargetCatalog();
            break;
          default:
            break;
        }
      }
      result->ok = true;
      result->error = makeError(robot_system_interfaces::msg::ErrorCode::SUCCESS, "success");
      result->diagnostic = nlohmann::json({
        {"stage", publicStageName(goal->execution_stage)},
        {"box_ids", snapshot.box_ids},
        {"frames", frames.size()},
        {"backend", execution_backend_}}).dump();
      goal_handle->succeed(result);
      finishPublicGoal();
    } catch (const std::exception& error) {
      abort(robot_system_interfaces::msg::ErrorCode::MOTION_INTERNAL_ERROR,
        "motion stage internal error", error.what());
    }
  }

  void publishReadiness()
  {
    if (!readiness_publisher_) return;
    robot_system_interfaces::msg::DomainReadiness message;
    message.header.stamp = now();
    message.domain = "motion";
    message.readiness_name = "execute_stage";
    std::string blocker;
    if (fjt_outcome_unknown_) blocker = "FJT_OUTCOME_UNKNOWN";
    else if (session_requires_reset_) blocker = "INCOMPLETE_STAGE_REQUIRES_RESET";
    else if (execution_backend_ == "fjt" && !readFjtState(nullptr, nullptr))
      blocker = "FULL_JOINT_STATE_UNAVAILABLE";
    else if (execution_backend_ == "fjt" && !readSafetyState(nullptr))
      blocker = "SAFETY_STATE_UNAVAILABLE";
    else if (execution_backend_ == "fjt" &&
        (!fjt_client_ || !fjt_client_->action_server_is_ready()))
      blocker = "FJT_SERVER_UNAVAILABLE";
    message.ready = blocker.empty();
    message.status = message.ready ?
      robot_system_interfaces::msg::DomainReadiness::STATUS_HEALTHY :
      robot_system_interfaces::msg::DomainReadiness::STATUS_UNAVAILABLE;
    if (!message.ready) message.blockers.push_back(blocker);
    message.operational_state = public_goal_active_ ? "BUSY" : "IDLE";
    message.producer_instance_id = publisher_id_;
    readiness_publisher_->publish(message);
  }

  void publishWallTargetCatalog()
  {
    if (!wall_target_catalog_publisher_ || !distance_demo_ ||
        !robot_model_->hasLinkModel("base_link")) return;
    const Eigen::Isometry3d base_from_world =
      initial_state_->getGlobalLinkTransform("base_link").inverse();
    nlohmann::json catalog = {{"frame_id", "base_link"}, {"rows", 5}, {"columns", 5},
      {"wall_distance", wall_distance_}, {"removed_box_ids", removed_boxes_},
      {"boxes", nlohmann::json::array()}};
    for (int box_id = 0; box_id < 25; ++box_id) {
      const Eigen::Vector3d center = wallBoxCenter(wall_distance_, box_id);
      catalog["boxes"].push_back({
        {"box_id", box_id},
        {"left_side", poseJson(base_from_world * contactPose(center, "left", false))},
        {"right_side", poseJson(base_from_world * contactPose(center, "right", false))},
        {"left_top", poseJson(base_from_world * contactPose(center, "left", true))},
        {"right_top", poseJson(base_from_world * contactPose(center, "right", true))},
      });
    }
    std_msgs::msg::String message;
    message.data = catalog.dump();
    wall_target_catalog_publisher_->publish(message);
  }

  void loadEnvironment()
  {
    environment_objects_.clear();
    environment_json_ = {{"enabled", getParameter<bool>("check_environment", true)},
      {"frame_id", world_frame_}, {"boxes", nlohmann::json::array()}};
    if (!environment_json_.at("enabled").get<bool>()) {
      RCLCPP_WARN(get_logger(), "ENVIRONMENT COLLISIONS DISABLED: regression-only wall scene");
      return;
    }
    try {
      const auto config = nlohmann::json::parse(getParameter<std::string>("environment_json", ""));
      if (config.at("frame_id").get<std::string>() != world_frame_)
        throw std::invalid_argument("frame_id must match planning world frame");
      const auto anchor = config.value("anchor", std::string("world"));
      if (anchor != "world" && anchor != "box_wall_back")
        throw std::invalid_argument("anchor must be world or box_wall_back");
      const Eigen::Vector3d offset = anchor == "box_wall_back" ?
        // Same micrometre tolerance as tool contact: avoid false penetration on attachment.
        Eigen::Vector3d(chassis_front_x_ + wall_distance_ + box_depth_ + contact_numerical_gap_,
                        wall_center_y_, 0.0) :
        Eigen::Vector3d::Zero();
      environment_json_["anchor"] = anchor;
      const auto& boxes = config.at("boxes");
      if (!boxes.is_array() || boxes.empty() || boxes.size() > 128)
        throw std::invalid_argument("boxes must contain 1..128 axis-aligned boxes, including ground");
      std::set<std::string> ids;
      for (const auto& box : boxes) {
        const auto id = box.at("id").get<std::string>();
        if (id.empty() || id.find_first_not_of(
              "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != std::string::npos ||
            !ids.insert(id).second)
          throw std::invalid_argument("box ids must be unique nonempty letters/digits/_/-");
        // Reject unsupported rotation rather than silently drawing/checking a different obstacle.
        for (auto it = box.begin(); it != box.end(); ++it)
          if (it.key() != "id" && it.key() != "center" && it.key() != "size")
            throw std::invalid_argument("box supports only id, center, size (world-axis aligned)");
        for (const auto* key : {"center", "size"}) {
          const auto& values = box.at(key);
          if (!values.is_array() || values.size() != 3)
            throw std::invalid_argument("center/size must be 3 numbers in metres");
          for (const auto& value : values) {
            if (!value.is_number() || !std::isfinite(value.get<double>()) ||
                std::abs(value.get<double>()) > 10000.0 ||
                (std::string(key) == "size" && value.get<double>() <= 0.0))
              throw std::invalid_argument("finite center/positive size required, magnitude <=10000m");
          }
        }
        moveit_msgs::msg::CollisionObject object;
        object.header.frame_id = world_frame_;
        object.id = "environment_" + id;
        object.operation = moveit_msgs::msg::CollisionObject::ADD;
        shape_msgs::msg::SolidPrimitive primitive;
        primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
        primitive.dimensions = {box.at("size")[0].get<double>(),
          box.at("size")[1].get<double>(), box.at("size")[2].get<double>()};
        geometry_msgs::msg::Pose pose;
        pose.orientation.w = 1.0;
        pose.position.x = box.at("center")[0].get<double>() + offset.x();
        pose.position.y = box.at("center")[1].get<double>() + offset.y();
        pose.position.z = box.at("center")[2].get<double>();
        object.primitives.push_back(primitive);
        object.primitive_poses.push_back(pose);
        environment_objects_.push_back(object);
        environment_json_["boxes"].push_back({{"id", object.id},
          {"center", {pose.position.x, pose.position.y, pose.position.z}}, {"size", box.at("size")}});
      }
      if (!ids.count("ground")) throw std::invalid_argument("a box named ground is required");
      environment_json_["description"] = config.at("description").get<std::string>();
    } catch (const std::exception& error) {
      throw std::invalid_argument(std::string("invalid environment configuration: ") + error.what());
    }
    RCLCPP_WARN(get_logger(), "Environment collision checking ON: %zu objects; %s. "
      "Configured geometry only, not sensed/calibrated surroundings.", environment_objects_.size(),
      environment_json_.at("description").get<std::string>().c_str());
  }

  double modelChassisFrontX(bool rear = false) const
  {
    constexpr std::array<const char*, 11> chassis_links = {
      "model_base", "chassis_base", "active_suspension_carriage",
      "caster01", "caster02", "caster03", "caster04",
      "wheel01", "wheel02", "wheel03", "wheel04"};
    double front = rear ? std::numeric_limits<double>::infinity() :
      -std::numeric_limits<double>::infinity();
    for (const char* link_name : chassis_links) {
      const auto* link = robot_model_->getLinkModel(link_name);
      if (!link) throw std::runtime_error(std::string(link_name) + " missing from V3 chassis model");
      const auto& shapes = link->getShapes();
      const auto& origins = link->getCollisionOriginTransforms();
      for (size_t i = 0; i < shapes.size(); ++i) {
        if (shapes[i]->type != shapes::MESH) {
          throw std::runtime_error(std::string(link_name) + " chassis collision must be a mesh");
        }
        const auto* mesh = static_cast<const shapes::Mesh*>(shapes[i].get());
        const Eigen::Isometry3d transform = initial_state_->getGlobalLinkTransform(link) * origins[i];
        for (unsigned int v = 0; v < mesh->vertex_count; ++v) {
          const Eigen::Vector3d point(
            mesh->vertices[3*v], mesh->vertices[3*v+1], mesh->vertices[3*v+2]);
          front = rear ? std::min(front, (transform * point).x()) :
            std::max(front, (transform * point).x());
        }
      }
    }
    if (!std::isfinite(front)) throw std::runtime_error("no V3 chassis collision vertices");
    return front;
  }

  Eigen::Vector3d wallBoxCenter(double x, int box_id) const
  {
    return {chassis_front_x_ + x + box_depth_ / 2.0,
      wall_center_y_ + (box_id % 5 - 2) * (box_width_ + wall_gap_),
      wall_bottom_z_ + box_height_ / 2.0 + (box_id / 5) * (box_height_ + wall_gap_)};
  }

  void updateWallTarget(double x, int box_id)
  {
    if (!std::isfinite(x) || x <= 0.0 || box_id < 0 || box_id >= 25) {
      throw std::invalid_argument("x must be finite/positive and box_id must be 0..24");
    }
    const Eigen::Vector3d center = wallBoxCenter(x, box_id);
    if (!center.allFinite()) throw std::invalid_argument("wall coordinates overflow");
    wall_distance_ = x;
    wall_target_row_ = box_id / 5;
    wall_target_column_ = box_id % 5;
    box_center_ = center;
    loadEnvironment();
    if (wall_target_catalog_publisher_) publishWallTargetCatalog();
  }

  void selectArm(const std::string& side)
  {
    height_clearance_ = {{"checked", false}};
    side_ = side;
    tool_link_ = side + "_tool0";
    planning_group_name_ = side + "_arm";
    planning_group_ = robot_model_->getJointModelGroup(planning_group_name_);
    solver_ = std::make_unique<V3RedundantArmAnalyticIk>(
      side == "left" ? V3RedundantArmModel::V311Left : V3RedundantArmModel::V311Right);
  }

  void createBoxMarker()
  {
    const Eigen::Vector3d handle_position = controlHandlePosition(box_center_);
    InteractiveMarker marker;
    marker.header.frame_id = world_frame_;
    marker.name = kMarkerName;
    marker.description = "箱堆外侧XYZ控制球：拖动后右键确认计算";
    marker.scale = 0.65;
    marker.pose.position.x = handle_position.x();
    marker.pose.position.y = handle_position.y();
    marker.pose.position.z = handle_position.z();
    marker.pose.orientation.w = 1.0;

    InteractiveMarkerControl body;
    body.always_visible = true;
    body.interaction_mode = InteractiveMarkerControl::MOVE_3D;
    Marker handle;
    handle.type = Marker::SPHERE;
    handle.scale.x = handle.scale.y = handle.scale.z = 0.09;
    handle.color = color(0.10F, 0.85F, 1.0F, 0.95F);
    body.markers.push_back(handle);
    marker.controls.push_back(body);
    marker.controls.push_back(axisControl("move_x", 1.0, 0.0, 0.0));
    marker.controls.push_back(axisControl("move_y", 0.0, 1.0, 0.0));
    marker.controls.push_back(axisControl("move_z", 0.0, 0.0, 1.0));

    marker_server_->insert(
      marker,
      [this](const Feedback::ConstSharedPtr& feedback) {handleMarkerFeedback(feedback);});
  }

  Eigen::Vector3d controlHandlePosition(const Eigen::Vector3d& center) const
  {
    const double lateral_sign = side_ == "left" ? -1.0 : 1.0;
    return center + Eigen::Vector3d(
      -(box_depth_ * 0.5 + control_handle_clearance_),
      lateral_sign * control_handle_lateral_offset_,
      0.0);
  }

  void handleMarkerFeedback(const Feedback::ConstSharedPtr& feedback)
  {
    if (feedback->event_type != Feedback::POSE_UPDATE &&
        feedback->event_type != Feedback::MOUSE_UP) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(box_mutex_);
      box_center_ = Eigen::Vector3d(
        feedback->pose.position.x + box_depth_ * 0.5 + control_handle_clearance_,
        feedback->pose.position.y +
          (side_ == "left" ? control_handle_lateral_offset_ : -control_handle_lateral_offset_),
        feedback->pose.position.z);
    }
    publishPreview("箱位已更新，右键确认后才开始计算");
    publishSceneMarkers();
  }

  void resetBoxPose()
  {
    {
      std::lock_guard<std::mutex> lock(box_mutex_);
      box_center_ = initial_box_center_;
    }
    const Eigen::Vector3d handle_position = controlHandlePosition(box_center_);
    geometry_msgs::msg::Pose pose;
    pose.position.x = handle_position.x();
    pose.position.y = handle_position.y();
    pose.position.z = handle_position.z();
    pose.orientation.w = 1.0;
    marker_server_->setPose(kMarkerName, pose);
    marker_server_->applyChanges();
    publishPreview("已恢复默认箱位");
    publishSceneMarkers();
  }

  bool busy() const
  {
    return planning_active_.load() || planning_requested_.load() || sequence_playback_ ||
      public_goal_active_.load() || public_flow_state_.load() != PublicFlowState::Idle;
  }

  TaskResult planWithFallback(const Eigen::Vector3d& center, bool allow_opposite_arm = false)
  {
    attempts_ = nlohmann::json::array();
    if (!distance_demo_) return planTask(center);
    TaskResult result, deepest_failure;
    std::string failure_side;
    bool failure_top = false;
    nlohmann::json failure_height;
    PlanningMetrics metrics;
    double elapsed = 0.0;
    for (const auto& [top, side] : alfa_robot::motion::wallGraspAttempts(
        alfa_robot::motion::isBottomBox(center.z(), box_height_, wall_bottom_z_), requested_arm_,
        requested_suction_mode_ == "top", allow_opposite_arm)) {
      top_suction_ = top;
      selectArm(side);
      result = planTask(center);
      metrics.add(result.metrics);
      elapsed += result.total_ms;
      attempts_.push_back({{"arm", side}, {"suction_mode", top ? "top" : "front"},
        {"success", result.success}, {"failure_stage", result.failure_stage},
        {"failure_reason", result.failure_reason}, {"total_ms", result.total_ms},
        {"height_alignment", heightAlignment(center)}, {"height_selections", 1}});
      RCLCPP_INFO(get_logger(), "box=%d arm=%s suction=%s %s stage=%s reason=%s",
        wall_target_row_ * 5 + wall_target_column_, side.c_str(), top ? "top" : "front",
        result.success ? "SUCCESS" : "FAILED", result.failure_stage.c_str(), result.failure_reason.c_str());
      if (result.success) break;
      // A later unreachable fallback must not erase a real connected collision prefix.
      if (result.diagnostic_frames.size() >= deepest_failure.diagnostic_frames.size()) {
        deepest_failure = result;
        failure_side = side;
        failure_top = top;
        failure_height = height_clearance_;
      }
    }
    if (!result.success && !failure_side.empty()) {
      result = std::move(deepest_failure);
      top_suction_ = failure_top;
      selectArm(failure_side);
      height_clearance_ = std::move(failure_height);
    }
    result.metrics = metrics;
    result.total_ms = elapsed;
    // Planning is transactional: never replay a failed front prefix before trying top.
    if (!result.success) result.frames.clear();
    return result;
  }

  static int transferPhase(const std::string& stage)
  {
    if (stage == "attach_box") return 3;
    if (stage == "cartesian_approach") return 2;
    if (stage == "cartesian_retreat" || stage == "cartesian_lift") return 4;
    if (stage == "rrt_return") return 5;
    if (stage == "updown_return" || stage == "loaded_home") return 6;
    if (stage == "rear_placement") return 6;
    if (stage == "release_box") return 7;
    if (stage == "home_return" || stage == "home_updown") return 8;
    if (stage == "rrt_approach") return 1;
    return 0;
  }

  nlohmann::json carriedBoxJson(
    int box_id, const Eigen::Vector3d& center, const std::string& side, bool top,
    bool attached, bool visible) const
  {
    const auto transform = toolToBox(side, top);
    nlohmann::json rotation = nlohmann::json::array();
    for (int row = 0; row < 3; ++row)
      rotation.push_back({transform.linear()(row, 0), transform.linear()(row, 1),
        transform.linear()(row, 2)});
    return {{"box_id", box_id}, {"side", side}, {"tool_link", side + "_tool0"},
      {"box_center", {center.x(), center.y(), center.z()}}, {"attached", attached},
      {"visible", visible},
      {"tool_to_box_center", {transform.translation().x(), transform.translation().y(),
        transform.translation().z()}}, {"tool_to_box_rotation", rotation}};
  }

  double foldedElbowPosition(const std::string& side) const
  {
    const std::string joint = side + "_joint4";
    const auto& bounds = robot_model_->getVariableBounds(joint);
    return home_state_->getVariablePosition(joint) >= 0.0 ?
      bounds.max_position_ - degToRad(5) : bounds.min_position_ + degToRad(5);
  }

  std::array<double, 7> sideJoints(
    const moveit::core::RobotState& state, const std::string& side) const
  {
    std::vector<double> values;
    state.copyJointGroupPositions(robot_model_->getJointModelGroup(side + "_arm"), values);
    std::array<double, 7> output{};
    if (values.size() != output.size()) throw std::runtime_error("unexpected dual arm joint count");
    std::copy(values.begin(), values.end(), output.begin());
    return output;
  }

  void addPlacedBox(
    const planning_scene::PlanningScenePtr& scene, int box_id,
    const Eigen::Isometry3d& pose, const std::string& touching_side = "") const
  {
    moveit_msgs::msg::CollisionObject object;
    object.header.frame_id = world_frame_;
    object.id = "placed_wall_box_" + std::to_string(box_id);
    object.operation = moveit_msgs::msg::CollisionObject::ADD;
    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive.dimensions = {box_depth_, box_width_, box_height_};
    object.primitives.push_back(primitive);
    object.primitive_poses.push_back(eigenToPose(pose));
    if (!scene->processCollisionObjectMsg(object))
      throw std::runtime_error("failed to add " + object.id + " to planning scene");
    if (!touching_side.empty()) {
      scene->getAllowedCollisionMatrixNonConst().setEntry(
        object.id, touching_side + "_tool0", true);
      scene->getAllowedCollisionMatrixNonConst().setEntry(
        object.id, touching_side + "_link7", true);
    }
  }

  planning_scene::PlanningScenePtr makeDualScene(int left_id, int right_id) const
  {
    auto scene = std::make_shared<planning_scene::PlanningScene>(robot_model_);
    scene->setCurrentState(*initial_state_);
    for (const auto& object : environment_objects_)
      if (!scene->processCollisionObjectMsg(object))
        throw std::runtime_error("failed to add " + object.id + " to dual planning scene");
    for (int id = 0; id < 25; ++id) {
      if (removed_boxes_.count(id)) continue;
      moveit_msgs::msg::CollisionObject object;
      object.header.frame_id = world_frame_;
      object.id = id == left_id ? kCarriedBoxLeftId :
        (id == right_id ? kCarriedBoxRightId : "wall_box_" + std::to_string(id));
      object.operation = moveit_msgs::msg::CollisionObject::ADD;
      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
      primitive.dimensions = {box_depth_, box_width_, box_height_};
      const auto center = wallBoxCenter(wall_distance_, id);
      geometry_msgs::msg::Pose pose;
      pose.orientation.w = 1.0;
      pose.position.x = center.x(); pose.position.y = center.y(); pose.position.z = center.z();
      object.primitives.push_back(primitive);
      object.primitive_poses.push_back(pose);
      if (!scene->processCollisionObjectMsg(object))
        throw std::runtime_error("failed to add " + object.id + " to dual planning scene");
    }
    for (const auto& placed : placed_boxes_)
      addPlacedBox(scene, placed.first, placed.second);
    return scene;
  }

  bool validateDualFrames(
    std::vector<ReplayFrame>& frames, int left_id, int right_id, bool top,
    PlanningMetrics* metrics, std::string* reason) const
  {
    const auto scene = makeDualScene(left_id, right_id);
    // Each independent approach already validates its own final suction contact.
    // Preserve only those pairs while recombining; cross-arm and environment
    // collisions remain checked.
    scene->getAllowedCollisionMatrixNonConst().setEntry(kCarriedBoxLeftId, "left_tool0", true);
    scene->getAllowedCollisionMatrixNonConst().setEntry(kCarriedBoxLeftId, "left_link7", true);
    scene->getAllowedCollisionMatrixNonConst().setEntry(kCarriedBoxRightId, "right_tool0", true);
    scene->getAllowedCollisionMatrixNonConst().setEntry(kCarriedBoxRightId, "right_link7", true);
    auto loaded = planning_scene::PlanningScene::clone(scene);
    loaded->getWorldNonConst()->removeObject(kCarriedBoxLeftId);
    loaded->getWorldNonConst()->removeObject(kCarriedBoxRightId);
    auto released = planning_scene::PlanningScene::clone(loaded);
    bool placed_current_boxes = false;
    moveit::core::RobotStatePtr previous;
    bool previous_attached = false;
    bool previous_visible = true;
    for (const auto& frame : frames) {
      moveit::core::RobotState state(*initial_state_);
      for (size_t i = 0; i < all_joint_names_.size(); ++i)
        state.setVariablePosition(all_joint_names_[i], frame.joints.at(i));
      const bool attached = std::any_of(frame.carried_boxes.begin(), frame.carried_boxes.end(),
        [](const nlohmann::json& box) {return box.value("attached", false);});
      if (enable_stage_action_ && !attached && !frame.box_visible && !placed_current_boxes) {
        state.update(true);
        addPlacedBox(released, left_id,
          state.getGlobalLinkTransform("left_tool0") * toolToBox("left", top), "left");
        addPlacedBox(released, right_id,
          state.getGlobalLinkTransform("right_tool0") * toolToBox("right", top), "right");
        placed_current_boxes = true;
      }
      if (attached) {
        attachCarriedBox(state, kCarriedBoxLeftId, "left", top);
        attachCarriedBox(state, kCarriedBoxRightId, "right", top);
      }
      state.update(true);
      if (!state.satisfiesBounds()) { if (reason) *reason = "dual_joint_bounds"; return false; }
      const auto active_scene = attached ? loaded : (frame.box_visible ? scene : released);
      const std::string collision = collisionReason(active_scene, state, metrics);
      if (!collision.empty()) {
        if (reason) *reason = "dual_" + frame.stage + "_" + collision;
        return false;
      }
      if (previous) {
        if (!alfa_robot::motion::sameShoulderElbowBranch(sideJoints(*previous, "left"), sideJoints(state, "left")) ||
            !alfa_robot::motion::sameShoulderElbowBranch(sideJoints(*previous, "right"), sideJoints(state, "right"))) {
          if (reason) *reason = "dual_shoulder_elbow_branch_flip";
          return false;
        }
        if (attached == previous_attached) {
          double maximum_delta = 0.0;
          for (const auto& name : all_joint_names_)
            maximum_delta = std::max(maximum_delta, std::abs(
              state.getVariablePosition(name) - previous->getVariablePosition(name)));
          const size_t steps = std::max<size_t>(1, std::ceil(maximum_delta / edge_joint_resolution_));
          for (size_t step = 1; step < steps; ++step) {
            moveit::core::RobotState probe(*previous);
            const double ratio = static_cast<double>(step) / steps;
            for (const auto& name : all_joint_names_)
              probe.setVariablePosition(name, previous->getVariablePosition(name) +
                (state.getVariablePosition(name) - previous->getVariablePosition(name)) * ratio);
            if (attached) {
              attachCarriedBox(probe, kCarriedBoxLeftId, "left", top);
              attachCarriedBox(probe, kCarriedBoxRightId, "right", top);
            }
            probe.update(true);
            const auto edge_scene = attached ? loaded :
              ((frame.box_visible || previous_visible) ? scene : released);
            const std::string edge_collision = collisionReason(edge_scene, probe, metrics);
            if (!probe.satisfiesBounds() || !edge_collision.empty()) {
              if (reason) *reason = "dual_edge_to_" + frame.stage + "_" +
                (edge_collision.empty() ? std::string("bounds") : edge_collision);
              return false;
            }
          }
        }
      }
      previous = std::make_shared<moveit::core::RobotState>(state);
      previous_attached = attached;
      previous_visible = frame.box_visible;
    }
    return true;
  }

  TaskResult planDualPair(int left_id, int right_id, bool top, double x)
  {
    TaskResult result;
    const auto started = std::chrono::steady_clock::now();
    const auto common_joints = allJoints(*initial_state_);
    updateWallTarget(x, left_id);
    const auto left_center = box_center_;
    updateWallTarget(x, right_id);
    const auto right_center = box_center_;
    const auto& updown_bounds = robot_model_->getVariableBounds("updown");
    std::vector<std::pair<double, double>> shared_heights;
    if (enable_stage_action_ && !top) {
      const auto scene = makeDualScene(left_id, right_id);
      for (int height_index = 0; height_index <= 25; ++height_index) {
        const double height = std::min(updown_bounds.max_position_,
          updown_bounds.min_position_ + 0.04 * height_index);
        moveit::core::RobotState lifted(*initial_state_);
        const double initial_height = lifted.getVariablePosition("updown");
        const size_t lift_steps = std::max<size_t>(1,
          std::ceil(std::abs(height - initial_height) / 0.005));
        bool lift_clear = true;
        for (size_t step = 1; step <= lift_steps; ++step) {
          lifted.setVariablePosition("updown", initial_height +
            (height - initial_height) * step / lift_steps);
          lifted.update(true);
          if (!lifted.satisfiesBounds() || !collisionReason(scene, lifted, &result.metrics).empty()) {
            lift_clear = false;
            break;
          }
        }
        if (!lift_clear) continue;
        std::array<std::vector<AnalyticCandidate>, 2> candidates;
        for (size_t side_index = 0; side_index < 2; ++side_index) {
          const std::string side = side_index == 0 ? "left" : "right";
          const auto& center = side_index == 0 ? left_center : right_center;
          selectArm(side);
          top_suction_ = false;
          candidates[side_index] = solvePoseCandidates(
            precontactPose(center), lifted, false, scene, &result.metrics, false, nullptr);
        }
        if (candidates[0].empty() || candidates[1].empty()) continue;
        double best_cost = std::numeric_limits<double>::infinity();
        for (size_t left_index = 0; left_index < std::min<size_t>(4, candidates[0].size()); ++left_index)
          for (size_t right_index = 0; right_index < std::min<size_t>(4, candidates[1].size()); ++right_index) {
            moveit::core::RobotState paired(*candidates[0][left_index].state);
            for (int joint = 1; joint <= 7; ++joint) {
              const std::string name = "right_joint" + std::to_string(joint);
              paired.setVariablePosition(name,
                candidates[1][right_index].state->getVariablePosition(name));
            }
            paired.update(true);
            if (!paired.satisfiesBounds() ||
                !collisionReason(scene, paired, &result.metrics).empty()) continue;
            best_cost = std::min(best_cost, candidates[0][left_index].score +
              candidates[1][right_index].score);
          }
        if (std::isfinite(best_cost)) shared_heights.emplace_back(best_cost, height);
      }
      std::sort(shared_heights.begin(), shared_heights.end());
    } else {
      selectArm("left"); top_suction_ = top;
      height_clearance_ = {{"checked", false}};
      const double left_updown = heightAlignment(left_center).at("target_updown").get<double>();
      selectArm("right"); top_suction_ = top;
      height_clearance_ = {{"checked", false}};
      const double right_updown = heightAlignment(right_center).at("target_updown").get<double>();
      shared_heights.emplace_back(0.0, alfa_robot::motion::chooseSharedUpdown(
        left_updown, right_updown, updown_bounds.min_position_, updown_bounds.max_position_));
    }
    TaskResult left, right;
    for (const auto& [cost, height] : shared_heights) {
      synchronized_updown_ = height;
      RCLCPP_INFO(get_logger(), "dual shared height: boxes=%d,%d height=%.3fm score=%.3f",
        left_id, right_id, height, cost);
      updateWallTarget(x, left_id);
      selectArm("left"); top_suction_ = top;
      left = planTask(left_center);
      updateWallTarget(x, right_id);
      selectArm("right"); top_suction_ = top;
      right = planTask(right_center);
      result.metrics.add(left.metrics);
      result.metrics.add(right.metrics);
      if (left.success && right.success) break;
      if (!enable_stage_action_) break;
    }
    synchronized_updown_.reset();
    if (!left.success || !right.success) {
      result.failure_stage = shared_heights.empty() ? "shared_height_ik" : "dual_independent_plan";
      result.failure_reason = (shared_heights.empty() ? "no collision-free paired precontact IK; " : "") +
        std::string("left=") + (left.success ? std::string("ok") : left.failure_stage + ":" + left.failure_reason) +
        " right=" + (right.success ? std::string("ok") : right.failure_stage + ":" + right.failure_reason);
      result.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
      return result;
    }

    std::vector<double> left_hold = common_joints, right_hold = common_joints;
    bool left_attached = false, right_attached = false, left_visible = true, right_visible = true;
    for (int phase = 0; phase <= 8; ++phase) {
      std::vector<const ReplayFrame*> left_phase, right_phase;
      for (const auto& frame : left.frames) if (transferPhase(frame.stage) == phase) left_phase.push_back(&frame);
      for (const auto& frame : right.frames) if (transferPhase(frame.stage) == phase) right_phase.push_back(&frame);
      const size_t count = std::max(left_phase.size(), right_phase.size());
      if (!count) continue;
      for (size_t i = 0; i < count; ++i) {
        auto sample = [i, count](const std::vector<const ReplayFrame*>& phase_frames) -> const ReplayFrame* {
          if (phase_frames.empty()) return nullptr;
          const size_t index = count == 1 ? phase_frames.size() - 1 :
            std::min(phase_frames.size() - 1, i * phase_frames.size() / count);
          return phase_frames[index];
        };
        if (const auto* frame = sample(left_phase)) {
          left_hold = frame->joints; left_attached = frame->box_attached; left_visible = frame->box_visible;
        }
        if (const auto* frame = sample(right_phase)) {
          right_hold = frame->joints; right_attached = frame->box_attached; right_visible = frame->box_visible;
        }
        if (left_attached != right_attached || left_visible != right_visible) {
          result.failure_stage = "dual_stage_sync";
          result.failure_reason = "independent plans changed attachment at different phase boundaries";
          result.frames.clear();
          return result;
        }
        ReplayFrame merged;
        merged.stage = "dual_" + std::to_string(phase);
        merged.joints = common_joints;
        for (size_t j = 0; j < all_joint_names_.size(); ++j) {
          const auto& name = all_joint_names_[j];
          if (name.rfind("left_", 0) == 0) merged.joints[j] = left_hold[j];
          else if (name.rfind("right_", 0) == 0) merged.joints[j] = right_hold[j];
          else if (name == "updown") {
            const bool loaded_return = post_extract_policy_ == "loaded_home" &&
              (phase == 5 || phase == 6);
            if (phase > 0 && phase < 8 && !loaded_return &&
                std::abs(left_hold[j] - right_hold[j]) > 1e-4) {
              result.failure_stage = "dual_height_sync";
              result.failure_reason = "left/right updown goals differ after alignment";
              result.frames.clear();
              return result;
            }
            // The lift is shared. Independently sampled phase 0 alignment and phase 8
            // HOME return are resynchronized by common progress, then revalidated.
            merged.joints[j] = 0.5 * (left_hold[j] + right_hold[j]);
          } else merged.joints[j] = left_hold[j];
        }
        merged.box_attached = left_attached;
        merged.box_visible = left_visible;
        merged.carried_boxes.push_back(carriedBoxJson(
          left_id, left_center, "left", top, left_attached, left_visible));
        merged.carried_boxes.push_back(carriedBoxJson(
          right_id, right_center, "right", top, right_attached, right_visible));
        result.frames.push_back(std::move(merged));
      }
    }
    std::string validation_reason;
    if (!validateDualFrames(result.frames, left_id, right_id, top, &result.metrics, &validation_reason)) {
      result.failure_stage = "dual_combined_validation";
      result.failure_reason = validation_reason;
      result.frames.clear();
    } else {
      result.success = true;
      if (post_extract_policy_ == "loaded_home")
        for (size_t joint = 0; joint < common_joints.size(); ++joint)
          if (std::abs(result.frames.back().joints[joint] - common_joints[joint]) > 1e-8) {
            result.success = false;
            result.failure_stage = "loaded_home_endpoint";
            result.failure_reason = all_joint_names_[joint] + " did not return to its initial value";
            result.frames.clear();
            break;
          }
    }
    result.total_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return result;
  }

  void ensureFailurePlayback(TaskResult& result, const Eigen::Vector3d& target) const
  {
    if (result.success || !result.diagnostic_frames.empty()) return;
    result.diagnostic_frames = result.frames;
    if (result.diagnostic_frames.empty())
      result.diagnostic_frames.push_back({"FAILED_HOLD: " + result.failure_stage,
        allJoints(*initial_state_), false});
    result.diagnostic = {{"diagnostic_only", true}, {"freeze_at_end", true},
      {"stage", result.failure_stage}, {"reason", result.failure_reason},
      {"snapshot", "last_available_state_no_rejected_configuration"},
      {"target", {target.x(), target.y(), target.z()}}};
  }

  void resetInitialState()
  {
    *initial_state_ = *home_state_;
    // Explicit simulation initial condition, not a motion from a colliding home.
    // The default SRDF home and all hardware/model startup states stay unchanged.
    if (initial_pose_ == "arms_down")
      for (const std::string side : {"left", "right"})
        initial_state_->setVariablePosition(side + "_joint4", 0.0);
    initial_state_->update(true);
  }

  void runSequence()
  {
    const auto started = std::chrono::steady_clock::now();
    const uint64_t generation = ++generation_;
    removed_boxes_.clear();
    placed_boxes_.clear();
    resetInitialState();
    display_diagnostic_ = nlohmann::json::object();
    display_failure_frozen_ = false;
    playback_frames_.clear();
    playback_scenes_.clear();
    sequence_playback_ = false;
    const double x = get_parameter("x").as_double();
    requested_arm_ = get_parameter("arm").as_string();
    updateWallTarget(x, 20);
    selectArm("left");
    const auto first_scene = sceneJson(box_center_);
    publishPlanningStarted(generation, box_center_);

    TaskResult total;
    total.success = true;
    nlohmann::json boxes = nlohmann::json::array();
    nlohmann::json rounds_json = nlohmann::json::array();
    int failed_box = -1;
    size_t dual_success_count = 0;
    size_t fallback_count = 0;

    auto append_result = [&](TaskResult& result, const std::vector<int>& ids,
                             bool dual, const std::string& fallback_reason) {
      ensureFailurePlayback(result, box_center_);
      total.metrics.add(result.metrics);
      const size_t scene_index = playback_scenes_.size();
      auto context = sceneJson(box_center_);
      context["side"] = dual ? "dual" : side_;
      context["box_ids"] = ids;
      context["dual"] = dual;
      playback_scenes_.push_back(context);
      for (auto& frame : result.frames) frame.scene_index = scene_index;
      for (auto& frame : result.diagnostic_frames) frame.scene_index = scene_index;
      publishTaskResult(generation, box_center_, result, false);
      last_result_["side"] = dual ? "dual" : side_;
      last_result_["box_ids"] = ids;
      last_result_["dual"] = dual;
      if (!fallback_reason.empty()) last_result_["fallback_reason"] = fallback_reason;
      auto segment = last_result_;
      segment["kind"] = "segment";
      segment["sequence"] = true;
      segment["segment_index"] = scene_index;
      segment["frame_begin"] = total.frames.size();
      segment["frame_end"] = total.frames.size() +
        (result.success ? result.frames.size() : result.diagnostic_frames.size());
      segment["scenes"] = playback_scenes_;
      segment["planning_elapsed_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
      publishJson(segment, true);
      auto record = last_result_;
      record.erase("frames"); record.erase("diagnostic_frames");
      record["frame_begin"] = total.frames.size();
      if (result.success) {
        total.frames.insert(total.frames.end(), result.frames.begin(), result.frames.end());
        const auto& final = result.frames.back();
        for (size_t j = 0; j < all_joint_names_.size(); ++j)
          initial_state_->setVariablePosition(all_joint_names_[j], final.joints[j]);
        initial_state_->clearAttachedBodies();
        initial_state_->update(true);
        removed_boxes_.insert(ids.begin(), ids.end());
      } else {
        total.success = false;
        total.failure_stage = result.failure_stage;
        total.failure_reason = result.failure_reason;
        failed_box = ids.empty() ? -1 : ids.front();
        total.diagnostic_frames = total.frames;
        total.diagnostic_frames.insert(total.diagnostic_frames.end(),
          result.diagnostic_frames.begin(), result.diagnostic_frames.end());
        total.diagnostic = result.diagnostic;
      }
      record["frame_end"] = total.frames.size();
      boxes.push_back(record);
      RCLCPP_INFO(get_logger(), "SEQUENCE_SEGMENT_READY generation=%llu segment=%zu boxes=%s dual=%d success=%d",
        static_cast<unsigned long long>(generation), scene_index,
        nlohmann::json(ids).dump().c_str(), dual, result.success);
      return result.success;
    };

    for (const auto& round : alfa_robot::motion::wallTransferRounds()) {
      nlohmann::json round_record = {{"left_box", round.left_box}, {"right_box", round.right_box},
        {"dual_attempted", round.dual()}, {"dual_success", false}, {"fallback", false}};
      if (round.dual()) {
        TaskResult dual_result;
        std::string dual_failure;
        const bool bottom = round.left_box / 5 == 0;
        for (const bool top : bottom ? std::vector<bool>{true} : std::vector<bool>{false, true}) {
          dual_result = planDualPair(round.left_box, round.right_box, top, x);
          round_record["attempts"].push_back({{"suction_mode", top ? "top" : "front"},
            {"success", dual_result.success}, {"failure_stage", dual_result.failure_stage},
            {"failure_reason", dual_result.failure_reason}});
          if (dual_result.success) break;
          total.metrics.add(dual_result.metrics);
          dual_failure = dual_result.failure_stage + ": " + dual_result.failure_reason;
          RCLCPP_WARN(get_logger(), "DUAL_ATTEMPT boxes=%d,%d suction=%s failed: %s",
            round.left_box, round.right_box, top ? "top" : "front", dual_failure.c_str());
        }
        if (dual_result.success) {
          ++dual_success_count;
          round_record["dual_success"] = true;
          if (!append_result(dual_result, {round.left_box, round.right_box}, true, "")) break;
        } else {
          ++fallback_count;
          round_record["fallback"] = true;
          round_record["fallback_reason"] = dual_failure;
          for (const int id : {round.left_box, round.right_box}) {
            updateWallTarget(x, id);
            requested_arm_ = id == round.left_box ? "left" : "right";
            TaskResult single = planWithFallback(box_center_);
            if (!append_result(single, {id}, false, dual_failure)) break;
          }
        }
      } else {
        const int id = round.left_box >= 0 ? round.left_box : round.right_box;
        updateWallTarget(x, id);
        requested_arm_ = round.left_box >= 0 ? "left" : "right";
        TaskResult single = planWithFallback(box_center_, true);
        if (!append_result(single, {id}, false, "")) {
          rounds_json.push_back(round_record);
          break;
        }
      }
      rounds_json.push_back(round_record);
      if (!total.success) break;
    }

    if (total.frames.empty()) {
      playback_scenes_.push_back(first_scene);
      total.frames.push_back(ReplayFrame{"sequence_stopped", allJoints(*initial_state_), false});
    }
    total.total_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    publishTaskResult(generation, box_center_, total, false);
    for (auto it = first_scene.begin(); it != first_scene.end(); ++it) last_result_[it.key()] = it.value();
    last_result_["sequence"] = true;
    last_result_["segment_count"] = boxes.size();
    last_result_["frame_begin"] = 0;
    last_result_["frame_end"] = total.success ? total.frames.size() : total.diagnostic_frames.size();
    last_result_["sequence_order"] = alfa_robot::motion::wallSequenceOrder();
    last_result_["transfer_rounds"] = rounds_json;
    last_result_["dual_success_count"] = dual_success_count;
    last_result_["dual_target_count"] = 10;
    last_result_["fallback_count"] = fallback_count;
    last_result_["full_dual_pass"] = total.success && fallback_count == 0 && dual_success_count == 10;
    last_result_["completed_count"] = removed_boxes_.size();
    last_result_["remaining_count"] = 25 - removed_boxes_.size();
    last_result_["final_removed_box_ids"] = removed_boxes_;
    last_result_["failed_box_id"] = failed_box;
    last_result_["boxes"] = boxes;
    last_result_["scenes"] = playback_scenes_;
    last_result_["attempts"] = nlohmann::json::array();
    last_result_["scope"] = "simulation-only geometric dual-arm transfer; no suction dynamics validation";
    last_result_["final_joints"] = allJoints(*initial_state_);
    publishJson(last_result_);
    display_diagnostic_ = total.success ? nlohmann::json::object() : total.diagnostic;
    if (playback_enabled_) {
      playback_frames_ = total.success || total.diagnostic_frames.empty() ?
        std::move(total.frames) : std::move(total.diagnostic_frames);
      playback_index_ = 0;
      sequence_playback_ = true;
      display_scene_ = playback_scenes_.front();
    }
    publishStatus("SEQUENCE " + std::string(total.success ? "SUCCESS" : "STOPPED") +
      " completed=" + std::to_string(removed_boxes_.size()) + "/25 dual=" +
      std::to_string(dual_success_count) + "/10 fallback=" + std::to_string(fallback_count), total.success);
  }

  bool requestPlanning()
  {
    if (busy()) return false;
    if (planning_requested_.exchange(true)) {
      return false;
    }
    if (distance_demo_) selectArm(requested_arm_ == "auto" ? "left" : requested_arm_);
    publishStatus("PLANNING REQUESTED", true);
    return true;
  }

  void onWorkerTimer()
  {
    if (!planning_requested_.exchange(false)) {
      return;
    }
    if (planning_active_.exchange(true)) {
      return;
    }
    if (sequence_requested_) {
      sequence_requested_ = false;
      sequence_running_ = true;
      try { runSequence(); }
      catch (const std::exception& error) {
        TaskResult failure;
        failure.failure_stage = "exception";
        failure.failure_reason = error.what();
        publishTaskResult(generation_, box_center_, failure, false);
        last_result_["sequence"] = true;
        last_result_["completed_count"] = removed_boxes_.size();
        last_result_["remaining_count"] = 25 - removed_boxes_.size();
        last_result_["failed_box_id"] = wall_target_row_ * 5 + wall_target_column_;
        publishJson(last_result_);
        publishStatus(std::string("SEQUENCE ERROR: ") + error.what(), false);
        RCLCPP_ERROR(get_logger(), "sequence exception: %s", error.what());
      }
      sequence_running_ = false;
      planning_active_.store(false);
      return;
    }
    removed_boxes_.clear();
    placed_boxes_.clear();
    if (wall_context_ == "sequence_prefix") {
      for (int id : alfa_robot::motion::wallSequenceOrder()) {
        if (id == wall_target_row_ * 5 + wall_target_column_) break;
        removed_boxes_.insert(id);
      }
    }
    resetInitialState();
    top_suction_ = distance_demo_ && (requested_suction_mode_ == "top" ||
      alfa_robot::motion::isBottomBox(box_center_.z(), box_height_, wall_bottom_z_));
    playback_scenes_.clear();
    display_scene_ = nlohmann::json();
    display_box_visible_ = true;
    Eigen::Vector3d box_center;
    {
      std::lock_guard<std::mutex> lock(box_mutex_);
      box_center = box_center_;
    }
    if (direct_attach_) {
      box_center = (initial_state_->getGlobalLinkTransform(tool_link_) * toolToBox()).translation();
      std::lock_guard<std::mutex> lock(box_mutex_);
      box_center_ = box_center;
    }
    const uint64_t generation = ++generation_;
    publishPlanningStarted(generation, box_center);
    publishStatus("CALCULATING", true);
    RCLCPP_INFO(
      get_logger(),
      "[%llu] calculation started: box_center=[%.3f, %.3f, %.3f]",
      static_cast<unsigned long long>(generation),
      box_center.x(), box_center.y(), box_center.z());
    if (distance_demo_ && !direct_attach_) {
      const auto alignment = heightAlignment(box_center);
      RCLCPP_INFO(get_logger(),
        "height proposal (clearance unchecked): strategy=%s enabled=%s descent=%.6fm target_updown=%.6fm",
        height_strategy_.c_str(), align_height_ ? "true" : "false",
        alignment.at("descent").get<double>(), alignment.at("target_updown").get<double>());
    }

    {
      std::lock_guard<std::mutex> lock(display_mutex_);
      display_diagnostic_ = nlohmann::json::object();
      display_failure_frozen_ = false;
      playback_frames_.clear();
      display_box_attached_ = false;
      *display_state_ = *initial_state_;
    }
    publishSceneMarkers();
    TaskResult result;
    const auto request_started = std::chrono::steady_clock::now();
    try {
      result = direct_attach_ ? planDirectAttachedPlacement() : planWithFallback(box_center);
    } catch (const std::exception& error) {
      result.success = false;
      result.failure_stage = "exception";
      result.failure_reason = error.what();
    }
    if (distance_demo_) result.total_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - request_started).count();
    if (distance_demo_ && !direct_attach_ && !result.success)
      result.frames = {ReplayFrame{"planning_failed", allJoints(*initial_state_), false}};
    ensureFailurePlayback(result, box_center);
    display_scene_ = sceneJson(box_center);
    publishTaskResult(generation, box_center, result);
    if (playback_enabled_ && (!result.frames.empty() || !result.diagnostic_frames.empty())) {
      std::lock_guard<std::mutex> lock(display_mutex_);
      playback_frames_ = result.success || result.diagnostic_frames.empty() ? result.frames : result.diagnostic_frames;
      display_diagnostic_ = result.success ? nlohmann::json::object() : result.diagnostic;
      playback_index_ = 0;
    }
    if (result.success) {
      std::ostringstream status;
      status << "SUCCESS total=" << std::fixed << std::setprecision(1)
             << result.total_ms << "ms";
      if (distance_demo_) {
        status << " arm=" << side_ << " lift=" << std::setprecision(3)
               << heightAlignment(box_center).at("target_updown").get<double>() << "m";
      }
      publishStatus(status.str(), true);
      RCLCPP_INFO(
        get_logger(),
        "[%llu] calculation completed: SUCCESS total=%.3fms analytic=%.3fms "
        "rrt_to=%.3fms rrt_return=%.3fms ik_calls=%llu collision_checks=%llu",
        static_cast<unsigned long long>(generation), result.total_ms,
        result.metrics.analytic_path_ms, result.metrics.rrt_approach_ms,
        result.metrics.rrt_return_ms,
        static_cast<unsigned long long>(result.metrics.ik_calls),
        static_cast<unsigned long long>(result.metrics.collision_checks));
    } else {
      publishStatus(
        "FAILED " + result.failure_stage + ": " + result.failure_reason, false);
      RCLCPP_ERROR(
        get_logger(),
        "[%llu] calculation completed: FAILED total=%.3fms stage=%s reason=%s",
        static_cast<unsigned long long>(generation), result.total_ms,
        result.failure_stage.c_str(), result.failure_reason.c_str());
    }
    planning_active_.store(false);
  }

  Eigen::Vector3d boxSize() const { return {box_depth_, box_width_, box_height_}; }

  Eigen::Isometry3d contactPose(
    const Eigen::Vector3d& box_center, const std::string& side, bool top) const
  {
    if (distance_demo_)
      return alfa_robot::motion::wallContactPose(
        box_center, boxSize(), contact_numerical_gap_, top,
        home_state_->getGlobalLinkTransform(side + "_tool0").linear());
    auto pose = alfa_robot::motion::wallContactPose(box_center, boxSize(), contact_numerical_gap_, top);
    if (top)
      pose.linear() = Eigen::AngleAxisd(side == "left" ? kPi / 2.0 : -kPi / 2.0,
        Eigen::Vector3d::UnitZ()).toRotationMatrix() * pose.linear();
    return pose;
  }

  Eigen::Isometry3d contactPose(const Eigen::Vector3d& box_center) const
  {
    return contactPose(box_center, side_, top_suction_);
  }

  Eigen::Isometry3d toolToBox(const std::string& side, bool top) const
  {
    return contactPose(Eigen::Vector3d::Zero(), side, top).inverse();
  }

  Eigen::Isometry3d toolToBox() const { return toolToBox(side_, top_suction_); }
  Eigen::Vector3d toolToBoxCenter() const { return toolToBox().translation(); }

  Eigen::Isometry3d precontactPose(const Eigen::Vector3d& box_center) const
  {
    Eigen::Isometry3d pose = contactPose(box_center);
    pose.translation() -= pose.linear().col(2) * approach_distance_;
    return pose;
  }

  Eigen::Isometry3d retreatPose(const Eigen::Vector3d& box_center) const
  {
    Eigen::Isometry3d pose = contactPose(box_center);
    pose.translation().x() -= retreat_distance_;
    return pose;
  }

  std::vector<Eigen::Vector3d> neighborCenters(const Eigen::Vector3d& center) const
  {
    if (direct_attach_) return {};
    if (scene_layout_ == "wall_5x5") {
      if (wall_context_ == "target_only") {
        return {};
      }
      std::vector<Eigen::Vector3d> neighbors;
      neighbors.reserve(24);
      for (int row = 0; row < 5; ++row) {
        for (int column = 0; column < 5; ++column) {
          if ((row == wall_target_row_ && column == wall_target_column_) ||
              removed_boxes_.count(row * 5 + column)) continue;
          neighbors.push_back(center + Eigen::Vector3d(
            0.0, (column - wall_target_column_) * (box_width_ + wall_gap_),
            (row - wall_target_row_) * (box_height_ + wall_gap_)));
        }
      }
      return neighbors;
    }
    return {
      center + Eigen::Vector3d(0.0, box_width_, 0.0),
      center - Eigen::Vector3d(0.0, box_width_, 0.0),
      center + Eigen::Vector3d(0.0, 0.0, box_height_),
      center - Eigen::Vector3d(0.0, 0.0, box_height_),
    };
  }

  planning_scene::PlanningScenePtr makeScene(const Eigen::Vector3d& box_center) const
  {
    // Legacy preview omits the target; wall mode keeps it until attachment.
    auto scene = std::make_shared<planning_scene::PlanningScene>(robot_model_);
    scene->setCurrentState(*initial_state_);
    // Shared by initial state, lift, IK, approach, attachment, retreat and loaded RRT.
    // No robot/ground or carried-box/environment ACM exemptions.
    for (const auto& object : environment_objects_) {
      if (!scene->processCollisionObjectMsg(object))
        throw std::runtime_error("failed to add " + object.id + " to planning scene");
    }
    auto neighbors = neighborCenters(box_center);
    if (distance_demo_) neighbors.push_back(box_center);
    const double depth = std::max(0.001, box_depth_ - 2.0 * collision_inset_);
    const double width = std::max(0.001, box_width_ - 2.0 * collision_inset_);
    const double height = std::max(0.001, box_height_ - 2.0 * collision_inset_);
    for (size_t index = 0; index < neighbors.size(); ++index) {
      moveit_msgs::msg::CollisionObject object;
      object.header.frame_id = world_frame_;
      object.id = distance_demo_ && index == neighbors.size() - 1 ?
        kCarriedBoxId : "neighbor_box_" + std::to_string(index);
      object.operation = moveit_msgs::msg::CollisionObject::ADD;
      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
      primitive.dimensions = {depth, width, height};
      geometry_msgs::msg::Pose pose;
      pose.position.x = neighbors[index].x();
      pose.position.y = neighbors[index].y();
      pose.position.z = neighbors[index].z();
      pose.orientation.w = 1.0;
      object.primitives.push_back(primitive);
      object.primitive_poses.push_back(pose);
      if (!scene->processCollisionObjectMsg(object)) {
        throw std::runtime_error("failed to add " + object.id + " to planning scene");
      }
    }
    for (const auto& placed : placed_boxes_)
      addPlacedBox(scene, placed.first, placed.second);
    return scene;
  }

  planning_scene::PlanningScenePtr loadedScene(
    const planning_scene::PlanningSceneConstPtr& scene) const
  {
    auto loaded = planning_scene::PlanningScene::clone(scene);
    if (distance_demo_) loaded->getWorldNonConst()->removeObject(kCarriedBoxId);
    return loaded;
  }

  void attachCarriedBox(
    moveit::core::RobotState& state, const std::string& object_id,
    const std::string& side, bool top) const
  {
    if (state.hasAttachedBody(object_id)) return;
    const Eigen::Vector3d size = boxSize().array() - 2.0 * collision_inset_;
    std::vector<shapes::ShapeConstPtr> shapes{
      std::make_shared<shapes::Box>(size.x(), size.y(), size.z())};
    EigenSTL::vector_Isometry3d shape_poses{toolToBox(side, top)};
    const std::string tool = side + "_tool0";
    const std::string prefix = side + "_";
    state.attachBody(object_id, Eigen::Isometry3d::Identity(), shapes, shape_poses,
      distance_demo_ ? std::vector<std::string>{tool, prefix + "link7"} :
        std::vector<std::string>{tool, prefix + "link7", prefix + "link6"}, tool);
    state.update(true);
  }

  void attachCarriedBox(moveit::core::RobotState& state) const
  {
    attachCarriedBox(state, kCarriedBoxId, side_, top_suction_);
  }

  nlohmann::json directAttachedBoxes(const moveit::core::RobotState& state) const
  {
    auto boxes = nlohmann::json::array();
    for (const std::string side : {"left", "right"}) {
      const Eigen::Isometry3d pose = state.getGlobalLinkTransform(side + "_tool0") *
        toolToBox(side, false);
      auto box = carriedBoxJson(side == "left" ? 0 : 1,
        pose.translation(), side, false, true, true);
      box["box_rotation"] = nlohmann::json::array();
      for (int row = 0; row < 3; ++row)
        box["box_rotation"].push_back({pose.linear()(row, 0),
          pose.linear()(row, 1), pose.linear()(row, 2)});
      boxes.push_back(std::move(box));
    }
    return boxes;
  }

  moveit::core::RobotState directPlacementGoal(const moveit::core::RobotState& start) const
  {
    moveit::core::RobotState goal(start);
    if (!goal.setToDefaultValues(robot_model_->getJointModelGroup("dual_arm"), direct_placement_pose_))
      throw std::runtime_error("dual_arm/" + direct_placement_pose_ + " named pose is unavailable");
    goal.update(true);
    return goal;
  }

  TaskResult planDirectAttachedPlacement()
  {
    TaskResult result;
    const auto started = std::chrono::steady_clock::now();
    const auto scene = makeScene(box_center_);
    scene->getWorldNonConst()->removeObject(kCarriedBoxId);
    moveit::core::RobotState start(*initial_state_);
    attachCarriedBox(start, kCarriedBoxLeftId, "left", false);
    attachCarriedBox(start, kCarriedBoxRightId, "right", false);
    const auto append_frame = [&](const moveit::core::RobotState& state,
                                  const std::string& stage) {
      ReplayFrame frame{stage, allJoints(state), true};
      frame.carried_boxes = directAttachedBoxes(state);
      result.frames.push_back(std::move(frame));
    };
    append_frame(start, "direct_attach");
    const auto finish = [&]() {
      if (!result.success) {
        if (result.diagnostic_frames.empty()) {
          result.diagnostic_frames = result.frames;
          result.diagnostic_frames.back().stage = "FAILED_HOLD: " + result.failure_stage;
        }
        result.diagnostic = {{"diagnostic_only", true}, {"freeze_at_end", true},
          {"stage", result.failure_stage}, {"reason", result.failure_reason},
          {"snapshot", result.diagnostic_frames.back().stage == "REJECTED_SHORTCUT_PREVIEW_NOT_EXECUTED" ?
            "rejected_shortcut_preview_not_executed" : "last_connected_loaded_state"}};
        if (result.rejected_state) {
          result.diagnostic["rejected_joints"] = allJoints(*result.rejected_state);
          collision_detection::CollisionRequest request;
          collision_detection::CollisionResult contacts;
          request.contacts = true;
          request.max_contacts = 20;
          request.max_contacts_per_pair = 1;
          scene->checkCollision(request, contacts, *result.rejected_state);
          for (const auto& pair : contacts.contacts)
            for (const auto& contact : pair.second)
              result.diagnostic["contacts"].push_back({
                {"bodies", {pair.first.first, pair.first.second}},
                {"position", {contact.pos.x(), contact.pos.y(), contact.pos.z()}}});
        }
      }
      result.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
      return result;
    };
    const std::string start_collision = collisionReason(scene, start, &result.metrics);
    if (!start.satisfiesBounds() || !start_collision.empty()) {
      result.failure_stage = "direct_attach";
      result.failure_reason = start_collision.empty() ? "joint_bounds" : start_collision;
      result.rejected_state = std::make_shared<moveit::core::RobotState>(start);
      return finish();
    }
    const auto goal = directPlacementGoal(start);
    const auto* group = robot_model_->getJointModelGroup("dual_arm");
    if (!group || group->getVariableCount() != 14)
      throw std::runtime_error("direct placement requires the 14-axis dual_arm group");
    const auto path = planRrt(scene, start, goal, false, &result.metrics, group);
    result.metrics.rrt_return_ms = path.wall_ms;
    for (size_t index = 1; index < path.states.size(); ++index) {
      const auto segment = directArmPath(*path.states[index - 1], *path.states[index],
        false, 0.0, 0.005, group);
      for (size_t sample = 1; sample < segment.size(); ++sample)
        append_frame(*segment[sample], "loaded_transfer");
    }
    if (!path.success) {
      result.failure_stage = "loaded_transfer";
      result.failure_reason = path.reason;
      result.rejected_state = path.rejected_state;
      if (!path.states.empty()) {
        std::string blocked_reason;
        moveit::core::RobotStatePtr blocked;
        edgeClear(scene, *path.states.back(), goal, false, nullptr,
          &blocked_reason, &blocked, false, group);
        if (blocked) {
          result.diagnostic_frames = result.frames;
          const auto probes = directArmPath(*path.states.back(), *blocked,
            false, 0.0, 0.005, group);
          for (size_t sample = 1; sample < probes.size(); ++sample) {
            ReplayFrame frame{"REJECTED_SHORTCUT_PREVIEW_NOT_EXECUTED", allJoints(*probes[sample]), true};
            frame.carried_boxes = directAttachedBoxes(*probes[sample]);
            result.diagnostic_frames.push_back(std::move(frame));
          }
          result.rejected_state = blocked;
        }
      }
      return finish();
    }
    append_frame(goal, direct_placement_pose_);
    result.success = true;
    return finish();
  }

  std::array<double, 7> armJoints(const moveit::core::RobotState& state) const
  {
    std::vector<double> values;
    state.copyJointGroupPositions(planning_group_, values);
    if (values.size() != 7U) {
      throw std::runtime_error("unexpected arm joint count");
    }
    std::array<double, 7> output{};
    std::copy(values.begin(), values.end(), output.begin());
    return output;
  }

  std::vector<double> allJoints(const moveit::core::RobotState& state) const
  {
    std::vector<double> output;
    output.reserve(all_joint_names_.size());
    for (const auto& name : all_joint_names_) {
      output.push_back(state.getVariablePosition(name));
    }
    return output;
  }

  std::string collisionReason(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit::core::RobotState& state,
    PlanningMetrics* metrics) const
  {
    const auto started = std::chrono::steady_clock::now();
    const std::string reason = alfa_robot::motion::scene_collision_reason(
      scene, state, distance_demo_ ? nullptr : planning_group_);
    if (metrics) {
      ++metrics->collision_checks;
      const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
      metrics->collision_ms += elapsed_ms;
      if (direct_attach_) metrics->collision_sample_ms.push_back(elapsed_ms);
    }
    return reason;
  }

  bool edgeClear(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit::core::RobotState& from,
    const moveit::core::RobotState& to,
    bool attached,
    PlanningMetrics* metrics,
    std::string* reason,
    moveit::core::RobotStatePtr* rejected = nullptr,
    bool plan_updown = false,
    const moveit::core::JointModelGroup* motion_group = nullptr) const
  {
    const auto* group = motion_group ? motion_group : planning_group_;
    std::vector<double> from_joints, to_joints;
    from.copyJointGroupPositions(group, from_joints);
    to.copyJointGroupPositions(group, to_joints);
    double maximum_delta = 0.0;
    for (size_t index = 0; index < from_joints.size(); ++index)
      maximum_delta = std::max(maximum_delta, std::abs(distance_demo_ ?
        to_joints[index] - from_joints[index] :
        normalizedAngle(to_joints[index] - from_joints[index])));
    size_t steps = std::max<size_t>(
      1, static_cast<size_t>(std::ceil(maximum_delta / edge_joint_resolution_)));
    if (plan_updown)
      steps = std::max(steps, static_cast<size_t>(std::ceil(std::abs(
        to.getVariablePosition("updown") - from.getVariablePosition("updown")) / 0.005)));
    for (size_t step = 1; step <= steps; ++step) {
      const double ratio = static_cast<double>(step) / static_cast<double>(steps);
      std::vector<double> interpolated(from_joints.size());
      for (size_t index = 0; index < interpolated.size(); ++index) {
        interpolated[index] = from_joints[index] +
          (distance_demo_ ? to_joints[index] - from_joints[index] :
           normalizedAngle(to_joints[index] - from_joints[index])) * ratio;
      }
      moveit::core::RobotState probe(from);
      probe.setJointGroupPositions(group, interpolated.data());
      if (plan_updown)
        probe.setVariablePosition("updown", from.getVariablePosition("updown") +
          (to.getVariablePosition("updown") - from.getVariablePosition("updown")) * ratio);
      if (attached && !probe.hasAttachedBody(kCarriedBoxId)) {
        attachCarriedBox(probe);
      } else if (!attached && probe.hasAttachedBody(kCarriedBoxId)) {
        probe.clearAttachedBody(kCarriedBoxId);
      }
      probe.update(true);
      if (!(plan_updown ? probe.satisfiesBounds() : probe.satisfiesBounds(group))) {
        if (reason) *reason = "joint_bounds";
        if (rejected) *rejected = std::make_shared<moveit::core::RobotState>(probe);
        return false;
      }
      const std::string collision = collisionReason(scene, probe, metrics);
      if (!collision.empty()) {
        if (reason) *reason = collision;
        if (rejected) *rejected = std::make_shared<moveit::core::RobotState>(probe);
        return false;
      }
    }
    return true;
  }

  RrtPlanResult moveUpdown(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit::core::RobotState& start,
    double target,
    bool attached,
    PlanningMetrics* metrics) const
  {
    const auto started = std::chrono::steady_clock::now();
    RrtPlanResult result;
    result.states.push_back(std::make_shared<moveit::core::RobotState>(start));
    const double initial = start.getVariablePosition("updown");
    const size_t steps = std::max<size_t>(
      1, static_cast<size_t>(std::ceil(std::abs(target - initial) / 0.005)));
    for (size_t index = 1; index <= steps; ++index) {
      auto state = std::make_shared<moveit::core::RobotState>(start);
      state->setVariablePosition(
        "updown",
        initial + (target - initial) * static_cast<double>(index) /
        static_cast<double>(steps));
      if (attached && !state->hasAttachedBody(kCarriedBoxId)) {
        attachCarriedBox(*state);
      }
      state->update(true);
      if (!state->satisfiesBounds()) {
        result.reason = "joint_bounds";
        result.rejected_state = state;
        break;
      }
      const std::string collision = collisionReason(scene, *state, metrics);
      if (!collision.empty()) {
        result.reason = collision;
        result.rejected_state = state;
        break;
      }
      result.states.push_back(std::move(state));
    }
    result.success = result.states.size() == steps + 1U;
    result.wall_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return result;
  }

  std::vector<moveit::core::RobotStatePtr> directArmPath(
    const moveit::core::RobotState& from, const moveit::core::RobotState& to,
    bool plan_updown = false, double angular_step = 0.0, double updown_step = 0.005,
    const moveit::core::JointModelGroup* motion_group = nullptr) const
  {
    const auto* group = motion_group ? motion_group : planning_group_;
    std::vector<double> from_joints, to_joints;
    from.copyJointGroupPositions(group, from_joints);
    to.copyJointGroupPositions(group, to_joints);
    std::vector<double> deltas(from_joints.size());
    double maximum_delta = 0.0;
    for (size_t i = 0; i < deltas.size(); ++i) {
      deltas[i] = distance_demo_ ? to_joints[i] - from_joints[i] :
        normalizedAngle(to_joints[i] - from_joints[i]);
      maximum_delta = std::max(maximum_delta, std::abs(deltas[i]));
    }
    if (angular_step <= 0.0) angular_step = edge_joint_resolution_;
    size_t steps = std::max<size_t>(1, std::ceil(maximum_delta / angular_step));
    if (plan_updown)
      steps = std::max(steps, static_cast<size_t>(std::ceil(std::abs(
        to.getVariablePosition("updown") - from.getVariablePosition("updown")) / updown_step)));
    std::vector<moveit::core::RobotStatePtr> states;
    states.reserve(steps + 1);
    for (size_t step = 0; step <= steps; ++step) {
      const double ratio = static_cast<double>(step) / steps;
      std::vector<double> joints(from_joints.size());
      for (size_t i = 0; i < joints.size(); ++i)
        joints[i] = from_joints[i] + deltas[i] * ratio;
      auto state = std::make_shared<moveit::core::RobotState>(from);
      state->setJointGroupPositions(group, joints.data());
      if (plan_updown)
        state->setVariablePosition("updown", from.getVariablePosition("updown") +
          (to.getVariablePosition("updown") - from.getVariablePosition("updown")) * ratio);
      state->update(true);
      states.push_back(std::move(state));
    }
    if (distance_demo_) {
      states.front() = std::make_shared<moveit::core::RobotState>(from);
      states.back() = std::make_shared<moveit::core::RobotState>(to);
    }
    return states;
  }

  std::vector<AnalyticCandidate> solvePoseCandidates(
    const Eigen::Isometry3d& target_world,
    const moveit::core::RobotState& seed_state,
    bool attached,
    const planning_scene::PlanningSceneConstPtr& scene,
    PlanningMetrics* metrics,
    bool enforce_step,
    std::string* rejection_summary,
    moveit::core::RobotStatePtr* rejected = nullptr) const
  {
    const Eigen::Isometry3d world_to_arm_base =
      seed_state.getGlobalLinkTransform(arm_base_link_).inverse();
    const Eigen::Isometry3d target_in_arm_base = world_to_arm_base * target_world;
    const auto seed_joints = armJoints(seed_state);
    std::vector<AnalyticCandidate> candidates;
    size_t bounds_rejects = 0;
    size_t jump_rejects = 0;
    size_t collision_rejects = 0;
    std::string last_collision;
    moveit::core::RobotStatePtr last_rejected;

    const int intervals = std::max(1, static_cast<int>(std::ceil(2.0 * kPi / psi_step_)));
    for (int index = 0; index < intervals; ++index) {
      V3RedundantIkRequest request;
      request.target_in_arm_base = target_in_arm_base;
      request.swivel_angle = -kPi + static_cast<double>(index) * 2.0 * kPi / intervals;
      request.seed = seed_joints;
      const auto started = std::chrono::steady_clock::now();
      const auto solutions = solver_->solveInArmBase(request);
      if (metrics) {
        ++metrics->ik_calls;
        metrics->ik_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count();
      }
      for (const auto& solution : solutions) {
        if (enforce_step &&
            (maximumJointDelta(seed_joints, solution.joints,
               connection_planner_ == "shortcut_local_rrt") > maximum_cartesian_joint_step_ ||
             !alfa_robot::motion::sameShoulderElbowBranch(seed_joints, solution.joints))) {
          ++jump_rejects;
          continue;
        }
        moveit::core::RobotState candidate(seed_state);
        candidate.setJointGroupPositions(planning_group_, solution.joints.data());
        if (attached) {
          attachCarriedBox(candidate);
        } else if (candidate.hasAttachedBody(kCarriedBoxId)) {
          candidate.clearAttachedBody(kCarriedBoxId);
        }
        candidate.update(true);
        if (!candidate.satisfiesBounds(planning_group_)) {
          ++bounds_rejects;
          continue;
        }
        const std::string collision = collisionReason(scene, candidate, metrics);
        if (!collision.empty()) {
          ++collision_rejects;
          last_collision = collision;
          last_rejected = std::make_shared<moveit::core::RobotState>(candidate);
          continue;
        }
        const bool duplicate = std::any_of(
          candidates.begin(), candidates.end(),
          [this, &solution](const AnalyticCandidate& existing) {
            return maximumJointDelta(existing.solution.joints, solution.joints,
              connection_planner_ == "shortcut_local_rrt") < 1e-5;
          });
        if (duplicate) {
          continue;
        }
        AnalyticCandidate output;
        output.state = std::make_shared<moveit::core::RobotState>(candidate);
        output.solution = solution;
        const double margin_penalty = 0.02 /
          std::max(0.01, solution.minimum_joint_limit_margin);
        output.score = alfa_robot::motion::naturalJointDistanceSquared(seed_joints, solution.joints,
          connection_planner_ == "shortcut_local_rrt") + margin_penalty;
        candidates.push_back(std::move(output));
      }
    }
    std::sort(
      candidates.begin(), candidates.end(),
      [](const AnalyticCandidate& lhs, const AnalyticCandidate& rhs) {
        return lhs.score < rhs.score;
      });
    if (rejection_summary) {
      std::ostringstream summary;
      summary << "candidates=" << candidates.size()
              << " bounds=" << bounds_rejects
              << " jump=" << jump_rejects
              << " collision=" << collision_rejects
             ;
      if (!last_collision.empty()) {
        summary << " last=" << last_collision;
      }
      *rejection_summary = summary.str();
    }
    if (rejected) *rejected = candidates.empty() ? last_rejected : nullptr;
    return candidates;
  }

  std::optional<AnalyticCandidate> solveNextPose(
    const Eigen::Isometry3d& target_world,
    const moveit::core::RobotState& seed_state,
    bool attached,
    const planning_scene::PlanningSceneConstPtr& scene,
    PlanningMetrics* metrics,
    std::string* reason,
    moveit::core::RobotStatePtr* rejected = nullptr) const
  {
    auto candidates = solvePoseCandidates(
      target_world, seed_state, attached, scene, metrics, true, reason, rejected);
    // Only the first usable Cartesian candidate is consumed. Check edges in
    // score order instead of densely checking every unused swivel solution.
    for (const auto& candidate : candidates) {
      std::string edge_reason;
      if (edgeClear(scene, seed_state, *candidate.state, attached, metrics, &edge_reason, rejected)) {
        if (rejected) rejected->reset();
        return candidate;
      }
      if (reason) *reason = "cartesian candidate edge rejected: " + edge_reason;
    }
    return std::nullopt;
  }

  bool traceCartesianPath(
    const Eigen::Vector3d& box_center,
    const moveit::core::RobotState& precontact_state,
    const planning_scene::PlanningSceneConstPtr& scene,
    PlanningMetrics* metrics,
    std::vector<moveit::core::RobotStatePtr>* approach_states,
    std::vector<moveit::core::RobotStatePtr>* retreat_states,
    std::string* failure_stage,
    std::string* failure_reason,
    TaskResult* diagnostic_result) const
  {
    if (!approach_states || !retreat_states) {
      return false;
    }
    approach_states->clear();
    retreat_states->clear();
    approach_states->push_back(std::make_shared<moveit::core::RobotState>(precontact_state));
    moveit::core::RobotStatePtr current = approach_states->back();
    const Eigen::Isometry3d contact = contactPose(box_center);
    auto contact_scene = planning_scene::PlanningScene::clone(scene);
    if (distance_demo_) {
      contact_scene->getAllowedCollisionMatrixNonConst().setEntry(kCarriedBoxId, tool_link_, true);
      contact_scene->getAllowedCollisionMatrixNonConst().setEntry(kCarriedBoxId, side_ + "_link7", true);
    }
    const auto loaded = loadedScene(scene);
    const size_t approach_steps = std::max<size_t>(
      1, static_cast<size_t>(std::ceil(approach_distance_ / cartesian_step_)));
    for (size_t step = 1; step <= approach_steps; ++step) {
      const double ratio = static_cast<double>(step) / static_cast<double>(approach_steps);
      Eigen::Isometry3d target = contact;
      target.translation() -= contact.linear().col(2) * approach_distance_ * (1.0 - ratio);
      std::string reason;
      auto next = solveNextPose(target, *current, false,
        distance_demo_ && step == approach_steps ? contact_scene : scene, metrics, &reason,
        &diagnostic_result->rejected_state);
      if (!next) {
        diagnostic_result->diagnostic["target"] = {target.translation().x(), target.translation().y(), target.translation().z()};
        if (failure_stage) *failure_stage = "cartesian_approach";
        if (failure_reason) {
          *failure_reason = "step " + std::to_string(step) + "/" +
            std::to_string(approach_steps) + " " + reason;
        }
        return false;
      }
      current = next->state;
      approach_states->push_back(current);
    }

    moveit::core::RobotState contact_attached(*current);
    attachCarriedBox(contact_attached);
    const std::string attach_collision = collisionReason(loaded, contact_attached, metrics);
    if (!attach_collision.empty()) {
      diagnostic_result->rejected_state = std::make_shared<moveit::core::RobotState>(contact_attached);
      if (failure_stage) *failure_stage = "attach_box";
      if (failure_reason) *failure_reason = attach_collision;
      return false;
    }
    current = std::make_shared<moveit::core::RobotState>(contact_attached);
    retreat_states->push_back(current);

    const size_t horizontal_steps = std::max<size_t>(
      1, static_cast<size_t>(std::ceil(retreat_distance_ / cartesian_step_)));
    const size_t lift_steps = top_suction_ ? std::max<size_t>(
      1, static_cast<size_t>(std::ceil(approach_distance_ / cartesian_step_))) : 0;
    const size_t retreat_steps = lift_steps + horizontal_steps;
    for (size_t step = 1; step <= retreat_steps; ++step) {
      Eigen::Isometry3d target = contact;
      // Lift off the support while backing away from the flush rear wall.
      // A purely vertical joint-interpolated segment can tip a box corner into
      // that wall; preserve both obstacles and create clearance through motion.
      const double backoff = top_suction_ ? std::min(0.02, retreat_distance_) : 0.0;
      if (lift_steps) {
        const double lifted = static_cast<double>(std::min(step, lift_steps)) / lift_steps;
        target.translation().z() += approach_distance_ * lifted;
        target.translation().x() -= backoff * lifted;
      }
      if (step > lift_steps) target.translation().x() -= (retreat_distance_ - backoff) *
        static_cast<double>(step - lift_steps) / horizontal_steps;
      std::string reason;
      auto next = solveNextPose(target, *current, true, loaded, metrics, &reason,
        &diagnostic_result->rejected_state);
      if (!next) {
        diagnostic_result->diagnostic["target"] = {target.translation().x(), target.translation().y(), target.translation().z()};
        if (failure_stage) *failure_stage = step <= lift_steps ? "cartesian_lift" : "cartesian_retreat";
        if (failure_reason) {
          *failure_reason = "step " + std::to_string(step) + "/" +
            std::to_string(retreat_steps) + " " + reason;
        }
        return false;
      }
      current = next->state;
      retreat_states->push_back(current);
    }
    return true;
  }

  RrtPlanResult planRrtConnect(
    const planning_scene::PlanningSceneConstPtr& base_scene,
    const moveit::core::RobotState& start_state,
    const moveit::core::RobotState& goal_state,
    bool plan_updown = false, bool local_patch = false,
    const moveit::core::JointModelGroup* motion_group = nullptr) const
  {
    RrtPlanResult result;
    const auto wall_started = std::chrono::steady_clock::now();
    auto scene = planning_scene::PlanningScene::clone(base_scene);
    scene->setCurrentState(start_state);
    const auto* search_group = motion_group ? motion_group : (plan_updown ?
      robot_model_->getJointModelGroup(side_ + "_arm_with_updown") : planning_group_);
    if (!search_group) throw std::runtime_error("arm/lift planning group unavailable");
    const std::string start_collision = alfa_robot::motion::scene_collision_reason(
      scene, start_state, distance_demo_ ? nullptr : planning_group_);
    if (!start_collision.empty()) {
      result.rejected_state = std::make_shared<moveit::core::RobotState>(start_state);
      result.reason = "start_" + start_collision;
      return result;
    }
    const std::string goal_collision = alfa_robot::motion::scene_collision_reason(
      scene, goal_state, distance_demo_ ? nullptr : planning_group_);
    if (!goal_collision.empty()) {
      result.rejected_state = std::make_shared<moveit::core::RobotState>(goal_state);
      result.reason = "goal_" + goal_collision;
      return result;
    }

    planning_interface::MotionPlanRequest request;
    request.group_name = search_group->getName();
    request.planner_id = local_patch ? "RRTConnectLocalPatchkConfigDefault" : "RRTConnectkConfigDefault";
    request.allowed_planning_time = local_patch ? local_rrt_planning_time_ : rrt_planning_time_;
    request.num_planning_attempts = rrt_planning_attempts_;
    request.max_velocity_scaling_factor = 1.0;
    request.max_acceleration_scaling_factor = 1.0;
    moveit::core::robotStateToRobotStateMsg(start_state, request.start_state, true);
    request.goal_constraints.push_back(
      kinematic_constraints::constructGoalConstraints(goal_state, search_group, 1e-3));

    planning_interface::MotionPlanResponse response;
    const bool generated = planning_pipeline_->generatePlan(scene, request, response);
    result.wall_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - wall_started).count();
    result.planner_ms = response.planning_time_ * 1000.0;
    if (!generated || response.error_code_.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS ||
        !response.trajectory_) {
      result.reason = "RRTConnect code=" + std::to_string(response.error_code_.val) + " " +
        alfa_robot::motion::direct_pipeline_failure_diagnostic(
          scene, start_state, goal_state, search_group);
      return result;
    }
    for (size_t index = 0; index < response.trajectory_->getWayPointCount(); ++index) {
      result.states.push_back(std::make_shared<moveit::core::RobotState>(
        response.trajectory_->getWayPoint(index)));
    }
    if (result.states.empty()) {
      result.reason = "RRTConnect returned empty trajectory";
      return result;
    }
    if (distance_demo_) {
      const bool attached = start_state.hasAttachedBody(kCarriedBoxId);
      for (const auto& state : result.states) {
        if (!state->satisfiesBounds() || state->hasAttachedBody(kCarriedBoxId) != attached) {
          result.rejected_state = state;
          result.reason = "rrt_invalid_bounds_or_attachment attached=" +
            std::to_string(state->hasAttachedBody(kCarriedBoxId)) + " expected=" + std::to_string(attached);
          for (const auto& name : all_joint_names_) {
            const auto& bounds = robot_model_->getVariableBounds(name);
            const double value = state->getVariablePosition(name);
            if (bounds.position_bounded_ && (value < bounds.min_position_ || value > bounds.max_position_))
              result.reason += " " + name + "=" + std::to_string(value);
          }
          return result;
        }
        for (const auto* payload_id : {kCarriedBoxLeftId, kCarriedBoxRightId}) {
          if (state->hasAttachedBody(payload_id) != start_state.hasAttachedBody(payload_id)) {
            result.rejected_state = state;
            result.reason = "rrt_changed_payload_attachment:" + std::string(payload_id);
            return result;
          }
        }
        for (const auto& name : all_joint_names_) {
          if (std::find(search_group->getVariableNames().begin(),
              search_group->getVariableNames().end(), name) == search_group->getVariableNames().end() &&
              std::abs(state->getVariablePosition(name) - start_state.getVariablePosition(name)) > 1e-8) {
            result.rejected_state = state;
            result.reason = "rrt_changed_fixed_joint";
            return result;
          }
        }
      }
      // Check and replay exact boundary bridges, not an adapter-repaired teleport.
      result.states.insert(result.states.begin(), std::make_shared<moveit::core::RobotState>(start_state));
      result.states.push_back(std::make_shared<moveit::core::RobotState>(goal_state));
      for (size_t i = 1; i < result.states.size(); ++i) {
        if (!edgeClear(scene, *result.states[i-1], *result.states[i],
                       start_state.hasAttachedBody(kCarriedBoxId), nullptr, &result.reason,
                       &result.rejected_state, plan_updown, motion_group)) {
          result.reason = "rrt_edge_" + result.reason;
          return result;
        }
      }
    }
    std::vector<std::array<double, 7>> natural_path;
    natural_path.reserve(result.states.size());
    for (const auto& state : result.states) natural_path.push_back(armJoints(*state));
    if (!(direct_attach_ && motion_group) &&
        !alfa_robot::motion::naturalJointPath(natural_path, 8.0, 3.0, local_patch)) {
      result.reason = "rrt_unnatural_branch_flip_or_detour";
      result.success = false;
      return result;
    }
    result.success = true;
    return result;
  }

  RrtPlanResult planRrt(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit::core::RobotState& start,
    const moveit::core::RobotState& goal,
    bool plan_updown = false, PlanningMetrics* metrics = nullptr,
    const moveit::core::JointModelGroup* motion_group = nullptr) const
  {
    if (connection_planner_ == "rrt_connect")
      return planRrtConnect(scene, start, goal, plan_updown, false, motion_group);
    const auto started = std::chrono::steady_clock::now();
    RrtPlanResult result;
    if (metrics) ++metrics->shortcut_connections;
    const auto finish = [&]() {
      result.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
      return result;
    };
    const auto* search_group = motion_group ? motion_group : (plan_updown ?
      robot_model_->getJointModelGroup(side_ + "_arm_with_updown") : planning_group_);
    if (!search_group) throw std::runtime_error("arm/lift planning group unavailable");
    for (const auto& name : all_joint_names_)
      if (std::find(search_group->getVariableNames().begin(), search_group->getVariableNames().end(),
          name) == search_group->getVariableNames().end() &&
          std::abs(goal.getVariablePosition(name) - start.getVariablePosition(name)) > 1e-8) {
        result.reason = "shortcut_changed_fixed_joint:" + name;
        result.rejected_state = std::make_shared<moveit::core::RobotState>(goal);
        return finish();
      }
    auto shortcut = directArmPath(start, goal, plan_updown, shortcut_step_,
      shortcut_updown_step_, motion_group);
    std::vector<bool> nodes_valid(shortcut.size(), true), edges_valid(shortcut.size() - 1, true);
    for (size_t index = 0; index < shortcut.size(); ++index) {
      const std::string collision = collisionReason(scene, *shortcut[index], metrics);
      nodes_valid[index] = shortcut[index]->satisfiesBounds() && collision.empty();
      if (!nodes_valid[index] && (index == 0 || index + 1 == shortcut.size())) {
        result.rejected_state = shortcut[index];
        result.reason = (index == 0 ? "shortcut_start_" : "shortcut_goal_") +
          (collision.empty() ? std::string("joint_bounds") : collision);
        return finish();
      }
    }
    for (size_t edge = 0; edge < edges_valid.size(); ++edge) {
      std::string reason;
      edges_valid[edge] = nodes_valid[edge] && nodes_valid[edge + 1] &&
        edgeClear(scene, *shortcut[edge], *shortcut[edge + 1],
          start.hasAttachedBody(kCarriedBoxId), metrics, &reason, nullptr,
          plan_updown, motion_group);
      if (!edges_valid[edge] && metrics) ++metrics->shortcut_blocked_edges;
    }
    const auto windows = alfa_robot::motion::shortcutRepairWindows(
      nodes_valid, edges_valid, shortcut_padding_points_);
    if (windows.empty() && metrics) ++metrics->shortcut_direct_successes;
    result.states.push_back(shortcut.front());
    size_t cursor = 0;
    const auto append_straight = [&](size_t end) {
      for (; cursor < end; ++cursor) {
        const auto states = directArmPath(*shortcut[cursor], *shortcut[cursor + 1],
          plan_updown, 0.0, 0.005, motion_group);
        result.states.insert(result.states.end(), states.begin() + 1, states.end());
      }
    };
    for (const auto& window : windows) {
      append_straight(window.begin);
      if (metrics) ++metrics->local_rrt_calls;
      const auto patch_started = std::chrono::steady_clock::now();
      const auto patch = planRrtConnect(scene, *shortcut[window.begin], *shortcut[window.end],
        plan_updown, true, motion_group);
      const double patch_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - patch_started).count();
      if (metrics) metrics->local_rrt_wall_ms += patch_ms;
      result.planner_ms += patch.planner_ms;
      if (!patch.success) {
        if (metrics) ++metrics->local_rrt_failures;
        result.reason = "shortcut_local_rrt[" + std::to_string(window.begin) + "," +
          std::to_string(window.end) + "]_" + patch.reason;
        result.rejected_state = patch.rejected_state;
        return finish();
      }
      result.states.insert(result.states.end(), patch.states.begin() + 1, patch.states.end());
      cursor = window.end;
    }
    append_straight(shortcut.size() - 1);
    std::vector<std::array<double, 7>> natural_path;
    for (const auto& state : result.states) natural_path.push_back(armJoints(*state));
    if (!(direct_attach_ && motion_group) &&
        !alfa_robot::motion::naturalJointPath(natural_path, 8.0, 3.0, true)) {
      result.reason = "shortcut_repaired_unnatural_branch_flip_or_detour";
      return finish();
    }
    result.success = true;
    return finish();
  }

  void appendStates(
    const std::vector<moveit::core::RobotStatePtr>& states,
    const std::string& stage,
    bool attached,
    bool skip_first,
    std::vector<ReplayFrame>* frames) const
  {
    if (!frames) return;
    for (size_t index = skip_first && !states.empty() ? 1U : 0U; index < states.size(); ++index) {
      ReplayFrame frame;
      frame.stage = stage;
      frame.joints = allJoints(*states[index]);
      frame.box_attached = attached;
      frames->push_back(std::move(frame));
    }
  }

  nlohmann::json heightAlignment(const Eigen::Vector3d& box_center) const
  {
    const double current_shoulder_z = distance_demo_ ?
      (initial_state_->getGlobalLinkTransform(arm_base_link_) * solver_->modelShoulderCenterInArmBase()).z() : initial_shoulder_z_;
    const double difference = current_shoulder_z - (box_center.z() + shoulder_box_offset_);
    const double descent = align_height_ ? std::max(0.0, difference) : 0.0;
    const double initial = initial_state_->getVariablePosition("updown");
    const auto& bounds = robot_model_->getVariableBounds("updown");
    if (distance_demo_ && top_suction_) {
      // The 23.8cm wrist-to-TCP extension points DOWN for top suction. A TCP-based
      // radius suitable for front suction incorrectly lowers the shoulder below the wrist.
      const auto shoulder = (initial_state_->getGlobalLinkTransform(arm_base_link_) *
        solver_->modelShoulderCenterInArmBase()).eval();
      const auto seed = armJoints(*initial_state_);
      const auto tcp = solver_->forwardInArmBase(seed);
      const Eigen::Vector3d tcp_to_wrist = tcp.linear().transpose() *
        (solver_->wristCenterInArmBase(seed) - tcp.translation());
      const Eigen::Vector3d wrist = contactPose(box_center) * tcp_to_wrist;
      const double lower = height_clearance_.value("lower", bounds.min_position_);
      const double upper = height_clearance_.value("upper", bounds.max_position_);
      const double wrist_aligned = initial + wrist.z() - shoulder.z();
      const double ideal = wrist_aligned + top_shoulder_above_wrist_;
      const double selected = !align_height_ ? initial :
        synchronized_updown_.value_or(std::clamp(ideal, lower, upper));
      const double xy = (wrist - shoulder).head<2>().norm();
      return {{"enabled", align_height_}, {"strategy", "top_wrist_alignment"}, {"arm", side_},
        {"reference", "independent top-suction shoulder-above-wrist height; vertical tool offset included"},
        {"shoulder_above_wrist", top_shoulder_above_wrist_},
        {"return_policy", "checked rear placement and release; retain final posture for next box"},
        {"target_world", {wrist.x(), wrist.y(), wrist.z()}},
        {"initial_updown", initial}, {"target_updown", selected}, {"ideal_updown", ideal},
        {"dual_shared_updown", synchronized_updown_.has_value()},
        {"descent", initial - selected}, {"lower_limit", bounds.min_position_},
        {"upper_limit", bounds.max_position_}, {"reachable_lift", height_clearance_},
        {"collision_sample_step_m", 0.005}, {"xy", xy}, {"arm_length", solver_->modelArmLength()},
        {"actual_ratio", std::hypot(xy, selected - wrist_aligned) / solver_->modelArmLength()},
        {"inside_band", std::hypot(xy, selected - wrist_aligned) <= solver_->modelArmLength()},
        {"outside_reason", xy > solver_->modelArmLength() ? "top_wrist_xy_beyond_arm_reach" :
          (std::hypot(xy, selected - wrist_aligned) > solver_->modelArmLength() ? "top_wrist_lift_clearance_limited" : "")}};
    }
    if (distance_demo_ && height_strategy_ == "comfort_radius") {
      const auto shoulder = (initial_state_->getGlobalLinkTransform(arm_base_link_) *
        solver_->modelShoulderCenterInArmBase()).eval();
      const Eigen::Vector3d target = contactPose(box_center).translation();
      const Eigen::Vector3d delta = target - shoulder;
      const double xy = delta.head<2>().norm();
      const double length = solver_->modelArmLength();
      auto choice = alfa_robot::motion::chooseComfortHeight(shoulder.z(), target.z(), xy,
        length, initial, height_clearance_.value("lower", bounds.min_position_),
        height_clearance_.value("upper", bounds.max_position_),
        comfort_min_, comfort_preferred_, comfort_max_, comfort_branch_);
      if (!align_height_) {
        choice.position = choice.ideal_position = initial;
        choice.ratio = delta.norm() / length;
        choice.projected = false;
      } else if (synchronized_updown_) {
        choice.position = *synchronized_updown_;
        const double dz = shoulder.z() + choice.position - initial - target.z();
        choice.ratio = std::hypot(xy, dz) / length;
        choice.projected = std::abs(choice.position - choice.ideal_position) > 1e-9;
      }
      const double final_dz = shoulder.z() + choice.position - initial - target.z();
      const bool inside = choice.ratio >= comfort_min_ - 1e-9 && choice.ratio <= comfort_max_ + 1e-9;
      return {{"enabled", align_height_}, {"strategy", height_strategy_}, {"arm", side_},
        {"reference", "selected arm shoulder common-axis center to contact TCP in world"},
        {"shoulder_world", {shoulder.x(), shoulder.y(), shoulder.z()}},
        {"target_world", {target.x(), target.y(), target.z()}},
        {"delta_world", {delta.x(), delta.y(), delta.z()}}, {"xy", xy},
        {"arm_length", length}, {"initial_ratio", delta.norm() / length},
        {"ratio_min", comfort_min_}, {"ratio_preferred", comfort_preferred_},
        {"ratio_max", comfort_max_}, {"actual_ratio", choice.ratio}, {"inside_band", inside},
        {"branch_policy", comfort_branch_}, {"branch", final_dz >= -1e-9 ? "above" : "below"},
        {"ideal_updown", choice.ideal_position}, {"projected", choice.projected},
        {"dual_shared_updown", synchronized_updown_.has_value()},
        {"outside_reason", inside ? "" : (!align_height_ ? "alignment_disabled" :
          (xy > comfort_max_ * length ? "xy_exceeds_band" :
            (height_clearance_.value("checked", false) ? "reachable_lift_limits" : "lift_limits")))},
        {"initial_shoulder_z", shoulder.z()}, {"box_center_z", box_center.z()},
        {"descent", initial - choice.position}, {"initial_updown", initial},
        {"target_updown", choice.position}, {"lower_limit", bounds.min_position_},
        {"upper_limit", bounds.max_position_}, {"reachable_lift", height_clearance_},
        {"collision_sample_step_m", 0.005},
        {"return_policy", post_extract_policy_}};
    }
    return {{"enabled", align_height_}, {"strategy", "fixed_offset"},
      {"reference", "midpoint of left/right shoulder common-axis centers in world Z"},
      {"initial_shoulder_z", current_shoulder_z}, {"box_center_z", box_center.z()},
      {"shoulder_box_offset", shoulder_box_offset_}, {"height_difference", difference},
      {"descent", descent}, {"initial_updown", initial}, {"target_updown", initial - descent},
      {"lower_limit", bounds.min_position_}, {"upper_limit", bounds.max_position_},
      {"collision_sample_step_m", 0.005}, {"return_policy", post_extract_policy_}};
  }

  void checkHeightClearance(
    const planning_scene::PlanningScenePtr& scene, const moveit::core::RobotState& lift_start,
    PlanningMetrics& metrics)
  {
    if (!distance_demo_ || !align_height_ || (!top_suction_ && height_strategy_ != "comfort_radius")) return;
    const double initial = initial_state_->getVariablePosition("updown");
    const auto& bounds = robot_model_->getVariableBounds("updown");
    height_clearance_ = {{"checked", true}, {"lower", initial}, {"upper", initial},
      {"blocked", nlohmann::json::array()}};
    // Conservatively sample the component reachable from home. Never skip an
    // obstacle to select a lower safe island. Same <=5mm sampling as replay validation.
    for (const auto& endpoint : {std::make_pair("lower", bounds.min_position_),
                                 std::make_pair("upper", bounds.max_position_)}) {
      moveit::core::RobotState state(lift_start);
      const size_t steps = static_cast<size_t>(std::ceil(std::abs(endpoint.second - initial) / 0.005));
      for (size_t step = 1; step <= steps; ++step) {
        const double q = initial + (endpoint.second - initial) * step / steps;
        state.setVariablePosition("updown", q);
        state.update(true);
        const auto collision = collisionReason(scene, state, &metrics);
        if (!state.satisfiesBounds() || !collision.empty()) {
          height_clearance_["blocked"].push_back({{"direction", endpoint.first},
            {"updown", q}, {"reason", collision.empty() ? "joint bounds violated" : collision}});
          break;
        }
        height_clearance_[endpoint.first] = q;
      }
    }
  }

  bool alignHeight(
    const Eigen::Vector3d& box_center, const planning_scene::PlanningScenePtr& scene,
    moveit::core::RobotState& grasp_start, TaskResult& result) const
  {
    if (!distance_demo_ || !align_height_) return true;
    const auto alignment = heightAlignment(box_center);
    const double target = alignment.at("target_updown").get<double>();
    const auto& bounds = robot_model_->getVariableBounds("updown");
    if (height_clearance_.value("checked", false) &&
        (target < height_clearance_.at("lower").get<double>() - 1e-9 ||
         target > height_clearance_.at("upper").get<double>() + 1e-9)) {
      result.failure_stage = "height_alignment_clearance";
      result.failure_reason = "shared updown=" + std::to_string(target) +
        " outside collision-free interval [" +
        std::to_string(height_clearance_.at("lower").get<double>()) + ", " +
        std::to_string(height_clearance_.at("upper").get<double>()) + "] metres";
      return false;
    }
    if (target < bounds.min_position_ || target > bounds.max_position_) {
      result.failure_stage = "height_alignment_limits";
      result.failure_reason = "required updown=" + std::to_string(target) +
        " outside [" + std::to_string(bounds.min_position_) + ", " +
        std::to_string(bounds.max_position_) + "] metres; not clamped";
      return false;
    }
    const double initial = grasp_start.getVariablePosition("updown");
    if (target == initial) return true;
    const std::string stage = height_strategy_ == "comfort_radius" ?
      "move_to_grasp_height" : "lower_to_box_height";
    std::vector<ReplayFrame> prefix{ReplayFrame{stage, allJoints(grasp_start), false}};
    const size_t steps = static_cast<size_t>(std::ceil(std::abs(target - initial) / 0.005));
    for (size_t step = 1; step <= steps; ++step) {
      grasp_start.setVariablePosition("updown", initial + (target - initial) * step / steps);
      grasp_start.update(true);
      const std::string collision = collisionReason(scene, grasp_start, &result.metrics);
      if (!grasp_start.satisfiesBounds() || !collision.empty()) {
        result.failure_stage = "height_alignment_collision";
        result.failure_reason = "updown=" +
          std::to_string(grasp_start.getVariablePosition("updown")) + ": " +
          (collision.empty() ? "joint bounds violated" : collision);
        result.diagnostic_frames = prefix;
        result.rejected_state = std::make_shared<moveit::core::RobotState>(grasp_start);
        result.diagnostic["snapshot"] = "first_rejected_lift_sample";
        // Execution still rejects the complete descent; diagnostics are display-only.
        return false;
      }
      prefix.push_back(ReplayFrame{stage, allJoints(grasp_start), false});
    }
    result.frames = std::move(prefix);
    return true;
  }

  TaskResult planTask(const Eigen::Vector3d& box_center)
  {
    height_clearance_ = {{"checked", false}};
    TaskResult result;
    const auto total_started = std::chrono::steady_clock::now();
    const auto scene = makeScene(box_center);
    const auto loaded = loadedScene(scene);
    auto finish = [&]() {
      if (!result.success) {
        if (result.diagnostic_frames.empty()) result.diagnostic_frames = result.frames;
        if (result.diagnostic_frames.empty())
          result.diagnostic_frames.push_back({"initial_state", allJoints(*initial_state_), false});
        result.diagnostic["diagnostic_only"] = true;
        result.diagnostic["stage"] = result.failure_stage;
        result.diagnostic["reason"] = result.failure_reason;
        result.diagnostic["freeze_at_end"] = true;
        result.diagnostic["contacts"] = nlohmann::json::array();
        if (result.rejected_state) {
          const moveit::core::RobotState state(*result.rejected_state);
          // A search/goal snapshot is not a continuation of the robot motion. In particular,
          // never attach a box at an unvisited home pose just to show a rejected return goal.
          const bool connected = result.failure_stage == "initial_state" ||
            result.failure_stage == "height_alignment_collision" ||
            result.failure_stage == "cartesian_approach" || result.failure_stage == "cartesian_retreat" ||
            result.failure_stage == "cartesian_lift" ||
            result.failure_stage == "updown_return" ||
            result.failure_stage == "attach_box";
          result.diagnostic["rejected_joints"] = allJoints(state);
          result.diagnostic["rejected_box_attached"] = state.hasAttachedBody(kCarriedBoxId);
          if (connected) {
            moveit::core::RobotState previous(*initial_state_);
            const auto& frame = result.diagnostic_frames.back();
            for (size_t i = 0; i < all_joint_names_.size(); ++i)
              previous.setVariablePosition(all_joint_names_[i], frame.joints[i]);
            previous.update(true);
            moveit::core::RobotStatePtr first_rejected;
            std::string ignored;
            if (result.failure_stage != "height_alignment_collision" &&
                result.failure_stage != "updown_return") edgeClear(state.hasAttachedBody(kCarriedBoxId) ? loaded : scene, previous, state,
              state.hasAttachedBody(kCarriedBoxId), nullptr, &ignored, &first_rejected);
            if (first_rejected) result.rejected_state = first_rejected;
            result.diagnostic_frames.push_back({"FAILED_SAMPLE: " + result.failure_stage,
              allJoints(*result.rejected_state), result.rejected_state->hasAttachedBody(kCarriedBoxId)});
            result.diagnostic["snapshot"] = result.failure_stage == "height_alignment_collision" ?
              "first_rejected_lift_sample" : "first_rejected_path_sample";
          } else {
            result.diagnostic_frames.back().stage = "FAILED_HOLD: " + result.failure_stage;
            result.diagnostic["snapshot"] = "unconnected_rejected_state_not_replayed";
            const auto target = state.getGlobalLinkTransform(tool_link_).translation().eval();
            if (!result.diagnostic.contains("target"))
              result.diagnostic["target"] = {target.x(), target.y(), target.z()};
          }
          collision_detection::CollisionRequest request;
          collision_detection::CollisionResult contacts;
          request.contacts = true;
          request.max_contacts = 20;
          request.max_contacts_per_pair = 1;
          if (!distance_demo_) request.group_name = planning_group_name_;
          (result.rejected_state->hasAttachedBody(kCarriedBoxId) ? loaded : scene)->checkCollision(request, contacts, *result.rejected_state);
          for (const auto& pair : contacts.contacts)
            for (const auto& contact : pair.second)
              result.diagnostic[connected ? "contacts" : "rejected_contacts"].push_back({{"bodies", {pair.first.first, pair.first.second}},
                {"position", {contact.pos.x(), contact.pos.y(), contact.pos.z()}}});
        } else {
          result.diagnostic_frames.back().stage = "FAILED_HOLD: " + result.failure_stage;
          result.diagnostic["snapshot"] = "last_available_state_no_rejected_configuration";
        }
      }
      result.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_started).count();
      return result;
    };
    const std::string initial_collision = collisionReason(scene, *initial_state_, &result.metrics);
    if (!initial_state_->satisfiesBounds() || !initial_collision.empty()) {
      result.rejected_state = std::make_shared<moveit::core::RobotState>(*initial_state_);
      result.failure_stage = "initial_state";
      result.failure_reason = initial_collision.empty() ? "joint bounds violated" : initial_collision;
      result.frames.push_back(ReplayFrame{"initial_state", allJoints(*initial_state_), false});
      return finish();
    }

    moveit::core::RobotState grasp_start(*initial_state_);
    std::vector<ReplayFrame> preparation;
    if (distance_demo_ && top_suction_) {
      const auto geometry = heightAlignment(box_center);
      if (geometry.at("xy").get<double>() > geometry.at("arm_length").get<double>()) {
        result.failure_stage = "precontact_ik";
        result.failure_reason = "top-suction wrist horizontal distance=" +
          std::to_string(geometry.at("xy").get<double>()) + "m exceeds arm length=" +
          std::to_string(geometry.at("arm_length").get<double>()) + "m; changing lift cannot fix XY reach";
        const auto target = precontactPose(box_center).translation().eval();
        result.diagnostic["target"] = {target.x(), target.y(), target.z()};
        return finish();
      }
      checkHeightClearance(scene, grasp_start, result.metrics);
      const auto height = heightAlignment(box_center);
      const auto& lift_bounds = robot_model_->getVariableBounds("updown");
      const double ideal = std::clamp(height.at("ideal_updown").get<double>(),
        lift_bounds.min_position_, lift_bounds.max_position_);
      // Keep an already retracted rear-release posture when it can reach the
      // working height. Do not force it back through a now-obstructed home fold.
      if (std::abs(height.at("target_updown").get<double>() - ideal) > 1e-6) {
        // Retract BOTH empty arms before lowering. Descending in forward-facing home
        // otherwise hits the lower rows with the idle suction plate and blocks valid IK heights.
        const std::string selected_side = side_;
        preparation.push_back({"initial_state", allJoints(grasp_start), false});
        for (const std::string side : {"left", "right"}) {
          selectArm(side);
          moveit::core::RobotState folded(grasp_start);
          folded.setJointGroupPositions(planning_group_, armJoints(*home_state_).data());
          folded.setVariablePosition(side + "_joint4", foldedElbowPosition(side));
          folded.update(true);
          const auto path = planRrt(scene, grasp_start, folded, false, &result.metrics);
          result.metrics.rrt_approach_ms += path.wall_ms;
          if (!path.success) {
            result.failure_stage = "prepare_top_suction";
            result.failure_reason = side + ": " + path.reason;
            result.rejected_state = path.rejected_state;
            result.frames = preparation;
            selectArm(selected_side);
            return finish();
          }
          appendStates(path.states, "fold_arms_before_lift", false, true, &preparation);
          grasp_start = folded;
        }
        selectArm(selected_side);
      }
    }
    checkHeightClearance(scene, grasp_start, result.metrics);
    if (height_clearance_.value("checked", false)) {
      const auto alignment = heightAlignment(box_center);
      RCLCPP_INFO(get_logger(),
        "height selected: arm=%s updown=%.6fm reachable=[%.6f, %.6f]m ratio=%.6f inside_band=%s",
        side_.c_str(), alignment.at("target_updown").get<double>(),
        height_clearance_.at("lower").get<double>(), height_clearance_.at("upper").get<double>(),
        alignment.at("actual_ratio").get<double>(), alignment.at("inside_band").get<bool>() ? "true" : "false");
    }
    if (!alignHeight(box_center, scene, grasp_start, result)) {
      result.diagnostic_frames.insert(result.diagnostic_frames.begin(), preparation.begin(), preparation.end());
      result.frames = {ReplayFrame{"initial_state", allJoints(*initial_state_), false}};
      return finish();
    }
    result.frames.insert(result.frames.begin(), preparation.begin(), preparation.end());
    const auto lift_prefix = result.frames;
    std::string precontact_rejections;
    auto precontact_candidates = solvePoseCandidates(
      precontactPose(box_center), grasp_start, false, scene, &result.metrics,
      false, &precontact_rejections, &result.rejected_state);
    if (precontact_candidates.empty()) {
      const auto target = precontactPose(box_center).translation().eval();
      result.diagnostic["target"] = {target.x(), target.y(), target.z()};
      result.failure_stage = "precontact_ik";
      result.failure_reason = precontact_rejections;
      if (top_suction_) {
        const auto geometry = heightAlignment(box_center);
        if (geometry.at("xy").get<double>() > geometry.at("arm_length").get<double>())
          result.failure_reason += " top-suction wrist horizontal distance=" +
            std::to_string(geometry.at("xy").get<double>()) + "m exceeds arm length=" +
            std::to_string(geometry.at("arm_length").get<double>()) + "m; changing lift cannot fix XY reach";
      }
      if (result.frames.empty())
        result.frames.push_back(ReplayFrame{"initial_state", allJoints(grasp_start), false});
      return finish();
    }
    if (precontact_candidates.size() > precontact_candidate_limit_) {
      precontact_candidates.resize(precontact_candidate_limit_);
    }

    std::string last_failure_stage = "candidate_search";
    std::string last_failure_reason = "no candidate attempted";
    std::vector<ReplayFrame> best_partial = lift_prefix;
    for (size_t candidate_index = 0; candidate_index < precontact_candidates.size(); ++candidate_index) {
      result.rejected_state.reset();
      result.diagnostic = nlohmann::json::object();
      result.diagnostic_frames.clear();
      best_partial = lift_prefix;
      std::vector<ReplayFrame> executable_prefix = lift_prefix;
      std::vector<moveit::core::RobotStatePtr> retreat_states;
      std::string analytic_failure_stage;
      std::string analytic_failure_reason;
      bool analytic_ok = true;
        const auto approach_rrt = planRrt(
          scene, grasp_start, *precontact_candidates[candidate_index].state, false, &result.metrics);
        result.metrics.rrt_approach_ms += approach_rrt.wall_ms;
        if (!approach_rrt.success) {
          result.rejected_state = approach_rrt.rejected_state;
          const auto target = precontactPose(box_center).translation().eval();
          result.diagnostic["target"] = {target.x(), target.y(), target.z()};
          last_failure_stage = "rrt_to_precontact";
          last_failure_reason = "candidate " + std::to_string(candidate_index) + " " +
            approach_rrt.reason;
          continue;
        }
        appendStates(approach_rrt.states, "rrt_to_precontact", false, false, &executable_prefix);
        const auto analytic_started = std::chrono::steady_clock::now();
        std::vector<moveit::core::RobotStatePtr> approach_states;
        const bool traced = traceCartesianPath(
          box_center, *precontact_candidates[candidate_index].state, scene,
          &result.metrics, &approach_states, &retreat_states,
          &analytic_failure_stage, &analytic_failure_reason, &result);
        analytic_ok = traced;
        result.metrics.analytic_path_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - analytic_started).count();
        appendStates(approach_states, "cartesian_approach", false, true, &executable_prefix);
        if (!retreat_states.empty()) {
          executable_prefix.push_back(
            ReplayFrame{"attach_box", allJoints(*retreat_states.front()), true});
        }
        const size_t retreat_begin = executable_prefix.size();
        appendStates(retreat_states, "cartesian_retreat", true, true, &executable_prefix);
        if (top_suction_) {
          const size_t lift_frames = std::max<size_t>(1, std::ceil(approach_distance_ / cartesian_step_));
          for (size_t i = retreat_begin; i < std::min(executable_prefix.size(), retreat_begin + lift_frames); ++i)
            executable_prefix[i].stage = "cartesian_lift";
        }
      best_partial = executable_prefix;
      if (!analytic_ok) {
        last_failure_stage = analytic_failure_stage;
        last_failure_reason = "candidate " + std::to_string(candidate_index) + " " + analytic_failure_reason;
        continue;
      }

      last_failure_stage.clear();
      moveit::core::RobotState return_goal(grasp_start);
      if (post_extract_policy_ == "loaded_home") {
        const auto initial_arm = armJoints(*initial_state_);
        return_goal.setJointGroupPositions(planning_group_, initial_arm.data());
        return_goal.setVariablePosition("updown", initial_state_->getVariablePosition("updown"));
      }
      attachCarriedBox(return_goal);
      if (distance_demo_ && post_extract_policy_ == "rear_release") {
        if (rear_placement_strategy_ == "named_unloading") {
          moveit::core::RobotState unloading(*home_state_);
          if (!unloading.setToDefaultValues(
              robot_model_->getJointModelGroup("whole_body"), "unloading")) {
            last_failure_stage = "rear_placement";
            last_failure_reason = "whole_body/unloading named pose is unavailable";
            continue;
          }
          const auto unloading_joints = armJoints(unloading);
          return_goal.setJointGroupPositions(planning_group_, unloading_joints.data());
          return_goal.update(true);
          const std::string unloading_collision = collisionReason(loaded, return_goal, &result.metrics);
          if (!return_goal.satisfiesBounds(planning_group_) || !unloading_collision.empty()) {
            result.rejected_state = std::make_shared<moveit::core::RobotState>(return_goal);
            last_failure_stage = "rear_placement";
            last_failure_reason = unloading_collision.empty() ?
              "named unloading pose violates joint bounds" : unloading_collision;
            continue;
          }
        } else {
          // Position the entire rotated payload behind the chassis, not just the TCP.
          // A folded top-suction payload can extend forward of its TCP.
          moveit::core::RobotState rear_reference(*home_state_);
          rear_reference.setVariablePosition("updown", grasp_start.getVariablePosition("updown"));
          if (top_suction_)
            rear_reference.setVariablePosition(side_ + "_joint4", foldedElbowPosition(side_));
          rear_reference.update(true);
          const auto rear = alfa_robot::motion::wallRearPlacementPose(
            rear_reference.getGlobalLinkTransform(tool_link_), toolToBox(), boxSize(),
            chassis_rear_x_, rear_clearance_, wall_bottom_z_ + contact_numerical_gap_);
          std::string reason;
          auto candidates = solvePoseCandidates(
            rear, grasp_start, true, loaded, &result.metrics, false, &reason,
            &result.rejected_state);
          result.diagnostic["target"] = {
            rear.translation().x(), rear.translation().y(), rear.translation().z()};
          bool found = false;
          for (size_t i = 0; i < std::min(candidates.size(), precontact_candidate_limit_); ++i) {
            const auto& goal = *candidates[i].state;
            if (!alfa_robot::motion::boxBehindChassis(
                goal.getGlobalLinkTransform(tool_link_) * toolToBox(), boxSize(), chassis_rear_x_)) {
              reason = "rear target does not put the entire box at least 1cm behind chassis";
              continue;
            }
            return_goal = goal;
            found = true;
            break;
          }
          if (!found) {
            last_failure_stage = "rear_placement";
            last_failure_reason = reason;
            continue;
          }
        }
      } else {
        const auto reason = collisionReason(loaded, return_goal, &result.metrics);
        if (!reason.empty()) {
          result.rejected_state = std::make_shared<moveit::core::RobotState>(return_goal);
          last_failure_stage = "return_goal";
          last_failure_reason = reason;
          continue;
        }
      }

      RrtPlanResult return_plan;
      std::string direct_reason;
      const bool return_with_updown = post_extract_policy_ == "loaded_home";
      if (connection_planner_ == "shortcut_local_rrt") {
        return_plan = planRrt(loaded, *retreat_states.back(), return_goal,
          return_with_updown, &result.metrics);
        result.metrics.rrt_return_ms += return_plan.wall_ms;
      } else if (edgeClear(loaded, *retreat_states.back(), return_goal, true, &result.metrics,
          &direct_reason, &return_plan.rejected_state, return_with_updown) &&
          alfa_robot::motion::sameShoulderElbowBranch(
            armJoints(*retreat_states.back()), armJoints(return_goal))) {
        return_plan.success = true;
        return_plan.states = directArmPath(*retreat_states.back(), return_goal, return_with_updown);
      } else {
        return_plan = planRrt(loaded, *retreat_states.back(), return_goal, return_with_updown);
        result.metrics.rrt_return_ms += return_plan.wall_ms;
      }
      if (!return_plan.success) {
        result.rejected_state = return_plan.rejected_state;
        const auto target = return_goal.getGlobalLinkTransform(tool_link_).translation().eval();
        result.diagnostic["target"] = {target.x(), target.y(), target.z()};
        last_failure_stage = "rrt_return";
        last_failure_reason = "candidate " + std::to_string(candidate_index) + " " +
          (return_plan.reason.empty() ? direct_reason : return_plan.reason);
        continue;
      }

      result.frames = std::move(executable_prefix);
      appendStates(return_plan.states, "rrt_return", true, true, &result.frames);
      if (post_extract_policy_ == "loaded_home") {
        auto loaded_return = return_plan.states.back();
        best_partial = result.frames;
        if (!synchronized_updown_) {
          const std::string carrying_side = side_;
          selectArm(carrying_side == "left" ? "right" : "left");
          moveit::core::RobotState idle_goal(*loaded_return);
          const auto initial_idle = armJoints(*initial_state_);
          idle_goal.setJointGroupPositions(planning_group_, initial_idle.data());
          idle_goal.update(true);
          RrtPlanResult idle_return;
          std::string idle_reason;
          if (connection_planner_ == "shortcut_local_rrt") {
            idle_return = planRrt(loaded, *loaded_return, idle_goal, false, &result.metrics);
            result.metrics.rrt_return_ms += idle_return.wall_ms;
          } else if (edgeClear(loaded, *loaded_return, idle_goal, true, &result.metrics,
              &idle_reason, &idle_return.rejected_state)) {
            idle_return.success = true;
            idle_return.states = directArmPath(*loaded_return, idle_goal);
          } else {
            idle_return = planRrt(loaded, *loaded_return, idle_goal);
            result.metrics.rrt_return_ms += idle_return.wall_ms;
          }
          selectArm(carrying_side);
          if (!idle_return.success) {
            result.rejected_state = idle_return.rejected_state;
            last_failure_stage = "loaded_home_idle_return";
            last_failure_reason = idle_return.reason.empty() ? idle_reason : idle_return.reason;
            continue;
          }
          appendStates(idle_return.states, "loaded_home_idle_return", true, true, &result.frames);
          loaded_return = idle_return.states.back();
          best_partial = result.frames;
        }
        const auto updown_return = moveUpdown(
          loaded, *loaded_return,
          initial_state_->getVariablePosition("updown"), true, &result.metrics);
        if (!updown_return.success) {
          appendStates(updown_return.states, "updown_return", true, true, &result.frames);
          best_partial = result.frames;
          result.rejected_state = updown_return.rejected_state;
          last_failure_stage = "updown_return";
          last_failure_reason = updown_return.reason;
          continue;
        }
        appendStates(updown_return.states, "updown_return", true, true, &result.frames);
        result.frames.push_back(
          ReplayFrame{"loaded_home", allJoints(*updown_return.states.back()), true});
      } else if (distance_demo_) {
        // The loaded path is checked through this last visible pose. Removal is
        // a separate frame with identical joints, never an obstacle workaround.
        result.frames.push_back(ReplayFrame{"rear_placement", allJoints(return_goal), true});
        result.frames.push_back(ReplayFrame{"release_box", allJoints(return_goal), false, false});

        if (enable_stage_action_) {
          auto released_scene = planning_scene::PlanningScene::clone(loaded);
          addPlacedBox(released_scene, wall_target_row_ * 5 + wall_target_column_,
            return_goal.getGlobalLinkTransform(tool_link_) * toolToBox(), side_);
          moveit::core::RobotState released(return_goal);
          released.clearAttachedBody(kCarriedBoxId);
          released.update(true);
          moveit::core::RobotState arm_home(released);
          const auto home_joints = armJoints(*home_state_);
          arm_home.setJointGroupPositions(planning_group_, home_joints.data());
          arm_home.update(true);

          RrtPlanResult home_plan;
          std::string home_direct_reason;
          if (connection_planner_ == "shortcut_local_rrt") {
            home_plan = planRrt(released_scene, released, arm_home, false, &result.metrics);
            result.metrics.rrt_return_ms += home_plan.wall_ms;
          } else if (edgeClear(released_scene, released, arm_home, false, &result.metrics,
              &home_direct_reason, &home_plan.rejected_state) &&
              alfa_robot::motion::sameShoulderElbowBranch(
                armJoints(released), armJoints(arm_home))) {
            home_plan.success = true;
            home_plan.states = directArmPath(released, arm_home);
          } else {
            home_plan = planRrt(released_scene, released, arm_home);
            result.metrics.rrt_return_ms += home_plan.wall_ms;
          }
          if (!home_plan.success) {
            result.rejected_state = home_plan.rejected_state;
            last_failure_stage = "home_return";
            last_failure_reason = home_plan.reason.empty() ? home_direct_reason : home_plan.reason;
            continue;
          }
          const size_t home_begin = result.frames.size();
          appendStates(home_plan.states, "home_return", false, true, &result.frames);
          for (size_t index = home_begin; index < result.frames.size(); ++index)
            result.frames[index].box_visible = false;

          const auto home_updown = moveUpdown(
            released_scene, *home_plan.states.back(), home_state_->getVariablePosition("updown"),
            false, &result.metrics);
          if (!home_updown.success) {
            result.rejected_state = home_updown.rejected_state;
            last_failure_stage = "home_updown";
            last_failure_reason = home_updown.reason;
            continue;
          }
          const size_t updown_begin = result.frames.size();
          appendStates(home_updown.states, "home_updown", false, true, &result.frames);
          for (size_t index = updown_begin; index < result.frames.size(); ++index)
            result.frames[index].box_visible = false;
        }
      }
      result.success = true;
      return finish();
    }

    result.failure_stage = last_failure_stage;
    result.failure_reason = last_failure_reason;
    result.frames = std::move(best_partial);
    if (result.frames.empty()) {
      result.frames.push_back(ReplayFrame{"initial_state", allJoints(grasp_start), false});
    }
    return finish();
  }

  nlohmann::json sceneJson(const Eigen::Vector3d& box_center) const
  {
    const auto neighbors = neighborCenters(box_center);
    nlohmann::json output;
    output["box_center"] = {box_center.x(), box_center.y(), box_center.z()};
    output["box_size"] = {box_depth_, box_width_, box_height_};
    output["collision_inset"] = collision_inset_;
    output["world_frame"] = world_frame_;
    output["joint_names"] = all_joint_names_;
    output["initial_joints"] = allJoints(*initial_state_);
    if (distance_demo_) {
      output["distance_demo"] = true;
      output["post_extract_policy"] = post_extract_policy_;
      output["connection_planner"] = connection_planner_;
      output["shortcut_padding_points"] = shortcut_padding_points_;
      output["shortcut_step_deg"] = radToDeg(shortcut_step_);
      output["shortcut_updown_step_m"] = shortcut_updown_step_;
      output["local_rrt_planning_time_s"] = local_rrt_planning_time_;
      output["bounded_joint_distance"] = connection_planner_ == "shortcut_local_rrt";
      output["rear_placement_strategy"] = direct_attach_ ? "named_unloading" : rear_placement_strategy_;
      output["release_after_transfer"] = !direct_attach_ && post_extract_policy_ == "rear_release";
      output["initial_pose"] = initial_pose_;
      output["direct_attach"] = direct_attach_;
      output["planning_seed"] = planning_seed_;
      output["edge_collision_resolution_deg"] = edge_joint_resolution_ * 180.0 / kPi;
      output["lift_after_attach_m"] = top_suction_ ? approach_distance_ : 0.0;
      output["lift_backoff_m"] = top_suction_ ? std::min(0.02, retreat_distance_) : 0.0;
      output["environment"] = environment_json_;
      output["height_alignment"] = direct_attach_ ?
        nlohmann::json{{"enabled", false}, {"strategy", "direct_attach"}} : heightAlignment(box_center);
      output["contact_numerical_gap"] = contact_numerical_gap_;
      output["requested_arm"] = requested_arm_;
      output["requested_suction_mode"] = requested_suction_mode_;
      output["wall_center_y"] = wall_center_y_;
      output["wall_bottom_z"] = wall_bottom_z_;
      output["scope"] = "simulation geometric extraction and loaded return; not hardware execution";
      output["x"] = wall_distance_;
      output["chassis_front_x"] = chassis_front_x_;
      output["box_id"] = wall_target_row_ * 5 + wall_target_column_;
      output["distance_reference"] = "world +X chassis front plane to wall near face";
    }
    output["tool_link"] = tool_link_;
    output["scene_layout"] = scene_layout_;
    output["wall_context"] = sequence_running_ ? "live_sequence" : wall_context_;
    if (scene_layout_ == "wall_5x5") {
      output["wall"] = {{"rows", 5}, {"columns", 5}, {"gap", wall_gap_},
        {"target_row", wall_target_row_}, {"target_column", wall_target_column_}};
    }
    output["neighbor_centers"] = nlohmann::json::array();
    for (const auto& center : neighbors) {
      output["neighbor_centers"].push_back({center.x(), center.y(), center.z()});
    }
    if (!direct_attach_) {
      const auto precontact = precontactPose(box_center);
      const auto contact = contactPose(box_center);
      const auto retreat = retreatPose(box_center);
      output["precontact"] = {
        precontact.translation().x(), precontact.translation().y(), precontact.translation().z()};
      output["contact"] = {
        contact.translation().x(), contact.translation().y(), contact.translation().z()};
      output["retreat"] = {
        retreat.translation().x(), retreat.translation().y(), retreat.translation().z()};
    } else {
      output["direct_attached_boxes"] = directAttachedBoxes(*initial_state_);
      output["direct_placement_pose"] = direct_placement_pose_;
      output["side"] = "dual";
      output["distance_reference"] = "synthetic boxes at current tools; no fixed-wall grasp target";
      output["scope"] = "simulation-only dual-arm direct attach to SRDF dual_arm/" + direct_placement_pose_;
      const auto unloading = directPlacementGoal(*initial_state_);
      output["goal_joints"] = allJoints(unloading);
    }
    const auto offset = toolToBoxCenter();
    output["tool_to_box_center"] = {offset.x(), offset.y(), offset.z()};
    output["tool_to_box_rotation"] = nlohmann::json::array();
    for (int row = 0; row < 3; ++row)
      output["tool_to_box_rotation"].push_back({toolToBox().linear()(row, 0),
        toolToBox().linear()(row, 1), toolToBox().linear()(row, 2)});
    output["suction_mode"] = top_suction_ ? "top" : "front";
    output["removed_box_ids"] = removed_boxes_;
    output["chassis_rear_x"] = chassis_rear_x_;
    output["rear_clearance"] = rear_clearance_;
    return output;
  }

  void publishJson(nlohmann::json& payload, bool segment = false)
  {
    payload["publisher_id"] = publisher_id_;
    if (payload.contains("generation"))
      payload["task_id"] = publisher_id_ + ":" + std::to_string(payload["generation"].get<uint64_t>());
    std_msgs::msg::String message;
    message.data = payload.dump();
    (segment ? segment_publisher_ : task_publisher_)->publish(message);
  }

  void publishPreview(const std::string& status)
  {
    Eigen::Vector3d center;
    {
      std::lock_guard<std::mutex> lock(box_mutex_);
      center = box_center_;
    }
    nlohmann::json payload = sceneJson(center);
    payload["kind"] = "preview";
    payload["side"] = direct_attach_ ? "dual" : side_;
    payload["status"] = status;
    publishJson(payload);
  }

  void publishPlanningStarted(uint64_t generation, const Eigen::Vector3d& box_center)
  {
    nlohmann::json payload = sceneJson(box_center);
    payload["kind"] = "planning";
    payload["generation"] = generation;
    payload["side"] = direct_attach_ ? "dual" : side_;
    payload["status"] = "计算开始";
    publishJson(payload);
  }

  void publishTaskResult(
    uint64_t generation,
    const Eigen::Vector3d& box_center,
    const TaskResult& result, bool publish = true)
  {
    last_result_ = sceneJson(box_center);
    auto& payload = last_result_;
    payload["kind"] = "result";
    payload["generation"] = generation;
    payload["side"] = direct_attach_ ? "dual" : side_;
    payload["tool_link"] = tool_link_;
    payload["success"] = result.success;
    payload["failure_stage"] = result.failure_stage;
    payload["failure_reason"] = result.failure_reason;
    payload["total_ms"] = result.total_ms;
    payload["metrics"] = {
      {"ik_calls", result.metrics.ik_calls},
      {"ik_ms", result.metrics.ik_ms},
      {"collision_checks", result.metrics.collision_checks},
      {"collision_ms", result.metrics.collision_ms},
      {"analytic_path_ms", result.metrics.analytic_path_ms},
      {"rrt_approach_ms", result.metrics.rrt_approach_ms},
      {"rrt_return_ms", result.metrics.rrt_return_ms},
      {"shortcut_connections", result.metrics.shortcut_connections},
      {"shortcut_direct_successes", result.metrics.shortcut_direct_successes},
      {"shortcut_blocked_edges", result.metrics.shortcut_blocked_edges},
      {"local_rrt_calls", result.metrics.local_rrt_calls},
      {"local_rrt_failures", result.metrics.local_rrt_failures},
      {"local_rrt_wall_ms", result.metrics.local_rrt_wall_ms},
    };
    if (!result.metrics.collision_sample_ms.empty()) {
      auto samples = result.metrics.collision_sample_ms;
      std::sort(samples.begin(), samples.end());
      payload["metrics"]["collision_latency_ms"] = {
        {"samples", samples.size()}, {"min", samples.front()},
        {"median", samples[samples.size() / 2]},
        {"p95", samples[static_cast<size_t>(0.95 * (samples.size() - 1))]},
        {"max", samples.back()},
      };
    }
    payload["joint_names"] = all_joint_names_;
    payload["attempts"] = attempts_;
    payload["verdict"] = result.success ? "path_found" : "no_path_found";
    payload["frames"] = nlohmann::json::array();
    payload["frames"].get_ref<nlohmann::json::array_t&>().reserve(result.frames.size());
    for (const auto& frame : result.frames) {
      payload["frames"].push_back({
        {"stage", frame.stage},
        {"joints", frame.joints},
        {"box_attached", frame.box_attached},
        {"box_visible", frame.box_visible},
        {"scene_index", frame.scene_index},
        {"carried_boxes", frame.carried_boxes},
      });
    }
    payload["diagnostic"] = result.diagnostic;
    payload["diagnostic_frames"] = nlohmann::json::array();
    payload["diagnostic_frames"].get_ref<nlohmann::json::array_t&>().reserve(
      result.diagnostic_frames.size());
    for (const auto& frame : result.diagnostic_frames)
      payload["diagnostic_frames"].push_back({{"stage", frame.stage}, {"joints", frame.joints},
        {"box_attached", frame.box_attached}, {"box_visible", frame.box_visible},
        {"scene_index", frame.scene_index}, {"carried_boxes", frame.carried_boxes},
        {"diagnostic_only", true}});
    if (publish) publishJson(payload);
  }

  void publishStatus(const std::string& status, bool good)
  {
    visualization_msgs::msg::MarkerArray markers;
    Marker text;
    text.header.frame_id = world_frame_;
    text.header.stamp = now();
    text.ns = "v3_single_arm_box_extract_status";
    text.id = 0;
    text.type = Marker::TEXT_VIEW_FACING;
    text.action = Marker::ADD;
    {
      std::lock_guard<std::mutex> lock(box_mutex_);
      text.pose.position.x = box_center_.x();
      text.pose.position.y = box_center_.y();
      text.pose.position.z = box_center_.z() + box_height_ * 0.75;
    }
    text.pose.orientation.w = 1.0;
    text.scale.z = 0.035;
    text.color = good ? color(0.15F, 1.0F, 0.25F) : color(1.0F, 0.12F, 0.12F);
    text.text = status;
    if (distance_demo_) text.text = "box=" + std::to_string(wall_target_row_ * 5 + wall_target_column_) +
      " x=" + std::to_string(wall_distance_) + "m arm=" + side_ + "\n" + status;
    if (distance_demo_ && height_strategy_ == "comfort_radius") {
      Eigen::Vector3d center;
      { std::lock_guard<std::mutex> lock(box_mutex_); center = box_center_; }
      const auto h = heightAlignment(center);
      std::ostringstream geometry;
      geometry << std::fixed << std::setprecision(3) << "\nxy=" << h.at("xy").get<double>()
        << " rho=" << h.at("actual_ratio").get<double>() << " band=[" << comfort_min_
        << "," << comfort_max_ << "] lift=" << h.at("target_updown").get<double>()
        << " " << h.value("branch", "wrist_aligned") << " " << h.at("outside_reason").get<std::string>()
        << (height_clearance_.value("checked", false) ? " lift_checked" : " proposal_unchecked");
      text.text += geometry.str();
    }
    markers.markers.push_back(text);
    status_marker_publisher_->publish(markers);
  }

  void publishSceneMarkers()
  {
    Eigen::Vector3d center;
    {
      std::lock_guard<std::mutex> lock(box_mutex_);
      center = box_center_;
    }
    const auto snapshot = display_scene_.is_object() ? display_scene_ : sceneJson(center);
    if (distance_demo_) {
      const auto& p = snapshot.at("box_center");
      center = Eigen::Vector3d(p[0].get<double>(), p[1].get<double>(), p[2].get<double>());
    }
    visualization_msgs::msg::MarkerArray markers;
    if (distance_demo_) {
      // A restarted planner may publish fewer obstacles; do not retain old RViz markers.
      Marker clear;
      clear.action = Marker::DELETEALL;
      markers.markers.push_back(clear);
    }
    for (size_t index = 0; index < environment_objects_.size(); ++index) {
      const auto& object = environment_objects_[index];
      Marker obstacle;
      obstacle.header.frame_id = world_frame_;
      obstacle.header.stamp = now();
      obstacle.ns = "environment";
      obstacle.id = static_cast<int>(index);
      obstacle.type = Marker::CUBE;
      obstacle.action = Marker::ADD;
      obstacle.pose = object.primitive_poses.front();
      const auto& size = object.primitives.front().dimensions;
      obstacle.scale.x = size[0];
      obstacle.scale.y = size[1];
      obstacle.scale.z = size[2];
      obstacle.color = color(0.45F, 0.55F, 0.65F, 0.20F);
      markers.markers.push_back(obstacle);
    }
    if (direct_attach_) {
      int marker_id = 0;
      for (const std::string side : {"left", "right"}) {
        Marker payload;
        payload.header.frame_id = world_frame_;
        payload.header.stamp = now();
        payload.ns = "direct_attached_boxes";
        payload.id = marker_id++;
        payload.type = Marker::CUBE;
        payload.action = Marker::ADD;
        payload.pose = eigenToPose(display_state_->getGlobalLinkTransform(side + "_tool0") *
          toolToBox(side, false));
        payload.scale.x = box_depth_;
        payload.scale.y = box_width_;
        payload.scale.z = box_height_;
        payload.color = color(0.20F, 0.85F, 0.25F, 0.72F);
        markers.markers.push_back(payload);
      }
      alfa_robot::motion::appendDemoFailureMarkers(
        markers, display_diagnostic_, display_failure_frozen_, world_frame_, now());
      scene_marker_publisher_->publish(markers);
      return;
    }
    Marker target;
    target.header.frame_id = world_frame_;
    target.header.stamp = now();
    target.ns = "target_box";
    target.id = 0;
    target.type = Marker::CUBE;
    target.action = Marker::ADD;
    target.pose.position.x = center.x();
    target.pose.position.y = center.y();
    target.pose.position.z = center.z();
    target.pose.orientation.w = 1.0;
    target.scale.x = box_depth_;
    target.scale.y = box_width_;
    target.scale.z = box_height_;
    target.color = color(0.20F, 0.85F, 0.25F, 0.72F);
    if (display_box_attached_) {
      const auto& tool = display_state_->getGlobalLinkTransform(snapshot.at("tool_link").get<std::string>());
      Eigen::Isometry3d offset = Eigen::Isometry3d::Identity();
      for (int row = 0; row < 3; ++row) {
        offset.translation()[row] = snapshot.at("tool_to_box_center")[row].get<double>();
        for (int col = 0; col < 3; ++col)
          offset.linear()(row, col) = snapshot.at("tool_to_box_rotation")[row][col].get<double>();
      }
      target.pose = eigenToPose(tool * offset);
    }
    if (display_box_visible_ || !distance_demo_) markers.markers.push_back(target);

    Marker control_link;
    control_link.header.frame_id = world_frame_;
    control_link.header.stamp = now();
    control_link.ns = "box_control_handle";
    control_link.id = 0;
    control_link.type = Marker::LINE_STRIP;
    control_link.action = Marker::ADD;
    control_link.pose.orientation.w = 1.0;
    control_link.scale.x = 0.008;
    control_link.color = color(0.10F, 0.85F, 1.0F, 0.85F);
    geometry_msgs::msg::Point handle_point;
    const Eigen::Vector3d handle_position = controlHandlePosition(center);
    handle_point.x = handle_position.x();
    handle_point.y = handle_position.y();
    handle_point.z = handle_position.z();
    control_link.points.push_back(handle_point);
    geometry_msgs::msg::Point center_point;
    center_point.x = center.x();
    center_point.y = center.y();
    center_point.z = center.z();
    control_link.points.push_back(center_point);
    if (!distance_demo_) markers.markers.push_back(control_link);

    std::vector<Eigen::Vector3d> neighbors;
    for (const auto& p : snapshot.at("neighbor_centers"))
      neighbors.emplace_back(p[0].get<double>(), p[1].get<double>(), p[2].get<double>());
    for (size_t index = 0; index < neighbors.size(); ++index) {
      Marker box;
      box.header.frame_id = world_frame_;
      box.header.stamp = now();
      box.ns = "neighbor_boxes";
      box.id = static_cast<int>(index);
      box.type = Marker::CUBE;
      box.action = Marker::ADD;
      box.pose.position.x = neighbors[index].x();
      box.pose.position.y = neighbors[index].y();
      box.pose.position.z = neighbors[index].z();
      box.pose.orientation.w = 1.0;
      box.scale.x = box_depth_;
      box.scale.y = box_width_;
      box.scale.z = box_height_;
      box.color = color(1.0F, 0.48F, 0.08F, 0.45F);
      markers.markers.push_back(box);
    }

    const std::array<std::pair<Eigen::Vector3d, std::string>, 3> points = {{
      {precontactPose(center).translation(), "precontact"},
      {contactPose(center).translation(), "contact"},
      {retreatPose(center).translation(), "retreat"},
    }};
    for (size_t index = 0; index < points.size(); ++index) {
      Marker point;
      point.header.frame_id = world_frame_;
      point.header.stamp = now();
      point.ns = "task_points";
      point.id = static_cast<int>(index);
      point.type = Marker::SPHERE;
      point.action = Marker::ADD;
      const auto& position = snapshot.at(points[index].second);
      point.pose.position.x = position[0].get<double>();
      point.pose.position.y = position[1].get<double>();
      point.pose.position.z = position[2].get<double>();
      point.pose.orientation.w = 1.0;
      point.scale.x = point.scale.y = point.scale.z = 0.035;
      point.color = index == 0 ?
        color(0.15F, 0.55F, 1.0F, 0.9F) :
        (index == 1 ? color(0.2F, 1.0F, 0.2F, 0.9F) : color(0.95F, 0.2F, 0.95F, 0.9F));
      markers.markers.push_back(point);
    }
    if (distance_demo_) {
      const auto removed = snapshot.at("removed_box_ids").get<std::set<int>>();
      for (int id = 0; id < 25; ++id) {
        if (removed.count(id) || (id == snapshot.at("box_id").get<int>() &&
            (!display_box_visible_ || display_box_attached_))) continue;
        Marker label;
        label.header = target.header;
        label.ns = "wall_box_ids";
        label.id = id;
        label.type = Marker::TEXT_VIEW_FACING;
        label.action = Marker::ADD;
        label.pose.orientation.w = 1.0;
        label.pose.position.x = chassis_front_x_ + wall_distance_ - 0.02;
        label.pose.position.y = wall_center_y_ + (id % 5 - 2) * (box_width_ + wall_gap_);
        label.pose.position.z = wall_bottom_z_ + box_height_ / 2.0 + (id / 5) * (box_height_ + wall_gap_);
        label.scale.z = 0.07;
        label.color = color(1.0F, 1.0F, 1.0F);
        label.text = std::to_string(id);
        markers.markers.push_back(label);
      }
    }
    alfa_robot::motion::appendDemoFailureMarkers(
      markers, display_diagnostic_, display_failure_frozen_, world_frame_, now());
    scene_marker_publisher_->publish(markers);
  }

  void publishDisplayState()
  {
    if (execution_backend_ == "fjt") {
      std::map<std::string, double> feedback;
      if (readFjtState(&feedback, nullptr)) {
        std::lock_guard<std::mutex> lock(display_mutex_);
        for (const auto& name : robot_model_->getVariableNames())
          display_state_->setVariablePosition(name, feedback.at(name));
        display_state_->update(true);
      }
      publishSceneMarkers();
      return;
    }
    std::lock_guard<std::mutex> lock(display_mutex_);
    if (!playback_frames_.empty()) {
      display_failure_frozen_ = !display_diagnostic_.empty() && playback_index_ + 1U == playback_frames_.size();
      const auto& frame = playback_frames_[playback_index_];
      for (size_t index = 0; index < all_joint_names_.size() && index < frame.joints.size(); ++index) {
        display_state_->setVariablePosition(all_joint_names_[index], frame.joints[index]);
      }
      display_state_->update(true);
      display_box_attached_ = frame.box_attached;
      display_box_visible_ = frame.box_visible;
      if (!playback_scenes_.empty()) display_scene_ = playback_scenes_.at(frame.scene_index);
      if (playback_index_ + 1U == playback_frames_.size()) sequence_playback_ = false;
      if (display_failure_frozen_)
        publishStatus("DIAGNOSTIC ONLY - FROZEN (not executable)\n" +
          display_diagnostic_.value("stage", "") + "\n" + display_diagnostic_.value("reason", ""), false);
      playback_index_ = (distance_demo_ || !display_diagnostic_.empty()) ? std::min(playback_index_ + 1U, playback_frames_.size() - 1U) :
        (playback_index_ + 1U) % playback_frames_.size();
    }
    sensor_msgs::msg::JointState message;
    message.header.stamp = now();
    message.name = all_joint_names_;
    message.position = allJoints(*display_state_);
    joint_state_publisher_->publish(message);
    publishSceneMarkers();
  }

  bool display_failure_frozen_ = false;
  nlohmann::json display_diagnostic_ = nlohmann::json::object();
  std::string wall_context_ = "full";
  bool top_suction_ = false;
  bool sequence_mode_ = false;
  bool playback_enabled_ = true;
  bool enable_stage_action_ = false;
  std::string execution_backend_ = "replay";
  std::string follow_joint_trajectory_action_ = "/whole_body_jtc/follow_joint_trajectory";
  std::string trajectory_cache_file_;
  nlohmann::json trajectory_cache_;
  double target_match_tolerance_ = 0.06;
  double target_orientation_tolerance_ = degToRad(5.0);
  double maximum_rotary_velocity_ = degToRad(20.0);
  double maximum_updown_velocity_ = 0.15;
  double minimum_trajectory_step_s_ = 0.05;
  double display_rate_hz_ = 20.0;
  std::string initial_pose_ = "home";
  double top_shoulder_above_wrist_ = 0.10;  // Offline-calibrated top policy, independent of front comfort ratio.
  bool sequence_running_ = false;
  bool sequence_requested_ = false;
  bool sequence_playback_ = false;
  bool display_box_visible_ = true;
  double chassis_rear_x_ = 0.0;
  double rear_clearance_ = 0.02;
  std::set<int> removed_boxes_;
  std::map<int, Eigen::Isometry3d> placed_boxes_;
  mutable std::mutex feedback_mutex_;
  std::map<std::string, double> latest_feedback_;
  std::chrono::steady_clock::time_point feedback_received_at_;
  std::chrono::steady_clock::time_point safety_received_at_;
  bool latest_safety_ready_ = false;
  std::atomic<bool> fjt_outcome_unknown_{false};
  std::atomic<bool> session_requires_reset_{false};
  moveit::core::RobotStatePtr home_state_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr sequence_service_;
  std::vector<nlohmann::json> playback_scenes_;
  nlohmann::json display_scene_;
  bool display_box_attached_ = false;
  std::vector<moveit_msgs::msg::CollisionObject> environment_objects_;
  nlohmann::json environment_json_;
  bool distance_demo_ = false;
  bool direct_attach_ = false;
  std::string direct_placement_pose_ = "unloading";
  std::string post_extract_policy_ = "rear_release";
  std::string rear_placement_strategy_ = "geometric";
  bool align_height_ = false;
  std::string height_strategy_ = "fixed_offset";
  nlohmann::json height_clearance_ = {{"checked", false}};
  std::optional<double> synchronized_updown_;
  std::string comfort_branch_ = "auto";
  double comfort_min_ = 0.8, comfort_preferred_ = 0.8, comfort_max_ = 0.8;
  int planning_seed_ = 0;
  double shoulder_box_offset_ = 0.25;
  double initial_shoulder_z_ = 0.0;
  double chassis_front_x_ = 0.0;
  double wall_center_y_ = 0.0;
  double wall_bottom_z_ = 0.0;
  double wall_distance_ = 0.0;
  std::string requested_arm_ = "auto";
  std::string requested_suction_mode_ = "auto";
  nlohmann::json last_result_;
  nlohmann::json attempts_;
  rclcpp::Service<WallRequest>::SharedPtr wall_service_;
  std::string side_;
  std::string world_frame_;
  std::string arm_base_link_;
  double contact_numerical_gap_ = 0.0;
  std::string planning_group_name_;
  std::string tool_link_;
  Eigen::Vector3d box_center_{0.88, -0.20, 0.55};
  Eigen::Vector3d initial_box_center_{0.88, -0.20, 0.55};
  std::string scene_layout_ = "cross";
  int wall_target_row_ = 0;
  int wall_target_column_ = 2;
  double wall_gap_ = 0.01;
  double box_depth_ = 0.30;
  double box_width_ = 0.40;
  double box_height_ = 0.40;
  double control_handle_clearance_ = 0.25;
  double control_handle_lateral_offset_ = 0.90;
  double approach_distance_ = 0.05;
  double retreat_distance_ = 0.35;
  double cartesian_step_ = 0.01;
  double collision_inset_ = 0.002;
  double psi_step_ = degToRad(5.0);
  double maximum_cartesian_joint_step_ = degToRad(15.0);
  double edge_joint_resolution_ = degToRad(2.5);
  size_t precontact_candidate_limit_ = 8;
  double rrt_planning_time_ = 1.0;
  int rrt_planning_attempts_ = 1;
  std::string connection_planner_ = "rrt_connect";
  size_t shortcut_padding_points_ = 5;
  double shortcut_step_ = degToRad(5.0);
  double shortcut_updown_step_ = 0.01;
  double local_rrt_planning_time_ = 8.0;
  bool auto_run_once_ = false;

  std::shared_ptr<robot_model_loader::RobotModelLoader> robot_model_loader_;
  moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::JointModelGroup* planning_group_ = nullptr;
  planning_pipeline::PlanningPipelinePtr planning_pipeline_;
  moveit::core::RobotStatePtr initial_state_;
  moveit::core::RobotStatePtr display_state_;
  std::unique_ptr<V3RedundantArmAnalyticIk> solver_;
  std::vector<std::string> all_joint_names_;

  std::unique_ptr<interactive_markers::InteractiveMarkerServer> marker_server_;
  interactive_markers::MenuHandler menu_handler_;
  interactive_markers::MenuHandler::EntryHandle confirm_menu_entry_ = 0;
  interactive_markers::MenuHandler::EntryHandle reset_menu_entry_ = 0;
  const std::string publisher_id_ = std::to_string(
    std::chrono::system_clock::now().time_since_epoch().count());
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr segment_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr task_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr wall_target_catalog_publisher_;
  rclcpp::Publisher<robot_system_interfaces::msg::DomainReadiness>::SharedPtr readiness_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr feedback_subscription_;
  rclcpp::Subscription<robot_rt_control_interfaces::msg::SafetyState>::SharedPtr safety_subscription_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr scene_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr status_marker_publisher_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr run_service_;
  rclcpp::TimerBase::SharedPtr worker_timer_;
  rclcpp::TimerBase::SharedPtr display_timer_;
  rclcpp::TimerBase::SharedPtr auto_run_timer_;
  rclcpp::TimerBase::SharedPtr readiness_timer_;
  rclcpp_action::Server<StageAction>::SharedPtr stage_action_server_;
  rclcpp_action::Client<FollowJointTrajectory>::SharedPtr fjt_client_;
  FollowJointTrajectoryGoalHandle::SharedPtr fjt_goal_handle_;

  std::mutex box_mutex_;
  std::mutex display_mutex_;
  mutable std::mutex public_flow_mutex_;
  std::mutex fjt_goal_mutex_;
  std::optional<CachedPublicFlow> cached_public_flow_;
  std::atomic<PublicFlowState> public_flow_state_{PublicFlowState::Idle};
  std::vector<ReplayFrame> playback_frames_;
  size_t playback_index_ = 0;
  std::atomic<bool> planning_requested_{false};
  std::atomic<bool> planning_active_{false};
  std::atomic<bool> public_goal_active_{false};
  std::atomic<bool> public_cancel_requested_{false};
  uint64_t generation_ = 0;
};

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<V3SingleArmBoxExtractDemo>(options);
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  std::thread spin_thread([&executor]() {executor.spin();});
  try {
    node->init();
    spin_thread.join();
  } catch (const std::exception& error) {
    RCLCPP_FATAL(node->get_logger(), "single-arm extract demo init failed: %s", error.what());
    executor.cancel();
    if (spin_thread.joinable()) spin_thread.join();
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
