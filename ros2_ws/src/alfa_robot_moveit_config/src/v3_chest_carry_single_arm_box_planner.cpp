#include <alfa_robot_analytic_ik/v3_redundant_analytic_ik.hpp>
#include <alfa_robot_moveit_config/planning_diagnostics.hpp>
#ifdef ALFA_HAS_VAMP
#include <alfa_robot_moveit_config/vamp_full_online_rrt.hpp>
#endif
#ifdef ALFA_HAS_VAMP_PAPER_KERNEL
#include <alfa_robot_moveit_config/vamp_paper_native.hpp>
#endif

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
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
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
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
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
using Feedback = visualization_msgs::msg::InteractiveMarkerFeedback;
using InteractiveMarker = visualization_msgs::msg::InteractiveMarker;
using InteractiveMarkerControl = visualization_msgs::msg::InteractiveMarkerControl;
using Marker = visualization_msgs::msg::Marker;
namespace ob = ompl::base;
namespace og = ompl::geometric;

#ifdef ALFA_HAS_VAMP_PAPER_KERNEL
class PaperHaltonSequence
{
public:
  explicit PaperHaltonSequence(size_t skip) : skip_(skip) {reset();}

  void reset()
  {
    halton_.reset();
    for (size_t index = 0U; index < skip_; ++index) {
      static_cast<void>(halton_.next());
    }
  }

  auto next() {return halton_.next();}

private:
  size_t skip_;
  vamp::rng::Halton<vamp::robots::AlfaV322> halton_;
};

class PaperHaltonStateSampler final : public ob::StateSampler
{
public:
  PaperHaltonStateSampler(
    const ob::StateSpace* space,
    std::shared_ptr<PaperHaltonSequence> sequence,
    std::shared_ptr<size_t> sample_count)
  : ob::StateSampler(space),
    sequence_(std::move(sequence)), sample_count_(std::move(sample_count))
  {}

  void sampleUniform(ob::State* state) override
  {
    ++(*sample_count_);
    const auto values = sequence_->next().to_array();
    auto* output = state->as<ob::RealVectorStateSpace::StateType>();
    for (size_t index = 0U; index < alfa_robot::motion::kPaperVampDimension; ++index) {
      output->values[index] = values[index];
    }
  }

  void sampleUniformNear(ob::State* state, const ob::State*, double) override
  {
    sampleUniform(state);
  }

  void sampleGaussian(ob::State* state, const ob::State*, double) override
  {
    sampleUniform(state);
  }

private:
  std::shared_ptr<PaperHaltonSequence> sequence_;
  std::shared_ptr<size_t> sample_count_;
};

class PaperCorridorSequence
{
public:
  using State = std::array<float, alfa_robot::motion::kPaperVampDimension>;

  PaperCorridorSequence(
    const State& start,
    const State& goal,
    const State& lower,
    const State& upper,
    uint32_t seed,
    size_t skip)
  : engine_(seed), start_(start), goal_(goal), lower_(lower), upper_(upper)
  {
    for (size_t index = 0U; index < skip; ++index) static_cast<void>(next());
  }

  State next()
  {
    State values{};
    if (uniform_(engine_) < 0.9F) {
      const float progress = uniform_(engine_);
      for (size_t index = 0U; index < values.size(); ++index) {
        values[index] = std::abs(goal_[index] - start_[index]) <= 1e-6F ?
          start_[index] : start_[index] + progress * (goal_[index] - start_[index]);
      }
    } else {
      for (size_t index = 0U; index < values.size(); ++index) {
        values[index] = std::abs(goal_[index] - start_[index]) <= 1e-6F ?
          start_[index] : lower_[index] + uniform_(engine_) * (upper_[index] - lower_[index]);
      }
    }
    return values;
  }

private:
  std::mt19937 engine_;
  std::uniform_real_distribution<float> uniform_{0.0F, 1.0F};
  State start_{};
  State goal_{};
  State lower_{};
  State upper_{};
};

class PaperCorridorStateSampler final : public ob::StateSampler
{
public:
  PaperCorridorStateSampler(
    const ob::StateSpace* space,
    std::shared_ptr<PaperCorridorSequence> sequence,
    std::shared_ptr<size_t> sample_count)
  : ob::StateSampler(space),
    sequence_(std::move(sequence)), sample_count_(std::move(sample_count))
  {}

  void sampleUniform(ob::State* state) override
  {
    ++(*sample_count_);
    const auto values = sequence_->next();
    auto* output = state->as<ob::RealVectorStateSpace::StateType>();
    for (size_t index = 0U; index < values.size(); ++index) {
      output->values[index] = values[index];
    }
  }

  void sampleUniformNear(ob::State* state, const ob::State*, double) override
  {
    sampleUniform(state);
  }

  void sampleGaussian(ob::State* state, const ob::State*, double) override
  {
    sampleUniform(state);
  }

private:
  std::shared_ptr<PaperCorridorSequence> sequence_;
  std::shared_ptr<size_t> sample_count_;
};
#endif

constexpr double kPi = 3.14159265358979323846;
constexpr char kMarkerName[] = "extract_box";
constexpr char kCarriedBoxId[] = "carried_target_box";
constexpr char kGroundId[] = "ground";
constexpr std::array<double, 7> kNaturalJointWeights = {
  1.0, 1.1, 1.0, 1.05, 1.2, 1.15, 1.3};

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
  const std::array<double, 7>& to)
{
  double maximum = 0.0;
  for (size_t index = 0; index < from.size(); ++index) {
    maximum = std::max(maximum, std::abs(normalizedAngle(to[index] - from[index])));
  }
  return maximum;
}

double squaredJointDistance(
  const std::array<double, 7>& from,
  const std::array<double, 7>& to)
{
  double distance = 0.0;
  for (size_t index = 0; index < from.size(); ++index) {
    const double delta = normalizedAngle(to[index] - from[index]);
    distance += delta * delta;
  }
  return distance;
}

double weightedSquaredJointDistance(
  const std::array<double, 7>& from,
  const std::array<double, 7>& to,
  const std::array<double, 7>& weights)
{
  double distance = 0.0;
  for (size_t index = 0; index < from.size(); ++index) {
    const double delta = normalizedAngle(to[index] - from[index]);
    distance += weights[index] * delta * delta;
  }
  return distance;
}

double weightedJointTravel(
  const std::array<double, 7>& from,
  const std::array<double, 7>& to,
  const std::array<double, 7>& weights)
{
  double travel = 0.0;
  for (size_t index = 0; index < from.size(); ++index) {
    travel += weights[index] * std::abs(normalizedAngle(to[index] - from[index]));
  }
  return travel;
}

std::array<double, 7> jointDelta(
  const std::array<double, 7>& from,
  const std::array<double, 7>& to)
{
  std::array<double, 7> delta{};
  for (size_t index = 0; index < from.size(); ++index) {
    delta[index] = normalizedAngle(to[index] - from[index]);
  }
  return delta;
}

double weightedSquaredJointDeltaError(
  const std::array<double, 7>& actual,
  const std::array<double, 7>& expected,
  const std::array<double, 7>& weights)
{
  double error = 0.0;
  for (size_t index = 0; index < actual.size(); ++index) {
    const double delta_error = normalizedAngle(actual[index] - expected[index]);
    error += weights[index] * delta_error * delta_error;
  }
  return error;
}

double jointLimitBarrier(
  const std::array<double, 7>& joints,
  const std::array<double, 7>& lower,
  const std::array<double, 7>& upper)
{
  constexpr double kFreeRatio = 0.65;
  constexpr std::array<double, 7> kWeights = {0.5, 3.0, 0.7, 0.5, 1.5, 1.2, 1.5};
  double cost = 0.0;
  for (size_t index = 0; index < joints.size(); ++index) {
    const double half_range = 0.5 * (upper[index] - lower[index]);
    if (half_range <= 1e-9) {
      continue;
    }
    const double center = 0.5 * (upper[index] + lower[index]);
    const double normalized = std::abs(joints[index] - center) / half_range;
    if (normalized <= kFreeRatio) {
      continue;
    }
    const double excess = (normalized - kFreeRatio) / (1.0 - kFreeRatio);
    const double barrier = excess * excess / std::max(1e-3, 1.0 - normalized);
    cost += kWeights[index] * barrier;
  }
  return cost;
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

std::set<int> parseBoxIds(const std::string& value)
{
  std::set<int> ids;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    size_t consumed = 0;
    const int id = std::stoi(token, &consumed);
    if (consumed != token.size()) {
      throw std::invalid_argument("invalid box id: " + token);
    }
    ids.insert(id);
  }
  return ids;
}

std::vector<double> parseNumberList(const std::string& value)
{
  std::vector<double> values;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    size_t consumed = 0;
    const double number = std::stod(token, &consumed);
    if (consumed != token.size()) {
      throw std::invalid_argument("invalid numeric value: " + token);
    }
    values.push_back(number);
  }
  return values;
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
  double tcp_shortcut_ms = 0.0;
  double reduced_rrt_ms = 0.0;
  double joint_rrt_ms = 0.0;
  size_t reduced_rrt_samples = 0U;
  size_t parallel_workers_launched = 0U;
  size_t parallel_workers_completed = 0U;
  int parallel_winner_seed = -1;
  std::vector<size_t> repaired_joint_indices;
  std::string reason;
  std::string strategy;
  std::vector<moveit::core::RobotStatePtr> states;
  nlohmann::json backend_audit = nlohmann::json::object();
};

struct PlanningMetrics
{
  uint64_t ik_calls = 0;
  double ik_ms = 0.0;
  uint64_t collision_checks = 0;
  double collision_ms = 0.0;
  double analytic_path_ms = 0.0;
  double rrt_approach_ms = 0.0;
  double rrt_return_ms = 0.0;
  uint64_t rrt_shortcut_edges_checked = 0;
  double rrt_shortcut_ms = 0.0;
  uint64_t cartesian_transfer_attempts = 0;
  uint64_t cartesian_guided_segments = 0;
  uint64_t joint_fallback_segments = 0;
  uint64_t cartesian_transfer_budget_exhaustions = 0;
  double tcp_shortcut_ms = 0.0;
  double reduced_rrt_ms = 0.0;
  double joint_rrt_ms = 0.0;
  uint64_t tcp_shortcut_attempts = 0;
  uint64_t tcp_shortcut_successes = 0;
  uint64_t reduced_rrt_attempts = 0;
  uint64_t reduced_rrt_samples = 0;
  uint64_t reduced_rrt_successes = 0;
  uint64_t parallel_rrt_workers_launched = 0;
  uint64_t parallel_rrt_workers_completed = 0;
  uint64_t parallel_rrt_wins = 0;
  uint64_t joint_rrt_fallbacks = 0;
};

struct TaskResult
{
  bool success = false;
  std::string failure_stage;
  std::string failure_reason;
  double total_ms = 0.0;
  PlanningMetrics metrics;
  std::vector<ReplayFrame> frames;
  std::optional<Eigen::Isometry3d> achieved_place_tcp;
  std::string transition_group;
  std::string transition_strategy;
  std::string transition_diagnostic;
  nlohmann::json segment_search = nlohmann::json::array();
  nlohmann::json paper_benchmark = nlohmann::json::object();
};

#ifdef ALFA_HAS_VAMP_PAPER_KERNEL
struct PaperPairedSegment
{
  bool success{false};
  RrtPlanResult ompl;
  RrtPlanResult vamp;
  double ompl_accumulated_planner_ms{0.0};
  double vamp_accumulated_planner_ms{0.0};
  double ompl_accumulated_wall_ms{0.0};
  double vamp_accumulated_wall_ms{0.0};
  nlohmann::json ompl_attempts = nlohmann::json::array();
  nlohmann::json vamp_attempts = nlohmann::json::array();
};
#endif

struct SceneBox
{
  std::string id;
  Eigen::Vector3d center;
  Eigen::Vector3d size;
};

class V3SingleArmBoxExtractDemo : public rclcpp::Node
{
public:
  explicit V3SingleArmBoxExtractDemo(const rclcpp::NodeOptions& options)
  : Node("v3_chest_carry_single_arm_box_planner", options)
  {}

  void init()
  {
    side_ = getParameter<std::string>("side", "left");
    world_frame_ = getParameter<std::string>("world_frame", "map");
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
    box_center_ = Eigen::Vector3d(
      getParameter<double>("initial_box_x", initial_box[0]),
      getParameter<double>("initial_box_y", initial_box[1]),
      getParameter<double>("initial_box_z", initial_box[2]));
    const double initial_updown = getParameter<double>("initial_updown", 0.0);
    initial_arm_pose_ = getParameter<std::string>("initial_arm_pose", "zero");
    const auto initial_left_arm_joints_deg = parseNumberList(
      getParameter<std::string>("initial_left_arm_joints_deg", ""));
    const auto initial_right_arm_joints_deg = parseNumberList(
      getParameter<std::string>("initial_right_arm_joints_deg", ""));
    box_depth_ = getParameter<double>("box_depth", 0.30);
    box_width_ = getParameter<double>("box_width", 0.40);
    box_height_ = getParameter<double>("box_height", 0.40);
    control_handle_clearance_ = std::max(
      0.10, getParameter<double>("control_handle_clearance", 0.25));
    control_handle_lateral_offset_ = std::max(
      0.50, getParameter<double>("control_handle_lateral_offset", 0.90));
    approach_distance_ = getParameter<double>("approach_distance", 0.05);
    retreat_distance_ = getParameter<double>("retreat_distance", 0.35);
    cartesian_step_ = getParameter<double>("cartesian_step", 0.01);
    collision_inset_ = getParameter<double>("collision_inset", 0.002);
    full_box_wall_scene_ = getParameter<bool>("full_box_wall_scene", false);
    box_grid_columns_ = std::max(1, getParameter<int>("box_grid_columns", 5));
    box_grid_rows_ = std::max(1, getParameter<int>("box_grid_rows", 5));
    box_grid_center_y_ = getParameter<double>("box_grid_center_y", 0.0);
    box_grid_bottom_z_ = getParameter<double>("box_grid_bottom_z", 0.0);
    target_box_id_ = getParameter<int>("target_box_id", 0);
    removed_box_ids_ = parseBoxIds(getParameter<std::string>("removed_box_ids", ""));
    grasp_mode_ = getParameter<std::string>("grasp_mode", "front");
    front_suction_y_offset_ = getParameter<double>("front_suction_y_offset", 0.0);
    front_suction_z_offset_ = getParameter<double>("front_suction_z_offset", 0.0);
    top_suction_x_offset_ = getParameter<double>("top_suction_x_offset", 0.0);
    contact_tool_roll_ = degToRad(getParameter<double>("contact_tool_roll_deg", 0.0));
    ground_enabled_ = getParameter<bool>("ground_enabled", true);
    ground_surface_z_ = getParameter<double>("ground_surface_z", box_grid_bottom_z_);
    ground_clearance_ = std::max(0.0, getParameter<double>("ground_clearance", 0.005));
    ground_size_x_ = getParameter<double>("ground_size_x", 6.0);
    ground_size_y_ = getParameter<double>("ground_size_y", 6.0);
    ground_thickness_ = getParameter<double>("ground_thickness", 0.10);
    warehouse_enabled_ = getParameter<bool>("warehouse_enabled", false);
    warehouse_opening_x_ = getParameter<double>("warehouse_opening_x", -1.18);
    warehouse_center_y_ = getParameter<double>("warehouse_center_y", 0.0);
    warehouse_floor_z_ = getParameter<double>("warehouse_floor_z", 0.0);
    warehouse_length_ = getParameter<double>("warehouse_length", 2.38);
    warehouse_width_ = getParameter<double>("warehouse_width", 2.38);
    warehouse_height_ = getParameter<double>("warehouse_height", 2.35);
    warehouse_wall_thickness_ = getParameter<double>("warehouse_wall_thickness", 0.05);
    psi_step_ = degToRad(getParameter<double>("psi_step_deg", 5.0));
    maximum_cartesian_joint_step_ = degToRad(
      getParameter<double>("maximum_cartesian_joint_step_deg", 15.0));
    edge_joint_resolution_ = degToRad(
      getParameter<double>("edge_joint_resolution_deg", 2.5));
    precontact_candidate_limit_ = static_cast<size_t>(std::max(
      1, getParameter<int>("precontact_candidate_limit", 8)));
    rrt_planning_time_ = std::max(0.05, getParameter<double>("rrt_planning_time", 1.0));
    rrt_planning_attempts_ = std::max(1, getParameter<int>("rrt_planning_attempts", 1));
    pure_rrt_full_flow_ = getParameter<bool>("pure_rrt_full_flow", false);
    pure_rrt_backend_ = getParameter<std::string>("pure_rrt_backend", "ompl");
    vamp_workers_ = getParameter<int>("vamp_workers", 4);
    vamp_max_iterations_ = getParameter<int>("vamp_max_iterations", 100000);
    vamp_max_samples_ = getParameter<int>("vamp_max_samples", 200000);
    vamp_range_ = getParameter<double>("vamp_range", 0.08);
    vamp_seed_ = getParameter<int64_t>("vamp_seed", 20261019);
    paper_halton_skip_ = static_cast<size_t>(std::max(
      0, getParameter<int>("paper_halton_skip", 0)));
    paper_search_retries_ = static_cast<size_t>(std::max(
      1, getParameter<int>("paper_search_retries", 3)));
    paper_shared_ik_path_ = getParameter<std::string>("paper_shared_ik_path", "");
    ompl_simplify_solutions_ = getParameter<bool>("ompl.simplify_solutions", true);
    natural_motion_enabled_ = getParameter<bool>("natural_motion_enabled", true);
    natural_swivel_weight_ = getParameter<double>("natural_swivel_weight", 0.02);
    natural_wrist_singularity_weight_ = getParameter<double>(
      "natural_wrist_singularity_weight", 0.03);
    natural_wrist_neutral_weight_ = getParameter<double>("natural_wrist_neutral_weight", 0.02);
    natural_joint_limit_weight_ = getParameter<double>("natural_joint_limit_weight", 0.05);
    natural_joint_wrap_weight_ = getParameter<double>("natural_joint_wrap_weight", 0.0);
    natural_seed_swivel_sampling_ = getParameter<bool>(
      "natural_seed_swivel_sampling", false);
    natural_seed_swivel_step_ = degToRad(
      getParameter<double>("natural_seed_swivel_step_deg", 1.0));
    natural_seed_swivel_neighbor_steps_ = std::max(
      0, getParameter<int>("natural_seed_swivel_neighbor_steps", 2));
    natural_joint_acceleration_weight_ = getParameter<double>(
      "natural_joint_acceleration_weight", 0.0);
    natural_place_return_weight_ = getParameter<double>(
      "natural_place_return_weight", 0.0);
    natural_cartesian_replay_step_ = degToRad(
      getParameter<double>("natural_cartesian_replay_step_deg", 0.0));
    natural_rrt_shortcut_enabled_ = getParameter<bool>(
      "natural_rrt_shortcut_enabled", false);
    natural_rrt_shortcut_tcp_cost_ = getParameter<bool>(
      "natural_rrt_shortcut_tcp_cost", false);
    natural_rrt_shortcut_max_nodes_ = getParameter<int>(
      "natural_rrt_shortcut_max_nodes", 0);
    cartesian_transfer_search_enabled_ = getParameter<bool>(
      "cartesian_transfer_search_enabled", false);
    cartesian_transfer_translation_step_ = getParameter<double>(
      "cartesian_transfer_translation_step", 0.02);
    cartesian_transfer_rotation_step_ = degToRad(getParameter<double>(
      "cartesian_transfer_rotation_step_deg", 2.0));
    cartesian_transfer_max_search_attempts_ = getParameter<int>(
      "cartesian_transfer_max_search_attempts", 0);
    shortcut_repair_rrt_enabled_ = getParameter<bool>("shortcut_repair_rrt_enabled", false);
    shortcut_repair_rrt_budget_ms_ = getParameter<double>(
      "shortcut_repair_rrt_budget_ms", 1800.0);
    shortcut_repair_rrt_max_samples_ = getParameter<int>(
      "shortcut_repair_rrt_max_samples", 240);
    parallel_rrt_workers_ = std::max(
      1, getParameter<int>("parallel_rrt_workers", 1));
    shortcut_repair_max_joint_offset_ = degToRad(getParameter<double>(
      "shortcut_repair_max_joint_offset_deg", 35.0));
    place_updown_enabled_ = getParameter<bool>("place_updown_enabled", false);
    place_updown_ = getParameter<double>("place_updown", initial_updown);
    top_loaded_transfer_direct_only_ = getParameter<bool>(
      "top_loaded_transfer_direct_only", false);
    loaded_transfer_defer_place_updown_ = getParameter<bool>(
      "loaded_transfer_defer_place_updown", false);
    auto_safe_opposite_arm_ = getParameter<bool>("auto_safe_opposite_arm", false);
    natural_max_proximal_step_ = degToRad(
      getParameter<double>("natural_max_proximal_step_deg", 12.0));
    natural_max_wrist_step_ = degToRad(
      getParameter<double>("natural_max_wrist_step_deg", 8.0));
    natural_rrt_replay_step_ = degToRad(
      getParameter<double>("natural_rrt_replay_step_deg", 3.0));
    auto_run_once_ = getParameter<bool>("auto_run_once", false);
    analytic_path_only_ = getParameter<bool>("analytic_path_only", false);
    task_mode_ = getParameter<std::string>("task_mode", "full_extract");
    ignore_opposite_arm_ = getParameter<bool>("ignore_opposite_arm", false);
    continuous_sequence_ = getParameter<bool>("continuous_sequence", false);
    continuous_plan_approach_ = getParameter<bool>("continuous_plan_approach", false);
    maximum_carried_box_tilt_ = degToRad(
      getParameter<double>("maximum_carried_box_tilt_deg", 180.0));
    maximum_carried_box_center_z_ = getParameter<double>(
      "maximum_carried_box_center_z", 1000000.0);
    result_json_path_ = getParameter<std::string>("result_json_path", "");
    place_tcp_pose_ = parseNumberList(getParameter<std::string>("place_tcp_pose", ""));
    place_arm_joints_deg_ = parseNumberList(
      getParameter<std::string>("place_arm_joints_deg", ""));
    precontact_arm_joints_deg_ = parseNumberList(
      getParameter<std::string>("precontact_arm_joints_deg", ""));
    loaded_transfer_joint_waypoints_deg_ = parseNumberList(
      getParameter<std::string>("loaded_transfer_joint_waypoints_deg", ""));
    loaded_transfer_waypoint_start_deg_ = parseNumberList(
      getParameter<std::string>("loaded_transfer_waypoint_start_deg", ""));
    loaded_transfer_arm_updown_waypoints_ = parseNumberList(
      getParameter<std::string>("loaded_transfer_arm_updown_waypoints", ""));
    loaded_transfer_arm_updown_waypoint_start_ = parseNumberList(
      getParameter<std::string>("loaded_transfer_arm_updown_waypoint_start", ""));
    transition_joint_waypoints_ = parseNumberList(
      getParameter<std::string>("transition_joint_waypoints", ""));
    transition_waypoint_start_ = parseNumberList(
      getParameter<std::string>("transition_waypoint_start", ""));
    transition_whole_body_waypoints_ = parseNumberList(
      getParameter<std::string>("transition_whole_body_waypoints", ""));
    transition_whole_body_waypoint_start_ = parseNumberList(
      getParameter<std::string>("transition_whole_body_waypoint_start", ""));
    transition_from_joints_ = parseNumberList(
      getParameter<std::string>("transition_from_joints", ""));
    transition_to_joints_ = parseNumberList(
      getParameter<std::string>("transition_to_joints", ""));

    if (side_ != "left" && side_ != "right") {
      throw std::invalid_argument("side must be left or right");
    }
    if (!std::isfinite(maximum_carried_box_tilt_) ||
        maximum_carried_box_tilt_ <= 0.0 || maximum_carried_box_tilt_ > kPi) {
      throw std::invalid_argument("maximum_carried_box_tilt_deg must be in (0, 180]");
    }
    if (!std::isfinite(maximum_carried_box_center_z_)) {
      throw std::invalid_argument("maximum_carried_box_center_z must be finite");
    }
    if (!std::isfinite(contact_tool_roll_)) {
      throw std::invalid_argument("contact_tool_roll_deg must be finite");
    }
    if (!std::isfinite(natural_joint_wrap_weight_) || natural_joint_wrap_weight_ < 0.0) {
      throw std::invalid_argument("natural_joint_wrap_weight must be non-negative and finite");
    }
    if (natural_rrt_shortcut_max_nodes_ < 0 || natural_rrt_shortcut_max_nodes_ == 1) {
      throw std::invalid_argument("natural_rrt_shortcut_max_nodes must be zero or at least two");
    }
    if (!std::isfinite(cartesian_transfer_translation_step_) ||
        cartesian_transfer_translation_step_ <= 0.0 ||
        !std::isfinite(cartesian_transfer_rotation_step_) ||
        cartesian_transfer_rotation_step_ <= 0.0) {
      throw std::invalid_argument("Cartesian transfer steps must be positive and finite");
    }
    if (cartesian_transfer_max_search_attempts_ < 0) {
      throw std::invalid_argument("cartesian_transfer_max_search_attempts must be non-negative");
    }
    if (!std::isfinite(shortcut_repair_rrt_budget_ms_) ||
        shortcut_repair_rrt_budget_ms_ <= 0.0 || shortcut_repair_rrt_max_samples_ < 1 ||
        parallel_rrt_workers_ > 32 ||
        !std::isfinite(shortcut_repair_max_joint_offset_) ||
        shortcut_repair_max_joint_offset_ <= 0.0) {
      throw std::invalid_argument(
        "shortcut repair RRT limits must be positive and workers must be <= 32");
    }
    if (place_updown_enabled_ &&
        (!std::isfinite(place_updown_) || place_updown_ < -1.0 || place_updown_ > 0.0)) {
      throw std::invalid_argument("place_updown must be in [-1.0, 0.0]");
    }
    if (task_mode_ != "contact_reachability" && task_mode_ != "full_extract" &&
        task_mode_ != "state_transition" && task_mode_ != "paper_rrt_vamp_benchmark" &&
        task_mode_ != "paper_adaptive_multistage_benchmark" &&
        task_mode_ != "paper_shared_ik_fullflow_benchmark" &&
        task_mode_ != "paper_sphere_world_fullflow_benchmark") {
      throw std::invalid_argument(
        "task_mode must be contact_reachability, full_extract, state_transition, "
        "paper_rrt_vamp_benchmark, paper_adaptive_multistage_benchmark, or "
        "paper_shared_ik_fullflow_benchmark, or paper_sphere_world_fullflow_benchmark");
    }
    if (pure_rrt_full_flow_ && ompl_simplify_solutions_) {
      throw std::invalid_argument(
        "pure_rrt_full_flow requires ompl.simplify_solutions=false");
    }
    if (pure_rrt_backend_ != "ompl" && pure_rrt_backend_ != "vamp") {
      throw std::invalid_argument("pure_rrt_backend must be ompl or vamp");
    }
    if (vamp_workers_ < 1 || vamp_workers_ > 32 ||
        vamp_max_iterations_ < 1 || vamp_max_samples_ < 4 ||
        !std::isfinite(vamp_range_) || vamp_range_ <= 0.0 || vamp_seed_ < 0) {
      throw std::invalid_argument("VAMP worker, budget, range, and seed are invalid");
    }
#ifndef ALFA_HAS_VAMP
    if (pure_rrt_backend_ == "vamp") {
      throw std::invalid_argument("pure_rrt_backend=vamp requires vamp::vamp");
    }
#endif
    if (pure_rrt_full_flow_ &&
        (natural_rrt_shortcut_enabled_ || cartesian_transfer_search_enabled_ ||
        shortcut_repair_rrt_enabled_ || !loaded_transfer_joint_waypoints_deg_.empty() ||
        !loaded_transfer_arm_updown_waypoints_.empty() ||
        !transition_joint_waypoints_.empty() ||
        !transition_whole_body_waypoints_.empty() || top_loaded_transfer_direct_only_)) {
      throw std::invalid_argument(
        "pure_rrt_full_flow forbids shortcut, Cartesian, repair, direct-only, and waypoint inputs");
    }
    if (grasp_mode_ != "front" && grasp_mode_ != "top_suction") {
      throw std::invalid_argument("grasp_mode must be front or top_suction");
    }
    if (initial_arm_pose_ != "zero" && initial_arm_pose_ != "v3_home") {
      throw std::invalid_argument("initial_arm_pose must be zero or v3_home");
    }
    if (!place_tcp_pose_.empty()) {
      if ((task_mode_ != "full_extract" &&
          task_mode_ != "paper_adaptive_multistage_benchmark") ||
          place_tcp_pose_.size() != 7U ||
          !std::all_of(place_tcp_pose_.begin(), place_tcp_pose_.end(),
            [](double value) {return std::isfinite(value);})) {
        throw std::invalid_argument(
                "place_tcp_pose requires seven finite x,y,z,qx,qy,qz,qw values");
      }
      const Eigen::Quaterniond quaternion(
        place_tcp_pose_[6], place_tcp_pose_[3], place_tcp_pose_[4], place_tcp_pose_[5]);
      if (std::abs(quaternion.norm() - 1.0) > 1e-3) {
        throw std::invalid_argument("place_tcp_pose quaternion must be normalized");
      }
    }
    if (!place_arm_joints_deg_.empty() &&
        (task_mode_ != "full_extract" || place_arm_joints_deg_.size() != 7U ||
        !std::all_of(place_arm_joints_deg_.begin(), place_arm_joints_deg_.end(),
          [](double value) {return std::isfinite(value);})) ) {
      throw std::invalid_argument(
              "place_arm_joints_deg requires seven finite degree values in full_extract");
    }
    if (!precontact_arm_joints_deg_.empty() &&
        (task_mode_ != "full_extract" || precontact_arm_joints_deg_.size() != 7U ||
        !std::all_of(
          precontact_arm_joints_deg_.begin(), precontact_arm_joints_deg_.end(),
          [](double value) {return std::isfinite(value);})) ) {
      throw std::invalid_argument(
              "precontact_arm_joints_deg requires seven finite degree values in full_extract");
    }
    if (!place_tcp_pose_.empty() && !place_arm_joints_deg_.empty()) {
      throw std::invalid_argument(
              "place_tcp_pose and place_arm_joints_deg are mutually exclusive");
    }
    if (loaded_transfer_joint_waypoints_deg_.size() % 7U != 0U ||
        !std::all_of(
          loaded_transfer_joint_waypoints_deg_.begin(),
          loaded_transfer_joint_waypoints_deg_.end(),
          [](double value) {return std::isfinite(value);})) {
      throw std::invalid_argument(
              "loaded_transfer_joint_waypoints_deg requires finite groups of seven values");
    }
    if (!loaded_transfer_waypoint_start_deg_.empty() &&
        (loaded_transfer_waypoint_start_deg_.size() != 7U ||
        !std::all_of(
          loaded_transfer_waypoint_start_deg_.begin(),
          loaded_transfer_waypoint_start_deg_.end(),
          [](double value) {return std::isfinite(value);}))) {
      throw std::invalid_argument(
              "loaded_transfer_waypoint_start_deg requires seven finite values");
    }
    if (loaded_transfer_arm_updown_waypoints_.size() % 8U != 0U ||
        !std::all_of(
          loaded_transfer_arm_updown_waypoints_.begin(),
          loaded_transfer_arm_updown_waypoints_.end(),
          [](double value) {return std::isfinite(value);})) {
      throw std::invalid_argument(
              "loaded_transfer_arm_updown_waypoints requires finite groups of updown plus seven degrees");
    }
    if (!loaded_transfer_arm_updown_waypoint_start_.empty() &&
        (loaded_transfer_arm_updown_waypoint_start_.size() != 8U ||
        !std::all_of(
          loaded_transfer_arm_updown_waypoint_start_.begin(),
          loaded_transfer_arm_updown_waypoint_start_.end(),
          [](double value) {return std::isfinite(value);})) ) {
      throw std::invalid_argument(
              "loaded_transfer_arm_updown_waypoint_start requires updown plus seven degrees");
    }
    if (transition_joint_waypoints_.size() % 8U != 0U ||
        !std::all_of(
          transition_joint_waypoints_.begin(), transition_joint_waypoints_.end(),
          [](double value) {return std::isfinite(value);})) {
      throw std::invalid_argument(
              "transition_joint_waypoints requires finite groups of updown plus seven degrees");
    }
    if (!transition_waypoint_start_.empty() &&
        (transition_waypoint_start_.size() != 8U ||
        !std::all_of(
          transition_waypoint_start_.begin(), transition_waypoint_start_.end(),
          [](double value) {return std::isfinite(value);})) ) {
      throw std::invalid_argument(
              "transition_waypoint_start requires updown plus seven degrees");
    }
    if (transition_whole_body_waypoints_.size() % 15U != 0U ||
        !std::all_of(
          transition_whole_body_waypoints_.begin(),
          transition_whole_body_waypoints_.end(),
          [](double value) {return std::isfinite(value);})) {
      throw std::invalid_argument(
              "transition_whole_body_waypoints requires finite groups of updown plus fourteen degrees");
    }
    if (!transition_whole_body_waypoint_start_.empty() &&
        (transition_whole_body_waypoint_start_.size() != 15U ||
        !std::all_of(
          transition_whole_body_waypoint_start_.begin(),
          transition_whole_body_waypoint_start_.end(),
          [](double value) {return std::isfinite(value);})) ) {
      throw std::invalid_argument(
              "transition_whole_body_waypoint_start requires updown plus fourteen degrees");
    }
    if (!transition_joint_waypoints_.empty() &&
        !transition_whole_body_waypoints_.empty()) {
      throw std::invalid_argument(
              "single-arm and whole-body transition waypoints are mutually exclusive");
    }
    if ((!initial_left_arm_joints_deg.empty() && initial_left_arm_joints_deg.size() != 7U) ||
        (!initial_right_arm_joints_deg.empty() && initial_right_arm_joints_deg.size() != 7U)) {
      throw std::invalid_argument("explicit initial arm joint lists must contain seven degrees");
    }
    if (std::abs(front_suction_y_offset_) >= box_width_ * 0.5 ||
        std::abs(front_suction_z_offset_) >= box_height_ * 0.5 ||
        std::abs(top_suction_x_offset_) >= box_depth_ * 0.5) {
      throw std::invalid_argument("suction offset must stay inside its box face");
    }
    const int box_count = box_grid_columns_ * box_grid_rows_;
    if (target_box_id_ < 0 || target_box_id_ > box_count) {
      throw std::invalid_argument("target_box_id must be zero or a valid box-grid id");
    }
    for (const int id : removed_box_ids_) {
      if (id < 1 || id > box_count) {
        throw std::invalid_argument("removed_box_ids contains an out-of-range id");
      }
    }
    if (box_depth_ <= 0.0 || box_width_ <= 0.0 || box_height_ <= 0.0 ||
        approach_distance_ <= 0.0 || retreat_distance_ <= 0.0 || cartesian_step_ <= 0.0 ||
        ground_size_x_ <= 0.0 || ground_size_y_ <= 0.0 || ground_thickness_ <= 0.0 ||
        warehouse_length_ <= 0.0 || warehouse_width_ <= 0.0 || warehouse_height_ <= 0.0 ||
        warehouse_wall_thickness_ <= 0.0 || natural_max_proximal_step_ <= 0.0 ||
        natural_max_wrist_step_ <= 0.0 || natural_rrt_replay_step_ <= 0.0) {
      throw std::invalid_argument("box dimensions and Cartesian distances must be positive");
    }
    if (natural_swivel_weight_ < 0.0 || natural_wrist_singularity_weight_ < 0.0 ||
        natural_wrist_neutral_weight_ < 0.0 || natural_joint_limit_weight_ < 0.0 ||
        natural_joint_acceleration_weight_ < 0.0 || natural_seed_swivel_step_ <= 0.0 ||
        natural_place_return_weight_ < 0.0 ||
        natural_cartesian_replay_step_ < 0.0 ||
        natural_max_proximal_step_ <= 0.0 || natural_max_wrist_step_ <= 0.0 ||
        natural_rrt_replay_step_ <= 0.0) {
      throw std::invalid_argument(
              "natural-motion weights must be non-negative and steps positive");
    }

    robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(
      shared_from_this(), "robot_description");
    robot_model_ = robot_model_loader_->getModel();
    if (!robot_model_) {
      throw std::runtime_error("failed to load robot model");
    }
    if (world_frame_ != robot_model_->getModelFrame()) {
      throw std::invalid_argument("world_frame must match the robot model frame");
    }
    planar_root_joint_ = robot_model_->getJointModel("map_to_base_footprint");
    if (!planar_root_joint_ || planar_root_joint_->getVariableCount() != 3U) {
      throw std::runtime_error("V3 MoveIt model requires map_to_base_footprint planar root");
    }
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    planning_group_ = robot_model_->getJointModelGroup(planning_group_name_);
    if (!planning_group_ || planning_group_->getVariableCount() != 7U) {
      throw std::runtime_error("planning group must be a seven-axis arm: " + planning_group_name_);
    }
    if (!robot_model_->hasLinkModel(tool_link_) || !robot_model_->hasLinkModel(arm_base_link_)) {
      throw std::runtime_error("missing tool or arm base link");
    }

    all_joint_names_.reserve(16);
    if (robot_model_->hasJointModel("updown")) {
      all_joint_names_.push_back("updown");
    }
    if (robot_model_->hasJointModel("head_joint")) {
      all_joint_names_.push_back("head_joint");
    }
    for (const std::string arm_side : {std::string("left"), std::string("right")}) {
      for (int index = 1; index <= 7; ++index) {
        all_joint_names_.push_back(arm_side + "_joint" + std::to_string(index));
      }
    }

    initial_state_ = std::make_shared<moveit::core::RobotState>(robot_model_);
    initial_state_->setToDefaultValues();
    for (const auto& name : all_joint_names_) {
      if (robot_model_->hasJointModel(name)) {
        initial_state_->setVariablePosition(name, 0.0);
      }
    }
    if (task_mode_ == "state_transition") {
      if (transition_from_joints_.size() != all_joint_names_.size() ||
          transition_to_joints_.size() != all_joint_names_.size() ||
          !std::all_of(transition_from_joints_.begin(), transition_from_joints_.end(),
            [](double value) {return std::isfinite(value);}) ||
          !std::all_of(transition_to_joints_.begin(), transition_to_joints_.end(),
            [](double value) {return std::isfinite(value);})) {
        throw std::invalid_argument("state_transition requires two finite 16-joint position lists");
      }
      for (size_t index = 0; index < all_joint_names_.size(); ++index) {
        initial_state_->setVariablePosition(
          all_joint_names_[index], transition_from_joints_[index]);
      }
    } else {
    if (initial_arm_pose_ == "v3_home") {
      const std::array<double, 7> left_home = {
        degToRad(155.0), degToRad(-105.0), degToRad(20.0), degToRad(90.0),
        degToRad(-90.0), degToRad(-40.0), 0.0};
      const std::array<double, 7> right_home = {
        degToRad(-155.0), degToRad(-105.0), degToRad(-20.0), degToRad(90.0),
        degToRad(90.0), degToRad(40.0), 0.0};
      for (const auto& [arm_side, home] : {
          std::pair<std::string, std::array<double, 7>>{"left", left_home},
          std::pair<std::string, std::array<double, 7>>{"right", right_home}}) {
        for (size_t index = 0; index < home.size(); ++index) {
          initial_state_->setVariablePosition(
            arm_side + "_joint" + std::to_string(index + 1), home[index]);
        }
      }
    }
    const auto apply_explicit_arm = [this](
      const std::string& arm_side, const std::vector<double>& values_deg) {
        for (size_t index = 0; index < values_deg.size(); ++index) {
          initial_state_->setVariablePosition(
            arm_side + "_joint" + std::to_string(index + 1), degToRad(values_deg[index]));
        }
      };
    apply_explicit_arm("left", initial_left_arm_joints_deg);
    apply_explicit_arm("right", initial_right_arm_joints_deg);
    if (robot_model_->hasJointModel("updown")) {
      initial_state_->setVariablePosition("updown", initial_updown);
    }
    }
    // Named poses may sit exactly on a joint bound; normalize sub-epsilon
    // degree-to-radian roundoff before collision and transition checks.
    initial_state_->enforceBounds();
    initial_state_->update(true);
    display_state_ = std::make_shared<moveit::core::RobotState>(*initial_state_);

    solver_ = std::make_unique<V3RedundantArmAnalyticIk>(
      side_ == "left" ? V3RedundantArmModel::V322Left : V3RedundantArmModel::V322Right);

    const std::vector<std::string> request_adapters = {
      "default_planner_request_adapters/AddTimeOptimalParameterization",
      "default_planner_request_adapters/ResolveConstraintFrames",
      "default_planner_request_adapters/FixWorkspaceBounds",
      "default_planner_request_adapters/FixStartStateBounds",
      "default_planner_request_adapters/FixStartStateCollision",
      "default_planner_request_adapters/FixStartStatePathConstraints",
    };
    planning_pipeline_ = std::make_shared<planning_pipeline::PlanningPipeline>(
      robot_model_, shared_from_this(), "ompl", "ompl_interface/OMPLPlanner", request_adapters);
    planning_pipeline_->displayComputedMotionPlans(false);
    planning_pipeline_->publishReceivedRequests(false);
    planning_pipeline_->checkSolutionPaths(true);

    task_publisher_ = create_publisher<std_msgs::msg::String>(
      "~/task_json", rclcpp::QoS(1).reliable().transient_local());
    joint_state_publisher_ = create_publisher<sensor_msgs::msg::JointState>("~/joint_states", 10);
    scene_marker_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "~/scene_markers", rclcpp::QoS(1).reliable().transient_local());
    status_marker_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "~/status_markers", rclcpp::QoS(1).reliable().transient_local());
    run_service_ = create_service<std_srvs::srv::Trigger>(
      "~/run_current_box",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        response->success = requestPlanning();
        response->message = response->success ?
          "planning request accepted" : "planner is already running";
      });

    marker_server_ = std::make_unique<interactive_markers::InteractiveMarkerServer>(
      "v3_chest_carry_single_arm_box_planner_marker",
      get_node_base_interface(),
      get_node_clock_interface(),
      get_node_logging_interface(),
      get_node_topics_interface(),
      get_node_services_interface());
    createBoxMarker();
    confirm_menu_entry_ = menu_handler_.insert(
      "确认并计算当前箱位",
      [this](const Feedback::ConstSharedPtr&) {requestPlanning();});
    reset_menu_entry_ = menu_handler_.insert(
      "恢复默认箱位",
      [this](const Feedback::ConstSharedPtr&) {resetBoxPose();});
    (void)confirm_menu_entry_;
    (void)reset_menu_entry_;
    menu_handler_.apply(*marker_server_, kMarkerName);
    marker_server_->applyChanges();

    worker_timer_ = create_wall_timer(
      std::chrono::milliseconds(25), [this]() {onWorkerTimer();});
    display_timer_ = create_wall_timer(
      std::chrono::milliseconds(50), [this]() {publishDisplayState();});
    publishPreview("拖动箱体XYZ；右键箱体并选择“确认并计算当前箱位”");
    publishSceneMarkers();
    publishStatus("READY", true);

    if (auto_run_once_) {
      auto_run_timer_ = create_wall_timer(
        std::chrono::milliseconds(500), [this]() {
          if (auto_run_timer_) {
            auto_run_timer_->cancel();
          }
          requestPlanning();
        });
    }

    RCLCPP_INFO(
      get_logger(),
      "V3 single-arm box extract demo ready: side=%s box=(%.2f,%.2f,%.2f) "
      "grasp=%s approach=%.2fm retreat=%.2fm ground=%s warehouse=%s natural=%s "
      "psi_step=%.1fdeg RRT=%.2fs",
      side_.c_str(), box_depth_, box_width_, box_height_, grasp_mode_.c_str(),
      approach_distance_, retreat_distance_, ground_enabled_ ? "on" : "off",
      warehouse_enabled_ ? "on" : "off", natural_motion_enabled_ ? "on" : "off",
      radToDeg(psi_step_), rrt_planning_time_);
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
      box_center_ = Eigen::Vector3d(0.88, -0.20, 0.55);
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

  bool requestPlanning()
  {
    if (planning_active_.load() || planning_requested_.exchange(true)) {
      return false;
    }
    publishStatus("PLANNING REQUESTED", true);
    return true;
  }

  bool refreshPlanarRootFromTf(std::string* reason)
  {
    try {
      const auto transform = tf_buffer_->lookupTransform(
        world_frame_, "base_footprint", tf2::TimePointZero, tf2::durationFromSec(1.0));
      const rclcpp::Time transform_stamp(transform.header.stamp);
      if (transform_stamp.nanoseconds() == 0 ||
          (now() - transform_stamp).seconds() > 0.5) {
        throw tf2::TransformException("map to base_footprint TF is older than 500ms");
      }
      const auto& rotation = transform.transform.rotation;
      const double sin_yaw = 2.0 * (rotation.w * rotation.z + rotation.x * rotation.y);
      const double cos_yaw = 1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z);
      const double values[] = {
        transform.transform.translation.x,
        transform.transform.translation.y,
        std::atan2(sin_yaw, cos_yaw),
      };
      initial_state_->setJointPositions(planar_root_joint_, values);
      initial_state_->update(true);
      {
        std::lock_guard<std::mutex> lock(display_mutex_);
        display_state_->setJointPositions(planar_root_joint_, values);
        display_state_->update(true);
      }
      base_pose_map_ = {values[0], values[1], values[2]};
      base_pose_ready_ = true;
      return true;
    } catch (const tf2::TransformException& error) {
      base_pose_ready_ = false;
      if (reason) {
        *reason = std::string("base pose TF unavailable: ") + error.what();
      }
      return false;
    }
  }

  void onWorkerTimer()
  {
    if (!planning_requested_.exchange(false)) {
      return;
    }
    if (planning_active_.exchange(true)) {
      return;
    }
    Eigen::Vector3d box_center;
    {
      std::lock_guard<std::mutex> lock(box_mutex_);
      box_center = box_center_;
    }
    const uint64_t generation = ++generation_;
    std::string base_pose_error;
    const bool base_pose_available = refreshPlanarRootFromTf(&base_pose_error);
    publishPlanningStarted(generation, box_center);
    publishStatus("CALCULATING", true);
    RCLCPP_INFO(
      get_logger(),
      "[%llu] calculation started: box_center=[%.3f, %.3f, %.3f]",
      static_cast<unsigned long long>(generation),
      box_center.x(), box_center.y(), box_center.z());

    TaskResult result;
    if (!base_pose_available) {
      result.success = false;
      result.failure_stage = "base_pose";
      result.failure_reason = base_pose_error;
    } else {
      try {
        result = task_mode_ == "state_transition" ? planStateTransition(box_center) :
          (task_mode_ == "contact_reachability" ? planContactReachability(box_center) :
            (task_mode_ == "paper_rrt_vamp_benchmark" ? planPaperBenchmark(box_center) :
            (task_mode_ == "paper_adaptive_multistage_benchmark" ?
              planPaperAdaptiveMultistageBenchmark(box_center) :
            (task_mode_ == "paper_shared_ik_fullflow_benchmark" ?
              planPaperSharedIkFullflowBenchmark(box_center, false) :
            (task_mode_ == "paper_sphere_world_fullflow_benchmark" ?
              planPaperSharedIkFullflowBenchmark(box_center, true) : planTask(box_center))))));
      } catch (const std::exception& error) {
        result.success = false;
        result.failure_stage = "exception";
        result.failure_reason = error.what();
      }
    }
    publishTaskResult(generation, box_center, result);
    if (!result.frames.empty()) {
      std::lock_guard<std::mutex> lock(display_mutex_);
      playback_frames_ = result.frames;
      playback_index_ = 0;
    }
    if (result.success) {
      std::ostringstream status;
      status << "SUCCESS total=" << std::fixed << std::setprecision(1)
             << result.total_ms << "ms";
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

  Eigen::Matrix3d contactRotation() const
  {
    const double signed_roll = side_ == "right" ? -contact_tool_roll_ : contact_tool_roll_;
    const Eigen::Matrix3d face_normal_rotation = grasp_mode_ == "top_suction" ?
      Eigen::AngleAxisd(kPi, Eigen::Vector3d::UnitX()).toRotationMatrix() :
      Eigen::AngleAxisd(kPi / 2.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
    return face_normal_rotation *
      Eigen::AngleAxisd(signed_roll, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  }

  Eigen::Isometry3d contactPose(const Eigen::Vector3d& box_center) const
  {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    if (grasp_mode_ == "top_suction") {
      pose.translation() = box_center + Eigen::Vector3d(
        top_suction_x_offset_, 0.0, box_height_ * 0.5);
    } else {
      pose.translation() = box_center + Eigen::Vector3d(
        -box_depth_ * 0.5, front_suction_y_offset_, front_suction_z_offset_);
    }
    pose.linear() = contactRotation();
    return pose;
  }

  Eigen::Vector3d contactNormal() const
  {
    if (grasp_mode_ == "top_suction") {
      return Eigen::Vector3d(0.0, 0.0, -1.0);
    }
    return Eigen::Vector3d::UnitX();
  }

  Eigen::Vector3d toolToBoxCenter() const
  {
    const Eigen::Vector3d offset_world = grasp_mode_ == "top_suction" ?
      Eigen::Vector3d(-top_suction_x_offset_, 0.0, -box_height_ * 0.5) :
      Eigen::Vector3d(
        box_depth_ * 0.5, -front_suction_y_offset_, -front_suction_z_offset_);
    return contactRotation().transpose() * offset_world;
  }

  Eigen::Matrix3d toolToBoxRotation() const
  {
    return contactRotation().transpose();
  }

  Eigen::Vector3d carriedBoxSizeInTool() const
  {
    return Eigen::Vector3d(box_depth_, box_width_, box_height_);
  }

  Eigen::Isometry3d precontactPose(const Eigen::Vector3d& box_center) const
  {
    Eigen::Isometry3d pose = contactPose(box_center);
    pose.translation() -= contactNormal() * approach_distance_;
    return pose;
  }

  Eigen::Isometry3d retreatPose(const Eigen::Vector3d& box_center) const
  {
    Eigen::Isometry3d pose = contactPose(box_center);
    pose.translation() -= contactNormal() * retreat_distance_;
    return pose;
  }

  Eigen::Isometry3d placeTcpPose() const
  {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(
      place_tcp_pose_[0], place_tcp_pose_[1], place_tcp_pose_[2]);
    pose.linear() = Eigen::Quaterniond(
      place_tcp_pose_[6], place_tcp_pose_[3], place_tcp_pose_[4], place_tcp_pose_[5])
      .toRotationMatrix();
    return pose;
  }

  bool hasPlaceGoal() const
  {
    return !place_tcp_pose_.empty() || !place_arm_joints_deg_.empty();
  }

  std::vector<Eigen::Vector3d> obstacleBoxCenters(const Eigen::Vector3d& target) const
  {
    if (!full_box_wall_scene_) {
      return {
        target + Eigen::Vector3d(0.0, box_width_, 0.0),
        target - Eigen::Vector3d(0.0, box_width_, 0.0),
        target + Eigen::Vector3d(0.0, 0.0, box_height_),
        target - Eigen::Vector3d(0.0, 0.0, box_height_),
      };
    }

    std::vector<Eigen::Vector3d> centers;
    centers.reserve(static_cast<size_t>(box_grid_columns_ * box_grid_rows_ - 1));
    for (int row = 0; row < box_grid_rows_; ++row) {
      const double z = box_grid_bottom_z_ + (static_cast<double>(row) + 0.5) * box_height_;
      for (int column = 0; column < box_grid_columns_; ++column) {
        const int box_id =
          (box_grid_rows_ - 1 - row) * box_grid_columns_ + column + 1;
        if (box_id == target_box_id_ || removed_box_ids_.count(box_id) != 0U) {
          continue;
        }
        const double y = box_grid_center_y_ +
          (0.5 * static_cast<double>(box_grid_columns_ - 1) - static_cast<double>(column)) *
          box_width_;
        const Eigen::Vector3d center(target.x(), y, z);
        if ((center - target).cwiseAbs().maxCoeff() < 1.0e-6) {
          continue;
        }
        centers.push_back(center);
      }
    }
    return centers;
  }

  std::vector<SceneBox> warehousePanels() const
  {
    if (!warehouse_enabled_) {
      return {};
    }
    const double center_x = warehouse_opening_x_ + warehouse_length_ * 0.5;
    const double center_z = warehouse_floor_z_ + warehouse_height_ * 0.5;
    const double half_thickness = warehouse_wall_thickness_ * 0.5;
    return {
      SceneBox{
        "warehouse_left_wall",
        Eigen::Vector3d(
          center_x, warehouse_center_y_ + warehouse_width_ * 0.5 + half_thickness, center_z),
        Eigen::Vector3d(warehouse_length_, warehouse_wall_thickness_, warehouse_height_)},
      SceneBox{
        "warehouse_right_wall",
        Eigen::Vector3d(
          center_x, warehouse_center_y_ - warehouse_width_ * 0.5 - half_thickness, center_z),
        Eigen::Vector3d(warehouse_length_, warehouse_wall_thickness_, warehouse_height_)},
      SceneBox{
        "warehouse_ceiling",
        Eigen::Vector3d(
          center_x, warehouse_center_y_,
          warehouse_floor_z_ + warehouse_height_ + half_thickness),
        Eigen::Vector3d(
          warehouse_length_, warehouse_width_ + 2.0 * warehouse_wall_thickness_,
          warehouse_wall_thickness_)},
      SceneBox{
        "warehouse_rear_wall",
        Eigen::Vector3d(
          warehouse_opening_x_ + warehouse_length_ + half_thickness,
          warehouse_center_y_, center_z),
        Eigen::Vector3d(
          warehouse_wall_thickness_, warehouse_width_ + 2.0 * warehouse_wall_thickness_,
          warehouse_height_)},
    };
  }

  planning_scene::PlanningScenePtr makeScene(const Eigen::Vector3d& box_center) const
  {
    auto scene = std::make_shared<planning_scene::PlanningScene>(robot_model_);
    scene->setCurrentState(*initial_state_);
    if (ignore_opposite_arm_) {
      const std::string opposite_side = side_ == "left" ? "right" : "left";
      const auto* opposite_group = robot_model_->getJointModelGroup(opposite_side + "_arm");
      if (!opposite_group) {
        throw std::runtime_error("missing opposite arm group");
      }
      auto& acm = scene->getAllowedCollisionMatrixNonConst();
      for (const auto& tested_link : planning_group_->getLinkModelNames()) {
        for (const auto& opposite_link : opposite_group->getLinkModelNames()) {
          acm.setEntry(tested_link, opposite_link, true);
        }
      }
      for (const auto& opposite_link : opposite_group->getLinkModelNames()) {
        acm.setEntry(kCarriedBoxId, opposite_link, true);
      }
    }
    if (ground_enabled_) {
      moveit_msgs::msg::CollisionObject ground;
      ground.header.frame_id = world_frame_;
      ground.id = kGroundId;
      ground.operation = moveit_msgs::msg::CollisionObject::ADD;
      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
      primitive.dimensions = {ground_size_x_, ground_size_y_, ground_thickness_};
      geometry_msgs::msg::Pose pose;
      pose.position.z = ground_surface_z_ - ground_clearance_ - ground_thickness_ * 0.5;
      pose.orientation.w = 1.0;
      ground.primitives.push_back(primitive);
      ground.primitive_poses.push_back(pose);
      if (!scene->processCollisionObjectMsg(ground)) {
        throw std::runtime_error("failed to add ground to planning scene");
      }
      scene->getAllowedCollisionMatrixNonConst().setEntry(kGroundId, "model_base", true);
    }
    for (const auto& panel : warehousePanels()) {
      moveit_msgs::msg::CollisionObject object;
      object.header.frame_id = world_frame_;
      object.id = panel.id;
      object.operation = moveit_msgs::msg::CollisionObject::ADD;
      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
      primitive.dimensions = {panel.size.x(), panel.size.y(), panel.size.z()};
      geometry_msgs::msg::Pose pose;
      pose.position.x = panel.center.x();
      pose.position.y = panel.center.y();
      pose.position.z = panel.center.z();
      pose.orientation.w = 1.0;
      object.primitives.push_back(primitive);
      object.primitive_poses.push_back(pose);
      if (!scene->processCollisionObjectMsg(object)) {
        throw std::runtime_error("failed to add " + panel.id + " to planning scene");
      }
    }
    auto neighbors = obstacleBoxCenters(box_center);
    const double depth = std::max(0.001, box_depth_ - 2.0 * collision_inset_);
    const double width = std::max(0.001, box_width_ - 2.0 * collision_inset_);
    const double height = std::max(0.001, box_height_ - 2.0 * collision_inset_);
    for (size_t index = 0; index < neighbors.size(); ++index) {
      moveit_msgs::msg::CollisionObject object;
      object.header.frame_id = world_frame_;
      object.id = "neighbor_box_" + std::to_string(index);
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
    return scene;
  }

  void attachCarriedBox(moveit::core::RobotState& state) const
  {
    if (state.hasAttachedBody(kCarriedBoxId)) {
      return;
    }
    const Eigen::Vector3d local_size = carriedBoxSizeInTool();
    const double local_x = std::max(0.001, local_size.x() - 2.0 * collision_inset_);
    const double local_y = std::max(0.001, local_size.y() - 2.0 * collision_inset_);
    const double local_z = std::max(0.001, local_size.z() - 2.0 * collision_inset_);
    std::vector<shapes::ShapeConstPtr> shapes;
    shapes.push_back(std::make_shared<shapes::Box>(local_x, local_y, local_z));
    EigenSTL::vector_Isometry3d shape_poses;
    Eigen::Isometry3d shape_pose = Eigen::Isometry3d::Identity();
    shape_pose.translation() = toolToBoxCenter();
    shape_pose.linear() = toolToBoxRotation();
    shape_poses.push_back(shape_pose);
    const std::string prefix = side_ + "_";
    state.attachBody(
      kCarriedBoxId,
      Eigen::Isometry3d::Identity(),
      shapes,
      shape_poses,
      std::vector<std::string>{tool_link_, prefix + "link7", prefix + "link6"},
      tool_link_);
    state.update(true);
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
    if (state.hasAttachedBody(kCarriedBoxId)) {
      if (!carriedBoxUpright(state)) return "carried_box_tilt_limit";
      if (!carriedBoxWithinHeight(state)) return "carried_box_height_limit";
    }
    const auto started = std::chrono::steady_clock::now();
    const std::string reason = alfa_robot::motion::scene_collision_reason(
      scene, state, planning_group_);
    if (metrics) {
      ++metrics->collision_checks;
      metrics->collision_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    }
    return reason;
  }

  std::string fullCollisionReason(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit::core::RobotState& state,
    PlanningMetrics* metrics) const
  {
    if (state.hasAttachedBody(kCarriedBoxId)) {
      if (!carriedBoxUpright(state)) return "carried_box_tilt_limit";
      if (!carriedBoxWithinHeight(state)) return "carried_box_height_limit";
    }
    const auto started = std::chrono::steady_clock::now();
    const std::string reason = alfa_robot::motion::scene_collision_reason(
      scene, state, nullptr);
    if (metrics) {
      ++metrics->collision_checks;
      metrics->collision_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    }
    return reason;
  }

  double carriedBoxTilt(const moveit::core::RobotState& state) const
  {
    const Eigen::Vector3d up_in_tool = toolToBoxRotation() * Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d up_world =
      state.getGlobalLinkTransform(tool_link_).linear() * up_in_tool;
    return std::acos(std::clamp(up_world.z(), -1.0, 1.0));
  }

  bool carriedBoxUpright(const moveit::core::RobotState& state) const
  {
    return carriedBoxTilt(state) <= maximum_carried_box_tilt_ + 1e-8;
  }

  bool carriedBoxWithinHeight(const moveit::core::RobotState& state) const
  {
    const auto& tool = state.getGlobalLinkTransform(tool_link_);
    const double center_z = (tool * toolToBoxCenter()).z();
    return center_z <= maximum_carried_box_center_z_ + 1e-8;
  }

  bool carriedBoxFeasible(const moveit::core::RobotState& state) const
  {
    return carriedBoxUpright(state) && carriedBoxWithinHeight(state);
  }

  bool edgeClear(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit::core::RobotState& from,
    const moveit::core::RobotState& to,
    bool attached,
    PlanningMetrics* metrics,
    std::string* reason) const
  {
    const auto from_joints = armJoints(from);
    const auto to_joints = armJoints(to);
    double maximum_delta = 0.0;
    for (size_t index = 0U; index < from_joints.size(); ++index) {
      const double delta = pure_rrt_full_flow_ ?
        to_joints[index] - from_joints[index] :
        normalizedAngle(to_joints[index] - from_joints[index]);
      maximum_delta = std::max(maximum_delta, std::abs(delta));
    }
    const size_t steps = std::max<size_t>(
      1, static_cast<size_t>(std::ceil(maximum_delta / edge_joint_resolution_)));
    for (size_t step = 1; step <= steps; ++step) {
      const double ratio = static_cast<double>(step) / static_cast<double>(steps);
      moveit::core::RobotState probe(from);
      if (pure_rrt_full_flow_) {
        std::array<double, 7> interpolated{};
        for (size_t index = 0; index < interpolated.size(); ++index) {
          interpolated[index] = from_joints[index] +
            (to_joints[index] - from_joints[index]) * ratio;
        }
        probe.setJointGroupPositions(planning_group_, interpolated.data());
        probe.zeroVelocities();
        probe.zeroAccelerations();
      } else {
        std::array<double, 7> interpolated{};
        for (size_t index = 0; index < interpolated.size(); ++index) {
          interpolated[index] = from_joints[index] +
            normalizedAngle(to_joints[index] - from_joints[index]) * ratio;
        }
        probe.setJointGroupPositions(planning_group_, interpolated.data());
      }
      if (attached) {
        attachCarriedBox(probe);
      } else if (probe.hasAttachedBody(kCarriedBoxId)) {
        probe.clearAttachedBody(kCarriedBoxId);
      }
      probe.update(true);
      if (!probe.satisfiesBounds(planning_group_)) {
        if (reason) {
          *reason = "joint_bounds";
          bool detailed = false;
          for (const auto& variable : planning_group_->getVariableNames()) {
            const auto& bounds = robot_model_->getVariableBounds(variable);
            const double value = probe.getVariablePosition(variable);
            if (bounds.position_bounded_ &&
                (value < bounds.min_position_ || value > bounds.max_position_)) {
              *reason += ":" + variable + "=" + std::to_string(value) +
                " not_in[" + std::to_string(bounds.min_position_) + "," +
                std::to_string(bounds.max_position_) + "]";
              detailed = true;
              break;
            }
          }
          if (!detailed) {
            *reason += ":group_rejected_values=";
            for (const auto& variable : planning_group_->getVariableNames()) {
              *reason += variable + "=" +
                std::to_string(probe.getVariablePosition(variable)) + ",";
            }
          }
        }
        return false;
      }
      const std::string collision = collisionReason(scene, probe, metrics);
      if (!collision.empty()) {
        if (reason) *reason = collision;
        return false;
      }
    }
    return true;
  }

  std::vector<AnalyticCandidate> solvePoseCandidates(
    const Eigen::Isometry3d& target_world,
    const moveit::core::RobotState& seed_state,
    bool attached,
    const planning_scene::PlanningSceneConstPtr& scene,
    PlanningMetrics* metrics,
    bool enforce_step,
    std::string* rejection_summary,
    const std::array<double, 7>* previous_joint_delta = nullptr,
    const double* preferred_swivel = nullptr,
    bool preferred_swivel_only = false) const
  {
    const Eigen::Isometry3d world_to_arm_base =
      seed_state.getGlobalLinkTransform(arm_base_link_).inverse();
    const Eigen::Isometry3d target_in_arm_base = world_to_arm_base * target_world;
    const auto seed_joints = armJoints(seed_state);
    const double raw_seed_swivel = solver_->swivelAngle(seed_joints);
    const double seed_swivel = std::isfinite(raw_seed_swivel) ? raw_seed_swivel : 0.0;
    const auto lower_limits = solver_->jointLowerLimits();
    const auto upper_limits = solver_->jointUpperLimits();
    std::vector<AnalyticCandidate> candidates;
    size_t bounds_rejects = 0;
    size_t jump_rejects = 0;
    size_t fk_rejects = 0;
    size_t collision_rejects = 0;
    size_t edge_rejects = 0;
    std::string last_collision;

    std::vector<double> swivel_samples;
    const auto add_swivel_sample = [&swivel_samples](double sample) {
      const double normalized = normalizedAngle(sample);
      const bool duplicate = std::any_of(
        swivel_samples.begin(), swivel_samples.end(),
        [normalized](double existing) {
          return std::abs(normalizedAngle(existing - normalized)) < 1e-8;
        });
      if (!duplicate) {
        swivel_samples.push_back(normalized);
      }
    };
    if (preferred_swivel) {
      add_swivel_sample(*preferred_swivel);
    }
    if (!preferred_swivel_only) {
      if (natural_motion_enabled_ && natural_seed_swivel_sampling_ && enforce_step) {
        add_swivel_sample(seed_swivel);
        for (int neighbor = 1; neighbor <= natural_seed_swivel_neighbor_steps_; ++neighbor) {
          const double offset = static_cast<double>(neighbor) * natural_seed_swivel_step_;
          add_swivel_sample(seed_swivel - offset);
          add_swivel_sample(seed_swivel + offset);
        }
      }
      const int intervals = std::max(1, static_cast<int>(std::ceil(2.0 * kPi / psi_step_)));
      for (int index = 0; index < intervals; ++index) {
        add_swivel_sample(-kPi + static_cast<double>(index) * 2.0 * kPi / intervals);
      }
    }
    for (const double swivel_sample : swivel_samples) {
      V3RedundantIkRequest request;
      request.target_in_arm_base = target_in_arm_base;
      request.swivel_angle = swivel_sample;
      request.seed = seed_joints;
      const auto started = std::chrono::steady_clock::now();
      const auto solutions = solver_->solveInArmBase(request);
      if (metrics) {
        ++metrics->ik_calls;
        metrics->ik_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count();
      }
      for (const auto& solution : solutions) {
        if (enforce_step) {
          bool step_too_large =
            maximumJointDelta(seed_joints, solution.joints) > maximum_cartesian_joint_step_;
          if (natural_motion_enabled_) {
            for (size_t joint_index = 0; joint_index < solution.joints.size(); ++joint_index) {
              const double limit = joint_index < 4U ?
                natural_max_proximal_step_ : natural_max_wrist_step_;
              if (std::abs(normalizedAngle(
                    solution.joints[joint_index] - seed_joints[joint_index])) > limit) {
                step_too_large = true;
                break;
              }
            }
          }
          if (step_too_large) {
            ++jump_rejects;
            continue;
          }
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
        const auto& actual_tcp = candidate.getGlobalLinkTransform(tool_link_);
        if ((actual_tcp.translation() - target_world.translation()).norm() > 0.001 ||
            Eigen::Quaterniond(actual_tcp.linear()).angularDistance(
              Eigen::Quaterniond(target_world.linear())) > degToRad(0.5)) {
          ++fk_rejects;
          continue;
        }
        const std::string collision = collisionReason(scene, candidate, metrics);
        if (!collision.empty()) {
          ++collision_rejects;
          last_collision = collision;
          continue;
        }
        if (enforce_step) {
          std::string edge_reason;
          if (!edgeClear(scene, seed_state, candidate, attached, metrics, &edge_reason)) {
            ++edge_rejects;
            last_collision = edge_reason;
            continue;
          }
        }
        const bool duplicate = std::any_of(
          candidates.begin(), candidates.end(),
          [&solution](const AnalyticCandidate& existing) {
            return maximumJointDelta(existing.solution.joints, solution.joints) < 1e-5;
          });
        if (duplicate) {
          continue;
        }
        AnalyticCandidate output;
        output.state = std::make_shared<moveit::core::RobotState>(candidate);
        output.solution = solution;
        const double margin_penalty = 0.02 /
          std::max(0.01, solution.minimum_joint_limit_margin);
        if (natural_motion_enabled_) {
          const double swivel_delta = normalizedAngle(solution.swivel_angle - seed_swivel);
          const double wrist_sine = std::sin(solution.joints[5]);
          const double wrist_singularity_penalty =
            0.01 / (wrist_sine * wrist_sine + 0.01);
          const double wrist_neutral_penalty =
            solution.joints[4] * solution.joints[4] +
            0.25 * solution.joints[5] * solution.joints[5] +
            0.5 * solution.joints[6] * solution.joints[6];
          output.score = weightedSquaredJointDistance(
              seed_joints, solution.joints, kNaturalJointWeights) +
            natural_swivel_weight_ * swivel_delta * swivel_delta +
            natural_wrist_singularity_weight_ * wrist_singularity_penalty +
            natural_wrist_neutral_weight_ * wrist_neutral_penalty +
            natural_joint_limit_weight_ * jointLimitBarrier(
              solution.joints, lower_limits, upper_limits) +
            margin_penalty;
          if (natural_joint_wrap_weight_ > 0.0) {
            for (size_t joint_index = 0U; joint_index < solution.joints.size(); ++joint_index) {
              const double raw_delta = solution.joints[joint_index] - seed_joints[joint_index];
              const double wrap_excess = std::max(
                0.0, std::abs(raw_delta) - std::abs(normalizedAngle(raw_delta)));
              output.score += natural_joint_wrap_weight_ * wrap_excess * wrap_excess;
            }
          }
          if (previous_joint_delta && natural_joint_acceleration_weight_ > 0.0) {
            output.score += natural_joint_acceleration_weight_ *
              weightedSquaredJointDeltaError(
                jointDelta(seed_joints, solution.joints),
                *previous_joint_delta,
                kNaturalJointWeights);
          }
        } else {
          output.score = squaredJointDistance(seed_joints, solution.joints) + margin_penalty;
        }
        if (preferred_swivel) {
          const double preferred_delta = normalizedAngle(
            solution.swivel_angle - *preferred_swivel);
          output.score += 2.0 * preferred_delta * preferred_delta;
        }
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
              << " fk=" << fk_rejects
              << " collision=" << collision_rejects
              << " edge=" << edge_rejects;
      if (!last_collision.empty()) {
        summary << " last=" << last_collision;
      }
      *rejection_summary = summary.str();
    }
    return candidates;
  }

  std::optional<AnalyticCandidate> solveNextPose(
    const Eigen::Isometry3d& target_world,
    const moveit::core::RobotState& seed_state,
    bool attached,
    const planning_scene::PlanningSceneConstPtr& scene,
    PlanningMetrics* metrics,
    std::string* reason,
    const std::array<double, 7>* previous_joint_delta = nullptr,
    const double* preferred_swivel = nullptr,
    bool preferred_swivel_only = false) const
  {
    double continuation_swivel = solver_->swivelAngle(armJoints(seed_state));
    const double* fast_swivel = preferred_swivel;
    if (!fast_swivel && std::isfinite(continuation_swivel)) {
      fast_swivel = &continuation_swivel;
    }
    auto candidates = solvePoseCandidates(
      target_world, seed_state, attached, scene, metrics, true, reason,
      previous_joint_delta, fast_swivel, true);
    if (candidates.empty() && !preferred_swivel_only) {
      candidates = solvePoseCandidates(
        target_world, seed_state, attached, scene, metrics, true, reason,
        previous_joint_delta, preferred_swivel, false);
    }
    if (candidates.empty()) {
      return std::nullopt;
    }
    return candidates.front();
  }

  bool traceCartesianPath(
    const Eigen::Vector3d& box_center,
    const moveit::core::RobotState& precontact_state,
    const planning_scene::PlanningSceneConstPtr& scene,
    PlanningMetrics* metrics,
    std::vector<moveit::core::RobotStatePtr>* approach_states,
    std::vector<moveit::core::RobotStatePtr>* retreat_states,
    std::string* failure_stage,
    std::string* failure_reason) const
  {
    if (!approach_states || !retreat_states) {
      return false;
    }
    approach_states->clear();
    retreat_states->clear();
    approach_states->push_back(std::make_shared<moveit::core::RobotState>(precontact_state));
    moveit::core::RobotStatePtr current = approach_states->back();
    std::optional<std::array<double, 7>> previous_joint_delta;
    const Eigen::Isometry3d contact = contactPose(box_center);
    const Eigen::Vector3d outward = -contactNormal();
    const size_t approach_steps = std::max<size_t>(
      1, static_cast<size_t>(std::ceil(approach_distance_ / cartesian_step_)));
    for (size_t step = 1; step <= approach_steps; ++step) {
      const double ratio = static_cast<double>(step) / static_cast<double>(approach_steps);
      Eigen::Isometry3d target = contact;
      target.translation() += outward * approach_distance_ * (1.0 - ratio);
      std::string reason;
      const auto current_joints = armJoints(*current);
      auto next = solveNextPose(
        target, *current, false, scene, metrics, &reason,
        previous_joint_delta ? &*previous_joint_delta : nullptr);
      if (!next) {
        if (failure_stage) *failure_stage = "cartesian_approach";
        if (failure_reason) {
          *failure_reason = "step " + std::to_string(step) + "/" +
            std::to_string(approach_steps) + " " + reason;
        }
        return false;
      }
      previous_joint_delta = jointDelta(current_joints, armJoints(*next->state));
      current = next->state;
      approach_states->push_back(current);
    }

    moveit::core::RobotState contact_attached(*current);
    attachCarriedBox(contact_attached);
    const std::string attach_collision = collisionReason(scene, contact_attached, metrics);
    if (!attach_collision.empty()) {
      if (failure_stage) *failure_stage = "attach_box";
      if (failure_reason) *failure_reason = attach_collision;
      return false;
    }
    current = std::make_shared<moveit::core::RobotState>(contact_attached);
    retreat_states->push_back(current);
    previous_joint_delta.reset();

    const size_t retreat_steps = std::max<size_t>(
      1, static_cast<size_t>(std::ceil(retreat_distance_ / cartesian_step_)));
    for (size_t step = 1; step <= retreat_steps; ++step) {
      const double ratio = static_cast<double>(step) / static_cast<double>(retreat_steps);
      Eigen::Isometry3d target = contact;
      target.translation() += outward * retreat_distance_ * ratio;
      std::string reason;
      const auto current_joints = armJoints(*current);
      auto next = solveNextPose(
        target, *current, true, scene, metrics, &reason,
        previous_joint_delta ? &*previous_joint_delta : nullptr);
      if (!next) {
        if (failure_stage) *failure_stage = "cartesian_retreat";
        if (failure_reason) {
          *failure_reason = "step " + std::to_string(step) + "/" +
            std::to_string(retreat_steps) + " " + reason;
        }
        return false;
      }
      previous_joint_delta = jointDelta(current_joints, armJoints(*next->state));
      current = next->state;
      retreat_states->push_back(current);
    }
    return true;
  }

  bool assignCollisionFreeOppositeArm(
    const planning_scene::PlanningSceneConstPtr& scene,
    std::vector<moveit::core::RobotStatePtr>* approach_states,
    std::vector<moveit::core::RobotStatePtr>* retreat_states,
    PlanningMetrics* metrics,
    std::string* reason) const
  {
    if (!approach_states || !retreat_states || approach_states->empty() ||
        retreat_states->empty()) {
      if (reason) *reason = "opposite-arm search has no task states";
      return false;
    }
    const std::string opposite_side = side_ == "left" ? "right" : "left";
    const auto* opposite_group = robot_model_->getJointModelGroup(opposite_side + "_arm");
    if (!opposite_group || opposite_group->getVariableCount() != 7U) {
      if (reason) *reason = "missing opposite-arm group";
      return false;
    }
    const auto candidate_clear = [&](const std::vector<double>& joints) {
        for (const auto* states : {approach_states, retreat_states}) {
          for (const auto& state : *states) {
            moveit::core::RobotState probe(*state);
            probe.setJointGroupPositions(opposite_group, joints);
            probe.update(true);
            if (!probe.satisfiesBounds(opposite_group) ||
                !fullCollisionReason(scene, probe, metrics).empty()) {
              return false;
            }
          }
        }
        return true;
      };
    const auto apply_candidate = [&](const std::vector<double>& joints) {
        for (auto* states : {approach_states, retreat_states}) {
          for (auto& state : *states) {
            state->setJointGroupPositions(opposite_group, joints);
            state->update(true);
          }
        }
      };

    std::vector<double> initial_joints;
    approach_states->front()->copyJointGroupPositions(opposite_group, initial_joints);
    if (candidate_clear(initial_joints)) {
      apply_candidate(initial_joints);
      return true;
    }
    constexpr size_t kOppositeArmSamples = 2048U;
    constexpr size_t kOppositeArmAcceptedTarget = 32U;
    size_t accepted = 0U;
    double best_score = std::numeric_limits<double>::infinity();
    std::vector<double> best_joints;
    for (size_t sample = 0U; sample < kOppositeArmSamples; ++sample) {
      moveit::core::RobotState candidate(*approach_states->front());
      candidate.setToRandomPositions(opposite_group);
      std::vector<double> joints;
      candidate.copyJointGroupPositions(opposite_group, joints);
      if (!candidate_clear(joints)) continue;
      ++accepted;
      double score = 0.0;
      for (size_t index = 0U; index < joints.size(); ++index) {
        const double delta = normalizedAngle(joints[index] - initial_joints[index]);
        score += delta * delta;
      }
      if (score < best_score) {
        best_score = score;
        best_joints = std::move(joints);
      }
      if (accepted >= kOppositeArmAcceptedTarget) break;
    }
    if (!best_joints.empty()) {
      apply_candidate(best_joints);
      return true;
    }
    if (reason) {
      *reason = "no fixed opposite-arm pose cleared all task states after " +
        std::to_string(kOppositeArmSamples) + " samples";
    }
    return false;
  }

  std::vector<moveit::core::RobotStatePtr> densifyReplay(
    const std::vector<moveit::core::RobotStatePtr>& states,
    double maximum_step) const
  {
    if (!natural_motion_enabled_ || states.size() < 2U || maximum_step <= 0.0) {
      return states;
    }
    std::vector<moveit::core::RobotStatePtr> dense;
    dense.push_back(std::make_shared<moveit::core::RobotState>(*states.front()));
    for (size_t segment = 1; segment < states.size(); ++segment) {
      const auto from_joints = armJoints(*states[segment - 1]);
      const auto to_joints = armJoints(*states[segment]);
      double maximum_delta = 0.0;
      for (size_t index = 0U; index < from_joints.size(); ++index) {
        const double delta = pure_rrt_full_flow_ ?
          to_joints[index] - from_joints[index] :
          normalizedAngle(to_joints[index] - from_joints[index]);
        maximum_delta = std::max(maximum_delta, std::abs(delta));
      }
      const size_t steps = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(maximum_delta / maximum_step)));
      for (size_t step = 1; step <= steps; ++step) {
        const double ratio = static_cast<double>(step) / static_cast<double>(steps);
        auto state = std::make_shared<moveit::core::RobotState>(*states[segment - 1]);
        if (pure_rrt_full_flow_) {
          std::array<double, 7> interpolated{};
          for (size_t index = 0; index < interpolated.size(); ++index) {
            interpolated[index] = from_joints[index] +
              (to_joints[index] - from_joints[index]) * ratio;
          }
          state->setJointGroupPositions(planning_group_, interpolated.data());
          state->zeroVelocities();
          state->zeroAccelerations();
        } else {
          std::array<double, 7> interpolated{};
          for (size_t index = 0; index < interpolated.size(); ++index) {
            interpolated[index] = from_joints[index] +
              normalizedAngle(to_joints[index] - from_joints[index]) * ratio;
          }
          state->setJointGroupPositions(planning_group_, interpolated.data());
        }
        state->update(true);
        dense.push_back(std::move(state));
      }
    }
    return dense;
  }

  std::vector<moveit::core::RobotStatePtr> densifyRrtReplay(
    const std::vector<moveit::core::RobotStatePtr>& states) const
  {
    return densifyReplay(states, natural_rrt_replay_step_);
  }

  double pathJointTravel(
    const std::vector<moveit::core::RobotStatePtr>& states) const
  {
    double travel = 0.0;
    for (size_t index = 1U; index < states.size(); ++index) {
      travel += weightedJointTravel(
        armJoints(*states[index - 1U]), armJoints(*states[index]), kNaturalJointWeights);
    }
    return travel;
  }

  std::vector<moveit::core::RobotStatePtr> densifyCartesianReplay(
    const std::vector<moveit::core::RobotStatePtr>& states) const
  {
    return densifyReplay(states, natural_cartesian_replay_step_);
  }

  bool traceUpdownTransition(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit::core::RobotState& start_state,
    double target_updown,
    bool attached,
    PlanningMetrics* metrics,
    std::vector<moveit::core::RobotStatePtr>* states,
    std::string* reason) const
  {
    if (!states || !robot_model_->hasJointModel("updown")) {
      if (reason) *reason = "updown joint is unavailable";
      return false;
    }
    const double start_updown = start_state.getVariablePosition("updown");
    const size_t steps = std::max<size_t>(
      1, static_cast<size_t>(std::ceil(std::abs(target_updown - start_updown) / 0.02)));
    states->clear();
    states->reserve(steps + 1U);
    states->push_back(std::make_shared<moveit::core::RobotState>(start_state));
    for (size_t step = 1; step <= steps; ++step) {
      auto state = std::make_shared<moveit::core::RobotState>(start_state);
      state->setVariablePosition(
        "updown",
        start_updown + (target_updown - start_updown) *
        static_cast<double>(step) / static_cast<double>(steps));
      if (attached) {
        attachCarriedBox(*state);
      } else if (state->hasAttachedBody(kCarriedBoxId)) {
        state->clearAttachedBody(kCarriedBoxId);
      }
      state->update(true);
      if (!state->satisfiesBounds()) {
        if (reason) *reason = "updown transition violates joint bounds";
        return false;
      }
      const std::string collision = fullCollisionReason(scene, *state, metrics);
      if (!collision.empty()) {
        if (reason) {
          *reason = "updown transition step " + std::to_string(step) + "/" +
            std::to_string(steps) + " " + collision;
        }
        return false;
      }
      states->push_back(std::move(state));
    }
    return true;
  }

  bool traceToolSpaceSegment(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit::core::RobotState& start_state,
    const moveit::core::RobotState& goal_state,
    bool attached,
    PlanningMetrics* metrics,
    std::vector<moveit::core::RobotStatePtr>* states,
    std::string* reason,
    bool exhaustive_redundancy = false) const
  {
    if (!states) return false;
    states->clear();
    states->push_back(std::make_shared<moveit::core::RobotState>(start_state));
    const Eigen::Isometry3d start_pose = start_state.getGlobalLinkTransform(tool_link_);
    const Eigen::Isometry3d goal_pose = goal_state.getGlobalLinkTransform(tool_link_);
    const Eigen::Quaterniond start_rotation(start_pose.linear());
    const Eigen::Quaterniond goal_rotation(goal_pose.linear());
    const double translation = (goal_pose.translation() - start_pose.translation()).norm();
    const double rotation = start_rotation.angularDistance(goal_rotation);
    const double start_swivel = solver_->swivelAngle(armJoints(start_state));
    const double goal_swivel = solver_->swivelAngle(armJoints(goal_state));
    if (!std::isfinite(start_swivel) || !std::isfinite(goal_swivel)) {
      if (reason) *reason = "Cartesian endpoint swivel is undefined";
      states->clear();
      return false;
    }
    const double swivel_delta = normalizedAngle(goal_swivel - start_swivel);
    const size_t steps = std::max<size_t>({
      1U,
      static_cast<size_t>(std::ceil(translation / cartesian_transfer_translation_step_)),
      static_cast<size_t>(std::ceil(rotation / cartesian_transfer_rotation_step_)),
    });
    moveit::core::RobotStatePtr current = states->front();
    std::optional<std::array<double, 7>> previous_joint_delta;
    for (size_t step = 1U; step < steps; ++step) {
      const double ratio = static_cast<double>(step) / static_cast<double>(steps);
      Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
      target.translation() = start_pose.translation() +
        (goal_pose.translation() - start_pose.translation()) * ratio;
      target.linear() = start_rotation.slerp(ratio, goal_rotation).normalized().toRotationMatrix();
      const auto current_joints = armJoints(*current);
      const double target_swivel = normalizedAngle(start_swivel + swivel_delta * ratio);
      std::string step_reason;
      auto next = solveNextPose(
        target, *current, attached, scene, metrics, &step_reason,
        previous_joint_delta ? &*previous_joint_delta : nullptr,
        &target_swivel, !exhaustive_redundancy);
      if (!next && !exhaustive_redundancy) {
        std::string fallback_reason;
        next = solveNextPose(
          target, *current, attached, scene, metrics, &fallback_reason,
          previous_joint_delta ? &*previous_joint_delta : nullptr,
          &target_swivel, false);
        if (!next) {
          step_reason += "; redundant-angle fallback: " + fallback_reason;
        }
      }
      if (!next) {
        if (reason) {
          *reason = "Cartesian step " + std::to_string(step) + "/" +
            std::to_string(steps) + " " + step_reason;
        }
        states->clear();
        return false;
      }
      previous_joint_delta = jointDelta(current_joints, armJoints(*next->state));
      current = next->state;
      states->push_back(current);
    }
    std::string final_reason;
    const auto current_joints = armJoints(*current);
    const auto goal_joints = armJoints(goal_state);
    for (size_t index = 0; index < goal_joints.size(); ++index) {
      const double limit = natural_motion_enabled_ ?
        (index < 4U ? natural_max_proximal_step_ : natural_max_wrist_step_) :
        maximum_cartesian_joint_step_;
      if (std::abs(normalizedAngle(goal_joints[index] - current_joints[index])) > limit) {
        if (reason) *reason = "Cartesian final edge exceeds natural joint step";
        states->clear();
        return false;
      }
    }
    if (!edgeClear(scene, *current, goal_state, attached, metrics, &final_reason)) {
      if (reason) *reason = "Cartesian final edge " + final_reason;
      states->clear();
      return false;
    }
    states->push_back(std::make_shared<moveit::core::RobotState>(goal_state));
    return true;
  }

  bool searchToolSpaceWaypointPath(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit::core::RobotState& start_state,
    const moveit::core::RobotState& goal_state,
    bool attached,
    PlanningMetrics* metrics,
    size_t maximum_attempts,
    size_t* search_attempts,
    std::vector<moveit::core::RobotStatePtr>* states,
    std::string* reason) const
  {
    if (!states || !search_attempts) return false;
    states->clear();
    const Eigen::Isometry3d start_pose = start_state.getGlobalLinkTransform(tool_link_);
    const Eigen::Isometry3d goal_pose = goal_state.getGlobalLinkTransform(tool_link_);
    const Eigen::Quaterniond start_rotation(start_pose.linear());
    const Eigen::Quaterniond goal_rotation(goal_pose.linear());
    const double start_swivel = solver_->swivelAngle(armJoints(start_state));
    const double goal_swivel = solver_->swivelAngle(armJoints(goal_state));
    if (!std::isfinite(start_swivel) || !std::isfinite(goal_swivel)) {
      if (reason) *reason = "Cartesian waypoint endpoint swivel is undefined";
      return false;
    }
    const double side_sign = side_ == "left" ? 1.0 : -1.0;
    std::vector<std::pair<Eigen::Vector3d, double>> waypoint_specs = {
      // The wall normal is +X: after suction, pull away along -X and fold
      // inward toward the body centerline (-side*Y). The legacy signs sent
      // the carried box back toward the wall and farther outward.
      {start_pose.translation() + Eigen::Vector3d(-0.10, 0.0, 0.0), 0.0},
      {start_pose.translation() + Eigen::Vector3d(-0.20, 0.0, 0.0), 0.0},
      {start_pose.translation() + Eigen::Vector3d(-0.10, -side_sign * 0.10, 0.0), 0.0},
      {start_pose.translation() + Eigen::Vector3d(-0.20, -side_sign * 0.20, 0.0), 0.0},
      {start_pose.translation() + Eigen::Vector3d(-0.20, -side_sign * 0.25, -0.05), 0.0},
      {start_pose.translation() + Eigen::Vector3d(-0.25, -side_sign * 0.20, -0.10), 0.1},
    };
    const std::vector<std::pair<Eigen::Vector3d, double>> goal_waypoint_specs = {
      {goal_pose.translation() + Eigen::Vector3d(-0.20, 0.0, 0.0), 1.0},
      {goal_pose.translation() + Eigen::Vector3d(-0.30, 0.0, 0.0), 1.0},
      {{goal_pose.translation().x(), goal_pose.translation().y(), start_pose.translation().z()},
        1.0},
      {{goal_pose.translation().x() - 0.10,
        goal_pose.translation().y() - side_sign * 0.05,
        start_pose.translation().z()}, 1.0},
      {goal_pose.translation() + Eigen::Vector3d(-0.20, -side_sign * 0.10, -0.20), 0.8},
      {goal_pose.translation() + Eigen::Vector3d(-0.20, -side_sign * 0.10, -0.20), 1.0},
      {goal_pose.translation() + Eigen::Vector3d(-0.25, -side_sign * 0.20, -0.25), 0.8},
      {goal_pose.translation() + Eigen::Vector3d(-0.10, 0.0, -0.30), 1.0},
      {goal_pose.translation() + Eigen::Vector3d(-0.20, side_sign * 0.10, -0.10), 1.0},
    };
    std::string last_reason = "no waypoint candidate";
    size_t waypoint_ik_successes = 0U;
    size_t first_segment_successes = 0U;
    size_t goal_waypoint_ik_successes = 0U;
    size_t middle_segment_successes = 0U;
    std::string middle_success_description;
    for (const auto& [waypoint_position, ratio] : waypoint_specs) {
        if (maximum_attempts > 0U && *search_attempts >= maximum_attempts) {
          if (reason) *reason = "Cartesian waypoint search budget exhausted: " + last_reason;
          return !states->empty();
        }
        ++*search_attempts;
        if (metrics) ++metrics->cartesian_transfer_attempts;
        Eigen::Isometry3d waypoint_pose = Eigen::Isometry3d::Identity();
        waypoint_pose.translation() = waypoint_position;
        waypoint_pose.linear() = start_rotation.slerp(
          ratio, goal_rotation).normalized().toRotationMatrix();
        const double waypoint_swivel = normalizedAngle(
          start_swivel + normalizedAngle(goal_swivel - start_swivel) * ratio);
        std::string waypoint_reason;
        auto waypoint_candidates = solvePoseCandidates(
          waypoint_pose, start_state, attached, scene, metrics, false,
          &waypoint_reason, nullptr, &waypoint_swivel, true);
        if (waypoint_candidates.empty()) {
          waypoint_candidates = solvePoseCandidates(
            waypoint_pose, start_state, attached, scene, metrics, false,
            &waypoint_reason, nullptr, &waypoint_swivel, false);
        }
        if (waypoint_candidates.empty()) {
          last_reason = "waypoint IK: " + waypoint_reason;
          continue;
        }
        ++waypoint_ik_successes;
        for (size_t candidate_index = 0U;
            candidate_index < std::min<size_t>(waypoint_candidates.size(), 2U);
            ++candidate_index) {
          std::vector<moveit::core::RobotStatePtr> first;
          std::vector<moveit::core::RobotStatePtr> second;
          std::string segment_reason;
          if (!traceToolSpaceSegment(
                scene, start_state, *waypoint_candidates[candidate_index].state,
                attached, metrics, &first, &segment_reason)) {
            last_reason = "to waypoint: " + segment_reason;
            continue;
          }
          ++first_segment_successes;
          std::string direct_tail_reason;
          if (edgeClear(
                scene, *waypoint_candidates[candidate_index].state, goal_state,
                attached, metrics, &direct_tail_reason)) {
            auto direct_tail = densifyRrtReplay({
              waypoint_candidates[candidate_index].state,
              std::make_shared<moveit::core::RobotState>(goal_state),
            });
            std::vector<moveit::core::RobotStatePtr> candidate = first;
            candidate.insert(candidate.end(), direct_tail.begin() + 1U, direct_tail.end());
            if (metrics) ++metrics->joint_fallback_segments;
            *states = std::move(candidate);
            return true;
          }
          std::vector<moveit::core::RobotStatePtr> scheduled;
          std::string schedule_reason;
          if (searchToolSpaceOrientationSchedule(
                scene, *waypoint_candidates[candidate_index].state, goal_state,
                attached, metrics, &scheduled, &schedule_reason)) {
            std::vector<moveit::core::RobotStatePtr> candidate = first;
            candidate.insert(candidate.end(), scheduled.begin() + 1U, scheduled.end());
            *states = std::move(candidate);
            return true;
          }
          last_reason = "orientation schedule: " + schedule_reason;
          if (traceToolSpaceSegment(
                scene, *waypoint_candidates[candidate_index].state, goal_state,
                attached, metrics, &second, &segment_reason)) {
            std::vector<moveit::core::RobotStatePtr> candidate = first;
            candidate.insert(candidate.end(), second.begin() + 1U, second.end());
            *states = std::move(candidate);
            return true;
          }
          last_reason = "from waypoint: " + segment_reason;
          std::vector<std::pair<Eigen::Vector3d, double>> second_waypoint_specs = {
            {waypoint_pose.translation(), 0.25},
            {waypoint_pose.translation(), 0.50},
            {waypoint_pose.translation(), 0.75},
            {waypoint_pose.translation(), 1.00},
          };
          second_waypoint_specs.insert(
            second_waypoint_specs.end(),
            goal_waypoint_specs.begin(), goal_waypoint_specs.end());
          for (const auto& [goal_waypoint_position, goal_orientation_ratio] :
              second_waypoint_specs) {
            if (maximum_attempts > 0U && *search_attempts >= maximum_attempts) break;
            ++*search_attempts;
            if (metrics) ++metrics->cartesian_transfer_attempts;
            Eigen::Isometry3d goal_waypoint_pose = Eigen::Isometry3d::Identity();
            goal_waypoint_pose.translation() = goal_waypoint_position;
            goal_waypoint_pose.linear() = start_rotation.slerp(
              goal_orientation_ratio, goal_rotation).normalized().toRotationMatrix();
            const double goal_waypoint_swivel = normalizedAngle(
              start_swivel + normalizedAngle(goal_swivel - start_swivel) *
              goal_orientation_ratio);
            std::string goal_waypoint_reason;
            auto goal_waypoint_candidates = solvePoseCandidates(
              goal_waypoint_pose, *waypoint_candidates[candidate_index].state,
              attached, scene, metrics, false, &goal_waypoint_reason,
              nullptr, &goal_waypoint_swivel, false);
            if (goal_waypoint_candidates.empty()) {
              last_reason = "goal waypoint IK: " + goal_waypoint_reason;
              continue;
            }
            ++goal_waypoint_ik_successes;
            for (size_t goal_candidate_index = 0U;
                goal_candidate_index < std::min<size_t>(goal_waypoint_candidates.size(), 2U);
                ++goal_candidate_index) {
              std::vector<moveit::core::RobotStatePtr> middle;
              std::vector<moveit::core::RobotStatePtr> final;
              if (!traceToolSpaceSegment(
                    scene, *waypoint_candidates[candidate_index].state,
                    *goal_waypoint_candidates[goal_candidate_index].state,
                    attached, metrics, &middle, &segment_reason)) {
                last_reason = "between waypoints: " + segment_reason;
                continue;
              }
              ++middle_segment_successes;
              {
                std::ostringstream description;
                description << " first=[" << waypoint_pose.translation().transpose()
                            << "] first_orientation_ratio=" << ratio
                            << " second=[" << goal_waypoint_pose.translation().transpose()
                            << "] second_orientation_ratio=" << goal_orientation_ratio;
                middle_success_description = description.str();
              }
              std::vector<moveit::core::RobotStatePtr> task_space_tail;
              std::string task_space_tail_reason;
              if (searchShortcutRepairRrt(
                    scene, *goal_waypoint_candidates[goal_candidate_index].state,
                    goal_state, attached, metrics,
                    &task_space_tail, &task_space_tail_reason)) {
                std::vector<moveit::core::RobotStatePtr> candidate = first;
                candidate.insert(candidate.end(), middle.begin() + 1U, middle.end());
                candidate.insert(
                  candidate.end(), task_space_tail.begin() + 1U, task_space_tail.end());
                *states = std::move(candidate);
                return true;
              }
              last_reason = "task-space tail: " + task_space_tail_reason;
              if (edgeClear(
                    scene, *goal_waypoint_candidates[goal_candidate_index].state,
                    goal_state, attached, metrics, &direct_tail_reason)) {
                auto direct_tail = densifyRrtReplay({
                  goal_waypoint_candidates[goal_candidate_index].state,
                  std::make_shared<moveit::core::RobotState>(goal_state),
                });
                std::vector<moveit::core::RobotStatePtr> candidate = first;
                candidate.insert(candidate.end(), middle.begin() + 1U, middle.end());
                candidate.insert(
                  candidate.end(), direct_tail.begin() + 1U, direct_tail.end());
                if (metrics) ++metrics->joint_fallback_segments;
                *states = std::move(candidate);
                return true;
              }
              if (traceToolSpaceSegment(
                    scene, *goal_waypoint_candidates[goal_candidate_index].state,
                    goal_state, attached, metrics, &final, &segment_reason)) {
                std::vector<moveit::core::RobotStatePtr> candidate = first;
                candidate.insert(candidate.end(), middle.begin() + 1U, middle.end());
                candidate.insert(candidate.end(), final.begin() + 1U, final.end());
                *states = std::move(candidate);
                return true;
              }
              last_reason = "from goal waypoint: " + segment_reason;
              for (const auto& [final_waypoint_position, final_orientation_ratio] :
                  goal_waypoint_specs) {
                if (maximum_attempts > 0U && *search_attempts >= maximum_attempts) break;
                ++*search_attempts;
                if (metrics) ++metrics->cartesian_transfer_attempts;
                Eigen::Isometry3d final_waypoint_pose = Eigen::Isometry3d::Identity();
                final_waypoint_pose.translation() = final_waypoint_position;
                final_waypoint_pose.linear() = start_rotation.slerp(
                  final_orientation_ratio, goal_rotation).normalized().toRotationMatrix();
                const double final_waypoint_swivel = normalizedAngle(
                  start_swivel + normalizedAngle(goal_swivel - start_swivel) *
                  final_orientation_ratio);
                std::string final_waypoint_reason;
                auto final_waypoint_candidates = solvePoseCandidates(
                  final_waypoint_pose,
                  *goal_waypoint_candidates[goal_candidate_index].state,
                  attached, scene, metrics, false, &final_waypoint_reason,
                  nullptr, &final_waypoint_swivel, false);
                for (size_t final_candidate_index = 0U;
                    final_candidate_index < std::min<size_t>(
                      final_waypoint_candidates.size(), 2U);
                    ++final_candidate_index) {
                  std::vector<moveit::core::RobotStatePtr> third;
                  std::vector<moveit::core::RobotStatePtr> fourth;
                  if (!traceToolSpaceSegment(
                        scene, *goal_waypoint_candidates[goal_candidate_index].state,
                        *final_waypoint_candidates[final_candidate_index].state,
                        attached, metrics, &third, &segment_reason)) {
                    last_reason = "to final waypoint: " + segment_reason;
                    continue;
                  }
                  if (!traceToolSpaceSegment(
                        scene, *final_waypoint_candidates[final_candidate_index].state,
                        goal_state, attached, metrics, &fourth, &segment_reason)) {
                    last_reason = "from final waypoint: " + segment_reason;
                    continue;
                  }
                  std::vector<moveit::core::RobotStatePtr> candidate = first;
                  candidate.insert(candidate.end(), middle.begin() + 1U, middle.end());
                  candidate.insert(candidate.end(), third.begin() + 1U, third.end());
                  candidate.insert(candidate.end(), fourth.begin() + 1U, fourth.end());
                  *states = std::move(candidate);
                  return true;
                }
              }
            }
          }
        }
    }
    if (states->empty()) {
      if (reason) {
        *reason = "waypoint IK successes=" + std::to_string(waypoint_ik_successes) +
          " first-segment successes=" + std::to_string(first_segment_successes) +
          " goal-waypoint IK successes=" + std::to_string(goal_waypoint_ik_successes) +
          " middle-segment successes=" + std::to_string(middle_segment_successes) +
          middle_success_description +
          ": " + last_reason;
      }
      return false;
    }
    return true;
  }

  bool searchToolSpaceOrientationSchedule(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit::core::RobotState& start_state,
    const moveit::core::RobotState& goal_state,
    bool attached,
    PlanningMetrics* metrics,
    std::vector<moveit::core::RobotStatePtr>* states,
    std::string* reason) const
  {
    struct BeamNode
    {
      moveit::core::RobotStatePtr state;
      std::vector<moveit::core::RobotStatePtr> path;
      double orientation_ratio = 0.0;
      double score = 0.0;
    };
    if (!states) return false;
    states->clear();
    const Eigen::Isometry3d start_pose = start_state.getGlobalLinkTransform(tool_link_);
    const Eigen::Isometry3d goal_pose = goal_state.getGlobalLinkTransform(tool_link_);
    const Eigen::Quaterniond start_rotation(start_pose.linear());
    const Eigen::Quaterniond goal_rotation(goal_pose.linear());
    const double rotation = start_rotation.angularDistance(goal_rotation);
    const double translation = (goal_pose.translation() - start_pose.translation()).norm();
    const double orientation_step = degToRad(10.0);
    const size_t steps = std::max<size_t>({
      2U,
      static_cast<size_t>(std::ceil(translation / cartesian_transfer_translation_step_)),
      static_cast<size_t>(std::ceil(rotation / orientation_step)),
    });
    const double maximum_ratio_step = rotation > 1e-9 ? orientation_step / rotation : 1.0;
    const double start_swivel = solver_->swivelAngle(armJoints(start_state));
    const double goal_swivel = solver_->swivelAngle(armJoints(goal_state));
    if (!std::isfinite(start_swivel) || !std::isfinite(goal_swivel)) {
      if (reason) *reason = "orientation schedule endpoint swivel is undefined";
      return false;
    }
    std::vector<BeamNode> beam;
    beam.push_back({
      std::make_shared<moveit::core::RobotState>(start_state),
      {std::make_shared<moveit::core::RobotState>(start_state)},
      0.0,
      0.0,
    });
    for (size_t step = 1U; step < steps; ++step) {
      const double position_ratio = static_cast<double>(step) / static_cast<double>(steps);
      const size_t remaining_steps = steps - step;
      std::vector<BeamNode> next_beam;
      for (const auto& node : beam) {
        const double minimum_ratio = std::max(
          0.0,
          1.0 - static_cast<double>(remaining_steps) * maximum_ratio_step);
        const double maximum_ratio = std::min(
          1.0, node.orientation_ratio + maximum_ratio_step);
        const std::array<double, 5> raw_ratios = {
          std::clamp(position_ratio, minimum_ratio, maximum_ratio),
          minimum_ratio,
          maximum_ratio,
          std::clamp(node.orientation_ratio, minimum_ratio, maximum_ratio),
          0.5 * (minimum_ratio + maximum_ratio),
        };
        std::vector<double> orientation_ratios;
        for (const double candidate_ratio : raw_ratios) {
          if (std::none_of(
                orientation_ratios.begin(), orientation_ratios.end(),
                [candidate_ratio](double existing) {
                  return std::abs(existing - candidate_ratio) < 1e-8;
                })) {
            orientation_ratios.push_back(candidate_ratio);
          }
        }
        for (const double orientation_ratio : orientation_ratios) {
          Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
          target.translation() = start_pose.translation() +
            (goal_pose.translation() - start_pose.translation()) * position_ratio;
          target.linear() = start_rotation.slerp(
            orientation_ratio, goal_rotation).normalized().toRotationMatrix();
          const double current_swivel = solver_->swivelAngle(armJoints(*node.state));
          const double target_swivel = normalizedAngle(
            current_swivel + normalizedAngle(goal_swivel - current_swivel) /
            static_cast<double>(remaining_steps + 1U));
          std::string candidate_reason;
          auto candidate = solveNextPose(
            target, *node.state, attached, scene, metrics, &candidate_reason,
            nullptr, &target_swivel, true);
          if (!candidate) continue;
          BeamNode next;
          next.state = candidate->state;
          next.path = node.path;
          next.path.push_back(candidate->state);
          next.orientation_ratio = orientation_ratio;
          next.score = node.score + weightedJointTravel(
            armJoints(*node.state), armJoints(*candidate->state), kNaturalJointWeights);
          next_beam.push_back(std::move(next));
        }
      }
      if (next_beam.empty()) {
        if (reason) {
          *reason = "no Cartesian beam candidate at step " + std::to_string(step) + "/" +
            std::to_string(steps);
        }
        return false;
      }
      std::sort(
        next_beam.begin(), next_beam.end(),
        [](const BeamNode& lhs, const BeamNode& rhs) {return lhs.score < rhs.score;});
      if (next_beam.size() > 10U) next_beam.resize(10U);
      beam = std::move(next_beam);
    }
    for (const auto& node : beam) {
      std::vector<moveit::core::RobotStatePtr> final;
      std::string final_reason;
      if (!traceToolSpaceSegment(
            scene, *node.state, goal_state, attached, metrics, &final, &final_reason, true)) {
        continue;
      }
      *states = node.path;
      states->insert(states->end(), final.begin() + 1U, final.end());
      return true;
    }
    if (reason) *reason = "Cartesian beam could not connect to exact goal";
    return false;
  }

  bool searchShortcutRepairRrt(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit::core::RobotState& start_state,
    const moveit::core::RobotState& goal_state,
    bool attached,
    PlanningMetrics* metrics,
    std::vector<moveit::core::RobotStatePtr>* states,
    std::string* reason,
    std::vector<size_t>* repaired_joint_indices = nullptr,
    uint32_t seed_offset = 0U,
    std::atomic_bool* stop_requested = nullptr) const
  {
    struct RepairNode
    {
      moveit::core::RobotStatePtr state;
      size_t parent = 0U;
      double progress = 0.0;
      std::array<double, 2> offsets{0.0, 0.0};
    };
    if (!states) return false;
    states->clear();
    const auto started = std::chrono::steady_clock::now();
    const auto start_joints = armJoints(start_state);
    const auto goal_joints = armJoints(goal_state);
    const size_t shortcut_steps = std::max<size_t>(
      2U, static_cast<size_t>(std::ceil(
        maximumJointDelta(start_joints, goal_joints) / edge_joint_resolution_)));
    const auto elapsed_ms = [&]() {
      return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    };
    const auto cancelled = [&]() {
      return stop_requested && stop_requested->load(std::memory_order_acquire);
    };
    if (cancelled()) {
      if (reason) *reason = "parallel shortcut repair cancelled before start";
      return false;
    }
    const auto nominal_joints = [&](double progress) {
      std::array<double, 7> joints{};
      for (size_t index = 0U; index < joints.size(); ++index) {
        joints[index] = start_joints[index] +
          normalizedAngle(goal_joints[index] - start_joints[index]) * progress;
      }
      return joints;
    };
    const auto make_state = [&](
      double progress, const std::vector<size_t>& active_joints,
      const std::array<double, 2>& offsets) {
        auto joints = nominal_joints(progress);
        for (size_t index = 0U; index < active_joints.size(); ++index) {
          joints[active_joints[index]] += offsets[index];
        }
        auto state = std::make_shared<moveit::core::RobotState>(start_state);
        state->setJointGroupPositions(planning_group_, joints.data());
        if (attached) {
          attachCarriedBox(*state);
        } else if (state->hasAttachedBody(kCarriedBoxId)) {
          state->clearAttachedBody(kCarriedBoxId);
        }
        state->update(true);
        return state;
      };
    const auto tcp_in_corridor = [&](
      const moveit::core::RobotState& state,
      const moveit::core::RobotState& nominal) {
        const auto& actual_tcp = state.getGlobalLinkTransform(tool_link_);
        const auto& nominal_tcp = nominal.getGlobalLinkTransform(tool_link_);
        return (actual_tcp.translation() - nominal_tcp.translation()).norm() <= 0.18 + 1e-9 &&
               Eigen::Quaterniond(actual_tcp.linear()).angularDistance(
                 Eigen::Quaterniond(nominal_tcp.linear())) <= degToRad(35.0) + 1e-9;
      };
    const auto valid_state = [&](
      const moveit::core::RobotState& state,
      const moveit::core::RobotState& nominal) {
        return state.satisfiesBounds(planning_group_) &&
               (!attached || carriedBoxFeasible(state)) &&
               tcp_in_corridor(state, nominal) &&
               collisionReason(scene, state, metrics).empty();
      };
    const auto valid_edge = [&](
      const moveit::core::RobotState& from, double from_progress,
      const moveit::core::RobotState& to, double to_progress,
      std::string* edge_reason) {
        const size_t steps = std::max<size_t>(
          1U, static_cast<size_t>(std::ceil(
            maximumJointDelta(armJoints(from), armJoints(to)) / edge_joint_resolution_)));
        for (size_t step = 1U; step <= steps; ++step) {
          const double ratio = static_cast<double>(step) / static_cast<double>(steps);
          moveit::core::RobotState probe(from);
          from.interpolate(to, ratio, probe, planning_group_);
          probe.update(true);
          auto nominal = make_state(
            from_progress + (to_progress - from_progress) * ratio, {}, {0.0, 0.0});
          if (!probe.satisfiesBounds(planning_group_) ||
              (attached && !carriedBoxFeasible(probe)) ||
              !tcp_in_corridor(probe, *nominal)) {
            if (edge_reason) *edge_reason = "joint bounds, tilt, or TCP corridor";
            return false;
          }
          const std::string collision = collisionReason(scene, probe, metrics);
          if (!collision.empty()) {
            if (edge_reason) *edge_reason = collision;
            return false;
          }
        }
        return true;
      };

    std::vector<moveit::core::RobotStatePtr> shortcut;
    shortcut.reserve(shortcut_steps + 1U);
    std::vector<size_t> collision_indices;
    for (size_t step = 0U; step <= shortcut_steps; ++step) {
      const double progress = static_cast<double>(step) / static_cast<double>(shortcut_steps);
      auto state = make_state(progress, {}, {0.0, 0.0});
      if (!state->satisfiesBounds(planning_group_) ||
          (attached && !carriedBoxFeasible(*state)) ||
          !collisionReason(scene, *state, metrics).empty()) {
        collision_indices.push_back(step);
      }
      shortcut.push_back(std::move(state));
    }
    shortcut.back() = std::make_shared<moveit::core::RobotState>(goal_state);
    if (collision_indices.empty()) {
      if (reason) *reason = "joint shortcut is already collision-free";
      return false;
    }
    if (collision_indices.front() == 0U || collision_indices.back() == shortcut_steps) {
      if (reason) *reason = "shortcut collision reaches a fixed endpoint";
      return false;
    }
    const size_t repair_start_index = collision_indices.front() > 6U ?
      collision_indices.front() - 6U : 0U;
    const size_t repair_goal_index = std::min(
      shortcut_steps, collision_indices.back() + 6U);
    const double repair_start_progress = static_cast<double>(repair_start_index) /
      static_cast<double>(shortcut_steps);
    const double repair_goal_progress = static_cast<double>(repair_goal_index) /
      static_cast<double>(shortcut_steps);
    const auto& repair_goal_state = *shortcut[repair_goal_index];
    const auto offset_limit_at = [&](double progress) {
      const double ratio = (progress - repair_start_progress) /
        (repair_goal_progress - repair_start_progress);
      return shortcut_repair_max_joint_offset_ *
        std::sin(kPi * std::clamp(ratio, 0.0, 1.0));
    };

    const size_t probe_index = collision_indices[collision_indices.size() / 2U];
    const double probe_progress = static_cast<double>(probe_index) /
      static_cast<double>(shortcut_steps);
    const auto& probe_nominal = *shortcut[probe_index];
    const std::array<double, 5> probe_magnitudes = {
      degToRad(5.0), degToRad(10.0), degToRad(15.0),
      degToRad(25.0), shortcut_repair_max_joint_offset_};
    std::vector<size_t> active_joints;
    double best_single_score = std::numeric_limits<double>::infinity();
    for (size_t joint = 0U; joint < start_joints.size(); ++joint) {
      for (const double magnitude : probe_magnitudes) {
        for (const double sign : {-1.0, 1.0}) {
          if (cancelled()) {
            if (reason) *reason = "parallel shortcut repair cancelled during joint probes";
            return false;
          }
          if (elapsed_ms() >= shortcut_repair_rrt_budget_ms_) {
            if (reason) *reason = "shortcut repair budget exhausted during joint probes";
            return false;
          }
          const std::array<double, 2> offsets{sign * magnitude, 0.0};
          auto candidate = make_state(probe_progress, {joint}, offsets);
          if (!valid_state(*candidate, probe_nominal)) continue;
          const double score = magnitude + 0.1 * weightedJointTravel(
            armJoints(probe_nominal), armJoints(*candidate), kNaturalJointWeights);
          if (score < best_single_score) {
            best_single_score = score;
            active_joints = {joint};
          }
        }
      }
    }
    if (active_joints.empty()) {
      double best_pair_score = std::numeric_limits<double>::infinity();
      const std::array<double, 3> pair_magnitudes = {
        degToRad(10.0), degToRad(20.0), shortcut_repair_max_joint_offset_};
      for (size_t first = 0U; first < start_joints.size(); ++first) {
        for (size_t second = first + 1U; second < start_joints.size(); ++second) {
          for (const double first_magnitude : pair_magnitudes) {
            for (const double second_magnitude : pair_magnitudes) {
              for (const double first_sign : {-1.0, 1.0}) {
                for (const double second_sign : {-1.0, 1.0}) {
                  if (cancelled()) {
                    if (reason) {
                      *reason = "parallel shortcut repair cancelled during joint probes";
                    }
                    return false;
                  }
                  if (elapsed_ms() >= shortcut_repair_rrt_budget_ms_) {
                    if (reason) *reason = "shortcut repair budget exhausted during joint probes";
                    return false;
                  }
                  const std::array<double, 2> offsets{
                    first_sign * first_magnitude, second_sign * second_magnitude};
                  auto candidate = make_state(probe_progress, {first, second}, offsets);
                  if (!valid_state(*candidate, probe_nominal)) continue;
                  const double score = first_magnitude + second_magnitude +
                    0.1 * weightedJointTravel(
                    armJoints(probe_nominal), armJoints(*candidate), kNaturalJointWeights);
                  if (score < best_pair_score) {
                    best_pair_score = score;
                    active_joints = {first, second};
                  }
                }
              }
            }
          }
        }
      }
    }
    if (active_joints.empty()) {
      if (reason) {
        *reason = "shortcut collision could not be cleared by one or two joint probes";
      }
      return false;
    }
    if (repaired_joint_indices) *repaired_joint_indices = active_joints;

    uint32_t random_seed = side_ == "left" ? 0xA11F309U : 0xA11F310U;
    for (const double value : start_joints) {
      random_seed ^= static_cast<uint32_t>(std::llround((value + 4.0) * 10000.0));
      random_seed = random_seed * 1664525U + 1013904223U;
    }
    random_seed ^= 0x9E3779B9U * seed_offset;
    std::mt19937 random_engine(random_seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> signed_unit(-1.0, 1.0);

    std::vector<RepairNode> tree;
    tree.reserve(static_cast<size_t>(shortcut_repair_rrt_max_samples_) + 1U);
    tree.push_back({
      shortcut[repair_start_index], 0U, repair_start_progress, {0.0, 0.0}});
    size_t solved_nodes = 0U;
    size_t attempted_samples = 0U;
    std::string last_reason = "no valid repair node";
    const double progress_step = 1.0 / static_cast<double>(shortcut_steps);
    const double offset_step = degToRad(8.0);

    const auto connect_goal = [&](
      const RepairNode& node,
      std::vector<moveit::core::RobotStatePtr>* terminal) {
        terminal->clear();
        terminal->push_back(node.state);
        const double remaining = repair_goal_progress - node.progress;
        const size_t steps = std::max<size_t>(
          1U, static_cast<size_t>(std::ceil(remaining / progress_step)));
        moveit::core::RobotStatePtr current = node.state;
        for (size_t step = 1U; step <= steps; ++step) {
          if (cancelled()) {
            last_reason = "parallel shortcut repair cancelled during goal connection";
            terminal->clear();
            return false;
          }
          if (elapsed_ms() >= shortcut_repair_rrt_budget_ms_) {
            last_reason = "repair budget exhausted during goal connection";
            terminal->clear();
            return false;
          }
          const double ratio = static_cast<double>(step) / static_cast<double>(steps);
          const double progress = node.progress + remaining * ratio;
          std::array<double, 2> offsets{
            node.offsets[0] * (1.0 - ratio), node.offsets[1] * (1.0 - ratio)};
          auto nominal = make_state(progress, {}, {0.0, 0.0});
          auto next = make_state(progress, active_joints, offsets);
          std::string edge_reason;
          if (!valid_state(*next, *nominal) ||
              !valid_edge(*current, node.progress + remaining *
                static_cast<double>(step - 1U) / static_cast<double>(steps),
                *next, progress, &edge_reason)) {
            last_reason = "goal repair edge: " + edge_reason;
            terminal->clear();
            return false;
          }
          current = next;
          terminal->push_back(current);
        }
        std::string exact_reason;
        if (!valid_edge(*terminal->back(), repair_goal_progress,
              repair_goal_state, repair_goal_progress, &exact_reason)) {
          last_reason = "exact goal: " + exact_reason;
          terminal->clear();
          return false;
        }
        terminal->back() = shortcut[repair_goal_index];
        return true;
      };

    for (int iteration = 0; iteration < shortcut_repair_rrt_max_samples_; ++iteration) {
      if (cancelled()) {
        last_reason = "parallel shortcut repair cancelled by another seed";
        break;
      }
      if (elapsed_ms() >= shortcut_repair_rrt_budget_ms_) break;
      ++attempted_samples;
      if (metrics) ++metrics->reduced_rrt_samples;
      const bool goal_bias = iteration % 4 == 0;
      const double sample_progress = goal_bias ? repair_goal_progress :
        repair_start_progress +
        (repair_goal_progress - repair_start_progress) * unit(random_engine);
      const double offset_limit = offset_limit_at(sample_progress);
      std::array<double, 2> sample_offsets{0.0, 0.0};
      if (goal_bias) {
        const auto& furthest = *std::max_element(
          tree.begin(), tree.end(),
          [](const RepairNode& left, const RepairNode& right) {
            return left.progress < right.progress;
          });
        sample_offsets = furthest.offsets;
      } else {
        for (size_t index = 0U; index < active_joints.size(); ++index) {
          sample_offsets[index] = signed_unit(random_engine) * offset_limit;
        }
      }

      size_t nearest_index = 0U;
      double nearest_distance = std::numeric_limits<double>::infinity();
      for (size_t index = 0U; index < tree.size(); ++index) {
        if (tree[index].progress > sample_progress + 1e-9) continue;
        double distance = 6.0 * (sample_progress - tree[index].progress);
        for (size_t active = 0U; active < active_joints.size(); ++active) {
          distance += 0.5 * std::abs(normalizedAngle(
            sample_offsets[active] - tree[index].offsets[active]));
        }
        if (distance < nearest_distance) {
          nearest_distance = distance;
          nearest_index = index;
        }
      }
      const RepairNode& nearest = tree[nearest_index];
      if (sample_progress <= nearest.progress + 1e-5) continue;
      const double progress = std::min(sample_progress, nearest.progress + progress_step);
      std::array<double, 2> offsets = nearest.offsets;
      const double tapered_limit = offset_limit_at(progress);
      for (size_t active = 0U; active < active_joints.size(); ++active) {
        const double delta = std::clamp(
          normalizedAngle(sample_offsets[active] - nearest.offsets[active]),
          -offset_step, offset_step);
        offsets[active] = std::clamp(
          nearest.offsets[active] + delta, -tapered_limit, tapered_limit);
      }
      auto nominal = make_state(progress, {}, {0.0, 0.0});
      auto next = make_state(progress, active_joints, offsets);
      std::string edge_reason;
      if (!valid_state(*next, *nominal) ||
          !valid_edge(*nearest.state, nearest.progress, *next, progress, &edge_reason)) {
        last_reason = edge_reason.empty() ? "invalid repair state" : edge_reason;
        continue;
      }
      ++solved_nodes;
      tree.push_back({
        next, nearest_index, progress, offsets});
      const size_t new_index = tree.size() - 1U;
      if (progress < repair_start_progress +
          0.5 * (repair_goal_progress - repair_start_progress)) continue;
      std::vector<moveit::core::RobotStatePtr> terminal;
      if (!connect_goal(tree.back(), &terminal)) continue;

      std::vector<size_t> route;
      for (size_t index = new_index;; index = tree[index].parent) {
        route.push_back(index);
        if (index == 0U) break;
      }
      std::reverse(route.begin(), route.end());
      std::vector<moveit::core::RobotStatePtr> repaired_path{tree[route.front()].state};
      for (size_t from = 0U; from + 1U < route.size();) {
        if (cancelled()) {
          if (reason) *reason = "parallel shortcut repair cancelled during path shortcutting";
          return false;
        }
        size_t selected = from + 1U;
        for (size_t to = route.size() - 1U; to > from; --to) {
          std::string shortcut_reason;
          if (valid_edge(
                *tree[route[from]].state, tree[route[from]].progress,
                *tree[route[to]].state, tree[route[to]].progress,
                &shortcut_reason)) {
            selected = to;
            break;
          }
        }
        repaired_path.push_back(tree[route[selected]].state);
        from = selected;
      }
      *states = std::vector<moveit::core::RobotStatePtr>(
        shortcut.begin(), shortcut.begin() + repair_start_index);
      states->insert(states->end(), repaired_path.begin(), repaired_path.end());
      states->insert(states->end(), terminal.begin() + 1U, terminal.end());
      states->insert(states->end(),
        shortcut.begin() + repair_goal_index + 1U, shortcut.end());
      if (reason) {
        *reason = "shortcut repair joints=" + std::to_string(active_joints[0] + 1U) +
          (active_joints.size() == 2U ? "," + std::to_string(active_joints[1] + 1U) : "") +
          " window=" + std::to_string(repair_start_index) + "-" +
          std::to_string(repair_goal_index) + "/" + std::to_string(shortcut_steps) +
          " samples=" +
          std::to_string(attempted_samples) + " nodes=" + std::to_string(tree.size()) +
          " elapsed_ms=" + std::to_string(elapsed_ms());
      }
      if (stop_requested) {
        stop_requested->store(true, std::memory_order_release);
      }
      return true;
    }
    if (reason) {
      *reason = "shortcut repair RRT exhausted joints=" +
        std::to_string(active_joints[0] + 1U) +
        (active_joints.size() == 2U ? "," + std::to_string(active_joints[1] + 1U) : "") +
        " window=" + std::to_string(repair_start_index) + "-" +
        std::to_string(repair_goal_index) + "/" + std::to_string(shortcut_steps) +
        " samples=" +
        std::to_string(attempted_samples) +
        " nodes=" + std::to_string(tree.size()) +
        " projected=" + std::to_string(solved_nodes) +
        " elapsed_ms=" + std::to_string(elapsed_ms()) + ": " + last_reason;
    }
    return false;
  }

  bool searchShortcutRepairRrtParallel(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit::core::RobotState& start_state,
    const moveit::core::RobotState& goal_state,
    bool attached,
    PlanningMetrics* metrics,
    std::vector<moveit::core::RobotStatePtr>* states,
    std::string* reason,
    std::vector<size_t>* repaired_joint_indices,
    size_t* workers_launched,
    size_t* workers_completed,
    int* winner_seed) const
  {
    const size_t worker_count = static_cast<size_t>(parallel_rrt_workers_);
    if (workers_launched) *workers_launched = worker_count;
    if (workers_completed) *workers_completed = 0U;
    if (winner_seed) *winner_seed = -1;
    if (metrics) metrics->parallel_rrt_workers_launched += worker_count;
    if (worker_count == 1U) {
      const bool success = searchShortcutRepairRrt(
        scene, start_state, goal_state, attached, metrics, states, reason,
        repaired_joint_indices);
      if (workers_completed) *workers_completed = 1U;
      if (winner_seed && success) *winner_seed = 0;
      if (metrics) {
        ++metrics->parallel_rrt_workers_completed;
        if (success) ++metrics->parallel_rrt_wins;
      }
      return success;
    }

    struct TrialResult
    {
      bool success = false;
      size_t seed = 0U;
      double completion_ms = 0.0;
      PlanningMetrics metrics;
      std::vector<moveit::core::RobotStatePtr> states;
      std::vector<size_t> repaired_joint_indices;
      std::string reason;
    };

    const auto started = std::chrono::steady_clock::now();
    std::atomic_bool stop_requested{false};
    std::vector<planning_scene::PlanningScenePtr> worker_scenes;
    worker_scenes.reserve(worker_count);
    for (size_t worker = 0U; worker < worker_count; ++worker) {
      auto worker_scene = planning_scene::PlanningScene::clone(scene);
      worker_scene->setCurrentState(start_state);
      worker_scenes.push_back(std::move(worker_scene));
    }
    std::vector<std::future<TrialResult>> futures;
    futures.reserve(worker_count);
    for (size_t worker = 0U; worker < worker_count; ++worker) {
      auto worker_scene = worker_scenes[worker];
      futures.push_back(std::async(
        std::launch::async,
        [this, worker_scene, &start_state, &goal_state, attached, worker,
        started, &stop_requested]() {
          TrialResult trial;
          trial.seed = worker;
          trial.success = searchShortcutRepairRrt(
            worker_scene, start_state, goal_state, attached, &trial.metrics,
            &trial.states, &trial.reason, &trial.repaired_joint_indices,
            static_cast<uint32_t>(worker), &stop_requested);
          trial.completion_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
          return trial;
        }));
    }

    std::vector<TrialResult> trials;
    trials.reserve(worker_count);
    for (auto& future : futures) {
      trials.push_back(future.get());
    }
    if (workers_completed) *workers_completed = trials.size();
    if (metrics) {
      metrics->parallel_rrt_workers_completed += trials.size();
      for (const auto& trial : trials) {
        metrics->collision_checks += trial.metrics.collision_checks;
        metrics->collision_ms += trial.metrics.collision_ms;
        metrics->reduced_rrt_samples += trial.metrics.reduced_rrt_samples;
      }
    }

    const auto winner = std::min_element(
      trials.begin(), trials.end(),
      [](const TrialResult& left, const TrialResult& right) {
        if (left.success != right.success) return left.success > right.success;
        return left.completion_ms < right.completion_ms;
      });
    if (winner == trials.end() || !winner->success) {
      if (reason) {
        std::ostringstream stream;
        stream << "parallel shortcut repair exhausted " << worker_count << " seeds";
        for (const auto& trial : trials) {
          stream << "; seed" << trial.seed << "=" << trial.reason;
        }
        *reason = stream.str();
      }
      return false;
    }

    if (states) *states = winner->states;
    if (repaired_joint_indices) {
      *repaired_joint_indices = winner->repaired_joint_indices;
    }
    if (winner_seed) *winner_seed = static_cast<int>(winner->seed);
    if (metrics) ++metrics->parallel_rrt_wins;
    if (reason) {
      *reason = "parallel_seed=" + std::to_string(winner->seed) + "/" +
        std::to_string(worker_count) + " completion_ms=" +
        std::to_string(winner->completion_ms) + "; " + winner->reason;
    }
    return true;
  }

  RrtPlanResult planRrt(
    const planning_scene::PlanningSceneConstPtr& base_scene,
    const moveit::core::RobotState& start_state,
    const moveit::core::RobotState& goal_state,
    bool direct_only = false,
    PlanningMetrics* metrics = nullptr,
    bool force_cartesian_search = false) const
  {
    if (pure_rrt_full_flow_ && pure_rrt_backend_ == "vamp") {
      return planPureArmUpdownRrt(
        base_scene, start_state, goal_state,
        start_state.hasAttachedBody(kCarriedBoxId), metrics);
    }
    RrtPlanResult result;
    const auto wall_started = std::chrono::steady_clock::now();
    auto scene = planning_scene::PlanningScene::clone(base_scene);
    scene->setCurrentState(start_state);
    const bool loaded = start_state.hasAttachedBody(kCarriedBoxId);
    std::string cartesian_diagnostic;
    size_t cartesian_search_attempts = 0U;
    bool cartesian_budget_exhausted = false;
    const auto cartesian_search_available = [&]() {
      return cartesian_transfer_max_search_attempts_ == 0 ||
             cartesian_search_attempts <
             static_cast<size_t>(cartesian_transfer_max_search_attempts_);
    };
    const auto record_cartesian_attempt = [&]() {
      ++cartesian_search_attempts;
      if (metrics) ++metrics->cartesian_transfer_attempts;
    };
    if (loaded) {
      scene->setStateFeasibilityPredicate(
        [this](const moveit::core::RobotState& state, bool) {
          return carriedBoxFeasible(state);
        });
      if (!carriedBoxFeasible(start_state) || !carriedBoxFeasible(goal_state)) {
        result.reason = "carried_box_pose_limit";
        return result;
      }
    }
    const std::string start_collision = alfa_robot::motion::scene_collision_reason(
      scene, start_state, planning_group_);
    if (!start_collision.empty()) {
      result.reason = "start_" + start_collision;
      return result;
    }
    const std::string goal_collision = alfa_robot::motion::scene_collision_reason(
      scene, goal_state, planning_group_);
    if (!goal_collision.empty()) {
      result.reason = "goal_" + goal_collision;
      return result;
    }

    if (!pure_rrt_full_flow_ &&
        shortcut_repair_rrt_enabled_ && cartesian_transfer_search_enabled_ &&
        !direct_only && !force_cartesian_search) {
      const auto start_joints = armJoints(start_state);
      const auto goal_joints = armJoints(goal_state);
      const size_t steps = std::max<size_t>(
        1U, static_cast<size_t>(std::ceil(
          maximumJointDelta(start_joints, goal_joints) / edge_joint_resolution_)));
      std::vector<moveit::core::RobotStatePtr> shortcut;
      shortcut.push_back(std::make_shared<moveit::core::RobotState>(start_state));
      bool shortcut_clear = true;
      for (size_t step = 1U; step <= steps; ++step) {
        const double progress = static_cast<double>(step) / static_cast<double>(steps);
        std::array<double, 7> joints{};
        for (size_t index = 0U; index < joints.size(); ++index) {
          joints[index] = start_joints[index] +
            normalizedAngle(goal_joints[index] - start_joints[index]) * progress;
        }
        auto state = std::make_shared<moveit::core::RobotState>(start_state);
        state->setJointGroupPositions(planning_group_, joints.data());
        state->update(true);
        if (!state->satisfiesBounds(planning_group_) ||
            (loaded && !carriedBoxFeasible(*state)) ||
            !collisionReason(scene, *state, metrics).empty()) {
          shortcut_clear = false;
          break;
        }
        shortcut.push_back(std::move(state));
      }
      if (shortcut_clear) {
        shortcut.back() = std::make_shared<moveit::core::RobotState>(goal_state);
        result.states = std::move(shortcut);
        result.success = true;
        result.strategy = "joint_shortcut";
        result.reason = result.strategy;
        result.wall_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - wall_started).count();
        return result;
      }

      const auto repair_started = std::chrono::steady_clock::now();
      if (metrics) ++metrics->reduced_rrt_attempts;
      const uint64_t samples_before = metrics ? metrics->reduced_rrt_samples : 0U;
      std::vector<moveit::core::RobotStatePtr> repaired;
      std::string repair_reason;
      const bool repair_success = searchShortcutRepairRrtParallel(
        scene, start_state, goal_state, loaded, metrics, &repaired, &repair_reason,
        &result.repaired_joint_indices, &result.parallel_workers_launched,
        &result.parallel_workers_completed, &result.parallel_winner_seed);
      result.reduced_rrt_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - repair_started).count();
      result.reduced_rrt_samples = static_cast<size_t>(
        (metrics ? metrics->reduced_rrt_samples : samples_before) - samples_before);
      if (metrics) metrics->reduced_rrt_ms += result.reduced_rrt_ms;
      if (repair_success) {
        if (metrics) ++metrics->reduced_rrt_successes;
        if (metrics) ++metrics->cartesian_guided_segments;
        result.states = densifyRrtReplay(repaired);
        result.success = true;
        result.strategy = "shortcut_repair_rrt";
        result.reason = repair_reason;
        result.wall_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - wall_started).count();
        return result;
      }
      cartesian_diagnostic = repair_reason;
    }

    if (!pure_rrt_full_flow_ &&
        cartesian_transfer_search_enabled_ && cartesian_search_available()) {
      const auto shortcut_started = std::chrono::steady_clock::now();
      record_cartesian_attempt();
      if (metrics) ++metrics->tcp_shortcut_attempts;
      std::vector<moveit::core::RobotStatePtr> cartesian_states;
      std::string cartesian_reason;
      const bool shortcut_success = traceToolSpaceSegment(
            scene, start_state, goal_state, loaded, metrics,
            &cartesian_states, &cartesian_reason) &&
          (force_cartesian_search ||
          pathJointTravel(cartesian_states) <= 1.25 * weightedJointTravel(
            armJoints(start_state), armJoints(goal_state), kNaturalJointWeights) + 1e-6);
      result.tcp_shortcut_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - shortcut_started).count();
      if (metrics) metrics->tcp_shortcut_ms += result.tcp_shortcut_ms;
      if (shortcut_success) {
        if (metrics) ++metrics->cartesian_guided_segments;
        if (metrics) ++metrics->tcp_shortcut_successes;
        result.wall_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - wall_started).count();
        result.states = std::move(cartesian_states);
        result.success = true;
        result.reason = "direct_cartesian_analytic_ik";
        result.strategy = "tcp_shortcut";
        return result;
      }
      cartesian_diagnostic = "direct Cartesian: " + cartesian_reason;
    }
    if (!pure_rrt_full_flow_ &&
        cartesian_transfer_search_enabled_ && cartesian_search_available()) {
      const auto waypoint_started = std::chrono::steady_clock::now();
      std::vector<moveit::core::RobotStatePtr> waypoint_states;
      std::string waypoint_reason;
      const bool waypoint_success = searchToolSpaceWaypointPath(
        scene, start_state, goal_state, loaded, metrics,
        static_cast<size_t>(cartesian_transfer_max_search_attempts_),
        &cartesian_search_attempts, &waypoint_states, &waypoint_reason);
      const double waypoint_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - waypoint_started).count();
      result.tcp_shortcut_ms += waypoint_ms;
      if (metrics) metrics->tcp_shortcut_ms += waypoint_ms;
      if (waypoint_success && !waypoint_states.empty()) {
        if (metrics) {
          ++metrics->cartesian_guided_segments;
          ++metrics->tcp_shortcut_successes;
        }
        result.states = std::move(waypoint_states);
        result.success = true;
        result.reason = "inward Cartesian waypoint path: " + waypoint_reason;
        result.strategy = "tcp_waypoint_shortcut";
        result.wall_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - wall_started).count();
        return result;
      }
      cartesian_diagnostic += "; waypoint Cartesian: " + waypoint_reason;
    }
    if (!pure_rrt_full_flow_ && force_cartesian_search &&
        shortcut_repair_rrt_enabled_ && !direct_only) {
      const auto repair_started = std::chrono::steady_clock::now();
      if (metrics) ++metrics->reduced_rrt_attempts;
      const uint64_t samples_before = metrics ? metrics->reduced_rrt_samples : 0U;
      std::vector<moveit::core::RobotStatePtr> repaired;
      std::string repair_reason;
      const bool repair_success = searchShortcutRepairRrtParallel(
        scene, start_state, goal_state, loaded, metrics, &repaired, &repair_reason,
        &result.repaired_joint_indices, &result.parallel_workers_launched,
        &result.parallel_workers_completed, &result.parallel_winner_seed);
      result.reduced_rrt_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - repair_started).count();
      result.reduced_rrt_samples = static_cast<size_t>(
        (metrics ? metrics->reduced_rrt_samples : samples_before) - samples_before);
      if (metrics) metrics->reduced_rrt_ms += result.reduced_rrt_ms;
      if (repair_success) {
        if (metrics) {
          ++metrics->reduced_rrt_successes;
          ++metrics->cartesian_guided_segments;
        }
        result.states = densifyRrtReplay(repaired);
        result.success = true;
        result.strategy = "shortcut_repair_rrt";
        result.reason = repair_reason + " after " + cartesian_diagnostic;
        result.wall_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - wall_started).count();
        return result;
      }
      cartesian_diagnostic += "; shortcut repair: " + repair_reason;
    }
    const auto start_joints = armJoints(start_state);
    const auto goal_joints = armJoints(goal_state);
    std::vector<moveit::core::RobotStatePtr> direct_states;
    bool direct_clear = false;
    if (!pure_rrt_full_flow_) {
      const size_t direct_steps = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(
          maximumJointDelta(start_joints, goal_joints) / edge_joint_resolution_)));
      direct_states.reserve(direct_steps + 1U);
      direct_states.push_back(std::make_shared<moveit::core::RobotState>(start_state));
      direct_clear = true;
      for (size_t step = 1; step <= direct_steps; ++step) {
        const double ratio = static_cast<double>(step) / static_cast<double>(direct_steps);
        std::array<double, 7> interpolated{};
        for (size_t index = 0; index < interpolated.size(); ++index) {
          interpolated[index] = start_joints[index] +
            normalizedAngle(goal_joints[index] - start_joints[index]) * ratio;
        }
        auto state = std::make_shared<moveit::core::RobotState>(start_state);
        state->setJointGroupPositions(planning_group_, interpolated.data());
        state->update(true);
        if (!state->satisfiesBounds(planning_group_) ||
            (loaded && !carriedBoxFeasible(*state)) ||
            !alfa_robot::motion::scene_collision_reason(
              scene, *state, planning_group_).empty()) {
          direct_clear = false;
          break;
        }
        direct_states.push_back(std::move(state));
      }
    }
    if (!pure_rrt_full_flow_ && direct_clear) {
      if (metrics) ++metrics->joint_fallback_segments;
      result.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_started).count();
      result.states = std::move(direct_states);
      result.success = true;
      result.reason = "direct_joint_interpolation";
      result.strategy = "joint_shortcut";
      return result;
    }
    if (!pure_rrt_full_flow_ && direct_only) {
      result.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_started).count();
      result.reason = "direct_joint_interpolation_in_collision";
      return result;
    }

    const auto joint_rrt_started = std::chrono::steady_clock::now();
    planning_interface::MotionPlanRequest request;
    request.group_name = planning_group_name_;
    request.planner_id = "RRTConnectkConfigDefault";
    request.allowed_planning_time = rrt_planning_time_;
    request.num_planning_attempts = rrt_planning_attempts_;
    request.max_velocity_scaling_factor = 1.0;
    request.max_acceleration_scaling_factor = 1.0;
    moveit::core::robotStateToRobotStateMsg(start_state, request.start_state, true);
    request.goal_constraints.push_back(
      kinematic_constraints::constructGoalConstraints(goal_state, planning_group_, 1e-3));

    planning_interface::MotionPlanResponse response;
    const bool generated = planning_pipeline_->generatePlan(scene, request, response);
    result.joint_rrt_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - joint_rrt_started).count();
    if (metrics) metrics->joint_rrt_ms += result.joint_rrt_ms;
    result.wall_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - wall_started).count();
    result.planner_ms = response.planning_time_ * 1000.0;
    if (!generated || response.error_code_.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS ||
        !response.trajectory_) {
      if (!pure_rrt_full_flow_ && direct_clear) {
        if (metrics) ++metrics->joint_fallback_segments;
        result.states = std::move(direct_states);
        result.success = true;
        result.reason = "direct_joint_fallback_after_cartesian_search";
        result.strategy = "joint_shortcut_fallback";
        if (metrics) ++metrics->joint_rrt_fallbacks;
        return result;
      }
      result.reason = "RRTConnect code=" + std::to_string(response.error_code_.val) + " " +
        alfa_robot::motion::direct_pipeline_failure_diagnostic(
          scene, start_state, goal_state, planning_group_);
      if (!cartesian_diagnostic.empty()) {
        result.reason += "; " + cartesian_diagnostic;
      }
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
    if (pure_rrt_full_flow_) {
      std::vector<moveit::core::RobotStatePtr> raw_path;
      raw_path.push_back(std::make_shared<moveit::core::RobotState>(start_state));
      raw_path.insert(raw_path.end(), result.states.begin(), result.states.end());
      const auto& last = *raw_path.back();
      double goal_error = 0.0;
      for (const auto& variable : planning_group_->getVariableNames()) {
        goal_error = std::max(goal_error, std::abs(
          last.getVariablePosition(variable) - goal_state.getVariablePosition(variable)));
      }
      if (goal_error > 1e-8) {
        raw_path.push_back(std::make_shared<moveit::core::RobotState>(goal_state));
      }
      result.states = densifyRrtReplay(raw_path);
      for (size_t index = 1U; index < result.states.size(); ++index) {
        if (!edgeClear(
              scene, *result.states[index - 1U], *result.states[index],
              loaded, nullptr, &result.reason)) {
          result.reason = "pure_rrt_path_validation: " + result.reason;
          result.states.clear();
          return result;
        }
      }
      result.success = true;
      result.reason = "pure_rrt_full_flow exact solution";
      result.strategy = "pure_ompl_rrt_connect";
      result.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_started).count();
      return result;
    }
    if (natural_rrt_shortcut_enabled_ && result.states.size() > 2U) {
      const auto shortcut_started = std::chrono::steady_clock::now();
      std::vector<moveit::core::RobotStatePtr> original;
      original.reserve(result.states.size() + 1U);
      original.push_back(std::make_shared<moveit::core::RobotState>(start_state));
      original.insert(original.end(), result.states.begin(), result.states.end());
      std::vector<moveit::core::RobotStatePtr> shortcut_nodes;
      if (natural_rrt_shortcut_max_nodes_ > 1 &&
          original.size() > static_cast<size_t>(natural_rrt_shortcut_max_nodes_)) {
        const size_t maximum_nodes = static_cast<size_t>(natural_rrt_shortcut_max_nodes_);
        const size_t stride = std::max<size_t>(
          1U, (original.size() - 1U + maximum_nodes - 2U) / (maximum_nodes - 1U));
        for (size_t index = 0U; index < original.size(); index += stride) {
          shortcut_nodes.push_back(original[index]);
        }
        if (shortcut_nodes.back() != original.back()) shortcut_nodes.push_back(original.back());
      } else {
        shortcut_nodes = original;
      }

      const bool attached = start_state.hasAttachedBody(kCarriedBoxId);
      std::vector<double> costs(
        shortcut_nodes.size(), std::numeric_limits<double>::infinity());
      std::vector<size_t> predecessors(shortcut_nodes.size(), 0U);
      costs.front() = 0.0;
      for (size_t to = 1U; to < shortcut_nodes.size(); ++to) {
        for (size_t from = 0U; from < to; ++from) {
          if (metrics) ++metrics->rrt_shortcut_edges_checked;
          if (!std::isfinite(costs[from]) ||
              !edgeClear(
                scene, *shortcut_nodes[from], *shortcut_nodes[to], attached, nullptr, nullptr)) {
            continue;
          }
          double edge_cost = weightedJointTravel(
            armJoints(*shortcut_nodes[from]), armJoints(*shortcut_nodes[to]),
            kNaturalJointWeights);
          if (force_cartesian_search || natural_rrt_shortcut_tcp_cost_) {
            const auto dense_edge = densifyRrtReplay({
              shortcut_nodes[from], shortcut_nodes[to]});
            double tcp_position_travel = 0.0;
            double tcp_orientation_travel = 0.0;
            for (size_t index = 1U; index < dense_edge.size(); ++index) {
              const auto& edge_from =
                dense_edge[index - 1U]->getGlobalLinkTransform(tool_link_);
              const auto& edge_to = dense_edge[index]->getGlobalLinkTransform(tool_link_);
              tcp_position_travel +=
                (edge_to.translation() - edge_from.translation()).norm();
              tcp_orientation_travel += Eigen::Quaterniond(edge_from.linear()).angularDistance(
                Eigen::Quaterniond(edge_to.linear()));
            }
            edge_cost = tcp_position_travel + 0.15 * tcp_orientation_travel;
          }
          const double cost = costs[from] + edge_cost;
          if (cost < costs[to]) {
            costs[to] = cost;
            predecessors[to] = from;
          }
        }
      }
      if (std::isfinite(costs.back())) {
        std::vector<moveit::core::RobotStatePtr> shortened;
        for (size_t index = shortcut_nodes.size() - 1U;; index = predecessors[index]) {
          shortened.push_back(shortcut_nodes[index]);
          if (index == 0U) {
            break;
          }
        }
        std::reverse(shortened.begin(), shortened.end());
        result.states = std::move(shortened);
      }
      if (metrics) {
        metrics->rrt_shortcut_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - shortcut_started).count();
      }
    }
    std::vector<moveit::core::RobotStatePtr> sparse_states;
    sparse_states.push_back(std::make_shared<moveit::core::RobotState>(start_state));
    for (const auto& state : result.states) {
      if (maximumJointDelta(armJoints(*sparse_states.back()), armJoints(*state)) > 1e-8) {
        sparse_states.push_back(state);
      }
    }
    if (maximumJointDelta(armJoints(*sparse_states.back()), goal_joints) > 1e-8) {
      sparse_states.push_back(std::make_shared<moveit::core::RobotState>(goal_state));
    }
    std::vector<moveit::core::RobotStatePtr> hybrid_states{sparse_states.front()};
    size_t from = 0U;
    while (from + 1U < sparse_states.size()) {
      size_t selected_to = from + 1U;
      std::vector<moveit::core::RobotStatePtr> segment_states;
      bool guided = false;
      if (cartesian_transfer_search_enabled_) {
        double original_travel = 0.0;
        for (size_t index = from + 1U; index < sparse_states.size(); ++index) {
          original_travel += weightedJointTravel(
            armJoints(*sparse_states[index - 1U]), armJoints(*sparse_states[index]),
            kNaturalJointWeights);
        }
        for (size_t to = sparse_states.size() - 1U; to > from; --to) {
          if (!cartesian_search_available()) {
            cartesian_budget_exhausted = true;
            break;
          }
          record_cartesian_attempt();
          std::vector<moveit::core::RobotStatePtr> candidate_states;
          std::string segment_reason;
          if (traceToolSpaceSegment(
                scene, *sparse_states[from], *sparse_states[to], loaded, metrics,
                &candidate_states, &segment_reason) &&
              pathJointTravel(candidate_states) <= 1.25 * original_travel + 1e-6) {
            selected_to = to;
            segment_states = std::move(candidate_states);
            guided = true;
            break;
          }
          original_travel -= weightedJointTravel(
            armJoints(*sparse_states[to - 1U]), armJoints(*sparse_states[to]),
            kNaturalJointWeights);
        }
      }
      if (guided) {
        if (metrics) ++metrics->cartesian_guided_segments;
      } else {
        if (metrics) ++metrics->joint_fallback_segments;
        segment_states = densifyRrtReplay({sparse_states[from], sparse_states[selected_to]});
      }
      hybrid_states.insert(hybrid_states.end(), segment_states.begin() + 1U, segment_states.end());
      from = selected_to;
    }
    if (cartesian_budget_exhausted && metrics) {
      ++metrics->cartesian_transfer_budget_exhaustions;
    }
    result.states = std::move(hybrid_states);
    for (size_t index = 1U; index < result.states.size(); ++index) {
      if (!edgeClear(
            scene, *result.states[index - 1U], *result.states[index],
            loaded, nullptr, &result.reason)) {
        result.reason = "rrt_path_validation: " + result.reason;
        result.states.clear();
        return result;
      }
    }
    result.success = true;
    result.reason = pure_rrt_full_flow_ ?
      "pure_rrt_full_flow exact solution" : "rrt_cartesian_hybrid";
    result.strategy = pure_rrt_full_flow_ ?
      "pure_ompl_rrt_connect" : "joint_rrt_fallback";
    if (metrics) ++metrics->joint_rrt_fallbacks;
    if (!cartesian_diagnostic.empty()) result.reason += " after " + cartesian_diagnostic;
    result.wall_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - wall_started).count();
    return result;
  }

  RrtPlanResult planPureArmUpdownRrt(
    const planning_scene::PlanningSceneConstPtr& base_scene,
    const moveit::core::RobotState& start_state,
    const moveit::core::RobotState& goal_state,
    bool attached,
    PlanningMetrics* metrics,
    const std::string& backend_override = "",
    std::optional<size_t> paper_skip_override = std::nullopt,
    std::optional<uint32_t> vamp_seed_override = std::nullopt,
    bool endpoint_local_bounds = false) const
  {
    RrtPlanResult result;
    const auto started = std::chrono::steady_clock::now();
    const std::string group_name = "dual_arm_with_updown";
    const auto* group = robot_model_->getJointModelGroup(group_name);
    if (!group || group->getVariableCount() != 15U) {
      result.reason = "missing pure RRT group " + group_name;
      return result;
    }
    auto scene = planning_scene::PlanningScene::clone(base_scene);
    scene->setCurrentState(start_state);
    if (attached) {
      scene->setStateFeasibilityPredicate(
        [this](const moveit::core::RobotState& state, bool) {
          return carriedBoxFeasible(state);
        });
    }
    const auto state_valid = [&](const moveit::core::RobotState& state) {
        return state.satisfiesBounds(group) &&
          (!attached || carriedBoxFeasible(state)) &&
          fullCollisionReason(scene, state, metrics).empty();
      };
    if (!state_valid(start_state)) {
      result.reason = "pure arm+updown RRT start state invalid";
      return result;
    }
    if (!state_valid(goal_state)) {
      result.reason = "pure arm+updown RRT goal state invalid";
      return result;
    }

    const auto& variables = group->getVariableNames();
    const bool paper_ompl = backend_override == "ompl_paper";
    const bool paper_corridor = backend_override == "ompl_corridor";
    const std::string selected_backend = (paper_ompl || paper_corridor) ? "ompl" :
      (backend_override.empty() ? pure_rrt_backend_ : backend_override);
    auto space = std::make_shared<ob::RealVectorStateSpace>(variables.size());
    ob::RealVectorBounds bounds(variables.size());
    for (size_t index = 0U; index < variables.size(); ++index) {
      const auto& limit = robot_model_->getVariableBounds(variables[index]);
      if (!limit.position_bounded_ || !(limit.min_position_ < limit.max_position_)) {
        result.reason = "pure RRT variable lacks finite bounds: " + variables[index];
        return result;
      }
      double low = limit.min_position_;
      double high = limit.max_position_;
      if (endpoint_local_bounds) {
        const double start_value = start_state.getVariablePosition(variables[index]);
        const double goal_value = goal_state.getVariablePosition(variables[index]);
        const bool changes = std::abs(goal_value - start_value) > 1e-6;
        const double margin = variables[index] == "updown" ?
          (changes ? 0.15 : 0.03) : degToRad(changes ? 35.0 : 5.0);
        low = std::max(low, std::min(start_value, goal_value) - margin);
        high = std::min(high, std::max(start_value, goal_value) + margin);
      }
      bounds.setLow(index, low);
      bounds.setHigh(index, high);
    }
#ifdef ALFA_HAS_VAMP
    if (selected_backend == "vamp") {
      using alfa_robot::motion::kFullOnlineVampDimension;
      std::array<float, kFullOnlineVampDimension> start_values{};
      std::array<float, kFullOnlineVampDimension> goal_values{};
      std::array<float, kFullOnlineVampDimension> lows{};
      std::array<float, kFullOnlineVampDimension> highs{};
      if (variables.size() != kFullOnlineVampDimension) {
        result.reason = "VAMP state dimension does not match dual_arm_with_updown";
        return result;
      }
      for (size_t index = 0U; index < variables.size(); ++index) {
        start_values[index] = static_cast<float>(
          start_state.getVariablePosition(variables[index]));
        goal_values[index] = static_cast<float>(
          goal_state.getVariablePosition(variables[index]));
        lows[index] = static_cast<float>(bounds.low[index]);
        highs[index] = static_cast<float>(bounds.high[index]);
      }
      auto state_template = std::make_shared<moveit::core::RobotState>(start_state);
      const auto validator_factory = [this, base_scene, state_template,
          variables, group, attached]() {
          auto worker_scene = planning_scene::PlanningScene::clone(base_scene);
          auto worker_state = std::make_shared<moveit::core::RobotState>(*state_template);
          return [this, worker_scene, worker_state, variables, group, attached](
              const std::array<float,
                alfa_robot::motion::kFullOnlineVampDimension>& values) mutable {
              for (size_t index = 0U; index < variables.size(); ++index) {
                worker_state->setVariablePosition(variables[index], values[index]);
              }
              worker_state->zeroVelocities();
              worker_state->zeroAccelerations();
              if (attached) attachCarriedBox(*worker_state);
              else if (worker_state->hasAttachedBody(kCarriedBoxId)) {
                worker_state->clearAttachedBody(kCarriedBoxId);
              }
              worker_state->update(true);
              return worker_state->satisfiesBounds(group) &&
                (!attached || carriedBoxFeasible(*worker_state)) &&
                fullCollisionReason(worker_scene, *worker_state, nullptr).empty();
            };
        };
      const auto search_started = std::chrono::steady_clock::now();
      auto trials = alfa_robot::motion::solveFullOnlineVampRrtc(
        start_values, goal_values, lows, highs, validator_factory,
        static_cast<size_t>(vamp_workers_),
        static_cast<size_t>(vamp_max_iterations_),
        static_cast<size_t>(vamp_max_samples_),
        static_cast<float>(vamp_range_),
        vamp_seed_override.value_or(static_cast<uint32_t>(vamp_seed_)));
      result.joint_rrt_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - search_started).count();
      if (metrics) metrics->joint_rrt_ms += result.joint_rrt_ms;
      result.planner_ms = result.joint_rrt_ms;
      result.parallel_workers_launched = trials.size();
      result.parallel_workers_completed = trials.size();
      result.backend_audit = {
        {"backend", "vamp_rrt_connect_15d"},
        {"state_dimension", kFullOnlineVampDimension},
        {"endpoint_source", task_mode_ == "state_transition" ?
          "current_run_release_state_and_named_home" : "online_pose_ik"},
        {"shortcut_used", false},
        {"fallback_used", false},
        {"endpoint_local_bounds", endpoint_local_bounds},
        {"lower", lows},
        {"upper", highs},
        {"workers", nlohmann::json::array()},
      };
      for (const auto& trial : trials) {
        result.backend_audit["workers"].push_back({
          {"seed", trial.seed}, {"solved", trial.solved},
          {"iterations", trial.iterations}, {"tree_nodes", trial.tree_nodes},
          {"validity_batches", trial.validity_batches},
          {"validity_states", trial.validity_states},
          {"search_ms", trial.search_ms}, {"fcl_ms", trial.fcl_ms},
          {"sparse_states", trial.path.size()},
        });
      }
      const auto winner = std::min_element(
        trials.begin(), trials.end(), [](const auto& left, const auto& right) {
          if (left.solved != right.solved) return left.solved > right.solved;
          if (left.path.size() != right.path.size())
            return left.path.size() < right.path.size();
          return left.search_ms < right.search_ms;
        });
      if (winner == trials.end() || !winner->solved) {
        result.reason = "VAMP 15D RRT-Connect exhausted exact-solution budget";
        result.wall_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count();
        return result;
      }
      result.parallel_winner_seed = static_cast<int>(winner->seed);
      std::vector<moveit::core::RobotStatePtr> sparse;
      sparse.push_back(std::make_shared<moveit::core::RobotState>(start_state));
      for (const auto& values : winner->path) {
        auto path_state = std::make_shared<moveit::core::RobotState>(start_state);
        for (size_t index = 0U; index < variables.size(); ++index) {
          path_state->setVariablePosition(variables[index], values[index]);
        }
        path_state->zeroVelocities();
        path_state->zeroAccelerations();
        if (attached) attachCarriedBox(*path_state);
        else if (path_state->hasAttachedBody(kCarriedBoxId)) {
          path_state->clearAttachedBody(kCarriedBoxId);
        }
        path_state->update(true);
        sparse.push_back(std::move(path_state));
      }
      sparse.push_back(std::make_shared<moveit::core::RobotState>(goal_state));
      result.states.push_back(std::make_shared<moveit::core::RobotState>(start_state));
      for (size_t edge = 1U; edge < sparse.size(); ++edge) {
        const auto& from = *sparse[edge - 1U];
        const auto& to = *sparse[edge];
        double maximum_ratio = 0.0;
        for (const auto& variable : variables) {
          const double resolution = variable == "updown" ? 0.01 : degToRad(1.0);
          maximum_ratio = std::max(
            maximum_ratio,
            std::abs(to.getVariablePosition(variable) -
            from.getVariablePosition(variable)) / resolution);
        }
        const size_t steps = std::max<size_t>(
          1U, static_cast<size_t>(std::ceil(maximum_ratio)));
        for (size_t step = 1U; step <= steps; ++step) {
          const double ratio = static_cast<double>(step) / static_cast<double>(steps);
          auto probe = std::make_shared<moveit::core::RobotState>(from);
          for (const auto& variable : variables) {
            probe->setVariablePosition(
              variable, from.getVariablePosition(variable) +
              (to.getVariablePosition(variable) - from.getVariablePosition(variable)) * ratio);
          }
          probe->zeroVelocities();
          probe->zeroAccelerations();
          if (attached) attachCarriedBox(*probe);
          else if (probe->hasAttachedBody(kCarriedBoxId)) {
            probe->clearAttachedBody(kCarriedBoxId);
          }
          probe->update(true);
          if (!state_valid(*probe)) {
            result.states.clear();
            result.reason = "VAMP 15D path failed dense MoveIt/FCL validation";
            result.wall_ms = std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - started).count();
            return result;
          }
          result.states.push_back(std::move(probe));
        }
      }
      result.success = true;
      result.strategy = "pure_vamp_rrt_connect_dual_arm_updown";
      result.reason = "pure full-flow VAMP exact 15D solution";
      result.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
      return result;
    }
#endif
    space->setBounds(bounds);
#ifdef ALFA_HAS_VAMP_PAPER_KERNEL
    auto paper_sample_count = std::make_shared<size_t>(0U);
    std::shared_ptr<PaperHaltonSequence> paper_sequence;
    std::shared_ptr<PaperCorridorSequence> corridor_sequence;
    if (paper_ompl) {
      const size_t skip = paper_skip_override.value_or(paper_halton_skip_);
      paper_sequence = std::make_shared<PaperHaltonSequence>(skip);
      space->setStateSamplerAllocator(
        [paper_sequence, paper_sample_count](const ob::StateSpace* allocated_space) {
          return std::make_shared<PaperHaltonStateSampler>(
            allocated_space, paper_sequence, paper_sample_count);
        });
    }
    if (paper_corridor) {
      PaperCorridorSequence::State corridor_start{};
      PaperCorridorSequence::State corridor_goal{};
      PaperCorridorSequence::State corridor_lower{};
      PaperCorridorSequence::State corridor_upper{};
      for (size_t index = 0U; index < variables.size(); ++index) {
        corridor_start[index] = static_cast<float>(
          start_state.getVariablePosition(variables[index]));
        corridor_goal[index] = static_cast<float>(
          goal_state.getVariablePosition(variables[index]));
        corridor_lower[index] = static_cast<float>(bounds.low[index]);
        corridor_upper[index] = static_cast<float>(bounds.high[index]);
      }
      const size_t skip = paper_skip_override.value_or(paper_halton_skip_);
      corridor_sequence = std::make_shared<PaperCorridorSequence>(
        corridor_start, corridor_goal, corridor_lower, corridor_upper,
        static_cast<uint32_t>(skip + 1U), skip);
      space->setStateSamplerAllocator(
        [corridor_sequence, paper_sample_count](const ob::StateSpace* allocated_space) {
          return std::make_shared<PaperCorridorStateSampler>(
            allocated_space, corridor_sequence, paper_sample_count);
        });
    }
#endif
    og::SimpleSetup setup(space);
    auto validity_probe = std::make_shared<moveit::core::RobotState>(start_state);
    size_t paper_validity_checks = 0U;
    setup.setStateValidityChecker(
      [&, validity_probe](const ob::State* state) mutable {
        if (paper_ompl || paper_corridor) ++paper_validity_checks;
        const auto* values = state->as<
          ob::RealVectorStateSpace::StateType>()->values;
        for (size_t index = 0U; index < variables.size(); ++index) {
          validity_probe->setVariablePosition(variables[index], values[index]);
        }
        validity_probe->zeroVelocities();
        validity_probe->zeroAccelerations();
        if (attached) attachCarriedBox(*validity_probe);
        else if (validity_probe->hasAttachedBody(kCarriedBoxId)) {
          validity_probe->clearAttachedBody(kCarriedBoxId);
        }
        validity_probe->update(true);
        return state_valid(*validity_probe);
      });
    const double paper_segment_length = 1.0 / 64.0;
    setup.getSpaceInformation()->setStateValidityCheckingResolution(
      (paper_ompl || paper_corridor) ?
      paper_segment_length / space->getMaximumExtent() : 0.0005);
    ob::ScopedState<> start(space);
    ob::ScopedState<> goal(space);
    for (size_t index = 0U; index < variables.size(); ++index) {
      start[index] = start_state.getVariablePosition(variables[index]);
      goal[index] = goal_state.getVariablePosition(variables[index]);
    }
    setup.setStartAndGoalStates(start, goal);
    auto planner = std::make_shared<og::RRTConnect>(setup.getSpaceInformation());
    planner->setRange((paper_ompl || paper_corridor) ? vamp_range_ : 0.12);
    setup.setPlanner(planner);
    if (paper_ompl) {
      setup.setup();
      paper_sequence->reset();
      *paper_sample_count = 0U;
    }
    const auto search_started = std::chrono::steady_clock::now();
    bool paper_direct_motion = false;
    ob::PlannerStatus status;
    if (paper_ompl && setup.getSpaceInformation()->checkMotion(start.get(), goal.get())) {
      auto direct_path = std::make_shared<og::PathGeometric>(setup.getSpaceInformation());
      direct_path->append(start.get());
      direct_path->append(goal.get());
      setup.getProblemDefinition()->addSolutionPath(
        direct_path, false, 0.0, planner->getName());
      status = ob::PlannerStatus::EXACT_SOLUTION;
      paper_direct_motion = true;
    } else {
      status = setup.solve(rrt_planning_time_);
    }
    result.joint_rrt_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - search_started).count();
    if (metrics) metrics->joint_rrt_ms += result.joint_rrt_ms;
    result.planner_ms = result.joint_rrt_ms;
    if (endpoint_local_bounds) {
      result.backend_audit = {
        {"backend", "ompl_rrt_connect_15d"},
        {"collision_backend", "moveit_fcl_mesh"},
        {"endpoint_local_bounds", true},
        {"range", 0.12},
        {"lower", bounds.low},
        {"upper", bounds.high},
        {"shortcut_used", false},
        {"fallback_used", false},
      };
    }
#ifdef ALFA_HAS_VAMP_PAPER_KERNEL
    if (paper_ompl) {
      result.backend_audit = {
        {"backend", "ompl_rrt_connect_15d"},
        {"collision_backend", "moveit_fcl_mesh"},
        {"halton_skip", paper_skip_override.value_or(paper_halton_skip_)},
        {"halton_samples", *paper_sample_count},
        {"validity_checks", paper_validity_checks},
        {"range", vamp_range_},
        {"edge_checks_per_configuration_unit", 64},
        {"planner_setup_inside_timed_region", false},
        {"direct_motion", paper_direct_motion},
        {"shortcut_used", false},
        {"fallback_used", false},
      };
    }
    if (paper_corridor) {
      result.backend_audit = {
        {"backend", "ompl_rrt_connect_15d"},
        {"collision_backend", "moveit_fcl_mesh"},
        {"sampler", "shared_goal_corridor_mt19937"},
        {"corridor_probability", 0.9},
        {"halton_skip", paper_skip_override.value_or(paper_halton_skip_)},
        {"samples", *paper_sample_count},
        {"validity_checks", paper_validity_checks},
        {"range", vamp_range_},
        {"edge_checks_per_configuration_unit", 64},
        {"endpoint_local_bounds", endpoint_local_bounds},
        {"shortcut_used", false},
        {"fallback_used", false},
      };
    }
#endif
    if (status != ob::PlannerStatus::EXACT_SOLUTION) {
      result.reason = "pure dual-arm+updown RRTConnect status=" +
        status.asString();
      result.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
      return result;
    }

    std::vector<moveit::core::RobotStatePtr> sparse;
    sparse.push_back(std::make_shared<moveit::core::RobotState>(start_state));
    const auto& path = setup.getSolutionPath();
    for (size_t path_index = 0U; path_index < path.getStateCount(); ++path_index) {
      const auto* values = path.getState(path_index)->as<
        ob::RealVectorStateSpace::StateType>()->values;
      auto path_state = std::make_shared<moveit::core::RobotState>(start_state);
      for (size_t index = 0U; index < variables.size(); ++index) {
        path_state->setVariablePosition(variables[index], values[index]);
      }
      path_state->zeroVelocities();
      path_state->zeroAccelerations();
      if (attached) attachCarriedBox(*path_state);
      else if (path_state->hasAttachedBody(kCarriedBoxId)) {
        path_state->clearAttachedBody(kCarriedBoxId);
      }
      path_state->update(true);
      sparse.push_back(std::move(path_state));
    }
    sparse.push_back(std::make_shared<moveit::core::RobotState>(goal_state));
    result.states.push_back(std::make_shared<moveit::core::RobotState>(start_state));
    for (size_t edge = 1U; edge < sparse.size(); ++edge) {
      const auto& from = *sparse[edge - 1U];
      const auto& to = *sparse[edge];
      double maximum_ratio = 0.0;
      for (const auto& variable : group->getVariableNames()) {
        const double resolution = variable == "updown" ? 0.01 : degToRad(1.0);
        maximum_ratio = std::max(
          maximum_ratio,
          std::abs(to.getVariablePosition(variable) -
            from.getVariablePosition(variable)) / resolution);
      }
      const size_t steps = std::max<size_t>(
        1U, static_cast<size_t>(std::ceil(maximum_ratio)));
      for (size_t step = 1U; step <= steps; ++step) {
        const double ratio =
          static_cast<double>(step) / static_cast<double>(steps);
        auto probe = std::make_shared<moveit::core::RobotState>(from);
        for (const auto& variable : group->getVariableNames()) {
          probe->setVariablePosition(
            variable, from.getVariablePosition(variable) +
            (to.getVariablePosition(variable) - from.getVariablePosition(variable)) * ratio);
        }
        probe->zeroVelocities();
        probe->zeroAccelerations();
        if (attached) attachCarriedBox(*probe);
        else if (probe->hasAttachedBody(kCarriedBoxId)) {
          probe->clearAttachedBody(kCarriedBoxId);
        }
        probe->update(true);
        if (!state_valid(*probe)) {
          result.states.clear();
          result.reason = "pure arm+updown RRT path failed dense FCL validation";
          result.wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
          return result;
        }
        result.states.push_back(std::move(probe));
      }
    }
    result.success = true;
    result.strategy = "pure_ompl_rrt_connect_dual_arm_updown";
    result.reason = "pure_rrt_full_flow exact dual-arm+updown solution";
    result.wall_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return result;
  }

  RrtPlanResult planLoadedTransfer(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit::core::RobotState& start_state,
    const moveit::core::RobotState& goal_state,
    bool direct_only,
    PlanningMetrics* metrics,
    bool prefer_cartesian = false) const
  {
    if (loaded_transfer_joint_waypoints_deg_.empty()) {
      return planRrt(
        scene, start_state, goal_state, direct_only, metrics, prefer_cartesian);
    }

    const auto started = std::chrono::steady_clock::now();
    RrtPlanResult result;
    if (!loaded_transfer_waypoint_start_deg_.empty()) {
      const auto start_joints = armJoints(start_state);
      double maximum_error = 0.0;
      for (size_t index = 0U; index < start_joints.size(); ++index) {
        maximum_error = std::max(maximum_error, std::abs(normalizedAngle(
          start_joints[index] - degToRad(loaded_transfer_waypoint_start_deg_[index]))));
      }
      if (maximum_error > degToRad(0.01)) {
        result.reason = "validated waypoint start mismatch " +
          std::to_string(radToDeg(maximum_error)) + "deg";
        result.wall_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count();
        return result;
      }
    }
    result.states.push_back(std::make_shared<moveit::core::RobotState>(start_state));
    const size_t waypoint_count = loaded_transfer_joint_waypoints_deg_.size() / 7U;
    for (size_t waypoint_index = 0U; waypoint_index <= waypoint_count; ++waypoint_index) {
      moveit::core::RobotStatePtr next;
      if (waypoint_index == waypoint_count) {
        next = std::make_shared<moveit::core::RobotState>(goal_state);
      } else {
        std::array<double, 7> joints{};
        for (size_t joint_index = 0U; joint_index < joints.size(); ++joint_index) {
          joints[joint_index] = degToRad(
            loaded_transfer_joint_waypoints_deg_[waypoint_index * 7U + joint_index]);
        }
        next = std::make_shared<moveit::core::RobotState>(start_state);
        next->setJointGroupPositions(planning_group_, joints.data());
        attachCarriedBox(*next);
        next->update(true);
      }
      std::string edge_reason;
      if (!next->satisfiesBounds(planning_group_)) {
        edge_reason = "joint_bounds";
      } else if (!carriedBoxUpright(*next)) {
        edge_reason = "carried_box_tilt";
      } else if (!carriedBoxWithinHeight(*next)) {
        edge_reason = "carried_box_height";
      } else {
        edge_reason = collisionReason(scene, *next, metrics);
      }
      if (edge_reason.empty() && !edgeClear(
            scene, *result.states.back(), *next, true, metrics, &edge_reason)) {
        // edgeClear supplies the rejection reason.
      }
      if (!edge_reason.empty()) {
        if (!loaded_transfer_waypoint_start_deg_.empty()) {
          result.reason = "validated waypoint edge " +
            std::to_string(waypoint_index + 1U) + " failed: " + edge_reason;
          result.wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
          return result;
        }
        auto fallback = planRrt(
          scene, start_state, goal_state, direct_only, metrics, prefer_cartesian);
        fallback.reason += "; validated waypoint edge " +
          std::to_string(waypoint_index + 1U) + " failed: " + edge_reason;
        return fallback;
      }
      auto dense = densifyRrtReplay({result.states.back(), next});
      result.states.insert(result.states.end(), dense.begin() + 1U, dense.end());
    }
    result.success = true;
    result.strategy = "validated_joint_waypoints";
    result.reason = result.strategy + " count=" + std::to_string(waypoint_count);
    result.wall_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return result;
  }

  RrtPlanResult planCoupledLoadedArmUpdown(
    const planning_scene::PlanningSceneConstPtr& base_scene,
    const moveit::core::RobotState& start_state,
    const moveit::core::RobotState& goal_state,
    PlanningMetrics* metrics) const
  {
    RrtPlanResult result;
    const auto started = std::chrono::steady_clock::now();
    const std::string group_name = side_ + "_arm_with_updown";
    const auto* group = robot_model_->getJointModelGroup(group_name);
    if (!group || group->getVariableCount() != 8U) {
      result.reason = "missing coupled group " + group_name;
      return result;
    }
    auto scene = planning_scene::PlanningScene::clone(base_scene);
    scene->setCurrentState(start_state);
    scene->setStateFeasibilityPredicate(
      [this](const moveit::core::RobotState& state, bool) {
        return carriedBoxFeasible(state);
      });
    for (const auto* state : {&start_state, &goal_state}) {
      if (!state->satisfiesBounds(group) || !carriedBoxFeasible(*state)) {
        result.reason = "coupled endpoint violates bounds or carried-box pose";
        return result;
      }
      const std::string collision = fullCollisionReason(scene, *state, metrics);
      if (!collision.empty()) {
        result.reason = "coupled endpoint " + collision;
        return result;
      }
    }
    if (!loaded_transfer_arm_updown_waypoints_.empty()) {
      if (!loaded_transfer_arm_updown_waypoint_start_.empty()) {
        double start_error = std::abs(
          start_state.getVariablePosition("updown") -
          loaded_transfer_arm_updown_waypoint_start_[0]);
        const auto start_arm = armJoints(start_state);
        for (size_t joint = 0U; joint < start_arm.size(); ++joint) {
          start_error = std::max(start_error, std::abs(normalizedAngle(
            start_arm[joint] - degToRad(
              loaded_transfer_arm_updown_waypoint_start_[joint + 1U]))));
        }
        if (start_error > degToRad(0.01)) {
          result.reason = "validated coupled waypoint start mismatch " +
            std::to_string(radToDeg(start_error)) + "deg";
          return result;
        }
      }
      const auto sample_edge = [&](const moveit::core::RobotState& from,
          const moveit::core::RobotState& to,
          std::vector<moveit::core::RobotStatePtr>* output) {
          double maximum_ratio = 0.0;
          for (const auto& variable : group->getVariableNames()) {
            const double resolution = variable == "updown" ? 0.01 : degToRad(1.0);
            const double delta = variable == "updown" ?
              to.getVariablePosition(variable) - from.getVariablePosition(variable) :
              normalizedAngle(
                to.getVariablePosition(variable) - from.getVariablePosition(variable));
            maximum_ratio = std::max(maximum_ratio, std::abs(delta) / resolution);
          }
          const size_t steps = std::max<size_t>(
            1U, static_cast<size_t>(std::ceil(maximum_ratio)));
          for (size_t step = 1U; step <= steps; ++step) {
            auto probe = std::make_shared<moveit::core::RobotState>(from);
            from.interpolate(
              to, static_cast<double>(step) / static_cast<double>(steps),
              *probe, group);
            attachCarriedBox(*probe);
            probe->update(true);
            const std::string collision = fullCollisionReason(scene, *probe, metrics);
            if (!probe->satisfiesBounds(group) || !collision.empty()) return false;
            output->push_back(std::move(probe));
          }
          return true;
        };
      result.states.push_back(std::make_shared<moveit::core::RobotState>(start_state));
      const size_t waypoint_count = loaded_transfer_arm_updown_waypoints_.size() / 8U;
      for (size_t waypoint = 0U; waypoint <= waypoint_count; ++waypoint) {
        auto next = std::make_shared<moveit::core::RobotState>(*result.states.back());
        if (waypoint == waypoint_count) {
          *next = goal_state;
        } else {
          const size_t offset = waypoint * 8U;
          next->setVariablePosition(
            "updown", loaded_transfer_arm_updown_waypoints_[offset]);
          std::array<double, 7> joints{};
          for (size_t joint = 0U; joint < joints.size(); ++joint) {
            joints[joint] = degToRad(
              loaded_transfer_arm_updown_waypoints_[offset + joint + 1U]);
          }
          next->setJointGroupPositions(planning_group_, joints.data());
          attachCarriedBox(*next);
          next->update(true);
        }
        std::vector<moveit::core::RobotStatePtr> edge;
        if (!sample_edge(*result.states.back(), *next, &edge)) {
          result.states.clear();
          result.reason = "coupled loaded waypoint edge " +
            std::to_string(waypoint + 1U) + " failed";
          return result;
        }
        result.states.insert(result.states.end(), edge.begin(), edge.end());
      }
      result.success = true;
      result.strategy = loaded_transfer_arm_updown_waypoint_start_.empty() ?
        "revalidated_generic_coupled_loaded_waypoints" :
        "validated_coupled_loaded_waypoints";
      result.reason = result.strategy + " count=" + std::to_string(waypoint_count);
      result.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
      return result;
    }

    planning_interface::MotionPlanRequest request;
    request.group_name = group_name;
    request.planner_id = "RRTConnectkConfigDefault";
    request.allowed_planning_time = std::max(5.0, rrt_planning_time_);
    request.num_planning_attempts = std::max(4, rrt_planning_attempts_);
    request.max_velocity_scaling_factor = 1.0;
    request.max_acceleration_scaling_factor = 1.0;
    moveit::core::robotStateToRobotStateMsg(start_state, request.start_state, true);
    request.goal_constraints.push_back(
      kinematic_constraints::constructGoalConstraints(goal_state, group, 1e-3));
    planning_interface::MotionPlanResponse response;
    const auto rrt_started = std::chrono::steady_clock::now();
    const bool generated = planning_pipeline_->generatePlan(scene, request, response);
    result.joint_rrt_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - rrt_started).count();
    if (metrics) metrics->joint_rrt_ms += result.joint_rrt_ms;
    result.planner_ms = response.planning_time_ * 1000.0;
    if (!generated || response.error_code_.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS ||
        !response.trajectory_ || response.trajectory_->getWayPointCount() == 0U) {
      result.reason = "coupled RRTConnect code=" +
        std::to_string(response.error_code_.val);
      result.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
      return result;
    }

    result.states.push_back(std::make_shared<moveit::core::RobotState>(start_state));
    for (size_t waypoint = 0U;
        waypoint < response.trajectory_->getWayPointCount(); ++waypoint) {
      auto target = std::make_shared<moveit::core::RobotState>(
        response.trajectory_->getWayPoint(waypoint));
      attachCarriedBox(*target);
      target->update(true);
      const auto& from = *result.states.back();
      double maximum_ratio = 0.0;
      for (const auto& variable : group->getVariableNames()) {
        const double resolution = variable == "updown" ? 0.01 : degToRad(1.0);
        const double delta = variable == "updown" ?
          target->getVariablePosition(variable) - from.getVariablePosition(variable) :
          normalizedAngle(
            target->getVariablePosition(variable) - from.getVariablePosition(variable));
        maximum_ratio = std::max(maximum_ratio, std::abs(delta) / resolution);
      }
      const size_t steps = std::max<size_t>(
        1U, static_cast<size_t>(std::ceil(maximum_ratio)));
      for (size_t step = 1U; step <= steps; ++step) {
        auto probe = std::make_shared<moveit::core::RobotState>(from);
        from.interpolate(
          *target, static_cast<double>(step) / static_cast<double>(steps),
          *probe, group);
        attachCarriedBox(*probe);
        probe->update(true);
        const std::string collision = fullCollisionReason(scene, *probe, metrics);
        if (!probe->satisfiesBounds(group) || !collision.empty()) {
          result.states.clear();
          result.reason = "coupled path validation " +
            (collision.empty() ? std::string("joint_bounds") : collision);
          result.wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
          return result;
        }
        result.states.push_back(std::move(probe));
      }
    }
    result.states.back() = std::make_shared<moveit::core::RobotState>(goal_state);
    result.success = true;
    result.strategy = "coupled_loaded_arm_updown_rrt";
    result.reason = result.strategy;
    result.wall_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return result;
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

  TaskResult planTask(const Eigen::Vector3d& box_center)
  {
    TaskResult result;
    const auto total_started = std::chrono::steady_clock::now();
    auto finish = [&]() {
      result.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_started).count();
      return result;
    };
    const auto record_search = [this, &result](
      const std::string& stage, const RrtPlanResult& plan) {
      nlohmann::json repaired_joints = nlohmann::json::array();
      for (const size_t index : plan.repaired_joint_indices) {
        repaired_joints.push_back(side_ + "_joint" + std::to_string(index + 1U));
      }
      result.segment_search.push_back({
        {"stage", stage},
        {"success", plan.success},
        {"strategy", plan.strategy},
        {"total_ms", plan.wall_ms},
        {"tcp_shortcut_ms", plan.tcp_shortcut_ms},
        {"shortcut_repair_rrt_ms", plan.reduced_rrt_ms},
        {"shortcut_repair_samples", plan.reduced_rrt_samples},
        {"parallel_rrt_workers_launched", plan.parallel_workers_launched},
        {"parallel_rrt_workers_completed", plan.parallel_workers_completed},
        {"parallel_rrt_winner_seed", plan.parallel_winner_seed},
        {"repaired_joints", repaired_joints},
        {"joint_rrt_ms", plan.joint_rrt_ms},
        {"backend_audit", plan.backend_audit},
        {"diagnostic", plan.reason},
      });
    };
    const auto scene = makeScene(box_center);

    const std::string initial_collision = collisionReason(scene, *initial_state_, &result.metrics);
    if ((!continuous_sequence_ || continuous_plan_approach_) && !initial_collision.empty()) {
      result.failure_stage = "initial_state";
      result.failure_reason = initial_collision;
      result.frames.push_back(ReplayFrame{"initial_state", allJoints(*initial_state_), false});
      return finish();
    }

    moveit::core::RobotState return_goal(*initial_state_);
    if (!hasPlaceGoal() && !analytic_path_only_) {
      attachCarriedBox(return_goal);
      const std::string return_goal_collision = collisionReason(scene, return_goal, &result.metrics);
      if (!return_goal_collision.empty()) {
        result.failure_stage = "return_goal";
        result.failure_reason = "initial pose cannot carry box: " + return_goal_collision;
        result.frames.push_back(ReplayFrame{"initial_state", allJoints(*initial_state_), false});
        return finish();
      }
    }

    std::string precontact_rejections;
    std::vector<AnalyticCandidate> precontact_candidates;
    if (!precontact_arm_joints_deg_.empty()) {
      std::array<double, 7> joints{};
      std::transform(
        precontact_arm_joints_deg_.begin(), precontact_arm_joints_deg_.end(),
        joints.begin(), [](double value) {return degToRad(value);});
      auto state = std::make_shared<moveit::core::RobotState>(*initial_state_);
      state->setJointGroupPositions(planning_group_, joints.data());
      state->update(true);
      const auto& actual = state->getGlobalLinkTransform(tool_link_);
      const auto target = precontactPose(box_center);
      if (!state->satisfiesBounds(planning_group_)) {
        precontact_rejections = "fixed precontact violates joint bounds";
      } else if ((actual.translation() - target.translation()).norm() > 0.001 ||
          Eigen::Quaterniond(actual.linear()).angularDistance(
            Eigen::Quaterniond(target.linear())) > degToRad(0.5)) {
        precontact_rejections = "fixed precontact FK error";
      } else {
        const auto collision = collisionReason(scene, *state, &result.metrics);
        if (!collision.empty()) {
          precontact_rejections = "fixed precontact " + collision;
        } else {
          AnalyticCandidate candidate;
          candidate.state = std::move(state);
          candidate.solution.joints = joints;
          candidate.score = 0.0;
          precontact_candidates.push_back(std::move(candidate));
        }
      }
    } else if (pure_rrt_full_flow_) {
      const auto& lift_bounds = robot_model_->getVariableBounds("updown");
      std::vector<double> lift_samples{
        initial_state_->getVariablePosition("updown")};
      constexpr double kLiftGoalStep = 0.25;
      for (double lift = lift_bounds.max_position_;
          lift >= lift_bounds.min_position_ - 1e-9; lift -= kLiftGoalStep) {
        if (std::none_of(
              lift_samples.begin(), lift_samples.end(),
              [lift](double existing) {return std::abs(existing - lift) < 1e-9;})) {
          lift_samples.push_back(lift);
        }
      }
      std::sort(
        lift_samples.begin(), lift_samples.end(),
        [this](double left, double right) {
          const double current = initial_state_->getVariablePosition("updown");
          return std::abs(left - current) < std::abs(right - current);
        });
      std::ostringstream lift_diagnostics;
      std::vector<AnalyticCandidate> whole_body_candidates;
      for (const double lift : lift_samples) {
        moveit::core::RobotState lift_seed(*initial_state_);
        lift_seed.setVariablePosition("updown", lift);
        lift_seed.update(true);
        std::string rejection;
        auto candidates = solvePoseCandidates(
          precontactPose(box_center), lift_seed, false, scene, &result.metrics,
          false, &rejection);
        for (auto& candidate : candidates) {
          candidate.score += 0.1 * std::abs(
            lift - initial_state_->getVariablePosition("updown"));
          std::vector<moveit::core::RobotStatePtr> target_states{
            candidate.state};
          std::string opposite_reason;
          if (!assignCollisionFreeOppositeArm(
                scene, &target_states, &target_states,
                &result.metrics, &opposite_reason)) {
            precontact_rejections += " opposite=[" + opposite_reason + "]";
            continue;
          }
          candidate.state = target_states.front();
          whole_body_candidates.push_back(std::move(candidate));
          if (whole_body_candidates.size() >= precontact_candidate_limit_) break;
        }
        lift_diagnostics << " h=" << lift << "[" << rejection << "]";
        if (whole_body_candidates.size() >= precontact_candidate_limit_) break;
      }
      std::stable_sort(
        whole_body_candidates.begin(), whole_body_candidates.end(),
        [](const AnalyticCandidate& left, const AnalyticCandidate& right) {
          return left.score < right.score;
        });
      precontact_rejections = lift_diagnostics.str();
      precontact_candidates = std::move(whole_body_candidates);
    } else {
      precontact_candidates = solvePoseCandidates(
        precontactPose(box_center), *initial_state_, false, scene, &result.metrics,
        false, &precontact_rejections);
    }
    if (precontact_candidates.empty()) {
      result.failure_stage = "precontact_ik";
      result.failure_reason = precontact_rejections;
      result.frames.push_back(ReplayFrame{"initial_state", allJoints(*initial_state_), false});
      return finish();
    }
    if (precontact_candidates.size() > precontact_candidate_limit_) {
      precontact_candidates.resize(precontact_candidate_limit_);
    }

    std::string last_failure_stage = "candidate_search";
    std::string last_failure_reason = "no candidate attempted";
    std::vector<ReplayFrame> best_partial;
    for (size_t candidate_index = 0; candidate_index < precontact_candidates.size(); ++candidate_index) {
      const auto analytic_started = std::chrono::steady_clock::now();
      std::vector<moveit::core::RobotStatePtr> approach_states;
      std::vector<moveit::core::RobotStatePtr> retreat_states;
      std::string analytic_failure_stage;
      std::string analytic_failure_reason;
      RrtPlanResult pure_precontact_rrt;
      if (pure_rrt_full_flow_) {
        pure_precontact_rrt = planPureArmUpdownRrt(
          scene, *initial_state_, *precontact_candidates[candidate_index].state,
          false, &result.metrics);
        record_search("to_precontact", pure_precontact_rrt);
        result.metrics.rrt_approach_ms += pure_precontact_rrt.wall_ms;
        if (!pure_precontact_rrt.success) {
          last_failure_stage = "rrt_to_precontact";
          last_failure_reason = "candidate " + std::to_string(candidate_index) +
            " " + pure_precontact_rrt.reason;
          continue;
        }
      }
      bool analytic_ok = false;
      if (pure_rrt_full_flow_) {
        std::string contact_rejections;
        const auto contact_candidates = solvePoseCandidates(
          contactPose(box_center), *precontact_candidates[candidate_index].state,
          false, scene, &result.metrics, false, &contact_rejections);
        for (const auto& contact_candidate : contact_candidates) {
          auto to_contact = planRrt(
            scene, *precontact_candidates[candidate_index].state,
            *contact_candidate.state, false, &result.metrics);
          record_search("precontact_to_contact", to_contact);
          if (!to_contact.success) {
            analytic_failure_reason = to_contact.reason;
            continue;
          }
          moveit::core::RobotState loaded_contact(*contact_candidate.state);
          attachCarriedBox(loaded_contact);
          loaded_contact.update(true);
          std::string retreat_rejections;
          const auto retreat_candidates = solvePoseCandidates(
            retreatPose(box_center), loaded_contact, true, scene,
            &result.metrics, false, &retreat_rejections);
          for (const auto& retreat_candidate : retreat_candidates) {
            auto to_retreat = planRrt(
              scene, loaded_contact, *retreat_candidate.state,
              false, &result.metrics);
            record_search("contact_to_retreat_loaded", to_retreat);
            if (!to_retreat.success) {
              analytic_failure_reason = to_retreat.reason;
              continue;
            }
            approach_states = std::move(to_contact.states);
            retreat_states = std::move(to_retreat.states);
            if (retreat_states.empty() ||
                !retreat_states.front()->hasAttachedBody(kCarriedBoxId)) {
              retreat_states.insert(
                retreat_states.begin(),
                std::make_shared<moveit::core::RobotState>(loaded_contact));
            }
            analytic_ok = true;
            break;
          }
          if (analytic_ok) break;
          analytic_failure_reason = retreat_rejections;
        }
        analytic_failure_stage = analytic_ok ? "" : "pure_rrt_grasp_motion";
        if (!analytic_ok && analytic_failure_reason.empty()) {
          analytic_failure_reason = contact_rejections;
        }
      } else {
        analytic_ok = traceCartesianPath(
          box_center, *precontact_candidates[candidate_index].state, scene,
          &result.metrics, &approach_states, &retreat_states,
          &analytic_failure_stage, &analytic_failure_reason);
      }
      result.metrics.analytic_path_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - analytic_started).count();
      if (!analytic_ok) {
        last_failure_stage = analytic_failure_stage;
        last_failure_reason = "candidate " + std::to_string(candidate_index) + " " +
          analytic_failure_reason;
        continue;
      }
      if (auto_safe_opposite_arm_) {
        std::string opposite_reason;
        if (!assignCollisionFreeOppositeArm(
              scene, &approach_states, &retreat_states,
              &result.metrics, &opposite_reason)) {
          last_failure_stage = "opposite_arm_safe_pose";
          last_failure_reason = "candidate " + std::to_string(candidate_index) +
            " " + opposite_reason;
          continue;
        }
        precontact_candidates[candidate_index].state = approach_states.front();
      }

      if (analytic_path_only_) {
        appendStates(
          densifyCartesianReplay(approach_states),
          "cartesian_approach", false, true, &result.frames);
        if (!retreat_states.empty()) {
          result.frames.push_back(
            ReplayFrame{"attach_box", allJoints(*retreat_states.front()), true});
        }
        appendStates(
          densifyCartesianReplay(retreat_states),
          "cartesian_retreat", true, true, &result.frames);
        result.success = true;
        return finish();
      }

      RrtPlanResult approach_rrt;
      if (pure_rrt_full_flow_) {
        approach_rrt = std::move(pure_precontact_rrt);
      } else if (
          continuous_sequence_ && !continuous_plan_approach_) {
        // The sequence planner connects the last release directly to this
        // precontact state with whole-body collision checking.
        approach_rrt.success = true;
        approach_rrt.states.push_back(precontact_candidates[candidate_index].state);
      } else {
        approach_rrt = planRrt(
          scene, *initial_state_, *precontact_candidates[candidate_index].state,
          false, &result.metrics, continuous_plan_approach_);
        record_search("to_precontact", approach_rrt);
      }
      if (!pure_rrt_full_flow_) {
        result.metrics.rrt_approach_ms += approach_rrt.wall_ms;
      }
      if (!approach_rrt.success) {
        last_failure_stage = "rrt_to_precontact";
        last_failure_reason = "candidate " + std::to_string(candidate_index) + " " +
          approach_rrt.reason;
        continue;
      }

      std::vector<ReplayFrame> executable_prefix;
      const std::string approach_stage = pure_rrt_full_flow_ ?
        "rrt_to_precontact" :
        (continuous_plan_approach_ ? "tcp_to_precontact" :
        (continuous_sequence_ ? "precontact" : "rrt_to_precontact"));
      appendStates(
        approach_rrt.states, approach_stage, false, false, &executable_prefix);
      appendStates(
        pure_rrt_full_flow_ ? approach_states :
          densifyCartesianReplay(approach_states),
        pure_rrt_full_flow_ ? "rrt_to_contact" : "cartesian_approach",
        false, true, &executable_prefix);
      if (!retreat_states.empty()) {
        executable_prefix.push_back(
          ReplayFrame{"attach_box", allJoints(*retreat_states.front()), true});
      }
      appendStates(
        pure_rrt_full_flow_ ? retreat_states :
          densifyCartesianReplay(retreat_states),
        pure_rrt_full_flow_ ? "rrt_loaded_retreat" : "cartesian_retreat",
        true, true, &executable_prefix);
      if (executable_prefix.size() > best_partial.size()) {
        best_partial = executable_prefix;
      }

      if (!hasPlaceGoal()) {
        const auto return_rrt = planRrt(
          scene, *retreat_states.back(), return_goal, false, &result.metrics);
        record_search("return", return_rrt);
        result.metrics.rrt_return_ms += return_rrt.wall_ms;
        if (!return_rrt.success) {
          last_failure_stage = "rrt_return";
          last_failure_reason = "candidate " + std::to_string(candidate_index) + " " +
            return_rrt.reason;
          continue;
        }
        result.frames = std::move(executable_prefix);
        appendStates(return_rrt.states, "rrt_return", true, true, &result.frames);
        result.success = true;
        return finish();
      }

      std::vector<moveit::core::RobotStatePtr> to_updown_safe;
      std::vector<moveit::core::RobotStatePtr> updown_to_place;
      const double pickup_updown = retreat_states.back()->getVariablePosition("updown");
      const bool low_clearance_staging = place_updown_enabled_ &&
        !place_arm_joints_deg_.empty() && loaded_transfer_defer_place_updown_ &&
        pickup_updown < place_updown_ - 1e-9;
      const bool automatic_top_defer = grasp_mode_ == "top_suction" &&
        pickup_updown > place_updown_;
      const bool defer_place_updown = place_updown_enabled_ &&
        !place_arm_joints_deg_.empty() && !low_clearance_staging &&
        (loaded_transfer_defer_place_updown_ || automatic_top_defer);
      moveit::core::RobotStatePtr place_seed = retreat_states.back();
      if (place_updown_enabled_ && !defer_place_updown && !low_clearance_staging) {
        std::string updown_reason;
        if (!traceUpdownTransition(
              scene, *retreat_states.back(), place_updown_, true, &result.metrics,
              &updown_to_place, &updown_reason)) {
          moveit::core::RobotState safe_goal(*initial_state_);
          attachCarriedBox(safe_goal);
          const auto safe_plan = planRrt(
            scene, *retreat_states.back(), safe_goal, false, &result.metrics);
          record_search("to_updown_safe", safe_plan);
          result.metrics.rrt_return_ms += safe_plan.wall_ms;
          if (!safe_plan.success) {
            last_failure_stage = "rrt_to_updown_safe";
            last_failure_reason = "candidate " + std::to_string(candidate_index) + " " +
              safe_plan.reason + " after " + updown_reason +
              " defer=" + (loaded_transfer_defer_place_updown_ ? "true" : "false") +
              " pickup_updown=" + std::to_string(pickup_updown) +
              " place_updown=" + std::to_string(place_updown_);
            continue;
          }
          to_updown_safe = safe_plan.states;
          if (!traceUpdownTransition(
                scene, *to_updown_safe.back(), place_updown_, true, &result.metrics,
                &updown_to_place, &updown_reason)) {
            last_failure_stage = "updown_to_place";
            last_failure_reason = "candidate " + std::to_string(candidate_index) + " " +
              updown_reason;
            continue;
          }
        }
        place_seed = updown_to_place.back();
      }

      std::string place_rejections;
      std::vector<AnalyticCandidate> place_candidates;
      if (!place_arm_joints_deg_.empty()) {
        std::array<double, 7> named_joints{};
        std::transform(
          place_arm_joints_deg_.begin(), place_arm_joints_deg_.end(),
          named_joints.begin(), [](double value) {return degToRad(value);});
        moveit::core::RobotState named_place(*place_seed);
        named_place.setJointGroupPositions(planning_group_, named_joints.data());
        if (low_clearance_staging) {
          named_place.setVariablePosition("updown", place_updown_);
        }
        named_place.enforceBounds(planning_group_);
        attachCarriedBox(named_place);
        named_place.update(true);
        if (!named_place.satisfiesBounds(planning_group_)) {
          place_rejections = "named unloading pose violates joint bounds";
        } else {
          const std::string collision = collisionReason(scene, named_place, &result.metrics);
          if (!collision.empty()) {
            place_rejections = "named unloading pose collision: " + collision;
          } else {
            AnalyticCandidate candidate;
            candidate.state = std::make_shared<moveit::core::RobotState>(named_place);
            candidate.solution.joints = named_joints;
            candidate.score = 0.0;
            place_candidates.push_back(std::move(candidate));
          }
        }
      } else if (pure_rrt_full_flow_) {
        const auto& lift_bounds = robot_model_->getVariableBounds("updown");
        std::vector<double> lift_samples{
          place_seed->getVariablePosition("updown")};
        constexpr double kLiftGoalStep = 0.25;
        for (double lift = lift_bounds.max_position_;
            lift >= lift_bounds.min_position_ - 1e-9; lift -= kLiftGoalStep) {
          if (std::none_of(
                lift_samples.begin(), lift_samples.end(),
                [lift](double existing) {return std::abs(existing - lift) < 1e-9;})) {
            lift_samples.push_back(lift);
          }
        }
        std::sort(
          lift_samples.begin(), lift_samples.end(),
          [place_seed](double left, double right) {
            const double current = place_seed->getVariablePosition("updown");
            return std::abs(left - current) < std::abs(right - current);
          });
        std::ostringstream diagnostics;
        for (const double lift : lift_samples) {
          moveit::core::RobotState lift_seed(*place_seed);
          lift_seed.setVariablePosition("updown", lift);
          attachCarriedBox(lift_seed);
          lift_seed.update(true);
          std::string rejection;
          auto candidates = solvePoseCandidates(
            placeTcpPose(), lift_seed, true, scene, &result.metrics,
            false, &rejection);
          for (auto& candidate : candidates) {
            if (!fullCollisionReason(scene, *candidate.state, &result.metrics).empty()) {
              continue;
            }
            candidate.score += 0.1 * std::abs(
              lift - place_seed->getVariablePosition("updown"));
            place_candidates.push_back(std::move(candidate));
            if (place_candidates.size() >= 2U) break;
          }
          diagnostics << " h=" << lift << "[" << rejection << "]";
          if (place_candidates.size() >= 2U) break;
        }
        std::stable_sort(
          place_candidates.begin(), place_candidates.end(),
          [](const AnalyticCandidate& left, const AnalyticCandidate& right) {
            return left.score < right.score;
          });
        place_rejections = diagnostics.str();
      } else {
        place_candidates = solvePoseCandidates(
          placeTcpPose(), *place_seed, true, scene, &result.metrics,
          false, &place_rejections);
      }
      if (place_candidates.empty()) {
        last_failure_stage = "place_tcp_ik";
        last_failure_reason = place_rejections;
        continue;
      }
      if (natural_motion_enabled_ && grasp_mode_ == "front" &&
          natural_place_return_weight_ > 0.0) {
        const auto ready_joints = armJoints(*initial_state_);
        for (auto& candidate : place_candidates) {
          candidate.score += natural_place_return_weight_ * weightedSquaredJointDistance(
            armJoints(*candidate.state), ready_joints, kNaturalJointWeights);
        }
        std::stable_sort(
          place_candidates.begin(), place_candidates.end(),
          [](const AnalyticCandidate& lhs, const AnalyticCandidate& rhs) {
            return lhs.score < rhs.score;
          });
      }
      const size_t place_candidate_limit =
        grasp_mode_ == "front" && !place_tcp_pose_.empty() ? 2U : 8U;
      for (size_t place_index = 0; place_index <
          std::min(place_candidates.size(), place_candidate_limit); ++place_index) {
        const auto& place_state = *place_candidates[place_index].state;
        RrtPlanResult to_place;
        if (pure_rrt_full_flow_) {
          to_place = planPureArmUpdownRrt(
            scene, *retreat_states.back(), place_state, true, &result.metrics);
          record_search("loaded_transfer_dual_arm_updown", to_place);
          result.metrics.rrt_return_ms += to_place.wall_ms;
        } else if (low_clearance_staging) {
          moveit::core::RobotState low_place(place_state);
          low_place.setVariablePosition("updown", pickup_updown);
          low_place.update(true);
          bool staged = false;
          std::string staged_reason = "no collision-free low carry posture";
          auto coupled = planCoupledLoadedArmUpdown(
            scene, *retreat_states.back(), place_state, &result.metrics);
          record_search("coupled_loaded_arm_updown", coupled);
          result.metrics.rrt_return_ms += coupled.wall_ms;
          if (coupled.success) {
            to_place = std::move(coupled);
            staged = true;
          } else {
            staged_reason = coupled.reason;
          }
          const std::string coupled_reason = staged_reason;
          struct LowCarryCandidate
          {
            double score{0.0};
            moveit::core::RobotStatePtr state;
            std::vector<moveit::core::RobotStatePtr> lift_states;
          };
          std::vector<LowCarryCandidate> low_candidates;
          const auto collect_low_candidate = [&](moveit::core::RobotStatePtr candidate) {
              attachCarriedBox(*candidate);
              candidate->update(true);
              const std::string low_collision = collisionReason(
                scene, *candidate, &result.metrics);
              if (!low_collision.empty()) {
                staged_reason = "low candidate collision: " + low_collision;
                return;
              }
              std::vector<moveit::core::RobotStatePtr> lift_states;
              std::string lift_reason;
              if (!traceUpdownTransition(
                    scene, *candidate, place_updown_, true, &result.metrics,
                    &lift_states, &lift_reason)) {
                staged_reason = "low candidate lift: " + lift_reason;
                return;
              }
              const double score = weightedSquaredJointDistance(
                armJoints(*retreat_states.back()), armJoints(*candidate),
                kNaturalJointWeights) + weightedSquaredJointDistance(
                armJoints(*candidate), armJoints(place_state), kNaturalJointWeights);
              low_candidates.push_back(
                LowCarryCandidate{score, std::move(candidate), std::move(lift_states)});
            };
          constexpr size_t kLowCarrySamples = 32U;
          for (size_t sample = 1U; !staged && sample < kLowCarrySamples; ++sample) {
            const double ratio = static_cast<double>(sample) /
              static_cast<double>(kLowCarrySamples);
            auto low_safe = std::make_shared<moveit::core::RobotState>(
              *retreat_states.back());
            retreat_states.back()->interpolate(
              low_place, ratio, *low_safe, planning_group_);
            collect_low_candidate(std::move(low_safe));
          }
          constexpr size_t kRandomLowCarrySamples = 1024U;
          for (size_t sample = 0U; !staged && sample < kRandomLowCarrySamples; ++sample) {
            auto low_safe = std::make_shared<moveit::core::RobotState>(
              *retreat_states.back());
            low_safe->setToRandomPositions(planning_group_);
            collect_low_candidate(std::move(low_safe));
          }
          std::stable_sort(
            low_candidates.begin(), low_candidates.end(),
            [](const LowCarryCandidate& lhs, const LowCarryCandidate& rhs) {
              return lhs.score < rhs.score;
            });
          if (!staged && low_candidates.empty()) {
            staged_reason += " after " + std::to_string(
              kLowCarrySamples - 1U + kRandomLowCarrySamples) + " samples";
          }
          const size_t candidate_limit = std::min<size_t>(low_candidates.size(), 48U);
          for (size_t low_index = 0U;
              !staged && low_index < candidate_limit; ++low_index) {
            auto& candidate = low_candidates[low_index];
            auto to_low_safe = planLoadedTransfer(
              scene, *retreat_states.back(), *candidate.state, false, &result.metrics);
            record_search("loaded_to_low_safe", to_low_safe);
            result.metrics.rrt_return_ms += to_low_safe.wall_ms;
            if (!to_low_safe.success) {
              staged_reason = "low candidate transfer: " + to_low_safe.reason;
              continue;
            }
            auto after_lift = planLoadedTransfer(
              scene, *candidate.lift_states.back(), place_state,
              top_loaded_transfer_direct_only_ && grasp_mode_ == "top_suction",
              &result.metrics);
            record_search("loaded_after_lift", after_lift);
            result.metrics.rrt_return_ms += after_lift.wall_ms;
            if (!after_lift.success) {
              staged_reason = "after-lift transfer: " + after_lift.reason;
              continue;
            }
            to_updown_safe = std::move(to_low_safe.states);
            updown_to_place = std::move(candidate.lift_states);
            to_place = std::move(after_lift);
            staged = true;
            break;
          }
          if (!staged) {
            last_failure_stage = "loaded_low_clearance_staging";
            last_failure_reason = "candidate " + std::to_string(candidate_index) +
              " place " + std::to_string(place_index) + " coupled=[" +
              coupled_reason + "] fallback=[" + staged_reason + "]";
            continue;
          }
        } else {
          to_place = planLoadedTransfer(
            scene, *place_seed, place_state,
            top_loaded_transfer_direct_only_ && grasp_mode_ == "top_suction",
            &result.metrics, !place_tcp_pose_.empty());
          record_search("loaded_transfer", to_place);
          result.metrics.rrt_return_ms += to_place.wall_ms;
        }
        if (!to_place.success) {
          last_failure_stage = "rrt_to_place";
          last_failure_reason = "candidate " + std::to_string(candidate_index) +
            " place " + std::to_string(place_index) + " " + to_place.reason;
          continue;
        }
        std::vector<moveit::core::RobotStatePtr> updown_at_place;
        moveit::core::RobotState final_place(place_state);
        if (defer_place_updown) {
          std::string updown_reason;
          if (!traceUpdownTransition(
                scene, place_state, place_updown_, true, &result.metrics,
                &updown_at_place, &updown_reason)) {
            last_failure_stage = "updown_at_place";
            last_failure_reason = "candidate " + std::to_string(candidate_index) +
              " place " + std::to_string(place_index) + " " + updown_reason;
            continue;
          }
          final_place = *updown_at_place.back();
        }
        moveit::core::RobotState released(final_place);
        released.clearAttachedBody(kCarriedBoxId);
        released.update(true);
        if (continuous_sequence_) {
          result.frames = executable_prefix;
          appendStates(to_updown_safe, "rrt_to_updown_safe", true, true, &result.frames);
          appendStates(updown_to_place, "updown_to_place", true, true, &result.frames);
          appendStates(to_place.states, "rrt_to_place", true, true, &result.frames);
          appendStates(updown_at_place, "updown_at_place", true, true, &result.frames);
          result.frames.push_back(ReplayFrame{"release_at_place", allJoints(released), false});
          result.achieved_place_tcp = final_place.getGlobalLinkTransform(tool_link_);
          result.success = true;
          return finish();
        }
        moveit::core::RobotState place_ready(*initial_state_);
        if (place_updown_enabled_) {
          place_ready.setVariablePosition("updown", place_updown_);
          place_ready.update(true);
        }
        const auto back_to_ready = planRrt(
          scene, released, place_ready, false, &result.metrics);
        record_search("back_to_ready", back_to_ready);
        result.metrics.rrt_return_ms += back_to_ready.wall_ms;
        if (!back_to_ready.success) {
          last_failure_stage = "rrt_to_ready";
          last_failure_reason = "candidate " + std::to_string(candidate_index) +
            " place " + std::to_string(place_index) + " " + back_to_ready.reason;
          continue;
        }
        std::vector<moveit::core::RobotStatePtr> updown_to_task;
        if (place_updown_enabled_) {
          std::string updown_reason;
          if (!traceUpdownTransition(
                scene, *back_to_ready.states.back(),
                initial_state_->getVariablePosition("updown"), false, &result.metrics,
                &updown_to_task, &updown_reason)) {
            last_failure_stage = "updown_to_task";
            last_failure_reason = "candidate " + std::to_string(candidate_index) +
              " place " + std::to_string(place_index) + " " + updown_reason;
            continue;
          }
        }
        result.frames = executable_prefix;
        appendStates(
          to_updown_safe, "rrt_to_updown_safe", true, true, &result.frames);
        appendStates(
          updown_to_place, "updown_to_place", true, true, &result.frames);
        appendStates(to_place.states, "rrt_to_place", true, true, &result.frames);
        appendStates(updown_at_place, "updown_at_place", true, true, &result.frames);
        result.frames.push_back(ReplayFrame{"release_at_place", allJoints(released), false});
        appendStates(back_to_ready.states, "rrt_to_ready", false, true, &result.frames);
        appendStates(
          updown_to_task, "updown_to_task", false, true, &result.frames);
        result.achieved_place_tcp = final_place.getGlobalLinkTransform(tool_link_);
        result.success = true;
        return finish();
      }
    }

    result.failure_stage = last_failure_stage;
    result.failure_reason = last_failure_reason;
    result.frames = std::move(best_partial);
    if (result.frames.empty()) {
      result.frames.push_back(ReplayFrame{"initial_state", allJoints(*initial_state_), false});
    }
    return finish();
  }

  TaskResult planStateTransition(const Eigen::Vector3d& box_center)
  {
    TaskResult result;
    const auto started = std::chrono::steady_clock::now();
    const auto finish = [&]() {
      result.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
      return result;
    };
    const auto* whole_body = robot_model_->getJointModelGroup("whole_body");
    if (!whole_body || ignore_opposite_arm_) {
      result.failure_stage = "transition_contract";
      result.failure_reason = "whole_body group and both-arm collision checks are required";
      return finish();
    }
    auto scene = makeScene(box_center);
    moveit::core::RobotState goal(*initial_state_);
    for (size_t index = 0; index < all_joint_names_.size(); ++index) {
      goal.setVariablePosition(all_joint_names_[index], transition_to_joints_[index]);
    }
    goal.enforceBounds(whole_body);
    goal.update(true);
    const std::string opposite_side = side_ == "left" ? "right" : "left";
    bool opposite_unchanged = true;
    for (int index = 1; index <= 7; ++index) {
      const std::string name = opposite_side + "_joint" + std::to_string(index);
      opposite_unchanged = opposite_unchanged && std::abs(
        initial_state_->getVariablePosition(name) - goal.getVariablePosition(name)) < 1e-6;
    }
    const std::string transition_group_name = opposite_unchanged ?
      side_ + "_arm_with_updown" : "whole_body";
    const auto* transition_group = robot_model_->getJointModelGroup(transition_group_name);
    if (!transition_group) {
      result.failure_stage = "transition_contract";
      result.failure_reason = "missing transition group " + transition_group_name;
      return finish();
    }
    result.transition_group = transition_group_name;
    for (const auto& state : {initial_state_.get(), &goal}) {
      if (!state->satisfiesBounds(whole_body)) {
        result.failure_stage = "transition_bounds";
        result.failure_reason = "start or goal violates whole_body joint limits";
        return finish();
      }
      const auto collision = alfa_robot::motion::scene_collision_reason(scene, *state, nullptr);
      if (!collision.empty()) {
        result.failure_stage = "transition_endpoint";
        result.failure_reason = collision;
        return finish();
      }
    }

    if (pure_rrt_full_flow_ && pure_rrt_backend_ == "vamp") {
      auto plan = planPureArmUpdownRrt(
        scene, *initial_state_, goal, false, &result.metrics);
      result.segment_search.push_back({
        {"stage", "state_transition"},
        {"success", plan.success},
        {"strategy", plan.strategy},
        {"total_ms", plan.wall_ms},
        {"joint_rrt_ms", plan.joint_rrt_ms},
        {"backend_audit", plan.backend_audit},
        {"diagnostic", plan.reason},
      });
      if (!plan.success) {
        result.failure_stage = "transition_vamp";
        result.failure_reason = plan.reason;
        return finish();
      }
      appendStates(plan.states, "between_boxes", false, false, &result.frames);
      result.transition_group = "dual_arm_with_updown";
      result.transition_strategy = "pure_vamp_rrt_connect_dual_arm_updown";
      result.transition_diagnostic = plan.reason;
      result.success = true;
      return finish();
    }

    const auto sample_edge = [&](
      const moveit::core::RobotState& from,
      const moveit::core::RobotState& to,
      std::vector<moveit::core::RobotStatePtr>* states) {
        double max_ratio = 0.0;
        for (const auto& name : transition_group->getVariableNames()) {
          const double delta = std::abs(to.getVariablePosition(name) - from.getVariablePosition(name));
          max_ratio = std::max(max_ratio, delta /
            (name == "updown" ? 0.01 : edge_joint_resolution_));
        }
        const size_t steps = std::max<size_t>(1, static_cast<size_t>(std::ceil(max_ratio)));
        for (size_t step = 1; step <= steps; ++step) {
          auto state = std::make_shared<moveit::core::RobotState>(from);
          from.interpolate(to, static_cast<double>(step) / steps, *state, transition_group);
          state->update(true);
          if (!state->satisfiesBounds(whole_body) ||
              !alfa_robot::motion::scene_collision_reason(scene, *state, nullptr).empty()) {
            return false;
          }
          states->push_back(std::move(state));
        }
        return true;
      };

    const auto validate_full_path = [&sample_edge](
      const std::vector<moveit::core::RobotStatePtr>& input,
      std::vector<moveit::core::RobotStatePtr>* output) {
        if (!output || input.empty()) return false;
        output->clear();
        output->push_back(std::make_shared<moveit::core::RobotState>(*input.front()));
        for (size_t index = 1U; index < input.size(); ++index) {
          std::vector<moveit::core::RobotStatePtr> edge;
          if (!sample_edge(*output->back(), *input[index], &edge)) return false;
          output->insert(output->end(), edge.begin(), edge.end());
        }
        return true;
      };

    if (!transition_whole_body_waypoints_.empty()) {
      if (transition_group_name != "whole_body") {
        result.failure_stage = "transition_contract";
        result.failure_reason = "whole-body waypoints require both arms to change";
        return finish();
      }
      if (transition_whole_body_waypoint_start_.empty()) {
        result.failure_stage = "transition_contract";
        result.failure_reason = "validated whole-body waypoints require a start fingerprint";
        return finish();
      }
      const auto* left_group = robot_model_->getJointModelGroup("left_arm");
      const auto* right_group = robot_model_->getJointModelGroup("right_arm");
      if (!left_group || !right_group) {
        result.failure_stage = "transition_contract";
        result.failure_reason = "missing arm group for whole-body waypoints";
        return finish();
      }
      const double updown_error = std::abs(
        initial_state_->getVariablePosition("updown") -
        transition_whole_body_waypoint_start_[0]);
      double maximum_joint_error = 0.0;
      for (size_t joint = 0U; joint < 7U; ++joint) {
        maximum_joint_error = std::max(maximum_joint_error, std::abs(normalizedAngle(
          initial_state_->getVariablePosition("left_joint" + std::to_string(joint + 1U)) -
          degToRad(transition_whole_body_waypoint_start_[joint + 1U]))));
        maximum_joint_error = std::max(maximum_joint_error, std::abs(normalizedAngle(
          initial_state_->getVariablePosition("right_joint" + std::to_string(joint + 1U)) -
          degToRad(transition_whole_body_waypoint_start_[joint + 8U]))));
      }
      if (updown_error > 1e-5 || maximum_joint_error > degToRad(0.01)) {
        result.failure_stage = "transition_waypoint_start";
        result.failure_reason = "validated whole-body waypoint start mismatch " +
          std::to_string(updown_error) + "m/" +
          std::to_string(radToDeg(maximum_joint_error)) + "deg";
        return finish();
      }
      std::vector<moveit::core::RobotStatePtr> validated{
        std::make_shared<moveit::core::RobotState>(*initial_state_)};
      const size_t waypoint_count = transition_whole_body_waypoints_.size() / 15U;
      for (size_t waypoint = 0U; waypoint <= waypoint_count; ++waypoint) {
        auto next = std::make_shared<moveit::core::RobotState>(*validated.back());
        if (waypoint == waypoint_count) {
          *next = goal;
        } else {
          const size_t offset = waypoint * 15U;
          next->setVariablePosition(
            "updown", transition_whole_body_waypoints_[offset]);
          std::array<double, 7> left_joints{};
          std::array<double, 7> right_joints{};
          for (size_t joint = 0U; joint < 7U; ++joint) {
            left_joints[joint] = degToRad(
              transition_whole_body_waypoints_[offset + joint + 1U]);
            right_joints[joint] = degToRad(
              transition_whole_body_waypoints_[offset + joint + 8U]);
          }
          next->setJointGroupPositions(left_group, left_joints.data());
          next->setJointGroupPositions(right_group, right_joints.data());
          next->update(true);
        }
        std::vector<moveit::core::RobotStatePtr> edge;
        if (!sample_edge(*validated.back(), *next, &edge)) {
          result.failure_stage = "transition_waypoint_edge";
          result.failure_reason = "validated whole-body transition waypoint edge " +
            std::to_string(waypoint + 1U) + " failed";
          return finish();
        }
        validated.insert(validated.end(), edge.begin(), edge.end());
      }
      result.transition_strategy = "validated_transition_whole_body_waypoints";
      result.transition_diagnostic = "count=" + std::to_string(waypoint_count);
      appendStates(validated, "between_boxes", false, false, &result.frames);
      result.success = true;
      return finish();
    }

    if (!transition_joint_waypoints_.empty()) {
      if (transition_waypoint_start_.empty()) {
        result.failure_stage = "transition_contract";
        result.failure_reason = "validated transition waypoints require a start fingerprint";
        return finish();
      }
      const auto start_arm = armJoints(*initial_state_);
      double maximum_start_error = std::abs(
        initial_state_->getVariablePosition("updown") - transition_waypoint_start_[0]);
      for (size_t index = 0U; index < start_arm.size(); ++index) {
        maximum_start_error = std::max(
          maximum_start_error,
          std::abs(normalizedAngle(
            start_arm[index] - degToRad(transition_waypoint_start_[index + 1U]))));
      }
      if (maximum_start_error > degToRad(0.01)) {
        result.failure_stage = "transition_waypoint_start";
        result.failure_reason = "validated transition waypoint start mismatch " +
          std::to_string(radToDeg(maximum_start_error)) + "deg";
        return finish();
      }
      std::vector<moveit::core::RobotStatePtr> validated{
        std::make_shared<moveit::core::RobotState>(*initial_state_)};
      const size_t waypoint_count = transition_joint_waypoints_.size() / 8U;
      for (size_t waypoint = 0U; waypoint <= waypoint_count; ++waypoint) {
        auto next = std::make_shared<moveit::core::RobotState>(*validated.back());
        if (waypoint == waypoint_count) {
          *next = goal;
        } else {
          next->setVariablePosition(
            "updown", transition_joint_waypoints_[waypoint * 8U]);
          std::array<double, 7> joints{};
          for (size_t joint = 0U; joint < joints.size(); ++joint) {
            joints[joint] = degToRad(
              transition_joint_waypoints_[waypoint * 8U + joint + 1U]);
          }
          next->setJointGroupPositions(planning_group_, joints.data());
          next->update(true);
        }
        std::vector<moveit::core::RobotStatePtr> edge;
        if (!sample_edge(*validated.back(), *next, &edge)) {
          result.failure_stage = "transition_waypoint_edge";
          result.failure_reason = "validated transition waypoint edge " +
            std::to_string(waypoint + 1U) + " failed";
          return finish();
        }
        validated.insert(validated.end(), edge.begin(), edge.end());
      }
      result.transition_strategy = "validated_transition_joint_waypoints";
      result.transition_diagnostic = "count=" + std::to_string(waypoint_count);
      appendStates(validated, "between_boxes", false, false, &result.frames);
      result.success = true;
      return finish();
    }
    const auto tcp_path_cost = [this](
      const std::vector<moveit::core::RobotStatePtr>& path) {
        double position_travel = 0.0;
        double orientation_travel = 0.0;
        for (size_t index = 1U; index < path.size(); ++index) {
          const auto& from = path[index - 1U]->getGlobalLinkTransform(tool_link_);
          const auto& to = path[index]->getGlobalLinkTransform(tool_link_);
          position_travel += (to.translation() - from.translation()).norm();
          orientation_travel += Eigen::Quaterniond(from.linear()).angularDistance(
            Eigen::Quaterniond(to.linear()));
        }
        return position_travel + 0.15 * orientation_travel;
      };

    std::vector<moveit::core::RobotStatePtr> joint_fallback_states{
      std::make_shared<moveit::core::RobotState>(*initial_state_)};
    bool joint_transition_available = sample_edge(
      *initial_state_, goal, &joint_fallback_states);
    if (!joint_transition_available) {
      for (const bool move_updown_first : {false, true}) {
        moveit::core::RobotState waypoint(
          move_updown_first ? *initial_state_ : goal);
        waypoint.setVariablePosition(
          "updown", move_updown_first ? goal.getVariablePosition("updown") :
          initial_state_->getVariablePosition("updown"));
        waypoint.update(true);
        std::vector<moveit::core::RobotStatePtr> first_edge;
        std::vector<moveit::core::RobotStatePtr> second_edge;
        if (sample_edge(*initial_state_, waypoint, &first_edge) &&
            sample_edge(waypoint, goal, &second_edge)) {
          joint_fallback_states.clear();
          joint_fallback_states.push_back(
            std::make_shared<moveit::core::RobotState>(*initial_state_));
          joint_fallback_states.insert(
            joint_fallback_states.end(), first_edge.begin(), first_edge.end());
          joint_fallback_states.insert(
            joint_fallback_states.end(), second_edge.begin(), second_edge.end());
          joint_transition_available = true;
          break;
        }
      }
    }
    std::vector<moveit::core::RobotStatePtr> states;
    bool joint_transition_connected = false;
    bool cartesian_transition_connected = false;
    std::string best_transition_strategy;
    std::string cartesian_transition_diagnostic = "no Cartesian transition candidate attempted";
    if (!joint_transition_available &&
        (continuous_plan_approach_ || shortcut_repair_rrt_enabled_) &&
        cartesian_transfer_search_enabled_ && opposite_unchanged) {
      double best_cost = std::numeric_limits<double>::infinity();
      for (const bool move_updown_first : {false, true}) {
        std::vector<moveit::core::RobotStatePtr> first_updown;
        std::vector<moveit::core::RobotStatePtr> second_updown;
        moveit::core::RobotState arm_start(*initial_state_);
        moveit::core::RobotState arm_goal(goal);
        std::string updown_reason;

        if (move_updown_first) {
          if (!traceUpdownTransition(
                scene, *initial_state_, goal.getVariablePosition("updown"), false,
                &result.metrics, &first_updown, &updown_reason)) {
            cartesian_transition_diagnostic = "updown first: " + updown_reason;
            continue;
          }
          arm_start = *first_updown.back();
        } else {
          arm_goal.setVariablePosition("updown", initial_state_->getVariablePosition("updown"));
          arm_goal.update(true);
        }

        const uint64_t guided_before = result.metrics.cartesian_guided_segments;
        const auto arm_plan = planRrt(
          scene, arm_start, arm_goal, false, &result.metrics, true);
        result.segment_search.push_back({
          {"stage", "transition_arm"},
          {"success", arm_plan.success},
          {"strategy", arm_plan.strategy},
          {"total_ms", arm_plan.wall_ms},
          {"tcp_shortcut_ms", arm_plan.tcp_shortcut_ms},
          {"shortcut_repair_rrt_ms", arm_plan.reduced_rrt_ms},
          {"shortcut_repair_samples", arm_plan.reduced_rrt_samples},
          {"repaired_joints", [&]() {
            nlohmann::json names = nlohmann::json::array();
            for (const size_t index : arm_plan.repaired_joint_indices) {
              names.push_back(side_ + "_joint" + std::to_string(index + 1U));
            }
            return names;
          }()},
          {"joint_rrt_ms", arm_plan.joint_rrt_ms},
          {"diagnostic", arm_plan.reason},
        });
        cartesian_transition_diagnostic = arm_plan.reason;
        if (!arm_plan.success ||
            (!shortcut_repair_rrt_enabled_ &&
             result.metrics.cartesian_guided_segments == guided_before)) {
          cartesian_transition_diagnostic = "arm Cartesian search: " + arm_plan.reason;
          if (arm_plan.success) {
            cartesian_transition_diagnostic += " (no Cartesian-guided segment)";
          }
          continue;
        }
        std::vector<moveit::core::RobotStatePtr> validated_arm_path;
        if (!validate_full_path(arm_plan.states, &validated_arm_path)) {
          cartesian_transition_diagnostic = "arm Cartesian path failed full-scene validation";
          continue;
        }
        if (!move_updown_first && !traceUpdownTransition(
              scene, *validated_arm_path.back(), goal.getVariablePosition("updown"), false,
              &result.metrics, &second_updown, &updown_reason)) {
          cartesian_transition_diagnostic = "updown after arm: " + updown_reason;
          continue;
        }

        std::vector<moveit::core::RobotStatePtr> candidate;
        if (move_updown_first) {
          candidate = first_updown;
          candidate.insert(
            candidate.end(), validated_arm_path.begin() + 1U, validated_arm_path.end());
        } else {
          candidate = validated_arm_path;
          candidate.insert(
            candidate.end(), second_updown.begin() + 1U, second_updown.end());
        }
        const double cost = tcp_path_cost(candidate);
        if (cost < best_cost) {
          best_cost = cost;
          states = std::move(candidate);
          cartesian_transition_connected = true;
          best_transition_strategy = arm_plan.strategy;
        }
      }
    }

    if (cartesian_transition_connected) {
      result.transition_strategy = best_transition_strategy.empty() ?
        "cartesian_tcp_hybrid" : best_transition_strategy;
      result.transition_diagnostic = cartesian_transition_diagnostic;
    } else if (joint_transition_available) {
      states = std::move(joint_fallback_states);
      joint_transition_connected = true;
      result.transition_strategy = "direct_joint_interpolation";
      result.transition_diagnostic = cartesian_transition_diagnostic;
    } else {
      result.transition_strategy = "joint_space_fallback";
      result.transition_diagnostic = cartesian_transition_diagnostic;
      states.push_back(std::make_shared<moveit::core::RobotState>(*initial_state_));
    }
    if (!cartesian_transition_connected && !joint_transition_connected) {
      states.clear();
      states.push_back(std::make_shared<moveit::core::RobotState>(*initial_state_));
      planning_interface::MotionPlanRequest request;
      request.group_name = transition_group_name;
      request.planner_id = "RRTConnectkConfigDefault";
      request.allowed_planning_time = rrt_planning_time_;
      request.num_planning_attempts = rrt_planning_attempts_;
      moveit::core::robotStateToRobotStateMsg(*initial_state_, request.start_state, true);
      request.goal_constraints.push_back(
        kinematic_constraints::constructGoalConstraints(goal, transition_group, 1e-3));
      planning_interface::MotionPlanResponse response;
      const bool generated = planning_pipeline_->generatePlan(scene, request, response);
      if (!generated || response.error_code_.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS ||
          !response.trajectory_ || response.trajectory_->getWayPointCount() == 0U) {
        result.failure_stage = "transition_ompl";
        result.failure_reason = transition_group_name + " RRTConnect code=" +
          std::to_string(response.error_code_.val);
        return finish();
      }
      states.clear();
      states.push_back(std::make_shared<moveit::core::RobotState>(*initial_state_));
      for (size_t index = 0; index < response.trajectory_->getWayPointCount(); ++index) {
        std::vector<moveit::core::RobotStatePtr> edge;
        if (!sample_edge(
              *states.back(), response.trajectory_->getWayPoint(index), &edge)) {
          result.failure_stage = "transition_path_collision";
          result.failure_reason = transition_group_name +
            " RRT path failed interpolated collision check";
          return finish();
        }
        states.insert(states.end(), edge.begin(), edge.end());
      }
      std::vector<moveit::core::RobotStatePtr> final_edge;
      if (!sample_edge(*states.back(), goal, &final_edge)) {
        result.failure_stage = "transition_path_collision";
        result.failure_reason = transition_group_name +
          " RRT path cannot reach exact goal";
        return finish();
      }
      states.insert(states.end(), final_edge.begin(), final_edge.end());
      result.transition_strategy = "tcp_cost_joint_rrt_fallback";
    }
    if (!cartesian_transition_connected && !joint_transition_connected &&
        natural_rrt_shortcut_enabled_ && states.size() > 2U) {
      // Bound the shortcut graph while retaining the validated path as fallback.
      std::vector<moveit::core::RobotStatePtr> nodes;
      const size_t maximum_nodes = static_cast<size_t>(
        natural_rrt_shortcut_max_nodes_ > 1 ? natural_rrt_shortcut_max_nodes_ : 20);
      const size_t stride = std::max<size_t>(
        1U, (states.size() - 1U + maximum_nodes - 2U) / (maximum_nodes - 1U));
      for (size_t index = 0U; index < states.size(); index += stride) {
        nodes.push_back(states[index]);
      }
      if (nodes.back() != states.back()) nodes.push_back(states.back());
      std::vector<double> costs(nodes.size(), std::numeric_limits<double>::infinity());
      std::vector<size_t> previous(nodes.size(), 0U);
      costs.front() = 0.0;
      for (size_t to = 1U; to < nodes.size(); ++to) {
        for (size_t from = 0U; from < to; ++from) {
          if (!std::isfinite(costs[from])) continue;
          std::vector<moveit::core::RobotStatePtr> edge;
          if (sample_edge(*nodes[from], *nodes[to], &edge)) {
            std::vector<moveit::core::RobotStatePtr> tcp_edge{nodes[from]};
            tcp_edge.insert(tcp_edge.end(), edge.begin(), edge.end());
            const double cost = costs[from] + tcp_path_cost(tcp_edge);
            if (cost >= costs[to]) continue;
            costs[to] = cost;
            previous[to] = from;
          }
        }
      }
      if (std::isfinite(costs.back())) {
        std::vector<size_t> route;
        for (size_t index = nodes.size() - 1U;; index = previous[index]) {
          route.push_back(index);
          if (index == 0U) break;
        }
        std::reverse(route.begin(), route.end());
        std::vector<moveit::core::RobotStatePtr> shortened{nodes.front()};
        for (size_t index = 1U; index < route.size(); ++index) {
          std::vector<moveit::core::RobotStatePtr> edge;
          if (!sample_edge(*nodes[route[index - 1U]], *nodes[route[index]], &edge)) {
            result.failure_stage = "transition_path_collision";
            result.failure_reason = "shortcut failed full-scene revalidation";
            return finish();
          }
          shortened.insert(shortened.end(), edge.begin(), edge.end());
        }
        states = std::move(shortened);
      }
    }
    if (!cartesian_transition_connected && !joint_transition_connected &&
        result.transition_strategy.empty()) {
      result.transition_strategy = "tcp_cost_joint_rrt_fallback";
    }
    appendStates(states, "between_boxes", false, false, &result.frames);
    result.success = true;
    return finish();
  }

  TaskResult planContactReachability(const Eigen::Vector3d& box_center)
  {
    TaskResult result;
    const auto total_started = std::chrono::steady_clock::now();
    auto finish = [&]() {
      result.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_started).count();
      return result;
    };
    const auto scene = makeScene(box_center);
    const std::string initial_collision = collisionReason(scene, *initial_state_, &result.metrics);
    if (!initial_collision.empty()) {
      result.failure_stage = "initial_state";
      result.failure_reason = initial_collision;
      result.frames.push_back(ReplayFrame{"initial_state", allJoints(*initial_state_), false});
      return finish();
    }

    std::string rejections;
    const auto candidates = solvePoseCandidates(
      contactPose(box_center), *initial_state_, false, scene, &result.metrics, false, &rejections);
    if (candidates.empty()) {
      result.failure_stage = "contact_ik";
      result.failure_reason = rejections;
      result.frames.push_back(ReplayFrame{"initial_state", allJoints(*initial_state_), false});
      return finish();
    }

    result.success = true;
    result.frames.push_back(ReplayFrame{
      "contact_reachability", allJoints(*candidates.front().state), false});
    return finish();
  }

#ifdef ALFA_HAS_VAMP_PAPER_KERNEL
  std::vector<alfa_robot::motion::PaperVampCuboid> paperNativeCuboids(
    const Eigen::Vector3d& box_center,
    const moveit::core::RobotState& state) const
  {
    std::vector<alfa_robot::motion::PaperVampCuboid> output;
    const Eigen::Isometry3d world_to_root =
      state.getGlobalLinkTransform("base_footprint").inverse();
    const Eigen::Vector3d euler = world_to_root.linear().eulerAngles(0, 1, 2);
    const auto append = [&](const Eigen::Vector3d& center, const Eigen::Vector3d& size) {
        const Eigen::Vector3d local_center = world_to_root * center;
        alfa_robot::motion::PaperVampCuboid cuboid;
        for (size_t axis = 0U; axis < 3U; ++axis) {
          cuboid.center[axis] = static_cast<float>(local_center[axis]);
          cuboid.euler_xyz[axis] = static_cast<float>(euler[axis]);
          cuboid.size[axis] = static_cast<float>(size[axis]);
        }
        output.push_back(cuboid);
      };

    if (ground_enabled_) {
      append(
        Eigen::Vector3d(
          0.0, 0.0,
          ground_surface_z_ - ground_clearance_ - ground_thickness_ * 0.5),
        Eigen::Vector3d(ground_size_x_, ground_size_y_, ground_thickness_));
    }
    const Eigen::Isometry3d root_to_world =
      state.getGlobalLinkTransform("base_footprint");
    append(
      root_to_world * Eigen::Vector3d(0.0, 0.0, 0.15),
      Eigen::Vector3d(
        std::max(0.001, 1.016 - 2.0 * collision_inset_),
        std::max(0.001, 1.022 - 2.0 * collision_inset_),
        std::max(0.001, 0.300 - 2.0 * collision_inset_)));
    for (const auto& panel : warehousePanels()) {
      append(panel.center, panel.size);
    }
    const double depth = std::max(0.001, box_depth_ - 2.0 * collision_inset_);
    const double width = std::max(0.001, box_width_ - 2.0 * collision_inset_);
    const double height = std::max(0.001, box_height_ - 2.0 * collision_inset_);
    for (const auto& center : obstacleBoxCenters(box_center)) {
      append(center, Eigen::Vector3d(depth, width, height));
    }
    return output;
  }

  RrtPlanResult planNativePaperRrt(
    const planning_scene::PlanningSceneConstPtr& base_scene,
    const moveit::core::RobotState& start_state,
    const moveit::core::RobotState& goal_state,
    const Eigen::Vector3d& box_center,
    bool attached = false,
    std::optional<size_t> paper_skip_override = std::nullopt,
    bool sphere_world_only = false) const
  {
    RrtPlanResult result;
    const auto wall_started = std::chrono::steady_clock::now();
    const auto* group = robot_model_->getJointModelGroup("dual_arm_with_updown");
    if (!group || group->getVariableCount() != alfa_robot::motion::kPaperVampDimension) {
      result.reason = "paper VAMP requires the 15D dual_arm_with_updown group";
      return result;
    }
    const auto& variables = group->getVariableNames();
    for (size_t index = 0U; index < variables.size(); ++index) {
      if (variables[index] != vamp::robots::AlfaV322::joint_names[index]) {
        result.reason = "paper VAMP joint order mismatch at index " + std::to_string(index);
        return result;
      }
    }

    alfa_robot::motion::PaperVampState start{};
    alfa_robot::motion::PaperVampState goal{};
    for (size_t index = 0U; index < variables.size(); ++index) {
      start[index] = static_cast<float>(start_state.getVariablePosition(variables[index]));
      goal[index] = static_cast<float>(goal_state.getVariablePosition(variables[index]));
    }
    const auto cuboids = paperNativeCuboids(box_center, start_state);
    const size_t effective_skip = paper_skip_override.value_or(paper_halton_skip_);
    alfa_robot::motion::PaperVampState local_lower{};
    alfa_robot::motion::PaperVampState local_upper{};
    if (sphere_world_only) {
      for (size_t index = 0U; index < variables.size(); ++index) {
        const auto& limit = robot_model_->getVariableBounds(variables[index]);
        const double start_value = start_state.getVariablePosition(variables[index]);
        const double goal_value = goal_state.getVariablePosition(variables[index]);
        const bool changes = std::abs(goal_value - start_value) > 1e-6;
        const double margin = variables[index] == "updown" ?
          (changes ? 0.15 : 0.03) : degToRad(changes ? 35.0 : 5.0);
        local_lower[index] = static_cast<float>(std::max(
          limit.min_position_, std::min(start_value, goal_value) - margin));
        local_upper[index] = static_cast<float>(std::min(
          limit.max_position_, std::max(start_value, goal_value) + margin));
      }
    }
    std::optional<alfa_robot::motion::PaperVampAttachment> attachment;
    if (attached) {
      attachment.emplace();
      const Eigen::Vector3d center = toolToBoxCenter();
      const Eigen::Quaterniond orientation(toolToBoxRotation());
      const Eigen::Vector3d size = carriedBoxSizeInTool();
      attachment->center = {
        static_cast<float>(center.x()), static_cast<float>(center.y()),
        static_cast<float>(center.z())};
      attachment->quaternion_xyzw = {
        static_cast<float>(orientation.x()), static_cast<float>(orientation.y()),
        static_cast<float>(orientation.z()), static_cast<float>(orientation.w())};
      attachment->size = {
        static_cast<float>(std::max(0.001, size.x() - 2.0 * collision_inset_)),
        static_cast<float>(std::max(0.001, size.y() - 2.0 * collision_inset_)),
        static_cast<float>(std::max(0.001, size.z() - 2.0 * collision_inset_))};
    }
    const auto native = side_ == "right" ?
      alfa_robot::motion::solvePaperVampRrtcRight(
        start, goal, cuboids, static_cast<float>(vamp_range_),
        static_cast<size_t>(vamp_max_iterations_),
        static_cast<size_t>(vamp_max_samples_), effective_skip,
        attachment ? &*attachment : nullptr,
        sphere_world_only ? &local_lower : nullptr,
        sphere_world_only ? &local_upper : nullptr) :
      alfa_robot::motion::solvePaperVampRrtc(
        start, goal, cuboids, static_cast<float>(vamp_range_),
        static_cast<size_t>(vamp_max_iterations_),
        static_cast<size_t>(vamp_max_samples_), effective_skip,
        attachment ? &*attachment : nullptr,
        sphere_world_only ? &local_lower : nullptr,
        sphere_world_only ? &local_upper : nullptr);
    result.planner_ms = static_cast<double>(native.search_ns) / 1.0e6;
    result.joint_rrt_ms = result.planner_ms;
    result.backend_audit = {
      {"backend", "native_vamp_rrt_connect_15d"},
      {"collision_backend", "cricket_simd_spheres"},
      {"sphere_world_only", sphere_world_only},
      {"attachment_sphere_count", attached ? 48 : 0},
      {"attachment_tool_link", attached ? tool_link_ : ""},
      {"base_collision_model", "lower_chassis_cuboid"},
      {"endpoint_local_bounds", sphere_world_only},
      {"sphere_count", vamp::robots::AlfaV322::n_spheres},
      {"environment_cuboids", cuboids.size()},
      {"halton_skip", effective_skip},
      {"range", vamp_range_},
      {"sampler", sphere_world_only ? "shared_goal_corridor_mt19937" : "halton"},
      {"corridor_probability", sphere_world_only ? 0.9 : 0.0},
      {"iterations", native.iterations},
      {"tree_nodes", native.tree_nodes},
      {"sparse_states", native.path.size()},
      {"path_cost", native.cost},
      {"planner_ms", result.planner_ms},
      {"start_environment_collisions", native.start_environment_collisions},
      {"start_self_collisions", native.start_self_collisions},
      {"goal_environment_collisions", native.goal_environment_collisions},
      {"goal_self_collisions", native.goal_self_collisions},
      {"goal_self_collision_details", native.goal_self_collision_details},
      {"start_sphere_world_valid", native.start_valid},
      {"goal_sphere_world_valid", native.goal_valid},
      {"start_robot_attachment_valid", native.start_robot_attachment_valid},
      {"goal_robot_attachment_valid", native.goal_robot_attachment_valid},
      {"start_robot_self_valid", native.start_robot_self_valid},
      {"goal_robot_self_valid", native.goal_robot_self_valid},
      {"start_robot_environment_collisions", native.start_robot_environment_collisions},
      {"goal_robot_environment_collisions", native.goal_robot_environment_collisions},
      {"start_attachment_environment_collisions",
        native.start_attachment_environment_collisions},
      {"goal_attachment_environment_collisions",
        native.goal_attachment_environment_collisions},
      {"start_attachment_center_root", native.start_attachment_center},
      {"goal_attachment_center_root", native.goal_attachment_center},
      {"start_attachment_min_root", native.start_attachment_min},
      {"start_attachment_max_root", native.start_attachment_max},
      {"attachment_sphere_radius", native.attachment_sphere_radius},
      {"shortcut_used", false},
      {"fallback_used", false},
      {"endpoint_source", "same_online_pose_ik"},
      {"fixed_base_mesh_checked_by_fcl_before_search", true},
      {"dense_fcl_path_audit_outside_timed_region", true},
      {"attached_payload_in_native_kernel", attached},
      {"attached_payload_in_dense_fcl_audit", attached},
    };
    if (!native.solved) {
      result.reason = "native VAMP exhausted the exact RRT-Connect budget";
      result.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_started).count();
      return result;
    }

    auto scene = planning_scene::PlanningScene::clone(base_scene);
    auto state_valid = [&](const moveit::core::RobotState& state) {
        return state.satisfiesBounds(group) &&
          (!attached || carriedBoxFeasible(state)) &&
          fullCollisionReason(scene, state, nullptr).empty();
      };
    const auto audit_started = std::chrono::steady_clock::now();
    size_t audit_states = 0U;
    std::vector<moveit::core::RobotStatePtr> sparse;
    sparse.reserve(native.path.size());
    for (const auto& values : native.path) {
      auto path_state = std::make_shared<moveit::core::RobotState>(start_state);
      for (size_t index = 0U; index < variables.size(); ++index) {
        path_state->setVariablePosition(variables[index], values[index]);
      }
      path_state->zeroVelocities();
      path_state->zeroAccelerations();
      if (attached) attachCarriedBox(*path_state);
      path_state->update(true);
      sparse.push_back(std::move(path_state));
    }
    if (sparse.empty()) {
      result.reason = "native VAMP returned an empty path";
      return result;
    }
    bool fcl_audit_valid = true;
    size_t first_invalid_fcl_state = 0U;
    size_t invalid_fcl_states = 0U;
    result.states.push_back(std::make_shared<moveit::core::RobotState>(*sparse.front()));
    for (size_t edge = 1U; edge < sparse.size(); ++edge) {
      const auto& from = *sparse[edge - 1U];
      const auto& to = *sparse[edge];
      double maximum_ratio = 0.0;
      for (const auto& variable : variables) {
        const double resolution = variable == "updown" ? 0.01 : degToRad(1.0);
        maximum_ratio = std::max(
          maximum_ratio,
          std::abs(to.getVariablePosition(variable) -
          from.getVariablePosition(variable)) / resolution);
      }
      const size_t steps = std::max<size_t>(
        1U, static_cast<size_t>(std::ceil(maximum_ratio)));
      for (size_t step = 1U; step <= steps; ++step) {
        const double ratio = static_cast<double>(step) / static_cast<double>(steps);
        auto probe = std::make_shared<moveit::core::RobotState>(from);
        for (const auto& variable : variables) {
          probe->setVariablePosition(
            variable, from.getVariablePosition(variable) +
            (to.getVariablePosition(variable) - from.getVariablePosition(variable)) * ratio);
        }
        probe->zeroVelocities();
        probe->zeroAccelerations();
        if (attached) attachCarriedBox(*probe);
        probe->update(true);
        ++audit_states;
        if (!sphere_world_only && !state_valid(*probe)) {
          if (fcl_audit_valid) first_invalid_fcl_state = audit_states;
          fcl_audit_valid = false;
          ++invalid_fcl_states;
        }
        result.states.push_back(std::move(probe));
      }
    }
    result.backend_audit["fcl_audit_performed"] = !sphere_world_only;
    result.backend_audit["fcl_audit_valid"] = sphere_world_only ?
      nlohmann::json(nullptr) : nlohmann::json(fcl_audit_valid);
    result.backend_audit["fcl_audit_states"] = audit_states;
    result.backend_audit["first_invalid_fcl_state"] = first_invalid_fcl_state;
    result.backend_audit["invalid_fcl_states"] = invalid_fcl_states;
    result.backend_audit["fcl_audit_ms"] =
      std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - audit_started).count();
    if (!sphere_world_only && !fcl_audit_valid) {
      result.reason = "native VAMP path failed independent dense MoveIt/FCL audit";
      result.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_started).count();
      return result;
    }
    result.success = true;
    result.strategy = sphere_world_only ?
      "native_vamp_rrt_connect_sphere_world" :
      "native_vamp_rrt_connect_cricket_simd";
    result.reason = sphere_world_only ?
      "native VAMP sphere-world solution" :
      "native VAMP solution passed independent dense MoveIt/FCL audit";
    result.wall_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - wall_started).count();
    return result;
  }

  nlohmann::json paperReplayStates(const RrtPlanResult& plan) const
  {
    const auto* group = robot_model_->getJointModelGroup("dual_arm_with_updown");
    nlohmann::json frames = nlohmann::json::array();
    for (const auto& state : plan.states) {
      nlohmann::json values = nlohmann::json::array();
      for (const auto& variable : group->getVariableNames()) {
        values.push_back(state->getVariablePosition(variable));
      }
      frames.push_back(std::move(values));
    }
    return frames;
  }

  PaperPairedSegment planPaperPairedSegment(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit::core::RobotState& start_state,
    const moveit::core::RobotState& goal_state,
    const Eigen::Vector3d& box_center,
    bool attached,
    size_t segment_index,
    bool endpoint_local_bounds = false,
    bool vamp_sphere_world = false) const
  {
    PaperPairedSegment output;
    bool ompl_solved = false;
    bool vamp_solved = false;
    for (size_t attempt = 0U;
      attempt < paper_search_retries_ && (!ompl_solved || !vamp_solved); ++attempt)
    {
      const size_t skip = paper_halton_skip_ + segment_index * 7919U + attempt * 997U;
      const bool vamp_first = (skip / 997U) % 2U == 1U;
      const auto run_ompl = [&]() {
          if (ompl_solved) return;
          auto plan = planPureArmUpdownRrt(
            scene, start_state, goal_state, attached, nullptr,
            vamp_sphere_world ? "ompl_corridor" : "ompl",
            vamp_sphere_world ? std::optional<size_t>(skip) : std::nullopt,
            std::nullopt, endpoint_local_bounds);
          output.ompl_accumulated_planner_ms += plan.planner_ms;
          output.ompl_accumulated_wall_ms += plan.wall_ms;
          output.ompl_attempts.push_back({
            {"attempt", attempt}, {"halton_skip", skip}, {"success", plan.success},
            {"planner_ms", plan.planner_ms}, {"wall_ms", plan.wall_ms},
            {"reason", plan.reason}, {"audit", plan.backend_audit},
          });
          output.ompl = plan;
          ompl_solved = plan.success;
        };
      const auto run_vamp = [&]() {
          if (vamp_solved) return;
          auto plan = vamp_sphere_world ?
            planNativePaperRrt(
              scene, start_state, goal_state, box_center, attached, skip, true) :
            planPureArmUpdownRrt(
              scene, start_state, goal_state, attached, nullptr, "vamp",
              std::nullopt, static_cast<uint32_t>(vamp_seed_) + static_cast<uint32_t>(skip),
              endpoint_local_bounds);
          output.vamp_accumulated_planner_ms += plan.planner_ms;
          output.vamp_accumulated_wall_ms += plan.wall_ms;
          output.vamp_attempts.push_back({
            {"attempt", attempt}, {"halton_skip", skip}, {"success", plan.success},
            {"planner_ms", plan.planner_ms}, {"wall_ms", plan.wall_ms},
            {"reason", plan.reason}, {"audit", plan.backend_audit},
          });
          output.vamp = plan;
          vamp_solved = plan.success;
        };
      if (vamp_first) {
        run_vamp();
        run_ompl();
      } else {
        run_ompl();
        run_vamp();
      }
    }
    output.success = ompl_solved && vamp_solved;
    return output;
  }

  nlohmann::json paperPairedSegmentJson(
    const std::string& stage,
    size_t precontact_index,
    size_t contact_index,
    size_t retreat_index,
    size_t transport_index,
    bool attached,
    const PaperPairedSegment& pair) const
  {
    nlohmann::json output = {
      {"stage", stage},
      {"precontact_candidate", precontact_index},
      {"contact_candidate", contact_index},
      {"retreat_candidate", retreat_index},
      {"transport_candidate", transport_index},
      {"attached", attached},
      {"paired_success", pair.success},
      {"ompl", {
        {"success", pair.ompl.success},
        {"accumulated_planner_ms", pair.ompl_accumulated_planner_ms},
        {"accumulated_wall_ms", pair.ompl_accumulated_wall_ms},
        {"attempts", pair.ompl_attempts},
        {"replay_states", paperReplayStates(pair.ompl)},
      }},
      {"vamp", {
        {"success", pair.vamp.success},
        {"accumulated_planner_ms", pair.vamp_accumulated_planner_ms},
        {"accumulated_wall_ms", pair.vamp_accumulated_wall_ms},
        {"attempts", pair.vamp_attempts},
        {"replay_states", paperReplayStates(pair.vamp)},
      }},
    };
    if (pair.success && pair.vamp_accumulated_planner_ms > 0.0) {
      output["speedup_ompl_over_vamp"] =
        pair.ompl_accumulated_planner_ms / pair.vamp_accumulated_planner_ms;
    }
    return output;
  }
#endif

  TaskResult planPaperSharedIkFullflowBenchmark(
    const Eigen::Vector3d& box_center, bool sphere_world)
  {
    TaskResult result;
    const auto total_started = std::chrono::steady_clock::now();
    const auto finish = [&]() {
        result.total_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - total_started).count();
        return result;
      };
#ifndef ALFA_HAS_VAMP_PAPER_KERNEL
    result.failure_stage = "paper_kernel";
    result.failure_reason = "shared-IK benchmark requires the paper benchmark build";
    return finish();
#else
    if (paper_shared_ik_path_.empty()) {
      result.failure_stage = "shared_ik_input";
      result.failure_reason = "paper_shared_ik_path is required";
      return finish();
    }
    std::ifstream input(paper_shared_ik_path_);
    if (!input) {
      result.failure_stage = "shared_ik_input";
      result.failure_reason = "cannot open shared IK source: " + paper_shared_ik_path_;
      return finish();
    }
    nlohmann::json source;
    input >> source;
    if (!source.value("success", false) || !source.contains("joint_names") ||
        !source.contains("frames") || source["frames"].empty()) {
      result.failure_stage = "shared_ik_input";
      result.failure_reason = "shared IK source is not a successful task payload";
      return finish();
    }
    if (source.value("target_box_id", target_box_id_) != target_box_id_) {
      result.failure_stage = "shared_ik_input";
      result.failure_reason = "shared IK source target box does not match request";
      return finish();
    }
    const auto* group = robot_model_->getJointModelGroup("dual_arm_with_updown");
    const auto& variables = group->getVariableNames();
    const auto source_names = source["joint_names"].get<std::vector<std::string>>();
    const auto state_from_frame = [&](const nlohmann::json& frame, bool attached) {
        const auto values = frame.at("joints").get<std::vector<double>>();
        if (values.size() != source_names.size()) {
          throw std::runtime_error("shared IK frame joint count mismatch");
        }
        auto state = std::make_shared<moveit::core::RobotState>(*initial_state_);
        for (const auto& variable : variables) {
          const auto found = std::find(source_names.begin(), source_names.end(), variable);
          if (found == source_names.end()) {
            throw std::runtime_error("shared IK source lacks variable " + variable);
          }
          const size_t index = static_cast<size_t>(std::distance(source_names.begin(), found));
          state->setVariablePosition(variable, values[index]);
        }
        if (attached) attachCarriedBox(*state);
        else if (state->hasAttachedBody(kCarriedBoxId)) state->clearAttachedBody(kCarriedBoxId);
        state->zeroVelocities();
        state->zeroAccelerations();
        state->update(true);
        return state;
      };
    const auto last_stage_frame = [&](const std::string& stage) -> const nlohmann::json& {
        const nlohmann::json* selected = nullptr;
        for (const auto& frame : source["frames"]) {
          if (frame.value("stage", "") == stage) selected = &frame;
        }
        if (!selected) throw std::runtime_error("shared IK source lacks stage " + stage);
        return *selected;
      };

    auto start = state_from_frame(source["frames"].front(), false);
    auto precontact = state_from_frame(last_stage_frame("rrt_to_precontact"), false);
    auto contact = state_from_frame(last_stage_frame("rrt_to_contact"), false);
    auto loaded_contact = std::make_shared<moveit::core::RobotState>(*contact);
    attachCarriedBox(*loaded_contact);
    loaded_contact->update(true);
    auto retreat = state_from_frame(last_stage_frame("rrt_loaded_retreat"), true);
    auto transport = state_from_frame(last_stage_frame("rrt_to_place"), true);
    auto released_transport = std::make_shared<moveit::core::RobotState>(*transport);
    released_transport->clearAttachedBody(kCarriedBoxId);
    released_transport->update(true);

    result.paper_benchmark = {
      {"schema", sphere_world ?
        "alfa.v322.vamp_sphere_world_fullflow_pair.v1" :
        "alfa.v322.vamp_shared_ik_fullflow_pair.v1"},
      {"metric", "rrt_connect_search_over_shared_validated_stage_endpoints_ms"},
      {"vamp_collision_backend", sphere_world ?
        "cricket_simd_sphere_world" : "moveit_fcl_mesh"},
      {"vamp_online_fcl_used", !sphere_world},
      {"shared_ik_source", paper_shared_ik_path_},
      {"source_frame_count", source["frames"].size()},
      {"source_frames_used_as_paths", false},
      {"source_data_used", "first_state_and_last_state_of_each_named_stage_only"},
      {"joint_names", variables},
      {"ik_inside_timed_region", false},
      {"endpoint_local_bounds", true},
      {"same_sampler_distribution_per_backend", sphere_world},
      {"same_sampler_seed_per_backend", sphere_world},
      {"ompl_sampler", sphere_world ?
        "shared_goal_corridor_mt19937" : "standard_ompl_uniform"},
      {"vamp_sampler", sphere_world ?
        "shared_goal_corridor_mt19937" : "goal_corridor_biased"},
      {"rrt_range", sphere_world ? vamp_range_ : 0.12},
      {"edge_checks_per_configuration_unit", sphere_world ? 64 : 0},
      {"changed_joint_margin_deg", 35.0},
      {"unchanged_joint_margin_deg", 5.0},
      {"changed_updown_margin_m", 0.15},
      {"unchanged_updown_margin_m", 0.03},
      {"shortcut_used", false},
      {"cartesian_used", false},
      {"historical_path_used", false},
      {"fallback_used", false},
      {"selected_segments", nlohmann::json::array()},
      {"all_segment_attempts", nlohmann::json::array()},
    };
    const auto endpoint_values = [&](const moveit::core::RobotState& state) {
        nlohmann::json values = nlohmann::json::array();
        for (const auto& variable : variables) values.push_back(state.getVariablePosition(variable));
        return values;
      };
    result.paper_benchmark["shared_endpoints"] = {
      {"start", endpoint_values(*start)},
      {"precontact", endpoint_values(*precontact)},
      {"contact", endpoint_values(*contact)},
      {"retreat", endpoint_values(*retreat)},
      {"transport", endpoint_values(*transport)},
      {"return_home", endpoint_values(*start)},
    };

    struct StageSpec
    {
      std::string name;
      moveit::core::RobotStatePtr from;
      moveit::core::RobotStatePtr to;
      bool attached{false};
    };
    const std::vector<StageSpec> stages = {
      {"to_precontact", start, precontact, false},
      {"precontact_to_contact", precontact, contact, false},
      {"contact_to_retreat_loaded", loaded_contact, retreat, true},
      {"retreat_to_transport_loaded", retreat, transport, true},
      {"released_transport_to_home", released_transport, start, false},
    };
    double ompl_ms = 0.0;
    double vamp_ms = 0.0;
    for (size_t index = 0U; index < stages.size(); ++index) {
      const auto& stage = stages[index];
      auto pair = planPaperPairedSegment(
        makeScene(box_center), *stage.from, *stage.to,
        box_center, stage.attached, index, true, sphere_world);
      auto stage_json = paperPairedSegmentJson(
        stage.name, 0U, 0U, 0U, 0U, stage.attached, pair);
      result.paper_benchmark["all_segment_attempts"].push_back(stage_json);
      if (!pair.success) {
        result.failure_stage = stage.name;
        result.failure_reason = "OMPL: " + pair.ompl.reason + "; VAMP: " + pair.vamp.reason;
        result.paper_benchmark["accumulated_search"] = {
          {"ompl_planner_ms", ompl_ms + pair.ompl_accumulated_planner_ms},
          {"vamp_planner_ms", vamp_ms + pair.vamp_accumulated_planner_ms},
        };
        return finish();
      }
      result.paper_benchmark["selected_segments"].push_back(stage_json);
      ompl_ms += pair.ompl_accumulated_planner_ms;
      vamp_ms += pair.vamp_accumulated_planner_ms;
      appendStates(pair.ompl.states, stage.name, stage.attached, true, &result.frames);
      result.segment_search.push_back({
        {"stage", stage.name}, {"success", true},
        {"ompl_planner_ms", pair.ompl_accumulated_planner_ms},
        {"vamp_planner_ms", pair.vamp_accumulated_planner_ms},
      });
    }
    result.paper_benchmark["accumulated_search"] = {
      {"ompl_planner_ms", ompl_ms}, {"vamp_planner_ms", vamp_ms},
      {"speedup_ompl_over_vamp", vamp_ms > 0.0 ? ompl_ms / vamp_ms : 0.0},
    };
    result.metrics.joint_rrt_ms = ompl_ms + vamp_ms;
    result.achieved_place_tcp = transport->getGlobalLinkTransform(tool_link_);
    result.success = true;
    return finish();
#endif
  }

  TaskResult planPaperAdaptiveMultistageBenchmark(const Eigen::Vector3d& box_center)
  {
    TaskResult result;
    const auto total_started = std::chrono::steady_clock::now();
    const auto finish = [&]() {
        result.total_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - total_started).count();
        return result;
      };
#ifndef ALFA_HAS_VAMP_PAPER_KERNEL
    result.failure_stage = "paper_kernel";
    result.failure_reason = "Cricket-generated V3.2.2 kernel was not compiled";
    return finish();
#else
    if (!hasPlaceGoal()) {
      result.failure_stage = "transport_goal";
      result.failure_reason = "adaptive benchmark requires an online transport TCP pose";
      return finish();
    }
    const auto scene = makeScene(box_center);
    const std::string initial_collision = collisionReason(scene, *initial_state_, &result.metrics);
    if (!initial_collision.empty()) {
      result.failure_stage = "initial_state";
      result.failure_reason = initial_collision;
      return finish();
    }

    result.paper_benchmark = {
      {"schema", "alfa.v322.vamp_paper_adaptive_multistage.v1"},
      {"metric", "accumulated_rrt_connect_search_with_exact_fcl_to_first_valid_path_ms"},
      {"joint_names", robot_model_->getJointModelGroup(
        "dual_arm_with_updown")->getVariableNames()},
      {"endpoint_source", "online_pose_ik_for_every_stage"},
      {"ik_inside_timed_region", false},
      {"candidate_policy_inside_timed_region", false},
      {"same_endpoint_per_backend", true},
      {"same_sampler_per_backend", false},
      {"ompl_sampler", "standard_ompl_uniform"},
      {"vamp_sampler", "goal_corridor_biased"},
      {"ompl_collision_backend", "moveit_fcl_mesh"},
      {"vamp_collision_backend", "moveit_fcl_mesh"},
      {"native_simd_paper_metric_reported_separately", true},
      {"search_retries", paper_search_retries_},
      {"shortcut_used", false},
      {"cartesian_used", false},
      {"historical_path_used", false},
      {"fallback_used", false},
      {"all_segment_attempts", nlohmann::json::array()},
      {"selected_segments", nlohmann::json::array()},
    };

    const auto pose_candidates = [&](
      const Eigen::Isometry3d& pose,
      const moveit::core::RobotState& seed,
      bool attached,
      bool search_lift,
      bool assign_opposite) {
        std::vector<double> lifts{seed.getVariablePosition("updown")};
        if (search_lift) {
          const auto& bounds = robot_model_->getVariableBounds("updown");
          for (double lift = bounds.max_position_;
            lift >= bounds.min_position_ - 1e-9; lift -= 0.25)
          {
            if (std::none_of(lifts.begin(), lifts.end(), [lift](double existing) {
              return std::abs(existing - lift) < 1e-9;
            })) lifts.push_back(lift);
          }
          std::stable_sort(lifts.begin(), lifts.end(), [&](double left, double right) {
            const double current = seed.getVariablePosition("updown");
            return std::abs(left - current) < std::abs(right - current);
          });
        }
        std::vector<AnalyticCandidate> output;
        for (const double lift : lifts) {
          moveit::core::RobotState lift_seed(seed);
          lift_seed.setVariablePosition("updown", lift);
          if (attached) attachCarriedBox(lift_seed);
          lift_seed.update(true);
          std::string rejection;
          auto candidates = solvePoseCandidates(
            pose, lift_seed, attached, scene, &result.metrics, false, &rejection);
          for (auto& candidate : candidates) {
            if (assign_opposite) {
              std::vector<moveit::core::RobotStatePtr> targets{candidate.state};
              std::string opposite_reason;
              if (!assignCollisionFreeOppositeArm(
                  scene, &targets, &targets, &result.metrics, &opposite_reason)) continue;
              candidate.state = targets.front();
            }
            candidate.score += 0.1 * std::abs(lift - seed.getVariablePosition("updown"));
            output.push_back(std::move(candidate));
            if (output.size() >= precontact_candidate_limit_) break;
          }
          if (output.size() >= precontact_candidate_limit_) break;
        }
        std::stable_sort(output.begin(), output.end(), [](const auto& left, const auto& right) {
          return left.score < right.score;
        });
        return output;
      };

    auto precontact_candidates = pose_candidates(
      precontactPose(box_center), *initial_state_, false, true, true);
    if (precontact_candidates.empty()) {
      result.failure_stage = "precontact_ik";
      result.failure_reason = "no online precontact IK candidate";
      return finish();
    }

    struct SelectedStage
    {
      std::string name;
      bool attached{false};
      PaperPairedSegment pair;
    };
    std::vector<SelectedStage> selected;
    double all_ompl_ms = 0.0;
    double all_vamp_ms = 0.0;
    size_t segment_index = 0U;
    bool found = false;
    moveit::core::RobotStatePtr selected_transport_goal;
    for (size_t pre_index = 0U; pre_index < precontact_candidates.size() && !found; ++pre_index) {
      auto pre_pair = planPaperPairedSegment(
        scene, *initial_state_, *precontact_candidates[pre_index].state,
        box_center, false, segment_index++);
      all_ompl_ms += pre_pair.ompl_accumulated_planner_ms;
      all_vamp_ms += pre_pair.vamp_accumulated_planner_ms;
      result.paper_benchmark["all_segment_attempts"].push_back(paperPairedSegmentJson(
        "to_precontact", pre_index, 0U, 0U, 0U, false, pre_pair));
      if (!pre_pair.success) continue;

      std::string contact_rejection;
      auto contact_candidates = solvePoseCandidates(
        contactPose(box_center), *precontact_candidates[pre_index].state,
        false, scene, &result.metrics, false, &contact_rejection);
      if (contact_candidates.size() > precontact_candidate_limit_)
        contact_candidates.resize(precontact_candidate_limit_);
      for (size_t contact_index = 0U;
        contact_index < contact_candidates.size() && !found; ++contact_index)
      {
        auto contact_pair = planPaperPairedSegment(
          scene, *precontact_candidates[pre_index].state,
          *contact_candidates[contact_index].state, box_center, false, segment_index++);
        all_ompl_ms += contact_pair.ompl_accumulated_planner_ms;
        all_vamp_ms += contact_pair.vamp_accumulated_planner_ms;
        result.paper_benchmark["all_segment_attempts"].push_back(paperPairedSegmentJson(
          "precontact_to_contact", pre_index, contact_index, 0U, 0U, false, contact_pair));
        if (!contact_pair.success) continue;

        moveit::core::RobotState loaded_contact(*contact_candidates[contact_index].state);
        attachCarriedBox(loaded_contact);
        loaded_contact.update(true);
        std::string retreat_rejection;
        auto retreat_candidates = solvePoseCandidates(
          retreatPose(box_center), loaded_contact, true, scene,
          &result.metrics, false, &retreat_rejection);
        if (retreat_candidates.size() > precontact_candidate_limit_)
          retreat_candidates.resize(precontact_candidate_limit_);
        for (size_t retreat_index = 0U;
          retreat_index < retreat_candidates.size() && !found; ++retreat_index)
        {
          auto retreat_pair = planPaperPairedSegment(
            scene, loaded_contact, *retreat_candidates[retreat_index].state,
            box_center, true, segment_index++);
          all_ompl_ms += retreat_pair.ompl_accumulated_planner_ms;
          all_vamp_ms += retreat_pair.vamp_accumulated_planner_ms;
          result.paper_benchmark["all_segment_attempts"].push_back(paperPairedSegmentJson(
            "contact_to_retreat_loaded", pre_index, contact_index,
            retreat_index, 0U, true, retreat_pair));
          if (!retreat_pair.success) continue;

          auto transport_candidates = pose_candidates(
            placeTcpPose(), *retreat_candidates[retreat_index].state, true, true, false);
          if (transport_candidates.size() > precontact_candidate_limit_)
            transport_candidates.resize(precontact_candidate_limit_);
          for (size_t transport_index = 0U;
            transport_index < transport_candidates.size() && !found; ++transport_index)
          {
            auto transport_pair = planPaperPairedSegment(
              scene, *retreat_candidates[retreat_index].state,
              *transport_candidates[transport_index].state,
              box_center, true, segment_index++);
            all_ompl_ms += transport_pair.ompl_accumulated_planner_ms;
            all_vamp_ms += transport_pair.vamp_accumulated_planner_ms;
            result.paper_benchmark["all_segment_attempts"].push_back(paperPairedSegmentJson(
              "retreat_to_transport_loaded", pre_index, contact_index,
              retreat_index, transport_index, true, transport_pair));
            if (!transport_pair.success) continue;

            moveit::core::RobotState released_transport(
              *transport_candidates[transport_index].state);
            released_transport.clearAttachedBody(kCarriedBoxId);
            released_transport.update(true);
            auto return_pair = planPaperPairedSegment(
              scene, released_transport, *initial_state_,
              box_center, false, segment_index++);
            all_ompl_ms += return_pair.ompl_accumulated_planner_ms;
            all_vamp_ms += return_pair.vamp_accumulated_planner_ms;
            result.paper_benchmark["all_segment_attempts"].push_back(paperPairedSegmentJson(
              "released_transport_to_home", pre_index, contact_index,
              retreat_index, transport_index, false, return_pair));
            if (!return_pair.success) continue;
            selected = {
              {"to_precontact", false, pre_pair},
              {"precontact_to_contact", false, contact_pair},
              {"contact_to_retreat_loaded", true, retreat_pair},
              {"retreat_to_transport_loaded", true, transport_pair},
              {"released_transport_to_home", false, return_pair},
            };
            selected_transport_goal = transport_candidates[transport_index].state;
            result.paper_benchmark["selected_candidate"] = {
              {"precontact", pre_index}, {"contact", contact_index},
              {"retreat", retreat_index}, {"transport", transport_index},
            };
            found = true;
          }
        }
      }
    }

    result.paper_benchmark["accumulated_candidate_search"] = {
      {"ompl_planner_ms", all_ompl_ms}, {"vamp_planner_ms", all_vamp_ms},
    };
    if (all_vamp_ms > 0.0) {
      result.paper_benchmark["accumulated_candidate_search"]["speedup_ompl_over_vamp"] =
        all_ompl_ms / all_vamp_ms;
    }
    if (!found) {
      result.failure_stage = "paired_multistage_search";
      result.failure_reason = "no candidate chain succeeded in both matched backends";
      return finish();
    }

    double selected_ompl_ms = 0.0;
    double selected_vamp_ms = 0.0;
    for (const auto& stage : selected) {
      result.paper_benchmark["selected_segments"].push_back(paperPairedSegmentJson(
        stage.name, 0U, 0U, 0U, 0U, stage.attached, stage.pair));
      selected_ompl_ms += stage.pair.ompl_accumulated_planner_ms;
      selected_vamp_ms += stage.pair.vamp_accumulated_planner_ms;
      appendStates(stage.pair.ompl.states, stage.name, stage.attached, true, &result.frames);
      result.segment_search.push_back({
        {"stage", stage.name}, {"success", true},
        {"ompl_planner_ms", stage.pair.ompl_accumulated_planner_ms},
        {"vamp_planner_ms", stage.pair.vamp_accumulated_planner_ms},
      });
    }
    result.paper_benchmark["selected_chain"] = {
      {"ompl_planner_ms", selected_ompl_ms},
      {"vamp_planner_ms", selected_vamp_ms},
      {"speedup_ompl_over_vamp", selected_vamp_ms > 0.0 ?
        selected_ompl_ms / selected_vamp_ms : 0.0},
    };
    result.metrics.joint_rrt_ms = all_ompl_ms + all_vamp_ms;
    result.achieved_place_tcp = selected_transport_goal->getGlobalLinkTransform(tool_link_);
    result.success = true;
    return finish();
#endif
  }

  TaskResult planPaperBenchmark(const Eigen::Vector3d& box_center)
  {
    TaskResult result;
    const auto total_started = std::chrono::steady_clock::now();
    const auto finish = [&]() {
        result.total_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - total_started).count();
        return result;
      };
#ifndef ALFA_HAS_VAMP_PAPER_KERNEL
    result.failure_stage = "paper_kernel";
    result.failure_reason = "Cricket-generated V3.2.2 kernel was not compiled";
    return finish();
#else
    const auto scene = makeScene(box_center);
    const std::string initial_collision = collisionReason(scene, *initial_state_, &result.metrics);
    if (!initial_collision.empty()) {
      result.failure_stage = "initial_state";
      result.failure_reason = initial_collision;
      return finish();
    }
    std::string rejections;
    auto candidates = solvePoseCandidates(
      precontactPose(box_center), *initial_state_, false, scene,
      &result.metrics, false, &rejections);
    if (candidates.empty()) {
      result.failure_stage = "precontact_ik";
      result.failure_reason = rejections;
      return finish();
    }
    const auto& goal = *candidates.front().state;

    RrtPlanResult ompl_result;
    RrtPlanResult vamp_result;
    const bool vamp_first = (paper_halton_skip_ / 997U) % 2U == 1U;
    if (vamp_first) {
      vamp_result = planNativePaperRrt(scene, *initial_state_, goal, box_center);
      ompl_result = planPureArmUpdownRrt(
        scene, *initial_state_, goal, false, nullptr, "ompl_paper");
    } else {
      ompl_result = planPureArmUpdownRrt(
        scene, *initial_state_, goal, false, nullptr, "ompl_paper");
      vamp_result = planNativePaperRrt(scene, *initial_state_, goal, box_center);
    }

    const auto* group = robot_model_->getJointModelGroup("dual_arm_with_updown");
    const auto& variables = group->getVariableNames();
    nlohmann::json start_values = nlohmann::json::array();
    nlohmann::json goal_values = nlohmann::json::array();
    for (const auto& variable : variables) {
      start_values.push_back(initial_state_->getVariablePosition(variable));
      goal_values.push_back(goal.getVariablePosition(variable));
    }
    const auto replay_states = [&](const RrtPlanResult& plan) {
        nlohmann::json frames = nlohmann::json::array();
        for (const auto& state : plan.states) {
          nlohmann::json values = nlohmann::json::array();
          for (const auto& variable : variables) {
            values.push_back(state->getVariablePosition(variable));
          }
          frames.push_back(std::move(values));
        }
        return frames;
      };
    const auto ompl_replay_states = replay_states(ompl_result);
    const auto vamp_replay_states = replay_states(vamp_result);
    result.paper_benchmark = {
      {"schema", "alfa.v322.vamp_paper_rrtc_pair.v1"},
      {"metric", "rrt_connect_planning_time_to_first_collision_valid_path_ms"},
      {"dimension", variables.size()},
      {"joint_names", variables},
      {"start", start_values},
      {"goal", goal_values},
      {"endpoint_source", "same_online_pose_ik"},
      {"ik_inside_timed_region", false},
      {"sampler", "same_vamp_halton_sequence"},
      {"halton_skip", paper_halton_skip_},
      {"execution_order", vamp_first ?
        nlohmann::json::array({"vamp", "ompl"}) :
        nlohmann::json::array({"ompl", "vamp"})},
      {"range", vamp_range_},
      {"max_iterations", vamp_max_iterations_},
      {"max_samples", vamp_max_samples_},
      {"shortcut_used", false},
      {"cartesian_used", false},
      {"fallback_used", false},
      {"ompl", {
        {"success", ompl_result.success},
        {"planner_ms", ompl_result.planner_ms},
        {"total_with_fcl_audit_ms", ompl_result.wall_ms},
        {"reason", ompl_result.reason},
        {"audit", ompl_result.backend_audit},
        {"replay_states", ompl_replay_states},
        {"replay_state_count", ompl_replay_states.size()},
      }},
      {"vamp", {
        {"success", vamp_result.success},
        {"planner_ms", vamp_result.planner_ms},
        {"total_with_fcl_audit_ms", vamp_result.wall_ms},
        {"reason", vamp_result.reason},
        {"audit", vamp_result.backend_audit},
        {"replay_states", vamp_replay_states},
        {"replay_state_count", vamp_replay_states.size()},
      }},
    };
    if (ompl_result.success && vamp_result.success && vamp_result.planner_ms > 0.0) {
      result.paper_benchmark["speedup_ompl_over_vamp"] =
        ompl_result.planner_ms / vamp_result.planner_ms;
    }
    result.segment_search.push_back({
      {"stage", "paper_same_endpoint_ompl"},
      {"success", ompl_result.success},
      {"joint_rrt_ms", ompl_result.planner_ms},
      {"diagnostic", ompl_result.reason},
    });
    result.segment_search.push_back({
      {"stage", "paper_same_endpoint_vamp"},
      {"success", vamp_result.success},
      {"joint_rrt_ms", vamp_result.planner_ms},
      {"diagnostic", vamp_result.reason},
      {"backend_audit", vamp_result.backend_audit},
    });
    result.success = ompl_result.success && vamp_result.success;
    if (!result.success) {
      result.failure_stage = "paper_paired_search";
      result.failure_reason = "OMPL: " + ompl_result.reason + "; VAMP: " + vamp_result.reason;
    }
    result.metrics.joint_rrt_ms = ompl_result.planner_ms + vamp_result.planner_ms;
    return finish();
#endif
  }

  nlohmann::json sceneJson(const Eigen::Vector3d& box_center) const
  {
    const auto neighbors = obstacleBoxCenters(box_center);
    nlohmann::json output;
    output["box_center"] = {box_center.x(), box_center.y(), box_center.z()};
    output["world_frame"] = world_frame_;
    output["base_pose_ready"] = base_pose_ready_;
    output["base_pose_map"] = {
      base_pose_map_[0], base_pose_map_[1], base_pose_map_[2]};
    output["box_size"] = {box_depth_, box_width_, box_height_};
    output["collision_inset"] = collision_inset_;
    output["full_box_wall_scene"] = full_box_wall_scene_;
    output["target_box_id"] = target_box_id_;
    output["removed_box_ids"] = removed_box_ids_;
    output["grasp_mode"] = grasp_mode_;
    output["place_tcp_pose"] = place_tcp_pose_;
    output["place_arm_joints_deg"] = place_arm_joints_deg_;
    output["precontact_arm_joints_deg"] = precontact_arm_joints_deg_;
    output["loaded_transfer_joint_waypoints_deg"] = loaded_transfer_joint_waypoints_deg_;
    output["loaded_transfer_waypoint_start_deg"] = loaded_transfer_waypoint_start_deg_;
    output["transition_joint_waypoints"] = transition_joint_waypoints_;
    output["transition_waypoint_start"] = transition_waypoint_start_;
    output["transition_whole_body_waypoints"] = transition_whole_body_waypoints_;
    output["transition_whole_body_waypoint_start"] =
      transition_whole_body_waypoint_start_;
    output["return_to_ready"] = !continuous_sequence_;
    output["maximum_carried_box_tilt_deg"] = radToDeg(maximum_carried_box_tilt_);
    output["maximum_carried_box_center_z"] = maximum_carried_box_center_z_;
    output["contact_tool_roll_deg"] = radToDeg(
      side_ == "right" ? -contact_tool_roll_ : contact_tool_roll_);
    output["contact_tool_roll_magnitude_deg"] = radToDeg(contact_tool_roll_);
    output["initial_arm_pose"] = initial_arm_pose_;
    output["front_suction_y_offset"] = front_suction_y_offset_;
    output["front_suction_z_offset"] = front_suction_z_offset_;
    output["top_suction_x_offset"] = top_suction_x_offset_;
    output["contact_normal"] = {
      contactNormal().x(), contactNormal().y(), contactNormal().z()};
    output["ground"] = {
      {"enabled", ground_enabled_},
      {"surface_z", ground_surface_z_},
      {"collision_top_z", ground_surface_z_ - ground_clearance_},
      {"size", {ground_size_x_, ground_size_y_, ground_thickness_}},
    };
    output["warehouse"] = {
      {"enabled", warehouse_enabled_},
      {"opening_x", warehouse_opening_x_},
      {"center_y", warehouse_center_y_},
      {"floor_z", warehouse_floor_z_},
      {"length", warehouse_length_},
      {"width", warehouse_width_},
      {"height", warehouse_height_},
      {"wall_thickness", warehouse_wall_thickness_},
      {"panels", nlohmann::json::array()},
    };
    output["natural_motion"] = {
      {"enabled", natural_motion_enabled_},
      {"swivel_weight", natural_swivel_weight_},
      {"wrist_singularity_weight", natural_wrist_singularity_weight_},
      {"wrist_neutral_weight", natural_wrist_neutral_weight_},
      {"joint_limit_weight", natural_joint_limit_weight_},
      {"joint_wrap_weight", natural_joint_wrap_weight_},
      {"seed_swivel_sampling", natural_seed_swivel_sampling_},
      {"seed_swivel_step_deg", radToDeg(natural_seed_swivel_step_)},
      {"seed_swivel_neighbor_steps", natural_seed_swivel_neighbor_steps_},
      {"joint_acceleration_weight", natural_joint_acceleration_weight_},
      {"place_return_weight", natural_place_return_weight_},
      {"cartesian_replay_step_deg", radToDeg(natural_cartesian_replay_step_)},
      {"rrt_shortcut_enabled", natural_rrt_shortcut_enabled_},
      {"rrt_shortcut_tcp_cost", natural_rrt_shortcut_tcp_cost_},
      {"rrt_shortcut_max_nodes", natural_rrt_shortcut_max_nodes_},
      {"cartesian_transfer_search_enabled", cartesian_transfer_search_enabled_},
      {"cartesian_transfer_max_search_attempts", cartesian_transfer_max_search_attempts_},
      {"shortcut_repair_rrt_enabled", shortcut_repair_rrt_enabled_},
      {"shortcut_repair_rrt_budget_ms", shortcut_repair_rrt_budget_ms_},
      {"shortcut_repair_rrt_max_samples", shortcut_repair_rrt_max_samples_},
      {"parallel_rrt_workers", parallel_rrt_workers_},
      {"shortcut_repair_max_joint_offset_deg", radToDeg(shortcut_repair_max_joint_offset_)},
      {"place_updown_enabled", place_updown_enabled_},
      {"place_updown_m", place_updown_},
      {"top_loaded_transfer_direct_only", top_loaded_transfer_direct_only_},
      {"loaded_transfer_defer_place_updown", loaded_transfer_defer_place_updown_},
      {"max_proximal_step_deg", radToDeg(natural_max_proximal_step_)},
      {"max_wrist_step_deg", radToDeg(natural_max_wrist_step_)},
      {"rrt_replay_step_deg", radToDeg(natural_rrt_replay_step_)},
    };
    for (const auto& panel : warehousePanels()) {
      output["warehouse"]["panels"].push_back({
        {"id", panel.id},
        {"center", {panel.center.x(), panel.center.y(), panel.center.z()}},
        {"size", {panel.size.x(), panel.size.y(), panel.size.z()}},
      });
    }
    output["neighbor_centers"] = nlohmann::json::array();
    for (const auto& center : neighbors) {
      output["neighbor_centers"].push_back({center.x(), center.y(), center.z()});
    }
    const auto precontact = precontactPose(box_center);
    const auto contact = contactPose(box_center);
    const auto retreat = retreatPose(box_center);
    output["precontact"] = {
      precontact.translation().x(), precontact.translation().y(), precontact.translation().z()};
    output["contact"] = {
      contact.translation().x(), contact.translation().y(), contact.translation().z()};
    output["retreat"] = {
      retreat.translation().x(), retreat.translation().y(), retreat.translation().z()};
    const auto tool_offset = toolToBoxCenter();
    output["tool_to_box_center"] = {tool_offset.x(), tool_offset.y(), tool_offset.z()};
    const auto tool_to_box_rotation = toolToBoxRotation();
    output["tool_to_box_rotation"] = {
      {tool_to_box_rotation(0, 0), tool_to_box_rotation(0, 1), tool_to_box_rotation(0, 2)},
      {tool_to_box_rotation(1, 0), tool_to_box_rotation(1, 1), tool_to_box_rotation(1, 2)},
      {tool_to_box_rotation(2, 0), tool_to_box_rotation(2, 1), tool_to_box_rotation(2, 2)},
    };
    const auto carried_size = carriedBoxSizeInTool();
    output["carried_box_size_tool"] = {carried_size.x(), carried_size.y(), carried_size.z()};
    return output;
  }

  void publishJson(const nlohmann::json& payload)
  {
    std_msgs::msg::String message;
    message.data = payload.dump();
    task_publisher_->publish(message);
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
    payload["side"] = side_;
    payload["status"] = status;
    publishJson(payload);
  }

  void publishPlanningStarted(uint64_t generation, const Eigen::Vector3d& box_center)
  {
    nlohmann::json payload = sceneJson(box_center);
    payload["kind"] = "planning";
    payload["generation"] = generation;
    payload["side"] = side_;
    payload["status"] = "计算开始";
    publishJson(payload);
  }

  void publishTaskResult(
    uint64_t generation,
    const Eigen::Vector3d& box_center,
    const TaskResult& result)
  {
    nlohmann::json payload = sceneJson(box_center);
    payload["kind"] = "result";
    payload["generation"] = generation;
    payload["schema"] = "alfa.v3_single_arm_box_test.v1";
    payload["task_mode"] = task_mode_;
    payload["grasp_mode"] = grasp_mode_;
    payload["side"] = side_;
    payload["tool_link"] = tool_link_;
    payload["initial_updown"] = initial_state_->getVariablePosition("updown");
    payload["ignore_opposite_arm"] = ignore_opposite_arm_;
    payload["pure_rrt_full_flow"] = pure_rrt_full_flow_;
    payload["ompl_simplify_solutions"] = ompl_simplify_solutions_;
    payload["success"] = result.success;
    payload["failure_stage"] = result.failure_stage;
    payload["failure_reason"] = result.failure_reason;
    if (!result.transition_group.empty()) {
      payload["transition_group"] = result.transition_group;
    }
    if (!result.transition_strategy.empty()) {
      payload["transition_strategy"] = result.transition_strategy;
    }
    if (!result.transition_diagnostic.empty()) {
      payload["transition_diagnostic"] = result.transition_diagnostic;
    }
    if (result.achieved_place_tcp) {
      const auto& actual = *result.achieved_place_tcp;
      const Eigen::Quaterniond orientation(actual.linear());
      payload["achieved_place_tcp_pose"] = {
        actual.translation().x(), actual.translation().y(), actual.translation().z(),
        orientation.x(), orientation.y(), orientation.z(), orientation.w()};
    }
    payload["total_ms"] = result.total_ms;
    payload["segment_search"] = result.segment_search;
    if (!result.paper_benchmark.empty()) {
      payload["paper_benchmark"] = result.paper_benchmark;
    }
    payload["metrics"] = {
      {"ik_calls", result.metrics.ik_calls},
      {"ik_ms", result.metrics.ik_ms},
      {"collision_checks", result.metrics.collision_checks},
      {"collision_ms", result.metrics.collision_ms},
      {"analytic_path_ms", result.metrics.analytic_path_ms},
      {"rrt_approach_ms", result.metrics.rrt_approach_ms},
      {"rrt_return_ms", result.metrics.rrt_return_ms},
      {"rrt_shortcut_edges_checked", result.metrics.rrt_shortcut_edges_checked},
      {"rrt_shortcut_ms", result.metrics.rrt_shortcut_ms},
      {"cartesian_transfer_attempts", result.metrics.cartesian_transfer_attempts},
      {"cartesian_guided_segments", result.metrics.cartesian_guided_segments},
      {"joint_fallback_segments", result.metrics.joint_fallback_segments},
      {"cartesian_transfer_budget_exhaustions",
        result.metrics.cartesian_transfer_budget_exhaustions},
      {"tcp_shortcut_attempts", result.metrics.tcp_shortcut_attempts},
      {"tcp_shortcut_successes", result.metrics.tcp_shortcut_successes},
      {"tcp_shortcut_ms", result.metrics.tcp_shortcut_ms},
      {"shortcut_repair_rrt_attempts", result.metrics.reduced_rrt_attempts},
      {"shortcut_repair_rrt_successes", result.metrics.reduced_rrt_successes},
      {"shortcut_repair_rrt_samples", result.metrics.reduced_rrt_samples},
      {"shortcut_repair_rrt_ms", result.metrics.reduced_rrt_ms},
      {"parallel_rrt_workers_launched", result.metrics.parallel_rrt_workers_launched},
      {"parallel_rrt_workers_completed", result.metrics.parallel_rrt_workers_completed},
      {"parallel_rrt_wins", result.metrics.parallel_rrt_wins},
      {"joint_rrt_fallbacks", result.metrics.joint_rrt_fallbacks},
      {"joint_rrt_ms", result.metrics.joint_rrt_ms},
    };
    for (const auto& frame : result.frames) {
      if (frame.stage != "attach_box") continue;
      moveit::core::RobotState state(*initial_state_);
      for (size_t index = 0; index < all_joint_names_.size(); ++index) {
        state.setVariablePosition(all_joint_names_[index], frame.joints[index]);
      }
      state.update(true);
      const auto& tool = state.getGlobalLinkTransform(tool_link_);
      const double face_normal_error = std::acos(std::clamp(
        (tool.linear() * Eigen::Vector3d::UnitZ()).dot(contactNormal()), -1.0, 1.0));
      const Eigen::Vector3d scoop_long_axis = tool.linear() * Eigen::Vector3d::UnitX();
      const double horizontal_error = std::asin(std::clamp(
        std::abs(scoop_long_axis.z()), 0.0, 1.0));
      payload["contact_tool_metrics"] = {
        {"face_normal_error_deg", radToDeg(face_normal_error)},
        {"scoop_horizontal_error_deg", radToDeg(horizontal_error)},
        {"joint7_deg", radToDeg(state.getVariablePosition(side_ + "_joint7"))},
      };
      break;
    }
    payload["joint_names"] = all_joint_names_;
    payload["frames"] = nlohmann::json::array();
    for (const auto& frame : result.frames) {
      payload["frames"].push_back({
        {"stage", frame.stage},
        {"joints", frame.joints},
        {"box_attached", frame.box_attached},
      });
      if (frame.box_attached) {
        moveit::core::RobotState state(*initial_state_);
        for (size_t index = 0; index < all_joint_names_.size(); ++index) {
          state.setVariablePosition(all_joint_names_[index], frame.joints[index]);
        }
        state.update(true);
        payload["frames"].back()["carried_box_tilt_deg"] = radToDeg(carriedBoxTilt(state));
      }
    }
    if (task_mode_ == "state_transition" && result.frames.size() >= 2U) {
      std::vector<Eigen::Isometry3d> tool_poses;
      tool_poses.reserve(result.frames.size());
      for (const auto& frame : result.frames) {
        moveit::core::RobotState state(*initial_state_);
        for (size_t index = 0; index < all_joint_names_.size(); ++index) {
          state.setVariablePosition(all_joint_names_[index], frame.joints[index]);
        }
        state.update(true);
        tool_poses.push_back(state.getGlobalLinkTransform(tool_link_));
      }
      double position_travel = 0.0;
      double orientation_travel = 0.0;
      for (size_t index = 1U; index < tool_poses.size(); ++index) {
        position_travel += (
          tool_poses[index].translation() - tool_poses[index - 1U].translation()).norm();
        orientation_travel += Eigen::Quaterniond(tool_poses[index - 1U].linear()).angularDistance(
          Eigen::Quaterniond(tool_poses[index].linear()));
      }
      const Eigen::Vector3d start = tool_poses.front().translation();
      const Eigen::Vector3d axis = tool_poses.back().translation() - start;
      const double direct_distance = axis.norm();
      const double axis_squared = axis.squaredNorm();
      double maximum_line_deviation = 0.0;
      for (const auto& pose : tool_poses) {
        const double ratio = axis_squared > 1e-12 ? std::clamp(
          (pose.translation() - start).dot(axis) / axis_squared, 0.0, 1.0) : 0.0;
        maximum_line_deviation = std::max(
          maximum_line_deviation,
          (pose.translation() - (start + axis * ratio)).norm());
      }
      payload["transition_tcp_motion"] = {
        {"path_length_m", position_travel},
        {"direct_distance_m", direct_distance},
        {"excess_distance_m", position_travel - direct_distance},
        {"max_line_deviation_m", maximum_line_deviation},
        {"orientation_travel_deg", radToDeg(orientation_travel)},
      };
    }
    writeResultFile(payload);
    publishJson(payload);
  }

  void writeResultFile(const nlohmann::json& payload) const
  {
    if (result_json_path_.empty()) {
      return;
    }
    const std::filesystem::path output_path(result_json_path_);
    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    std::filesystem::path temporary_path = output_path;
    temporary_path += ".tmp";
    {
      std::ofstream stream(temporary_path);
      if (!stream) {
        throw std::runtime_error("failed to open result JSON: " + temporary_path.string());
      }
      stream << std::setw(2) << payload << '\n';
    }
    std::error_code error;
    std::filesystem::rename(temporary_path, output_path, error);
    if (error) {
      std::filesystem::remove(output_path, error);
      error.clear();
      std::filesystem::rename(temporary_path, output_path, error);
    }
    if (error) {
      throw std::runtime_error("failed to publish result JSON: " + error.message());
    }
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
    visualization_msgs::msg::MarkerArray markers;
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
    markers.markers.push_back(target);

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
    markers.markers.push_back(control_link);

    const auto neighbors = obstacleBoxCenters(center);
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

    if (ground_enabled_) {
      Marker ground;
      ground.header.frame_id = world_frame_;
      ground.header.stamp = now();
      ground.ns = "ground";
      ground.id = 0;
      ground.type = Marker::CUBE;
      ground.action = Marker::ADD;
      ground.pose.position.z = ground_surface_z_ - ground_thickness_ * 0.5;
      ground.pose.orientation.w = 1.0;
      ground.scale.x = ground_size_x_;
      ground.scale.y = ground_size_y_;
      ground.scale.z = ground_thickness_;
      ground.color = color(0.32F, 0.35F, 0.38F, 0.80F);
      markers.markers.push_back(ground);
    }

    const auto warehouse_panels = warehousePanels();
    for (size_t index = 0; index < warehouse_panels.size(); ++index) {
      const auto& panel_spec = warehouse_panels[index];
      Marker panel;
      panel.header.frame_id = world_frame_;
      panel.header.stamp = now();
      panel.ns = "warehouse";
      panel.id = static_cast<int>(index);
      panel.type = Marker::CUBE;
      panel.action = Marker::ADD;
      panel.pose.position.x = panel_spec.center.x();
      panel.pose.position.y = panel_spec.center.y();
      panel.pose.position.z = panel_spec.center.z();
      panel.pose.orientation.w = 1.0;
      panel.scale.x = panel_spec.size.x();
      panel.scale.y = panel_spec.size.y();
      panel.scale.z = panel_spec.size.z();
      panel.color = color(0.20F, 0.55F, 0.78F, 0.16F);
      markers.markers.push_back(panel);
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
      point.pose.position.x = points[index].first.x();
      point.pose.position.y = points[index].first.y();
      point.pose.position.z = points[index].first.z();
      point.pose.orientation.w = 1.0;
      point.scale.x = point.scale.y = point.scale.z = 0.035;
      point.color = index == 0 ?
        color(0.15F, 0.55F, 1.0F, 0.9F) :
        (index == 1 ? color(0.2F, 1.0F, 0.2F, 0.9F) : color(0.95F, 0.2F, 0.95F, 0.9F));
      markers.markers.push_back(point);
    }
    scene_marker_publisher_->publish(markers);
  }

  void publishDisplayState()
  {
    std::lock_guard<std::mutex> lock(display_mutex_);
    if (!playback_frames_.empty()) {
      const auto& frame = playback_frames_[playback_index_];
      for (size_t index = 0; index < all_joint_names_.size() && index < frame.joints.size(); ++index) {
        display_state_->setVariablePosition(all_joint_names_[index], frame.joints[index]);
      }
      display_state_->update(true);
      playback_index_ = (playback_index_ + 1U) % playback_frames_.size();
    }
    sensor_msgs::msg::JointState message;
    message.header.stamp = now();
    message.name = all_joint_names_;
    message.position = allJoints(*display_state_);
    joint_state_publisher_->publish(message);
  }

  std::string side_;
  std::string world_frame_;
  std::string arm_base_link_;
  std::string planning_group_name_;
  std::string tool_link_;
  Eigen::Vector3d box_center_{0.88, -0.20, 0.55};
  double box_depth_ = 0.30;
  double box_width_ = 0.40;
  double box_height_ = 0.40;
  double control_handle_clearance_ = 0.25;
  double control_handle_lateral_offset_ = 0.90;
  double approach_distance_ = 0.05;
  double retreat_distance_ = 0.35;
  double cartesian_step_ = 0.01;
  double collision_inset_ = 0.002;
  bool full_box_wall_scene_ = false;
  int box_grid_columns_ = 5;
  int box_grid_rows_ = 5;
  double box_grid_center_y_ = 0.0;
  double box_grid_bottom_z_ = 0.0;
  int target_box_id_ = 0;
  std::set<int> removed_box_ids_;
  std::string grasp_mode_ = "front";
  std::string initial_arm_pose_ = "zero";
  double front_suction_y_offset_ = 0.0;
  double front_suction_z_offset_ = 0.0;
  double top_suction_x_offset_ = 0.0;
  double contact_tool_roll_ = 0.0;
  bool ground_enabled_ = true;
  double ground_surface_z_ = 0.0;
  double ground_clearance_ = 0.005;
  double ground_size_x_ = 6.0;
  double ground_size_y_ = 6.0;
  double ground_thickness_ = 0.10;
  bool warehouse_enabled_ = false;
  double warehouse_opening_x_ = -1.18;
  double warehouse_center_y_ = 0.0;
  double warehouse_floor_z_ = 0.0;
  double warehouse_length_ = 2.38;
  double warehouse_width_ = 2.38;
  double warehouse_height_ = 2.35;
  double warehouse_wall_thickness_ = 0.05;
  double psi_step_ = degToRad(5.0);
  double maximum_cartesian_joint_step_ = degToRad(15.0);
  double edge_joint_resolution_ = degToRad(2.5);
  size_t precontact_candidate_limit_ = 8;
  double rrt_planning_time_ = 1.0;
  int rrt_planning_attempts_ = 1;
  bool pure_rrt_full_flow_ = false;
  std::string pure_rrt_backend_ = "ompl";
  int vamp_workers_ = 4;
  int vamp_max_iterations_ = 100000;
  int vamp_max_samples_ = 200000;
  double vamp_range_ = 0.08;
  int64_t vamp_seed_ = 20261019;
  size_t paper_halton_skip_ = 0U;
  size_t paper_search_retries_ = 3U;
  std::string paper_shared_ik_path_;
  bool ompl_simplify_solutions_ = true;
  bool natural_motion_enabled_ = true;
  double natural_swivel_weight_ = 0.02;
  double natural_wrist_singularity_weight_ = 0.03;
  double natural_wrist_neutral_weight_ = 0.02;
  double natural_joint_limit_weight_ = 0.05;
  double natural_joint_wrap_weight_ = 0.0;
  bool natural_seed_swivel_sampling_ = false;
  double natural_seed_swivel_step_ = degToRad(1.0);
  int natural_seed_swivel_neighbor_steps_ = 2;
  double natural_joint_acceleration_weight_ = 0.0;
  double natural_place_return_weight_ = 0.0;
  double natural_cartesian_replay_step_ = 0.0;
  bool natural_rrt_shortcut_enabled_ = false;
  bool natural_rrt_shortcut_tcp_cost_ = false;
  int natural_rrt_shortcut_max_nodes_ = 0;
  bool cartesian_transfer_search_enabled_ = false;
  double cartesian_transfer_translation_step_ = 0.02;
  double cartesian_transfer_rotation_step_ = degToRad(2.0);
  int cartesian_transfer_max_search_attempts_ = 0;
  bool shortcut_repair_rrt_enabled_ = false;
  double shortcut_repair_rrt_budget_ms_ = 1800.0;
  int shortcut_repair_rrt_max_samples_ = 240;
  int parallel_rrt_workers_ = 1;
  double shortcut_repair_max_joint_offset_ = degToRad(35.0);
  bool place_updown_enabled_ = false;
  double place_updown_ = 0.0;
  bool top_loaded_transfer_direct_only_ = false;
  bool loaded_transfer_defer_place_updown_ = false;
  bool auto_safe_opposite_arm_ = false;
  double natural_max_proximal_step_ = degToRad(12.0);
  double natural_max_wrist_step_ = degToRad(8.0);
  double natural_rrt_replay_step_ = degToRad(3.0);
  double maximum_carried_box_center_z_ = 1000000.0;
  bool auto_run_once_ = false;
  bool analytic_path_only_ = false;
  std::string task_mode_ = "full_extract";
  bool ignore_opposite_arm_ = false;
  bool continuous_sequence_ = false;
  bool continuous_plan_approach_ = false;
  double maximum_carried_box_tilt_ = kPi;
  std::string result_json_path_;
  std::vector<double> place_tcp_pose_;
  std::vector<double> place_arm_joints_deg_;
  std::vector<double> precontact_arm_joints_deg_;
  std::vector<double> loaded_transfer_joint_waypoints_deg_;
  std::vector<double> loaded_transfer_waypoint_start_deg_;
  std::vector<double> loaded_transfer_arm_updown_waypoints_;
  std::vector<double> loaded_transfer_arm_updown_waypoint_start_;
  std::vector<double> transition_joint_waypoints_;
  std::vector<double> transition_waypoint_start_;
  std::vector<double> transition_whole_body_waypoints_;
  std::vector<double> transition_whole_body_waypoint_start_;
  std::vector<double> transition_from_joints_;
  std::vector<double> transition_to_joints_;

  std::shared_ptr<robot_model_loader::RobotModelLoader> robot_model_loader_;
  moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::JointModel* planar_root_joint_ = nullptr;
  const moveit::core::JointModelGroup* planning_group_ = nullptr;
  planning_pipeline::PlanningPipelinePtr planning_pipeline_;
  moveit::core::RobotStatePtr initial_state_;
  moveit::core::RobotStatePtr display_state_;
  std::unique_ptr<V3RedundantArmAnalyticIk> solver_;
  std::vector<std::string> all_joint_names_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::array<double, 3> base_pose_map_{0.0, 0.0, 0.0};
  bool base_pose_ready_ = false;

  std::unique_ptr<interactive_markers::InteractiveMarkerServer> marker_server_;
  interactive_markers::MenuHandler menu_handler_;
  interactive_markers::MenuHandler::EntryHandle confirm_menu_entry_ = 0;
  interactive_markers::MenuHandler::EntryHandle reset_menu_entry_ = 0;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr task_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr scene_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr status_marker_publisher_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr run_service_;
  rclcpp::TimerBase::SharedPtr worker_timer_;
  rclcpp::TimerBase::SharedPtr display_timer_;
  rclcpp::TimerBase::SharedPtr auto_run_timer_;

  std::mutex box_mutex_;
  std::mutex display_mutex_;
  std::vector<ReplayFrame> playback_frames_;
  size_t playback_index_ = 0;
  std::atomic<bool> planning_requested_{false};
  std::atomic<bool> planning_active_{false};
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
