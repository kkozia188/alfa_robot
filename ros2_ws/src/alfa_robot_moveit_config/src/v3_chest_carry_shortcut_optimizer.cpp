#include <alfa_robot_moveit_config/planning_diagnostics.hpp>

#include <geometric_shapes/shapes.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <nlohmann/json.hpp>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/util/RandomNumbers.h>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#ifdef ALFA_HAS_VAMP
#include <vamp/collision/factory.hh>
#include <vamp/collision/validity.hh>
#include <vamp/planning/rrtc.hh>
#include <vamp/random/halton.hh>
#include <vamp/robots/sphere.hh>
#endif

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ob = ompl::base;
namespace og = ompl::geometric;

constexpr double kPi = 3.14159265358979323846;
constexpr char kLeftBoxId[] = "dual_carried_box_left";
constexpr char kRightBoxId[] = "dual_carried_box_right";
constexpr size_t kFullStateDimension = 17U;
#ifdef ALFA_HAS_VAMP
constexpr bool kVampCompiled = true;
#else
constexpr bool kVampCompiled = false;
#endif

struct AttachmentSpec {
  std::string id;
  std::string side;
  std::string tool_link;
  Eigen::Vector3d center_in_tool = Eigen::Vector3d::Zero();
  Eigen::Matrix3d rotation_in_tool = Eigen::Matrix3d::Identity();
  Eigen::Vector3d size = Eigen::Vector3d::Zero();
};

struct LocalSearchStats {
  std::string backend;
  size_t state_dimension = 0U;
  size_t grid_state_checks = 0U;
  size_t occupied_cells = 0U;
  double grid_build_ms = 0.0;
  double search_wall_ms = 0.0;
  double search_core_ms = 0.0;
  double search_fcl_ms = 0.0;
  double candidate_fcl_ms = 0.0;
  size_t validity_batches = 0U;
  size_t validity_states = 0U;
  size_t search_iterations = 0U;
  size_t search_tree_nodes = 0U;
  size_t candidate_sparse_states = 0U;
  double candidate_base_path_m = 0.0;
  double candidate_base_yaw_path_deg = 0.0;
  double candidate_max_base_offset_m = 0.0;
  double candidate_max_base_yaw_offset_deg = 0.0;
  double candidate_updown_path_m = 0.0;
  double candidate_arm_path_deg = 0.0;
  size_t workers = 1U;
  size_t solved_workers = 0U;
  int winner_seed = -1;
  size_t refinements = 0U;
  size_t rejected_candidates = 0U;
  std::string last_rejection_reason;
  double last_rejection_progress = 0.0;
  double last_rejection_alpha = 0.0;
};

#ifdef ALFA_HAS_VAMP
struct FullStateVampContext {
  std::function<bool(const std::array<float, kFullStateDimension> &)> validator;
  size_t validity_batches = 0U;
  size_t validity_states = 0U;
  double fcl_ms = 0.0;
};

thread_local FullStateVampContext *g_full_state_vamp_context = nullptr;

struct FullStateVampRobot {
  static constexpr char name[] = "alfa_v322_contract17_fcl";
  static constexpr size_t dimension = kFullStateDimension;
  static constexpr size_t n_spheres = 0U;
  static constexpr size_t resolution = 8U;
  static constexpr float min_radius = 0.0F;
  static constexpr float max_radius = 0.0F;

  using Configuration = vamp::FloatVector<dimension>;
  using ConfigurationArray = std::array<vamp::FloatT, dimension>;
  template <size_t rake>
  using ConfigurationBlock = vamp::FloatVector<rake, dimension>;
  struct alignas(vamp::FloatVectorAlignment) ConfigurationBuffer
    : std::array<float, Configuration::num_scalars_rounded> {};

  inline static std::array<float, dimension> lows{};
  inline static std::array<float, dimension> highs{};

  static void setBounds(const std::array<float, dimension> &new_lows,
                        const std::array<float, dimension> &new_highs) noexcept {
    lows = new_lows;
    highs = new_highs;
  }

  static void scale_configuration(Configuration &configuration) noexcept {
    const Configuration lower(lows);
    const Configuration upper(highs);
    configuration = configuration * (upper - lower) + lower;
  }

  static void descale_configuration(Configuration &configuration) noexcept {
    const Configuration lower(lows);
    const Configuration upper(highs);
    configuration = (configuration - lower) / (upper - lower);
  }

  template <size_t rake>
  static void scale_configuration_block(
      ConfigurationBlock<rake> &configuration) noexcept {
    for (size_t index = 0U; index < dimension; ++index) {
      configuration[index] = lows[index] +
        configuration[index] * (highs[index] - lows[index]);
    }
  }

  template <size_t rake>
  static void descale_configuration_block(
      ConfigurationBlock<rake> &configuration) noexcept {
    for (size_t index = 0U; index < dimension; ++index) {
      configuration[index] =
        (configuration[index] - lows[index]) / (highs[index] - lows[index]);
    }
  }

  static float space_measure() noexcept {
    float squared = 0.0F;
    for (size_t index = 0U; index < dimension; ++index) {
      const float extent = highs[index] - lows[index];
      squared += extent * extent;
    }
    return std::sqrt(squared);
  }

  template <size_t rake>
  static bool fkcc(
      const vamp::collision::Environment<vamp::FloatVector<rake>> &,
      const ConfigurationBlock<rake> &block) noexcept {
    if (!g_full_state_vamp_context ||
        !g_full_state_vamp_context->validator) {
      return false;
    }
    const auto started = std::chrono::steady_clock::now();
    ++g_full_state_vamp_context->validity_batches;
    std::array<std::array<float, rake>, dimension> rows{};
    for (size_t axis = 0U; axis < dimension; ++axis) {
      rows[axis] = block[axis].to_array();
    }
    try {
      for (size_t lane = 0U; lane < rake; ++lane) {
        std::array<float, dimension> state{};
        for (size_t axis = 0U; axis < dimension; ++axis) {
          state[axis] = rows[axis][lane];
        }
        ++g_full_state_vamp_context->validity_states;
        if (!g_full_state_vamp_context->validator(state)) {
          g_full_state_vamp_context->fcl_ms +=
            std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - started).count();
          return false;
        }
      }
    } catch (...) {
      g_full_state_vamp_context->fcl_ms +=
        std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count();
      return false;
    }
    g_full_state_vamp_context->fcl_ms +=
      std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return true;
  }

  template <size_t rake>
  static bool fkcc_attach(
      const vamp::collision::Environment<vamp::FloatVector<rake>> &environment,
      const ConfigurationBlock<rake> &block) noexcept {
    return fkcc(environment, block);
  }
};

struct FullStateVampSampler : public vamp::rng::RNG<FullStateVampRobot> {
  explicit FullStateVampSampler(uint32_t seed) noexcept
      : initial_seed(seed), engine(seed) {}

  void reset() noexcept override {
    engine.seed(initial_seed);
  }

  FullStateVampRobot::Configuration next() noexcept override {
    std::array<float, kFullStateDimension> values{};
    for (auto &value : values) value = distribution(engine);
    FullStateVampRobot::Configuration configuration(values);
    FullStateVampRobot::scale_configuration(configuration);
    return configuration;
  }

  uint32_t initial_seed;
  std::mt19937 engine;
  std::uniform_real_distribution<float> distribution{0.0F, 1.0F};
};
#endif

Eigen::Vector3d vector3(const nlohmann::json &value, const std::string &label) {
  if (!value.is_array() || value.size() != 3U) {
    throw std::invalid_argument(label + " must contain three numbers");
  }
  return Eigen::Vector3d(value.at(0).get<double>(), value.at(1).get<double>(),
                         value.at(2).get<double>());
}

Eigen::Matrix3d matrix3(const nlohmann::json &value, const std::string &label) {
  if (!value.is_array() || value.size() != 3U) {
    throw std::invalid_argument(label + " must contain three rows");
  }
  Eigen::Matrix3d output;
  for (size_t row = 0U; row < 3U; ++row) {
    if (!value.at(row).is_array() || value.at(row).size() != 3U) {
      throw std::invalid_argument(label + " rows must contain three numbers");
    }
    for (size_t column = 0U; column < 3U; ++column) {
      output(static_cast<Eigen::Index>(row),
             static_cast<Eigen::Index>(column)) =
          value.at(row).at(column).get<double>();
    }
  }
  return output;
}

std::array<double, 3> basePose(const nlohmann::json &operation,
                               const nlohmann::json *frame = nullptr) {
  const nlohmann::json *value = nullptr;
  if (frame && frame->contains("base_pose_map")) {
    value = &frame->at("base_pose_map");
  } else if (operation.contains("base_pose_map")) {
    value = &operation.at("base_pose_map");
  }
  if (!value) {
    return {0.0, 0.0, 0.0};
  }
  if (!value->is_array() || value->size() != 3U) {
    throw std::invalid_argument("base_pose_map must contain three values");
  }
  std::array<double, 3> output{value->at(0).get<double>(),
                               value->at(1).get<double>(),
                               value->at(2).get<double>()};
  if (!std::all_of(output.begin(), output.end(),
                   [](double item) { return std::isfinite(item); })) {
    throw std::invalid_argument(
        "base_pose_map must contain three finite values");
  }
  return output;
}

double shortestAngleDelta(double from, double to) {
  return std::atan2(std::sin(to - from), std::cos(to - from));
}

class ChestCarryShortcutOptimizer : public rclcpp::Node {
public:
  explicit ChestCarryShortcutOptimizer(const rclcpp::NodeOptions &options)
      : Node("v3_chest_carry_shortcut_optimizer", options) {
    ompl::RNG::setSeed(0xC4357U);
    replay_path_ = getParameter<std::string>("replay_json_path", "");
    output_path_ = getParameter<std::string>("output_replay_json_path", "");
    result_path_ = getParameter<std::string>("optimization_report_json_path", "");
    edge_joint_step_ =
        getParameter<double>("edge_joint_step_deg", 1.0) * kPi / 180.0;
    edge_updown_step_ = getParameter<double>("edge_updown_step_m", 0.01);
    collision_inset_ = getParameter<double>("collision_inset", 0.002);
    maximum_tilt_ = getParameter<double>("maximum_carried_box_tilt_deg", 95.0) *
                    kPi / 180.0;
    local_rrt_mode_ = getParameter<std::string>("local_rrt_mode", "full");
    visibility_graph_mode_ = getParameter<std::string>(
      "visibility_graph_mode", "full");
    local_rrt_time_s_ = getParameter<double>("local_rrt_time_s", 0.95);
    manifold_alpha_max_ = getParameter<double>("manifold_alpha_max", 1.25);
    manifold_levels_ = getParameter<int>("manifold_levels", 64);
    manifold_alpha_samples_ = getParameter<int>("manifold_alpha_samples", 11);
    manifold_alpha_neighbor_steps_ = getParameter<int>(
      "manifold_alpha_neighbor_steps", 1);
    vamp_workers_ = getParameter<int>("vamp_workers", 4);
    vamp_max_iterations_ = getParameter<int>("vamp_max_iterations", 1000);
    vamp_max_samples_ = getParameter<int>("vamp_max_samples", 2000);
    vamp_secondary_max_iterations_ = getParameter<int>(
      "vamp_secondary_max_iterations", 300);
    vamp_range_ = getParameter<double>("vamp_range", 0.08);
    vamp_max_refinements_ = getParameter<int>("vamp_max_refinements", 8);
    full_vamp_base_xy_margin_ =
      getParameter<double>("full_vamp_base_xy_margin_m", 0.15);
    full_vamp_base_yaw_margin_ = getParameter<double>(
      "full_vamp_base_yaw_margin_deg", 5.0) * kPi / 180.0;
    full_vamp_arm_margin_ = getParameter<double>(
      "full_vamp_arm_margin_deg", 35.0) * kPi / 180.0;
    if (replay_path_.empty() || output_path_.empty() || result_path_.empty()) {
      throw std::invalid_argument(
          "replay_json_path, output_replay_json_path and "
          "optimization_report_json_path are required");
    }
    if (edge_joint_step_ <= 0.0 || edge_updown_step_ <= 0.0 ||
        maximum_tilt_ <= 0.0 || maximum_tilt_ > kPi ||
        local_rrt_time_s_ <= 0.0 || manifold_alpha_max_ < 1.0 ||
        manifold_levels_ < 2 || manifold_alpha_samples_ < 2 ||
        manifold_alpha_neighbor_steps_ < 1 ||
        vamp_workers_ < 1 || vamp_workers_ > 32 ||
        vamp_max_iterations_ < 1 || vamp_max_samples_ < 4 || vamp_range_ <= 0.0 ||
        vamp_secondary_max_iterations_ < 1 ||
        vamp_secondary_max_iterations_ > vamp_max_iterations_ ||
        vamp_max_refinements_ < 0 || vamp_max_refinements_ > 128 ||
        full_vamp_base_xy_margin_ <= 0.0 ||
        full_vamp_base_yaw_margin_ <= 0.0 || full_vamp_arm_margin_ <= 0.0 ||
        (visibility_graph_mode_ != "full" &&
        visibility_graph_mode_ != "fast_families") ||
        (local_rrt_mode_ != "full" && local_rrt_mode_ != "golden_manifold" &&
        local_rrt_mode_ != "golden_lattice" && local_rrt_mode_ != "golden_vamp" &&
        local_rrt_mode_ != "contract17_vamp")) {
      throw std::invalid_argument(
          "validation steps, tilt, RRT time, manifold bounds, and mode are invalid");
    }
  }

  void init() {
    robot_model_loader_ =
        std::make_shared<robot_model_loader::RobotModelLoader>(
            shared_from_this(), "robot_description");
    robot_model_ = robot_model_loader_->getModel();
    if (!robot_model_) {
      throw std::runtime_error("failed to load robot model");
    }
    whole_body_ = robot_model_->getJointModelGroup("whole_body");
    left_arm_ = robot_model_->getJointModelGroup("left_arm");
    right_arm_ = robot_model_->getJointModelGroup("right_arm");
    if (!whole_body_ || !left_arm_ || !right_arm_) {
      throw std::runtime_error(
          "whole_body and both seven-axis arm groups are required");
    }
    planar_root_ = robot_model_->getJointModel("map_to_base_footprint");
    if (!planar_root_ || planar_root_->getVariableCount() != 3U) {
      throw std::runtime_error("map_to_base_footprint planar root is required");
    }
  }

  bool run() {
    auto output = readJson(replay_path_);
    nlohmann::json result = {
        {"schema", "alfa.v322_chest_golden_shortcut.v1"},
        {"input", replay_path_},
        {"output", output_path_},
        {"success", true},
        {"cache_loading_counted", false},
        {"local_search_mode", local_rrt_mode_},
        {"visibility_graph_mode", visibility_graph_mode_},
        {"vamp_compiled", kVampCompiled},
        {"vamp_workers", vamp_workers_},
        {"vamp_secondary_max_iterations", vamp_secondary_max_iterations_},
        {"cycles", nlohmann::json::array()},
        {"maximum_task_core_ms", 0.0},
        {"boxes_within_3s", 0},
    };
    if (local_rrt_mode_ == "contract17_vamp") {
      result["full_state_contract"] = {
        {"dimension", kFullStateDimension},
        {"axes", {
          "base_x", "base_y", "base_yaw",
          "left_joint1", "left_joint2", "left_joint3", "left_joint4",
          "left_joint5", "left_joint6", "left_joint7",
          "right_joint1", "right_joint2", "right_joint3", "right_joint4",
          "right_joint5", "right_joint6", "right_joint7"}},
        {"fixed_replay_axes", {"head_joint"}},
        {"dependent_replay_axes", {"updown"}},
        {"updown_mapping", "start-goal projection in the 17-D state"},
        {"collision_backend", "MoveIt/FCL callback per VAMP edge sample"},
        {"vamp_tree_backend", "VAMP RRT-Connect"},
      };
    }
    try {
      validateInput(output);
      const auto joint_names = output.at("joint_names").get<std::vector<std::string>>();
      nlohmann::json optimized_operations = nlohmann::json::array();
      std::map<int, double> task_times;
      for (const auto &source_operation : output.at("operations")) {
        if (source_operation.value("kind", "") == "base_transition") {
          optimized_operations.push_back(source_operation);
          continue;
        }
        nlohmann::json operation_result;
        auto optimized = optimizeOperation(
          source_operation, joint_names, &operation_result);
        const double core_ms = operation_result.at("planning_core_ms").get<double>();
        result["maximum_task_core_ms"] = std::max(
          result.at("maximum_task_core_ms").get<double>(), core_ms);
        for (const auto side : {"left", "right"}) {
          const int box_id = source_operation.value(std::string(side) + "_box_id", 0);
          if (box_id > 0) task_times[box_id] = core_ms;
        }
        result["cycles"].push_back(operation_result);
        optimized_operations.push_back(std::move(optimized));
      }
      output["operations"] = std::move(optimized_operations);
      for (auto &item : output.at("box_planning")) {
        const int box_id = item.at("box_id").get<int>();
        item["golden_source_search_ms"] = item.value("selected_task_core_ms", 0.0);
        item["selected_task_core_ms"] = task_times.at(box_id);
        item["source"] = local_rrt_mode_ == "golden_vamp" ?
          "golden_curve_vamp_parallel_2d" :
          (local_rrt_mode_ == "contract17_vamp" ?
          "contract17_vamp_parallel" :
          "golden_chest_curve_local_shortcut");
        if (task_times.at(box_id) < 3000.0) {
          result["boxes_within_3s"] =
            result.at("boxes_within_3s").get<int>() + 1;
        }
      }
      std::map<int, nlohmann::json> planning_by_id;
      for (const auto &item : output.at("box_planning")) {
        planning_by_id[item.at("box_id").get<int>()] = item;
      }
      for (auto &operation : output.at("operations")) {
        if (operation.value("kind", "") == "base_transition") continue;
        nlohmann::json planning = nlohmann::json::array();
        for (const auto &item : operation.at("box_planning")) {
          planning.push_back(planning_by_id.at(item.at("box_id").get<int>()));
        }
        operation["box_planning"] = std::move(planning);
      }
      output["golden_curve_optimization"] = result;
      if (task_times.size() != 25U || result.at("boxes_within_3s").get<int>() != 25) {
        throw std::runtime_error("not all 25 box task cores are below 3 seconds");
      }
      writeJson(output_path_, output);
    } catch (const std::exception &error) {
      result["success"] = false;
      result["failure_reason"] = error.what();
    }

    writeJson(result_path_, result);
    if (result.at("success").get<bool>()) {
      RCLCPP_INFO(get_logger(),
                  "Chest golden shortcut passed: boxes=25/25 max_core=%.3fms",
                  result.at("maximum_task_core_ms").get<double>());
      return true;
    }
    RCLCPP_ERROR(get_logger(), "Chest golden shortcut failed: %s",
                 result.value("failure_reason", "operation failure").c_str());
    return false;
  }

private:
  template <typename T>
  T getParameter(const std::string &name, const T &fallback) {
    if (!has_parameter(name)) {
      declare_parameter<T>(name, fallback);
    }
    return get_parameter(name).get_value<T>();
  }

  static nlohmann::json readJson(const std::string &path) {
    std::ifstream stream(path);
    if (!stream) {
      throw std::runtime_error("failed to open replay JSON: " + path);
    }
    nlohmann::json value;
    stream >> value;
    return value;
  }

  static void writeJson(const std::string &path, const nlohmann::json &value) {
    const std::filesystem::path output(path);
    std::filesystem::create_directories(output.parent_path());
    const auto temporary = output.string() + ".tmp";
    {
      std::ofstream stream(temporary);
      if (!stream) {
        throw std::runtime_error("failed to open validation output: " +
                                 temporary);
      }
      stream << value.dump(2) << '\n';
    }
    std::filesystem::rename(temporary, output);
  }

  static void validateInput(const nlohmann::json &input) {
    if (input.value("schema", "") != "alfa.v3_scoop_5x5_dual_replay.v1") {
      throw std::invalid_argument("unexpected dual replay schema");
    }
    if (!input.contains("joint_names") || !input.at("joint_names").is_array() ||
        !input.contains("operations") || !input.at("operations").is_array()) {
      throw std::invalid_argument(
          "dual replay requires joint_names and operations");
    }
  }

  static AttachmentSpec attachmentSpec(const nlohmann::json &value,
                                       const std::string &side) {
    AttachmentSpec spec;
    spec.id = side == "left" ? kLeftBoxId : kRightBoxId;
    spec.side = side;
    spec.tool_link = value.at("tool_link").get<std::string>();
    spec.center_in_tool =
        vector3(value.at("center_in_tool"), side + " center_in_tool");
    spec.rotation_in_tool =
        matrix3(value.at("rotation_in_tool"), side + " rotation_in_tool");
    spec.size = vector3(value.at("size"), side + " size");
    return spec;
  }

  planning_scene::PlanningScenePtr
  makeScene(const nlohmann::json &operation) const {
    auto scene = std::make_shared<planning_scene::PlanningScene>(robot_model_);
    moveit::core::RobotState default_state(robot_model_);
    default_state.setToDefaultValues();
    setPlanarRoot(default_state, basePose(operation));
    default_state.update(true);
    scene->setCurrentState(default_state);

    for (const auto &obstacle : operation.at("obstacles")) {
      moveit_msgs::msg::CollisionObject object;
      object.header.frame_id = robot_model_->getModelFrame();
      object.id = obstacle.at("id").get<std::string>();
      object.operation = moveit_msgs::msg::CollisionObject::ADD;
      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
      const auto size = vector3(obstacle.at("size"), object.id + " size");
      primitive.dimensions = {size.x(), size.y(), size.z()};
      const auto center = vector3(obstacle.at("center"), object.id + " center");
      geometry_msgs::msg::Pose pose;
      pose.position.x = center.x();
      pose.position.y = center.y();
      pose.position.z = center.z();
      pose.orientation.w = 1.0;
      object.primitives.push_back(primitive);
      object.primitive_poses.push_back(pose);
      if (!scene->processCollisionObjectMsg(object)) {
        throw std::runtime_error("failed to add obstacle " + object.id);
      }
    }
    if (operation.value("allow_ground_model_base", false)) {
      scene->getAllowedCollisionMatrixNonConst().setEntry("ground",
                                                          "model_base", true);
    }
    return scene;
  }

  void attachBox(moveit::core::RobotState &state,
                 const AttachmentSpec &spec) const {
    const double x = std::max(0.001, spec.size.x() - 2.0 * collision_inset_);
    const double y = std::max(0.001, spec.size.y() - 2.0 * collision_inset_);
    const double z = std::max(0.001, spec.size.z() - 2.0 * collision_inset_);
    std::vector<shapes::ShapeConstPtr> shapes{
        std::make_shared<shapes::Box>(x, y, z)};
    EigenSTL::vector_Isometry3d poses;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = spec.center_in_tool;
    pose.linear() = spec.rotation_in_tool;
    poses.push_back(pose);
    state.attachBody(spec.id, Eigen::Isometry3d::Identity(), shapes, poses,
                     std::vector<std::string>{spec.tool_link,
                                              spec.side + "_link7",
                                              spec.side + "_link6"},
                     spec.tool_link);
  }

  moveit::core::RobotStatePtr
  makeState(const std::vector<std::string> &names, const nlohmann::json &frame,
            const nlohmann::json &operation, const AttachmentSpec &left_spec,
            const AttachmentSpec &right_spec) const {
    const auto positions = frame.at("joints").get<std::vector<double>>();
    if (positions.size() != names.size()) {
      throw std::invalid_argument(
          "frame joint count does not match joint_names");
    }
    auto state = std::make_shared<moveit::core::RobotState>(robot_model_);
    state->setToDefaultValues();
    setPlanarRoot(*state, basePose(operation, &frame));
    for (size_t index = 0U; index < names.size(); ++index) {
      if (!std::isfinite(positions[index]) ||
          !robot_model_->hasJointModel(names[index])) {
        throw std::invalid_argument("invalid replay joint " + names[index]);
      }
      state->setVariablePosition(names[index], positions[index]);
    }
    if (frame.value("left_attached", false))
      attachBox(*state, left_spec);
    if (frame.value("right_attached", false))
      attachBox(*state, right_spec);
    state->update(true);
    return state;
  }

  void setPlanarRoot(moveit::core::RobotState &state,
                     const std::array<double, 3> &pose) const {
    state.setJointPositions(planar_root_, pose.data());
  }

  double boxTilt(const moveit::core::RobotState &state,
                 const AttachmentSpec &spec) const {
    const Eigen::Vector3d up_in_tool =
        spec.rotation_in_tool * Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d up_world =
        state.getGlobalLinkTransform(spec.tool_link).linear() * up_in_tool;
    return std::acos(std::clamp(up_world.z(), -1.0, 1.0));
  }

  bool validateState(const planning_scene::PlanningSceneConstPtr &scene,
                     const moveit::core::RobotState &state, bool left_attached,
                     bool right_attached, const AttachmentSpec &left_spec,
                     const AttachmentSpec &right_spec, std::string *reason,
                     nlohmann::json *aggregate) const {
    if (!state.satisfiesBounds()) {
      if (reason)
        *reason = "joint_bounds";
      return false;
    }
    if (left_attached) {
      const double tilt = boxTilt(state, left_spec);
      (*aggregate)["maximum_left_tilt_deg"] =
          std::max(aggregate->at("maximum_left_tilt_deg").get<double>(),
                   tilt * 180.0 / kPi);
      if (tilt > maximum_tilt_ + 1e-9) {
        if (reason)
          *reason = "left_carried_box_tilt";
        return false;
      }
    }
    if (right_attached) {
      const double tilt = boxTilt(state, right_spec);
      (*aggregate)["maximum_right_tilt_deg"] =
          std::max(aggregate->at("maximum_right_tilt_deg").get<double>(),
                   tilt * 180.0 / kPi);
      if (tilt > maximum_tilt_ + 1e-9) {
        if (reason)
          *reason = "right_carried_box_tilt";
        return false;
      }
    }
    const std::string collision =
        alfa_robot::motion::scene_collision_reason(scene, state, nullptr);
    if (!collision.empty()) {
      if (reason)
        *reason = collision;
      return false;
    }
    return true;
  }

  size_t edgeSteps(const moveit::core::RobotState &from,
                   const moveit::core::RobotState &to) const {
    double maximum_ratio = 1.0;
    for (const auto &name : whole_body_->getVariableNames()) {
      const double delta = std::abs(to.getVariablePosition(name) -
                                    from.getVariablePosition(name));
      const double step =
          name == "updown" ? edge_updown_step_ : edge_joint_step_;
      maximum_ratio = std::max(maximum_ratio, delta / step);
    }
    const double *from_base = from.getJointPositions(planar_root_);
    const double *to_base = to.getJointPositions(planar_root_);
    maximum_ratio =
        std::max(maximum_ratio, std::abs(to_base[0] - from_base[0]) /
                                    edge_base_translation_step_);
    maximum_ratio =
        std::max(maximum_ratio, std::abs(to_base[1] - from_base[1]) /
                                    edge_base_translation_step_);
    maximum_ratio = std::max(
        maximum_ratio, std::abs(shortestAngleDelta(from_base[2], to_base[2])) /
                           edge_base_yaw_step_);
    return static_cast<size_t>(std::ceil(maximum_ratio));
  }

  void interpolateState(const moveit::core::RobotState &from,
                        const moveit::core::RobotState &to, double ratio,
                        moveit::core::RobotState *output) const {
    from.interpolate(to, ratio, *output, whole_body_);
    const double *from_base = from.getJointPositions(planar_root_);
    const double *to_base = to.getJointPositions(planar_root_);
    const std::array<double, 3> base{
        from_base[0] + (to_base[0] - from_base[0]) * ratio,
        from_base[1] + (to_base[1] - from_base[1]) * ratio,
        from_base[2] + shortestAngleDelta(from_base[2], to_base[2]) * ratio,
    };
    setPlanarRoot(*output, base);
    output->update(true);
  }

  bool shortcutEdgeValid(
      const planning_scene::PlanningSceneConstPtr &scene,
      const nlohmann::json &operation,
      const std::vector<std::string> &joint_names,
      const nlohmann::json &from_frame,
      const nlohmann::json &to_frame,
      const AttachmentSpec &left_spec,
      const AttachmentSpec &right_spec,
      size_t *checked_samples) const {
    const bool left_attached = from_frame.value("left_attached", false);
    const bool right_attached = from_frame.value("right_attached", false);
    if (left_attached != to_frame.value("left_attached", false) ||
        right_attached != to_frame.value("right_attached", false)) {
      return false;
    }
    auto from = makeState(
      joint_names, from_frame, operation, left_spec, right_spec);
    auto to = makeState(
      joint_names, to_frame, operation, left_spec, right_spec);
    nlohmann::json aggregate = {
      {"maximum_left_tilt_deg", 0.0}, {"maximum_right_tilt_deg", 0.0}};
    std::string reason;
    if (!validateState(scene, *from, left_attached, right_attached,
          left_spec, right_spec, &reason, &aggregate) ||
        !validateState(scene, *to, left_attached, right_attached,
          left_spec, right_spec, &reason, &aggregate)) {
      return false;
    }
    const size_t steps = edgeSteps(*from, *to);
    for (size_t step = 1U; step < steps; ++step) {
      moveit::core::RobotState probe(*from);
      interpolateState(
        *from, *to, static_cast<double>(step) / static_cast<double>(steps),
        &probe);
      ++*checked_samples;
      if (!validateState(scene, probe, left_attached, right_attached,
            left_spec, right_spec, &reason, &aggregate)) {
        return false;
      }
    }
    return true;
  }

  nlohmann::json densifyShortcut(
      const nlohmann::json &golden,
      const std::vector<size_t> &anchors,
      const nlohmann::json &operation,
      const std::vector<std::string> &joint_names,
      const AttachmentSpec &left_spec,
      const AttachmentSpec &right_spec) const {
    nlohmann::json output = nlohmann::json::array();
    output.push_back(golden.at(anchors.front()));
    for (size_t edge = 0U; edge + 1U < anchors.size(); ++edge) {
      const auto &from = golden.at(anchors[edge]);
      const auto &to = golden.at(anchors[edge + 1U]);
      auto from_state = makeState(
        joint_names, from, operation, left_spec, right_spec);
      auto to_state = makeState(
        joint_names, to, operation, left_spec, right_spec);
      const size_t steps = edgeSteps(*from_state, *to_state);
      for (size_t step = 1U; step <= steps; ++step) {
        const double ratio =
          static_cast<double>(step) / static_cast<double>(steps);
        moveit::core::RobotState probe(*from_state);
        interpolateState(*from_state, *to_state, ratio, &probe);
        auto frame = to;
        std::vector<double> joints(joint_names.size(), 0.0);
        for (size_t index = 0U; index < joints.size(); ++index) {
          joints[index] = probe.getVariablePosition(joint_names[index]);
        }
        frame["joints"] = joints;
        if (from.contains("base_pose_map") || to.contains("base_pose_map")) {
          const double *base = probe.getJointPositions(planar_root_);
          frame["base_pose_map"] = {base[0], base[1], base[2]};
        }
        frame["stage"] =
          edge + 2U == anchors.size() && step == steps ?
          "chest_carry_ready" : "chest_carry_retract";
        output.push_back(std::move(frame));
      }
    }
    return output;
  }

  nlohmann::json rebuildOperation(
      const nlohmann::json &source,
      const nlohmann::json &optimized_chest,
      const nlohmann::json &metadata,
      size_t chest_start) const {
    auto output = source;
    const auto &frames = source.at("frames");
    size_t base_start = frames.size();
    size_t release = frames.size();
    size_t stow_last = frames.size();
    for (size_t index = 0U; index < frames.size(); ++index) {
      const std::string stage = frames.at(index).value("stage", "");
      if (stage == "base_backoff_after_chest_carry" && base_start == frames.size())
        base_start = index;
      if (stage == "release_at_right_conveyor") release = index;
      if (stage == "post_release_stow_outside_warehouse") stow_last = index;
    }
    if (base_start == frames.size() || release == frames.size() ||
        stow_last == frames.size()) {
      throw std::runtime_error("operation lacks base/release/stow boundaries");
    }
    nlohmann::json forward = nlohmann::json::array();
    for (size_t index = 0U; index < chest_start; ++index)
      forward.push_back(frames.at(index));
    for (const auto &frame : optimized_chest) forward.push_back(frame);
    nlohmann::json rebuilt = forward;
    for (size_t index = base_start; index <= release; ++index)
      rebuilt.push_back(frames.at(index));
    const auto conveyor_pose = frames.at(release).at("base_pose_map");
    for (size_t index = forward.size() - 1U; index-- > 0U;) {
      auto frame = forward.at(index);
      frame["stage"] = "post_release_shortcut_stow_outside_warehouse";
      frame["left_attached"] = false;
      frame["right_attached"] = false;
      frame["base_pose_map"] = conveyor_pose;
      rebuilt.push_back(std::move(frame));
    }
    for (size_t index = stow_last + 1U; index < frames.size(); ++index)
      rebuilt.push_back(frames.at(index));
    output["frames"] = std::move(rebuilt);
    output["chest_shortcut"] = metadata;
    return output;
  }

  nlohmann::json tryLocalRrt(
      const planning_scene::PlanningSceneConstPtr &scene,
      const nlohmann::json &operation,
      const std::vector<std::string> &joint_names,
      const nlohmann::json &golden,
      const AttachmentSpec &left_spec,
      const AttachmentSpec &right_spec,
      double *raw_cost) const {
    std::vector<size_t> controlled_indices{0U};
    for (size_t index = 2U; index < joint_names.size(); ++index)
      controlled_indices.push_back(index);
    auto space = std::make_shared<ob::RealVectorStateSpace>(
      controlled_indices.size());
    ob::RealVectorBounds bounds(controlled_indices.size());
    for (size_t dimension = 0U; dimension < controlled_indices.size(); ++dimension) {
      const size_t joint_index = controlled_indices[dimension];
      const std::string &name = joint_names[joint_index];
      double lower = std::numeric_limits<double>::infinity();
      double upper = -std::numeric_limits<double>::infinity();
      for (const auto &frame : golden) {
        const double value = frame.at("joints").at(joint_index).get<double>();
        lower = std::min(lower, value);
        upper = std::max(upper, value);
      }
      const double margin = name == "updown" ? 0.08 : 35.0 * kPi / 180.0;
      lower -= margin;
      upper += margin;
      const auto &variable_bounds = robot_model_->getVariableBounds(name);
      if (variable_bounds.position_bounded_) {
        lower = std::max(lower, variable_bounds.min_position_);
        upper = std::min(upper, variable_bounds.max_position_);
      }
      if (!(lower < upper)) return nlohmann::json::array();
      bounds.setLow(dimension, lower);
      bounds.setHigh(dimension, upper);
    }
    space->setBounds(bounds);
    og::SimpleSetup setup(space);
    const auto start_robot = makeState(
      joint_names, golden.front(), operation, left_spec, right_spec);
    const bool left_attached = golden.front().value("left_attached", false);
    const bool right_attached = golden.front().value("right_attached", false);
    setup.setStateValidityChecker(
      [&](const ob::State *state) {
        const auto *values = state->as<ob::RealVectorStateSpace::StateType>();
        moveit::core::RobotState probe(*start_robot);
        for (size_t dimension = 0U; dimension < controlled_indices.size(); ++dimension)
          probe.setVariablePosition(
            joint_names[controlled_indices[dimension]], values->values[dimension]);
        probe.update(true);
        nlohmann::json aggregate = {
          {"maximum_left_tilt_deg", 0.0}, {"maximum_right_tilt_deg", 0.0}};
        std::string reason;
        return validateState(
          scene, probe, left_attached, right_attached,
          left_spec, right_spec, &reason, &aggregate);
      });
    setup.getSpaceInformation()->setStateValidityCheckingResolution(0.001);
    ob::ScopedState<> start(space);
    ob::ScopedState<> goal_state(space);
    for (size_t dimension = 0U; dimension < controlled_indices.size(); ++dimension) {
      const size_t joint_index = controlled_indices[dimension];
      start[dimension] = golden.front().at("joints").at(joint_index).get<double>();
      goal_state[dimension] = golden.back().at("joints").at(joint_index).get<double>();
    }
    setup.setStartAndGoalStates(start, goal_state);
    auto planner = std::make_shared<og::RRTConnect>(setup.getSpaceInformation());
    planner->setRange(0.15);
    setup.setPlanner(planner);
    const auto status = setup.solve(local_rrt_time_s_);
    if (status != ob::PlannerStatus::EXACT_SOLUTION)
      return nlohmann::json::array();
    setup.simplifySolution(0.08);
    const auto &path = setup.getSolutionPath();
    nlohmann::json sparse = nlohmann::json::array();
    const auto base_joints = golden.front().at("joints").get<std::vector<double>>();
    for (size_t path_index = 0U; path_index < path.getStateCount(); ++path_index) {
      auto joints = base_joints;
      const auto *values = path.getState(path_index)->as<
        ob::RealVectorStateSpace::StateType>();
      for (size_t dimension = 0U; dimension < controlled_indices.size(); ++dimension)
        joints[controlled_indices[dimension]] = values->values[dimension];
      auto frame = golden.front();
      frame["joints"] = joints;
      frame["stage"] = "chest_carry_retract";
      sparse.push_back(std::move(frame));
    }
    sparse.front() = golden.front();
    sparse.back() = golden.back();
    std::vector<size_t> anchors(sparse.size(), 0U);
    for (size_t index = 0U; index < anchors.size(); ++index) anchors[index] = index;
    auto dense = densifyShortcut(
      sparse, anchors, operation, joint_names, left_spec, right_spec);
    *raw_cost = 0.0;
    for (size_t index = 1U; index < sparse.size(); ++index) {
      const auto a = sparse.at(index - 1U).at("joints").get<std::vector<double>>();
      const auto b = sparse.at(index).at("joints").get<std::vector<double>>();
      *raw_cost += std::abs(b.at(0) - a.at(0)) * 100.0;
      for (size_t joint = 2U; joint < a.size(); ++joint)
        *raw_cost += std::abs(b[joint] - a[joint]) * 180.0 / kPi;
    }
    return dense;
  }

  // alpha=0 follows the endpoint chord; alpha=1 reproduces the golden curve.
  nlohmann::json manifoldFrame(
      const nlohmann::json &golden,
      const std::vector<std::string> &joint_names,
      double progress,
      double alpha) const {
    if (golden.size() < 2U) return golden.front();
    progress = std::clamp(progress, 0.0, 1.0);
    const double scaled = progress * static_cast<double>(golden.size() - 1U);
    const size_t from_index = std::min(
      golden.size() - 2U, static_cast<size_t>(std::floor(scaled)));
    const size_t to_index = from_index + 1U;
    const double local_ratio = scaled - static_cast<double>(from_index);
    const auto start = golden.front().at("joints").get<std::vector<double>>();
    const auto goal = golden.back().at("joints").get<std::vector<double>>();
    const auto from = golden.at(from_index).at("joints").get<std::vector<double>>();
    const auto to = golden.at(to_index).at("joints").get<std::vector<double>>();
    if (start.size() != joint_names.size() || goal.size() != joint_names.size() ||
        from.size() != joint_names.size() || to.size() != joint_names.size()) {
      throw std::invalid_argument("golden manifold joint count mismatch");
    }
    std::vector<double> joints(joint_names.size(), 0.0);
    for (size_t index = 0U; index < joints.size(); ++index) {
      const double golden_value = from[index] + (to[index] - from[index]) * local_ratio;
      const double chord_value = start[index] + (goal[index] - start[index]) * progress;
      joints[index] = chord_value + alpha * (golden_value - chord_value);
    }
    auto frame = golden.at(from_index);
    frame["joints"] = joints;
    if (golden.at(from_index).contains("base_pose_map") &&
        golden.at(to_index).contains("base_pose_map")) {
      const auto from_base = golden.at(from_index).at("base_pose_map")
        .get<std::vector<double>>();
      const auto to_base = golden.at(to_index).at("base_pose_map")
        .get<std::vector<double>>();
      if (from_base.size() == 3U && to_base.size() == 3U) {
        frame["base_pose_map"] = {
          from_base[0] + (to_base[0] - from_base[0]) * local_ratio,
          from_base[1] + (to_base[1] - from_base[1]) * local_ratio,
          from_base[2] + shortestAngleDelta(from_base[2], to_base[2]) * local_ratio,
        };
      }
    }
    frame["stage"] = progress >= 1.0 - 1e-9 ?
      "chest_carry_ready" : "chest_carry_retract";
    return frame;
  }

  nlohmann::json tryGoldenManifoldRrt(
      const planning_scene::PlanningSceneConstPtr &scene,
      const nlohmann::json &operation,
      const std::vector<std::string> &joint_names,
      const nlohmann::json &golden,
      const AttachmentSpec &left_spec,
      const AttachmentSpec &right_spec,
      double *raw_cost) const {
    if (golden.size() < 2U) return nlohmann::json::array();
    auto space = std::make_shared<ob::RealVectorStateSpace>(2U);
    ob::RealVectorBounds bounds(2U);
    bounds.setLow(0U, 0.0);
    bounds.setHigh(0U, 1.0);
    bounds.setLow(1U, 0.0);
    bounds.setHigh(1U, manifold_alpha_max_);
    space->setBounds(bounds);
    og::SimpleSetup setup(space);
    const bool left_attached = golden.front().value("left_attached", false);
    const bool right_attached = golden.front().value("right_attached", false);
    setup.setStateValidityChecker(
      [&](const ob::State *state) {
        const auto *values = state->as<ob::RealVectorStateSpace::StateType>();
        const auto frame = manifoldFrame(
          golden, joint_names, values->values[0], values->values[1]);
        auto probe = makeState(
          joint_names, frame, operation, left_spec, right_spec);
        nlohmann::json aggregate = {
          {"maximum_left_tilt_deg", 0.0}, {"maximum_right_tilt_deg", 0.0}};
        std::string reason;
        return validateState(
          scene, *probe, left_attached, right_attached,
          left_spec, right_spec, &reason, &aggregate);
      });
    setup.getSpaceInformation()->setStateValidityCheckingResolution(0.001);
    ob::ScopedState<> start(space);
    ob::ScopedState<> goal_state(space);
    start[0] = 0.0;
    start[1] = 0.0;
    goal_state[0] = 1.0;
    goal_state[1] = 0.0;
    setup.setStartAndGoalStates(start, goal_state);
    auto planner = std::make_shared<og::RRTConnect>(setup.getSpaceInformation());
    planner->setRange(0.08);
    setup.setPlanner(planner);
    const auto status = setup.solve(local_rrt_time_s_);
    if (status != ob::PlannerStatus::EXACT_SOLUTION)
      return nlohmann::json::array();
    setup.simplifySolution(std::min(0.08, 0.25 * local_rrt_time_s_));
    const auto &path = setup.getSolutionPath();
    if (path.getStateCount() < 2U) return nlohmann::json::array();

    nlohmann::json dense = nlohmann::json::array();
    for (size_t edge = 0U; edge + 1U < path.getStateCount(); ++edge) {
      const auto *from = path.getState(edge)->as<ob::RealVectorStateSpace::StateType>();
      const auto *to = path.getState(edge + 1U)->as<ob::RealVectorStateSpace::StateType>();
      const double progress_delta = std::abs(to->values[0] - from->values[0]);
      const double alpha_delta = std::abs(to->values[1] - from->values[1]);
      const size_t steps = std::max<size_t>({
        1U,
        static_cast<size_t>(std::ceil(
          progress_delta * static_cast<double>(golden.size() - 1U))),
        static_cast<size_t>(std::ceil(alpha_delta / 0.02)),
      });
      for (size_t step = edge == 0U ? 0U : 1U; step <= steps; ++step) {
        const double ratio = static_cast<double>(step) / static_cast<double>(steps);
        dense.push_back(manifoldFrame(
          golden, joint_names,
          from->values[0] + (to->values[0] - from->values[0]) * ratio,
          from->values[1] + (to->values[1] - from->values[1]) * ratio));
      }
    }
    if (dense.empty()) return dense;
    dense.front() = golden.front();
    dense.back() = golden.back();
    *raw_cost = 0.0;
    for (size_t index = 1U; index < dense.size(); ++index) {
      const auto a = dense.at(index - 1U).at("joints").get<std::vector<double>>();
      const auto b = dense.at(index).at("joints").get<std::vector<double>>();
      *raw_cost += std::abs(b.at(0) - a.at(0)) * 100.0;
      for (size_t joint = 2U; joint < a.size(); ++joint)
        *raw_cost += std::abs(b[joint] - a[joint]) * 180.0 / kPi;
    }
    return dense;
  }

  nlohmann::json manifoldEdgeFrames(
      const nlohmann::json &golden,
      const std::vector<std::string> &joint_names,
      double from_progress,
      double from_alpha,
      double to_progress,
      double to_alpha,
      bool include_start) const {
    const size_t steps = std::max<size_t>({
      1U,
      static_cast<size_t>(std::ceil(
        std::abs(to_progress - from_progress) *
        static_cast<double>(golden.size() - 1U))),
      static_cast<size_t>(std::ceil(std::abs(to_alpha - from_alpha) / 0.05)),
    });
    nlohmann::json frames = nlohmann::json::array();
    for (size_t step = include_start ? 0U : 1U; step <= steps; ++step) {
      const double ratio = static_cast<double>(step) / static_cast<double>(steps);
      frames.push_back(manifoldFrame(
        golden, joint_names,
        from_progress + (to_progress - from_progress) * ratio,
        from_alpha + (to_alpha - from_alpha) * ratio));
    }
    return frames;
  }

  nlohmann::json tryGoldenManifoldLattice(
      const planning_scene::PlanningSceneConstPtr &scene,
      const nlohmann::json &operation,
      const std::vector<std::string> &joint_names,
      const nlohmann::json &golden,
      const AttachmentSpec &left_spec,
      const AttachmentSpec &right_spec,
      double *raw_cost) const {
    if (golden.size() < 2U) return nlohmann::json::array();
    const size_t levels = std::min<size_t>(
      static_cast<size_t>(manifold_levels_), golden.size());
    const size_t alpha_samples = static_cast<size_t>(manifold_alpha_samples_);
    const bool left_attached = golden.front().value("left_attached", false);
    const bool right_attached = golden.front().value("right_attached", false);
    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<std::vector<bool>> valid(
      levels, std::vector<bool>(alpha_samples, false));
    std::vector<std::vector<double>> costs(
      levels, std::vector<double>(alpha_samples, infinity));
    std::vector<std::vector<size_t>> previous(
      levels, std::vector<size_t>(alpha_samples, 0U));
    const auto progress_at = [levels](size_t level) {
        return static_cast<double>(level) / static_cast<double>(levels - 1U);
      };
    const auto alpha_at = [this, alpha_samples](size_t sample) {
        return manifold_alpha_max_ * static_cast<double>(sample) /
          static_cast<double>(alpha_samples - 1U);
      };
    const auto state_valid = [&](size_t level, size_t sample) {
        const auto frame = manifoldFrame(
          golden, joint_names, progress_at(level), alpha_at(sample));
        auto state = makeState(
          joint_names, frame, operation, left_spec, right_spec);
        nlohmann::json aggregate = {
          {"maximum_left_tilt_deg", 0.0}, {"maximum_right_tilt_deg", 0.0}};
        std::string reason;
        return validateState(
          scene, *state, left_attached, right_attached,
          left_spec, right_spec, &reason, &aggregate);
      };
    const auto frame_cost = [](const nlohmann::json &from, const nlohmann::json &to) {
        const auto a = from.at("joints").get<std::vector<double>>();
        const auto b = to.at("joints").get<std::vector<double>>();
        double cost = std::abs(b.at(0) - a.at(0)) * 100.0;
        for (size_t joint = 2U; joint < a.size(); ++joint)
          cost += std::abs(b[joint] - a[joint]) * 180.0 / kPi;
        return cost;
      };
    const auto edge_cost = [&](size_t from_level, size_t from_sample,
        size_t to_level, size_t to_sample, double *cost) {
        const auto edge = manifoldEdgeFrames(
          golden, joint_names,
          progress_at(from_level), alpha_at(from_sample),
          progress_at(to_level), alpha_at(to_sample), true);
        *cost = 0.0;
        nlohmann::json previous_frame;
        for (const auto &frame : edge) {
          auto state = makeState(
            joint_names, frame, operation, left_spec, right_spec);
          nlohmann::json aggregate = {
            {"maximum_left_tilt_deg", 0.0}, {"maximum_right_tilt_deg", 0.0}};
          std::string reason;
          if (!validateState(
                scene, *state, left_attached, right_attached,
                left_spec, right_spec, &reason, &aggregate)) {
            return false;
          }
          if (!previous_frame.is_null()) *cost += frame_cost(previous_frame, frame);
          previous_frame = frame;
        }
        return true;
      };

    for (size_t level = 0U; level < levels; ++level) {
      for (size_t sample = 0U; sample < alpha_samples; ++sample)
        valid[level][sample] = state_valid(level, sample);
    }
    for (size_t sample = 0U; sample < alpha_samples; ++sample) {
      if (valid.front()[sample]) costs.front()[sample] = 0.0;
    }
    for (size_t level = 1U; level < levels; ++level) {
      for (size_t to_sample = 0U; to_sample < alpha_samples; ++to_sample) {
        if (!valid[level][to_sample]) continue;
        const size_t first = to_sample > static_cast<size_t>(manifold_alpha_neighbor_steps_) ?
          to_sample - static_cast<size_t>(manifold_alpha_neighbor_steps_) : 0U;
        const size_t last = std::min(
          alpha_samples - 1U,
          to_sample + static_cast<size_t>(manifold_alpha_neighbor_steps_));
        for (size_t from_sample = first; from_sample <= last; ++from_sample) {
          if (!std::isfinite(costs[level - 1U][from_sample])) continue;
          double transition_cost = 0.0;
          if (!edge_cost(
                level - 1U, from_sample, level, to_sample, &transition_cost)) {
            continue;
          }
          const double candidate = costs[level - 1U][from_sample] + transition_cost;
          if (candidate < costs[level][to_sample]) {
            costs[level][to_sample] = candidate;
            previous[level][to_sample] = from_sample;
          }
        }
      }
    }
    const auto best = std::min_element(costs.back().begin(), costs.back().end());
    if (best == costs.back().end() || !std::isfinite(*best))
      return nlohmann::json::array();
    std::vector<size_t> route(levels, 0U);
    route.back() = static_cast<size_t>(std::distance(costs.back().begin(), best));
    for (size_t level = levels - 1U; level > 0U; --level)
      route[level - 1U] = previous[level][route[level]];

    nlohmann::json dense = nlohmann::json::array();
    for (size_t level = 1U; level < levels; ++level) {
      const auto edge = manifoldEdgeFrames(
        golden, joint_names,
        progress_at(level - 1U), alpha_at(route[level - 1U]),
        progress_at(level), alpha_at(route[level]), level == 1U);
      for (const auto &frame : edge) dense.push_back(frame);
    }
    dense.front() = golden.front();
    dense.back() = golden.back();
    *raw_cost = 0.0;
    for (size_t index = 1U; index < dense.size(); ++index)
      *raw_cost += frame_cost(dense.at(index - 1U), dense.at(index));
    return dense;
  }

  nlohmann::json tryContract17Vamp(
      const planning_scene::PlanningSceneConstPtr &scene,
      const nlohmann::json &operation,
      const std::vector<std::string> &joint_names,
      const nlohmann::json &golden,
      const AttachmentSpec &left_spec,
      const AttachmentSpec &right_spec,
      double *raw_cost,
      LocalSearchStats *stats) const {
#ifndef ALFA_HAS_VAMP
    (void)scene;
    (void)operation;
    (void)joint_names;
    (void)golden;
    (void)left_spec;
    (void)right_spec;
    (void)raw_cost;
    if (stats) stats->backend = "vamp_unavailable";
    throw std::runtime_error(
      "local_rrt_mode=contract17_vamp requires building with vamp::vamp");
#else
    using Robot = FullStateVampRobot;
    static constexpr size_t kRake = vamp::FloatVectorWidth;
    using Rrt = vamp::planning::RRTC<Robot, kRake, Robot::resolution>;
    using VectorEnvironment =
      vamp::collision::Environment<vamp::FloatVector<kRake>>;
    struct Trial {
      bool solved = false;
      size_t seed = 0U;
      size_t nanoseconds = 0U;
      size_t iterations = 0U;
      size_t tree_nodes = 0U;
      size_t validity_batches = 0U;
      size_t validity_states = 0U;
      double fcl_ms = 0.0;
      double task_cost = std::numeric_limits<double>::infinity();
      std::vector<std::array<float, kFullStateDimension>> path;
    };
    if (golden.size() < 2U) return nlohmann::json::array();
    if (joint_names.size() != 16U) {
      throw std::invalid_argument(
        "contract17 VAMP expects the 16-axis replay contract");
    }
    const auto find_joint = [&joint_names](const std::string &name) {
        const auto found = std::find(joint_names.begin(), joint_names.end(), name);
        if (found == joint_names.end()) {
          throw std::invalid_argument("full-state VAMP missing replay joint " + name);
        }
        return static_cast<size_t>(std::distance(joint_names.begin(), found));
      };
    const size_t updown_index = find_joint("updown");
    std::vector<std::string> arm_names = left_arm_->getVariableNames();
    const auto right_names = right_arm_->getVariableNames();
    arm_names.insert(arm_names.end(), right_names.begin(), right_names.end());
    if (arm_names.size() != 14U) {
      throw std::runtime_error("contract17 VAMP requires two seven-axis arms");
    }
    std::array<size_t, 14U> arm_indices{};
    for (size_t index = 0U; index < arm_names.size(); ++index) {
      arm_indices[index] = find_joint(arm_names[index]);
    }

    const auto frame_state = [&](const nlohmann::json &frame) {
        std::array<float, kFullStateDimension> state{};
        const auto base = basePose(operation, &frame);
        state[0] = static_cast<float>(base[0]);
        state[1] = static_cast<float>(base[1]);
        state[2] = static_cast<float>(base[2]);
        for (size_t index = 0U; index < arm_indices.size(); ++index) {
          state[index + 3U] =
            frame.at("joints").at(arm_indices[index]).get<float>();
        }
        return state;
      };
    const auto start_values = frame_state(golden.front());
    const auto goal_values = frame_state(golden.back());
    const double start_updown =
      golden.front().at("joints").at(updown_index).get<double>();
    const double goal_updown =
      golden.back().at("joints").at(updown_index).get<double>();
    double projection_norm = 0.0;
    for (size_t axis = 0U; axis < kFullStateDimension; ++axis) {
      const double delta = static_cast<double>(goal_values[axis] - start_values[axis]);
      projection_norm += delta * delta;
    }
    if (projection_norm <= 1e-12) {
      throw std::runtime_error("contract17 start and goal are indistinguishable");
    }
    const auto state_progress = [start_values, goal_values, projection_norm](
        const std::array<float, kFullStateDimension> &state) {
        double projection = 0.0;
        for (size_t axis = 0U; axis < kFullStateDimension; ++axis) {
          projection += static_cast<double>(state[axis] - start_values[axis]) *
            static_cast<double>(goal_values[axis] - start_values[axis]);
        }
        return std::clamp(projection / projection_norm, 0.0, 1.0);
      };
    const auto updown_at = [start_updown, goal_updown, state_progress](
        const std::array<float, kFullStateDimension> &state) {
        return start_updown +
          (goal_updown - start_updown) * state_progress(state);
      };
    const auto task_cost = [updown_at](const auto &path) {
        double cost = 0.0;
        for (size_t index = 1U; index < path.size(); ++index) {
          const auto &from = path[index - 1U];
          const auto &to = path[index];
          cost += std::hypot(
            static_cast<double>(to[0] - from[0]),
            static_cast<double>(to[1] - from[1])) * 100.0;
          cost += std::abs(shortestAngleDelta(from[2], to[2])) * 180.0 / kPi;
          cost += std::abs(updown_at(to) - updown_at(from)) * 100.0;
          for (size_t axis = 3U; axis < kFullStateDimension; ++axis) {
            cost += std::abs(static_cast<double>(to[axis] - from[axis])) *
              180.0 / kPi;
          }
        }
        return cost;
      };

    std::array<float, kFullStateDimension> lows{};
    std::array<float, kFullStateDimension> highs{};
    lows.fill(std::numeric_limits<float>::infinity());
    highs.fill(-std::numeric_limits<float>::infinity());
    for (const auto &frame : golden) {
      const auto state = frame_state(frame);
      for (size_t axis = 0U; axis < kFullStateDimension; ++axis) {
        lows[axis] = std::min(lows[axis], state[axis]);
        highs[axis] = std::max(highs[axis], state[axis]);
      }
    }
    for (size_t axis = 0U; axis < 2U; ++axis) {
      lows[axis] -= static_cast<float>(full_vamp_base_xy_margin_);
      highs[axis] += static_cast<float>(full_vamp_base_xy_margin_);
    }
    lows[2] -= static_cast<float>(full_vamp_base_yaw_margin_);
    highs[2] += static_cast<float>(full_vamp_base_yaw_margin_);
    for (size_t axis = 3U; axis < kFullStateDimension; ++axis) {
      lows[axis] -= static_cast<float>(full_vamp_arm_margin_);
      highs[axis] += static_cast<float>(full_vamp_arm_margin_);
    }
    const auto clamp_to_bounds = [&](size_t axis, const std::string &name) {
        const auto &bounds = robot_model_->getVariableBounds(name);
        if (bounds.position_bounded_) {
          lows[axis] = std::max(lows[axis], static_cast<float>(bounds.min_position_));
          highs[axis] = std::min(highs[axis], static_cast<float>(bounds.max_position_));
        }
        if (!(lows[axis] < highs[axis])) {
          throw std::runtime_error("empty contract17 VAMP bound for " + name);
        }
      };
    for (size_t index = 0U; index < arm_names.size(); ++index) {
      clamp_to_bounds(index + 3U, arm_names[index]);
    }
    Robot::setBounds(lows, highs);
    const Robot::Configuration start(start_values);
    const Robot::Configuration goal(goal_values);
    vamp::planning::RRTCSettings settings;
    settings.range = static_cast<float>(vamp_range_);
    settings.dynamic_domain = false;
    settings.max_iterations = static_cast<size_t>(vamp_max_iterations_);
    settings.max_samples = static_cast<size_t>(vamp_max_samples_);
    const VectorEnvironment environment;
    const bool left_attached = golden.front().value("left_attached", false);
    const bool right_attached = golden.front().value("right_attached", false);

    const auto batch_started = std::chrono::steady_clock::now();
    std::vector<std::future<Trial>> futures;
    futures.reserve(static_cast<size_t>(vamp_workers_));
    for (int worker = 0; worker < vamp_workers_; ++worker) {
      auto worker_scene = planning_scene::PlanningScene::clone(scene);
      auto worker_state = makeState(
        joint_names, golden.front(), operation, left_spec, right_spec);
      futures.push_back(std::async(
        std::launch::async,
        [this, worker, worker_scene, worker_state, start, goal, environment,
        settings, arm_names, left_attached, right_attached, left_spec,
        right_spec, task_cost, updown_at]() mutable {
          Trial trial;
          trial.seed = static_cast<size_t>(worker);
          FullStateVampContext context;
          context.validator = [this, worker_scene, worker_state, arm_names,
            left_attached, right_attached, left_spec, right_spec, updown_at](
              const std::array<float, kFullStateDimension> &state) {
              const std::array<double, 3> base{
                state[0], state[1], state[2]};
              setPlanarRoot(*worker_state, base);
              worker_state->setVariablePosition("updown", updown_at(state));
              for (size_t index = 0U; index < arm_names.size(); ++index) {
                worker_state->setVariablePosition(
                  arm_names[index], state[index + 3U]);
              }
              worker_state->update(true);
              if (!worker_state->satisfiesBounds()) return false;
              if (left_attached && boxTilt(*worker_state, left_spec) >
                  maximum_tilt_ + 1e-9) return false;
              if (right_attached && boxTilt(*worker_state, right_spec) >
                  maximum_tilt_ + 1e-9) return false;
              return alfa_robot::motion::scene_collision_reason(
                worker_scene, *worker_state, nullptr).empty();
            };
          g_full_state_vamp_context = &context;
          auto rng = std::make_shared<FullStateVampSampler>(
            static_cast<uint32_t>(0xA17FA11U + worker * 0x9E3779B9U));
          auto worker_settings = settings;
          if (worker > 0) {
            worker_settings.max_iterations = static_cast<size_t>(
              vamp_secondary_max_iterations_);
            worker_settings.max_samples = std::min(
              worker_settings.max_samples,
              static_cast<size_t>(2 * vamp_secondary_max_iterations_));
          }
          const auto result = Rrt::solve(
            start, goal, environment, worker_settings, rng);
          g_full_state_vamp_context = nullptr;
          trial.solved = result.path.size() >= 2U;
          trial.nanoseconds = result.nanoseconds;
          trial.iterations = result.iterations;
          trial.tree_nodes = std::accumulate(
            result.size.begin(), result.size.end(), size_t{0U});
          trial.validity_batches = context.validity_batches;
          trial.validity_states = context.validity_states;
          trial.fcl_ms = context.fcl_ms;
          for (const auto &configuration : result.path) {
            const auto values = configuration.to_array();
            std::array<float, kFullStateDimension> state{};
            std::copy_n(values.begin(), kFullStateDimension, state.begin());
            trial.path.push_back(state);
          }
          if (trial.solved) trial.task_cost = task_cost(trial.path);
          return trial;
        }));
    }
    std::vector<Trial> trials;
    trials.reserve(futures.size());
    for (auto &future : futures) trials.push_back(future.get());
    if (stats) {
      stats->backend = "vamp_rrtc_contract17_fcl_callback";
      stats->state_dimension = kFullStateDimension;
      stats->workers = trials.size();
      stats->search_wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - batch_started).count();
      for (const auto &trial : trials) {
        stats->search_core_ms +=
          static_cast<double>(trial.nanoseconds) / 1.0e6;
        stats->search_fcl_ms += trial.fcl_ms;
        stats->validity_batches += trial.validity_batches;
        stats->validity_states += trial.validity_states;
        stats->search_iterations += trial.iterations;
        stats->search_tree_nodes += trial.tree_nodes;
        if (trial.solved) ++stats->solved_workers;
      }
    }
    const auto winner = std::min_element(
      trials.begin(), trials.end(), [](const Trial &left, const Trial &right) {
        if (left.solved != right.solved) return left.solved > right.solved;
        if (left.task_cost != right.task_cost)
          return left.task_cost < right.task_cost;
        return left.nanoseconds < right.nanoseconds;
      });
    if (winner == trials.end() || !winner->solved) {
      return nlohmann::json::array();
    }
    if (stats) {
      stats->winner_seed = static_cast<int>(winner->seed);
      stats->candidate_sparse_states = winner->path.size();
      const auto &origin = winner->path.front();
      for (size_t index = 1U; index < winner->path.size(); ++index) {
        const auto &from = winner->path[index - 1U];
        const auto &to = winner->path[index];
        stats->candidate_base_path_m += std::hypot(
          static_cast<double>(to[0] - from[0]),
          static_cast<double>(to[1] - from[1]));
        stats->candidate_base_yaw_path_deg +=
          std::abs(shortestAngleDelta(from[2], to[2])) * 180.0 / kPi;
        stats->candidate_updown_path_m +=
          std::abs(updown_at(to) - updown_at(from));
        for (size_t axis = 3U; axis < kFullStateDimension; ++axis) {
          stats->candidate_arm_path_deg +=
            std::abs(static_cast<double>(to[axis] - from[axis])) *
            180.0 / kPi;
        }
      }
      for (const auto &state : winner->path) {
        stats->candidate_max_base_offset_m = std::max(
          stats->candidate_max_base_offset_m,
          std::hypot(
            static_cast<double>(state[0] - origin[0]),
            static_cast<double>(state[1] - origin[1])));
        stats->candidate_max_base_yaw_offset_deg = std::max(
          stats->candidate_max_base_yaw_offset_deg,
          std::abs(shortestAngleDelta(origin[2], state[2])) * 180.0 / kPi);
      }
    }

    nlohmann::json sparse = nlohmann::json::array();
    for (const auto &state : winner->path) {
      auto frame = golden.front();
      auto joints = frame.at("joints").get<std::vector<double>>();
      joints[updown_index] = updown_at(state);
      for (size_t index = 0U; index < arm_indices.size(); ++index) {
        joints[arm_indices[index]] = state[index + 3U];
      }
      frame["joints"] = std::move(joints);
      frame["base_pose_map"] = {state[0], state[1], state[2]};
      frame["stage"] = "chest_carry_retract";
      sparse.push_back(std::move(frame));
    }
    sparse.front() = golden.front();
    sparse.back() = golden.back();
    std::vector<size_t> anchors(sparse.size(), 0U);
    std::iota(anchors.begin(), anchors.end(), 0U);
    *raw_cost = 0.0;
    for (size_t index = 1U; index < sparse.size(); ++index) {
      const auto from_joints = sparse.at(index - 1U).at("joints")
        .get<std::vector<double>>();
      const auto to_joints = sparse.at(index).at("joints")
        .get<std::vector<double>>();
      const auto from_base = basePose(operation, &sparse.at(index - 1U));
      const auto to_base = basePose(operation, &sparse.at(index));
      *raw_cost += std::hypot(
        to_base[0] - from_base[0], to_base[1] - from_base[1]) * 100.0;
      *raw_cost += std::abs(shortestAngleDelta(from_base[2], to_base[2])) *
        180.0 / kPi;
      *raw_cost += std::abs(
        to_joints[updown_index] - from_joints[updown_index]) * 100.0;
      for (const size_t arm_index : arm_indices) {
        *raw_cost += std::abs(
          to_joints[arm_index] - from_joints[arm_index]) * 180.0 / kPi;
      }
    }
    return densifyShortcut(
      sparse, anchors, operation, joint_names, left_spec, right_spec);
#endif
  }

  nlohmann::json tryGoldenVamp(
      const planning_scene::PlanningSceneConstPtr &scene,
      const nlohmann::json &operation,
      const std::vector<std::string> &joint_names,
      const nlohmann::json &golden,
      const AttachmentSpec &left_spec,
      const AttachmentSpec &right_spec,
      double *raw_cost,
      LocalSearchStats *stats) const {
#ifndef ALFA_HAS_VAMP
    (void)scene;
    (void)operation;
    (void)joint_names;
    (void)golden;
    (void)left_spec;
    (void)right_spec;
    (void)raw_cost;
    if (stats) stats->backend = "vamp_unavailable";
    throw std::runtime_error(
      "local_rrt_mode=golden_vamp requires building with vamp::vamp");
#else
    using Robot = vamp::robots::Sphere;
    static constexpr size_t kRake = vamp::FloatVectorWidth;
    using VectorEnvironment = vamp::collision::Environment<vamp::FloatVector<kRake>>;
    // The occupancy cells are narrower than the stock sphere resolution.
    using Rrt = vamp::planning::RRTC<Robot, kRake, 256U>;
    struct Trial {
      bool solved = false;
      size_t seed = 0U;
      size_t nanoseconds = 0U;
      size_t iterations = 0U;
      float cost = std::numeric_limits<float>::infinity();
      std::vector<std::array<float, 3>> path;
    };
    if (golden.size() < 2U) return nlohmann::json::array();
    if (stats) {
      stats->backend = "vamp_sphere_rrtc_avx2";
      stats->workers = static_cast<size_t>(vamp_workers_);
    }
    const size_t levels = std::min<size_t>(
      static_cast<size_t>(manifold_levels_), golden.size());
    const size_t alpha_samples = static_cast<size_t>(manifold_alpha_samples_);
    const bool left_attached = golden.front().value("left_attached", false);
    const bool right_attached = golden.front().value("right_attached", false);
    const auto progress_at = [levels](size_t level) {
        return static_cast<double>(level) / static_cast<double>(levels - 1U);
      };
    const auto alpha_at = [this, alpha_samples](size_t sample) {
        return manifold_alpha_max_ * static_cast<double>(sample) /
          static_cast<double>(alpha_samples - 1U);
      };

    const auto grid_started = std::chrono::steady_clock::now();
    std::vector<std::vector<bool>> valid(
      levels, std::vector<bool>(alpha_samples, false));
    for (size_t level = 0U; level < levels; ++level) {
      for (size_t sample = 0U; sample < alpha_samples; ++sample) {
        const auto frame = manifoldFrame(
          golden, joint_names, progress_at(level), alpha_at(sample));
        auto state = makeState(
          joint_names, frame, operation, left_spec, right_spec);
        nlohmann::json aggregate = {
          {"maximum_left_tilt_deg", 0.0}, {"maximum_right_tilt_deg", 0.0}};
        std::string reason;
        valid[level][sample] = validateState(
          scene, *state, left_attached, right_attached,
          left_spec, right_spec, &reason, &aggregate);
        if (stats) ++stats->grid_state_checks;
      }
    }

    vamp::collision::Environment<float> scalar_environment;
    const float progress_step = 1.0F / static_cast<float>(levels - 1U);
    const float alpha_step = static_cast<float>(manifold_alpha_max_) /
      static_cast<float>(alpha_samples - 1U);
    for (size_t level = 1U; level + 1U < levels; ++level) {
      for (size_t sample = 0U; sample < alpha_samples; ++sample) {
        if (valid[level][sample]) continue;
        scalar_environment.z_aligned_cuboids.emplace_back(
          vamp::collision::factory::cuboid::flat(
            static_cast<float>(progress_at(level)),
            static_cast<float>(alpha_at(sample)), 0.5F,
            0.0F, 0.0F, 0.0F,
            0.45F * progress_step, 0.45F * alpha_step, 1.0F));
        if (stats) ++stats->occupied_cells;
      }
    }
    if (stats) {
      stats->grid_build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - grid_started).count();
    }

    Robot::set_lows({0.0F, 0.0F, 0.5F});
    Robot::set_highs({1.0F, static_cast<float>(manifold_alpha_max_), 0.5F});
    Robot::set_radius(0.005F);
    vamp::planning::RRTCSettings settings;
    settings.range = static_cast<float>(vamp_range_);
    settings.dynamic_domain = false;
    settings.max_iterations = static_cast<size_t>(vamp_max_iterations_);
    settings.max_samples = static_cast<size_t>(vamp_max_samples_);

    for (int refinement = 0; refinement <= vamp_max_refinements_; ++refinement) {
      scalar_environment.sort();
      const VectorEnvironment environment(scalar_environment);
      const auto batch_started = std::chrono::steady_clock::now();
      std::vector<std::future<Trial>> futures;
      futures.reserve(static_cast<size_t>(vamp_workers_));
      for (int worker = 0; worker < vamp_workers_; ++worker) {
        futures.push_back(std::async(
          std::launch::async,
          [worker, refinement, this, &environment, &settings]() {
            Trial trial;
            trial.seed = static_cast<size_t>(worker);
            const float endpoint_alpha = vamp_workers_ == 1 ? 0.0F :
              static_cast<float>(manifold_alpha_max_) *
              static_cast<float>(worker) / static_cast<float>(vamp_workers_ - 1);
            const Robot::Configuration worker_start(
              Robot::ConfigurationArray{0.0F, endpoint_alpha, 0.5F});
            const Robot::Configuration worker_goal(
              Robot::ConfigurationArray{1.0F, endpoint_alpha, 0.5F});
            const float offset = static_cast<float>(worker * 6 + refinement * 24);
            auto rng = std::make_shared<vamp::rng::Halton<Robot>>(
              std::initializer_list<float>{
                3.0F + offset, 5.0F + offset, 7.0F + offset});
            auto result = Rrt::solve(
              worker_start, worker_goal, environment, settings, rng);
            trial.solved = result.path.size() >= 2U;
            trial.nanoseconds = result.nanoseconds;
            trial.iterations = result.iterations;
            trial.cost = result.cost;
            for (const auto &configuration : result.path) {
              const auto values = configuration.to_array();
              trial.path.push_back({values[0], values[1], values[2]});
            }
            return trial;
          }));
      }
      std::vector<Trial> trials;
      trials.reserve(futures.size());
      for (auto &future : futures) trials.push_back(future.get());
      if (stats) {
        stats->search_wall_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - batch_started).count();
        for (const auto &trial : trials) {
          stats->search_core_ms += static_cast<double>(trial.nanoseconds) / 1.0e6;
          if (trial.solved) ++stats->solved_workers;
        }
      }
      const auto winner = std::min_element(
        trials.begin(), trials.end(), [](const Trial &left, const Trial &right) {
          if (left.solved != right.solved) return left.solved > right.solved;
          if (left.cost != right.cost) return left.cost < right.cost;
          return left.nanoseconds < right.nanoseconds;
        });
      if (winner == trials.end() || !winner->solved) break;
      if (stats) stats->winner_seed = static_cast<int>(winner->seed);

      nlohmann::json dense = nlohmann::json::array();
      std::vector<std::array<double, 2>> dense_coordinates;
      for (size_t edge = 0U; edge + 1U < winner->path.size(); ++edge) {
        const auto &from = winner->path[edge];
        const auto &to = winner->path[edge + 1U];
        const size_t steps = std::max<size_t>({
          1U,
          static_cast<size_t>(std::ceil(
            std::abs(to[0] - from[0]) * static_cast<double>(golden.size() - 1U))),
          static_cast<size_t>(std::ceil(std::abs(to[1] - from[1]) / 0.02)),
        });
        for (size_t step = edge == 0U ? 0U : 1U; step <= steps; ++step) {
          const double ratio = static_cast<double>(step) / static_cast<double>(steps);
          const double progress = from[0] + (to[0] - from[0]) * ratio;
          const double alpha = from[1] + (to[1] - from[1]) * ratio;
          dense.push_back(manifoldFrame(golden, joint_names, progress, alpha));
          dense_coordinates.push_back({progress, alpha});
        }
      }
      if (dense.empty()) break;
      dense.front() = golden.front();
      dense.back() = golden.back();
      nlohmann::json candidate = operation;
      candidate["frames"] = dense;
      nlohmann::json validation = {
        {"maximum_left_tilt_deg", 0.0}, {"maximum_right_tilt_deg", 0.0},
        {"checked_frames", 0}, {"checked_edge_samples", 0}};
      nlohmann::json validation_result;
      const auto candidate_validation_started = std::chrono::steady_clock::now();
      if (validateOperation(
            candidate, joint_names, &validation_result, &validation)) {
        *raw_cost = 0.0;
        for (size_t index = 1U; index < dense.size(); ++index) {
          const auto a = dense.at(index - 1U).at("joints").get<std::vector<double>>();
          const auto b = dense.at(index).at("joints").get<std::vector<double>>();
          *raw_cost += std::abs(b.at(0) - a.at(0)) * 100.0;
          for (size_t joint = 2U; joint < a.size(); ++joint)
            *raw_cost += std::abs(b[joint] - a[joint]) * 180.0 / kPi;
        }
        if (stats) {
          stats->refinements = static_cast<size_t>(refinement);
          stats->candidate_fcl_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - candidate_validation_started).count();
        }
        return dense;
      }

      if (stats) {
        stats->candidate_fcl_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - candidate_validation_started).count();
        ++stats->rejected_candidates;
      }
      const size_t failed_frame = std::min<size_t>(
        validation_result.value("failure_frame", dense_coordinates.size() / 2U),
        dense_coordinates.size() - 1U);
      auto failed = dense_coordinates[failed_frame];
      if (failed_frame > 0U && validation_result.contains("failure_edge_step")) {
        auto from_state = makeState(
          joint_names, dense.at(failed_frame - 1U), operation, left_spec, right_spec);
        auto to_state = makeState(
          joint_names, dense.at(failed_frame), operation, left_spec, right_spec);
        const size_t steps = edgeSteps(*from_state, *to_state);
        const size_t failed_step = std::min<size_t>(
          validation_result.at("failure_edge_step").get<size_t>(), steps);
        const double ratio = static_cast<double>(failed_step) /
          static_cast<double>(std::max<size_t>(1U, steps));
        failed[0] = dense_coordinates[failed_frame - 1U][0] +
          (dense_coordinates[failed_frame][0] -
          dense_coordinates[failed_frame - 1U][0]) * ratio;
        failed[1] = dense_coordinates[failed_frame - 1U][1] +
          (dense_coordinates[failed_frame][1] -
          dense_coordinates[failed_frame - 1U][1]) * ratio;
      }
      if (stats) {
        stats->last_rejection_reason = validation_result.value(
          "failure_reason", "unknown FCL rejection");
        stats->last_rejection_progress = failed[0];
        stats->last_rejection_alpha = failed[1];
      }
      if (failed[0] <= 1e-6 || failed[0] >= 1.0 - 1e-6) break;
      scalar_environment.z_aligned_cuboids.emplace_back(
        vamp::collision::factory::cuboid::flat(
          static_cast<float>(failed[0]), static_cast<float>(failed[1]), 0.5F,
          0.0F, 0.0F, 0.0F,
          0.75F * progress_step, 0.75F * alpha_step, 1.0F));
      if (stats) {
        ++stats->occupied_cells;
        stats->refinements = static_cast<size_t>(refinement + 1);
      }
    }
    return nlohmann::json::array();
#endif
  }

  nlohmann::json optimizeOperation(
      const nlohmann::json &operation,
      const std::vector<std::string> &joint_names,
      nlohmann::json *operation_result) const {
    const auto &frames = operation.at("frames");
    size_t chest_start = frames.size();
    size_t chest_end = frames.size();
    for (size_t index = 0U; index < frames.size(); ++index) {
      const std::string stage = frames.at(index).value("stage", "");
      if (stage == "chest_carry_retract" && chest_start == frames.size())
        chest_start = index;
      if (stage == "chest_carry_ready") chest_end = index;
    }
    if (chest_start == frames.size() || chest_end < chest_start)
      throw std::runtime_error("operation lacks chest-carry boundaries");
    nlohmann::json golden = nlohmann::json::array();
    for (size_t index = chest_start; index <= chest_end; ++index)
      golden.push_back(frames.at(index));
    const auto left_spec = attachmentSpec(operation.at("left_attachment"), "left");
    const auto right_spec = attachmentSpec(operation.at("right_attachment"), "right");
    const auto scene = makeScene(operation);

    const auto started = std::chrono::steady_clock::now();
    size_t visibility_checks = 0U;
    size_t checked_samples = 0U;
    const size_t goal = golden.size() - 1U;
    const auto frame_distance = [](const nlohmann::json &left,
        const nlohmann::json &right) {
        const auto a = left.at("joints").get<std::vector<double>>();
        const auto b = right.at("joints").get<std::vector<double>>();
        double cost = std::abs(b.at(0) - a.at(0)) * 100.0;
        for (size_t index = 2U; index < a.size(); ++index)
          cost += std::abs(b[index] - a[index]) * 180.0 / kPi;
        return cost;
      };
    const auto golden_cost = [&](size_t from, size_t to) {
        double cost = 0.0;
        for (size_t index = from; index < to; ++index)
          cost += frame_distance(golden.at(index), golden.at(index + 1U));
        return cost;
      };

    const size_t representative_count = std::min<size_t>(80U, golden.size());
    std::vector<size_t> representatives;
    representatives.reserve(representative_count);
    for (size_t index = 0U; index < representative_count; ++index) {
      const size_t source = static_cast<size_t>(std::llround(
        static_cast<double>(index) * static_cast<double>(goal) /
        static_cast<double>(representative_count - 1U)));
      if (representatives.empty() || representatives.back() != source)
        representatives.push_back(source);
    }
    struct RouteEdge {size_t from; size_t to; bool direct;};
    std::vector<RouteEdge> route;
    double graph_cost = golden_cost(0U, goal);
    size_t visibility_cache_hits = 0U;
    std::string visibility_strategy = "full_dag";
    const size_t last_representative = representatives.size() - 1U;
    const auto golden_representative_route = [&]() {
        std::vector<RouteEdge> output;
        for (size_t index = 1U; index < representatives.size(); ++index) {
          output.push_back(RouteEdge{
            representatives[index - 1U], representatives[index], false});
        }
        return output;
      };
    route = golden_representative_route();

    std::map<std::pair<size_t, size_t>, bool> visibility_cache;
    const auto edge_valid = [&](size_t from, size_t to) {
        const auto key = std::make_pair(from, to);
        const auto cached = visibility_cache.find(key);
        if (cached != visibility_cache.end()) {
          ++visibility_cache_hits;
          return cached->second;
        }
        ++visibility_checks;
        const bool valid = shortcutEdgeValid(
          scene, operation, joint_names,
          golden.at(representatives[from]), golden.at(representatives[to]),
          left_spec, right_spec, &checked_samples);
        visibility_cache.emplace(key, valid);
        return valid;
      };

    bool run_full_dag = visibility_graph_mode_ == "full";
    if (visibility_graph_mode_ == "fast_families") {
      struct FamilyCandidate {double cost; size_t middle;};
      const auto by_cost = [last_representative](
          const FamilyCandidate &left, const FamilyCandidate &right) {
          if (std::abs(left.cost - right.cost) > 1e-9)
            return left.cost < right.cost;
          const auto center = static_cast<double>(last_representative) / 2.0;
          return std::abs(static_cast<double>(left.middle) - center) <
            std::abs(static_cast<double>(right.middle) - center);
        };
      const auto direct_cost = [&](size_t from, size_t to) {
          return frame_distance(
            golden.at(representatives[from]), golden.at(representatives[to]));
        };
      const double lower_bound = direct_cost(0U, last_representative);
      bool family_improved = false;
      if (lower_bound >= graph_cost - 1e-9) {
        visibility_strategy = "golden_at_metric_lower_bound";
        run_full_dag = false;
      } else if (edge_valid(0U, last_representative)) {
        graph_cost = lower_bound;
        route = {RouteEdge{0U, goal, true}};
        visibility_strategy = "direct_lower_bound";
        family_improved = true;
        run_full_dag = false;
      } else {
        std::vector<FamilyCandidate> suffixes;
        std::vector<FamilyCandidate> prefixes;
        const std::array<int, 7U> center_offsets{0, -1, 1, -2, 2, -3, 3};
        std::vector<size_t> probed_centers;
        const int center = static_cast<int>(last_representative / 2U);
        for (const int offset : center_offsets) {
          const int probe = center + offset;
          if (probe <= 0 ||
              probe >= static_cast<int>(last_representative)) continue;
          const size_t middle = static_cast<size_t>(probe);
          if (std::find(probed_centers.begin(), probed_centers.end(), middle) !=
              probed_centers.end()) continue;
          probed_centers.push_back(middle);
          const double cost = direct_cost(0U, middle) +
            direct_cost(middle, last_representative);
          if (cost >= graph_cost - 1e-9) continue;
          if (!edge_valid(0U, middle) ||
              !edge_valid(middle, last_representative)) continue;
          graph_cost = cost;
          route = {
            RouteEdge{0U, representatives[middle], true},
            RouteEdge{representatives[middle], goal, true}};
          visibility_strategy = "center_waypoint";
          family_improved = true;
        }
        if (!family_improved) {
          for (size_t middle = 1U; middle < last_representative; ++middle) {
            suffixes.push_back(
                FamilyCandidate{golden_cost(0U, representatives[middle]) +
                                    direct_cost(middle, last_representative),
                                middle});
            prefixes.push_back(
                FamilyCandidate{direct_cost(0U, middle) +
                                    golden_cost(representatives[middle], goal),
                                middle});
          }
          std::sort(suffixes.begin(), suffixes.end(), by_cost);
          std::sort(prefixes.begin(), prefixes.end(), by_cost);
          for (const auto &candidate : suffixes) {
            if (candidate.cost >= graph_cost - 1e-9)
              break;
            if (!edge_valid(candidate.middle, last_representative))
              continue;
            graph_cost = candidate.cost;
            route.clear();
            for (size_t index = 1U; index <= candidate.middle; ++index) {
              route.push_back(RouteEdge{representatives[index - 1U],
                                        representatives[index], false});
            }
            route.push_back(
                RouteEdge{representatives[candidate.middle], goal, true});
            visibility_strategy = "golden_prefix_direct_suffix";
            family_improved = true;
            break;
          }
          for (const auto &candidate : prefixes) {
            if (candidate.cost >= graph_cost - 1e-9)
              break;
            if (!edge_valid(0U, candidate.middle))
              continue;
            graph_cost = candidate.cost;
            route = {RouteEdge{0U, representatives[candidate.middle], true}};
            for (size_t index = candidate.middle + 1U;
                 index <= last_representative; ++index) {
              route.push_back(RouteEdge{representatives[index - 1U],
                                        representatives[index], false});
            }
            visibility_strategy = "direct_prefix_golden_suffix";
            family_improved = true;
            break;
          }
          std::vector<FamilyCandidate> one_waypoints;
          for (size_t middle = 1U; middle < last_representative; ++middle) {
            one_waypoints.push_back(
                FamilyCandidate{direct_cost(0U, middle) +
                                    direct_cost(middle, last_representative),
                                middle});
          }
          std::sort(one_waypoints.begin(), one_waypoints.end(), by_cost);
          for (const auto &candidate : one_waypoints) {
            if (candidate.cost >= graph_cost - 1e-9)
              break;
            if (!edge_valid(0U, candidate.middle) ||
                !edge_valid(candidate.middle, last_representative))
              continue;
            graph_cost = candidate.cost;
            route = {RouteEdge{0U, representatives[candidate.middle], true},
                     RouteEdge{representatives[candidate.middle], goal, true}};
            visibility_strategy = "one_waypoint";
            family_improved = true;
            break;
          }
        }
        run_full_dag = !family_improved;
      }
    }

    if (run_full_dag) {
      const double infinity = std::numeric_limits<double>::infinity();
      std::vector<double> costs(representatives.size(), infinity);
      std::vector<size_t> previous(representatives.size(), 0U);
      std::vector<bool> direct_edge(representatives.size(), false);
      costs.front() = 0.0;
      for (size_t to = 1U; to < representatives.size(); ++to) {
        const size_t prior = to - 1U;
        costs[to] = costs[prior] + golden_cost(
          representatives[prior], representatives[to]);
        previous[to] = prior;
        direct_edge[to] = false;
        for (size_t from = 0U; from < to; ++from) {
          const double direct_cost = frame_distance(
            golden.at(representatives[from]), golden.at(representatives[to]));
          if (costs[from] + direct_cost >= costs[to] - 1e-9) continue;
          if (edge_valid(from, to)) {
            costs[to] = costs[from] + direct_cost;
            previous[to] = from;
            direct_edge[to] = true;
          }
        }
      }
      graph_cost = costs.back();
      route.clear();
      for (size_t current = last_representative; current > 0U;) {
        const size_t from = previous[current];
        route.push_back(RouteEdge{
          representatives[from], representatives[current], direct_edge[current]});
        current = from;
      }
      std::reverse(route.begin(), route.end());
      visibility_strategy = visibility_graph_mode_ == "fast_families" ?
        "fast_families_fallback_full_dag" : "full_dag";
    }
    nlohmann::json optimized = nlohmann::json::array();
    optimized.push_back(golden.at(0));
    std::vector<size_t> anchors{0U};
    size_t shortcut_edges = 0U;
    size_t golden_windows = 0U;
    for (const auto &edge : route) {
      anchors.push_back(edge.to);
      if (edge.direct) {
        ++shortcut_edges;
        const auto dense = densifyShortcut(
          golden, std::vector<size_t>{edge.from, edge.to}, operation,
          joint_names, left_spec, right_spec);
        for (size_t index = 1U; index < dense.size(); ++index)
          optimized.push_back(dense.at(index));
      } else {
        ++golden_windows;
        for (size_t index = edge.from + 1U; index <= edge.to; ++index)
          optimized.push_back(golden.at(index));
      }
    }
    for (auto &frame : optimized) frame["stage"] = "chest_carry_retract";
    optimized.back()["stage"] = "chest_carry_ready";
    double selected_cost = graph_cost;
    double local_rrt_cost = std::numeric_limits<double>::infinity();
    bool local_rrt_success = false;
    bool local_candidate_fcl_valid = false;
    LocalSearchStats local_search_stats;
    local_search_stats.backend = local_rrt_mode_;
    const auto local_rrt_started = std::chrono::steady_clock::now();
    nlohmann::json local_rrt;
    if (local_rrt_mode_ == "golden_manifold") {
      local_rrt = tryGoldenManifoldRrt(
        scene, operation, joint_names, golden, left_spec, right_spec,
        &local_rrt_cost);
    } else if (local_rrt_mode_ == "golden_lattice") {
      local_rrt = tryGoldenManifoldLattice(
        scene, operation, joint_names, golden, left_spec, right_spec,
        &local_rrt_cost);
    } else if (local_rrt_mode_ == "golden_vamp") {
      local_rrt = tryGoldenVamp(
        scene, operation, joint_names, golden, left_spec, right_spec,
        &local_rrt_cost, &local_search_stats);
    } else if (local_rrt_mode_ == "contract17_vamp") {
      local_rrt = tryContract17Vamp(
        scene, operation, joint_names, golden, left_spec, right_spec,
        &local_rrt_cost, &local_search_stats);
    } else {
      local_rrt = tryLocalRrt(
        scene, operation, joint_names, golden, left_spec, right_spec,
        &local_rrt_cost);
    }
    const double local_rrt_wall_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - local_rrt_started).count();
    if (!local_rrt.empty()) {
      nlohmann::json rrt_candidate = operation;
      rrt_candidate["frames"] = local_rrt;
      nlohmann::json rrt_validation = {
        {"maximum_left_tilt_deg", 0.0}, {"maximum_right_tilt_deg", 0.0},
        {"checked_frames", 0}, {"checked_edge_samples", 0}};
      nlohmann::json rrt_result;
      const auto candidate_fcl_started = std::chrono::steady_clock::now();
      local_candidate_fcl_valid = validateOperation(
        rrt_candidate, joint_names, &rrt_result, &rrt_validation);
      if (local_rrt_mode_ == "contract17_vamp") {
        local_search_stats.candidate_fcl_ms +=
          std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - candidate_fcl_started).count();
        if (!local_candidate_fcl_valid) {
          ++local_search_stats.rejected_candidates;
          local_search_stats.last_rejection_reason = rrt_result.value(
            "failure_reason", "full-state candidate failed FCL validation");
        }
      }
      if (local_candidate_fcl_valid && local_rrt_cost < selected_cost - 1e-9) {
        optimized = std::move(local_rrt);
        selected_cost = local_rrt_cost;
        local_rrt_success = true;
      }
    }
    nlohmann::json candidate = operation;
    candidate["frames"] = optimized;
    nlohmann::json validation = {
      {"maximum_left_tilt_deg", 0.0}, {"maximum_right_tilt_deg", 0.0},
      {"checked_frames", 0}, {"checked_edge_samples", 0}};
    nlohmann::json validation_result;
    if (!validateOperation(
          candidate, joint_names, &validation_result, &validation)) {
      throw std::runtime_error(
        "optimized chest segment failed: " +
        validation_result.value("failure_reason", "unknown"));
    }
    const double core_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    nlohmann::json metadata = {
      {"strategy", local_rrt_success ? "bounded_local_rrt_with_shortcut" :
        (golden_windows == 0U && shortcut_edges == 1U ?
        "direct_local_shortcut" :
        (local_rrt_mode_ == "golden_vamp" ?
        "representative_graph_shortcut_with_golden_vamp" :
        (local_rrt_mode_ == "contract17_vamp" ?
        "representative_graph_shortcut_with_contract17_vamp" :
        "representative_graph_shortcut_with_golden_local_rrt")))},
      {"cache_loading_counted", false},
      {"source_frame_count", golden.size()},
      {"optimized_frame_count", optimized.size()},
      {"anchor_count", anchors.size()},
      {"anchors", anchors},
      {"representative_node_count", representatives.size()},
      {"shortcut_edge_count", shortcut_edges},
      {"golden_rrt_window_count", golden_windows},
      {"source_joint_travel_cost", golden_cost(0U, goal)},
      {"graph_joint_travel_cost", graph_cost},
      {"local_rrt_attempted", true},
      {"local_rrt_mode", local_rrt_mode_},
      {"local_rrt_state_dimension", local_rrt_mode_ == "contract17_vamp" ?
        static_cast<int>(kFullStateDimension) :
        (local_rrt_mode_ != "full" ? 2 :
        static_cast<int>(joint_names.size() - 1U))},
      {"local_rrt_wall_ms", local_rrt_wall_ms},
      {"local_candidate_fcl_valid", local_candidate_fcl_valid},
      {"local_search_backend", local_search_stats.backend},
      {"local_search_grid_state_checks", local_search_stats.grid_state_checks},
      {"local_search_occupied_cells", local_search_stats.occupied_cells},
      {"local_search_grid_build_ms", local_search_stats.grid_build_ms},
      {"local_search_planner_wall_ms", local_search_stats.search_wall_ms},
      {"local_search_planner_core_ms", local_search_stats.search_core_ms},
      {"local_search_fcl_ms", local_search_stats.search_fcl_ms},
      {"local_search_candidate_fcl_ms", local_search_stats.candidate_fcl_ms},
      {"local_search_validity_batches", local_search_stats.validity_batches},
      {"local_search_validity_states", local_search_stats.validity_states},
      {"local_search_iterations", local_search_stats.search_iterations},
      {"local_search_tree_nodes", local_search_stats.search_tree_nodes},
      {"local_search_candidate_sparse_states",
        local_search_stats.candidate_sparse_states},
      {"local_search_candidate_base_path_m",
        local_search_stats.candidate_base_path_m},
      {"local_search_candidate_base_yaw_path_deg",
        local_search_stats.candidate_base_yaw_path_deg},
      {"local_search_candidate_max_base_offset_m",
        local_search_stats.candidate_max_base_offset_m},
      {"local_search_candidate_max_base_yaw_offset_deg",
        local_search_stats.candidate_max_base_yaw_offset_deg},
      {"local_search_candidate_updown_path_m",
        local_search_stats.candidate_updown_path_m},
      {"local_search_candidate_arm_path_deg",
        local_search_stats.candidate_arm_path_deg},
      {"local_search_workers", local_search_stats.workers},
      {"local_search_solved_workers", local_search_stats.solved_workers},
      {"local_search_winner_seed", local_search_stats.winner_seed},
      {"local_search_refinements", local_search_stats.refinements},
      {"local_search_rejected_candidates", local_search_stats.rejected_candidates},
      {"local_search_last_rejection_reason", local_search_stats.last_rejection_reason},
      {"local_search_last_rejection_progress", local_search_stats.last_rejection_progress},
      {"local_search_last_rejection_alpha", local_search_stats.last_rejection_alpha},
      {"local_rrt_success", local_rrt_success},
      {"local_rrt_joint_travel_cost",
        std::isfinite(local_rrt_cost) ? nlohmann::json(local_rrt_cost) :
        nlohmann::json(nullptr)},
      {"optimized_joint_travel_cost", selected_cost},
      {"visibility_checks", visibility_checks},
      {"visibility_cache_hits", visibility_cache_hits},
      {"visibility_strategy", visibility_strategy},
      {"visibility_graph_mode", visibility_graph_mode_},
      {"checked_edge_samples", checked_samples},
      {"planning_core_ms", core_ms},
      {"within_3s", core_ms < 3000.0},
    };
    *operation_result = metadata;
    (*operation_result)["label"] = operation.value("label", "unnamed");
    return rebuildOperation(operation, optimized, metadata, chest_start);
  }

  bool validateOperation(const nlohmann::json &operation,
                         const std::vector<std::string> &joint_names,
                         nlohmann::json *operation_result,
                         nlohmann::json *aggregate) const {
    operation_result->update({
        {"label", operation.value("label", "unnamed")},
        {"success", false},
        {"checked_frames", 0},
        {"checked_edge_samples", 0},
        {"base_pose_map",
         operation.value("base_pose_map", std::vector<double>{0.0, 0.0, 0.0})},
    });
    const auto frames = operation.at("frames");
    if (!frames.is_array() || frames.empty()) {
      (*operation_result)["failure_reason"] = "operation has no frames";
      return false;
    }
    const auto left_spec =
        attachmentSpec(operation.at("left_attachment"), "left");
    const auto right_spec =
        attachmentSpec(operation.at("right_attachment"), "right");
    const auto scene = makeScene(operation);
    moveit::core::RobotStatePtr previous;
    bool previous_left_attached = false;
    bool previous_right_attached = false;

    for (size_t frame_index = 0U; frame_index < frames.size(); ++frame_index) {
      const auto &frame = frames.at(frame_index);
      const bool left_attached = frame.value("left_attached", false);
      const bool right_attached = frame.value("right_attached", false);
      auto current =
          makeState(joint_names, frame, operation, left_spec, right_spec);
      std::string reason;

      if (previous) {
        if (left_attached != previous_left_attached ||
            right_attached != previous_right_attached) {
          const size_t steps = edgeSteps(*previous, *current);
          if (steps > 1U) {
            (*operation_result)["failure_reason"] =
                "attachment changed while joints moved";
            (*operation_result)["failure_frame"] = frame_index;
            return false;
          }
        } else {
          const size_t steps = edgeSteps(*previous, *current);
          for (size_t step = 1U; step < steps; ++step) {
            const double ratio =
                static_cast<double>(step) / static_cast<double>(steps);
            moveit::core::RobotState probe(*previous);
            interpolateState(*previous, *current, ratio, &probe);
            if (!validateState(scene, probe, left_attached, right_attached,
                               left_spec, right_spec, &reason, aggregate)) {
              (*operation_result)["failure_reason"] = "edge " + reason;
              (*operation_result)["failure_frame"] = frame_index;
              (*operation_result)["failure_edge_step"] = step;
              return false;
            }
            (*operation_result)["checked_edge_samples"] =
                operation_result->at("checked_edge_samples").get<size_t>() + 1U;
            (*aggregate)["checked_edge_samples"] =
                aggregate->at("checked_edge_samples").get<size_t>() + 1U;
          }
        }
      }

      if (!validateState(scene, *current, left_attached, right_attached,
                         left_spec, right_spec, &reason, aggregate)) {
        (*operation_result)["failure_reason"] = reason;
        (*operation_result)["failure_frame"] = frame_index;
        (*operation_result)["failure_stage"] = frame.value("stage", "");
        return false;
      }
      (*operation_result)["checked_frames"] =
          operation_result->at("checked_frames").get<size_t>() + 1U;
      (*aggregate)["checked_frames"] =
          aggregate->at("checked_frames").get<size_t>() + 1U;
      previous = std::move(current);
      previous_left_attached = left_attached;
      previous_right_attached = right_attached;
    }
    (*operation_result)["success"] = true;
    return true;
  }

  std::string replay_path_;
  std::string output_path_;
  std::string result_path_;
  double edge_joint_step_ = kPi / 180.0;
  double edge_updown_step_ = 0.01;
  double collision_inset_ = 0.002;
  double maximum_tilt_ = 95.0 * kPi / 180.0;
  std::string local_rrt_mode_ = "full";
  std::string visibility_graph_mode_ = "full";
  double local_rrt_time_s_ = 0.95;
  double manifold_alpha_max_ = 1.25;
  int manifold_levels_ = 64;
  int manifold_alpha_samples_ = 11;
  int manifold_alpha_neighbor_steps_ = 1;
  int vamp_workers_ = 4;
  int vamp_max_iterations_ = 1000;
  int vamp_max_samples_ = 2000;
  int vamp_secondary_max_iterations_ = 300;
  double vamp_range_ = 0.08;
  int vamp_max_refinements_ = 8;
  double full_vamp_base_xy_margin_ = 0.15;
  double full_vamp_base_yaw_margin_ = 5.0 * kPi / 180.0;
  double full_vamp_arm_margin_ = 35.0 * kPi / 180.0;
  double edge_base_translation_step_ = 0.01;
  double edge_base_yaw_step_ = kPi / 180.0;
  std::shared_ptr<robot_model_loader::RobotModelLoader> robot_model_loader_;
  moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::JointModelGroup *whole_body_ = nullptr;
  const moveit::core::JointModelGroup *left_arm_ = nullptr;
  const moveit::core::JointModelGroup *right_arm_ = nullptr;
  const moveit::core::JointModel *planar_root_ = nullptr;
};

} // namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  bool success = false;
  try {
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);
    auto node = std::make_shared<ChestCarryShortcutOptimizer>(options);
    node->init();
    success = node->run();
  } catch (const std::exception &error) {
    std::cerr << "chest shortcut optimizer exception: " << error.what() << '\n';
  }
  rclcpp::shutdown();
  return success ? 0 : 1;
}
