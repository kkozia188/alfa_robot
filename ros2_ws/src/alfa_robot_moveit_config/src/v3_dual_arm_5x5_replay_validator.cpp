#include <alfa_robot_moveit_config/planning_diagnostics.hpp>

#include <geometric_shapes/shapes.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr char kLeftBoxId[] = "dual_carried_box_left";
constexpr char kRightBoxId[] = "dual_carried_box_right";

struct AttachmentSpec {
  std::string id;
  std::string side;
  std::string tool_link;
  Eigen::Vector3d center_in_tool = Eigen::Vector3d::Zero();
  Eigen::Matrix3d rotation_in_tool = Eigen::Matrix3d::Identity();
  Eigen::Vector3d size = Eigen::Vector3d::Zero();
};

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

class DualReplayValidator : public rclcpp::Node {
public:
  explicit DualReplayValidator(const rclcpp::NodeOptions &options)
      : Node("v3_dual_arm_5x5_replay_validator", options) {
    replay_path_ = getParameter<std::string>("replay_json_path", "");
    result_path_ = getParameter<std::string>("result_json_path", "");
    edge_joint_step_ =
        getParameter<double>("edge_joint_step_deg", 1.0) * kPi / 180.0;
    edge_updown_step_ = getParameter<double>("edge_updown_step_m", 0.01);
    collision_inset_ = getParameter<double>("collision_inset", 0.002);
    maximum_tilt_ = getParameter<double>("maximum_carried_box_tilt_deg", 95.0) *
                    kPi / 180.0;
    if (replay_path_.empty() || result_path_.empty()) {
      throw std::invalid_argument(
          "replay_json_path and result_json_path are required");
    }
    if (edge_joint_step_ <= 0.0 || edge_updown_step_ <= 0.0 ||
        maximum_tilt_ <= 0.0 || maximum_tilt_ > kPi) {
      throw std::invalid_argument(
          "validation steps and tilt limit must be positive");
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
    const auto input = readJson(replay_path_);
    nlohmann::json result = {
        {"schema", "alfa.v3_scoop_5x5_dual_validation.v1"},
        {"input", replay_path_},
        {"success", true},
        {"checked_frames", 0},
        {"checked_edge_samples", 0},
        {"maximum_left_tilt_deg", 0.0},
        {"maximum_right_tilt_deg", 0.0},
        {"operations", nlohmann::json::array()},
    };

    try {
      validateInput(input);
      const auto joint_names =
          input.at("joint_names").get<std::vector<std::string>>();
      for (const auto &operation : input.at("operations")) {
        nlohmann::json operation_result;
        if (!validateOperation(operation, joint_names, &operation_result,
                               &result)) {
          result["success"] = false;
          result["operations"].push_back(std::move(operation_result));
          break;
        }
        result["operations"].push_back(std::move(operation_result));
      }
    } catch (const std::exception &error) {
      result["success"] = false;
      result["failure_reason"] = error.what();
    }

    writeJson(result_path_, result);
    if (result.at("success").get<bool>()) {
      RCLCPP_INFO(get_logger(),
                  "Dual replay validation passed: frames=%zu edge_samples=%zu",
                  result.at("checked_frames").get<size_t>(),
                  result.at("checked_edge_samples").get<size_t>());
      return true;
    }
    RCLCPP_ERROR(get_logger(), "Dual replay validation failed: %s",
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
  std::string result_path_;
  double edge_joint_step_ = kPi / 180.0;
  double edge_updown_step_ = 0.01;
  double collision_inset_ = 0.002;
  double maximum_tilt_ = 95.0 * kPi / 180.0;
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
    auto node = std::make_shared<DualReplayValidator>(options);
    node->init();
    success = node->run();
  } catch (const std::exception &error) {
    std::cerr << "dual replay validator exception: " << error.what() << '\n';
  }
  rclcpp::shutdown();
  return success ? 0 : 1;
}
