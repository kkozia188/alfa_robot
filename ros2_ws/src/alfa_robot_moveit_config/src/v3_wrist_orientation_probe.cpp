#include <alfa_robot_analytic_ik/v3_redundant_analytic_ik.hpp>
#include <alfa_robot_moveit_config/wall_sequence.hpp>

#include <geometric_shapes/shapes.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <nlohmann/json.hpp>
#include <ompl/base/PlannerData.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/util/RandomNumbers.h>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
using Json = nlohmann::json;
using Solver = alfa_robot::analytic_ik::V3RedundantArmAnalyticIk;
using Model = alfa_robot::analytic_ik::V3RedundantArmModel;
namespace ob = ompl::base;
namespace og = ompl::geometric;
constexpr double kPi = 3.14159265358979323846;

void addBox(const planning_scene::PlanningScenePtr& scene, const std::string& name,
  const Json& box)
{
  moveit_msgs::msg::CollisionObject object;
  object.header.frame_id = "world";
  object.id = name;
  object.operation = moveit_msgs::msg::CollisionObject::ADD;
  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
  const auto size = box.at("size").get<std::vector<double>>();
  if (size.size() != 3) throw std::invalid_argument("box size must have three elements");
  primitive.dimensions = {size[0], size[1], size[2]};
  object.primitives.push_back(primitive);
  geometry_msgs::msg::Pose pose;
  const auto center = box.at("center").get<std::vector<double>>();
  pose.position.x = center.at(0);
  pose.position.y = center.at(1);
  pose.position.z = center.at(2);
  pose.orientation.w = 1.0;
  object.primitive_poses.push_back(pose);
  if (!scene->processCollisionObjectMsg(object))
    throw std::runtime_error("cannot add " + name);
}

std::string collisionReason(const planning_scene::PlanningScenePtr& scene,
  const moveit::core::RobotState& state)
{
  collision_detection::CollisionRequest request;
  request.contacts = true;
  request.max_contacts = 12;
  request.max_contacts_per_pair = 1;
  collision_detection::CollisionResult result;
  scene->checkCollision(request, result, state);
  std::string reason;
  for (const auto& [pair, contacts] : result.contacts) {
    if (!reason.empty()) reason += ',';
    reason += pair.first + "<->" + pair.second;
  }
  return reason;
}

class WristEdgeValidator : public ob::MotionValidator
{
public:
  using Lift = std::function<std::shared_ptr<moveit::core::RobotState>(const ob::State*)>;

  WristEdgeValidator(const ob::SpaceInformationPtr& information, Lift lift,
    planning_scene::PlanningScenePtr scene, size_t& checked_edges,
    size_t& jump_rejections, size_t& bridge_checks, size_t& bridge_successes,
    size_t& coarse_rejections, size_t& near_refinements,
    double& bridge_collision_ms, double& clearance_ms,
    double fifth_step = .01, bool allow_wrist_bridge = true,
    std::array<double, 4> joint_scales = {1.0, 1.0, 1.0, 1.0})
  : ob::MotionValidator(information.get()), lift_(std::move(lift)),
    scene_(std::move(scene)), checked_edges_(checked_edges),
    jump_rejections_(jump_rejections), bridge_checks_(bridge_checks),
    bridge_successes_(bridge_successes), bridge_collision_ms_(bridge_collision_ms)
    , coarse_rejections_(coarse_rejections), near_refinements_(near_refinements),
    clearance_ms_(clearance_ms), fifth_step_(fifth_step),
    allow_wrist_bridge_(allow_wrist_bridge), joint_scales_(joint_scales)
  {
  }

  bool checkMotion(const ob::State* from, const ob::State* to) const override
  {
    return check(from, to, nullptr);
  }

  bool checkMotion(const ob::State* from, const ob::State* to,
    std::pair<ob::State*, double>& last_valid) const override
  {
    return check(from, to, &last_valid);
  }

private:
  bool check(const ob::State* from, const ob::State* to,
    std::pair<ob::State*, double>* last_valid) const
  {
    ++checked_edges_;
    auto previous = lift_(from);
    if (!previous) return false;
    const auto* start = from->as<ob::RealVectorStateSpace::StateType>()->values;
    const auto* goal = to->as<ob::RealVectorStateSpace::StateType>()->values;
    double max_delta = 0.0;
    for (size_t joint = 0; joint < (si_->getStateSpace()->getDimension() == 8 ? 8U : 4U); ++joint)
      max_delta = std::max(max_delta,
        std::abs(goal[joint] - start[joint]) *
          (joint < 4 ? std::abs(joint_scales_[joint]) : 1.0));
    if (si_->getStateSpace()->getDimension() == 5)
      max_delta = std::max(max_delta,
        std::abs(goal[4] - start[4]) * kPi / (180.0 * fifth_step_));
    const int steps = std::max(1, static_cast<int>(
      std::ceil(max_delta / (1.0 * kPi / 180.0))));
    auto* probe = si_->getStateSpace()->allocState();
    const int coarse_steps = std::max(1, static_cast<int>(
      std::ceil(max_delta / (4.0 * kPi / 180.0))));
    for (int coarse_step = 1; coarse_step <= coarse_steps; ++coarse_step) {
      si_->getStateSpace()->interpolate(from, to,
        static_cast<double>(coarse_step) / coarse_steps, probe);
      if (!lift_(probe)) {
        ++coarse_rejections_;
        if (last_valid) {
          last_valid->second = 0.0;
          if (last_valid->first)
            si_->getStateSpace()->interpolate(from, to, 0.0, last_valid->first);
        }
        si_->getStateSpace()->freeState(probe);
        return false;
      }
    }
    for (int step = 1; step <= steps; ++step) {
      const double progress = static_cast<double>(step) / steps;
      si_->getStateSpace()->interpolate(from, to, progress, probe);
      auto next = lift_(probe);
      bool valid = static_cast<bool>(next);
      if (valid) {
        bool wrist_jump = false;
        for (const std::string side : {"left", "right"})
          for (int joint = 1; joint <= 7; ++joint) {
            const auto name = side + "_joint" + std::to_string(joint);
            if (std::abs(next->getVariablePosition(name) -
                previous->getVariablePosition(name)) > 10.0 * kPi / 180.0) {
              wrist_jump = true;
            }
          }
        if (wrist_jump) {
          ++bridge_checks_;
          valid = allow_wrist_bridge_ && validateWristBridge(*previous, *next);
          if (valid) ++bridge_successes_;
          else ++jump_rejections_;
        }
      }
      if (!valid) {
        if (last_valid) {
          last_valid->second = static_cast<double>(step - 1) / steps;
          if (last_valid->first)
            si_->getStateSpace()->interpolate(from, to, last_valid->second, last_valid->first);
        }
        si_->getStateSpace()->freeState(probe);
        return false;
      }
      previous = std::move(next);
    }
    si_->getStateSpace()->freeState(probe);
    return true;
  }

  bool validateWristBridge(const moveit::core::RobotState& from,
    const moveit::core::RobotState& to) const
  {
    moveit::core::RobotState intermediate(from);
    double max_delta = 0.0;
    for (const std::string side : {"left", "right"})
      for (int joint = 1; joint <= 7; ++joint) {
        const auto name = side + "_joint" + std::to_string(joint);
        max_delta = std::max(max_delta,
          std::abs(to.getVariablePosition(name) - from.getVariablePosition(name)));
      }
    const auto check_sample = [&](double alpha, double* clearance) {
        for (const std::string side : {"left", "right"})
          for (int joint = 1; joint <= 7; ++joint) {
            const auto name = side + "_joint" + std::to_string(joint);
            intermediate.setVariablePosition(name,
              from.getVariablePosition(name) * (1.0 - alpha) +
              to.getVariablePosition(name) * alpha);
          }
        intermediate.update(true);
        if (!intermediate.satisfiesBounds()) return false;
        const auto started = std::chrono::steady_clock::now();
        const bool collision = scene_->isStateColliding(intermediate);
        bridge_collision_ms_ += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count();
        if (collision) return false;
        if (clearance) {
          const auto distance_started = std::chrono::steady_clock::now();
          *clearance = scene_->distanceToCollision(intermediate);
          clearance_ms_ += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - distance_started).count();
        }
        return true;
    };
    const int coarse = std::max(1, static_cast<int>(
      std::ceil(max_delta / (4.0 * kPi / 180.0))));
    double previous_clearance = scene_->distanceToCollision(from);
    for (int segment = 1; segment <= coarse; ++segment) {
      const double start = static_cast<double>(segment - 1) / coarse;
      const double end = static_cast<double>(segment) / coarse;
      double clearance = previous_clearance;
      if (segment < coarse && !check_sample(end, &clearance)) return false;
      if (segment == coarse) clearance = scene_->distanceToCollision(to);
      if (std::min(previous_clearance, clearance) < 0.15) {
        ++near_refinements_;
        for (int inner = 1; inner < 4; ++inner)
          if (!check_sample(start + (end - start) * inner / 4, nullptr)) return false;
      }
      previous_clearance = clearance;
    }
    return true;
  }

  Lift lift_;
  planning_scene::PlanningScenePtr scene_;
  size_t& checked_edges_;
  size_t& jump_rejections_;
  size_t& bridge_checks_;
  size_t& bridge_successes_;
  double& bridge_collision_ms_;
  size_t& coarse_rejections_;
  size_t& near_refinements_;
  double& clearance_ms_;
  double fifth_step_;
  bool allow_wrist_bridge_;
  std::array<double, 4> joint_scales_;
};

class Probe
{
public:
  Probe(const rclcpp::Node::SharedPtr& node, const moveit::core::RobotModelPtr& model)
  : model_(model), home_(model), left_(Model::V311Left), right_(Model::V311Right)
  {
    const Json snapshot = Json::parse(node->get_parameter("scene_json").as_string());
    if (snapshot.at("kind") != "v3_initial_transition_direct_shortcut_diagnostic" ||
        snapshot.at("moving_joints") != "both_arms_j1_to_j4")
      throw std::invalid_argument("expected V3 first-four scene snapshot");
    names_ = snapshot.at("joint_names").get<std::vector<std::string>>();
    const auto values = snapshot.at("home_joints").get<std::vector<double>>();
    if (names_.size() != 17 || values.size() != names_.size())
      throw std::invalid_argument("invalid home state");
    goal_values_ = snapshot.value("goal_joints", values);
    if (goal_values_.size() != names_.size())
      throw std::invalid_argument("invalid goal state");
    home_.setToDefaultValues();
    for (size_t index = 0; index < names_.size(); ++index)
      home_.setVariablePosition(names_[index], values[index]);
    home_.update(true);
    scene_ = std::make_shared<planning_scene::PlanningScene>(model_);
    scene_->setCurrentState(home_);
    benchmark_environment_ = snapshot.at("environment").at("boxes");
    wall_boxes_ = snapshot.at("wall_boxes");
    for (const auto& box : snapshot.at("environment").at("boxes"))
      addBox(scene_, "environment_" + box.at("id").get<std::string>(), box);
    for (const auto& box : snapshot.at("wall_boxes"))
      addBox(scene_, "wall_box_" + std::to_string(box.at("box_id").get<int>()), box);
    for (size_t side = 0; side < 2; ++side) {
      const std::string prefix = side == 0 ? "left" : "right";
      const Eigen::Isometry3d carriage = home_.getGlobalLinkTransform("arm_carriage");
      const Eigen::Isometry3d tool = home_.getGlobalLinkTransform(prefix + "_tool0");
      target_rotations_[side] = carriage.linear().transpose() * tool.linear();
      for (size_t joint = 0; joint < 3; ++joint)
        previous_wrist_[side][joint] = home_.getVariablePosition(
          prefix + "_joint" + std::to_string(joint + 5));
    }
    if (!collisionReason(scene_, home_).empty())
      throw std::runtime_error("home pose starts in collision");
  }

  Json evaluate(const Json& request)
  {
    const std::string side = request.at("side").get<std::string>();
    if (side != "left" && side != "right")
      throw std::invalid_argument("side must be left or right");
    const auto degrees = request.at("first_four_deg").get<std::vector<double>>();
    if (degrees.size() != 4)
      throw std::invalid_argument("first_four_deg requires four values");
    const size_t index = side == "left" ? 0 : 1;
    const Solver& solver = index == 0 ? left_ : right_;
    const auto bounds = model_->getJointModelGroup(side + "_arm");
    std::array<double, 4> prefix{};
    moveit::core::RobotState state(home_);
    for (size_t joint = 0; joint < prefix.size(); ++joint) {
      if (!std::isfinite(degrees[joint]))
        throw std::invalid_argument("non-finite joint angle");
      prefix[joint] = degrees[joint] * kPi / 180.0;
      state.setVariablePosition(side + "_joint" + std::to_string(joint + 1), prefix[joint]);
    }
    state.update(true);
    const auto started = std::chrono::steady_clock::now();
    const auto solutions = solver.solveWristOrientation(
      prefix, target_rotations_[index], previous_wrist_[index]);
    const double solve_us = std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now() - started).count();
    Json answer = {{"side", side}, {"requested_first_four_deg", degrees},
      {"wrist_solve_us", solve_us}, {"wrist_branches", solutions.size()}};
    if (solutions.empty()) {
      answer["status"] = "no_wrist_solution";
      answer["preview_only"] = true;
    } else {
      const auto& solution = solutions.front();
      for (size_t joint = 0; joint < 3; ++joint)
        state.setVariablePosition(side + "_joint" + std::to_string(joint + 5),
          solution.wrist[joint]);
      state.update(true);
      previous_wrist_[index] = solution.wrist;
      const Eigen::Matrix3d actual =
        home_.getGlobalLinkTransform("arm_carriage").linear().transpose() *
        state.getGlobalLinkTransform(side + "_tool0").linear();
      answer["orientation_error_rad"] = std::abs(Eigen::AngleAxisd(
        target_rotations_[index].transpose() * actual).angle());
      const std::string collision = collisionReason(scene_, state);
      answer["collision_reason"] = collision;
      answer["status"] = collision.empty() ? "ok" : "collision";
      answer["preview_only"] = false;
    }
    answer["within_joint_limits"] = state.satisfiesBounds(bounds);
    std::vector<double> joints;
    for (const auto& name : names_)
      joints.push_back(state.getVariablePosition(name));
    answer["joints"] = std::move(joints);
    return answer;
  }

  void configureManualLoaded(const Json& context)
  {
    if (context.at("initial_pose") != "home" || context.at("wall_context") != "target_only")
      throw std::invalid_argument("expected home direct-attach target_only context");
    manual_scene_ = std::make_shared<planning_scene::PlanningScene>(model_);
    for (const auto& box : context.at("environment").at("boxes"))
      addBox(manual_scene_, box.at("id").get<std::string>(), box);
    manual_start_ = std::make_unique<moveit::core::RobotState>(home_);
    manual_boxes_.clear();
    const auto size = context.at("box_size").get<std::vector<double>>();
    if (size.size() != 3 || context.at("direct_attached_boxes").size() != 2)
      throw std::invalid_argument("expected two 3D attached boxes");
    const std::vector<shapes::ShapeConstPtr> shapes{
      std::make_shared<shapes::Box>(size[0], size[1], size[2])};
    for (const auto& item : context.at("direct_attached_boxes")) {
      const std::string side = item.at("side").get<std::string>();
      if (side != "left" && side != "right") throw std::invalid_argument("invalid attached side");
      Eigen::Isometry3d offset = Eigen::Isometry3d::Identity();
      const auto center = item.at("tool_to_box_center").get<std::vector<double>>();
      const auto rotation = item.at("tool_to_box_rotation");
      offset.translation() = Eigen::Vector3d(center.at(0), center.at(1), center.at(2));
      for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
          offset.linear()(row, col) = rotation.at(row).at(col).get<double>();
      const std::string tool = side + "_tool0";
      const std::vector<std::string> touch{tool, side + "_joint7"};
      manual_start_->attachBody("carried_" + side, Eigen::Isometry3d::Identity(),
        shapes, EigenSTL::vector_Isometry3d{offset}, touch, tool);
      manual_boxes_.push_back({side, offset});
    }
    manual_start_->update(true);
    manual_goal_ = std::make_unique<moveit::core::RobotState>(*manual_start_);
    if (!manual_goal_->setToDefaultValues(
      model_->getJointModelGroup("whole_body"), "unloading"))
      throw std::runtime_error("missing unloading named pose");
    manual_goal_->update(true);
  }

  Json evaluateManualLoaded(const Json& request)
  {
    if (!manual_start_ || !manual_goal_) throw std::runtime_error("manual context not initialized");
    const auto progress = request.at("progress_percent").get<std::vector<double>>();
    if (progress.size() != 7 ||
        std::any_of(progress.begin(), progress.end(), [](double value) {
          return !std::isfinite(value) || value < 0.0 || value > 100.0;
        }))
      throw std::invalid_argument("seven progress values in [0,100] required");
    const auto started = std::chrono::steady_clock::now();
    moveit::core::RobotState state(*manual_start_);
    for (const std::string side : {"left", "right"})
      for (int joint = 1; joint <= 7; ++joint) {
        const auto name = side + "_joint" + std::to_string(joint);
        const double fraction = progress[joint - 1] / 100.0;
        state.setVariablePosition(name,
          manual_start_->getVariablePosition(name) * (1.0 - fraction) +
          manual_goal_->getVariablePosition(name) * fraction);
      }
    state.update(true);
    Json joints = Json::array();
    for (const auto& name : names_) joints.push_back(state.getVariablePosition(name));
    Json boxes = Json::array();
    for (const auto& [side, offset] : manual_boxes_) {
      const Eigen::Isometry3d transform = state.getGlobalLinkTransform(side + "_tool0") * offset;
      Json rotation = Json::array();
      for (int row = 0; row < 3; ++row)
        rotation.push_back({transform.linear()(row, 0), transform.linear()(row, 1),
          transform.linear()(row, 2)});
      boxes.push_back({{"side", side},
        {"center", {transform.translation().x(), transform.translation().y(),
          transform.translation().z()}}, {"rotation", rotation}});
    }
    const bool bounded = state.satisfiesBounds();
    const std::string collision = collisionReason(manual_scene_, state);
    return {{"progress_percent", progress}, {"joint_names", names_}, {"joints", joints},
      {"within_joint_limits", bounded}, {"collision_pairs", collision},
      {"status", !bounded ? "joint_limit" : collision.empty() ? "ok" : "collision"},
      {"boxes", boxes}, {"wall_ms", std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count()}};
  }

  Json inspectSharedHeightFaceIk(const Json& request)
  {
    const auto started = std::chrono::steady_clock::now();
    const auto face_world = [&](const Json& pose) {
      Eigen::Isometry3d local = Eigen::Isometry3d::Identity();
      local.translation() = Eigen::Vector3d(
        pose.at("x").get<double>(), pose.at("y").get<double>(), pose.at("z").get<double>());
      local.linear() = (Eigen::AngleAxisd(pose.at("yaw").get<double>(), Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pose.at("pitch").get<double>(), Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(pose.at("roll").get<double>(), Eigen::Vector3d::UnitX())).toRotationMatrix();
      return home_.getGlobalLinkTransform("base_link") * local;
    };
    const Eigen::Isometry3d left_face = face_world(request.at("left"));
    const bool has_right = request.contains("right") && !request.at("right").is_null();
    const Eigen::Isometry3d right_face = has_right ?
      face_world(request.at("right")) : Eigen::Isometry3d::Identity();
    Json output = {{"kind", "v3_front_face_shared_height_ik_probe"},
      {"success", false}, {"left_pose_in_base_link", request.at("left")},
      {"right_pose_in_base_link", has_right ? request.at("right") : Json(nullptr)},
      {"height_step_m", .04}, {"psi_step_deg", 10.0}, {"initials", Json::array()}};
    const auto* whole = model_->getJointModelGroup("whole_body");
    const auto& updown_bounds = model_->getVariableBounds("updown");
    for (const std::string home : {"home", "second_home"}) {
      moveit::core::RobotState named(home_);
      if (!named.setToDefaultValues(whole, home))
        throw std::runtime_error("missing named home state");
      named.update(true);
      Json variant = {{"home", home}, {"shared_heights", Json::array()},
        {"left_contact_solutions", 0}, {"right_contact_solutions", 0},
        {"left_pregrasp_solutions", 0}, {"right_pregrasp_solutions", 0},
        {"analytic_calls", 0}};
      for (int height_index = 0; height_index <= 25; ++height_index) {
        const double height = std::min(updown_bounds.max_position_,
          updown_bounds.min_position_ + .04 * height_index);
        moveit::core::RobotState height_state(named);
        height_state.setVariablePosition("updown", height);
        height_state.update(true);
        Json best_pair = {{"height_m", height}, {"home", home}};
        bool pair_has_ik = true;
        for (size_t side_index = 0; side_index < (has_right ? 2U : 1U); ++side_index) {
          const std::string side = side_index == 0 ? "left" : "right";
          const auto& solver = side_index == 0 ? left_ : right_;
          const Eigen::Isometry3d& face = side_index == 0 ? left_face : right_face;
          Eigen::Isometry3d contact = alfa_robot::motion::toolPoseFromFrontFace(
            face, named.getGlobalLinkTransform(side + "_tool0").linear());
          contact.translation() -= 1e-6 * face.linear().col(2);
          Eigen::Isometry3d pregrasp = contact;
          pregrasp.translation() -= .05 * face.linear().col(2);
          const Eigen::Isometry3d arm_inverse =
            height_state.getGlobalLinkTransform("arm_carriage").inverse();
          std::array<double, 7> seed{};
          for (int joint = 0; joint < 7; ++joint)
            seed[joint] = named.getVariablePosition(side + "_joint" + std::to_string(joint + 1));
          double best_cost = std::numeric_limits<double>::infinity();
          Json best = nullptr;
          int local_contacts = 0;
          for (int psi_index = 0; psi_index < 36; ++psi_index) {
            alfa_robot::analytic_ik::V3RedundantIkRequest query;
            query.seed = seed;
            query.swivel_angle = -kPi + psi_index * 10.0 * kPi / 180.0;
            query.target_in_arm_base = arm_inverse * contact;
            const auto contacts = solver.solveInArmBase(query);
            variant["analytic_calls"] = variant["analytic_calls"].get<int>() + 1;
            local_contacts += contacts.size();
            if (contacts.empty()) continue;
            query.target_in_arm_base = arm_inverse * pregrasp;
            const auto pregrasp_solutions = solver.solveInArmBase(query);
            variant["analytic_calls"] = variant["analytic_calls"].get<int>() + 1;
            variant[side + "_pregrasp_solutions"] =
              variant[side + "_pregrasp_solutions"].get<int>() + pregrasp_solutions.size();
            for (const auto& before : pregrasp_solutions)
              for (const auto& after : contacts) {
                if (before.shoulder_branch != after.shoulder_branch ||
                    before.elbow_branch != after.elbow_branch ||
                    before.wrist_branch != after.wrist_branch) continue;
                double cost = 0.0;
                Json joints = Json::array();
                for (int joint = 0; joint < 7; ++joint) {
                  cost += std::pow(before.joints[joint] - seed[joint], 2);
                  joints.push_back(before.joints[joint]);
                }
                if (cost < best_cost) {
                  best_cost = cost;
                  best = {{"psi_deg", query.swivel_angle * 180.0 / kPi},
                    {"cost", cost}, {"pregrasp_joints", joints}};
                }
              }
          }
          variant[side + "_contact_solutions"] =
            variant[side + "_contact_solutions"].get<int>() + local_contacts;
          if (best.is_null()) pair_has_ik = false;
          else best_pair[side] = best;
        }
        if (pair_has_ik) {
          best_pair["cost"] = best_pair["left"]["cost"].get<double>() +
            (has_right ? best_pair["right"]["cost"].get<double>() : 0.0);
          moveit::core::RobotState combined(height_state);
          for (size_t side_index = 0; side_index < (has_right ? 2U : 1U); ++side_index) {
            const std::string side = side_index == 0 ? "left" : "right";
            const auto& joints = best_pair.at(side).at("pregrasp_joints");
            for (int joint = 0; joint < 7; ++joint)
              combined.setVariablePosition(side + "_joint" + std::to_string(joint + 1),
                joints.at(joint).get<double>());
          }
          combined.update(true);
          best_pair["pregrasp_collision"] = collisionReason(scene_, combined);
          best_pair["pregrasp_within_bounds"] = combined.satisfiesBounds();
          variant["shared_heights"].push_back(best_pair);
        }
      }
      output["initials"].push_back(std::move(variant));
    }
    output["success"] = std::any_of(output["initials"].begin(), output["initials"].end(),
      [](const Json& item) {return !item.at("shared_heights").empty();});
    output["wall_ms"] = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return output;
  }

  Json planBestSharedHeightPregrasp(const Json& candidates, double budget_s)
  {
    if (candidates.at("kind") != "v3_front_face_shared_height_ik_probe" ||
        !candidates.at("success").get<bool>() || !std::isfinite(budget_s) || budget_s <= 0.0)
      throw std::invalid_argument("expected shared-height IK result and positive budget");
    const Json* best = nullptr;
    for (const auto& variant : candidates.at("initials"))
      for (const auto& pair : variant.at("shared_heights"))
        if (!best || pair.at("cost").get<double>() < best->at("cost").get<double>())
          best = &pair;
    if (!best) throw std::runtime_error("no shared-height pair");
    const auto started = std::chrono::steady_clock::now();
    moveit::core::RobotState initial(home_);
    if (!initial.setToDefaultValues(model_->getJointModelGroup("whole_body"),
        best->at("home").get<std::string>()))
      throw std::runtime_error("unknown initial named pose");
    initial.update(true);
    moveit::core::RobotState target(initial);
    target.setVariablePosition("updown", best->at("height_m").get<double>());
    for (const std::string side : {"left", "right"})
      for (int joint = 1; joint <= 7; ++joint)
        target.setVariablePosition(side + "_joint" + std::to_string(joint),
          best->at(side).at("pregrasp_joints").at(joint - 1).get<double>());
    target.update(true);
    Json result = {{"kind", "v3_dual_front_face_best_pair_pregrasp_plan"},
      {"success", false}, {"stage", "pregrasp"}, {"initial_pose", best->at("home")},
      {"height_m", best->at("height_m")}, {"pair_cost", best->at("cost")},
      {"budget_s", budget_s}, {"candidate_scope", "one global lowest-cost pair only"},
      {"initial_collision", collisionReason(scene_, initial)},
      {"pregrasp_collision", collisionReason(scene_, target)}, {"frames", Json::array()}};
    if (!initial.satisfiesBounds() || !target.satisfiesBounds() ||
        !result["initial_collision"].get<std::string>().empty() ||
        !result["pregrasp_collision"].get<std::string>().empty()) {
      result["failure_stage"] = "invalid_endpoint";
      return result;
    }
    auto space = std::make_shared<ob::RealVectorStateSpace>(15);
    ob::RealVectorBounds bounds(15);
    for (size_t axis = 0; axis < 15; ++axis) {
      const auto& limit = model_->getVariableBounds(names_[axis]);
      bounds.setLow(axis, limit.min_position_);
      bounds.setHigh(axis, limit.max_position_);
    }
    space->setBounds(bounds);
    og::SimpleSetup setup(space);
    size_t checks = 0;
    const auto toRobot = [&](const ob::State* state) {
      moveit::core::RobotState robot(initial);
      const auto* q = state->as<ob::RealVectorStateSpace::StateType>()->values;
      for (size_t axis = 0; axis < 15; ++axis) robot.setVariablePosition(names_[axis], q[axis]);
      robot.update(true);
      return robot;
    };
    setup.setStateValidityChecker([&](const ob::State* state) {
      auto robot = toRobot(state);
      ++checks;
      return robot.satisfiesBounds() && !scene_->isStateColliding(robot);
    });
    setup.getSpaceInformation()->setStateValidityCheckingResolution(.0002);
    ob::ScopedState<> from(space), to(space);
    for (size_t axis = 0; axis < 15; ++axis) {
      from[axis] = initial.getVariablePosition(names_[axis]);
      to[axis] = target.getVariablePosition(names_[axis]);
    }
    setup.setStartAndGoalStates(from, to, 1e-9);
    setup.setup();
    const auto shortcut_started = std::chrono::steady_clock::now();
    result["shortcut_valid"] = setup.getSpaceInformation()->checkMotion(from.get(), to.get());
    result["shortcut_ms"] = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - shortcut_started).count();
    if (!result["shortcut_valid"].get<bool>()) {
      ompl::RNG::setSeed(1);
      auto planner = std::make_shared<og::RRTConnect>(setup.getSpaceInformation());
      planner->setRange(.25);
      setup.setPlanner(planner);
      const auto deadline = started + std::chrono::duration<double>(budget_s);
      const ob::PlannerTerminationCondition stop([&]() {
        return std::chrono::steady_clock::now() >= deadline;
      });
      const auto rrt_started = std::chrono::steady_clock::now();
      result["ompl_exact"] = setup.solve(stop) == ob::PlannerStatus::EXACT_SOLUTION;
      result["rrt_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - rrt_started).count();
    } else {
      result["ompl_exact"] = true;
      result["rrt_ms"] = 0.0;
    }
    if (result["ompl_exact"].get<bool>()) {
      const auto validation_started = std::chrono::steady_clock::now();
      const auto* path = result["shortcut_valid"].get<bool>() ? nullptr : &setup.getSolutionPath();
      const size_t segments = path ? path->getStateCount() - 1 : 1;
      bool valid = true;
      for (size_t segment = 0; segment < segments && valid; ++segment) {
        const auto* a = path ? path->getState(segment) : from.get();
        const auto* b = path ? path->getState(segment + 1) : to.get();
        const auto* qa = a->as<ob::RealVectorStateSpace::StateType>()->values;
        const auto* qb = b->as<ob::RealVectorStateSpace::StateType>()->values;
        double max_steps = std::abs(qb[14] - qa[14]) / .005;
        for (size_t axis = 0; axis < 14; ++axis)
          max_steps = std::max(max_steps,
            std::abs(qb[axis] - qa[axis]) / (0.5 * kPi / 180.0));
        const int steps = std::max(1, static_cast<int>(std::ceil(max_steps)));
        auto* sample = space->allocState();
        for (int step = segment == 0 ? 0 : 1; step <= steps; ++step) {
          space->interpolate(a, b, static_cast<double>(step) / steps, sample);
          auto robot = toRobot(sample);
          ++checks;
          if (!robot.satisfiesBounds() || scene_->isStateColliding(robot)) {
            valid = false;
            result["failure_stage"] = "post_validation_collision";
            break;
          }
          Json joints = Json::array();
          for (const auto& name : names_) joints.push_back(robot.getVariablePosition(name));
          result["frames"].push_back({{"stage", "pregrasp_transfer"}, {"joints", joints}});
        }
        space->freeState(sample);
      }
      result["success"] = valid;
      result["validation_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - validation_started).count();
    } else {
      result["failure_stage"] = "rrt_timeout_or_no_path";
    }
    result["collision_checks"] = checks;
    result["wall_ms"] = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return result;
  }

  Json approachSharedHeightPair(const Json& candidates, const Json& pregrasp_plan)
  {
    if (!pregrasp_plan.at("success").get<bool>() || pregrasp_plan.at("frames").empty())
      throw std::invalid_argument("pregrasp transfer must be verified before Cartesian approach");
    const Json* best = nullptr;
    for (const auto& variant : candidates.at("initials"))
      for (const auto& pair : variant.at("shared_heights"))
        if (!best || pair.at("cost").get<double>() < best->at("cost").get<double>())
          best = &pair;
    if (!best || best->at("home") != pregrasp_plan.at("initial_pose") ||
        std::abs(best->at("height_m").get<double>() -
          pregrasp_plan.at("height_m").get<double>()) > 1e-9)
      throw std::runtime_error("pregrasp candidate identity changed");
    const auto face_world = [&](const Json& pose) {
      Eigen::Isometry3d local = Eigen::Isometry3d::Identity();
      local.translation() = Eigen::Vector3d(
        pose.at("x").get<double>(), pose.at("y").get<double>(), pose.at("z").get<double>());
      local.linear() = (Eigen::AngleAxisd(pose.at("yaw").get<double>(), Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pose.at("pitch").get<double>(), Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(pose.at("roll").get<double>(), Eigen::Vector3d::UnitX())).toRotationMatrix();
      return home_.getGlobalLinkTransform("base_link") * local;
    };
    std::array<Eigen::Isometry3d, 2> face{
      face_world(candidates.at("left_pose_in_base_link")),
      face_world(candidates.at("right_pose_in_base_link"))};
    moveit::core::RobotState named(home_);
    if (!named.setToDefaultValues(model_->getJointModelGroup("whole_body"),
        best->at("home").get<std::string>()))
      throw std::runtime_error("unknown initial pose for tool orientation");
    named.update(true);
    const std::array<Eigen::Isometry3d, 2> contact_tools{
      alfa_robot::motion::toolPoseFromFrontFace(
        face[0], named.getGlobalLinkTransform("left_tool0").linear()),
      alfa_robot::motion::toolPoseFromFrontFace(
        face[1], named.getGlobalLinkTransform("right_tool0").linear())};
    auto contact_scene = planning_scene::PlanningScene::clone(scene_);
    for (size_t side_index = 0; side_index < 2; ++side_index) {
      const std::string side = side_index == 0 ? "left" : "right";
      const Eigen::Vector3d target_center = face[side_index].translation() +
        .15 * face[side_index].linear().col(2);
      double minimum = std::numeric_limits<double>::infinity();
      int target_id = -1;
      for (const auto& box : wall_boxes_) {
        const auto center = box.at("center").get<std::vector<double>>();
        const double distance = (target_center - Eigen::Vector3d(center[0], center[1], center[2])).norm();
        if (distance < minimum) {
          minimum = distance;
          target_id = box.at("box_id").get<int>();
        }
      }
      if (minimum > .01 || target_id < 0)
        throw std::invalid_argument("front-face pose does not match a scene box");
      const std::string object = "wall_box_" + std::to_string(target_id);
      contact_scene->getAllowedCollisionMatrixNonConst().setEntry(object, side + "_tool0", true);
      contact_scene->getAllowedCollisionMatrixNonConst().setEntry(object, side + "_link7", true);
    }
    moveit::core::RobotState state(home_);
    const auto& pregrasp_joints = pregrasp_plan.at("frames").back().at("joints");
    for (size_t axis = 0; axis < names_.size(); ++axis)
      state.setVariablePosition(names_[axis], pregrasp_joints.at(axis).get<double>());
    state.update(true);
    Json result = {{"kind", "v3_two_front_faces_shared_height_cartesian_approach"},
      {"success", false}, {"height_m", state.getVariablePosition("updown")},
      {"stage", "approach_5cm"}, {"frames", Json::array()},
      {"start_collision", collisionReason(scene_, state)}};
    if (!result["start_collision"].get<std::string>().empty()) {
      result["failure_stage"] = "pregrasp_start_collision";
      return result;
    }
    for (int step = 0; step <= 5; ++step) {
      if (step > 0) {
        const double distance = .05 - .01 * step;
        std::array<std::vector<alfa_robot::analytic_ik::V3RedundantIkSolution>, 2> solutions;
        for (size_t side_index = 0; side_index < 2; ++side_index) {
          const std::string side = side_index == 0 ? "left" : "right";
          auto target = contact_tools[side_index];
          target.translation() -= (distance + 1e-6) * target.linear().col(2);
          alfa_robot::analytic_ik::V3RedundantIkRequest request;
          request.target_in_arm_base =
            state.getGlobalLinkTransform("arm_carriage").inverse() * target;
          request.swivel_angle = best->at(side).at("psi_deg").get<double>() * kPi / 180.0;
          for (int joint = 0; joint < 7; ++joint)
            request.seed[joint] = state.getVariablePosition(side + "_joint" + std::to_string(joint + 1));
          solutions[side_index] = (side_index == 0 ? left_ : right_).solveInArmBase(request);
        }
        bool found = false;
        std::string reason = solutions[0].empty() || solutions[1].empty() ?
          "analytic_ik_no_solution" : "joint_jump_or_collision";
        double minimum_cost = std::numeric_limits<double>::infinity();
        moveit::core::RobotState selected(state);
        for (const auto& left : solutions[0])
          for (const auto& right : solutions[1]) {
            moveit::core::RobotState candidate(state);
            bool jump = false;
            double cost = 0.0;
            for (size_t side_index = 0; side_index < 2; ++side_index) {
              const std::string side = side_index == 0 ? "left" : "right";
              const auto& joint_solution = side_index == 0 ? left.joints : right.joints;
              for (int joint = 0; joint < 7; ++joint) {
                const auto name = side + "_joint" + std::to_string(joint + 1);
                const double delta = joint_solution[joint] - state.getVariablePosition(name);
                if (std::abs(delta) > 10.0 * kPi / 180.0) jump = true;
                cost += delta * delta;
                candidate.setVariablePosition(name, joint_solution[joint]);
              }
            }
            if (jump || cost >= minimum_cost) continue;
            candidate.update(true);
            if (!candidate.satisfiesBounds()) {
              reason = "joint_limit";
              continue;
            }
            const std::string collision = collisionReason(contact_scene, candidate);
            if (!collision.empty()) {
              reason = collision;
              continue;
            }
            minimum_cost = cost;
            selected = std::move(candidate);
            found = true;
          }
        if (!found) {
          result["failure_stage"] = "cartesian_approach";
          result["failure_step"] = step;
          result["failure_reason"] = reason;
          break;
        }
        state = std::move(selected);
      }
      Json joints = Json::array();
      for (const auto& name : names_) joints.push_back(state.getVariablePosition(name));
      result["frames"].push_back({{"stage", "approach_5cm"},
        {"distance_to_contact_m", .05 - .01 * step}, {"joints", joints}});
    }
    result["success"] = result["frames"].size() == 6;
    return result;
  }

  Json retreatAttachedSharedHeightPair(const Json& candidates, const Json& pregrasp_plan,
    const Json& approach, const std::string& placement_reference_path = "",
    int unloaded_retract_steps = 5)
  {
    if (unloaded_retract_steps < 1 || unloaded_retract_steps > 20)
      throw std::invalid_argument("unloaded retract must be 1..20 centimetres");
    if (!approach.at("success").get<bool>() || approach.at("frames").size() != 6)
      throw std::invalid_argument("verified contact is required before attaching boxes");
    const Json* best = nullptr;
    for (const auto& variant : candidates.at("initials"))
      for (const auto& pair : variant.at("shared_heights"))
        if (!best || pair.at("cost").get<double>() < best->at("cost").get<double>())
          best = &pair;
    if (!best || best->at("home") != pregrasp_plan.at("initial_pose") ||
        std::abs(best->at("height_m").get<double>() -
          pregrasp_plan.at("height_m").get<double>()) > 1e-9)
      throw std::invalid_argument("contact and shared-height candidate do not match");

    auto loaded_scene = planning_scene::PlanningScene::clone(scene_);
    moveit::core::RobotState state(home_);
    const auto& contact_joints = approach.at("frames").back().at("joints");
    for (size_t axis = 0; axis < names_.size(); ++axis)
      state.setVariablePosition(names_[axis], contact_joints.at(axis).get<double>());
    state.update(true);
    std::array<Eigen::Isometry3d, 2> contact_tools{
      state.getGlobalLinkTransform("left_tool0"),
      state.getGlobalLinkTransform("right_tool0")};
    const auto face_world = [&](const Json& pose) {
      Eigen::Isometry3d local = Eigen::Isometry3d::Identity();
      local.translation() = Eigen::Vector3d(
        pose.at("x").get<double>(), pose.at("y").get<double>(), pose.at("z").get<double>());
      local.linear() = (Eigen::AngleAxisd(pose.at("yaw").get<double>(), Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pose.at("pitch").get<double>(), Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(pose.at("roll").get<double>(), Eigen::Vector3d::UnitX())).toRotationMatrix();
      return home_.getGlobalLinkTransform("base_link") * local;
    };
    const std::array<Eigen::Isometry3d, 2> faces{
      face_world(candidates.at("left_pose_in_base_link")),
      face_world(candidates.at("right_pose_in_base_link"))};
    std::array<Eigen::Isometry3d, 2> attachment_offsets;
    Json result = {{"kind", "v3_two_front_faces_attached_cartesian_retreat"},
      {"success", false}, {"stage", "retreat_35cm"}, {"frames", Json::array()}};
    for (size_t side_index = 0; side_index < 2; ++side_index) {
      const std::string side = side_index == 0 ? "left" : "right";
      const Eigen::Vector3d center = faces[side_index].translation() +
        .15 * faces[side_index].linear().col(2);
      double minimum = std::numeric_limits<double>::infinity();
      const Json* target_box = nullptr;
      for (const auto& box : wall_boxes_) {
        const auto position = box.at("center").get<std::vector<double>>();
        const double distance = (center - Eigen::Vector3d(
          position[0], position[1], position[2])).norm();
        if (distance < minimum) {
          minimum = distance;
          target_box = &box;
        }
      }
      if (!target_box || minimum > .01)
        throw std::invalid_argument("front face does not match attached box geometry");
      const std::string object = "wall_box_" + std::to_string(target_box->at("box_id").get<int>());
      loaded_scene->getWorldNonConst()->removeObject(object);
      const auto size = target_box->at("size").get<std::vector<double>>();
      const Eigen::Isometry3d tool_to_box = contact_tools[side_index].inverse() *
        Eigen::Translation3d(center);
      attachment_offsets[side_index] = tool_to_box;
      std::vector<shapes::ShapeConstPtr> shapes{
        std::make_shared<shapes::Box>(size.at(0), size.at(1), size.at(2))};
      EigenSTL::vector_Isometry3d shape_poses{tool_to_box};
      const std::string tool = side + "_tool0";
      state.attachBody(object, Eigen::Isometry3d::Identity(), shapes, shape_poses,
        std::vector<std::string>{tool, side + "_link7"}, tool);
    }
    state.update(true);
    const std::string initial_collision = collisionReason(loaded_scene, state);
    if (!initial_collision.empty()) {
      result["failure_stage"] = "attached_contact";
      result["failure_reason"] = initial_collision;
      return result;
    }
    for (int step = 0; step <= 35; ++step) {
      if (step > 0) {
        std::array<std::vector<alfa_robot::analytic_ik::V3RedundantIkSolution>, 2> solutions;
        for (size_t side_index = 0; side_index < 2; ++side_index) {
          const std::string side = side_index == 0 ? "left" : "right";
          Eigen::Isometry3d target = contact_tools[side_index];
          target.translation() -= .01 * step * faces[side_index].linear().col(2);
          alfa_robot::analytic_ik::V3RedundantIkRequest request;
          request.target_in_arm_base =
            state.getGlobalLinkTransform("arm_carriage").inverse() * target;
          request.swivel_angle = best->at(side).at("psi_deg").get<double>() * kPi / 180.0;
          for (int joint = 0; joint < 7; ++joint)
            request.seed[joint] = state.getVariablePosition(side + "_joint" + std::to_string(joint + 1));
          solutions[side_index] = (side_index == 0 ? left_ : right_).solveInArmBase(request);
        }
        bool found = false;
        std::string reason = solutions[0].empty() || solutions[1].empty() ?
          "analytic_ik_no_solution" : "joint_jump_or_collision";
        double minimum_cost = std::numeric_limits<double>::infinity();
        moveit::core::RobotState selected(state);
        for (const auto& left : solutions[0])
          for (const auto& right : solutions[1]) {
            moveit::core::RobotState candidate(state);
            bool jump = false;
            double cost = 0.0;
            for (size_t side_index = 0; side_index < 2; ++side_index) {
              const std::string side = side_index == 0 ? "left" : "right";
              const auto& joints = side_index == 0 ? left.joints : right.joints;
              for (int joint = 0; joint < 7; ++joint) {
                const std::string name = side + "_joint" + std::to_string(joint + 1);
                const double delta = joints[joint] - state.getVariablePosition(name);
                jump |= std::abs(delta) > 10.0 * kPi / 180.0;
                cost += delta * delta;
                candidate.setVariablePosition(name, joints[joint]);
              }
            }
            if (jump || cost >= minimum_cost) continue;
            candidate.update(true);
            if (!candidate.satisfiesBounds()) {
              reason = "joint_limit";
              continue;
            }
            const std::string collision = collisionReason(loaded_scene, candidate);
            if (!collision.empty()) {
              reason = collision;
              continue;
            }
            minimum_cost = cost;
            selected = std::move(candidate);
            found = true;
          }
        if (!found) {
          result["failure_stage"] = "loaded_cartesian_retreat";
          result["failure_step"] = step;
          result["failure_reason"] = reason;
          break;
        }
        state = std::move(selected);
      }
      Json joints = Json::array();
      for (const auto& name : names_) joints.push_back(state.getVariablePosition(name));
      result["frames"].push_back({{"stage", "retreat_35cm"},
        {"retreat_m", .01 * step}, {"joints", joints}});
    }
    result["success"] = result["frames"].size() == 36;
    if (result["success"].get<bool>()) {
      moveit::core::RobotState named(home_);
      if (!named.setToDefaultValues(model_->getJointModelGroup("whole_body"),
          "second_home"))
        throw std::runtime_error("missing named second_home");
      named.update(true);
      const std::array<Eigen::Isometry3d, 2> desired{
        named.getGlobalLinkTransform("left_tool0"),
        named.getGlobalLinkTransform("right_tool0")};
      Json diagnostics = Json::array();
      for (int height_index = 0; height_index <= 25; ++height_index) {
        const double height = -1.0 + .04 * height_index;
        moveit::core::RobotState height_state(state);
        height_state.setVariablePosition("updown", height);
        height_state.update(true);
        moveit::core::RobotState combined(height_state);
        Json row = {{"height_m", height}};
        bool both_solved = true;
        for (size_t side_index = 0; side_index < 2; ++side_index) {
          const std::string side = side_index == 0 ? "left" : "right";
          Eigen::Isometry3d target = desired[side_index];
          target.linear() = contact_tools[side_index].linear();
          alfa_robot::analytic_ik::V3RedundantIkRequest query;
          query.target_in_arm_base =
            height_state.getGlobalLinkTransform("arm_carriage").inverse() * target;
          double lowest = std::numeric_limits<double>::infinity();
          std::array<double, 7> selected{};
          for (int joint = 0; joint < 7; ++joint)
            query.seed[joint] = state.getVariablePosition(side + "_joint" + std::to_string(joint + 1));
          for (int psi_index = 0; psi_index < 36; ++psi_index) {
            query.swivel_angle = -kPi + psi_index * 10.0 * kPi / 180.0;
            for (const auto& solution : (side_index == 0 ? left_ : right_).solveInArmBase(query)) {
              double cost = 0.0;
              for (int joint = 0; joint < 7; ++joint) {
                const double delta = solution.joints[joint] -
                  named.getVariablePosition(side + "_joint" + std::to_string(joint + 1));
                cost += delta * delta;
              }
              if (cost < lowest) {
                lowest = cost;
                selected = solution.joints;
              }
            }
          }
          if (!std::isfinite(lowest)) {
            both_solved = false;
            row[side + "_ik"] = false;
            continue;
          }
          row[side + "_ik"] = true;
          row[side + "_joint_cost"] = lowest;
          for (int joint = 0; joint < 7; ++joint)
            combined.setVariablePosition(side + "_joint" + std::to_string(joint + 1), selected[joint]);
        }
        if (!both_solved) {
          diagnostics.push_back(row);
          continue;
        }
        combined.update(true);
        row["joint_cost"] = row["left_joint_cost"].get<double>() +
          row["right_joint_cost"].get<double>();
        row["within_bounds"] = combined.satisfiesBounds();
        row["collision"] = collisionReason(loaded_scene, combined);
        if (row["within_bounds"].get<bool>() && row["collision"].get<std::string>().empty()) {
          Json joints = Json::array();
          for (const auto& name : names_) joints.push_back(combined.getVariablePosition(name));
          row["joints"] = std::move(joints);
        }
        diagnostics.push_back(std::move(row));
      }
      result["return_goal_candidates"] = std::move(diagnostics);
      const Json* goal_row = nullptr;
      for (const auto& row : result["return_goal_candidates"])
        if (row.contains("joints") &&
            std::abs(row.at("height_m").get<double>() - state.getVariablePosition("updown")) < 1e-8) {
          goal_row = &row;
          break;
        }
      Json transfer = {{"kind", "v3_loaded_orientation_preserving_return"},
        {"success", false}, {"frames", Json::array()}, {"budget_s", 2.0}};
      if (!goal_row) {
        transfer["failure_stage"] = "no_collision_free_same_height_goal";
      } else {
        moveit::core::RobotState goal(state);
        for (size_t axis = 0; axis < names_.size(); ++axis)
          goal.setVariablePosition(names_[axis], goal_row->at("joints").at(axis).get<double>());
        goal.update(true);
        const Eigen::Matrix3d carriage_rotation =
          state.getGlobalLinkTransform("arm_carriage").linear();
        const std::array<Eigen::Matrix3d, 2> tool_rotations{
          carriage_rotation.transpose() * state.getGlobalLinkTransform("left_tool0").linear(),
          carriage_rotation.transpose() * state.getGlobalLinkTransform("right_tool0").linear()};
        std::array<std::string, 8> axes;
        for (size_t index = 0; index < 8; ++index)
          axes[index] = std::string(index < 4 ? "left_joint" : "right_joint") +
            std::to_string(index % 4 + 1);
        auto space = std::make_shared<ob::RealVectorStateSpace>(8);
        ob::RealVectorBounds bounds(8);
        for (size_t axis = 0; axis < axes.size(); ++axis) {
          const auto& limit = model_->getVariableBounds(axes[axis]);
          bounds.setLow(axis, limit.min_position_);
          bounds.setHigh(axis, limit.max_position_);
        }
        space->setBounds(bounds);
        using Key = std::array<long long, 8>;
        struct KeyHash {
          size_t operator()(const Key& key) const {
            size_t hash = 0;
            for (const auto value : key)
              hash ^= std::hash<long long>{}(value) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
            return hash;
          }
        };
        std::unordered_map<Key, std::shared_ptr<moveit::core::RobotState>, KeyHash> cache;
        size_t fcl_checks = 0, wrist_rejections = 0, checked_edges = 0;
        size_t jump_rejections = 0, bridge_checks = 0, bridge_successes = 0;
        size_t coarse_rejections = 0, near_refinements = 0;
        double bridge_collision_ms = 0.0, clearance_ms = 0.0;
        const auto lift = [&](const ob::State* value) -> std::shared_ptr<moveit::core::RobotState> {
          const auto* q = value->as<ob::RealVectorStateSpace::StateType>()->values;
          Key key{};
          for (size_t axis = 0; axis < key.size(); ++axis)
            key[axis] = std::llround(q[axis] * 1e9);
          const auto cached = cache.find(key);
          if (cached != cache.end()) return cached->second;
          auto robot = std::make_shared<moveit::core::RobotState>(state);
          for (size_t axis = 0; axis < axes.size(); ++axis)
            robot->setVariablePosition(axes[axis], q[axis]);
          robot->update(true);
          if (!robot->satisfiesBounds())
            return cache.emplace(key, nullptr).first->second;
          std::array<std::vector<alfa_robot::analytic_ik::V3WristOrientationSolution>, 2> wrist;
          for (size_t side_index = 0; side_index < 2; ++side_index) {
            const std::string side = side_index == 0 ? "left" : "right";
            std::array<double, 4> prefix{};
            std::array<double, 3> seed{};
            double progress = 0.0;
            for (size_t axis = 0; axis < 4; ++axis) {
              prefix[axis] = q[side_index * 4 + axis];
              const double travel = goal.getVariablePosition(axes[side_index * 4 + axis]) -
                state.getVariablePosition(axes[side_index * 4 + axis]);
              if (std::abs(travel) > 1e-6)
                progress += (prefix[axis] - state.getVariablePosition(axes[side_index * 4 + axis])) /
                  travel / 4.0;
            }
            for (int joint = 0; joint < 3; ++joint) {
              const std::string name = side + "_joint" + std::to_string(joint + 5);
              seed[joint] = state.getVariablePosition(name) * (1.0 - progress) +
                goal.getVariablePosition(name) * progress;
            }
            wrist[side_index] = (side_index == 0 ? left_ : right_).solveWristOrientation(
              prefix, tool_rotations[side_index], seed);
          }
          if (wrist[0].empty() || wrist[1].empty()) {
            ++wrist_rejections;
            return cache.emplace(key, nullptr).first->second;
          }
          double best_cost = std::numeric_limits<double>::infinity();
          std::shared_ptr<moveit::core::RobotState> selected;
          for (const auto& left_wrist : wrist[0])
            for (const auto& right_wrist : wrist[1]) {
              auto candidate = std::make_shared<moveit::core::RobotState>(*robot);
              double cost = 0.0;
              for (size_t side_index = 0; side_index < 2; ++side_index) {
                const std::string side = side_index == 0 ? "left" : "right";
                const auto& solution = side_index == 0 ? left_wrist : right_wrist;
                for (int joint = 0; joint < 3; ++joint) {
                  const std::string name = side + "_joint" + std::to_string(joint + 5);
                  const double delta = solution.wrist[joint] - state.getVariablePosition(name);
                  cost += delta * delta;
                  candidate->setVariablePosition(name, solution.wrist[joint]);
                }
              }
              if (cost >= best_cost) continue;
              candidate->update(true);
              if (!candidate->satisfiesBounds()) continue;
              ++fcl_checks;
              if (loaded_scene->isStateColliding(*candidate)) continue;
              best_cost = cost;
              selected = std::move(candidate);
            }
          return cache.emplace(key, std::move(selected)).first->second;
        };
        og::SimpleSetup setup(space);
        setup.setStateValidityChecker([&](const ob::State* value) {
          return static_cast<bool>(lift(value));
        });
        setup.getSpaceInformation()->setMotionValidator(
          std::make_shared<WristEdgeValidator>(setup.getSpaceInformation(), lift, loaded_scene,
            checked_edges, jump_rejections, bridge_checks, bridge_successes,
            coarse_rejections, near_refinements, bridge_collision_ms, clearance_ms,
            .01, false));
        ob::ScopedState<> from(space), to(space);
        for (size_t axis = 0; axis < axes.size(); ++axis) {
          from[axis] = state.getVariablePosition(axes[axis]);
          to[axis] = goal.getVariablePosition(axes[axis]);
        }
        setup.setStartAndGoalStates(from, to, 1e-9);
        setup.setup();
        const auto lifted_start = lift(from.get());
        const auto lifted_goal = lift(to.get());
        transfer["start_valid"] = static_cast<bool>(lifted_start);
        transfer["goal_valid"] = static_cast<bool>(lifted_goal);
        if (!lifted_start || !lifted_goal) {
          transfer["failure_stage"] = "wrist_ik_or_endpoint_collision";
        } else {
          const auto started = std::chrono::steady_clock::now();
          transfer["shortcut_valid"] = setup.getSpaceInformation()->checkMotion(from.get(), to.get());
          transfer["shortcut_ms"] = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
          if (!transfer["shortcut_valid"].get<bool>()) {
            ompl::RNG::setSeed(1);
            auto planner = std::make_shared<og::RRTConnect>(setup.getSpaceInformation());
            planner->setRange(.25);
            setup.setPlanner(planner);
            const auto deadline = started + std::chrono::duration<double>(2.0);
            const ob::PlannerTerminationCondition stop([&]() {
              return std::chrono::steady_clock::now() >= deadline;
            });
            transfer["rrt_exact"] = setup.solve(stop) == ob::PlannerStatus::EXACT_SOLUTION;
          } else {
            transfer["rrt_exact"] = true;
          }
          if (transfer["rrt_exact"].get<bool>()) {
            const auto* path = transfer["shortcut_valid"].get<bool>() ? nullptr : &setup.getSolutionPath();
            const size_t segments = path ? path->getStateCount() - 1 : 1;
            bool valid = true;
            for (size_t segment = 0; segment < segments && valid; ++segment) {
              const auto* first = path ? path->getState(segment) : from.get();
              const auto* last = path ? path->getState(segment + 1) : to.get();
              const auto* qa = first->as<ob::RealVectorStateSpace::StateType>()->values;
              const auto* qb = last->as<ob::RealVectorStateSpace::StateType>()->values;
              double max_delta = 0.0;
              for (size_t axis = 0; axis < axes.size(); ++axis)
                max_delta = std::max(max_delta, std::abs(qb[axis] - qa[axis]));
              const int steps = std::max(1, static_cast<int>(
                std::ceil(max_delta / (.5 * kPi / 180.0))));
              auto* sample = space->allocState();
              for (int step = segment == 0 ? 0 : 1; step <= steps; ++step) {
                space->interpolate(first, last, static_cast<double>(step) / steps, sample);
                auto robot = lift(sample);
                if (!robot) {
                  valid = false;
                  transfer["failure_stage"] = "return_post_validation";
                  break;
                }
                Json joints = Json::array();
                for (const auto& name : names_) joints.push_back(robot->getVariablePosition(name));
                transfer["frames"].push_back({{"stage", "loaded_return"}, {"joints", joints}});
              }
              space->freeState(sample);
            }
            transfer["success"] = valid;
          } else {
            transfer["failure_stage"] = "return_rrt_timeout_or_no_path";
          }
          transfer["wall_ms"] = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        }
        transfer["fcl_checks"] = fcl_checks;
        transfer["checked_edges"] = checked_edges;
        transfer["wrist_rejections"] = wrist_rejections;
        transfer["jump_rejections"] = jump_rejections;
      }
      result["loaded_return"] = std::move(transfer);
      Json bridge = {{"kind", "v3_loaded_return_to_named_second_home_bridge"},
        {"success", false}, {"frames", Json::array()}};
      if (result["loaded_return"].at("success").get<bool>()) {
        moveit::core::RobotState from(state);
        const auto& last = result["loaded_return"].at("frames").back().at("joints");
        for (size_t axis = 0; axis < names_.size(); ++axis)
          from.setVariablePosition(names_[axis], last.at(axis).get<double>());
        from.update(true);
        moveit::core::RobotState to(from);
        if (!to.setToDefaultValues(model_->getJointModelGroup("whole_body"), "second_home"))
          throw std::runtime_error("missing named second_home for loaded bridge");
        to.update(true);
        bridge["goal_collision"] = collisionReason(loaded_scene, to);
        double max_steps = 0.0;
        for (size_t axis = 0; axis < names_.size(); ++axis) {
          const double delta = std::abs(to.getVariablePosition(names_[axis]) -
            from.getVariablePosition(names_[axis]));
          max_steps = std::max(max_steps, delta / (axis == 14 ? .005 : .5 * kPi / 180.0));
        }
        const int steps = std::max(1, static_cast<int>(std::ceil(max_steps)));
        for (int step = 0; step <= steps; ++step) {
          moveit::core::RobotState sample(from);
          const double fraction = static_cast<double>(step) / steps;
          for (size_t axis = 0; axis < names_.size(); ++axis)
            sample.setVariablePosition(names_[axis],
              from.getVariablePosition(names_[axis]) * (1.0 - fraction) +
              to.getVariablePosition(names_[axis]) * fraction);
          sample.update(true);
          const std::string collision = collisionReason(loaded_scene, sample);
          if (!sample.satisfiesBounds() || !collision.empty()) {
            bridge["failure_step"] = step;
            bridge["failure_reason"] = collision.empty() ? "joint_limit" : collision;
            break;
          }
          Json joints = Json::array();
          for (const auto& name : names_) joints.push_back(sample.getVariablePosition(name));
          bridge["frames"].push_back({{"stage", "loaded_bridge"}, {"joints", joints}});
        }
        bridge["success"] = bridge["frames"].size() == static_cast<size_t>(steps + 1);
        bridge["maximum_orientation_change_deg"] = std::max(
          Eigen::AngleAxisd(from.getGlobalLinkTransform("left_tool0").linear().transpose() *
            to.getGlobalLinkTransform("left_tool0").linear()).angle(),
          Eigen::AngleAxisd(from.getGlobalLinkTransform("right_tool0").linear().transpose() *
            to.getGlobalLinkTransform("right_tool0").linear()).angle()) * 180.0 / kPi;
      }
      result["placement_entry_bridge"] = std::move(bridge);
      if (result["placement_entry_bridge"].at("success").get<bool>()) {
        moveit::core::RobotState from(state);
        const auto& entry = result["placement_entry_bridge"].at("frames").back().at("joints");
        for (size_t axis = 0; axis < names_.size(); ++axis)
          from.setVariablePosition(names_[axis], entry.at(axis).get<double>());
        from.update(true);
        moveit::core::RobotState to(from);
        if (!to.setToDefaultValues(model_->getJointModelGroup("whole_body"), "second_unloading"))
          throw std::runtime_error("missing second_unloading named pose");
        to.update(true);
        Json placement = {{"kind", "v3_named_loaded_placement_shortcut"},
          {"success", false}, {"frames", Json::array()},
          {"goal_collision", collisionReason(loaded_scene, to)}};
        double max_steps = 0.0;
        for (size_t axis = 0; axis < names_.size(); ++axis) {
          const double delta = std::abs(to.getVariablePosition(names_[axis]) -
            from.getVariablePosition(names_[axis]));
          max_steps = std::max(max_steps, delta / (axis == 14 ? .005 : .5 * kPi / 180.0));
        }
        const int steps = std::max(1, static_cast<int>(std::ceil(max_steps)));
        for (int step = 0; step <= steps; ++step) {
          moveit::core::RobotState sample(from);
          const double fraction = static_cast<double>(step) / steps;
          for (size_t axis = 0; axis < names_.size(); ++axis)
            sample.setVariablePosition(names_[axis],
              from.getVariablePosition(names_[axis]) * (1.0 - fraction) +
              to.getVariablePosition(names_[axis]) * fraction);
          sample.update(true);
          const std::string collision = collisionReason(loaded_scene, sample);
          if (!sample.satisfiesBounds() || !collision.empty()) {
            placement["failure_step"] = step;
            placement["failure_reason"] = collision.empty() ? "joint_limit" : collision;
            break;
          }
          Json joints = Json::array();
          for (const auto& name : names_) joints.push_back(sample.getVariablePosition(name));
          placement["frames"].push_back({{"stage", "loaded_placement"}, {"joints", joints}});
        }
        placement["success"] = placement["frames"].size() == static_cast<size_t>(steps + 1);
        placement["total_steps"] = steps;
        result["named_placement_shortcut"] = std::move(placement);
      }
      if (!placement_reference_path.empty() &&
          result["placement_entry_bridge"].at("success").get<bool>()) {
        std::ifstream file(placement_reference_path);
        if (!file) throw std::runtime_error("cannot read fixed placement reference");
        const Json reference = Json::parse(file);
        if (!reference.at("success").get<bool>() ||
            reference.at("joint_names").get<std::vector<std::string>>() != names_)
          throw std::invalid_argument("invalid fixed placement trajectory reference");
        Json placement = {{"kind", "v3_actual_attached_boxes_fixed_placement_check"},
          {"success", false}, {"reference", placement_reference_path},
          {"checked_frames", 0}};
        const auto& frames = reference.at("frames");
        if (frames.empty()) throw std::invalid_argument("empty fixed placement trajectory");
        const auto& entry = result["placement_entry_bridge"].at("frames").back().at("joints");
        for (size_t axis = 0; axis < names_.size(); ++axis)
          if (std::abs(frames.front().at("joints").at(axis).get<double>() -
              entry.at(axis).get<double>()) > 1e-5)
            throw std::invalid_argument("fixed placement does not start at named second_home");
        moveit::core::RobotState sample(state);
        for (size_t index = 0; index < frames.size(); ++index) {
          for (size_t axis = 0; axis < names_.size(); ++axis)
            sample.setVariablePosition(names_[axis],
              frames.at(index).at("joints").at(axis).get<double>());
          sample.update(true);
          const std::string collision = collisionReason(loaded_scene, sample);
          if (!sample.satisfiesBounds() || !collision.empty()) {
            placement["failure_index"] = index;
            placement["failure_stage"] = frames.at(index).value("stage", "unknown");
            placement["failure_reason"] = collision.empty() ? "joint_limit" : collision;
            break;
          }
          placement["checked_frames"] = index + 1;
        }
        placement["success"] = placement["checked_frames"].get<size_t>() == frames.size();
        placement["total_frames"] = frames.size();
        result["placement_reference_check"] = std::move(placement);
        if (result["placement_reference_check"].at("success").get<bool>()) {
          moveit::core::RobotState release(state);
          const auto& end_joints = frames.back().at("joints");
          for (size_t axis = 0; axis < names_.size(); ++axis)
            release.setVariablePosition(names_[axis], end_joints.at(axis).get<double>());
          release.update(true);
          auto released_scene = planning_scene::PlanningScene::clone(loaded_scene);
          for (size_t side_index = 0; side_index < 2; ++side_index) {
            const std::string side = side_index == 0 ? "left" : "right";
            const Eigen::Isometry3d box_pose =
              release.getGlobalLinkTransform(side + "_tool0") * attachment_offsets[side_index];
            moveit_msgs::msg::CollisionObject placed;
            placed.header.frame_id = "world";
            placed.id = "placed_" + side;
            placed.operation = moveit_msgs::msg::CollisionObject::ADD;
            shape_msgs::msg::SolidPrimitive primitive;
            primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
            primitive.dimensions = {.30, .40, .40};
            placed.primitives.push_back(primitive);
            geometry_msgs::msg::Pose pose;
            pose.position.x = box_pose.translation().x();
            pose.position.y = box_pose.translation().y();
            pose.position.z = box_pose.translation().z();
            const Eigen::Quaterniond rotation(box_pose.linear());
            pose.orientation.x = rotation.x();
            pose.orientation.y = rotation.y();
            pose.orientation.z = rotation.z();
            pose.orientation.w = rotation.w();
            placed.primitive_poses.push_back(pose);
            if (!released_scene->processCollisionObjectMsg(placed))
              throw std::runtime_error("failed to insert placed box");
          }
          release.clearAttachedBodies();
          release.update(true);
          Json reset = {{"kind", "v3_placed_boxes_unloaded_retract_and_return"},
            {"success", false}, {"frames", Json::array()},
            {"release_collision", collisionReason(released_scene, release)}};
          auto contact_scene = planning_scene::PlanningScene::clone(released_scene);
          const std::array<Eigen::Isometry3d, 2> release_tools{
            release.getGlobalLinkTransform("left_tool0"),
            release.getGlobalLinkTransform("right_tool0")};
          std::array<double, 2> swivel{};
          for (size_t side_index = 0; side_index < 2; ++side_index) {
            const std::string side = side_index == 0 ? "left" : "right";
            const std::string object = "placed_" + side;
            contact_scene->getAllowedCollisionMatrixNonConst().setEntry(object, side + "_tool0", true);
            contact_scene->getAllowedCollisionMatrixNonConst().setEntry(object, side + "_link7", true);
            std::array<double, 7> joints{};
            for (size_t joint = 0; joint < joints.size(); ++joint)
              joints[joint] = release.getVariablePosition(
                side + "_joint" + std::to_string(joint + 1));
            swivel[side_index] = (side_index == 0 ? left_ : right_).swivelAngle(joints);
          }
          moveit::core::RobotState withdrawn(release);
          for (int step = 0; step <= unloaded_retract_steps; ++step) {
            if (step > 0) {
              std::array<std::vector<alfa_robot::analytic_ik::V3RedundantIkSolution>, 2> solutions;
              for (size_t side_index = 0; side_index < 2; ++side_index) {
                const std::string side = side_index == 0 ? "left" : "right";
                Eigen::Isometry3d target = release_tools[side_index];
                target.translation() -= .01 * step * target.linear().col(2);
                alfa_robot::analytic_ik::V3RedundantIkRequest request;
                request.target_in_arm_base =
                  withdrawn.getGlobalLinkTransform("arm_carriage").inverse() * target;
                request.swivel_angle = swivel[side_index];
                for (int joint = 0; joint < 7; ++joint)
                  request.seed[joint] = withdrawn.getVariablePosition(
                    side + "_joint" + std::to_string(joint + 1));
                solutions[side_index] = (side_index == 0 ? left_ : right_).solveInArmBase(request);
              }
              double best_cost = std::numeric_limits<double>::infinity();
              std::string reason = solutions[0].empty() || solutions[1].empty() ?
                "analytic_ik_no_solution" : "collision_or_joint_jump";
              moveit::core::RobotState best(withdrawn);
              for (const auto& left : solutions[0])
                for (const auto& right : solutions[1]) {
                  moveit::core::RobotState candidate(withdrawn);
                  double cost = 0.0;
                  bool jumped = false;
                  for (size_t side_index = 0; side_index < 2; ++side_index) {
                    const std::string side = side_index == 0 ? "left" : "right";
                    const auto& joints = side_index == 0 ? left.joints : right.joints;
                    for (int joint = 0; joint < 7; ++joint) {
                      const std::string name = side + "_joint" + std::to_string(joint + 1);
                      const double delta = joints[joint] - withdrawn.getVariablePosition(name);
                      jumped |= std::abs(delta) > 10.0 * kPi / 180.0;
                      cost += delta * delta;
                      candidate.setVariablePosition(name, joints[joint]);
                    }
                  }
                  if (jumped || cost >= best_cost) continue;
                  candidate.update(true);
                  if (!candidate.satisfiesBounds()) {
                    reason = "joint_limit";
                    continue;
                  }
                  const auto collision = collisionReason(contact_scene, candidate);
                  if (!collision.empty()) {
                    reason = collision;
                    continue;
                  }
                  best_cost = cost;
                  best = std::move(candidate);
                }
              if (!std::isfinite(best_cost)) {
                reset["failure_stage"] = "unloaded_retract";
                reset["failure_step"] = step;
                reset["failure_reason"] = reason;
                break;
              }
              withdrawn = std::move(best);
            }
            Json joints = Json::array();
            for (const auto& name : names_) joints.push_back(withdrawn.getVariablePosition(name));
            reset["frames"].push_back({{"stage", "unloaded_retract"}, {"joints", joints}});
          }
          if (reset["frames"].size() == static_cast<size_t>(unloaded_retract_steps + 1)) {
            reset["retract_end_strict_collision"] = collisionReason(released_scene, withdrawn);
            if (!reset["retract_end_strict_collision"].get<std::string>().empty()) {
              reset["failure_stage"] = "retract_still_touching";
              reset["failure_reason"] = reset["retract_end_strict_collision"];
            } else {
              moveit::core::RobotState goal(withdrawn);
              if (!goal.setToDefaultValues(model_->getJointModelGroup("whole_body"), "second_home"))
                throw std::runtime_error("missing second_home reset goal");
              goal.update(true);
              reset["goal_collision"] = collisionReason(released_scene, goal);
              double maximum = 0.0;
              for (size_t axis = 0; axis < names_.size(); ++axis) {
                const double delta = std::abs(goal.getVariablePosition(names_[axis]) -
                  withdrawn.getVariablePosition(names_[axis]));
                maximum = std::max(maximum, delta / (axis == 14 ? .005 : .5 * kPi / 180.0));
              }
              const int count = std::max(1, static_cast<int>(std::ceil(maximum)));
              for (int sample_index = 1; sample_index <= count; ++sample_index) {
                moveit::core::RobotState sample(withdrawn);
                const double fraction = static_cast<double>(sample_index) / count;
                for (size_t axis = 0; axis < names_.size(); ++axis)
                  sample.setVariablePosition(names_[axis],
                    withdrawn.getVariablePosition(names_[axis]) * (1.0 - fraction) +
                    goal.getVariablePosition(names_[axis]) * fraction);
                sample.update(true);
                const std::string collision = collisionReason(released_scene, sample);
                if (!sample.satisfiesBounds() || !collision.empty()) {
                  reset["failure_stage"] = "unloaded_return_shortcut";
                  reset["failure_step"] = sample_index;
                  reset["failure_reason"] = collision.empty() ? "joint_limit" : collision;
                  break;
                }
                Json joints = Json::array();
                for (const auto& name : names_) joints.push_back(sample.getVariablePosition(name));
                reset["frames"].push_back({{"stage", "unloaded_return"}, {"joints", joints}});
              }
              reset["success"] = reset["frames"].size() ==
                static_cast<size_t>(unloaded_retract_steps + 1 + count);
              if (!reset["success"].get<bool>() &&
                  reset.value("failure_stage", "") == "unloaded_return_shortcut") {
                reset["frames"].erase(reset["frames"].begin() + unloaded_retract_steps + 1,
                  reset["frames"].end());
                auto space = std::make_shared<ob::RealVectorStateSpace>(15);
                ob::RealVectorBounds bounds(15);
                for (size_t axis = 0; axis < 15; ++axis) {
                  const auto& limit = model_->getVariableBounds(names_[axis]);
                  bounds.setLow(axis, limit.min_position_);
                  bounds.setHigh(axis, limit.max_position_);
                }
                space->setBounds(bounds);
                og::SimpleSetup setup(space);
                const auto state_from_ompl = [&](const ob::State* sample) {
                  moveit::core::RobotState state(withdrawn);
                  const auto* positions = sample->as<ob::RealVectorStateSpace::StateType>()->values;
                  for (size_t axis = 0; axis < 15; ++axis)
                    state.setVariablePosition(names_[axis], positions[axis]);
                  state.update(true);
                  return state;
                };
                setup.setStateValidityChecker([&](const ob::State* sample) {
                  const auto candidate = state_from_ompl(sample);
                  return candidate.satisfiesBounds() && !released_scene->isStateColliding(candidate);
                });
                setup.getSpaceInformation()->setStateValidityCheckingResolution(.0002);
                ob::ScopedState<> start(space), target(space);
                for (size_t axis = 0; axis < 15; ++axis) {
                  start[axis] = withdrawn.getVariablePosition(names_[axis]);
                  target[axis] = goal.getVariablePosition(names_[axis]);
                }
                setup.setStartAndGoalStates(start, target, 1e-9);
                ompl::RNG::setSeed(1);
                auto planner = std::make_shared<og::RRTConnect>(setup.getSpaceInformation());
                planner->setRange(.25);
                setup.setPlanner(planner);
                setup.setup();
                const auto started = std::chrono::steady_clock::now();
                const auto deadline = started + std::chrono::duration<double>(2.0);
                const ob::PlannerTerminationCondition stop([&]() {
                  return std::chrono::steady_clock::now() >= deadline;
                });
                reset["rrt_exact"] = setup.solve(stop) == ob::PlannerStatus::EXACT_SOLUTION;
                reset["rrt_ms"] = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - started).count();
                if (reset["rrt_exact"].get<bool>()) {
                  bool valid = true;
                  const auto& path = setup.getSolutionPath();
                  for (size_t segment = 0; segment + 1 < path.getStateCount() && valid; ++segment) {
                    const auto* from = path.getState(segment);
                    const auto* to = path.getState(segment + 1);
                    const auto* first = from->as<ob::RealVectorStateSpace::StateType>()->values;
                    const auto* last = to->as<ob::RealVectorStateSpace::StateType>()->values;
                    double maximum = std::abs(last[14] - first[14]) / .005;
                    for (size_t axis = 0; axis < 14; ++axis)
                      maximum = std::max(maximum,
                        std::abs(last[axis] - first[axis]) / (.5 * kPi / 180.0));
                    const int steps = std::max(1, static_cast<int>(std::ceil(maximum)));
                    auto* probe = space->allocState();
                    for (int index = 1; index <= steps; ++index) {
                      space->interpolate(from, to, static_cast<double>(index) / steps, probe);
                      const auto candidate = state_from_ompl(probe);
                      const std::string collision = collisionReason(released_scene, candidate);
                      if (!candidate.satisfiesBounds() || !collision.empty()) {
                        valid = false;
                        reset["failure_stage"] = "unloaded_rrt_post_validation";
                        reset["failure_reason"] = collision.empty() ? "joint_limit" : collision;
                        break;
                      }
                      Json joints = Json::array();
                      for (const auto& name : names_)
                        joints.push_back(candidate.getVariablePosition(name));
                      reset["frames"].push_back({{"stage", "unloaded_return_rrt"}, {"joints", joints}});
                    }
                    space->freeState(probe);
                  }
                  reset["success"] = valid;
                  if (valid) {
                    reset.erase("failure_stage");
                    reset.erase("failure_step");
                    reset.erase("failure_reason");
                  }
                }
              }
            }
          }
          result["unloaded_reset_check"] = std::move(reset);
        }
      }
    }
    return result;
  }

  Json planManualLoadedFourD(double budget_s)
  {
    if (!manual_scene_ || !manual_start_ || !manual_goal_ ||
        !std::isfinite(budget_s) || budget_s <= 0.0)
      throw std::invalid_argument("manual loaded context or budget invalid");
    const auto started = std::chrono::steady_clock::now();
    const std::array<int, 4> joints{1, 3, 4, 6};
    std::array<double, 4> joint_deltas{};
    for (size_t axis = 0; axis < joints.size(); ++axis) {
      const auto name = "left_joint" + std::to_string(joints[axis]);
      joint_deltas[axis] = manual_goal_->getVariablePosition(name) -
        manual_start_->getVariablePosition(name);
    }
    auto space = std::make_shared<ob::RealVectorStateSpace>(4);
    ob::RealVectorBounds bounds(4);
    bounds.setLow(0.0);
    bounds.setHigh(1.0);
    space->setBounds(bounds);
    using StateKey = std::array<long long, 4>;
    struct Hash {
      size_t operator()(const StateKey& key) const {
        size_t hash = 0;
        for (const auto value : key)
          hash ^= std::hash<long long>{}(value) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
        return hash;
      }
    };
    size_t fcl_checks = 0, collision_rejections = 0, checked_edges = 0;
    size_t jump_rejections = 0, bridge_checks = 0, bridge_successes = 0;
    size_t coarse_rejections = 0, near_refinements = 0;
    double bridge_collision_ms = 0.0, clearance_ms = 0.0;
    std::unordered_map<StateKey, std::shared_ptr<moveit::core::RobotState>, Hash> cache;
    const auto lift = [&](const ob::State* value) -> std::shared_ptr<moveit::core::RobotState> {
      const auto* progress = value->as<ob::RealVectorStateSpace::StateType>()->values;
      StateKey key{};
      for (size_t axis = 0; axis < joints.size(); ++axis)
        key[axis] = std::llround(progress[axis] * 1e9);
      const auto cached = cache.find(key);
      if (cached != cache.end()) return cached->second;
      auto candidate = std::make_shared<moveit::core::RobotState>(*manual_start_);
      for (const std::string side : {"left", "right"})
        for (size_t axis = 0; axis < joints.size(); ++axis) {
          const auto name = side + "_joint" + std::to_string(joints[axis]);
          candidate->setVariablePosition(name,
            manual_start_->getVariablePosition(name) * (1.0 - progress[axis]) +
            manual_goal_->getVariablePosition(name) * progress[axis]);
        }
      candidate->update(true);
      if (!candidate->satisfiesBounds()) return cache.emplace(key, nullptr).first->second;
      ++fcl_checks;
      if (manual_scene_->isStateColliding(*candidate)) {
        ++collision_rejections;
        return cache.emplace(key, nullptr).first->second;
      }
      return cache.emplace(key, candidate).first->second;
    };
    ompl::RNG::setSeed(1);
    og::SimpleSetup setup(space);
    setup.setStateValidityChecker([&lift](const ob::State* state) {
      return static_cast<bool>(lift(state));
    });
    setup.getSpaceInformation()->setMotionValidator(
      std::make_shared<WristEdgeValidator>(setup.getSpaceInformation(), lift,
        manual_scene_, checked_edges, jump_rejections, bridge_checks, bridge_successes,
        coarse_rejections, near_refinements, bridge_collision_ms, clearance_ms,
        .01, false, joint_deltas));
    ob::ScopedState<> from(space), to(space);
    for (size_t axis = 0; axis < joints.size(); ++axis) {
      from[axis] = 0.0;
      to[axis] = 1.0;
    }
    setup.setStartAndGoalStates(from, to, 1e-9);
    setup.setup();
    Json result = {{"kind", "v3_loaded_four_independent_joint_progress_rrt"},
      {"success", false}, {"joint_numbers", joints},
      {"scope", "offline target_only, two attached boxes; each of J1,J3,J4,J6 in [0,1]"},
      {"budget_s", budget_s}, {"start_collision", collisionReason(manual_scene_, *manual_start_)},
      {"goal_collision", collisionReason(manual_scene_, *manual_goal_)},
      {"frames", Json::array()}};
    const bool start_valid = static_cast<bool>(lift(from.get()));
    const bool goal_valid = static_cast<bool>(lift(to.get()));
    result["start_valid"] = start_valid;
    result["goal_valid"] = goal_valid;
    if (start_valid && goal_valid) {
      const auto shortcut_started = std::chrono::steady_clock::now();
      result["direct_valid"] = setup.getSpaceInformation()->checkMotion(from.get(), to.get());
      result["shortcut_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - shortcut_started).count();
      if (!result["direct_valid"].get<bool>()) {
        auto planner = std::make_shared<og::RRTConnect>(setup.getSpaceInformation());
        planner->setRange(.25);
        setup.setPlanner(planner);
        const auto deadline = started + std::chrono::duration<double>(budget_s);
        const ob::PlannerTerminationCondition stop([&]() {
          return std::chrono::steady_clock::now() >= deadline;
        });
        const auto rrt_started = std::chrono::steady_clock::now();
        result["ompl_exact"] = setup.solve(stop) == ob::PlannerStatus::EXACT_SOLUTION;
        result["rrt_ms"] = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - rrt_started).count();
      } else {
        result["ompl_exact"] = true;
        result["rrt_ms"] = 0.0;
      }
      ob::PlannerData data(setup.getSpaceInformation());
      if (!result["direct_valid"].get<bool>()) setup.getPlannerData(data);
      result["tree_vertices"] = data.numVertices();
      if (result["ompl_exact"].get<bool>()) {
        const auto validation_started = std::chrono::steady_clock::now();
        const auto* path = result["direct_valid"].get<bool>() ? nullptr : &setup.getSolutionPath();
        const size_t segments = path ? path->getStateCount() - 1 : 1;
        bool valid = true;
        for (size_t segment = 0; segment < segments && valid; ++segment) {
          const auto* first = path ? path->getState(segment) : from.get();
          const auto* last = path ? path->getState(segment + 1) : to.get();
          const auto* first_values = first->as<ob::RealVectorStateSpace::StateType>()->values;
          const auto* last_values = last->as<ob::RealVectorStateSpace::StateType>()->values;
          double max_delta = 0.0;
          for (size_t axis = 0; axis < joints.size(); ++axis)
            max_delta = std::max(max_delta,
              std::abs(last_values[axis] - first_values[axis]) * std::abs(joint_deltas[axis]));
          const int steps = std::max(1, static_cast<int>(
            std::ceil(max_delta / (0.5 * kPi / 180.0))));
          auto* sample = space->allocState();
          for (int index = segment == 0 ? 0 : 1; index <= steps; ++index) {
            space->interpolate(first, last, static_cast<double>(index) / steps, sample);
            auto candidate = lift(sample);
            if (!candidate) {
              valid = false;
              result["failure_stage"] = "post_validation_collision";
              break;
            }
            Json values = Json::array();
            for (const auto& name : names_)
              values.push_back(candidate->getVariablePosition(name));
            Json phases = Json::array();
            const auto* progress = sample->as<ob::RealVectorStateSpace::StateType>()->values;
            for (size_t axis = 0; axis < joints.size(); ++axis)
              phases.push_back(progress[axis]);
            result["frames"].push_back({{"joints", values}, {"progress", phases}});
          }
          space->freeState(sample);
        }
        result["success"] = valid;
        result["validation_ms"] = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - validation_started).count();
      } else {
        result["failure_stage"] = "rrt_budget_or_no_path";
      }
    } else {
      result["failure_stage"] = "invalid_endpoint";
    }
    result["fcl_checks"] = fcl_checks;
    result["collision_rejections"] = collision_rejections;
    result["checked_edges"] = checked_edges;
    result["coarse_rejections"] = coarse_rejections;
    result["jump_rejections"] = jump_rejections;
    result["wall_ms"] = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return result;
  }

  Json previewExtractedBoxWristTurn(const Json& recorded, int box_id)
  {
    const auto& tasks = recorded.at("tasks");
    const Json* task = nullptr;
    for (const auto& item : tasks)
      if (item.at("box_id").get<int>() == box_id) {
        task = &item;
        break;
      }
    if (!task || !task->at("success").get<bool>())
      throw std::invalid_argument("no successful recorded box task");
    const auto& attempt = task->at("attempts").at(task->at("selected_attempt").get<size_t>());
    if (attempt.at("initial_pose") != "home" || attempt.at("grasp_mode") != "front" ||
        !attempt.at("loaded_stage").at("success").get<bool>())
      throw std::invalid_argument("expected successful Home 1 side-suction loaded stage");
    const auto& history = attempt.at("loaded_stage").at("frames");
    const Json* extraction = nullptr;
    for (const auto& frame : history)
      if (frame.at("stage") == "horizontal_extract") extraction = &frame;
    if (!extraction || !extraction->at("attached").get<bool>())
      throw std::invalid_argument("missing attached horizontal extraction endpoint");
    auto scene = std::make_shared<planning_scene::PlanningScene>(model_);
    for (const auto& box : recorded.at("environment").at("boxes"))
      addBox(scene, "environment_" + box.at("id").get<std::string>(), box);
    std::set<int> removed;
    for (const auto& id : task->at("removed_box_ids")) removed.insert(id.get<int>());
    removed.insert(box_id);
    for (const auto& box : recorded.at("wall_boxes")) {
      const int id = box.at("box_id").get<int>();
      if (!removed.count(id)) addBox(scene, "wall_box_" + std::to_string(id), box);
    }
    moveit::core::RobotState start(home_);
    const auto& values = extraction->at("joints");
    for (size_t index = 0; index < names_.size(); ++index)
      start.setVariablePosition(names_[index], values.at(index).get<double>());
    start.update(true);
    Eigen::Isometry3d box_world = Eigen::Isometry3d::Identity();
    const auto& box_pose = extraction->at("box_pose");
    const auto position = box_pose.at("position").get<std::vector<double>>();
    const auto quaternion = box_pose.at("quaternion_xyzw").get<std::vector<double>>();
    box_world.translation() = Eigen::Vector3d(position[0], position[1], position[2]);
    box_world.linear() = Eigen::Quaterniond(quaternion[3], quaternion[0],
      quaternion[1], quaternion[2]).normalized().toRotationMatrix();
    const auto tool_to_box = start.getGlobalLinkTransform("left_tool0").inverse() * box_world;
    const std::vector<shapes::ShapeConstPtr> shapes{
      std::make_shared<shapes::Box>(0.30, 0.40, 0.40)};
    const std::vector<std::string> touch{"left_tool0", "left_joint7", "left_joint6"};
    start.attachBody("carried_box", Eigen::Isometry3d::Identity(), shapes,
      EigenSTL::vector_Isometry3d{tool_to_box}, touch, "left_tool0");
    start.update(true);
    Json output = {{"kind", "v3_extracted_box_wrist_only_orientation_preview"},
      {"box_id", box_id}, {"initial_pose", "home"}, {"grasp_mode", "front"},
      {"success", false}, {"joint_names", names_}, {"frames", Json::array()},
      {"recorded_extract_m", .30}, {"requested_extract_m", .35},
      {"start_collision", collisionReason(scene, start)},
      {"scope", "offline single-arm preview; 30-to-35cm extract then wrist-only turn; no loaded return"}};
    if (!output["start_collision"].get<std::string>().empty()) {
      output["failure_stage"] = "attached_start_collision";
      return output;
    }
    const auto append_frame = [&](const moveit::core::RobotState& state,
      const std::string& stage, double fraction, double angle) {
      const Eigen::Isometry3d box = state.getGlobalLinkTransform("left_tool0") * tool_to_box;
      const Eigen::Quaterniond box_rotation(box.linear());
      Json joints = Json::array();
      for (const auto& name : names_) joints.push_back(state.getVariablePosition(name));
      output["frames"].push_back({{"stage", stage}, {"joints", joints},
        {"fraction", fraction}, {"rotation_deg", angle},
        {"box_position", {box.translation().x(), box.translation().y(), box.translation().z()}},
        {"box_quaternion_xyzw", {box_rotation.x(), box_rotation.y(),
          box_rotation.z(), box_rotation.w()}}});
    };
    append_frame(start, "extract_30_to_35cm", 0.0, 0.0);
    const Eigen::Isometry3d initial_tool = start.getGlobalLinkTransform("left_tool0");
    for (int extract_step = 1; extract_step <= 5; ++extract_step) {
      Eigen::Isometry3d target = initial_tool;
      target.translation().x() -= 0.01 * extract_step;
      alfa_robot::analytic_ik::V3RedundantIkRequest request;
      request.target_in_arm_base = start.getGlobalLinkTransform("arm_carriage").inverse() * target;
      std::array<double, 7> seed{};
      for (int joint = 0; joint < 7; ++joint)
        seed[joint] = start.getVariablePosition("left_joint" + std::to_string(joint + 1));
      request.seed = seed;
      request.swivel_angle = left_.swivelAngle(seed);
      const auto solutions = left_.solveInArmBase(request);
      bool valid = false;
      std::string reason = solutions.empty() ? "no_analytic_solution" : "joint_jump_or_collision";
      for (const auto& solution : solutions) {
        moveit::core::RobotState candidate(start);
        bool jump = false;
        for (int joint = 0; joint < 7; ++joint) {
          if (std::abs(solution.joints[joint] - seed[joint]) > 10.0 * kPi / 180.0)
            jump = true;
          candidate.setVariablePosition("left_joint" + std::to_string(joint + 1),
            solution.joints[joint]);
        }
        if (jump) continue;
        candidate.update(true);
        if (!candidate.satisfiesBounds()) {
          reason = "joint_limit";
          continue;
        }
        reason = collisionReason(scene, candidate);
        if (!reason.empty()) continue;
        start = std::move(candidate);
        valid = true;
        break;
      }
      if (!valid) {
        output["failure_stage"] = "extract_30_to_35cm";
        output["failure_reason"] = reason;
        output["failure_step"] = extract_step;
        return output;
      }
      append_frame(start, "extract_30_to_35cm", extract_step / 5.0, 0.0);
    }
    box_world = start.getGlobalLinkTransform("left_tool0") * tool_to_box;
    moveit::core::RobotState named_home(home_);
    if (!named_home.setToDefaultValues(model_->getJointModelGroup("whole_body"), "home"))
      throw std::runtime_error("missing Home 1 named state");
    named_home.update(true);
    const Eigen::Matrix3d carriage = start.getGlobalLinkTransform("arm_carriage").linear();
    const Eigen::Quaterniond from(carriage.transpose() *
      start.getGlobalLinkTransform("left_tool0").linear());
    const Eigen::Quaterniond to(carriage.transpose() *
      named_home.getGlobalLinkTransform("left_tool0").linear());
    const double rotation = from.angularDistance(to);
    const int steps = std::max(1, static_cast<int>(std::ceil(rotation / (0.5 * kPi / 180.0))));
    output["requested_rotation_deg"] = rotation * 180.0 / kPi;
    moveit::core::RobotState current(start);
    const Eigen::Vector3d first_box = box_world.translation();
    double max_box_displacement = 0.0;
    size_t wrist_frames = 0;
    for (int step = 0; step <= steps; ++step) {
      const double fraction = static_cast<double>(step) / steps;
      const double eased = fraction * fraction * (3.0 - 2.0 * fraction);
      const Eigen::Matrix3d target = from.slerp(eased, to).toRotationMatrix();
      std::array<double, 4> prefix{};
      std::array<double, 3> seed{};
      for (int joint = 0; joint < 4; ++joint)
        prefix[joint] = start.getVariablePosition("left_joint" + std::to_string(joint + 1));
      for (int joint = 0; joint < 3; ++joint)
        seed[joint] = current.getVariablePosition("left_joint" + std::to_string(joint + 5));
      const auto solutions = left_.solveWristOrientation(prefix, target, seed);
      bool valid = false;
      std::string reason = solutions.empty() ? "no_analytic_wrist_solution" : "wrist_jump_or_collision";
      moveit::core::RobotState chosen(current);
      for (const auto& solution : solutions) {
        moveit::core::RobotState candidate(current);
        bool jump = false;
        for (int joint = 0; joint < 3; ++joint) {
          if (std::abs(solution.wrist[joint] - seed[joint]) > 8.0 * kPi / 180.0)
            jump = true;
          candidate.setVariablePosition("left_joint" + std::to_string(joint + 5),
            solution.wrist[joint]);
        }
        if (jump) continue;
        candidate.update(true);
        if (!candidate.satisfiesBounds()) {
          reason = "joint_limit";
          continue;
        }
        reason = collisionReason(scene, candidate);
        if (!reason.empty()) continue;
        chosen = std::move(candidate);
        valid = true;
        break;
      }
      if (!valid) {
        output["failure_stage"] = "wrist_turn_step";
        output["failure_reason"] = reason;
        output["failure_step"] = step;
        break;
      }
      current = std::move(chosen);
      const Eigen::Isometry3d box = current.getGlobalLinkTransform("left_tool0") * tool_to_box;
      max_box_displacement = std::max(max_box_displacement,
        (box.translation() - first_box).norm());
      append_frame(current, "wrist_only_turn", fraction, rotation * eased * 180.0 / kPi);
      ++wrist_frames;
    }
    output["success"] = wrist_frames == static_cast<size_t>(steps + 1);
    output["max_box_center_displacement_m"] = max_box_displacement;
    return output;
  }

  Json previewDualExtractedWristTurn(const Json& segment)
  {
    if (segment.at("initial_pose") != "home" || segment.at("suction_mode") != "front" ||
        !segment.at("success").get<bool>() || segment.at("box_ids").size() != 2)
      throw std::invalid_argument("expected successful dual front-suction Home 1 segment");
    const Json* retreat = nullptr;
    for (const auto& frame : segment.at("frames"))
      if (frame.at("stage") == "dual_4") retreat = &frame;
    if (!retreat) throw std::invalid_argument("no dual 35cm retreat endpoint");
    auto scene = planning_scene::PlanningScene::clone(scene_);
    for (const auto& id : segment.at("removed_box_ids"))
      scene->getWorldNonConst()->removeObject("wall_box_" + std::to_string(id.get<int>()));
    for (const auto& id : segment.at("box_ids"))
      scene->getWorldNonConst()->removeObject("wall_box_" + std::to_string(id.get<int>()));
    moveit::core::RobotState start(home_);
    for (size_t joint = 0; joint < names_.size(); ++joint)
      start.setVariablePosition(names_[joint], retreat->at("joints").at(joint).get<double>());
    start.update(true);
    struct Box
    {
      std::string side;
      Eigen::Isometry3d offset;
      Eigen::Vector3d first_center;
    };
    std::vector<Box> boxes;
    const auto size = segment.at("box_size").get<std::vector<double>>();
    const std::vector<shapes::ShapeConstPtr> shapes{
      std::make_shared<shapes::Box>(size[0], size[1], size[2])};
    for (const auto& item : retreat->at("carried_boxes")) {
      const std::string side = item.at("side").get<std::string>();
      Eigen::Isometry3d offset = Eigen::Isometry3d::Identity();
      const auto translation = item.at("tool_to_box_center").get<std::vector<double>>();
      offset.translation() = Eigen::Vector3d(translation[0], translation[1], translation[2]);
      const auto rotation = item.at("tool_to_box_rotation");
      for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
          offset.linear()(row, col) = rotation.at(row).at(col).get<double>();
      const std::string tool = side + "_tool0";
      const std::vector<std::string> touch{tool, side + "_joint7"};
      start.attachBody("carried_" + side, Eigen::Isometry3d::Identity(), shapes,
        EigenSTL::vector_Isometry3d{offset}, touch, tool);
      boxes.push_back({side, offset,
        (start.getGlobalLinkTransform(tool) * offset).translation()});
    }
    start.update(true);
    moveit::core::RobotState named_home(home_);
    if (!named_home.setToDefaultValues(model_->getJointModelGroup("whole_body"), "home"))
      throw std::runtime_error("missing Home 1 named pose");
    named_home.update(true);
    const Eigen::Matrix3d carriage = start.getGlobalLinkTransform("arm_carriage").linear();
    std::array<Eigen::Quaterniond, 2> from, to;
    double maximum_rotation = 0.0;
    for (size_t side_index = 0; side_index < 2; ++side_index) {
      const std::string side = side_index == 0 ? "left" : "right";
      from[side_index] = Eigen::Quaterniond(carriage.transpose() *
        start.getGlobalLinkTransform(side + "_tool0").linear());
      to[side_index] = Eigen::Quaterniond(carriage.transpose() *
        named_home.getGlobalLinkTransform(side + "_tool0").linear());
      maximum_rotation = std::max(maximum_rotation,
        from[side_index].angularDistance(to[side_index]));
    }
    const int steps = std::max(1, static_cast<int>(
      std::ceil(maximum_rotation / (0.5 * kPi / 180.0))));
    Json output = {{"kind", "v3_dual_wrist_only_loaded_preview"},
      {"box_ids", segment.at("box_ids")}, {"success", false},
      {"joint_names", names_}, {"box_size", size}, {"frames", Json::array()},
      {"requested_rotation_deg", maximum_rotation * 180.0 / kPi},
      {"start_collision", collisionReason(scene, start)},
      {"scope", "offline dual side-suction after recorded 35cm retreat; wrist-only, no return path"}};
    if (!output["start_collision"].get<std::string>().empty() || boxes.size() != 2) {
      output["failure_stage"] = "attached_start_invalid";
      return output;
    }
    moveit::core::RobotState current(start);
    std::array<double, 2> max_displacement{0.0, 0.0};
    for (int step = 0; step <= steps; ++step) {
      const double fraction = static_cast<double>(step) / steps;
      const double eased = fraction * fraction * (3.0 - 2.0 * fraction);
      std::array<std::vector<alfa_robot::analytic_ik::V3WristOrientationSolution>, 2> solutions;
      for (size_t side_index = 0; side_index < 2; ++side_index) {
        const std::string side = side_index == 0 ? "left" : "right";
        std::array<double, 4> prefix{};
        std::array<double, 3> seed{};
        for (int joint = 0; joint < 4; ++joint)
          prefix[joint] = start.getVariablePosition(side + "_joint" + std::to_string(joint + 1));
        for (int joint = 0; joint < 3; ++joint)
          seed[joint] = current.getVariablePosition(side + "_joint" + std::to_string(joint + 5));
        const Eigen::Matrix3d desired = from[side_index].slerp(eased, to[side_index]).toRotationMatrix();
        solutions[side_index] = (side_index == 0 ? left_ : right_).solveWristOrientation(
          prefix, desired, seed);
      }
      bool valid = false;
      std::string reason = solutions[0].empty() || solutions[1].empty() ?
        "no_analytic_wrist_solution" : "wrist_jump_or_collision";
      for (const auto& left : solutions[0]) {
        if (valid) break;
        for (const auto& right : solutions[1]) {
          moveit::core::RobotState candidate(current);
          bool jump = false;
          for (size_t side_index = 0; side_index < 2; ++side_index) {
            const std::string side = side_index == 0 ? "left" : "right";
            const auto& wrist = side_index == 0 ? left.wrist : right.wrist;
            for (int joint = 0; joint < 3; ++joint) {
              const auto name = side + "_joint" + std::to_string(joint + 5);
              if (std::abs(wrist[joint] - current.getVariablePosition(name)) > 8.0 * kPi / 180.0)
                jump = true;
              candidate.setVariablePosition(name, wrist[joint]);
            }
          }
          if (jump) continue;
          candidate.update(true);
          if (!candidate.satisfiesBounds()) {
            reason = "joint_limit";
            continue;
          }
          reason = collisionReason(scene, candidate);
          if (!reason.empty()) continue;
          current = std::move(candidate);
          valid = true;
          break;
        }
      }
      if (!valid) {
        output["failure_stage"] = "dual_wrist_turn_step";
        output["failure_step"] = step;
        output["failure_reason"] = reason;
        break;
      }
      Json joints = Json::array();
      for (const auto& name : names_) joints.push_back(current.getVariablePosition(name));
      Json carried = Json::array();
      for (size_t side_index = 0; side_index < boxes.size(); ++side_index) {
        const auto& box = boxes[side_index];
        const Eigen::Isometry3d world = current.getGlobalLinkTransform(box.side + "_tool0") * box.offset;
        const Eigen::Quaterniond orientation(world.linear());
        max_displacement[side_index] = std::max(max_displacement[side_index],
          (world.translation() - box.first_center).norm());
        carried.push_back({{"side", box.side},
          {"center", {world.translation().x(), world.translation().y(), world.translation().z()}},
          {"quaternion_xyzw", {orientation.x(), orientation.y(), orientation.z(), orientation.w()}}});
      }
      output["frames"].push_back({{"stage", "dual_wrist_only_turn"},
        {"fraction", fraction}, {"rotation_deg", maximum_rotation * eased * 180.0 / kPi},
        {"joints", joints}, {"carried", carried}});
    }
    output["success"] = output["frames"].size() == static_cast<size_t>(steps + 1);
    output["max_box_displacement_m"] = max_displacement;
    return output;
  }

  Json shortcutManualLoadedFourD(const Json& original)
  {
    if (!manual_scene_ || !manual_start_ ||
        original.at("kind") != "v3_loaded_four_independent_joint_progress_rrt" ||
        !original.at("success").get<bool>())
      throw std::invalid_argument("expected a validated four-progress loaded path");
    const auto& source = original.at("frames");
    if (source.size() < 2) throw std::invalid_argument("path requires at least two frames");
    const auto started = std::chrono::steady_clock::now();
    size_t checks = 0, edges = 0;
    const auto interpolate = [&](size_t from, size_t to, double amount) {
      moveit::core::RobotState state(*manual_start_);
      const auto& first = source.at(from).at("joints");
      const auto& last = source.at(to).at("joints");
      for (size_t joint = 0; joint < names_.size(); ++joint)
        state.setVariablePosition(names_[joint],
          first.at(joint).get<double>() * (1.0 - amount) +
          last.at(joint).get<double>() * amount);
      state.update(true);
      return state;
    };
    const auto sample_count = [&](size_t from, size_t to, double step_degrees) {
      double max_delta = 0.0;
      const auto& first = source.at(from).at("joints");
      const auto& last = source.at(to).at("joints");
      for (size_t joint = 0; joint < 14; ++joint)
        max_delta = std::max(max_delta,
          std::abs(last.at(joint).get<double>() - first.at(joint).get<double>()));
      return std::max(1, static_cast<int>(
        std::ceil(max_delta / (step_degrees * kPi / 180.0))));
    };
    const auto edge_clear = [&](size_t from, size_t to, double step_degrees) {
      const int count = sample_count(from, to, step_degrees);
      for (int sample = 1; sample <= count; ++sample) {
        auto state = interpolate(from, to, static_cast<double>(sample) / count);
        ++checks;
        if (!state.satisfiesBounds() || manual_scene_->isStateColliding(state)) return false;
      }
      return true;
    };
    Json result = {{"kind", "v3_loaded_four_progress_farthest_shortcut"},
      {"success", false}, {"source_frames", source.size()},
      {"waypoints", Json::array()}, {"frames", Json::array()}};
    std::vector<size_t> waypoints{0};
    size_t current = 0;
    while (current + 1 < source.size()) {
      size_t farthest = current;
      for (size_t candidate = source.size() - 1; candidate > current; --candidate) {
        ++edges;
        if (edge_clear(current, candidate, 4.0) &&
            edge_clear(current, candidate, 0.5)) {
          farthest = candidate;
          break;
        }
      }
      if (farthest == current) {
        result["failure_stage"] = "no_collision_free_next_frame";
        break;
      }
      waypoints.push_back(farthest);
      current = farthest;
    }
    if (current == source.size() - 1) {
      for (size_t segment = 1; segment < waypoints.size(); ++segment) {
        const size_t from = waypoints[segment - 1];
        const size_t to = waypoints[segment];
        result["waypoints"].push_back({from, to});
        const int count = sample_count(from, to, 0.5);
        for (int sample = segment == 1 ? 0 : 1; sample <= count; ++sample) {
          const double amount = static_cast<double>(sample) / count;
          auto state = interpolate(from, to, amount);
          Json joints = Json::array();
          for (const auto& name : names_)
            joints.push_back(state.getVariablePosition(name));
          Json progress = Json::array();
          for (size_t axis = 0; axis < 4; ++axis) {
            const double first = source.at(from).at("progress").at(axis).get<double>();
            const double last = source.at(to).at("progress").at(axis).get<double>();
            progress.push_back(first * (1.0 - amount) + last * amount);
          }
          result["frames"].push_back({{"joints", joints}, {"progress", progress},
            {"source_from", from}, {"source_to", to}});
        }
      }
      const auto validation_started = std::chrono::steady_clock::now();
      bool valid = true;
      for (const auto& frame : result["frames"]) {
        moveit::core::RobotState state(*manual_start_);
        for (size_t joint = 0; joint < names_.size(); ++joint)
          state.setVariablePosition(names_[joint], frame.at("joints").at(joint).get<double>());
        state.update(true);
        ++checks;
        if (!state.satisfiesBounds() || manual_scene_->isStateColliding(state)) {
          valid = false;
          result["failure_stage"] = "post_validation_collision";
          break;
        }
      }
      result["success"] = valid;
      result["validation_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - validation_started).count();
    }
    result["checked_edges"] = edges;
    result["collision_checks"] = checks;
    result["wall_ms"] = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return result;
  }

  Json plan(double time_limit_s, int collision_budget, bool keep_wrists_fixed = false,
    bool plan_updown = false)
  {
    if (!std::isfinite(time_limit_s) || time_limit_s <= 0.0 || collision_budget < 1)
      throw std::invalid_argument("invalid RRT budget");
    const auto started = std::chrono::steady_clock::now();
    const auto initial = home_;
    auto space = std::make_shared<ob::RealVectorStateSpace>(plan_updown ? 5 : 4);
    ob::RealVectorBounds bounds(plan_updown ? 5 : 4);
    moveit::core::RobotState named_goal(initial);
    if (plan_updown && !named_goal.setToDefaultValues(
      model_->getJointModelGroup("whole_body"), "second_home"))
      throw std::runtime_error("missing named second_home");
    if (plan_updown) {
      bounds.setLow(4, 0.0);
      bounds.setHigh(4, 1.0);
    }
    std::array<double, 4> signs{};
    for (size_t axis = 0; axis < 4; ++axis) {
      const auto& limit = model_->getVariableBounds("left_joint" + std::to_string(axis + 1));
      bounds.setLow(axis, limit.min_position_);
      bounds.setHigh(axis, limit.max_position_);
      const double left_start = initial.getVariablePosition("left_joint" + std::to_string(axis + 1));
      const double right_start = initial.getVariablePosition("right_joint" + std::to_string(axis + 1));
      const double left_goal = plan_updown ?
        named_goal.getVariablePosition("left_joint" + std::to_string(axis + 1)) : goal_values_[axis];
      const double right_goal = plan_updown ?
        named_goal.getVariablePosition("right_joint" + std::to_string(axis + 1)) : goal_values_[7 + axis];
      if (std::abs(left_goal - left_start) > 1e-8)
        signs[axis] = (right_goal - right_start) / (left_goal - left_start);
      else if (std::abs(left_start) > 1e-8)
        signs[axis] = right_start / left_start;
      else
        signs[axis] = -1.0;
      if (std::abs(std::abs(signs[axis]) - 1.0) > 1e-6)
        throw std::runtime_error("right-arm coupling must have +/-1 sign");
    }
    space->setBounds(bounds);
    size_t fcl_checks = 0;
    size_t wrist_calls = 0;
    size_t checked_edges = 0;
    size_t jump_rejections = 0;
    size_t bound_rejections = 0;
    size_t wrist_rejections = 0;
    size_t collision_rejections = 0;
    size_t bridge_checks = 0;
    size_t bridge_successes = 0;
    size_t coarse_rejections = 0;
    size_t near_refinements = 0;
    double state_update_ms = 0.0;
    double wrist_solve_ms = 0.0;
    double collision_check_ms = 0.0;
    double bridge_collision_ms = 0.0;
    double clearance_ms = 0.0;
    using StateKey = std::array<long long, 5>;
    struct KeyHash {
      size_t operator()(const StateKey& key) const {
        size_t hash = 0;
        for (const auto value : key)
          hash ^= std::hash<long long>{}(value) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
        return hash;
      }
    };
    std::unordered_map<StateKey, std::shared_ptr<moveit::core::RobotState>, KeyHash> cache;
    std::array<Eigen::Quaterniond, 2> transition_start_rotation, transition_goal_rotation;
    std::array<std::array<double, 3>, 2> transition_start_wrist, transition_goal_wrist;
    if (plan_updown) {
      const Eigen::Matrix3d carriage = initial.getGlobalLinkTransform("arm_carriage").linear();
      for (size_t side_index = 0; side_index < 2; ++side_index) {
        const std::string side = side_index == 0 ? "left" : "right";
        transition_start_rotation[side_index] = Eigen::Quaterniond(
          carriage.transpose() * initial.getGlobalLinkTransform(side + "_tool0").linear());
        transition_goal_rotation[side_index] = Eigen::Quaterniond(
          carriage.transpose() * named_goal.getGlobalLinkTransform(side + "_tool0").linear());
        for (size_t wrist = 0; wrist < 3; ++wrist) {
          const std::string name = side + "_joint" + std::to_string(wrist + 5);
          transition_start_wrist[side_index][wrist] = initial.getVariablePosition(name);
          transition_goal_wrist[side_index][wrist] = named_goal.getVariablePosition(name);
        }
      }
    }
    auto lift = [&](const ob::State* value) -> std::shared_ptr<moveit::core::RobotState> {
      const auto* first_four = value->as<ob::RealVectorStateSpace::StateType>()->values;
      StateKey key{};
      for (size_t axis = 0; axis < space->getDimension(); ++axis)
        key[axis] = std::llround(first_four[axis] * 1e9);
      const auto cached = cache.find(key);
      if (cached != cache.end()) return cached->second;
      auto candidate = std::make_shared<moveit::core::RobotState>(initial);
      for (size_t axis = 0; axis < 4; ++axis) {
        const auto number = std::to_string(axis + 1);
        candidate->setVariablePosition("left_joint" + number, first_four[axis]);
        candidate->setVariablePosition("right_joint" + number,
          initial.getVariablePosition("right_joint" + number) + signs[axis] *
          (first_four[axis] - initial.getVariablePosition("left_joint" + number)));
      }
      if (plan_updown) {
        const double phase = std::clamp(first_four[4], 0.0, 1.0);
        const double smooth = phase * phase * (3.0 - 2.0 * phase);
        candidate->setVariablePosition("updown",
          initial.getVariablePosition("updown") * (1.0 - smooth) +
          named_goal.getVariablePosition("updown") * smooth);
      }
      if (plan_updown) {
        bool at_start = std::abs(first_four[4]) < 1e-10;
        bool at_goal = std::abs(first_four[4] - 1.0) < 1e-10;
        for (size_t axis = 0; axis < 4; ++axis) {
          at_start &= std::abs(first_four[axis] - initial.getVariablePosition(
            "left_joint" + std::to_string(axis + 1))) < 1e-10;
          at_goal &= std::abs(first_four[axis] - named_goal.getVariablePosition(
            "left_joint" + std::to_string(axis + 1))) < 1e-10;
        }
        if (at_start || at_goal) {
          *candidate = at_start ? initial : named_goal;
          candidate->update(true);
          ++fcl_checks;
          if (candidate->satisfiesBounds() && !scene_->isStateColliding(*candidate))
            return cache.emplace(key, candidate).first->second;
          return cache.emplace(key, nullptr).first->second;
        }
      }
      auto timed = std::chrono::steady_clock::now();
      candidate->update(true);
      state_update_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - timed).count();
      if (!candidate->satisfiesBounds()) {
        ++bound_rejections;
        return cache.emplace(key, nullptr).first->second;
      }
      if (keep_wrists_fixed) {
        ++fcl_checks;
        const auto collision_started = std::chrono::steady_clock::now();
        const bool colliding = scene_->isStateColliding(*candidate);
        collision_check_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - collision_started).count();
        if (!colliding) return cache.emplace(key, candidate).first->second;
        ++collision_rejections;
        return cache.emplace(key, nullptr).first->second;
      }
      timed = std::chrono::steady_clock::now();
      double orientation_phase = 0.0;
      if (plan_updown) orientation_phase = std::clamp(first_four[4], 0.0, 1.0);
      const Eigen::Matrix3d left_target = plan_updown ?
        transition_start_rotation[0].slerp(
          orientation_phase, transition_goal_rotation[0]).toRotationMatrix() :
        target_rotations_[0];
      const Eigen::Matrix3d right_target = plan_updown ?
        transition_start_rotation[1].slerp(
          orientation_phase, transition_goal_rotation[1]).toRotationMatrix() :
        target_rotations_[1];
      std::array<double, 3> left_seed = previous_wrist_[0];
      std::array<double, 3> right_seed = previous_wrist_[1];
      if (plan_updown)
        for (size_t wrist = 0; wrist < 3; ++wrist) {
          left_seed[wrist] = transition_start_wrist[0][wrist] * (1.0 - orientation_phase) +
            transition_goal_wrist[0][wrist] * orientation_phase;
          right_seed[wrist] = transition_start_wrist[1][wrist] * (1.0 - orientation_phase) +
            transition_goal_wrist[1][wrist] * orientation_phase;
        }
      const auto left_wrist = left_.solveWristOrientation(
        {first_four[0], first_four[1], first_four[2], first_four[3]},
        left_target, left_seed);
      const std::array<double, 4> right_prefix{
        candidate->getVariablePosition("right_joint1"),
        candidate->getVariablePosition("right_joint2"),
        candidate->getVariablePosition("right_joint3"),
        candidate->getVariablePosition("right_joint4")};
      const auto right_wrist = right_.solveWristOrientation(
        right_prefix, right_target, right_seed);
      wrist_solve_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - timed).count();
      wrist_calls += 2;
      if (left_wrist.empty() || right_wrist.empty()) {
        ++wrist_rejections;
        return cache.emplace(key, nullptr).first->second;
      }
      for (const auto& left : left_wrist)
        for (const auto& right : right_wrist) {
          for (size_t joint = 0; joint < 3; ++joint) {
            candidate->setVariablePosition("left_joint" + std::to_string(joint + 5), left.wrist[joint]);
            candidate->setVariablePosition("right_joint" + std::to_string(joint + 5), right.wrist[joint]);
          }
          timed = std::chrono::steady_clock::now();
          candidate->update(true);
          state_update_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - timed).count();
          if (!candidate->satisfiesBounds()) continue;
          ++fcl_checks;
          timed = std::chrono::steady_clock::now();
          const bool colliding = scene_->isStateColliding(*candidate);
          collision_check_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - timed).count();
          if (!colliding)
            return cache.emplace(key, candidate).first->second;
          if (fcl_checks >= static_cast<size_t>(collision_budget)) break;
        }
      ++collision_rejections;
      return cache.emplace(key, nullptr).first->second;
    };

    og::SimpleSetup setup(space);
    setup.setStateValidityChecker([&lift](const ob::State* value) {
      return static_cast<bool>(lift(value));
    });
    setup.getSpaceInformation()->setMotionValidator(
      std::make_shared<WristEdgeValidator>(setup.getSpaceInformation(), lift, scene_,
        checked_edges, jump_rejections, bridge_checks, bridge_successes,
        coarse_rejections, near_refinements, bridge_collision_ms, clearance_ms));
    ob::ScopedState<> start(space);
    ob::ScopedState<> goal(space);
    for (size_t axis = 0; axis < 4; ++axis) {
      start[axis] = initial.getVariablePosition("left_joint" + std::to_string(axis + 1));
      goal[axis] = plan_updown ?
        named_goal.getVariablePosition("left_joint" + std::to_string(axis + 1)) : goal_values_[axis];
    }
    if (plan_updown) {
      start[4] = 0.0;
      goal[4] = 1.0;
    }
    setup.setStartAndGoalStates(start, goal, 1e-9);
    setup.setup();
    Json result = {{"kind", plan_updown ? "v3_home_to_second_five_axis_rrt" :
        keep_wrists_fixed ? "v3_four_axis_rrt_fixed_wrist_joints" :
          "v3_four_axis_rrt_fixed_tool_orientation"},
      {"success", false}, {"direct_valid", false},
      {"frames", Json::array()}, {"scope", plan_updown ?
        "offline only; mirrored J1-J4 plus updown, J5-J7 fixed, full 25-box wall" :
        keep_wrists_fixed ? "offline only; J5-J7 fixed, tool orientation may change" :
        "offline only; wrists hold home tool orientations"},
      {"goal_first_four_deg", Json::array()}, {"goal_original_wrist_deg", Json::array()},
      {"right_coupling_signs", signs}};
    for (size_t axis = 0; axis < 4; ++axis)
      result["goal_first_four_deg"].push_back(goal[axis] * 180.0 / kPi);
    for (size_t side = 0; side < 2; ++side)
      for (size_t joint = 4; joint < 7; ++joint)
        result["goal_original_wrist_deg"].push_back(goal_values_[side * 7 + joint] * 180.0 / kPi);
    result["goal_valid"] = static_cast<bool>(lift(goal.get()));
    result["start_valid"] = static_cast<bool>(lift(start.get()));
    const auto initial_state = lift(start.get());
    const auto final_state = lift(goal.get());
    const auto collect_joints = [&](const moveit::core::RobotState& state) {
      std::vector<double> values;
      for (const auto& name : names_) values.push_back(state.getVariablePosition(name));
      return values;
    };
    if (initial_state) result["start_joints"] = collect_joints(*initial_state);
    if (final_state) result["goal_joints"] = collect_joints(*final_state);
    if (result["goal_valid"].get<bool>() && result["start_valid"].get<bool>()) {
      const auto shortcut_started = std::chrono::steady_clock::now();
      const bool direct = setup.getSpaceInformation()->checkMotion(start.get(), goal.get());
      result["shortcut_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - shortcut_started).count();
      result["direct_valid"] = direct;
      if (!direct) {
        auto planner = std::make_shared<og::RRTConnect>(setup.getSpaceInformation());
        planner->setRange(0.25);
        setup.setPlanner(planner);
        const auto deadline = started + std::chrono::duration<double>(time_limit_s);
        const ob::PlannerTerminationCondition stop([&]() {
          return fcl_checks >= static_cast<size_t>(collision_budget) ||
            std::chrono::steady_clock::now() >= deadline;
        });
        const auto rrt_started = std::chrono::steady_clock::now();
        result["ompl_exact"] = setup.solve(stop) == ob::PlannerStatus::EXACT_SOLUTION;
        result["rrt_ms"] = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - rrt_started).count();
      } else {
        result["ompl_exact"] = true;
      }
    } else {
      result["ompl_exact"] = false;
      result["failure_stage"] = "invalid_endpoint";
    }
    ob::PlannerData data(setup.getSpaceInformation());
    if (!result["direct_valid"].get<bool>() && result["goal_valid"].get<bool>() &&
        result["start_valid"].get<bool>()) setup.getPlannerData(data);
    result["tree_vertices"] = data.numVertices();
    if (data.numVertices()) {
      result["tree_nodes"] = Json::array();
      double nearest = std::numeric_limits<double>::infinity();
      std::queue<unsigned int> pending;
      std::vector<bool> visited(data.numVertices(), false);
      const unsigned int start_index = data.getStartIndex(0);
      if (start_index < data.numVertices()) {
        pending.push(start_index);
        visited[start_index] = true;
      }
      size_t start_vertices = 0;
      while (!pending.empty()) {
        const auto vertex = pending.front();
        pending.pop();
        ++start_vertices;
        const auto* state = data.getVertex(vertex).getState()->as<ob::RealVectorStateSpace::StateType>();
        double distance = 0.0;
        for (size_t axis = 0; axis < 4; ++axis)
          distance += std::pow(state->values[axis] - goal[axis], 2);
        if (distance < nearest) {
          nearest = distance;
          result["nearest_tree_state_deg"] = {
            state->values[0] * 180.0 / kPi, state->values[1] * 180.0 / kPi,
            state->values[2] * 180.0 / kPi, state->values[3] * 180.0 / kPi};
          auto robot = lift(data.getVertex(vertex).getState());
          if (robot) result["nearest_tree_joints"] = collect_joints(*robot);
        }
        result["tree_nodes"].push_back({{"joints_deg", {
            state->values[0] * 180.0 / kPi, state->values[1] * 180.0 / kPi,
            state->values[2] * 180.0 / kPi, state->values[3] * 180.0 / kPi}},
          {"start_tree", true}});
        std::vector<unsigned int> neighbors;
        data.getEdges(vertex, neighbors);
        std::vector<unsigned int> incoming;
        data.getIncomingEdges(vertex, incoming);
        neighbors.insert(neighbors.end(), incoming.begin(), incoming.end());
        for (const auto neighbor : neighbors)
          if (neighbor < visited.size() && !visited[neighbor]) {
            visited[neighbor] = true;
            pending.push(neighbor);
          }
      }
      result["start_tree_vertices"] = start_vertices;
      for (size_t vertex = 0; vertex < data.numVertices(); ++vertex) {
        if (visited[vertex]) continue;
        const auto* state = data.getVertex(vertex).getState()->as<ob::RealVectorStateSpace::StateType>();
        result["tree_nodes"].push_back({{"joints_deg", {
            state->values[0] * 180.0 / kPi, state->values[1] * 180.0 / kPi,
            state->values[2] * 180.0 / kPi, state->values[3] * 180.0 / kPi}},
          {"start_tree", false}});
      }
      if (std::isfinite(nearest))
        result["nearest_tree_distance_deg"] = std::sqrt(nearest) * 180.0 / kPi;
    }
    if (result["ompl_exact"].get<bool>()) {
      const auto validation_started = std::chrono::steady_clock::now();
      const auto* path = result["direct_valid"].get<bool>() ? nullptr : &setup.getSolutionPath();
      const size_t segments = path ? path->getStateCount() - 1 : 1;
      size_t before_validation = fcl_checks;
      bool valid = true;
      for (size_t segment = 0; segment < segments && valid; ++segment) {
        const auto* from = path ? path->getState(segment) : start.get();
        const auto* to = path ? path->getState(segment + 1) : goal.get();
        const auto* start_values = from->as<ob::RealVectorStateSpace::StateType>()->values;
        const auto* goal_values = to->as<ob::RealVectorStateSpace::StateType>()->values;
        double max_delta = 0.0;
        for (size_t axis = 0; axis < 4; ++axis)
          max_delta = std::max(max_delta, std::abs(goal_values[axis] - start_values[axis]));
        if (plan_updown)
          max_delta = std::max(max_delta,
            std::abs(goal_values[4] - start_values[4]) * kPi / (180.0 * .01));
        const int steps = std::max(1, static_cast<int>(
          std::ceil(max_delta / (0.5 * kPi / 180.0))));
        auto* probe = space->allocState();
        for (int step = segment == 0 ? 0 : 1; step <= steps; ++step) {
          space->interpolate(from, to, static_cast<double>(step) / steps, probe);
          auto state = lift(probe);
          if (!state) {
            valid = false;
            result["failure_stage"] = "post_validation_invalid_state";
            break;
          }
          if (!result["frames"].empty()) {
            const Json previous = result["frames"].back()["joints"];
            bool wrist_jump = false;
            for (size_t index = 0; index < 14; ++index)
              if (std::abs(state->getVariablePosition(names_[index]) -
                previous[index].get<double>()) > 10.0 * kPi / 180.0) {
                wrist_jump = true;
              }
            if (wrist_jump) {
              moveit::core::RobotState from(*state);
              for (size_t index = 0; index < 14; ++index)
                from.setVariablePosition(names_[index], previous[index].get<double>());
              from.update(true);
              double max_delta = 0.0;
              for (size_t index = 0; index < 14; ++index)
                max_delta = std::max(max_delta,
                  std::abs(state->getVariablePosition(names_[index]) - previous[index].get<double>()));
              const int bridge_steps = std::max(1, static_cast<int>(
                std::ceil(max_delta / (0.5 * kPi / 180.0))));
              for (int bridge_step = 1; bridge_step < bridge_steps; ++bridge_step) {
                const double alpha = static_cast<double>(bridge_step) / bridge_steps;
                moveit::core::RobotState bridge(from);
                std::vector<double> joints;
                for (size_t index = 0; index < names_.size(); ++index) {
                  const double value = index < 14 ?
                    previous[index].get<double>() * (1.0 - alpha) +
                      state->getVariablePosition(names_[index]) * alpha :
                    state->getVariablePosition(names_[index]);
                  bridge.setVariablePosition(names_[index], value);
                  joints.push_back(value);
                }
                bridge.update(true);
                if (!bridge.satisfiesBounds() || scene_->isStateColliding(bridge)) {
                  valid = false;
                  result["failure_stage"] = "post_validation_bridge_collision";
                  break;
                }
                result["frames"].push_back({{"joints", joints}, {"wrist_transition", true}});
              }
            }
          }
          if (!valid) break;
          std::vector<double> joints;
          for (const auto& name : names_) joints.push_back(state->getVariablePosition(name));
          result["frames"].push_back({{"joints", joints},
            {"left_tcp_position", {state->getGlobalLinkTransform("left_tool0").translation().x(),
              state->getGlobalLinkTransform("left_tool0").translation().y(),
              state->getGlobalLinkTransform("left_tool0").translation().z()}}});
        }
        space->freeState(probe);
      }
      result["success"] = valid;
      result["post_validation_fcl_checks"] = fcl_checks - before_validation;
      result["validation_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - validation_started).count();
    }
    if (!result["success"].get<bool>() && !result.contains("failure_stage"))
      result["failure_stage"] = "rrt_budget_or_no_path";
    result["wrist_calls"] = wrist_calls;
    result["fcl_checks"] = fcl_checks;
    result["checked_edges"] = checked_edges;
    result["jump_rejections"] = jump_rejections;
    result["bound_rejections"] = bound_rejections;
    result["wrist_rejections"] = wrist_rejections;
    result["collision_rejections"] = collision_rejections;
    result["bridge_checks"] = bridge_checks;
    result["bridge_successes"] = bridge_successes;
    result["bridge_collision_ms"] = bridge_collision_ms;
    result["coarse_rejections"] = coarse_rejections;
    result["near_refinements"] = near_refinements;
    result["clearance_ms"] = clearance_ms;
    result["state_update_ms"] = state_update_ms;
    result["wrist_solve_ms"] = wrist_solve_ms;
    result["collision_check_ms"] = collision_check_ms;
    result["wall_ms"] = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return result;
  }

  Json wristBranchExample()
  {
    Json example = { {"kind", "v3_fixed_orientation_wrist_branch_example"},
      {"found", false}, {"joint_names", names_}, {"environment", Json::array()},
      {"wall_boxes", Json::array()} };
    const auto home_prefix = [&](const std::string& side) {
      std::array<double, 4> joints{};
      for (size_t axis = 0; axis < 4; ++axis)
        joints[axis] = home_.getVariablePosition(side + "_joint" + std::to_string(axis + 1));
      return joints;
    };
    for (int side_index = 0; side_index < 2; ++side_index) {
      const std::string side = side_index == 0 ? "left" : "right";
      const auto& solver = side_index == 0 ? left_ : right_;
      auto prefix = home_prefix(side);
      const std::array<double, 3> seed = previous_wrist_[side_index];
      // Fixed, deterministic nearby poses; this is a branch illustration, not a motion plan.
      for (int row = 0; row <= 30; ++row) {
        for (int col = 0; col <= 30; ++col) {
          prefix = home_prefix(side);
          prefix[0] += (row - 15) * kPi / 180.0;
          prefix[1] += (col - 15) * kPi / 180.0;
          const auto solutions = solver.solveWristOrientation(
            prefix, target_rotations_[side_index], seed);
          if (solutions.size() < 2) continue;
          Json poses = Json::array();
          for (const auto& solution : solutions) {
            moveit::core::RobotState state(home_);
            for (size_t axis = 0; axis < 4; ++axis)
              state.setVariablePosition(side + "_joint" + std::to_string(axis + 1), prefix[axis]);
            for (size_t axis = 0; axis < 3; ++axis)
              state.setVariablePosition(side + "_joint" + std::to_string(axis + 5), solution.wrist[axis]);
            state.update(true);
            if (!state.satisfiesBounds()) continue;
            Json joints = Json::array();
            for (const auto& name : names_) joints.push_back(state.getVariablePosition(name));
            poses.push_back({{"joints", joints}, {"collision", collisionReason(scene_, state)},
              {"wrist_deg", {solution.wrist[0] * 180.0 / kPi,
                solution.wrist[1] * 180.0 / kPi, solution.wrist[2] * 180.0 / kPi}}});
          }
          if (poses.size() != 2) continue;
          const bool both_clear = poses[0]["collision"].get<std::string>().empty() &&
            poses[1]["collision"].get<std::string>().empty();
          if (!both_clear) continue;
          example["found"] = true;
          example["side"] = side;
          example["first_four_deg"] = {prefix[0] * 180.0 / kPi,
            prefix[1] * 180.0 / kPi, prefix[2] * 180.0 / kPi, prefix[3] * 180.0 / kPi};
          example["branches"] = poses;
          return example;
        }
      }
    }
    return example;
  }

  Json optimizePath(const Json& original, bool choose_short_wrist)
  {
    if (!original.at("success").get<bool>())
      throw std::invalid_argument("shortcut input must be a validated path");
    const auto& source = original.at("frames");
    if (source.size() < 2) throw std::invalid_argument("shortcut path is too short");
    std::vector<std::vector<double>> positions;
    positions.reserve(source.size());
    for (const auto& frame : source) {
      auto joints = frame.at("joints").get<std::vector<double>>();
      if (joints.size() != names_.size()) throw std::invalid_argument("joint count mismatch");
      positions.push_back(std::move(joints));
    }
    size_t collision_checks = 0;
    size_t attempted_edges = 0;
    const auto check_edge = [&](size_t from, size_t to, bool dense) {
      double max_delta = 0.0;
      for (size_t joint = 0; joint < names_.size(); ++joint)
        max_delta = std::max(max_delta,
          std::abs(positions[to][joint] - positions[from][joint]));
      const double resolution = (dense ? 0.5 : 4.0) * kPi / 180.0;
      const int samples = std::max(1, static_cast<int>(std::ceil(max_delta / resolution)));
      moveit::core::RobotState probe(home_);
      for (int sample = 1; sample <= samples; ++sample) {
        const double fraction = static_cast<double>(sample) / samples;
        for (size_t joint = 0; joint < names_.size(); ++joint)
          probe.setVariablePosition(names_[joint], positions[from][joint] * (1.0 - fraction) +
            positions[to][joint] * fraction);
        probe.update(true);
        ++collision_checks;
        if (!probe.satisfiesBounds() || scene_->isStateColliding(probe)) return false;
      }
      return true;
    };
    const auto started = std::chrono::steady_clock::now();
    std::vector<size_t> waypoints{0};
    Json result = { {"kind", "v3_four_axis_farthest_collision_free_shortcut"},
      {"success", false}, {"frames", Json::array()}, {"waypoints", Json::array()} };
    const auto travel = [&](const std::vector<std::vector<double>>& frames, size_t joint) {
      double sum = 0.0;
      for (size_t frame = 1; frame < frames.size(); ++frame)
        sum += std::abs(frames[frame][joint] - frames[frame - 1][joint]);
      return sum * 180.0 / kPi;
    };
    const size_t left_j7 = 6;
    const size_t right_j7 = 13;
    result["original_j7_travel_deg"] = {travel(positions, left_j7),
      travel(positions, right_j7)};
    result["original_goal_joints"] = positions.back();
    result["goal_wrist_options"] = Json::array();
    const auto goal_prefix = [&](const std::string& side) {
      std::array<double, 4> joints{};
      for (size_t joint = 0; joint < 4; ++joint)
        joints[joint] = positions.back()[side == "left" ? joint : joint + 7];
      return joints;
    };
    const auto left_options = left_.solveWristOrientation(
      goal_prefix("left"), target_rotations_[0], previous_wrist_[0]);
    const auto right_options = right_.solveWristOrientation(
      goal_prefix("right"), target_rotations_[1], previous_wrist_[1]);
    double best_cost = std::numeric_limits<double>::infinity();
    std::vector<double> chosen_goal;
    for (const auto& left : left_options)
      for (const auto& right : right_options) {
        moveit::core::RobotState state(home_);
        for (size_t joint = 0; joint < names_.size(); ++joint)
          state.setVariablePosition(names_[joint], positions.back()[joint]);
        for (size_t joint = 0; joint < 3; ++joint) {
          state.setVariablePosition("left_joint" + std::to_string(joint + 5), left.wrist[joint]);
          state.setVariablePosition("right_joint" + std::to_string(joint + 5), right.wrist[joint]);
        }
        state.update(true);
        const std::string collision = collisionReason(scene_, state);
        result["goal_wrist_options"].push_back({
          {"left_wrist_deg", {left.wrist[0] * 180.0 / kPi,
            left.wrist[1] * 180.0 / kPi, left.wrist[2] * 180.0 / kPi}},
          {"right_wrist_deg", {right.wrist[0] * 180.0 / kPi,
            right.wrist[1] * 180.0 / kPi, right.wrist[2] * 180.0 / kPi}},
          {"collision", collision}, {"within_bounds", state.satisfiesBounds()}});
        const double cost = std::abs(left.wrist[2] - positions.front()[left_j7]) +
          std::abs(right.wrist[2] - positions.front()[right_j7]);
        if (collision.empty() && state.satisfiesBounds() && cost < best_cost) {
          best_cost = cost;
          chosen_goal.clear();
          for (const auto& name : names_)
            chosen_goal.push_back(state.getVariablePosition(name));
        }
      }
    if (choose_short_wrist) {
      if (chosen_goal.empty()) throw std::runtime_error("no valid alternate wrist goal");
      positions.back() = chosen_goal;
    }
    result["selected_goal_joints"] = positions.back();
    size_t current = 0;
    while (current + 1 < positions.size()) {
      size_t farthest = current;
      for (size_t candidate = positions.size() - 1; candidate > current; --candidate) {
        ++attempted_edges;
        if (!check_edge(current, candidate, false)) continue;
        if (!check_edge(current, candidate, true)) continue;
        farthest = candidate;
        break;
      }
      if (farthest == current) {
        result["failure_stage"] = "no_collision_free_next_frame";
        break;
      }
      waypoints.push_back(farthest);
      current = farthest;
    }
    if (choose_short_wrist && current == positions.size() - 1 && waypoints.size() == 3) {
      const size_t original_midpoint = waypoints[1];
      const auto wrist_travel = [&](size_t midpoint) {
        return std::abs(positions[midpoint][left_j7] - positions.front()[left_j7]) +
          std::abs(positions.back()[left_j7] - positions[midpoint][left_j7]) +
          std::abs(positions[midpoint][right_j7] - positions.front()[right_j7]) +
          std::abs(positions.back()[right_j7] - positions[midpoint][right_j7]);
      };
      double best_travel = wrist_travel(original_midpoint);
      auto try_midpoint = [&](size_t midpoint) {
        if (wrist_travel(midpoint) >= best_travel - 1e-9) return;
        ++attempted_edges;
        if (!check_edge(midpoint, positions.size() - 1, false) ||
            !check_edge(midpoint, positions.size() - 1, true)) return;
        ++attempted_edges;
        if (!check_edge(0, midpoint, false) || !check_edge(0, midpoint, true)) return;
        best_travel = wrist_travel(midpoint);
        waypoints[1] = midpoint;
      };
      for (size_t midpoint = 10; midpoint < original_midpoint; midpoint += 10)
        try_midpoint(midpoint);
      const size_t center = waypoints[1];
      for (size_t midpoint = center > 9 ? center - 9 : 1;
           midpoint < std::min(original_midpoint, center + 10); ++midpoint)
        try_midpoint(midpoint);
      result["farthest_only_midpoint"] = original_midpoint;
      result["low_wrist_travel_midpoint"] = waypoints[1];
    }
    if (current == positions.size() - 1) {
      for (size_t segment = 1; segment < waypoints.size(); ++segment) {
        const size_t from = waypoints[segment - 1];
        const size_t to = waypoints[segment];
        double max_delta = 0.0;
        for (size_t joint = 0; joint < names_.size(); ++joint)
          max_delta = std::max(max_delta,
            std::abs(positions[to][joint] - positions[from][joint]));
        const int samples = std::max(1, static_cast<int>(
          std::ceil(max_delta / (0.5 * kPi / 180.0))));
        for (int sample = segment == 1 ? 0 : 1; sample <= samples; ++sample) {
          const double fraction = static_cast<double>(sample) / samples;
          std::vector<double> joints;
          joints.reserve(names_.size());
          for (size_t joint = 0; joint < names_.size(); ++joint)
            joints.push_back(positions[from][joint] * (1.0 - fraction) +
              positions[to][joint] * fraction);
          result["frames"].push_back({{"joints", joints},
            {"source_from", from}, {"source_to", to}});
        }
        result["waypoints"].push_back({from, to});
      }
      result["success"] = true;
    }
    if (result["success"].get<bool>()) {
      std::vector<std::vector<double>> optimized;
      for (const auto& frame : result["frames"])
        optimized.push_back(frame.at("joints").get<std::vector<double>>());
      result["optimized_j7_travel_deg"] = {travel(optimized, left_j7),
        travel(optimized, right_j7)};
    }
    result["attempted_edges"] = attempted_edges;
    result["collision_checks"] = collision_checks;
    result["wall_ms"] = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return result;
  }

  Json validateSynchronizedUpdown(const Json& shortcut, double destination)
  {
    if (!shortcut.at("success").get<bool>() || !std::isfinite(destination))
      throw std::invalid_argument("expected a validated shortcut and finite updown goal");
    const auto& source = shortcut.at("frames");
    if (source.size() < 2) throw std::invalid_argument("shortcut requires at least two frames");
    const size_t updown_index = std::find(names_.begin(), names_.end(), "updown") - names_.begin();
    if (updown_index == names_.size()) throw std::runtime_error("missing updown joint");
    const double initial_updown = source.front().at("joints").at(updown_index).get<double>();
    Json result = { {"kind", "v3_home_transition_synchronized_updown"},
      {"success", true}, {"forward_frames", Json::array()}, {"return_frames", Json::array()},
      {"joint_names", names_}, {"updown_start", initial_updown}, {"updown_end", destination},
      {"updown_profile", "smoothstep"} };
    size_t checks = 0;
    const auto inspect = [&](size_t index, const std::string& direction) {
      auto joints = source.at(index).at("joints").get<std::vector<double>>();
      if (joints.size() != names_.size()) throw std::invalid_argument("joint count mismatch");
      const double progress = static_cast<double>(index) / (source.size() - 1);
      const double eased = progress * progress * (3.0 - 2.0 * progress);
      joints[updown_index] = initial_updown + (destination - initial_updown) * eased;
      moveit::core::RobotState state(home_);
      for (size_t joint = 0; joint < names_.size(); ++joint)
        state.setVariablePosition(names_[joint], joints[joint]);
      state.update(true);
      ++checks;
      const bool bounded = state.satisfiesBounds();
      const std::string collision = collisionReason(scene_, state);
      if (!bounded || !collision.empty()) {
        result["success"] = false;
        result["failure"] = { {"direction", direction}, {"source_frame", index},
          {"updown", joints[updown_index]}, {"within_joint_limits", bounded},
          {"collision_pairs", collision}, {"joints", joints} };
      } else {
        result[direction + "_frames"].push_back({{"joints", joints}, {"source_frame", index}});
      }
    };
    for (size_t index = 0; index < source.size() && result["success"].get<bool>(); ++index)
      inspect(index, "forward");
    for (size_t index = source.size(); index > 0 && result["success"].get<bool>(); --index)
      inspect(index - 1, "return");
    result["collision_checks"] = checks;
    return result;
  }

  Json checkNamedShortcut(const std::string& from_name, const std::string& to_name,
    bool attach_boxes)
  {
    const auto* whole = model_->getJointModelGroup("whole_body");
    if (!whole) throw std::runtime_error("missing whole_body group");
    moveit::core::RobotState start(home_);
    moveit::core::RobotState goal(home_);
    if (!start.setToDefaultValues(whole, from_name) ||
        !goal.setToDefaultValues(whole, to_name))
      throw std::invalid_argument("named shortcut endpoint missing");
    start.update(true);
    goal.update(true);
    double maximum = 0.0;
    for (const auto& name : names_)
      if (name != "updown")
        maximum = std::max(maximum,
          std::abs(goal.getVariablePosition(name) - start.getVariablePosition(name)));
    const int segments = std::max(1, static_cast<int>(
      std::ceil(maximum / (0.5 * kPi / 180.0))));
    Json result = {{"kind", "v3_named_pose_joint_shortcut_collision_diagnostic"},
      {"from", from_name}, {"to", to_name}, {"segments", segments},
      {"joint_names", names_}, {"frames", Json::array()},
      {"collision_frames", 0}, {"out_of_bounds_frames", 0},
      {"attached_boxes", attach_boxes},
      {"box_size_m", {0.30, 0.40, 0.40}}};
    Eigen::Isometry3d tool_to_box = Eigen::Isometry3d::Identity();
    tool_to_box.linear() = Eigen::AngleAxisd(-kPi / 2.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
    tool_to_box.translation() = Eigen::Vector3d(0.0, 0.0, 0.150001);
    const std::vector<shapes::ShapeConstPtr> shapes{
      std::make_shared<shapes::Box>(0.30, 0.40, 0.40)};
    const EigenSTL::vector_Isometry3d poses{tool_to_box};
    for (int index = 0; index <= segments; ++index) {
      moveit::core::RobotState state(home_);
      start.interpolate(goal, static_cast<double>(index) / segments, state);
      state.update(true);
      if (attach_boxes)
        for (const std::string side : {"left", "right"}) {
          const std::string tool = side + "_tool0";
          const std::vector<std::string> touch_links{
            tool, side + "_joint7", side + "_joint6"};
          state.attachBody("carried_" + side, Eigen::Isometry3d::Identity(), shapes,
            poses, touch_links, tool);
        }
      state.update(true);
      const bool bounded = state.satisfiesBounds();
      const std::string contacts = collisionReason(scene_, state);
      if (!bounded) result["out_of_bounds_frames"] = result["out_of_bounds_frames"].get<int>() + 1;
      if (!contacts.empty()) result["collision_frames"] = result["collision_frames"].get<int>() + 1;
      Json values = Json::array();
      for (const auto& name : names_)
        values.push_back(state.getVariablePosition(name));
      result["frames"].push_back({{"index", index}, {"joints", values},
        {"collision_pairs", contacts}, {"within_joint_limits", bounded}});
    }
    result["success"] = result["collision_frames"].get<int>() == 0 &&
      result["out_of_bounds_frames"].get<int>() == 0;
    result["direct_success"] = result["success"];
    if (!result["success"].get<bool>() && !attach_boxes) {
      auto space = std::make_shared<ob::RealVectorStateSpace>(15);
      ob::RealVectorBounds bounds(15);
      for (size_t axis = 0; axis < 15; ++axis) {
        const auto& limit = model_->getVariableBounds(names_[axis]);
        bounds.setLow(axis, limit.min_position_);
        bounds.setHigh(axis, limit.max_position_);
      }
      space->setBounds(bounds);
      og::SimpleSetup setup(space);
      const auto convert = [&](const ob::State* value) {
        moveit::core::RobotState state(start);
        const auto* joints = value->as<ob::RealVectorStateSpace::StateType>()->values;
        for (size_t axis = 0; axis < 15; ++axis)
          state.setVariablePosition(names_[axis], joints[axis]);
        state.update(true);
        return state;
      };
      setup.setStateValidityChecker([&](const ob::State* value) {
        const auto state = convert(value);
        return state.satisfiesBounds() && !scene_->isStateColliding(state);
      });
      setup.getSpaceInformation()->setStateValidityCheckingResolution(.0002);
      ob::ScopedState<> from(space), to(space);
      for (size_t axis = 0; axis < 15; ++axis) {
        from[axis] = start.getVariablePosition(names_[axis]);
        to[axis] = goal.getVariablePosition(names_[axis]);
      }
      setup.setStartAndGoalStates(from, to, 1e-9);
      ompl::RNG::setSeed(1);
      auto planner = std::make_shared<og::RRTConnect>(setup.getSpaceInformation());
      planner->setRange(.25);
      setup.setPlanner(planner);
      setup.setup();
      result["rrt_exact"] = setup.solve(8.0) == ob::PlannerStatus::EXACT_SOLUTION;
      if (result["rrt_exact"].get<bool>()) {
        result["frames"] = Json::array();
        const auto& path = setup.getSolutionPath();
        bool valid = true;
        for (size_t segment = 0; segment + 1 < path.getStateCount() && valid; ++segment) {
          const auto* first = path.getState(segment);
          const auto* last = path.getState(segment + 1);
          const auto* a = first->as<ob::RealVectorStateSpace::StateType>()->values;
          const auto* b = last->as<ob::RealVectorStateSpace::StateType>()->values;
          double maximum = std::abs(b[14] - a[14]) / .005;
          for (size_t axis = 0; axis < 14; ++axis)
            maximum = std::max(maximum,
              std::abs(b[axis] - a[axis]) / (.5 * kPi / 180.0));
          const int count = std::max(1, static_cast<int>(std::ceil(maximum)));
          auto* sample = space->allocState();
          for (int index = segment == 0 ? 0 : 1; index <= count; ++index) {
            space->interpolate(first, last, static_cast<double>(index) / count, sample);
            const auto state = convert(sample);
            const std::string contacts = collisionReason(scene_, state);
            if (!state.satisfiesBounds() || !contacts.empty()) {
              valid = false;
              result["failure_stage"] = "rrt_post_validation";
              result["failure_reason"] = contacts.empty() ? "joint_limit" : contacts;
              break;
            }
            Json values = Json::array();
            for (const auto& name : names_)
              values.push_back(state.getVariablePosition(name));
            result["frames"].push_back({{"index", result["frames"].size()},
              {"joints", values}, {"collision_pairs", ""}, {"within_joint_limits", true}});
          }
          space->freeState(sample);
        }
        result["success"] = valid;
      }
    }
    return result;
  }

  Json benchmarkCollision(int samples)
  {
    if (samples < 1 || samples > 10000) throw std::invalid_argument("invalid benchmark count");
    auto test_scene = std::make_shared<planning_scene::PlanningScene>(model_);
    test_scene->setCurrentState(home_);
    Json rows = Json::array();
    const auto measure = [&](const std::string& label, bool contacts, bool attached,
      const std::string& scope) {
      moveit::core::RobotState state(home_);
      auto diagnostic_acm = test_scene->getAllowedCollisionMatrix();
      if (scope == "self_without_head" || scope == "self_without_body")
        diagnostic_acm.setEntry("head", true);
      if (scope == "self_without_body") {
        for (const std::string link : {"model_base", "chassis_base", "arm_carriage",
          "caster01", "caster02", "caster03", "caster04",
          "wheel01", "wheel02", "wheel03", "wheel04"})
          diagnostic_acm.setEntry(link, true);
      }
      if (scope == "self_without_joint7") {
        diagnostic_acm.setEntry("left_joint7", true);
        diagnostic_acm.setEntry("right_joint7", true);
      }
      if (scope == "self_without_model_base" || scope == "self_without_base_and_carriage")
        diagnostic_acm.setEntry("model_base", true);
      if (scope == "self_without_arm_carriage" || scope == "self_without_base_and_carriage")
        diagnostic_acm.setEntry("arm_carriage", true);
      if (scope == "self_without_chassis_base")
        diagnostic_acm.setEntry("chassis_base", true);
      if (scope == "self_without_wheels")
        for (const std::string link : {"caster01", "caster02", "caster03", "caster04",
          "wheel01", "wheel02", "wheel03", "wheel04"})
          diagnostic_acm.setEntry(link, true);
      for (int index = 1; index <= 4; ++index) {
        const std::string suffix = "0" + std::to_string(index);
        const std::string caster = "caster" + suffix;
        const std::string wheel = "wheel" + suffix;
        if (scope == "self_without_casters") diagnostic_acm.setEntry(caster, true);
        if (scope == "self_without_road_wheels") diagnostic_acm.setEntry(wheel, true);
        if (scope == "self_allow_chassis_casters")
          diagnostic_acm.setEntry("chassis_base", caster, true);
        if (scope == "self_allow_chassis_caster02" && index == 2)
          diagnostic_acm.setEntry("chassis_base", caster, true);
        if (scope == "self_allow_chassis_caster03" && index == 3)
          diagnostic_acm.setEntry("chassis_base", caster, true);
        if (scope == "self_allow_chassis_wheels")
          diagnostic_acm.setEntry("chassis_base", wheel, true);
      }
      if (attached) {
        const std::vector<shapes::ShapeConstPtr> shapes{
          std::make_shared<shapes::Box>(0.30, 0.40, 0.40)};
        Eigen::Isometry3d offset = Eigen::Isometry3d::Identity();
        offset.translation().z() = .150001;
        const EigenSTL::vector_Isometry3d poses{offset};
        for (const std::string side : {"left", "right"}) {
          const std::string tool = side + "_tool0";
          const std::vector<std::string> touch{tool, side + "_joint7", side + "_joint6"};
          state.attachBody("benchmark_box_" + side, Eigen::Isometry3d::Identity(),
            shapes, poses, touch, tool);
        }
      }
      const double initial = state.getVariablePosition("left_joint1");
      std::vector<double> durations;
      durations.reserve(samples);
      int collisions = 0;
      std::string first_contact;
      for (int sample = -64; sample < samples; ++sample) {
        state.setVariablePosition("left_joint1", initial + .01 * std::sin(sample * .07));
        state.update(true);
        collision_detection::CollisionRequest request;
        request.contacts = contacts;
        request.max_contacts = 5;
        request.max_contacts_per_pair = 1;
        collision_detection::CollisionResult collision;
        const auto started = std::chrono::steady_clock::now();
        if (scope == "self" || scope.rfind("self_without_", 0) == 0 ||
            scope.rfind("self_allow_", 0) == 0)
          test_scene->checkSelfCollision(request, collision, state, diagnostic_acm);
        else if (scope == "world")
          test_scene->getCollisionEnv()->checkRobotCollision(
            request, collision, state, test_scene->getAllowedCollisionMatrix());
        else
          test_scene->checkCollision(request, collision, state);
        const double duration = std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - started).count();
        if (sample >= 0) {
          durations.push_back(duration);
          if (collision.collision) {
            ++collisions;
            if (first_contact.empty() && !collision.contacts.empty()) {
              const auto& pair = collision.contacts.begin()->first;
              first_contact = pair.first + "<->" + pair.second;
            }
          }
        }
      }
      std::sort(durations.begin(), durations.end());
      double sum = 0.0;
      for (double duration : durations) sum += duration;
      rows.push_back({{"mode", label}, {"samples", samples}, {"collisions", collisions},
        {"min_us", durations.front()}, {"median_us", durations[durations.size() / 2]},
        {"p95_us", durations[static_cast<size_t>(.95 * (durations.size() - 1))]},
        {"max_us", durations.back()}, {"mean_us", sum / durations.size()},
        {"first_contact", first_contact}});
    };
    measure("v3_robot_only_contacts", true, false, "full");
    measure("v3_robot_only_self", true, false, "self");
    measure("v3_robot_only_self_without_head", true, false, "self_without_head");
    measure("v3_robot_only_self_without_body", true, false, "self_without_body");
    measure("v3_robot_only_self_without_joint7", true, false, "self_without_joint7");
    measure("v3_robot_only_self_without_model_base", true, false, "self_without_model_base");
    measure("v3_robot_only_self_without_arm_carriage", true, false, "self_without_arm_carriage");
    measure("v3_robot_only_self_without_chassis_base", true, false, "self_without_chassis_base");
    measure("v3_robot_only_self_without_wheels", true, false, "self_without_wheels");
    measure("v3_robot_only_self_without_casters", true, false, "self_without_casters");
    measure("v3_robot_only_self_without_road_wheels", true, false, "self_without_road_wheels");
    measure("v3_robot_only_self_allow_chassis_casters", true, false, "self_allow_chassis_casters");
    measure("v3_robot_only_self_allow_chassis_caster02", true, false, "self_allow_chassis_caster02");
    measure("v3_robot_only_self_allow_chassis_caster03", true, false, "self_allow_chassis_caster03");
    measure("v3_robot_only_self_allow_chassis_wheels", true, false, "self_allow_chassis_wheels");
    measure("v3_robot_only_self_without_base_and_carriage", true, false,
      "self_without_base_and_carriage");
    measure("v3_robot_only_world", true, false, "world");
    for (const auto& box : benchmark_environment_)
      addBox(test_scene, "environment_" + box.at("id").get<std::string>(), box);
    measure("v3_five_walls_two_boxes_contacts", true, true, "full");
    measure("v3_five_walls_two_boxes_self", true, true, "self");
    measure("v3_five_walls_two_boxes_world", true, true, "world");
    measure("v3_five_walls_two_boxes_boolean", false, true, "full");
    return {{"kind", "v3_fixed_state_collision_microbenchmark"}, {"rows", rows}};
  }

  Json scanChassisCasters()
  {
    auto test_scene = std::make_shared<planning_scene::PlanningScene>(model_);
    Json pairs = Json::array();
    for (const std::string caster : {"caster02", "caster03"}) {
      auto acm = test_scene->getAllowedCollisionMatrix();
      acm.setEntry(true);
      acm.setEntry("chassis_base", caster, false);
      Json row = {{"pair", "chassis_base<->" + caster},
        {"samples", 0}, {"collisions", 0}, {"first_collision", nullptr},
        {"minimum_clearance_m", std::numeric_limits<double>::infinity()},
        {"closest_sample", nullptr}};
      moveit::core::RobotState state(home_);
      for (int height_step = 0; height_step <= 15; ++height_step)
        for (int yaw_step = 0; yaw_step < 72; ++yaw_step) {
          const double suspension = -0.01 * height_step;
          const double yaw = -kPi + (2.0 * kPi * yaw_step) / 72.0;
          state.setVariablePosition("active_suspension_joint", suspension);
          state.setVariablePosition(caster + "_joint", yaw);
          state.update(true);
          collision_detection::CollisionRequest request;
          request.contacts = true;
          request.max_contacts = 1;
          collision_detection::CollisionResult result;
          test_scene->checkSelfCollision(request, result, state, acm);
          const double clearance = test_scene->getCollisionEnvUnpadded()->distanceSelf(state, acm);
          if (clearance < row["minimum_clearance_m"].get<double>()) {
            row["minimum_clearance_m"] = clearance;
            row["closest_sample"] = {{"suspension_m", suspension},
              {"caster_yaw_deg", yaw * 180.0 / kPi}};
          }
          row["samples"] = row["samples"].get<int>() + 1;
          if (result.collision) {
            row["collisions"] = row["collisions"].get<int>() + 1;
            if (row["first_collision"].is_null())
              row["first_collision"] = {{"suspension_m", suspension},
                {"caster_yaw_deg", yaw * 180.0 / kPi}};
          }
        }
      pairs.push_back(std::move(row));
    }
    return {{"kind", "v3_chassis_caster_pair_range_scan"}, {"pairs", pairs},
      {"scope", "offline 1cm lift and 5deg yaw sampling; not continuous-space proof"}};
  }

  Json planLoadedOrientationPhase(const Json& context, double budget_s)
  {
    if (context.at("initial_pose") != "home" || context.at("wall_context") != "target_only" ||
        !std::isfinite(budget_s) || budget_s <= 0.0)
      throw std::invalid_argument("expected home direct-attach context and positive budget");
    const auto started = std::chrono::steady_clock::now();
    auto scene = std::make_shared<planning_scene::PlanningScene>(model_);
    for (const auto& box : context.at("environment").at("boxes"))
      addBox(scene, box.at("id").get<std::string>(), box);
    moveit::core::RobotState start(home_);
    const auto& attached = context.at("direct_attached_boxes");
    if (attached.size() != 2) throw std::invalid_argument("expected two carried boxes");
    const auto size = context.at("box_size").get<std::vector<double>>();
    if (size.size() != 3) throw std::invalid_argument("invalid box dimensions");
    const std::vector<shapes::ShapeConstPtr> shapes{
      std::make_shared<shapes::Box>(size[0], size[1], size[2])};
    for (const auto& item : attached) {
      const std::string side = item.at("side").get<std::string>();
      if (side != "left" && side != "right") throw std::invalid_argument("invalid attached arm");
      Eigen::Isometry3d offset = Eigen::Isometry3d::Identity();
      const auto center = item.at("tool_to_box_center").get<std::vector<double>>();
      const auto rotation = item.at("tool_to_box_rotation");
      offset.translation() = Eigen::Vector3d(center.at(0), center.at(1), center.at(2));
      for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
          offset.linear()(row, col) = rotation.at(row).at(col).get<double>();
      const std::string tool = side + "_tool0";
      const std::vector<std::string> touch{tool, side + "_joint7"};
      start.attachBody("carried_" + side, Eigen::Isometry3d::Identity(), shapes,
        EigenSTL::vector_Isometry3d{offset}, touch, tool);
    }
    start.update(true);
    moveit::core::RobotState goal(start);
    if (!goal.setToDefaultValues(model_->getJointModelGroup("whole_body"), "unloading"))
      throw std::runtime_error("missing unloading named pose");
    goal.update(true);
    const auto carriage = start.getGlobalLinkTransform("arm_carriage").linear();
    std::array<Eigen::Quaterniond, 2> orientation_start, orientation_goal;
    std::array<std::array<double, 3>, 2> wrist_start, wrist_goal;
    for (size_t index = 0; index < 2; ++index) {
      const std::string side = index == 0 ? "left" : "right";
      const std::string tool = side + "_tool0";
      orientation_start[index] = Eigen::Quaterniond(
        carriage.transpose() * start.getGlobalLinkTransform(tool).linear());
      orientation_goal[index] = Eigen::Quaterniond(
        carriage.transpose() * goal.getGlobalLinkTransform(tool).linear());
      for (int joint = 0; joint < 3; ++joint) {
        const auto name = side + "_joint" + std::to_string(joint + 5);
        wrist_start[index][joint] = start.getVariablePosition(name);
        wrist_goal[index][joint] = goal.getVariablePosition(name);
      }
    }
    Json output = {{"kind", "v3_loaded_four_joint_plus_orientation_phase_rrt"},
      {"success", false}, {"frames", Json::array()},
      {"scope", "offline target_only; two side-suction boxes, mirrored J1-J4, common SO3 phase"},
      {"budget_s", budget_s}, {"start_collision", collisionReason(scene, start)},
      {"goal_collision", collisionReason(scene, goal)}};
    if (!start.satisfiesBounds() || !goal.satisfiesBounds() ||
        !output["start_collision"].get<std::string>().empty() ||
        !output["goal_collision"].get<std::string>().empty()) {
      output["failure_stage"] = "invalid_endpoint";
      return output;
    }

    auto space = std::make_shared<ob::RealVectorStateSpace>(5);
    ob::RealVectorBounds bounds(5);
    std::array<double, 4> mirror{};
    for (int joint = 0; joint < 4; ++joint) {
      const auto name = "left_joint" + std::to_string(joint + 1);
      const auto& limit = model_->getVariableBounds(name);
      bounds.setLow(joint, limit.min_position_);
      bounds.setHigh(joint, limit.max_position_);
      const double delta = goal.getVariablePosition(name) - start.getVariablePosition(name);
      const std::string opposite = "right_joint" + std::to_string(joint + 1);
      mirror[joint] = std::abs(delta) > 1e-9 ?
        (goal.getVariablePosition(opposite) - start.getVariablePosition(opposite)) / delta :
        (joint == 0 || joint == 2 ? -1.0 : 1.0);
      if (std::abs(std::abs(mirror[joint]) - 1.0) > 1e-6)
        throw std::runtime_error("named poses violate mirror coupling");
    }
    bounds.setLow(4, 0.0);
    bounds.setHigh(4, 1.0);
    space->setBounds(bounds);
    const double orientation_angle = std::max(
      orientation_start[0].angularDistance(orientation_goal[0]),
      orientation_start[1].angularDistance(orientation_goal[1]));
    const double phase_step = (kPi / 180.0) / orientation_angle;
    size_t fcl_checks = 0, wrist_calls = 0, collision_rejections = 0;
    size_t wrist_rejections = 0, checked_edges = 0, jump_rejections = 0;
    size_t bridge_checks = 0, bridge_successes = 0, coarse_rejections = 0;
    size_t near_refinements = 0;
    double bridge_collision_ms = 0.0, clearance_ms = 0.0;
    using StateKey = std::array<long long, 5>;
    struct Hash {
      size_t operator()(const StateKey& key) const {
        size_t hash = 0;
        for (auto value : key)
          hash ^= std::hash<long long>{}(value) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
        return hash;
      }
    };
    std::unordered_map<StateKey, std::shared_ptr<moveit::core::RobotState>, Hash> cache;
    const auto lift = [&](const ob::State* state) -> std::shared_ptr<moveit::core::RobotState> {
      const auto* value = state->as<ob::RealVectorStateSpace::StateType>()->values;
      StateKey key{};
      for (size_t axis = 0; axis < key.size(); ++axis)
        key[axis] = std::llround(value[axis] * 1e9);
      const auto cached = cache.find(key);
      if (cached != cache.end()) return cached->second;
      auto robot = std::make_shared<moveit::core::RobotState>(start);
      for (int joint = 0; joint < 4; ++joint) {
        const auto suffix = std::to_string(joint + 1);
        robot->setVariablePosition("left_joint" + suffix, value[joint]);
        robot->setVariablePosition("right_joint" + suffix,
          start.getVariablePosition("right_joint" + suffix) + mirror[joint] *
          (value[joint] - start.getVariablePosition("left_joint" + suffix)));
      }
      robot->update(true);
      if (!robot->satisfiesBounds()) return cache.emplace(key, nullptr).first->second;
      const double phase = value[4];
      const double eased = phase * phase * (3.0 - 2.0 * phase);
      const auto left_rotation = orientation_start[0].slerp(eased, orientation_goal[0]).toRotationMatrix();
      const auto right_rotation = orientation_start[1].slerp(eased, orientation_goal[1]).toRotationMatrix();
      std::array<double, 3> left_seed, right_seed;
      for (int joint = 0; joint < 3; ++joint) {
        left_seed[joint] = wrist_start[0][joint] * (1.0 - eased) + wrist_goal[0][joint] * eased;
        right_seed[joint] = wrist_start[1][joint] * (1.0 - eased) + wrist_goal[1][joint] * eased;
      }
      const auto left = left_.solveWristOrientation(
        {value[0], value[1], value[2], value[3]}, left_rotation, left_seed);
      const auto right = right_.solveWristOrientation(
        {robot->getVariablePosition("right_joint1"), robot->getVariablePosition("right_joint2"),
          robot->getVariablePosition("right_joint3"), robot->getVariablePosition("right_joint4")},
        right_rotation, right_seed);
      wrist_calls += 2;
      if (left.empty() || right.empty()) {
        ++wrist_rejections;
        return cache.emplace(key, nullptr).first->second;
      }
      for (const auto& left_solution : left)
        for (const auto& right_solution : right) {
          for (int joint = 0; joint < 3; ++joint) {
            robot->setVariablePosition("left_joint" + std::to_string(joint + 5),
              left_solution.wrist[joint]);
            robot->setVariablePosition("right_joint" + std::to_string(joint + 5),
              right_solution.wrist[joint]);
          }
          robot->update(true);
          if (!robot->satisfiesBounds()) continue;
          ++fcl_checks;
          if (!scene->isStateColliding(*robot))
            return cache.emplace(key, robot).first->second;
        }
      ++collision_rejections;
      return cache.emplace(key, nullptr).first->second;
    };
    ompl::RNG::setSeed(1);
    og::SimpleSetup setup(space);
    setup.setStateValidityChecker([&lift](const ob::State* state) {
      return static_cast<bool>(lift(state));
    });
    setup.getSpaceInformation()->setMotionValidator(
      std::make_shared<WristEdgeValidator>(setup.getSpaceInformation(), lift, scene,
        checked_edges, jump_rejections, bridge_checks, bridge_successes,
        coarse_rejections, near_refinements, bridge_collision_ms, clearance_ms,
        phase_step, false));
    ob::ScopedState<> from(space);
    ob::ScopedState<> to(space);
    for (int joint = 0; joint < 4; ++joint) {
      const auto name = "left_joint" + std::to_string(joint + 1);
      from[joint] = start.getVariablePosition(name);
      to[joint] = goal.getVariablePosition(name);
    }
    from[4] = 0.0;
    to[4] = 1.0;
    setup.setStartAndGoalStates(from, to, 1e-9);
    setup.setup();
    const auto lifted_start = lift(from.get());
    const auto lifted_goal = lift(to.get());
    output["start_valid"] = static_cast<bool>(lifted_start);
    output["goal_valid"] = static_cast<bool>(lifted_goal);
    const auto endpoint_matches = [&](const moveit::core::RobotState& actual,
      const moveit::core::RobotState& expected) {
      for (const auto& name : names_)
        if (std::abs(actual.getVariablePosition(name) - expected.getVariablePosition(name)) > 1e-5)
          return false;
      return true;
    };
    if (!lifted_start || !lifted_goal || !endpoint_matches(*lifted_start, start) ||
        !endpoint_matches(*lifted_goal, goal)) {
      output["failure_stage"] = "analytic_endpoint_mismatch_or_collision";
    } else {
      const auto shortcut_started = std::chrono::steady_clock::now();
      output["direct_valid"] = setup.getSpaceInformation()->checkMotion(from.get(), to.get());
      output["shortcut_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - shortcut_started).count();
      if (!output["direct_valid"].get<bool>()) {
        auto planner = std::make_shared<og::RRTConnect>(setup.getSpaceInformation());
        planner->setRange(.25);
        setup.setPlanner(planner);
        const auto deadline = started + std::chrono::duration<double>(budget_s);
        const ob::PlannerTerminationCondition stop([&]() {
          return std::chrono::steady_clock::now() >= deadline;
        });
        const auto rrt_started = std::chrono::steady_clock::now();
        output["ompl_exact"] = setup.solve(stop) == ob::PlannerStatus::EXACT_SOLUTION;
        output["rrt_ms"] = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - rrt_started).count();
      } else {
        output["ompl_exact"] = true;
        output["rrt_ms"] = 0.0;
      }
      ob::PlannerData tree(setup.getSpaceInformation());
      if (!output["direct_valid"].get<bool>()) setup.getPlannerData(tree);
      output["tree_vertices"] = tree.numVertices();
      if (output["ompl_exact"].get<bool>()) {
        const auto validation_started = std::chrono::steady_clock::now();
        const auto* path = output["direct_valid"].get<bool>() ? nullptr : &setup.getSolutionPath();
        const size_t segments = path ? path->getStateCount() - 1 : 1;
        bool valid = true;
        double last_phase = 0.0;
        for (size_t segment = 0; segment < segments && valid; ++segment) {
          const auto* first = path ? path->getState(segment) : from.get();
          const auto* last = path ? path->getState(segment + 1) : to.get();
          const auto* first_q = first->as<ob::RealVectorStateSpace::StateType>()->values;
          const auto* last_q = last->as<ob::RealVectorStateSpace::StateType>()->values;
          double steps_metric = std::abs(last_q[4] - first_q[4]) / (phase_step * .5);
          for (int joint = 0; joint < 4; ++joint)
            steps_metric = std::max(steps_metric,
              std::abs(last_q[joint] - first_q[joint]) / (0.5 * kPi / 180.0));
          const int steps = std::max(1, static_cast<int>(std::ceil(steps_metric)));
          auto* sample = space->allocState();
          for (int index = segment == 0 ? 0 : 1; index <= steps; ++index) {
            space->interpolate(first, last, static_cast<double>(index) / steps, sample);
            const double phase = sample->as<ob::RealVectorStateSpace::StateType>()->values[4];
            auto state = lift(sample);
            if (!state) {
              valid = false;
              output["failure_stage"] = "post_validation_collision_or_wrist_ik";
              break;
            }
            if (!output["frames"].empty()) {
              if (phase < last_phase - 1e-6) {
                valid = false;
                output["failure_stage"] = "orientation_progress_reversed";
                break;
              }
              const auto& prior = output["frames"].back()["joints"];
              for (size_t joint = 0; joint < 14; ++joint)
                if (std::abs(state->getVariablePosition(names_[joint]) -
                    prior[joint].get<double>()) > 5.0 * kPi / 180.0) {
                  valid = false;
                  output["failure_stage"] = "wrist_branch_jump";
                  break;
                }
            }
            if (!valid) break;
            Json joints = Json::array();
            for (const auto& name : names_)
              joints.push_back(state->getVariablePosition(name));
            output["frames"].push_back({{"joints", joints}, {"orientation_progress", phase}});
            last_phase = phase;
          }
          space->freeState(sample);
        }
        output["success"] = valid && last_phase >= 1.0 - 1e-6;
        output["validation_ms"] = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - validation_started).count();
      } else {
        output["failure_stage"] = "rrt_budget_or_no_path";
      }
    }
    output["fcl_checks"] = fcl_checks;
    output["wrist_calls"] = wrist_calls;
    output["wrist_rejections"] = wrist_rejections;
    output["collision_rejections"] = collision_rejections;
    output["checked_edges"] = checked_edges;
    output["jump_rejections"] = jump_rejections;
    output["coarse_rejections"] = coarse_rejections;
    output["wall_ms"] = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return output;
  }

private:
  struct ManualBox
  {
    std::string side;
    Eigen::Isometry3d offset;
  };
  moveit::core::RobotModelPtr model_;
  moveit::core::RobotState home_;
  planning_scene::PlanningScenePtr scene_;
  Solver left_;
  Solver right_;
  std::vector<std::string> names_;
  std::vector<double> goal_values_;
  std::array<Eigen::Matrix3d, 2> target_rotations_{};
  std::array<std::array<double, 3>, 2> previous_wrist_{};
  Json benchmark_environment_;
  Json wall_boxes_;
  planning_scene::PlanningScenePtr manual_scene_;
  std::unique_ptr<moveit::core::RobotState> manual_start_;
  std::unique_ptr<moveit::core::RobotState> manual_goal_;
  std::vector<ManualBox> manual_boxes_;
};
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try {
    const auto node = std::make_shared<rclcpp::Node>("v3_wrist_orientation_probe",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
    auto loader = std::make_shared<robot_model_loader::RobotModelLoader>(node, "robot_description");
    if (!loader->getModel()) throw std::runtime_error("failed to load V3 model");
    Probe probe(node, loader->getModel());
    if (node->has_parameter("dual_attached_retreat_output_path")) {
      const Json candidates = Json::parse(
        node->get_parameter("dual_cartesian_candidates_json").as_string());
      const Json planned = Json::parse(
        node->get_parameter("dual_cartesian_pregrasp_plan_json").as_string());
      const Json approach = Json::parse(
        node->get_parameter("dual_cartesian_approach_json").as_string());
      const std::string reference = node->has_parameter("dual_placement_reference_path") ?
        node->get_parameter("dual_placement_reference_path").as_string() : "";
      const int retract_steps = node->has_parameter("dual_unloaded_retract_steps") ?
        static_cast<int>(node->get_parameter("dual_unloaded_retract_steps").as_int()) : 5;
      const Json result = probe.retreatAttachedSharedHeightPair(
        candidates, planned, approach, reference, retract_steps);
      std::ofstream file(node->get_parameter("dual_attached_retreat_output_path").as_string());
      if (!file) throw std::runtime_error("failed to write attached retreat");
      file << result.dump(2) << '\n';
      std::cout << "DUAL_ATTACHED_RETREAT_RESULT " << result.at("success") << '\n' << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    if (node->has_parameter("dual_cartesian_approach_output_path")) {
      const Json candidates = Json::parse(
        node->get_parameter("dual_cartesian_candidates_json").as_string());
      const Json planned = Json::parse(
        node->get_parameter("dual_cartesian_pregrasp_plan_json").as_string());
      const Json result = probe.approachSharedHeightPair(candidates, planned);
      const std::string output = node->get_parameter("dual_cartesian_approach_output_path").as_string();
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write dual approach");
      file << result.dump(2) << '\n';
      std::cout << "DUAL_CARTESIAN_APPROACH_RESULT " << result.at("success") << '\n' << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    if (node->has_parameter("dual_pregrasp_plan_output_path")) {
      const Json candidates = Json::parse(
        node->get_parameter("dual_pregrasp_candidates_json").as_string());
      const Json result = probe.planBestSharedHeightPregrasp(candidates,
        node->get_parameter("dual_pregrasp_budget_s").as_double());
      const std::string output = node->get_parameter("dual_pregrasp_plan_output_path").as_string();
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write dual pregrasp plan");
      file << result.dump(2) << '\n';
      std::cout << "DUAL_PREGRASP_PLAN_RESULT " << result.at("success") << '\n' << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    if (node->has_parameter("dual_face_ik_output_path")) {
      const Json request = Json::parse(node->get_parameter("dual_face_ik_request_json").as_string());
      const Json result = probe.inspectSharedHeightFaceIk(request);
      const std::string output = node->get_parameter("dual_face_ik_output_path").as_string();
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write dual face IK probe");
      file << result.dump(2) << '\n';
      std::cout << "DUAL_FACE_IK_RESULT " << result.at("success") << '\n' << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    if (node->has_parameter("dual_wrist_preview_output_path")) {
      const Json segment = Json::parse(node->get_parameter("dual_wrist_preview_segment_json").as_string());
      const Json result = probe.previewDualExtractedWristTurn(segment);
      const std::string output = node->get_parameter("dual_wrist_preview_output_path").as_string();
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write dual wrist preview");
      file << result.dump(2) << '\n';
      std::cout << "DUAL_WRIST_PREVIEW_RESULT " << result.at("success") << '\n' << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    if (node->has_parameter("wrist_preview_output_path")) {
      const Json recorded = Json::parse(node->get_parameter("wrist_preview_recorded_json").as_string());
      const int box_id = static_cast<int>(node->get_parameter("wrist_preview_box_id").as_int());
      const Json result = probe.previewExtractedBoxWristTurn(recorded, box_id);
      const std::string output = node->get_parameter("wrist_preview_output_path").as_string();
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write wrist preview");
      file << result.dump(2) << '\n';
      std::cout << "WRIST_PREVIEW_RESULT " << result.at("success") << '\n' << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    if (node->has_parameter("manual_loaded_shortcut_output_path")) {
      const Json context = Json::parse(
        node->get_parameter("manual_loaded_four_axis_context_json").as_string());
      probe.configureManualLoaded(context);
      const std::string input = node->get_parameter("manual_loaded_shortcut_input_path").as_string();
      const std::string output = node->get_parameter("manual_loaded_shortcut_output_path").as_string();
      const Json result = probe.shortcutManualLoadedFourD(Json::parse(std::ifstream(input)));
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write loaded shortcut");
      file << result.dump(2) << '\n';
      std::cout << "LOADED_SHORTCUT_RESULT " << result.at("success") << '\n' << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    if (node->has_parameter("manual_loaded_four_axis_output_path")) {
      const Json context = Json::parse(
        node->get_parameter("manual_loaded_four_axis_context_json").as_string());
      probe.configureManualLoaded(context);
      const double budget = node->get_parameter("manual_loaded_four_axis_budget_s").as_double();
      const std::string output = node->get_parameter("manual_loaded_four_axis_output_path").as_string();
      const Json result = probe.planManualLoadedFourD(budget);
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write four-axis loaded search");
      file << result.dump(2) << '\n';
      std::cout << "FOUR_AXIS_LOADED_RESULT " << result.at("success") << '\n' << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    const bool manual_loaded = node->has_parameter("manual_loaded_context_json");
    if (manual_loaded)
      probe.configureManualLoaded(Json::parse(node->get_parameter("manual_loaded_context_json").as_string()));
    if (node->has_parameter("loaded_orientation_output_path")) {
      const std::string output = node->get_parameter("loaded_orientation_output_path").as_string();
      const Json context = Json::parse(node->get_parameter("loaded_orientation_context_json").as_string());
      const double budget = node->get_parameter("loaded_orientation_budget_s").as_double();
      const Json result = probe.planLoadedOrientationPhase(context, budget);
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write loaded orientation search");
      file << result.dump(2) << '\n';
      std::cout << "LOADED_ORIENTATION_RESULT " << result.at("success") << '\n' << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    if (node->has_parameter("caster_scan_output_path")) {
      const std::string output = node->get_parameter("caster_scan_output_path").as_string();
      const Json result = probe.scanChassisCasters();
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write caster range scan");
      file << result.dump(2) << '\n';
      std::cout << "CASTER_RANGE_SCAN " << result.dump() << '\n' << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    if (node->has_parameter("collision_benchmark_output_path")) {
      const std::string output = node->get_parameter("collision_benchmark_output_path").as_string();
      const int samples = static_cast<int>(node->get_parameter("collision_benchmark_samples").as_int());
      const Json result = probe.benchmarkCollision(samples);
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write collision benchmark");
      file << result.dump(2) << '\n';
      std::cout << "COLLISION_BENCHMARK " << result.dump() << '\n' << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    if (node->has_parameter("named_shortcut_output_path")) {
      const std::string output = node->get_parameter("named_shortcut_output_path").as_string();
      const Json result = probe.checkNamedShortcut(
        node->get_parameter("named_shortcut_from").as_string(),
        node->get_parameter("named_shortcut_to").as_string(),
        node->has_parameter("named_shortcut_attached") &&
          node->get_parameter("named_shortcut_attached").as_bool());
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write named shortcut diagnostic");
      file << result.dump(2) << '\n';
      std::cout << "NAMED_SHORTCUT_RESULT " << result.at("success") << '\n' << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    if (node->has_parameter("synchronized_updown_input_path")) {
      const auto input = node->get_parameter("synchronized_updown_input_path").as_string();
      const auto output = node->get_parameter("synchronized_updown_output_path").as_string();
      const double destination = node->get_parameter("synchronized_updown_goal").as_double();
      const Json result = probe.validateSynchronizedUpdown(
        Json::parse(std::ifstream(input)), destination);
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write updown validation");
      file << result.dump(2) << '\n';
      std::cout << "UPDOWN_RESULT " << result.at("success") << ' '
        << result.at("collision_checks") << " checks\n" << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    if (node->has_parameter("optimize_input_path")) {
      const auto input = node->get_parameter("optimize_input_path").as_string();
      const auto output = node->get_parameter("optimize_output_path").as_string();
      const bool short_wrist = node->has_parameter("optimize_short_wrist_goal") &&
        node->get_parameter("optimize_short_wrist_goal").as_bool();
      const Json result = probe.optimizePath(Json::parse(std::ifstream(input)), short_wrist);
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write optimized path");
      file << result.dump(2) << '\n';
      std::cout << "OPTIMIZE_RESULT " << result.at("success") << ' '
        << result.at("wall_ms") << "ms\n" << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    if (node->has_parameter("branch_output_path")) {
      const std::string output = node->get_parameter("branch_output_path").as_string();
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write branch example");
      file << probe.wristBranchExample().dump(2) << '\n';
      rclcpp::shutdown();
      return 0;
    }
    if (node->has_parameter("plan_output_path")) {
      const std::string output = node->get_parameter("plan_output_path").as_string();
      const double time_limit = node->get_parameter("plan_time_s").as_double();
      const int collision_limit = static_cast<int>(node->get_parameter("plan_fcl_budget").as_int());
      const bool keep_wrists_fixed = node->has_parameter("plan_keep_wrists_fixed") &&
        node->get_parameter("plan_keep_wrists_fixed").as_bool();
      const bool plan_updown = node->has_parameter("plan_include_updown") &&
        node->get_parameter("plan_include_updown").as_bool();
      Json result = probe.plan(time_limit, collision_limit, keep_wrists_fixed, plan_updown);
      const auto folder = std::filesystem::path(output).parent_path();
      if (!folder.empty()) std::filesystem::create_directories(folder);
      std::ofstream file(output);
      if (!file) throw std::runtime_error("failed to write four-axis RRT result");
      file << result.dump(2) << '\n';
      std::cout << "RRT_RESULT " << result.dump() << '\n' << std::flush;
      rclcpp::shutdown();
      return 0;
    }
    std::cout << "WRIST_READY\n" << std::flush;
    std::string line;
    while (std::getline(std::cin, line)) {
      try {
        std::cout << "WRIST_RESULT " << (manual_loaded ?
          probe.evaluateManualLoaded(Json::parse(line)) : probe.evaluate(Json::parse(line))).dump() << '\n'
                  << std::flush;
      } catch (const std::exception& error) {
        std::cout << "WRIST_ERROR " << error.what() << '\n' << std::flush;
      }
    }
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
}
