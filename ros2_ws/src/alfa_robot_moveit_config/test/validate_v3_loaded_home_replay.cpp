#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <geometric_shapes/shapes.h>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

using Json = nlohmann::json;

static void require(bool condition, const std::string& message)
{
  if (!condition) throw std::runtime_error(message);
}

static std::string readFile(const std::string& path)
{
  std::ifstream input(path);
  require(input.good(), "cannot read " + path);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

static Eigen::Vector3d vector3(const Json& value)
{
  return {value.at(0).get<double>(), value.at(1).get<double>(), value.at(2).get<double>()};
}

static void addBox(planning_scene::PlanningScene& scene, const std::string& id,
                   const Eigen::Vector3d& center, const Eigen::Vector3d& size)
{
  moveit_msgs::msg::CollisionObject object;
  object.header.frame_id = "world";
  object.id = id;
  object.operation = moveit_msgs::msg::CollisionObject::ADD;
  shape_msgs::msg::SolidPrimitive shape;
  shape.type = shape_msgs::msg::SolidPrimitive::BOX;
  shape.dimensions = {size.x(), size.y(), size.z()};
  geometry_msgs::msg::Pose pose;
  pose.orientation.w = 1.0;
  pose.position.x = center.x(); pose.position.y = center.y(); pose.position.z = center.z();
  object.primitives.push_back(shape);
  object.primitive_poses.push_back(pose);
  require(scene.processCollisionObjectMsg(object), "cannot add " + id);
}

static Json validateCycle(const Json& cycle, const Json& frames,
                          const moveit::core::RobotModelPtr& model,
                          const std::vector<Eigen::Vector3d>& wall)
{
  const auto names = cycle.at("joint_names").get<std::vector<std::string>>();
  const auto initial = cycle.at("initial_joints").get<std::vector<double>>();
  const Eigen::Vector3d size = vector3(cycle.at("box_size"));
  const auto ids = cycle.at("box_ids").get<std::vector<int>>();
  const auto removed = cycle.at("removed_box_ids").get<std::set<int>>();
  require(!frames.empty(), "missing cycle frames");
  planning_scene::PlanningScene scene(model);
  require(cycle.at("environment").at("enabled"), "environment disabled");
  for (const auto& obstacle : cycle.at("environment").at("boxes"))
    addBox(scene, obstacle.at("id"), vector3(obstacle.at("center")), vector3(obstacle.at("size")));
  for (int id = 0; id < 25; ++id)
    if (!removed.count(id)) addBox(scene, "wall_" + std::to_string(id), wall.at(id), size);

  Json payloads = frames.front().at("carried_boxes");
  if (payloads.empty()) payloads.push_back({
    {"box_id", ids.at(0)}, {"side", cycle.at("side")}, {"tool_link", cycle.at("tool_link")},
    {"box_center", cycle.at("box_center")}, {"tool_to_box_center", cycle.at("tool_to_box_center")},
    {"tool_to_box_rotation", cycle.at("tool_to_box_rotation")}});
  std::vector<Eigen::Isometry3d> offsets;
  for (const auto& payload : payloads) {
    const int id = payload.at("box_id");
    const std::string side = payload.at("side"), tool = payload.at("tool_link");
    scene.getAllowedCollisionMatrixNonConst().setEntry("wall_" + std::to_string(id), tool, true);
    scene.getAllowedCollisionMatrixNonConst().setEntry("wall_" + std::to_string(id), side + "_joint7", true);
    Eigen::Isometry3d offset = Eigen::Isometry3d::Identity();
    offset.translation() = vector3(payload.at("tool_to_box_center"));
    for (int row = 0; row < 3; ++row)
      for (int column = 0; column < 3; ++column)
        offset.linear()(row, column) = payload.at("tool_to_box_rotation").at(row).at(column).get<double>();
    offsets.push_back(offset);
  }
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  for (const auto& name : model->getVariableNames()) state.setVariablePosition(name, 0.0);
  bool attached = false;
  size_t probes = 0;
  Json contacts = Json::array();
  int first_invalid = -1;
  auto previous = initial;
  for (size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
    const auto& frame = frames.at(frame_index);
    const auto joints = frame.at("joints").get<std::vector<double>>();
    require(joints.size() == names.size(), "wrong joint count");
    const bool now_attached = frame.at("box_attached");
    if (now_attached != attached) {
      require(now_attached, "payload released before loaded home");
      for (size_t joint = 0; joint < names.size(); ++joint)
        require(std::abs(joints[joint] - previous[joint]) < 1e-8, "attachment teleport");
      for (size_t joint = 0; joint < names.size(); ++joint) state.setVariablePosition(names[joint], joints[joint]);
      state.update(true);
      for (size_t payload_index = 0; payload_index < payloads.size(); ++payload_index) {
        const auto& payload = payloads.at(payload_index);
        const std::string tool = payload.at("tool_link"), side = payload.at("side");
        const int id = payload.at("box_id");
        const Eigen::Isometry3d world_box = state.getGlobalLinkTransform(tool) * offsets.at(payload_index);
        require((world_box.translation() - wall.at(id)).norm() < 1e-5 &&
          (world_box.linear() - Eigen::Matrix3d::Identity()).norm() < 1e-5, "payload jumped at contact");
        scene.getWorldNonConst()->removeObject("wall_" + std::to_string(id));
        EigenSTL::vector_Isometry3d poses{offsets.at(payload_index)};
        state.attachBody("wall_" + std::to_string(id), Eigen::Isometry3d::Identity(),
          {shapes::ShapeConstPtr(new shapes::Box(size.x(), size.y(), size.z()))}, poses,
          std::set<std::string>{tool, side + "_joint7"}, tool);
      }
      attached = true;
    }
    require(frame.at("box_visible"), "payload disappeared before completion");
    const std::string stage = frame.at("stage");
    require(stage != "rear_placement" && stage != "release_box" && stage != "dual_7" &&
      stage != "dual_8", "placement stages still present");
    size_t steps = 1;
    for (size_t joint = 0; joint < names.size(); ++joint)
      steps = std::max(steps, static_cast<size_t>(std::ceil(std::abs(joints[joint] - previous[joint]) /
        (names[joint] == "updown" ? 0.0025 : 0.5 * std::acos(-1.0) / 180.0))));
    for (size_t step = 1; step <= steps; ++step) {
      for (size_t joint = 0; joint < names.size(); ++joint)
        state.setVariablePosition(names[joint], previous[joint] +
          (joints[joint] - previous[joint]) * static_cast<double>(step) / steps);
      state.update(true);
      collision_detection::CollisionRequest request;
      collision_detection::CollisionResult result;
      request.contacts = true;
      request.max_contacts = 20;
      request.max_contacts_per_pair = 1;
      scene.checkCollision(request, result, state);
      ++probes;
      if (result.collision || !state.satisfiesBounds()) {
        first_invalid = static_cast<int>(frame_index);
        for (const auto& pair : result.contacts)
          for (const auto& contact : pair.second)
            contacts.push_back({{"bodies", {pair.first.first, pair.first.second}},
              {"depth_m", contact.depth}, {"position", {contact.pos.x(), contact.pos.y(), contact.pos.z()}}});
        require(!cycle.at("success").get<bool>(), "successful trajectory collides at frame " +
          std::to_string(frame_index) + " " + contacts.dump());
        require(frame_index + 1 == frames.size(), "failed diagnostic crosses a rejected state");
        break;
      }
    }
    if (first_invalid >= 0) break;
    previous = joints;
  }
  if (cycle.at("success")) {
    require(attached, "successful cycle has no payload");
    for (size_t joint = 0; joint < names.size(); ++joint)
      require(std::abs(previous[joint] - initial[joint]) < 1e-8,
        "loaded return did not restore " + names[joint]);
  }
  return {{"box_ids", ids}, {"planned_success", cycle.at("success")},
    {"checked_frames", frames.size()}, {"probes", probes}, {"first_invalid_frame", first_invalid},
    {"contacts", contacts}, {"successful_path_verified", cycle.at("success")}};
}

int main(int argc, char** argv)
{
  try {
    require(argc == 5, "usage: validator sequence.json robot.urdf robot.srdf report.json");
    const auto task = Json::parse(readFile(argv[1]));
    require(task.at("post_extract_policy") == "loaded_home" &&
      !task.at("release_after_transfer").get<bool>(), "wrong completion policy");
    auto urdf = urdf::parseURDF(readFile(argv[2]));
    auto srdf = std::make_shared<srdf::Model>();
    require(urdf && srdf->initString(*urdf, readFile(argv[3])), "invalid model");
    auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
    std::vector<Eigen::Vector3d> wall{vector3(task.at("box_center"))};
    for (const auto& center : task.at("neighbor_centers")) wall.push_back(vector3(center));
    require(wall.size() == 25, "incomplete initial wall");
    std::sort(wall.begin(), wall.end(), [](const Eigen::Vector3d& first, const Eigen::Vector3d& second) {
      return first.z() == second.z() ? first.y() < second.y() : first.z() < second.z();
    });
    Json checks = Json::array();
    for (const auto& cycle : task.at("boxes")) {
      Json frames = Json::array();
      if (cycle.at("success")) {
        const size_t begin = cycle.at("frame_begin"), end = cycle.at("frame_end");
        for (size_t frame = begin; frame < end; ++frame) frames.push_back(task.at("frames").at(frame));
      } else {
        const auto& all_frames = task.at("diagnostic_frames");
        const size_t scene = all_frames.back().at("scene_index");
        for (const auto& frame : all_frames)
          if (frame.at("scene_index").get<size_t>() == scene) frames.push_back(frame);
      }
      checks.push_back(validateCycle(cycle, frames, model, wall));
    }
    Json report{{"successful_paths_valid", true}, {"cycles", checks},
      {"sampling", "whole robot + wall + container + payloads; <=0.5deg / 2.5mm"}};
    std::ofstream output(argv[4]);
    require(output.good(), "cannot write report");
    output << report.dump(2) << '\n';
    std::cout << report.dump(2) << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
