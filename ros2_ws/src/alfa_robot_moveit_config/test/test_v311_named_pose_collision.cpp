#include <ament_index_cpp/get_package_share_directory.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <srdfdom/model.h>
#include <urdf/model.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::string readFile(const std::string& path)
{
  std::ifstream input(path);
  require(input.good(), "failed to open " + path);
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

std::string renderInstalledUrdf()
{
  const std::string path =
    ament_index_cpp::get_package_share_directory("alfa_robot_description") +
    "/urdf/alfa_robot.urdf.xacro";
  const std::string command = "xacro '" + path + "' end_effector:=suction";
  std::unique_ptr<FILE, int (*)(FILE*)> pipe(popen(command.c_str(), "r"), pclose);
  require(pipe != nullptr, "failed to launch xacro");
  std::string output;
  std::array<char, 8192> buffer{};
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
    output += buffer.data();
  }
  require(!output.empty(), "xacro returned an empty URDF");
  return output;
}

moveit::core::RobotModelPtr makeRobotModel()
{
  auto urdf_model = std::make_shared<urdf::Model>();
  require(urdf_model->initString(renderInstalledUrdf()), "failed to parse installed URDF");
  auto srdf_model = std::make_shared<srdf::Model>();
  const std::string path =
    ament_index_cpp::get_package_share_directory("alfa_robot_moveit_config") +
    "/config/alfa_robot.srdf";
  require(srdf_model->initString(*urdf_model, readFile(path)), "failed to parse installed SRDF");
  return std::make_shared<moveit::core::RobotModel>(urdf_model, srdf_model);
}

std::string contacts(const collision_detection::CollisionResult& result)
{
  std::ostringstream output;
  for (const auto& [links, values] : result.contacts) {
    if (output.tellp() > 0) {
      output << ',';
    }
    output << links.first << "<->" << links.second << '(' << values.size() << ')';
  }
  return output.str();
}

}  // namespace

int main()
{
  const auto model = makeRobotModel();
  require(model->getModelFrame() == "map", "MoveIt model frame must be map");
  const auto* root = model->getJointModel("map_to_base_footprint");
  require(root != nullptr, "missing planar root joint");
  require(root->getVariableCount() == 3, "planar root must expose x/y/theta");
  const auto* group = model->getJointModelGroup("whole_body");
  require(group != nullptr, "missing whole_body planning group");
  planning_scene::PlanningScene scene(model);

  for (const std::string pose : {"home", "second_home", "unloading", "second_unloading"}) {
    moveit::core::RobotState state(model);
    state.setToDefaultValues();
    require(state.setToDefaultValues(group, pose), "missing named pose " + pose);
    state.update(true);
    require(state.satisfiesBounds(group), pose + " violates joint bounds");

    collision_detection::CollisionRequest request;
    request.contacts = true;
    request.max_contacts = 100;
    collision_detection::CollisionResult result;
    scene.checkSelfCollision(
      request, result, state, scene.getAllowedCollisionMatrix());
    require(!result.collision, pose + " is self-colliding: " + contacts(result));
  }

  moveit::core::RobotState origin(model);
  origin.setToDefaultValues();
  require(origin.setToDefaultValues(group, "home"), "missing home pose");
  origin.update(true);
  const Eigen::Isometry3d origin_base = origin.getGlobalLinkTransform("base_link");
  const Eigen::Isometry3d origin_tool = origin.getGlobalLinkTransform("left_tool0");

  moveit::core::RobotState moved(origin);
  const double root_values[] = {1.2, -0.4, 0.35};
  moved.setJointPositions(root, root_values);
  moved.update(true);
  const Eigen::Isometry3d moved_base = moved.getGlobalLinkTransform("base_link");
  const Eigen::Isometry3d moved_tool = moved.getGlobalLinkTransform("left_tool0");
  Eigen::Isometry3d expected = Eigen::Isometry3d::Identity();
  expected.translation() = Eigen::Vector3d(root_values[0], root_values[1], 0.0);
  expected.linear() = Eigen::AngleAxisd(root_values[2], Eigen::Vector3d::UnitZ()).toRotationMatrix();
  require((moved_base.matrix() - (expected * origin_base).matrix()).norm() < 1e-9,
    "base_link did not follow the planar root");
  require((moved_tool.matrix() - (expected * origin_tool).matrix()).norm() < 1e-9,
    "TCP did not follow the planar root");
  require(((moved_base.inverse() * moved_tool).matrix() -
    (origin_base.inverse() * origin_tool).matrix()).norm() < 1e-9,
    "base_link-relative FK changed after moving the planar root");
  return 0;
}
