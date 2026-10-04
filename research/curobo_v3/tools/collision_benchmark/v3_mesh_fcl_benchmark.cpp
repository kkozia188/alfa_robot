#include <geometric_shapes/shapes.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
constexpr std::array<const char*, 15> kJointNames{
  "updown",
  "left_joint1", "left_joint2", "left_joint3", "left_joint4",
  "left_joint5", "left_joint6", "left_joint7",
  "right_joint1", "right_joint2", "right_joint3", "right_joint4",
  "right_joint5", "right_joint6", "right_joint7"};

std::string readText(const std::string& path)
{
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open " + path);
  return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

std::vector<std::array<double, 15>> readStates(const std::string& path)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path);
  uint64_t count = 0;
  stream.read(reinterpret_cast<char*>(&count), sizeof(count));
  std::vector<std::array<double, 15>> states(count);
  stream.read(reinterpret_cast<char*>(states.data()),
    static_cast<std::streamsize>(count * sizeof(states.front())));
  if (!stream) throw std::runtime_error("invalid state file " + path);
  return states;
}

void addBox(planning_scene::PlanningScene& scene, const std::string& name,
  const Eigen::Vector3d& center, const Eigen::Vector3d& size)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = center;
  scene.getWorldNonConst()->addToObject(
    name, std::make_shared<shapes::Box>(size.x(), size.y(), size.z()), pose);
}

void addScene(planning_scene::PlanningScene& scene)
{
  constexpr double front = 1.4;
  for (int box = 0; box < 25; ++box) {
    addBox(scene, "wall_box_" + std::to_string(box),
      Eigen::Vector3d(front + 0.15, (box % 5 - 2) * 0.41, 0.20 + (box / 5) * 0.41),
      Eigen::Vector3d(0.30, 0.40, 0.40));
  }
  const double wall_back = front + 0.30 + 1e-6;
  addBox(scene, "left_wall", Eigen::Vector3d(wall_back - 2.0, -1.25, 1.2),
    Eigen::Vector3d(4.0, 0.1, 2.4));
  addBox(scene, "right_wall", Eigen::Vector3d(wall_back - 2.0, 1.25, 1.2),
    Eigen::Vector3d(4.0, 0.1, 2.4));
  addBox(scene, "front_wall", Eigen::Vector3d(wall_back + 0.05, 0.0, 1.2),
    Eigen::Vector3d(0.1, 2.6, 2.4));
  addBox(scene, "ceiling", Eigen::Vector3d(wall_back - 2.0, 0.0, 2.45),
    Eigen::Vector3d(4.2, 2.6, 0.1));
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc != 6) {
    std::cerr << "usage: v3_mesh_fcl_benchmark URDF SRDF STATES DURATION_S THREADS\n";
    return 2;
  }
  const auto urdf_model = urdf::parseURDF(readText(argv[1]));
  if (!urdf_model) throw std::runtime_error("failed to parse URDF");
  auto srdf_model = std::make_shared<srdf::Model>();
  if (!srdf_model->initString(*urdf_model, readText(argv[2])))
    throw std::runtime_error("failed to parse SRDF");
  auto robot_model = std::make_shared<moveit::core::RobotModel>(urdf_model, srdf_model);
  const auto states = readStates(argv[3]);
  const double duration_s = std::stod(argv[4]);
  const size_t thread_count = static_cast<size_t>(std::stoul(argv[5]));
  std::vector<uint64_t> thread_checks(thread_count, 0);
  std::vector<uint64_t> thread_collisions(thread_count, 0);
  std::atomic<size_t> ready{0};
  std::atomic<bool> run{false};
  std::vector<std::thread> workers;
  workers.reserve(thread_count);
  for (size_t thread_index = 0; thread_index < thread_count; ++thread_index) {
    workers.emplace_back([&, thread_index]() {
      planning_scene::PlanningScene scene(robot_model);
      addScene(scene);
      moveit::core::RobotState state(robot_model);
      state.setToDefaultValues();
      state.setVariablePosition("head_joint", 0.0);
      state.setVariablePosition("head_pitch_joint", 0.0);
      collision_detection::CollisionRequest request;
      request.contacts = false;
      request.distance = false;
      request.verbose = false;
      collision_detection::CollisionResult result;
      size_t index = thread_index * 257 % states.size();
      for (size_t warmup = 0; warmup < 256; ++warmup) {
        const auto& values = states[index];
        for (size_t joint = 0; joint < kJointNames.size(); ++joint)
          state.setVariablePosition(kJointNames[joint], values[joint]);
        state.update(true);
        result.clear();
        scene.checkCollision(request, result, state);
        index = (index + 1) % states.size();
      }
      ready.fetch_add(1);
      while (!run.load()) std::this_thread::yield();
      const auto local_deadline = std::chrono::steady_clock::now() +
        std::chrono::duration<double>(duration_s);
      while (std::chrono::steady_clock::now() < local_deadline) {
        const auto& values = states[index];
        for (size_t joint = 0; joint < kJointNames.size(); ++joint)
          state.setVariablePosition(kJointNames[joint], values[joint]);
        state.update(true);
        result.clear();
        scene.checkCollision(request, result, state);
        thread_collisions[thread_index] += result.collision ? 1 : 0;
        ++thread_checks[thread_index];
        index = (index + 1) % states.size();
      }
    });
  }
  while (ready.load() != thread_count) std::this_thread::yield();
  const auto benchmark_started = std::chrono::steady_clock::now();
  run.store(true);
  for (auto& worker : workers) worker.join();
  uint64_t checks = 0;
  uint64_t collisions = 0;
  for (size_t index = 0; index < thread_count; ++index) {
    checks += thread_checks[index];
    collisions += thread_collisions[index];
  }
  const double elapsed = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - benchmark_started).count();
  size_t collision_shapes = 0;
  size_t collision_links = 0;
  for (const auto* link : robot_model->getLinkModelsWithCollisionGeometry()) {
    ++collision_links;
    collision_shapes += link->getShapes().size();
  }
  std::cout << "{\"backend\":\"moveit_fcl_mesh\",\"checks\":" << checks
            << ",\"collisions\":" << collisions
            << ",\"elapsed_s\":" << elapsed
            << ",\"checks_per_s\":" << checks / elapsed
            << ",\"threads\":" << thread_count
            << ",\"collision_links\":" << collision_links
            << ",\"collision_shapes\":" << collision_shapes
            << ",\"states\":" << states.size() << "}\n";
  return 0;
}
