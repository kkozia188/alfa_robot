#pragma once

#ifdef ALFA_HAS_VAMP_PAPER_KERNEL

#include <alfa_v322_fk.hh>
#include <alfa_v322_right_fk.hh>

#include <vamp/collision/environment.hh>
#include <vamp/collision/factory.hh>
#include <vamp/planning/rrtc.hh>
#include <vamp/random/halton.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace alfa_robot::motion
{

constexpr std::size_t kPaperVampDimension = vamp::robots::AlfaV322::dimension;
using PaperVampState = std::array<float, kPaperVampDimension>;

struct PaperVampCuboid
{
  std::array<float, 3> center{};
  std::array<float, 3> euler_xyz{};
  std::array<float, 3> size{};
};

struct PaperVampAttachment
{
  std::array<float, 3> center{};
  std::array<float, 4> quaternion_xyzw{0.0F, 0.0F, 0.0F, 1.0F};
  std::array<float, 3> size{};
  std::array<std::size_t, 3> grid{3U, 4U, 4U};
};

struct PaperVampResult
{
  bool solved{false};
  bool start_valid{false};
  bool goal_valid{false};
  bool start_robot_attachment_valid{false};
  bool goal_robot_attachment_valid{false};
  bool start_robot_self_valid{false};
  bool goal_robot_self_valid{false};
  std::vector<std::string> start_robot_environment_collisions;
  std::vector<std::string> goal_robot_environment_collisions;
  std::vector<std::string> start_attachment_environment_collisions;
  std::vector<std::string> goal_attachment_environment_collisions;
  std::array<float, 3> start_attachment_center{};
  std::array<float, 3> goal_attachment_center{};
  std::array<float, 3> start_attachment_min{};
  std::array<float, 3> start_attachment_max{};
  float attachment_sphere_radius{0.0F};
  std::uint64_t search_ns{0U};
  std::size_t iterations{0U};
  std::size_t tree_nodes{0U};
  float cost{0.0F};
  std::vector<std::string> start_environment_collisions;
  std::vector<std::pair<std::size_t, std::size_t>> start_self_collisions;
  std::vector<std::string> goal_environment_collisions;
  std::vector<std::pair<std::size_t, std::size_t>> goal_self_collisions;
  std::vector<std::array<float, 5>> goal_self_collision_details;
  std::vector<PaperVampState> path;
};

template<typename Robot>
class PaperBoundedHalton final : public vamp::rng::RNG<Robot>
{
public:
  PaperBoundedHalton(const PaperVampState& lower, const PaperVampState& upper)
  : lower_(lower), upper_(upper)
  {}

  void reset() noexcept override {halton_.reset();}

  auto next() noexcept -> vamp::FloatVector<Robot::dimension> override
  {
    auto normalized = halton_.next();
    Robot::descale_configuration(normalized);
    auto values = normalized.to_array();
    for (std::size_t index = 0U; index < Robot::dimension; ++index) {
      values[index] = lower_[index] + values[index] * (upper_[index] - lower_[index]);
    }
    const typename Robot::Configuration output(values.data());
    return output;
  }

private:
  PaperVampState lower_{};
  PaperVampState upper_{};
  vamp::rng::Halton<Robot> halton_;
};

template<typename Robot>
class PaperCorridorSampler final : public vamp::rng::RNG<Robot>
{
public:
  PaperCorridorSampler(
    const PaperVampState& start,
    const PaperVampState& goal,
    const PaperVampState& lower,
    const PaperVampState& upper,
    std::uint32_t seed)
  : initial_seed_(seed), engine_(seed), start_(start), goal_(goal),
    lower_(lower), upper_(upper)
  {}

  void reset() noexcept override
  {
    engine_.seed(initial_seed_);
  }

  auto next() noexcept -> vamp::FloatVector<Robot::dimension> override
  {
    PaperVampState values{};
    if (uniform_(engine_) < 0.9F) {
      const float progress = uniform_(engine_);
      for (std::size_t index = 0U; index < Robot::dimension; ++index) {
        values[index] = std::abs(goal_[index] - start_[index]) <= 1e-6F ?
          start_[index] :
          start_[index] + progress * (goal_[index] - start_[index]);
      }
    } else {
      for (std::size_t index = 0U; index < Robot::dimension; ++index) {
        values[index] = std::abs(goal_[index] - start_[index]) <= 1e-6F ?
          start_[index] : lower_[index] +
          uniform_(engine_) * (upper_[index] - lower_[index]);
      }
    }
    return typename Robot::Configuration(values);
  }

private:
  std::uint32_t initial_seed_;
  std::mt19937 engine_;
  std::uniform_real_distribution<float> uniform_{0.0F, 1.0F};
  PaperVampState start_{};
  PaperVampState goal_{};
  PaperVampState lower_{};
  PaperVampState upper_{};
};

template<typename Robot>
inline auto solvePaperVampRrtcFor(
  const PaperVampState& start,
  const PaperVampState& goal,
  const std::vector<PaperVampCuboid>& cuboids,
  const PaperVampAttachment* attachment,
  float range,
  std::size_t max_iterations,
  std::size_t max_samples,
  std::size_t halton_skip,
  const PaperVampState* lower,
  const PaperVampState* upper) -> PaperVampResult
{
  static constexpr std::size_t kRake = vamp::FloatVectorWidth;
  using VectorEnvironment =
    vamp::collision::Environment<vamp::FloatVector<kRake>>;
  using Rrtc = vamp::planning::RRTC<Robot, kRake, Robot::resolution>;

  vamp::collision::Environment<float> scalar_environment;
  scalar_environment.cuboids.reserve(cuboids.size());
  for (std::size_t cuboid_index = 0U; cuboid_index < cuboids.size(); ++cuboid_index) {
    const auto& cuboid = cuboids[cuboid_index];
    const std::array<float, 3> half_extents{
      0.5F * cuboid.size[0],
      0.5F * cuboid.size[1],
      0.5F * cuboid.size[2],
    };
    scalar_environment.cuboids.emplace_back(
      vamp::collision::factory::cuboid::array(
        cuboid.center, cuboid.euler_xyz, half_extents));
    scalar_environment.cuboids.back().name = "cuboid_" + std::to_string(cuboid_index);
  }
  if (attachment) {
    Eigen::Isometry3f transform = Eigen::Isometry3f::Identity();
    transform.translation() = Eigen::Vector3f(
      attachment->center[0], attachment->center[1], attachment->center[2]);
    transform.linear() = Eigen::Quaternionf(
      attachment->quaternion_xyzw[3], attachment->quaternion_xyzw[0],
      attachment->quaternion_xyzw[1], attachment->quaternion_xyzw[2]).toRotationMatrix();
    scalar_environment.attachments.emplace(transform);
    const std::array<float, 3> cell{
      attachment->size[0] / static_cast<float>(attachment->grid[0]),
      attachment->size[1] / static_cast<float>(attachment->grid[1]),
      attachment->size[2] / static_cast<float>(attachment->grid[2]),
    };
    const float radius = 0.5F * std::sqrt(
      cell[0] * cell[0] + cell[1] * cell[1] + cell[2] * cell[2]);
    for (std::size_t ix = 0U; ix < attachment->grid[0]; ++ix) {
      for (std::size_t iy = 0U; iy < attachment->grid[1]; ++iy) {
        for (std::size_t iz = 0U; iz < attachment->grid[2]; ++iz) {
          scalar_environment.attachments->spheres.emplace_back(
            -0.5F * attachment->size[0] + (static_cast<float>(ix) + 0.5F) * cell[0],
            -0.5F * attachment->size[1] + (static_cast<float>(iy) + 0.5F) * cell[1],
            -0.5F * attachment->size[2] + (static_cast<float>(iz) + 0.5F) * cell[2],
            radius);
        }
      }
    }
  }
  scalar_environment.sort();
  const VectorEnvironment environment(scalar_environment);
  auto scalar_robot_environment = scalar_environment;
  scalar_robot_environment.spheres.clear();
  scalar_robot_environment.capsules.clear();
  scalar_robot_environment.z_aligned_capsules.clear();
  scalar_robot_environment.cylinders.clear();
  scalar_robot_environment.cuboids.clear();
  scalar_robot_environment.z_aligned_cuboids.clear();
  scalar_robot_environment.heightfields.clear();
  scalar_robot_environment.pointclouds.clear();
  const VectorEnvironment robot_environment(scalar_robot_environment);
  auto scalar_bare_robot_environment = scalar_robot_environment;
  scalar_bare_robot_environment.attachments.reset();
  const VectorEnvironment bare_robot_environment(scalar_bare_robot_environment);

  const auto state_valid = [](const VectorEnvironment& checked_environment,
      const PaperVampState& values) {
      const typename Robot::Configuration configuration(values);
      typename Robot::template ConfigurationBlock<kRake> block;
      for (std::size_t axis = 0U; axis < Robot::dimension; ++axis) {
        block[axis] = configuration.broadcast(axis);
      }
      return checked_environment.attachments ?
        Robot::template fkcc_attach<kRake>(checked_environment, block) :
        Robot::template fkcc<kRake>(checked_environment, block);
    };

  const auto debug_state = [&](const PaperVampState& values) {
      const typename Robot::Configuration configuration(values);
      typename Robot::template ConfigurationBlock<kRake> block;
      for (std::size_t axis = 0U; axis < Robot::dimension; ++axis) {
        block[axis] = configuration.broadcast(axis);
      }
      return Robot::template fkcc_debug<kRake>(environment, block);
    };

  std::shared_ptr<vamp::rng::RNG<Robot>> rng;
  if (lower && upper) {
    rng = std::make_shared<PaperCorridorSampler<Robot>>(
      start, goal, *lower, *upper, static_cast<std::uint32_t>(halton_skip + 1U));
  } else {
    rng = std::make_shared<vamp::rng::Halton<Robot>>();
  }
  for (std::size_t index = 0U; index < halton_skip; ++index) {
    static_cast<void>(rng->next());
  }
  vamp::planning::RRTCSettings settings;
  settings.range = range;
  settings.max_iterations = max_iterations;
  settings.max_samples = max_samples;
  settings.dynamic_domain = false;

  PaperVampResult result;
  result.start_valid = state_valid(environment, start);
  if (environment.attachments) {
    result.start_attachment_min.fill(std::numeric_limits<float>::infinity());
    result.start_attachment_max.fill(-std::numeric_limits<float>::infinity());
    for (const auto& sphere : environment.attachments->posed_spheres) {
      const std::array<float, 3> center{
        sphere.x[{0, 0}], sphere.y[{0, 0}], sphere.z[{0, 0}]};
      for (std::size_t axis = 0U; axis < 3U; ++axis) {
        result.start_attachment_center[axis] += center[axis];
        result.start_attachment_min[axis] = std::min(
          result.start_attachment_min[axis], center[axis]);
        result.start_attachment_max[axis] = std::max(
          result.start_attachment_max[axis], center[axis]);
      }
      result.attachment_sphere_radius = sphere.r[{0, 0}];
    }
    if (!environment.attachments->posed_spheres.empty()) {
      const float inverse_count = 1.0F / static_cast<float>(
        environment.attachments->posed_spheres.size());
      for (auto& value : result.start_attachment_center) value *= inverse_count;
    }
    for (const auto& sphere : environment.attachments->posed_spheres) {
      auto collisions = vamp::sphere_environment_get_collisions(
        environment, sphere.x, sphere.y, sphere.z, sphere.r[{0, 0}]);
      result.start_attachment_environment_collisions.insert(
        result.start_attachment_environment_collisions.end(),
        collisions.begin(), collisions.end());
    }
    std::sort(
      result.start_attachment_environment_collisions.begin(),
      result.start_attachment_environment_collisions.end());
    result.start_attachment_environment_collisions.erase(
      std::unique(
        result.start_attachment_environment_collisions.begin(),
        result.start_attachment_environment_collisions.end()),
      result.start_attachment_environment_collisions.end());
  }
  result.goal_valid = state_valid(environment, goal);
  if (environment.attachments) {
    for (const auto& sphere : environment.attachments->posed_spheres) {
      result.goal_attachment_center[0] += sphere.x[{0, 0}];
      result.goal_attachment_center[1] += sphere.y[{0, 0}];
      result.goal_attachment_center[2] += sphere.z[{0, 0}];
    }
    if (!environment.attachments->posed_spheres.empty()) {
      const float inverse_count = 1.0F / static_cast<float>(
        environment.attachments->posed_spheres.size());
      for (auto& value : result.goal_attachment_center) value *= inverse_count;
    }
    for (const auto& sphere : environment.attachments->posed_spheres) {
      auto collisions = vamp::sphere_environment_get_collisions(
        environment, sphere.x, sphere.y, sphere.z, sphere.r[{0, 0}]);
      result.goal_attachment_environment_collisions.insert(
        result.goal_attachment_environment_collisions.end(),
        collisions.begin(), collisions.end());
    }
    std::sort(
      result.goal_attachment_environment_collisions.begin(),
      result.goal_attachment_environment_collisions.end());
    result.goal_attachment_environment_collisions.erase(
      std::unique(
        result.goal_attachment_environment_collisions.begin(),
        result.goal_attachment_environment_collisions.end()),
      result.goal_attachment_environment_collisions.end());
  }
  result.start_robot_attachment_valid = state_valid(robot_environment, start);
  result.goal_robot_attachment_valid = state_valid(robot_environment, goal);
  result.start_robot_self_valid = state_valid(bare_robot_environment, start);
  result.goal_robot_self_valid = state_valid(bare_robot_environment, goal);
  if (result.start_robot_self_valid || result.goal_robot_self_valid) {
    for (const auto& cuboid : scalar_environment.cuboids) {
      vamp::collision::Environment<float> scalar_single_environment;
      scalar_single_environment.cuboids.push_back(cuboid);
      const VectorEnvironment single_environment(scalar_single_environment);
      if (result.start_robot_self_valid && !state_valid(single_environment, start)) {
        result.start_robot_environment_collisions.push_back(cuboid.name);
      }
      if (result.goal_robot_self_valid && !state_valid(single_environment, goal)) {
        result.goal_robot_environment_collisions.push_back(cuboid.name);
      }
    }
  }
  const auto start_debug = debug_state(start);
  const auto goal_debug = debug_state(goal);
  if (!start_debug.first.empty()) {
    result.start_environment_collisions = start_debug.first.front();
  }
  result.start_self_collisions = start_debug.second;
  if (!goal_debug.first.empty()) {
    result.goal_environment_collisions = goal_debug.first.front();
  }
  result.goal_self_collisions = goal_debug.second;
  if (!result.goal_self_collisions.empty()) {
    typename Robot::template ConfigurationBlock<kRake> block;
    const typename Robot::Configuration configuration(goal);
    for (std::size_t axis = 0U; axis < Robot::dimension; ++axis) {
      block[axis] = configuration.broadcast(axis);
    }
    typename Robot::template Spheres<kRake> spheres;
    Robot::template sphere_fk<kRake>(block, spheres);
    for (const auto& pair : result.goal_self_collisions) {
      const float dx = spheres.x[pair.first][{0, 0}] - spheres.x[pair.second][{0, 0}];
      const float dy = spheres.y[pair.first][{0, 0}] - spheres.y[pair.second][{0, 0}];
      const float dz = spheres.z[pair.first][{0, 0}] - spheres.z[pair.second][{0, 0}];
      const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
      const float radii = spheres.r[pair.first][{0, 0}] + spheres.r[pair.second][{0, 0}];
      result.goal_self_collision_details.push_back({
        static_cast<float>(pair.first), static_cast<float>(pair.second),
        distance, radii, radii - distance});
    }
  }
  if (!result.start_valid || !result.goal_valid) {
    return result;
  }

  const typename Robot::Configuration start_configuration(start);
  const typename Robot::Configuration goal_configuration(goal);
  const auto raw = Rrtc::solve(
    start_configuration, goal_configuration, environment, settings, rng);
  result.solved = raw.path.size() >= 2U;
  result.search_ns = raw.nanoseconds;
  result.iterations = raw.iterations;
  result.tree_nodes = std::accumulate(
    raw.size.begin(), raw.size.end(), std::size_t{0U});
  result.cost = raw.cost;
  result.path.reserve(raw.path.size());
  for (const auto& configuration : raw.path) {
    const auto values = configuration.to_array();
    PaperVampState state{};
    std::copy_n(values.begin(), state.size(), state.begin());
    result.path.push_back(state);
  }
  return result;
}

inline auto solvePaperVampRrtc(
  const PaperVampState& start,
  const PaperVampState& goal,
  const std::vector<PaperVampCuboid>& cuboids,
  float range,
  std::size_t max_iterations,
  std::size_t max_samples,
  std::size_t halton_skip,
  const PaperVampAttachment* attachment = nullptr,
  const PaperVampState* lower = nullptr,
  const PaperVampState* upper = nullptr) -> PaperVampResult
{
  return solvePaperVampRrtcFor<vamp::robots::AlfaV322>(
    start, goal, cuboids, attachment, range, max_iterations, max_samples, halton_skip,
    lower, upper);
}

inline auto solvePaperVampRrtcRight(
  const PaperVampState& start,
  const PaperVampState& goal,
  const std::vector<PaperVampCuboid>& cuboids,
  float range,
  std::size_t max_iterations,
  std::size_t max_samples,
  std::size_t halton_skip,
  const PaperVampAttachment* attachment = nullptr,
  const PaperVampState* lower = nullptr,
  const PaperVampState* upper = nullptr) -> PaperVampResult
{
  return solvePaperVampRrtcFor<vamp::robots::AlfaV322Right>(
    start, goal, cuboids, attachment, range, max_iterations, max_samples, halton_skip,
    lower, upper);
}

}  // namespace alfa_robot::motion

#endif  // ALFA_HAS_VAMP_PAPER_KERNEL
