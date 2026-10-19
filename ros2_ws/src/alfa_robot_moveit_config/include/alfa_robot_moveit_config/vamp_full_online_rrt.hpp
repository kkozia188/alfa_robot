#pragma once

#ifdef ALFA_HAS_VAMP

#include <vamp/collision/validity.hh>
#include <vamp/planning/rrtc.hh>
#include <vamp/random/rng.hh>
#include <vamp/robots/sphere.hh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace alfa_robot::motion {

constexpr size_t kFullOnlineVampDimension = 15U;

struct FullOnlineVampContext
{
  std::function<bool(
    const std::array<float, kFullOnlineVampDimension>&)> validator;
  size_t validity_batches{0U};
  size_t validity_states{0U};
  double fcl_ms{0.0};
};

inline thread_local FullOnlineVampContext* g_full_online_vamp_context = nullptr;

struct FullOnlineVampRobot
{
  static constexpr char name[] = "alfa_v322_full_online_15d_fcl";
  static constexpr size_t dimension = kFullOnlineVampDimension;
  static constexpr size_t n_spheres = 0U;
  static constexpr size_t resolution = 64U;
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

  static void setBounds(
    const std::array<float, dimension>& lower,
    const std::array<float, dimension>& upper) noexcept
  {
    lows = lower;
    highs = upper;
  }

  static void scale_configuration(Configuration& configuration) noexcept
  {
    const Configuration lower(lows);
    const Configuration upper(highs);
    configuration = configuration * (upper - lower) + lower;
  }

  static void descale_configuration(Configuration& configuration) noexcept
  {
    const Configuration lower(lows);
    const Configuration upper(highs);
    configuration = (configuration - lower) / (upper - lower);
  }

  template <size_t rake>
  static void scale_configuration_block(
    ConfigurationBlock<rake>& configuration) noexcept
  {
    for (size_t index = 0U; index < dimension; ++index) {
      configuration[index] = lows[index] +
        configuration[index] * (highs[index] - lows[index]);
    }
  }

  template <size_t rake>
  static void descale_configuration_block(
    ConfigurationBlock<rake>& configuration) noexcept
  {
    for (size_t index = 0U; index < dimension; ++index) {
      configuration[index] =
        (configuration[index] - lows[index]) / (highs[index] - lows[index]);
    }
  }

  static float space_measure() noexcept
  {
    float squared = 0.0F;
    for (size_t index = 0U; index < dimension; ++index) {
      const float extent = highs[index] - lows[index];
      squared += extent * extent;
    }
    return std::sqrt(squared);
  }

  template <size_t rake>
  static bool fkcc(
    const vamp::collision::Environment<vamp::FloatVector<rake>>&,
    const ConfigurationBlock<rake>& block) noexcept
  {
    if (!g_full_online_vamp_context ||
      !g_full_online_vamp_context->validator) {
      return false;
    }
    const auto started = std::chrono::steady_clock::now();
    ++g_full_online_vamp_context->validity_batches;
    std::array<std::array<float, rake>, dimension> rows{};
    for (size_t axis = 0U; axis < dimension; ++axis) {
      rows[axis] = block[axis].to_array();
    }
    for (size_t lane = 0U; lane < rake; ++lane) {
      std::array<float, dimension> state{};
      for (size_t axis = 0U; axis < dimension; ++axis) {
        state[axis] = rows[axis][lane];
      }
      ++g_full_online_vamp_context->validity_states;
      if (!g_full_online_vamp_context->validator(state)) {
        g_full_online_vamp_context->fcl_ms +=
          std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count();
        return false;
      }
    }
    g_full_online_vamp_context->fcl_ms +=
      std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return true;
  }

  template <size_t rake>
  static bool fkcc_attach(
    const vamp::collision::Environment<vamp::FloatVector<rake>>& environment,
    const ConfigurationBlock<rake>& block) noexcept
  {
    return fkcc(environment, block);
  }
};

class FullOnlineVampSampler : public vamp::rng::RNG<FullOnlineVampRobot>
{
public:
  FullOnlineVampSampler(
    uint32_t seed,
    const std::array<float, kFullOnlineVampDimension>& start_normalized,
    const std::array<float, kFullOnlineVampDimension>& goal_normalized) noexcept
  : initial_seed_(seed), engine_(seed), start_(start_normalized), goal_(goal_normalized)
  {}

  void reset() noexcept override
  {
    engine_.seed(initial_seed_);
    normal_.reset();
  }

  FullOnlineVampRobot::Configuration next() noexcept override
  {
    std::array<float, kFullOnlineVampDimension> values{};
    if (uniform_(engine_) < 0.8F) {
      const float progress = uniform_(engine_);
      for (size_t axis = 0U; axis < values.size(); ++axis) {
        values[axis] = std::clamp(
          start_[axis] + progress * (goal_[axis] - start_[axis]) +
          0.08F * normal_(engine_), 0.0F, 1.0F);
      }
    } else {
      for (auto& value : values) value = uniform_(engine_);
    }
    FullOnlineVampRobot::Configuration configuration(values);
    FullOnlineVampRobot::scale_configuration(configuration);
    return configuration;
  }

private:
  uint32_t initial_seed_;
  std::mt19937 engine_;
  std::uniform_real_distribution<float> uniform_{0.0F, 1.0F};
  std::normal_distribution<float> normal_{0.0F, 1.0F};
  std::array<float, kFullOnlineVampDimension> start_{};
  std::array<float, kFullOnlineVampDimension> goal_{};
};

struct FullOnlineVampTrial
{
  bool solved{false};
  uint32_t seed{0U};
  size_t iterations{0U};
  size_t tree_nodes{0U};
  size_t validity_batches{0U};
  size_t validity_states{0U};
  double search_ms{0.0};
  double fcl_ms{0.0};
  std::vector<std::array<float, kFullOnlineVampDimension>> path;
};

inline std::vector<FullOnlineVampTrial> solveFullOnlineVampRrtc(
  const std::array<float, kFullOnlineVampDimension>& start_values,
  const std::array<float, kFullOnlineVampDimension>& goal_values,
  const std::array<float, kFullOnlineVampDimension>& lows,
  const std::array<float, kFullOnlineVampDimension>& highs,
  const std::function<bool(
    const std::array<float, kFullOnlineVampDimension>&)>& validator,
  size_t workers,
  size_t max_iterations,
  size_t max_samples,
  float range,
  uint32_t base_seed)
{
  using Robot = FullOnlineVampRobot;
  static constexpr size_t kRake = vamp::FloatVectorWidth;
  using Rrt = vamp::planning::RRTC<Robot, kRake, Robot::resolution>;
  using Environment =
    vamp::collision::Environment<vamp::FloatVector<kRake>>;
  Robot::setBounds(lows, highs);
  std::array<float, kFullOnlineVampDimension> start_normalized{};
  std::array<float, kFullOnlineVampDimension> goal_normalized{};
  for (size_t axis = 0U; axis < Robot::dimension; ++axis) {
    const float extent = highs[axis] - lows[axis];
    start_normalized[axis] = (start_values[axis] - lows[axis]) / extent;
    goal_normalized[axis] = (goal_values[axis] - lows[axis]) / extent;
  }
  const Robot::Configuration start(start_values);
  const Robot::Configuration goal(goal_values);
  vamp::planning::RRTCSettings settings;
  settings.range = range;
  settings.dynamic_domain = false;
  settings.max_iterations = max_iterations;
  settings.max_samples = max_samples;
  const Environment environment;
  std::vector<std::future<FullOnlineVampTrial>> futures;
  for (size_t worker = 0U; worker < workers; ++worker) {
    futures.push_back(std::async(
      std::launch::async,
      [=]() {
        FullOnlineVampTrial trial;
        trial.seed = base_seed + static_cast<uint32_t>(worker) * 0x9E3779B9U;
        FullOnlineVampContext context;
        context.validator = validator;
        g_full_online_vamp_context = &context;
        auto rng = std::make_shared<FullOnlineVampSampler>(
          trial.seed, start_normalized, goal_normalized);
        const auto result = Rrt::solve(start, goal, environment, settings, rng);
        g_full_online_vamp_context = nullptr;
        trial.solved = result.path.size() >= 2U;
        trial.iterations = result.iterations;
        trial.tree_nodes = std::accumulate(
          result.size.begin(), result.size.end(), size_t{0U});
        trial.validity_batches = context.validity_batches;
        trial.validity_states = context.validity_states;
        trial.search_ms = static_cast<double>(result.nanoseconds) / 1.0e6;
        trial.fcl_ms = context.fcl_ms;
        for (const auto& configuration : result.path) {
          const auto values = configuration.to_array();
          std::array<float, kFullOnlineVampDimension> state{};
          std::copy_n(values.begin(), Robot::dimension, state.begin());
          trial.path.push_back(state);
        }
        return trial;
      }));
  }
  std::vector<FullOnlineVampTrial> trials;
  for (auto& future : futures) trials.push_back(future.get());
  return trials;
}

}  // namespace alfa_robot::motion

#endif  // ALFA_HAS_VAMP
