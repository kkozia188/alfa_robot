#include "alfa_robot_moveit_config/optimized_ik_pipeline.hpp"

#include <cassert>
#include <cmath>

namespace
{

ik_benchmark::UpdownAwareIkCandidate make_candidate(
  bool legal,
  double score,
  size_t h_index,
  size_t seed_index,
  double h,
  double joint1)
{
  ik_benchmark::UpdownAwareIkCandidate candidate;
  candidate.legal = legal;
  candidate.score = score;
  candidate.h_index = h_index;
  candidate.seed_index = seed_index;
  candidate.h = h;
  candidate.full_joint_names = {
    "left_v5_joint1", "left_v5_joint2", "left_v5_joint3",
    "left_v5_joint4", "left_v5_joint5", "left_v5_joint6",
    "right_v5_joint1", "right_v5_joint2", "right_v5_joint3",
    "right_v5_joint4", "right_v5_joint5", "right_v5_joint6",
  };
  candidate.full_joint_values = {
    joint1, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  };
  return candidate;
}

}  // namespace

int main()
{
  using alfa_robot::motion::IkCandidateSelectionStats;
  using alfa_robot::motion::IkCandidateSelector;
  using alfa_robot::motion::IkCandidateSelectorConfig;

  ik_benchmark::UpdownAwareIkResult result;
  result.candidates.push_back(make_candidate(true, 3.0, 0, 2, 0.3, 0.20));
  result.candidates.push_back(make_candidate(false, 0.1, 0, 0, 0.3, 0.00));
  result.candidates.push_back(make_candidate(true, 1.0, 2, 4, 0.3, 0.00));
  result.candidates.push_back(make_candidate(true, 1.0, 1, 3, 0.3, 0.00));
  result.candidates.push_back(make_candidate(true, 2.0, 0, 5, 0.3, 0.004));

  IkCandidateSelectorConfig config;
  config.dedup_enabled = true;
  config.joint_threshold = 1.0 * M_PI / 180.0;
  config.h_threshold = 0.005;
  config.candidate_limit = 2;

  IkCandidateSelector selector(config);
  IkCandidateSelectionStats stats;
  const auto selected = selector.selectLegalFromResult(result, &stats);

  assert(selected.size() == 2);
  assert(selected[0].score == 1.0);
  assert(selected[0].h_index == 1);
  assert(selected[0].seed_index == 3);
  assert(selected[1].score == 3.0);
  assert(stats.enabled);
  assert(stats.input_count == 4);
  assert(stats.unique_count == 2);
  assert(stats.selected_count == 2);
  assert(stats.removed_count == 2);

  return 0;
}
