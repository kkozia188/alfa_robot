#pragma once

#include "alfa_robot_moveit_config/extract_planning_pipeline.hpp"
#include "robot_motion_scene_service/motion_core/scene_geometry.hpp"
#include "ik_benchmark/parallel_updown_aware_ik_solver.h"

#include <moveit/robot_state/robot_state.h>

#include <memory>
#include <string>
#include <vector>

namespace alfa_robot::motion
{

enum class ExtractMonitorPhase
{
  ReadyForIk,
  ReadyForExtract,
  ReadyForLoaded,
  ReadyForFinal,
  Done,
};

struct ExtractMonitorState
{
  int left_box_id = 0;
  int right_box_id = 0;
  std::string prefix;
  AttachedBoxSpec left_box;
  AttachedBoxSpec right_box;
  moveit::core::RobotStatePtr seed_state;
  moveit::core::RobotStatePtr loaded_start_state;
  ik_benchmark::UpdownAwareIkResult ik_result;
  std::vector<ik_benchmark::UpdownAwareIkCandidate> legal_candidates;
  std::vector<moveit::core::RobotStatePtr> candidate_states;
  std::vector<ExtractRolloutTiming> timings;
};

}  // namespace alfa_robot::motion
