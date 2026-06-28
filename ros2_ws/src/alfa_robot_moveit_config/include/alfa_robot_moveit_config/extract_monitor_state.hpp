#pragma once

#include "alfa_robot_moveit_config/extract_planning_pipeline.hpp"
#include "robot_motion_scene_service/motion_core/scene_geometry.hpp"
#include "ik_benchmark/parallel_updown_aware_ik_solver.h"

#include <moveit/robot_state/robot_state.h>

#include <array>
#include <functional>
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

enum class ExtractMonitorStage
{
  Ik,
  Extract,
  Loaded,
  Final,
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

using ExtractMonitorStageRunner = std::function<bool(std::string*)>;

struct ExtractMonitorStageCallbacks
{
  ExtractMonitorStageRunner ik;
  ExtractMonitorStageRunner extract;
  ExtractMonitorStageRunner loaded;
  ExtractMonitorStageRunner final;
};

struct ExtractMonitorFullRunResult
{
  bool success = false;
  std::string message;
  std::array<double, 4> stage_elapsed_ms{0.0, 0.0, 0.0, 0.0};
  double total_elapsed_ms = 0.0;
};

const char* extract_monitor_stage_failure_label(ExtractMonitorStage stage);

ExtractMonitorStage extract_monitor_stage_for_phase(ExtractMonitorPhase phase);

ExtractMonitorPhase extract_monitor_phase_before_running(ExtractMonitorPhase phase);

ExtractMonitorPhase extract_monitor_next_phase_after(ExtractMonitorStage stage);

bool run_extract_monitor_stage(
  ExtractMonitorStage stage,
  const ExtractMonitorStageCallbacks& callbacks,
  const std::function<double()>& last_stage_ms,
  double* elapsed_ms,
  std::string* message);

ExtractMonitorFullRunResult run_extract_monitor_full_sequence(
  const ExtractMonitorStageCallbacks& callbacks,
  const std::function<double()>& last_stage_ms);

}  // namespace alfa_robot::motion
