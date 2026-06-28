#pragma once

#include "alfa_robot_moveit_config/extract_planning_pipeline.hpp"
#include "robot_motion_scene_service/motion_core/scene_geometry.hpp"
#include "ik_benchmark/parallel_updown_aware_ik_solver.h"

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <nlohmann/json.hpp>

#include <map>
#include <string>
#include <vector>

namespace alfa_robot::motion
{

nlohmann::json attached_boxes_json(const std::vector<AttachedBoxSpec>& specs);

nlohmann::json robot_state_json(const moveit::core::RobotState& state);

nlohmann::json trajectory_json(const moveit::planning_interface::MoveGroupInterface::Plan& plan);

nlohmann::json extract_monitor_stage_json(
  const std::string& stage_name,
  const moveit::planning_interface::MoveGroupInterface::Plan& plan,
  const moveit::core::RobotState& start_state,
  const moveit::core::RobotState& goal_state,
  const std::vector<std::string>& target_names,
  const std::vector<AttachedBoxSpec>& attached_boxes,
  const nlohmann::json& static_box_obstacles,
  const nlohmann::json& extra);

nlohmann::json extract_monitor_candidate_json(
  const ik_benchmark::UpdownAwareIkCandidate& candidate,
  size_t display_index,
  const moveit::core::RobotState& state);

nlohmann::json extract_monitor_timing_json(
  const ExtractRolloutTiming& timing,
  size_t display_index,
  const moveit::core::RobotState& state,
  const std::string& prefix,
  int left_box_id,
  int right_box_id,
  const std::vector<std::string>& target_names,
  const std::vector<AttachedBoxSpec>& carried_boxes,
  const nlohmann::json& static_box_obstacles);

nlohmann::json failure_counts_json(const std::map<std::string, size_t>& failure_counts);

}  // namespace alfa_robot::motion
