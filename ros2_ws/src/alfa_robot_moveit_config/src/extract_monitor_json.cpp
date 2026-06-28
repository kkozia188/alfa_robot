#include "alfa_robot_moveit_config/extract_monitor_json.hpp"

#include "alfa_robot_moveit_config/motion_core/pose_math.hpp"

#include <rclcpp/duration.hpp>

namespace alfa_robot::motion
{

nlohmann::json attached_boxes_json(const std::vector<AttachedBoxSpec>& specs)
{
  nlohmann::json boxes = nlohmann::json::array();
  for (const auto& box : specs) {
    boxes.push_back({
      {"id", box.id},
      {"link_name", box.link_name},
      {"center_in_link", {box.center_in_link[0], box.center_in_link[1], box.center_in_link[2]}},
      {"size", {box.size[0], box.size[1], box.size[2]}},
    });
  }
  return boxes;
}

nlohmann::json robot_state_json(const moveit::core::RobotState& state)
{
  const auto& names = state.getRobotModel()->getVariableNames();
  std::vector<double> values;
  values.reserve(names.size());
  for (const auto& name : names) {
    values.push_back(state.getVariablePosition(name));
  }
  return {{"joint_names", names}, {"joint_values", values}, {"joint_map", names_values_json(names, values)}};
}

nlohmann::json trajectory_json(const moveit::planning_interface::MoveGroupInterface::Plan& plan)
{
  const auto& traj = plan.trajectory_.joint_trajectory;
  nlohmann::json points = nlohmann::json::array();
  for (const auto& point : traj.points) {
    points.push_back({
      {"time_from_start_sec", rclcpp::Duration(point.time_from_start).seconds()},
      {"positions", point.positions},
      {"velocities", point.velocities}
    });
  }
  return {
    {"joint_names", traj.joint_names},
    {"point_count", traj.points.size()},
    {"points", points}
  };
}

nlohmann::json extract_monitor_stage_json(
  const std::string& stage_name,
  const moveit::planning_interface::MoveGroupInterface::Plan& plan,
  const moveit::core::RobotState& start_state,
  const moveit::core::RobotState& goal_state,
  const std::vector<std::string>& target_names,
  const std::vector<AttachedBoxSpec>& attached_boxes,
  const nlohmann::json& static_box_obstacles,
  const nlohmann::json& extra)
{
  return {
    {"type", "stage"},
    {"stage", stage_name},
    {"trajectory", trajectory_json(plan)},
    {"target_names", target_names},
    {"start_state", robot_state_json(start_state)},
    {"goal_state", robot_state_json(goal_state)},
    {"attached_boxes", attached_boxes_json(attached_boxes)},
    {"static_box_obstacles", static_box_obstacles},
    {"extra", extra}
  };
}

nlohmann::json extract_monitor_candidate_json(
  const ik_benchmark::UpdownAwareIkCandidate& candidate,
  size_t display_index,
  const moveit::core::RobotState& state)
{
  return {
    {"display_index", display_index},
    {"h", candidate.h},
    {"h_index", candidate.h_index},
    {"seed_index", candidate.seed_index},
    {"score", candidate.score},
    {"solve_ms", candidate.solve_ms},
    {"solver_path", candidate.solver_path},
    {"target_order", candidate.target_order},
    {"updown_delta", candidate.updown_delta},
    {"joint_delta", candidate.joint_delta},
    {"state", robot_state_json(state)}
  };
}

nlohmann::json extract_monitor_timing_json(
  const ExtractRolloutTiming& timing,
  size_t display_index,
  const moveit::core::RobotState& state,
  const std::string& prefix,
  int left_box_id,
  int right_box_id,
  const std::vector<std::string>& target_names,
  const std::vector<AttachedBoxSpec>& carried_boxes,
  const nlohmann::json& static_box_obstacles)
{
  nlohmann::json replay_stages = nlohmann::json::array();
  for (const auto& stage : timing.rollout_records) {
    replay_stages.push_back(stage);
  }
  for (const auto& shift_stage : timing.lateral_shift_replay_stages) {
    if (!shift_stage.start_state || !shift_stage.goal_state) {
      continue;
    }
    nlohmann::json extra = shift_stage.extra;
    extra["candidate_order"] = timing.candidate_order;
    extra["loaded_plan_rank"] = timing.loaded_plan_rank;
    extra["loaded_plan_success"] = timing.loaded_plan_success;
    extra["loaded_plan_failure_reason"] = timing.loaded_plan_failure_reason;
    extra["left_box_id"] = left_box_id;
    extra["right_box_id"] = right_box_id;
    replay_stages.push_back(extract_monitor_stage_json(
      shift_stage.stage_name,
      shift_stage.plan,
      *shift_stage.start_state,
      *shift_stage.goal_state,
      target_names,
      carried_boxes,
      static_box_obstacles,
      extra));
  }
  if (timing.loaded_start_state && timing.loaded_goal_state &&
      !timing.loaded_plan.trajectory_.joint_trajectory.points.empty()) {
    nlohmann::json extra = {
      {"stage_kind", "monitor_loaded_plan_attempt_replay"},
      {"valid", timing.loaded_plan_success},
      {"candidate_order", timing.candidate_order},
      {"loaded_plan_rank", timing.loaded_plan_rank},
      {"loaded_plan_success", timing.loaded_plan_success},
      {"loaded_plan_failure_reason", timing.loaded_plan_failure_reason},
      {"loaded_plan_ms", timing.loaded_plan_ms},
      {"loaded_plan_points", timing.loaded_plan_points},
      {"loaded_plan_trajectory_distance", timing.loaded_plan_trajectory_distance},
      {"left_box_id", left_box_id},
      {"right_box_id", right_box_id}
    };
    replay_stages.push_back(extract_monitor_stage_json(
      prefix + "/candidate_" + std::to_string(timing.candidate_order) + "/loaded_plan_attempt",
      timing.loaded_plan,
      *timing.loaded_start_state,
      *timing.loaded_goal_state,
      target_names,
      carried_boxes,
      static_box_obstacles,
      extra));
  }
  return {
    {"display_index", display_index},
    {"candidate_order", timing.candidate_order},
    {"h", timing.h},
    {"h_index", timing.h_index},
    {"seed_index", timing.seed_index},
    {"ik_score", timing.ik_score},
    {"ik_solve_ms", timing.ik_solve_ms},
    {"rollout_ms", timing.rollout_ms},
    {"interval_ms", timing.interval_ms},
    {"accepted_steps", timing.accepted_steps},
    {"failed_steps", timing.failed_steps},
    {"final_retreat_x", timing.final_retreat_x},
    {"final_lift_z", timing.final_lift_z},
    {"final_pitch_deg", timing.final_pitch_deg},
    {"right_final_retreat_x", timing.right_final_retreat_x},
    {"right_final_lift_z", timing.right_final_lift_z},
    {"right_final_pitch_deg", timing.right_final_pitch_deg},
    {"success", timing.success},
    {"failure_reason", timing.failure_reason},
    {"loaded_plan_attempted", timing.loaded_plan_attempted},
    {"loaded_plan_success", timing.loaded_plan_success},
    {"lateral_shift_attempted", timing.lateral_shift_attempted},
    {"lateral_shift_success", timing.lateral_shift_success},
    {"lateral_shift_ms", timing.lateral_shift_ms},
    {"lateral_shift_reached_distance", timing.lateral_shift_reached_distance},
    {"lateral_shift_points", timing.lateral_shift_points},
    {"loaded_plan_rank", timing.loaded_plan_rank},
    {"loaded_plan_ms", timing.loaded_plan_ms},
    {"loaded_plan_points", timing.loaded_plan_points},
    {"loaded_plan_trajectory_distance", timing.loaded_plan_trajectory_distance},
    {"loaded_plan_failure_reason", timing.loaded_plan_failure_reason},
    {"loaded_pose_distance_sum", timing.loaded_pose_distance_sum},
    {"loaded_pose_distance_l2", timing.loaded_pose_distance_l2},
    {"loaded_pose_max_joint_delta", timing.loaded_pose_max_joint_delta},
    {"loaded_plan_selected", timing.loaded_plan_selected},
    {"state", robot_state_json(state)},
    {"replay_stage_count", replay_stages.size()},
    {"replay_stages", replay_stages}
  };
}

nlohmann::json failure_counts_json(const std::map<std::string, size_t>& failure_counts)
{
  nlohmann::json failure_json = nlohmann::json::object();
  for (const auto& [reason, count_value] : failure_counts) {
    failure_json[reason] = count_value;
  }
  return failure_json;
}

nlohmann::json extract_monitor_snapshot_base(
  const std::string& phase,
  const std::string& phase_label,
  double elapsed_ms,
  int left_box_id,
  int right_box_id,
  double box_front_x,
  double scene_y_shift)
{
  return {
    {"type", "extract_monitor_snapshot"},
    {"phase", phase},
    {"phase_label", phase_label},
    {"elapsed_ms", elapsed_ms},
    {"left_box_id", left_box_id},
    {"right_box_id", right_box_id},
    {"box_front_x", box_front_x},
    {"scene_y_shift", scene_y_shift}
  };
}

nlohmann::json extract_monitor_ik_snapshot(
  const std::string& snapshot_path,
  double elapsed_ms,
  int left_box_id,
  int right_box_id,
  double box_front_x,
  double scene_y_shift,
  const ik_benchmark::UpdownAwareIkResult& ik_result,
  const IkCandidateSelectionStats& dedup_stats,
  const nlohmann::json& rejection_counts,
  const nlohmann::json& records)
{
  auto snapshot = extract_monitor_snapshot_base(
    "ik_candidates", "不重复 IK 候选", elapsed_ms, left_box_id, right_box_id, box_front_x, scene_y_shift);
  snapshot["snapshot_path"] = snapshot_path;
  snapshot["ik_trial_count"] = ik_result.trial_count;
  snapshot["ik_legal_count"] = ik_result.legal_count;
  snapshot["ik_wall_ms"] = ik_result.wall_ms;
  snapshot["ik_dedup_enabled"] = dedup_stats.enabled;
  snapshot["ik_dedup_input_count"] = dedup_stats.input_count;
  snapshot["ik_dedup_unique_count"] = dedup_stats.unique_count;
  snapshot["ik_dedup_removed_count"] = dedup_stats.removed_count;
  snapshot["ik_dedup_selected_count"] = dedup_stats.selected_count;
  snapshot["ik_dedup_ms"] = dedup_stats.elapsed_ms;
  snapshot["rejection_counts"] = rejection_counts;
  snapshot["records"] = records;
  return snapshot;
}

nlohmann::json extract_monitor_extract_snapshot(
  double elapsed_ms,
  int left_box_id,
  int right_box_id,
  double box_front_x,
  double scene_y_shift,
  size_t input_candidate_count,
  size_t success_count,
  size_t worker_count,
  const std::map<std::string, size_t>& failure_counts,
  const nlohmann::json& records)
{
  auto snapshot = extract_monitor_snapshot_base(
    "extract_successes", "抽离成功候选", elapsed_ms, left_box_id, right_box_id, box_front_x, scene_y_shift);
  snapshot["input_candidate_count"] = input_candidate_count;
  snapshot["success_count"] = success_count;
  snapshot["worker_count"] = worker_count;
  snapshot["failure_counts"] = failure_counts_json(failure_counts);
  snapshot["records"] = records;
  return snapshot;
}

nlohmann::json extract_monitor_loaded_snapshot(
  double elapsed_ms,
  int left_box_id,
  int right_box_id,
  double box_front_x,
  double scene_y_shift,
  size_t extract_success_count,
  size_t attempted_count,
  size_t success_count,
  double loaded_plan_batch_wall_ms,
  size_t loaded_parallel_workers,
  size_t loaded_candidate_limit,
  const std::map<std::string, size_t>& failure_counts,
  const nlohmann::json& records)
{
  auto snapshot = extract_monitor_snapshot_base(
    "loaded_plan_successes", "负重规划成功候选", elapsed_ms, left_box_id, right_box_id, box_front_x, scene_y_shift);
  snapshot["extract_success_count"] = extract_success_count;
  snapshot["attempted_count"] = attempted_count;
  snapshot["success_count"] = success_count;
  snapshot["loaded_plan_batch_wall_ms"] = loaded_plan_batch_wall_ms;
  snapshot["loaded_parallel_workers"] = loaded_parallel_workers;
  snapshot["loaded_candidate_limit"] = loaded_candidate_limit;
  snapshot["failure_counts"] = failure_counts_json(failure_counts);
  snapshot["records"] = records;
  return snapshot;
}

}  // namespace alfa_robot::motion
