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

}  // namespace alfa_robot::motion
