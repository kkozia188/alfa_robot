#include "alfa_robot_moveit_config/extract_monitor_replay_builder.hpp"

namespace alfa_robot::motion
{

nlohmann::json ExtractMonitorReplayBuilder::build(
  ExtractRolloutTiming& selected,
  const ExtractMonitorReplayBuildRequest& request) const
{
  if (ensure_extract_replay) {
    ensure_extract_replay(selected);
  }

  nlohmann::json replay_stages = nlohmann::json::array();
  if (request.loaded_start_state && request.ik_goal_state) {
    const auto transition_t0 = now ? now() : std::chrono::steady_clock::now();
    const auto transition = transition_planner.plan(*request.loaded_start_state, *request.ik_goal_state);
    const auto transition_t1 = now ? now() : std::chrono::steady_clock::now();
    const double transition_ms = std::chrono::duration<double, std::milli>(
      transition_t1 - transition_t0).count();
    replay_stages.push_back(extract_monitor_stage_json(
      request.prefix + "/selected_pre_attach_loaded_to_ik",
      transition.plan,
      *request.loaded_start_state,
      *request.ik_goal_state,
      request.target_names,
      {},
      request.static_box_obstacles,
      extract_monitor_pre_attach_replay_extra(
        selected,
        request.left_box_id,
        request.right_box_id,
        transition.valid,
        transition.method,
        transition_ms,
        transition.failure_reason)));
  }

  for (const auto& stage : selected.rollout_records) {
    replay_stages.push_back(stage);
  }

  const auto shift_replay_stages = extract_monitor_selected_lateral_shift_replay_stages(
    selected,
    request.left_box_id,
    request.right_box_id,
    request.target_names,
    request.carried_boxes,
    request.static_box_obstacles);
  for (const auto& stage : shift_replay_stages) {
    replay_stages.push_back(stage);
  }

  const auto loaded_stage = extract_monitor_selected_loaded_plan_replay_stage(
    request.prefix,
    selected,
    request.left_box_id,
    request.right_box_id,
    request.target_names,
    request.carried_boxes,
    request.static_box_obstacles);
  if (!loaded_stage.is_null()) {
    replay_stages.push_back(loaded_stage);
  }

  return replay_stages;
}

}  // namespace alfa_robot::motion
