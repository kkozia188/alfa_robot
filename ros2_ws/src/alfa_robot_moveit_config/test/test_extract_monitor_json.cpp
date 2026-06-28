#include "alfa_robot_moveit_config/extract_monitor_json.hpp"

#include <array>
#include <cassert>

int main()
{
  using alfa_robot::motion::AttachedBoxSpec;
  using alfa_robot::motion::attached_boxes_json;
  using alfa_robot::motion::extract_monitor_extract_snapshot;
  using alfa_robot::motion::extract_monitor_final_snapshot;
  using alfa_robot::motion::extract_monitor_full_selected_snapshot;
  using alfa_robot::motion::extract_monitor_loaded_snapshot;
  using alfa_robot::motion::extract_monitor_pre_attach_replay_extra;
  using alfa_robot::motion::extract_monitor_replay_context_json;
  using alfa_robot::motion::extract_monitor_snapshot_base;
  using alfa_robot::motion::failure_counts_json;

  AttachedBoxSpec box;
  box.id = "carried_left_box_6";
  box.link_name = "left_v5_tool0";
  box.center_in_link = {0.1, 0.2, 0.3};
  box.size = {0.4, 0.5, 0.6};

  const auto boxes = attached_boxes_json({box});
  assert(boxes.is_array());
  assert(boxes.size() == 1);
  assert(boxes[0].at("id") == "carried_left_box_6");
  assert(boxes[0].at("link_name") == "left_v5_tool0");
  assert(boxes[0].at("center_in_link").size() == 3);
  assert(boxes[0].at("size").size() == 3);

  const auto failures = failure_counts_json({{"left_kdl_no_solution", 2}, {"unknown", 1}});
  assert(failures.is_object());
  assert(failures.at("left_kdl_no_solution") == 2);
  assert(failures.at("unknown") == 1);

  const auto snapshot = extract_monitor_snapshot_base(
    "extract_successes", "抽离成功候选", 12.5, 6, 8, 0.925, -0.4);
  assert(snapshot.at("type") == "extract_monitor_snapshot");
  assert(snapshot.at("phase") == "extract_successes");
  assert(snapshot.at("left_box_id") == 6);
  assert(snapshot.at("right_box_id") == 8);
  assert(snapshot.at("box_front_x") == 0.925);

  const auto extract_snapshot = extract_monitor_extract_snapshot(
    23.0, 6, 8, 0.925, -0.4, 64, 12, 8, {{"collision", 3}}, nlohmann::json::array());
  assert(extract_snapshot.at("phase") == "extract_successes");
  assert(extract_snapshot.at("input_candidate_count") == 64);
  assert(extract_snapshot.at("success_count") == 12);
  assert(extract_snapshot.at("worker_count") == 8);
  assert(extract_snapshot.at("failure_counts").at("collision") == 3);

  const auto loaded_snapshot = extract_monitor_loaded_snapshot(
    34.0, 6, 8, 0.925, -0.4, 12, 4, 1, 30.0, 8, 8, {{"plan_failed", 3}}, nlohmann::json::array());
  assert(loaded_snapshot.at("phase") == "loaded_plan_successes");
  assert(loaded_snapshot.at("extract_success_count") == 12);
  assert(loaded_snapshot.at("attempted_count") == 4);
  assert(loaded_snapshot.at("success_count") == 1);
  assert(loaded_snapshot.at("loaded_parallel_workers") == 8);

  alfa_robot::motion::ExtractRolloutTiming timing;
  timing.candidate_order = 12;
  timing.loaded_plan_rank = 3;
  const auto replay_context = extract_monitor_replay_context_json(timing, 6, 8);
  assert(replay_context.at("candidate_order") == 12);
  assert(replay_context.at("loaded_plan_rank") == 3);
  assert(replay_context.at("left_box_id") == 6);
  assert(replay_context.at("right_box_id") == 8);

  const auto pre_attach_ok = extract_monitor_pre_attach_replay_extra(
    timing, 6, 8, true, "joint_interpolation", 5.0, "ignored_failure");
  assert(pre_attach_ok.at("stage_kind") == "monitor_selected_pre_attach_loaded_to_ik_replay");
  assert(pre_attach_ok.at("candidate_order") == 12);
  assert(pre_attach_ok.at("loaded_plan_rank") == 3);
  assert(pre_attach_ok.at("valid") == true);
  assert(pre_attach_ok.at("method") == "joint_interpolation");
  assert(pre_attach_ok.at("transition_ms") == 5.0);
  assert(pre_attach_ok.at("failure_reason") == "");

  const auto pre_attach_failed = extract_monitor_pre_attach_replay_extra(
    timing, 6, 8, false, "rrt", 7.5, "collision");
  assert(pre_attach_failed.at("valid") == false);
  assert(pre_attach_failed.at("method") == "rrt");
  assert(pre_attach_failed.at("failure_reason") == "collision");

  const auto final_snapshot = extract_monitor_final_snapshot(
    45.0,
    6,
    8,
    0.925,
    -0.4,
    nlohmann::json{{"candidate_order", 12}},
    nlohmann::json::array({nlohmann::json{{"stage", "selected_loaded_plan"}}}));
  assert(final_snapshot.at("phase") == "final_selected");
  assert(final_snapshot.at("phase_label") == "最终采用方案");
  assert(final_snapshot.at("records").size() == 1);
  assert(final_snapshot.at("records")[0].at("candidate_order") == 12);
  assert(final_snapshot.at("replay_stages").size() == 1);
  assert(final_snapshot.at("replay_stages")[0].at("stage") == "selected_loaded_plan");

  const auto full_snapshot = extract_monitor_full_selected_snapshot(
    final_snapshot, 0.925, -0.4, 123.0, std::array<double, 4>{1.0, 2.0, 3.0, 4.0});
  assert(full_snapshot.at("phase") == "full_selected");
  assert(full_snapshot.at("phase_label") == "完整流程最终采用方案");
  assert(full_snapshot.at("box_front_x") == 0.925);
  assert(full_snapshot.at("scene_y_shift") == -0.4);
  assert(full_snapshot.at("elapsed_ms") == 123.0);
  assert(full_snapshot.at("ik_elapsed_ms") == 1.0);
  assert(full_snapshot.at("extract_elapsed_ms") == 2.0);
  assert(full_snapshot.at("loaded_elapsed_ms") == 3.0);
  assert(full_snapshot.at("final_elapsed_ms") == 4.0);
  assert(full_snapshot.at("records").size() == 1);

  return 0;
}
