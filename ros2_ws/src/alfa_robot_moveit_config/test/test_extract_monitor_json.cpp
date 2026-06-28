#include "alfa_robot_moveit_config/extract_monitor_json.hpp"

#include <cassert>

int main()
{
  using alfa_robot::motion::AttachedBoxSpec;
  using alfa_robot::motion::attached_boxes_json;
  using alfa_robot::motion::extract_monitor_extract_snapshot;
  using alfa_robot::motion::extract_monitor_loaded_snapshot;
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

  return 0;
}
