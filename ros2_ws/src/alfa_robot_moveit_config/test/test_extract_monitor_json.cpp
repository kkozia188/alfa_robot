#include "alfa_robot_moveit_config/extract_monitor_json.hpp"

#include <cassert>

int main()
{
  using alfa_robot::motion::AttachedBoxSpec;
  using alfa_robot::motion::attached_boxes_json;
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

  return 0;
}
