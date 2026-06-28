#include "alfa_robot_moveit_config/extract_monitor_json.hpp"

#include <cassert>

int main()
{
  using alfa_robot::motion::AttachedBoxSpec;
  using alfa_robot::motion::attached_boxes_json;

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

  return 0;
}
