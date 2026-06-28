#include "robot_motion_scene_service/motion_core/scene_geometry.hpp"

#include <cassert>
#include <iostream>

int main()
{
  using namespace alfa_robot::motion;

  ContainerGeometryConfig container;
  container.center_x = 0.8;
  container.center_y = 0.0;
  container.width = 2.2;
  container.height = 2.4;
  container.length = 4.0;
  container.wall_thickness = 0.02;
  container.floor_z = 0.0;
  const auto panels = make_container_panels(container);
  assert(panels.size() == 3);
  assert(panels[0].id == "container_left_wall");
  assert(panels[1].id == "container_right_wall");
  assert(panels[2].id == "container_ceiling");

  BoxWallGeometryConfig wall;
  wall.box_front_x = 0.925;
  wall.scene_y_shift = -0.4;
  wall.container_center_y = -0.4;
  wall.container_width = 2.2;
  wall.container_floor_z = 0.0;
  const auto obstacles = make_box_wall_obstacles_for_opening(6, 8, wall);
  assert(!obstacles.empty());

  const auto left_box = make_attached_box_spec("left", 6, false, CarriedBoxGeometryConfig{});
  assert(left_box.id == "carried_left_box_6");
  assert(left_box.link_name == "left_v5_tool0");
  assert(left_box.size[0] > 0.0);

  const AxisAlignedBox a{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}};
  const AxisAlignedBox b{{0.4, 0.0, 0.0}, {1.0, 1.0, 1.0}};
  const AxisAlignedBox c{{2.0, 0.0, 0.0}, {1.0, 1.0, 1.0}};
  assert(aabb_overlaps(a, b));
  assert(!aabb_overlaps(a, c));

  std::cout << "scene geometry smoke passed\n";
  return 0;
}
