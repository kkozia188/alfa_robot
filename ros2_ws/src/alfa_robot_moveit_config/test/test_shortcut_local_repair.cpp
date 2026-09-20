#include <alfa_robot_moveit_config/shortcut_local_repair.hpp>
#include <cassert>

int main()
{
  using namespace alfa_robot::motion;
  std::vector<bool> nodes(21, true), edges(20, true);
  assert(shortcutRepairWindows(nodes, edges, 5).empty());
  nodes[5] = false;
  nodes[15] = false;
  auto windows = shortcutRepairWindows(nodes, edges, 1);
  assert(windows.size() == 2);
  assert(windows[0].begin == 3 && windows[0].end == 7);
  assert(windows[1].begin == 13 && windows[1].end == 17);
  nodes.assign(11, true); edges.assign(10, true);
  nodes[4] = false; nodes[7] = false;
  windows = shortcutRepairWindows(nodes, edges, 2);
  assert(windows.size() == 1 && windows[0].begin == 1 && windows[0].end == 10);
  nodes.assign(8, true); edges.assign(7, true);
  edges[2] = false;
  windows = shortcutRepairWindows(nodes, edges, 2);
  assert(windows.size() == 1 && windows[0].begin == 0 && windows[0].end == 5);
  windows = shortcutRepairWindows(nodes, edges, 100);
  assert(windows.size() == 1 && windows[0].begin == 0 && windows[0].end == 7);
  assert(shortcutRepairWindows({true}, {}, 5).empty());
  bool rejected = false;
  try { shortcutRepairWindows({false, true}, {false}, 5); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
  rejected = false;
  try { shortcutRepairWindows({true, true}, {}, 5); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}
