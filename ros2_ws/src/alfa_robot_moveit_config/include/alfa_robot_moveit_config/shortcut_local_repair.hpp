#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace alfa_robot::motion
{
struct ShortcutRepairWindow
{
  size_t begin = 0;
  size_t end = 0;
};

inline std::vector<ShortcutRepairWindow> shortcutRepairWindows(
  const std::vector<bool>& nodes_valid, const std::vector<bool>& edges_valid,
  size_t padding)
{
  if (nodes_valid.empty() || edges_valid.size() + 1 != nodes_valid.size() ||
      !nodes_valid.front() || !nodes_valid.back())
    throw std::invalid_argument("shortcut requires valid endpoints and matching edge flags");
  std::vector<ShortcutRepairWindow> windows;
  size_t edge = 0;
  const auto failed = [&](size_t index) {
    return !nodes_valid[index] || !nodes_valid[index + 1] || !edges_valid[index];
  };
  while (edge < edges_valid.size()) {
    if (!failed(edge)) { ++edge; continue; }
    const size_t first = edge;
    while (edge < edges_valid.size() && failed(edge)) ++edge;
    ShortcutRepairWindow window;
    window.begin = first > padding ? first - padding : 0;
    window.end = std::min(nodes_valid.size() - 1, edge + padding);
    while (!nodes_valid[window.begin]) --window.begin;
    while (!nodes_valid[window.end]) ++window.end;
    if (!windows.empty() && window.begin <= windows.back().end)
      windows.back().end = std::max(windows.back().end, window.end);
    else windows.push_back(window);
  }
  return windows;
}
}
