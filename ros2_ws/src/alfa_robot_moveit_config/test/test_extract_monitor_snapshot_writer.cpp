#include "alfa_robot_moveit_config/extract_monitor_snapshot_writer.hpp"

#include <array>
#include <cassert>
#include <filesystem>
#include <string>

int main()
{
  const auto path = std::filesystem::temp_directory_path() /
    "alfa_extract_monitor_snapshot_writer_test" / "nested" / "snapshot.json";
  std::filesystem::remove_all(path.parent_path().parent_path());

  alfa_robot::motion::ExtractMonitorSnapshotWriter writer(path.string());
  const std::string formatted_error = writer.writeError("disk full");
  assert(formatted_error.find(path.string()) != std::string::npos);
  assert(formatted_error.find("disk full") != std::string::npos);

  std::string error;
  const bool ok = writer.write({{"phase", "ik_candidates"}, {"count", 3}}, &error);
  assert(ok);
  assert(error.empty());

  const auto loaded = writer.readOrEmpty();
  assert(loaded.is_object());
  assert(loaded.at("phase") == "ik_candidates");
  assert(loaded.at("count") == 3);

  const bool full_ok = writer.writeFullSelectedSnapshot(
    alfa_robot::motion::ExtractMonitorFullSelectedSnapshotRequest{
      0.925,
      -0.4,
      123.0,
      std::array<double, 4>{1.0, 2.0, 3.0, 4.0}},
    &error);
  assert(full_ok);
  const auto full_loaded = writer.readOrEmpty();
  assert(full_loaded.at("phase") == "full_selected");
  assert(full_loaded.at("phase_label") == "完整流程最终采用方案");
  assert(full_loaded.at("box_front_x") == 0.925);
  assert(full_loaded.at("scene_y_shift") == -0.4);
  assert(full_loaded.at("elapsed_ms") == 123.0);
  assert(full_loaded.at("ik_elapsed_ms") == 1.0);
  assert(full_loaded.at("extract_elapsed_ms") == 2.0);
  assert(full_loaded.at("loaded_elapsed_ms") == 3.0);
  assert(full_loaded.at("final_elapsed_ms") == 4.0);

  const auto message = writer.appendSnapshotPath("done");
  assert(message.find("done snapshot=") == 0);
  assert(message.find(path.string()) != std::string::npos);

  std::filesystem::remove_all(path.parent_path().parent_path());
  return 0;
}
