#include "alfa_robot_moveit_config/extract_monitor_snapshot_writer.hpp"

#include <cassert>
#include <filesystem>

int main()
{
  const auto path = std::filesystem::temp_directory_path() /
    "alfa_extract_monitor_snapshot_writer_test" / "nested" / "snapshot.json";
  std::filesystem::remove_all(path.parent_path().parent_path());

  alfa_robot::motion::ExtractMonitorSnapshotWriter writer(path.string());
  std::string error;
  const bool ok = writer.write({{"phase", "ik_candidates"}, {"count", 3}}, &error);
  assert(ok);
  assert(error.empty());

  const auto loaded = writer.readOrEmpty();
  assert(loaded.is_object());
  assert(loaded.at("phase") == "ik_candidates");
  assert(loaded.at("count") == 3);

  std::filesystem::remove_all(path.parent_path().parent_path());
  return 0;
}
