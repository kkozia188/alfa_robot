#include "alfa_robot_moveit_config/extract_monitor_snapshot_writer.hpp"

#include <filesystem>
#include <fstream>
#include <utility>

namespace alfa_robot::motion
{

ExtractMonitorSnapshotWriter::ExtractMonitorSnapshotWriter(std::string path)
: path_(std::move(path))
{}

void ExtractMonitorSnapshotWriter::setPath(std::string path)
{
  path_ = std::move(path);
}

bool ExtractMonitorSnapshotWriter::write(const nlohmann::json& snapshot, std::string* error) const
{
  try {
    const std::filesystem::path path(path_);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
      if (error) *error = "failed to open " + path_;
      return false;
    }
    out << snapshot.dump(2) << '\n';
    return true;
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return false;
  }
}

nlohmann::json ExtractMonitorSnapshotWriter::readOrEmpty() const
{
  try {
    std::ifstream in(path_);
    if (!in) return nlohmann::json::object();
    return nlohmann::json::parse(in);
  } catch (const std::exception&) {
    return nlohmann::json::object();
  }
}

}  // namespace alfa_robot::motion
