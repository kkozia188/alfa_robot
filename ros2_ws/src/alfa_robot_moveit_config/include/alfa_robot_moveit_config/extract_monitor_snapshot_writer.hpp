#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace alfa_robot::motion
{

class ExtractMonitorSnapshotWriter
{
public:
  explicit ExtractMonitorSnapshotWriter(std::string path = {});

  const std::string& path() const { return path_; }
  void setPath(std::string path);

  bool write(const nlohmann::json& snapshot, std::string* error = nullptr) const;
  std::string writeError(const std::string& error) const;
  nlohmann::json readOrEmpty() const;

private:
  std::string path_;
};

}  // namespace alfa_robot::motion
