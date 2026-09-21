#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/ros2_ws"
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH CMAKE_PREFIX_PATH AMENT_PREFIX_PATH LD_LIBRARY_PATH
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
source /opt/ros/humble/setup.bash

if [[ ! -d src/central_robot_interfaces ]]; then
  vcs import src --skip-existing < src/dependency.repos
fi

/usr/bin/colcon build \
  --packages-up-to robot_motion_runtime \
  --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
