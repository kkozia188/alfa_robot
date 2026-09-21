#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/ros2_ws"
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH CMAKE_PREFIX_PATH AMENT_PREFIX_PATH LD_LIBRARY_PATH
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
set +u
source /opt/ros/humble/setup.bash
set -u

if [[ ! -d src/central_robot_interfaces ]]; then
  git clone https://github.com/SevenovaHangzhou/robot_interfaces.git src/central_robot_interfaces
  git -C src/central_robot_interfaces checkout 92d6ff2ed0b45684d7da2170d96703ca8be569f4
fi

/usr/bin/colcon build \
  --packages-up-to robot_motion_runtime \
  --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
