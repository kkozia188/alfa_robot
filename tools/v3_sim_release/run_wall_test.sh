#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/ros2_ws"
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH CMAKE_PREFIX_PATH AMENT_PREFIX_PATH LD_LIBRARY_PATH
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

ros2 run robot_motion_runtime v3_motion_stage_wall_client \
  --wall \
  --max-rounds 11 \
  --grasp-mode side \
  --summary /tmp/v3_motion_stage_wall_summary.json
