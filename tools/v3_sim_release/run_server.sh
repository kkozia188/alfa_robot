#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/ros2_ws"
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH CMAKE_PREFIX_PATH AMENT_PREFIX_PATH LD_LIBRARY_PATH
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
source /opt/ros/humble/setup.bash
source install/setup.bash

BACKEND="${1:-replay}"
RECORDING="${2:-/tmp/v3_motion_stage_wall.rrd}"
ros2 launch alfa_robot_moveit_config v3_motion_stage_wall_demo.launch.py \
  x:=0.5 \
  execution_backend:="$BACKEND" \
  follow_joint_trajectory_action:=/whole_body_jtc/follow_joint_trajectory \
  start_rerun:=true \
  start_rviz:=false \
  spawn_viewer:=true \
  rerun_recording_path:="$RECORDING"
