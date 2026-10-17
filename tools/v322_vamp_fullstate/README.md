# V3.2.2 full-state VAMP experiment

This experiment is isolated from the saved two-dimensional VAMP branch. It
adds an opt-in `local_rrt_mode:=full_vamp` to the chest shortcut optimizer.

The state is 18-dimensional:

- chassis `x`, `y`, and `yaw`;
- shared lift `updown`;
- seven left-arm and seven right-arm joints.

`head_joint` is fixed because it is zero at every chest-segment endpoint and is
outside the grasp planning control contract. VAMP runs RRT-Connect, nearest
neighbor search, state arithmetic, and eight-state edge batches. Each sampled
state is mapped to the V3.2.2 MoveIt robot and checked by FCL. This is a real
full-state VAMP tree, but it is not a generated V3 SIMD collision kernel.

## Build

```bash
source /opt/ros/humble/setup.bash
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash
colcon --log-base log-vamp-fullstate build \
  --base-paths ros2_ws/src \
  --build-base build-vamp-fullstate \
  --install-base install-vamp-fullstate \
  --packages-select alfa_robot_moveit_config \
  --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
    -Dvamp_DIR=/home/tim/.cache/alfa-vamp/prefix/share/cmake/vamp
```

## Run

Use the direct-pull source replay, not an already optimized replay:

```bash
source install-vamp-fullstate/setup.bash
ros2 launch alfa_robot_moveit_config \
  v3_chest_carry_shortcut_optimizer.launch.py \
  replay_json_path:=/home/tim/alfa_robot-v322-chest-carry-fast-base/results/v322_chest_carry_fast_base/v322-chest-carry-direct-pull-v2-replay.json \
  output_replay_json_path:=$PWD/results/v322_vamp_fullstate/full18d-replay.json \
  optimization_report_json_path:=$PWD/results/v322_vamp_fullstate/full18d-report.json \
  local_rrt_mode:=full_vamp \
  vamp_workers:=4 \
  vamp_max_iterations:=1000 \
  vamp_max_samples:=2000 \
  vamp_range:=0.15
```

Generate the aggregate report after three runs of each configuration:

```bash
/usr/bin/python3 tools/v322_vamp_fullstate/summarize_fullstate_vamp.py
```

The authoritative safety check remains
`v3_dual_arm_5x5_replay_validator.launch.py` with 1-degree joint, 1-centimeter
lift/base edge sampling and an 8-degree carried-box tilt limit.

## Measured result

Three sequential repetitions use the same direct-pull input and a 1000
iteration / 2000 sample budget. Median results are:

| Planner | Search wall | Planning core | Validated states/s |
|---|---:|---:|---:|
| OMPL full 15-D | 3.989 s | 7.038 s | n/a |
| VAMP full 18-D, 1 worker | 0.328 s | 3.377 s | 18.4k |
| VAMP full 18-D, 4 workers | 0.387 s | 3.475 s | 53.7k |

Both VAMP configurations solve 15/15 cycles. Four workers solve 58/60 seeds
and raise aggregate FCL-validated state throughput by 2.92x, but their search
latency is 17.9% higher and CPU work is 3.93x. MoveIt/FCL consumes 99.88% of
single-worker search core time, so one worker is the better latency default.

All 15 winning full-state candidates pass FCL. None beats the visibility graph
under the common double-precision weighted-travel gate. The final replay passes
28,836 frames plus 1,071 edge samples. The safe-turn Rerun passes 25,731 frames
plus 4,011 edge samples and keeps the existing layout and world-origin orbit
center.
