# V3.2.2 mandatory contract-17 VAMP search

This project is isolated from the prior-guided low-latency branch. It adds
`local_rrt_mode:=contract17_vamp_search` with these hard requirements:

- the search receives only the chest-segment start and goal states;
- bounds come from those endpoints, explicit margins and robot limits;
- every cycle must call VAMP RRT-Connect;
- a failed search fails the cycle; no golden or visibility-graph fallback exists;
- the selected path must come from the search tree;
- post-processing only densifies search-tree edges for execution and validation;
- every state/edge and final replay remains authoritative under MoveIt/FCL.

The source golden segment is retained outside the search interface only to report
a comparison cost. It cannot affect bounds, samples, waypoints or path selection.

## Build

```bash
export PATH=/home/tim/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/humble/setup.bash
source /home/tim/alfa_robot-v322-vamp-17d-parallel/install-vamp-17d/setup.bash
colcon --log-base log-vamp-search-clean build \
  --base-paths ros2_ws/src \
  --build-base build-vamp-search-clean \
  --install-base install-vamp-search-clean \
  --packages-select alfa_robot_moveit_config \
  --symlink-install \
  --cmake-args -DBUILD_TESTING=ON \
    -Dvamp_DIR=/home/tim/.cache/alfa-vamp/prefix/share/cmake/vamp \
    -DPython3_EXECUTABLE=/usr/bin/python3
```

## Search

Run three independent base seeds (`169343505`, `271828183`, `314159265`). The first
run command is:

```bash
source install-vamp-search-clean/setup.bash
ros2 launch alfa_robot_moveit_config \
  v3_chest_carry_shortcut_optimizer.launch.py \
  replay_json_path:=/home/tim/alfa_robot-v322-chest-carry-fast-base/results/v322_chest_carry_fast_base/v322-chest-carry-direct-pull-v2-replay.json \
  output_replay_json_path:=$PWD/results/v322_vamp_17d_search/seed-169343505-replay.json \
  optimization_report_json_path:=$PWD/results/v322_vamp_17d_search/seed-169343505-report.json \
  local_rrt_mode:=contract17_vamp_search \
  vamp_workers:=4 \
  vamp_seed:=169343505 \
  vamp_max_iterations:=1000 \
  vamp_secondary_max_iterations:=1000 \
  vamp_max_samples:=4000 \
  vamp_range:=0.25 \
  vamp_prune_at_metric_lower_bound:=false \
  full_vamp_base_xy_margin_m:=0.15 \
  full_vamp_base_yaw_margin_deg:=5.0 \
  full_vamp_arm_margin_deg:=35.0
```

## Results

- Three seeds complete 45/45 mandatory searches and 75/75 boxes.
- No cycle uses a golden intermediate frame, visibility graph or fallback.
- Twelve cycles are collision-free direct planner connections.
- Boxes 3, 16+20 and 21+25 require RRT tree expansion for all three seeds.
- The complete replay and safe-turn replay pass MoveIt/FCL.
- Across 180 worker trials, 178 solve the tree problem, 176 pass dense FCL,
  and two coarse candidates are explicitly rejected before winner selection.
- Median 15-cycle tree-search wall time is `379.651 ms`; median complete
  planning core is `3.282 s` and the median maximum cycle is `443.080 ms`.
- Safe-turn validation covers 25,739 frames plus 3,985 edge samples; maximum
  carried-box tilt is `7.708 / 7.935 deg` and equal-height error is
  `1.06998e-10 m`.

Generate and check the report:

```bash
/usr/bin/python3 tools/v322_vamp_17d_search/summarize_search.py
/usr/bin/python3 tools/v322_vamp_17d_search/test_contract.py
```

Open the verified Rerun:

```bash
/usr/bin/python3 tools/v322_vamp_17d_search/open_rerun.py
```
