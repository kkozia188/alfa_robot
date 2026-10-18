# V3.2.2 contract-17 VAMP optimization

This experiment keeps the saved 18-D benchmark unchanged and implements the
project's 17-variable planning contract as:

- chassis `x`, `y`, and `yaw`;
- seven left-arm joints;
- seven right-arm joints.

The shared lift is recovered continuously from the state's projection along
the 17-D start-to-goal direction. `head_joint` stays fixed. Four VAMP
RRT-Connect workers search the complete 17-D bounds; worker 0 receives the full
1000-iteration budget and three concurrent scout workers receive 300 iterations
to avoid failed seeds extending tail latency. Every sampled state and edge is
checked by MoveIt/FCL.

The planning-time fix is independent of the VAMP collision checks. The previous
80-node visibility graph tested almost every improving DAG edge. The new opt-in
`visibility_graph_mode:=fast_families` checks, in order:

1. the direct metric lower bound;
2. center waypoints;
3. golden-prefix/direct-suffix routes;
4. direct-prefix/golden-suffix routes;
5. one-waypoint routes;
6. the original full DAG as a fallback.

This matches the previous graph cost for every cycle. The complete audit mode
solves 57/60 worker seeds and validates 15/15 winning candidates. Production
mode additionally skips VAMP only when the selected graph path has already
reached the admissible weighted-L1 endpoint lower bound. Fourteen cycles meet
that proof; `16+20` retains real four-worker 17-D search. The slow cycles fall
from `436/130/552/1243/588 ms` to approximately `17/12/6/91/16 ms` for boxes
`3`, `8`, `11+15`, `16+20`, and `21+25`.

## Reproduce

Build with the same VAMP package used by the parent experiment, then launch:

```bash
ros2 launch alfa_robot_moveit_config \
  v3_chest_carry_shortcut_optimizer.launch.py \
  replay_json_path:=/home/tim/alfa_robot-v322-chest-carry-fast-base/results/v322_chest_carry_fast_base/v322-chest-carry-direct-pull-v2-replay.json \
  output_replay_json_path:=$PWD/results/v322_vamp_17d/contract17-replay.json \
  optimization_report_json_path:=$PWD/results/v322_vamp_17d/contract17-report.json \
  local_rrt_mode:=contract17_vamp \
  visibility_graph_mode:=fast_families \
  vamp_workers:=4 \
  vamp_max_iterations:=1000 \
  vamp_secondary_max_iterations:=300 \
  vamp_prune_at_metric_lower_bound:=true \
  vamp_max_samples:=2000 \
  vamp_range:=0.25
```

Generate the before/after report with:

```bash
/usr/bin/python3 tools/v322_vamp_17d/summarize_contract17.py
```
