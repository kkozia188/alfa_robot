# V3.2.2 full-state VAMP benchmark

State: chassis `x/y/yaw` + `updown` + 14 arm joints = 18 dimensions; `head_joint` remains fixed at zero.

| Planner | Search wall median | Planning core median | Max cycle | Solved | Validated states/s |
|---|---:|---:|---:|---:|---:|
| OMPL full 15-D | 3.989s | 7.038s | 1.433s | 15/15 | n/a |
| VAMP full 18-D, 1 worker | 0.328s | 3.377s | 1.229s | 15/15 | 18.4k |
| VAMP full 18-D, 4 workers | 0.387s | 3.475s | 1.238s | 15/15 (58/60 seeds) | 53.7k |

- One-worker VAMP search is **12.16x** faster than the OMPL full-space stage; end-to-end planning is **2.08x** faster.
- Four workers raise validated-state throughput by **2.92x**, but search latency is **17.9% higher** and CPU work is **3.93x**.
- MoveIt/FCL consumes **99.88%** of single-worker search core time. The next acceleration target is collision validation, not nearest-neighbor/tree expansion.
- Full validation passed **28,836 frames + 1,071 edge samples**; 15/15 full-state candidates passed FCL, but the visibility graph retained lower weighted travel.
- Safe-turn Rerun passed **25,731 frames + 4,011 edge samples** with 25/25 boxes, 180-degree unload and equal-height release.

This experiment uses VAMP for the 18-D RRT-Connect tree and batched edge sampling. Collision checks are authoritative MoveIt/FCL callbacks, not a Cricket-generated V3 SIMD sphere kernel.
