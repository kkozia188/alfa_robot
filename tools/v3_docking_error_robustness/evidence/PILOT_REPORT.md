# MOTION-274 3D Pilot Report

Date: 2026-10-02

## Scope

This is a pilot, not the final 125-by-15 acceptance matrix.

- Model: `robot_v3.2.2-suction`
- Tool0 local +Z offset: `0.151 m`
- Upstream baseline: `d9c330cef72981390d81ac2b1cd5a6eb9e892195`
- Samples: `3 x 3 x 3 = 27`
- Errors: `x/y = {-0.05, 0, +0.05} m`, `yaw = {-3, 0, +3} deg`
- Cycles: top-row outer pair, inner pair, and center box (`3` cycles)
- Evaluated cycle tasks: `81`
- Planning seed: `104729`
- Design fingerprint: `8046b965a90b88d07bc3ce26ea28cd03c69409875c7ab2dcd5895d0be39af61c`
- Trajectory cache: disabled; cache loading is not planning time

Each sample publishes a fresh `map -> odom -> base_footprint` transform. The
planner then republishes current `base_link` 6D target poses and rejects TF
older than 500 ms. Successful Action diagnostics are checked against the exact
injected planar root to `1e-4`.

## Result

| Metric | Baseline | Dual-phase repair | Delta |
| --- | ---: | ---: | ---: |
| Complete cycle success | 66/81 (81.48%) | 74/81 (91.36%) | +9.88 pp |
| IK | 81/81 (100.00%) | 81/81 (100.00%) | 0.00 pp |
| PREGRASP | 81/81 (100.00%) | 81/81 (100.00%) | 0.00 pp |
| EXTRACT | 66/81 (81.48%) | 74/81 (91.36%) | +9.88 pp |
| RETURN | 66/81 (81.48%) | 74/81 (91.36%) | +9.88 pp |
| Collision-class failures | 15 | 7 | -8 |

All seven optimized failures are cycle 2 (`boxes 2+4`). Cycles 1 and 3 are
`27/27`; cycle 2 improves to `20/27`. The dominant defect is not IK reachability
but collision introduced when independently planned left/right entry or return
paths are time-aligned. The optimization replans only those colliding phase 0
or phase 8 segments with `dual_arm_with_updown`, then reruns complete MoveIt/FCL
state and edge validation. It does not disable cross-arm, chassis, neighbor-box,
or environment collisions.

## Axis slices after optimization

| Axis value | Success |
| --- | ---: |
| x = -0.05 m | 25/27 (92.59%) |
| x = 0.00 m | 25/27 (92.59%) |
| x = +0.05 m | 24/27 (88.89%) |
| y = -0.05 m | 23/27 (85.19%) |
| y = 0.00 m | 26/27 (96.30%) |
| y = +0.05 m | 25/27 (92.59%) |
| yaw = -3 deg | 24/27 (88.89%) |
| yaw = 0 deg | 25/27 (92.59%) |
| yaw = +3 deg | 25/27 (92.59%) |

Pilot recommendation: keep the requested docking lateral offset centered at
`base_y = 0`; do not intentionally bias to robot-right (`y=-0.05 m`). Avoid a
positive X bias until the full 15-cycle matrix is complete. This recommendation
is provisional because only the top row was sampled.

## Reproducible Rerun

The optimized nominal sample completed `3/3` cycles and produced:

- File: `data/ik_benchmark/v3_docking_error/motion-274-nominal-rerun/rerun/dx_p000_dy_p000_yaw_p0000.rrd`
- Size: `3,739,235 bytes`
- SHA-256: `7bad95eeb68f1a2d4700e1f9df755bd123ee75f9e07441ef9fd2cebc074c716d`
- Rerun verification: passed
- Rerun statistics: `674` chunks, `103` entity paths, `82,668` rows

The recording contains the V3.2.2 production robot meshes, wall boxes,
environment geometry, carried boxes, stage labels, and all replay frames.
The `world/boxes/placed` path contains recursive clear commands only; no placed
box geometry is retained after the external handoff.

## Evidence hashes

| Artifact | SHA-256 |
| --- | --- |
| pilot baseline summary | `fda09c9aa812c6996ebae4d9f6989a5bc05095d19ff215780c9b4a31b52205fd` |
| pilot optimized summary | `77ea9cefa6f6aeb16e8b81d2fa1a66bb8b28f8ac3c4079ea4edde472006d51ac` |
| pilot comparison | `ef25a19fd88e915ecc7b8bb0e7bacddfb13c4151ad3d0b3c6408534ec6dee041` |

## Remaining acceptance work

- Run the configured 125-point matrix over all 15 cycles (`1,875` cycle tasks)
  for baseline and optimized strategies.
- Recheck successful samples with the final independent replay validator and
  publish the full failure taxonomy and selected boundary Reruns.
- Decide the final guaranteed docking-error envelope only from that complete
  result; the pilot rates must not be advertised as full-wall acceptance.
