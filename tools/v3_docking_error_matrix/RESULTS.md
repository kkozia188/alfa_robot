# Preliminary Result: 2026-10-08

This is the first batch from the independent `MOTION-257` X/Y/Yaw matrix
project. It validates the experiment machinery and records real failure modes;
it is not yet the final 75-case complete joint-matrix certificate.

## Dataset State

- Designed unique poses: `131`.
- Imported complete baselines: `11` aligned X cases from MOTION-261 and `11`
  Yaw cases from MOTION-257.
- Newly screened Y cases: `21`, from `-0.20 m` to `+0.20 m` in `0.02 m`
  steps.
- Newly attempted complete cases: `4`.
- Aggregate result rows after scope-aware de-duplication: `47`.
- Existing and new full validations represented by the aggregate:
  `629,105` MoveIt/FCL frames and `23,783` interpolated edges.

The default full joint matrix contains `75` cases:

```text
dx   = {-0.05, 0, +0.05} m
dy   = {-0.10, -0.05, 0, +0.05, +0.10} m
yaw  = {-5, -2.5, 0, +2.5, +5} deg
```

## Y Baseline Screen

The eight-group screen passes continuously from `dy=-0.14 m` through
`dy=+0.10 m`. The observed brackets are:

- negative: pass `-0.14 m`, fail `-0.16 m`;
- positive: pass `+0.10 m`, fail `+0.12 m`.

The positive boundary is not a simple reach limit: at `+0.12 m`, box 23 has
a candidate until Cartesian retreat, then collides
`base_link <-> left_link5`. At `+0.14 m` retreat becomes infeasible without a
reported collision candidate, and at `+0.16 m` there is no precontact IK.

The negative side first loses box 13 precontact IK at `-0.16 m` in the current
screen. This asymmetry is why Y error must be measured in both directions.

## Complete-Task Findings

### Nominal pose: complete but slow bridge

The nominal `dx=0, dy=0, yaw=0` complete task passes:

- `25/25` boxes, 15 cycles, 10 simultaneous dual cycles;
- `28,632` frames and `972` interpolated edges;
- maximum joint step `2.998282 deg`, flips `0`;
- maximum carried tilt `0.192639 deg`;
- maximum task core `0.727836 s`;
- maximum whole-body bridge `5.616908 s`.

It is classified `degraded`, not failed, because the bridge exceeds the
3-second performance target.

### `dx=+0.20 m`: complete planning, validation collision

All pickups and entry bridges plan, but the complete FCL replay fails at box
23 during `post_release_stow_outside_warehouse`:

```text
collision:left_link7<->warehouse_right_wall
```

This demonstrates why pickup-only screening cannot replace full replay
validation.

### `dy=+0.10 m`: screen pass, carried-box collision

The high-risk pickup screen passes. The complete replay then fails in dual
group `22+24`, frame `331`:

```text
edge collision:dual_carried_box_left<->wall_box_23
```

### Joint effects

`dx=+0.05 m, dy=+0.10 m, yaw=+5 deg` fails on box 23 with no precontact IK for
any tested front/top suction and lift combination, even though `dy=+0.10 m`
alone passes the pickup screen.

The opposite corner `dx=-0.05 m, dy=-0.10 m, yaw=-5 deg` passes the original
seven-group screen but fails the complete planner at box 13. The sentinel set
was consequently expanded to include group 9 / box 13.

## Failure Classes Observed

- `ik_or_reachability`;
- `retreat`;
- robot self-collision;
- carried box against remaining wall box;
- robot against warehouse wall;
- planning-time target exceeded.

No carried-box tilt failure has been observed in this batch. The project keeps
5-degree warning and 15-degree failure gates so future wider cases will be
classified if this occurs; no synthetic tilt failure is reported.

## Next Batch

1. Run complete tasks at `dy=-0.14, -0.10, +0.05 m` to distinguish screen and
   full-task safe regions.
2. Bisect the Y brackets near `-0.15 m` and `+0.11 m`.
3. Screen all 75 joint-matrix points, then execute complete tasks for every
   screen pass and every point adjacent to a pass/fail boundary.
4. Optimize repeated dominant failures and rerun the same immutable matrix for
   before/after success-rate comparison.

The task remains `In Progress` until the full joint matrix and optimization
comparison are complete.
