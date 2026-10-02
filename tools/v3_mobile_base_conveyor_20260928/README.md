# V3 Mobile-Base 5x5 Synchronized Pickup

This technical release preserves the validated moving-base V3 5x5 replay for
Linear issue `MOTION-261`. The complete 25-box evidence was generated against
the V3.1.1 hybrid model. The branch has since been rebased onto the V3.2.2
production suction model and now carries the model-specific IK migration and
compatibility evidence described below.

## Task Contract

- Process rows from top to bottom.
- For every row, pick the outer pair first, the adjacent inner pair second, and
  the center box last with one arm.
- Execute all ten two-box groups with both arms attached simultaneously.
- Keep the base at `Y=0` while entering and picking from the wall.
- Use a front clearance of `0.85 m` for the upper three rows and `0.60 m` for
  the lower two rows by default. The certified input ranges are `0.80-0.90 m`
  and `0.55-0.65 m`, measured from the vehicle front contact plane to the box
  front contact plane and rounded half-up to the nearest centimeter.

## Base Route

Every pickup cycle uses the same rectangular route:

```text
pickup at Y=0
-> reverse 2.35 m along -X
-> move 1.50 m to robot-right along -Y
-> release at the conveyor reference
-> return 1.50 m along +Y
-> move forward 2.35 m along +X
```

The reverse leg clears the warehouse side wall before lateral travel. The
conveyor cuboid in Rerun is a visual handoff reference, not calibrated site
geometry or an active collision object.

## Historical V3.1.1 Result

- Completed boxes / rows / cycles: `25/25`, `5/5`, `15/15`
- Simultaneous dual-arm cycles: `10/10`
- One-sided attachment frames in dual cycles: `0`
- Replay frames: `28,084`
- Additional MoveIt/FCL edge samples: `6,778`
- Maximum replay joint step: `2.996970 deg`
- Joint flip events: `0`
- Maximum carried-box tilt: left `94.786 deg`, right `93.848 deg`
- Boxes below the 3-second task-core target: `25/25`
- Maximum task-core planning time: `1.504 s`
- Maximum new bridge planning time: `1.663 s`
- Total base travel: `115.75 m`
- `rerun rrd verify`: passed

These metrics remain valid for `robot_v3.1.1-hybrid` only. They are not a
V3.2.2 certificate.

## V3.2.2 Migration Status

The branch is rebased onto `alfa_v3_dev@d9c330ce`, which introduces
`robot_v3.2.2-suction` and the authoritative Tool0 local `+Z 0.151 m` offset.

- Added exact V3.2.2 left/right fixed transforms to the redundant seven-axis
  IK model.
- Added a deterministic bounded numerical solver for the V3.2.2 geometry and
  seed-continuous Cartesian solving; the V3.1.1 analytic path remains unchanged.
- MoveIt/FK comparison passed for 256 random samples per arm. Maximum FK
  position error was below `6.2e-12 m`.
- The old 25-box replay is deliberately rejected on V3.2.2 at operation 1,
  frame 18: `base_link <-> right_link5`.
- The complete V3.2.2 task finishes `25/25` boxes in 15 cycles. All `10/10`
  pair cycles are simultaneous, with zero one-sided attachment frames.
- Fresh pickup task core total / average / maximum: `8.594 / 0.344 / 1.352 s`;
  `25/25` boxes are below 3 seconds.
- Deterministic per-operation MoveIt/FCL validation, including process startup,
  is below 3 seconds for `16/16` operations; maximum `0.565 s`. Cache loading
  is not counted.
- The final replay passes `27,372` frames and `1,020` additional edge samples,
  with maximum joint step `2.943380 deg` and zero joint flips.
- The complete Rerun uses the V3.2.2 production shell (`49` visual meshes).
- All `11 x 11 = 121` upper/lower clearance pairs pass complete MoveIt/FCL
  replay validation: `3,288,544` frames and `123,074` interpolated edge
  samples in total.
- All `275/275` fresh selected profile tasks are below the 3-second core
  planning target; the maximum is `2.696674 s`. Cache loading is excluded.
- Every pair keeps pickup entry at `Y=0`, has `10/10` simultaneous dual-arm
  groups with zero one-sided attachment frames, reverses `2.35 m`, moves
  `1.50 m` to robot-right, releases, and returns to the pickup station.

## Evidence

The GitHub pre-release attached to tag
`v3-mobile-base-conveyor-2026.09.28` contains:

- `v3-scoop-u085-l060-back235-y150-conveyor.rrd`: complete interactive replay.
- `v3-scoop-u085-l060-back235-y150-conveyor-replay.json`: complete joint and
  base replay data.
- `v3-scoop-u085-l060-back235-y150-conveyor-validation.json`: full MoveIt/FCL
  validation report.
- `v3-scoop-u085-l060-back235-y150-conveyor-summary.json`: task summary.
- `v3-scoop-u085-l060-back235-y150-conveyor-metrics.csv`: replay metrics.
- `v3-scoop-u085-l060-back235-y150-conveyor-box-planning.csv`: per-box planning
  measurements.
- `rerun-back235-conveyor-y150.png`: final Rerun overview.
- `v3-mobile-base-conveyor-source-snapshot.tar.gz`: exact research source
  snapshot used to produce the evidence.
- `MANIFEST.json` and `SHA256SUMS`: certificate and integrity checks.

The complete V3.2.2 replacement is attached to tag
`v3-mobile-base-conveyor-v322-tool0151-2026.10.01`. Its Rerun is
`v322-conveyor-production-shell.rrd`; its exact source snapshot, plan cache,
certified replay, full validation, per-operation runtime validation and hashes
are published alongside it.

## Replay

```bash
sha256sum -c SHA256SUMS
rerun --new v3-scoop-u085-l060-back235-y150-conveyor.rrd
```

This command opens the historical V3.1.1 replay. Do not execute or validate it
as a V3.2.2 trajectory.

For V3.2.2, download the newer pre-release and run:

```bash
sha256sum -c SHA256SUMS
rerun --new v322-conveyor-production-shell.rrd
```

After placing the range certificate directory at
`data/ik_benchmark/v3_scoop_5x5/range_certification_v322_tool0151`, the
interactive certified-range entry is:

```bash
cd /home/tim/alfa_robot-alfa_v3_dev
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run alfa_robot_moveit_config v3_scoop_5x5_conveyor_demo.py
```

It prompts for the first-three-row and last-two-row clearances, verifies the
V3.2.2 certificate and replay hashes, records the selected RRD when needed,
verifies it, and opens Rerun.

## Scope Boundary

This is simulation and planning evidence, not a hardware execution trajectory.
It does not replace the independent conveyor-vehicle work in `MOTION-246` or
the stopped-conveyor pickup workflow in `MOTION-248`. Suction hardware,
calibrated conveyor geometry, navigation localization, and simultaneous vehicle
coordination remain outside this release.

V3.2.2 migration reports are stored in `v322_migration/`.

Linear tracking: `MOTION-261`. Owner: Zhang Zelin.
