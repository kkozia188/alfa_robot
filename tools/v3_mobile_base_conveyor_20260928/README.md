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
  the lower two rows.

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

The branch is rebased onto `alfa_v3_dev@7ac01bd7`, which introduces
`robot_v3.2.2-suction`.

- Added exact V3.2.2 left/right fixed transforms to the redundant seven-axis
  IK model.
- Added a deterministic bounded numerical fallback for the V3.2.2 geometry;
  the V3.1.1 analytic path remains unchanged.
- MoveIt/FK comparison passed for 256 random samples per arm. Maximum FK
  position error was below `6.2e-12 m`.
- The old 25-box replay is deliberately rejected on V3.2.2 at operation 1,
  frame 18: `base_link <-> right_link5`.
- A newly planned box-1 pickup completed in `1.703 s`. Its full route through
  `2.35 m` backoff and `1.50 m` right shuttle passed `812` frames and `625`
  additional MoveIt/FCL edge samples.

The box-1 result is a migration smoke test, not a replacement 25-box
certificate. The remaining rows, synchronized pairs, inter-group transitions,
and complete Rerun must be replanned and fully validated before this pull
request is ready to merge.

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

## Replay

```bash
sha256sum -c SHA256SUMS
rerun --new v3-scoop-u085-l060-back235-y150-conveyor.rrd
```

This command opens the historical V3.1.1 replay. Do not execute or validate it
as a V3.2.2 trajectory.

## Scope Boundary

This is simulation and planning evidence, not a hardware execution trajectory.
It does not replace the independent conveyor-vehicle work in `MOTION-246` or
the stopped-conveyor pickup workflow in `MOTION-248`. Suction hardware,
calibrated conveyor geometry, navigation localization, and simultaneous vehicle
coordination remain outside this release.

V3.2.2 migration reports are stored in `v322_migration/`.

Linear tracking: `MOTION-261`. Owner: Zhang Zelin.
