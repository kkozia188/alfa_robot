# V3 Mobile-Base 5x5 Synchronized Pickup

This technical release preserves the validated moving-base V3 5x5 replay for
Linear issue `MOTION-261`. It extends the row-wise dual-arm pickup sequence with
a collision-clearing base route to a conveyor handoff area.

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

## Validated Result

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

## Scope Boundary

This is simulation and planning evidence, not a hardware execution trajectory.
It does not replace the independent conveyor-vehicle work in `MOTION-246` or
the stopped-conveyor pickup workflow in `MOTION-248`. Suction hardware,
calibrated conveyor geometry, navigation localization, and simultaneous vehicle
coordination remain outside this release.

Linear tracking: `MOTION-261`. Owner: Zhang Zelin.
