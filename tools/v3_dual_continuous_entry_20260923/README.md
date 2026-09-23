# V3 Dual-Arm Continuous-Entry Hybrid Planner

This release snapshot preserves the validated V3.1.1 scoop 5x5 row-wise dual-arm
planning result. It is a technical research artifact for the legacy
`alfa_robot/MoveIt` line, not a hardware execution trajectory.

## Planner

```text
analytic redundant IK
-> bounded-joint shortcut
-> TCP-cost path selection
-> collision-window detection
-> 1/2-joint reduced-DOF RRT repair
-> full-joint RRT fallback
-> whole-body waypoint validation
-> complete MoveIt/FCL edge validation
```

The whole-body transition contract covers the lift and both seven-axis arms.
Every deterministic replay checks the exact start fingerprint and revalidates
interpolated edges before accepting cached waypoints.

## Continuous Entry

- Box 2 no longer returns the left arm to `home`: the lift and both arms move
  directly from the previous release state to the box-2/4 precontact states.
- Box 25 no longer returns the right arm to `home`: the validated box-21
  left-arm/lift corridor is retained while a TCP-cost RRT moves the right arm
  directly around the warehouse wall to box 25.
- Bounded revolute joints use executable travel, preventing false `+/-pi`
  shortcuts and near-360-degree physical winding.

## Validated Result

- Completed boxes / rows / groups: `25/25`, `5/5`, `15/15`
- Replay frames: `4,678`
- Interpolated edge samples: `6,685` at `1 degree / 1 cm`
- Selected task-core total / average / maximum: `33.72 / 1.35 / 2.56 s`
- Boxes below the 3-second task-core target: `25/25`
- Total joint travel: `40,060.29 deg`
- Maximum replay joint step: `2.999 deg`
- Joint flip events: `0`
- Maximum carried-box tilt: left `93.584 deg`, right `94.391 deg`

Box 2 continuous entry:

- left TCP travel: `4.512 -> 2.863 m`
- left-arm excess travel: `512.8 -> 227.3 deg`
- reversals: `6 -> 2`
- validated waypoint planning: `0.368 s`

Box 25 continuous entry:

- right TCP travel: `3.006 -> 2.114 m`
- right-arm excess travel: `100.3 -> 89.3 deg`
- reversals: `3 -> 1`
- validated waypoint planning: `0.507 s`

## Evidence

- `v3-scoop-x075-dual-row-wise.rrd`: complete interactive Rerun replay.
- `dual-arm-5x5-complete-sequence.mp4`: complete 25-box page-playable video.
- `v3-dual-continuous-entry-rerun-demo.rrd`: focused box-2/25 Rerun replay.
- `v3-dual-continuous-entry-rerun-demo.mp4`: Rerun Viewer screen recording.
- `rerun-box02-continuous-direct.png`: box-2 continuous-entry visualization.
- `rerun-box25-continuous-direct.png`: box-25 continuous-entry visualization.
- `v3-scoop-x075-dual-row-wise-replay.json`: complete joint replay.
- `v3-scoop-x075-dual-row-wise-validation.json`: MoveIt/FCL report.
- `dual-source-snapshot.tar.gz`: exact implementation snapshot.

## Replay

```bash
sha256sum -c SHA256SUMS
rerun --new v3-scoop-x075-dual-row-wise.rrd
```

Linear tracking: `MOTION-235`.
