# Preliminary Result: 2026-10-08

This is the first batch from the independent `MOTION-257` single-variable
X/Y/Yaw project. It validates the experiment machinery and records real
failure modes. Only one error variable is nonzero in official results.

## Dataset State

- Official single-axis poses after aggressive expansion: `73`.
- Imported complete baselines: `11` aligned X cases from MOTION-261 and `11`
  Yaw cases from MOTION-257.
- Newly screened Y cases: `21`, from `-0.20 m` to `+0.20 m` in `0.02 m`
  steps.
- New complete single-axis case rows: `23`.
- Official aggregate rows after scope-aware de-duplication: `107`.
- Existing and new full validations represented by the aggregate:
  `848,811` MoveIt/FCL frames and `32,267` interpolated edges.

Historical combined-error runs are excluded from the official success rate,
range report, and evidence package.

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
3. Complete denser independent X/Y/Yaw samples around each pass/fail boundary.
4. Optimize repeated dominant failures and rerun the same immutable single-axis matrix for
   before/after success-rate comparison.

The task remains `In Progress` until the independent sweeps and optimization
comparison are complete.

## Historical Common-dx Range Search: 2026-10-09 (Superseded)

These historical common-base-X results were:

- X: `-0.150 .. +0.0875 m`, with nearest failures at `-0.175/+0.100 m`;
- Y: `-0.120 .. +0.0875 m`, with nearest failures at `-0.130/+0.100 m`;
- Yaw: `-5.0 .. +5.0 deg`, with nearest failures at
  `-5.15625/+5.3125 deg`.

They are retained only as provenance and are not part of the current official X
claim. The current report is in `SINGLE_AXIS_RANGE.md` and
`SINGLE_AXIS_RANGE.json`.

## X Definition Correction: 2026-10-09

The previous common-`dx` X result is superseded. X now means physical clearance
from the vehicle-front contact plane to the box-front contact plane, with the two
row families varied independently.

- rows 1-3, rows 4-5 fixed at 0.60 m: complete at `0.74..1.02 m`; adjacent
  failures at `0.73/1.03 m`;
- rows 4-5, rows 1-3 fixed at 0.85 m: complete at `0.36..0.79 m`; adjacent
  failures at `0.35/0.80 m`;
- the original `0.80..0.90 m` and `0.55..0.65 m` ranges remain the strongest
  cross-grid certificate and the recommended operating inputs.

The extension was executed in one-centimeter increments. Details and quality
layers are in `X_CLEARANCE_RANGE.md`. `Y_ASYMMETRY.md` records why the measured
Y limits are not symmetric.

## Maximum Interactive Input Envelope: 2026-10-10

The unified terminal entry now accepts the full independent bounds: rows 1-3 X
`0.74..1.02 m`, rows 4-5 X `0.36..0.79 m`, Y
`-0.120..+0.0875 m`, and Yaw `-5..+5 deg`. X extension points reuse complete
one-centimeter results. Uncached Y and non-integer Yaw inputs run the complete
25-box planner and MoveIt/FCL validator before Rerun opens.

This is an accepted attempt envelope, not a continuous Y/Yaw certificate. A
fresh `Y=+0.080 m` on-demand run failed group 22+24 full edge validation with
`dual_carried_box_left <-> wall_box_23`, although separately planned
`+0.075 m` and `+0.0875 m` cases pass. Failed inputs are recorded and Rerun is
not opened.
