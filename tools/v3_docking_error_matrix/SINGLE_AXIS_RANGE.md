# Independent X-Clearance, Y, and Yaw Ranges

Date: 2026-10-09

X is the physical distance from the vehicle-front contact plane to the box-front
contact plane. It is not a common base `dx`. Y and Yaw remain independent base
pose errors. Inputs finer than 1 cm are rounded half-up to the nearest centimeter.

“Complete” means all 25 boxes, 15 cycles, 10 simultaneous dual-arm cycles, and
complete MoveIt/FCL frame plus interpolated-edge validation.

| Experiment | Complete functional range | Nearest tested failures |
| --- | ---: | ---: |
| Rows 1-3 X clearance; rows 4-5 fixed at 0.60 m | `0.74 .. 1.02 m` | `0.73 / 1.03 m` |
| Rows 4-5 X clearance; rows 1-3 fixed at 0.85 m | `0.36 .. 0.79 m` | `0.35 / 0.80 m` |
| Y; X clearances nominal and Yaw=0 | tested pass extrema `-0.120 / +0.0875 m` | `-0.130 / +0.080 / +0.100 m` |
| Yaw; X clearances nominal and Y=0 | `-5.0 .. +5.0 deg` | `-5.15625 / +5.3125 deg` |

## X Guarantee Levels

The recommended input ranges remain:

- rows 1-3: `0.80..0.90 m`, nominal `0.85 m`;
- rows 4-5: `0.55..0.65 m`, nominal `0.60 m`.

These ranges have the strongest evidence: all `121` cross-combinations passed,
with `3,288,544` validated frames, `123,074` interpolated edges, and all `275`
selected pickup task cores below 3 seconds; maximum `2.697 s`.

The independent one-centimeter extension establishes wider functional limits:

- rows 1-3 at `0.74..1.02 m`; `0.74..0.76 m` completes but raises a carried-box
  tilt warning, so the posture-qualified interval is `0.77..1.02 m`;
- rows 4-5 at `0.36..0.79 m`; posture remains below the warning gate, but
  `0.51/0.52 m` exceed 3 seconds in selected task-core planning. The contiguous
  posture-plus-task-core-under-3s interval around nominal is `0.53..0.79 m`.

Every fresh extension run has at least one whole-body entry bridge over 3
seconds. The wider intervals are functional robustness results, not an
all-planning-under-3s guarantee. Details are in `X_CLEARANCE_RANGE.md`.

## X Boundary Failures

- rows 1-3 at `0.73 m`: `left_link7 <-> warehouse_right_wall` during
  post-release stow;
- rows 1-3 at `1.03 m`: box 13 has no precontact IK;
- rows 4-5 at `0.35 m`: both link7 bodies collide with the warehouse rear wall
  at the 17+19 transition endpoint;
- rows 4-5 at `0.80 m`: box 21 top-suction precontact has no IK.

## Y Sweep and Asymmetry

- `Y=-0.120 m`: complete pass; maximum task core `1.037 s`, maximum bridge
  `10.024 s`;
- `Y=-0.130 m`: box 18 right-arm Cartesian approach fails;
- `Y=+0.0875 m`: complete pass; maximum task core `1.133 s`, maximum bridge
  `8.278 s`;
- `Y=+0.080 m`: a later fresh on-demand plan fails full edge validation in
  group 22+24, `dual_carried_box_left <-> wall_box_23`; this proves that the
  passing extrema are not a continuous all-values certificate;
- `Y=+0.100 m`: group 22+24 collides
  `dual_carried_box_left <-> wall_box_23`.

Y is not symmetric because the task policy is not mirrored: center boxes
3/8/13/18 use the right arm, box 23 uses the left arm, the removal order is
fixed, and every loaded cycle moves to robot-right. `Y_ASYMMETRY.md` records the
code contract, non-monotonic interior result, and different failure mechanisms.

## Yaw Sweep

- every integer Yaw from `-5 deg` through `+5 deg` completes;
- `Yaw=-5.15625 deg`: carried right box collides with wall box 23 during backoff;
- `Yaw=+5.3125 deg`: carried left box collides with wall box 23 during backoff.

## Evidence Coverage

- physical X rows: `77` (`73` complete, `4` failed), `1,989,100` frames and
  `77,072` interpolated edges;
- independent Y/Yaw rows: `75`, `467,459` frames and `17,598` interpolated
  edges;
- total official rows: `152`; common-`dx` X and combined-error rows: `0`;
- joint flips in successful X extension cases: `0`;
- carried-box tilt failures: `0`; X warning maximum: `6.093 deg`.
- one later on-demand Y probe at `+0.080 m` adds `24,392` checked frames and
  `906` edges and records the interior collision; official fixed-matrix counts
  above remain unchanged.

These are sampled empirical ranges for the current robot model, task order,
warehouse, box wall, suction modes, and conveyor route. They do not certify
navigation/localization error, combined errors, or physical hardware execution.
