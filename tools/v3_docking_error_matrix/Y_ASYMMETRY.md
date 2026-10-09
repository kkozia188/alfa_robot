# Why the Y Range Is Asymmetric

Date: 2026-10-09

The measured complete-task Y range is `-0.120 .. +0.0875 m`, with the nearest
failures at `-0.130 m` and `+0.100 m`. This is not evidence of a coordinate-sign
bug. The scene geometry is close to symmetric, but the executed task is not a
left/right mirror experiment.

## Task Asymmetries

- The center boxes use a fixed arm policy: boxes 3, 8, 13, and 18 use the right
  arm; box 23 uses the left arm.
- Each row is removed in the fixed order outer pair, inner pair, then center.
  During the 22+24 dual pickup, box 23 is still present.
- Every loaded cycle backs away and then moves 1.50 m to robot-right toward the
  conveyor. The route is not alternated or mirrored for positive and negative Y.
- Bottom-row boxes use top suction and different retreat distances, so the two Y
  signs exercise different arm, carried-box, and remaining-box clearances.

## Observed Failure Evidence

- At `Y=-0.130 m`, box 18, assigned to the right arm, fails the final Cartesian
  approach sample. The failure is kinematic/approach feasibility, before replay
  validation.
- At `Y=+0.100 m`, pickup planning completes, but full edge validation fails in
  group 22+24: `dual_carried_box_left <-> wall_box_23`.
- At `Y=+0.120 m`, the sentinel screen reaches box 23 and then reports
  `base_link <-> left_link5` during Cartesian retreat.

The positive and negative boundaries therefore fail in different operations and
with different mechanisms. A symmetric numeric Y interval would discard real
task evidence. Any future claim of symmetric Y tolerance requires a mirrored
arm-allocation and mirrored conveyor-route experiment, not just symmetric input
samples.
