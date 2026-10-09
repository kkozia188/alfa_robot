# V3.2.2 Docking Error Matrix

This is the independent single-variable experiment project for Linear
`MOTION-257`. X, Y, and Yaw are scanned separately; the other two errors are
always fixed at zero.

## Coordinates

Errors are applied in `map` to both pickup stations:

- upper nominal base pose: `[-0.60, 0.00, 0.00]`;
- lower nominal base pose: `[-0.35, 0.00, 0.00]`;
- case pose: `[nominal_x + dx, dy, yaw]`.

The wall, boxes, warehouse, and conveyor references remain fixed in `map`.
Positive `dx` moves the vehicle closer to the wall. Positive `dy` moves it to
map-left. Yaw is counter-clockwise in degrees.

## Dataset Design

The default specification contains three independent datasets:

1. **Certified X baseline**: `dx=-0.05..+0.05 m`, 1 cm steps. These 11 aligned
   upper/lower cases are imported from the MOTION-261 121-pair certificate.
2. **Y baseline**: `dy=-0.20..+0.20 m`, 2 cm steps at nominal X/Yaw. These
   cases must be screened and then run as complete 25-box tasks.
3. **Existing Yaw baseline**: `-5..+5 deg`, 1 degree steps from the current
   MOTION-257 certificate.
A wider boundary probe covers the current experiment limits of `dx +/-40 cm`,
`dy +/-50 cm`, and `yaw +/-15 deg`. It is deliberately outside the intended certification domain
so that the experiment records real reachability, collision, and timeout
failures. Boundary points use eight high-risk groups first; a pass there is not
reported as a complete-task success.

## Failure Taxonomy

Every result records the first concrete failure and one normalized class:

- `ik_or_reachability` / `contact_or_suction`;
- `collision` / `retreat` / `whole_body_transition`;
- `planning_timeout` / `case_timeout` / `planner_process`;
- `trajectory_validation`;
- `carried_box_tilt`, `joint_flip`, or joint-step quality failure.

Successful trajectories are still marked `degraded` when any selected task or
entry bridge exceeds the 3-second target. Carried-box tilt warns at 5 degrees
and fails the experiment quality gate at 15 degrees. These are stricter
evaluation thresholds, not changes to the underlying planner safety limits.

## Generate The Matrix

```bash
cd /home/tim/alfa_robot-motion257-docking-matrix
/usr/bin/python3 tools/v3_docking_error_matrix/generate_matrix.py \
  --output-dir data/ik_benchmark/v3_docking_error_matrix/design
```

## Run Boundary Screens

Source the same V3.2.2/Yaw overlay used by MOTION-257, then run one axis at a
time. Start with one worker; each case launches ROS processes and owns a domain.

```bash
source /opt/ros/humble/setup.bash
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash
source /home/tim/alfa_robot-motion274-docking-error/install-motion274-yaw/setup.bash

/usr/bin/python3 tools/v3_docking_error_matrix/run_matrix.py \
  --matrix data/ik_benchmark/v3_docking_error_matrix/design/matrix-cases.json \
  --execution screen \
  --phases boundary_probe_y \
  --baseline-cache /home/tim/alfa_robot-alfa_v3_dev/data/ik_benchmark/v3_scoop_5x5/releases/2026-10-01-v322-tool0151-mobile-base-conveyor/v322-plan-cache.json \
  --output-root data/ik_benchmark/v3_docking_error_matrix/probes \
  --workers 1 --resume
```

Generate pass/fail brackets and suggested bisection cases:

```bash
/usr/bin/python3 tools/v3_docking_error_matrix/find_boundaries.py \
  --results-root data/ik_benchmark/v3_docking_error_matrix/probes \
  --output data/ik_benchmark/v3_docking_error_matrix/release/boundaries.json
```

## Run A Complete Single-Variable Case

```bash
/usr/bin/python3 tools/v3_docking_error_matrix/run_pose_case.py \
  --base-dx-m 0 --base-dy-m 0.0875 --yaw-deg 0 \
  --baseline-cache /home/tim/alfa_robot-alfa_v3_dev/data/ik_benchmark/v3_scoop_5x5/releases/2026-10-01-v322-tool0151-mobile-base-conveyor/v322-plan-cache.json \
  --output-root data/ik_benchmark/v3_docking_error_matrix/full
```

Only `case-result.json` values with all 25 boxes, 15 cycles, 10 simultaneous
dual cycles, and successful MoveIt/FCL validation count as complete results.
Official batch execution additionally rejects cases with more than one nonzero
error variable.

## Compact Evidence

```bash
/usr/bin/python3 tools/v3_docking_error_matrix/package_evidence.py \
  --data-root data/ik_benchmark/v3_docking_error_matrix \
  --output-dir data/ik_benchmark/v3_docking_error_matrix/evidence
```

The compact archive excludes candidate payloads, caches, and replay JSONs. It
contains the immutable matrix, result tables, per-case outcome records, failed
or successful validation reports, and a SHA-256 manifest.

## Scope

This project evaluates planning robustness. It does not certify localization,
navigation execution, suction hardware, or the physical accuracy of a docking
controller.

The latest result is documented in `SINGLE_AXIS_RANGE.md` and
`SINGLE_AXIS_RANGE.json`. Historical combined-error runs are excluded from
official aggregation and evidence.
