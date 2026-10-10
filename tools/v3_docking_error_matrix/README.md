# V3.2.2 Docking Error Matrix

This is the independent single-variable experiment project for Linear
`MOTION-257`. Physical X clearance, Y, and Yaw are scanned separately.

## Coordinates

Y and Yaw errors are applied in `map` to both pickup stations:

- upper nominal base pose: `[-0.60, 0.00, 0.00]`;
- lower nominal base pose: `[-0.35, 0.00, 0.00]`;
- case pose: `[nominal_x, dy, yaw]`.

X is not a common `dx` shift. It is the physical distance from the
vehicle-front contact plane to the box-front contact plane:

- rows 1-3 vary their clearance while rows 4-5 stay at `0.60 m`;
- rows 4-5 vary their clearance while rows 1-3 stay at `0.85 m`;
- terminal inputs are rounded half-up to the nearest `0.01 m`.

The wall, boxes, warehouse, and conveyor references remain fixed in `map`.
Positive `dy` moves the vehicle to map-left. Yaw is counter-clockwise in degrees.

## Dataset Design

The default specification contains three independent datasets:

1. **Certified X clearance baseline**: rows 1-3 use `0.80..0.90 m` with rows
   4-5 fixed at `0.60 m`; rows 4-5 use `0.55..0.65 m` with rows 1-3 fixed at
   `0.85 m`. The source is the MOTION-261 121-pair certificate.
2. **Y baseline**: `dy=-0.20..+0.20 m`, 2 cm steps at nominal X/Yaw. These
   cases must be screened and then run as complete 25-box tasks.
3. **Existing Yaw baseline**: `-5..+5 deg`, 1 degree steps from the current
   MOTION-257 certificate.
A wider boundary probe covers `dy +/-50 cm` and `yaw +/-15 deg`. X uses a
dedicated one-centimeter clearance sweep. It is deliberately outside the intended certification domain
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

## Run A Physical X-Clearance Sweep

```bash
/usr/bin/python3 tools/v3_docking_error_matrix/run_x_clearance_sweep.py \
  --family upper --values 0.79,0.78 \
  --baseline-cache /home/tim/alfa_robot-alfa_v3_dev/data/ik_benchmark/v3_scoop_5x5/releases/2026-10-01-v322-tool0151-mobile-base-conveyor/v322-plan-cache.json \
  --output-root data/ik_benchmark/v3_docking_error_matrix/x-clearance-upper \
  --workers 1 --ros-domain-base 120 --resume
```

Use `--family lower` for rows 4-5. The runner fixes the other row family at its
nominal clearance and rejects changing both families in one experiment.

## Open A Certified Rerun From Terminal Input

Run the unified interactive entry point:

```bash
cd /home/tim/alfa_robot-motion257-docking-matrix
/usr/bin/python3 tools/v3_docking_error_matrix/open_single_axis_rerun.py
```

It prompts for rows 1-3 X clearance, rows 4-5 X clearance, Y, and Yaw, then
opens the matching complete Rerun. Only one of physical X, Y, or Yaw may differ
from nominal in one run. The maximum accepted ranges are:

- rows 1-3 X: `0.74..1.02 m`; rows 4-5 must remain `0.60 m` outside the
  `0.80..0.90 x 0.55..0.65 m` 121-pair strong grid;
- rows 4-5 X: `0.36..0.79 m`; rows 1-3 must remain `0.85 m` outside the strong
  grid;
- Y: `-0.120..+0.0875 m`, rounded to the stable 1 mm case-ID grid while
  preserving the measured `+0.0875 m` boundary;
- Yaw: `-5..+5 deg`, rounded to `0.1 deg`.

Existing complete cases open immediately. A Y or non-integer Yaw point without
a cache first runs the complete 25-box planner and MoveIt/FCL validator and only
opens Rerun after it passes. These bounds are therefore the maximum **accepted
attempt envelope**, not a continuous success certificate. For example, a fresh
`Y=+0.080 m` run reached full replay validation but collided
`dual_carried_box_left <-> wall_box_23`; the entry point records that failure
and does not open its Rerun. Use `--retry-failed` only when a fresh randomized
replan is explicitly desired. Use CLI arguments to skip the prompts:

```bash
/usr/bin/python3 tools/v3_docking_error_matrix/open_single_axis_rerun.py \
  --upper-x-m 0.86 --lower-x-m 0.60 --y-m 0 --yaw-deg 0
```

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

The latest result is documented in `SINGLE_AXIS_RANGE.md`,
`X_CLEARANCE_RANGE.md`, and `Y_ASYMMETRY.md`. Historical common-`dx` and
combined-error runs are excluded from official aggregation and evidence.
