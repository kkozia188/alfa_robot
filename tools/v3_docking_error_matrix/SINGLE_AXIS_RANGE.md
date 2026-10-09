# Independent X, Y, and Yaw Ranges

Date: 2026-10-09

Each experiment varies exactly one base-pose error. The other two variables
remain zero. Historical combined-error cases are not included in this report.

“Pass” means the complete 25-box task, 15 cycles, and complete MoveIt/FCL frame
plus interpolated-edge validation.

| Experiment | Complete functional range | Nearest tested failure |
| --- | ---: | ---: |
| X, with Y=0/Yaw=0 | `-0.150 .. +0.0875 m` | `-0.175 / +0.100 m` |
| Y, with X=0/Yaw=0 | `-0.120 .. +0.0875 m` | `-0.130 / +0.100 m` |
| Yaw, with X=0/Y=0 | `-5.0 .. +5.0 deg` | `-5.15625 / +5.3125 deg` |

Positive X moves the vehicle closer to the wall. Positive Y is map-left and
positive Yaw is counter-clockwise.

## X Sweep

- `X=-0.150 m`: complete 25/25 and FCL pass; maximum task core `1.040 s`,
  maximum fresh bridge `6.131 s`.
- `X=-0.175 m`: box 13 has no precontact IK in the enhanced screen.
- `X=+0.0875 m`: complete pass; maximum task core `3.625 s`, maximum bridge
  `8.135 s`.
- `X=+0.100 m`: box 18 pickup search exceeds the 2400-second case limit.

## Y Sweep

- `Y=-0.120 m`: complete pass; maximum task core `1.037 s`, maximum bridge
  `10.024 s`.
- `Y=-0.130 m`: box 18 Cartesian approach fails.
- `Y=+0.0875 m`: complete pass; maximum task core `1.133 s`, maximum bridge
  `8.278 s`.
- `Y=+0.100 m`: group 22+24 collides
  `dual_carried_box_left <-> wall_box_23`.

## Yaw Sweep

- Existing complete baseline: every integer Yaw from `-5 deg` through
  `+5 deg` passes.
- `Yaw=-5.15625 deg`: backoff collides
  `dual_carried_box_right <-> wall_box_23`.
- `Yaw=+5.3125 deg`: backoff collides
  `dual_carried_box_left <-> wall_box_23`.

## Planning-Time Qualification

The functional ranges do not satisfy an all-planning-below-3-seconds claim:

- the nominal zero-error case has a `5.617 s` fresh whole-body bridge;
- the positive X pass boundary has a `3.625 s` selected task core;
- the Y pass boundaries have fresh bridges up to `10.024 s`;
- the existing Yaw sweep has one selected task core at `3.275 s`.

These are functional ranges, not performance-qualified ranges.

## Tilt

No carried-box tilt failure was observed. The 5-degree warning and 15-degree
failure gates remain active; no synthetic failure is reported.

## Evidence Coverage

- official result rows: `107`;
- screen rows: `62`;
- new complete single-axis cases: `23`;
- imported complete X/Yaw baselines: `22`;
- combined-error rows: `0`;
- MoveIt/FCL frames: `848,811`;
- interpolated edge samples: `32,267`.

## Scope

The values are sampled empirical ranges for the current task order, warehouse,
box wall, route, robot model, and planner. They do not imply any combined-error
guarantee and do not certify navigation, localization, or hardware execution.
