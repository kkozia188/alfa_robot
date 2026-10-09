# Maximum Observed Docking-Error Range

Date: 2026-10-09
Model: `robot_v3.2.2-suction`, Tool0 local `+Z 0.151 m`

“Pass” here means the complete 25-box task, all 15 cycles, and complete
MoveIt/FCL frame plus interpolated-edge validation. Pickup screens are used to
find boundaries but are never counted as complete success.

## Single-Axis Functional Range

With the other two errors fixed at zero, the largest complete ranges observed
in this experiment are:

| Axis | Complete pass range | Nearest tested failure |
| --- | ---: | ---: |
| X | `-0.150 .. +0.0875 m` | `-0.175 / +0.100 m` |
| Y | `-0.120 .. +0.0875 m` | `-0.130 / +0.100 m` |
| Yaw | `-5.0 .. +5.0 deg` | `-5.15625 / +5.3125 deg` |

Positive X moves the vehicle closer to the wall. Positive Y is map-left and
positive Yaw is counter-clockwise.

Boundary failures are concrete:

- X `-0.175 m`: box 13 has no precontact IK in the enhanced screen.
- X `+0.100 m`: complete pickup planning reaches box 18 and exceeds the
  2400-second case limit.
- Y `-0.130 m`: box 18 Cartesian approach fails.
- Y `+0.100 m`: complete group 22+24 collides
  `dual_carried_box_left <-> wall_box_23`.
- Yaw `-5.15625 deg`: backoff collides
  `dual_carried_box_right <-> wall_box_23`.
- Yaw `+5.3125 deg`: backoff collides
  `dual_carried_box_left <-> wall_box_23`.

These are axis-specific ranges. They cannot be multiplied into a rectangular
joint guarantee.

## Largest Fully Tested Joint Box

All eight corners pass for:

```text
|dx|   <= 0.0075 m   (7.5 mm)
|dy|   <= 0.0125 m   (12.5 mm)
|yaw|  <= 0.75 deg
```

Aggregate validation over the eight corners:

- completed corner cases: `8/8`;
- MoveIt/FCL frames: `175,756`;
- interpolated edges: `8,142`;
- maximum joint step: `2.987157 deg`;
- joint flips: `0`;
- maximum carried-box tilt: `0.435294 deg`.

The next tested nested box was:

```text
|dx|   <= 0.00875 m
|dy|   <= 0.01625 m
|yaw|  <= 0.875 deg
```

All four corners on its positive-Yaw face failed during base backoff with the
same collision: `dual_carried_box_left <-> wall_box_23`. Therefore the largest
fully tested joint guarantee remains the smaller 7.5 mm / 12.5 mm / 0.75 deg
box. This is an empirical sampled guarantee, not a proof for every continuous
interior pose.

## Planning-Time Boundary

Functional success and the 3-second planning target are different:

- maximum selected task core in the joint box: `3.310273 s`;
- maximum fresh whole-body bridge core: `11.048966 s`;
- the nominal zero-error case itself has a `5.616908 s` fresh bridge.

Consequently no non-empty “all fresh planning stages below 3 seconds” region
can be claimed with the current bridge algorithm. The joint box is functionally
valid but performance-degraded.

## Tilt

No carried-box tilt failure was observed. The maximum tilt in the largest
joint box is `0.435294 deg`, below the experiment’s 5-degree warning and
15-degree failure gates. No synthetic tilt failure is reported.

## Engineering Implication

The dominant joint-error limit is not raw arm reach. It is the current fixed
local backoff route while boxes 22/24 are carried past box 23 and the warehouse
side walls. Increasing the guaranteed joint box requires replanning the loaded
base route or changing extraction/order geometry, followed by rerunning the
same immutable matrix.
