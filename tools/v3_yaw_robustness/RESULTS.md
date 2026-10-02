# MOTION-274 Result

The corrected implementation uses the V3.2.2 production URDF and the complete
MOTION-261 5x5 dual-arm task.

## Acceptance Result

- Yaw cases: `-5 deg .. +5 deg`, 1 degree steps, `11/11` passed.
- Boxes: `275/275`.
- Cycles: `165/165`.
- Simultaneous dual cycles: `110/110`.
- One-sided attachment frames: `0`.
- MoveIt/FCL frames: `301,518`.
- Additional 1 degree / 1 cm edge samples: `11,540`.
- Maximum joint step: `2.998284 deg`.
- Joint flips: `0`.
- Maximum carried-box tilt, left/right: `0.417698 / 0.800591 deg`.
- Task core average/max: `0.363 / 3.275 s`.
- Task cores below 3 s: `274/275`.
- Base travel per yaw case: `115.75 m`.
- Cache loading counted as planning time: `false`.

Detailed per-yaw results are generated at:

- `data/ik_benchmark/v3_yaw_robustness/release/yaw-certificate.json`
- `data/ik_benchmark/v3_yaw_robustness/release/yaw-results.csv`
- `data/ik_benchmark/v3_yaw_robustness/release/REPORT.md`

## Production-Shell Rerun

Each recording contains 21 links and 49 visual meshes from
`robot_v3.2.2-suction`.

| Case | Size | SHA-256 |
| --- | ---: | --- |
| yaw -5 deg | 27,975,418 | `a5e954ca5c2e7d893d9c587df7cb559b8b79f32545545bdac4c53d26e72b74d1` |
| yaw 0 deg | 28,192,140 | `a1c745c302fc7c0ad2585d9de49c3fd2c5c66955392d6c80df043906e39ad7e1` |
| yaw +5 deg | 28,413,242 | `0f18c0fa0d827100223fb972d03dfa345a6725d6a034205e9e6337823661d64c` |

All three files pass `rerun rrd verify`.

Production-shell RRDs are also available locally for every intermediate integer
yaw. Open any certified angle through the interactive terminal selector:

```bash
cd /home/tim/alfa_robot-motion274-docking-error && \
/usr/bin/python3 tools/v3_yaw_robustness/v3_yaw_rerun_demo.py
```

Aggregate certificate SHA-256:

- `yaw-certificate.json`: `85e615f91a6c95c60f0150989bd9bafee4fe980c72658460d5a2e584c39bfb39`
- `yaw-results.csv`: `a68a9831c25ce118ab4ff129911378c54c24d5dd82ee2443f7cea45a26549dfa`

## Software Verification

- `alfa_robot_moveit_config`: 25/25 relevant tests passed, including the new
  yaw contract. The pre-existing `test_v311_analytic_moveit_fk` is excluded
  because it loads the current V3.2.2 description while asserting V3.1.1 FK.
- `alfa_robot_analytic_ik`: 3/3 passed, including 256 random MoveIt FK/IK
  comparisons per V3.2.2 arm.
- `alfa_robot_rerun`: 2/2 passed.
- Yaw orchestration helpers: 4/4 passed.
- The existing Stage Action source and launch are byte-identical to `610d612`.

## Boundary

This result does not claim navigation, localization, suction hardware, or
execution-control acceptance. The certified condition is fixed station x/y
with actual base yaw in `[-5 deg, +5 deg]` at integer-degree samples.
