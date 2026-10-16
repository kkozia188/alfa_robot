# VAMP parallel experiment

## Result

- Official VAMP benchmark: 10000/10000 solved.
- Four-process throughput: 2528.1 plans/s, 1.81x versus serial.
- This number is for VAMP's precompiled Panda model, not ALFA V3.2.2.

## V3 profile

- 25-box task maximum: 2.228s; 25/25 under 3s.
- IK share of task core: 94.0%.
- FCL collision-check share of task core: 3.8%.
- Box 2 local-RRT A/B total speedup: 0.920x.
- Default-one-worker regression: 25/25 success, 25/25 under 3s, maximum 2.460s.

## Decision

Keep the production default at one worker. Four-worker local RRT remains an explicit experiment for difficult repair windows. A production VAMP backend requires a reviewed sphere model, Cricket-generated V3 kernel, attached-box support, and zero-false-negative comparison against MoveIt/FCL.
