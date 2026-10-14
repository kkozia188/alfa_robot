# V3.2.2 Chest Carry + Fast Base

Independent research project built from the certified MOTION-261 replay. It
keeps the original model, wall, pickup order, base route, and validation
contract, but changes the cycle order to:

1. attach and retreat from the wall;
2. move the upright box to a close chest carry pose;
3. stop both arms at the chest-ready pose;
4. reverse `2.35 m` to the certified safe exit;
5. rotate `180 deg`, release both boxes at equal height, stow, turn back, and
   return without the former `1.50 m` lateral shuttle.

The source MOTION-261/MOTION-257 caches and certificates are read-only inputs.
All generated payloads, replay files, validation reports, and Rerun recordings
must be written to this project's own result directory.

## Direct-Pull Geometry

- Normal front-suction box center in `base_footprint`: `X=1.00 m`,
  `Y=+/-0.60 m`, `Z=1.20 m`.
- Top-row center box 3 uses `X=1.00 m`, `Y=-0.45 m`, `Z=1.20 m`.
- Bottom front-suction boxes 21/25 use `X=1.05 m`, `Y=+/-0.45 m`,
  `Z=1.10 m`.
- Top-suction box center: `X=0.90 m`, `Y=+/-0.40 m`, `Z=0.85 m`.
- Shared carry lift: `updown=-0.30 m`.
- The arm/lift motion finishes before any base backoff frame.
- The MOTION-258 route reverses `2.35 m`, rotates `180 deg`, and releases at
  the safe exit; lateral shuttle is `0 m`.
- Base translation speed is `0.25 m/s`; yaw and joint edges are checked at
  `1 deg`, translation and lift edges at `1 cm`.
- Every dual release has an FK-based equal-height gate of `1e-6 m`.

Each box now solves a fresh TCP pose from its own retreat state. Shared chest
joint profiles are disabled. The planner tries a direct Cartesian pull first,
then short inward TCP waypoints, a reduced local repair, and finally a bounded
joint fallback.

The optimized result uses the validated chest paths as golden curves, builds
an 80-node FCL visibility graph, applies local shortcuts, and retains the
original local-RRT windows where no direct edge is valid. A bounded local RRT
is also attempted, but approximate solutions are rejected. Final results:

- 25/25 selected task cores below 3 seconds;
- maximum selected task core 2.502 seconds;
- chest joint-travel cost reduced 6.5%;
- 43,366 frames plus 5,901 extra 1 degree / 1 cm edge samples pass MoveIt/FCL;
- cache loading is not counted as planning time.

The direct-pull result contains 25/25 boxes and 15/15 cycles. All 25 fresh box
plans are below 3 seconds (average `1.042 s`, maximum `2.228 s`). Twenty-one
pullbacks are direct Cartesian paths, two use short Cartesian waypoints, and
two use near-straight joint shortcuts. The worst TCP path/direct ratio is
`1.214`, with `7.16 cm` maximum line deviation. Full validation covers 29,299
frames plus 3,554 extra 1 degree / 1 cm edge samples. See
`results/v322_chest_carry_fast_base/direct-pull-v2-acceptance-report.md`.

The current safe-turn result composes that direct pullback with the route from
`MOTION-258@fb5cf97`. It passes 25/25 boxes and 15/15 cycles, with 26,194
MoveIt/FCL frames plus 6,494 edge samples. The maximum dual release height
error is `0.000000107 mm`; all 25 task cores remain below 3 seconds, with a
maximum of `2.228 s`. Released boxes remain visible in Rerun with their center
Z values so equal height can be inspected directly.

## Open Rerun

```bash
cd /home/tim/alfa_robot-v322-chest-turn-place
/usr/bin/python3 tools/v322_chest_carry_fast_base/open_rerun.py
```

Use `--keyframe` to inspect the first dual release after the 2.35 m backoff and
180-degree turn. Both entry points verify the RRD before opening it.

## Reproduce

Use the isolated build overlay in `install-chest-carry`. The workflow is:

1. `plan_chest_payloads.py --fresh-pose-goals --fast-probe` generates 25 fresh
   pickup-to-chest paths without shared joint profiles.
2. `plan_entry_bridges.py` generates 15 fresh whole-body entry bridges.
3. `assemble_replay.py` adds the chest-before-base task order.
4. `build_safe_turn_unload_replay.py` applies the certified MOTION-258 safe
   backoff/turn route without changing the chest pullback.
5. `validate_safe_turn_contract.py` verifies stage order, distance, yaw,
   zero lateral shuttle, and FK-derived equal release height.
6. `v3_dual_arm_5x5_replay_validator.launch.py` performs full MoveIt/FCL
   validation.
7. `v3_scoop_5x5_dual_rerun.py` records the production-shell RRD.

Run the local contract checks with:

```bash
/usr/bin/python3 tools/v322_chest_carry_fast_base/test_contract.py
/usr/bin/python3 tools/v322_chest_carry_fast_base/test_safe_turn_contract.py
```
