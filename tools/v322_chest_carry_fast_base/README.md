# V3.2.2 Chest Carry + Fast Base

Independent research project built from the certified MOTION-261 replay. It
keeps the original model, wall, pickup order, base route, and validation
contract, but changes the cycle order to:

1. attach and retreat from the wall;
2. move the upright box to a close chest carry pose;
3. stop both arms;
4. move the base backward and then to the right-side conveyor;
5. release, stow outside the warehouse, and return.

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
- Base route stays `2.35 m` backward then `1.50 m` robot-right.
- Base replay speed is `0.35 m/s` instead of `0.10 m/s`; route geometry and
  the 1 cm collision sampling are unchanged.

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

## Open Rerun

```bash
cd /home/tim/alfa_robot-v322-chest-carry-fast-base
/usr/bin/python3 tools/v322_chest_carry_fast_base/open_rerun.py
```

Use `--keyframe` to inspect the first direct-pull dual chest-carry-ready pose
and its pullback curves. Both entry
points verify the RRD before opening it.

## Reproduce

Use the isolated build overlay in `install-chest-carry`. The workflow is:

1. `plan_chest_payloads.py --fresh-pose-goals --fast-probe` generates 25 fresh
   pickup-to-chest paths without shared joint profiles.
2. `plan_entry_bridges.py` generates 15 fresh whole-body entry bridges.
3. `assemble_replay.py` adds the chest-before-base task order and 0.35 m/s
   timing metadata.
4. `v3_dual_arm_5x5_replay_validator.launch.py` performs full MoveIt/FCL
   validation.
5. `v3_scoop_5x5_dual_rerun.py` records the production-shell RRD.
6. `summarize_result.py` generates the acceptance report.

Run the local contract checks with:

```bash
/usr/bin/python3 tools/v322_chest_carry_fast_base/test_contract.py
```
