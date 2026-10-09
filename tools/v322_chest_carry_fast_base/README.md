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

## Certified Geometry

- Front-suction box center in `base_footprint`: `X=0.75 m`,
  `Y=+/-0.32 m`, `Z=0.95 m`.
- Top-suction box center: `X=0.90 m`, `Y=+/-0.40 m`, `Z=0.85 m`.
- Shared carry lift: `updown=-0.30 m`.
- The arm/lift motion finishes before any base backoff frame.
- Base route stays `2.35 m` backward then `1.50 m` robot-right.
- Base replay speed is `0.35 m/s` instead of `0.10 m/s`; route geometry and
  the 1 cm collision sampling are unchanged.

The certified result contains 25/25 boxes, 15/15 cycles, 37,529 MoveIt/FCL
frames, and 14,442 extra 1 degree / 1 cm edge samples. See
`results/v322_chest_carry_fast_base/acceptance-report.md`.

## Open Rerun

```bash
cd /home/tim/alfa_robot-v322-chest-carry-fast-base
/usr/bin/python3 tools/v322_chest_carry_fast_base/open_rerun.py
```

Use `--keyframe` to inspect the first dual chest-carry-ready pose. Both entry
points verify the RRD before opening it.

## Reproduce

Use the isolated build overlay in `install-chest-carry`. The workflow is:

1. `plan_chest_payloads.py` generates 25 fresh pickup-to-chest paths.
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
