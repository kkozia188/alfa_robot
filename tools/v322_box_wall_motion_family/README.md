# V3.2.2 Pose-Driven Box-Wall Motion Family

This isolated research project generalizes the certified V3.2.2 box-wall
planner from fixed box-number rules to a geometry contract. It is based on
`robot_v3.2.2-suction`, upstream commit `d9c330ce`, and the local Tool0 `+Z
0.151 m` offset.

The MOTION-261 worktree, validation cache, and 2026-10-01 release are read-only
inputs. Development lives in:

- Git worktree: `/home/tim/alfa_robot-v322-box-wall-motion-family`
- Validation workspace: `/home/tim/alfa_robot-v322-box-wall-motion-family-validation`

## Contract

The public task contains no box ID. It provides:

- box-center 6D Pose in the configured wall frame;
- box dimensions and wall geometry;
- front/top/left/right free-space measurements and clearance channels;
- allowed arms;
- current 17-axis state in the order emitted by `compile_family.py`.

The compiler derives row, column, a private collision-scene slot, grasp
template, arm, lift height, opposite-arm policy, contact Pose, and ranked
candidates. The private slot only opens the matching collision cell; it is not
used in action scoring.

Two initial templates are implemented:

- `front_face_extract`: front approach and 0.35 m loaded retreat;
- `top_face_extract`: top approach and 0.20 m vertical loaded retreat.

Both continue with the certified mobile-base policy: back off 2.35 m, move 1.50
m to robot-right, release, stow outside the warehouse, and return.

## Current Result

Five representative tasks, one per row, freshly planned `5/5`. The complete
continuous replay contains five full cycles and one base-position transition.

- Pickup planning mean/max: `398.807 / 598.449 ms`
- Entry bridge mean/max: `3120.566 / 7388.848 ms`
- Continuous MoveIt/FCL: `9002` frames plus `294` 1-degree/1-centimetre edge samples
- Maximum joint step: `1.681040 deg`; joint flips: `0`
- Maximum carried-box tilt, left/right: `0.004607 / 0.002494 deg`
- Continuous operation boundaries: `5/5`
- Production shell: `21` links and `49` visual meshes

The 25-box certified reference remains `25/25`, with every row at `5/5`.

## Quick Checks

```bash
cd /home/tim/alfa_robot-v322-box-wall-motion-family/tools/v322_box_wall_motion_family
python3 -m unittest discover -s tests -v
python3 compile_family.py \
  --tasks examples/representative_tasks.json \
  --family config/action_family.json \
  --output /tmp/v322-family-plan.json
```

Open the complete production-shell Rerun:

```bash
rerun --new \
  /home/tim/alfa_robot-v322-box-wall-motion-family-validation/results/v322-box-wall-motion-family-full.rrd
```

The consolidated machine-readable and human-readable results are
`results/acceptance-report.json` and `results/acceptance-report.md` in the
validation workspace.

## Limits

- Fresh acceptance currently uses five axis-aligned representative poses. The
  backend rejects box-frame orientation deviations over 5 degrees.
- The lower-row entry bridges are the current timing hotspots; the maximum is
  `7.389 s`.
- Navigation localization, suction IO, conveyor control, and hardware
  execution remain outside this simulation certificate.
- Any URDF, SRDF, Tool0, wall geometry, station Pose, or start-state change
  invalidates these trajectories and requires fresh planning plus full FCL.
