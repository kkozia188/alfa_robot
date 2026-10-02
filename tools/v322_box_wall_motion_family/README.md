# V3.2.2 Pose-Driven Box-Wall Motion Family

This isolated research project generalizes the certified V3.2.2 box-wall
planner from fixed trajectories to ID-to-Pose planning. It is based on
`robot_v3.2.2-suction`, upstream commit `d9c330ce`, and the local Tool0 `+Z
0.151 m` offset.

The MOTION-261 worktree, validation cache, and 2026-10-01 release are read-only
inputs. Development lives in:

- Git worktree: `/home/tim/alfa_robot-v322-box-wall-motion-family`
- Validation workspace: `/home/tim/alfa_robot-v322-box-wall-motion-family-validation`

## Two-Box ID Contract

The primary request contains exactly two box IDs. The system maps each ID to a
box-center 6D Pose from the configured wall geometry, then selects:

- left/right arm assignment;
- one common shared-lift position;
- one common front/top suction template;
- approach and retreat distance;
- mobile-base station;
- IK candidates seeded from the current 17-axis state.

IDs are geometry lookup keys. No per-ID trajectory or per-ID action override is
used. Current wall occupancy is supplied by the scene state, not encoded into
the two-ID command. See `pair_contract.schema.json` and
`examples/cross_row_pair_requests.json`.

The lower-level single-Pose contract remains available for reachability and
template development.

Two initial templates are implemented:

- `front_face_extract`: front approach and a preferred 0.35 m loaded retreat,
  with 0.30/0.25/0.20 m space-constrained fallbacks;
- `top_face_extract`: top approach and 0.20 m vertical loaded retreat.

Both continue with the certified mobile-base policy: back off 2.35 m, move 1.50
m to robot-right, release, stow outside the warehouse, and return.

## Cross-Row Result

Three representative cross-row ID pairs freshly planned and completed:

- `1 + 10`: rows 1/2, common lift `0.00 m`, retreat `0.35 m`
- `12 + 20`: rows 3/4, common lift `-0.50 m`, retreat `0.25 m`
- `17 + 25`: rows 4/5, common lift `-0.75 m`, retreat `0.35 m`

Results:

- Cross-row requests: `3/3`; boxes: `6/6`
- One-sided attachment frames: `0`
- Continuous operation boundaries: `3/3`
- MoveIt/FCL: `5722` frames plus `366` 1-degree/1-centimetre edge samples
- Maximum joint step: `2.953128 deg`; joint flips: `0`
- Maximum carried-box tilt, left/right: `2.295180 / 0.065597 deg`
- Production shell: `21` links and `49` visual meshes

## Single-Pose Reference

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
python3 compile_pair_family.py \
  --requests examples/cross_row_pair_requests.json \
  --family config/action_family.json \
  --output /tmp/v322-pair-plan.json
```

Open the complete production-shell Rerun:

```bash
rerun --new \
  /home/tim/alfa_robot-v322-box-wall-motion-family-validation/results/v322-cross-row-pair-full.rrd
```

The pair acceptance results are `results/cross-row-pair-acceptance.json/.md` in
the validation workspace. The earlier single-Pose result remains in
`results/acceptance-report.json/.md`.

## Limits

- Three representative cross-row pairs are certified. This does not yet prove
  all `25 choose 2` combinations; infeasible pairs return candidate failure
  reasons.
- The backend currently assumes the configured axis-aligned wall geometry.
- The lower-row entry bridges are the current timing hotspots; the maximum is
  `7.389 s`.
- Navigation localization, suction IO, conveyor control, and hardware
  execution remain outside this simulation certificate.
- Any URDF, SRDF, Tool0, wall geometry, station Pose, or start-state change
  invalidates these trajectories and requires fresh planning plus full FCL.
