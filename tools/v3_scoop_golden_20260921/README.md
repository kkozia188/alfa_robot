# V3.1.1 Scoop 5x5 Golden Research Baseline

This directory indexes the user-approved `alfa_robot/MoveIt` research baseline:

`2026-09-21-golden-box7-front-box20-fast`

The binary recording, plan cache, reports, and exact source snapshot are published as assets on
the GitHub pre-release `v3-scoop-golden-2026.09.21` rather than committed to normal Git history.

## Algorithm

The planning chain is:

```text
analytic redundant IK
-> 7-axis joint shortcut
-> collision-window detection
-> 1-joint local RRT
-> 2-joint local RRT
-> TCP-guided interpolation
-> full 7-axis RRT fallback
-> shortcut smoothing and full validation
```

Local RRT keeps all unselected joints on the nominal shortcut and searches only path progress plus
one or two joint offsets. Every accepted edge is rechecked for joint bounds, full MoveIt/FCL
collision, attached-box tilt, and TCP corridor limits.

Bounded revolute joints are scored using executable travel rather than treating `+/-pi` as a free
wrap boundary. This removed the first-box J1 motion from `+155deg -> -151deg` and selected
`+155deg -> +167.1deg`, eliminating roughly 306 degrees of physical winding.

## Golden Result

- Complete grasp tasks: `25/25`
- Validated inter-box transitions: `25/25`
- Selected task core planning time: `25/25 <= 5s`
- Core planning average / median / maximum: `1.69s / 1.22s / 4.90s`
- Box 7: front suction at `updown=0.00m`, core planning `1.56s`
- Box 20: collision-validated waypoint replay, core planning `0.72s`
- Complete sequence joint travel: `41476.71deg`
- Carried-box inverted frames: `0`
- Large joint-flip events: `0`

Validation completed locally: 20/20 CTest targets, 28/28 policy tests, RRD verification, and
SHA-256 verification.

## Boundary

This is an archived research implementation and migration reference. It does not replace or
silently modify the current MotionSystem Stage Action/FJT runtime on `alfa_v3_dev`. In particular,
the separate MotionSystem Preview warning about J7 branch winding remains in force until these
ideas are reimplemented and revalidated against the current Profile, Scene, and Stage contracts.

The exact implementation used to generate the golden result is in the release asset
`source-snapshot.tar.gz`.

## Replaying

After downloading the release assets into one directory:

```bash
sha256sum -c SHA256SUMS
/home/tim/.local/bin/rerun --new v3-scoop-x075-sequence.rrd
```

Linear tracking: `MOTION-223`.
