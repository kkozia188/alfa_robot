# VAMP parallel planning experiment

This directory is isolated from the certified V3.2.2 trajectory artifacts. It
contains two separate experiments:

1. `benchmark_vamp_parallel.py` measures the official VAMP AVX2 planner with a
   precompiled Panda model, both serially and as process-parallel batches.
2. The chest-carry planner exposes `parallel_rrt_workers`. It races independent
   local shortcut-repair seeds on cloned MoveIt scenes and keeps MoveIt/FCL as
   the authoritative state and edge validator.

The stock VAMP package does not contain the ALFA V3.2.2 robot. Run
`inspect_v3_readiness.py` to produce the explicit integration-gap report. A
V3 production integration requires a sphere-only collision approximation and a
Cricket-generated SIMD kernel, followed by decision-by-decision comparison
against the existing MoveIt/FCL certificate.

Typical commands:

```bash
python3 -m venv /home/tim/.cache/alfa-vamp/venv
/home/tim/.cache/alfa-vamp/venv/bin/pip install \
  -r tools/v322_vamp_parallel/requirements-vamp.txt

/home/tim/.cache/alfa-vamp/venv/bin/python \
  tools/v322_vamp_parallel/benchmark_vamp_parallel.py

/home/tim/.cache/alfa-vamp/venv/bin/python \
  tools/v322_vamp_parallel/inspect_v3_readiness.py
```

The planner and chest payload generator both default to
`parallel_rrt_workers=1`. Pass `--parallel-rrt-workers 4` to the payload
generator for the explicit A/B experiment. The VAMP source and custom-robot
workflow are documented by the
[official project](https://github.com/KavrakiLab/vamp) and
[VAMP-MR](https://github.com/vamp-mr/vamp-mr).

## Golden-curve manifold search

The shortcut optimizer also has two opt-in reduced search modes:

- `local_rrt_mode:=golden_manifold` runs a two-dimensional OMPL RRT over
  `(progress, alpha)`.
- `local_rrt_mode:=golden_lattice` runs a monotonic layered search over the
  same two-dimensional manifold. `alpha=0` is the endpoint chord and
  `alpha=1` is the certified golden curve.

The `32 x 9` lattice is the current experiment baseline. It returns an exact
candidate for all 15 cycles, keeps the visibility graph as the selected path,
and cuts total planning core time from `26.150 s` to `14.079 s`. Its selected
replay matches all 16 operation frame hashes from the certified optimized
replay and passes 43,366 MoveIt/FCL frames plus 5,901 edge samples. This is a
VAMP-ready state-space reduction, not yet a VAMP collision backend.

### Real VAMP backend

`local_rrt_mode:=golden_vamp` is the real VAMP path. It uses VAMP's AVX2
RRT-Connect in the reduced `(progress, alpha)` space with four concurrent
Halton workers. A MoveIt/FCL-generated occupancy map is converted to VAMP
cuboid obstacles; the winning path is mapped back to all 16 replay joints and
must pass a second full MoveIt/FCL validation before it can compete with the
visibility-graph path.

Build with the installed VAMP CMake package:

```bash
colcon build --packages-select alfa_robot_moveit_config \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -Dvamp_DIR=/home/tim/.cache/alfa-vamp/prefix/share/cmake/vamp
```

The latest direct-pull run solved 60/60 workers. Pure VAMP search took
`5.648 ms` across all 15 cycles; FCL map construction took `210.020 ms` and
winning-candidate FCL confirmation took `107.132 ms`. The final safe-turn
replay passes 25,731 frames and 4,011 edge samples. The visibility graph still
wins the joint-travel quality gate; VAMP is a validated concurrent fallback,
not a forced replacement for a better path.
