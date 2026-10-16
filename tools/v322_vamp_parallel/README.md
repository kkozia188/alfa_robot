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
