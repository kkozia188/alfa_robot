# V3.2.2 VAMP paper reproduction

This experiment compares exactly two single-worker RRT-Connect backends:

1. OMPL RRT-Connect with MoveIt/FCL validation.
2. VAMP RRT-Connect with a Cricket-generated SIMD FK/collision kernel.

Both backends receive the same online-IK start/goal pair. Timed planning starts
at the RRT-Connect solve call and ends at the first collision-valid path. IK,
path simplification, ROS startup, recording, and independent final FCL audit are
outside the paper-compatible timing interval.

No historical path, intermediate waypoint, Cartesian planner, shortcut,
visibility graph, or fallback is allowed.

## Model preparation

```bash
/usr/bin/python3 tools/v322_vamp_paper_repro/prepare_model.py
```

The generated manifest locks the certified V3.2.2 source hashes and verifies
the exact 15D order: `updown`, seven left-arm joints, and seven right-arm
joints. The two head joints are fixed at zero for this benchmark.

The project driver uses Foam's normal mesh-repair and SphereTree pipeline with
a fixed branch count. This avoids Foam's signed-volume heuristic, which is not
defined for the V3.2.2 thin-shell collision meshes:

```bash
PYTHONPATH=/home/tim/.cache/alfa-vamp/foam \
  /home/tim/.cache/alfa-vamp/venv/bin/python \
  tools/v322_vamp_paper_repro/foam_spherize_model.py \
  results/v322_vamp_paper_repro/model/alfa-v322-15d-mesh.urdf \
  results/v322_vamp_paper_repro/model/alfa-v322-15d-foam-tuned.urdf \
  --exclude-link base_link \
  --branch-override part_002_solid_002.stl=12 \
  --branch-override part_058_solid_058.stl=12 \
  --branch-override part_050_solid_050.stl=20 \
  --branch-override part_046_solid_046.stl=12 \
  --branch-override left_link2=12 \
  --branch-override left_link5=12 \
  --branch-override left_link7=12 \
  --branch-override right_link2=20 \
  --branch-override right_link5=12 \
  --branch-override right_link7=12
```

`base_link` is invariant in this 15D experiment. Its FCL validity is checked
before either timed solve; excluding it prevents a coarse fixed-base sphere
from overlapping the floor while leaving every motion-dependent collision in
both planners.

The sphere URDF preserves the same kinematic tree and limits but is an
approximation, not a replacement robot model. Record its label disagreement
against the unchanged collision meshes before making a performance claim:

```bash
/home/tim/miniconda3/envs/alfa-vamp-cricket/bin/python \
  tools/v322_vamp_paper_repro/audit_sphere_model.py \
  results/v322_vamp_paper_repro/model/alfa-v322-15d-mesh.urdf \
  results/v322_vamp_paper_repro/model/alfa-v322-15d-foam-tuned.urdf \
  results/v322_vamp_paper_repro/model/alfa-v322-15d.srdf \
  results/v322_vamp_paper_repro/model/sphere-audit-10000.json \
  --samples 10000
```

A native VAMP path is accepted only after an independent dense FCL audit. A
nonzero sphere false-negative rate must be disclosed and means the SIMD search
cannot be treated as an exact substitute for the mesh model.

If the official Foam mesh-repair binaries are unavailable, the deterministic
local fallback can generate a conservative surface-covering sphere model by
recursively partitioning triangle faces and bounding every partition:

```bash
PYTHONPATH=/home/tim/.cache/alfa-vamp/foam \
  /home/tim/.cache/alfa-vamp/venv/bin/python \
  tools/v322_vamp_paper_repro/spherize_model.py
```

This fallback is not accepted for performance claims until state and motion
labels have zero false negatives against the certified MoveIt/FCL model.

## SIMD kernel generation

```bash
/home/tim/miniconda3/bin/conda run -n alfa-vamp-cricket \
  /home/tim/.cache/alfa-vamp/cricket/build-alfa/fkcc_gen \
  tools/v322_vamp_paper_repro/cricket-alfa-v322.json
```

For the full 5x5 sphere-world experiment, generate both attachment kernels from
the repository root:

```bash
/home/tim/miniconda3/bin/conda run -n alfa-vamp-cricket \
  /home/tim/.cache/alfa-vamp/cricket/build-alfa/fkcc_gen \
  tools/v322_vamp_paper_repro/cricket-alfa-v322-sphere-world-left.json
/home/tim/miniconda3/bin/conda run -n alfa-vamp-cricket \
  /home/tim/.cache/alfa-vamp/cricket/build-alfa/fkcc_gen \
  tools/v322_vamp_paper_repro/cricket-alfa-v322-right.json
```

Build the benchmark in Release mode with the generated kernel directory:

```bash
source /opt/ros/humble/setup.bash
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash
export CMAKE_PREFIX_PATH="/home/tim/.cache/alfa-vamp/prefix:$CMAKE_PREFIX_PATH"
colcon --log-base log-paper build --packages-select alfa_robot_moveit_config \
  --build-base build-paper --install-base install-paper \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DALFA_VAMP_PAPER_KERNEL_DIR="$PWD/results/v322_vamp_paper_repro/model" \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3
```

Run the paired 5x5 trials only after sourcing `install-paper/setup.bash`:

```bash
python3 tools/v322_vamp_paper_repro/run_paired_benchmark.py \
  results/v322_vamp_paper_repro/paper-paired-5x5
```

Run the shared-IK full-task sphere-world benchmark with an explicit endpoint
directory. Only the named stage endpoints are read; historical path samples are
not used:

```bash
python3 tools/v322_vamp_paper_repro/run_shared_ik_paired.py \
  results/v322_vamp_paper_repro/sphere-world-release-final-5x5 \
  --shared-ik-dir /path/to/pure-rrt/final_payloads \
  --boxes 1-25 --rounds 3 --rrt-seconds 30 --search-retries 3 \
  --vamp-workers 1 --vamp-max-iterations 3000000 \
  --sphere-world --collision-inset 0.03
python3 tools/v322_vamp_paper_repro/summarize_shared_ik_paired.py \
  results/v322_vamp_paper_repro/sphere-world-release-final-5x5
```

The final recorded experiment completed 75/75 tasks. Aggregate RRT search time
was 19472.668 ms for OMPL/FCL and 861.470 ms for VAMP/SIMD (22.604x). This is
an experimental planning geometry result, not a production safety claim: both
backends use 30 mm contracted box geometry, the chassis is represented only by
a lower 0-0.30 m cuboid, and the native VAMP path is not accepted by an online
FCL gate. Production removal of FCL still requires zero-false-negative state
and continuous-edge audits against the original-size meshes.

## Rerun and browser report

The final timing JSON is kept unchanged. A separate representative capture run
persists the actual online RRT-Connect path states under the same planner
contract. The two Rerun files animate only those returned states, show the
complete 5x5 scene, per-box median search times, paired speedup, and backend
success/FCL status. Failed searches hold the common start state; no trajectory
is synthesized.

```bash
source /opt/ros/humble/setup.bash
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash
source install-paper/setup.bash
/usr/bin/python3 tools/v322_vamp_paper_repro/capture_replay_paths.py \
  results/v322_vamp_paper_repro/paper-paired-5x5-final/trials.csv \
  results/v322_vamp_paper_repro/replay-capture-final
```

```bash
source /opt/ros/humble/setup.bash
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash
source install-paper/setup.bash
/usr/bin/python3 tools/v322_vamp_paper_repro/create_paper_reruns.py \
  results/v322_vamp_paper_repro/paper-paired-5x5-final \
  results/v322_vamp_paper_repro/model/alfa-v322-15d-mesh.urdf \
  results/v322_vamp_paper_repro/replay-capture-final \
  results/v322_vamp_paper_repro/paper-paired-5x5-final/rerun
```

All Rerun-visible text is ASCII English because the current Rerun viewer does
not render Chinese glyphs. Build the self-contained Chinese browser report with:

```bash
/usr/bin/python3 tools/v322_vamp_paper_repro/build_paper_web_report.py \
  results/v322_vamp_paper_repro/paper-paired-5x5-final \
  results/v322_vamp_paper_repro/paper-paired-5x5-final/rerun \
  results/v322_vamp_paper_repro/paper-paired-5x5-final/index.html
```

## Adaptive full-task comparison

The adaptive experiment keeps the paper metric but uses one shared online IK
result for every matched stage. Both backends receive the exact same 15D
start/goal for precontact, contact, loaded retreat, loaded transport, and the
released return to Home. IK is never included in either search time.

This complete-task comparison uses exact MoveIt/FCL validity in both OMPL and
VAMP so every displayed path carries the same collision semantics. It is a
functional full-flow comparison, not the native-SIMD paper metric above. The
native sphere kernel currently has nonzero FCL disagreement and therefore
cannot honestly certify all 25 loaded manipulation chains.

The full-task experiment compares the integrated backends: standard OMPL
uniform sampling and range `0.12` versus VAMP's goal-corridor sampler and range
`0.08`. Start/goal IK and exact FCL semantics are shared, but the sampling
sequences are intentionally backend-native. Therefore this industrial metric
must not be presented as the paper's matched-sampler SIMD metric.

The shared task policy may try both base stations (`-0.60m`, `-0.35m`), both
arms, and front/top suction. Every failed search, retry, and policy candidate
is retained. No historical path, shortcut, Cartesian planner, or fallback is
allowed.

```bash
source /opt/ros/humble/setup.bash
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash
source install-paper/setup.bash
/usr/bin/python3 tools/v322_vamp_paper_repro/run_adaptive_multistage.py \
  results/v322_vamp_paper_repro/adaptive-multistage-5x5-final \
  --boxes 1-25 --rounds 3 --rrt-seconds 3.0 --search-retries 3 \
  --vamp-max-iterations 1000000 --vamp-workers 1 \
  --candidate-limit 1 --trial-timeout 300 --max-configs 8
```

```bash
/usr/bin/python3 tools/v322_vamp_paper_repro/summarize_adaptive_multistage.py \
  results/v322_vamp_paper_repro/adaptive-multistage-5x5-final
```

The summary reports both the final successful RRT attempt and cumulative RRT
time from the first candidate until the first complete five-stage solution.
Create the two continuous 5x5 Reruns with:

```bash
/usr/bin/python3 tools/v322_vamp_paper_repro/create_adaptive_reruns.py \
  results/v322_vamp_paper_repro/adaptive-multistage-5x5-final \
  results/v322_vamp_paper_repro/model/alfa-v322-15d-mesh.urdf \
  results/v322_vamp_paper_repro/adaptive-multistage-5x5-final/rerun
```
