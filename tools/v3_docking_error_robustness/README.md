# MOTION-274 V3 Docking-Error Planning Benchmark

This directory evaluates how planar docking error changes the V3.2.2 dual-arm
wall-pick planning result. It is isolated from the MOTION-261 release and never
loads the certified trajectory cache as a planning result.

## Measurement contract

- Model: `robot_v3.2.2-suction`, local Tool0 `+Z 0.151 m`.
- Task order and arm assignment: the 15 MOTION-261 cycles, including top suction
  only for boxes 22, 24, and 23.
- Nominal upper/lower base poses: `[-0.60, 0, 0]` and `[-0.35, 0, 0]` in `map`.
- Wall near face: fixed at map `X=0.75 m`; errors move only the robot TF.
- Every sample publishes a fresh `map -> odom -> base_footprint` transform and
  obtains the current full 6D target poses from the planner's `base_link`
  catalog before sending PREGRASP.
- The trajectory cache is explicitly empty. Cache loading is not planning time.
- Released boxes leave the scene after the external handoff, matching the
  MOTION-261 conveyor contract; the runtime default still retains placed boxes.
- The evaluated return is an empty-arm return after the certified conveyor
  handoff boundary. Base navigation and the conveyor motion remain out of scope.
- Planning phases are reported as IK, PREGRASP, EXTRACT, and RETURN. Action
  execution stages are also retained as PREGRASP, APPROACH, PLACE, and HOME.
- A fixed OMPL seed is used so before/after comparisons share one deterministic
  experiment design.

The initial full matrix is a provisional engineering envelope of `x/y = +/-5
cm` at `2.5 cm` spacing and `yaw = +/-3 deg` at `1.5 deg` spacing. It is an
experiment configuration, not a localization or navigation requirement.

## Build

The external public interface package is supplied by the already validated
MOTION-261 underlay. Build this branch into a separate overlay:

```bash
cd /home/tim/alfa_robot-motion274-docking-error && \
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH PKG_CONFIG_PATH PYTHONPATH && \
export PATH=/home/tim/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin && \
source /opt/ros/humble/setup.bash && \
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash && \
colcon --log-base log-motion274 build \
  --base-paths ros2_ws/src \
  --packages-select alfa_robot_analytic_ik alfa_robot_moveit_config alfa_robot_rerun \
  --build-base build-motion274 \
  --install-base install-motion274 \
  --symlink-install \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DPython3_EXECUTABLE=/usr/bin/python3
```

## Smoke run

Use an unused ROS domain so another simulation cannot publish a competing base
TF. The command writes only under this branch's ignored `data/` directory:

```bash
cd /home/tim/alfa_robot-motion274-docking-error && \
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH PKG_CONFIG_PATH PYTHONPATH && \
export PATH=/home/tim/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin && \
source /opt/ros/humble/setup.bash && \
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash && \
source /home/tim/alfa_robot-motion274-docking-error/install-motion274/setup.bash && \
export ROS_DOMAIN_ID=174 && \
/usr/bin/python3 tools/v3_docking_error_robustness/evaluate_docking_error.py \
  --config tools/v3_docking_error_robustness/config/smoke-baseline.json \
  --output-dir data/ik_benchmark/v3_docking_error/motion-274-smoke-baseline
```

The output contains one raw JSON per error sample, `cycle-results.csv`,
`summary.json`, `REPORT.md`, planner logs, and the configured Rerun recordings.
Rerun files are closed through SIGINT before their planner process exits.

## Full baseline and comparison

```bash
cd /home/tim/alfa_robot-motion274-docking-error && \
source /opt/ros/humble/setup.bash && \
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash && \
source /home/tim/alfa_robot-motion274-docking-error/install-motion274/setup.bash && \
export ROS_DOMAIN_ID=174 && \
/usr/bin/python3 tools/v3_docking_error_robustness/evaluate_docking_error.py \
  --config tools/v3_docking_error_robustness/config/full-baseline.json \
  --output-dir data/ik_benchmark/v3_docking_error/motion-274-full-baseline \
  --resume
```

An optimized config must keep the matrix, cycles, model, poses, and baseline
commit unchanged. Compare the reports with:

```bash
cd /home/tim/alfa_robot-motion274-docking-error && \
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH PKG_CONFIG_PATH PYTHONPATH && \
export PATH=/home/tim/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin && \
/usr/bin/python3 tools/v3_docking_error_robustness/compare_reports.py \
  --before data/ik_benchmark/v3_docking_error/motion-274-full-baseline/summary.json \
  --after data/ik_benchmark/v3_docking_error/motion-274-full-optimized/summary.json \
  --output-dir data/ik_benchmark/v3_docking_error/motion-274-comparison
```

## Scope boundary

This benchmark measures motion planning robustness. It does not evaluate or
control navigation, localization, chassis execution, suction hardware, or a
physical conveyor.
