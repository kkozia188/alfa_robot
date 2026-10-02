# V3.2.2 5x5 Dual-Arm Yaw Robustness

This toolset extends the certified MOTION-261 task, not the legacy Stage wall
demo. Every yaw case runs the same complete task:

- `robot_v3.2.2-suction`, Tool0 local `+Z 0.151 m`;
- 25 boxes, 5 rows, and 15 top-to-bottom cycles;
- outer pair, inner pair, then center box per row;
- 10 simultaneous dual pickups with zero one-sided attachment frames;
- upper/lower pickup poses `[-0.60, 0, yaw]` and `[-0.35, 0, yaw]` in `map`;
- 2.35 m backoff, 1.50 m robot-right shuttle, release, stow, and return;
- top suction for boxes 22, 24, and 23;
- complete MoveIt/FCL frame validation plus 1 degree / 1 cm edge sampling.

The existing `/motion/execute_stage` server is unchanged. The research planner
is isolated as `v3_yaw_single_arm_box_extract_demo` and its own launch file.

## Algorithm

For every yaw, the planner consumes the current planar TF and re-solves the
world-fixed box 6D targets. A yaw-zero or adjacent certified-yaw precontact
state is only a numerical IK seed. The returned joints must still match the
current target through FK, satisfy bounds, and pass collision checks.

Candidate selection searches one collision-valid shared lift and suction mode
for each pair. All 15 whole-body entry bridges are rebuilt for the current yaw;
difficult bridges may use a validated earlier bridge as a prefix and a newly
planned local suffix. No yaw-zero trajectory is accepted as a yaw-error result.

Boundary improvements found by the sweep:

- At `-5 deg`, group `2+4` fails retreat at its old `0.00 m` lift and succeeds
  at `-0.25 m`.
- At `+5 deg`, top-suction box 23 fails at its old `-0.75 m` lift and succeeds
  at `-1.00 m`.

## Build

```bash
cd /home/tim/alfa_robot-motion274-docking-error && \
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH PKG_CONFIG_PATH PYTHONPATH && \
export PATH=/home/tim/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin && \
source /opt/ros/humble/setup.bash && \
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash && \
colcon --log-base log-motion274-yaw build \
  --base-paths ros2_ws/src \
  --packages-select alfa_robot_analytic_ik alfa_robot_moveit_config alfa_robot_rerun \
  --build-base build-motion274-yaw \
  --install-base install-motion274-yaw \
  --symlink-install \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DPython3_EXECUTABLE=/usr/bin/python3
```

## Reproduce The Sweep

The following is the exact continuation order used for the certificate. Each
chain needs its own ROS Domain.

```bash
cd /home/tim/alfa_robot-motion274-docking-error && \
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH PKG_CONFIG_PATH PYTHONPATH && \
export PATH=/home/tim/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin && \
source /opt/ros/humble/setup.bash && \
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash && \
source /home/tim/alfa_robot-motion274-docking-error/install-motion274-yaw/setup.bash && \
export ROS_DOMAIN_ID=178 && \
/usr/bin/python3 tools/v3_yaw_robustness/run_yaw_case.py \
  --yaw-deg -5 \
  --baseline-cache /home/tim/alfa_robot-alfa_v3_dev/data/ik_benchmark/v3_scoop_5x5/releases/2026-10-01-v322-tool0151-mobile-base-conveyor/v322-plan-cache.json \
  --output-root data/ik_benchmark/v3_yaw_robustness/cases
```

Run the negative chain from the certified `-5 deg` cache:

```bash
cd /home/tim/alfa_robot-motion274-docking-error && \
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH PKG_CONFIG_PATH PYTHONPATH && \
export PATH=/home/tim/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin && \
source /opt/ros/humble/setup.bash && \
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash && \
source /home/tim/alfa_robot-motion274-docking-error/install-motion274-yaw/setup.bash && \
export ROS_DOMAIN_ID=180 && \
/usr/bin/python3 tools/v3_yaw_robustness/run_yaw_sweep.py \
  --yaws=-4,-3,-2,-1,0 \
  --baseline-cache /home/tim/alfa_robot-alfa_v3_dev/data/ik_benchmark/v3_scoop_5x5/releases/2026-10-01-v322-tool0151-mobile-base-conveyor/v322-plan-cache.json \
  --initial-continuation-cache data/ik_benchmark/v3_yaw_robustness/cases/yaw-m05/v322-yaw-plan-cache.json \
  --output-root data/ik_benchmark/v3_yaw_robustness/cases
```

Run the positive boundary and continuation chain with a different ROS Domain:

```bash
cd /home/tim/alfa_robot-motion274-docking-error && \
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH PKG_CONFIG_PATH PYTHONPATH && \
export PATH=/home/tim/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin && \
source /opt/ros/humble/setup.bash && \
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash && \
source /home/tim/alfa_robot-motion274-docking-error/install-motion274-yaw/setup.bash && \
export ROS_DOMAIN_ID=179 && \
/usr/bin/python3 tools/v3_yaw_robustness/run_yaw_case.py \
  --yaw-deg 5 \
  --baseline-cache /home/tim/alfa_robot-alfa_v3_dev/data/ik_benchmark/v3_scoop_5x5/releases/2026-10-01-v322-tool0151-mobile-base-conveyor/v322-plan-cache.json \
  --output-root data/ik_benchmark/v3_yaw_robustness/cases
```

```bash
cd /home/tim/alfa_robot-motion274-docking-error && \
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH PKG_CONFIG_PATH PYTHONPATH && \
export PATH=/home/tim/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin && \
source /opt/ros/humble/setup.bash && \
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash && \
source /home/tim/alfa_robot-motion274-docking-error/install-motion274-yaw/setup.bash && \
export ROS_DOMAIN_ID=181 && \
/usr/bin/python3 tools/v3_yaw_robustness/run_yaw_sweep.py \
  --yaws=4,3,2,1 \
  --baseline-cache /home/tim/alfa_robot-alfa_v3_dev/data/ik_benchmark/v3_scoop_5x5/releases/2026-10-01-v322-tool0151-mobile-base-conveyor/v322-plan-cache.json \
  --initial-continuation-cache data/ik_benchmark/v3_yaw_robustness/cases/yaw-p05/v322-yaw-plan-cache.json \
  --output-root data/ik_benchmark/v3_yaw_robustness/cases
```

Generate the aggregate certificate with:

```bash
cd /home/tim/alfa_robot-motion274-docking-error && \
/usr/bin/python3 tools/v3_yaw_robustness/summarize_yaw_sweep.py \
  --cases-root data/ik_benchmark/v3_yaw_robustness/cases \
  --baseline-cache /home/tim/alfa_robot-alfa_v3_dev/data/ik_benchmark/v3_scoop_5x5/releases/2026-10-01-v322-tool0151-mobile-base-conveyor/v322-plan-cache.json \
  --output-dir data/ik_benchmark/v3_yaw_robustness/release
```

## Production-Shell Rerun

For normal review, start the interactive selector. It prompts for one certified
integer yaw from `-5` through `+5`, verifies the matching RRD, and opens Rerun:

```bash
cd /home/tim/alfa_robot-motion274-docking-error && \
/usr/bin/python3 tools/v3_yaw_robustness/v3_yaw_rerun_demo.py
```

An automation caller can skip the prompt with, for example,
`--yaw-deg=-3`.

To rebuild a production-shell RRD from a certified case:

```bash
cd /home/tim/alfa_robot-motion274-docking-error && \
source /opt/ros/humble/setup.bash && \
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash && \
source /home/tim/alfa_robot-motion274-docking-error/install-motion274-yaw/setup.bash && \
/usr/bin/python3 tools/v3_yaw_robustness/record_yaw_rerun.py \
  --case-dir data/ik_benchmark/v3_yaw_robustness/cases/yaw-m05 \
  --output-dir data/ik_benchmark/v3_yaw_robustness/release/rerun
```

The recorder expands the current description package and rejects non-V3.2.2
URDFs. A successful recording reports `root=base_footprint`, `links=21`, and
`meshes=49`.

## Scope Boundary

The certificate covers motion planning and collision validation for the fixed
station x/y, box wall, and conveyor replay. Navigation, localization, suction
hardware, and execution control remain outside MOTION-274.
