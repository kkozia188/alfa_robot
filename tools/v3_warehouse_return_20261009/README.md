# V3.2.2 Warehouse Return Navigation Simulation

Linear issue: `MOTION-291`

Owner: Zhang Zelin

GitHub tag: `v3-warehouse-return-navigation-sim-2026.10.09`

This delivery adds an interactive Rerun simulation for returning the folded
V3.2.2 production robot from a safe warehouse pose to the defined origin.

## Coordinate Contract

The relative origin is the pose where the vehicle-front contact plane is
`0.90 m` from the 5x5 box contact plane, `Y=0`, and `Yaw=0 deg`.

The conservative input range valid for every Yaw is:

```text
x   = [-0.530000, +0.616523] m
y   = [-0.406523, +0.406523] m
Yaw = [-180, +180] deg
```

The range uses the certified `robot_v3.2.2-suction` Home-pose collision mesh,
its `0.733476558 m` maximum XY rotation radius, and a `0.05 m` side-wall and
box-contact-plane safety clearance.

## Run

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run alfa_robot_moveit_config v3_warehouse_return_demo.py
```

The program first records and opens a READY RRD at the requested pose. It does
not move until the terminal receives `start`. The return then uses minimum-angle
in-place alignment, the geometrically shortest straight center path with
forward/reverse selection by total rotation, and a final in-place Yaw reset.

## Validation

- Exact expanded URDF SHA-256:
  `537d380a0d8a14a931bd2ca153759879eaeaec8e5f78f9cdf3f5a58c0a122893`
- Production shell: `21 links / 49 visual meshes`
- Repository V3.2.2 description assets checked against the 129-file upstream lock
- Boundary matrix: `225/225` closed-loop returns collision-free
- Maximum final position error: `0.0491 mm`
- Maximum final Yaw error: `0.00972 deg`
- Package build and dedicated pytest/CTest: passed
- READY and full-return RRD verification and 1600x900 render inspection: passed

The release assets contain READY/full RRDs, screenshots, summary,
`matrix-validation.json`, and `SHA256SUMS`.

## Scope Boundary

This is rigid-body planning and closed-loop trajectory simulation in a static
warehouse. It is not validation of Nav2, hardware IMU/LiDAR fusion, wheel slip,
actuator delay, `/cmd_vel` execution, or the physical emergency-stop chain.
