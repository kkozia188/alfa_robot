# MOTION-233 cuRobo RRT种子轨迹优化与时间化

## 当前合同

在PR #53基线`bb4ac23`上显式启用`--trajopt-rrt`后：

1. `approach`、`transport`、`return_home`将现有15维RRT作为固定cuRobo main `78fd485`的`TrajectoryOptimizer.solve_cspace()` seed。
2. cuRobo输出`q/qdot/qddot/jerk/dt`；结果仍须通过项目`GpuValidity`节点和0.5°边门禁，不通过时回退原RRT几何路径。
3. 回退RRT使用逐路点停稳的五次时间律，严格保留已验碰的关节空间直边。
4. 解析`extract`与loaded/empty yaw保持原几何节点，以C2样条和五次时间律时间化，并对样条密采样重新验碰；extract另复核笛卡尔走廊。
5. 输出包含完整周期`time_from_start_s/velocities/accelerations/jerks/motion_duration_s`，home/attach/release为零时长静止事件。
6. Viser按真实时间回放，显示每关节`q/qdot`、选中关节曲线和按速度上限归一化的关节轴矢量。

这仍是冻结代理模型下的研究产品：`load_dynamics=false`，不加载箱体质量/COM/惯量，不运行RNEA或torque约束，不可直接下发实机。

## 完整0.75m墙规划

以下命令可在新终端完整复制。`prepare_checkout.py`会把本工作树中的模型绝对路径重定位到当前目录并编译解析IK桥；因此应在专用工作树执行。

```bash
cd /home/astesia/Sevenova/.motion-233-curobo-trajopt/research/curobo_v3 && \
/home/astesia/Sevenova/curobo_v2_ros/bin/python tools/prepare_checkout.py --compiler g++ && \
PYTHONPATH=/home/astesia/Sevenova/golden_curobo_adapter/mentor-repro-20261006/curobo-main \
PYTORCH_ALLOC_CONF=expandable_segments:True \
/home/astesia/Sevenova/curobo_v2_ros/bin/python tools/v3_plan_cycle.py \
  --sequence \
  --wall-distance 0.75 \
  --trajopt-rrt \
  --trajopt-interpolation-dt 0.025 \
  --output /tmp/curobo-trajopt-full-wall-075.json
```

## 离线产品门禁

```bash
cd /home/astesia/Sevenova/.motion-233-curobo-trajopt && \
/home/astesia/Sevenova/curobo_v2_ros/bin/python tools/verify_motion233_timed_trajectory.py \
  --result /tmp/curobo-trajopt-full-wall-075.json \
  --mobile-robot-config /home/astesia/Sevenova/.motion-233-curobo-trajopt/research/curobo_v3/generated/v3_analytic_071cb95/mobile18/alfa_v322_suction_mobile18.yml \
  --output /tmp/curobo-trajopt-full-wall-summary.json
```

门禁检查：规划/全部轮次成功、关节顺序、时间与导数数组对齐、数值有限、时间不倒退、速度/加速度/jerk不超过冻结配置、home/attach/release静止、各轮`timed_validation`通过。

## Viser只读回放

```bash
cd /home/astesia/Sevenova/.motion-233-curobo-trajopt/research/curobo_v3 && \
PYTHONPATH=/home/astesia/Sevenova/golden_curobo_adapter/mentor-repro-20261006/curobo-main \
/home/astesia/Sevenova/curobo_v2_ros/bin/python tools/v3_full_cycle_demo.py \
  --port 8090 \
  --sequence-json /tmp/curobo-trajopt-full-wall-075.json \
  --replay-only \
  --wall-distance 0.75
```

浏览器打开`http://127.0.0.1:8090`。页面中的时间、速度和曲线来自规划结果，不由回放帧差推测。

## 当前完整墙证据

2026-10-10当前严格源码已合入上游`alfa_v3_curobo@8e396a3`的距离合同，并在0.75m墙完成15/15轮、25/25箱：39552个真实时间采样帧，规划164.428s，名义动作997.665s。TrajOpt累计13.084s；warm中位数为approach 94.79ms、transport 359.98ms、return-home 95.54ms。15轮中分别采纳7/12/11段，未采纳段保持原已验碰直边，并以25ms或更细的五次时间律输出每帧真实`q/qdot/qddot/jerk`。速度、加速度、jerk冻结限制最大比例为0.952、0.996、0.868。结果SHA256 `4b3e7cba0301ab2dcc34dee898e72426de4457ff6522a8937448528bfe1b099d`，完整索引见`generated/trajopt_validation/summary.json`。
