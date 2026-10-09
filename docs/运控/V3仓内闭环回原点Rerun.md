# V3.2.2 仓内闭环回原点 Rerun

## 坐标与输入范围

输入是相对原点的平面位姿 `(x, y, Yaw)`：

- 原点：车头前接触面至 5x5 箱体接触面净距 `0.90m`、`y=0m`、`Yaw=0deg`。
- `+x` 指向箱墙，`+y` 指向仓库左侧，`+Yaw` 为逆时针。
- 仓库内界：绝对 `X=[-1.18, 1.20]m`、`Y=[-1.19, 1.19]m`，开口位于 `X=-1.18m`。

使用认证的 `robot_v3.2.2-suction` Home 姿态生产碰撞网格。模型为 21 个 link、
49 个 visual mesh，碰撞网格 XY 最大旋转半径为 `0.733476558m`。按任意 Yaw
都成立的旋转圆包络并增加 `0.05m` 墙/箱面安全余量后，统一输入范围为：

```text
x = [-0.529999997, +0.616523444] m
y = [-0.406523442, +0.406523442] m
Yaw = [-180, +180] deg
```

这是保守的统一矩形。范围内的任意 `(x,y,Yaw)` 均可先原地旋转，且后续到原点
的直线中心路径始终处于同一个凸安全区。仓库开口不是碰撞墙；`x` 下界约束
`base_footprint` 原点不越过开口，转弯时外壳可以伸出开口。

## 交互演示

```bash
cd /home/tim/alfa_robot-alfa_v3_dev/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run alfa_robot_moveit_config v3_warehouse_return_demo.py
```

入口先依次读取 `x`、`y`、`Yaw`，校验生产模型和起点碰撞，然后生成并打开
`*-ready.rrd`。机器人只出现在输入位姿，不运动。终端收到 `start` 后才生成并
打开完整 `*-return.rrd`。

无人值守复现：

```bash
ros2 run alfa_robot_moveit_config v3_warehouse_return_demo.py \
  --x 0.45 --y 0.30 --yaw-deg 135 --start --no-open
```

默认产物位于 `data/ik_benchmark/v3_warehouse_return/`，包含 READY RRD、完整回放
RRD 和 JSON 摘要。

## 规划与闭环

路径由三段组成：

1. 在起点原地按最小角旋转到直线路径航向。
2. 沿起点到原点的几何最短直线闭环跟踪；比较正向和倒向方案，选择总旋转角更小者。
3. 到达原点后原地按最小角把 Yaw 归零。

各段使用五次 smoothstep 参考轨迹和速度/加速度限幅，位置及航向误差每 `0.04s`
反馈修正。限制为线速度 `0.25m/s`、角速度 `0.40rad/s`、线加速度
`0.20m/s^2`、角加速度 `0.30rad/s^2`。每个控制采样都用折叠整机生产碰撞网格
的二维凸包复核；采样步长不超过 `1cm / 0.916deg`。

## 边界

这是无动态障碍的刚体运动学和闭环轨迹 Rerun 演示。它验证生产外壳、静态仓库、
输入范围、控制限幅和几何碰撞，不代表 IMU/LiDAR 噪声、轮胎打滑、定位漂移、
执行器延迟或实机急停链路已经验收。接入实机时仍须由定位系统提供唯一
`map -> odom -> base_footprint`，导航速度经 `/cmd_vel_nav -> twist_mux -> /cmd_vel`。
