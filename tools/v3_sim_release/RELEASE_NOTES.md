# V3 Motion Stage Simulation Preview 0.2.0

## 内容

- `/motion/execute_stage` 四阶段 Action：`PREGRASP -> APPROACH -> PLACE -> HOME`。
- replay 与完整 17 轴 FollowJointTrajectory 后端。
- FJT 模式强制检查完整 26 轴新鲜状态、`/control/safety_state`、轨迹起点和执行末态。
- 固定 5x5 箱墙前 11 轮侧吸轨迹缓存：11/11 成功，包含3轮单臂 `NO_MOVE`。
- 每轮依次执行候选预抓取、5cm接近、35cm抽离、携箱回初始、固定放置、释放和空载回初始。
- 第一/第二初始姿态按缓存自动切换；放置确认后箱子从 Motion 地图删除。
- Rerun 回放、缓存生成器、合同测试和仿真部门交接脚本。

## 验证

- 最新 V3 suction description。
- 13 个相关 ROS 包 Release 构建通过。
- Action replay 连续收到 11 轮请求，44个阶段Action全部成功。
- 完整序列18,341帧，首尾均为第一初始姿态；缓存旋转轴单帧不超过0.5度，Updown不超过2.86mm。
- `v3-motion-stage-11-rounds.rrd` 通过 `rerun rrd verify`。

## 限制

- 这是前11轮侧吸任务Preview，不是25箱全部成功版本。
- 外部仿真 FJT 尚未验收；仿真必须先升级到 26 轴 `/joint_states`、17 轴 FJT 和 SafetyState 合同。
- Motion 不控制吸盘，阶段之间的吸取/释放由上层编排。

## 使用

解压后执行：

```bash
cd alfa_robot
./tools/v3_sim_release/build.sh
./tools/v3_sim_release/run_server.sh replay
```

另开终端：

```bash
cd alfa_robot
./tools/v3_sim_release/run_wall_test.sh
```

仿真方完成 26/17 接口升级后，将服务端命令中的 `replay` 改为 `fjt`。
