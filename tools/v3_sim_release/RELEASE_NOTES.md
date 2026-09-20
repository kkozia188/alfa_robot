# V3 Motion Stage Simulation Preview 0.1.0

## 内容

- `/motion/execute_stage` 四阶段 Action：`PREGRASP -> APPROACH -> PLACE -> HOME`。
- replay 与完整 17 轴 FollowJointTrajectory 后端。
- FJT 模式强制检查完整 26 轴新鲜状态、`/control/safety_state`、轨迹起点和执行末态。
- 固定 5x5 箱墙轨迹缓存：成功轮 0、1、3、4；其余轮次在运动前失败并可跳过。
- 第一轮从第一初始姿态开始，后续成功轮从第二初始姿态开始；放置确认后箱子从 Motion 地图删除。
- Rerun 回放、缓存生成器、合同测试和仿真部门交接脚本。

## 验证

- 最新 V3 suction description。
- 13 个相关 ROS 包 Release 构建通过。
- 4 个相关包共 54 项测试通过。
- Action replay 收到 15 轮请求：4 轮四阶段成功，11 轮 PREGRASP 前失败并跳过。
- 阶段/任务边界最大关节差为 0；缓存旋转轴单帧不超过 0.5 度，Updown 不超过 5 mm。
- `v3_motion222_wall_action_smooth.rrd` 通过 `rerun rrd verify`。

## 限制

- 这是 Preview，不是 25 箱全部成功版本。
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
