# V3 Motion Stage 仿真联调包

本包固定使用当前 V3 吸盘 description：完整状态 26 轴，Motion FJT 17 轴。
旧仿真 25 轴反馈、14 轴 FJT 加独立 Updown 的接口不能直接使用。

## 构建

```bash
cd alfa_robot
./tools/v3_sim_release/build.sh
```

## 本地 Replay 验证

终端一：

```bash
cd alfa_robot
./tools/v3_sim_release/run_server.sh replay
```

终端二：

```bash
cd alfa_robot
./tools/v3_sim_release/run_wall_test.sh
```

当前固定墙缓存完整通过前 11 轮侧吸任务，包含 8 轮双臂与 3 轮单臂
`NO_MOVE`。测试客户端默认只发送这 11 轮。
放置确认后箱子从 Motion 地图删除。成功轮的阶段/任务边界完全连续，缓存内部旋转轴
单帧变化不超过 0.5 度，Updown 单帧变化不超过 5 mm。
默认将回放写入 `/tmp/v3_motion_stage_wall.rrd`，可使用
`rerun /tmp/v3_motion_stage_wall.rrd` 打开；`run_server.sh` 的第二个参数可覆盖路径。

## 接仿真执行器

仿真方先按 `docs/运控/IK/V3吸盘机仿真接口交付合同.md` 提供 26 轴
`/joint_states` 和完整 17 轴 `/whole_body_jtc/follow_joint_trajectory`，再运行：

```bash
cd alfa_robot
./tools/v3_sim_release/run_server.sh fjt
```

`PREGRASP -> APPROACH -> PLACE -> HOME` 之间的吸取和释放由上层调用仿真吸盘接口。
本程序不会自动控制真空。缺轴、状态过期、轨迹起点不一致、Action 不可用或末态不收敛时
均拒绝继续执行。

本 Release 已完成 Replay Action 验收，未声称外部仿真 FJT 已完成验收。仿真方接入时先运行
`replay` 核对 11 轮请求与阶段边界，再切换 `fjt` 验证真实 26 轴反馈、SafetyState 和17轴执行。
