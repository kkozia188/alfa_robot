# 测试杆同点多朝向打表：开发暂停

## 更正：底盘重复检测已定位并修复

用户指出权威底盘早已使用cuboid。ProbeCollision直接加载fixed.yaml后遗漏了正式BallFullCyclePlanner中的replace_arm_base_pairs，导致保留旧base_link球与手臂自碰撞，同时又启用BaseCuboidLayer。已复用正式替代逻辑：仅由cuboid层负责手臂与底盘，其他自碰撞不变。重新测零位/Home：self cost均0、重叠球对均0、完整GpuValidity均True。因此下文原始阻塞不是实体碰撞证据，现已解除，GUI/执行尚未完成。修正证据/tmp/probe_collision_pairs_corrected.json，旧球诊断保留/tmp/probe_collision_pairs_legacy_base_spheres.json；8161已更新显示修正后结果。

分支：feature/probe-analytic-touch。用户确认固定Updown=-0.3m；TCP相对left_link7为[0,0,0.070]m，姿态不偏转；拖拽位置/朝向，生成原朝向及同一位置局部XYZ轴±25°候选，枚举解析冗余角和分支，供用户选择。之后从实时左7轴起点生成关节插值轨迹，先预览、后明确点击发送既定FJT。不是末端笛卡尔直线运动。两个功能分别是IK候选和插值运动。

已完成：probe_touch_ik.py固定升降/临时TCP转换至原tool0解析器，3度冗余角采样及当前冗余角、去重、限位、URDF FK复核；nearby_targets保持TCP位置、朝向偏转25度。probe_touch_motion.py五次时间律生成10Hz轨迹，精确核验RT插值速度≤15度/s（默认10）、加速度≤30度/s²、jerk≤150度/s³，最长60秒；无关节±2π偷换、无自动RRT绕行。

新碰撞门禁集成测试未通过，遵循用户遇非预期问题先停的要求暂停，尚未创建可执行前端、未部署新IPC接口、未获取控制权/使能/发送动作。原8155/8156/8160保持各自职责。

失败证据：/tmp/probe_touch_collision_blocker.json。独立初始化当前fixed.yaml碰撞检查器，固定右臂为配置Home参考、升降-.3，左臂零位及Home的self collision cost分别为0.0028553/0.0073689；world、bounds、base成本均0。该成本不是穿透深度，不能直接按米解释。尚未定位具体球对或证明实体碰撞，不能说机器人真实Home不可用，也不能绕过门禁。ground_min包含被豁免的base_link球，不据此判穿地。

后续先核对碰撞球对/当前球模型与权威配置、参考右臂状态、检查器自碰撞语义，待结论明确再接GUI和发送门禁。外部接触目标/工作台及实测右臂状态尚未建模，不能声称本轮门禁覆盖真实完整环境。保持原URDF和旧吸盘tool0未改，仅临时TCP转换。
