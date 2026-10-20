# 随机误差规划Demo：8171

2026-10-20，分支feature/random-error-planning-demo。统一入口http://127.0.0.1:8171/；不替换8152或测试杆8162，不连接实机。

## 后续诊断更正：目标分配与外观显示

初版runner错误硬编码世界箱心left Y=+.495/right Y=-.495、X=-2.3，把转身前后左右关系及输送机中心/放箱点混用。原ConveyorCyclePlanner.conveyor_reference_targets实际为left Y=-.425/right Y=+.425、X=-1.45（-2.3+.85），高度1.082。已删除重复常量，runner和显示共同复用原定义，并核验两臂能共享同一停车位，否则拒绝而不平均掩盖差异。

独立CPU诊断：错误目标在导航后footprint中left工具Y=-.495、right工具Y=+.495，肩到腕最短采样距离约1.061m，超过.98m臂长；21高度×5度ψ采样均0解。仅恢复原流程的横向目标后两臂各6816个限位内解析解（统计未去重、不含碰撞，也不是全周期成功率）。证据/tmp/random_error_place_target_diagnostic_20261020.json和/tmp/random_error_place_correct_reference_20261020.json。新回归test_random_error_placement.py用真实canonical方法证明转身后局部left Y=+.425/right Y=-.425，且保持.8m前伸、同一底盘位；另验不一致停车需求拒绝，2项通过。

双龙头车不见的原因是ReadOnlyScene.reference_conveyors返回的根节点默认visible=false，原UI忽略返回值。已保留根节点、默认显示、加“显示双龙头车外观”，光圈按相同canonical目标更新；这是修显示，不是把外观当作碰撞几何。

复测后缀出现独立阻塞：共享fixed.yaml新增顶层coordinate_revision元数据，当前cuRobo RobotCfg.__init__不接受，检查器初始化报TypeError，未进入修正目标后的IK。按用户遇非预期问题先停要求，未剥离/覆盖其他工程师模型元数据，尚不宣称完整动作成功。同期左J5零位+180及Home更新也未覆盖。

当前8171服务为alfa-random-error-view-8171，默认载入/tmp/random_error_failure_rebased_view_20261020.json：这是原失败现场，left J5按共享新零位减pi转换以保持同一物理姿态，原JSON保留，页面明确“历史错误目标现场／复测阻塞”。目标修正后的新轨迹尚未获得。原alfa-random-error-planning-8171已停止，不重复占端口。HTTP200/服务运行已验证，未绕过浏览器限制、未实机发送。

## 使用

页面使用当前重零模型及当前Home。选择原119任务之一或重新随机，生成停车半径5cm/Yaw±5度、箱体Roll/Pitch±2度/Yaw±5度；数值显示在页面。点击“规划当前随机场景”，依次尝试接触、world -X抽离35cm、收束运输、模拟导航到位、外部放箱、释放及Home。成功或失败都保留返回轨迹，可播放、拖帧、跳转阶段。切换输入会清除旧轨迹；规划时禁用选择，防止新场景混用旧结果。

目前默认载入一次真实计算的失败样本，不伪称完整成功：L24/R21，抓取/抽离/收束成功，模拟导航后“龙头车放置目标IK”失败，returned64、solver_success0、collision_valid0。842帧，最后阶段base_reposition；没有生成放箱/释放/Home，不会补画。失败不等于实体机械臂物理不可达，尚未进一步定位或改目标/放宽约束。初始规划约3.680秒、放箱尝试约1.703秒，前者日志未包含后缀失败时间，不能把3.680秒当全流程总耗时。

底层箱绕几何中心微倾可能穿地的问题仍存在，未得到用户确认前不自动抬箱或关闭地面碰撞。该Demo仍是实验接入，不宣称全119任务或随机误差成功率已验收。

## 代码与证据

- v3_tcp_scene_preview.py：场景输入、统一UI、异步规划、当前模型、载荷随动及失败前缀播放。
- v3_tcp_rezero_pipeline.py：恢复feature/tcp-motion-pipeline中的TcpMotionPlanner请求/阶段逻辑，以RezeroTaskPlanner为基础接当前解析映射和已转换候选库。接触工具面内roll/逆附着和场景恢复一致；不把测试杆70mm TCP用于搬箱，此处仍是原吸盘tool0。
- v3_random_error_runner.py：绑定TCP请求、保存输入、运行抓到收束、模拟导航及外部放箱/释放确认；失败立即返回现场，不继续伪造后续阶段。
- test_tcp_rezero_contract.py：60组侧/顶吸扰动下roll变换、目标箱恢复、世界抽离方向等价。连同原合同共3测试通过。
- /tmp/random_error_rezero_initial_20261020/：input.json、pick.json、placement.json、result.json。新请求保存research/curobo_v3/generated/random_error_demo_runs/<时间戳>/。临时证据不是长期归档。
- /tmp/random_error_ui_smoke_20261020.log：真实ReadOnlyScene接口119任务各两次、共239次更新通过。另修复轨迹仅18轴而显示还需头部固定轴的适配；未改规划自由度。

systemd用户服务alfa-random-error-planning-8171；原纯场景alfa-random-error-demo-8171已停。HTTP200和服务运行已验证；本轮浏览器自动访问受既有策略限制，未声称按钮已通过浏览器实点验收。现场用户可刷新测试。

启动（服务已在运行，不要重复占端口）：

```bash
cd /mnt/mydisk/ALFA/alfa_robot_v3 && \
/mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python \
  research/curobo_v3/tools/v3_tcp_scene_preview.py \
  --port 8171 \
  --result /tmp/random_error_rezero_initial_20261020/result.json
```

若临时result丢失，去掉--result即可生成新场景，需用户点击规划，不会套用旧录制。
