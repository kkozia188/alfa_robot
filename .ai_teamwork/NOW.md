# 当前项目状态

## 一句话概况

ALFA Robot 是 ROS2 双臂工业机器人项目；当前仓库只保留运控、电控、机器人模型及其必要运行适配，历史感知、导航和仿真实现已退出主线。

## 开工入口

- 先读 `.ai_teamwork/START.md`。
- 再读本文件和 `.ai_teamwork/TASKS.md`。
- `.ai_teamwork/archive/` 默认不读；只有追溯旧方案、验收证据、责任边界时再查。

## 当前推进重点

- 当前主线：`v5_dev` 已收口左右箱体正面中心 6D 位姿任务合同；功能分支正在接入 V3 七轴双臂模型。
- V3 模型工作跟踪：Linear `MOTION-94`。本地描述已同步
  `kkozia188/alfa_robot:alfa_v3_dev@eb898a9`（PR #31）的对称零位、解析 IK
  固定变换及 `home/unloading` 命名姿态；description 快照锁定在
  `robot_v3.1.1-hybrid@510694697e`，受管模型资产由
  `tools/sync_v311_description.py --check-local` 校验。
- V3 基础动作跟踪：Linear `MOTION-154`。分支 `motion-154-v3-dual-arm-simple-motion` 已完成首个40cm箱双臂同步解析笛卡尔平移Demo，待用户交互验收后继续旋转和异构握持任务。
- 正式任务输入只包含 `request_id`、左右正面中心 `pose_6d` 和 `execute`；算法内部按高度容差识别排数、吸附方式和抽离策略，禁止从箱号或外部吸附模式获取帮助。
- 旧 `/robot_motion/run_box_pair_task` 仅保留为显式兼容入口，默认完整栈不启动 `box_pair_task_adapter_node`。
- 当前任务表只保留未完成/需确认事项：T-0030/T-0031/T-0032/T-0037。
- 已完成/已同步 Linear 的长过程已归档到 `.ai_teamwork/archive/2026-05-18_v5_dev_collaboration_cleanup/`。
- PM 必须持续把完成任务移出当前表，避免后续 AI 误认为仍需处理。
- 架构目标、包职责和调试代码准入规则见 `docs/运控/系统架构与包职责边界.md`。
- 可视化架构入口见 `docs/system_portal/architecture.html`。

## 当前仍需注意

- V3.1.1 吸盘整机现有26个可动关节，包含双臂14轴、升降、双轴头部、主动悬挂及脚轮/车轮；升降范围仍是 `[-1,0]m`。单臂搬箱回放仍输出原16个规划相关关节，其余轴保持模型默认值；历史执行桥的16轴合同尚未迁移到实机 V3.1.1，不可直接用于硬件执行。
- V3.1.1 `home`：`updown=-0.3m`，左臂 `[155,-105,20,90,-90,-40,0]°`、右臂 `[-155,-105,-20,90,90,40,0]°`；模型查看器、mock ros2_control、MoveIt 初始状态和 SRDF 应保持一致。
- V3.1.1 `unloading`：左臂 `[130,-105,-180,20,-90,-30,0]°`、右臂 `[-130,-105,180,20,90,30,0]°`，同样 `updown=-0.3m`。它的末端工具轴仍近乎水平，不满足旧版本“末端竖直朝下”的要求；当前仿真按 PR #31 权威固定姿态释放。
- Linear/Git 关联提交标题优先使用 `Refs MOTION-xx: ...`；只写 `MOTION-xx:` 不稳定。
- 一个 issue 只对创建时的验收目标负责；后续探索/测试应拆新 issue 或放 Backlog，不要让已达标 issue 永远开着。
- `alfa_robot_moveit_config` 已不再编译或包含 `scripts/ik_benchmark/` 的头文件；公共 IK 候选类型已迁入 `robot_motion_core`，Rerun 公共实现已迁入 `alfa_robot_rerun`。
- `dual_arm_planner_node` 仍承载完整候选排序、抽离和负重规划适配；这些实现尚未全部迁入独立 core/planning service。
- 历史 `bio_ik`、仓库内 `alfa_robot_hardware` 和旧 `alfa_robot_bringup` 已退出主线；实机硬件与生命周期由外部 `rt-control` 域负责。
- V3.1.1 模型、解析IK、MoveIt FK 与 `home/unloading` 自碰撞测试已通过；
  独立铲式 X=0.75m 5×5 任务完成25箱真实抓取、25次整机过渡与Rerun录制。
  当前序列每箱释放后直接规划到下一箱，不再逐箱回`home`；携箱倾角限制95°，
  已验证携箱帧无上下倒置。每排第1/2列固定左手，第3/4/5列固定右手；
  箱间过渡按单臂分段执行，不让双臂同时参与搜索。
  速度版把RRT捷径图限制为64个代表节点、单次笛卡尔段搜索限制为128次；
  正面长距离转运保留连续 swivel 解析IK引导，顶吸带箱转运改用更稳定的关节路径。
  抓取接触滚转按
  左`+90°`/右`-90°`镜像，接触法向与吸盘水平误差均小于`4e-6°`。
  完整25箱冷启动约350.67s，计划缓存后重建RRD约14.32s；默认时间轴135.25s，
  较上一版V3.1.1缩短38.28%。速度版产物位于
  `data/ik_benchmark/v3_scoop_5x5/2026-09-18-speed-optimized/`。
  9月19日新增 shortcut 局部关节RRT：先定位7轴shortcut碰撞窗，只在1轴、必要时
  2轴偏移上搜索，其余轴保持主体路径；完整25箱25/25，选中轨迹核心规划平均
  2.60s、中位1.31s，22/25不超过5s。9号带箱只动右J4（171.76ms），8号过渡
  只动右J1（7.68ms）；大步关节翻转0、携箱倒置0。结果和可视报告位于
  `data/ik_benchmark/v3_scoop_5x5/2026-09-19-joint-shortcut-rrt/`。
  9月21日修复首箱左J1从`+155°`绕到`-151°`的问题：有限位关节候选评分不再把
  跨`±180°`当作可执行近路，新分支为`+155°→+167.1°`；首段78帧/755.45°，
  首箱完整换向4次。完整25箱结果位于
  `data/ik_benchmark/v3_scoop_5x5/2026-09-21-first-grasp-direct/`。
  同日进一步压缩4号和6号低位放箱路径：4号完整动作`1968.66°→918.23°`，
  6号`1459.63°→728.03°`；两箱选中任务核心规划分别`4.90s/4.23s`，完整
  序列总行程`42689.83°`。新默认结果位于
  `data/ik_benchmark/v3_scoop_5x5/2026-09-21-box4-box6-low-transfer/`。
  随后7号改为左臂正吸（唯一完整可达高度`updown=0.00m`），核心规划
  `10.15s→1.56s`；20号固定工位使用逐边FCL复核的关节路标热启动，核心规划
  `9.42s→0.72s`。当前25箱选中核心规划全部不超过5秒，平均1.69s、最大4.90s，
  总行程`41476.71°`。新默认结果位于
  `data/ik_benchmark/v3_scoop_5x5/2026-09-21-box7-front-box20-fast/`。
  用户已认可该版本，黄金归档为
  `data/ik_benchmark/v3_scoop_5x5/baselines/2026-09-21-golden-box7-front-box20-fast/`，
  包含完整产物、源码快照和独立SHA-256校验；后续实验不得覆盖。
  PR #31 前用户认可的25/25版本保存在
  `data/ik_benchmark/v3_scoop_5x5/baselines/2026-09-18-approved-column-split/`。
  首箱从对称零位到预接触位采用无碰撞关节拓扑加TCP路径代价选优；正式首箱
  TCP路径`3.495m`、姿态累计`427.73°`，较受控旧版`3.906m/541.15°`分别下降
  10.5%/21.0%。PR #31首箱优化前结果另存
  `data/ik_benchmark/v3_scoop_5x5/baselines/2026-09-18-pr31-before-first-tcp/`。
  旧吸盘站位及历史 V3.0.9 可达性结论不自动迁移，详情见
  `docs/运控/IK/V3铲式5x5抓取仿真.md`。

## 当前主要模块速查

- 接口契约：`ros2_ws/src/robot_motion_interfaces/`
- 纯算法公共核心：`ros2_ws/src/robot_motion_core/`
- 核心运行时：`ros2_ws/src/robot_motion_runtime/`
- 场景能力：`ros2_ws/src/robot_motion_scene_service/`
- 运动算法与 MoveIt 适配：`ros2_ws/src/alfa_robot_analytic_ik/`、`ros2_ws/src/alfa_robot_moveit_config/`
- 执行适配：`ros2_ws/src/alfa_robot_execution_bridge/`；真实硬件由外部 `rt-control` 域负责
- 模型与启动编排：`ros2_ws/src/alfa_robot_description/`、`ros2_ws/src/robot_motion_runtime/launch/`
- 可视化：`ros2_ws/src/alfa_robot_rerun/`
- 实验资产：`scripts/ik_benchmark/`，仅用于验证和迁移，不应反向成为运行时职责来源。

## 常用追溯入口

- 本次归档摘要：`.ai_teamwork/archive/2026-05-18_v5_dev_collaboration_cleanup/COMPLETED_SUMMARY.md`
- 本次归档前完整日志：`.ai_teamwork/archive/2026-05-18_v5_dev_collaboration_cleanup/LOG.before_archive.md`
- 控制层硬编码：`docs/CONTROL_LAYER_HARDCODED_PARAMS.md`
- 控制架构梳理：`docs/REFACTOR_ARCHITECTURE_NOTES.md`
- IK 服务说明：`docs/运控/IK/ik_service.md`
