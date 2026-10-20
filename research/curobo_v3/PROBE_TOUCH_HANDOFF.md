# 测试杆同点多朝向打表与左臂实机联调交接

更新时间：2026-10-20。本文是本功能的接手入口；历史排障看 `PROBE_TOUCH_STATUS.md`，不要将其中早期“开发暂停”视为最新结论。

## 当前零位修订：仅左J5，新0度＝此前URDF 180度

2026-10-20用户确认电控已经调整，仅左臂J5需要同步。当前坐标版本 `rezero-left-j5-plus180-20261020`；分支 `feature/left-j5-zero-180`。以下历史章节中的旧Home左J5=0、旧分支名均需结合此节阅读。

- 左J5关节origin旋转额外180度，轴方向、数值限位、tool0和70mm测试TCP定义不变；右臂完全不变。实机反馈/命令已是新零位，因此直接使用，不再额外加减180度。
- 重零生成器左臂旧解析坐标offset由90度变为270度；固定/planar URDF、cuRobo配置的坐标版本、命名姿态和side/top候选库已同步。每个旧姿态的左J5换算为wrap(旧角−180度)。原物理Home保持不变，因此新的左Home是[65,-75,20,-90,-180,40,90]度；右Home仍[-65,75,-20,90,0,-40,-90]度。全零测试中的左J5=0则是新的物理零位，不是旧Home。
- 本机8156与8162已重启加载，刷新后重新对齐目标、求IK/锁定轴并计算预览。旧预览和保存的角度不能直接继续发送。8155旧吸附库保持原始数据不改写，借助tools/ipc_left_prefix/coordinate_policy.json显式禁止旧回放预览/发送，待另行迁移规划验证，不静默重用旧轨迹。
- IPC18122已更新server.py及coordinate_policy.json，重启前确认无动作运行；备份server.py.before_left_j5_20261020。prepare和send均校验坐标版本；旧preview返回409。重启会清理旧prepared/控制台会话，用户需重新获取控制权；原18121未改。
- 验证：30个随机状态新角与旧角+180度的全link FK一致，右臂一致；固定/planar生成器各100组等价检查通过；新零位解析FK/IK回归、固定轴回归和旧预览拒绝，33项相关测试通过。部署后后端坐标版本正确、motion_publications=0，未自动使能/发动作。尚未做新零位实机运动验收。
- 旧研究结果、8160保存前缀及历史JSON仍为旧零位；不能当新模型轨迹。8171是其他研究入口，运行中的进程尚未由本轮重启/验收；其负责工程师需重启资源并重新生成结果，不可混用内存中的旧模型与新磁盘文件。
- 源码新增migrate_left_j5_zero.py为一次性候选资产迁移工具，有版本防重复；不要重复运行或对已变更源模型再次叠加180度。旧底层模型目录保留原定义，当前权威联调模型是zero_model_candidate_20261018中的新资产。

## 本会话后续职责：实机联调

用户于2026-10-20明确指定本会话后续主要负责联调。重点是运控算法输出到RT-Control执行接口的适配、模型/TCP与关节反馈映射核对、轨迹预览和手动发送、控制权与状态管理、问题复现和故障定位、测试证据与交接记录。按具体联调需求修改程序，算法研究或大范围架构重构另按用户任务推进；不因职责调整自行改变控制范围、碰撞提示策略或其他既定限制。

当前工作入口：8155录制吸附前缀控制、8156左臂实时显示、8162测试杆多朝向/固定轴打表，原18121由电控提供。页面和服务可能停止，接手先核对实际状态。实机动作继续由用户先预览再明确点击发送；不自动使能或发送。发现问题要区分算法、适配、轨迹和RT执行证据，未定位时不猜责任方。

## 最新增量：固定目标轴与独立全零测试

分支 `feature/probe-fixed-joint-ik`，在原功能上增加：

- 选择J1～J7，点击“读取当前角度并锁定”，取新鲜反馈快照。只约束IK终点，运动途中允许该轴变化。归零不重新采集锁定角；只有主动再次读取或改变锁定轴才更换定义。页面重启/刷新不保证保留会话，重要记录以保存的IK/轨迹文件为准。
- 当前目标每个朝向均在ψ上1度扫描、同分支变号区间用Brent求根；接近零的小残差区间尝试有界极小化。内层依然是解析IK；不是七轴数值IK。所有输出重新检查锁定角≤1e-6rad和FK位置/朝向≤2e-6。有限扫描及近奇异分支跟踪不保证穷尽；没有解不证明物理不可达。
- 不是固定“必有8解”。尤其当前球肩肘腕机构固定J4时可能仍保留连续冗余，返回的是采样构型而非有限根全集。
- “单独预览：当前姿态→左七轴全零”生成独立动作，仍要完整预览、确认、手动发送。没有串行自动执行；不在归零后自动去目标。
- 全零后等待实机停稳，不点击重新读取锁定角，原候选继续保留；点击“从实时姿态生成关节插值轨迹”，将实际零位作为新的起点，独立预览/发送到选中解。
- 预览payload保存motion_context，区分all_seven_zero与selected_ik，记录归零时的saved_lock或目标endpoint_lock；已有哈希与起点门禁继续生效。归零动作不强制满足终点锁定角，否则无法七轴全零。
- 长距离在默认5度/s下可能超过60秒而拒绝；只能由用户自行选择允许的10/15度/s后重新预览，不能偷偷提速或放宽时长。原参考碰撞仅提示策略及其他限位/控制权/反馈门禁不变。

回归：新增test_probe_fixed_joint.py覆盖J1～J7、不同seed已知目标恢复、锁定角/FK验收、非法锁定限位拒绝、全零→原锁定角候选的两段独立构造。合计8项probe测试＋15项prefix测试通过。恢复了8162及本机8155隧道，未更改/重启工控机后端、未获取控制权/使能/发送动作；检查时实机IDLE、控制器inactive。浏览器自动访问受策略限制，本轮未完成新按钮的浏览器交互验收，由用户刷新8162试用，不声称已实机验收。

## 1. 目标与当前状态

用户需要两个功能：

1. 拖动末端位置、朝向，用解析IK求多个构型；也生成同一TCP位置、附近不同朝向的解，供人工挑选。
2. 选中解后，从新读取的实机左臂关节状态生成平滑关节插值轨迹；先预览，由用户明确点击后才执行。

开发仓库：`/mnt/mydisk/ALFA/alfa_robot_v3`，分支 `feature/probe-analytic-touch`，功能提交 `23c3d3af`，底盘重复检测修复 `7655bdba`。尚不能把本地提交等同于已推送、PR合并或正式Release。

2026-10-19已部署过可测试版：本机 `http://127.0.0.1:8162/`。完成20项CPU回归、浏览器求解/选解/预览及工控机真实准备接口验证；本次开发验证未获取控制权、未使能、未发送实机动作。未记录用户后续实机验收结果，不可声称已通过实机验收。

2026-10-20文档整理时只读检查：本机8155、8156、8162均无监听，8162用户服务非active。本文保存的是可恢复方案，不承诺这些URL始终在线；本轮没有重新启动或操作机器人。

## 2. 不可混淆的坐标与模型约定

| 项目 | 确认值／边界 |
| --- | --- |
| 实机控制范围 | 左臂J1～J7；不发送右臂、Updown、底盘、真空命令 |
| Updown | 固定−0.300m，是模型参考值，不是实时读取的升降反馈 |
| 临时TCP | 相对 `left_link7` 平移 `[0,0,0.070]`m，旋转单位矩阵 |
| 70mm方向 | Joint7局部＋Z，不是世界＋Z；随Joint7姿态变化 |
| 尺寸来源 | 用户确认的临时杆尖位置；并非通过多姿态实测完成的TCP标定 |
| 被替代的早期信息 | “法兰面伸出130.2mm”不是当前实现参数；以之后确认的Joint7＋Z 70mm为准 |
| 原吸盘tool0 | 保留原定义：Joint7局部＋Z 151mm；没有为此修改共享URDF |
| 目标输入 | 当前模型 `base_footprint` 下的XYZ米、RPY度；代码按SciPy `xyz` 欧拉约定 |
| J6 | 沿用已确认的新方向；不回退旧符号；左右J6限位±110度 |
| 左J2 | 当前模型−90～120度；其他限位以当前模型及实机软限位共同约束 |
| 模型 | 当前人工重零的旧几何/旧安装/tool候选，不是未经适配的同事3.3.1原模型 |
| 右臂 | 显示及参考验碰取配置Home，不是实机右臂反馈 |

模型目录：`research/curobo_v3/generated/zero_model_candidate_20261018/`。使用 `robot_rezero_candidate.urdf`、`fixed.yaml`。左臂Home为 `[65,-75,20,-90,0,40,90]`度，右臂Home为 `[-65,75,-20,90,0,-40,-90]`度。联调不强制从Home出发，以读取的实际左臂状态为起点。

Joint7中心到Joint6旋转轴的垂距为99.2mm；不是Joint6/7两个原点的距离，也不应叠加到70mm参数中。

## 3. 算法与执行链路

```text
8162拖动/输入TCP目标
  → 固定Updown、用实时左臂状态作IK参考
  → 原朝向与局部X/Y/Z各±25°目标（默认25°，可调0～30°）
  → 每个目标位置完全相同；不是每个欧拉角同时加25°
  → ψ每3°枚举，加当前ψ；每个解析分支分别保留
  → 去重、模型限位、当前URDF FK复核
  → 朝向分组／同组按对实时参考姿态的关节距离排序
  → 用户选构型
  → 重新读实时左臂作为轨迹起点
  → 关节空间直线＋整段五次时间律
  → 10Hz位置/速度/加速度＋实际RT五次插值峰值检查
  → 模型碰撞结果提示、轨迹播放、用户确认
  → 获取既有控制台控制权
  → /api/prepare保存并验证预览哈希
  → 用户点击发送 → /api/move，起点复核
  → /left_arm_jtc/follow_joint_trajectory
```

解析器仍以旧tool0为目标，所以要先做临时TCP→Joint7→旧tool0的刚体转换，再转到 `arm_carriage` 坐标。不要只减世界Z或重复加151mm。原临时TCP位置、朝向最后用当前URDF FK复核。

路径是关节空间插值，**不是TCP直线、不保证绕障、不自动RRT重搜**。同点多朝向只约束候选终点，运动途中杆尖不必保持在该点。后续若要“绕固定接触点连续变姿态”，应另做路径约束，不可把本工具当作已支持。

## 4. 操作步骤

1. 现场确认机器人周围、杆尖附近无障碍，硬件急停可用。使能只能由现场人员在原18121控制台完成。
2. 打开8162，点击“目标对齐实时杆尖／清除旧预览”；拖动坐标轴/旋转环，或输入XYZ/RPY。
3. 点击“1 求解解析IK候选”，选择朝向分组和构型。查看预览构型、七轴角度、FK误差。
4. 默认先用5度/s，点击“2 从实时姿态生成关节插值轨迹”。查看时长、点数、参考碰撞警告。
5. 完整播放轨迹预览。目标、构型、速度变化会清掉旧预览，需重新计算。
6. 原18121控制台停稳并释放控制权；在8162获取控制权。不要同时操作8155旧前缀发送入口。
7. 勾选现场确认后，点击“发送已预览轨迹”。若起点变化导致拒绝，重新生成预览；不硬接旧轨迹。
8. “取消本后端Action”只发送Action取消请求，**不是硬件急停**。异常按现场停机流程处理。

## 5. 门禁与已知安全边界

- 前端默认5度/s，可选10/15；最高15度/s、30度/s²、150度/s³。
- 严格0.1秒采样，整段Action一次提交；最长60秒、最多1000点。不是以10Hz重复发送Action。
- 首末点速度/加速度为0；内部点不强制全部停稳。实际五次Hermite插值内部导数峰值和软限位也需检查。
- 发送前检查新鲜反馈、停稳、控制器active、Action可用、RT ENABLED、控制权、没有冲突任务。
- 实际起点与预览起点任一轴偏差超过0.05度拒绝，不自动加入衔接段；不自动使能、抢占控制权、重发或执行后续动作。
- 用户明确要求参考碰撞只提示，不作发送硬门禁。报告保留是否通过、首个异常时刻及帧数；仅 `task=probe_touch` 且 `collision_advisory=true` 允许这一策略，不推广到正式搬箱。
- 参考检查包含模型自碰撞、底盘cuboid、地面；未建模真实工作台、接触物、杆体和实测右臂。因此碰撞通过也不等于实机安全，碰撞提示也不能直接说实物一定相撞。
- 底盘已由cuboid负责与手臂的检查，必须在创建检查器前调用 `replace_arm_base_pairs`。早期遗漏此步骤，旧底盘球与cuboid重复检测产生错误阻塞；已修复，**不能把修复解释成关闭底盘检查**。
- 本工具执行FJT位置轨迹，不是力控、柔顺控制或接触检测；没有接触力/触碰成功判据。预览页面不是实时执行反馈图，实际状态可看8156和原控制台。

## 6. 代码、数据和验证入口

以下路径相对 `/mnt/mydisk/ALFA/alfa_robot_v3/`：

| 文件 | 职责 |
| --- | --- |
| `research/curobo_v3/tools/probe_touch_app.py` | 8162前端、任务状态、朝向分组、预览与手动发送 |
| `research/curobo_v3/tools/probe_touch_ik.py` | 70mm TCP变换、多朝向目标、解析IK与FK复核 |
| `research/curobo_v3/tools/probe_touch_motion.py` | 五次时间律、参考碰撞检查、预览数据与哈希 |
| `research/curobo_v3/tools/v3_rezero_analytic.py` | 当前重零坐标与底层解析库映射 |
| `research/curobo_v3/tools/ipc_left_prefix/server.py` | 工控机ROS Action、控制状态、准备/发送/取消门禁 |
| `research/curobo_v3/tools/ipc_left_prefix/smooth_timing.py` | RT插值重建、峰值核验、预览校验 |
| `research/curobo_v3/tools/ipc_left_prefix/console_lease.py` | 18121认证、控制权和软限位读取；不在文档中保存令牌 |
| `research/curobo_v3/tools/ipc_left_prefix/live_viewer.py` | 8156左臂实测反馈、Joint7和70mm杆尖标记 |
| `research/curobo_v3/tools/probe_collision_viewer.py` | 8161离线碰撞诊断，不是运动入口 |

IK表：`research/curobo_v3/generated/probe_touch_runs/latest_ik.json`，会被下一次求解覆盖。

轨迹：同目录 `preview_<哈希前16位>.json`，含七轴q/v/a/时间、目标TCP、固定升降、参考碰撞结果。长期验收前请另存相应IK表与轨迹，不能仅依赖latest或/tmp。

回归命令（不发实机动作）：

```bash
cd /mnt/mydisk/ALFA/alfa_robot_v3 && \
/mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python -m unittest discover \
  -s research/curobo_v3/tests -p 'test_*prefix*.py' && \
/mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python -m unittest discover \
  -s research/curobo_v3/tests -p test_probe_touch_ik.py
```

当时结果15＋5项通过，覆盖IK/FK、同点偏转、插值导数峰值、预览篡改/超时拒绝、准确发送预览点及起点变化拒绝。后端publisher使用mock；真实 `/api/prepare` 返回ready，但没有调用实际 `/api/move`。示例解数量约4128、偏转候选轨迹11.8秒，只是当时样本，不是每个目标的保证。

## 7. 部署与恢复

| 位置 | 配置 |
| --- | --- |
| 工控机 | SSH `ar@192.168.100.40` |
| 原控制台 | `http://192.168.100.40:18121/`，保持既有实现 |
| 自建执行后端 | 工控机localhost:18122，目录 `/home/ar/alfa_left_prefix_control_20261018` |
| ROS | Humble，ROS_DOMAIN_ID=77；左7轴反馈与 `/left_arm_jtc/follow_joint_trajectory` |
| 本机8155 | SSH隧道转发到工控机localhost:18122；不是另一个规划器 |
| 本机8162 | 打表前端、解析IK和本机GPU参考验碰 |

后端历史用户服务：`alfa-left-prefix-control.service`；隧道：`alfa-left-prefix-tunnel-8155.service`；前端：`alfa-probe-touch-8162.service`。先检查服务和端口，不重复启动。旧服务以transient方式创建，重启电脑后可能不存在；仓库代码和本文才是恢复依据。

恢复隧道（在本机新终端；8155空闲且SSH可达时）：

```bash
ssh -N \
  -o BatchMode=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -L 127.0.0.1:8155:127.0.0.1:18122 \
  ar@192.168.100.40
```

恢复工控机后端（新终端；先确认旧后端没有动作运行、18122空闲）：

```bash
ssh -t ar@192.168.100.40 'bash -lc "cd /home/ar/alfa_left_prefix_control_20261018 && source /opt/ros/humble/setup.bash && source /home/ar/rt-control-electri127/install/setup.bash && export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=0 && exec python3 server.py --library /home/ar/alfa_left_prefix_control_20261018/prefixes.json --port 18122 --scope left7_joint_replay"'
```

恢复前端（本机新终端；8162空闲）：

```bash
cd /mnt/mydisk/ALFA/alfa_robot_v3 && \
/mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python \
  research/curobo_v3/tools/probe_touch_app.py \
  --port 8162 \
  --backend http://127.0.0.1:8155
```

IPC中部署过的 `server.py`、`smooth_timing.py` 对应本功能版本，依赖NumPy/SciPy；旧server备份 `server.py.before_probe_touch_20261019`。不要在运动中覆盖/重启后端，也不要只回滚server而忽略模块兼容性。原18121包未被本功能修改。

接手工程师先读本文和代码、核对当前软限位/模型与反馈映射，再运行离线回归。实机复测必须由用户现场确认并明确点击，不得因恢复服务或回归测试而自动运动。
