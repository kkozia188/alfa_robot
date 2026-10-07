# 双臂箱墙抓放初版方案

2026-10-07。本PR固化从同事工作树同步的94组任务Demo及本轮确认的无硬倾角策略，
作为cuRobo算法研究基线。94组是可选组合数量，不是已验收成功数量。

## 流程

- 上三层60组侧吸，底两层34组顶吸；同排/相邻排配对，不跨第三与第四层。
- 根据最低任务层生成连续填充障碍，保留当前目标箱。这不是25箱全部独立保留的场景。
- 顶吸先检查底盘前移0.3m；侧吸保持当前底盘。数值IK生成32接触候选。
- 固定updown/ψ解析后抽35cm筛选；通过后用Informed Connect到3cm预抓取，直线贴近吸附。
- 抽离后携箱到统一吸盘Pose：局部x=0.6m、y=±0.425m、z=1.0m，吸盘朝下。
- 携箱yaw180°、释放并删除箱体、空载转回、回原上身Home；顶吸最终底盘x=0.3m。

自由搜索每段2s上限，Informed Connect首个精检合法路径提前返回，无Shortcut/TrajOpt。
本轮只在TaskCyclePlanner设置max_box_tilt_deg=180，允许任意箱体倾角；暂保留
稳定性软代价。旧通用FullCyclePlanner的默认倾角策略未改。

## 必须保留的边界

预抓取到位对目标箱独立检测；最后3cm贴近排除两个目标箱碰撞，仍检查机器人和其他
环境。抽离途中延后附着箱碰撞，35cm终点启用。40球是内接近似，无法保证完整箱体
真实网格无碰撞；不把这些豁免描述为严格接触模型。独立臂/箱对底盘STL收束层未接入。

顶吸在底盘前移后的局部坐标系规划。输出18轴帧/生成障碍是map坐标，但内部
snapshot身份及预测事件仍为局部语义。它们不可不经转换直接用于生产执行。
所有帧未做时间参数化，没有ROS Action/RT执行、吸力或动力学验收。

## 已验证

| 本轮任务 | 策略 | 结果 | 计算时间 | 轨迹帧 |
| --- | --- | --- | --- | --- |
| L4/R0 | 顶吸，89°旧策略 | 完整成功，候选7 | 3.956s | 2258 |
| L19/R15 | 侧吸，无硬倾角限制 | 完整成功，候选1 | 4.075s | 1574 |

两条上身Home误差均6.38e-8，最大旋转相邻步长≤0.5°。顶吸文件保留原89°policy，
不能把它解释为取消倾角后的重验；当前在线计算使用180°策略。数值依赖当次运行，
不作为性能排名。L19/R15此前失败原因是统一向下吸盘目标使侧吸箱倾角90°而超过89°。

## 可复制命令

先运行prepare_checkout定位当前模型路径并编译解析桥，再启动我方端口8120。
源工程师8100/8110/8111保持独立，本会话使用8120～8129，启动前检查端口占用。

```bash
cd /mnt/mydisk/ALFA/alfa_robot_v3/research/curobo_v3
/mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python tools/prepare_checkout.py \
  --compiler g++
PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python tools/v3_task_full_cycle_demo.py \
  --port 8120 \
  --result generated/task_demo_sync/no_tilt_L19_R15.json \
  generated/task_demo_sync/reproduced_L4_R0.json
```

打开http://127.0.0.1:8120，勾选播放查看已计算结果。切换组合任务后点击
“计算完整流程并播放”在线重算；失败显示真实阶段与可用诊断，不用其他轨迹代替。

```bash
cd /mnt/mydisk/ALFA/alfa_robot_v3/research/curobo_v3
PYTHONDONTWRITEBYTECODE=1 \
  /mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python -m unittest discover \
  -s tests \
  -p 'test_core_*.py' \
  -v
```

源目录alfa_robot_v3_curobo的未提交算法保持原样，不把本PR理解为其全部后续研究合并。
本PR只收口依赖模块、入口与两份本轮复现；歪斜箱/18轴IK、圆柱收束及历史失效回放不包含。
