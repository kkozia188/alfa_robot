# 解析抽离完整流程 Demo

启动：`cd /mnt/mydisk/ALFA/curobo_v2_ws`，运行 `/mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python tools/v3_full_cycle_demo.py --port 8090`。

页面为 http://localhost:8090/，选择组合任务后点击“计算完整流程并播放”。一次规划完整链路，全部成功后才自动播放；失败保留阶段与前缀，不自动更换策略。

cuRobo接触IK使用512种子返回最多32个成功解，并按到第一Home的加权关节距离排序。逐候选以该姿态固定Updown和左右ψ，调用V322Left/Right原生C++解析IK实现同步后退35cm，以相邻关节角选择连续分支，独立URDF FK检查直线及朝向。首个抽离和Home到接触RRT均通过的候选进入携箱搬运，不为后续失败再回退选起点。

到位、携箱运输、空载回位均使用cuRobo GPU碰撞加速的批量RRT；保持每段2秒搜索预算。携箱原地0→180°，释放时附着箱与地图中的任务箱消失，再空载180→0°并双臂回第一Home。放置目标使用冻结完整6D Pose（已X减5cm）。模型为解析代理071cb95，负载使用确认的40球模型。

抽离检查沿用既定接触合同：抽离途中检查机器人、自碰撞、集装箱、地面及箱体朝向，附着箱球碰撞在35cm完成后启用。这不是抽离途中真实箱体OBB完整碰撞验收。

第一组合L24/R20验证：32接触候选，第1候选通过；实时全链路11.456s，1453帧。固定Updown约-0.001m，抽离最大线偏差0.03255mm、朝向偏差0.00463°；密化相邻臂关节步长≤0.5°，阶段接续检查通过，最终Home误差0。其他59组未作批量验收。

证据：`full_cycle_first_pair.json`（独立冒烟），`full_cycle_latest.json`（页面实时计算，后续运行覆盖）。C++接口`tools/v3_analytic_bridge.cpp`编译为本目录`libv3_analytic_bridge.so`；解析源码来源`/mnt/mydisk/ALFA/alfa_robot_v3/ros2_ws/src/alfa_robot_analytic_ik/`，仅依赖Eigen，可在当前研究环境中调用。
