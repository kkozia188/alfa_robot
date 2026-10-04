# 四规划器完整流程对比

2026-10-04，当前解析代理071cb95与40球模型，L24/R20第一组合。运输目标为冻结完整6D Pose（X减5cm）。三段自由搜索均使用所选规划器：Home到接触、抽离后到放置、释放转回后到Home。解析35cm抽离和固定yaw180/回0保持相同。

Informed基线先通过32候选选择第1接触姿态；对比其他方案时冻结同一个接触/抽离构型，仍记录数值接触IK返回32解，避免将起点分支变化误认为规划器效果。后续放置采用同一完整6D目标及最多16IK构型。各段2秒预算，无Shortcut/TrajOpt；PRM和RRTConnect未加无效重试或放宽碰撞。

| 方案 | 热态全部规划 | 帧数 | 完整结果 |
| --- | --- | --- | --- |
| Informed RRT | 6.396s | 1453 | 成功 |
| Batched RRT | 6.448s | 1475 | 成功 |
| Batched RRTConnect | 6.303s | 2044 | 成功 |
| GPU PRM | 6.298s | 2682 | 成功 |

Informed首次冷启动约10.59s，与另外三项热态不同，不纳入表中速度排名；Informed热态单独复测后替换预载结果。只有首组单次热态样本，不证明总体哪一种优越。帧数来自统一密化和360+360固定转身，不能当作执行时间或自然性评分。

验证：四条均从原Home出发、到位后吸附、固定Updown解析退35cm、携箱到目标、转身180释放箱消失、转回0、空载回原Home；最终Home误差0、底盘x/y/yaw最终0，臂相邻步长≤0.5°、释放后payload为false。沿用暂缓抽离附着箱检查的既定合同，未补真实OBB全周期验收。验收使用当前GPU球检测、地面、限位、稳定性。

数据：generated/v3_analytic_071cb95/full_cycle_four_planners.json。隔离工作目录避免影响motion-276当前未提交开发。prepare_checkout只为隔离目录重定位；路径改写不作为本次代码提交。

本机页面命令：

```bash
cd /mnt/mydisk/ALFA/alfa_robot_curobo_compare/research/curobo_v3
/mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python tools/v3_full_cycle_demo.py \
  --port 8090 \
  --comparison-json generated/v3_analytic_071cb95/full_cycle_four_planners.json
```

页面规划器下拉用于切换已计算结果回放，阶段下拉/帧条用于检查衔接；“计算完整流程并播放”对当前所选规划器在线重算，使用同一起点。切换任务清除该任务的对比数据，随后可逐个实时计算；未测试全部60组合。

刷新本机首组对比：使用同环境执行tools/v3_cycle_compare.py，启动服务前先停止旧服务避免两个GPU进程叠加OOM。该脚本先跑Informed基线再冻结起点，不将失败方案换算法兜底。
