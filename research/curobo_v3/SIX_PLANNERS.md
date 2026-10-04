# 完整流程增加 Connect＋Informed 和 BIT* GPU

2026-10-04，Refs MOTION-275。基于既有四规划器首组对比，增加两个选择；CBS本轮不测。实现同步到远端核心封装 a2aa28a4，入口为 CuroboBackend.search_path。

## 相同实验条件

首组 L24/R20，冻结原Informed基线选中的同一个接触构型、固定Updown解析退35cm、同一完整6D放置目标、40球附着箱、地面/集装箱场景。三段自由运动均使用所选新规划器：Home→接触，抽离终点→放置，空载→Home。携箱yaw180、释放消失、转回0也全部包含。每段最多2s搜索，不加Shortcut、TrajOpt或更换算法兜底。

新核心封装下首组热态结果（单次样本，不是全任务成功率）：

| 方案 | 全流程规划 | 帧数 | 接近/运输/回Home首条粗检通路 | 结果 |
| --- | --- | --- | --- | --- |
| Connect＋Informed | 6.318s | 2044 | 44.2 / 98.0 / 31.5ms | 完整成功 |
| BIT* GPU批量变体 | 6.227s | 1516 | 13.5 / 14.7 / 8.4ms | 完整成功 |

首条通路时间不含后续2s内继续择优及最终0.5°离散验收；不能作为可执行完整轨迹的交付耗时。冷启动Connect为10.64s，未混入表中。两条最终Home误差均为0，双臂相邻帧关节步长最大0.499963°。沿用目标箱从接近场景排除、抽离阶段暂缓附着箱检测的既定研究合同；不代表真实OBB或连续碰撞/动力学验收。

BIT*接近段检查569条粗边、换父节点7次，运输段231条、回Home31条。运输与回Home各为2个原始路点，播放帧包含密化、解析抽离与转身；1516帧不等于1516个搜索或TrajOpt控制点。

## 算法区别与边界

Connect＋Informed沿用当前双向批量树，交替扩展并跨树尝试直连；找到路径后用代价上界生成起终点加权椭球样本，保留30%全域采样。它不是带换父节点的Informed RRT*；当前Connect也不是OMPL逐步贪心extend-until-blocked的原版实现。

BIT*采用原论文的两种队列思想：节点队列决定何时找近邻，边队列按累计代价＋边长＋到目标下界排序；隐式随机几何图按批次新增样本，延迟验边，改善路径时换父节点并更新后代代价，再剪去不可能改善的连接。几何长度下界适用于现有非负箱体倾斜罚项。参考：[BIT*论文](https://personalrobotics.cs.washington.edu/publications/gammell2020bitstar.pdf)。

GPU承担批量FK/碰撞和近邻距离；Python/CPU维护两队列和父子树。每批512个候选状态、最多64条边一起验，近邻数随log(N)增长，节点上限12000。保留全域探索、队列边分组处理、有限节点上限和离散碰撞使其属于自研GPU批量变体，本次不声称原论文的渐近最优保证。它不是cuRobo官方BIT*封装，也不是PRM改名。

本轮没有证明“BIT*速度爆炸”：三段仍按2s择优，总耗时接近其他方案。BIT*接近段近邻/队列相关扩展约1.01s、GPU验边约0.58s；回Home验边仅约3.44ms，剩余预算主要花在采样/筛选/近邻管理。更大批量会增加这些开销，不能只凭并行能力推断更快。先看页面动作，再决定是否改变预算和采样策略。

## 页面与重算

```bash
cd /mnt/mydisk/ALFA/alfa_robot_curobo_compare/research/curobo_v3
/mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python tools/v3_full_cycle_demo.py \
  --port 8090 \
  --comparison-json generated/v3_analytic_071cb95/full_cycle_four_planners.json \
    generated/v3_analytic_071cb95/full_cycle_extra_planners.json
```

下拉增加Connect＋Informed、BIT*＋GPU，切换可直接回放同起点结果。“计算完整流程并播放”仍为实时规划。换任务清除预载对比数据，不将首组结果套到其他任务。

先停止占GPU的旧页面，再重算两种：

```bash
cd /mnt/mydisk/ALFA/alfa_robot_curobo_compare/research/curobo_v3
/mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python tools/v3_cycle_compare.py \
  --planners informed_connect bitstar informed_connect \
  --baseline generated/v3_analytic_071cb95/full_cycle_four_planners.json \
  --output generated/v3_analytic_071cb95/full_cycle_six_planners.json
```

重复最后一项只是取得热态Connect测量；前一冷态仍输出日志，不用于调参或失败兜底。JSON保留每段统计与完整帧。程序原始空批次错误已修复；GPU小场景测试覆盖无障碍直连、墙体绕行、同起终点。未额外遍历全部任务，也未验证本页浏览器截图。

## TrajOpt路点数量

RRT输出的上千帧可能只是少量路点插值密化。TrajOpt计算量主要取决于实际参与优化的控制节点、种子和迭代次数；执行密化帧多，不代表优化变量多。所有搜索路点都变成控制变量会增加代价，过度压缩又可能跨过障碍。后续应在保持碰撞绕行形状的条件下重采样，而非一律压成32点。本次两种新增方案不做TrajOpt。
