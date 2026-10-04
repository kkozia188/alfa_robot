# 工控机与本机cuRobo持续GPU压测

2026-10-04，Refs MOTION-275。目的是检查连续GPU高负载下升温是否导致吞吐下降，不比较规划器成功率或完整任务耗时。

## 结论

工控机RTX A2000 Laptop出现明确的温度降频。本机RTX4060 Laptop在本轮约2分钟内稳定。

| 指标 | 本机RTX4060 Laptop | 工控机RTX A2000 Laptop |
| --- | --- | --- |
| 实际持续负载 | 121.05s | 120.77s |
| GPU平均利用率 | 98.42% | 99.17% |
| 起始/末尾温度 | 56 / 59°C | 70 / 88°C |
| 最高温度 | 60°C | 89°C |
| 前两窗口吞吐 | 525,936状态/s | 457,072状态/s |
| 末两窗口吞吐 | 530,350状态/s | 310,272状态/s |
| 前后变化 | +0.84% | -32.12% |
| 整段平均吞吐 | 528,399状态/s | 373,356状态/s |
| 首/末SM频率 | 2085 / 2100MHz | 1687 / 742MHz |
| 温度降频采样 | 0/119 | 97/118 |

工控机约21.66秒、86°C时首次记录到软件温度降频Active；后续温度在87～89°C附近，频率与功耗逐级下降，GPU利用率保持接近100%，吞吐同步降低。因此有驱动明确降频标志支持，不是仅根据温度猜测。末段吞吐约为本机末段的58.5%；整个2分钟均值约为本机70.7%。

前后比较采用实际完整采样窗口加权吞吐：本机前0～21.93s、末105.23～121.05s；工控机前0～22.94s、末103.87～120.77s。最后一个窗口是结束时的短窗口，不能把两个区间写成精确等长20秒。原始每窗口数据与逐秒遥测保留。两轮各一次，不能外推到所有工控机或数小时稳态；工控机末尾仍有性能继续下降迹象。

## 工作负载与对齐

- 同一15轴解析代理、机器人与两个附着箱共340个球，场景25箱墙＋4个集装箱结构，共29个cuboid。
- 每批16384状态，4批确定性随机合法关节数据交替；每次做实际cuRobo FK、自碰撞、世界球碰撞和关节边界检查。没有CPU RRT树管理、近邻队列、IK或抽离。
- 常驻检查器完成预热后用CUDA Graph重复真实计算，64次replay为一个同步计时块，不做空GPU矩阵乘法或模拟碰撞。采样随机数据提前生成，避免CPU采样成为温度测试瓶颈。
- 普通调用与Graph在四批输入上的碰撞判定逐项一致，浮点代价使用atol=1e-6、rtol=1e-5核对。首次逐位相等检查仅有2.38e-7数值误差，修正测试比较方法，没有修改cuRobo计算结果。
- 两机模型（剔除部署路径）、输入数据、脚本SHA256、球数、障碍数、PyTorch2.8.0+cu128和CUDA12.8均一致，比较工具会校验这些字段。
- 本机驱动580.178.04；工控机610.57.04、Ubuntu22.04 realtime内核。比较的是当前两台平台，不是严格只改变GPU的实验。未测室温，也未更改功耗、风扇、驱动或系统CUDA。
- 本机已有显示/Sunshine负载保留；工控机测试前只有桌面GPU进程。没有其他cuRobo规划计算并发。环境传输会占CPU/网络，但不占GPU，正式GPU利用率与吞吐曲线已记录。
- 工控机原无PyTorch/cuRobo；在/home/sev_v3/curobo_thermal/venv隔离目录部署本机同版本site-packages（约7.4GB），只使用现有Python3.10和NVIDIA驱动，不替换系统环境。测试结束时已确认压测进程退出、利用率回到1%、温度下降且温度降频解除。

## 证据

generated/thermal_20261004包含local_120s.json、ipc_120s.json、summary.json、comparison.png、两份5秒短测及冻结robot.yml/robot.urdf；本地另保留两机hardware.xml，不纳入Git。脚本位于tools/collision_benchmark/v3_gpu_thermal_stress.py；汇总图工具v3_compare_thermal.py。

## 复跑命令

本机：

```bash
cd /mnt/mydisk/ALFA/alfa_robot_curobo_compare/research/curobo_v3
/mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python tools/collision_benchmark/v3_gpu_thermal_stress.py \
  --robot-config generated/thermal_20261004/robot.yml \
  --urdf generated/thermal_20261004/robot.urdf \
  --output generated/thermal_20261004/local_120s.json \
  --duration 120 \
  --window 10 \
  --batch-size 16384 \
  --replays-per-block 64
```

工控机（在本机终端直接执行）：

```bash
ssh sev_v3@192.168.100.28 \
  'cd /home/sev_v3/curobo_thermal && venv/bin/python v3_gpu_thermal_stress.py \
    --robot-config robot.yml \
    --urdf robot.urdf \
    --output ipc_120s.json \
    --duration 120 \
    --window 10 \
    --batch-size 16384 \
    --replays-per-block 64'
```

本次结果说明热机状态会显著影响GPU FK/碰撞效率，但不直接证明完整RRT运行时间必然增加32%。建议先核对工控机散热、进出风和风扇工作状态；不修改降频保护。若后续调整散热，复跑同一负载看温度、驱动降频原因和末段吞吐是否改善。
