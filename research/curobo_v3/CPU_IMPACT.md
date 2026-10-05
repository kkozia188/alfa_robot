# 工控机GPU持续压测对CPU和其他程序的资源影响

2026-10-05，MOTION-275。重新运行上一轮同一GPU FK/碰撞负载121.44秒，新增20秒背景基线与10秒退出恢复监测。工控机i7-14700（20物理核、28逻辑核）、RTX A2000 Laptop。没有同时进行全核CPU压力测试，也没有更改进程优先级、CPU亲和性、风扇或温度保护。

## 结果

| 指标 | 背景基线 | 压测进程运行期（含初始化） | 退出恢复 |
| --- | --- | --- | --- |
| 整机CPU平均（28逻辑核总计100%） | 3.21% | 6.20% | 2.08% |
| 整机CPU P95 | 3.56% | 6.81% | 2.16% |
| 整机CPU采样峰值 | 4.53% | 18.16% | 2.17% |
| CPU Package平均温度 | 94.70°C | 89.54°C | 95.60°C |
| CPU Package最高温度 | 100°C | 97°C | 99°C |
| Package热降频计数增量 | 2 | 31 | 1 |

cuRobo压测进程运行期平均CPU101.95%（单个逻辑核=100%），初始化峰值427.17%即约4.27核，进入持续GPU负载后基本约100%。77个线程不等于77核负载；主要计算/等待线程在CPU8/9/10等逻辑核间迁移。每个逻辑核占用、频率、拓扑、可用温度、热降频计数、CPU/内存/IO压力、内存状态与所有活跃进程每秒保存。cpu_impact.png提供全部28核的热力图。

背景gnome-shell已有平均75.75%单核占用，运行期51.19%，退出期46.95%。占用降低不能解释成它被cuRobo抢占或变快，因为没有测它的响应时间或输入负载。CPU平均上升约2.99个百分点，也不能直接作为压测进程CPU总占用，因为背景进程同时变化。监测器自身约1.9%单核CPU，开销已单独记录，约占整机0.07%。

GPU正式计算121.44秒，平均38.85万状态/s，再次出现GPU温度降频，最后88°C/937MHz。与昨天不是同一个环境初温/背景负载，不作为严格重复性排名。

## 对其他程序的含义

- CPU未被整体吃满，但长时间占用约1个逻辑核，同核执行的其他任务可能产生竞争；初始化短暂更高。真实RRT还有CPU树管理/近邻等，本次没有覆盖这部分。
- GPU持续接近100%，共享GPU的视觉/推理/其他cuRobo工作可能受影响；本次没有运行其真实业务请求，无法量化延迟或deadline违约率。
- CPU在压测前就高温并发生热降频，运行期又有新计数；恢复后温度重新接近99°C。不能从这轮结果推断GPU压测导致CPU升温，也不能因为总体CPU占用低就保证其他程序不受影响。
- 热降频计数按phase统计，同一个Package计数会出现在所有逻辑核sysfs文件中；不能把28份重复Package计数相加。整段Package增量34次，其中baseline2、benchmark31、recovery1。
- CPU压力PSI有等待，但没有出现全CPU堵塞；没有真实业务周期耗时，所以CPU占用/频率/温度均为资源证据而非业务实时性证明。

建议优先检查工控机CPU/GPU散热与背景桌面负载，再以真实视觉/控制任务的周期耗时、P95/P99、deadline miss与本GPU负载并行测量，才可明确对别的程序的业务影响。当前未擅自停掉gnome-shell/远程桌面，也未绑定CPU或修改RT调度。

## 证据与复跑

generated/cpu_impact_20261005：ipc_cpu_20261005.json全部每秒监测；ipc_gpu_20261005.json完整GPU吞吐遥测；cpu_summary.json逐核/进程汇总；cpu_impact.png图；ipc_cpu_20261005.benchmark.log原输出。启动和退出确认成功，所有153份CPU监测样本均包含28逻辑核。

```bash
ssh sev_v3@192.168.100.28 \
  'cd /home/sev_v3/curobo_thermal && python3 v3_cpu_impact_monitor.py \
    --output ipc_cpu_20261005.json \
    --baseline 20 \
    --recovery 10 \
    -- venv/bin/python v3_gpu_thermal_stress.py \
    --robot-config robot.yml \
    --urdf robot.urdf \
    --output ipc_gpu_20261005.json \
    --duration 120 \
    --window 10 \
    --batch-size 16384 \
    --replays-per-block 64'
```

CPU频率是sysfs当前频率采样，不是APERF/MPERF有效频率；温度来自coretemp可见的Package/物理核传感器，不伪称每个超线程有独立温度。初始化和数值校验包含在CPU监测运行期，GPU吞吐正式计时仍在预热完成后。监测未采集其他进程敏感参数，只保存PID、comm、CPU、RSS及线程数。
