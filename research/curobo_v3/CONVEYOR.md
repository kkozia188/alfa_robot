# 龙头车实时实例与规划快照

实时场景中的机器人和龙头车组合独立拥有map位姿。两台龙头车共用组合根，相对
间距5cm；默认组合根在机器人后方1.9m。机器人yaw不会自动带动组合。
复用MOTION-246旧原型的STL，转存为ASCII，米制顶点和三角面不变：每台
2.40×0.94×0.732m、12个三角面、封闭长方体。这是当前资产本身的精度。

`LiveScene.capture()`将当前实例原子展开为不可变SceneSnapshot，Mesh顶点/三角面
直接冻结到快照，cuRobo使用真实Mesh通道而非AABB替代。对象时间戳过期、未来时间、
缺失所需实例或map变换缺失会拒绝捕获。显示专用对象不进入碰撞快照；关闭龙头车
碰撞选项仍可看到实时实例，冻结快照中则没有它们。

当前演示是本地模拟状态刷新，不是ROS TF/驱动接入。碰撞检查覆盖机器人球与世界
长方体/Mesh、自碰撞、限位和地面，不覆盖龙头车与集装箱之间的环境-环境碰撞，
不生成导航轨迹。完整机械臂任务继续由MOTION-248跟踪。

## 打开交互程序

```bash
cd /mnt/mydisk/ALFA/alfa_robot_v3_curobo/research/curobo_v3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python tools/v3_conveyor_snapshot_demo.py \
  --port 8094 \
  --mesh models/conveyor/roller_conveyor_collision_solid_v2.stl \
  --gap-m 0.05 \
  --follow-distance-m 1.9 \
  --snapshot-output /tmp/motion246_snapshot.json
```

浏览器打开`http://127.0.0.1:8094`：

1. 点击“播放驶入与后方跟随”观察实时运动，或输入双方x/y/yaw独立摆位。
2. 把组合x设为-4m，点击“冻结当前状态”。再把实时组合x设为0，旧快照不变。
3. “显示内容”切到“本次规划快照”，看到冻结的机器人和碰撞Mesh；点击检测应通过。
4. 再冻结x=0的重叠状态，检测应报告环境碰撞。
5. 取消“龙头车参与下次规划碰撞”，重新冻结并检测，应恢复无龙头车的结果。

碰撞按钮始终使用冻结快照，修改实时控件不会改变它。此按钮检查单一机器人状态，
不代表整条抓取轨迹无碰撞。快照保存到指定JSON，检测另存同名`.collision.json`。
模拟播放从x=-3到0只用于状态/可视化验证，不宣称是导航生成或无碰撞路线。

## 无界面检查

```bash
cd /mnt/mydisk/ALFA/alfa_robot_v3_curobo/research/curobo_v3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python tools/v3_conveyor_snapshot_check.py \
  --output /tmp/motion246_conveyor_mesh_check.json
```

该检查验证远离、重叠、禁用、快照隔离及GPU缓存随几何身份变更。全周期PlanRequest
也可包含这些Mesh，但当前全周期仍限制原点固定底盘，不把有龙头车就能完成取放当作已验收。
