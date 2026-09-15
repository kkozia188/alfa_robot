# V3.1.1 MoveIt Demo

当前通用 Demo 使用内网 `robot_description/robot_v3` 的固定消费者快照：

- 模型版本：`robot_v3.1.1`
- 上游提交：`b0c53aa8ca0bba1701d2f6680115012ed38079c0`
- 27个运动关节：双臂14轴、升降、头部、两侧夹爪、后悬架和8个轮组关节
- `updown`：`[-1,0]m`，数值增大时向上；Demo默认`-0.5m`，即物理中位高度
- 两侧夹爪：`[0,0.1]m`
- 后悬架：来源范围`[0,0.17]m`；Demo home采用`0.085m`，避免来源零位下两个后转向件与底座相交
- 七轴限位：J1/J3/J5/J7 ±180°、J2 ±105°、J4 ±145°、J6 ±110°

V3.1.1 尚未冻结抓取TCP，因此 `left_arm/right_arm` 的KDL链止于J7。通用MoveIt
手动规划可用；依赖旧`tool0`或V3.0.9解析常量的抓箱、可达性和冗余解专项Demo，
不能视为已经适配V3.1.1。

## 构建并验证

```bash
cd /mnt/mydisk/ALFA/alfa_robot_v3
./build_v3_moveit_demo.sh
```

构建入口会先检查本地消费者快照是否仍与权威 description 提交一致，然后执行Release
构建及description/MoveIt回归测试。`tools/check_v3_demo_install.py`会检查源码与install
中的模型锁、活动Xacro、MoveIt限位和SRDF是否一致。

## 启动

```bash
cd /mnt/mydisk/ALFA/alfa_robot_v3
./run_v3_moveit_demo.sh
```

启动脚本默认使用`ROS_DOMAIN_ID=79`。如果install仍是旧模型，会先自动重建，避免继续
显示V3.0.9。也可以直接执行`ros2 launch alfa_robot_moveit_config demo.launch.py`，
但必须先source本工作区；Launch会拒绝不是V3.1.1的description锁。

在RViz中选择`left_arm`、`right_arm`、`dual_arm`或`dual_arm_with_updown`进行手动规划。
`whole`保留全部27个来源自由度，用于模型检查，不表示推荐一次规划轮组、夹爪和双臂。
来源`source_zero`不是无碰撞姿态：后悬架为0时会出现两对底座碰撞，因此没有作为Demo home。

## 快照来源

更新消费者快照：

```bash
/usr/bin/python3 tools/sync_v311_description.py
```

仅核对而不修改：

```bash
/usr/bin/python3 tools/sync_v311_description.py --check
```

锁文件为`ros2_ws/src/alfa_robot_description/config/upstream_description.lock.json`。
不要手工复制单个URDF或只修改MoveIt限位；模型、SRDF、ros2_control和控制器清单必须
作为一组通过回归测试。
