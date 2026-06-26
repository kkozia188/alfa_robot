# alfa_robot_execution_bridge

统一的 ALFA 机器人执行接口包。当前只实现 `mock` 后端，用于无电机、无 EtherCAT 主栈时调通上层任务编排、MoveIt 和执行接口。

## 接口

- Action server: `/alfa_execution/execute_joint_trajectory`
- Action type: `control_msgs/action/FollowJointTrajectory`
- State topic: `/joint_states`
- Joint order:
  - `right_joint1` ~ `right_joint6`
  - `left_joint1` ~ `left_joint6`
  - `turn`

发送的轨迹必须包含完整 13 个关节，位置单位是 `rad`，目标含义是 ROS 机械零位坐标下的绝对关节位置。

## 运行 mock

```bash
cd /mnt/mydisk/ALFA/alfa_robot/ros2_ws
colcon build --packages-select alfa_robot_execution_bridge --symlink-install
source install/setup.bash
ros2 launch alfa_robot_execution_bridge execution_bridge.launch.py
```

另一个终端发送测试轨迹：

```bash
cd /mnt/mydisk/ALFA/alfa_robot/ros2_ws
source install/setup.bash
ros2 run alfa_robot_execution_bridge send_mock_trajectory --duration-s 4 --amplitude-deg 15
```

## 后续真实后端

真实 EtherCAT 主栈后端应继续复用同一个 action 和 joint state 话题，只把 `mock` 后端替换为 ros2_control / EtherCAT 写入实现。
`config/execution_bridge.yaml` 中的 `direction_signs` 记录了当前实机方向标定关系；mock 后端不使用该符号翻转，只发布 ROS 语义下的关节角。
