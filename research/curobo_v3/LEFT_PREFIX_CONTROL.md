# 左臂初始至接触录制回放

feature/ipc-left-prefix-control，用户已确认只回放左7轴，不控制升降/右臂、不承诺TCP到原箱面。基于contact-roll-alignment实验，未PR合并或实机运动验收。

## 已完成

- generated/left_prefix_control_20261018/prefixes.json保存六组左臂前缀。截断在attach之前最后contact_approach帧，完全排除吸附、抽离、运输、放箱、释放；回Home只逆序同一前缀。
- 工控机/home/ar/alfa_left_prefix_control_20261018存放独立标准库脚本和数据；alfa-left-prefix-control.service运行在127.0.0.1:18122，ROS_DOMAIN_ID=77。没有更改RT控制源码、没有自动使能。
- 本机alfa-left-prefix-tunnel-8155.service将localhost:8155转发到上述后端。前端可选择六组任务、查看真实反馈/目标角度、最高速度、预计时长、获取/释放18121控制权及受控停止。
- 最新控制台反馈左[65,-75,20,-90,0,40,90]°，J3/J6与自制模型同号。旧反号映射已经取消。
- 后端通过18121既有认证建立独立会话、SSE心跳及lease互斥，令牌只在工控机读取、不会发给前端或记录内容。实际软限位从18121读取，每次发送再检查；不跳过起点/新鲜反馈/停稳/active/ENABLED/订阅者门禁。
- 停止沿当前五次插值段到下一个原记录点零速度停稳，不是硬件急停；停稳后允许沿已走前缀逆序回Home。页面失去心跳时尝试先受控停稳再释放lease。

## 当前真实阻塞

已读取实际ROS图：/left_arm_jtc/joint_trajectory没有订阅者，left_arm_jtc仅订阅/parameter_events。18121真正执行接口为/left_arm_jtc/follow_joint_trajectory Action。

当前脚本按用户原要求使用JointTrajectory话题，不能驱动此RT实例。8155明确显示该阻塞并禁止发送；实机motion_publications=0，未实际移动。不能绕过缺失订阅者门禁，也不能擅自改RT控制器。

已异步询问：是否改接18121现有ROS2 Action（推荐，不改RT）或必须等待RT提供JointTrajectory订阅。用户尚未回答此项；此前“确认”只确认左7轴回放范围。下一工程师不得将其当成允许切换接口或已完成实机执行验收。

## 验证

六项标准库单元测试通过（导出截断、仅7轴/映射、返回方向、速度/加速度/jerk、拒绝后段、状态门禁）。
实际ROS2收发在隔离Domain231完成，脚本integration_smoke.py拒绝在其他Domain运行。模拟前往、返回、途中停止再返回共5条topic消息，全部只含左7轴，最终Home误差0。第二次运行退出码0，无真实Domain77运动。
证据/tmp/left_prefix_ros_smoke_20261018.log；这不等于当前真实RT支持话题执行。

时间参数化保留每段原几何，逐记录点停稳，默认最高5°/s、10°/s²、50°/s³；L18/R22约218.6秒。禁止为加速静默简化路径。

## 重启

工控机服务已设置完整ROS环境，令牌文件会自动定位当前18121进程：

```bash
ssh ar@192.168.100.40 'systemctl --user restart alfa-left-prefix-control.service'
systemctl --user restart alfa-left-prefix-tunnel-8155.service
```

当前真实接口问题解决前，重启也不会开放发送。原18121使能和现场停止流程保持独立。
