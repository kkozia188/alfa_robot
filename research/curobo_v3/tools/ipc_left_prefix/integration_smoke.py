import json
import math
import os
import threading
import time

from model import JOINT_NAMES
from server import State,attach_ros,detach_ros


def main():
    if os.environ.get('ROS_DOMAIN_ID')!='231':
        raise RuntimeError('Isolation required: ROS_DOMAIN_ID must be 231; no test publication permitted otherwise')
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from diagnostic_msgs.msg import DiagnosticArray,DiagnosticStatus,KeyValue
    from trajectory_msgs.msg import JointTrajectory
    from controller_manager_msgs.srv import ListControllers
    from controller_manager_msgs.msg import ControllerState

    home=[math.radians(value) for value in [65,-75,20,-90,0,40,90]]
    middle=home.copy();middle[0]+=math.radians(.1)
    target=home.copy();target[0]+=math.radians(.2)
    record=dict(model_positions=[home,middle,target],phases=['home','approach','contact_approach'])
    state=State(dict(tasks=dict(sample=record)),'left7_joint_replay')
    state.lease_client=object();state.motor_bounds=[(-math.pi,math.pi)]*7
    backend=attach_ros(state,'/left_arm_jtc/joint_trajectory')
    fixture=Node('isolated_prefix_mock_robot')
    feedback=fixture.create_publisher(JointState,'/joint_states',10)
    diagnostics=fixture.create_publisher(DiagnosticArray,'/diagnostics',10)
    robot={'positions':home.copy(),'deadline':0.,'target':home.copy()}
    messages=[]

    def receive(message):
        assert list(message.joint_names)==JOINT_NAMES
        assert all(len(point.positions)==len(point.velocities)==len(point.accelerations)==7 for point in message.points)
        durations=[point.time_from_start.sec+point.time_from_start.nanosec*1e-9 for point in message.points]
        assert all(end>start for start,end in zip(durations,durations[1:]))
        assert all(abs(value)<=math.radians(5.)+1e-7 for point in message.points for value in point.velocities)
        robot['deadline']=time.monotonic()+durations[-1]
        robot['target']=list(message.points[-1].positions)
        messages.append(dict(names=list(message.joint_names),points=len(message.points),duration_s=durations[-1],target=robot['target']))

    def controllers(_request,response):
        response.controller=[ControllerState(name='left_arm_jtc',state='active')]
        return response

    def tick():
        if time.monotonic()>=robot['deadline']:
            robot['positions']=robot['target'].copy()
        message=JointState();message.header.stamp=fixture.get_clock().now().to_msg()
        message.name=JOINT_NAMES;message.position=robot['positions'];feedback.publish(message)
        diagnostics.publish(DiagnosticArray(status=[DiagnosticStatus(name='/robot/rt_control/enable_manager',values=[KeyValue(key='state',value='ENABLED')])]))
        state.console_status={'lease':'you'};state.console_received=time.monotonic()

    fixture.create_subscription(JointTrajectory,'/left_arm_jtc/joint_trajectory',receive,10)
    fixture.create_service(ListControllers,'/controller_manager/list_controllers',controllers)
    fixture.create_timer(.02,tick)
    executor=SingleThreadedExecutor();executor.add_node(fixture)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()

    def wait_ready():
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            if not state.reason():
                return
            time.sleep(.03)
        raise AssertionError(state.status())

    try:
        wait_ready();state.send('sample','forward',5.);wait_ready()
        assert max(abs(a-b) for a,b in zip(state.positions,target))<1e-8
        state.send('sample','home',5.);wait_ready()
        assert max(abs(a-b) for a,b in zip(state.positions,home))<1e-8
        state.send('sample','forward',5.)
        time.sleep(.15)
        stopped=state.stop();assert stopped['stopped'];wait_ready()
        assert max(abs(a-b) for a,b in zip(state.positions,middle))<1e-8
        state.send('sample','home',5.);wait_ready()
        assert max(abs(a-b) for a,b in zip(state.positions,home))<1e-8
        assert len(messages)==5,len(messages)
        print(json.dumps(dict(success=True,domain=231,messages=messages,final_home_max_error=0.),ensure_ascii=False),flush=True)
    finally:
        executor.shutdown();thread.join(timeout=3);fixture.destroy_node();detach_ros(backend)


if __name__=='__main__':
    main()
