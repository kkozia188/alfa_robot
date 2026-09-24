import math
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros import Buffer, TransformListener

from robot_motion_runtime.vehicle_pose_source_node import VehiclePoseSourceNode


def test_vehicle_pose_source_publishes_composable_map_odom_base_chain():
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    source = VehiclePoseSourceNode()
    listener_node = Node("vehicle_pose_source_test_listener")
    buffer = Buffer()
    listener = TransformListener(buffer, listener_node)
    executor = SingleThreadedExecutor()
    executor.add_node(source)
    executor.add_node(listener_node)
    try:
        source.set_parameters([
            Parameter("map_to_odom_x", value=0.2),
            Parameter("map_to_odom_y", value=-0.1),
            Parameter("map_to_odom_yaw", value=0.1),
            Parameter("x", value=1.3),
            Parameter("y", value=0.4),
            Parameter("yaw", value=-0.25),
        ])
        deadline = time.monotonic() + 3.0
        transform = None
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
            if buffer.can_transform("map", "base_footprint", rclpy.time.Time()):
                transform = buffer.lookup_transform("map", "base_footprint", rclpy.time.Time())
                break
        assert transform is not None
        expected_x = 0.2 + math.cos(0.1) * 1.3 - math.sin(0.1) * 0.4
        expected_y = -0.1 + math.sin(0.1) * 1.3 + math.cos(0.1) * 0.4
        assert abs(transform.transform.translation.x - expected_x) < 1e-6
        assert abs(transform.transform.translation.y - expected_y) < 1e-6
        yaw = 2.0 * math.atan2(
            transform.transform.rotation.z, transform.transform.rotation.w
        )
        assert abs(yaw - (-0.15)) < 1e-6
    finally:
        executor.remove_node(source)
        executor.remove_node(listener_node)
        source.destroy_node()
        listener_node.destroy_node()
        del listener
        if owns_context and rclpy.ok():
            rclpy.shutdown()
