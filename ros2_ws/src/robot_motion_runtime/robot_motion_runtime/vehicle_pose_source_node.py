from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from robot_motion_runtime.common import RuntimeStatusPublisher


class VehiclePoseSourceNode(Node):
    """Simulation stand-in for localization and odometry TF ownership."""

    def __init__(self) -> None:
        super().__init__("vehicle_pose_source")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("map_to_odom_x", 0.0)
        self.declare_parameter("map_to_odom_y", 0.0)
        self.declare_parameter("map_to_odom_yaw", 0.0)
        self.declare_parameter("x", 0.0)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("yaw", 0.0)

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if publish_rate_hz <= 0.0:
            raise ValueError(f"publish_rate_hz must be > 0, got {publish_rate_hz}")

        self.broadcaster = TransformBroadcaster(self)
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.status = RuntimeStatusPublisher(
            self,
            "vehicle_pose_source",
            f"simulated {self.map_frame}->{self.odom_frame}->{self.base_frame}; "
            "placeholder for localization and navigation odometry",
        )
        self.status.mark_ready(f"x={self.x:.3f} y={self.y:.3f} yaw_deg={math.degrees(self.yaw):.2f}")
        self.timer = self.create_timer(1.0 / publish_rate_hz, self.publish_transform)

    @property
    def x(self) -> float:
        return float(self.get_parameter("x").value)

    @property
    def y(self) -> float:
        return float(self.get_parameter("y").value)

    @property
    def yaw(self) -> float:
        return float(self.get_parameter("yaw").value)

    def _on_set_parameters(self, params):
        from rcl_interfaces.msg import SetParametersResult

        for param in params:
            if param.name == "publish_rate_hz" and param.value <= 0.0:
                return SetParametersResult(successful=False, reason="publish_rate_hz must be > 0")
        return SetParametersResult(successful=True)

    def publish_transform(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.broadcaster.sendTransform([
            self._transform(
                stamp,
                self.map_frame,
                self.odom_frame,
                float(self.get_parameter("map_to_odom_x").value),
                float(self.get_parameter("map_to_odom_y").value),
                float(self.get_parameter("map_to_odom_yaw").value),
            ),
            self._transform(stamp, self.odom_frame, self.base_frame, self.x, self.y, self.yaw),
        ])

    @staticmethod
    def _transform(stamp, parent: str, child: str, x: float, y: float, yaw: float):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent
        transform.child_frame_id = child
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        half_yaw = yaw / 2.0
        transform.transform.rotation.z = math.sin(half_yaw)
        transform.transform.rotation.w = math.cos(half_yaw)
        return transform


def main() -> None:
    rclpy.init()
    node = VehiclePoseSourceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
