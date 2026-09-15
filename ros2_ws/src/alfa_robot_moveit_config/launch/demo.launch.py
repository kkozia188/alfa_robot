import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch


def generate_launch_description():
    lock_path = Path(get_package_share_directory("alfa_robot_description")) / "config/upstream_description.lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("model_revision") != "robot_v3.1.1":
        raise RuntimeError(
            f"MoveIt demo requires robot_v3.1.1, installed description is {lock.get('model_revision')}"
        )
    moveit_config = MoveItConfigsBuilder("alfa_robot", package_name="alfa_robot_moveit_config").to_moveit_configs()
    return generate_demo_launch(moveit_config)
