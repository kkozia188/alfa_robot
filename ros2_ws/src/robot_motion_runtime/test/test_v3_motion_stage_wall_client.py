from robot_motion_runtime.v3_motion_stage_wall_client import pose_from_json, wall_rounds


def test_wall_rounds_cover_every_box_once():
    rounds = wall_rounds()
    assert len(rounds) == 15
    boxes = [box for pair in rounds for box in pair if box is not None]
    assert sorted(boxes) == list(range(25))
    assert rounds[:3] == [(24, 20), (23, 21), (22, None)]
    assert rounds[-3:] == [(4, 0), (3, 1), (2, None)]


def test_catalog_pose_preserves_position_and_orientation():
    pose = pose_from_json({
        "position": [0.8, -0.4, 1.2],
        "orientation": [0.0, 0.707106781, 0.0, 0.707106781],
    })
    assert (pose.position.x, pose.position.y, pose.position.z) == (0.8, -0.4, 1.2)
    assert (pose.orientation.x, pose.orientation.y, pose.orientation.z,
            pose.orientation.w) == (0.0, 0.707106781, 0.0, 0.707106781)
