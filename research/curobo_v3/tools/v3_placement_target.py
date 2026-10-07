def placement_tool_poses():
    return {side + '_tool0': {'position': [0.6, sign * 0.425, 1.0], 'quaternion': [0.0, 1.0, 0.0, 0.0]}
            for side, sign in [('left', 1), ('right', -1)]}
