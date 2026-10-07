import numpy as np


def wall_center(front, distance, box_id):
    return np.array([front + distance + 0.15, (box_id % 5 - 2) * 0.41,
                     0.20 + (box_id // 5) * 0.41])


def row_obstacles(front, boxes):
    centers = {side: wall_center(front, 0.9, box_id) for side, box_id in boxes.items()}
    for center in centers.values():
        center[2] += 0.001
    lowest = min(centers.values(), key=lambda center: center[2])
    intervals = [(-1.02, 1.02)]
    for center in centers.values():
        if abs(center[2] - lowest[2]) > 1e-6:
            continue
        remaining = []
        cut_lower, cut_upper = center[1] - 0.2, center[1] + 0.2
        for lower, upper in intervals:
            if cut_upper <= lower or cut_lower >= upper:
                remaining.append((lower, upper))
            else:
                if cut_lower > lower:
                    remaining.append((lower, cut_lower))
                if cut_upper < upper:
                    remaining.append((cut_upper, upper))
        intervals = remaining
    obstacles = [dict(center=[lowest[0], (lower + upper) / 2, lowest[2]],
                      dimensions=[0.3, upper - lower, 0.4]) for lower, upper in intervals if upper - lower > 1e-8]
    bottom_height = lowest[2] - 0.2
    if bottom_height - 0.001 > 1e-8:
        obstacles.append(dict(center=[lowest[0], 0.0, (bottom_height + 0.001) / 2], dimensions=[0.3, 2.04, bottom_height - 0.001]))
    return centers, obstacles


def grasp_transform(side, top):
    if not top:
        matrix = np.eye(4)
        matrix[:3, :3] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
        matrix[2, 3] = 0.15
        return matrix
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
    matrix[2, 3] = 0.2
    return matrix
