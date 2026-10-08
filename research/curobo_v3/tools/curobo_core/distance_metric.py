import math


UPDOWN_DISTANCE_WEIGHT = math.radians(15.0) / 0.1


def joint_distance_weights(joint_names, weights=None):
    names = tuple(joint_names)
    values = list(weights) if weights is not None else [1.0] * len(names)
    if len(values) != len(names):
        raise ValueError("distance weights must match joint names")
    for index, name in enumerate(names):
        if name == "updown":
            values[index] = UPDOWN_DISTANCE_WEIGHT
    return values
