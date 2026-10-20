import json
from pathlib import Path

import numpy as np


POLICY = json.loads((Path(__file__).parent / 'coordinate_policy.json').read_text())
EXECUTION_REVISION = POLICY['required_coordinate_revision']
MOTOR_MINUS_MODEL_RAD = np.deg2rad(POLICY['motor_minus_model_deg'])


def motor_to_model(values):
    values = np.asarray(values, dtype=float)
    if values.shape[-1:] != (7,) or not np.isfinite(values).all():
        raise ValueError('需要七轴有限数值')
    return values - MOTOR_MINUS_MODEL_RAD


def model_to_motor(values):
    values = np.asarray(values, dtype=float)
    if values.shape[-1:] != (7,) or not np.isfinite(values).all():
        raise ValueError('需要七轴有限数值')
    return values + MOTOR_MINUS_MODEL_RAD
