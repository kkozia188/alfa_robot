import hashlib
import json
import math
from pathlib import Path


JOINT_NAMES=[f'left_joint{index}' for index in range(1,8)]
MOTOR_SIGNS=[1,1,1,1,1,1,1]
MODEL_LIMITS_DEG=[[-180,180],[-90,120],[-180,180],[-145,145],[-180,180],[-110,110],[-180,180]]


def motor_positions(model_positions):
    return [float(value)*sign for value,sign in zip(model_positions,MOTOR_SIGNS)]


def read_library(path):
    library=json.loads(Path(path).read_text())
    if library['joint_names']!=JOINT_NAMES or library['motor_signs']!=MOTOR_SIGNS:
        raise ValueError('Unsupported joint names or coordinate mapping')
    for record in library['tasks'].values():
        rows=record['model_positions']
        if len(rows)<2 or any(len(row)!=7 for row in rows):
            raise ValueError('Every recorded point must contain exactly seven left joints')
        if record['phases'][-1]!='contact_approach' or len(record['phases'])!=len(rows):
            raise ValueError('Recording must end at contact approach')
        if not set(record['phases'])<={'home','base_pick_reposition','approach','contact_approach'}:
            raise ValueError('Forbidden post-contact phase in recording')
        for row in rows:
            for value,(lower,upper) in zip(row,MODEL_LIMITS_DEG):
                if not math.isfinite(value) or not math.radians(lower)-1e-6<=value<=math.radians(upper)+1e-6:
                    raise ValueError('Recorded joint is outside model limits')
    return library


def export_prefix(source):
    result=json.loads(source.read_text())
    if not result.get('success') or not result.get('contact_roll_policy'):
        raise ValueError('Use a successful recording with the corrected contact roll policy')
    boundary=result['phases'].index('attach')
    phases=result['phases'][:boundary]
    names=result['joint_names']
    indices=[names.index(name) for name in JOINT_NAMES]
    rows=[[float(row[index]) for index in indices] for row in result['frames'][:boundary]]
    if phases[-1]!='contact_approach':
        raise ValueError('Last movement before attach is not contact approach')
    if max(abs(rows[-1][joint]-result['frames'][boundary][index]) for joint,index in enumerate(indices))>1e-6:
        raise ValueError('Recorded prefix does not reach the contact configuration')
    updown=[row[names.index('updown')] for row in result['frames'][:boundary]]
    right=[names.index(f'right_joint{index}') for index in range(1,8)]
    return dict(task=result['task'],source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                model_positions=rows,phases=phases,source_end_frame=boundary-1,
                updown_reference=dict(start_m=updown[0],end_m=updown[-1],minimum_m=min(updown),maximum_m=max(updown)),
                right_reference_max_motion_deg=math.degrees(max(abs(row[index]-result['frames'][0][index]) for row in result['frames'][:boundary] for index in right)),
                end_policy='Before attach; no vacuum, extraction, transport, placement or navigation commands')


def select_path(record,feedback,direction,tolerance_deg=.5):
    rows=[motor_positions(row) for row in record['model_positions']]
    tolerance=math.radians(tolerance_deg)
    if direction=='forward':
        if max(abs(value-reference) for value,reference in zip(feedback,rows[0]))>tolerance:
            raise ValueError('当前左臂不在记录的初始姿态，请先用原控制台调整到初始姿态')
        return [list(feedback)]+rows
    if direction!='home':
        raise ValueError('Only forward and home commands are permitted')
    distances=[max(abs(value-reference) for value,reference in zip(feedback,row)) for row in rows]
    nearest=min(range(len(rows)),key=distances.__getitem__)
    if distances[nearest]>tolerance:
        raise ValueError('当前位置偏离已记录路径，不能自动沿原轨迹返回')
    return [list(feedback)]+rows[nearest::-1]


def time_parameterize(rows,speed_deg_s=5.,acceleration_deg_s2=10.,jerk_deg_s3=50.,sample_period=.02):
    if not .1<=speed_deg_s<=10.:
        raise ValueError('Speed must be between 0.1 and 10 deg/s')
    velocity=math.radians(speed_deg_s)
    acceleration=math.radians(acceleration_deg_s2)
    jerk=math.radians(jerk_deg_s3)
    if acceleration<=0 or jerk<=0 or sample_period<=0:
        raise ValueError('Timing limits must be positive')
    points=[dict(time_s=.1,positions=list(rows[0]),velocities=[0.]*7,accelerations=[0.]*7)]
    elapsed=.1
    for start,end in zip(rows,rows[1:]):
        delta=[target-origin for origin,target in zip(start,end)]
        distance=max(abs(value) for value in delta)
        if distance<1e-10:
            continue
        duration=max(1.875*distance/velocity,math.sqrt(5.773503*distance/acceleration),(60*distance/jerk)**(1/3),sample_period)
        count=math.ceil(duration/sample_period)
        for step in range(1,count+1):
            fraction=step/count
            progress=10*fraction**3-15*fraction**4+6*fraction**5
            rate=(30*fraction**2-60*fraction**3+30*fraction**4)/duration
            change=(60*fraction-180*fraction**2+120*fraction**3)/duration**2
            points.append(dict(time_s=elapsed+duration*fraction,
                positions=[origin+value*progress for origin,value in zip(start,delta)],
                velocities=[value*rate for value in delta],accelerations=[value*change for value in delta]))
        elapsed+=duration
    return points
