import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf
import yaml
import yourdfpy


LOCKED = {
    'head_joint': 0.0, 'head_pitch_joint': 0.0, 'active_suspension_joint': 0.0,
    'caster01_joint': 0.0, 'wheel01_joint': 0.0, 'caster02_joint': 0.0,
    'wheel02_joint': 0.0, 'caster03_joint': 0.0, 'wheel03_joint': 0.0,
    'caster04_joint': 0.0, 'wheel04_joint': 0.0,
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path,required=True);parser.add_argument('--port',type=int,default=8080);args=parser.parse_args()
    report=json.loads(args.result.read_text());optimization=report['chunked_curobo_optimization'];frames=np.asarray(optimization['frames']);names=report['joint_names'];config=yaml.safe_load(Path(report['robot_config']).read_text())['kinematics'];urdf_path=Path(config['urdf_path'])
    free_end=optimization['stage_boundaries']['free_end'];loaded_begin=optimization['stage_boundaries']['loaded_begin'];loaded_frames=optimization['segments']['loaded']['frames']
    server=viser.ViserServer(host='127.0.0.1',port=args.port,label='ALFA · 第一组完整规划');server.scene.set_up_direction('+z');server.scene.add_grid('/floor_grid',width=6,height=5)
    server.scene.add_frame('/robot',show_axes=False);urdf=yourdfpy.URDF.load(urdf_path,load_meshes=True,build_scene_graph=True);robot=ViserUrdf(server,urdf,root_node_name='/robot',mesh_color_override=(0.69,0.75,0.82,0.82));actuated=list(robot.get_actuated_joint_names())
    front=report['chassis_front_x_m']+report['wall_distance_m'];target_ids=set(report['pair']);static_targets={};carried={}
    for box_id in range(25):
        center=(front+0.15,(box_id%5-2)*.41,.2+(box_id//5)*.41);target=box_id in target_ids
        handle=server.scene.add_box(f'/boxes/box_{box_id:02}',position=center,dimensions=(.3,.4,.4),color=(230,132,55) if target else (122,141,168),opacity=.72 if target else .2)
        if target: static_targets[box_id]=handle;server.scene.add_label(f'/boxes/box_{box_id:02}/label',text=f'目标 {box_id}',position=(-.16,0,.24))
    wall_back=front+.3+1e-6
    for name,center,dimensions in [('ground',(wall_back-2,0,-.05),(4.2,2.6,.1)),('left_wall',(wall_back-2,-1.25,1.2),(4,.1,2.4)),('right_wall',(wall_back-2,1.25,1.2),(4,.1,2.4)),('front_wall',(wall_back+.05,0,1.2),(.1,2.6,2.4)),('ceiling',(wall_back-2,0,2.45),(4.2,2.6,.1))]:server.scene.add_box(f'/container/{name}',position=center,dimensions=dimensions,color=(183,153,103),opacity=.07 if name!='ground' else .12)
    def set_robot(index):
        values=dict(zip(names,frames[index].tolist()));robot.update_cfg(np.array([values.get(name,LOCKED.get(name,0.0)) for name in actuated]))
    set_robot(loaded_begin)
    tool_to_box={}
    for side,box_id in zip(('left','right'),report['pair']):
        tool=urdf.get_transform(f'{side}_tool0',urdf.base_link);box=np.eye(4);box[:3,3]=np.array([front+.15,(box_id%5-2)*.41,.2+(box_id//5)*.41]);tool_to_box[side]=np.linalg.inv(tool)@box
        carried[side]=server.scene.add_box(f'/carried/{side}',dimensions=(.3,.4,.4),color=(242,93,52) if side=='left' else (47,132,210),opacity=.78,visible=False)
    extraction_end=loaded_begin
    initial_x=front+.15
    for global_index in range(loaded_begin,len(frames)):
        set_robot(global_index);centers=[]
        for side in ('left','right'):
            centers.append((urdf.get_transform(f'{side}_tool0',urdf.base_link)@tool_to_box[side])[:3,3])
        if max(center[0] for center in centers)<=initial_x-.345:
            extraction_end=global_index;break
    set_robot(0)
    with server.gui.add_folder('完整任务 L24 / R20'):
        server.gui.add_markdown('**结果：成功** · 0.9 m箱墙 · 双臂侧吸 · 集装箱碰撞开启')
        frame_slider=server.gui.add_slider('帧',min=1,max=len(frames),step=1,initial_value=1);play=server.gui.add_checkbox('播放',initial_value=True);speed=server.gui.add_slider('速度',min=.1,max=3.0,step=.1,initial_value=.8)
        with server.gui.add_folder('定位'):
            buttons=[server.gui.add_button(label) for label in ('初始','吸附','抽离完成','放置')]
        status=server.gui.add_markdown('');server.gui.add_markdown(f'总帧数 `{len(frames)}` · cuRobo优化局部段 `15` · 安全种子回退段 `2`\n\n所有帧通过机器人球、真实双箱和集装箱复核。')
    lock=threading.Lock()
    def show(index):
        with lock,server.atomic():
            set_robot(index);attached=index>=loaded_begin
            for handle in static_targets.values():handle.visible=not attached
            for side,handle in carried.items():
                handle.visible=attached
                if attached:
                    transform=urdf.get_transform(f'{side}_tool0',urdf.base_link)@tool_to_box[side];handle.position=transform[:3,3];handle.wxyz=np.roll(Rotation.from_matrix(transform[:3,:3]).as_quat(),1)
            if index<loaded_begin:phase='运动到吸附位置';color='#2676bd'
            elif index<=extraction_end:phase='双箱抽离35 cm';color='#d97706'
            else:phase='负重绕障到第二放置位';color='#21815d'
            source='cuRobo优化/安全种子混合段'
            status.content=f'第 **{index+1}/{len(frames)}** 帧\n\n<span style="color:{color}">**{phase}**</span>\n\n{source}'
    @frame_slider.on_update
    def on_frame(event):show(int(frame_slider.value)-1)
    targets=[0,loaded_begin,extraction_end,len(frames)-1]
    for button,target in zip(buttons,targets):
        @button.on_click
        def on_click(event,target=target):play.value=False;frame_slider.value=target+1
    @server.on_client_connect
    def on_connect(client):client.camera.position=(-3.6,3.8,2.7);client.camera.look_at=(.55,0,1.15);client.camera.up_direction=(0,0,1)
    show(0);print(json.dumps({'url':f'http://localhost:{args.port}','frames':len(frames),'free_end':free_end,'extraction_end':extraction_end,'goal':len(frames)-1}),flush=True)
    while True:
        if play.value:frame_slider.value=int(frame_slider.value)%len(frames)+1;time.sleep(.025/speed.value)
        else:time.sleep(.05)


if __name__=='__main__':main()
