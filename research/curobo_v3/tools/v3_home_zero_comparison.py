import argparse
import json
from pathlib import Path
import queue
import time

import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf
import yaml
import yourdfpy

from v3_zero_model_candidate import ROOT,OLD,OFFSETS_DEG


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    home=yaml.safe_load((ROOT/'models/robot_description_analytic_071cb95/config/named_poses_analytic_proxy.yaml').read_text())['named_poses']['home']
    original_home={name:float(value) for name,value in home.items()}
    converted=original_home.copy()
    for side in ('left','right'):
        names=[f'{side}_joint{index}' for index in range(1,8)]
        values=(np.asarray([home[name] for name in names])-np.deg2rad(OFFSETS_DEG[side]))*np.array([1,-1,1,1,1,1,1])
        values=(values+np.pi)%(2*np.pi)-np.pi
        converted.update(dict(zip(names,values)))
    server=viser.ViserServer(host='127.0.0.1',port=8152,label='原Home vs 自制重零URDF · 不夹限位的姿态核对')
    paths={'old':OLD,'candidate':ROOT/'generated/zero_model_candidate_20261018/robot_rezero_candidate.urdf'}
    models={};robots={};roots={};markers=[];measurements={}
    for label,path in paths.items():
        model=yourdfpy.URDF.load(path,load_meshes=True,build_scene_graph=True,filename_handler=lambda fname,path=path:str(path.parent/fname) if not Path(fname).is_absolute() else fname)
        models[label]=model;roots[label]=server.scene.add_frame('/'+label,show_axes=False)
        robots[label]=ViserUrdf(server,model,root_node_name='/'+label,mesh_color_override=(70,150,245) if label=='old' else (245,150,40))
    mode=server.gui.add_dropdown('显示',options=['同footprint重合','并排比较','只看自制新URDF'],initial_value='并排比较')
    preset=server.gui.add_dropdown('自制新URDF姿态',options=['原Home严格等效换算（诊断）','尝试左J2=-75°/右J2=+75°','尝试左J2=+75°/右J2=-75°','自制新零位：双臂全0'],initial_value='原Home严格等效换算（诊断）')
    edits=queue.Queue();controls={}
    for side in ('left','right'):
        with server.gui.add_folder('自制新URDF '+side+' 七轴角度'):
            for index in range(1,8):
                name=f'{side}_joint{index}'
                slider=server.gui.add_slider(name+' °',min=-180.,max=180.,step=.1,initial_value=float(np.rad2deg(converted[name])))
                number=server.gui.add_number(name+' 精确°',initial_value=slider.value,step=.1)
                controls[name]=(slider,number)
                def slide(_event,name=name):edits.put((name,controls[name][0].value))
                def enter(_event,name=name):edits.put((name,controls[name][1].value))
                slider.on_update(slide);number.on_update(enter)
    server.gui.add_markdown('蓝色：原URDF文件内Home；橙色：我们自己的重零候选URDF，不是同事3.3.1。默认两者应重现同一实体姿态。滑槽允许超出候选限位供诊断，超限会明确显示；不修改URDF限制，不做规划或实机控制。Updown都固定-0.3m。')
    status=server.gui.add_markdown('')

    def render():
        for node in markers:node.remove()
        markers.clear()
        poses={}
        violations=[]
        output=[]
        for label,model in models.items():
            values={name:original_home.get(name,0.) for name in model.actuated_joint_names}
            if label=='candidate':values.update({name:np.deg2rad(slider.value) for name,(slider,number) in controls.items()})
            robots[label].update_cfg(np.asarray([values[name] for name in model.actuated_joint_names]))
            offset=np.array([0.,1.2 if label=='old' else -1.2,0.]) if mode.value=='并排比较' else np.zeros(3)
            roots[label].position=offset;roots[label].visible=not(mode.value=='只看自制新URDF' and label=='old')
            poses[label]={}
            for side in ('left','right'):
                tool=model.get_transform(side+'_tool0',model.base_link)
                poses[label][side]=tool.copy()
                if roots[label].visible:
                    markers.append(server.scene.add_frame(f'/markers/{label}_{side}_tcp',position=tool[:3,3]+offset,wxyz=np.roll(Rotation.from_matrix(tool[:3,:3]).as_quat(),1),axes_length=.16,axes_radius=.003))
            if label=='candidate':
                for name,(slider,number) in controls.items():
                    joint=model.joint_map[name]
                    if values[name]<joint.limit.lower-1e-7 or values[name]>joint.limit.upper+1e-7:violations.append(name+f'={slider.value:+.2f}°')
            output.append(f"**{'旧Home' if label=='old' else '自制新URDF'}** 左臂：`{np.round(np.rad2deg([values[f'left_joint{i}'] for i in range(1,8)]),2).tolist()}`°；右臂：`{np.round(np.rad2deg([values[f'right_joint{i}'] for i in range(1,8)]),2).tolist()}`°。")
        metrics={}
        for side in ('left','right'):
            distance=float(np.linalg.norm(poses['old'][side][:3,3]-poses['candidate'][side][:3,3])*1000)
            angle=float(np.rad2deg((Rotation.from_matrix(poses['old'][side][:3,:3]).inv()*Rotation.from_matrix(poses['candidate'][side][:3,:3])).magnitude()))
            metrics[side]=dict(tool_position_difference_mm=distance,tool_rotation_difference_deg=angle)
            output.append(f'{side} tool0相对旧Home：位置差 **{distance:.6f}mm**，朝向差 **{angle:.6f}°**。')
        output.append('**诊断超限：'+', '.join(violations)+'；仅展示，不能执行。**' if violations else '候选关节值在其URDF限位内；未验碰，不代表可执行。')
        status.content='\n\n'.join(output)
        measurements[preset.value]=dict(metrics=metrics,violations=violations,angles_deg={name:slider.value for name,(slider,number) in controls.items()})

    def set_preset(value):
        angles={name:float(np.rad2deg(converted[name])) for name in controls}
        if value.startswith('尝试左J2=-75'):angles.update(left_joint2=-75.,right_joint2=75.)
        elif value.startswith('尝试左J2=+75'):angles.update(left_joint2=75.,right_joint2=-75.)
        elif value.startswith('自制新零位'):angles={name:0. for name in controls}
        for name,angle in angles.items():
            controls[name][0].value=angle;controls[name][1].value=angle

    @preset.on_update
    def preset_changed(_event):edits.put(('preset',preset.value))
    @mode.on_update
    def mode_changed(_event):edits.put((None,None))
    @server.on_client_connect
    def connected(client):
        client.camera.position=(4.,-4.,2.6);client.camera.look_at=(0.,0.,1.1);client.camera.up_direction=(0.,0.,1.)
    render()
    if args.smoke_test:
        for value in preset.options:
            preset.value=value;set_preset(value);render()
        target=Path('/tmp/v331_model_compare_20261018/home_reproduction.json')
        target.write_text(json.dumps(measurements,ensure_ascii=False,indent=2))
        exact=measurements['原Home严格等效换算（诊断）']['metrics']
        assert all(value['tool_position_difference_mm']<.001 and value['tool_rotation_difference_deg']<.001 for value in exact.values())
        print('SMOKE PASS: all four presets, exact Home FK match',flush=True);server.stop();return
    print('Ready http://127.0.0.1:8152 · original vs our re-zero Home diagnostic',flush=True)
    while True:
        changed=False
        while not edits.empty():
            name,value=edits.get_nowait()
            if name=='preset':set_preset(value)
            elif name:
                slider,number=controls[name];value=float(np.clip(value,-180,180))
                if slider.value!=value:slider.value=value
                if number.value!=value:number.value=value
            changed=True
        if changed:render()
        time.sleep(.03)


if __name__=='__main__':main()
