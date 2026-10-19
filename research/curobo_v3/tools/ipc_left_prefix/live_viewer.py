import argparse
import json
from pathlib import Path
import time
from urllib.request import urlopen

import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--status-url',default='http://127.0.0.1:8155/api/status')
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--urdf',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8156)
    args=parser.parse_args()
    library=json.loads(args.library.read_text())
    model=yourdfpy.URDF.load(args.urdf,load_meshes=True)
    for name,link in model.link_map.items():
        if name.startswith('right_'):
            link.visuals=[]
    model._scene=model._create_scene()
    server=viser.ViserServer(host='127.0.0.1',port=args.port,label='左臂实测运动 · 只读显示')
    server.gui.configure_theme(control_width='small',show_share_button=False)
    robot=ViserUrdf(server,model,root_node_name='/feedback',mesh_color_override=(140,185,220))
    server.scene.add_grid('/ground',width=3,height=3)
    marker=server.scene.add_icosphere('/target_tcp',radius=.025,color=(70,230,140),visible=False)
    state_text=server.gui.add_markdown('等待左臂实测反馈…')
    server.gui.add_markdown('机器人运动来自实测 /joint_states，不是预设播放。绿球表示所选轨迹的接触目标TCP；仅左7轴。Updown固定-0.3m只用于显示参考，未反馈的轴不代表实机状态。')

    @server.on_client_connect
    def connected(client):
        client.camera.position=(2.3,2.7,2.1)
        client.camera.look_at=(.25,.25,1.25)
        client.camera.up_direction=(0.,0.,1.)

    last=None
    while True:
        try:
            with urlopen(args.status_url,timeout=1) as response:
                status=json.load(response)
            task=status.get('last_task') if status.get('busy') else status.get('view_task')
            task=task if task in library['tasks'] else next(iter(library['tasks']))
            degrees=status.get('feedback_deg')
            if degrees is None or status.get('feedback_age_s',999)>.5:
                state_text.content='**反馈过期：保持最后画面，不当作实时状态**'
            else:
                if task!=last:
                    row=library['tasks'][task]['model_positions'][-1]
                    mapping=dict(zip(library['joint_names'],row));mapping['updown']=-.3
                    model.update_cfg({name:mapping.get(name,0.) for name in model.actuated_joint_names})
                    marker.position=model.get_transform('left_tool0',model.base_link)[:3,3]
                    marker.visible=True;last=task
                values=dict(zip(library['joint_names'],np.deg2rad(degrees)));values['updown']=-.3
                robot.update_cfg(np.array([values.get(name,0.) for name in model.actuated_joint_names]))
                state_text.content=f"**左臂实测 · {task} · {status.get('action_phase','idle')}**\n\n反馈年龄 {status['feedback_age_s']:.3f}s\n\n{status.get('message','')}"
        except Exception as error:
            state_text.content='**后端连接中断，画面不再更新**\n\n'+str(error)
        time.sleep(.05)


if __name__=='__main__':
    main()
