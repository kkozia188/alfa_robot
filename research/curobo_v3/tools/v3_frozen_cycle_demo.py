import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gzip
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import viser
from alfa_motion_viser import ReadOnlyScene,default_visual_assets
from curobo_core.scene import SceneSnapshot
from v3_frozen_cycle import FrozenCyclePlanner,PROFILE
from v3_current_urdf_forward import current_assets
from v3_frozen_cycle_validate import OUTPUT,save
from v3_visualization_dependency import verify_frontend_package


def main(default_port=8243):
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=default_port)
    parser.add_argument('--result',type=Path)
    assets=current_assets()
    for name in ('robot_config','mobile_robot_config','urdf','named_poses','box_fit'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,default=getattr(assets,name))
    args=parser.parse_args()
    planner=FrozenCyclePlanner(args)
    verify_frontend_package()
    server=viser.ViserServer(host='127.0.0.1',port=args.port,label='固定全流程 · 解析抽离/RRT运输/LM放箱')
    renderer=ReadOnlyScene(server,planner.args.urdf,default_visual_assets())
    server.gui.add_markdown('**已固定流程**：Home→3cm预接触→解析贴近吸附→解析直线抽离35cm→多根多目标RRT运输→导航瞬移→128点15轴连续LM放箱→释放→从实际末态RRT回Home。携箱仅用原40球模型，失败停止，不回退RRT放箱。仅离线规划与回放。')
    choices=dict(planner.task_options)
    task=server.gui.add_dropdown('任务',options=list(choices))
    run=server.gui.add_button('计算完整流程')
    play=server.gui.add_checkbox('播放完整流程',initial_value=False)
    frame=server.gui.add_slider('轨迹帧',min=0,max=1,step=1,initial_value=0)
    phase=server.gui.add_dropdown('跳转阶段',options=['home'])
    cutaway=server.gui.add_checkbox('集装箱剖视',initial_value=True)
    status=server.gui.add_markdown('选择任务后计算。')
    detail=server.gui.add_markdown('')
    state={}
    nodes=[]
    source_boxes=[]
    held={}
    for side in ('left','right'):
        held[side]=server.scene.add_frame('/held/'+side,show_axes=False,visible=False)
        renderer.carton_object('/held/'+side+'/box',(.3,.4,.4))
    planner.configure_task(choices[task.value])
    renderer.assembled_container(tuple(replace(item,pose=replace(item.pose,position=(item.pose.position[0]+planner.base_advance,*item.pose.position[1:]))) for item in planner.snapshot.objects))
    renderer.set_scene_view(False,True)
    renderer.reference_conveyors().visible=True
    renderer.apply_state(dict(planner.home,base_x=planner.base_advance,base_y=0.,base_yaw=0.))
    executor=ThreadPoolExecutor(max_workers=1)

    @server.on_client_connect
    def connect(client):
        client.camera.position=(1.,-3.2,2.1)
        client.camera.look_at=(.3,0.,1.)
        client.camera.up_direction=(0.,0.,1.)

    def update():
        if 'result' not in state:return
        result=state['result']
        index=min(int(frame.value),len(result['frames'])-1)
        values=dict(result['visual_home'])
        values.update(zip(result['joint_names'],result['frames'][index]))
        renderer.apply_state(values)
        for side,node in held.items():
            node.visible=bool(result['payload'][index]) and side in result['task']
            if node.visible:
                matrix=renderer.tool_poses[side]@np.asarray(result['visual_tool_to_box'][side])
                node.position=matrix[:3,3]
                node.wxyz=np.roll(Rotation.from_matrix(matrix[:3,:3]).as_quat(),1)
        picked=any(result['payload'][:index+1])
        for node in source_boxes:node.visible=not picked
        status.content=f"**{state['result_label']} · {result['phases'][index]} · 帧{index}/{len(result['frames'])-1}**\n\n"+('全流程通过。' if result['success'] else f"停止于{result.get('blocked_stage')}：{result.get('blocked_details')}。仅显示已通过前缀。")

    def load(result):
        if result.get('cycle_profile')!=PROFILE:raise ValueError('recording belongs to another planning profile')
        state['result']=result
        state['result_label']=next(label for label,boxes in choices.items() if boxes==result['task'])
        play.value=False
        frame.max=max(1,len(result['frames'])-1)
        frame.value=0
        phase.options=list(dict.fromkeys(result['phases']))
        phase.value=phase.options[0]
        for node in nodes:node.remove()
        nodes.clear()
        source_boxes.clear()
        snapshot=SceneSnapshot.from_dict(result['visual_scene_snapshot'])
        for item in snapshot.objects:
            if item.object_id in ('front_wall','left_wall','right_wall','ceiling') or item.object_id.startswith('conveyor_pair'):continue
            node=server.scene.add_box('/environment/'+item.object_id,dimensions=item.dimensions_m,
                position=np.asarray(item.pose.position)+[result['visual_base_advance'],0,0],wxyz=item.pose.quaternion_wxyz,color=(180,125,60),opacity=.4)
            nodes.append(node)
            if item.object_id.startswith('wall_box_'):source_boxes.append(node)
        if result.get('conveyor_box_targets'):renderer.update_conveyor_targets(result['conveyor_box_targets'])
        lm=result.get('conveyor_place_lm')
        detail.content=f"本次全流程计算{result['wall_ms']/1000:.3f}s。"
        if lm:detail.content+=f" LM放箱{lm['total_ms']/1000:.3f}s，进度{lm['completed_progress']*100:.1f}%；终点关节重新求解。"
        update()

    def compute():
        if state.get('busy'):return
        state['busy']=True
        task.disabled=True
        run.disabled=True
        play.value=False
        selected=dict(choices[task.value])
        future=executor.submit(planner.full_cycle,selected,lambda text:renderer.submit(lambda text=text:setattr(status,'content',text)))
        def complete(future):
            def finish():
                try:
                    result=future.result()
                    save(OUTPUT/'live'/f'{time.time_ns()}.json.gz',result)
                    load(result)
                except Exception as error:
                    status.content='计算错误，停止：'+str(error)
                finally:
                    state['busy']=False
                    task.disabled=False
                    run.disabled=False
            renderer.submit(finish)
        future.add_done_callback(complete)

    run.on_click(lambda event:renderer.submit(compute))
    task.on_update(lambda event:setattr(play,'value',False))
    frame.on_update(lambda event:renderer.submit(update))
    phase.on_update(lambda event:setattr(frame,'value',state['result']['phases'].index(phase.value)) if 'result' in state else None)
    cutaway.on_update(lambda event:[setattr(node,'visible',not cutaway.value) for node in renderer.container_walls])
    if args.result:
        result=json.loads(gzip.decompress(args.result.read_bytes()) if args.result.suffix=='.gz' else args.result.read_bytes())
        task.value=next(label for label,boxes in choices.items() if boxes==result['task'])
        load(result)
    print(f'Ready: http://127.0.0.1:{args.port}/',flush=True)
    tick=time.perf_counter()
    while True:
        renderer.drain()
        if play.value and 'result' in state and time.perf_counter()>=tick:
            result=state['result']
            index=int(frame.value)
            if index>=len(result['frames'])-1:play.value=False
            else:
                columns=[column for column,name in enumerate(result['joint_names']) if '_joint' in name]
                delta=max(abs(result['frames'][index+1][column]-result['frames'][index][column]) for column in columns)
                frame.value=index+1
                tick=time.perf_counter()+max(.015,delta/np.deg2rad(30))
        time.sleep(.005)


if __name__=='__main__':main()
