import numpy as np
import torch

from curobo_core.planner import CycleBlocked
from v3_lm_conveyor import SequentialPlacementLm
from v3_restored_analytic_cycle import RestoredAnalyticPlanner
from v3_wall_ik_benchmark import ACTIVE_JOINTS


PROFILE='20261031-analytic-extraction-rrt-transport-lm-placement-frozen40-v1'


class FrozenCyclePlanner(RestoredAnalyticPlanner):
    pipeline_label='解析抽离＋RRT运输＋连续LM放箱'
    pipeline_description='旧多候选8/16/32/64分批解析抽离35cm，多根多目标RRT运输；导航瞬移后128点15轴连续LM放箱，无RRT回退；释放后从实际放箱末态用RRT回Home。全程原40球模型。'
    conveyor_placement_description='导航到位瞬移完成；底盘冻结，使用15轴连续LM放箱，启用携箱40球与龙头车碰撞检查'
    navigation_export_enabled=False

    def __init__(self,args=None):
        super().__init__(args)
        self.placement_lm_solvers={}

    def placement_solver(self):
        sides=tuple(sorted(self.selected_task))
        if sides not in self.placement_lm_solvers:
            self.placement_lm_solvers[sides]=SequentialPlacementLm(self,seeds=32,iterations=32,samples=128,active_sides=sides)
        return self.placement_lm_solvers[sides]

    def conveyor_ik_solver(self,boxes):
        return None

    def plan_conveyor_target(self,start,targets,checker,validity,solver,report,append):
        attachments={side:np.asarray(matrix) for side,matrix in checker.task_tool_to_box.items()}
        tools=self._tool_poses(checker,torch.tensor(np.asarray(start),device='cuda',dtype=torch.float32))
        case=dict(start=np.asarray(start,dtype=np.float32),checker=checker,attachments=attachments,
            starts={side:tools[side]@matrix for side,matrix in attachments.items()},
            ends={side:targets[side]@matrix for side,matrix in attachments.items()})
        engine=self.placement_solver()
        result=engine.plan(case)
        report['conveyor_place_mode']='continuous15axis LM, box-center line and orientation interpolation, free terminal joints, no RRT fallback'
        report['conveyor_place_lm']={key:value for key,value in result.items() if key!='frames'}
        report['conveyor_place_lm']['solver_setup_ms']=engine.setup_ms
        names=report['joint_names']
        frames=np.repeat(np.asarray(report['frames'][-1])[None],len(result['frames']),axis=0)
        frames[:,[names.index(name) for name in ACTIVE_JOINTS]]=result['frames']
        append(frames,'conveyor_place',True)
        if not result['success']:
            raise CycleBlocked('龙头车放置LM',f"进度{result['completed_progress']*100:.2f}%：{result['failure']}",report)
        return np.asarray(result['frames'][-1],dtype=np.float32)

    def full_cycle(self,boxes,progress=lambda text:None):
        result=super().full_cycle(boxes,progress)
        result.update(algorithm='frozen_cycle',cycle_profile=PROFILE,
            strategy=dict(contact='historical multi-candidate',extraction='fixed Updown/psi analytic35cm',
                transport='multi-root multi-goal RRT',placement='sequential15axis LM128points32seeds32iterations',
                return_home='RRT from actual LM terminal state',payload_model='frozen40',rrt_placement_fallback=False))
        return result
