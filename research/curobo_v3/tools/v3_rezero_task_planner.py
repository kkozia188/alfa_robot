from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from curobo_core.contracts import PlannerAssets
from v3_adaptive_raised_cycle_planner import AdaptiveRaisedCyclePlanner
from v3_prepare_rezero_profiles import convert_row
from v3_rezero_analytic import RezeroAnalytic
from v3_zero_model_candidate import ROOT


FOLDER=ROOT/'generated/zero_model_candidate_20261018'


def migrated_assets():
    return replace(PlannerAssets(),robot_config=FOLDER/'fixed.yaml',mobile_robot_config=FOLDER/'mobile.yaml',
        urdf=FOLDER/'robot_rezero_candidate.urdf',named_poses=FOLDER/'named_poses.yaml')


class RezeroTaskPlanner(AdaptiveRaisedCyclePlanner):
    def __init__(self,args=None):
        super().__init__(args or migrated_assets())
        self.analytic=RezeroAnalytic()
        self.target_values=convert_row(['updown']+[f'{side}_joint{index}' for side in ('left','right') for index in range(1,8)],self.target_values)
        self.placement_atlas_override=FOLDER/'top_atlas.npz'

    def _atlas(self,planner):
        return FOLDER/('top_atlas.npz' if planner.top_grasp else 'side_atlas.npz')

    def full_cycle(self,boxes,progress=lambda message:None):
        self.configure_task(boxes)
        with ExitStack() as stack:
            for module in ('v3_research_assets','v3_all_extraction_roots_test','v3_side_upward_prepare','v3_lazy_extraction_planner','v3_single_arm_planner'):
                stack.enter_context(patch(module+'.atlas_path',self._atlas))
            result=super().full_cycle(boxes,progress)
        result['coordinate_model']='Rezero with legacy tool: left J2 [-90,120], right [-120,90]'
        return result


def migrate_atlases():
    names=['updown']+[f'{side}_joint{index}' for side in ('left','right') for index in range(1,8)]
    sources={'side':ROOT/'generated/selected_placement_20261013/all_solutions.npz',
             'top':ROOT/'generated/adaptive_raised_cycle_20261016/top_raised20.npz'}
    for label,source in sources.items():
        with np.load(source,allow_pickle=False) as stored:
            arrays={key:stored[key] for key in stored.files}
        arrays['joints']=convert_row(names,arrays['joints']).astype(np.float32)
        np.savez_compressed(FOLDER/(label+'_atlas.npz'),**arrays)
        print('ATLAS',label,len(arrays['joints']),flush=True)
