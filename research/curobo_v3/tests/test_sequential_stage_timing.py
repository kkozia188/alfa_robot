"""Stage clocks are measured, additive across revisits, and never inferred from playback."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from curobo_core.sequential import _SequentialTask
from v3_stage_timing import STAGES, stage_label, stage_method, timing_markdown


class StageTimingTests(unittest.TestCase):
    def test_clock_partitions_wall_time_and_keeps_collision_nested(self):
        task=object.__new__(_SequentialTask)
        task.torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))
        task.started=task.stage_started=0.
        task.stage_collision_checkpoint=0.
        task.phase='initialize'
        checker=SimpleNamespace(collision_ms=0.)
        task.validities=[checker]
        task.progress=lambda phase:None
        task.report={'stage_timing_ms':{},'stage_breakdown_ms':{},'stage_timing_spans':[]}
        with patch('curobo_core.sequential.time.perf_counter',side_effect=[1.,3.,5.,7.]):
            task.enter('right_contact')
            checker.collision_ms=120.
            task.record_operation_time('cartesian',1.)
            task.enter('initialize')
            task.close_stage_timing()
        self.assertEqual(task.report['stage_timing_ms'],{'initialize':3000.,'right_contact':4000.})
        self.assertEqual(sum(task.report['stage_timing_ms'].values()),7000.)
        self.assertEqual(task.report['stage_breakdown_ms']['right_contact'],{'cartesian':2000.,'collision':120.})
        self.assertEqual([r['start_ms'] for r in task.report['stage_timing_spans']],[0.,1000.,5000.])

    def test_legacy_results_show_missing_not_invented_stage_totals(self):
        result={'total_ms':60000.,'phases':['left_place'], 'frames':[[0.]*15]*100,
                'attempts':[{'stage':'left_place','operation':'rrtconnect','stats':{'search_ms':2000.}}],
                'timing_ms':{'search':2000.,'collision':10000.}}
        text=timing_markdown(result)
        self.assertIn('60.000 秒',text)
        self.assertIn('旧结果未记录逐阶段总耗时',text)
        self.assertIn('| 下箱搬运至左后卸载区 | 未记录 |',text)
        self.assertIn('| 下箱搬运至左后卸载区 | 2.000 |',text)
        self.assertNotIn('| 下箱搬运至左后卸载区 | 60.000 |',text)

    def test_restored_methods_and_ik_costs_do_not_relabel_history(self):
        current = {'cartesian_strategy': 'f044_fixed_swivel', 'phases': ['left_extract'],
                   'attempts': [{'stage': 'left_extract', 'initialization_ms': 2.,
                                 'solve_ms': 40., 'filter_ms': 1., 'cache_hit': True}]}
        text = timing_markdown(current)
        self.assertIn('终点引导选解析分支 + 正向固定冗余角 IK', text)
        self.assertNotIn('逆向解析 IK', text)
        self.assertIn('| 0.002 | 0.040 | 0.001 | 是 |', text)
        self.assertIn('逆向解析 IK', stage_method('left_extract'))

    def test_chinese_labels_and_actual_methods_are_explicit(self):
        self.assertEqual(stage_label('right_lower'),'上箱竖直下降至下层')
        self.assertIn('无 RRT',stage_method('right_lower'))
        self.assertIn('RRTConnect',stage_method('left_place'))
        self.assertIn('解析 IK',stage_method('left_extract'))
        self.assertEqual(len({label for label,_ in STAGES.values()}),len(STAGES))
        text=timing_markdown({'total_ms':5000.,'phases':['left_place'],
            'stage_timing_ms':{'left_place':5000.},
            'stage_breakdown_ms':{'left_place':{'ik':1000.,'search':2000.,'audit':1000.,'collision':500.}}})
        self.assertIn('| 下箱搬运至左后卸载区 | 5.000 |',text)
        self.assertIn('| 下箱搬运至左后卸载区 | 1.000 | 2.000 | 0.000 | 1.000 | 1.000 |',text)
        self.assertIn('嵌套子项，不可再相加',text)
        self.assertIn('| 下箱搬运至左后卸载区 | 0.500 |',text)


if __name__=='__main__':
    unittest.main()
