"""CPU audit regression against this experiment's measured complete trace.

Recorded rows are used only to test the verifier, never to seed planning.
"""
import copy
import json
from pathlib import Path
import sys
import unittest

import yourdfpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from curobo_core.contracts import PlanRequest, PlannerAssets
from v3_verify_sequential import verify

FIXTURE = ROOT/'artifacts/sequential-unload/20261007/seed11-safe-parking.json'


@unittest.skipUnless(FIXTURE.exists(), 'requires the measured simulation result, not a fabricated fixture')
class IndependentResultAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        document = json.loads(FIXTURE.read_text())
        cls.request = PlanRequest.from_dict(document['request'])
        cls.result = document['results'][0]
        cls.robot = yourdfpy.URDF.load(PlannerAssets().urdf, load_meshes=False, build_scene_graph=True)

    def test_measured_complete_trace_passes_independent_fk(self):
        report = verify(self.request, self.result, self.robot)
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['release_checks']), 2)
        self.assertEqual(len(report['line_checks']), 3)

    def test_forged_release_pose_is_rejected(self):
        changed = copy.deepcopy(self.result)
        event = next(item for item in changed['predicted_scene_events'] if item['phase']=='left_release')
        event['world_pose']['position'][0] += .01
        with self.assertRaisesRegex(AssertionError, 'release not actual FK'):
            verify(self.request, changed, self.robot)

    def test_support_drift_is_rejected(self):
        changed = copy.deepcopy(self.result)
        i = changed['phases'].index('left_place')
        changed['frames'][i][8] += 2e-6
        with self.assertRaises(AssertionError):
            verify(self.request, changed, self.robot)

    def test_partial_release_cannot_pass_as_complete(self):
        changed = copy.deepcopy(self.result)
        changed['predicted_scene_events'].pop()
        with self.assertRaises(AssertionError):
            verify(self.request, changed, self.robot)


if __name__ == '__main__':
    unittest.main()
