from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from probe_touch_motion import interpolate_motion
from probe_preview_state import activate_preview, quick_send_ready


class QuickPreviewStateTest(unittest.TestCase):
    def payload(self):
        points, report = interpolate_motion(np.zeros(7), np.ones(7) * .05)
        return dict(points=points, report=report, quick_ik_ms=1.)

    def test_quick_result_automatically_plays_from_start(self):
        state = dict(payload=None, curves=None, seen=False)
        timeline = SimpleNamespace(max=1., value=.5)
        play = SimpleNamespace(value=False)
        payload = self.payload()
        activate_preview(state, payload, timeline, play, 'quick_preview')
        self.assertTrue(play.value)
        self.assertEqual(timeline.value, 0.)
        self.assertIs(state['payload'], payload)
        np.testing.assert_allclose([curve(timeline.max) for curve in state['curves']], payload['points'][-1]['positions'])

    def test_no_quick_send_before_result_even_when_controller_ready(self):
        state = dict(payload=None, curves=None, status=dict(ready=True), future=None)
        self.assertFalse(quick_send_ready(state, True))

    def test_send_requires_current_quick_result_confirmation_and_no_computation(self):
        state = dict(payload=self.payload(), curves=[object()], status=dict(ready=True), future=None)
        self.assertTrue(quick_send_ready(state, True))
        self.assertFalse(quick_send_ready(state, False))
        state['future'] = object()
        self.assertFalse(quick_send_ready(state, True))
        state['future'] = None
        state['payload'] = dict(points=[], report={})
        self.assertFalse(quick_send_ready(state, True))


if __name__ == '__main__':
    unittest.main()
