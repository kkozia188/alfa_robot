"""CPU check of baseline IK configuration and ownership on invalidation."""
import gc
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from curobo_core.backend import CuroboBackend


class Solver:
    def __init__(self, config):
        self.config = config
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


class IkCacheTests(unittest.TestCase):
    def test_context_replacement_releases_solver_but_same_context_reuses_it(self):
        backend = object.__new__(CuroboBackend)
        backend.snapshot = SimpleNamespace(objects=(), meshes=())
        backend.robot = {'model': 'original'}
        backend.box_fit = {}
        backend._cached_task = None
        backend._cached_solver = None
        backend._cached_solver_key = None
        backend._cached_checkers = {}
        backend.scene = lambda boxes: 'scene'
        with patch('curobo_core.backend.resource_key', return_value='scene-key'), \
             patch('curobo_core.backend.InverseKinematics', Solver), \
             patch('curobo_core.backend.InverseKinematicsCfg.create', side_effect=lambda **kw: kw):
            first = backend.ik_solver({})
            self.assertIs(backend.ik_solver({}), first)
            self.assertTrue(backend.ik_cache_hit)
            config = first.config
            self.assertEqual(config['num_seeds'], 512)
            self.assertEqual(config['override_iters_for_multi_link_ik'], 500)
            self.assertEqual(config['position_tolerance'], .002)
            self.assertEqual(config['orientation_tolerance'], math.radians(1))
            self.assertEqual(config['optimizer_collision_activation_distance'], .005)
            seen = weakref.ref(first)
            second = backend.ik_solver({}, random_seed=11)
            self.assertTrue(first.destroyed)
            self.assertFalse(backend.ik_cache_hit)
            del first
            gc.collect()
            self.assertIsNone(seen())
            self.assertIs(backend.ik_solver({}, random_seed=11), second)
            backend._release_ik_solver()
            self.assertTrue(second.destroyed)
            self.assertIsNone(backend._cached_solver_key)


if __name__ == '__main__':
    unittest.main()
