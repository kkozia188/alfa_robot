import unittest

import torch

from curobo._src.graph_planner.graph.connector_linear import LinearConnector
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class ConnectorCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = PRMGraphPlannerCfg.create(robot="franka.yml", scene_model="collision_test.yml")
        cls.connector = LinearConnector(config)
        cls.connector.set_dependencies(
            action_dim=1,
            cspace_distance_weight=torch.ones(1, device="cuda"),
            check_feasibility_fn=lambda values: torch.ones(len(values), device=values.device, dtype=torch.bool),
            preallocated_idx_buffer=torch.arange(6, device="cuda", dtype=torch.int64),
        )

    def check_endpoint(self, mask, expected):
        line = torch.arange(6, device="cuda", dtype=torch.float32).reshape(1, 6, 1)
        result = self.connector._find_last_feasible_point(line, torch.tensor([mask], device="cuda"), 6)
        self.assertEqual(result[0, 0].item(), expected)

    def test_first_collision_prevents_skipping_to_later_free_region(self):
        self.check_endpoint([True, True, False, False, True, True], 1)

    def test_collision_free_edge_reaches_endpoint(self):
        self.check_endpoint([True] * 6, 5)

    def test_colliding_start_does_not_jump_to_endpoint(self):
        self.check_endpoint([False, True, True, True, True, True], 0)


if __name__ == "__main__":
    unittest.main()
