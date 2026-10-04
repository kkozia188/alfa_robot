import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from v3_gpu_bitstar import batched_bitstar_multi_goal


class WallValidity:
    def __init__(self, blocked):
        self.weights = torch.ones(2, device="cuda")
        self.stability_weight = 0.0
        self.blocked = blocked

    def mask(self, values):
        if not self.blocked:
            return torch.ones(len(values), device=values.device, dtype=torch.bool)
        return ~((values[:, 0].abs() < 0.15) & (values[:, 1].abs() < 0.65))

    def edges(self, starts, goals, return_stability=False):
        fractions = torch.linspace(0, 1, 121, device=starts.device)
        values = starts[:, None, :] + fractions[None, :, None] * (goals - starts)[:, None, :]
        accepted = self.mask(values.reshape(-1, 2)).reshape(len(starts), -1).all(1)
        return accepted, torch.zeros(len(starts), device=starts.device)


@unittest.skipUnless(torch.cuda.is_available(), "GPU search requires CUDA")
class BitstarTests(unittest.TestCase):
    def solve(self, validity, identical=False):
        start = torch.tensor([-1.0, 0.0], device="cuda")
        goals = start.reshape(1, -1) if identical else torch.tensor([[1.0, 0.0]], device="cuda")
        lower = torch.tensor([-1.2, -1.2], device="cuda")
        path, stats = batched_bitstar_multi_goal(start, goals, lower, -lower, validity, 2, 42)
        self.assertIsNotNone(path, stats)
        self.assertTrue(torch.allclose(torch.as_tensor(path[0], device="cuda"), start))
        self.assertTrue(torch.allclose(torch.as_tensor(path[-1], device="cuda"), goals[0]))
        self.assertTrue(validity.edges(torch.as_tensor(path[:-1], device="cuda"),
                                       torch.as_tensor(path[1:], device="cuda"))[0].all().item())
        return path, stats

    def test_wall_requires_detour(self):
        path, stats = self.solve(WallValidity(True))
        self.assertGreater(max(abs(path[:, 1])), 0.65)
        self.assertGreater(stats["edges_checked"], 0)

    def test_direct_solution_and_empty_informed_set(self):
        path, stats = self.solve(WallValidity(False))
        self.assertEqual(len(path), 2)
        self.assertAlmostEqual(stats["best_weighted_length"], 2.0)

    def test_identical_start_goal(self):
        _, stats = self.solve(WallValidity(False), identical=True)
        self.assertEqual(stats["edges_checked"], 0)
        self.assertEqual(stats["best_weighted_length"], 0.0)


if __name__ == "__main__":
    unittest.main()
