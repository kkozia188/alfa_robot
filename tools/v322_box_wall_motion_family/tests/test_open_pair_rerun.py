#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from open_pair_rerun import pair_result, selected_paths  # noqa: E402


class OpenPairRerunTest(unittest.TestCase):
    def test_pair_lookup_is_unique(self) -> None:
        summary = {
            "results": [
                {"pair": [1, 6], "success": True},
                {"pair": [1, 21], "success": False},
            ]
        }
        self.assertTrue(pair_result(summary, [1, 6])["success"])
        self.assertFalse(pair_result(summary, [1, 21])["success"])
        with self.assertRaisesRegex(RuntimeError, "no unique pair"):
            pair_result(summary, [2, 3])

    def test_selected_paths_support_standard_and_rescued_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            standard = {
                "pair": [1, 10],
                "selected": {"candidate_rank": 1},
            }
            rescued = {
                "pair": [1, 6],
                "selected": {"candidate_rank": 190, "bridge_rescue_retry": 1},
            }
            self.assertEqual(
                selected_paths(root, standard),
                (
                    root / "pair-01-10/candidate-001/replay.json",
                    root / "pair-01-10/candidate-001/validation.json",
                ),
            )
            self.assertEqual(
                selected_paths(root, rescued),
                (
                    root / "pair-01-06/candidate-190/replay-rescue-1.json",
                    root / "pair-01-06/candidate-190/validation-rescue-1.json",
                ),
            )


if __name__ == "__main__":
    unittest.main()
