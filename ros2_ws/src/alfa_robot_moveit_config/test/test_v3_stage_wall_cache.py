#!/usr/bin/env python3
import json
import math
from pathlib import Path


CACHE = Path(__file__).parents[1] / "config" / "v3_stage_wall_cache.json"


def test_verified_cache_is_continuous_and_bounded():
    data = json.loads(CACHE.read_text())
    assert data["schema_version"] == 1
    assert data["kind"] == "v3_fixed_wall_motion_stage_cache"
    assert len(data["joint_names"]) == 17
    successful = [entry for entry in data["entries"] if entry["success"]]
    assert [entry["round"] for entry in successful] == [1, 2, 4, 5]
    prior = None
    for entry in successful:
        combined = []
        for stage in ("pregrasp", "approach", "place", "home"):
            frames = entry[stage]
            assert frames
            if combined:
                assert frames[0]["joints"] == combined[-1]["joints"]
            combined.extend(frames)
        if prior is not None:
            assert combined[0]["joints"] == prior
        for first, last in zip(combined, combined[1:]):
            assert max(abs(a - b) for a, b in zip(
                first["joints"][:14], last["joints"][:14])) <= math.radians(0.5) + 1e-9
            assert abs(first["joints"][14] - last["joints"][14]) <= 0.005 + 1e-9
        prior = combined[-1]["joints"]


def test_failed_entries_do_not_contain_executable_frames():
    data = json.loads(CACHE.read_text())
    for entry in data["entries"]:
        if entry["success"]:
            continue
        assert "failure" in entry
        for stage in ("pregrasp", "approach", "place", "home"):
            assert stage not in entry
