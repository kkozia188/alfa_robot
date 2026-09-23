#!/usr/bin/env python3
"""Fail-closed gate for the immutable MOTION-224 25/25 golden baseline."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tools/v3_scoop_golden_20260921/RELEASE_MANIFEST.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(release_dir, manifest):
    for asset in manifest["assets"]:
        path = release_dir / asset["name"]
        assert path.is_file(), f"missing release asset: {path}"
        assert path.stat().st_size == asset["size"], f"size mismatch: {path.name}"
        assert sha256(path) == asset["sha256"], f"SHA-256 mismatch: {path.name}"


def verify_imported_snapshot(archive):
    members = []
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            if not member.isfile():
                continue
            members.append(member.name)
            extracted = stream.extractfile(member)
            assert extracted is not None
            assert extracted.read() == (ROOT / member.name).read_bytes(), \
                f"imported source differs from verified snapshot: {member.name}"
    assert len(members) == 14, f"unexpected snapshot file count: {len(members)}"
    return members


def verify_summary(release_dir, manifest):
    summary = json.loads((release_dir / "v3-scoop-x075-sequence-summary.json").read_text())
    cache = json.loads((release_dir / "v3-scoop-x075-sequence-plan-cache.json").read_text())
    rows = list(csv.DictReader((release_dir / "v3-scoop-x075-sequence-metrics.csv").open()))
    expected = manifest["validation"]

    assert summary["schema"] == manifest["result_schema"]
    assert summary["completed_boxes"] == expected["completed_boxes"] == 25
    assert summary["validated_transitions"] == expected["validated_transitions"] == 25
    assert len(summary["boxes"]) == len(cache["tasks"]) == len(rows) == 25
    assert sorted(box["box_id"] for box in summary["boxes"]) == list(range(1, 26))
    assert all(box["validated_transition_frames"] > 0 for box in summary["boxes"])

    times = [float(row["selected_planner_s"]) for row in rows]
    performance = summary["search_performance"]
    assert sum(value <= 5.0 for value in times) == expected["core_planning_at_or_below_5s"] == 25
    for key in ("core_planning_average_s", "core_planning_median_s", "core_planning_max_s"):
        assert math.isclose(performance[key], expected[key], abs_tol=1e-12)
    assert math.isclose(statistics.fmean(times), expected["core_planning_average_s"], abs_tol=0.001)
    assert math.isclose(statistics.median(times), expected["core_planning_median_s"], abs_tol=0.001)
    assert math.isclose(max(times), expected["core_planning_max_s"], abs_tol=0.001)
    assert math.isclose(summary["total_sequence_joint_travel_deg"],
                        expected["total_sequence_joint_travel_deg"], abs_tol=1e-9)

    flips = sum(int(row["joint_flip_events"]) for row in rows)
    inverted = sum(int(box["carried_box_orientation"]["inverted_frames"])
                   for box in summary["boxes"])
    assert flips == expected["joint_flip_events"] == 0
    assert inverted == expected["carried_box_inverted_frames"] == 0

    strategy_counts = {
        "joint_shortcut": 0,
        "shortcut_repair_rrt": 0,
        "tcp_shortcut": 0,
        "joint_rrt_fallback": 0,
    }
    for box in summary["boxes"]:
        for segment in box.get("segment_search", []):
            strategy = segment.get("strategy")
            if segment.get("success") and strategy in strategy_counts:
                strategy_counts[strategy] += 1

    return {
        "completed_boxes": 25,
        "validated_transitions": 25,
        "core_planning_average_s": performance["core_planning_average_s"],
        "core_planning_median_s": performance["core_planning_median_s"],
        "core_planning_max_s": performance["core_planning_max_s"],
        "core_planning_at_or_below_5s": sum(value <= 5.0 for value in times),
        "joint_flip_events": flips,
        "carried_box_inverted_frames": inverted,
        "strategy_counts": strategy_counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    release_dir = args.release_dir.resolve()
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["tag"] == "v3-scoop-golden-2026.09.21"
    verify_assets(release_dir, manifest)
    members = verify_imported_snapshot(release_dir / "source-snapshot.tar.gz")
    report = {
        "schema": "alfa.v3_scoop_golden_25_regression_gate.v1",
        "release": manifest["tag"],
        "golden_commit": "8eba011d056dcd48f5254d08e1b35e169e5828dc",
        "source_snapshot_sha256": sha256(release_dir / "source-snapshot.tar.gz"),
        "snapshot_files": members,
        "validation": verify_summary(release_dir, manifest),
        "current_l3_claimed": False,
        "current_l4_claimed": False,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
