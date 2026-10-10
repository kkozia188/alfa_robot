#!/usr/bin/env python3
"""Reapply the current normalized failure taxonomy to existing result files."""

from __future__ import annotations

import argparse
from pathlib import Path

from matrix_common import classify_failure, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    changed = 0
    for name in ("probe-result.json", "case-result.json"):
        for path in args.results_root.resolve().glob(f"**/{name}"):
            result = read_json(path)
            if not result.get("failure_stage") and not result.get("failure_reason"):
                continue
            updated = classify_failure(
                str(result.get("failure_stage", "")),
                str(result.get("failure_reason", "")),
            )
            if updated != result.get("failure_class"):
                result["failure_class"] = updated
                write_json(path, result)
                changed += 1
    print(f"RECLASSIFIED changed={changed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
