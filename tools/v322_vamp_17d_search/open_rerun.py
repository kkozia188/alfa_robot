#!/usr/bin/python3
"""Verify and open the mandatory-search safe-turn Rerun."""

from __future__ import annotations

import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
RRD = WORKSPACE / (
    "results/v322_vamp_17d_search/search-production-safe-turn.rrd"
)


def main() -> int:
    subprocess.run(["rerun", "rrd", "verify", str(RRD)], check=True)
    return subprocess.run(["rerun", "--new", str(RRD)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
