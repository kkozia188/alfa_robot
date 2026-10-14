#!/usr/bin/env python3
"""Compile a box-wall task contract into ranked action-family candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motion_family import ContractError, compile_contract, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = compile_contract(load_json(args.tasks), load_json(args.family))
    except ContractError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = plan["summary"]
    print(
        f"compiled={summary['compiled']}/{summary['requested']} "
        f"failed={summary['failed']} output={args.output}"
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
