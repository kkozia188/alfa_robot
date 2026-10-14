#!/usr/bin/python3
"""Record pose-driven action-family selections and certified metrics to Rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import rerun as rr
import rerun.blueprint as rrb


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-markdown", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    report = read_json(args.report)
    wall = plan["wall"]
    size = [float(value) for value in wall["box_size_m"]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rr.init("v322_box_wall_motion_family", spawn=False)
    rr.save(args.output)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", contents=["/world/**"], name="Pose-driven actions"),
            rrb.Vertical(
                rrb.TextDocumentView(origin="/status", name="Selected action"),
                rrb.BarChartView(origin="/planning/representative_ms", name="Representative planning ms"),
                rrb.TextDocumentView(origin="/report", name="Certified baseline report"),
                row_shares=[0.22, 0.20, 0.58],
            ),
            column_shares=[0.66, 0.34],
        ),
        collapse_panels=True,
    ))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "report",
        rr.TextDocument(
            args.report_markdown.read_text(encoding="utf-8"),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )
    representatives = report["representative_tasks"]
    rr.log(
        "planning/representative_ms",
        rr.BarChart(
            [float(item["planning_time_ms"]) for item in representatives],
            abscissa=[int(item["row_from_top"]) for item in representatives],
            widths=[0.65] * len(representatives),
            color=[30, 145, 205, 230],
        ),
        static=True,
    )
    centers = wall_centers(wall)
    rr.log(
        "world/box_wall",
        rr.Boxes3D(
            centers=centers,
            half_sizes=[[0.5 * value for value in size]],
            colors=[[170, 176, 182, 105]],
            show_labels=False,
        ),
        static=True,
    )

    evidence = {item["request_id"]: item for item in representatives}
    for tick, task in enumerate(plan["tasks"]):
        rr.set_time("representative", sequence=tick)
        candidate = task["selected_candidate"]
        box_position = [float(value) for value in task["box_pose"]["position_m"]]
        contact = [float(value) for value in candidate["contact_pose"]["position_m"]]
        normal = [0.0, 0.0, -1.0] if candidate["grasp_mode"] == "top_suction" else [1.0, 0.0, 0.0]
        precontact = [
            value - normal[index] * float(candidate["approach_distance_m"])
            for index, value in enumerate(contact)
        ]
        retreat = [
            value - normal[index] * float(candidate["retreat_distance_m"])
            for index, value in enumerate(contact)
        ]
        rr.log("world/selected", rr.Clear(recursive=True))
        rr.log(
            "world/selected/box",
            rr.Boxes3D(
                centers=[box_position],
                half_sizes=[[0.5 * value for value in size]],
                colors=[[35, 185, 105, 235] if candidate["arm"] == "left" else [35, 145, 225, 235]],
                labels=[task["request_id"]],
                show_labels=True,
            ),
        )
        rr.log(
            "world/selected/approach",
            rr.Arrows3D(
                origins=[precontact],
                vectors=[[contact[index] - precontact[index] for index in range(3)]],
                colors=[[40, 205, 115, 255]],
                radii=[0.012],
                labels=["pregrasp -> contact"],
            ),
        )
        rr.log(
            "world/selected/retreat",
            rr.Arrows3D(
                origins=[contact],
                vectors=[[retreat[index] - contact[index] for index in range(3)]],
                colors=[[235, 155, 40, 255]],
                radii=[0.012],
                labels=["loaded retreat"],
            ),
        )
        item = evidence[task["request_id"]]
        status = "\n".join([
            f"# {task['request_id']}",
            "",
            f"- Row/column: {task['derived_geometry']['row_from_top']} / {task['derived_geometry']['column_from_left']}",
            f"- Template: `{candidate['template']}`",
            f"- Arm / lift: `{candidate['arm']}` / `{candidate['updown_m']:.2f} m`",
            f"- Certified planning time: `{float(item['planning_time_ms']):.3f} ms`",
            f"- Pose-matched FCL evidence: `{'PASS' if item['success'] else 'FAIL'}`",
            "",
            "Scene slot is derived from pose and is not an external action-selection input.",
        ])
        rr.log("status", rr.TextDocument(status, media_type=rr.MediaType.MARKDOWN))

    rr.disconnect()
    print(f"recorded={len(plan['tasks'])} output={args.output}")
    return 0


def wall_centers(wall: dict[str, Any]) -> list[list[float]]:
    rows = int(wall["rows"])
    columns = int(wall["columns"])
    depth, width, height = (float(value) for value in wall["box_size_m"])
    gap_y = float(wall["gap_y_m"])
    gap_z = float(wall["gap_z_m"])
    centers: list[list[float]] = []
    for row_from_bottom in range(rows):
        for column in range(columns):
            centers.append([
                0.9,
                float(wall["center_y_m"]) + (0.5 * (columns - 1) - column) * (width + gap_y),
                float(wall["bottom_z_m"]) + (row_from_bottom + 0.5) * (height + gap_z),
            ])
    return centers


if __name__ == "__main__":
    raise SystemExit(main())
