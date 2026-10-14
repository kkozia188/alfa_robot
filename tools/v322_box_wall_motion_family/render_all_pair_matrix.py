#!/usr/bin/python3
"""Render a dependency-free PNG summary of the 25-choose-2 pair matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    failed = {tuple(item["pair"]) for item in report["failed_pairs"]}
    width, height = 1800, 1500
    image = Image.new("RGB", (width, height), "#f7f8f6")
    draw = ImageDraw.Draw(image)
    draw.text((80, 55), "V3.2.2 Dual-Box Pair Matrix", fill="#18211d", font=font(44, True))
    draw.text(
        (80, 112),
        f"{report['successes']} / {report['pair_count']} certified  |  "
        f"{report['collision_validation']['checked_frames']:,} FCL frames  |  "
        f"one-sided attachment frames: {report['collision_validation']['one_sided_attachment_frames']}",
        fill="#3c4a43",
        font=font(23),
    )

    origin_x, origin_y, cell = 125, 235, 40
    success_color = "#2f9d67"
    failure_color = "#d34f4f"
    diagonal_color = "#b8bfbb"
    inactive_color = "#e3e7e4"
    grid_color = "#ffffff"
    label_font = font(17, True)
    for value in range(1, 26):
        x = origin_x + (value - 1) * cell + cell // 2
        y = origin_y + (value - 1) * cell + cell // 2
        text = str(value)
        bbox = draw.textbbox((0, 0), text, font=label_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, origin_y - 34), text, fill="#26322c", font=label_font)
        draw.text((origin_x - 42, y - 10), text, fill="#26322c", font=label_font)
    for row in range(1, 26):
        for column in range(1, 26):
            x0 = origin_x + (column - 1) * cell
            y0 = origin_y + (row - 1) * cell
            if row == column:
                color = diagonal_color
            elif row < column:
                color = failure_color if (row, column) in failed else success_color
            else:
                color = inactive_color
            draw.rectangle(
                (x0, y0, x0 + cell - 1, y0 + cell - 1),
                fill=color,
                outline=grid_color,
                width=1,
            )

    panel_x = 1190
    draw.text((panel_x, 235), "Unsupported pairs", fill="#18211d", font=font(30, True))
    for index, item in enumerate(report["failed_pairs"]):
        y = 300 + index * 71
        pair = item["pair"]
        draw.rounded_rectangle(
            (panel_x, y, 1680, y + 52), radius=5,
            fill="#fff0ee", outline="#d34f4f", width=2,
        )
        draw.text(
            (panel_x + 18, y + 11),
            f"{pair[0]} + {pair[1]}",
            fill="#9f2f2f",
            font=font(22, True),
        )
        draw.text(
            (panel_x + 145, y + 13),
            f"rows {item['rows_from_top'][0]}/{item['rows_from_top'][1]}  "
            f"col {item['columns_from_left'][0]}  "
            f"{item['candidate_attempts']} candidates",
            fill="#5a403c",
            font=font(18),
        )

    legend_y = 1300
    for x, color, label in (
        (125, success_color, "Certified full cycle"),
        (425, failure_color, "Unsupported by current family"),
        (820, diagonal_color, "Same box / not a pair"),
    ):
        draw.rectangle((x, legend_y, x + 30, legend_y + 30), fill=color)
        draw.text((x + 43, legend_y + 2), label, fill="#26322c", font=font(20))
    draw.text(
        (125, 1370),
        "Strict scene: all 23 non-target boxes remain present. Each green cell has an independent full-cycle MoveIt/FCL certificate.",
        fill="#4a5851",
        font=font(19),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
