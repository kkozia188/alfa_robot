#!/usr/bin/python3
"""Record the five representative full cycles with the V3.2.2 production shell."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_dual(script_dir: Path) -> ModuleType:
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    path = script_dir / "v3_scoop_5x5_dual_rerun.py"
    spec = importlib.util.spec_from_file_location("v322_family_dual_rerun", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Rerun recorder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--report-markdown", type=Path, required=True)
    parser.add_argument("--script-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--playback-speed", type=float, default=3.0)
    args = parser.parse_args()
    replay = read_json(args.replay)
    validation = read_json(args.validation)
    if not validation.get("success"):
        parser.error("validation report is not successful")
    dual = load_dual(args.script_dir)
    urdf_text = args.urdf.read_text(encoding="utf-8")
    dual.sequence.render_current_urdf = lambda _mappings=None: urdf_text
    operations = list(replay["operations"])
    box_planning = list(replay["box_planning"])
    selected_ids = {int(item["box_id"]) for item in box_planning}

    class FamilyRecorder(dual.DualSequenceRecorder):
        def __init__(self) -> None:
            super().__init__(
                args.output.resolve(), args.playback_speed, box_planning, operations
            )
            self.total_representatives = len(selected_ids)
            self.remaining = set(selected_ids)
            self.completed_boxes = 0
            self.task_title = "Pose-driven box-wall action family"
            rr.send_blueprint(rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        origin="/world", contents=["/world/**"],
                        name="V3.2.2 representative full cycles",
                    ),
                    rrb.Vertical(
                        rrb.TextDocumentView(origin="/report", name="Acceptance report"),
                        rrb.BarChartView(
                            origin="/planning/task_core_s",
                            name="Fresh representative planning time (s)",
                        ),
                        rrb.TextDocumentView(origin="/transition_status", name="Current cycle"),
                        rrb.TextDocumentView(origin="/status", name="Current stage"),
                        row_shares=[0.42, 0.18, 0.18, 0.22],
                    ),
                    column_shares=[0.67, 0.33],
                ),
                collapse_panels=True,
            ))
            rr.log(
                "report",
                rr.TextDocument(
                    args.report_markdown.read_text(encoding="utf-8"),
                    media_type=rr.MediaType.MARKDOWN,
                ),
                static=True,
            )

        def log_status(
            self,
            operation: dict[str, Any],
            group_index: int,
            stage: str,
            metrics: dict[str, Any],
            path_phase: str,
        ) -> None:
            rr.log(
                "status",
                rr.TextDocument(
                    "\n".join([
                        "# Pose-driven box-wall action family",
                        "",
                        f"- Representative: **{group_index}/{self.total_representatives}**",
                        f"- Request: **{operation.get('pose_driven_request_id', operation['label'])}**",
                        f"- Row: **{operation['row']}/5**",
                        f"- Stage: `{stage}`",
                        f"- Path phase: **{path_phase}**",
                        f"- Completed representatives: **{self.completed_boxes}/{self.total_representatives}**",
                        f"- Base Y: `{self.base_transform[1, 3]:.3f} m`",
                        f"- Fresh task core: **{float(metrics['selected_task_core_s']):.3f}s**",
                        f"- Fresh entry bridge: **{float(metrics['selected_transition_core_s']):.3f}s**",
                        "- Validation: `MoveIt/FCL, 1 deg / 1 cm edges`",
                    ]),
                    media_type=rr.MediaType.MARKDOWN,
                ),
            )

        def play_base_transition(
            self, operation: dict[str, Any], joint_names: list[str]
        ) -> dict[str, Any]:
            metrics = dual.operation_metrics(operation, joint_names)
            frames = list(operation["frames"])
            previous_position: np.ndarray | None = None
            previous_joints = self.last_joints
            for frame_index, frame in enumerate(frames, start=1):
                joints = {
                    name: float(value)
                    for name, value in zip(joint_names, frame["joints"])
                }
                self.base_transform = dual.frame_base_transform(operation, frame)
                position = self.base_transform[:3, 3].copy()
                self.set_time()
                rr.log(
                    "world/robot",
                    rr.Transform3D(
                        translation=position.tolist(),
                        quaternion=dual.sequence.matrix_to_quaternion(
                            self.base_transform[:3, :3]
                        ),
                    ),
                )
                dual.log_robot_state(self.robot, joints, "world/robot")
                self.log_boxes()
                rr.log(
                    "status",
                    rr.TextDocument(
                        "\n".join([
                            "# Chassis reposition",
                            "",
                            f"- Progress: **{frame_index}/{len(frames)}**",
                            f"- Completed representatives: **{self.completed_boxes}/{self.total_representatives}**",
                            f"- Base X: **{position[0]:.3f} m**",
                            f"- Target X: **{float(operation['base_pose_goal_map'][0]):.3f} m**",
                        ]),
                        media_type=rr.MediaType.MARKDOWN,
                    ),
                )
                delay = self.minimum_frame_interval_s
                if previous_position is not None:
                    delay = max(delay, float(np.linalg.norm(position - previous_position)) / 0.10)
                if previous_joints is not None:
                    delay = max(
                        delay,
                        max(abs(value - previous_joints.get(name, value)) for name, value in joints.items())
                        / self.transition_joint_speed_rad_s,
                    )
                self.advance(delay)
                previous_position = position
                previous_joints = joints
            self.last_joints = previous_joints
            self.advance(0.25)
            return metrics

        def finish_family(self) -> None:
            self.set_time()
            rr.log("world/boxes/carried", rr.Clear(recursive=True))
            rr.log("world/boxes/targets", rr.Clear(recursive=True))
            rr.log("world/tcp_paths", rr.Clear(recursive=True))
            self.log_boxes()
            rr.log(
                "status",
                rr.TextDocument(
                    "\n".join([
                        "# Pose-driven box-wall action family",
                        "",
                        f"- Completed representatives: **{self.completed_boxes}/{self.total_representatives}**",
                        "- Rows represented: **5/5**",
                        "- Full-cycle MoveIt/FCL: **PASS**",
                    ]),
                    media_type=rr.MediaType.MARKDOWN,
                ),
            )
            self.advance(0.5)
            rr.disconnect()

    recorder = FamilyRecorder()
    group_index = 0
    metrics = []
    for operation in operations:
        if operation.get("kind") != "base_transition":
            group_index += 1
        metrics.append(recorder.play_operation(operation, group_index, replay["joint_names"]))
    recorder.finish_family()
    print(
        f"RESULT representatives={recorder.completed_boxes}/{len(selected_ids)} "
        f"frames={sum(item['frame_count'] for item in metrics)} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
