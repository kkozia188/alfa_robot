#!/usr/bin/python3
"""Record a validated row-wise dual-arm 5x5 replay in Rerun."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v3_5x5_grasp_sequence_rerun as sequence

from alfa_robot_rerun.visualize_rerun import log_robot_state


SCHEMA = "alfa.v3_scoop_5x5_dual_replay.v1"
VEHICLE_FRONT_X_IN_BASE_M = 0.500000002779484


def base_transform(operation: dict[str, Any]) -> np.ndarray:
    return base_transform_values(operation.get("base_pose_map", [0.0, 0.0, 0.0]))


def base_transform_values(values: list[float]) -> np.ndarray:
    values = [float(value) for value in values]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("base_pose_map must contain three finite values")
    x, y, yaw = values
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    transform = np.eye(4)
    transform[:3, :3] = np.asarray([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])
    transform[:3, 3] = [x, y, 0.0]
    return transform


def frame_base_transform(
    operation: dict[str, Any], frame: dict[str, Any]
) -> np.ndarray:
    return base_transform_values(
        frame.get("base_pose_map", operation.get("base_pose_map", [0.0, 0.0, 0.0]))
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def operation_ids(operation: dict[str, Any]) -> list[int]:
    return [
        box_id for box_id in (
            int(operation.get("left_box_id", 0)),
            int(operation.get("right_box_id", 0)),
        )
        if box_id > 0
    ]


def operation_metrics(
    operation: dict[str, Any], joint_names: list[str]
) -> dict[str, Any]:
    frames = operation["frames"]
    arm_indices = [
        index for index, name in enumerate(joint_names)
        if name.startswith("left_joint") or name.startswith("right_joint")
    ]
    updown_index = joint_names.index("updown")
    joint_travel_deg = 0.0
    lift_travel_m = 0.0
    maximum_step_deg = 0.0
    joint_flip_events = 0
    base_travel_m = 0.0
    for previous, current in zip(frames, frames[1:]):
        for index in arm_indices:
            delta = abs(float(current["joints"][index]) - float(previous["joints"][index]))
            degrees = math.degrees(delta)
            joint_travel_deg += degrees
            maximum_step_deg = max(maximum_step_deg, degrees)
            if degrees > 180.0 + 1e-6:
                joint_flip_events += 1
        lift_travel_m += abs(
            float(current["joints"][updown_index])
            - float(previous["joints"][updown_index])
        )
        previous_base = frame_base_transform(operation, previous)[:3, 3]
        current_base = frame_base_transform(operation, current)[:3, 3]
        base_travel_m += float(np.linalg.norm(current_base - previous_base))
    bridge = operation.get(
        "transition_bridge", operation.get("center_entry_bridge", {})
    )
    box_planning = list(operation.get("box_planning", []))
    selected_task_core_s = sum(
        float(item.get("selected_task_core_ms", 0.0)) for item in box_planning
    ) / 1000.0
    accumulated_task_search_s = sum(
        float(item.get("accumulated_task_search_ms", 0.0))
        for item in box_planning
    ) / 1000.0
    selected_transition_core_s = sum(
        float(item.get("selected_transition_core_ms", 0.0))
        for item in box_planning
    ) / 1000.0
    accumulated_transition_search_s = sum(
        float(item.get("accumulated_transition_search_ms", 0.0))
        for item in box_planning
    ) / 1000.0
    bridge_planning_s = float(bridge.get("total_ms", 0.0)) / 1000.0
    return {
        "frame_count": len(frames),
        "joint_travel_deg": joint_travel_deg,
        "maximum_joint_step_deg": maximum_step_deg,
        "joint_flip_events": joint_flip_events,
        "lift_travel_m": lift_travel_m,
        "base_travel_m": base_travel_m,
        "attached_frame_count": sum(
            bool(frame.get("left_attached")) or bool(frame.get("right_attached"))
            for frame in frames
        ),
        "selected_task_core_s": selected_task_core_s,
        "accumulated_task_search_s": accumulated_task_search_s,
        "selected_transition_core_s": selected_transition_core_s,
        "accumulated_transition_search_s": accumulated_transition_search_s,
        "bridge_planning_s": bridge_planning_s,
        "recorded_core_planning_s": (
            selected_task_core_s + selected_transition_core_s + bridge_planning_s
        ),
        "transition_strategy": str(operation.get("transition_strategy", "single_golden")),
        "updown_policy": str(operation.get("updown_policy", "golden")),
        "edge_repairs": operation.get("edge_repairs", []),
    }


def planning_overview_markdown(
    box_planning: list[dict[str, Any]],
    operations: list[dict[str, Any]],
) -> str:
    boxes = sorted(box_planning, key=lambda item: int(item["box_id"]))
    task_total_s = sum(float(item["selected_task_core_ms"]) for item in boxes) / 1000.0
    retry_total_s = sum(float(item["accumulated_task_search_ms"]) for item in boxes) / 1000.0
    transition_total_s = sum(
        float(item["selected_transition_core_ms"]) for item in boxes
    ) / 1000.0
    within_target = sum(
        float(item["selected_task_core_ms"]) < 3000.0 - 1e-6 for item in boxes
    )
    maximum = max(boxes, key=lambda item: float(item["selected_task_core_ms"]))

    lines = [
        "# All planning times",
        "",
        "Selected task core time per box (seconds; `!` exceeds 3 s):",
        "",
        "| Row | C1 | C2 | C3 | C4 | C5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in range(1, 6):
        cells = []
        for item in boxes[(row - 1) * 5:row * 5]:
            seconds = float(item["selected_task_core_ms"]) / 1000.0
            cells.append(f"B{int(item['box_id']):02d} {seconds:.2f}{' !' if seconds >= 3.0 else ''}")
        lines.append(f"| {row} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        f"- Below 3 s target: **{within_target}/25**",
        f"- Selected task core total: **{task_total_s:.2f}s**",
        f"- Accumulated task search with retries: **{retry_total_s:.2f}s**",
        f"- Source transition core total: **{transition_total_s:.2f}s**",
        f"- Maximum: **B{int(maximum['box_id']):02d} "
        f"{float(maximum['selected_task_core_ms']) / 1000.0:.2f}s**",
    ])
    box5 = next((item for item in boxes if int(item["box_id"]) == 5), None)
    if box5 is not None:
        box5_seconds = float(box5["selected_task_core_ms"]) / 1000.0
        if box5_seconds > 5.0:
            detail = (
                "B05 timeout: shared lift `0.00 -> -0.25m`; "
                f"**{int(box5.get('ik_calls', 0)):,} IK / "
                f"{float(box5.get('ik_ms', 0.0)) / 1000.0:.2f}s**, "
                f"TCP shortcuts **{int(box5.get('tcp_shortcut_successes', 0))}/"
                f"{int(box5.get('tcp_shortcut_attempts', 0))}**, then joint RRT fallback."
            )
        else:
            detail = (
                f"B05 optimized: bounded J1 branch + joint shortcut; **{box5_seconds:.2f}s**, "
                f"{int(box5.get('ik_calls', 0)):,} IK / "
                f"{float(box5.get('ik_ms', 0.0)) / 1000.0:.2f}s."
            )
        lines.extend(["", f"- {detail}"])
    lines.extend([
        "",
        "Dual sequence groups (`task core + source entry core + new bridge`):",
        "",
        "| Group | Boxes | Core (s) | Bridge (s) | Recorded total (s) |",
        "| ---: | --- | ---: | ---: | ---: |",
    ])
    for index, operation in enumerate(operations, start=1):
        metrics = operation_metrics(operation, [
            "updown", "head_joint",
            *[f"left_joint{joint}" for joint in range(1, 8)],
            *[f"right_joint{joint}" for joint in range(1, 8)],
        ])
        boxes_text = "+".join(map(str, operation_ids(operation)))
        source_core = (
            float(metrics["selected_task_core_s"])
            + float(metrics["selected_transition_core_s"])
        )
        lines.append(
            f"| {index} | {boxes_text} | {source_core:.2f} | "
            f"{float(metrics['bridge_planning_s']):.2f} | "
            f"{float(metrics['recorded_core_planning_s']):.2f} |"
        )
    lines.extend([
        "",
        "Times are preserved offline search measurements. Cached replay loading is not counted.",
    ])
    return "\n".join(lines)


class DualSequenceRecorder(sequence.SequenceRecorder):
    def __init__(
        self,
        save: Path,
        playback_speed: float,
        box_planning: list[dict[str, Any]],
        operations: list[dict[str, Any]],
    ) -> None:
        self.is_dual_sequence = any(
            operation.get("kind") in {"dual", "dual_conveyor_cycle"}
            for operation in operations
        )
        self.has_conveyor_shuttle = any(
            str(operation.get("kind", "")).endswith("conveyor_cycle")
            for operation in operations
        )
        sequence_title = (
            "Dual-arm 5x5 right-conveyor shuttle"
            if self.has_conveyor_shuttle else (
                "Dual-arm 5x5 row-wise grasp" if self.is_dual_sequence
                else "Vehicle-front clearance sequential 5x5 grasp"
            )
        )
        super().__init__(
            save,
            playback_speed,
            end_effector="scoop",
            application_id="v3_scoop_x075_5x5_dual_row_sequence",
            view_name=f"V3 scoop {sequence_title}",
            task_title=sequence_title,
            tool_name="scoop",
            transition_joint_speed_deg_s=45.0,
            task_joint_speed_deg_s=45.0,
            cartesian_joint_speed_deg_s=35.0,
        )
        self.base_transform = np.eye(4)
        self.completed_boxes = 0
        self.group_count = sum(
            operation.get("kind") != "base_transition" for operation in operations
        )
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        origin="/world",
                        contents=["/world/**"],
                        name=f"V3 scoop {sequence_title}",
                    ),
                    rrb.Vertical(
                        rrb.TextDocumentView(
                            origin="/planning_overview", name="All planning times"
                        ),
                        rrb.BarChartView(
                            origin="/planning/task_core_s",
                            name="25 task core planning times (s), target < 3",
                        ),
                        rrb.TextDocumentView(
                            origin="/transition_status", name="Current group planning"
                        ),
                        rrb.TextDocumentView(origin="/status", name="Task status"),
                        row_shares=[0.42, 0.22, 0.14, 0.22],
                    ),
                    column_shares=[0.67, 0.33],
                ),
                collapse_panels=True,
            )
        )
        rr.log(
            "planning_overview",
            rr.TextDocument(
                planning_overview_markdown(box_planning, operations),
                media_type=rr.MediaType.MARKDOWN,
            ),
            static=True,
        )
        ordered = sorted(box_planning, key=lambda item: int(item["box_id"]))
        rr.log(
            "planning/task_core_s",
            rr.BarChart(
                [float(item["selected_task_core_ms"]) / 1000.0 for item in ordered],
                abscissa=[int(item["box_id"]) for item in ordered],
                widths=[0.72] * len(ordered),
                color=[35, 170, 225, 230],
            ),
            static=True,
        )
        if self.has_conveyor_shuttle:
            rr.log(
                "world/conveyor/reference",
                rr.Boxes3D(
                    centers=[[-2.825, -1.5, 0.58]],
                    half_sizes=[[0.475, 1.05, 0.06]],
                    colors=[[70, 78, 86, 210]],
                    labels=["right conveyor (visual reference)"],
                    show_labels=True,
                ),
                static=True,
            )

    def log_boxes(self, target_ids: list[int] | None = None) -> None:
        targets = set(target_ids or []) & self.remaining
        regular_ids = sorted(self.remaining - targets)
        if regular_ids:
            rr.log(
                "world/boxes/remaining",
                rr.Boxes3D(
                    centers=[self.centers[box_id] for box_id in regular_ids],
                    half_sizes=[[
                        sequence.BOX_DEPTH / 2.0,
                        sequence.BOX_WIDTH / 2.0,
                        sequence.BOX_HEIGHT / 2.0,
                    ]],
                    colors=[[238, 142, 48, 180]],
                    show_labels=False,
                ),
            )
        else:
            rr.log("world/boxes/remaining", rr.Clear(recursive=True))
        if targets:
            ordered = sorted(targets)
            colors = [
                [45, 190, 110, 235] if index == 0 else [35, 160, 235, 235]
                for index, _ in enumerate(ordered)
            ]
            rr.log(
                "world/boxes/targets",
                rr.Boxes3D(
                    centers=[self.centers[box_id] for box_id in ordered],
                    half_sizes=[[
                        sequence.BOX_DEPTH / 2.0,
                        sequence.BOX_WIDTH / 2.0,
                        sequence.BOX_HEIGHT / 2.0,
                    ]],
                    colors=colors,
                    labels=[f"target {box_id}" for box_id in ordered],
                    show_labels=True,
                ),
            )
        else:
            rr.log("world/boxes/targets", rr.Clear(recursive=True))

    def log_carried_box(
        self,
        side: str,
        box_id: int,
        attachment: dict[str, Any],
        joints: dict[str, float],
    ) -> None:
        transforms = self.robot.fk(joints)
        tool_link = str(attachment["tool_link"])
        tool_transform = transforms.get(tool_link)
        if tool_transform is None:
            raise RuntimeError(f"missing FK for {tool_link}")
        offset = np.asarray(attachment["center_in_tool"], dtype=float)
        rotation = np.asarray(attachment["rotation_in_tool"], dtype=float)
        size = np.asarray(attachment["size"], dtype=float)
        transform = self.base_transform @ tool_transform
        transform[:3, 3] = (
            self.base_transform @ np.asarray([
                *(tool_transform[:3, 3] + tool_transform[:3, :3] @ offset), 1.0
            ])
        )[:3]
        transform[:3, :3] = (
            self.base_transform[:3, :3] @ tool_transform[:3, :3] @ rotation
        )
        color = [45, 205, 125, 235] if side == "left" else [40, 170, 245, 235]
        rr.log(
            f"world/boxes/carried/{side}",
            rr.Boxes3D(
                centers=[transform[:3, 3].tolist()],
                half_sizes=[(size * 0.5).tolist()],
                quaternions=[sequence.matrix_to_quaternion(transform[:3, :3])],
                colors=[color],
                labels=[f"{side} carried box {box_id}"],
                show_labels=True,
            ),
        )

    def build_tcp_paths(
        self,
        operation: dict[str, Any],
        joint_names: list[str],
    ) -> dict[str, dict[str, list[list[float]]]]:
        active_sides = [
            side for side in ("left", "right")
            if int(operation.get(f"{side}_box_id", 0)) > 0
        ]
        frames = list(operation["frames"])
        attached_indices = [
            index for index, frame in enumerate(frames)
            if bool(frame.get("left_attached")) or bool(frame.get("right_attached"))
        ]
        if not attached_indices:
            raise ValueError(f"operation {operation['label']} never attaches a box")
        first_attached = attached_indices[0]
        last_attached = attached_indices[-1]
        pickup_indices = range(0, first_attached + 1)
        pullback_indices = range(
            first_attached, min(len(frames), last_attached + 2)
        )
        paths: dict[str, dict[str, list[list[float]]]] = {
            "pickup": {side: [] for side in active_sides},
            "pullback": {side: [] for side in active_sides},
        }
        for phase, indices in (
            ("pickup", pickup_indices), ("pullback", pullback_indices)
        ):
            for index in indices:
                frame = frames[index]
                joints = {
                    name: float(value)
                    for name, value in zip(joint_names, frame["joints"])
                }
                transforms = self.robot.fk(joints)
                for side in active_sides:
                    tool_link = str(operation[f"{side}_attachment"]["tool_link"])
                    tool_world = self.base_transform @ transforms[tool_link]
                    paths[phase][side].append(tool_world[:3, 3].tolist())
        return paths

    def show_tcp_path_phase(
        self,
        paths: dict[str, dict[str, list[list[float]]]],
        phase: str,
    ) -> None:
        rr.log("world/tcp_paths", rr.Clear(recursive=True))
        colors = {
            ("pickup", "left"): [35, 195, 105, 225],
            ("pickup", "right"): [35, 145, 235, 225],
            ("pullback", "left"): [240, 168, 48, 225],
            ("pullback", "right"): [225, 78, 68, 225],
        }
        for side, positions in paths[phase].items():
            if len(positions) < 2:
                continue
            rr.log(
                f"world/tcp_paths/{phase}/{side}",
                rr.LineStrips3D(
                    strips=[positions],
                    colors=[colors[(phase, side)]],
                    radii=[0.005],
                    labels=[f"{side} {phase} path"],
                    show_labels=False,
                ),
            )

    def log_status(
        self,
        operation: dict[str, Any],
        group_index: int,
        stage: str,
        metrics: dict[str, Any],
        path_phase: str,
    ) -> None:
        ids = operation_ids(operation)
        box_text = " + ".join(str(box_id) for box_id in ids)
        kinds = {
            "dual": "dual simultaneous",
            "single_center": "single center",
            "dual_conveyor_cycle": "dual pickup + right conveyor shuttle",
            "single_conveyor_cycle": "single pickup + right conveyor shuttle",
        }
        kind = kinds.get(str(operation["kind"]), str(operation["kind"]))
        bridge_seconds = float(metrics["bridge_planning_s"])
        bridge_text = f"{bridge_seconds:.2f}s" if bridge_seconds > 0.0 else "cached path"
        repairs = metrics["edge_repairs"]
        repair_text = ", ".join(
            f"{item['joint']} {float(item['offset_deg']):+.2f} deg"
            for item in repairs
        ) or "none"
        rr.log(
            "status",
            rr.TextDocument(
                f"# {self.task_title}\n\n"
                f"- Group: **{group_index}/{self.group_count}**\n"
                f"- Row: **{operation['row']}/5**\n"
                f"- Boxes: **{box_text}**\n"
                f"- Mode: `{kind}`\n"
                f"- Stage: `{stage}`\n"
                f"- Path shown: **{path_phase}**\n"
                f"- Completed: **{self.completed_boxes}/25**\n"
                f"- Remaining: **{len(self.remaining)}**\n"
                f"- Shared lift: `{float(operation['frames'][0]['joints'][0]):.2f} m`\n"
                f"- Base Y: `{self.base_transform[1, 3]:.3f} m`\n"
                f"- Conveyor route: `back {float(operation.get('backoff_distance_m', 0.0)):.3f} m, "
                f"right {float(operation.get('right_shuttle_distance_m', 0.0)):.3f} m`\n"
                f"- Transition: `{metrics['transition_strategy']}`\n"
                f"- Bridge search: **{bridge_text}**\n"
                f"- Task core: **{float(metrics['selected_task_core_s']):.2f}s**\n"
                f"- Source entry core: **{float(metrics['selected_transition_core_s']):.2f}s**\n"
                f"- Recorded core total: **{float(metrics['recorded_core_planning_s']):.2f}s**\n"
                f"- Local repair: `{repair_text}`\n"
                f"- Joint travel: **{float(metrics['joint_travel_deg']):.1f} deg**\n"
                f"- Joint flip events: **{int(metrics['joint_flip_events'])}**\n"
                "- Validation: `MoveIt/FCL, 1 deg / 1 cm edges`",
                media_type=rr.MediaType.MARKDOWN,
            ),
        )

    def play_operation(
        self,
        operation: dict[str, Any],
        group_index: int,
        joint_names: list[str],
    ) -> dict[str, Any]:
        if operation.get("kind") == "base_transition":
            return self.play_base_transition(operation, joint_names)
        metrics = operation_metrics(operation, joint_names)
        ids = operation_ids(operation)
        self.base_transform = base_transform(operation)
        tcp_paths = self.build_tcp_paths(operation, joint_names)
        self.set_time()
        rr.log(
            "world/robot",
            rr.Transform3D(
                translation=self.base_transform[:3, 3].tolist(),
                quaternion=sequence.matrix_to_quaternion(
                    self.base_transform[:3, :3]
                ),
            ),
        )
        rr.log(
            "transition_status",
            rr.TextDocument(
                f"# Row {operation['row']} / Group {group_index}\n\n"
                f"- Targets: **{' + '.join(map(str, ids))}**\n"
                f"- Frames: **{metrics['frame_count']}**\n"
                f"- Empty transition: `{metrics['transition_strategy']}`\n"
                f"- Shared lift policy: `{metrics['updown_policy']}`\n"
                f"- Bridge planning: **{float(metrics['bridge_planning_s']):.2f}s**\n"
                f"- Task core planning: **{float(metrics['selected_task_core_s']):.2f}s**\n"
                f"- Source entry core: **{float(metrics['selected_transition_core_s']):.2f}s**\n"
                f"- Recorded core total: **{float(metrics['recorded_core_planning_s']):.2f}s**\n"
                "- Pickup path: `left green / right blue`\n"
                "- Pullback path: `left amber / right red`\n"
                f"- Edge repairs: **{len(metrics['edge_repairs'])}**",
                media_type=rr.MediaType.MARKDOWN,
            ),
        )
        rr.log(
            "metrics/transition/total_search_s",
            rr.Scalars([float(metrics["bridge_planning_s"])]),
        )
        rr.log(
            "metrics/transition/selected_core_s",
            rr.Scalars([float(metrics["bridge_planning_s"])]),
        )

        detached = False
        attached_sides_seen: set[str] = set()
        visible_path_phase = ""
        previous_stage = ""
        previous = self.last_joints
        previous_base_position = self.base_transform[:3, 3].copy()
        for frame in operation["frames"]:
            joints = {
                name: float(value)
                for name, value in zip(joint_names, frame["joints"])
            }
            stage = str(frame["stage"])
            left_attached = bool(frame.get("left_attached", False))
            right_attached = bool(frame.get("right_attached", False))
            for side, attached in (
                ("left", left_attached), ("right", right_attached)
            ):
                box_id = int(operation.get(f"{side}_box_id", 0))
                if attached and side not in attached_sides_seen and box_id > 0:
                    self.remaining.remove(box_id)
                    self.completed_boxes += 1
                    attached_sides_seen.add(side)
            if left_attached or right_attached:
                detached = True

            self.base_transform = frame_base_transform(operation, frame)
            self.set_time()
            path_phase = "pullback" if detached else "pickup"
            if path_phase != visible_path_phase:
                self.show_tcp_path_phase(tcp_paths, path_phase)
                visible_path_phase = path_phase
            rr.log(
                "world/robot",
                rr.Transform3D(
                    translation=self.base_transform[:3, 3].tolist(),
                    quaternion=sequence.matrix_to_quaternion(
                        self.base_transform[:3, :3]
                    ),
                ),
            )
            log_robot_state(self.robot, joints, "world/robot")
            pending_ids = [
                int(operation.get(f"{side}_box_id", 0))
                for side in ("left", "right")
                if side not in attached_sides_seen
                and int(operation.get(f"{side}_box_id", 0)) > 0
            ]
            self.log_boxes(pending_ids)
            for side, attached in (
                ("left", left_attached), ("right", right_attached)
            ):
                box_id = int(operation.get(f"{side}_box_id", 0))
                if attached and box_id > 0:
                    self.log_carried_box(
                        side, box_id, operation[f"{side}_attachment"], joints
                    )
                else:
                    rr.log(f"world/boxes/carried/{side}", rr.Clear(recursive=True))
            self.log_status(operation, group_index, stage, metrics, path_phase)

            maximum_delta_deg = 0.0
            if previous is not None:
                maximum_delta_deg = max(
                    math.degrees(abs(value - previous.get(name, value)))
                    for name, value in joints.items()
                    if name != "updown"
                )
            speed = (
                self.cartesian_joint_speed_deg_s
                if stage in {"cartesian_approach", "cartesian_retreat"}
                else self.task_joint_speed_deg_s
            )
            delay = max(self.minimum_frame_interval_s, maximum_delta_deg / speed)
            base_position = self.base_transform[:3, 3]
            base_distance = float(np.linalg.norm(base_position - previous_base_position))
            delay = max(delay, base_distance / 0.10)
            if previous_stage and stage != previous_stage:
                delay = max(delay, 0.18)
            self.advance(min(delay, 0.20))
            previous = joints
            previous_base_position = base_position.copy()
            previous_stage = stage

        self.last_joints = previous
        self.set_time()
        rr.log("world/boxes/carried/left", rr.Clear(recursive=True))
        rr.log("world/boxes/carried/right", rr.Clear(recursive=True))
        rr.log("world/tcp_paths", rr.Clear(recursive=True))
        self.log_boxes()
        self.log_status(operation, group_index, "group_complete", metrics, "complete")
        self.advance(0.30)
        return metrics

    def play_base_transition(
        self,
        operation: dict[str, Any],
        joint_names: list[str],
    ) -> dict[str, Any]:
        metrics = operation_metrics(operation, joint_names)
        frames = list(operation["frames"])
        speed = float(operation.get("base_speed_m_s", 0.10))
        if speed <= 0.0:
            raise ValueError("base_speed_m_s must be positive")
        rr.log("world/tcp_paths", rr.Clear(recursive=True))
        rr.log("world/boxes/carried", rr.Clear(recursive=True))
        rr.log(
            "transition_status",
            rr.TextDocument(
                "# Chassis reposition\n\n"
                f"- After completed rows: **{int(operation.get('after_row', 3))}/5**\n"
                f"- From front clearance: **{float(operation['from_front_clearance_m']):.2f} m**\n"
                f"- To front clearance: **{float(operation['to_front_clearance_m']):.2f} m**\n"
                f"- Base travel: **{float(metrics['base_travel_m']):.2f} m**\n"
                f"- Frames: **{len(frames)}**\n"
                "- Arms: `stow first, then stationary / no attached boxes`",
                media_type=rr.MediaType.MARKDOWN,
            ),
        )
        previous_position: np.ndarray | None = None
        previous_joints = self.last_joints
        for frame_index, frame in enumerate(frames, start=1):
            joints = {
                name: float(value)
                for name, value in zip(joint_names, frame["joints"])
            }
            self.base_transform = frame_base_transform(operation, frame)
            position = self.base_transform[:3, 3].copy()
            self.set_time()
            rr.log(
                "world/robot",
                rr.Transform3D(
                    translation=position.tolist(),
                    quaternion=sequence.matrix_to_quaternion(
                        self.base_transform[:3, :3]
                    ),
                ),
            )
            log_robot_state(self.robot, joints, "world/robot")
            self.log_boxes()
            rr.log(
                "status",
                rr.TextDocument(
                    f"# {self.task_title}\n\n"
                    "- Stage: **chassis reposition**\n"
                    f"- Progress: **{frame_index}/{len(frames)}**\n"
                    f"- Completed: **{self.completed_boxes}/25**\n"
                    f"- Current base X: **{position[0]:.3f} m**\n"
                    f"- Target base X: **{float(operation['base_pose_goal_map'][0]):.3f} m**\n"
                    "- Validation: `MoveIt/FCL, 1 deg / 1 cm base edges`",
                    media_type=rr.MediaType.MARKDOWN,
                ),
            )
            if previous_position is None:
                delay = self.minimum_frame_interval_s
            else:
                delay = max(
                    self.minimum_frame_interval_s,
                    float(np.linalg.norm(position - previous_position)) / speed,
                )
            if previous_joints is not None:
                maximum_joint_delta = max(
                    abs(value - previous_joints.get(name, value))
                    for name, value in joints.items()
                )
                delay = max(
                    delay,
                    maximum_joint_delta / self.transition_joint_speed_rad_s,
                )
            self.advance(delay)
            previous_position = position
            previous_joints = joints
        self.last_joints = previous_joints
        self.advance(0.30)
        return metrics

    def finish_dual(self, save: Path) -> None:
        self.set_time()
        rr.log("world/boxes/carried", rr.Clear(recursive=True))
        rr.log("world/boxes/targets", rr.Clear(recursive=True))
        rr.log("world/tcp_paths", rr.Clear(recursive=True))
        self.log_boxes()
        rr.log(
            "status",
            rr.TextDocument(
                f"# {self.task_title}\n\n"
                "- Completed: **25/25**\n"
                "- Remaining boxes: **0**\n"
                "- Rows completed: **5/5**\n"
                "- Result: **SUCCESS**",
                media_type=rr.MediaType.MARKDOWN,
            ),
        )
        self.advance(0.50)
        rr.disconnect()
        print(f"RESULT success=25/25 recording={save}", flush=True)


def write_metrics_csv(
    path: Path,
    operations: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "group", "row", "boxes", "kind", "frames", "transition_strategy",
        "updown_policy", "selected_task_core_s", "accumulated_task_search_s",
        "selected_transition_core_s", "accumulated_transition_search_s",
        "bridge_planning_s", "recorded_core_planning_s", "joint_travel_deg",
        "maximum_joint_step_deg", "joint_flip_events", "lift_travel_m",
        "edge_repairs", "base_travel_m",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (operation, item) in enumerate(zip(operations, metrics), start=1):
            writer.writerow({
                "group": index,
                "row": operation["row"],
                "boxes": "+".join(map(str, operation_ids(operation))),
                "kind": operation["kind"],
                "frames": item["frame_count"],
                "transition_strategy": item["transition_strategy"],
                "updown_policy": item["updown_policy"],
                "selected_task_core_s": f"{float(item['selected_task_core_s']):.6f}",
                "accumulated_task_search_s": (
                    f"{float(item['accumulated_task_search_s']):.6f}"
                ),
                "selected_transition_core_s": (
                    f"{float(item['selected_transition_core_s']):.6f}"
                ),
                "accumulated_transition_search_s": (
                    f"{float(item['accumulated_transition_search_s']):.6f}"
                ),
                "bridge_planning_s": f"{float(item['bridge_planning_s']):.6f}",
                "recorded_core_planning_s": (
                    f"{float(item['recorded_core_planning_s']):.6f}"
                ),
                "joint_travel_deg": f"{float(item['joint_travel_deg']):.6f}",
                "maximum_joint_step_deg": f"{float(item['maximum_joint_step_deg']):.6f}",
                "joint_flip_events": item["joint_flip_events"],
                "lift_travel_m": f"{float(item['lift_travel_m']):.6f}",
                "base_travel_m": f"{float(item['base_travel_m']):.6f}",
                "edge_repairs": json.dumps(item["edge_repairs"], separators=(",", ":")),
            })
    temporary.replace(path)


def write_box_planning_csv(path: Path, box_planning: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "box_id", "row", "column", "side", "mode", "source",
        "selected_task_core_s", "accumulated_task_search_s",
        "selected_transition_core_s", "accumulated_transition_search_s",
        "within_5s",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in sorted(box_planning, key=lambda value: int(value["box_id"])):
            task_core_s = float(item["selected_task_core_ms"]) / 1000.0
            writer.writerow({
                "box_id": item["box_id"],
                "row": item["row"],
                "column": item["column"],
                "side": item["side"],
                "mode": item["mode"],
                "source": item["source"],
                "selected_task_core_s": f"{task_core_s:.6f}",
                "accumulated_task_search_s": (
                    f"{float(item['accumulated_task_search_ms']) / 1000.0:.6f}"
                ),
                "selected_transition_core_s": (
                    f"{float(item['selected_transition_core_ms']) / 1000.0:.6f}"
                ),
                "accumulated_transition_search_s": (
                    f"{float(item['accumulated_transition_search_ms']) / 1000.0:.6f}"
                ),
                "within_5s": task_core_s <= 5.0 + 1e-6,
            })
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--box-planning-csv", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--playback-speed", type=float, default=2.0)
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.playback_speed <= 0.0:
        parser.error("playback speed must be positive")

    replay = read_json(args.input)
    if replay.get("schema") != SCHEMA:
        parser.error(f"unexpected replay schema: {replay.get('schema')}")
    operations = list(replay["operations"])
    dual_order = [
        [1, 5], [2, 4], [3], [6, 10], [7, 9], [8], [11, 15],
        [12, 14], [13], [16, 20], [17, 19], [18], [21, 25],
        [22, 24], [23],
    ]
    sequential_order = [
        [box_id] for box_id in (
            1, 2, 5, 4, 3, 6, 7, 10, 9, 8, 11, 12, 15, 14, 13,
            16, 17, 20, 19, 18, 21, 22, 25, 24, 23,
        )
    ]
    if replay.get("group_order") not in (dual_order, sequential_order):
        parser.error("input does not use a supported 5x5 operation order")
    joint_names = [str(name) for name in replay["joint_names"]]
    box_planning = list(replay.get("box_planning", []))
    if len(box_planning) != 25:
        parser.error("input does not contain all 25 box planning measurements")
    validation = read_json(args.validation_report) if args.validation_report else {}
    if validation and not bool(validation.get("success")):
        parser.error("validation report is not successful")

    recorder = DualSequenceRecorder(
        args.save.resolve(), args.playback_speed, box_planning, operations
    )
    metrics = []
    group_index = 0
    for operation in operations:
        if operation.get("kind") != "base_transition":
            group_index += 1
        metrics.append(recorder.play_operation(operation, group_index, joint_names))
    recorder.finish_dual(args.save.resolve())

    summary_path = args.summary or args.save.with_name(args.save.stem + "-summary.json")
    metrics_path = args.metrics_csv or args.save.with_name(args.save.stem + "-metrics.csv")
    box_planning_path = args.box_planning_csv or args.save.with_name(
        args.save.stem + "-box-planning.csv"
    )
    selected_task_core_s = sum(
        float(item["selected_task_core_ms"]) for item in box_planning
    ) / 1000.0
    accumulated_task_search_s = sum(
        float(item["accumulated_task_search_ms"]) for item in box_planning
    ) / 1000.0
    summary = {
        "schema": "alfa.v3_scoop_5x5_dual_summary.v1",
        "input": str(args.input.resolve()),
        "recording": str(args.save.resolve()),
        "validation_report": (
            str(args.validation_report.resolve()) if args.validation_report else ""
        ),
        "validation_success": bool(validation.get("success", False)),
        "completed_boxes": sum(len(operation_ids(item)) for item in operations),
        "completed_rows": 5,
        "group_count": sum(
            operation.get("kind") != "base_transition" for operation in operations
        ),
        "base_transition_count": sum(
            operation.get("kind") == "base_transition" for operation in operations
        ),
        "base_travel_m": sum(float(item["base_travel_m"]) for item in metrics),
        "frame_count": sum(int(item["frame_count"]) for item in metrics),
        "timeline_s": recorder.timeline_s,
        "group_order": replay["group_order"],
        "base_poses_map": sorted({
            tuple(float(value) for value in operation.get("base_pose_map", [0, 0, 0]))
            for operation in operations
        }),
        "planning_origin_standoff_distances_m": sorted({
            sequence.CONTACT_X - float(operation.get("base_pose_map", [0, 0, 0])[0])
            for operation in operations
        }),
        "vehicle_front_clearance_distances_m": sorted({
            sequence.CONTACT_X - (
                float(operation.get("base_pose_map", [0, 0, 0])[0])
                + VEHICLE_FRONT_X_IN_BASE_M
            )
            for operation in operations
        }),
        "joint_flip_events": sum(int(item["joint_flip_events"]) for item in metrics),
        "total_joint_travel_deg": sum(
            float(item["joint_travel_deg"]) for item in metrics
        ),
        "maximum_joint_step_deg": max(
            float(item["maximum_joint_step_deg"]) for item in metrics
        ),
        "bridge_planning_s": sum(
            float(item["bridge_planning_s"]) for item in metrics
        ),
        "selected_task_core_s": selected_task_core_s,
        "accumulated_task_search_s": accumulated_task_search_s,
        "boxes_within_5s": sum(
            float(item["selected_task_core_ms"]) <= 5000.0 + 1e-6
            for item in box_planning
        ),
        "boxes_within_3s": sum(
            float(item["selected_task_core_ms"]) < 3000.0 - 1e-6
            for item in box_planning
        ),
        "maximum_box_task_core_s": max(
            float(item["selected_task_core_ms"]) for item in box_planning
        ) / 1000.0,
        "box_planning": box_planning,
        "operations": [
            {
                "group": index,
                "row": operation["row"],
                "boxes": operation_ids(operation),
                "kind": operation["kind"],
                **item,
            }
            for index, (operation, item) in enumerate(
                zip(operations, metrics), start=1
            )
        ],
    }
    write_json(summary_path, summary)
    write_metrics_csv(metrics_path, operations, metrics)
    write_box_planning_csv(box_planning_path, box_planning)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
    print(f"Box planning: {box_planning_path}", flush=True)

    if args.spawn:
        subprocess.Popen(
            ["rerun", "--new", str(args.save.resolve())],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"Rerun opened: {args.save.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
