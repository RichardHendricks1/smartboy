#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import os
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _ensure_mplconfigdir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    mpl_dir = os.path.join(tempfile.gettempdir(), "planner_toolbox_mplconfig")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_dir


def _prefer_safe_gui_backend() -> None:
    if os.environ.get("MPLBACKEND"):
        return
    try:
        import tkinter  # noqa: F401
    except Exception:
        return
    os.environ["MPLBACKEND"] = "TkAgg"


_ensure_mplconfigdir()
_prefer_safe_gui_backend()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

import plot_planner_inputs as legacy_csv
import plotlog_0320 as legacy_log


ALL_EVENT_TYPES = (
    "search2park",
    "replan",
    "stop_reason",
    "status",
    "file_boundary",
)
TIMESTAMP_MATCH_TOLERANCE = 5_000_000
EVENT_TYPE_COLORS = {
    "search2park": "#E53935",
    "replan": "#FB8C00",
    "stop_reason": "#43A047",
    "status": "#3949AB",
    "file_boundary": "#00838F",
}
MATCH_STATE_COLORS = {
    "exact": "#43A047",
    "nearest": "#FB8C00",
    "no_match": "#CFD8DC",
}
BADGE_COLORS = {
    "neutral": "#ECEFF1",
    "info": "#E3F2FD",
    "stage": "#E8EAF6",
    "status": "#E8F5E9",
    "mode": "#F3E5F5",
    "timestamp": "#E8EAF6",
    "trajectory_yes": "#E8F5E9",
    "trajectory_no": "#F5F5F5",
    "warning": "#FFF3E0",
}


@dataclass
class LogEvent:
    event_type: str
    frame: int
    label: str


@dataclass
class TimestampMatch:
    source_index: int
    target_index: int
    source_timestamp: int
    target_timestamp: int
    delta: int
    match_kind: str


@dataclass
class LogFilters:
    start_frame: int = 0
    end_frame: Optional[int] = None
    stages: Optional[set[int]] = None
    statuses: Optional[set[int]] = None
    modes: Optional[set[int]] = None
    event_types: tuple[str, ...] = ALL_EVENT_TYPES


@dataclass
class LogDataset:
    source_path: str
    file_display_name: str
    payload: dict
    frame_count: int
    file_boundaries: list[tuple[int, int, str]]
    fixed_xlim: tuple[float, float]
    fixed_ylim: tuple[float, float]


@dataclass
class LogPageState:
    data: LogDataset
    filters: LogFilters
    pause_on_event: bool = False
    valid_frames: list[int] = field(default_factory=list)
    events: list[LogEvent] = field(default_factory=list)
    current_index: int = 0
    playing: bool = False
    show_arrows: bool = False
    speed: float = 1.0
    user_adjusted_view: bool = False
    view_xlim: Optional[tuple[float, float]] = None
    view_ylim: Optional[tuple[float, float]] = None

    def current_frame(self) -> Optional[int]:
        if not self.valid_frames:
            return None
        self.current_index = max(0, min(self.current_index, len(self.valid_frames) - 1))
        return self.valid_frames[self.current_index]

    def rebuild(self, preferred_frame: Optional[int] = None) -> None:
        explicit_target = preferred_frame is not None
        if preferred_frame is None:
            preferred_frame = self.current_frame()

        self.valid_frames = [
            frame
            for frame in range(self.data.frame_count)
            if _frame_matches_log_filters(self.data.payload, frame, self.filters)
        ]
        self.events = _build_log_events(self.data, self.filters, set(self.valid_frames))

        if not self.valid_frames:
            self.current_index = 0
            return

        if preferred_frame is None:
            target = _find_first_interesting_frame(self.valid_frames, self.data.payload, self.filters.start_frame)
            if target is None:
                target = self.filters.start_frame
        else:
            target = preferred_frame

        if target in self.valid_frames:
            self.current_index = self.valid_frames.index(target)
            return

        if not explicit_target:
            interesting_target = _find_first_interesting_frame(self.valid_frames, self.data.payload, target)
            if interesting_target in self.valid_frames:
                self.current_index = self.valid_frames.index(interesting_target)
                return

        insert_at = bisect.bisect_left(self.valid_frames, target)
        if insert_at >= len(self.valid_frames):
            self.current_index = len(self.valid_frames) - 1
        else:
            self.current_index = insert_at

    def jump_to_frame(self, frame: int) -> None:
        if not self.valid_frames:
            return
        if frame in self.valid_frames:
            self.current_index = self.valid_frames.index(frame)
            return
        insert_at = bisect.bisect_left(self.valid_frames, frame)
        if insert_at >= len(self.valid_frames):
            self.current_index = len(self.valid_frames) - 1
        else:
            self.current_index = insert_at


@dataclass
class CsvDataset:
    csv_path: str
    groups: list[dict]
    grid_size: int
    resolution: float


@dataclass
class CsvPageState:
    data: CsvDataset
    trajectory_filter: str = "all"
    filtered_indices: list[int] = field(default_factory=list)
    current_index: int = 0

    def current_group_index(self) -> Optional[int]:
        if not self.filtered_indices:
            return None
        self.current_index = max(0, min(self.current_index, len(self.filtered_indices) - 1))
        return self.filtered_indices[self.current_index]

    def rebuild(self, preferred_group: Optional[int] = None) -> None:
        if preferred_group is None:
            preferred_group = self.current_group_index()

        self.filtered_indices = [
            index
            for index, group in enumerate(self.data.groups)
            if _group_matches_trajectory_filter(group, self.trajectory_filter)
        ]

        if not self.filtered_indices:
            self.current_index = 0
            return

        if preferred_group in self.filtered_indices:
            self.current_index = self.filtered_indices.index(preferred_group)
            return

        target = 0 if preferred_group is None else preferred_group
        insert_at = bisect.bisect_left(self.filtered_indices, target)
        if insert_at >= len(self.filtered_indices):
            self.current_index = len(self.filtered_indices) - 1
        else:
            self.current_index = insert_at

    def jump_to_group(self, group_index: int) -> bool:
        if group_index in self.filtered_indices:
            self.current_index = self.filtered_indices.index(group_index)
            return True
        return False


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _safe_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(text)
    except (TypeError, ValueError):
        return None


def _build_sorted_timestamp_pairs(raw_pairs: list[tuple[int, object]]) -> list[tuple[int, int]]:
    normalized_pairs: list[tuple[int, int]] = []
    for index, raw_timestamp in raw_pairs:
        timestamp = _safe_int(raw_timestamp)
        if timestamp is None or timestamp <= 0:
            continue
        normalized_pairs.append((index, timestamp))
    normalized_pairs.sort(key=lambda item: (item[1], item[0]))
    return normalized_pairs


def discover_default_log_file(base_dir: str = ".") -> Optional[str]:
    candidates = []
    for pattern in ("planning.log.*", "*.log"):
        candidates.extend(str(path) for path in Path(base_dir).glob(pattern) if path.is_file())
    candidates = _dedupe_preserve_order(candidates)
    if not candidates:
        return None
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def discover_default_csv_file(base_dir: str = ".") -> Optional[str]:
    default_path = Path(base_dir) / "planner_inputs.csv"
    if default_path.is_file():
        return str(default_path)
    candidates = [str(path) for path in Path(base_dir).glob("*.csv") if path.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def _payload_value(payload: dict, key: str, frame: int):
    values = payload.get(key, [])
    if frame < len(values):
        return values[frame]
    return None


def _parse_named_filter(raw: Optional[str], mapping: dict[int, str]) -> Optional[set[int]]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None

    reverse_mapping = {name.lower(): value for value, name in mapping.items()}
    result = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lstrip("-").isdigit():
            result.add(int(token))
            continue
        lookup = reverse_mapping.get(token.lower())
        if lookup is None:
            raise ValueError(f"未知过滤值: {token}")
        result.add(lookup)
    return result or None


def _parse_event_types(raw: Optional[str]) -> tuple[str, ...]:
    if raw is None:
        return ALL_EVENT_TYPES
    raw = raw.strip()
    if not raw:
        return ALL_EVENT_TYPES
    result = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in ALL_EVENT_TYPES:
            raise ValueError(f"未知事件类型: {token}")
        result.append(token)
    if not result:
        return ALL_EVENT_TYPES
    return tuple(_dedupe_preserve_order(result))


def _frame_matches_log_filters(payload: dict, frame: int, filters: LogFilters) -> bool:
    if frame < filters.start_frame:
        return False
    if filters.end_frame is not None and frame > filters.end_frame:
        return False

    stage = _payload_value(payload, "parking_function_stages", frame)
    status = _payload_value(payload, "parking_function_statuses", frame)
    mode = _payload_value(payload, "parking_function_modes", frame)

    if filters.stages is not None and stage not in filters.stages:
        return False
    if filters.statuses is not None and status not in filters.statuses:
        return False
    if filters.modes is not None and mode not in filters.modes:
        return False
    return True


def _is_reasonable_point(x_coord: float, y_coord: float, limit: float = 50000.0) -> bool:
    return abs(x_coord) <= limit and abs(y_coord) <= limit


def _collect_log_frame_points(payload: dict, frame: int) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    def add_point(x_coord, y_coord, limit: float = 50000.0) -> None:
        if x_coord is None or y_coord is None:
            return
        if not _is_reasonable_point(float(x_coord), float(y_coord), limit=limit):
            return
        points.append((float(x_coord), float(y_coord)))

    def add_pose(pose) -> None:
        if not pose:
            return
        x_coord, y_coord, yaw_coord = pose
        for corner_x, corner_y in legacy_log.calculate_vehicle_corners(x_coord, y_coord, yaw_coord):
            add_point(corner_x, corner_y)
        add_point(x_coord, y_coord)

    vehicle_location = _payload_value(payload, "vehicle_locations", frame)
    if vehicle_location and (abs(vehicle_location[0]) > 1 or abs(vehicle_location[1]) > 1 or abs(vehicle_location[2]) > 1):
        add_pose(vehicle_location)

    for key in ("plan_stage_target_poses", "plan_final_target_poses"):
        pose = _payload_value(payload, key, frame)
        if pose and (abs(pose[0]) > 1 or abs(pose[1]) > 1 or abs(pose[2]) > 1):
            add_pose(pose)

    slot_corners = _payload_value(payload, "slot_corners", frame)
    if slot_corners:
        for x_coord, y_coord in slot_corners:
            if abs(x_coord) > 1 or abs(y_coord) > 1:
                add_point(x_coord, y_coord)

    coordinates = _payload_value(payload, "coordinates", frame)
    if coordinates:
        for point in coordinates:
            if len(point) >= 2 and (abs(point[0]) > 1 or abs(point[1]) > 1):
                add_point(point[0], point[1])

    p0_p5_points = _payload_value(payload, "p0_p5_points", frame)
    if p0_p5_points:
        p0, p5 = p0_p5_points
        if abs(p0[0]) > 1 or abs(p0[1]) > 1:
            add_point(p0[0], p0[1])
        if abs(p5[0]) > 1 or abs(p5[1]) > 1:
            add_point(p5[0], p5[1])

    realtime_points = _payload_value(payload, "realtime_parkingspaces", frame)
    if realtime_points:
        for x_coord, y_coord in realtime_points:
            if abs(x_coord) > 1 or abs(y_coord) > 1:
                add_point(x_coord, y_coord)

    chamfer_points = _payload_value(payload, "parking_space_chamfers", frame)
    if chamfer_points:
        for point in chamfer_points:
            if len(point) >= 2 and (abs(point[0]) > 1 or abs(point[1]) > 1):
                add_point(point[0], point[1], limit=20000.0)

    target_corners_abcd = _payload_value(payload, "target_corners_abcd", frame)
    if target_corners_abcd:
        for point in target_corners_abcd.values():
            if len(point) >= 2 and (abs(point[0]) > 1 or abs(point[1]) > 1):
                add_point(point[0], point[1])

    return points


def _frame_has_visible_content(payload: dict, frame: int) -> bool:
    return bool(_collect_log_frame_points(payload, frame))


def _find_first_interesting_frame(valid_frames: list[int], payload: dict, start_frame: int = 0) -> Optional[int]:
    if not valid_frames:
        return None
    for frame in valid_frames:
        if frame < start_frame:
            continue
        if _frame_has_visible_content(payload, frame):
            return frame
    for frame in valid_frames:
        if _frame_has_visible_content(payload, frame):
            return frame
    return None


def _compute_log_frame_view(payload: dict, frame: int, fallback_xlim: tuple[float, float], fallback_ylim: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = _collect_log_frame_points(payload, frame)
    if not points:
        return fallback_xlim, fallback_ylim

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    x_span = max(x_max - x_min, 4000.0)
    y_span = max(y_max - y_min, 4000.0)
    x_margin = x_span * 0.25
    y_margin = y_span * 0.25
    return (x_min - x_margin, x_max + x_margin), (y_min - y_margin, y_max + y_margin)


def _build_log_events(dataset: LogDataset, filters: LogFilters, valid_frames: set[int]) -> list[LogEvent]:
    payload = dataset.payload
    events: list[LogEvent] = []

    if "search2park" in filters.event_types:
        stages = payload.get("parking_function_stages", [])
        for frame in range(1, len(stages)):
            if frame not in valid_frames:
                continue
            if stages[frame - 1] == 1 and stages[frame] == 2:
                events.append(LogEvent("search2park", frame, "SEARCH -> PARK"))

    if "replan" in filters.event_types:
        replan_ids = payload.get("replan_ids", [])
        for frame in range(1, len(replan_ids)):
            if frame not in valid_frames:
                continue
            if replan_ids[frame] > replan_ids[frame - 1]:
                events.append(LogEvent("replan", frame, f"Replan {replan_ids[frame - 1]} -> {replan_ids[frame]}"))

    if "stop_reason" in filters.event_types:
        reasons = payload.get("vehicle_stop_reasons", [])
        for frame in range(1, len(reasons)):
            if frame not in valid_frames:
                continue
            if reasons[frame] != reasons[frame - 1]:
                prev_label = legacy_log.control_stop_reason_mapping.get(reasons[frame - 1], str(reasons[frame - 1]))
                curr_label = legacy_log.control_stop_reason_mapping.get(reasons[frame], str(reasons[frame]))
                events.append(LogEvent("stop_reason", frame, f"Stop {prev_label} -> {curr_label}"))

    if "status" in filters.event_types:
        statuses = payload.get("parking_function_statuses", [])
        for frame in range(1, len(statuses)):
            if frame not in valid_frames:
                continue
            if statuses[frame] != statuses[frame - 1]:
                prev_label = legacy_log.func_status_mapping.get(statuses[frame - 1], str(statuses[frame - 1]))
                curr_label = legacy_log.func_status_mapping.get(statuses[frame], str(statuses[frame]))
                events.append(LogEvent("status", frame, f"Status {prev_label} -> {curr_label}"))

    if "file_boundary" in filters.event_types:
        for start_frame, _end_frame, filename in dataset.file_boundaries[1:]:
            if start_frame in valid_frames:
                events.append(LogEvent("file_boundary", start_frame, f"Switch File: {filename}"))

    events.sort(key=lambda item: item.frame)
    return events


def _find_best_timestamp_match(
    source_index: int,
    source_timestamp: Optional[int],
    target_pairs: list[tuple[int, int]],
    tolerance: int = TIMESTAMP_MATCH_TOLERANCE,
) -> Optional[TimestampMatch]:
    if source_timestamp is None or source_timestamp <= 0 or not target_pairs:
        return None

    timestamps = [timestamp for _index, timestamp in target_pairs]
    insert_at = bisect.bisect_left(timestamps, source_timestamp)
    best_match = None

    for candidate_pos in (insert_at - 1, insert_at, insert_at + 1):
        if not (0 <= candidate_pos < len(target_pairs)):
            continue
        target_index, target_timestamp = target_pairs[candidate_pos]
        delta = target_timestamp - source_timestamp
        abs_delta = abs(delta)
        if abs_delta > tolerance:
            continue
        if best_match is None or abs_delta < abs(best_match.delta):
            match_kind = "exact" if delta == 0 else "nearest"
            best_match = TimestampMatch(
                source_index=source_index,
                target_index=target_index,
                source_timestamp=source_timestamp,
                target_timestamp=target_timestamp,
                delta=delta,
                match_kind=match_kind,
            )

    return best_match


def _compute_log_frame_count(payload: dict) -> int:
    keys = (
        "parking_function_statuses",
        "path_current_segment_ids",
        "vehicle_locations",
        "coordinates",
        "parking_function_stages",
        "parking_function_modes",
    )
    return max((len(payload.get(key, [])) for key in keys), default=0)


def _compute_fixed_view(payload: dict, frame_count: int) -> tuple[tuple[float, float], tuple[float, float]]:
    all_x = []
    all_y = []

    for frame in range(frame_count):
        parking_space = _payload_value(payload, "parking_spaces", frame)
        if parking_space:
            all_x.extend(point[0] for point in parking_space)
            all_y.extend(point[1] for point in parking_space)

        slot_corners = _payload_value(payload, "slot_corners", frame)
        if slot_corners:
            all_x.extend(point[0] for point in slot_corners)
            all_y.extend(point[1] for point in slot_corners)

        vehicle_location = _payload_value(payload, "vehicle_locations", frame)
        if vehicle_location:
            all_x.append(vehicle_location[0])
            all_y.append(vehicle_location[1])

        coordinates = _payload_value(payload, "coordinates", frame)
        if coordinates:
            all_x.extend(point[0] for point in coordinates if len(point) >= 2)
            all_y.extend(point[1] for point in coordinates if len(point) >= 2)

        p0_p5 = _payload_value(payload, "p0_p5_points", frame)
        if p0_p5:
            p0, p5 = p0_p5
            all_x.extend([p0[0], p5[0]])
            all_y.extend([p0[1], p5[1]])

    if not all_x or not all_y:
        return (-10000.0, 10000.0), (-10000.0, 10000.0)

    max_reasonable_coord = 50000
    filtered_x = [x for x in all_x if abs(x) < max_reasonable_coord]
    filtered_y = [y for y in all_y if abs(y) < max_reasonable_coord]
    if len(filtered_x) < len(all_x) * 0.5:
        max_reasonable_coord = 500000
        filtered_x = [x for x in all_x if abs(x) < max_reasonable_coord]
        filtered_y = [y for y in all_y if abs(y) < max_reasonable_coord]

    if filtered_x and filtered_y:
        x_min, x_max = min(filtered_x), max(filtered_x)
        y_min, y_max = min(filtered_y), max(filtered_y)
    else:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

    x_margin = (x_max - x_min) * 0.15 or 1500
    y_margin = (y_max - y_min) * 0.15 or 1500
    return (x_min - x_margin, x_max + x_margin), (y_min - y_margin, y_max + y_margin)


def _extract_single_log_payload(file_path: str) -> dict:
    return {
        "parking_spaces": legacy_log.extract_parking_spaces(file_path),
        "slot_corners": legacy_log.extract_slot_corners(file_path),
        "vehicle_locations": legacy_log.extract_vehicle_locations(file_path),
        "plan_frame_ids": legacy_log.extract_plan_frame_id(file_path),
        "parking_function_statuses": legacy_log.extract_parking_function_status(file_path),
        "vehicle_stop_reasons": legacy_log.extract_vehicle_stop_reason(file_path),
        "parking_space_chamfers": legacy_log.extract_parking_space_chamfer(file_path),
        "plan_stage_target_poses": legacy_log.extract_plan_stage_target_pose(file_path),
        "plan_final_target_poses": legacy_log.extract_plan_final_target_pose(file_path),
        "path_current_segment_ids": legacy_log.extract_path_current_segment_id(file_path),
        "replan_ids": legacy_log.extract_replan_id(file_path),
        "coordinates": legacy_log.extract_xy_coordinates(file_path),
        "timestamps": legacy_log.extract_timestamps(file_path),
        "p0_p5_points": legacy_log.extract_parking_space_p0_p5(file_path),
        "gear_info": legacy_log.extract_gear_info(file_path),
        "parking_function_stages": legacy_log.extract_parking_function_stage(file_path),
        "parking_function_modes": legacy_log.extract_parking_function_mode(file_path),
        "control_work_modes": legacy_log.extract_control_work_mode(file_path),
        "vehicle_moving_statuses": legacy_log.extract_vehicle_moving_status(file_path),
        "perception_fusion_timestamps": legacy_log.extract_perception_fusion_timestamps(file_path),
        "stopper_distances": legacy_log.extract_stopper_dis_record(file_path),
        "target_corners_abcd": legacy_log.extract_target_slot_corners_abcd(file_path),
        "realtime_parkingspaces": legacy_log.extract_realtime_parkingspace(file_path),
        "fork_star_starts": legacy_log.extract_fork_star_starts(file_path),
        "parking_tasks": legacy_log.extract_parking_tasks(file_path),
    }


def load_log_dataset(log_path: Optional[str], merge_enabled: bool = True) -> Optional[LogDataset]:
    resolved_path = log_path or discover_default_log_file()
    if not resolved_path:
        return None

    resolved_path = str(Path(resolved_path).resolve())
    log_files = legacy_log.find_and_sort_log_files(resolved_path)
    file_boundaries: list[tuple[int, int, str]] = []

    if merge_enabled and len(log_files) > 1:
        payload, file_boundaries = legacy_log.merge_log_data(log_files)
        file_display_name = f"{os.path.basename(log_files[0])} ~ {os.path.basename(log_files[-1])} ({len(log_files)} files)"
    else:
        payload = _extract_single_log_payload(resolved_path)
        file_display_name = os.path.basename(resolved_path)

    frame_count = _compute_log_frame_count(payload)
    fixed_xlim, fixed_ylim = _compute_fixed_view(payload, frame_count)
    return LogDataset(
        source_path=resolved_path,
        file_display_name=file_display_name,
        payload=payload,
        frame_count=frame_count,
        file_boundaries=file_boundaries,
        fixed_xlim=fixed_xlim,
        fixed_ylim=fixed_ylim,
    )


def _group_matches_trajectory_filter(group: dict, trajectory_filter: str) -> bool:
    has_trajectory = bool(group.get("has_trajectory"))
    if trajectory_filter == "with":
        return has_trajectory
    if trajectory_filter == "without":
        return not has_trajectory
    return True


def load_csv_dataset(csv_path: Optional[str], grid_size: int, resolution: float) -> Optional[CsvDataset]:
    resolved_path = csv_path or discover_default_csv_file()
    if not resolved_path:
        return None
    resolved_path = str(Path(resolved_path).resolve())
    groups = legacy_csv.split_csv_by_blank_lines(resolved_path)
    parsed_groups = [legacy_csv.parse_group(group_lines, grid_size=grid_size) for group_lines in groups]
    return CsvDataset(
        csv_path=resolved_path,
        groups=parsed_groups,
        grid_size=grid_size,
        resolution=resolution,
    )


def _get_current_log_name(dataset: LogDataset, frame: int) -> str:
    for start_frame, end_frame, filename in dataset.file_boundaries:
        if start_frame <= frame <= end_frame:
            return filename
    return os.path.basename(dataset.source_path)


def _collect_rear_axle_trail(payload: dict, frame: int, history_limit: int = 240) -> list[tuple[float, float]]:
    vehicle_locations = payload.get("vehicle_locations", [])
    segment_ids = payload.get("path_current_segment_ids", [])
    statuses = payload.get("parking_function_statuses", [])
    if frame >= len(vehicle_locations):
        return []

    current_segment = segment_ids[frame] if frame < len(segment_ids) else None
    previous_segment = None if current_segment is None else current_segment - 1
    points = []
    lower_bound = max(0, frame - history_limit)
    for index in range(frame, lower_bound - 1, -1):
        if index < len(statuses) and statuses[index] == 0 and index != frame:
            break
        if index >= len(vehicle_locations):
            continue
        segment_id = segment_ids[index] if index < len(segment_ids) else None
        if current_segment is not None and segment_id not in {current_segment, previous_segment}:
            continue
        x_vehicle, y_vehicle, _yaw_vehicle = vehicle_locations[index]
        points.append((x_vehicle, y_vehicle))
    points.reverse()
    return points


def _collect_arrow_history(payload: dict, frame: int, history_limit: int = 12) -> list[tuple[float, float, float]]:
    vehicle_locations = payload.get("vehicle_locations", [])
    if frame >= len(vehicle_locations):
        return []
    start = max(0, frame - history_limit)
    return vehicle_locations[start : frame + 1]


def _draw_slot_corners(ax, slot_points) -> None:
    if not slot_points:
        return
    x_slot = [point[0] for point in slot_points] + [slot_points[0][0]]
    y_slot = [point[1] for point in slot_points] + [slot_points[0][1]]
    ax.plot(
        x_slot,
        y_slot,
        marker="s",
        markersize=5,
        linestyle="--",
        color=legacy_log.COLORS["slot_corners"],
        linewidth=2,
        label="Slot Corners",
        markerfacecolor=legacy_log.COLORS["slot_corners"],
        markeredgecolor="white",
        markeredgewidth=1,
        alpha=0.9,
    )
    ax.fill(x_slot, y_slot, color=legacy_log.COLORS["slot_corners"], alpha=0.15)
    for label, (x_coord, y_coord) in zip(("A", "B", "C", "D"), slot_points):
        ax.text(
            x_coord,
            y_coord,
            label,
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="circle,pad=0.3",
                facecolor=legacy_log.COLORS["slot_corners"],
                edgecolor="white",
                linewidth=1.5,
                alpha=0.9,
            ),
            ha="center",
            va="center",
            zorder=15,
        )


def _draw_p0_p5(ax, p0_p5_points) -> None:
    if not p0_p5_points:
        return
    p0, p5 = p0_p5_points
    for label, point in (("P0", p0), ("P5", p5)):
        ax.plot(
            point[0],
            point[1],
            marker="D",
            color=legacy_log.COLORS["p0_p5_fused"],
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=10,
        )
        ax.text(
            point[0],
            point[1] + 200,
            label,
            color=legacy_log.COLORS["p0_p5_fused"],
            fontsize=10,
            ha="center",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )


def _draw_realtime_parkingspace(ax, realtime_points) -> None:
    if not realtime_points or len(realtime_points) != 8:
        return
    x_rt = [point[0] for point in realtime_points] + [realtime_points[0][0]]
    y_rt = [point[1] for point in realtime_points] + [realtime_points[0][1]]
    ax.plot(x_rt, y_rt, linestyle="-.", linewidth=2, color="#00CED1", alpha=0.7, zorder=13, label="Realtime Space")
    ax.fill(x_rt, y_rt, color="#00CED1", alpha=0.08, zorder=12)
    for x_coord, y_coord in realtime_points:
        if abs(x_coord) <= 1 and abs(y_coord) <= 1:
            continue
        ax.plot(
            x_coord,
            y_coord,
            marker="D",
            markersize=4,
            color="#00CED1",
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=14,
        )


def _draw_stopper_distance(ax, stopper_dist, corners_abcd) -> None:
    if stopper_dist is None or not corners_abcd or "C" not in corners_abcd or "D" not in corners_abcd:
        return
    c_point = corners_abcd["C"]
    d_point = corners_abcd["D"]
    cd_mid_x = (c_point[0] + d_point[0]) / 2
    cd_mid_y = (c_point[1] + d_point[1]) / 2
    cd_vec_x = c_point[0] - d_point[0]
    cd_vec_y = c_point[1] - d_point[1]
    cd_length = np.sqrt(cd_vec_x**2 + cd_vec_y**2)
    if cd_length <= 0:
        return
    normal_x = -cd_vec_y / cd_length
    normal_y = cd_vec_x / cd_length
    stopper_x = cd_mid_x + normal_x * stopper_dist
    stopper_y = cd_mid_y + normal_y * stopper_dist
    ax.plot(
        [stopper_x, cd_mid_x],
        [stopper_y, cd_mid_y],
        color="#FF6B6B",
        linewidth=2.2,
        alpha=0.85,
        zorder=16,
        label="Stopper Distance",
    )
    ax.plot(stopper_x, stopper_y, marker="o", color="#FF6B6B", markersize=7, markeredgecolor="white", markeredgewidth=1.5, zorder=17)
    ax.text(
        (stopper_x + cd_mid_x) / 2,
        (stopper_y + cd_mid_y) / 2,
        f"Stopper {stopper_dist:.1f}mm",
        fontsize=10,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FF6B6B", edgecolor="white", linewidth=1.4, alpha=0.9),
        ha="center",
        va="center",
        zorder=18,
    )


def _draw_chamfer(ax, chamfer_points) -> None:
    if not chamfer_points or len(chamfer_points) < 4:
        return

    def sanitize(point):
        x_coord, y_coord = point
        if abs(x_coord) > 20000:
            x_coord = 0
        if abs(y_coord) > 20000:
            y_coord = 0
        return x_coord, y_coord

    p0_aisle, p0_slot, p5_aisle, p5_slot = [sanitize(point) for point in chamfer_points[:4]]
    ax.plot([p0_aisle[0], p0_slot[0]], [p0_aisle[1], p0_slot[1]], linestyle="-", color="black", linewidth=1.5)
    ax.plot([p5_aisle[0], p5_slot[0]], [p5_aisle[1], p5_slot[1]], linestyle="-", color="black", linewidth=1.5)


def _draw_target_pose(ax, pose, color: str, linestyle: str, label: str, marker_size: int) -> None:
    if not pose:
        return
    x_target, y_target, yaw_target = pose
    target_corners = legacy_log.calculate_vehicle_corners(x_target, y_target, yaw_target)
    x_target_contour = [point[0] for point in target_corners]
    y_target_contour = [point[1] for point in target_corners]
    ax.plot(x_target_contour, y_target_contour, color=color, linestyle=linestyle, linewidth=2.4, label=label, alpha=0.8, zorder=4)
    ax.fill(x_target_contour, y_target_contour, color=color, alpha=0.2)
    ax.plot(x_target, y_target, marker="*", color=color, markersize=marker_size, markeredgecolor="white", markeredgewidth=1.5, zorder=5)


def _format_filter_values(values: Optional[set[int]], mapping: dict[int, str]) -> str:
    if not values:
        return "ALL"
    labels = [mapping.get(value, str(value)) for value in sorted(values)]
    return ",".join(labels)


def _build_log_info_lines(state: LogPageState, frame: int, linked_csv_match: Optional[TimestampMatch] = None) -> list[str]:
    payload = state.data.payload
    current_file = _get_current_log_name(state.data, frame)
    current_event = next((event for event in state.events if event.frame == frame), None)
    current_event_pos = next((index for index, event in enumerate(state.events) if event.frame == frame), None)

    lines = [
        f"Source: {state.data.file_display_name}",
        f"Current file: {current_file}",
        f"Frame: {frame}",
        f"Frame range: {state.filters.start_frame} - {state.filters.end_frame if state.filters.end_frame is not None else state.data.frame_count - 1}",
        f"Stage filter: {_format_filter_values(state.filters.stages, legacy_log.func_stage_mapping)}",
        f"Status filter: {_format_filter_values(state.filters.statuses, legacy_log.func_status_mapping)}",
        f"Mode filter: {_format_filter_values(state.filters.modes, legacy_log.func_mode_mapping)}",
        f"Event types: {', '.join(state.filters.event_types)}",
        f"Events: {len(state.events)} total",
        f"Current event: {current_event.label if current_event else 'None'}",
        f"Event index: {(current_event_pos + 1) if current_event_pos is not None else '-'} / {len(state.events)}",
        f"Pause on event: {'ON' if state.pause_on_event else 'OFF'}",
    ]

    if linked_csv_match is None:
        lines.append("Linked CSV: None")
    else:
        lines.append(
            f"Linked CSV: group {linked_csv_match.target_index + 1} ({linked_csv_match.match_kind}, dt={linked_csv_match.delta:+d})"
        )
        lines.append(f"Linked CSV ts: {linked_csv_match.target_timestamp}")

    timestamp = _payload_value(payload, "timestamps", frame)
    if timestamp is not None:
        lines.append(f"Vehicle time: {legacy_log.convert_timestamp_to_bj_time(timestamp)}")
    perception_time = _payload_value(payload, "perception_fusion_timestamps", frame)
    if perception_time is not None:
        lines.append(f"Perception ts: {perception_time}")
        lines.append(f"Perception time: {legacy_log.convert_timestamp_to_bj_time(perception_time)}")

    stage = _payload_value(payload, "parking_function_stages", frame)
    mode = _payload_value(payload, "parking_function_modes", frame)
    status = _payload_value(payload, "parking_function_statuses", frame)
    gear = _payload_value(payload, "gear_info", frame)
    stop_reason = _payload_value(payload, "vehicle_stop_reasons", frame)
    task = _payload_value(payload, "parking_tasks", frame)
    work_mode = _payload_value(payload, "control_work_modes", frame)
    moving_status = _payload_value(payload, "vehicle_moving_statuses", frame)
    replan_id = _payload_value(payload, "replan_ids", frame)
    segment_id = _payload_value(payload, "path_current_segment_ids", frame)
    stage_target = _payload_value(payload, "plan_stage_target_poses", frame)
    final_target = _payload_value(payload, "plan_final_target_poses", frame)
    vehicle_location = _payload_value(payload, "vehicle_locations", frame)

    lines.extend(
        [
            f"PF Stage: {legacy_log.func_stage_mapping.get(stage, stage)}",
            f"PF Mode: {legacy_log.func_mode_mapping.get(mode, mode)}",
            f"PF Status: {legacy_log.func_status_mapping.get(status, status)}",
            f"Gear: {gear}",
            f"Stop Reason: {legacy_log.control_stop_reason_mapping.get(stop_reason, stop_reason)}",
            f"Parking Task: {task}",
            f"Ctrl Mode: {legacy_log.control_work_mode_mapping.get(work_mode, work_mode)}",
            f"Veh Status: {legacy_log.vehicle_moving_status_mapping.get(moving_status, moving_status)}",
            f"Replan ID: {replan_id}",
            f"Path Segment: {segment_id}",
            f"Stage Target: {stage_target}",
            f"Final Target: {final_target}",
            f"Current Loc: {vehicle_location}",
        ]
    )
    return lines


def _build_csv_info_lines(state: CsvPageState, group_index: int, linked_log_match: Optional[TimestampMatch] = None) -> list[str]:
    group = state.data.groups[group_index]
    gridmap = group["gridmap"]

    lines = [
        f"CSV: {os.path.basename(state.data.csv_path)}",
        f"Group: {group_index + 1} / {len(state.data.groups)}",
        f"Filtered slot: {state.current_index + 1} / {len(state.filtered_indices)}",
        f"Trajectory filter: {state.trajectory_filter}",
        f"Timestamp: {group.get('timestamp') or '-'}",
        f"Has trajectory: {'YES' if group.get('has_trajectory') else 'NO'}",
        f"Trajectory points: {len(group.get('trajectory') or [])}",
        f"Occupied cells: {int(np.sum(gridmap == 128))}",
        f"Line count: {group.get('line_count')}",
        f"Ego pose: {group.get('ego_pose')}",
        f"Target pose: {group.get('target_pose')}",
    ]

    if linked_log_match is None:
        lines.insert(5, "Linked log: None")
        lines.insert(6, "Linked PF ts: -")
    else:
        lines.insert(
            5,
            f"Linked log: frame {linked_log_match.target_index} ({linked_log_match.match_kind}, dt={linked_log_match.delta:+d})",
        )
        lines.insert(6, f"Linked PF ts: {linked_log_match.target_timestamp}")

    return lines


def _describe_link_match(match: Optional[TimestampMatch], left_label: str, right_label: str) -> str:
    if match is None:
        return f"{left_label} -> {right_label}: no timestamp match"
    delta_text = "0" if match.delta == 0 else f"{match.delta:+d}"
    return (
        f"{left_label} -> {right_label} {match.target_index + 1 if right_label == 'CSV' else match.target_index} "
        f"({match.match_kind}, dt={delta_text})"
    )


def _match_state_name(match: Optional[TimestampMatch]) -> str:
    if match is None:
        return "no_match"
    return match.match_kind


def _match_state_label(match: Optional[TimestampMatch]) -> str:
    if match is None:
        return "No Match"
    if match.match_kind == "exact":
        return "Exact"
    return "Nearest"


def _shorten_text(value: object, limit: int = 24) -> str:
    text = "-" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _compact_event_badge(event: Optional[LogEvent]) -> str:
    if event is None:
        return "None"
    if event.event_type == "search2park":
        return "SEARCH>PARK"
    if event.event_type == "replan":
        return "Replan"
    if event.event_type == "stop_reason":
        return "Stop Change"
    if event.event_type == "status":
        return "Status Change"
    if event.event_type == "file_boundary":
        return "File Switch"
    return _shorten_text(event.label, limit=18)


def _csv_tick_positions(grid_size: int) -> np.ndarray:
    step = max(grid_size // 8, 1)
    ticks = np.arange(0, grid_size, step)
    if ticks.size == 0 or ticks[-1] != grid_size - 1:
        ticks = np.append(ticks, grid_size - 1)
    return np.unique(ticks.astype(int))


class PlannerToolboxApp:
    def __init__(self, log_state: Optional[LogPageState], csv_state: Optional[CsvPageState], start_page: str = "log"):
        self.log_state = log_state
        self.csv_state = csv_state
        self.current_page = "log"
        self.drag_start: Optional[tuple[float, float]] = None

        self.fig = plt.figure(figsize=(15, 10))
        self.fig.patch.set_facecolor("#F5F5F5")
        self.fig.canvas.manager.set_window_title("Parking Planning Toolbox")
        self.plot_ax = self.fig.add_axes([0.06, 0.22, 0.67, 0.68])
        self.detail_ax = self.fig.add_axes([0.55, 0.22, 0.18, 0.68])
        self.detail_ax.set_visible(False)
        self.log_timeline_ax = self.fig.add_axes([0.06, 0.13, 0.67, 0.05])
        self.log_timeline_ax.set_visible(False)
        self.csv_overview_ax = self.fig.add_axes([0.06, 0.13, 0.67, 0.05])
        self.csv_overview_ax.set_visible(False)
        self.header_text = self.fig.text(0.5, 0.975, "", ha="center", va="top", fontsize=13, fontweight="bold")
        self.footer_text = self.fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=9, color="#455A64")

        self.common_axes = []
        self.log_axes = []
        self.csv_axes = []
        self.log_slider: Optional[Slider] = None
        self.csv_slider: Optional[Slider] = None
        self.log_marker_lines = []
        self.synchronizing_log_slider = False
        self.synchronizing_csv_slider = False
        self.detail_open_by_page = {"log": False, "csv": False}
        self.csv_timestamp_pairs: list[tuple[int, int]] = []
        self.log_timestamp_pairs: list[tuple[int, int]] = []
        self.badge_artists = []
        self.auto_follow_exact = False
        self.syncing_auto_follow_controls = False

        self._build_common_controls()
        if self.log_state:
            self._build_log_controls()
            self._rebuild_log_slider()
        if self.csv_state:
            self._build_csv_controls()
            self._rebuild_csv_slider()
        self._build_timestamp_link_index()

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.animation = FuncAnimation(self.fig, self._on_animation_tick, interval=200, cache_frame_data=False)

        if start_page == "csv" and self.csv_state:
            self.switch_page("csv")
        elif self.log_state:
            self.switch_page("log")
        elif self.csv_state:
            self.switch_page("csv")
        else:
            raise RuntimeError("没有可用的数据页面")

    def _build_common_controls(self) -> None:
        self.page_log_button = None
        self.page_csv_button = None

        if self.log_state:
            ax_log = self.fig.add_axes([0.06, 0.92, 0.12, 0.04])
            self.page_log_button = Button(ax_log, "Log Replay", color="#90CAF9", hovercolor="#64B5F6")
            self.page_log_button.label.set_fontweight("bold")
            self.page_log_button.label.set_color("white")
            self.page_log_button.on_clicked(lambda _event: self.switch_page("log"))
            self.common_axes.append(ax_log)

        if self.csv_state:
            ax_csv = self.fig.add_axes([0.19, 0.92, 0.12, 0.04])
            self.page_csv_button = Button(ax_csv, "CSV Browser", color="#A5D6A7", hovercolor="#81C784")
            self.page_csv_button.label.set_fontweight("bold")
            self.page_csv_button.label.set_color("white")
            self.page_csv_button.on_clicked(lambda _event: self.switch_page("csv"))
            self.common_axes.append(ax_csv)

    def _build_log_controls(self) -> None:
        ax_prev_frame = self.fig.add_axes([0.76, 0.90, 0.05, 0.04])
        self.log_prev_frame_button = Button(ax_prev_frame, "<F", color="#B0BEC5", hovercolor="#90A4AE")
        self.log_prev_frame_button.on_clicked(lambda _event: self._step_log_frame(-1))

        ax_play = self.fig.add_axes([0.82, 0.90, 0.06, 0.04])
        self.log_play_button = Button(ax_play, "Play", color="#4CAF50", hovercolor="#43A047")
        self.log_play_button.label.set_color("white")
        self.log_play_button.label.set_fontweight("bold")
        self.log_play_button.on_clicked(lambda _event: self._toggle_log_play())

        ax_next_frame = self.fig.add_axes([0.89, 0.90, 0.05, 0.04])
        self.log_next_frame_button = Button(ax_next_frame, "F>", color="#B0BEC5", hovercolor="#90A4AE")
        self.log_next_frame_button.on_clicked(lambda _event: self._step_log_frame(1))

        ax_first_event = self.fig.add_axes([0.76, 0.84, 0.05, 0.04])
        self.log_first_event_button = Button(ax_first_event, "|<E", color="#FFCC80", hovercolor="#FFB74D")
        self.log_first_event_button.on_clicked(lambda _event: self._jump_log_event("first"))

        ax_prev_event = self.fig.add_axes([0.82, 0.84, 0.05, 0.04])
        self.log_prev_event_button = Button(ax_prev_event, "<E", color="#FFCC80", hovercolor="#FFB74D")
        self.log_prev_event_button.on_clicked(lambda _event: self._jump_log_event("prev"))

        ax_next_event = self.fig.add_axes([0.88, 0.84, 0.05, 0.04])
        self.log_next_event_button = Button(ax_next_event, "E>", color="#FFCC80", hovercolor="#FFB74D")
        self.log_next_event_button.on_clicked(lambda _event: self._jump_log_event("next"))

        ax_last_event = self.fig.add_axes([0.94, 0.84, 0.05, 0.04])
        self.log_last_event_button = Button(ax_last_event, "E>|", color="#FFCC80", hovercolor="#FFB74D")
        self.log_last_event_button.on_clicked(lambda _event: self._jump_log_event("last"))

        ax_speed = self.fig.add_axes([0.76, 0.78, 0.23, 0.03], facecolor="#ECEFF1")
        self.log_speed_slider = Slider(ax_speed, "Speed", 1.0, 10.0, valinit=self.log_state.speed, valstep=0.5, color="#FF9800")
        self.log_speed_slider.on_changed(self._on_log_speed_changed)

        ax_start = self.fig.add_axes([0.76, 0.72, 0.10, 0.04])
        self.log_start_box = TextBox(ax_start, "Start", initial=str(self.log_state.filters.start_frame))
        ax_end = self.fig.add_axes([0.89, 0.72, 0.10, 0.04])
        self.log_end_box = TextBox(ax_end, "End", initial="" if self.log_state.filters.end_frame is None else str(self.log_state.filters.end_frame))

        ax_stage = self.fig.add_axes([0.76, 0.66, 0.23, 0.04])
        self.log_stage_box = TextBox(ax_stage, "Stage", initial=_format_filter_values(self.log_state.filters.stages, legacy_log.func_stage_mapping))
        ax_status = self.fig.add_axes([0.76, 0.60, 0.23, 0.04])
        self.log_status_box = TextBox(ax_status, "Status", initial=_format_filter_values(self.log_state.filters.statuses, legacy_log.func_status_mapping))
        ax_mode = self.fig.add_axes([0.76, 0.54, 0.23, 0.04])
        self.log_mode_box = TextBox(ax_mode, "Mode", initial=_format_filter_values(self.log_state.filters.modes, legacy_log.func_mode_mapping))

        ax_apply = self.fig.add_axes([0.76, 0.48, 0.23, 0.04])
        self.log_apply_button = Button(ax_apply, "Apply Filters", color="#42A5F5", hovercolor="#1E88E5")
        self.log_apply_button.label.set_color("white")
        self.log_apply_button.label.set_fontweight("bold")
        self.log_apply_button.on_clicked(lambda _event: self._apply_log_filters())

        ax_events = self.fig.add_axes([0.76, 0.28, 0.23, 0.17])
        self.log_event_checks = CheckButtons(
            ax_events,
            labels=list(ALL_EVENT_TYPES),
            actives=[event_type in self.log_state.filters.event_types for event_type in ALL_EVENT_TYPES],
        )
        self.log_event_checks.on_clicked(self._toggle_log_event_type)
        ax_events.set_title("Events", fontsize=10, fontweight="bold")

        ax_pause = self.fig.add_axes([0.76, 0.23, 0.23, 0.04])
        self.log_pause_check = CheckButtons(ax_pause, labels=["pause on event"], actives=[self.log_state.pause_on_event])
        self.log_pause_check.on_clicked(self._toggle_pause_on_event)

        ax_auto_follow = self.fig.add_axes([0.76, 0.18, 0.23, 0.04])
        self.log_auto_follow_check = CheckButtons(ax_auto_follow, labels=["Auto Follow Exact"], actives=[self.auto_follow_exact])
        self.log_auto_follow_check.on_clicked(self._toggle_log_auto_follow)

        ax_yaw = self.fig.add_axes([0.76, 0.13, 0.23, 0.04])
        self.log_yaw_button = Button(ax_yaw, "Toggle Yaw Arrow", color="#26A69A", hovercolor="#00897B")
        self.log_yaw_button.label.set_color("white")
        self.log_yaw_button.label.set_fontweight("bold")
        self.log_yaw_button.on_clicked(lambda _event: self._toggle_log_yaw())

        ax_reset_view = self.fig.add_axes([0.76, 0.08, 0.23, 0.04])
        self.log_reset_view_button = Button(ax_reset_view, "Reset Auto View", color="#8D6E63", hovercolor="#6D4C41")
        self.log_reset_view_button.label.set_color("white")
        self.log_reset_view_button.label.set_fontweight("bold")
        self.log_reset_view_button.on_clicked(lambda _event: self._reset_log_view())

        ax_link_csv = self.fig.add_axes([0.76, 0.03, 0.11, 0.04])
        self.log_link_csv_button = Button(ax_link_csv, "Go CSV", color="#5C6BC0", hovercolor="#3F51B5")
        self.log_link_csv_button.label.set_color("white")
        self.log_link_csv_button.label.set_fontweight("bold")
        self.log_link_csv_button.on_clicked(lambda _event: self._jump_to_linked_csv())

        ax_detail_window = self.fig.add_axes([0.88, 0.03, 0.11, 0.04])
        self.log_detail_button = Button(ax_detail_window, "Show Details", color="#546E7A", hovercolor="#455A64")
        self.log_detail_button.label.set_color("white")
        self.log_detail_button.label.set_fontweight("bold")
        self.log_detail_button.on_clicked(lambda _event: self._toggle_detail_panel())

        self.log_axes.extend(
            [
                ax_prev_frame,
                ax_play,
                ax_next_frame,
                ax_first_event,
                ax_prev_event,
                ax_next_event,
                ax_last_event,
                ax_speed,
                ax_start,
                ax_end,
                ax_stage,
                ax_status,
                ax_mode,
                ax_apply,
                ax_events,
                ax_pause,
                ax_auto_follow,
                ax_yaw,
                ax_reset_view,
                ax_link_csv,
                ax_detail_window,
            ]
        )

    def _build_csv_controls(self) -> None:
        ax_prev = self.fig.add_axes([0.76, 0.90, 0.07, 0.04])
        self.csv_prev_button = Button(ax_prev, "<Group", color="#B0BEC5", hovercolor="#90A4AE")
        self.csv_prev_button.on_clicked(lambda _event: self._step_csv_group(-1))

        ax_next = self.fig.add_axes([0.84, 0.90, 0.07, 0.04])
        self.csv_next_button = Button(ax_next, "Group>", color="#B0BEC5", hovercolor="#90A4AE")
        self.csv_next_button.on_clicked(lambda _event: self._step_csv_group(1))

        ax_jump = self.fig.add_axes([0.92, 0.90, 0.07, 0.04])
        self.csv_jump_box = TextBox(ax_jump, "Jump", initial="1")
        self.csv_jump_box.on_submit(self._jump_csv_group_from_text)

        ax_radio = self.fig.add_axes([0.76, 0.73, 0.23, 0.12])
        self.csv_radio = RadioButtons(ax_radio, labels=("all", "with", "without"), active=("all", "with", "without").index(self.csv_state.trajectory_filter))
        self.csv_radio.on_clicked(self._set_csv_trajectory_filter)
        ax_radio.set_title("Trajectory Filter", fontsize=10, fontweight="bold")

        ax_go = self.fig.add_axes([0.76, 0.67, 0.23, 0.04])
        self.csv_go_button = Button(ax_go, "Go To Group", color="#42A5F5", hovercolor="#1E88E5")
        self.csv_go_button.label.set_color("white")
        self.csv_go_button.label.set_fontweight("bold")
        self.csv_go_button.on_clicked(lambda _event: self._jump_csv_group_from_text(self.csv_jump_box.text))

        ax_link_log = self.fig.add_axes([0.76, 0.61, 0.23, 0.04])
        self.csv_link_log_button = Button(ax_link_log, "Go Linked Log Frame", color="#5C6BC0", hovercolor="#3F51B5")
        self.csv_link_log_button.label.set_color("white")
        self.csv_link_log_button.label.set_fontweight("bold")
        self.csv_link_log_button.on_clicked(lambda _event: self._jump_to_linked_log())

        ax_auto_follow = self.fig.add_axes([0.76, 0.55, 0.23, 0.04])
        self.csv_auto_follow_check = CheckButtons(ax_auto_follow, labels=["Auto Follow Exact"], actives=[self.auto_follow_exact])
        self.csv_auto_follow_check.on_clicked(self._toggle_csv_auto_follow)

        ax_detail = self.fig.add_axes([0.76, 0.49, 0.23, 0.04])
        self.csv_detail_button = Button(ax_detail, "Show Details", color="#546E7A", hovercolor="#455A64")
        self.csv_detail_button.label.set_color("white")
        self.csv_detail_button.label.set_fontweight("bold")
        self.csv_detail_button.on_clicked(lambda _event: self._toggle_detail_panel())

        self.csv_axes.extend([ax_prev, ax_next, ax_jump, ax_radio, ax_go, ax_link_log, ax_auto_follow, ax_detail])

    def _build_timestamp_link_index(self) -> None:
        self.csv_timestamp_pairs = []
        self.log_timestamp_pairs = []

        if self.csv_state:
            self.csv_timestamp_pairs = _build_sorted_timestamp_pairs(
                [(group_index, group.get("timestamp")) for group_index, group in enumerate(self.csv_state.data.groups)]
            )

        if self.log_state:
            perception_timestamps = self.log_state.data.payload.get("perception_fusion_timestamps", [])
            self.log_timestamp_pairs = _build_sorted_timestamp_pairs(list(enumerate(perception_timestamps)))

    def _get_linked_csv_match(self, frame: Optional[int] = None) -> Optional[TimestampMatch]:
        if not self.log_state or not self.csv_state:
            return None
        if frame is None:
            frame = self.log_state.current_frame()
        timestamp = _safe_int(_payload_value(self.log_state.data.payload, "perception_fusion_timestamps", frame))
        return _find_best_timestamp_match(frame, timestamp, self.csv_timestamp_pairs)

    def _get_linked_log_match(self, group_index: Optional[int] = None) -> Optional[TimestampMatch]:
        if not self.csv_state or not self.log_state:
            return None
        if group_index is None:
            group_index = self.csv_state.current_group_index()
        if group_index is None:
            return None
        timestamp = _safe_int(self.csv_state.data.groups[group_index].get("timestamp"))
        return _find_best_timestamp_match(group_index, timestamp, self.log_timestamp_pairs)

    def _jump_to_linked_csv(self) -> None:
        if not self.csv_state or not self.log_state:
            return
        match = self._get_linked_csv_match()
        if match is None:
            print("当前日志帧没有找到可匹配的 CSV 组。")
            return
        if match.target_index not in self.csv_state.filtered_indices:
            self.csv_state.trajectory_filter = "all"
            self.csv_state.rebuild(preferred_group=match.target_index)
            self.csv_radio.set_active(0)
            self._rebuild_csv_slider()
        else:
            self.csv_state.jump_to_group(match.target_index)
        self.switch_page("csv", render_page=False, sync_context=False)
        self._sync_csv_slider()
        self.render()

    def _jump_to_linked_log(self) -> None:
        if not self.csv_state or not self.log_state:
            return
        match = self._get_linked_log_match()
        if match is None:
            print("当前 CSV 组没有找到可匹配的日志帧。")
            return

        if match.target_index not in self.log_state.valid_frames:
            self.log_state.filters.start_frame = 0
            self.log_state.filters.end_frame = None
            self.log_state.filters.stages = None
            self.log_state.filters.statuses = None
            self.log_state.filters.modes = None
            self.log_start_box.set_val("0")
            self.log_end_box.set_val("")
            self.log_stage_box.set_val("ALL")
            self.log_status_box.set_val("ALL")
            self.log_mode_box.set_val("ALL")
            self.log_state.rebuild(preferred_frame=match.target_index)
            self._rebuild_log_slider()
        else:
            self.log_state.jump_to_frame(match.target_index)

        self._focus_log_frame_view()
        self.switch_page("log", render_page=False, sync_context=False)
        self._sync_log_slider()
        self.render()

    def _maybe_auto_follow_from_log(self) -> None:
        if not self.auto_follow_exact or not self.log_state or not self.csv_state:
            return
        match = self._get_linked_csv_match()
        if match is None or match.match_kind != "exact":
            return
        if match.target_index not in self.csv_state.filtered_indices:
            return
        self.csv_state.jump_to_group(match.target_index)
        self._sync_csv_slider()

    def _maybe_auto_follow_from_csv(self) -> None:
        if not self.auto_follow_exact or not self.csv_state or not self.log_state:
            return
        match = self._get_linked_log_match()
        if match is None or match.match_kind != "exact":
            return
        if match.target_index not in self.log_state.valid_frames:
            return
        self.log_state.jump_to_frame(match.target_index)
        self._sync_log_slider()

    def _set_axes_visible(self, axes_list: list, visible: bool) -> None:
        for ax in axes_list:
            ax.set_visible(visible)

    def _current_detail_open(self) -> bool:
        return bool(self.detail_open_by_page.get(self.current_page, False))

    def _sync_auto_follow_controls(self) -> None:
        controls = []
        if hasattr(self, "log_auto_follow_check"):
            controls.append(self.log_auto_follow_check)
        if hasattr(self, "csv_auto_follow_check"):
            controls.append(self.csv_auto_follow_check)

        self.syncing_auto_follow_controls = True
        try:
            for control in controls:
                current_status = control.get_status()[0]
                if current_status != self.auto_follow_exact:
                    control.set_active(0)
        finally:
            self.syncing_auto_follow_controls = False

    def _set_auto_follow_exact(self, enabled: bool) -> None:
        self.auto_follow_exact = enabled
        self._sync_auto_follow_controls()
        if enabled:
            if self.current_page == "log":
                self._maybe_auto_follow_from_log()
            elif self.current_page == "csv":
                self._maybe_auto_follow_from_csv()
        self.render()

    def _toggle_log_auto_follow(self, _label: str) -> None:
        if self.syncing_auto_follow_controls:
            return
        self._set_auto_follow_exact(self.log_auto_follow_check.get_status()[0])

    def _toggle_csv_auto_follow(self, _label: str) -> None:
        if self.syncing_auto_follow_controls:
            return
        self._set_auto_follow_exact(self.csv_auto_follow_check.get_status()[0])

    def _refresh_page_buttons(self) -> None:
        if self.page_log_button:
            color = "#1E88E5" if self.current_page == "log" else "#90CAF9"
            self.page_log_button.ax.set_facecolor(color)
        if self.page_csv_button:
            color = "#2E7D32" if self.current_page == "csv" else "#A5D6A7"
            self.page_csv_button.ax.set_facecolor(color)

    def _refresh_detail_button_labels(self) -> None:
        if hasattr(self, "log_detail_button"):
            self.log_detail_button.label.set_text("Hide Details" if self.detail_open_by_page.get("log") else "Show Details")
        if hasattr(self, "csv_detail_button"):
            self.csv_detail_button.label.set_text("Hide Details" if self.detail_open_by_page.get("csv") else "Show Details")

    def _apply_content_layout(self) -> None:
        plot_left = 0.06
        plot_bottom = 0.22
        plot_height = 0.68
        plot_width = 0.47 if self._current_detail_open() else 0.67

        self.plot_ax.set_position([plot_left, plot_bottom, plot_width, plot_height])
        self.detail_ax.set_position([0.55, plot_bottom, 0.18, plot_height])
        self.detail_ax.set_visible(self._current_detail_open())

        self.log_timeline_ax.set_position([plot_left, 0.14, plot_width, 0.05])
        self.csv_overview_ax.set_position([plot_left, 0.14, plot_width, 0.05])
        self.log_timeline_ax.set_visible(self.current_page == "log")
        self.csv_overview_ax.set_visible(self.current_page == "csv")

        if self.log_slider:
            self.log_slider.ax.set_position([plot_left, 0.09, plot_width, 0.03])
        if self.csv_slider:
            self.csv_slider.ax.set_position([plot_left, 0.09, plot_width, 0.03])

        self._refresh_detail_button_labels()

    def switch_page(self, page: str, render_page: bool = True, sync_context: bool = True) -> None:
        if page == "log" and not self.log_state:
            return
        if page == "csv" and not self.csv_state:
            return
        previous_page = self.current_page
        if sync_context and previous_page != page:
            if previous_page == "log" and page == "csv":
                self._sync_csv_context_from_log()
            elif previous_page == "csv" and page == "log":
                self._sync_log_context_from_csv()
        self.current_page = page
        self._refresh_page_buttons()
        self._set_axes_visible(self.log_axes, page == "log")
        self._set_axes_visible(self.csv_axes, page == "csv")
        if self.log_slider:
            self.log_slider.ax.set_visible(page == "log")
        if self.csv_slider:
            self.csv_slider.ax.set_visible(page == "csv")
        self._apply_content_layout()
        if render_page:
            self.render()

    def _toggle_log_play(self) -> None:
        if not self.log_state or not self.log_state.valid_frames:
            return
        self.log_state.playing = not self.log_state.playing
        self.log_play_button.label.set_text("Pause" if self.log_state.playing else "Play")
        self.render()

    def _toggle_log_yaw(self) -> None:
        if not self.log_state:
            return
        self.log_state.show_arrows = not self.log_state.show_arrows
        self.render()

    def _reset_log_view(self) -> None:
        if not self.log_state:
            return
        self.log_state.user_adjusted_view = False
        self.log_state.view_xlim = None
        self.log_state.view_ylim = None
        self.render()

    def _focus_log_frame_view(self) -> None:
        if not self.log_state:
            return
        self.log_state.user_adjusted_view = False
        self.log_state.view_xlim = None
        self.log_state.view_ylim = None

    def _sync_csv_context_from_log(self) -> None:
        if not self.log_state or not self.csv_state or not self.log_state.valid_frames:
            return
        match = self._get_linked_csv_match(self.log_state.current_frame())
        if match is None or match.match_kind != "exact":
            return
        if match.target_index not in self.csv_state.filtered_indices:
            if self.csv_state.trajectory_filter != "all":
                self.csv_state.trajectory_filter = "all"
                if hasattr(self, "csv_radio"):
                    self.csv_radio.set_active(0)
                else:
                    self.csv_state.rebuild(preferred_group=match.target_index)
                    self._rebuild_csv_slider()
            elif match.target_index not in self.csv_state.filtered_indices:
                self.csv_state.rebuild(preferred_group=match.target_index)
                self._rebuild_csv_slider()
        self.csv_state.jump_to_group(match.target_index)
        self._sync_csv_slider()

    def _sync_log_context_from_csv(self) -> None:
        if not self.csv_state or not self.log_state or not self.csv_state.filtered_indices:
            return
        match = self._get_linked_log_match(self.csv_state.current_group_index())
        if match is None or match.match_kind != "exact":
            return
        if match.target_index not in self.log_state.valid_frames:
            self.log_state.filters.start_frame = 0
            self.log_state.filters.end_frame = None
            self.log_state.filters.stages = None
            self.log_state.filters.statuses = None
            self.log_state.filters.modes = None
            self.log_start_box.set_val("0")
            self.log_end_box.set_val("")
            self.log_stage_box.set_val("ALL")
            self.log_status_box.set_val("ALL")
            self.log_mode_box.set_val("ALL")
            self.log_state.rebuild(preferred_frame=match.target_index)
            self._rebuild_log_slider()
        else:
            self.log_state.jump_to_frame(match.target_index)
        self._focus_log_frame_view()
        self._sync_log_slider()

    def _toggle_detail_panel(self) -> None:
        self.detail_open_by_page[self.current_page] = not self.detail_open_by_page.get(self.current_page, False)
        self._apply_content_layout()
        self.render()

    def _step_log_frame(self, delta: int) -> None:
        if not self.log_state or not self.log_state.valid_frames:
            return
        self.log_state.playing = False
        self.log_play_button.label.set_text("Play")
        self.log_state.current_index = max(0, min(self.log_state.current_index + delta, len(self.log_state.valid_frames) - 1))
        self._sync_log_slider()
        self._maybe_auto_follow_from_log()
        self.render()

    def _jump_log_event(self, where: str) -> None:
        if not self.log_state or not self.log_state.events:
            return

        current_frame = self.log_state.current_frame()
        target_event = None
        if where == "first":
            target_event = self.log_state.events[0]
        elif where == "last":
            target_event = self.log_state.events[-1]
        elif where == "prev":
            for event in reversed(self.log_state.events):
                if current_frame is not None and event.frame < current_frame:
                    target_event = event
                    break
            if target_event is None:
                target_event = self.log_state.events[0]
        elif where == "next":
            for event in self.log_state.events:
                if current_frame is not None and event.frame > current_frame:
                    target_event = event
                    break
            if target_event is None:
                target_event = self.log_state.events[-1]

        if target_event is None:
            return

        self.log_state.playing = False
        self.log_play_button.label.set_text("Play")
        self.log_state.jump_to_frame(target_event.frame)
        self._sync_log_slider()
        self._maybe_auto_follow_from_log()
        self.render()

    def _on_log_speed_changed(self, value: float) -> None:
        if not self.log_state:
            return
        self.log_state.speed = value
        self.animation.event_source.interval = int(200 / value)

    def _toggle_log_event_type(self, label: str) -> None:
        if not self.log_state:
            return
        active = []
        for event_type, checked in zip(ALL_EVENT_TYPES, self.log_event_checks.get_status()):
            if checked:
                active.append(event_type)
        if not active:
            event_index = ALL_EVENT_TYPES.index(label)
            self.log_event_checks.set_active(event_index)
            return
        self.log_state.filters.event_types = tuple(active)
        self._recalculate_log_state()

    def _toggle_pause_on_event(self, _label: str) -> None:
        if not self.log_state:
            return
        self.log_state.pause_on_event = self.log_pause_check.get_status()[0]
        self.render()

    def _apply_log_filters(self) -> None:
        if not self.log_state:
            return
        try:
            start_text = self.log_start_box.text.strip()
            end_text = self.log_end_box.text.strip()
            self.log_state.filters.start_frame = int(start_text) if start_text else 0
            self.log_state.filters.end_frame = int(end_text) if end_text else None
            self.log_state.filters.stages = _parse_named_filter(self.log_stage_box.text, legacy_log.func_stage_mapping)
            self.log_state.filters.statuses = _parse_named_filter(self.log_status_box.text, legacy_log.func_status_mapping)
            self.log_state.filters.modes = _parse_named_filter(self.log_mode_box.text, legacy_log.func_mode_mapping)
        except ValueError as exc:
            print(f"过滤条件解析失败: {exc}")
            return
        self._recalculate_log_state()

    def _recalculate_log_state(self) -> None:
        if not self.log_state:
            return
        current_frame = self.log_state.current_frame()
        self.log_state.rebuild(preferred_frame=current_frame)
        self._rebuild_log_slider()
        self._maybe_auto_follow_from_log()
        self.render()

    def _rebuild_log_slider(self) -> None:
        if not self.log_state:
            return
        if self.log_slider:
            self.log_slider.ax.remove()
            self.log_slider = None
        self.log_marker_lines = []

        slider_ax = self.fig.add_axes([0.06, 0.08, 0.67, 0.03], facecolor="#E3F2FD")
        slider_max = max(len(self.log_state.valid_frames) - 1, 1)
        slider_value = min(self.log_state.current_index, slider_max)
        self.log_slider = Slider(slider_ax, "Log Frame", 0, slider_max, valinit=slider_value, valstep=1, color="#8E24AA")
        self.log_slider.on_changed(self._on_log_slider_changed)

        if self.current_page != "log":
            slider_ax.set_visible(False)

        position_map = {frame: index for index, frame in enumerate(self.log_state.valid_frames)}
        for event in self.log_state.events:
            position = position_map.get(event.frame)
            if position is None:
                continue
            color = {
                "search2park": "#E53935",
                "replan": "#FB8C00",
                "stop_reason": "#43A047",
                "status": "#3949AB",
                "file_boundary": "#00838F",
            }.get(event.event_type, "#455A64")
            line = slider_ax.axvline(position, color=color, linewidth=1.2, alpha=0.7, zorder=1)
            self.log_marker_lines.append(line)
        self._apply_content_layout()

    def _sync_log_slider(self) -> None:
        if not self.log_slider or not self.log_state:
            return
        self.synchronizing_log_slider = True
        try:
            self.log_slider.set_val(self.log_state.current_index)
        finally:
            self.synchronizing_log_slider = False

    def _on_log_slider_changed(self, value: float) -> None:
        if self.synchronizing_log_slider:
            return
        if not self.log_state or not self.log_state.valid_frames:
            return
        self.log_state.playing = False
        self.log_play_button.label.set_text("Play")
        self.log_state.current_index = int(value)
        self._maybe_auto_follow_from_log()
        self.render()

    def _step_csv_group(self, delta: int) -> None:
        if not self.csv_state or not self.csv_state.filtered_indices:
            return
        self.csv_state.current_index = max(0, min(self.csv_state.current_index + delta, len(self.csv_state.filtered_indices) - 1))
        self._sync_csv_slider()
        self._maybe_auto_follow_from_csv()
        self.render()

    def _set_csv_trajectory_filter(self, label: str) -> None:
        if not self.csv_state:
            return
        current_group = self.csv_state.current_group_index()
        self.csv_state.trajectory_filter = label
        self.csv_state.rebuild(preferred_group=current_group)
        self._rebuild_csv_slider()
        self._maybe_auto_follow_from_csv()
        self.render()

    def _jump_csv_group_from_text(self, text: str) -> None:
        if not self.csv_state:
            return
        try:
            group_number = int(text.strip())
        except ValueError:
            print(f"组号解析失败: {text}")
            return

        target_index = group_number - 1
        if target_index < 0 or target_index >= len(self.csv_state.data.groups):
            print(f"组号超出范围: {group_number}")
            return
        if not self.csv_state.jump_to_group(target_index):
            print(f"组 {group_number} 当前被轨迹筛选排除")
            return
        self._sync_csv_slider()
        self._maybe_auto_follow_from_csv()
        self.render()

    def _rebuild_csv_slider(self) -> None:
        if not self.csv_state:
            return
        if self.csv_slider:
            self.csv_slider.ax.remove()
            self.csv_slider = None
        slider_ax = self.fig.add_axes([0.06, 0.08, 0.67, 0.03], facecolor="#E8F5E9")
        slider_max = max(len(self.csv_state.filtered_indices) - 1, 1)
        slider_value = min(self.csv_state.current_index, slider_max)
        self.csv_slider = Slider(slider_ax, "CSV Group", 0, slider_max, valinit=slider_value, valstep=1, color="#2E7D32")
        self.csv_slider.on_changed(self._on_csv_slider_changed)
        if self.current_page != "csv":
            slider_ax.set_visible(False)
        self._apply_content_layout()

    def _sync_csv_slider(self) -> None:
        if not self.csv_slider or not self.csv_state:
            return
        self.synchronizing_csv_slider = True
        try:
            self.csv_slider.set_val(self.csv_state.current_index)
        finally:
            self.synchronizing_csv_slider = False

    def _on_csv_slider_changed(self, value: float) -> None:
        if self.synchronizing_csv_slider:
            return
        if not self.csv_state or not self.csv_state.filtered_indices:
            return
        self.csv_state.current_index = int(value)
        self._maybe_auto_follow_from_csv()
        self.render()

    def _jump_to_csv_group_from_overview(self, group_index: int) -> None:
        if not self.csv_state:
            return
        if not (0 <= group_index < len(self.csv_state.data.groups)):
            return

        if group_index not in self.csv_state.filtered_indices:
            if self.csv_state.trajectory_filter != "all":
                self.csv_radio.set_active(0)
            if group_index not in self.csv_state.filtered_indices:
                return

        self.csv_state.jump_to_group(group_index)
        self._sync_csv_slider()
        self._maybe_auto_follow_from_csv()
        self.render()

    def _handle_log_timeline_click(self, event) -> None:
        if not self.log_state or not self.log_state.valid_frames or event.xdata is None:
            return

        position_map = {frame: index for index, frame in enumerate(self.log_state.valid_frames)}
        event_positions = [(position_map[event_item.frame], event_item) for event_item in self.log_state.events if event_item.frame in position_map]
        target_position = max(0, min(int(round(event.xdata)), len(self.log_state.valid_frames) - 1))
        target_frame = self.log_state.valid_frames[target_position]

        if event_positions:
            nearest_position, nearest_event = min(event_positions, key=lambda item: abs(item[0] - event.xdata))
            if abs(nearest_position - event.xdata) <= 0.8:
                target_frame = nearest_event.frame

        self.log_state.playing = False
        self.log_play_button.label.set_text("Play")
        self.log_state.jump_to_frame(target_frame)
        self._sync_log_slider()
        self._maybe_auto_follow_from_log()
        self.render()

    def _handle_csv_overview_click(self, event) -> None:
        if not self.csv_state or event.xdata is None:
            return
        group_index = int(np.floor(event.xdata))
        self._jump_to_csv_group_from_overview(group_index)

    def _on_animation_tick(self, _frame_index) -> None:
        if self.current_page != "log" or not self.log_state or not self.log_state.playing or not self.log_state.valid_frames:
            return
        if self.log_state.current_index >= len(self.log_state.valid_frames) - 1:
            self.log_state.current_index = 0
        else:
            self.log_state.current_index += 1
        current_frame = self.log_state.current_frame()
        if self.log_state.pause_on_event and current_frame is not None:
            if any(event.frame == current_frame for event in self.log_state.events):
                self.log_state.playing = False
                self.log_play_button.label.set_text("Play")
        self._sync_log_slider()
        self._maybe_auto_follow_from_log()
        self.render()

    def _on_key_press(self, event) -> None:
        if event.key == "1" and self.log_state:
            self.switch_page("log")
            return
        if event.key == "2" and self.csv_state:
            self.switch_page("csv")
            return
        if event.key == "h":
            self._print_help()
            return

        if self.current_page == "log" and self.log_state:
            if event.key == " ":
                self._toggle_log_play()
            elif event.key == "left":
                self._step_log_frame(-1)
            elif event.key == "right":
                self._step_log_frame(1)
            elif event.key == "p":
                self._jump_log_event("prev")
            elif event.key == "n":
                self._jump_log_event("next")
            elif event.key == "f":
                self._jump_log_event("first")
            elif event.key == "l":
                self._jump_log_event("last")
            elif event.key == "a":
                self._toggle_log_yaw()
            elif event.key == "i":
                self._toggle_detail_panel()
            elif event.key == "c":
                self._jump_to_linked_csv()
        elif self.current_page == "csv" and self.csv_state:
            if event.key == "left":
                self._step_csv_group(-1)
            elif event.key == "right":
                self._step_csv_group(1)
            elif event.key == "i":
                self._toggle_detail_panel()
            elif event.key == "j":
                self._jump_to_linked_log()

    def _on_scroll(self, event) -> None:
        if self.current_page != "log" or not self.log_state:
            return
        if event.inaxes != self.plot_ax or event.xdata is None or event.ydata is None:
            return
        x_min, x_max = self.plot_ax.get_xlim()
        y_min, y_max = self.plot_ax.get_ylim()
        scale = 0.9 if event.button == "up" else 1.1
        new_width = (x_max - x_min) * scale
        new_height = (y_max - y_min) * scale
        center_x = event.xdata
        center_y = event.ydata
        self.log_state.view_xlim = (center_x - new_width / 2, center_x + new_width / 2)
        self.log_state.view_ylim = (center_y - new_height / 2, center_y + new_height / 2)
        self.log_state.user_adjusted_view = True
        self.render()

    def _on_mouse_press(self, event) -> None:
        if self.current_page == "log" and event.inaxes == self.log_timeline_ax and event.button == 1:
            self._handle_log_timeline_click(event)
            return
        if self.current_page == "csv" and event.inaxes == self.csv_overview_ax and event.button == 1:
            self._handle_csv_overview_click(event)
            return
        if self.current_page != "log" or not self.log_state:
            return
        if event.inaxes != self.plot_ax or event.button != 2 or event.xdata is None or event.ydata is None:
            return
        self.drag_start = (event.xdata, event.ydata)

    def _on_mouse_move(self, event) -> None:
        if self.current_page != "log" or not self.log_state or not self.drag_start:
            return
        if event.inaxes != self.plot_ax or event.xdata is None or event.ydata is None:
            return
        previous_x, previous_y = self.drag_start
        delta_x = event.xdata - previous_x
        delta_y = event.ydata - previous_y
        self.drag_start = (event.xdata, event.ydata)
        x_min, x_max = self.plot_ax.get_xlim()
        y_min, y_max = self.plot_ax.get_ylim()
        self.log_state.view_xlim = (x_min - delta_x, x_max - delta_x)
        self.log_state.view_ylim = (y_min - delta_y, y_max - delta_y)
        self.log_state.user_adjusted_view = True
        self.render()

    def _on_mouse_release(self, _event) -> None:
        self.drag_start = None

    def _print_help(self) -> None:
        print("\n" + "=" * 72)
        print("Parking Planning Toolbox 快捷键")
        print("  1 / 2          页面切换 (Log / CSV)")
        print("  h              显示帮助")
        print("  Log 页: 空格 播放暂停, 左右逐帧, p/n 上下事件, f/l 首末事件, a 切换航向箭头, i 开关详情侧栏, c 跳到匹配 CSV 组")
        print("  Log 页: 下方 Event Timeline 可点击跳事件/跳帧, Auto Follow Exact 仅在 exact 匹配时后台联动")
        print("  CSV 页: 左右切组, j 跳到匹配日志帧, i 开关详情侧栏, Group Overview 可点击跳组")
        print("  鼠标滚轮 / 中键拖动仅在 Log 页生效，用于缩放和平移")
        print("=" * 72 + "\n")

    def render(self) -> None:
        self._clear_status_badges()
        if self.current_page == "log" and self.log_state:
            self._render_log_page()
        elif self.current_page == "csv" and self.csv_state:
            self._render_csv_page()
        self._render_log_timeline()
        self._render_csv_overview()
        self._render_detail_panel()
        self.fig.canvas.draw_idle()

    def _clear_status_badges(self) -> None:
        for artist in self.badge_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self.badge_artists = []

    def _render_status_badges(self, badges: list[tuple[str, str, str]]) -> None:
        left = 0.06
        width = 0.67
        columns = 4
        row_y = [0.058, 0.032]
        col_width = width / columns

        for index, (label, value, color_key) in enumerate(badges[:8]):
            row = index // columns
            col = index % columns
            x_pos = left + col * col_width
            y_pos = row_y[min(row, len(row_y) - 1)]
            facecolor = MATCH_STATE_COLORS.get(
                color_key,
                EVENT_TYPE_COLORS.get(color_key, BADGE_COLORS.get(color_key, BADGE_COLORS["neutral"])),
            )
            artist = self.fig.text(
                x_pos,
                y_pos,
                f"{label}  {value}",
                ha="left",
                va="center",
                fontsize=8.4,
                fontweight="bold",
                color="#263238",
                bbox=dict(
                    boxstyle="round,pad=0.34",
                    facecolor=facecolor,
                    edgecolor="#90A4AE",
                    linewidth=1.0,
                    alpha=0.95,
                ),
            )
            self.badge_artists.append(artist)

    def _build_log_status_badges(self) -> list[tuple[str, str, str]]:
        frame = self.log_state.current_frame()
        payload = self.log_state.data.payload
        stage = legacy_log.func_stage_mapping.get(_payload_value(payload, "parking_function_stages", frame), "-")
        status = legacy_log.func_status_mapping.get(_payload_value(payload, "parking_function_statuses", frame), "-")
        mode = legacy_log.func_mode_mapping.get(_payload_value(payload, "parking_function_modes", frame), "-")
        current_event = next((event for event in self.log_state.events if event.frame == frame), None)
        linked_csv_match = self._get_linked_csv_match(frame)
        filtered_pos = f"{self.log_state.current_index + 1}/{len(self.log_state.valid_frames)}"
        return [
            ("Frame", str(frame), "neutral"),
            ("Filtered", filtered_pos, "info"),
            ("Stage", _shorten_text(stage, 12), "stage"),
            ("Status", _shorten_text(status, 14), "status"),
            ("Mode", _shorten_text(mode, 12), "mode"),
            ("PF ts", _shorten_text(_payload_value(payload, "perception_fusion_timestamps", frame), 16), "timestamp"),
            (
                "CSV Link",
                f"G{linked_csv_match.target_index + 1}" if linked_csv_match else "None",
                _match_state_name(linked_csv_match),
            ),
            (
                "Event",
                _shorten_text(_compact_event_badge(current_event), 16),
                current_event.event_type if current_event else "neutral",
            ),
        ]

    def _build_csv_status_badges(self) -> list[tuple[str, str, str]]:
        group_index = self.csv_state.current_group_index()
        group = self.csv_state.data.groups[group_index]
        linked_log_match = self._get_linked_log_match(group_index)
        filtered_pos = f"{self.csv_state.current_index + 1}/{len(self.csv_state.filtered_indices)}"
        trajectory_color = "trajectory_yes" if group.get("has_trajectory") else "trajectory_no"
        return [
            ("Group", f"{group_index + 1}/{len(self.csv_state.data.groups)}", "neutral"),
            ("Filtered", filtered_pos, "info"),
            ("Timestamp", _shorten_text(group.get("timestamp") or "-", 16), "timestamp"),
            (
                "Linked Log",
                f"F{linked_log_match.target_index}" if linked_log_match else "None",
                _match_state_name(linked_log_match),
            ),
            (
                "Linked PF ts",
                _shorten_text(linked_log_match.target_timestamp if linked_log_match else "-", 16),
                "timestamp",
            ),
            ("Trajectory", "YES" if group.get("has_trajectory") else "NO", trajectory_color),
            ("Match", _match_state_label(linked_log_match), _match_state_name(linked_log_match)),
            ("Filter", self.csv_state.trajectory_filter, "mode"),
        ]

    def _render_detail_panel(self) -> None:
        self.detail_ax.clear()
        self.detail_ax.set_axis_off()
        self.detail_ax.set_facecolor("#F7F9FB")

        if not self._current_detail_open():
            self.detail_ax.set_visible(False)
            return

        self.detail_ax.set_visible(True)

        if self.current_page == "log":
            if not self.log_state or not self.log_state.valid_frames:
                self.detail_ax.text(0.02, 0.98, "No log frame available.", va="top", ha="left", fontsize=10)
                return
            frame = self.log_state.current_frame()
            info_lines = _build_log_info_lines(self.log_state, frame, linked_csv_match=self._get_linked_csv_match(frame))
            title = "Log Details"
        else:
            if not self.csv_state or not self.csv_state.filtered_indices:
                self.detail_ax.text(0.02, 0.98, "No CSV group available.", va="top", ha="left", fontsize=10)
                return
            group_index = self.csv_state.current_group_index()
            info_lines = _build_csv_info_lines(self.csv_state, group_index, linked_log_match=self._get_linked_log_match(group_index))
            title = "CSV Details"

        wrapped_lines = []
        for line in info_lines:
            wrapped_lines.extend(textwrap.wrap(str(line), width=28) or [""])

        self.detail_ax.text(0.02, 0.98, title, va="top", ha="left", fontsize=11, fontweight="bold", color="#37474F")
        self.detail_ax.text(
            0.02,
            0.94,
            "\n".join(wrapped_lines),
            va="top",
            ha="left",
            fontsize=8.8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#90A4AE", alpha=0.96),
        )

    def _render_log_timeline(self) -> None:
        ax = self.log_timeline_ax
        ax.clear()
        ax.set_facecolor("#FAFAFA")

        if self.current_page != "log" or not self.log_state:
            ax.set_visible(False)
            return

        ax.set_visible(True)
        state = self.log_state
        if not state.valid_frames:
            ax.text(0.5, 0.5, "No timeline", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        total = len(state.valid_frames)
        ax.set_xlim(-0.5, max(total - 0.5, 0.5))
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_title("Event Timeline", fontsize=9, fontweight="bold", pad=4)
        ax.hlines(0.08, -0.5, total - 0.5, color="#B0BEC5", linewidth=1.1, zorder=1)

        tick_count = min(6, total)
        tick_positions = np.unique(np.linspace(0, total - 1, num=tick_count, dtype=int))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(state.valid_frames[position]) for position in tick_positions], fontsize=7)

        event_y = {
            "search2park": 0.82,
            "replan": 0.66,
            "stop_reason": 0.50,
            "status": 0.34,
            "file_boundary": 0.18,
        }
        position_map = {frame: index for index, frame in enumerate(state.valid_frames)}
        for event in state.events:
            position = position_map.get(event.frame)
            if position is None:
                continue
            color = EVENT_TYPE_COLORS.get(event.event_type, "#455A64")
            y_pos = event_y.get(event.event_type, 0.5)
            ax.vlines(position, 0.08, y_pos - 0.04, color=color, linewidth=1.2, alpha=0.4, zorder=2)
            ax.scatter([position], [y_pos], s=34, color=color, edgecolors="white", linewidths=0.8, zorder=3)

        ax.axvline(state.current_index, color="#212121", linewidth=1.2, alpha=0.9, zorder=4)
        ax.scatter([state.current_index], [0.96], marker="v", s=36, color="#212121", zorder=5)
        for spine in ax.spines.values():
            spine.set_color("#CFD8DC")

    def _render_csv_overview(self) -> None:
        ax = self.csv_overview_ax
        ax.clear()
        ax.set_facecolor("#FAFAFA")

        if self.current_page != "csv" or not self.csv_state:
            ax.set_visible(False)
            return

        ax.set_visible(True)
        state = self.csv_state
        group_count = len(state.data.groups)
        if group_count == 0:
            ax.text(0.5, 0.5, "No groups", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        ax.set_xlim(0, group_count)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Group Overview", fontsize=9, fontweight="bold", pad=4)

        current_group = state.current_group_index()
        linked_from_log_match = self._get_linked_csv_match(self.log_state.current_frame()) if self.log_state and self.log_state.valid_frames else None
        linked_group_index = linked_from_log_match.target_index if linked_from_log_match else None

        for group_index, group in enumerate(state.data.groups):
            linked_log_match = self._get_linked_log_match(group_index)
            match_color = MATCH_STATE_COLORS[_match_state_name(linked_log_match)]
            fill_color = "#81C784" if group.get("has_trajectory") else "#ECEFF1"
            alpha = 0.95 if group_index in state.filtered_indices else 0.35

            base_rect = Rectangle((group_index + 0.05, 0.16), 0.9, 0.62, facecolor=fill_color, edgecolor="#B0BEC5", linewidth=1.0, alpha=alpha)
            stripe_rect = Rectangle((group_index + 0.05, 0.80), 0.9, 0.12, facecolor=match_color, edgecolor="none", alpha=alpha)
            ax.add_patch(base_rect)
            ax.add_patch(stripe_rect)
            ax.text(group_index + 0.5, 0.47, str(group_index + 1), ha="center", va="center", fontsize=8, fontweight="bold", color="#263238")

            if current_group == group_index:
                ax.add_patch(Rectangle((group_index + 0.03, 0.14), 0.94, 0.80, fill=False, edgecolor="#212121", linewidth=2.2))
            if linked_group_index == group_index:
                ax.add_patch(
                    Rectangle(
                        (group_index + 0.09, 0.20),
                        0.82,
                        0.68,
                        fill=False,
                        edgecolor="#1E88E5",
                        linewidth=1.6,
                        linestyle="--",
                    )
                )

        for spine in ax.spines.values():
            spine.set_color("#CFD8DC")

    def _render_log_page(self) -> None:
        state = self.log_state
        ax = self.plot_ax
        ax.clear()

        self.header_text.set_text("Parking Planning Toolbox | Log Replay")
        self.footer_text.set_text("Space play/pause | Left/Right frame | P/N event | C linked CSV | I details | H help")

        if not state.valid_frames:
            ax.text(0.5, 0.5, "当前筛选条件下没有可显示帧", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            return

        frame = state.current_frame()
        payload = state.data.payload

        _draw_p0_p5(ax, _payload_value(payload, "p0_p5_points", frame))
        _draw_slot_corners(ax, _payload_value(payload, "slot_corners", frame))
        _draw_stopper_distance(ax, _payload_value(payload, "stopper_distances", frame), _payload_value(payload, "target_corners_abcd", frame))
        _draw_realtime_parkingspace(ax, _payload_value(payload, "realtime_parkingspaces", frame))

        vehicle_location = _payload_value(payload, "vehicle_locations", frame)
        if vehicle_location:
            x_vehicle, y_vehicle, yaw_vehicle = vehicle_location
            legacy_log.draw_detailed_vehicle(ax, x_vehicle, y_vehicle, yaw_vehicle)

            trail_points = _collect_rear_axle_trail(payload, frame)
            if trail_points:
                ax.plot(
                    [point[0] for point in trail_points],
                    [point[1] for point in trail_points],
                    marker="o",
                    color=legacy_log.COLORS["rear_axle"],
                    linestyle="",
                    markersize=4,
                    alpha=0.45,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    label="Rear Axle Trail",
                )

            if state.show_arrows:
                arrow_history = _collect_arrow_history(payload, frame)
                for index, (x_coord, y_coord, yaw_coord) in enumerate(arrow_history):
                    arrow_length = 800
                    arrow_dx = arrow_length * np.cos(np.radians(yaw_coord))
                    arrow_dy = arrow_length * np.sin(np.radians(yaw_coord))
                    alpha = 0.8 if index == len(arrow_history) - 1 else 0.25
                    linewidth = 2.2 if index == len(arrow_history) - 1 else 1.3
                    ax.arrow(
                        x_coord,
                        y_coord,
                        arrow_dx,
                        arrow_dy,
                        head_width=300,
                        head_length=400,
                        fc=legacy_log.COLORS["vehicle_arrow"],
                        ec=legacy_log.COLORS["vehicle_arrow"],
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=6,
                    )

        coordinates = _payload_value(payload, "coordinates", frame)
        if coordinates:
            x_coords = [point[0] for point in coordinates if len(point) >= 2]
            y_coords = [point[1] for point in coordinates if len(point) >= 2]
            if x_coords and y_coords:
                ax.plot(
                    x_coords,
                    y_coords,
                    marker="o",
                    color=legacy_log.COLORS["trajectory"],
                    linestyle="-",
                    markersize=4,
                    linewidth=2.3,
                    alpha=0.85,
                    markerfacecolor=legacy_log.COLORS["trajectory"],
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    label="Planned Trajectory",
                    zorder=3,
                )

        _draw_chamfer(ax, _payload_value(payload, "parking_space_chamfers", frame))
        _draw_target_pose(ax, _payload_value(payload, "plan_stage_target_poses", frame), legacy_log.COLORS["target_stage"], "--", "Stage Target", 12)
        _draw_target_pose(ax, _payload_value(payload, "plan_final_target_poses", frame), legacy_log.COLORS["target_final"], "-.", "Final Target", 15)

        plan_type = _payload_value(payload, "fork_star_starts", frame)
        if plan_type and vehicle_location:
            x_vehicle, y_vehicle, _yaw_vehicle = vehicle_location
            if plan_type == "PERP":
                star_color = "#FF00FF"
                label_text = "PERP FORK STAR!"
            else:
                star_color = "#00CCFF"
                label_text = "PARA FORK STAR!"
            ax.scatter([x_vehicle], [y_vehicle + 1500], marker="*", s=420, color=star_color, edgecolors="white", linewidths=2, zorder=20)
            ax.text(
                x_vehicle,
                y_vehicle + 2200,
                label_text,
                color=star_color,
                fontsize=13,
                fontweight="bold",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=star_color, alpha=0.25, edgecolor=star_color, linewidth=1.5),
                zorder=21,
            )

        frame_view_xlim, frame_view_ylim = _compute_log_frame_view(payload, frame, state.data.fixed_xlim, state.data.fixed_ylim)
        if state.user_adjusted_view and state.view_xlim and state.view_ylim:
            ax.set_xlim(state.view_xlim)
            ax.set_ylim(state.view_ylim)
        else:
            ax.set_xlim(frame_view_xlim)
            ax.set_ylim(frame_view_ylim)

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4, color=legacy_log.COLORS["grid"])
        ax.set_facecolor(legacy_log.COLORS["background"])
        ax.set_xlabel("X Position (mm)", fontsize=11, fontweight="bold", color=legacy_log.COLORS["text"])
        ax.set_ylabel("Y Position (mm)", fontsize=11, fontweight="bold", color=legacy_log.COLORS["text"])

        stage = _payload_value(payload, "parking_function_stages", frame)
        mode = _payload_value(payload, "parking_function_modes", frame)
        status = _payload_value(payload, "parking_function_statuses", frame)

        stage_str = legacy_log.func_stage_mapping.get(stage, stage)
        mode_str = legacy_log.func_mode_mapping.get(mode, mode)
        status_str = legacy_log.func_status_mapping.get(status, status)
        ax.set_title(
            f"Frame {frame} | Filtered {state.current_index + 1}/{len(state.valid_frames)} | Stage {stage_str} | Status {status_str} | Mode {mode_str}",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(
                loc="upper left",
                frameon=True,
                fancybox=True,
                shadow=False,
                fontsize=9,
                framealpha=0.9,
                edgecolor="#2E86AB",
                facecolor="white",
            )
            legend.get_frame().set_linewidth(1.0)

        self._render_status_badges(self._build_log_status_badges())

        if not _frame_has_visible_content(payload, frame):
            ax.text(
                0.5,
                0.08,
                "Current frame has no drawable geometry. Use Play / Next Frame / Next Event.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#B71C1C",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFEBEE", edgecolor="#EF9A9A", alpha=0.95),
            )

    def _render_csv_page(self) -> None:
        state = self.csv_state
        ax = self.plot_ax
        ax.clear()

        self.header_text.set_text("Parking Planning Toolbox | CSV Browser")
        self.footer_text.set_text("Left/Right group | J linked log | Overview click jump | I details | H help")

        if not state.filtered_indices:
            ax.text(0.5, 0.5, "当前轨迹筛选下没有可显示的组", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            return

        group_index = state.current_group_index()
        group = state.data.groups[group_index]
        gridmap = group["gridmap"]
        ax.imshow(gridmap, cmap="gray", vmin=0, vmax=255, origin="lower", interpolation="nearest")

        grid_size = state.data.grid_size
        resolution = state.data.resolution
        tick_positions = _csv_tick_positions(grid_size)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.grid(True, alpha=0.18, color="#90CAF9", linewidth=0.45)
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_xticklabels(
            [f"{tick * resolution / 1000:.0f}m" for tick in tick_positions],
            fontsize=6,
        )
        ax.set_yticklabels(
            [f"{tick * resolution / 1000:.0f}m" for tick in tick_positions],
            fontsize=6,
        )

        artists = []
        grid_origin = group.get("gridmap_origin")
        if grid_origin:
            if group.get("slot_points"):
                legacy_csv.draw_parking_slot(ax, artists, group["slot_points"], grid_origin, grid_size, resolution)
            if group.get("target_pose"):
                legacy_csv.draw_vehicle(ax, artists, group["target_pose"], grid_origin, grid_size, resolution, is_ego=False)
            if group.get("ego_pose"):
                legacy_csv.draw_vehicle(ax, artists, group["ego_pose"], grid_origin, grid_size, resolution, is_ego=True)
            if group.get("trajectory"):
                legacy_csv.draw_trajectory(ax, artists, group["trajectory"], grid_origin, grid_size, resolution)

        filtered_progress = f"{state.current_index + 1}/{len(state.filtered_indices)}"
        ax.set_title(
            f"Group {group_index + 1}/{len(state.data.groups)} | Filtered {filtered_progress} | Trajectory {state.trajectory_filter}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        self._render_status_badges(self._build_csv_status_badges())


    def show(self) -> None:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parking planning toolbox with unified log replay and CSV browser.")
    subparsers = parser.add_subparsers(dest="command")

    log_parser = subparsers.add_parser("log", help="Open log replay page")
    log_parser.add_argument("log_path", nargs="?", help="Path to a planning log file")
    log_parser.add_argument("--no-merge", action="store_true", help="Disable automatic merge of log series")
    log_parser.add_argument("--start-frame", type=int, default=0, help="Initial frame filter start")
    log_parser.add_argument("--end-frame", type=int, default=None, help="Initial frame filter end")
    log_parser.add_argument("--filter-stage", default=None, help="Stage filter, comma separated numbers or names")
    log_parser.add_argument("--filter-status", default=None, help="Status filter, comma separated numbers or names")
    log_parser.add_argument("--filter-mode", default=None, help="Mode filter, comma separated numbers or names")
    log_parser.add_argument("--event-types", default=",".join(ALL_EVENT_TYPES), help="Comma separated subset of search2park,replan,stop_reason,status,file_boundary")
    log_parser.add_argument("--pause-on-event", action="store_true", help="Pause automatically when stepping onto an indexed event")

    csv_parser = subparsers.add_parser("csv", help="Open CSV browser page")
    csv_parser.add_argument("--csv", dest="csv_path", default=None, help="Path to planner_inputs.csv")
    csv_parser.add_argument("--start-group", type=int, default=1, help="Initial 1-based group number")
    csv_parser.add_argument("--filter-trajectory", choices=("all", "with", "without"), default="all", help="Trajectory filter")
    csv_parser.add_argument("--resolution", type=float, default=100.0, help="Grid resolution in mm per cell")
    csv_parser.add_argument("--size", type=int, default=512, help="Grid size")

    return parser


def _build_log_state_from_args(args) -> Optional[LogPageState]:
    if args.command == "csv":
        log_dataset = load_log_dataset(discover_default_log_file(), merge_enabled=True)
    else:
        log_dataset = load_log_dataset(getattr(args, "log_path", None), merge_enabled=not getattr(args, "no_merge", False))
    if not log_dataset:
        return None

    filters = LogFilters(
        start_frame=getattr(args, "start_frame", 0),
        end_frame=getattr(args, "end_frame", None),
        stages=_parse_named_filter(getattr(args, "filter_stage", None), legacy_log.func_stage_mapping),
        statuses=_parse_named_filter(getattr(args, "filter_status", None), legacy_log.func_status_mapping),
        modes=_parse_named_filter(getattr(args, "filter_mode", None), legacy_log.func_mode_mapping),
        event_types=_parse_event_types(getattr(args, "event_types", None)),
    )
    state = LogPageState(data=log_dataset, filters=filters, pause_on_event=getattr(args, "pause_on_event", False))
    preferred_frame = filters.start_frame if filters.start_frame > 0 else None
    state.rebuild(preferred_frame=preferred_frame)
    return state


def _build_csv_state_from_args(args) -> Optional[CsvPageState]:
    grid_size = getattr(args, "size", 512)
    resolution = getattr(args, "resolution", 100.0)
    if args.command == "log":
        csv_dataset = load_csv_dataset(discover_default_csv_file(), grid_size=512, resolution=100.0)
    else:
        csv_dataset = load_csv_dataset(getattr(args, "csv_path", None), grid_size=grid_size, resolution=resolution)
    if not csv_dataset:
        return None

    state = CsvPageState(data=csv_dataset, trajectory_filter=getattr(args, "filter_trajectory", "all"))
    preferred_group = max(0, getattr(args, "start_group", 1) - 1)
    state.rebuild(preferred_group=preferred_group)
    return state


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        log_state = _build_log_state_from_args(args)
        csv_state = _build_csv_state_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    if not log_state and not csv_state:
        parser.error("No compatible log or CSV source was found in the current directory.")
        return 2

    start_page = "csv" if args.command == "csv" else "log"
    app = PlannerToolboxApp(log_state=log_state, csv_state=csv_state, start_page=start_page)
    app.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
