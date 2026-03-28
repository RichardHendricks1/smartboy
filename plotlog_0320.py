import re
import sys
import os
import glob
import tempfile
import numpy as np

if not os.environ.get("MPLCONFIGDIR"):
    _mplconfig_dir = os.path.join(tempfile.gettempdir(), "planner_toolbox_mplconfig")
    os.makedirs(_mplconfig_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _mplconfig_dir

if not os.environ.get("MPLBACKEND"):
    try:
        import tkinter  # noqa: F401
    except Exception:
        pass
    else:
        os.environ["MPLBACKEND"] = "TkAgg"

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from matplotlib.patches import FancyBboxPatch
from datetime import datetime, timezone, timedelta

# 使用深色网格样式，处理不同版本的兼容性
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        pass

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

COLORS = {
    'parking_space': '#2E86AB',      # 停车位 - 蓝色
    'slot_corners': '#A23B72',       # 停车位角点 - 紫红色
    'vehicle': '#F18F01',            # 车辆 - 橙色
    'vehicle_arrow': '#C73E1D',      # 车辆朝向箭头 - 红色
    'trajectory': '#06A77D',         # 轨迹 - 绿色
    'target_stage': '#6BCF7E',       # 阶段目标 - 浅绿色
    'target_final': '#FFD23F',       # 最终目标 - 金黄色
    'p0_p5_fused': '#E63946',        # P0/P5融合点 - 鲜红色
    'rear_axle': '#F18F01',          # 后轴中心点 - 橙色
    'grid': '#CCCCCC',               # 网格 - 浅灰色
    'background': '#F8F9FA',         # 背景 - 浅色
    'text': '#212529'                # 文本 - 深色
}

is_paused = False

current_frame_index = 0

updating_progress = False

all_rear_axle_centers = []
all_vehicle_arrows = []

prev_stop_reason = None
extra_frames_to_show = 0

file_boundaries = []  # 存储文件边界信息 [(start_frame, end_frame, filename), ...]
current_log_text = None  # 用于显示当前log文件名的文本对象

show_arrows = False # 设置为 True 显示箭头，设置为 False 不显示箭头

# 固定坐标轴范围
fixed_xlim = None
fixed_ylim = None
user_adjusted_view = False  # 标记用户是否手动调整过视角

animation_interval = 20

gear_mapping = {
    0: "null",
    1: "P",
    2: "R",
    3: "N",
    4: "D"
}

func_status_mapping = {
    0: "NULL",
    1: "RUN",
    2: "SUSPEND",
    3: "SLEEP"
}

func_stage_mapping = {
    0: "NULL",
    1: "SEARCH",
    2: "PARK"
}

func_mode_mapping = {
    0: "NULL",
    1: "VIS_PERP_REAR_IN",
    2: "VIS_PERP_FRONT_IN",
    3: "VIS_PARA_IN",
    4: "VIS_OBLI_REAR_IN",
    5: "VIS_OBLI_FRONT_IN",
    6: "USS_PERP_REAR_IN",
    7: "USS_PERP_FRONT_IN",
    8: "USS_PARA_IN",
    9: "USS_OBLI_REAR_IN",
    10: "USS_OBLI_FRONT_IN",
    11: "PARA_LEFT_OUT",
    12: "PARA_RIGHT_OUT",
    13: "PERP_RIGHT_FRONT_OUT",
    14: "PERP_LEFT_FRONT_OUT",
    15: "PERP_RIGHT_REAR_OUT",
    16: "PERP_LEFT_REAR_OUT",
    17: "OBLI_RIGHT_FRONT_OUT",
    18: "OBLI_LEFT_FRONT_OUT",
    19: "OBLI_RIGHT_REAR_OUT",
    20: "OBLI_LEFT_REAR_OUT"
}

control_work_mode_mapping = {
    0x00: "NULL",
    0x01: "GLOBAL",
    0x02: "DYNAMIC",
    0x03: "RESERVE"
}

vehicle_moving_status_mapping = {
    0: "MOVING",
    1: "FORWARD_FAIL",
    2: "BACKWARD_FAIL",
    3: "DEAD_LOCK"
}

control_stop_reason_mapping = {
    0: "NULL",
    1: "FRONT_ALERT",
    2: "REAR_ALERT",
    3: "LEFTSIDE_ALERT",
    4: "RIGHTSIDE_ALERT",
    5: "TARGET_CLOSING",
    6: "UNKNOWN_REASON",
    7: "MANUAL_SUSPEND",
    8: "APA_SUSPEND",
    9: "PLAN4STOP",
    10: "LEFTRVW_ALERT",
    11: "RIGHTRVW_ALERT",
    12: "TRACKING_FINISHED",
    13: "TRACK_LOSS",
    14: "SCANNING_FINISHED",
    15: "STANDBY"
}

def safe_read_file_lines(file_path):
    """安全读取文件行，自动处理编码错误"""
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            for line in file:
                lines.append(line)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return lines

def remove_log_prefix(line):
    """移除日志前缀，支持新旧两种格式
    新格式: [2026-03-02 18:41:39.644] [DEBUG] [planningComponent.cpp:146] [PID:2795251 TID:2795256]
    旧格式: [DecPlan Input ] ... 或直接以内容开头
    """
    # 跳过空行和非字符串行
    if not isinstance(line, str):
        return ""
    if not line.strip():
        return ""

    # 新格式前缀: [时间戳] [级别] [源文件:行号] [进程信息]
    # 模式: [\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[\w+\] \[[^\]]+\:\d+\] \[PID:\d+ TID:\d+\]
    prefix_pattern = r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[\w+\] \[[^\]]+\:\d+\] \[PID:\d+ TID:\d+\] \s*'
    match = re.match(prefix_pattern, line)
    if match:
        return line[match.end():]
    return line

def convert_timestamp_to_bj_time(timestamp):
    # 自动检测时间戳单位：纳秒(>1e15)、微秒(>1e12)、毫秒(>1e10)、秒(<=1e10)
    if timestamp > 1e15:  # 纳秒级
        timestamp = timestamp / 1e9
    elif timestamp > 1e12:  # 微秒级
        timestamp = timestamp / 1e6
    elif timestamp > 1e10:  # 毫秒级
        timestamp = timestamp / 1000.0
    # 秒级不需要转换

    utc_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    bj_time = utc_time + timedelta(hours=8)  # 北京时间比 UTC 时间快 8 小时
    return bj_time.strftime('%Y-%m-%d %H:%M:%S')

def extract_timestamps(file_path):
    # 新格式时间戳模式
    pattern_timestamp_new = r"\[/apa/loc/vehicle_pose_on_slot\]:\s*\(pub_timestamp,\s*sequence,\s*x,\s*y,\s*yaw,\s*timestamp_us\)\s*=\s*\((\d+)"
    # 旧格式时间戳模式（兼容）
    pattern_timestamp_old = r"Vehicle Location Time Stamp:\s*\[\s*(\d+)\s*\]"

    timestamps = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            # 优先尝试新格式
            match = re.search(pattern_timestamp_new, line)
            if match:
                timestamps.append(int(match.group(1)))
                continue
            # 兼容旧格式
            match = re.search(pattern_timestamp_old, line)
            if match:
                timestamps.append(int(match.group(1)))
    return timestamps

def extract_gear_info(file_path):
    pattern_gear = r"Path Segment Target Gear:\s*\[\s*(\d+)\s*\]"
    gear_info = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_gear, line)
            if match:
                gear_number = int(match.group(1))
                gear_string = gear_mapping.get(gear_number, "Unknown")
                gear_info.append(gear_string)
    return gear_info

def extract_parking_spaces(file_path):
    # 支持科学计数法的正则表达式: ([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)
    pattern_parking = r"Parking Space: P0\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*P1\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*P2\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*P3\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*P4\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*P5\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*P6\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*P7\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]"
    parking_spaces = []
    lines = safe_read_file_lines(file_path)
    for line in lines:
        line = remove_log_prefix(line)
        match = re.search(pattern_parking, line)
        if match:
            points = []
            for i in range(0, 16, 2):
                x = float(match.group(i + 1))  # 已经是mm，不需要转换
                y = float(match.group(i + 2))  # 已经是mm，不需要转换
                points.append((x, y))
            parking_spaces.append(points)
    return parking_spaces

def extract_realtime_parkingspace(file_path):
    """提取 Realtime updating parkingspace 的 p0-p7 点，与 Plan Frame ID 同步"""
    pattern_p0_p3 = r"Realtime updating parkingspace p0\[\s*([-+]?\d+\.?\d*)\s*mm,\s*([-+]?\d+\.?\d*)\s*mm\]\s*p1\[\s*([-+]?\d+\.?\d*)\s*mm,\s*([-+]?\d+\.?\d*)\s*mm\]\s*p2\[\s*([-+]?\d+\.?\d*)\s*mm,\s*([-+]?\d+\.?\d*)\s*mm\]\s*p3\[\s*([-+]?\d+\.?\d*)\s*mm,\s*([-+]?\d+\.?\d*)\s*mm\]"
    pattern_p4_p7 = r"Realtime updating parkingspace p4\[\s*([-+]?\d+\.?\d*)\s*mm,\s*([-+]?\d+\.?\d*)\s*mm\]\s*p5\[\s*([-+]?\d+\.?\d*)\s*mm,\s*([-+]?\d+\.?\d*)\s*mm\]\s*p6\[\s*([-+]?\d+\.?\d*)\s*mm,\s*([-+]?\d+\.?\d*)\s*mm\]\s*p7\[\s*([-+]?\d+\.?\d*)\s*mm,\s*([-+]?\d+\.?\d*)\s*mm\]"
    pattern_frame = r"Plan Frame ID \[\s*(\d+)\s*\]"
    realtime_spaces = []

    lines = safe_read_file_lines(file_path)

    current_p0_p3 = None
    current_p4_p7 = None
    frame_started = False

    for line in lines:
        line = remove_log_prefix(line)
        # 检测新的帧开始
        frame_match = re.search(pattern_frame, line)
        if frame_match:
            if frame_started:
                # 记录上一帧的 realtime parkingspace
                if current_p0_p3 and current_p4_p7:
                    # 合并 p0-p7
                    all_points = current_p0_p3 + current_p4_p7
                    realtime_spaces.append(all_points)
                else:
                    realtime_spaces.append(None)
            frame_started = True
            current_p0_p3 = None
            current_p4_p7 = None

        # 提取 p0-p3
        match_p0_p3 = re.search(pattern_p0_p3, line)
        if match_p0_p3 and current_p0_p3 is None:
            groups = match_p0_p3.groups()
            current_p0_p3 = [
                (float(groups[0]), float(groups[1])),  # p0 (已经是mm，不需要转换)
                (float(groups[2]), float(groups[3])),  # p1
                (float(groups[4]), float(groups[5])),  # p2
                (float(groups[6]), float(groups[7]))   # p3
            ]

        # 提取 p4-p7
        match_p4_p7 = re.search(pattern_p4_p7, line)
        if match_p4_p7 and current_p4_p7 is None:
            groups = match_p4_p7.groups()
            current_p4_p7 = [
                (float(groups[0]), float(groups[1])),  # p4 (已经是mm，不需要转换)
                (float(groups[2]), float(groups[3])),  # p5
                (float(groups[4]), float(groups[5])),  # p6
                (float(groups[6]), float(groups[7]))   # p7
            ]

    # 添加最后一帧
    if frame_started:
        if current_p0_p3 and current_p4_p7:
            all_points = current_p0_p3 + current_p4_p7
            realtime_spaces.append(all_points)
        else:
            realtime_spaces.append(None)

    return realtime_spaces

def extract_slot_corners(file_path):
    """提取 Slot corners after coordinate conversion
    注意：这个数据的单位已经是毫米，不需要再转换
    支持科学计数法：如 -2.71836e-07
    """
    # 支持科学计数法的正则表达式: ([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)
    pattern_slot = r"Slot corners after coordinate conversion A\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm,\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*B\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm,\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*C\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm,\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*D\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm,\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]"
    slot_corners = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_slot, line)
            if match:
                points = []
                for i in range(0, 8, 2):
                    x = float(match.group(i + 1))  # 已经是mm，不需要转换
                    y = float(match.group(i + 2))  # 已经是mm，不需要转换
                    points.append((x, y))
                slot_corners.append(points)
    return slot_corners

def extract_vehicle_locations(file_path):
    # 支持新旧两种格式
    # 旧格式: [DecPlan Input ] Vehicle Realtime Location: X[...]mm] Y[...]mm] Yaw[...]degree]
    # 新格式（前缀已被remove_log_prefix移除）: Vehicle Realtime Location: X[...]mm] Y[...]mm] Yaw[...]degree]
    pattern_vehicle = r"(?:\[DecPlan Input \] )?Vehicle Realtime Location: X\[\s*([-+]?\d+\.?\d*)\s*mm\]\s*Y\[\s*([-+]?\d+\.?\d*)\s*mm\]\s*Yaw\[\s*([-+]?\d+\.?\d*)\s*degree\]"
    vehicle_locations = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_vehicle, line)
            if match:
                x = float(match.group(1))  # 已经是mm，不需要转换
                y = float(match.group(2))  # 已经是mm，不需要转换
                yaw = float(match.group(3))
                vehicle_locations.append((x, y, yaw))
    return vehicle_locations

def extract_plan_frame_id(file_path):
    pattern_plan_frame_id = r"Plan Frame ID \[\s*(\d+)\s*\]"
    plan_frame_ids = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_plan_frame_id, line)
            if match:
                plan_frame_ids.append(int(match.group(1)))
    return plan_frame_ids

def extract_parking_function_status(file_path):
    pattern_parking_function_status = r"Parking Function Status:\s*\[?\s*(\d+)\s*\]?"
    parking_function_statuses = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_parking_function_status, line)
            if match:
                parking_function_statuses.append(int(match.group(1)))
    return parking_function_statuses

def extract_parking_function_stage(file_path):
    pattern_parking_function_stage = r"Parking Function Stage:\s*\[?\s*(\d+)\s*\]?"
    parking_function_stages = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_parking_function_stage, line)
            if match:
                parking_function_stages.append(int(match.group(1)))
    return parking_function_stages

def extract_parking_function_mode(file_path):
    pattern_parking_function_mode = r"Parking Function Mode:\s*\[?\s*(\d+)\s*\]?"
    parking_function_modes = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_parking_function_mode, line)
            if match:
                parking_function_modes.append(int(match.group(1)))
    return parking_function_modes

def extract_vehicle_stop_reason(file_path):
    # 支持两种格式：Control Stop Reason 和 Vehicle Stop Reason
    pattern_control_stop_reason = r"Control Stop Reason: \[\s*(\d+)\s*\]"
    pattern_vehicle_stop_reason = r"Vehicle Stop Reason: \[\s*(\d+)\s*\]"
    vehicle_stop_reasons = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            # 优先匹配 Control Stop Reason
            match = re.search(pattern_control_stop_reason, line)
            if not match:
                match = re.search(pattern_vehicle_stop_reason, line)
            if match:
                vehicle_stop_reasons.append(int(match.group(1)))
    return vehicle_stop_reasons

def extract_control_work_mode(file_path):
    pattern_control_work_mode = r"Control Work Mode: \[\s*(\d+)\s*\]"
    control_work_modes = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_control_work_mode, line)
            if match:
                control_work_modes.append(int(match.group(1)))
    return control_work_modes

def extract_vehicle_moving_status(file_path):
    pattern_vehicle_moving_status = r"Vehicle Moving Status: \[\s*(\d+)\s*\]"
    vehicle_moving_statuses = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_vehicle_moving_status, line)
            if match:
                vehicle_moving_statuses.append(int(match.group(1)))
    return vehicle_moving_statuses

def extract_perception_fusion_timestamps(file_path):
    pattern_perception_fusion_timestamp = r"Perception Fusion Time Stamp: \[\s*(\d+)\s*\]"
    perception_fusion_timestamps = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_perception_fusion_timestamp, line)
            if match:
                perception_fusion_timestamps.append(int(match.group(1)))
    return perception_fusion_timestamps

def extract_path_current_segment_id(file_path):
    pattern_path_current_segment_id = r"Path Current Segment ID: \[\s*(\d+)\s*\]"
    path_current_segment_ids = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_path_current_segment_id, line)
            if match:
                path_current_segment_ids.append(int(match.group(1)))
    return path_current_segment_ids

def extract_parking_space_chamfer(file_path):
    pattern_parking_space_chamfer = r"Parking Space Chamfer: P0 aisle\[\s*([-+]?\d+\.?\d*)\s*mm\s*([-+]?\d+\.?\d*)\s*mm\]\s*P0 slot\[\s*([-+]?\d+\.?\d*)\s*mm\s*([-+]?\d+\.?\d*)\s*mm\]\s*P5 aisle\[\s*([-+]?\d+\.?\d*)\s*mm\s*([-+]?\d+\.?\d*)\s*mm\]\s*P5 slot\[\s*([-+]?\d+\.?\d*)\s*mm\s*([-+]?\d+\.?\d*)\s*mm\]"
    parking_space_chamfers = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_parking_space_chamfer, line)
            if match:
                points = []
                for i in range(0, 8, 2):
                    x = float(match.group(i + 1))  # 已经是mm，不需要转换
                    y = float(match.group(i + 2))  # 已经是mm，不需要转换
                    points.append((x, y))
                parking_space_chamfers.append(points)
    return parking_space_chamfers

def extract_parking_space_p0_p5(file_path):
    # 支持科学计数法的正则表达式: ([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)
    pattern = r"Parking Space P0 & P5 from Fused Points: P0\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*P5\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]"
    p0_p5_points = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern, line)
            if match:
                p0 = (float(match.group(1)), float(match.group(2)))  # 已经是mm，不需要转换
                p5 = (float(match.group(3)), float(match.group(4)))  # 已经是mm，不需要转换
                p0_p5_points.append((p0, p5))
    return p0_p5_points

def extract_plan_stage_target_pose(file_path):
    # 支持科学计数法的正则表达式
    pattern_plan_stage_target_pose = r"Plan Stage Target Pose: X\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*Y\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*Yaw\[\s*([-+]?\d+\.?\d*)\s*degree\]"
    plan_stage_target_poses = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_plan_stage_target_pose, line)
            if match:
                x = float(match.group(1))  # 已经是mm，不需要转换
                y = float(match.group(2))  # 已经是mm，不需要转换
                yaw = float(match.group(3))
                plan_stage_target_poses.append((x, y, yaw))
    return plan_stage_target_poses

def extract_plan_final_target_pose(file_path):
    # 支持科学计数法的正则表达式
    pattern_plan_final_target_pose = r"Plan Final Target Pose: X\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*Y\[\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*mm\]\s*Yaw\[\s*([-+]?\d+\.?\d*)\s*degree\]"
    plan_final_target_poses = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_plan_final_target_pose, line)
            if match:
                x = float(match.group(1))  # 已经是mm，不需要转换
                y = float(match.group(2))  # 已经是mm，不需要转换
                yaw = float(match.group(3))
                plan_final_target_poses.append((x, y, yaw))
    return plan_final_target_poses

def extract_replan_id(file_path):
    pattern_replan_id = r"Replan type:\s*(\d+)"
    replan_ids = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = remove_log_prefix(line)
            match = re.search(pattern_replan_id, line)
            if match:
                replan_id = int(match.group(1))
                replan_ids.append(replan_id)
    return replan_ids

def extract_parking_tasks(file_path):
    """提取 Parking task 的状态，与 Plan Frame ID 同步"""
    pattern_parking_task = r"Parking task \[([A-Z_]+)\]"
    pattern_frame = r"Plan Frame ID \[\s*(\d+)\s*\]"
    parking_tasks = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()

    current_task = None
    frame_started = False

    for line in lines:
        line = remove_log_prefix(line)
        # 检测新的帧开始
        frame_match = re.search(pattern_frame, line)
        if frame_match:
            if frame_started:
                # 记录上一帧的 parking task
                parking_tasks.append(current_task)
            frame_started = True
            current_task = None

        # 提取parking task
        task_match = re.search(pattern_parking_task, line)
        if task_match and current_task is None:
            current_task = task_match.group(1)

    # 添加最后一帧
    if frame_started:
        parking_tasks.append(current_task)

    return parking_tasks

def extract_stopper_dis_record(file_path):
    """提取 Stopper dis record 的值，与 Plan Frame ID 同步"""
    pattern_stopper = r"Stopper dis record:\s*([\d.]+)"
    pattern_frame = r"Plan Frame ID \[\s*(\d+)\s*\]"
    stopper_distances = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()

    current_stopper = None
    frame_started = False

    for line in lines:
        line = remove_log_prefix(line)
        # 检测新的帧开始
        frame_match = re.search(pattern_frame, line)
        if frame_match:
            if frame_started:
                # 记录上一帧的stopper distance
                stopper_distances.append(current_stopper)
            frame_started = True
            current_stopper = None

        # 提取stopper distance
        stopper_match = re.search(pattern_stopper, line)
        if stopper_match and current_stopper is None:
            current_stopper = float(stopper_match.group(1))

    # 添加最后一帧
    if frame_started:
        stopper_distances.append(current_stopper)

    return stopper_distances

def extract_target_slot_corners_abcd(file_path):
    """提取 Target Slot Corners ABCD 坐标，与 Plan Frame ID 同步
    支持两种格式：
    1. DecPlan Output: A[0mm0mm] B[3000mm0mm]... (已经是mm，数值间无空格)
    2. DecPlan Input: A[-1.63mm -2.84mm]... (实际是m，数值间有空格)
    """
    pattern_frame = r"Plan Frame ID \[\s*(\d+)\s*\]"
    target_corners = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()

    current_corners = None
    frame_started = False

    for line in lines:
        line = remove_log_prefix(line)
        # 检测新的帧开始
        frame_match = re.search(pattern_frame, line)
        if frame_match:
            if frame_started:
                # 记录上一帧的corners
                target_corners.append(current_corners)
            frame_started = True
            current_corners = None

        # 优先尝试匹配 DecPlan Output 格式（已经是mm，数值间无空格）
        if current_corners is None and 'Target Slot Corners:' in line:
            # 使用更简单的方法：逐个提取ABCD点
            corners = {}
            try:
                # 提取A点: A[0mm0mm]
                a_match = re.search(r'A\[(\d+(?:\.\d+)?)mm(\-?\d+(?:\.\d+)?)mm\]', line)
                if a_match:
                    corners['A'] = (float(a_match.group(1)), float(a_match.group(2)))

                # 提取B点: B[3000mm0mm]
                b_match = re.search(r'B\[(\d+(?:\.\d+)?)mm(\-?\d+(?:\.\d+)?)mm\]', line)
                if b_match:
                    corners['B'] = (float(b_match.group(1)), float(b_match.group(2)))

                # 提取C点: C[3000mm-6000mm]
                c_match = re.search(r'C\[(\d+(?:\.\d+)?)mm(\-?\d+(?:\.\d+)?)mm\]', line)
                if c_match:
                    corners['C'] = (float(c_match.group(1)), float(c_match.group(2)))

                # 提取D点: D[0.0005mm-6000mm]
                d_match = re.search(r'D\[([\-]?\d+(?:\.\d+)?)mm([\-]?\d+(?:\.\d+)?)mm\]', line)
                if d_match:
                    corners['D'] = (float(d_match.group(1)), float(d_match.group(2)))

                # 如果成功提取了所有4个点，保存结果
                if len(corners) == 4:
                    current_corners = corners
            except Exception as e:
                pass  # 忽略解析错误

    # 添加最后一帧
    if frame_started:
        target_corners.append(current_corners)

    return target_corners

def extract_fork_star_starts(file_path):
    """提取 FORK STAR STARTS 标记，与 Plan Frame ID 同步
    返回: None-无标记, 'PERP'-垂直泊车, 'PARA'-水平泊车
    """
    pattern_perp = r"FORK STAR STARTS!"
    pattern_para = r"PARA FORK STAR STARTS!"
    pattern_frame = r"Plan Frame ID \[\s*(\d+)\s*\]"
    fork_star_flags = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()

    current_frame_type = None  # None, 'PERP', 'PARA'
    frame_started = False

    for line in lines:
        line = remove_log_prefix(line)
        # 检测新的帧开始
        frame_match = re.search(pattern_frame, line)
        if frame_match:
            if frame_started:
                # 记录上一帧的 fork star 类型
                fork_star_flags.append(current_frame_type)
            frame_started = True
            current_frame_type = None

        # 检测垂直泊车 FORK STAR STARTS
        if re.search(pattern_perp, line) and not re.search(pattern_para, line):
            current_frame_type = 'PERP'
        # 检测水平泊车 PARA FORK STAR STARTS
        elif re.search(pattern_para, line):
            current_frame_type = 'PARA'

    # 添加最后一帧
    if frame_started:
        fork_star_flags.append(current_frame_type)

    return fork_star_flags

def extract_xy_coordinates(file_path):
    pattern = r"No\[\d+\] x\[\s*([-+]?\d+\.?\d*)\s*mm\]\s*y\[\s*([-+]?\d+\.?\d*)\s*mm\]"
    frame_start_pattern = r"Path Segment Valid Point Num:"
    coordinates = []
    current_frame_coords = []
    is_started = False
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            for line in file:
                line = remove_log_prefix(line)
                if re.search(frame_start_pattern, line):
                    if is_started:
                        if current_frame_coords:
                            coordinates.append(current_frame_coords)
                        else:
                            coordinates.append([(0, 0)])
                        current_frame_coords = []
                    else:
                        is_started = True
                matches = re.findall(pattern, line)
                for match in matches:
                    x = float(match[0])  # 已经是mm，不需要转换
                    y = float(match[1])  # 已经是mm，不需要转换
                    current_frame_coords.append((x, y))
            if is_started:
                if current_frame_coords:
                    coordinates.append(current_frame_coords)
                else:
                    coordinates.append([(0, 0)])
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"处理文件时出现错误: {e}")
    return coordinates

def extract_log_sequence_number(filename):
    """从日志文件名中提取序号
    新格式: planning.log.YYYYMMDDHHMMSS -> 提取时间戳作为序号
    旧格式: Planning_C_cplanning_processSkel_plugin_<timestamp>_<sequence>.log -> 提取最后的sequence
    """
    # 新格式：planning.log.YYYYMMDDHHMMSS
    match = re.search(r'planning\.log\.(\d{14})$', filename)
    if match:
        return int(match.group(1))

    # 旧格式：提取最后的数字
    match = re.search(r'_(\d{2,4})\.log$', filename)
    if match:
        return int(match.group(1))
    return 0

def find_and_sort_log_files(file_path):
    """
    根据给定的日志文件，查找目录下所有同系列的日志文件，并按序号排序
    支持新旧两种格式：
    - 新格式：planning.log.YYYYMMDDHHMMSS
    - 旧格式：Planning_C_cplanning_processSkel_plugin_<timestamp>_<sequence>.log
    """
    directory = os.path.dirname(file_path) or '.'
    filename = os.path.basename(file_path)

    # 新格式：planning.log.YYYYMMDDHHMMSS
    if re.match(r'planning\.log\.\d{14}$', filename):
        # 查找所有 planning.log.* 文件
        pattern = os.path.join(directory, "planning.log.*")
        all_matching_files = glob.glob(pattern)
        # 过滤并只匹配 planning.log.14位数字 格式
        all_matching_files = [f for f in all_matching_files if re.search(r'planning\.log\.\d{14}$', os.path.basename(f))]
        all_matching_files.sort(key=lambda f: extract_log_sequence_number(f))
        if len(all_matching_files) == 0:
            return [file_path]
        return all_matching_files

    # 旧格式：Planning_C_cplanning_processSkel_plugin_<timestamp>_<sequence>.log
    match = re.match(r'(.+?)_\d{14}_\d{2,4}\.log$', filename)
    if match:
        base_prefix = match.group(1)  # 例如：Planning_C_cplanning_processSkel_plugin
        pattern = os.path.join(directory, f"{base_prefix}_*.log")
        all_matching_files = glob.glob(pattern)
        all_matching_files.sort(key=lambda f: extract_log_sequence_number(f))
        if len(all_matching_files) == 0:
            return [file_path]
        return all_matching_files

    # 如果不匹配任何已知模式，只返回单个文件
    return [file_path]

def merge_log_data(file_paths):
    """
    合并多个日志文件的数据
    返回：合并后的所有数据列表和文件边界信息
    """
    print("\n" + "="*70)
    print("正在合并以下日志文件:")
    for i, fp in enumerate(file_paths, 1):
        print(f"  {i}. {os.path.basename(fp)} (序号: {extract_log_sequence_number(fp)})")
    print("="*70 + "\n")
    
    merged_data = {
        'parking_spaces': [],
        'slot_corners': [],
        'vehicle_locations': [],
        'plan_frame_ids': [],
        'parking_function_statuses': [],
        'vehicle_stop_reasons': [],
        'parking_space_chamfers': [],
        'plan_stage_target_poses': [],
        'plan_final_target_poses': [],
        'path_current_segment_ids': [],
        'replan_ids': [],
        'coordinates': [],
        'timestamps': [],
        'p0_p5_points': [],
        'gear_info': [],
        'parking_function_stages': [],
        'parking_function_modes': [],
        'control_work_modes': [],
        'vehicle_moving_statuses': [],
        'perception_fusion_timestamps': [],
        'stopper_distances': [],
        'target_corners_abcd': [],
        'realtime_parkingspaces': [],
        'fork_star_starts': [],
        'parking_tasks': []
    }
    
    # 记录每个文件的帧范围
    file_boundaries = []  # [(start_frame, end_frame, filename), ...]
    current_frame = 0
    
    for file_path in file_paths:
        print(f"处理文件: {os.path.basename(file_path)}...", end=' ')
        
        parking_spaces = extract_parking_spaces(file_path)
        slot_corners = extract_slot_corners(file_path)
        vehicle_locations = extract_vehicle_locations(file_path)
        plan_frame_ids = extract_plan_frame_id(file_path)
        parking_function_statuses = extract_parking_function_status(file_path)
        vehicle_stop_reasons = extract_vehicle_stop_reason(file_path)
        parking_space_chamfers = extract_parking_space_chamfer(file_path)
        plan_stage_target_poses = extract_plan_stage_target_pose(file_path)
        plan_final_target_poses = extract_plan_final_target_pose(file_path)
        path_current_segment_ids = extract_path_current_segment_id(file_path)
        replan_ids = extract_replan_id(file_path)
        coordinates = extract_xy_coordinates(file_path)
        timestamps = extract_timestamps(file_path)
        p0_p5_points = extract_parking_space_p0_p5(file_path)
        gear_info = extract_gear_info(file_path)
        parking_function_stages = extract_parking_function_stage(file_path)
        parking_function_modes = extract_parking_function_mode(file_path)
        control_work_modes = extract_control_work_mode(file_path)
        vehicle_moving_statuses = extract_vehicle_moving_status(file_path)
        perception_fusion_timestamps = extract_perception_fusion_timestamps(file_path)
        stopper_distances = extract_stopper_dis_record(file_path)
        target_corners_abcd = extract_target_slot_corners_abcd(file_path)
        realtime_parkingspaces = extract_realtime_parkingspace(file_path)
        fork_star_starts = extract_fork_star_starts(file_path)
        parking_tasks = extract_parking_tasks(file_path)

        # Calculate frame_count based on key lists that should always exist
        # 使用核心数据列表计算帧数，避免因某些列表为空导致全部数据被丢弃
        list_lengths = [
            len(parking_function_statuses),
            len(vehicle_locations),
        ]
        non_empty_lengths = [l for l in list_lengths if l > 0]
        frame_count = min(non_empty_lengths) if non_empty_lengths else 0
        print(f"帧数: {frame_count} (基于核心数据列表)")

        # 记录当前文件的帧边界（只记录非空文件）
        if frame_count > 0:
            start_frame = current_frame
            end_frame = current_frame + frame_count - 1
            file_boundaries.append((start_frame, end_frame, os.path.basename(file_path)))
        current_frame += frame_count

        # 使用每个列表自己的长度进行扩展，而不是用统一的 frame_count
        # 这样可以保留每个文件中的所有数据，即使某些字段不完整
        merged_data['parking_spaces'].extend(parking_spaces)
        merged_data['slot_corners'].extend(slot_corners)
        merged_data['vehicle_locations'].extend(vehicle_locations)
        merged_data['plan_frame_ids'].extend(plan_frame_ids)
        merged_data['parking_function_statuses'].extend(parking_function_statuses)
        merged_data['vehicle_stop_reasons'].extend(vehicle_stop_reasons)
        merged_data['parking_space_chamfers'].extend(parking_space_chamfers)
        merged_data['plan_stage_target_poses'].extend(plan_stage_target_poses)
        merged_data['plan_final_target_poses'].extend(plan_final_target_poses)
        merged_data['path_current_segment_ids'].extend(path_current_segment_ids)
        merged_data['replan_ids'].extend(replan_ids)
        merged_data['coordinates'].extend(coordinates)
        merged_data['timestamps'].extend(timestamps)
        merged_data['p0_p5_points'].extend(p0_p5_points)
        merged_data['gear_info'].extend(gear_info)
        merged_data['parking_function_stages'].extend(parking_function_stages)
        merged_data['parking_function_modes'].extend(parking_function_modes)
        merged_data['control_work_modes'].extend(control_work_modes)
        merged_data['vehicle_moving_statuses'].extend(vehicle_moving_statuses)
        merged_data['perception_fusion_timestamps'].extend(perception_fusion_timestamps)
        merged_data['stopper_distances'].extend(stopper_distances)
        merged_data['target_corners_abcd'].extend(target_corners_abcd)
        merged_data['realtime_parkingspaces'].extend(realtime_parkingspaces)
        merged_data['fork_star_starts'].extend(fork_star_starts)
        merged_data['parking_tasks'].extend(parking_tasks)

    print(f"\n合并完成！总帧数: {len(merged_data['parking_function_statuses'])}\n")
    return merged_data, file_boundaries

def calculate_vehicle_corners(x_vehicle, y_vehicle, yaw_vehicle):
    vehicle_length = 5260  # 车长
    vehicle_width = 1980  # 车宽
    rear_to_center = 970  # 车辆后轴中心距离车辆后轮廓的距离
    cos_yaw = np.cos(np.radians(yaw_vehicle))
    sin_yaw = np.sin(np.radians(yaw_vehicle))
    rear_left = (x_vehicle - rear_to_center * cos_yaw - vehicle_width / 2 * sin_yaw, y_vehicle - rear_to_center * sin_yaw + vehicle_width / 2 * cos_yaw)
    rear_right = (x_vehicle - rear_to_center * cos_yaw + vehicle_width / 2 * sin_yaw, y_vehicle - rear_to_center * sin_yaw - vehicle_width / 2 * cos_yaw)
    front_left = (x_vehicle + (vehicle_length - rear_to_center) * cos_yaw - vehicle_width / 2 * sin_yaw, y_vehicle + (vehicle_length - rear_to_center) * sin_yaw + vehicle_width / 2 * cos_yaw)
    front_right = (x_vehicle + (vehicle_length - rear_to_center) * cos_yaw + vehicle_width / 2 * sin_yaw, y_vehicle + (vehicle_length - rear_to_center) * sin_yaw - vehicle_width / 2 * cos_yaw)
    corners = [
        rear_left,
        rear_right,
        front_right,
        front_left,
        rear_left
    ]
    return corners

def draw_detailed_vehicle(ax, x_vehicle, y_vehicle, yaw_vehicle):
    """绘制详细的车辆形状，包括车身、车窗、车轮等"""
    vehicle_length = 5260  # 车长 (mm)
    vehicle_width = 1980   # 车宽 (mm)
    rear_to_center = 970   # 后轴到车尾的距离 (mm)
    
    cos_yaw = np.cos(np.radians(yaw_vehicle))
    sin_yaw = np.sin(np.radians(yaw_vehicle))
    
    def rotate_point(x, y):
        """将局部坐标转换为全局坐标"""
        global_x = x_vehicle + x * cos_yaw - y * sin_yaw
        global_y = y_vehicle + x * sin_yaw + y * cos_yaw
        return global_x, global_y
    
    # 1. 绘制车身主体（深色填充）
    body_x_local = [-rear_to_center, vehicle_length - rear_to_center, 
                    vehicle_length - rear_to_center, -rear_to_center, -rear_to_center]
    body_y_local = [-vehicle_width/2, -vehicle_width/2, 
                    vehicle_width/2, vehicle_width/2, -vehicle_width/2]
    body_x, body_y = [], []
    for x, y in zip(body_x_local, body_y_local):
        gx, gy = rotate_point(x, y)
        body_x.append(gx)
        body_y.append(gy)
    
    ax.fill(body_x, body_y, color='#2C3E50', alpha=0.7, zorder=5)
    ax.plot(body_x, body_y, color='#34495E', linewidth=3, zorder=6)
    
    # 2. 绘制车窗（浅色）
    # 前挡风玻璃
    windshield_start = vehicle_length - rear_to_center - 1200
    windshield_end = vehicle_length - rear_to_center - 200
    windshield_x_local = [windshield_start, windshield_end, windshield_end, windshield_start, windshield_start]
    windshield_y_local = [-vehicle_width/2 * 0.6, -vehicle_width/2 * 0.6, 
                          vehicle_width/2 * 0.6, vehicle_width/2 * 0.6, -vehicle_width/2 * 0.6]
    windshield_x, windshield_y = [], []
    for x, y in zip(windshield_x_local, windshield_y_local):
        gx, gy = rotate_point(x, y)
        windshield_x.append(gx)
        windshield_y.append(gy)
    ax.fill(windshield_x, windshield_y, color='#3498DB', alpha=0.4, zorder=7)
    
    # 3. 绘制四个车轮（黑色矩形）
    wheel_length = 600
    wheel_width = 300
    front_wheel_x = vehicle_length - rear_to_center - 1000  # 前轮位置
    rear_wheel_x = 0  # 后轮位置（相对于后轴中心）
    
    # 左前轮
    wheel_corners = [
        (front_wheel_x - wheel_length/2, vehicle_width/2 - wheel_width/2),
        (front_wheel_x + wheel_length/2, vehicle_width/2 - wheel_width/2),
        (front_wheel_x + wheel_length/2, vehicle_width/2 + wheel_width/2),
        (front_wheel_x - wheel_length/2, vehicle_width/2 + wheel_width/2),
        (front_wheel_x - wheel_length/2, vehicle_width/2 - wheel_width/2)
    ]
    wheel_x, wheel_y = [], []
    for x, y in wheel_corners:
        gx, gy = rotate_point(x, y)
        wheel_x.append(gx)
        wheel_y.append(gy)
    ax.fill(wheel_x, wheel_y, color='#000000', alpha=0.8, zorder=8)
    
    # 右前轮
    wheel_corners = [
        (front_wheel_x - wheel_length/2, -vehicle_width/2 - wheel_width/2),
        (front_wheel_x + wheel_length/2, -vehicle_width/2 - wheel_width/2),
        (front_wheel_x + wheel_length/2, -vehicle_width/2 + wheel_width/2),
        (front_wheel_x - wheel_length/2, -vehicle_width/2 + wheel_width/2),
        (front_wheel_x - wheel_length/2, -vehicle_width/2 - wheel_width/2)
    ]
    wheel_x, wheel_y = [], []
    for x, y in wheel_corners:
        gx, gy = rotate_point(x, y)
        wheel_x.append(gx)
        wheel_y.append(gy)
    ax.fill(wheel_x, wheel_y, color='#000000', alpha=0.8, zorder=8)
    
    # 左后轮
    wheel_corners = [
        (rear_wheel_x - wheel_length/2, vehicle_width/2 - wheel_width/2),
        (rear_wheel_x + wheel_length/2, vehicle_width/2 - wheel_width/2),
        (rear_wheel_x + wheel_length/2, vehicle_width/2 + wheel_width/2),
        (rear_wheel_x - wheel_length/2, vehicle_width/2 + wheel_width/2),
        (rear_wheel_x - wheel_length/2, vehicle_width/2 - wheel_width/2)
    ]
    wheel_x, wheel_y = [], []
    for x, y in wheel_corners:
        gx, gy = rotate_point(x, y)
        wheel_x.append(gx)
        wheel_y.append(gy)
    ax.fill(wheel_x, wheel_y, color='#000000', alpha=0.8, zorder=8)
    
    # 右后轮
    wheel_corners = [
        (rear_wheel_x - wheel_length/2, -vehicle_width/2 - wheel_width/2),
        (rear_wheel_x + wheel_length/2, -vehicle_width/2 - wheel_width/2),
        (rear_wheel_x + wheel_length/2, -vehicle_width/2 + wheel_width/2),
        (rear_wheel_x - wheel_length/2, -vehicle_width/2 + wheel_width/2),
        (rear_wheel_x - wheel_length/2, -vehicle_width/2 - wheel_width/2)
    ]
    wheel_x, wheel_y = [], []
    for x, y in wheel_corners:
        gx, gy = rotate_point(x, y)
        wheel_x.append(gx)
        wheel_y.append(gy)
    ax.fill(wheel_x, wheel_y, color='#000000', alpha=0.8, zorder=8)
    
    # 4. 绘制车头方向指示器（小三角形）
    arrow_tip_x = vehicle_length - rear_to_center + 200
    arrow_base_x = vehicle_length - rear_to_center - 300
    arrow_corners = [
        (arrow_tip_x, 0),
        (arrow_base_x, -vehicle_width/4),
        (arrow_base_x, vehicle_width/4),
        (arrow_tip_x, 0)
    ]
    arrow_x, arrow_y = [], []
    for x, y in arrow_corners:
        gx, gy = rotate_point(x, y)
        arrow_x.append(gx)
        arrow_y.append(gy)
    ax.fill(arrow_x, arrow_y, color='#E74C3C', alpha=0.8, zorder=9)
    
    # 5. 标记后轴中心点
    ax.plot(x_vehicle, y_vehicle, marker='o', color='#FFD700', 
            markersize=6, markeredgecolor='white', markeredgewidth=1.5, zorder=10)

def update(frame, parking_spaces, slot_corners, vehicle_locations, plan_frame_ids, parking_function_statuses,
           vehicle_stop_reasons, parking_space_chamfers, plan_stage_target_poses, plan_final_target_poses,
           path_current_segment_ids, replan_ids, coordinates, ax, timestamps, p0_p5_points, gear_info,
           parking_function_stages, parking_function_modes, control_work_modes, vehicle_moving_statuses, perception_fusion_timestamps,
           stopper_distances, target_corners_abcd, realtime_parkingspaces, fork_star_starts, parking_tasks):

    global prev_stop_reason, extra_frames_to_show, all_rear_axle_centers, all_vehicle_arrows, show_arrows, file_boundaries, current_log_text, user_adjusted_view
    
    # 如果用户已调整视角，保存当前的坐标范围
    saved_xlim = None
    saved_ylim = None
    if user_adjusted_view:
        saved_xlim = ax.get_xlim()
        saved_ylim = ax.get_ylim()
    
    ax.cla()  # 清空坐标轴
    
    # 如果用户已调整视角，立即恢复保存的范围
    if user_adjusted_view and saved_xlim and saved_ylim:
        ax.set_xlim(saved_xlim)
        ax.set_ylim(saved_ylim)
    
    # 确定当前帧所属的log文件
    current_log_name = "Unknown"
    if file_boundaries:
        for start_frame, end_frame, filename in file_boundaries:
            if start_frame <= frame <= end_frame:
                current_log_name = filename
                break

    if frame < len(parking_function_statuses) and parking_function_statuses[frame] == 0:
        all_rear_axle_centers = []
        all_vehicle_arrows = []

    if frame < len(p0_p5_points):
        p0, p5 = p0_p5_points[frame]
        ax.plot(p0[0], p0[1], marker='D', color=COLORS['p0_p5_fused'], markersize=8,
                markeredgecolor='white', markeredgewidth=1.5, label='Fused P0/P5', zorder=10)
        ax.plot(p5[0], p5[1], marker='D', color=COLORS['p0_p5_fused'], markersize=8,
                markeredgecolor='white', markeredgewidth=1.5, zorder=10)
        # 只显示标签，不显示坐标
        ax.text(p0[0], p0[1] + 200, 'P0', color=COLORS['p0_p5_fused'], fontsize=11,
                ha='center', weight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        ax.text(p5[0], p5[1] + 200, 'P5', color=COLORS['p0_p5_fused'], fontsize=11,
                ha='center', weight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Parking Space (DecPlan Input) 显示已禁用
    # 原因：DecPlan Input 数据包含坐标转换前的原始值，P1-P4为(0,0)且位置不准确
    # 正确的停车位显示使用 Slot Corners (坐标转换后的数据)，见下方代码
    # if frame < len(parking_spaces):
    #     parking_points = parking_spaces[frame]
    #
    #     # 过滤掉 (0, 0) 的点，避免连接到原点
    #     valid_points = [(x, y) for x, y in parking_points if not (x == 0 and y == 0)]
    #
    #     if len(valid_points) >= 3:  # 至少需要3个点才能绘制多边形
    #         x_parking = [point[0] for point in valid_points]
    #         y_parking = [point[1] for point in valid_points]
    #         x_parking.append(valid_points[0][0])  # 闭合多边形
    #         y_parking.append(valid_points[0][1])
    #         ax.plot(x_parking, y_parking, marker='o', markersize=6, linestyle='-',
    #                 color=COLORS['parking_space'], linewidth=2.5, label='Parking Space',
    #                 markerfacecolor=COLORS['parking_space'], markeredgecolor='white',
    #                 markeredgewidth=1.5, alpha=0.8)
    #         ax.fill(x_parking, y_parking, color=COLORS['parking_space'], alpha=0.1)
    #
    #     # 标注 P0-P7 角点（只标注非零点）
    #     labels = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
    #     # 需要显示坐标的点索引（P0=0, P5=5, P6=6, P7=7）
    #     show_coord_indices = [0, 5, 6, 7]
    #     for i, (x, y) in enumerate(parking_points):
    #         if not (x == 0 and y == 0):  # 只标注非零点
    #             if i in show_coord_indices:
    #                 # 显示标签和坐标值
    #                 label_with_coord = f'{labels[i]}\n({x:.1f}, {y:.1f})'
    #                 ax.text(x, y, label_with_coord,
    #                        fontsize=9, fontweight='bold',
    #                        color='white',
    #                        bbox=dict(boxstyle='round,pad=0.4',
    #                                 facecolor=COLORS['parking_space'],
    #                                 edgecolor='white',
    #                                 linewidth=1.5,
    #                                 alpha=0.9),
    #                        ha='center', va='center',
    #                        zorder=14)
    #             else:
    #                 # 只显示标签
    #                 ax.text(x, y, labels[i],
    #                        fontsize=10, fontweight='bold',
    #                        color='white',
    #                        bbox=dict(boxstyle='circle,pad=0.25',
    #                                  facecolor=COLORS['parking_space'],
    #                                  edgecolor='white',
    #                                  linewidth=1.5,
    #                                  alpha=0.9),
    #                        ha='center', va='center',
    #                        zorder=14)

    if frame < len(slot_corners):
        slot_points = slot_corners[frame]
        x_slot = [point[0] for point in slot_points]
        y_slot = [point[1] for point in slot_points]
        x_slot.append(slot_points[0][0])  # 闭合四边形
        y_slot.append(slot_points[0][1])
        ax.plot(x_slot, y_slot, marker='s', markersize=5, linestyle='--', 
                color=COLORS['slot_corners'], linewidth=2, label='Slot Corners', 
                markerfacecolor=COLORS['slot_corners'], markeredgecolor='white', 
                markeredgewidth=1, alpha=0.9)
        ax.fill(x_slot, y_slot, color=COLORS['slot_corners'], alpha=0.15)
        
        # 标注 A, B, C, D 角点
        labels = ['A', 'B', 'C', 'D']
        for i, (x, y) in enumerate(slot_points):
            ax.text(x, y, labels[i], 
                   fontsize=12, fontweight='bold',
                   color='white', 
                   bbox=dict(boxstyle='circle,pad=0.3', 
                            facecolor=COLORS['slot_corners'], 
                            edgecolor='white', 
                            linewidth=2,
                            alpha=0.9),
                   ha='center', va='center',
                   zorder=15)
    
    # 绘制 Stopper Distance 到 CD 边的距离
    if (frame < len(stopper_distances) and frame < len(target_corners_abcd)):
        # 获取对应帧的数据（可能为空字典）
        if stopper_distances[frame] and target_corners_abcd[frame]:
            stopper_dist = stopper_distances[frame]
            corners_abcd = target_corners_abcd[frame]
        else:
            stopper_dist = None
            corners_abcd = None
    else:
        stopper_dist = None
        corners_abcd = None
    
    if stopper_dist is not None and corners_abcd is not None:
        
        if 'C' in corners_abcd and 'D' in corners_abcd:
            C = corners_abcd['C']
            D = corners_abcd['D']
            
            # 计算 CD 边的中点
            cd_mid_x = (C[0] + D[0]) / 2
            cd_mid_y = (C[1] + D[1]) / 2
            
            # 计算 CD 边的方向向量
            cd_vec_x = C[0] - D[0]
            cd_vec_y = C[1] - D[1]
            cd_length = np.sqrt(cd_vec_x**2 + cd_vec_y**2)
            
            if cd_length > 0:
                # 计算 CD 边的法向量（垂直方向，指向停车位内部）
                normal_x = -cd_vec_y / cd_length
                normal_y = cd_vec_x / cd_length
                
                # 计算 stopper distance 位置（从 CD 边沿法向量方向）
                stopper_x = cd_mid_x + normal_x * stopper_dist
                stopper_y = cd_mid_y + normal_y * stopper_dist
                
                # 绘制从 stopper 位置到 CD 边中点的线
                ax.plot([stopper_x, cd_mid_x], [stopper_y, cd_mid_y], 
                       color='#FF6B6B', linewidth=2.5, linestyle='-', 
                       alpha=0.8, zorder=16, label='Stopper Distance to CD')
                
                # 在 stopper 位置标记点
                ax.plot(stopper_x, stopper_y, marker='o', color='#FF6B6B', 
                       markersize=8, markeredgecolor='white', markeredgewidth=2, zorder=17)
                
                # 在距离线的中点添加文本标注
                text_x = (stopper_x + cd_mid_x) / 2
                text_y = (stopper_y + cd_mid_y) / 2
                ax.text(text_x, text_y, f'Stopper: {stopper_dist:.1f}mm', 
                       fontsize=11, fontweight='bold',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='#FF6B6B', 
                                edgecolor='white', 
                                linewidth=2,
                                alpha=0.9),
                       ha='center', va='center',
                       zorder=18)
    
    # 绘制 Realtime updating parkingspace (p0-p7)
    if frame < len(realtime_parkingspaces) and realtime_parkingspaces[frame] is not None:
        rt_points = realtime_parkingspaces[frame]
        if len(rt_points) == 8:  # 确保有完整的8个点 (p0-p7)
            # 绘制 p0-p7 形成的区域
            x_rt = [point[0] for point in rt_points]
            y_rt = [point[1] for point in rt_points]
            
            # 按顺序连接点形成多边形: p0->p1->p2->p3->p4->p5->p6->p7->p0
            x_rt.append(rt_points[0][0])  # 闭合多边形
            y_rt.append(rt_points[0][1])
            
            # 绘制边界线
            ax.plot(x_rt, y_rt, linestyle='-.', linewidth=2, 
                   color='#00CED1', alpha=0.7, zorder=13, label='Realtime Parkingspace')
            
            # 填充半透明区域
            ax.fill(x_rt, y_rt, color='#00CED1', alpha=0.08, zorder=12)
            
            # 标注点 p0-p7
            labels = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']
            for i, (x, y) in enumerate(rt_points):
                # 只标注非零点，避免太多标签
                if abs(x) > 1 or abs(y) > 1:
                    ax.plot(x, y, marker='D', markersize=4, 
                           color='#00CED1', markeredgecolor='white', 
                           markeredgewidth=1, zorder=14)
                    # 可选：添加点标签（如果需要）
                    # ax.text(x, y, labels[i], fontsize=8, color='#00CED1',
                    #        ha='right', va='bottom', zorder=14)

    if frame < len(vehicle_locations):
        x_vehicle, y_vehicle, yaw_vehicle = vehicle_locations[frame]
        
        # 使用详细的车辆绘制函数
        draw_detailed_vehicle(ax, x_vehicle, y_vehicle, yaw_vehicle)

        if show_arrows:
            arrow_length = 800  # 箭头长度，增加长度使其更明显
            arrow_dx = arrow_length * np.cos(np.radians(yaw_vehicle))
            arrow_dy = arrow_length * np.sin(np.radians(yaw_vehicle))
            arrow = ax.arrow(x_vehicle, y_vehicle, arrow_dx, arrow_dy, 
                           head_width=300, head_length=400, 
                           fc=COLORS['vehicle_arrow'], ec=COLORS['vehicle_arrow'],
                           linewidth=2.5, alpha=0.8, zorder=6)

            all_vehicle_arrows.append((x_vehicle, y_vehicle, arrow_dx, arrow_dy))

            for arrow_info in all_vehicle_arrows[:-1]:  # 不包括最后一个（当前箭头）
                ax.arrow(arrow_info[0], arrow_info[1], arrow_info[2], arrow_info[3], 
                        head_width=250, head_length=350,
                        fc=COLORS['vehicle_arrow'], ec=COLORS['vehicle_arrow'],
                        linewidth=1.5, alpha=0.3, zorder=4)

        # 安全地添加后轴中心点（处理 path_current_segment_ids 不完整的情况）
        if frame < len(path_current_segment_ids):
            all_rear_axle_centers.append((x_vehicle, y_vehicle, path_current_segment_ids[frame]))
            current_segment_id = path_current_segment_ids[frame]
            prev_segment_id = current_segment_id - 1

            filtered_centers = []
            for center in all_rear_axle_centers:
                if center[2] in [current_segment_id, prev_segment_id]:
                    filtered_centers.append(center[:2])

            if filtered_centers:
                x_centers = [point[0] for point in filtered_centers]
                y_centers = [point[1] for point in filtered_centers]
                ax.plot(x_centers, y_centers, marker='o', color=COLORS['rear_axle'],
                       linestyle='', markersize=4, alpha=0.5, markeredgecolor='white',
                       markeredgewidth=0.5, label='Rear Axle Trail')
       
        if frame < len(coordinates):
            current_coords = coordinates[frame]
            traj_x = []
            traj_y = []
            for point in current_coords:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    traj_x.append(point[0])
                    traj_y.append(point[1])
                    x = point[0]
                    y = point[1]
                else:
                    print(f"Invalid point in frame {frame}: {point}")
            if traj_x and traj_y:
                ax.plot(traj_x, traj_y, marker='o', color=COLORS['trajectory'],
                       linestyle='-', markersize=4, linewidth=2.5, alpha=0.85,
                       markerfacecolor=COLORS['trajectory'], markeredgecolor='white',
                       markeredgewidth=0.8, label='Planned Trajectory', zorder=3)
        # 注意：coordinates 可能不完整，跳过无坐标的帧（正常情况）

    # 显示 FORK STAR STARTS 规划开始标记
    if frame < len(fork_star_starts) and fork_star_starts[frame]:
        plan_type = fork_star_starts[frame]
        if frame < len(vehicle_locations):
            x_v, y_v, _ = vehicle_locations[frame]

            # 根据类型设置颜色和文字
            if plan_type == 'PERP':
                star_color = '#FF00FF'  # 紫色 - 垂直泊车
                label_text = 'PERP FORK STAR!'
                panel_text = 'PERPENDICULAR PLANNING'
            elif plan_type == 'PARA':
                star_color = '#00CCFF'  # 蓝色 - 水平泊车
                label_text = 'PARA FORK STAR!'
                panel_text = 'PARALLEL PLANNING'
            else:
                star_color = '#FF00FF'
                label_text = 'FORK STAR STARTS!'
                panel_text = 'FORWARD PLANNING START'

            # 显示星形标记 (用 scatter 绘制)
            ax.scatter([x_v], [y_v + 1500], marker='*', s=500, color=star_color,
                      edgecolors='white', linewidths=2, zorder=20)
            # 显示文字提示
            ax.text(x_v, y_v + 2200, label_text,
                   color=star_color, fontsize=14, fontweight='bold',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=star_color, alpha=0.3, edgecolor=star_color, linewidth=2),
                   zorder=21)
            # 在右侧信息面板也显示
            ax.text(0.95, 0.94, panel_text, color=star_color, fontsize=12, fontweight='bold',
                   transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=star_color, alpha=0.3, edgecolor=star_color, linewidth=2))

    if frame < len(timestamps):
        timestamp = timestamps[frame]
        bj_time = convert_timestamp_to_bj_time(timestamp)
        ax.text(0.95, 0.60, f"Vehicle Time: {bj_time}", color='black', fontsize=11, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='bottom')
    
    if frame < len(perception_fusion_timestamps):
        perception_timestamp = perception_fusion_timestamps[frame]
        perception_bj_time = convert_timestamp_to_bj_time(perception_timestamp)
        ax.text(0.95, 0.56, f"Perception Time: {perception_bj_time}  regular: {perception_timestamp}", color='darkblue', fontsize=11, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='bottom')
    
    if frame < len(parking_function_stages):
        stage = parking_function_stages[frame]
        stage_str = func_stage_mapping.get(stage, f"Unknown({stage})")
        ax.text(0.95, 0.52, f"PF Stage: {stage_str}", color='purple', fontsize=12, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='bottom')
    
    if frame < len(parking_function_modes):
        mode = parking_function_modes[frame]
        mode_str = func_mode_mapping.get(mode, f"Unknown({mode})")
        ax.text(0.95, 0.48, f"PF Mode: {mode_str}", color='purple', fontsize=12, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='bottom')
    
    if frame < len(parking_function_statuses):
        status = parking_function_statuses[frame]
        status_str = func_status_mapping.get(status, f"Unknown({status})")
        ax.text(0.95, 0.44, f"PF Status: {status_str}", color='purple', fontsize=12, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='bottom')
        
    if frame < len(gear_info):
        current_gear = gear_info[frame]
        ax.text(0.95, 0.4, f"Gear: {current_gear}", color='green', fontsize=12, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='bottom')

    if frame < len(replan_ids):
        replan_id = replan_ids[frame]
        ax.text(0.95, 0.35, f"Replan ID: {replan_id}", color='green', fontsize=12, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='bottom')
        
    if frame < len(vehicle_stop_reasons):
        current_stop_reason = vehicle_stop_reasons[frame]
        if current_stop_reason == 0 and prev_stop_reason is not None:
            show_stop_reason = prev_stop_reason
        else:
            show_stop_reason = current_stop_reason
            prev_stop_reason = current_stop_reason
        stop_reason_str = control_stop_reason_mapping.get(show_stop_reason, f"Unknown({show_stop_reason})")
        ax.text(0.95, 0.3, f"Stop Reason: {stop_reason_str}", color='green', fontsize=12,
                transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom')

    if frame < len(parking_tasks):
        task = parking_tasks[frame]
        if task:
            ax.text(0.95, 0.34, f"Parking Task: {task}", color='orange', fontsize=12,
                    transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom')

    if frame < len(control_work_modes):
        work_mode = control_work_modes[frame]
        work_mode_str = control_work_mode_mapping.get(work_mode, f"Unknown({work_mode})")
        ax.text(0.95, 0.26, f"Ctrl Mode: {work_mode_str}", color='darkorange', fontsize=12,
                transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom')
    
    if frame < len(vehicle_moving_statuses):
        moving_status = vehicle_moving_statuses[frame]
        moving_status_str = vehicle_moving_status_mapping.get(moving_status, f"Unknown({moving_status})")
        ax.text(0.95, 0.22, f"Veh Status: {moving_status_str}", color='darkorange', fontsize=12,
                transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom')

    if frame < len(path_current_segment_ids):
        path_current_segment_id = path_current_segment_ids[frame]
        ax.text(0.95, 0.18, f"Path Segment ID: {path_current_segment_id}", color='green', fontsize=12,
                transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom')

    if frame < len(plan_stage_target_poses):
       plan_stage_target_poses_ID = plan_stage_target_poses[frame]
       ax.text(0.95, 0.14, f"Stage Target: {plan_stage_target_poses_ID}", color='green', fontsize=11,
                transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom')
       
    if frame < len(plan_final_target_poses):
       plan_final_target_poses_ID = plan_final_target_poses[frame]
       ax.text(0.95, 0.10, f"Final Target: {plan_final_target_poses_ID}", color='limegreen', fontsize=11,
                transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom')
       
    if frame < len(vehicle_locations):
       vehicle_location_id = vehicle_locations[frame]
       ax.text(0.95, 0.06, f"Current Loc: {vehicle_location_id}", color='red', fontsize=11,
                transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom')
    if frame < len(p0_p5_points):
       p0_p5_points_id= p0_p5_points[frame]
       ax.text(0.95, 0.02, f"P0 P5: {p0_p5_points_id}", color='darkviolet', fontsize=10,
                transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom')

  
    if frame < len(parking_space_chamfers):
        chamfer_points = parking_space_chamfers[frame]
        if len(chamfer_points) >= 4:
            def process_point(point):
                if len(point) == 2:
                    x = point[0] if abs(point[0]) <= 20000 else 0
                    y = point[1] if abs(point[1]) <= 20000 else 0
                    return (x, y)
                return point

            p0_aisle = process_point(chamfer_points[0])
            p0_slot = process_point(chamfer_points[1])
            p5_aisle = process_point(chamfer_points[2])
            p5_slot = process_point(chamfer_points[3])

            ax.plot([p0_aisle[0], p0_slot[0]], [p0_aisle[1], p0_slot[1]], linestyle='-', color='black')
            ax.plot([p5_aisle[0], p5_slot[0]], [p5_aisle[1], p5_slot[1]], linestyle='-', color='black')
        else:
            print(f"Insufficient chamfer points in frame {frame}: {chamfer_points}")

    if frame < len(plan_stage_target_poses):
        x_target, y_target, yaw_target = plan_stage_target_poses[frame]
        target_corners = calculate_vehicle_corners(x_target, y_target, yaw_target)
        x_target_contour = [point[0] for point in target_corners]
        y_target_contour = [point[1] for point in target_corners]
        ax.plot(x_target_contour, y_target_contour, marker='', color=COLORS['target_stage'], 
               markersize=6, linestyle='--', linewidth=2.5, label='Stage Target', 
               alpha=0.7, zorder=4)
        ax.fill(x_target_contour, y_target_contour, color=COLORS['target_stage'], alpha=0.2)
        ax.plot(x_target, y_target, marker='*', color=COLORS['target_stage'], 
               markersize=12, markeredgecolor='white', markeredgewidth=1.5, zorder=5)

    if frame < len(plan_final_target_poses):
        x_final, y_final, yaw_final = plan_final_target_poses[frame]
        final_corners = calculate_vehicle_corners(x_final, y_final, yaw_final)
        x_final_contour = [point[0] for point in final_corners]
        y_final_contour = [point[1] for point in final_corners]
        ax.plot(x_final_contour, y_final_contour, marker='', color=COLORS['target_final'], 
               markersize=6, linestyle='-.', linewidth=2.5, label='Final Target',
               alpha=0.8, zorder=4)
        ax.fill(x_final_contour, y_final_contour, color=COLORS['target_final'], alpha=0.2)
        ax.plot(x_final, y_final, marker='*', color=COLORS['target_final'], 
               markersize=15, markeredgecolor='white', markeredgewidth=2, zorder=5)

    global current_frame_index, valid_frames, progress_slider, updating_progress
    try:
        current_frame_index = valid_frames.index(frame)
    except ValueError:
        current_frame_index = 0
    
    if 'progress_slider' in globals() and progress_slider is not None and not updating_progress:
        updating_progress = True
        progress_slider.set_val(current_frame_index)
        updating_progress = False
    
    progress_percent = (current_frame_index + 1) / len(valid_frames) * 100
    title_text = f"Parking Planning Visualization | Frame {frame} ({current_frame_index + 1}/{len(valid_frames)}) | Progress: {progress_percent:.1f}%"
    ax.set_title(title_text, fontsize=13, fontweight='bold', pad=15, 
                color=COLORS['text'], backgroundcolor='white', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F4F8', alpha=0.85, edgecolor='#2E86AB', linewidth=2))
    
    ax.set_xlabel('X Position (mm)', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Y Position (mm)', fontsize=11, fontweight='bold', color=COLORS['text'])
    
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4, color=COLORS['grid'])
    ax.set_facecolor(COLORS['background'])
    
    # 使用固定的坐标轴范围（仅在用户未手动调整时）
    global fixed_xlim, fixed_ylim
    if not user_adjusted_view:
        if fixed_xlim and fixed_ylim:
            ax.set_xlim(fixed_xlim)
            ax.set_ylim(fixed_ylim)
    # 如果用户已调整，范围已在ax.cla()之后恢复，这里不做处理
    
    ax.set_aspect('equal', adjustable='box')
    
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
                      ncol=2, fontsize=9, framealpha=0.9, edgecolor='#2E86AB', 
                      facecolor='white', title='Elements', title_fontsize=10)
    legend.get_title().set_fontweight('bold')
    
    # 显示当前正在播放的log文件
    if file_boundaries and current_log_name != "Unknown":
        ax.text(0.98, 0.98, f'Now we are playing : {current_log_name}', 
                transform=ax.transAxes, fontsize=11, weight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7, edgecolor='orange', linewidth=2))

    next_frame = frame + 1
    while next_frame < len(parking_function_statuses) and parking_function_statuses[next_frame] not in [1, 2]:
        if parking_function_statuses[next_frame] == 0:
            all_rear_axle_centers = []
        next_frame += 1
    return next_frame

def toggle_pause(event=None):
    global is_paused, current_frame_index, valid_frames, ani
    if is_paused:
        ani.event_source.start()  # 恢复动画
        button.label.set_text("Pause")  # 修改按钮标签
        ani.frame_seq = iter(valid_frames[current_frame_index:])
        ani.event_source.frame_seq = ani.frame_seq
    else:
        ani.event_source.stop()  # 暂停动画
        button.label.set_text("Play")  # 修改按钮标签
        current_frame_index = valid_frames.index(ani.frame_seq.__next__())  # 获取当前帧索引
    is_paused = not is_paused  # 切换状态

def next_frame(event=None):
    global current_frame_index, valid_frames, parking_spaces, slot_corners, vehicle_locations, plan_frame_ids, \
           parking_function_statuses, vehicle_stop_reasons, parking_space_chamfers, plan_stage_target_poses, \
           plan_final_target_poses, path_current_segment_ids, replan_ids, coordinates, ax, fig, is_paused, timestamps, p0_p5_points, gear_info, \
           parking_function_stages, parking_function_modes, control_work_modes, vehicle_moving_statuses, perception_fusion_timestamps, \
           stopper_distances, target_corners_abcd, realtime_parkingspaces, fork_star_starts, parking_tasks
    # 逐帧前进：无论当前是否暂停，均移动到下一帧并刷新显示。
    # 如果当前未暂停，先暂停动画以保证逐帧行为一致。
    if not is_paused:
        try:
            toggle_pause()
        except Exception:
            # 在极少数情况下 toggle_pause 可能会抛出异常，继续执行逐帧逻辑
            pass

    current_frame_index = (current_frame_index + 1) % len(valid_frames)
    frame = valid_frames[current_frame_index]
    update(frame, parking_spaces, slot_corners, vehicle_locations, plan_frame_ids,
           parking_function_statuses, vehicle_stop_reasons, parking_space_chamfers, plan_stage_target_poses,
           plan_final_target_poses, path_current_segment_ids, replan_ids, coordinates, ax, timestamps, p0_p5_points, gear_info,
           parking_function_stages, parking_function_modes, control_work_modes, vehicle_moving_statuses, perception_fusion_timestamps,
           stopper_distances, target_corners_abcd, realtime_parkingspaces, fork_star_starts, parking_tasks)
    try:
        fig.canvas.draw_idle()
    except Exception:
        pass

def prev_frame(event=None):
    global current_frame_index, valid_frames, parking_spaces, slot_corners, vehicle_locations, plan_frame_ids, \
           parking_function_statuses, vehicle_stop_reasons, parking_space_chamfers, plan_stage_target_poses, \
           plan_final_target_poses, path_current_segment_ids, replan_ids, coordinates, ax, fig, is_paused, timestamps, p0_p5_points, gear_info, \
           parking_function_stages, parking_function_modes, control_work_modes, vehicle_moving_statuses, perception_fusion_timestamps, \
           stopper_distances, target_corners_abcd, realtime_parkingspaces, fork_star_starts, parking_tasks
    # 逐帧后退：无论当前是否暂停，均移动到上一帧并刷新显示。
    if not is_paused:
        try:
            toggle_pause()
        except Exception:
            pass

    current_frame_index = (current_frame_index - 1) % len(valid_frames)
    frame = valid_frames[current_frame_index]
    update(frame, parking_spaces, slot_corners, vehicle_locations, plan_frame_ids,
           parking_function_statuses, vehicle_stop_reasons, parking_space_chamfers, plan_stage_target_poses,
           plan_final_target_poses, path_current_segment_ids, replan_ids, coordinates, ax, timestamps, p0_p5_points, gear_info,
           parking_function_stages, parking_function_modes, control_work_modes, vehicle_moving_statuses, perception_fusion_timestamps,
           stopper_distances, target_corners_abcd, realtime_parkingspaces, fork_star_starts, parking_tasks)
    try:
        fig.canvas.draw_idle()
    except Exception:
        pass
        fig.canvas.draw_idle()

def toggle_arrows(event=None):
    global show_arrows
    show_arrows = not show_arrows  # 切换箭头显示状态
    if show_arrows:
        button_arrows.label.set_text("Hide Yaw")  # 修改按钮标签
    else:
        button_arrows.label.set_text("Show Yaw")  # 修改按钮标签

    update(valid_frames[current_frame_index], parking_spaces, slot_corners, vehicle_locations, plan_frame_ids,
           parking_function_statuses, vehicle_stop_reasons, parking_space_chamfers, plan_stage_target_poses,
           plan_final_target_poses, path_current_segment_ids, replan_ids, coordinates, ax, timestamps, p0_p5_points, gear_info,
           parking_function_stages, parking_function_modes, control_work_modes, vehicle_moving_statuses, perception_fusion_timestamps,
           stopper_distances, target_corners_abcd, realtime_parkingspaces, fork_star_starts, parking_tasks)
    fig.canvas.draw_idle()

def on_mouse_scroll(event):
    global user_adjusted_view
    if event.button == 'up':  # 向上滚动，放大
        scale_factor = 0.9
    elif event.button == 'down':  # 向下滚动，缩小
        scale_factor = 1.1
    else:
        return

    xdata, ydata = event.xdata, event.ydata
    if xdata is None or ydata is None:
        return  # 鼠标指针不在图内

    new_xlim = [xdata + (xlim - xdata) * scale_factor for xlim in ax.get_xlim()]
    new_ylim = [ydata + (ylim - ydata) * scale_factor for ylim in ax.get_ylim()]

    ax.set_xlim(new_xlim)
    ax.set_ylim(new_ylim)
    user_adjusted_view = True  # 标记用户已手动调整视角
    fig.canvas.draw_idle()
    
def on_mouse_press(event):
    if event.button == 2:  # 鼠标滚轮按下
        global press_x, press_y
        press_x, press_y = event.xdata, event.ydata

def on_mouse_move(event):
    global user_adjusted_view
    if event.button == 2:  # 鼠标滚轮按下
        global press_x, press_y
        if press_x is not None and press_y is not None:
            dx = event.xdata - press_x
            dy = event.ydata - press_y
            press_x, press_y = event.xdata, event.ydata
            ax.set_xlim(ax.get_xlim()[0] - dx, ax.get_xlim()[1] - dx)
            ax.set_ylim(ax.get_ylim()[0] - dy, ax.get_ylim()[1] - dy)
            user_adjusted_view = True  # 标记用户已手动调整视角
            fig.canvas.draw_idle()

def select_log_file():
    """选择日志文件，默认优先最新文件，不依赖 tkinter 文件对话框"""
    # 支持新旧两种格式的log文件
    # 新格式: planning.log.YYYYMMDDHHMMSS
    # 旧格式: *.log
    log_files = glob.glob("*.log") + glob.glob("planning.log.*")

    # 去重并排序
    log_files = list(set(log_files))
    log_files = [f for f in log_files if os.path.isfile(f)]

    if not log_files:
        print("当前目录下没有找到 .log 文件")
        return None

    if len(log_files) == 1:
        print(f"自动选择唯一的日志文件: {log_files[0]}")
        return log_files[0]

    log_files.sort(key=os.path.getmtime, reverse=True)


    print("找到以下日志文件:")
    for i, file in enumerate(log_files[:10], 1):  # 只显示最新的10个
        mtime = datetime.fromtimestamp(os.path.getmtime(file))
        size = os.path.getsize(file) / 1024 / 1024  # MB
        print(f"{i}. {file} ({mtime.strftime('%Y-%m-%d %H:%M:%S')}, {size:.2f} MB)")

    while True:
        choice = input(f"\n请选择文件编号 (1-{min(10, len(log_files))}) 或按 Enter 选择最新文件: ")
        if choice == '':
            return log_files[0]
        elif choice.isdigit() and 1 <= int(choice) <= min(10, len(log_files)):

            return log_files[int(choice) - 1]
        else:
            print("无效的选择，请重试")

def change_speed(val):
    global animation_interval, ani
    animation_interval = int(200 / val)
    if ani is not None:
        ani.event_source.interval = animation_interval

def main():
    global ani, button, button_arrows, progress_slider, current_frame_index, valid_frames, parking_spaces, slot_corners, vehicle_locations, plan_frame_ids, \
           parking_function_statuses, vehicle_stop_reasons, parking_space_chamfers, plan_stage_target_poses, \
           plan_final_target_poses, path_current_segment_ids, replan_ids, coordinates, ax, fig, is_paused, press_x, press_y, timestamps, p0_p5_points, gear_info, \
           parking_function_stages, parking_function_modes, control_work_modes, vehicle_moving_statuses, perception_fusion_timestamps, file_boundaries, \
           fixed_xlim, fixed_ylim, user_adjusted_view, stopper_distances, target_corners_abcd, realtime_parkingspaces, fork_star_starts, parking_tasks

    # 解析命令行参数
    merge_enabled = '--no-merge' not in sys.argv
    
    # 获取文件路径（排除选项参数）
    file_path = None
    for arg in sys.argv[1:]:
        if not arg.startswith('--'):
            file_path = arg
            break
    
    if file_path:
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在")
            return
    else:
        file_path = select_log_file()
        if not file_path or not os.path.exists(file_path):
            print("未选择有效的文件")
            return
    
    log_files = find_and_sort_log_files(file_path)
    
    # 自动合并所有log文件（除非指定 --no-merge）
    if len(log_files) > 1 and merge_enabled:
        print(f"\n找到 {len(log_files)} 个同系列的日志文件，自动合并播放...")
        merged_data, file_boundaries = merge_log_data(log_files)
        parking_spaces = merged_data['parking_spaces']
        slot_corners = merged_data['slot_corners']
        vehicle_locations = merged_data['vehicle_locations']
        plan_frame_ids = merged_data['plan_frame_ids']
        parking_function_statuses = merged_data['parking_function_statuses']
        vehicle_stop_reasons = merged_data['vehicle_stop_reasons']
        parking_space_chamfers = merged_data['parking_space_chamfers']
        plan_stage_target_poses = merged_data['plan_stage_target_poses']
        plan_final_target_poses = merged_data['plan_final_target_poses']
        path_current_segment_ids = merged_data['path_current_segment_ids']
        replan_ids = merged_data['replan_ids']
        coordinates = merged_data['coordinates']
        timestamps = merged_data['timestamps']
        p0_p5_points = merged_data['p0_p5_points']
        gear_info = merged_data['gear_info']
        parking_function_stages = merged_data['parking_function_stages']
        parking_function_modes = merged_data['parking_function_modes']
        control_work_modes = merged_data['control_work_modes']
        vehicle_moving_statuses = merged_data['vehicle_moving_statuses']
        perception_fusion_timestamps = merged_data['perception_fusion_timestamps']
        stopper_distances = merged_data['stopper_distances']
        target_corners_abcd = merged_data['target_corners_abcd']
        realtime_parkingspaces = merged_data['realtime_parkingspaces']
        fork_star_starts = merged_data['fork_star_starts']
        parking_tasks = merged_data['parking_tasks']
        file_display_name = f"{os.path.basename(log_files[0])} ~ {os.path.basename(log_files[-1])} (共{len(log_files)}个文件)"
    else:
        # 单文件播放或禁用合并
        file_boundaries = []  # 单个文件时不需要边界信息
        parking_spaces = extract_parking_spaces(file_path)
        slot_corners = extract_slot_corners(file_path)
        vehicle_locations = extract_vehicle_locations(file_path)
        plan_frame_ids = extract_plan_frame_id(file_path)
        parking_function_statuses = extract_parking_function_status(file_path)
        vehicle_stop_reasons = extract_vehicle_stop_reason(file_path)
        parking_space_chamfers = extract_parking_space_chamfer(file_path)
        plan_stage_target_poses = extract_plan_stage_target_pose(file_path)
        plan_final_target_poses = extract_plan_final_target_pose(file_path)
        path_current_segment_ids = extract_path_current_segment_id(file_path)
        replan_ids = extract_replan_id(file_path)
        coordinates = extract_xy_coordinates(file_path)
        timestamps = extract_timestamps(file_path)
        p0_p5_points = extract_parking_space_p0_p5(file_path)
        gear_info = extract_gear_info(file_path)
        parking_function_stages = extract_parking_function_stage(file_path)
        parking_function_modes = extract_parking_function_mode(file_path)
        control_work_modes = extract_control_work_mode(file_path)
        vehicle_moving_statuses = extract_vehicle_moving_status(file_path)
        perception_fusion_timestamps = extract_perception_fusion_timestamps(file_path)
        stopper_distances = extract_stopper_dis_record(file_path)
        target_corners_abcd = extract_target_slot_corners_abcd(file_path)
        realtime_parkingspaces = extract_realtime_parkingspace(file_path)
        fork_star_starts = extract_fork_star_starts(file_path)
        parking_tasks = extract_parking_tasks(file_path)
        file_display_name = os.path.basename(file_path)
    
    fig, ax = plt.subplots()

    press_x, press_y = None, None

    # 使用最完整的列表作为基准，避免因某个列表不完整导致播放帧数减少
    list_lengths = {
        'parking_function_statuses': len(parking_function_statuses),
        'path_current_segment_ids': len(path_current_segment_ids),
        'vehicle_locations': len(vehicle_locations),
        'coordinates': len(coordinates)
    }

    # 使用最大的长度作为基准（通常 parking_function_statuses 是最完整的）
    max_length = max(list_lengths.values())
    valid_frames = list(range(max_length))

    print(f"各数据列表长度: {list_lengths}")
    print(f"使用最大长度作为有效帧数: {max_length}")
    print(f"注: 不完整的列表将在播放时跳过（如 coordinates 只有 {list_lengths['coordinates']} 条）")

    if valid_frames:
        print(f"总帧数: {len(valid_frames)} (播放所有帧，不做状态过滤)")
    else:
        print("没有找到有效的帧。")
        return
    
    # 计算所有数据的坐标范围，固定坐标轴
    global fixed_xlim, fixed_ylim
    all_x, all_y = [], []
    
    # 收集所有坐标点
    for i in range(max_length):
        if i < len(parking_spaces) and parking_spaces[i]:
            all_x.extend([p[0] for p in parking_spaces[i]])
            all_y.extend([p[1] for p in parking_spaces[i]])
        if i < len(slot_corners) and slot_corners[i]:
            all_x.extend([p[0] for p in slot_corners[i]])
            all_y.extend([p[1] for p in slot_corners[i]])
        if i < len(vehicle_locations) and vehicle_locations[i]:
            all_x.append(vehicle_locations[i][0])
            all_y.append(vehicle_locations[i][1])
        if i < len(coordinates) and coordinates[i]:
            all_x.extend([p[0] for p in coordinates[i]])
            all_y.extend([p[1] for p in coordinates[i]])
        if i < len(p0_p5_points) and p0_p5_points[i]:
            p0, p5 = p0_p5_points[i]
            all_x.extend([p0[0], p5[0]])
            all_y.extend([p0[1], p5[1]])

    if all_x and all_y:
        # 过滤异常值：去除绝对值过大的点（可能是无效数据）
        # 只使用硬阈值过滤，不使用百分位过滤
        # 原因：不同数据类型（slot corners、vehicle、coordinates）可能分布在不同区域
        # 百分位过滤会把少数区域的数据过滤掉
        MAX_REASONABLE_COORD = 50000  # 50米 = 50000毫米

        # 使用硬阈值过滤
        filtered_x = [x for x in all_x if abs(x) < MAX_REASONABLE_COORD]
        filtered_y = [y for y in all_y if abs(y) < MAX_REASONABLE_COORD]

        # 如果过滤后数据太少，放宽阈值
        if len(filtered_x) < len(all_x) * 0.5:
            MAX_REASONABLE_COORD = 500000  # 500米
            filtered_x = [x for x in all_x if abs(x) < MAX_REASONABLE_COORD]
            filtered_y = [y for y in all_y if abs(y) < MAX_REASONABLE_COORD]

        if filtered_x and filtered_y:
            x_min, x_max = min(filtered_x), max(filtered_x)
            y_min, y_max = min(filtered_y), max(filtered_y)
        else:
            # 如果过滤后没有数据，使用原始数据
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)

        # 添加边距（15%，增加一点边距）
        x_margin = (x_max - x_min) * 0.15
        y_margin = (y_max - y_min) * 0.15
        fixed_xlim = (x_min - x_margin, x_max + x_margin)
        fixed_ylim = (y_min - y_margin, y_max + y_margin)

        original_count = len(all_x)
        filtered_count = len(filtered_x)
        extreme_count = original_count - filtered_count
        print(f"坐标点统计: 原始{original_count}个点")
        print(f"  - 硬阈值过滤: 去除{extreme_count}个极端异常点 (|坐标|>={MAX_REASONABLE_COORD}mm)")
        print(f"  - 最终保留: {filtered_count}个点")
        print(f"固定坐标轴范围: X[{fixed_xlim[0]:.0f}, {fixed_xlim[1]:.0f}] Y[{fixed_ylim[0]:.0f}, {fixed_ylim[1]:.0f}]")
    else:
        fixed_xlim = (-10000, 10000)
        fixed_ylim = (-10000, 10000)

    current_frame_index = 0  # 初始化当前帧索引

    ani = FuncAnimation(fig, update, frames=valid_frames,
                        fargs=(parking_spaces, slot_corners, vehicle_locations, plan_frame_ids, parking_function_statuses,
                               vehicle_stop_reasons, parking_space_chamfers, plan_stage_target_poses, plan_final_target_poses,
                               path_current_segment_ids, replan_ids, coordinates, ax, timestamps, p0_p5_points, gear_info,
                               parking_function_stages, parking_function_modes, control_work_modes, vehicle_moving_statuses, perception_fusion_timestamps,
                               stopper_distances, target_corners_abcd, realtime_parkingspaces, fork_star_starts, parking_tasks),
                        interval=animation_interval, repeat=True)  

    fig.canvas.mpl_connect('scroll_event', on_mouse_scroll)
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    plt.subplots_adjust(top=0.85, bottom=0.18, left=0.08, right=0.95)
    
    fig.set_size_inches(14, 10)
    fig.patch.set_facecolor('#F5F5F5')

    ax_button = plt.axes([0.78, 0.92, 0.08, 0.045])
    button = Button(ax_button, 'Pause', 
                   color='#4CAF50', hovercolor='#45a049')
    button.label.set_fontsize(10)
    button.label.set_fontweight('bold')
    button.label.set_color('white')
    button.on_clicked(toggle_pause)

    ax_button_arrows = plt.axes([0.68, 0.92, 0.09, 0.045])
    button_arrows = Button(ax_button_arrows, 'Show Yaw', 
                          color='#2196F3', hovercolor='#0b7dda')
    button_arrows.label.set_fontsize(10)
    button_arrows.label.set_fontweight('bold')
    button_arrows.label.set_color('white')
    button_arrows.on_clicked(toggle_arrows)
    
    # 添加重置视角按钮
    ax_button_reset = plt.axes([0.57, 0.92, 0.10, 0.045])
    button_reset = Button(ax_button_reset, 'Reset View', 
                         color='#FF9800', hovercolor='#F57C00')
    button_reset.label.set_fontsize(10)
    button_reset.label.set_fontweight('bold')
    button_reset.label.set_color('white')
    
    def reset_view(event=None):
        global user_adjusted_view
        user_adjusted_view = False
        print("视角已重置，将自动适应数据范围")
    
    button_reset.on_clicked(reset_view)
    
    ax_speed = plt.axes([0.15, 0.05, 0.7, 0.025], facecolor='#E8F4F8')
    speed_slider = Slider(ax_speed, 'Speed', 1.0, 10.0, valinit=1.0, valstep=0.5,
                         color='#FF9800', track_color='#FFE0B2')
    speed_slider.label.set_fontweight('bold')
    speed_slider.label.set_fontsize(10)
    speed_slider.on_changed(change_speed)
    
    ax_progress = plt.axes([0.15, 0.09, 0.7, 0.025], facecolor='#E8F4F8')
    progress_slider = Slider(ax_progress, 'Progress', 0, len(valid_frames)-1, 
                            valinit=0, valstep=1, valfmt='%d',
                            color='#9C27B0', track_color='#E1BEE7')
    progress_slider.label.set_fontweight('bold')
    progress_slider.label.set_fontsize(10)
    
    stage_transition_frames = []  # SEARCH → PARK 切换
    replan_frames = []  # 重规划点
    
    for i in range(1, len(parking_function_stages)):
        if parking_function_stages[i-1] == 1 and parking_function_stages[i] == 2:
            stage_transition_frames.append(i)
            print(f"检测到状态切换: 帧 {i} (SEARCH → PARK)")
    
    if len(replan_ids) > 1:
        for i in range(1, len(replan_ids)):
            if replan_ids[i] > replan_ids[i-1]:
                replan_frames.append(i)
    
    if stage_transition_frames:
        for frame_idx in stage_transition_frames:
            ax_progress.axvline(x=frame_idx, color='red', linewidth=2.5, alpha=0.8, zorder=10)
        print(f"共检测到 {len(stage_transition_frames)} 个 SEARCH→PARK 切换点")
    
    if replan_frames:
        for frame_idx in replan_frames[:10]:  # 只显示前10个，避免过于密集
            ax_progress.axvline(x=frame_idx, color='orange', linewidth=1.5, alpha=0.6, 
                              linestyle='--', zorder=9)
        if len(replan_frames) > 10:
            print(f"检测到 {len(replan_frames)} 个重规划点（进度条仅显示前10个）")
        else:
            print(f"检测到 {len(replan_frames)} 个重规划点")
    
    def on_progress_change(val):
        global current_frame_index, is_paused, updating_progress
        if updating_progress:
            return
        
        if not is_paused:
            toggle_pause()
        
        current_frame_index = int(val)
        frame = valid_frames[current_frame_index]
        update(frame, parking_spaces, slot_corners, vehicle_locations, plan_frame_ids,
               parking_function_statuses, vehicle_stop_reasons, parking_space_chamfers, plan_stage_target_poses,
               plan_final_target_poses, path_current_segment_ids, replan_ids, coordinates, ax, timestamps, p0_p5_points, gear_info,
               parking_function_stages, parking_function_modes, control_work_modes, vehicle_moving_statuses, perception_fusion_timestamps,
               stopper_distances, target_corners_abcd, realtime_parkingspaces, fork_star_starts, parking_tasks)
        fig.canvas.draw_idle()

    progress_slider.on_changed(on_progress_change)
    
    ax_frame_info = plt.axes([0.15, 0.92, 0.3, 0.05])
    ax_frame_info.axis('off')
    frame_text = ax_frame_info.text(0.5, 0.5, f'Total Frames: {len(valid_frames)}', 
                                    ha='center', va='center', fontsize=10)

    def on_key_press(event):
        global is_paused
        if event.key == ' ':
            toggle_pause()
        elif event.key == 'right':
            # 无论是否暂停，都执行逐帧前进（函数会在必要时自动暂停）
            next_frame()
        elif event.key == 'left':
            # 无论是否暂停，都执行逐帧后退（函数会在必要时自动暂停）
            prev_frame()
        elif event.key == 'h':
            print("\n" + "="*50)
            print("快捷键说明:")
            print("  空格键      - 暂停/播放")
            print("  左箭头      - 上一帧（暂停时）")
            print("  右箭头      - 下一帧（暂停时）")
            print("  鼠标滚轮    - 缩放视图（缩放后视角自动锁定）")
            print("  鼠标中键拖动 - 平移视图（拖动后视角自动锁定）")
            print("  Reset View按钮 - 重置视角到自动范围")
            print("  Progress滑块 - 拖动跳转到指定帧")
            print("  H键         - 显示此帮助")
            print("="*50 + "\n")

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    print("\n" + "="*70)
    print(f"正在可视化日志文件: {file_display_name}")
    print(f"总帧数: {len(valid_frames)}")
    print("-"*70)
    print("控制说明:")
    print("  空格键       - 暂停/播放动画")
    print("  左/右箭头    - 逐帧浏览（需先暂停）")
    print("  鼠标滚轮     - 缩放视图（缩放后视角锁定）")
    print("  鼠标中键拖动  - 平移视图（拖动后视角锁定）")
    print("  Reset View按钮 - 重置视角到自动范围")
    print("  Progress滑块 - 拖动跳转到指定帧（显示当前播放进度）")
    print("  Speed滑块    - 调整播放速度 (1x-10x)")
    print("  Show Yaw按钮 - 显示/隐藏车辆朝向箭头")
    print("  H键          - 显示帮助信息")
    print("="*70 + "\n")

    plt.show()

def compatibility_main(argv=None):
    from planner_toolbox import main as planner_toolbox_main

    forwarded_args = ["log"]
    if argv is None:
        forwarded_args.extend(sys.argv[1:])
    else:
        forwarded_args.extend(argv)
    return planner_toolbox_main(forwarded_args)

if __name__ == "__main__":
    try:
        raise SystemExit(compatibility_main())
    except KeyboardInterrupt:
        print("\n程序已中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
