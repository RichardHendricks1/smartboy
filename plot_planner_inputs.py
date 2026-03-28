#!/usr/bin/env python3
"""
GridMap Multi-Frame Viewer
多帧网格地图查看器 - 将 planner_inputs.csv 中所有数据组分别显示在独立窗口中

每组数据 520 或 521 行：
  - 512 行 gridmap 数据
  - 第513行: 大坐标 + 时间戳
  - 第514行: 边界数据
  - 第515行: 自车位姿 (x, y, theta)
  - 第516行: 目标位姿 (x, y, theta)
  - 第517行: 车位ABCD坐标
  - 第518行: 2个值
  - 第519行: 5个值
  - 第520行: 2个值（标志位）
  - 第521行(可选): 轨迹数据

两组数据之间有一个空行。
"""

import numpy as np
import argparse
import os
import sys
import tempfile
from pathlib import Path

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
from matplotlib.patches import Polygon


def split_csv_by_blank_lines(csv_path: str) -> list:
    """按空行分割CSV文件，返回每组的行列表"""
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    groups = []
    current_group = []

    for line in lines:
        stripped = line.strip()
        if stripped == '':
            # 空行 -> 结束当前组
            if current_group:
                groups.append(current_group)
                current_group = []
        else:
            current_group.append(stripped)

    # 最后一组（文件末尾可能没有空行）
    if current_group:
        groups.append(current_group)

    return groups


def parse_group(group_lines: list, grid_size: int = 512) -> dict:
    """解析一组数据，提取 gridmap、元数据和轨迹

    Returns:
        dict with keys: gridmap, gridmap_origin, ego_pose, target_pose,
                        slot_points, trajectory, timestamp, has_trajectory, line_count
    """
    result = {
        'gridmap': None,
        'gridmap_origin': None,
        'ego_pose': None,
        'target_pose': None,
        'slot_points': None,
        'trajectory': None,
        'timestamp': '',
        'has_trajectory': False,
        'line_count': len(group_lines),
    }

    # 1) 解析 gridmap（前 512 行）
    gridmap_rows = []
    for i in range(min(grid_size, len(group_lines))):
        values = group_lines[i].split(',')
        row = []
        for v in values[:grid_size]:
            v = v.strip()
            if v:
                try:
                    row.append(int(v))
                except ValueError:
                    row.append(0)
            else:
                row.append(0)
        while len(row) < grid_size:
            row.append(0)
        gridmap_rows.append(row)

    while len(gridmap_rows) < grid_size:
        gridmap_rows.append([0] * grid_size)

    result['gridmap'] = np.array(gridmap_rows, dtype=np.uint8)

    n = len(group_lines)

    # 2) 第513行 (index 512): 大坐标 + 时间戳
    if n > 512:
        parts = group_lines[512].split(',')
        if len(parts) >= 4:
            result['timestamp'] = parts[3].strip()
            try:
                result['gridmap_origin_raw'] = {
                    'x_raw': float(parts[0]),
                    'y_raw': float(parts[1]),
                    'theta_raw': float(parts[2]),
                }
            except:
                pass

    # 3) 第515行 (index 514): 自车位姿 x,y,theta — 同时也是 gridmap_origin
    if n > 514:
        parts = group_lines[514].split(',')
        if len(parts) >= 3:
            try:
                ego = {
                    'x': float(parts[0]),
                    'y': float(parts[1]),
                    'theta': float(parts[2]),
                }
                result['ego_pose'] = ego
                result['gridmap_origin'] = ego.copy()
            except:
                pass

    # 4) 第516行 (index 515): 目标位姿
    if n > 515:
        parts = group_lines[515].split(',')
        if len(parts) >= 3:
            try:
                result['target_pose'] = {
                    'x': float(parts[0]),
                    'y': float(parts[1]),
                    'theta': float(parts[2]),
                }
            except:
                pass

    # 5) 第517行 (index 516): 车位 ABCD 坐标
    if n > 516:
        parts = group_lines[516].split(',')
        if len(parts) >= 8:
            try:
                result['slot_points'] = [
                    (float(parts[0]), float(parts[1])),  # A
                    (float(parts[2]), float(parts[3])),  # B
                    (float(parts[4]), float(parts[5])),  # C
                    (float(parts[6]), float(parts[7])),  # D
                ]
            except:
                pass

    # 6) 最后一行判断：如果行数 >= 521(即 index 520 存在)，认为有轨迹
    #    或者精确判断：行数 == 521
    #    更稳健的做法：如果最后一行包含很多逗号分隔的浮点数(>30个值)，认为是轨迹
    if n >= 521:
        last_line = group_lines[-1]
        vals = last_line.split(',')
        if len(vals) > 30:
            # 很大概率是轨迹数据
            result['has_trajectory'] = True
            trajectory = []
            for i in range(0, len(vals) - 2, 3):
                try:
                    x = float(vals[i])
                    y = float(vals[i + 1])
                    theta = float(vals[i + 2])
                    trajectory.append((x, y, theta))
                except:
                    break
            if trajectory:
                result['trajectory'] = trajectory

    return result


def slot_to_gridmap_pixel(slot_x, slot_y, gridmap_origin, grid_size=512, resolution=100.0):
    """将车位坐标系的点转换为 gridmap 像素坐标"""
    gx = gridmap_origin['x']
    gy = gridmap_origin['y']
    gtheta = gridmap_origin['theta']

    dx = slot_x - gx
    dy = slot_y - gy

    theta_rad = np.deg2rad(gtheta)
    cos_t = np.cos(-theta_rad)
    sin_t = np.sin(-theta_rad)

    rotated_x = dx * cos_t - dy * sin_t
    rotated_y = dx * sin_t + dy * cos_t

    pixel_x = rotated_x / resolution + grid_size / 2
    pixel_y = rotated_y / resolution + grid_size / 2

    return pixel_x, pixel_y


def draw_vehicle(ax, artists, pose, gridmap_origin, grid_size, resolution, is_ego=False):
    """绘制车辆轮廓"""
    vehicle_length = 5250   # mm
    vehicle_width = 2000    # mm
    rear_axle_to_rear = 1134  # mm

    rear_x = pose['x']
    rear_y = pose['y']
    theta = pose['theta']
    theta_rad = np.deg2rad(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    half_width = vehicle_width / 2
    front_length = vehicle_length - rear_axle_to_rear

    corners_local = [
        (-rear_axle_to_rear, -half_width),
        (front_length, -half_width),
        (front_length, half_width),
        (-rear_axle_to_rear, half_width),
    ]

    corners_px = []
    for lx, ly in corners_local:
        sx = rear_x + lx * cos_t - ly * sin_t
        sy = rear_y + lx * sin_t + ly * cos_t
        px, py = slot_to_gridmap_pixel(sx, sy, gridmap_origin, grid_size, resolution)
        corners_px.append((px, py))

    if is_ego:
        edge_color, face_color, label = 'lime', 'lime', 'START'
    else:
        edge_color, face_color, label = 'red', 'red', 'GOAL'

    poly = Polygon(corners_px, closed=True,
                   edgecolor=edge_color, facecolor=face_color,
                   alpha=0.4, linewidth=2, zorder=15)
    ax.add_patch(poly)
    artists.append(poly)

    rear_px, rear_py = slot_to_gridmap_pixel(rear_x, rear_y, gridmap_origin, grid_size, resolution)
    pt, = ax.plot(rear_px, rear_py, 'o', color=edge_color, markersize=8, zorder=17)
    artists.append(pt)

    txt = ax.text(rear_px + 3, rear_py + 3, label, fontsize=9, color='black',
                  fontweight='bold', zorder=18,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor=face_color, alpha=0.7))
    artists.append(txt)

    # 车头方向箭头
    front_x = rear_x + front_length * cos_t
    front_y = rear_y + front_length * sin_t
    front_px, front_py = slot_to_gridmap_pixel(front_x, front_y, gridmap_origin, grid_size, resolution)
    ddx = front_px - rear_px
    ddy = front_py - rear_py
    length = np.sqrt(ddx ** 2 + ddy ** 2)
    if length > 0:
        arrow_len = 20
        ddx = ddx / length * arrow_len
        ddy = ddy / length * arrow_len
        arr = ax.arrow(rear_px, rear_py, ddx, ddy,
                       head_width=6, head_length=8,
                       fc=edge_color, ec=edge_color, linewidth=1.5, zorder=17)
        artists.append(arr)


def draw_trajectory(ax, artists, trajectory, gridmap_origin, grid_size, resolution):
    """绘制轨迹"""
    if not trajectory or len(trajectory) < 2:
        return

    gtheta = gridmap_origin['theta']

    traj_px = []
    for x, y, theta in trajectory:
        px, py = slot_to_gridmap_pixel(x, y, gridmap_origin, grid_size, resolution)
        theta_gm = theta - gtheta
        traj_px.append((px, py, theta_gm))

    xs = [p[0] for p in traj_px]
    ys = [p[1] for p in traj_px]
    line, = ax.plot(xs, ys, 'b-', linewidth=1.5, alpha=0.7, zorder=8)
    artists.append(line)

    # 膨胀框
    veh_len_px = 5200 / resolution
    veh_wid_px = 2000 / resolution
    rear_to_rear_px = 1000 / resolution
    rear_to_front_px = veh_len_px - rear_to_rear_px
    half_w = veh_wid_px / 2
    box_interval = max(1, len(traj_px) // 40)

    for i in range(0, len(traj_px), box_interval):
        px, py, theta = traj_px[i]
        theta_rad = np.deg2rad(theta)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        corners_local = [
            (-rear_to_rear_px, -half_w),
            (rear_to_front_px, -half_w),
            (rear_to_front_px, half_w),
            (-rear_to_rear_px, half_w),
        ]
        corners = []
        for lx, ly in corners_local:
            rx = lx * cos_t - ly * sin_t
            ry = lx * sin_t + ly * cos_t
            corners.append((px + rx, py + ry))

        box = Polygon(corners, closed=True,
                      edgecolor='cyan', facecolor='cyan',
                      alpha=0.15, linewidth=0.8, zorder=5)
        ax.add_patch(box)
        artists.append(box)

    # 方向箭头
    arrow_interval = max(1, len(traj_px) // 15)
    for i in range(0, len(traj_px), arrow_interval):
        px, py, theta = traj_px[i]
        theta_rad = np.deg2rad(theta)
        adx = 1 * np.cos(theta_rad)
        ady = 1 * np.sin(theta_rad)
        arr = ax.arrow(px, py, adx, ady,
                       head_width=1.5, head_length=2,
                       fc='blue', ec='blue', linewidth=0.6, alpha=0.5, zorder=9)
        artists.append(arr)

    # 起点终点标注
    sp, = ax.plot(traj_px[0][0], traj_px[0][1], 'go', markersize=8, zorder=10)
    artists.append(sp)
    st = ax.text(traj_px[0][0] + 4, traj_px[0][1] + 4, 'Start', fontsize=9,
                 color='green', fontweight='bold', zorder=11,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    artists.append(st)

    ep, = ax.plot(traj_px[-1][0], traj_px[-1][1], 'ro', markersize=8, zorder=10)
    artists.append(ep)
    et = ax.text(traj_px[-1][0] + 4, traj_px[-1][1] + 4, 'End', fontsize=9,
                 color='red', fontweight='bold', zorder=11,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    artists.append(et)


def draw_parking_slot(ax, artists, slot_points, gridmap_origin, grid_size, resolution):
    """绘制车位"""
    if not slot_points or len(slot_points) != 4:
        return

    slot_px = []
    for sx, sy in slot_points:
        px, py = slot_to_gridmap_pixel(sx, sy, gridmap_origin, grid_size, resolution)
        slot_px.append((px, py))

    poly = Polygon(slot_px, closed=True,
                   edgecolor='red', facecolor='yellow',
                   alpha=0.3, linewidth=2, zorder=10)
    ax.add_patch(poly)
    artists.append(poly)

    # 边界线
    pts = slot_px + [slot_px[0]]
    xs_l = [p[0] for p in pts]
    ys_l = [p[1] for p in pts]
    ln, = ax.plot(xs_l, ys_l, 'r-', linewidth=2, zorder=11)
    artists.append(ln)

    # 标注 ABCD
    colors = ['red', 'green', 'blue', 'orange']
    for i, (px, py) in enumerate(slot_px):
        pt, = ax.plot(px, py, 'o', color=colors[i], markersize=6, zorder=12)
        artists.append(pt)
        txt = ax.text(px + 2, py + 2, chr(65 + i), fontsize=10, color=colors[i],
                      fontweight='bold', zorder=13,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        artists.append(txt)


def create_viewer_figure(group_data: dict, group_index: int,
                          grid_size: int = 512, resolution: float = 100.0):
    """为一组数据创建一个独立的查看窗口"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ts = group_data.get('timestamp', '')
    ts_short = ts[-10:] if len(ts) >= 10 else ts
    has_traj = group_data.get('has_trajectory', False)
    traj_info = " [有轨迹]" if has_traj else " [无轨迹]"
    title = f"Frame {group_index} | TS: {ts_short}{traj_info} | {group_data['line_count']} lines"
    fig.canvas.manager.set_window_title(title)

    gridmap = group_data['gridmap']
    ax.imshow(gridmap, cmap='gray', vmin=0, vmax=255,
              origin='lower', interpolation='nearest')

    # 网格线
    ax.set_xticks(np.arange(0, grid_size, 10))
    ax.set_yticks(np.arange(0, grid_size, 10))
    ax.grid(True, alpha=0.2, color='blue', linewidth=0.3)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)

    # 刻度标签
    x_labels = [f"{i * resolution / 1000:.0f}m" if i % 50 == 0 else ""
                for i in range(0, grid_size, 10)]
    ax.set_xticklabels(x_labels, fontsize=5)
    y_labels = [f"{i * resolution / 1000:.0f}m" if i % 50 == 0 else ""
                for i in range(0, grid_size, 10)]
    ax.set_yticklabels(y_labels, fontsize=5)

    ax.set_title(title, fontsize=10)

    artists = []
    gridmap_origin = group_data.get('gridmap_origin')

    if gridmap_origin:
        # 车位
        if group_data.get('slot_points'):
            draw_parking_slot(ax, artists, group_data['slot_points'],
                              gridmap_origin, grid_size, resolution)

        # 目标车辆
        if group_data.get('target_pose'):
            draw_vehicle(ax, artists, group_data['target_pose'],
                         gridmap_origin, grid_size, resolution, is_ego=False)

        # 自车
        if group_data.get('ego_pose'):
            draw_vehicle(ax, artists, group_data['ego_pose'],
                         gridmap_origin, grid_size, resolution, is_ego=True)

        # 轨迹
        if group_data.get('trajectory'):
            draw_trajectory(ax, artists, group_data['trajectory'],
                            gridmap_origin, grid_size, resolution)

    # 信息文本
    info_lines = []
    if group_data.get('ego_pose'):
        e = group_data['ego_pose']
        info_lines.append(f"Ego: ({e['x']:.1f}, {e['y']:.1f}, {e['theta']:.1f}°)")
    if group_data.get('target_pose'):
        t = group_data['target_pose']
        info_lines.append(f"Target: ({t['x']:.1f}, {t['y']:.1f}, {t['theta']:.1f}°)")
    info_lines.append(f"Occupied: {np.sum(gridmap == 128)}")
    if has_traj and group_data.get('trajectory'):
        info_lines.append(f"Traj pts: {len(group_data['trajectory'])}")

    info_str = '\n'.join(info_lines)
    ax.text(0.02, 0.98, info_str, transform=ax.transAxes,
            verticalalignment='top', fontsize=8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
            zorder=20)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='GridMap Multi-Frame Viewer - 将CSV中所有帧分别显示在独立窗口')
    parser.add_argument('--csv', type=str, default='planner_inputs.csv',
                        help='CSV文件路径 (默认: planner_inputs.csv)')
    parser.add_argument('--size', type=int, default=512,
                        help='网格大小 (默认: 512)')
    parser.add_argument('--resolution', type=float, default=100.0,
                        help='分辨率 mm/cell (默认: 100.0)')
    args = parser.parse_args()

    csv_path = args.csv
    if not Path(csv_path).exists():
        print(f"❌ 文件不存在: {csv_path}")
        sys.exit(1)

    print("=" * 60)
    print("GridMap Multi-Frame Viewer")
    print("=" * 60)
    print(f"CSV 文件: {csv_path}")

    # 1) 按空行分割
    groups = split_csv_by_blank_lines(csv_path)
    print(f"检测到 {len(groups)} 组数据（只显示前5组）")
    groups = groups[:5]  # 只取前5组
    for i, g in enumerate(groups):
        print(f"  组 {i}: {len(g)} 行  {'(有轨迹)' if len(g) >= 521 else '(无轨迹)'}")

    print("=" * 60)

    # 2) 解析每组数据并创建窗口
    figures = []
    for i, group_lines in enumerate(groups):
        print(f"\n--- 正在处理组 {i} ({len(group_lines)} 行) ---")
        group_data = parse_group(group_lines, grid_size=args.size)

        if group_data['gridmap'] is not None:
            fig = create_viewer_figure(group_data, i,
                                       grid_size=args.size,
                                       resolution=args.resolution)
            figures.append(fig)
            print(f"  ✅ 窗口已创建")
            if group_data.get('timestamp'):
                print(f"  时间戳: {group_data['timestamp']}")
            if group_data.get('trajectory'):
                print(f"  轨迹点数: {len(group_data['trajectory'])}")
        else:
            print(f"  ⚠️ 无法解析该组数据")

    print(f"\n{'=' * 60}")
    print(f"共创建 {len(figures)} 个窗口，显示中...")
    print(f"关闭所有窗口以退出程序")
    print(f"{'=' * 60}")

    plt.show()


def compatibility_main(argv=None):
    from planner_toolbox import main as planner_toolbox_main

    forwarded_args = ["csv"]
    if argv is None:
        forwarded_args.extend(sys.argv[1:])
    else:
        forwarded_args.extend(argv)
    return planner_toolbox_main(forwarded_args)


if __name__ == '__main__':
    raise SystemExit(compatibility_main())
