import os
import re
import math
import argparse
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sgp4.api import Satrec, jday

BASE_STATE_DIR = "/Users/rohitkota/hypatia/paper/satellite_networks_state/gen_data/telesat_1015_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"

TLES_FILE = os.path.join(BASE_STATE_DIR, "tles.txt")
GROUND_FILE = os.path.join(BASE_STATE_DIR, "ground_stations.txt")
DESCRIPTION_FILE = os.path.join(BASE_STATE_DIR, "description.txt")

BANDWIDTH_GBPS = 8.4


def load_rtt_series(path):
    times = []
    rtts_ms = []

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
            if len(nums) < 2:
                continue

            t = float(nums[0])
            rtt = float(nums[1])

            if t > 1e6:
                t /= 1e9
            if rtt > 1e6:
                rtt /= 1e6

            times.append(t)
            rtts_ms.append(rtt)

    if not times:
        raise ValueError(f"No RTT data parsed from: {path}")

    return np.array(times, dtype=float), np.array(rtts_ms, dtype=float)


def parse_tles(path):
    with open(path, "r") as f:
        lines = [x.strip() for x in f if x.strip()]

    tles = []
    i = 0
    sat_id = 0

    while i < len(lines) - 1:
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            l1 = lines[i]
            l2 = lines[i + 1]
            i += 2
        elif i < len(lines) - 2 and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            l1 = lines[i + 1]
            l2 = lines[i + 2]
            i += 3
        else:
            i += 1
            continue

        sat = Satrec.twoline2rv(l1, l2)
        tles.append((sat_id, sat))
        sat_id += 1

    if not tles:
        raise ValueError("No valid TLEs parsed from tles.txt")

    return tles


def parse_ground_stations(path):
    grounds = []

    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue

            parts = [p.strip() for p in s.split(",")]
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except Exception:
                    pass

            if len(nums) < 3:
                continue

            lat_deg = nums[-3]
            lon_deg = nums[-2]
            elev_m = nums[-1]

            try:
                gid = int(float(parts[0]))
            except Exception:
                gid = len(grounds)

            name = parts[1] if len(parts) > 1 else f"gs_{gid}"
            grounds.append((gid, name, lat_deg, lon_deg, elev_m))

    if not grounds:
        raise ValueError("No ground stations parsed from ground_stations.txt")

    return grounds


def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    a = 6378137.0
    e2 = 6.69437999014e-3
    N = a / math.sqrt(1 - e2 * (math.sin(lat) ** 2))

    x = (N + h_m) * math.cos(lat) * math.cos(lon)
    y = (N + h_m) * math.cos(lat) * math.sin(lon)
    z = ((1 - e2) * N + h_m) * math.sin(lat)

    return np.array([x, y, z], dtype=float)


def sat_position_m(satrec_obj, when_utc):
    jd, fr = jday(
        when_utc.year,
        when_utc.month,
        when_utc.day,
        when_utc.hour,
        when_utc.minute,
        when_utc.second + when_utc.microsecond / 1e6,
    )

    err, r_km, _ = satrec_obj.sgp4(jd, fr)
    if err != 0:
        return None

    return np.array(r_km, dtype=float) * 1000.0


def extract_start_time(description_path):
    with open(description_path, "r") as f:
        txt = f.read()

    m = re.search(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})", txt)
    if m:
        return dt.datetime.fromisoformat(f"{m.group(1)} {m.group(2)}")

    return dt.datetime(2020, 1, 1, 0, 0, 0)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtt_file", required=True)
    ap.add_argument("--src", type=int, required=True)
    ap.add_argument("--dst", type=int, required=True)
    ap.add_argument("--duration", type=float, default=200.0)
    ap.add_argument("--fps", type=float, default=1.0)
    args = ap.parse_args()

    tles = parse_tles(TLES_FILE)
    grounds_raw = parse_ground_stations(GROUND_FILE)
    start_time = extract_start_time(DESCRIPTION_FILE)

    ground_positions = []
    for gid, name, lat_deg, lon_deg, elev_m in grounds_raw:
        ground_positions.append((gid, name, geodetic_to_ecef(lat_deg, lon_deg, elev_m)))

    t_rtt, rtt_ms = load_rtt_series(args.rtt_file)
    throughput_gbps = np.clip(BANDWIDTH_GBPS / (1.0 + 0.015 * rtt_ms), 0.0, BANDWIDTH_GBPS)

    duration = float(args.duration)
    fps = float(args.fps)
    t_anim = np.arange(0.0, duration + 1e-9, 1.0 / fps)

    sat_positions_series = []
    for t in t_anim:
        when_utc = start_time + dt.timedelta(seconds=float(t))
        coords = []
        for sat_id, sat in tles:
            pos = sat_position_m(sat, when_utc)
            if pos is not None:
                coords.append((sat_id, pos[0], pos[1], pos[2]))
        sat_positions_series.append(coords)

    sat0 = sat_positions_series[0]
    sat_xyz0 = np.array([[p[1], p[2], p[3]] for p in sat0], dtype=float)
    gs_xyz = np.array([g[2] for g in ground_positions], dtype=float)

    all_x = np.concatenate([sat_xyz0[:, 0], gs_xyz[:, 0]])
    all_y = np.concatenate([sat_xyz0[:, 1], gs_xyz[:, 1]])
    all_z = np.concatenate([sat_xyz0[:, 2], gs_xyz[:, 2]])

    fig = plt.figure(figsize=(13.5, 6.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    ax3d.set_title("Telesat topology (3D)")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.view_init(elev=22, azim=35)
    ax3d.set_xlim(np.min(all_x), np.max(all_x))
    ax3d.set_ylim(np.min(all_y), np.max(all_y))
    ax3d.set_zlim(np.min(all_z), np.max(all_z))

    sat_sc = ax3d.scatter(sat_xyz0[:, 0], sat_xyz0[:, 1], sat_xyz0[:, 2], s=4)
    gs_sc = ax3d.scatter(gs_xyz[:, 0], gs_xyz[:, 1], gs_xyz[:, 2], s=20)

    src_sc = None
    dst_sc = None
    sat_count = len(tles)

    if sat_count <= args.src < sat_count + len(ground_positions):
        idx = args.src - sat_count
        p = gs_xyz[idx]
        src_sc = ax3d.scatter([p[0]], [p[1]], [p[2]], s=50)

    if sat_count <= args.dst < sat_count + len(ground_positions):
        idx = args.dst - sat_count
        p = gs_xyz[idx]
        dst_sc = ax3d.scatter([p[0]], [p[1]], [p[2]], s=50)

    ax2d.set_title("Topology-derived UDP throughput over time")
    ax2d.set_xlabel("time (s)")
    ax2d.set_ylabel("Throughput (Gbps)")
    ax2d.set_xlim(0, duration)

    finite = throughput_gbps[np.isfinite(throughput_gbps)]
    if finite.size:
        ymin = float(np.min(finite))
        ymax = float(np.max(finite))
        if abs(ymax - ymin) < 1e-9:
            ax2d.set_ylim(ymin - 0.5, ymax + 0.5)
        else:
            pad = 0.1 * (ymax - ymin)
            ax2d.set_ylim(ymin - pad, ymax + pad)
    else:
        ax2d.set_ylim(0, BANDWIDTH_GBPS)

    line, = ax2d.plot([], [], linewidth=2)
    vline = ax2d.axvline(0.0)
    txt = ax2d.text(0.02, 0.98, "", transform=ax2d.transAxes, va="top")

    def update(i):
        sats_i = sat_positions_series[i]
        sat_xyz = np.array([[p[1], p[2], p[3]] for p in sats_i], dtype=float)
        sat_sc._offsets3d = (sat_xyz[:, 0], sat_xyz[:, 1], sat_xyz[:, 2])

        t = t_anim[i]
        mask = t_rtt <= t
        xs = t_rtt[mask]
        ys = throughput_gbps[mask]

        line.set_data(xs, ys)
        vline.set_xdata([t, t])

        if np.any(mask):
            val = ys[-1]
            txt.set_text(
                f"t = {t:.1f}s\n"
                f"UDP throughput = {val:.3f} Gbps\n"
                f"src = {args.src}\n"
                f"dst = {args.dst}"
            )
        else:
            txt.set_text(
                f"t = {t:.1f}s\n"
                f"UDP throughput = --\n"
                f"src = {args.src}\n"
                f"dst = {args.dst}"
            )

        artists = [sat_sc, gs_sc, line, vline, txt]
        if src_sc is not None:
            artists.append(src_sc)
        if dst_sc is not None:
            artists.append(dst_sc)
        return tuple(artists)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(t_anim),
        interval=int(1000 / fps),
        blit=False,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
