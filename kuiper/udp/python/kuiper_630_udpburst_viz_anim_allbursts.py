import os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

RUN_DIR = os.getcwd()
LOGS = os.path.join(RUN_DIR, "logs_ns3")
VIZ_JSON = os.path.join(LOGS, "viz_timeseries.json")

UDP_PAYLOAD_BYTES = 1200
WINDOW = 10

with open(VIZ_JSON) as f:
    V = json.load(f)

raw_frames = V.get("frames", [])
frames = [fr for fr in raw_frames if isinstance(fr, dict) and "nodes" in fr]

def fr_time_s(fr):
    t = fr.get("t", None)
    if t is None:
        return None
    try:
        return float(t)
    except Exception:
        return None

def frame_positions(fr):
    nodes = fr.get("nodes", [])
    if not nodes:
        return np.zeros((0, 3), dtype=float)
    P = np.array([[n.get("x", 0.0), n.get("y", 0.0), n.get("z", 0.0)] for n in nodes], dtype=float)
    return P

frame_by_t = {}
all_max = 1.0
for fr in frames:
    ts = fr_time_s(fr)
    if ts is None:
        continue
    t_int = int(np.floor(ts))
    frame_by_t[t_int] = fr
    P = frame_positions(fr)
    if P.size:
        all_max = max(all_max, float(np.max(np.abs(P))))

def read_pkt_times_any(pattern):
    paths = sorted(glob.glob(os.path.join(LOGS, pattern)))
    if not paths:
        return pd.Series([], dtype=np.int64)
    parts = []
    for p in paths:
        df = pd.read_csv(p, header=None)
        if df.shape[1] < 3:
            continue
        col = pd.to_numeric(df.iloc[:, 2], errors="coerce").dropna().astype(np.int64)
        parts.append(col)
    if not parts:
        return pd.Series([], dtype=np.int64)
    return pd.concat(parts, ignore_index=True)

sent_ns = read_pkt_times_any("udp_burst_*_outgoing.csv")
recv_ns = read_pkt_times_any("udp_burst_*_incoming.csv")

def per_second_mbps(ns_series):
    if ns_series.size == 0:
        return np.zeros(1, dtype=float)
    sec = (ns_series // 1_000_000_000).astype(np.int64)
    counts = sec.value_counts().sort_index()
    end = int(counts.index.max())
    arr = np.zeros(end + 1, dtype=np.int64)
    arr[counts.index.to_numpy()] = counts.to_numpy()
    return (arr.astype(float) * (UDP_PAYLOAD_BYTES * 8) / 1e6)

out_mbps = per_second_mbps(sent_ns)
in_mbps = per_second_mbps(recv_ns)

t_end_frames = int(max(frame_by_t.keys())) if frame_by_t else 0
t_end_data = max(len(out_mbps) - 1, len(in_mbps) - 1)
T_END = int(max(t_end_frames, t_end_data))

def mbps_at(arr, t):
    if t < 0:
        return 0.0
    if t >= len(arr):
        return 0.0
    return float(arr[t])

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2, width_ratios=(2, 1), height_ratios=(1, 1))
ax3d = fig.add_subplot(gs[:, 0], projection="3d")
ax_top = fig.add_subplot(gs[0, 1])
ax_bot = fig.add_subplot(gs[1, 1])

def draw_at_time(t):
    fr = frame_by_t.get(t, None)
    if fr is None and frame_by_t:
        past = [k for k in frame_by_t.keys() if k <= t]
        if past:
            fr = frame_by_t[max(past)]
        else:
            fr = frame_by_t[min(frame_by_t.keys())]

    ax3d.cla()
    if fr is not None:
        P = frame_positions(fr)
        if P.size:
            ax3d.scatter(P[:, 0], P[:, 1], P[:, 2], s=1)
    ax3d.set_title(f"Constellation (t={t}s)")
    ax3d.set_xlim([-all_max, all_max])
    ax3d.set_ylim([-all_max, all_max])
    ax3d.set_zlim([-all_max, all_max])
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    ax_top.cla()
    xs = np.arange(0, T_END + 1, dtype=int)
    ys = np.array([mbps_at(in_mbps, i) for i in xs], dtype=float)
    m = xs <= t
    ax_top.plot(xs[m], ys[m])
    ax_top.axvline(t, linestyle="--")
    ax_top.set_title("Throughput over time")
    ax_top.set_xlabel("time (s)")
    ax_top.set_ylabel("Mb/s")
    if m.any():
        y0 = float(np.min(ys[m]))
        y1 = float(np.max(ys[m]))
        pad = 0.05 * (y1 - y0) if y1 > y0 else 1.0
        ax_top.set_ylim(y0 - pad, y1 + pad)

    ax_bot.cla()
    lo = max(0, t - WINDOW + 1)
    xs2 = np.arange(lo, t + 1, dtype=int)
    ys2 = np.array([mbps_at(in_mbps, i) for i in xs2], dtype=float)
    ax_bot.bar(xs2, ys2, width=0.9)
    ax_bot.set_xlim(lo - 0.5, t + 0.5)
    ax_bot.set_title(f"Throughput (last {WINDOW}s)")
    ax_bot.set_xlabel("time (s)")
    ax_bot.set_ylabel("Mb/s")
    if ys2.size:
        y0 = float(np.min(ys2))
        y1 = float(np.max(ys2))
        pad = 0.05 * (y1 - y0) if y1 > y0 else 1.0
        ax_bot.set_ylim(y0 - pad, y1 + pad)

def update(i):
    t = int(i)
    draw_at_time(t)
    return []

draw_at_time(0)
anim = FuncAnimation(fig, update, frames=range(0, T_END + 1), interval=1000, blit=False, repeat=False)
plt.tight_layout()
plt.show()
