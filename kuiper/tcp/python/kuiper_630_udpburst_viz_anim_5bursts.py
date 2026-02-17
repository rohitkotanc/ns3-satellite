import os, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

RUN_DIR = os.getcwd()
LOGS = os.path.join(RUN_DIR, "logs_ns3")
VIZ_JSON = os.path.join(LOGS, "viz_timeseries.json")
UDP_BURSTS_OUT = os.path.join(LOGS, "udp_bursts_outgoing.csv")

WINDOW = 10
USE_ONLY_FIRST_N_BURSTS = 5
USE_PAYLOAD_RATE = True

with open(VIZ_JSON, "r") as f:
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
        return np.zeros((0, 3)), {}
    P = np.array([[float(n.get("x", 0.0)), float(n.get("y", 0.0)), float(n.get("z", 0.0))] for n in nodes], dtype=float)
    ids = []
    for i, n in enumerate(nodes):
        nid = n.get("id", i)
        try:
            nid = int(nid)
        except Exception:
            nid = i
        ids.append(nid)
    idx = {ids[i]: i for i in range(len(ids))}
    return P, idx

frame_by_t = {}
all_max = 1.0
for fr in frames:
    ts = fr_time_s(fr)
    if ts is None:
        continue
    t_int = int(round(ts))
    frame_by_t[t_int] = fr
    P, _ = frame_positions(fr)
    if P.size:
        all_max = max(all_max, float(np.max(np.abs(P))))

def read_bursts(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            bid = int(parts[0])
            src = int(parts[1])
            dst = int(parts[2])
            target = float(parts[3])
            start_ns = int(parts[4])
            dur_ns = int(parts[5])
            rate_with_headers = float(parts[6])
            rate_payload = float(parts[7])
            rate = rate_payload if USE_PAYLOAD_RATE else rate_with_headers
            rows.append((bid, src, dst, target, start_ns, dur_ns, rate))
    rows.sort(key=lambda x: x[4])
    if USE_ONLY_FIRST_N_BURSTS is not None:
        rows = rows[:int(USE_ONLY_FIRST_N_BURSTS)]
    return rows

bursts = read_bursts(UDP_BURSTS_OUT)

t_end_viz = max(frame_by_t.keys()) if frame_by_t else 0
t_end_bursts = 0
if bursts:
    t_end_bursts = int(np.ceil(max((b[4] + b[5]) for b in bursts) / 1e9))
T_END = max(t_end_viz, t_end_bursts)

def throughput_at_second(t):
    t0 = t * 1e9
    t1 = (t + 1) * 1e9
    val = 0.0
    for (bid, src, dst, target, start_ns, dur_ns, rate) in bursts:
        a = start_ns
        b = start_ns + dur_ns
        if b <= t0 or a >= t1:
            continue
        val += rate
    return val

t_vals = np.arange(0, T_END + 1, dtype=int)
series = np.array([throughput_at_second(int(t)) for t in t_vals], dtype=float)

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
ax3d = fig.add_subplot(gs[:, 0], projection="3d")
ax_top = fig.add_subplot(gs[0, 1])
ax_bot = fig.add_subplot(gs[1, 1])

def draw_at_time(t):
    fr = frame_by_t.get(t, None)
    if fr is None:
        past = [k for k in frame_by_t.keys() if k <= t]
        fr = frame_by_t[max(past)] if past else (frames[0] if frames else {"nodes": []})

    P, _ = frame_positions(fr)

    ax3d.cla()
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
    m = t_vals <= t
    ax_top.plot(t_vals[m], series[m])
    ax_top.axvline(t, linestyle="--")
    ax_top.set_title("Throughput over time")
    ax_top.set_xlabel("time (s)")
    ax_top.set_ylabel("Mb/s")

    ax_bot.cla()
    lo = max(0, t - WINDOW + 1)
    xs = np.arange(lo, t + 1, dtype=int)
    ys = series[lo:t + 1]
    ax_bot.bar(xs, ys, width=0.9)
    ax_bot.set_xlim(lo - 0.5, t + 0.5)
    ax_bot.set_title(f"Throughput (last {WINDOW}s)")
    ax_bot.set_xlabel("time (s)")
    ax_bot.set_ylabel("Mb/s")

def update(i):
    t = int(i)
    draw_at_time(t)
    return []

draw_at_time(0)
anim = FuncAnimation(fig, update, frames=range(0, T_END + 1), interval=200, blit=False)
plt.tight_layout()
plt.show()
