import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

RUN_DIR = os.getcwd()
LOGS = os.path.join(RUN_DIR, "logs_ns3")
VIZ_JSON = os.path.join(LOGS, "viz_timeseries.json")
OUT_SUM = os.path.join(LOGS, "udp_bursts_outgoing.csv")
IN_SUM = os.path.join(LOGS, "udp_bursts_incoming.csv")

WINDOW = 10

def load_frames(path):
    with open(path, "r") as f:
        v = json.load(f)
    raw = v.get("frames", [])
    frames = []
    for fr in raw:
        if isinstance(fr, dict) and "nodes" in fr:
            t = fr.get("t", None)
            try:
                t = float(t)
            except Exception:
                continue
            nodes = fr.get("nodes", [])
            P = np.array([[n.get("x", 0.0), n.get("y", 0.0), n.get("z", 0.0)] for n in nodes], dtype=float)
            frames.append((t, P))
    frames.sort(key=lambda x: x[0])
    return frames

def read_udp_bursts(path):
    cols = [
        "burst_id","from_node","to_node","target_rate","start_ns","duration_ns",
        "rate_with_headers","rate_payload","pkts","data_with_headers","data_payload","metadata"
    ]
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 11:
        raise RuntimeError(f"Unexpected format in {path}")
    while df.shape[1] < len(cols):
        df[df.shape[1]] = ""
    df.columns = cols
    df["start_s"] = df["start_ns"].astype(float) / 1e9
    df["dur_s"] = df["duration_ns"].astype(float) / 1e9
    df["end_s"] = df["start_s"] + df["dur_s"]
    df = df.sort_values("start_s").reset_index(drop=True)
    return df

frames = load_frames(VIZ_JSON)
if len(frames) == 0:
    raise RuntimeError("No frames found in viz_timeseries.json")

out_df = read_udp_bursts(OUT_SUM)
in_df = read_udp_bursts(IN_SUM)

t_end = 0.0
t_end = max(t_end, float(frames[-1][0]))
if len(out_df) > 0:
    t_end = max(t_end, float(out_df["end_s"].max()))
if len(in_df) > 0:
    t_end = max(t_end, float(in_df["end_s"].max()))

T_END = int(np.floor(t_end))

def build_piecewise(df):
    edges = []
    vals = []
    for _, r in df.iterrows():
        a = float(r["start_s"])
        b = float(r["end_s"])
        v = float(r["rate_payload"])
        if b <= a:
            continue
        edges.append((a, b))
        vals.append(v)
    return edges, vals

out_edges, out_vals = build_piecewise(out_df)
in_edges, in_vals = build_piecewise(in_df)

def rate_at(t, edges, vals):
    for (a, b), v in zip(edges, vals):
        if a <= t < b:
            return v
    return 0.0

times = np.arange(0, T_END + 1, dtype=float)
series_in = np.array([rate_at(t, in_edges, in_vals) for t in times], dtype=float)

all_max = 1.0
for _, P in frames:
    if P.size:
        all_max = max(all_max, float(np.max(np.abs(P))))

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2, width_ratios=(2.1, 1.0), height_ratios=(1.0, 1.0))
ax3d = fig.add_subplot(gs[:, 0], projection="3d")
ax_top = fig.add_subplot(gs[0, 1])
ax_bot = fig.add_subplot(gs[1, 1])

frame_times = np.array([t for t, _ in frames], dtype=float)

def frame_at_time(t):
    k = int(np.searchsorted(frame_times, t, side="right") - 1)
    if k < 0:
        k = 0
    return frames[k][1]

def draw(ti):
    t = float(ti)

    P = frame_at_time(t)
    ax3d.cla()
    if P.size:
        ax3d.scatter(P[:, 0], P[:, 1], P[:, 2], s=1)
    ax3d.set_title(f"Constellation (t={int(t)}s)")
    ax3d.set_xlim([-all_max, all_max])
    ax3d.set_ylim([-all_max, all_max])
    ax3d.set_zlim([-all_max, all_max])
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    ax_top.cla()
    m = times <= t
    ax_top.plot(times[m], series_in[m])
    ax_top.axvline(t, linestyle="--")
    ax_top.set_title("Throughput over time")
    ax_top.set_xlabel("time (s)")
    ax_top.set_ylabel("Mb/s")

    ax_bot.cla()
    lo = max(0, int(t) - WINDOW + 1)
    xs = np.arange(lo, int(t) + 1, dtype=int)
    ys = series_in[lo:int(t) + 1]
    ax_bot.bar(xs, ys, width=0.9)
    ax_bot.set_xlim([lo - 0.5, int(t) + 0.5])
    ax_bot.set_title(f"Throughput (last {WINDOW}s)")
    ax_bot.set_xlabel("time (s)")
    ax_bot.set_ylabel("Mb/s")

def update(i):
    draw(i)
    return []

anim = FuncAnimation(fig, update, frames=range(0, T_END + 1), interval=250, blit=False)
plt.tight_layout()
plt.show()
