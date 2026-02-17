import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

RUN_DIR = os.getcwd()
LOGS = os.path.join(RUN_DIR, "logs_ns3")
VIZ_JSON = os.path.join(LOGS, "viz_timeseries.json")
OUT_PKTS = os.path.join(LOGS, "udp_burst_0_outgoing.csv")
IN_PKTS = os.path.join(LOGS, "udp_burst_0_incoming.csv")

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
        return np.zeros((0, 3)), {}
    P = np.array([[n.get("x", 0.0), n.get("y", 0.0), n.get("z", 0.0)] for n in nodes], dtype=float)
    ids = [int(n.get("id", i)) for i, n in enumerate(nodes)]
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

def read_pkt_times(path, colname):
    df = pd.read_csv(path, header=None, names=["burst_id", "seq", colname])
    df[colname] = pd.to_numeric(df[colname], errors="coerce")
    df = df.dropna(subset=[colname])
    df["_t"] = (df[colname] / 1e9).astype(float)
    df["_ts"] = np.floor(df["_t"]).astype(int)
    return df

out_df = read_pkt_times(OUT_PKTS, "sent_ns")
in_df  = read_pkt_times(IN_PKTS, "recv_ns")

t_end_frames = int(max(frame_by_t.keys())) if len(frame_by_t) else 0
t_end_data = 0
if len(out_df):
    t_end_data = max(t_end_data, int(out_df["_ts"].max()))
if len(in_df):
    t_end_data = max(t_end_data, int(in_df["_ts"].max()))
T_END = max(t_end_frames, t_end_data)

out_per = out_df.groupby("_ts").size().rename("pkts").to_frame()
in_per  = in_df.groupby("_ts").size().rename("pkts").to_frame()

idx = np.arange(0, T_END + 1, dtype=int)
out_per = out_per.reindex(idx, fill_value=0)
in_per  = in_per.reindex(idx, fill_value=0)

out_mbps = (out_per["pkts"].to_numpy() * UDP_PAYLOAD_BYTES * 8) / 1e6
in_mbps  = (in_per["pkts"].to_numpy()  * UDP_PAYLOAD_BYTES * 8) / 1e6
t_vals = idx

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
    ax_top.plot(t_vals[m], in_mbps[m])
    ax_top.axvline(t, linestyle="--")
    ax_top.set_title("Throughput over time")
    ax_top.set_xlabel("time (s)")
    ax_top.set_ylabel("Mb/s")
    ax_top.set_ylim(159.9, 160.02)
    ax_bot.cla()
    lo = max(0, t - WINDOW + 1)
    xs = np.arange(lo, t + 1, dtype=int)
    ys = in_mbps[lo:t + 1]
    ax_bot.bar(xs, ys, width=0.9)
    ax_bot.set_xlim(lo - 0.5, t + 0.5)
    ax_bot.set_ylim(159.9, 160.02)
    ax_bot.set_title(f"Throughput (last {WINDOW}s)")
    ax_bot.set_xlabel("time (s)")
    ax_bot.set_ylabel("Mb/s")

def update(i):
    t = int(i)
    draw_at_time(t)
    return []

draw_at_time(0)
anim = FuncAnimation(fig, update, frames=range(0, T_END + 1), interval=1000, blit=False)
plt.tight_layout()
plt.show()
