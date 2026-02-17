import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

RUN_DIR = os.getcwd()
LOGS = os.path.join(RUN_DIR, "logs_ns3")
VIZ_JSON = os.path.join(LOGS, "viz_timeseries.json")
PM_CSV = os.path.join(LOGS, "pingmesh.csv")
PING_PKT_BYTES = 64

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

frame_by_t = {}
all_max = 1.0

def frame_positions(fr):
    nodes = fr.get("nodes", [])
    if not nodes:
        return np.zeros((0, 3)), {}
    P = np.array([[n.get("x", 0.0), n.get("y", 0.0), n.get("z", 0.0)] for n in nodes], dtype=float)
    ids = [int(n.get("id", i)) for i, n in enumerate(nodes)]
    idx = {ids[i]: i for i in range(len(ids))}
    return P, idx

for fr in frames:
    ts = fr_time_s(fr)
    if ts is None:
        continue
    t_int = int(round(ts))
    frame_by_t[t_int] = fr
    P, _ = frame_positions(fr)
    if P.size:
        all_max = max(all_max, float(np.max(np.abs(P))))

DF = pd.read_csv(PM_CSV, header=None, comment="#")
ncol = DF.shape[1]
base_names = ["src","dst","seq","time_ns","sent_ns","recv_ns"]
extra = [f"c{i}" for i in range(6, ncol-1)]
names = base_names + extra + ["good"]
DF.columns = names[:ncol]

for c in ["src","dst","seq"]:
    if c in DF.columns:
        DF[c] = pd.to_numeric(DF[c], errors="coerce")
for c in ["time_ns","sent_ns","recv_ns"]:
    if c in DF.columns:
        DF[c] = pd.to_numeric(DF[c], errors="coerce")

if "good" in DF.columns:
    g = DF["good"]
    if g.dtype == object:
        g = g.astype(str).str.strip().str.lower()
        DF["good"] = g.isin(["1","true","t","yes","y"]).astype(int)
    else:
        DF["good"] = pd.to_numeric(g, errors="coerce").fillna(0).astype(int)
else:
    DF["good"] = 0

DF = DF.dropna(subset=["time_ns"])
DF["_ts"] = (DF["time_ns"] / 1e9).round().astype(int)
DF["delay_ms"] = (DF["recv_ns"] - DF["sent_ns"]) / 1e6
DF.loc[DF["good"] == 0, "delay_ms"] = np.nan

edges_by_t = {}
for t, g in DF.groupby("_ts"):
    sub = g[["src","dst"]].dropna().astype(int).to_numpy()
    edges_by_t[int(t)] = sub

per_sec = DF.groupby("_ts").agg(
    total_pkts=("seq","count"),
    good_pkts=("good","sum")
).sort_index()

per_sec["total_mbps"] = (per_sec["total_pkts"] * PING_PKT_BYTES * 8) / 1e6
per_sec["good_mbps"]  = (per_sec["good_pkts"]  * PING_PKT_BYTES * 8) / 1e6

t_vals = per_sec.index.to_numpy()
total_mbps = per_sec["total_mbps"].to_numpy()
good_mbps = per_sec["good_mbps"].to_numpy()

t_end_data = int(max(t_vals)) if len(t_vals) else 0
t_end_frames = int(max(frame_by_t.keys())) if len(frame_by_t) else 0
T_END = max(t_end_data, t_end_frames)

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
ax3d = fig.add_subplot(gs[:, 0], projection="3d")
ax_top = fig.add_subplot(gs[0, 1])
ax_bot = fig.add_subplot(gs[1, 1])

WINDOW = 10

def draw_at_time(t):
    fr = frame_by_t.get(t, None)
    if fr is None:
        past = [k for k in frame_by_t.keys() if k <= t]
        fr = frame_by_t[max(past)] if past else (frames[0] if frames else {"nodes":[]})

    P, idx = frame_positions(fr)

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

    E = edges_by_t.get(t, None)
    if E is not None and P.size:
        cnt = 0
        for s, d in E:
            if s in idx and d in idx:
                ps = P[idx[s]]
                pd_ = P[idx[d]]
                ax3d.plot([ps[0], pd_[0]], [ps[1], pd_[1]], [ps[2], pd_[2]], linewidth=0.5)
                cnt += 1
                if cnt > 400:
                    break

    ax_top.cla()
    if len(t_vals):
        m = t_vals <= t
        ax_top.plot(t_vals[m], total_mbps[m])
        ax_top.plot(t_vals[m], good_mbps[m])
    ax_top.axvline(t, linestyle="--")
    ax_top.set_title("Throughput over time")
    ax_top.set_xlabel("time (s)")
    ax_top.set_ylabel("Mb/s")

    ax_bot.cla()
    lo = max(0, t - WINDOW + 1)
    w = per_sec.loc[(per_sec.index >= lo) & (per_sec.index <= t)]
    if len(w):
        xs = w.index.to_numpy()
        ax_bot.bar(xs, w["good_mbps"].to_numpy(), width=0.9)
        ax_bot.set_xlim(lo - 0.5, t + 0.5)
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
