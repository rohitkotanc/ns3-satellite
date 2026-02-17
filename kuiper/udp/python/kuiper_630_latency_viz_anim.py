import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

run_dir = os.getcwd()
logs = os.path.join(run_dir, "logs_ns3")
viz_json = os.path.join(logs, "viz_timeseries.json")
pm_csv = os.path.join(logs, "pingmesh.csv")

with open(viz_json) as f:
    V = json.load(f)
frames = [fr for fr in V.get("frames", []) if isinstance(fr, dict) and "nodes" in fr]

def frame_positions(fr):
    nodes = fr["nodes"]
    if not nodes:
        return np.zeros((0, 3)), {}
    P = np.array([[n.get("x", 0.0), n.get("y", 0.0), n.get("z", 0.0)] for n in nodes], dtype=float)
    ids = [int(n.get("id", i)) for i, n in enumerate(nodes)]
    return P, {ids[i]: i for i in range(len(ids))}

all_max = 1.0
frame_t = []
for fr in frames:
    t = float(fr.get("t", 0.0))
    frame_t.append(t)
    P, _ = frame_positions(fr)
    if P.size:
        all_max = max(all_max, float(np.max(np.abs(P))))
frame_t = np.array(frame_t, dtype=float)

DF = pd.read_csv(
    pm_csv,
    header=None,
    comment="#",
    names=["src","dst","seq","time_ns","sent_ns","recv_ns","c6","c7","c8","success"]
)

DF["time_ns"] = pd.to_numeric(DF["time_ns"], errors="coerce")
DF["sent_ns"] = pd.to_numeric(DF["sent_ns"], errors="coerce")
DF["recv_ns"] = pd.to_numeric(DF["recv_ns"], errors="coerce")
DF = DF.dropna(subset=["time_ns", "sent_ns", "recv_ns"])

DF["t_s"] = DF["time_ns"] / 1e9
DF["_ts"] = DF["t_s"].round().astype(int)
DF["delay_ms"] = (DF["recv_ns"] - DF["sent_ns"]) / 1e6

DF = DF[np.isfinite(DF["delay_ms"]) & (DF["delay_ms"] >= 0)]

T_END = int(max(0, int(np.nanmax(DF["_ts"])) if not DF.empty else 0))
if len(frames) > 0:
    T_END = max(T_END, int(np.nanmax(frame_t)))

t_grid = np.arange(0, T_END + 1, 1, dtype=int)

ts = DF.groupby("_ts")["delay_ms"].mean().sort_index()
t_vals = ts.index.to_numpy(dtype=int)
y_vals = ts.to_numpy(dtype=float)

edges_by_t = {t: g[["src","dst"]].astype(int).to_numpy() for t, g in DF.groupby("_ts")}

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
ax3d = fig.add_subplot(gs[:, 0], projection="3d")
ax_top = fig.add_subplot(gs[0, 1])
ax_hist = fig.add_subplot(gs[1, 1])

def nearest_frame_index(t):
    if len(frame_t) == 0:
        return 0
    return int(np.argmin(np.abs(frame_t - t)))

def safe_ylim(y_segment):
    y_segment = np.array(y_segment, dtype=float)
    y_segment = y_segment[np.isfinite(y_segment)]
    if y_segment.size == 0:
        return (0.0, 1.0)
    lo = float(np.min(y_segment))
    hi = float(np.max(y_segment))
    if lo == hi:
        pad = max(1e-3, abs(lo) * 0.01)
        return (lo - pad, hi + pad)
    pad = (hi - lo) * 0.08
    return (lo - pad, hi + pad)

def draw_at_time(t):
    fr = frames[nearest_frame_index(t)] if len(frames) > 0 else {"t": t, "nodes": []}

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

    E = edges_by_t.get(int(t))
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
    ax_top.set_title("Delay over time")
    ax_top.set_xlabel("time (s)")
    ax_top.set_ylabel("delay (ms)")
    ax_top.set_xlim(0, T_END)

    if t_vals.size > 0:
        m = (t_vals <= int(t))
        ax_top.plot(t_vals[m], y_vals[m])
        ax_top.axvline(int(t), linestyle="--", linewidth=1)

        y_visible = y_vals[m]
        lo, hi = safe_ylim(y_visible)
        ax_top.set_ylim(lo, hi)
    else:
        ax_top.set_ylim(0, 1)

    ax_hist.cla()
    w = 10
    lo_t = max(0, int(t) - w)
    hi_t = int(t)
    cur = DF.loc[(DF["_ts"] >= lo_t) & (DF["_ts"] <= hi_t), "delay_ms"]
    if not cur.empty:
        ax_hist.hist(cur.to_numpy(dtype=float), bins=30)
    ax_hist.set_title(f"Delay distribution (t={lo_t}-{hi_t}s)")
    ax_hist.set_xlabel("delay (ms)")
    ax_hist.set_ylabel("count")

def update(i):
    t = int(t_grid[i])
    draw_at_time(t)
    return []

draw_at_time(int(t_grid[0]) if t_grid.size else 0)
anim = FuncAnimation(fig, update, frames=len(t_grid), interval=1000, blit=False, repeat=False)
plt.tight_layout()
plt.show()
