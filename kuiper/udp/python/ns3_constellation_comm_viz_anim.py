import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

run_dir = os.getcwd()
logs = os.path.join(run_dir, "logs_ns3")
viz_json = os.path.join(logs, "viz_timeseries.json")
pm_csv = os.path.join(logs, "pingmesh.csv")

with open(viz_json) as f:
    V = json.load(f)

frames = [fr for fr in V.get("frames", []) if isinstance(fr, dict) and "nodes" in fr and fr.get("nodes")]
if not frames:
    raise RuntimeError("No valid frames with nodes found in viz_timeseries.json")

def frame_positions(fr):
    nodes = fr["nodes"]
    P = np.array([[n.get("x", 0.0), n.get("y", 0.0), n.get("z", 0.0)] for n in nodes], dtype=float)
    ids = [int(n.get("id", i)) for i, n in enumerate(nodes)]
    return P, {ids[i]: i for i in range(len(ids))}

all_max = 1.0
t_in_frames = []
for fr in frames:
    t_in_frames.append(float(fr.get("t", 0.0)))
    P, _ = frame_positions(fr)
    if P.size:
        all_max = max(all_max, float(np.nanmax(np.abs(P))))

t_in_frames = np.array(t_in_frames, dtype=float)
t_min_frame = int(np.floor(np.nanmin(t_in_frames)))
t_max_frame = int(np.ceil(np.nanmax(t_in_frames)))

DF = pd.read_csv(
    pm_csv, header=None, comment="#",
    names=["src","dst","seq","time_ns","sent_ns","recv_ns","c6","c7","c8","success"]
)

DF["time_ns"] = pd.to_numeric(DF["time_ns"], errors="coerce")
DF["sent_ns"] = pd.to_numeric(DF["sent_ns"], errors="coerce")
DF["recv_ns"] = pd.to_numeric(DF["recv_ns"], errors="coerce")
DF = DF.dropna(subset=["time_ns","sent_ns","recv_ns"])

DF["delay_ms"] = (DF["recv_ns"] - DF["sent_ns"]) / 1e6
DF = DF.replace([np.inf, -np.inf], np.nan).dropna(subset=["delay_ms"])

DF["_ts"] = np.floor(DF["time_ns"] / 1e9).astype(int)
DF = DF.sort_values("_ts")

t_min_df = int(DF["_ts"].min()) if not DF.empty else t_min_frame
t_max_df = int(DF["_ts"].max()) if not DF.empty else t_max_frame

t_min = min(t_min_frame, t_min_df)
t_max = max(t_max_frame, t_max_df)

edges_by_t = {t: g[["src","dst"]].astype(int).to_numpy() for t, g in DF.groupby("_ts")}
ts_mean = DF.groupby("_ts")["delay_ms"].mean().sort_index()
t_vals = ts_mean.index.to_numpy()
y_vals = ts_mean.to_numpy()

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1, 1])

ax3d = fig.add_subplot(gs[:, 0], projection="3d")
ax_line = fig.add_subplot(gs[0, 1])
ax_hist = fig.add_subplot(gs[1, 1])

ax_line.set_title("Avg delay over time")
ax_line.set_xlabel("time (s)")
ax_line.set_ylabel("avg delay (ms)")

ax_hist.set_title("Delay distribution (sliding window)")
ax_hist.set_xlabel("delay (ms)")
ax_hist.set_ylabel("count")

times = np.arange(t_min, t_max + 1, dtype=int)
frame_by_t = {}
for fr in frames:
    tt = int(round(float(fr.get("t", 0.0))))
    if tt not in frame_by_t:
        frame_by_t[tt] = fr
sorted_frame_ts = np.array(sorted(frame_by_t.keys()), dtype=int)

def get_frame_for_t(t):
    if t in frame_by_t:
        return frame_by_t[t]
    if sorted_frame_ts.size == 0:
        return frames[0]
    j = int(np.clip(np.searchsorted(sorted_frame_ts, t), 0, sorted_frame_ts.size - 1))
    if sorted_frame_ts[j] != t and j > 0:
        if abs(sorted_frame_ts[j-1] - t) <= abs(sorted_frame_ts[j] - t):
            j = j - 1
    return frame_by_t[int(sorted_frame_ts[j])]

window_sec = 10
max_edges = 400

def draw(t):
    fr = get_frame_for_t(t)
    P, idx = frame_positions(fr)

    ax3d.cla()
    if P.size:
        ax3d.scatter(P[:,0], P[:,1], P[:,2], s=1)
    ax3d.set_title(f"Constellation (t={t}s)")
    ax3d.set_xlim([-all_max, all_max])
    ax3d.set_ylim([-all_max, all_max])
    ax3d.set_zlim([-all_max, all_max])
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    E = edges_by_t.get(t)
    if E is not None and P.size:
        cnt = 0
        for s, d in E:
            if s in idx and d in idx:
                ps = P[idx[s]]
                pd_ = P[idx[d]]
                ax3d.plot([ps[0], pd_[0]], [ps[1], pd_[1]], [ps[2], pd_[2]], linewidth=0.5)
                cnt += 1
                if cnt >= max_edges:
                    break

    ax_line.cla()
    if t_vals.size:
        m = t_vals <= t
        ax_line.plot(t_vals[m], y_vals[m])
        ax_line.axvline(t, linestyle="--")
        ax_line.set_xlim(t_min, t_max)
    ax_line.set_title("Avg delay over time")
    ax_line.set_xlabel("time (s)")
    ax_line.set_ylabel("avg delay (ms)")

    ax_hist.cla()
    lo = max(t_min, t - window_sec + 1)
    hi = t
    cur = DF[(DF["_ts"] >= lo) & (DF["_ts"] <= hi)]["delay_ms"]
    if len(cur):
        ax_hist.hist(cur.to_numpy(), bins=30)
    ax_hist.set_title(f"Delay distribution (t={lo}–{hi}s)")
    ax_hist.set_xlabel("delay (ms)")
    ax_hist.set_ylabel("count")

draw(times[0])

anim = FuncAnimation(fig, lambda i: (draw(int(times[i])), []), frames=len(times), interval=1000, blit=False)
plt.tight_layout()
plt.show()
