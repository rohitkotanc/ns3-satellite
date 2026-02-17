import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

with open("logs_ns3/viz_timeseries.json", "r") as f:
    data = json.load(f)

frames = data.get("frames", [])
example_frame = None
for fr in reversed(frames):
    if isinstance(fr, dict) and "nodes" in fr and isinstance(fr["nodes"], list) and len(fr["nodes"]) > 0:
        example_frame = fr
        break
if example_frame is None:
    raise RuntimeError("No valid frame with non-empty 'nodes' found in logs_ns3/viz_timeseries.json")

df = pd.read_csv("logs_ns3/pingmesh.csv", header=None)
df.columns = ["src", "dst", "seq", "time_ns", "send_ns", "recv_ns", "src_time_ns", "dst_time_ns", "path_time_ns", "good"]
df["delay_ns"] = (df["recv_ns"] - df["send_ns"]).astype(np.int64)
df["delay_ms"] = df["delay_ns"] / 1e6

avg_delay_over_time = df.groupby("time_ns")["delay_ms"].mean().reset_index()

t_end = int(df["time_ns"].max())
window_ns = int(10e9)
df_tail = df[df["time_ns"] >= (t_end - window_ns)].copy()
if df_tail.empty:
    df_tail = df.copy()

fig = plt.figure(figsize=(22, 10))
gs = GridSpec(2, 2, figure=fig, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.0])
fig.subplots_adjust(left=0.03, right=0.98, wspace=0.25, hspace=0.35)

ax1 = fig.add_subplot(gs[:, 0], projection="3d")
xs = [n["x"] for n in example_frame["nodes"] if "x" in n and "y" in n and "z" in n]
ys = [n["y"] for n in example_frame["nodes"] if "x" in n and "y" in n and "z" in n]
zs = [n["z"] for n in example_frame["nodes"] if "x" in n and "y" in n and "z" in n]
ax1.scatter(xs, ys, zs, s=2)
try:
    ax1.set_box_aspect((1, 1, 1))
except Exception:
    pass
ax1.view_init(elev=20, azim=35)
ax1.set_title(f"Constellation (t={example_frame.get('t','?')}s)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(avg_delay_over_time["time_ns"] / 1e9, avg_delay_over_time["delay_ms"])
ax2.set_title("Avg delay over time")
ax2.set_xlabel("time (s)")
ax2.set_ylabel("avg delay (ms)")

ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(df_tail["delay_ms"].values, bins=80)
ax3.set_title("Delay distribution (last 10 seconds)")
ax3.set_xlabel("delay (ms)")
ax3.set_ylabel("count")

plt.tight_layout()
plt.show()

print("frames:", len(frames))
print("picked frame keys:", list(example_frame.keys()))
print("tail window rows:", len(df_tail))
print("unique delays in tail:", int(df_tail["delay_ms"].nunique()))
print(df_tail["delay_ms"].describe())
