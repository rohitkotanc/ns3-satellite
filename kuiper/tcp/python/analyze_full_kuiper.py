import os, pandas as pd, matplotlib.pyplot as plt

run_dir = os.path.abspath(os.getcwd())
logs = os.path.join(run_dir, "logs_ns3")

topo_dir = "/Users/rohitkota/hypatia/paper/satellite_networks_state/gen_data/kuiper_630_isls_none_ground_stations_paris_moscow_grid_algorithm_free_one_only_gs_relays"

tles_file = os.path.join(topo_dir, "tles.txt")
gs_file  = os.path.join(topo_dir, "ground_stations.txt")

with open(tles_file) as f:
    num_sat = sum(1 for _ in f)

with open(gs_file) as f:
    num_gs = sum(1 for _ in f)

def node_type(n): return "sat" if n < num_sat else "gs"

df = pd.read_csv(os.path.join(logs, "pingmesh.csv"), header=None)

df["src"]  = df[0]
df["dst"]  = df[1]
df["rtt_ns"] = df[6]

df["one_way_ms"] = (df["rtt_ns"] / 1e6) / 2.0

df["class"] = [
    ("sat-sat" if node_type(int(s))=="sat" and node_type(int(d))=="sat"
     else "sat-gs" if (node_type(int(s))=="sat" and node_type(int(d))=="gs")
     or (node_type(int(s))=="gs" and node_type(int(d))=="sat")
     else "gs-gs")
    for s, d in zip(df["src"], df["dst"])
]

packet_bits = 64 * 8
df["rtt_s"] = df["rtt_ns"] / 1e9
df["throughput_mbps"] = (packet_bits / df["rtt_s"]) / 1e6

delay_by_class = df.groupby("class")["one_way_ms"].agg(["count","mean","median","min","max"])
thr_by_class   = df.groupby("class")["throughput_mbps"].agg(["count","mean","median","min","max"])

delay_by_class.to_csv(os.path.join(logs, "delay_summary_by_class.csv"))
thr_by_class.to_csv(os.path.join(logs, "throughput_summary_by_class.csv"))

plt.figure(figsize=(8,6))
for k, g in df.groupby("class"):
    g["one_way_ms"].plot(kind="hist", alpha=0.5, bins=50, label=k)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(logs, "delay_hist_by_class.png"))

plt.figure(figsize=(8,6))
groups = list(df.groupby("class"))
plt.boxplot([g[1]["one_way_ms"].values for g in groups],
            labels=[g[0] for g in groups], showfliers=False)
plt.tight_layout()
plt.savefig(os.path.join(logs, "delay_box_by_class.png"))

plt.figure(figsize=(8,6))
plt.plot(df.index, df["throughput_mbps"])
plt.tight_layout()
plt.savefig(os.path.join(logs, "throughput_timeseries.png"))

plt.figure(figsize=(8,6))
df["throughput_mbps"].plot(kind="hist", bins=50)
plt.tight_layout()
plt.savefig(os.path.join(logs, "throughput_hist.png"))

plt.figure(figsize=(8,6))
plt.scatter(df["one_way_ms"], df["throughput_mbps"], s=10)
plt.tight_layout()
plt.savefig(os.path.join(logs, "delay_vs_throughput_scatter.png"))

print("COMPLETE.")