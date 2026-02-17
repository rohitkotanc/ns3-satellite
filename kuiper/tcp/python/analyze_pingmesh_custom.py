import os, pandas as pd, matplotlib.pyplot as plt

run_dir = os.path.abspath(os.getcwd())
logs = os.path.join(run_dir, "logs_ns3")
csv_path = os.path.join(logs, "pingmesh.csv")

df = pd.read_csv(csv_path, header=None)
df.columns = ["src_id","dst_id","seq","time_s","sent_ns","recv_ns","x1","x2","x3","success"]

df["delay_ms"] = (df["recv_ns"] - df["sent_ns"]) / 1e6
df["class"] = df.apply(lambda r:
    "sat-gs" if (r["src_id"] >= 1156 or r["dst_id"] >= 1156)
    else "sat-sat", axis=1)

summary = df.groupby("class")["delay_ms"].agg(["count","mean","median","min","max"]).reset_index()
summary.to_csv(os.path.join(logs,"delay_summary_by_class.csv"), index=False)

plt.figure()
for cls, sub in df.groupby("class"):
    sub["delay_ms"].plot(kind="hist", bins=50, alpha=0.5, label=cls)
plt.xlabel("Delay (ms)")
plt.ylabel("Count")
plt.legend()
plt.title("Delay Distribution by Class")
plt.tight_layout()
plt.savefig(os.path.join(logs,"delay_hist_by_class.png"))

plt.figure()
groups = list(df.groupby("class"))
plt.boxplot([g[1]["delay_ms"].values for g in groups], labels=[g[0] for g in groups], showfliers=False)
plt.ylabel("Delay (ms)")
plt.title("Delay by Link Class")
plt.tight_layout()
plt.savefig(os.path.join(logs,"delay_box_by_class.png"))
print("OK — delay summaries and graphs saved.")
