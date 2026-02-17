import os
import re
import csv
import glob
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN_DIR = os.getcwd()
LOGS = os.path.join(RUN_DIR, "logs_ns3")

OUT_SUM = os.path.join(LOGS, "udp_bursts_outgoing.csv")
IN_SUM = os.path.join(LOGS, "udp_bursts_incoming.csv")

def read_summary(path):
    rows = []
    with open(path, "r") as f:
        r = csv.reader(f)
        for line in r:
            if not line:
                continue
            bid = int(line[0])
            src = int(line[1])
            dst = int(line[2])
            target = float(line[3])
            start_ns = int(line[4])
            dur_ns = int(line[5])
            rate_with_headers = float(line[6])
            rate_payload = float(line[7])
            pkts = int(line[8])
            rows.append((bid, src, dst, target, start_ns, dur_ns, rate_with_headers, rate_payload, pkts))
    return rows

out_rows = read_summary(OUT_SUM)
in_rows = read_summary(IN_SUM)

out_rows.sort(key=lambda x: x[4])
in_rows.sort(key=lambda x: x[4])

out_by_id = {r[0]: r for r in out_rows}
in_by_id = {r[0]: r for r in in_rows}

burst_ids = sorted(set(out_by_id.keys()) & set(in_by_id.keys()))
if not burst_ids:
    raise RuntimeError("No burst IDs found in udp_bursts_{outgoing,incoming}.csv")

t_end_ns = 0
for bid in burst_ids:
    r = out_by_id[bid]
    t_end_ns = max(t_end_ns, r[4] + r[5])
T_END = int(np.ceil(t_end_ns / 1e9))

times = np.arange(0, T_END + 1, dtype=int)
series = np.zeros_like(times, dtype=float)

for bid in burst_ids:
    r_in = in_by_id[bid]
    start_s = r_in[4] / 1e9
    dur_s = r_in[5] / 1e9
    end_s = start_s + dur_s
    v = float(r_in[7])
    a = int(np.floor(start_s))
    b = int(np.ceil(end_s))
    a = max(a, 0)
    b = min(b, T_END + 1)
    if b > a:
        series[a:b] = v

plt.figure()
plt.plot(times, series)
plt.xlabel("Time (s)")
plt.ylabel("Throughput (Mbit/s, payload)")
plt.title("Throughput timeseries (UDP bursts)")
plt.grid(True)
plt.savefig(os.path.join(LOGS, "throughput_timeseries.png"), dpi=200, bbox_inches="tight")
plt.close()

plt.figure()
bins = 40
plt.hist(series.astype(float), bins=bins)
plt.xlabel("Throughput (Mbit/s, payload)")
plt.ylabel("Count (per 1s sample)")
plt.title("Throughput histogram (UDP bursts)")
plt.grid(True)
plt.savefig(os.path.join(LOGS, "throughput_hist.png"), dpi=200, bbox_inches="tight")
plt.close()

delay_means_ms = []
thr_in_payload = []
labels = []

delay_hist_data = {}
for bid in burst_ids:
    out_path = os.path.join(LOGS, f"udp_burst_{bid}_outgoing.csv")
    in_path = os.path.join(LOGS, f"udp_burst_{bid}_incoming.csv")
    if not (os.path.exists(out_path) and os.path.exists(in_path)):
        continue

    out_df = pd.read_csv(out_path, header=None, names=["burst_id", "seq", "sent_ns"])
    in_df = pd.read_csv(in_path, header=None, names=["burst_id", "seq", "recv_ns"])

    out_df["seq"] = pd.to_numeric(out_df["seq"], errors="coerce")
    out_df["sent_ns"] = pd.to_numeric(out_df["sent_ns"], errors="coerce")
    in_df["seq"] = pd.to_numeric(in_df["seq"], errors="coerce")
    in_df["recv_ns"] = pd.to_numeric(in_df["recv_ns"], errors="coerce")

    out_df = out_df.dropna(subset=["seq", "sent_ns"])
    in_df = in_df.dropna(subset=["seq", "recv_ns"])

    out_df["seq"] = out_df["seq"].astype(np.int64)
    in_df["seq"] = in_df["seq"].astype(np.int64)

    m = out_df.merge(in_df, on="seq", how="inner")
    if len(m) == 0:
        continue

    delay_ms = (m["recv_ns"].to_numpy(dtype=np.float64) - m["sent_ns"].to_numpy(dtype=np.float64)) / 1e6
    delay_ms = delay_ms[np.isfinite(delay_ms)]
    delay_ms = delay_ms[delay_ms >= 0]

    if delay_ms.size == 0:
        continue

    delay_hist_data[bid] = delay_ms
    delay_means_ms.append(float(np.mean(delay_ms)))
    thr_in_payload.append(float(in_by_id[bid][7]))
    labels.append(str(bid))

if delay_means_ms:
    plt.figure()
    plt.plot(thr_in_payload, delay_means_ms, marker="o", linestyle="none")
    plt.xlabel("Incoming throughput (Mbit/s, payload)")
    plt.ylabel("Mean one-way delay (ms)")
    plt.title("Delay vs throughput (per burst)")
    plt.grid(True)
    plt.savefig(os.path.join(LOGS, "delay_vs_throughput_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close()
else:
    plt.figure()
    plt.plot([0.0], [0.0], marker="o", linestyle="none")
    plt.xlabel("Incoming throughput (Mbit/s, payload)")
    plt.ylabel("Mean one-way delay (ms)")
    plt.title("Delay vs throughput (per burst)")
    plt.grid(True)
    plt.savefig(os.path.join(LOGS, "delay_vs_throughput_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close()

if delay_hist_data:
    plt.figure()
    for bid in sorted(delay_hist_data.keys()):
        d = delay_hist_data[bid]
        if d.size > 200000:
            idx = np.random.choice(d.size, size=200000, replace=False)
            d = d[idx]
        plt.hist(d, bins=80, alpha=0.5, label=str(bid))
    plt.xlabel("One-way delay (ms)")
    plt.ylabel("Count")
    plt.title("Delay histogram by burst ID")
    plt.grid(True)
    plt.legend(title="Burst ID")
    plt.savefig(os.path.join(LOGS, "delay_hist_by_class.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    data = [delay_hist_data[bid] if delay_hist_data[bid].size <= 200000 else delay_hist_data[bid][np.random.choice(delay_hist_data[bid].size, size=200000, replace=False)] for bid in sorted(delay_hist_data.keys())]
    plt.boxplot(data, labels=[str(bid) for bid in sorted(delay_hist_data.keys())], showfliers=False)
    plt.xlabel("Burst ID")
    plt.ylabel("One-way delay (ms)")
    plt.title("Delay boxplot by burst ID")
    plt.grid(True)
    plt.savefig(os.path.join(LOGS, "delay_box_by_class.png"), dpi=200, bbox_inches="tight")
    plt.close()
else:
    plt.figure()
    plt.hist([0.0], bins=10)
    plt.xlabel("One-way delay (ms)")
    plt.ylabel("Count")
    plt.title("Delay histogram by burst ID")
    plt.grid(True)
    plt.savefig(os.path.join(LOGS, "delay_hist_by_class.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.boxplot([[0.0]], labels=["0"], showfliers=False)
    plt.xlabel("Burst ID")
    plt.ylabel("One-way delay (ms)")
    plt.title("Delay boxplot by burst ID")
    plt.grid(True)
    plt.savefig(os.path.join(LOGS, "delay_box_by_class.png"), dpi=200, bbox_inches="tight")
    plt.close()

to_open = [
    os.path.join(LOGS, "throughput_timeseries.png"),
    os.path.join(LOGS, "throughput_hist.png"),
    os.path.join(LOGS, "delay_vs_throughput_scatter.png"),
    os.path.join(LOGS, "delay_hist_by_class.png"),
    os.path.join(LOGS, "delay_box_by_class.png"),
]

for p in to_open:
    try:
        subprocess.run(["open", p], check=False)
    except Exception:
        pass
