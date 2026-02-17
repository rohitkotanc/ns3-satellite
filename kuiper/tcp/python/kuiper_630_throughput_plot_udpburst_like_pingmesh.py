import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs_ns3")
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

t0 = 0.0
t1 = 200.0

edges = []
values = []

for r in out_rows:
    start_s = r[4] / 1e9
    dur_s = r[5] / 1e9
    end_s = start_s + dur_s
    rate = r[7]
    edges.append((start_s, end_s))
    values.append(rate)

times = [t0]
series = [0.0]

for (a, b), v in zip(edges, values):
    times.append(a)
    series.append(series[-1])
    times.append(a)
    series.append(v)
    times.append(b)
    series.append(v)

times.append(t1)
series.append(series[-1] if len(series) else 0.0)

times = np.array(times, dtype=float)
series = np.array(series, dtype=float)

mask = (times >= t0) & (times <= t1)
times = times[mask]
series = series[mask]

plt.figure()
plt.plot(times, series)
plt.xlabel("Time (s)")
plt.ylabel("Throughput (Mbit/s, payload)")
plt.title("Throughput timeseries (UDP bursts)")
plt.grid(True)
plt.savefig(os.path.join(LOGS, "throughput_timeseries.png"), dpi=200, bbox_inches="tight")
plt.close()

samples = []
for (a, b), v in zip(edges, values):
    if b <= t0 or a >= t1:
        continue
    aa = max(a, t0)
    bb = min(b, t1)
    n = int(np.ceil((bb - aa) / 1.0))
    for _ in range(max(n, 1)):
        samples.append(v)

samples = np.array(samples, dtype=float)

plt.figure()
if samples.size == 0:
    plt.hist([0.0], bins=10)
else:
    bins = 40
    plt.hist(samples, bins=bins)
plt.xlabel("Throughput (Mbit/s, payload)")
plt.ylabel("Count (per 1s sample)")
plt.title("Throughput histogram (UDP bursts)")
plt.grid(True)
plt.savefig(os.path.join(LOGS, "throughput_hist.png"), dpi=200, bbox_inches="tight")
plt.close()

pairs = []
for o, i in zip(out_rows, in_rows):
    pairs.append((o[7], i[7]))
pairs = np.array(pairs, dtype=float)

plt.figure()
if pairs.size == 0:
    plt.plot([0.0], [0.0], marker="o", linestyle="none")
else:
    plt.plot(pairs[:, 0], pairs[:, 1], marker="o", linestyle="none")
plt.xlabel("Outgoing (Mbit/s, payload)")
plt.ylabel("Incoming (Mbit/s, payload)")
plt.title("Outgoing vs incoming (per burst)")
plt.grid(True)
plt.savefig(os.path.join(LOGS, "delay_vs_throughput_scatter.png"), dpi=200, bbox_inches="tight")
plt.close()

to_open = [
    os.path.join(LOGS, "throughput_timeseries.png"),
    os.path.join(LOGS, "throughput_hist.png"),
    os.path.join(LOGS, "delay_vs_throughput_scatter.png"),
]

for p in to_open:
    try:
        subprocess.run(["open", p], check=False)
    except Exception:
        pass
