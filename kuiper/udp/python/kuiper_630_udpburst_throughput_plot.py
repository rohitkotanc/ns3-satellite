import os
import csv
import numpy as np
import matplotlib.pyplot as plt

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
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

t_out = np.array([r[4] / 1e9 for r in out_rows], dtype=float)
dur_out = np.array([r[5] / 1e9 for r in out_rows], dtype=float)
y_out = np.array([r[7] for r in out_rows], dtype=float)

t_in = np.array([r[4] / 1e9 for r in in_rows], dtype=float)
dur_in = np.array([r[5] / 1e9 for r in in_rows], dtype=float)
y_in = np.array([r[7] for r in in_rows], dtype=float)

t_end_out = t_out + dur_out
t_end_in = t_in + dur_in

plt.figure()
plt.step(np.r_[t_out, t_end_out[-1]], np.r_[y_out, y_out[-1]], where="post")
plt.xlabel("Time (s)")
plt.ylabel("Outgoing throughput (Mbit/s, payload)")
plt.title("UDP burst throughput (outgoing)")
plt.grid(True)
plt.savefig(os.path.join(LOGS, "udpburst_throughput_outgoing_timeseries.png"), dpi=200, bbox_inches="tight")
plt.close()

plt.figure()
plt.step(np.r_[t_in, t_end_in[-1]], np.r_[y_in, y_in[-1]], where="post")
plt.xlabel("Time (s)")
plt.ylabel("Incoming throughput (Mbit/s, payload)")
plt.title("UDP burst throughput (incoming)")
plt.grid(True)
plt.savefig(os.path.join(LOGS, "udpburst_throughput_incoming_timeseries.png"), dpi=200, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(y_out, y_in, marker="o", linestyle="none")
plt.xlabel("Outgoing throughput (Mbit/s, payload)")
plt.ylabel("Incoming throughput (Mbit/s, payload)")
plt.title("Outgoing vs incoming (per burst)")
plt.grid(True)
plt.savefig(os.path.join(LOGS, "udpburst_throughput_scatter.png"), dpi=200, bbox_inches="tight")
plt.close()
