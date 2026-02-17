import os, json, glob, re, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter

RUN_DIR = os.getcwd()
LOGS = os.path.join(RUN_DIR, "logs_ns3")
VIZ_JSON = os.path.join(LOGS, "viz_timeseries.json")

WINDOW = 10

def load_frames():
    with open(VIZ_JSON) as f:
        V = json.load(f)
    raw_frames = V.get("frames", [])
    frames = [fr for fr in raw_frames if isinstance(fr, dict) and "nodes" in fr]
    frame_by_t = {}
    all_max = 1.0
    for fr in frames:
        t = fr.get("t", None)
        if t is None:
            continue
        try:
            ts = int(round(float(t)))
        except Exception:
            continue
        frame_by_t[ts] = fr
        nodes = fr.get("nodes", [])
        if nodes:
            P = np.array([[n.get("x", 0.0), n.get("y", 0.0), n.get("z", 0.0)] for n in nodes], dtype=float)
            all_max = max(all_max, float(np.max(np.abs(P))))
    return frame_by_t, all_max

def read_pkt_csv(path, colname):
    df = pd.read_csv(path, header=None, names=["burst_id", "seq", colname])
    df[colname] = pd.to_numeric(df[colname], errors="coerce")
    df = df.dropna(subset=[colname])
    df["t_s"] = (df[colname].astype(float) / 1e9)
    df["t_int"] = np.floor(df["t_s"]).astype(int)
    return df[["t_int"]]

def burst_bytes_per_pkt(summary_csv_path):
    m = {}
    if not os.path.exists(summary_csv_path):
        return m
    with open(summary_csv_path, "r") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            try:
                bid = int(row[0])
                pkts = int(row[8])
                payload_bytes = int(row[10])
            except Exception:
                continue
            if pkts > 0:
                m[bid] = payload_bytes / pkts
    return m

def sum_throughput_mbps(pattern, colname, bytes_per_pkt_map):
    files = sorted(glob.glob(os.path.join(LOGS, pattern)))
    if not files:
        return np.array([0.0]), np.array([0])

    per_bytes = {}
    t_end = 0

    for fp in files:
        m = re.search(r"udp_burst_(\d+)_", os.path.basename(fp))
        if not m:
            continue
        bid = int(m.group(1))
        bpp = bytes_per_pkt_map.get(bid, None)
        if bpp is None:
            continue

        df = read_pkt_csv(fp, colname)
        if len(df) == 0:
            continue

        t_end = max(t_end, int(df["t_int"].max()))
        g = df.groupby("t_int").size()
        for k, v in g.items():
            k = int(k)
            per_bytes[k] = per_bytes.get(k, 0.0) + float(v) * float(bpp)

    idx = np.arange(0, t_end + 1, dtype=int)
    bytes_arr = np.array([per_bytes.get(int(t), 0.0) for t in idx], dtype=float)
    mbps = (bytes_arr * 8.0) / 1e6
    return mbps, idx

frame_by_t, all_max = load_frames()

in_bpp = burst_bytes_per_pkt(os.path.join(LOGS, "udp_bursts_incoming.csv"))
out_bpp = burst_bytes_per_pkt(os.path.join(LOGS, "udp_bursts_outgoing.csv"))

in_mbps, t_vals_in = sum_throughput_mbps("udp_burst_*_incoming.csv", "recv_ns", in_bpp)
out_mbps, t_vals_out = sum_throughput_mbps("udp_burst_*_outgoing.csv", "sent_ns", out_bpp)

T_END = int(max(
    int(t_vals_in.max()) if len(t_vals_in) else 0,
    int(t_vals_out.max()) if len(t_vals_out) else 0,
    max(frame_by_t.keys()) if frame_by_t else 0
))

def get_positions_at(t):
    fr = frame_by_t.get(t, None)
    if fr is None:
        past = [k for k in frame_by_t.keys() if k <= t]
        if past:
            fr = frame_by_t[max(past)]
        else:
            return np.zeros((0, 3), dtype=float)
    nodes = fr.get("nodes", [])
    if not nodes:
        return np.zeros((0, 3), dtype=float)
    P = np.array([[n.get("x", 0.0), n.get("y", 0.0), n.get("z", 0.0)] for n in nodes], dtype=float)
    return P

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
ax3d = fig.add_subplot(gs[:, 0], projection="3d")
ax_top = fig.add_subplot(gs[0, 1])
ax_bot = fig.add_subplot(gs[1, 1])

fmt_plain = ScalarFormatter(useOffset=False)
fmt_plain.set_scientific(False)

def draw_at_time(t):
    ax3d.cla()
    P = get_positions_at(t)
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
    n = min(t + 1, len(in_mbps))
    ax_top.plot(t_vals_in[:n], in_mbps[:n])
    ax_top.axvline(t, linestyle="--")
    ax_top.set_xlim(0, T_END)
    ax_top.set_title("Throughput over time")
    ax_top.set_xlabel("time (s)")
    ax_top.set_ylabel("Mb/s")
    ax_top.yaxis.set_major_formatter(fmt_plain)
    ax_top.xaxis.set_major_formatter(fmt_plain)

    ax_bot.cla()
    lo = max(0, t - WINDOW + 1)
    xs = np.arange(lo, t + 1, dtype=int)
    if t + 1 <= len(in_mbps):
        ywin = in_mbps[lo:t + 1]
    else:
        ywin = np.pad(in_mbps[lo:], (0, t + 1 - len(in_mbps)), constant_values=0.0)
    ax_bot.bar(xs, ywin, width=0.9)
    ax_bot.set_xlim([lo - 0.5, t + 0.5])
    ax_bot.set_title(f"Throughput (last {WINDOW}s)")
    ax_bot.set_xlabel("time (s)")
    ax_bot.set_ylabel("Mb/s")
    ax_bot.yaxis.set_major_formatter(fmt_plain)
    ax_bot.xaxis.set_major_formatter(fmt_plain)

def update(i):
    t = int(i)
    draw_at_time(t)
    return []

draw_at_time(0)
anim = FuncAnimation(fig, update, frames=np.arange(0, T_END + 1, 1), interval=1000, blit=False)
plt.tight_layout()
plt.show()
