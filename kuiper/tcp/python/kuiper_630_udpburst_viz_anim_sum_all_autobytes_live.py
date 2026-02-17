import os, json, glob, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

RUN_DIR = os.getcwd()
LOGS = os.path.join(RUN_DIR, "logs_ns3")
VIZ_JSON = os.path.join(LOGS, "viz_timeseries.json")
BURSTS_IN_CSV = os.path.join(LOGS, "udp_bursts_incoming.csv")

WINDOW = 10
FPS = 1
SIM_SECONDS = 200

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

def burst_payload_bytes_per_pkt():
    m = {}
    with open(BURSTS_IN_CSV, "r") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            bid = int(row[0])
            pkts = int(row[8])
            payload_bytes = int(row[10])
            if pkts > 0:
                m[bid] = payload_bytes / pkts
    return m

def read_pkt_file(path):
    a = np.loadtxt(path, delimiter=",", dtype=np.int64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    bid = int(a[0, 0])
    seq = a[:, 1].astype(np.int64)
    t_ns = a[:, 2].astype(np.int64)
    t_s = t_ns / 1e9
    t_int = np.floor(t_s).astype(np.int64)
    return bid, seq, t_ns, t_int

def per_second_bytes(pattern, bytes_per_pkt_map):
    per = {}
    t_end = 0
    files = sorted(glob.glob(os.path.join(LOGS, pattern)))
    for fp in files:
        bid, seq, t_ns, t_int = read_pkt_file(fp)
        bpp = float(bytes_per_pkt_map.get(bid, 0.0))
        if bpp <= 0.0:
            continue
        t_end = max(t_end, int(t_int.max()))
        df = pd.DataFrame({"t_int": t_int})
        g = df.groupby("t_int").size()
        for k, v in g.items():
            per[int(k)] = per.get(int(k), 0.0) + float(v) * bpp
    idx = np.arange(0, t_end + 1, dtype=int) if t_end >= 0 else np.array([0], dtype=int)
    bytes_s = np.array([per.get(int(t), 0.0) for t in idx], dtype=float)
    return idx, bytes_s

def rolling_mbps(bytes_s, window):
    if len(bytes_s) == 0:
        return np.array([0.0], dtype=float)
    w = int(window)
    out = np.zeros_like(bytes_s, dtype=float)
    c = np.cumsum(bytes_s, dtype=float)
    for i in range(len(bytes_s)):
        j0 = max(0, i - w + 1)
        s = c[i] - (c[j0 - 1] if j0 > 0 else 0.0)
        out[i] = (s * 8.0) / 1e6 / float(w)
    return out

def build_delay_samples_by_second(bytes_per_pkt_map):
    out_files = sorted(glob.glob(os.path.join(LOGS, "udp_burst_*_outgoing.csv")))
    in_files = sorted(glob.glob(os.path.join(LOGS, "udp_burst_*_incoming.csv")))
    out_by_bid = {}
    for fp in out_files:
        bid, seq, t_ns, t_int = read_pkt_file(fp)
        out_by_bid[bid] = (seq, t_ns)
    samples = {}
    for fp in in_files:
        bid, seq_i, t_i_ns, t_i_int = read_pkt_file(fp)
        if bid not in out_by_bid:
            continue
        seq_o, t_o_ns = out_by_bid[bid]
        order_o = np.argsort(seq_o)
        seq_o_s = seq_o[order_o]
        t_o_s = t_o_ns[order_o]
        order_i = np.argsort(seq_i)
        seq_i_s = seq_i[order_i]
        t_i_s = t_i_ns[order_i]
        t_i_int_s = t_i_int[order_i]

        pos = np.searchsorted(seq_o_s, seq_i_s)
        ok = (pos >= 0) & (pos < len(seq_o_s)) & (seq_o_s[pos] == seq_i_s)
        pos = pos[ok]
        t_i_ok = t_i_s[ok]
        t_i_int_ok = t_i_int_s[ok]
        t_o_ok = t_o_s[pos]
        d_ms = (t_i_ok - t_o_ok).astype(np.float64) / 1e6

        for tsec, d in zip(t_i_int_ok.tolist(), d_ms.tolist()):
            if tsec not in samples:
                samples[tsec] = []
            samples[tsec].append(d)
    return samples

frame_by_t, all_max = load_frames()
bytes_per_pkt_map = burst_payload_bytes_per_pkt()

t_out, out_bytes_s = per_second_bytes("udp_burst_*_outgoing.csv", bytes_per_pkt_map)
t_in, in_bytes_s = per_second_bytes("udp_burst_*_incoming.csv", bytes_per_pkt_map)

t_max = int(max(t_out.max() if len(t_out) else 0, t_in.max() if len(t_in) else 0, SIM_SECONDS))
T = np.arange(0, t_max + 1, dtype=int)

out_full = np.zeros_like(T, dtype=float)
in_full = np.zeros_like(T, dtype=float)
if len(t_out):
    out_full[t_out] = out_bytes_s
if len(t_in):
    in_full[t_in] = in_bytes_s

out_mbps = rolling_mbps(out_full, WINDOW)
in_mbps = rolling_mbps(in_full, WINDOW)

delay_samples = build_delay_samples_by_second(bytes_per_pkt_map)

plt.rcParams["figure.figsize"] = (12, 7)
fig = plt.figure()

ax3d = fig.add_subplot(2, 2, 1, projection="3d")
ax_thr = fig.add_subplot(2, 2, 2)
ax_hist = fig.add_subplot(2, 1, 2)

xs = np.array([], dtype=float)
ys = np.array([], dtype=float)
zs = np.array([], dtype=float)
sc = ax3d.scatter(xs, ys, zs, s=2)

ax3d.set_xlim(-all_max, all_max)
ax3d.set_ylim(-all_max, all_max)
ax3d.set_zlim(-all_max, all_max)

line_out, = ax_thr.plot([], [], label="Outgoing (payload Mbps)")
line_in, = ax_thr.plot([], [], label="Incoming (payload Mbps)")
ax_thr.set_xlim(0, t_max)
ax_thr.set_ylim(0, max(1.0, float(np.max([out_mbps.max() if len(out_mbps) else 0.0, in_mbps.max() if len(in_mbps) else 0.0])) * 1.1))
ax_thr.set_xlabel("Time (s)")
ax_thr.set_ylabel(f"Mbps (rolling {WINDOW}s)")
ax_thr.legend(loc="upper right")

hist_bins = 60

def window_delays(t_now):
    d = []
    for tsec in range(max(0, t_now - WINDOW + 1), t_now + 1):
        if tsec in delay_samples:
            d.extend(delay_samples[tsec])
    return np.array(d, dtype=float) if d else np.array([], dtype=float)

def init():
    line_out.set_data([], [])
    line_in.set_data([], [])
    ax_hist.clear()
    return sc, line_out, line_in

def update(frame):
    t_now = int(frame)

    fr = frame_by_t.get(t_now, None)
    if fr is not None:
        nodes = fr.get("nodes", [])
        if nodes:
            P = np.array([[n.get("x", 0.0), n.get("y", 0.0), n.get("z", 0.0)] for n in nodes], dtype=float)
            sc._offsets3d = (P[:, 0], P[:, 1], P[:, 2])

    x = T[: min(t_now + 1, len(T))]
    y_out = out_mbps[: min(t_now + 1, len(out_mbps))]
    y_in = in_mbps[: min(t_now + 1, len(in_mbps))]
    line_out.set_data(x, y_out)
    line_in.set_data(x, y_in)

    ax_hist.clear()
    d = window_delays(t_now)
    if len(d):
        ax_hist.hist(d, bins=hist_bins)
        ax_hist.set_xlim(float(np.min(d)), float(np.max(d)))
    ax_hist.set_xlabel(f"Delay (ms) over last {WINDOW}s")
    ax_hist.set_ylabel("Packets")

    fig.suptitle(f"Kuiper UDP Bursts | t = {t_now}s", fontsize=12)
    return sc, line_out, line_in

anim = FuncAnimation(
    fig,
    update,
    frames=np.arange(0, min(t_max, SIM_SECONDS) + 1, 1),
    init_func=init,
    interval=int(1000 / FPS),
    blit=False,
    repeat=False
)

plt.tight_layout()
plt.show()
