import os
import re
import json
import math
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def pick_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def load_json_any(path):
    with open(path, "r") as f:
        return json.load(f)

def coerce_frames(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["frames", "timeline", "data", "samples"]:
            if k in obj:
                v = obj[k]
                if isinstance(v, list):
                    return v
                if isinstance(v, dict):
                    for kk in ["frames", "timeline", "data", "samples"]:
                        if kk in v and isinstance(v[kk], list):
                            return v[kk]
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], (dict, list)):
                return v
    return []

def try_get_time_ns(frame):
    if isinstance(frame, dict):
        for k in ["t_ns", "time_ns", "timestamp_ns", "sim_time_ns", "t", "time", "timestamp"]:
            if k in frame:
                v = frame[k]
                if isinstance(v, (int, float)):
                    if v < 1e6:
                        return int(v * 1e9)
                    return int(v)
                if isinstance(v, str):
                    try:
                        if "." in v:
                            return int(float(v) * 1e9)
                        return int(v)
                    except Exception:
                        pass
        for k in list(frame.keys()):
            lk = str(k).lower()
            if "time" in lk and isinstance(frame[k], (int, float)):
                v = frame[k]
                if v < 1e6:
                    return int(v * 1e9)
                return int(v)
    return None

def extract_node_quads_anywhere(frame):
    best = []

    def is_quad_list(lst):
        if not isinstance(lst, list) or not lst:
            return False
        a = lst[0]
        if not isinstance(a, (list, tuple)) or len(a) < 4:
            return False
        return all(isinstance(x, (int, float)) for x in a[:4])

    def is_dict_nodes(lst):
        if not isinstance(lst, list) or not lst:
            return False
        a = lst[0]
        return isinstance(a, dict) and any(k in a for k in ["id", "node_id", "nid"]) and any(k in a for k in ["x", "y", "z"])

    def dict_nodes_to_quads(lst):
        out = []
        for d in lst:
            if not isinstance(d, dict):
                continue
            nid = d.get("id", d.get("node_id", d.get("nid", None)))
            if nid is None:
                continue
            x = d.get("x", None)
            y = d.get("y", None)
            z = d.get("z", None)
            if x is None or y is None or z is None:
                if "pos" in d and isinstance(d["pos"], (list, tuple)) and len(d["pos"]) >= 3:
                    x, y, z = d["pos"][:3]
            try:
                out.append((int(nid), float(x), float(y), float(z)))
            except Exception:
                pass
        return out

    def walk(obj):
        nonlocal best
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, list) and (is_quad_list(v) or is_dict_nodes(v)):
                    cand = v if is_quad_list(v) else dict_nodes_to_quads(v)
                    if isinstance(cand, list) and len(cand) > len(best):
                        best = cand
                walk(v)
        elif isinstance(obj, list):
            if is_quad_list(obj):
                if len(obj) > len(best):
                    best = obj
            elif is_dict_nodes(obj):
                cand = dict_nodes_to_quads(obj)
                if len(cand) > len(best):
                    best = cand
            for v in obj[:100]:
                walk(v)

    walk(frame)

    out = []
    for it in best:
        if isinstance(it, (list, tuple)) and len(it) >= 4:
            try:
                out.append((int(it[0]), float(it[1]), float(it[2]), float(it[3])))
            except Exception:
                pass
        elif isinstance(it, dict):
            nid = it.get("id", it.get("node_id", it.get("nid", None)))
            if nid is None:
                continue
            try:
                x, y, z = float(it["x"]), float(it["y"]), float(it["z"])
                out.append((int(nid), x, y, z))
            except Exception:
                pass
    return out

def load_viz_timeseries(run_dir):
    p = pick_existing(
        os.path.join(run_dir, "logs_ns3", "viz_timeseries.json"),
        os.path.join(run_dir, "viz_timeseries.json"),
        os.path.join(run_dir, "logs_ns3", "viz_timeseries", "viz_timeseries.json"),
    )
    if not p:
        raise FileNotFoundError("Could not find viz_timeseries.json (tried logs_ns3/viz_timeseries.json and viz_timeseries.json).")
    data = load_json_any(p)
    frames = coerce_frames(data)
    if not isinstance(frames, list) or not frames:
        raise ValueError("viz_timeseries.json loaded but frames list could not be found.")
    return frames

def parse_tcp_flows_txt(path):
    lines = []
    with open(path, "r") as f:
        for l in f:
            l = l.rstrip("\n")
            if not l.strip():
                continue
            if l.lower().startswith("tcp flow id"):
                continue
            if not re.match(r"^\s*\d+\s", l):
                continue
            lines.append(l.strip())

    flows = []
    for l in lines:
        parts = l.split()
        try:
            fid = int(parts[0])
            src = int(parts[1])
            dst = int(parts[2])
        except Exception:
            continue

        try:
            i_mbit = next(i for i, t in enumerate(parts) if t.lower() == "mbit")
        except StopIteration:
            continue

        if i_mbit < 1:
            continue

        size_num = parts[i_mbit - 1]
        try:
            size_mbit = float(size_num)
        except Exception:
            m = re.search(r"(\d+(?:\.\d+)?)", size_num)
            size_mbit = float(m.group(1)) if m else None
        if size_mbit is None:
            continue

        if i_mbit + 2 >= len(parts):
            continue

        start_ns_s = parts[i_mbit + 1]
        end_ns_s = parts[i_mbit + 2]
        try:
            start_ns = int(float(start_ns_s))
            end_ns = int(float(end_ns_s))
        except Exception:
            continue

        dur_s = max((end_ns - start_ns) / 1e9, 1e-9)
        rate_mbps = (size_mbit / dur_s)

        flows.append(
            {
                "id": fid,
                "src": src,
                "dst": dst,
                "size_mbit": float(size_mbit),
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
                "rate_mbps": float(rate_mbps),
            }
        )
    if not flows:
        raise ValueError("Parsed 0 TCP flows from tcp_flows.txt (format unexpected).")
    return flows

def build_time_axis(frames, fps, duration_s):
    t_ns = []
    for fr in frames:
        tn = try_get_time_ns(fr)
        t_ns.append(tn)
    if all(v is None for v in t_ns):
        n = len(frames)
        dt = 1.0 / max(fps, 1e-9)
        t_s = np.arange(n) * dt
        if duration_s is not None:
            t_s = np.clip(t_s, 0.0, duration_s)
        return (t_s * 1e9).astype(np.int64)
    out = np.array([v if v is not None else -1 for v in t_ns], dtype=np.int64)
    last = None
    for i in range(len(out)):
        if out[i] >= 0:
            last = out[i]
        else:
            if last is None:
                out[i] = 0
                last = 0
            else:
                out[i] = last + int(1e9 / max(fps, 1e-9))
                last = out[i]
    if duration_s is not None:
        out = np.clip(out, 0, int(duration_s * 1e9))
    return out

def compute_throughput_series(flows, duration_s, dt_s):
    n = int(math.floor(duration_s / dt_s)) + 1
    ts = np.arange(n) * dt_s
    y = np.zeros(n, dtype=float)
    for fl in flows:
        s = fl["start_ns"] / 1e9
        e = fl["end_ns"] / 1e9
        if e <= s:
            continue
        i0 = max(int(math.floor(s / dt_s)), 0)
        i1 = min(int(math.ceil(e / dt_s)), n - 1)
        for i in range(i0, i1 + 1):
            t0 = ts[i]
            t1 = t0 + dt_s
            overlap = max(0.0, min(e, t1) - max(s, t0))
            if overlap > 0:
                y[i] += fl["rate_mbps"]
    return ts, y

def nearest_frame_index(t_frames_ns, target_ns):
    return int(np.argmin(np.abs(t_frames_ns - target_ns)))

def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--run_dir", default=".", type=str)
    ap.add_argument("--duration", default=200.0, type=float)
    ap.add_argument("--fps", default=1.0, type=float)
    ap.add_argument("--dt", default=1.0, type=float)
    ap.add_argument("--save_gif", action="store_true")
    ap.add_argument("--save_mp4", action="store_true")
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--out_dir", default=None, type=str)
    ap.add_argument("--src", default=None, type=int)
    ap.add_argument("--dst", default=None, type=int)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = args.out_dir or os.path.join(run_dir, "logs_ns3", "tcp_viz_out")
    os.makedirs(out_dir, exist_ok=True)

    tcp_txt = pick_existing(
        os.path.join(run_dir, "logs_ns3", "tcp_flows.txt"),
        os.path.join(run_dir, "tcp_flows.txt"),
    )
    if not tcp_txt:
        raise FileNotFoundError("Could not find tcp_flows.txt (tried logs_ns3/tcp_flows.txt and tcp_flows.txt).")
    flows = parse_tcp_flows_txt(tcp_txt)

    if args.src is None or args.dst is None:
        flows_sorted = sorted(flows, key=lambda d: d["start_ns"])
        src = flows_sorted[0]["src"]
        dst = flows_sorted[0]["dst"]
    else:
        src = int(args.src)
        dst = int(args.dst)

    frames = load_viz_timeseries(run_dir)
    t_frames_ns = build_time_axis(frames, args.fps, args.duration)

    sim_end_s = float(args.duration)
    for fl in flows:
        sim_end_s = max(sim_end_s, fl["end_ns"] / 1e9)
    sim_end_s = float(args.duration) if args.duration is not None else float(sim_end_s)
    sim_end_s = max(sim_end_s, 1.0)

    ts, thr = compute_throughput_series(flows, sim_end_s, args.dt)
    y_max = max(1.0, float(np.nanmax(thr)) * 1.05)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.35, hspace=0.35)

    ax0 = fig.add_subplot(gs[:, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    ax1.set_title("Throughput over time")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("Mbit/s")
    ax1.set_xlim(0, sim_end_s)
    ax1.set_ylim(0, y_max)

    ax2.set_title("Throughput (last 10s)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("Mbit/s")
    ax2.set_xlim(max(0, sim_end_s - 10), sim_end_s)
    ax2.set_ylim(0, y_max)

    line_full, = ax1.plot([], [], lw=1.5)
    vline1 = ax1.axvline(0.0, ls="--", lw=1.2)

    bars = ax2.bar([], [])

    txt = fig.text(0.06, 0.88, "", ha="left", va="top")

    sc_all = ax0.scatter([], [], [], s=1)
    sc_src = ax0.scatter([], [], [], s=40)
    sc_dst = ax0.scatter([], [], [], s=40)

    def set_3d_limits(nodes):
        xs = np.array([n[1] for n in nodes], dtype=float)
        ys = np.array([n[2] for n in nodes], dtype=float)
        zs = np.array([n[3] for n in nodes], dtype=float)
        if xs.size == 0:
            return
        pad = 0.05
        def lim(a):
            lo = float(np.nanmin(a))
            hi = float(np.nanmax(a))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo -= 1.0
                hi += 1.0
            r = hi - lo
            lo -= pad * r
            hi += pad * r
            return lo, hi
        x0, x1 = lim(xs)
        y0, y1 = lim(ys)
        z0, z1 = lim(zs)
        ax0.set_xlim(x0, x1)
        ax0.set_ylim(y0, y1)
        ax0.set_zlim(z0, z1)

    base_nodes = extract_node_quads_anywhere(frames[0])
    set_3d_limits(base_nodes)

    def update(frame_i):
        fr = frames[frame_i]
        t_ns = int(t_frames_ns[frame_i])
        t_s = t_ns / 1e9

        nodes = extract_node_quads_anywhere(fr)
        xs = np.array([n[1] for n in nodes], dtype=float)
        ys = np.array([n[2] for n in nodes], dtype=float)
        zs = np.array([n[3] for n in nodes], dtype=float)
        if xs.size and ys.size and zs.size:
            sc_all._offsets3d = (xs, ys, zs)

        src_xyz = None
        dst_xyz = None
        for nid, x, y, z in nodes:
            if nid == src:
                src_xyz = (x, y, z)
            if nid == dst:
                dst_xyz = (x, y, z)
        if src_xyz is not None:
            sc_src._offsets3d = (np.array([src_xyz[0]]), np.array([src_xyz[1]]), np.array([src_xyz[2]]))
        if dst_xyz is not None:
            sc_dst._offsets3d = (np.array([dst_xyz[0]]), np.array([dst_xyz[1]]), np.array([dst_xyz[2]]))

        ax0.set_title(f"Constellation (t={int(round(t_s))}s)")

        idx = int(np.searchsorted(ts, t_s, side="right"))
        idx = max(1, min(idx, len(ts)))

        line_full.set_data(ts[:idx], thr[:idx])
        vline1.set_xdata([t_s, t_s])

        w_end = t_s
        w_start = max(0.0, w_end - 10.0)

        i0 = int(np.searchsorted(ts, w_start, side="left"))
        i1 = int(np.searchsorted(ts, w_end, side="right"))
        i0 = max(0, min(i0, len(ts)))
        i1 = max(i0, min(i1, len(ts)))

        ax2.cla()
        ax2.set_title("Throughput (last 10s)")
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("Mbit/s")
        ax2.set_xlim(w_start, w_end if w_end > w_start else w_start + 10.0)
        ax2.set_ylim(0, y_max)

        if i1 - i0 > 0:
            tt = ts[i0:i1]
            yy = thr[i0:i1]
            ax2.bar(tt, yy, width=args.dt * 0.9, align="edge")

        active = 0
        for fl in flows:
            s = fl["start_ns"] / 1e9
            e = fl["end_ns"] / 1e9
            if s <= t_s < e:
                active += 1

        txt.set_text(f"t = {int(round(t_s))}s\nsrc={src} dst={dst}\nactive flows = {active}")

        return sc_all, sc_src, sc_dst, line_full, vline1, txt

    total_frames = int(round(sim_end_s * args.fps)) + 1
    if total_frames < 2:
        total_frames = 2

    target_t_ns = (np.arange(total_frames) / args.fps * 1e9).astype(np.int64)
    frame_map = [nearest_frame_index(t_frames_ns, tn) for tn in target_t_ns]

    anim = FuncAnimation(fig, lambda i: update(frame_map[i]), frames=len(frame_map), interval=int(1000 / max(args.fps, 1e-9)), blit=False)

    if args.save_mp4:
        out_mp4 = os.path.join(out_dir, "tcp_overview_live.mp4")
        try:
            anim.save(out_mp4, dpi=150)
        except Exception:
            pass

    if args.save_gif or (not args.no_show and not args.save_mp4):
        out_gif = os.path.join(out_dir, "tcp_overview_live.gif")
        try:
            anim.save(out_gif, dpi=150)
        except Exception:
            pass

    last_png = os.path.join(out_dir, f"tcp_overview_t{int(round(sim_end_s))}s.png")
    update(nearest_frame_index(t_frames_ns, int(sim_end_s * 1e9)))
    fig.savefig(last_png, dpi=150)

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    print("Wrote:", last_png)

if __name__ == "__main__":
    raise SystemExit(main())
