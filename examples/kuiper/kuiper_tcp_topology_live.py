import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

C = 299792458.0
RUN_DIR = os.path.abspath(".")
LOGS_DIR = os.path.join(RUN_DIR, "logs_ns3")
VIZ_JSON = os.path.join(LOGS_DIR, "viz_timeseries.json")

def load_frames(path):
    with open(path, "r") as f:
        j = json.load(f)
    if isinstance(j, dict) and "frames" in j:
        frames = j["frames"]
    elif isinstance(j, list):
        frames = j
    else:
        raise ValueError("Could not find frames in viz_timeseries.json")
    if not frames:
        raise ValueError("No frames found in viz_timeseries.json")
    return frames

def extract_nodes(frame, fallback_nodes=None):
    nodes = None
    if isinstance(frame, dict):
        if "nodes" in frame and isinstance(frame["nodes"], list):
            nodes = frame["nodes"]
        else:
            for v in frame.values():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    if "x" in v[0] and "y" in v[0] and "z" in v[0]:
                        nodes = v
                        break

    if nodes is None:
        if fallback_nodes is not None:
            return fallback_nodes
        raise ValueError("Could not parse nodes from viz_timeseries.json")

    out = []
    for i, n in enumerate(nodes):
        if not isinstance(n, dict):
            continue
        if "x" not in n or "y" not in n or "z" not in n:
            continue
        nid = n.get("id", n.get("node_id", n.get("nid", i)))
        out.append((int(nid), float(n["x"]), float(n["y"]), float(n["z"])))

    if not out:
        if fallback_nodes is not None:
            return fallback_nodes
        raise ValueError("Node list exists but no valid nodes were parsed")
    return out

def dist(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

def elevation_deg(gs_pos, sat_pos):
    gs = np.array(gs_pos, dtype=float)
    sat = np.array(sat_pos, dtype=float)
    los = sat - gs
    los_norm = np.linalg.norm(los)
    if los_norm == 0:
        return -90.0
    los_hat = los / los_norm
    zenith_hat = gs / np.linalg.norm(gs)
    elev_rad = np.arcsin(np.dot(los_hat, zenith_hat))
    return float(np.degrees(elev_rad))

def gsl_capacity_from_elevation(elev_deg, max_rate_gbps):
    if elev_deg < 10:
        scale = 0.15
    elif elev_deg < 20:
        scale = 0.35
    elif elev_deg < 30:
        scale = 0.55
    elif elev_deg < 40:
        scale = 0.72
    elif elev_deg < 50:
        scale = 0.84
    elif elev_deg < 60:
        scale = 0.92
    else:
        scale = 1.0
    return max_rate_gbps * scale

def build_graph(
    nodes,
    n_sats=630,
    k_neighbors=4,
    min_elevation_deg=25.0,
    gs_links=2,
    isl_rate_gbps=10.0,
    gsl_rate_gbps=10.0
):
    G = nx.Graph()
    pos = {nid: (x, y, z) for nid, x, y, z in nodes}

    sat_ids = [nid for nid in pos if nid < n_sats]
    gs_ids = [nid for nid in pos if nid >= n_sats]

    for nid in pos:
        G.add_node(nid)

    sat_positions = np.array([pos[nid] for nid in sat_ids], dtype=float)

    for i, nid in enumerate(sat_ids):
        p = sat_positions[i]
        dists = np.linalg.norm(sat_positions - p, axis=1)
        order = np.argsort(dists)
        nbr_count = 0
        for j in order:
            nid2 = sat_ids[j]
            if nid2 == nid:
                continue
            d = dist(pos[nid], pos[nid2])
            delay = d / C
            G.add_edge(
                nid,
                nid2,
                weight=delay,
                capacity_gbps=isl_rate_gbps,
                link_type="isl"
            )
            nbr_count += 1
            if nbr_count >= k_neighbors:
                break

    for gs in gs_ids:
        gsp = pos[gs]
        visible = []
        for sat in sat_ids:
            satp = pos[sat]
            elev = elevation_deg(gsp, satp)
            if elev >= min_elevation_deg:
                d = dist(gsp, satp)
                cap = gsl_capacity_from_elevation(elev, gsl_rate_gbps)
                visible.append((d, sat, elev, cap))
        visible.sort(key=lambda x: x[0])
        for d, sat, elev, cap in visible[:gs_links]:
            delay = d / C
            G.add_edge(
                gs,
                sat,
                weight=delay,
                capacity_gbps=cap,
                link_type="gsl",
                elevation_deg=elev
            )

    return G, pos, sat_ids, gs_ids

def compute_topology_metrics(G, src, dst, alpha=0.12):
    try:
        path = nx.shortest_path(G, source=src, target=dst, weight="weight")
    except Exception:
        return np.nan, None, np.nan, np.nan, np.nan

    caps = []
    delays = []
    for u, v in zip(path[:-1], path[1:]):
        caps.append(G[u][v]["capacity_gbps"])
        delays.append(G[u][v]["weight"])

    if not caps or not delays:
        return np.nan, path, np.nan, np.nan, np.nan

    bottleneck = min(caps)
    hops = len(path) - 1
    one_way_s = sum(delays)
    rtt_ms = 2.0 * one_way_s * 1000.0
    effective_capacity = bottleneck / (1.0 + alpha * max(0, hops - 1))

    return effective_capacity, path, bottleneck, hops, rtt_ms

def path_change_fraction(path, prev_path):
    if path is None or prev_path is None:
        return 0.0
    e1 = set(zip(path[:-1], path[1:]))
    e2 = set(zip(prev_path[:-1], prev_path[1:]))
    if not e1 and not e2:
        return 0.0
    union = e1 | e2
    inter = e1 & e2
    return 1.0 - (len(inter) / max(1, len(union)))

def compute_tcp_throughput_gbps(
    effective_capacity_gbps,
    rtt_ms,
    route_change_frac,
    tcp_window_mb=16.0,
    protocol_efficiency=0.9
):
    if not np.isfinite(effective_capacity_gbps) or not np.isfinite(rtt_ms) or rtt_ms <= 0:
        return np.nan

    window_bits = tcp_window_mb * 8.0 * 1e6
    rtt_s = rtt_ms / 1000.0

    route_penalty = 1.0 - 0.25 * route_change_frac
    route_penalty = max(0.6, route_penalty)

    window_limited_gbps = route_penalty * protocol_efficiency * (window_bits / rtt_s) / 1e9

    return min(effective_capacity_gbps, window_limited_gbps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=200.0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--n_sats", type=int, default=630)
    ap.add_argument("--k_neighbors", type=int, default=4)
    ap.add_argument("--min_elevation_deg", type=float, default=25.0)
    ap.add_argument("--gs_links", type=int, default=2)
    ap.add_argument("--isl_rate_gbps", type=float, default=10.0)
    ap.add_argument("--gsl_rate_gbps", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.12)
    ap.add_argument("--tcp_window_mb", type=float, default=16.0)
    ap.add_argument("--protocol_efficiency", type=float, default=0.9)
    ap.add_argument("--src", type=int, required=True)
    ap.add_argument("--dst", type=int, required=True)
    args = ap.parse_args()

    frames = load_frames(VIZ_JSON)

    duration = float(args.duration)
    fps = float(args.fps)
    t_anim = np.arange(0.0, duration + 1e-9, 1.0 / fps)
    frame_idxs = np.linspace(0, len(frames) - 1, len(t_anim)).astype(int)

    print(f"Loaded {len(frames)} frames")
    print(f"Using {len(frame_idxs)} frames for animation")

    node_series = []
    throughput_series = []
    bottleneck_series = []
    hops_series = []
    rtt_series = []
    route_change_series = []
    path_series = []

    last_good_nodes = None
    prev_path = None

    for j, idx in enumerate(frame_idxs):
        fr = frames[idx]
        nodes = extract_nodes(fr, fallback_nodes=last_good_nodes)
        last_good_nodes = nodes
        node_series.append(nodes)

        G, pos, sat_ids, gs_ids = build_graph(
            nodes,
            n_sats=args.n_sats,
            k_neighbors=args.k_neighbors,
            min_elevation_deg=args.min_elevation_deg,
            gs_links=args.gs_links,
            isl_rate_gbps=args.isl_rate_gbps,
            gsl_rate_gbps=args.gsl_rate_gbps,
        )

        effective_capacity, path, bottleneck, hops, rtt_ms = compute_topology_metrics(
            G,
            args.src,
            args.dst,
            alpha=args.alpha
        )

        route_change_frac = path_change_fraction(path, prev_path)

        thr = compute_tcp_throughput_gbps(
            effective_capacity_gbps=effective_capacity,
            rtt_ms=rtt_ms,
            route_change_frac=route_change_frac,
            tcp_window_mb=args.tcp_window_mb,
            protocol_efficiency=args.protocol_efficiency
        )

        prev_path = path

        throughput_series.append(thr)
        bottleneck_series.append(bottleneck)
        hops_series.append(hops)
        rtt_series.append(rtt_ms)
        route_change_series.append(route_change_frac)
        path_series.append(path)

        if j % 10 == 0 or j == len(frame_idxs) - 1:
            if np.isfinite(thr):
                print(
                    f"Computed frame {j+1}/{len(frame_idxs)}   "
                    f"tcp={thr:.3f} Gbps   hops={hops}   rtt={rtt_ms:.2f} ms   route_change={route_change_frac:.3f}"
                )
            else:
                print(f"Computed frame {j+1}/{len(frame_idxs)}   tcp=no path")

    throughput_series = np.array(throughput_series, dtype=float)

    fig = plt.figure(figsize=(13.5, 6.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    nodes0 = node_series[0]
    pos0 = {nid: (x, y, z) for nid, x, y, z in nodes0}

    sat0 = [(nid, *pos0[nid]) for nid in pos0 if nid < args.n_sats]
    gs0 = [(nid, *pos0[nid]) for nid in pos0 if nid >= args.n_sats]

    sat_x0 = np.array([p[1] for p in sat0], dtype=float)
    sat_y0 = np.array([p[2] for p in sat0], dtype=float)
    sat_z0 = np.array([p[3] for p in sat0], dtype=float)

    gs_x0 = np.array([p[1] for p in gs0], dtype=float) if gs0 else np.array([])
    gs_y0 = np.array([p[2] for p in gs0], dtype=float) if gs0 else np.array([])
    gs_z0 = np.array([p[3] for p in gs0], dtype=float) if gs0 else np.array([])

    all_x0 = np.array([p[1] for p in nodes0], dtype=float)
    all_y0 = np.array([p[2] for p in nodes0], dtype=float)
    all_z0 = np.array([p[3] for p in nodes0], dtype=float)

    ax3d.set_title("Kuiper topology (3D)")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.view_init(elev=22, azim=35)
    ax3d.set_xlim(np.min(all_x0), np.max(all_x0))
    ax3d.set_ylim(np.min(all_y0), np.max(all_y0))
    ax3d.set_zlim(np.min(all_z0), np.max(all_z0))

    sat_sc = ax3d.scatter(sat_x0, sat_y0, sat_z0, s=4)
    gs_sc = ax3d.scatter(gs_x0, gs_y0, gs_z0, s=20) if gs0 else None

    src_sc = None
    dst_sc = None
    if args.src in pos0:
        sx, sy, sz = pos0[args.src]
        src_sc = ax3d.scatter([sx], [sy], [sz], s=50)
    if args.dst in pos0:
        dx, dy, dz = pos0[args.dst]
        dst_sc = ax3d.scatter([dx], [dy], [dz], s=50)

    path_line = None
    p0 = path_series[0]
    if p0 is not None and len(p0) >= 2:
        px = [pos0[n][0] for n in p0 if n in pos0]
        py = [pos0[n][1] for n in p0 if n in pos0]
        pz = [pos0[n][2] for n in p0 if n in pos0]
        path_line, = ax3d.plot(px, py, pz, linewidth=2)

    ax2d.set_title("Topology-derived TCP throughput over time")
    ax2d.set_xlabel("time (s)")
    ax2d.set_ylabel("Throughput (Gbps)")
    ax2d.set_xlim(0, duration)

    finite = throughput_series[np.isfinite(throughput_series)]
    ymax = max(args.isl_rate_gbps, args.gsl_rate_gbps)
    if finite.size:
        ymin = max(0.0, float(np.min(finite)) - 0.1 * ymax)
        ymax_plot = float(np.max(finite)) + 0.1 * ymax
        ax2d.set_ylim(ymin, ymax_plot)
    else:
        ax2d.set_ylim(0, ymax)

    line, = ax2d.plot([], [], linewidth=2)
    vline = ax2d.axvline(0.0)
    txt = ax2d.text(0.02, 0.98, "", transform=ax2d.transAxes, va="top")

    def update(i):
        nonlocal path_line

        nodes = node_series[i]
        pos = {nid: (x, y, z) for nid, x, y, z in nodes}

        sat = [(nid, *pos[nid]) for nid in pos if nid < args.n_sats]
        sat_x = np.array([p[1] for p in sat], dtype=float)
        sat_y = np.array([p[2] for p in sat], dtype=float)
        sat_z = np.array([p[3] for p in sat], dtype=float)
        sat_sc._offsets3d = (sat_x, sat_y, sat_z)

        if gs_sc is not None:
            gs = [(nid, *pos[nid]) for nid in pos if nid >= args.n_sats]
            gs_x = np.array([p[1] for p in gs], dtype=float)
            gs_y = np.array([p[2] for p in gs], dtype=float)
            gs_z = np.array([p[3] for p in gs], dtype=float)
            gs_sc._offsets3d = (gs_x, gs_y, gs_z)

        if src_sc is not None and args.src in pos:
            sx, sy, sz = pos[args.src]
            src_sc._offsets3d = (np.array([sx]), np.array([sy]), np.array([sz]))
        if dst_sc is not None and args.dst in pos:
            dx, dy, dz = pos[args.dst]
            dst_sc._offsets3d = (np.array([dx]), np.array([dy]), np.array([dz]))

        if path_line is not None:
            path_line.remove()
            path_line = None

        pth = path_series[i]
        if pth is not None and len(pth) >= 2:
            px = [pos[n][0] for n in pth if n in pos]
            py = [pos[n][1] for n in pth if n in pos]
            pz = [pos[n][2] for n in pth if n in pos]
            path_line, = ax3d.plot(px, py, pz, linewidth=2)

        t = t_anim[i]
        xs = t_anim[: i + 1]
        ys = throughput_series[: i + 1]

        line.set_data(xs, ys)
        vline.set_xdata([t, t])

        val = throughput_series[i]
        hops = hops_series[i]
        bottleneck = bottleneck_series[i]
        rtt = rtt_series[i]
        rc = route_change_series[i]

        if np.isfinite(val):
            txt.set_text(
                f"t = {t:.1f}s\n"
                f"TCP throughput = {val:.3f} Gbps\n"
                f"bottleneck = {bottleneck:.2f} Gbps\n"
                f"hops = {hops}\n"
                f"RTT = {rtt:.2f} ms\n"
                f"route change = {rc:.3f}\n"
                f"src = {args.src}\n"
                f"dst = {args.dst}"
            )
        else:
            txt.set_text(
                f"t = {t:.1f}s\n"
                f"TCP throughput = no path\n"
                f"src = {args.src}\n"
                f"dst = {args.dst}"
            )

        artists = [sat_sc, line, vline, txt]
        if gs_sc is not None:
            artists.append(gs_sc)
        if src_sc is not None:
            artists.append(src_sc)
        if dst_sc is not None:
            artists.append(dst_sc)
        if path_line is not None:
            artists.append(path_line)
        return tuple(artists)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(t_anim),
        interval=int(1000 / fps),
        blit=False,
        repeat=False
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
