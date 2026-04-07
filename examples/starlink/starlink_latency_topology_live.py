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
        raise ValueError("Node list exists but no valid x/y/z nodes were parsed")

    return out

def dist(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

def visible_from_ground(gs_pos, sat_pos, min_elevation_deg=25.0):
    gs = np.array(gs_pos, dtype=float)
    sat = np.array(sat_pos, dtype=float)
    los = sat - gs
    los_norm = np.linalg.norm(los)
    if los_norm == 0:
        return False
    los_hat = los / los_norm
    zenith_hat = gs / np.linalg.norm(gs)
    elev_rad = np.arcsin(np.dot(los_hat, zenith_hat))
    elev_deg = np.degrees(elev_rad)
    return elev_deg >= min_elevation_deg

def build_graph(nodes, n_sats=1584, k_neighbors=4, min_elevation_deg=25.0, gs_links=2):
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
            G.add_edge(nid, nid2, weight=delay)
            nbr_count += 1
            if nbr_count >= k_neighbors:
                break

    for gs in gs_ids:
        gsp = pos[gs]
        visible = []
        for sat in sat_ids:
            satp = pos[sat]
            if visible_from_ground(gsp, satp, min_elevation_deg=min_elevation_deg):
                d = dist(gsp, satp)
                visible.append((d, sat))
        visible.sort(key=lambda x: x[0])
        for d, sat in visible[:gs_links]:
            delay = d / C
            G.add_edge(gs, sat, weight=delay)

    return G, pos, sat_ids, gs_ids

def choose_ground_pair(gs_ids):
    if len(gs_ids) < 2:
        raise ValueError("Need at least two ground stations")
    return gs_ids[0], gs_ids[-1]

def compute_rtt_ms(G, src, dst):
    try:
        one_way_s = nx.shortest_path_length(G, source=src, target=dst, weight="weight")
        return 2.0 * one_way_s * 1000.0
    except Exception:
        return np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=200.0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--n_sats", type=int, default=1584)
    ap.add_argument("--k_neighbors", type=int, default=4)
    ap.add_argument("--min_elevation_deg", type=float, default=25.0)
    ap.add_argument("--gs_links", type=int, default=2)
    ap.add_argument("--src", type=int, default=None)
    ap.add_argument("--dst", type=int, default=None)
    args = ap.parse_args()

    frames = load_frames(VIZ_JSON)

    duration = float(args.duration)
    fps = float(args.fps)
    t_anim = np.arange(0.0, duration + 1e-9, 1.0 / fps)
    frame_idxs = np.linspace(0, len(frames) - 1, len(t_anim)).astype(int)

    print(f"Loaded {len(frames)} frames")
    print(f"Using {len(frame_idxs)} frames for animation")

    node_series = []
    latency_series = []
    src = args.src
    dst = args.dst
    last_good_nodes = None

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
        )

        if src is None or dst is None:
            src, dst = choose_ground_pair(gs_ids)

        rtt_ms = compute_rtt_ms(G, src, dst)
        latency_series.append(rtt_ms)

        if j % 10 == 0 or j == len(frame_idxs) - 1:
            if np.isfinite(rtt_ms):
                print(f"Computed frame {j+1}/{len(frame_idxs)}   RTT={rtt_ms:.2f} ms")
            else:
                print(f"Computed frame {j+1}/{len(frame_idxs)}   RTT=no path")

    latency_series = np.array(latency_series, dtype=float)

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

    ax3d.set_title("Starlink topology (3D)")
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
    if src in pos0:
        sx, sy, sz = pos0[src]
        src_sc = ax3d.scatter([sx], [sy], [sz], s=50)
    if dst in pos0:
        dx, dy, dz = pos0[dst]
        dst_sc = ax3d.scatter([dx], [dy], [dz], s=50)

    ax2d.set_title("Topology-derived RTT over time")
    ax2d.set_xlabel("time (s)")
    ax2d.set_ylabel("RTT (ms)")
    ax2d.set_xlim(0, duration)

    finite = latency_series[np.isfinite(latency_series)]
    if finite.size:
        ymin = float(np.min(finite))
        ymax = float(np.max(finite))
        if abs(ymax - ymin) < 1e-6:
            center = float(np.mean(finite))
            ax2d.set_ylim(center - 5.0, center + 5.0)
        else:
            pad = 0.1 * (ymax - ymin)
            ax2d.set_ylim(ymin - pad, ymax + pad)
    else:
        ax2d.set_ylim(0, 100)

    line, = ax2d.plot([], [], linewidth=2)
    vline = ax2d.axvline(0.0)
    txt = ax2d.text(0.02, 0.98, "", transform=ax2d.transAxes, va="top")

    def update(i):
        nonlocal src_sc, dst_sc

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

        if src_sc is not None and src in pos:
            sx, sy, sz = pos[src]
            src_sc._offsets3d = (np.array([sx]), np.array([sy]), np.array([sz]))
        if dst_sc is not None and dst in pos:
            dx, dy, dz = pos[dst]
            dst_sc._offsets3d = (np.array([dx]), np.array([dy]), np.array([dz]))

        t = t_anim[i]
        xs = t_anim[: i + 1]
        ys = latency_series[: i + 1]

        line.set_data(xs, ys)
        vline.set_xdata([t, t])

        val = latency_series[i]
        if np.isfinite(val):
            txt.set_text(f"t = {t:.1f}s\nRTT = {val:.2f} ms\nsrc = {src}\ndst = {dst}")
        else:
            txt.set_text(f"t = {t:.1f}s\nRTT = no path\nsrc = {src}\ndst = {dst}")

        artists = [sat_sc, line, vline, txt]
        if gs_sc is not None:
            artists.append(gs_sc)
        if src_sc is not None:
            artists.append(src_sc)
        if dst_sc is not None:
            artists.append(dst_sc)
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
