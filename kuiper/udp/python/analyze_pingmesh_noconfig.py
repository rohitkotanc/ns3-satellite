import os, re, argparse, pandas as pd, matplotlib.pyplot as plt
ap=argparse.ArgumentParser()
ap.add_argument("--n_sat", type=int, required=True)
ap.add_argument("--n_gs", type=int, required=True)
ap.add_argument("--topo", type=str, default="")
args=ap.parse_args()
run_dir=os.path.abspath(os.getcwd())
logs=os.path.join(run_dir,"logs_ns3")
pm_csv=os.path.join(logs,"pingmesh.csv")
df=pd.read_csv(pm_csv)
cols=[c.lower() for c in df.columns]
def pick(vs):
    for v in vs:
        if v in cols: return df.columns[cols.index(v)]
    return None
c_src=pick(["src_node_id","src","source","from"])
c_dst=pick(["dst_node_id","dst","dest","to"])
c_rtt=pick(["rtt_ms","rtt","latency_ms"])
def node_type(n): return "sat" if n<args.n_sat else "gs"
plane={}
if args.topo:
    for cand in [os.path.join(args.topo,"satellites_with_plane_ids.txt"),
                 os.path.join(args.topo,"satellites.txt"),
                 os.path.join(args.topo,"tles.txt")]:
        if os.path.exists(cand):
            i=0
            with open(cand) as f:
                for line in f:
                    s=line.strip()
                    if not s: continue
                    mm=re.search(r"plane[_ ]?id\s*[:=]\s*(\d+)",s,re.I)
                    plane[i]=int(mm.group(1)) if mm else None
                    i+=1
            break
def classify(a,b):
    ta,tb=node_type(a),node_type(b)
    if ta=="sat" and tb=="sat":
        pa,pb=plane.get(a),plane.get(b)
        if pa is not None and pb is not None:
            return "sat-sat-same-orbit" if pa==pb else "sat-sat-cross-orbit"
        return "sat-sat-unknown-plane"
    if (ta=="sat" and tb=="gs") or (ta=="gs" and tb=="sat"):
        return "sat-gs"
    return "gs-gs"
df["one_way_ms"]=df[c_rtt]/2.0
df["class"]=[classify(int(s),int(d)) for s,d in zip(df[c_src],df[c_dst])]
by_class=df.groupby("class")["one_way_ms"].agg(["count","mean","median","min","max"]).reset_index()
by_pair=df.groupby([c_src,c_dst])["one_way_ms"].agg(["count","mean","median","min","max"]).reset_index()
by_class.to_csv(os.path.join(logs,"delay_summary_by_class.csv"),index=False)
by_pair.to_csv(os.path.join(logs,"delay_summary_by_pair.csv"),index=False)
plt.figure()
for k,g in df.groupby("class"):
    g["one_way_ms"].plot(kind="hist",bins=60,alpha=0.5,label=k)
plt.xlabel("One-way delay (ms)")
plt.ylabel("Count")
plt.legend()
plt.title("Delay distribution by link class")
plt.tight_layout()
plt.savefig(os.path.join(logs,"delay_hist_by_class.png"))
plt.figure()
groups=list(df.groupby("class"))
plt.boxplot([g[1]["one_way_ms"].values for g in groups],labels=[g[0] for g in groups],showfliers=False)
plt.ylabel("One-way delay (ms)")
plt.title("One-way delay by link class")
plt.tight_layout()
plt.savefig(os.path.join(logs,"delay_box_by_class.png"))
print("OK")
