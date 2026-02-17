import os,re,pandas as pd,matplotlib.pyplot as plt
run_dir=os.path.abspath(os.getcwd()); logs=os.path.join(run_dir,"logs_ns3"); cfg=os.path.join(run_dir,"config_ns3.txt")
with open(cfg,'r') as f: text=f.read()
def gi(k): m=re.search(rf"{k}\s*=\s*(\d+)",text); return int(m.group(1)) if m else None
N_sat=gi("num_satellites"); N_gs=gi("num_ground_stations")
def nt(n): return "sat" if n<N_sat else "gs"
m=re.search(r"satellite_network_dir\s*=\s*(.+)",text); topo=None
if m: topo=m.group(1).strip(); topo=os.path.abspath(os.path.join(run_dir,topo)) if not os.path.isabs(topo) else topo
plane={}
for cand in [os.path.join(topo,"satellites_with_plane_ids.txt"),os.path.join(topo,"satellites.txt"),os.path.join(topo,"tles.txt")]:
    if topo and os.path.exists(cand):
        i=0
        with open(cand) as f:
            for line in f:
                s=line.strip()
                if not s: continue
                mm=re.search(r"plane[_ ]?id\s*[:=]\s*(\d+)",s,re.I)
                plane[i]=int(mm.group(1)) if mm else None
                i+=1
        break
def cls(a,b):
    ta,tb=nt(a),nt(b)
    if ta=="sat" and tb=="sat":
        pa,pb=plane.get(a),plane.get(b)
        if pa is not None and pb is not None: return "sat-sat-same-orbit" if pa==pb else "sat-sat-cross-orbit"
        return "sat-sat-unknown-plane"
    if (ta=="sat" and tb=="gs") or (ta=="gs" and tb=="sat"): return "sat-gs"
    return "gs-gs"
df=pd.read_csv(os.path.join(logs,"pingmesh.csv"))
cols=[c.lower() for c in df.columns]
def pick(vs):
    for v in vs:
        if v in cols: return df.columns[cols.index(v)]
    return None
c_src=pick(["src_node_id","src","source","from"])
c_dst=pick(["dst_node_id","dst","dest","to"])
c_rtt=pick(["rtt_ms","rtt","latency_ms"])
df["one_way_ms"]=df[c_rtt]/2.0
df["class"]=[cls(int(s),int(d)) for s,d in zip(df[c_src],df[c_dst])]
by_class=df.groupby("class")["one_way_ms"].agg(["count","mean","median","min","max"]).reset_index()
by_pair=df.groupby([c_src,c_dst])["one_way_ms"].agg(["count","mean","median","min","max"]).reset_index()
by_class.to_csv(os.path.join(logs,"delay_summary_by_class.csv"),index=False)
by_pair.to_csv(os.path.join(logs,"delay_summary_by_pair.csv"),index=False)
plt.figure()
for k,g in df.groupby("class"): g["one_way_ms"].plot(kind="hist",bins=60,alpha=0.5,label=k)
plt.xlabel("One-way delay (ms)"); plt.ylabel("Count"); plt.legend(); plt.title("Delay distribution by link class"); plt.tight_layout()
plt.savefig(os.path.join(logs,"delay_hist_by_class.png"))
plt.figure()
gs=list(df.groupby("class"))
plt.boxplot([g[1]["one_way_ms"].values for g in gs],labels=[g[0] for g in gs],showfliers=False)
plt.ylabel("One-way delay (ms)"); plt.title("One-way delay by link class"); plt.tight_layout()
plt.savefig(os.path.join(logs,"delay_box_by_class.png"))
print("OK")
