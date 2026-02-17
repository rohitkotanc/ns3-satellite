import os, json
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday

tle_path = "starlink.txt"
lines = [ln.strip() for ln in open(tle_path) if ln.strip()]

def is_l1(s): return s.startswith("1 ")
def is_l2(s): return s.startswith("2 ")

pairs = []
i = 0
while i < len(lines)-1:
    # Accept either [name, l1, l2] or [l1, l2]
    if is_l1(lines[i]) and is_l2(lines[i+1]):
        l1, l2 = lines[i], lines[i+1]
        i += 2
    elif i+2 < len(lines) and is_l1(lines[i+1]) and is_l2(lines[i+2]):
        l1, l2 = lines[i+1], lines[i+2]  # skip the name line
        i += 3
    else:
        i += 1
        continue
    try:
        pairs.append(Satrec.twoline2rv(l1, l2))
    except Exception:
        pass

# Limit for speed; raise if you want more points
MAX_SATS = 200
sats = pairs[:MAX_SATS]
print(f"Loaded {len(sats)} satellites from {tle_path}")

start = datetime.utcnow()
frames = []
for step in range(0, 601, 30):  # 0..600s every 30s
    t = start + timedelta(seconds=step)
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond/1e6)
    nodes = []
    for s in sats:
        e, r, v = s.sgp4(jd, fr)
        if e == 0:
            # r is km in TEME; keep as-is for visualization
            nodes.append({"x": r[0], "y": r[1], "z": r[2]})
    frames.append({"t": step, "nodes": nodes})

os.makedirs("logs_ns3", exist_ok=True)
with open("logs_ns3/viz_timeseries.json", "w") as f:
    json.dump({"frames": frames}, f)
print("✅ Wrote logs_ns3/viz_timeseries.json")
