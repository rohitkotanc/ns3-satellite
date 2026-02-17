import os, json, math, time
import numpy as np
from sgp4.api import Satrec
from datetime import datetime, timedelta

# Read all TLEs
tle_file = "starlink.txt"
sats = []
with open(tle_file) as f:
    lines = f.readlines()
for i in range(0, len(lines), 3):
    name = lines[i].strip()
    if i+2 < len(lines):
        l1, l2 = lines[i+1].strip(), lines[i+2].strip()
        try:
            sats.append(Satrec.twoline2rv(l1, l2))
        except:
            continue

print(f"Loaded {len(sats)} Starlink satellites")

# Generate positions for 0–600 seconds (10 minutes)
frames = []
start = datetime.utcnow()
for step in range(0, 601, 30):
    t = start + timedelta(seconds=step)
    jd, fr = (t - datetime(2000,1,1,12,0,0)).days + 2451545, 0.0
    nodes = []
    for s in sats:
        e, r, v = s.sgp4(jd, fr)
        if e == 0:
            nodes.append({"x": r[0]*1.0, "y": r[1]*1.0, "z": r[2]*1.0})
    frames.append({"t": step, "nodes": nodes})

os.makedirs("logs_ns3", exist_ok=True)
with open("logs_ns3/viz_timeseries.json", "w") as f:
    json.dump({"frames": frames}, f)
print("✅ Generated logs_ns3/viz_timeseries.json using Starlink 5C data")
