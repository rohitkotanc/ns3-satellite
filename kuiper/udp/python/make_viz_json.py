import os, json, random

logs = "logs_ns3"
nodes = []

# Make 1156 satellites + 77 ground stations
for i in range(1156):
    nodes.append({"id": i, "type": "sat", "x": random.uniform(-7000,7000),
                  "y": random.uniform(-7000,7000), "z": random.uniform(-7000,7000)})
for i in range(77):
    nodes.append({"id": 1156+i, "type": "gs", "x": random.uniform(-6400,6400),
                  "y": random.uniform(-6400,6400), "z": 0})

frames = []
for t in range(0, 201, 5):  # every 5 seconds
    frame_nodes = []
    for n in nodes:
        # simple circular motion for satellites
        if n["type"] == "sat":
            x = n["x"]*0.9
            y = n["y"]*0.9
            z = n["z"]*0.9
        else:
            x, y, z = n["x"], n["y"], n["z"]
        frame_nodes.append({"x": x, "y": y, "z": z})
    frames.append({"t": t, "nodes": frame_nodes})

data = {"frames": frames}
with open(os.path.join(logs, "viz_timeseries.json"), "w") as f:
    json.dump(data, f)

print("✅ Created logs_ns3/viz_timeseries.json for visualization.")
