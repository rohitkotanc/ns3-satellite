import pandas as pd, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time, os

logs = os.path.join(os.getcwd(), "logs_ns3")
csv_path = os.path.join(logs, "pingmesh.csv")

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, 200)   # 200 s simulation
ax.set_ylim(0, 100)   # adjust depending on delay_ms range
ax.set_xlabel("Time (s)")
ax.set_ylabel("Delay (ms)")
ax.set_title("Real-time Ping Delay Visualization")

def init():
    line.set_data([], [])
    return line,

def update(frame):
    if not os.path.exists(csv_path):
        return line,
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 6:  # safety check
        return line,
    df.columns = ["src","dst","seq","time_s","sent_ns","recv_ns","x1","x2","x3","success"][:df.shape[1]]
    df["delay_ms"] = (df["recv_ns"] - df["sent_ns"]) / 1e6
    times = df["seq"] if "seq" in df else range(len(df))
    delays = df["delay_ms"]
    line.set_data(times, delays)
    ax.set_xlim(0, max(times)+1)
    ax.set_ylim(0, max(delays)*1.1 if len(delays)>0 else 100)
    return line,

ani = FuncAnimation(fig, update, init_func=init, interval=500, blit=False)
plt.show()
