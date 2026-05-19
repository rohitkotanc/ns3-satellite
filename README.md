## ns3-satellite

Analyzing Latency, UDP, and TCP throughput of different ns-3-based satellite topologies.
This repository has ns-3 and Hypatia used to analyze and visualize satellite network connectivity.
Topologies used are Starlink, Kuiper, and Telesat

Starlink:
Dense Low Earth Orbit Constellation
Very Dynamic Routing
Path Changes Frequently

Kuiper:
Moderately Dense Constellation
Smooth Transitions in Orbit
Moderate Routing Variability

Telesat:
More Stable Topology
Uses RTT data from Topology
Not Frequent Route Changes

Each example contains a 3D satellite topology and a real-time changing network metric.

## Requirements
Python3
Numpy
Matplotlib
SGP4

## How to Run

Go to any example directory: starlink, kuiper, and telesat
Select metric: latency, UDP, TCP


## Repository Structure

examples/
  starlink/
    base_starlink_latency.py
    base_starlink_udp.py
    base_starlink_tcp.py
  kuiper/
    base_kuiper_latency.py
    base_kuiper_udp.py
    base_kuiper_tcp.py
  telesat/
    base_telesat_latency.py
    base_telesat_udp.py
    base_telesat_tcp.py

## Starlink Examples

### base_starlink_latency.py

This example visualizes topology-derived round-trip latency across a dynamic Starlink satellite constellation. The left panel shows the moving 3D satellite topology, while the right panel plots round-trip time over a 200-second simulation. The latency curve changes as the selected route evolves through the moving satellite network, showing how orbital motion and path changes affect end-to-end delay.

[Watch Demo](videos/starlink_latency.mp4)

### base_starlink_udp.py

This example models UDP throughput across the Starlink topology. The simulation uses bottleneck link capacity and hop-count penalties to estimate throughput over time. Since UDP does not include TCP-style congestion control, the graph mainly reflects how route length, bottleneck capacity, and changing topology affect raw throughput.

[Watch Demo](videos/starlink_udp.mp4)

### base_starlink_tcp.py

This example models TCP throughput across the Starlink topology. Unlike UDP, the TCP model includes RTT sensitivity, route-change effects, protocol efficiency, and flow-control limits. This produces lower and more variable throughput, showing how TCP performance is strongly affected by latency and changing satellite routes.

[Watch Demo](videos/starlink_tcp.mp4)

## Kuiper Examples

### base_kuiper_latency.py

This example visualizes topology-derived round-trip latency across the Kuiper satellite constellation. The left panel shows the moving 3D Kuiper topology, while the right panel plots latency over time. The graph demonstrates how Kuiper’s orbital structure and routing changes affect end-to-end delay.

[Watch Demo](videos/kuiper_latency.mp4)

### base_kuiper_udp.py

This example models UDP throughput across the Kuiper topology. The throughput curve reflects the effect of changing path length, bottleneck links, and satellite motion. Compared with TCP, this UDP model focuses on topology and capacity limits without transport-layer congestion behavior.

[Watch Demo](videos/kuiper_udp.mp4)

### base_kuiper_tcp.py

This example models TCP throughput across the Kuiper constellation. The model incorporates topology-derived route behavior along with TCP-style limits such as RTT sensitivity and effective path capacity. This shows how TCP throughput changes as satellite paths evolve over time.

[Watch Demo](videos/kuiper_tcp.mp4)

## Telesat Examples

### base_telesat_latency.py

This example visualizes Telesat latency using topology-derived RTT data together with a 3D constellation view. The latency graph shows step-like changes over time, representing changes in path conditions and routing behavior across the Telesat network.

[Watch Demo](videos/telesat_latency.mp4)

### base_telesat_udp.py

This example models Telesat UDP throughput using topology-derived RTT behavior. The graph shows how throughput improves or decreases as the path becomes more or less favorable. Since this is UDP, the model emphasizes path quality and delay-based throughput changes rather than congestion-control behavior.

[Watch Demo](videos/telesat_udp.mp4)

### base_telesat_tcp.py

This example models Telesat TCP throughput using a more constrained transport-layer model than UDP. The throughput curve responds to delay and path-quality changes in the topology-derived data, showing how TCP performance can drop when network conditions become less favorable.

[Watch Demo](videos/telesat_tcp.mp4)
