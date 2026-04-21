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

This is an example of topology-based round-trip time latency analysis of a dynamically moving Starlink constellation network over a period of 200 seconds. On the left, you can see a 3D visualization of orbits, where satellites and ground stations are included. The figure on the right illustrates round-trip time between nodes 0 and 200. As observed in the figure below, there is significant round-trip time variability ranging from about 68 ms to 85 ms.

### base_starlink_udp.py

The following simulation illustrates the UDP throughput on the same Starlink configuration for 200 seconds, considering the bandwidth limitations at the bottleneck and hop costs. The 3D satellite network changes dynamically in real-time, whereas the throughput chart monitors the throughput between the source node (1584) and the destination node (1683). Throughput fluctuates within the range of 4.0 to 4.8 Gbps with step-wise changes due to the impact of routing and distance. The bottleneck capacity depicted in the graph is 8.40 Gbps, while the effective throughput decreases due to multiple hops.

### base_starlink_tcp.py

This particular example illustrates the throughput behavior of TCP for the Starlink satellite constellation based on a more realistic transport layer model, which takes into consideration the dependency on RTT, route change costs, protocol cost, and topological constraints. In the figure below, the left pane depicts the real-time 3D visualization of the satellite network, whereas the right pane represents the throughput behavior of TCP from source node 1584 to destination node 1683 over 200 seconds. It oscillates around 2.2 to 3.3 Gbps, much lower than UDP because of the delay incurred in the acknowledgment process and flow control.

## Demo Videos

### Starlink Latency
[Watch Video](videos/starlinkbaselatency.mov)

### Starlink UDP
[Watch Video](videos/starlinkbaseUDP.mov)

### Starlink TCP
[Watch Video](videos/starlinkbaseTCP.mov)
