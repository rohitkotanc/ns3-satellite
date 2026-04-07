# ns3-satellite

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


