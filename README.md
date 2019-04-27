[![Build Status](https://travis-ci.com/StanfordAHA/canal.svg?branch=master)](https://travis-ci.com/StanfordAHA/canal)

# Design Philosophy
In `canal`, the hardware is represented as a directed graph (DiGraph). In principle each node in the graph
will be turn into a hardware component based on the property of the node. The edge will be transformed
into a wire. 

One thing to notice is that staged generation happens on the graph level. That is, passes have to be operated
on the DiGraph, rather than the hardware generator itself. This allows better analysis and reduce the computation
overhead since `magma` is slow when constructing large hardware. In `canal`,  different `bit width` routing
network will have different DiGraph by design. This allows us to share the common passes on different DiGraph
while constructing on the same interconnect

# How to install
```
pip install .
```
Due to the `magma` backend, Python 3.7.2+ is required to install `canal`

# Relationship with CGRA_PNR (Thunder/Cyclone)
The internal DiGraph representation is identical to `Cyclone`. Due to the requirement of pure-Python implementation,
`canal` re-implemented the graph representation with more utility functions. The PnR information is interchangeable
with `Thunder` and `Cyclone` and thus can be used in place route directly.

There is a python utility package called `archipelago`, that can take `Interconnect` object in canal and perform
place and route directly. To install `archipelago`, you can use
```
pip install archipelago
```
It will use pre-built Python native wheels to speed up place and route.