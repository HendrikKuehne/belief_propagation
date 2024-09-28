# The structure of the tensor networks

Tensor networks are realised using the `networkx.MultiGraph` class.

Each node in the graph has a key `T` that holds the respective tensor. 

Edges in the network represent contractions, of course. This code distinguishes two kinds of edges, using the key `trace`. Edges with `trace=True` corresponds to contraction of a tensor with itself, i.e. "tracing out of dimensions". The edges where `trace=False` are contraction that involve two nodes, these are "default nodes".

The edges also hold information about which legs of the adjacent tensors are involved in the contraction. Trace-edges possess an `indices`-key, whose value is a set `{i1,i2}` that holds the two legs of the tensor that the trace runs over. Default legs possess a `legs`-key, whose value is a dictionary `{node1:i2,node2:i2}`, that gives the leg for the respective node. The `legs`-value of trace-edges is `None`, and vice-versa.[^1]

[^1]: Having two different ways of saving the leg indices and two kinds of legs seems convoluted, but this is necessary because `network_tools.contract_edge` needs to know if a given edge is a trace edge or not. The `indices`-values are sets as opposed to tuples or lists because the trace is symmetric with respect to the order of the legs.

# The `contract_edge` function

This function - defined in `networks.py` - does the heavy lifting of tensor network contraction. Given an edge `(node1,node2,key)`, it executes the respective contraction in the tensor network `G` in-place. This procedure is comparably easy if the edge in question is a trace edge. we take the respective trace, and save the resulting tensor in the node. Afterwards, the incident edges' `legs` or `indices` values need to be updated to reflect the changed indices of the tensor.

It gets a little more convoluted if we are contracting a default edge, because there are more cases. Each node might have trace indices attached. Furthermore, there could be multiple edges between `node1` and `node2`, and upon contraction of one of them the other ones turn into trace edges. See the image below for different scenarios.

<p align="center">
  <img width="300" height="300" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/imgs/contraction_cases.jpeg">
</p>

Contracting the two tensors is easy, and afterwards, the edge `(node1,node2,key)` is removed. `contract_edges` then devotes a substantial amount of attention to

* re-labeling the legs that are incident to `node1`,
* re-labeling the legs that are incident to `node2`, and
* connecting any nodes to `node1` that were previously connected to `node2`.

Each step requires distinguishing whether the edge under consideration is a trace edge or not.

The function `contract_edge` is used in `contract_network`, which has been tested against test cases (`utils.dummynet1` through `utils.dummynet5`) and against the plaquette code (`networks.grid_net`).

# Belief propagation in `BP.py`

each edge `(node1,node2)` has a key `msg`, that contains the messages on this edge.

```
    edge["msg"] = {
        node1:message_to_node_1,
        node2:message_to_node_2
    }
```

The message `G[node1][node2][0]["msg"][node1]` is thus the message from `node2` to `node1`. It is a vector which connects to leg `G[node1][node2][0]["legs"][node1]` of the tensor `G.nodes[node1]["T"]`.

# Belief propagation in `loopyNBP.py`

The loopy Neighborhood Belief Propagation (loopyNBP) algorithm is inspired by Kirkley et Al, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)). It considers the effect of small loops by explicitly contracting them. This is achieved through neighborhoods; let $N_i^{(r)}$ be the neighborhood around node $i$. $N_i^{(r)}$ contains all loops (i.e. nodes and edges) of length $r+2$ and shorter. For $r=2$, this implies $N_i^{(2)}={i}$, in which case the algorithm reduces to normal Belief Propagation. For $r=1$, the neighborhoods contain loops with length 3, i.e. triangles, and so forth.

Such loops render the BP algorithm inaccurate. The loopyNBP incorporates loops with length $r+2$ by passing messages not between individual nodes, but between neighborhoods. When calculating a new message iteratively, a network receives inbound messages at it's border. These entire neighborhood is contracted explicitly with it's inbound neighborhoods, and one outbound message emerges.

The function `loopyNBP.neighborhood` returns the neighborhood $N_i^{(r)}$ as two sets: One contains the edges in the neighborhood, and the other one the nodes. During the message passing iterations, in this neighborhood, edges within the neighborhood are contracted while inbound messages are identified as messages to nodes in $N_i^{(r)}$, which live in edges that are not in $N_i^{(r)}$.