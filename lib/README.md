# The structure of the tensor networks

Tensor networks are realised using the `networkx.MultiGraph` class.

Each node in the graph has a key `T` that holds the respective tensor. 

Edges between nodes represent contractions, of course. Which legs of the tensors are involved in which contraction is defined by the `legs` key that every edge possesses. Consider the edge `(u,v,i)`[^1]; it is equipped with the key-value pair `"legs":{u:r,v:l}`. The contraction that this edge represents involves leg `r` of `u`'s tensor and leg `l` of `v`'s tensor.

[^1]: What is `i`, an edge only connects to two nodes? Recall that the network is a `networkx.MultiGraph` object; `i` is the key that distinguishes between many edges that connect the same two nodes.

# The `contract_edge` function

![alt text](https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/imgs/contraction_cases.jpeg?raw=true)