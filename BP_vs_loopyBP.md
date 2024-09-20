# Belief Propagation and loopy Belief Propagation

Message Passing (MP) and loopy Belief Propagation (loopyBP) are two very different algorithms. Both iterate the equation

$$
    m_{a\rightarrow x} = \text{tr}\left(T_a\prod_{y\neq x} m_{y\rightarrow a}\right)
$$

on every node, where $\text{tr}$ denotes contraction.

The MP algorithm simply iterates this equation on every node of a tree. At convergence, the value of the complete contraction can be extracted form any node, since any node is a valid root. Contracting any node with the inflowing messages gives the network value, and contracting two messages on the same node gives the network value. This gives exact results on tree-shaped networks.

loopyBP adds a normalization step at two points wthin the algorithm:

* After one iteration step (i.e. applying the BP equation), each message is divided by the sum of it's elements.
* After convergence, on each edge, the messages are normalized s.t. their contraction is normalized to unity; on the edge $(a,b)$, we do

$$
    m_{a\leftrightarrow b}\rightarrow \frac{m_{a\leftrightarrow b}}{\sqrt{|m_{a\rightarrow b}m_{b\rightarrow a}|}}.
$$

The loopyBP algorithm is exact on trees, too. The approximation that it carries out is drastically different, however: MP directly implements the contraction of the network, s.t. each node holds the contraction value $\text{cntr}$. Let $\text{cntr}_a$ be the result of the contraction of a tensor with the surrounding nodes, then MP gives

$$
    \text{cntr} = \text{cntr}_x = \left(\prod_x\text{cntr}_x\right)^{1/L},
$$

where $L$ is the number of nodes. This works whether `psd=True` or `psd=False`.[^1] In loopyBP, the contraction value is the product of all node values:

[^1]: The value of `psd` has an influence on numerical accuracy, however. Whether `psd=True` or `psd=False`, the code in `lib/network_contraction` yields messages s.t. $\text{cntr}_a=\text{cntr}$ for every node $a$. Numerically verifying $\text{cntr} = \left(\prod_x\text{cntr}_x\right)^{1/L}$ only succeeds when `psd=True` (using the Numpy standard accuracy). If `psd=False`, the relative error is in $\mathcal{O}(10^{-1})$. Based on this, one could assume that the `psd` option in loopyBP only introduces numerical inaccuracies, too. This does not seem to be the case; loopyBP misses $\text{cntr}$ by one order of magnitude in real and imaginary part when `psd=False`, while relative errors in $\mathcal{O}(10^{-3})$ are achieved when `psd=True`.

$$
    \text{cntr} = \prod_x\text{cntr}_x
$$

Since loopyBP works differently, it also gives decent results on graphs with loops. MP cannot handle loops:

<p align="center">
  <img width="300" height="224" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/imgs/example2_short_loopy_graph.jpeg">
  <img width="300" height="224" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/imgs/example2_short_loopy_graph_eps.jpeg">
</p>

The presence of loops in the graph causes feedback loops in the passage of messages through the graph, s.t. their magnitudes diverge, as shown above for an example graph. The loopyBP algorithm is able to handle loops, because the approximation that it carries out is fundamentally different. Where MP tries to exactly contract the network, loopyBP approximates the marginals of each "stump" of a tensor and contracts the tensors with these, effectively neglecting all interaction between nodes.[^2]

<p align="center">
  <img width="630" height="200" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/imgs/loopyBP_approximation.jpeg">
</p>

[^2]: I like to phrase as an "inverse HOSVD", because every node becomes a core tensor that is surrounded by vectors. As opposed to the HOSVD, not the interior dimensions (i.e. singular values) are truncated but the outer dimensions. This truncation is extreme: All bonds between nodes become one-dimensional.