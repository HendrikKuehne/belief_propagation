# Message Passing and loopy Belief Propagation

Message Passing (MP) and Belief Propagation (BP) are two very different algorithms. Both iterate the equation

$$
    m_{a\rightarrow x} = \text{tr}\left(T_a\prod_{x\neq y\in N(a)} m_{y\rightarrow a}\right)
$$

on every node, where $\text{tr}$ denotes contraction and $N(a)$ is the neighborhood of $a$, i.e. it's adjacent nodes.

The MP algorithm simply iterates this equation on every node of a tree. At convergence, the value of the complete contraction can be extracted form any node, since any node is a valid root. Contracting any node with the inflowing messages gives the network value, and contracting two messages on the same node gives the network value. This gives exact results on tree-shaped networks.

BP adds a normalization step at two points in the algorithm:

* After one iteration step (i.e. applying the BP equation), each message is divided by the sum of it's elements.
* After convergence, on each edge, the messages are normalized s.t. their contraction is normalized to unity;[^1] on the edge $(a,b)$, we do

$$
    m_{a\leftrightarrow b}\rightarrow \frac{m_{a\leftrightarrow b}}{\sqrt{|m_{a\rightarrow b}m_{b\rightarrow a}|}}.
$$

[^1]: This can be interpreted statistically. When interpreting the contraction of a PEPS as Belief Propagation on a factor graph, the graph in question becomes a "double edge factor graph". The local marginals of it's variables can be calculated using eq. A3 of [Phys. Rev. Research 3, 023073 (2021)](https://doi.org/10.1103/PhysRevResearch.3.023073). By normalizing the messages in this particular way, we are normalizing the marginal.

The BP algorithm is exact on trees, too. The approximation that it carries out is drastically different, however: MP directly implements the contraction of the network, s.t. each node holds the contraction value $Z$. Let $\text{cntr}_a$ be the result of the contraction of tensor $a$ with it's incident messages, then MP gives

$$
    Z = \text{cntr}_x = \left(\prod_a\text{cntr}_a\right)^{1/L}
$$

for all nodes $x$, where $L$ is the number of nodes. This works whether `psd=True` or `psd=False`.[^2] In BP, the contraction value is the product of all node values:

[^2]: The value of `psd` has an influence on numerical accuracy, however. Whether `psd=True` or `psd=False`, the code in `belief_propagation/BP` yields messages s.t. $\text{cntr}_a=Z$ for every node $a$. Numerically verifying $Z = \left(\prod_x\text{cntr}_x\right)^{1/L}$ only succeeds when `psd=True` (using Numpy standard accuracy). If `psd=False`, the relative error is in $\mathcal{O}(10^{-1})$. Based on this, one could assume that the `psd` option in BP only introduces numerical inaccuracies, too. This does not seem to be the case; BP misses $Z$ by one order of magnitude in real and imaginary part when `psd=False`, while relative errors in $\mathcal{O}(10^{-3})$ are achieved when `psd=True`.

$$
    Z = \prod_x\text{cntr}_x
$$

This is a direct consequence of the normalization of the messages, both during iterations and before contracting them with the tensors. The normalization steps are such that we can associate with every node the value $Z$ and with every edge the value $1/Z$. Now since a tree with $N$ nodes has $N-1$ edges, we find

$$
    \prod_x\text{cntr}_x = Z^NZ^{1-N} = Z,
$$

i.e. BP is exact on trees. Note that the association of values to edges is not explicit in the algorithm. In actuality, each value $1/Z$ is broken up into factors that are contained in the nodes.

Since BP works differently, it also gives decent results on graphs with loops. MP cannot handle loops:

<p align="center">
  <img width="300" height="224" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/imgs/example2_short_loopy_graph.jpeg">
  <img width="300" height="224" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/imgs/example2_short_loopy_graph_eps.jpeg">
</p>

The presence of loops in the graph causes feedback loops in the passage of messages through the graph, s.t. their magnitudes diverge, as shown above for an example graph. The BP algorithm is able to handle loops, because the messages are normalized in each step, and the normalization is then accounted for by multiplying all nodes.[^3]

<p align="center">
  <img width="630" height="200" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/imgs/BP_approximation.jpeg">
</p>

[^3]: Note that, where MP tries to exactly contract the network, BP approximates the marginals of each "stump" of a tensor and contracts the tensors with these, effectively neglecting all interaction between nodes. I like to phrase this as an "inverse HOSVD", because every node becomes a core tensor that is surrounded by vectors. As opposed to the HOSVD, not the interior dimensions (i.e. singular values) are truncated but the outer dimensions. This truncation is extreme: All bonds between nodes become one-dimensional.

# The effect of the `psd` option

<p align="center">
  <img width="556" height="200" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/imgs/double_edge_factor_graph.png">
</p>

Arad (2021) ([Phys. Rev. Research 3, 023073 (2021)](https://doi.org/10.1103/PhysRevResearch.3.023073)) investigates so-called "double-edge factor graphs" (DEFG), which are constructed from a PEPS by stacking it and it's complex conjugate according to the picture above. This is of course what the option `psd=True` in this code accomplishes. They emphasize that the messages $m_{a\rightarrow b}(x_1,x_1')$, when interpreted as matrices, are positive semi-definite.

Recall also the mapping from a PEPS to a factor graph in the paper mentioned above. As explained [here](https://github.com/HendrikKuehne/belief_propagation/tree/main/doc#user-content-fn-1-fd873affe81f52cd3ae93dbdff2abae5), the normalization of messages in the final step of the BP algorithm can be seen as the normalization of the marginals of the variables of the DEFG.

The two properties that (I) the messages are positive semi-definite and that (II) the marginals of the variables are normalized are both affected by the `psd`-option, but not in the same way. Both are not true anymore when `psd=False`, yet (II) can be saved rather straightforwardly. If `psd=False`, the marginals are normalized not to unity but to a unit complex number, which prevents the statistical interpretation of the BP algorithm. By normalizing the messages according to

$$
    m_{a\leftrightarrow b}\rightarrow \frac{m_{a\leftrightarrow b}}{\sqrt{m_{a\rightarrow b}m_{b\rightarrow a}}}
$$

(removing the absolute value in the denominator), the marginals are once again normalized to unity. This does not make the algorithm work, however: (I) is also not true if `psd=False`.[^4] This behavior does not depend on the initialization of the messages.[^5] Arad et Al interpret the result of the BP algorithm as "the density matrix of some fictitious quantum state that 'lives on the edges of the PEPS'" (section A.2). This interpretation is lost if (I) does not hold anymore.

[^4]: It is, of course, fairly self-explanatory that (I) does not necessarily hold if the network does not originate from a PEPS.

[^5]: Tested using initialization of the messages using a normal distribution centered aroind zero.