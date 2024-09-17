# `tn_vs_pq.pdf`

<p align="center">
  <img width="800" height="480" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/plots/tn_vs_pq.jpeg">
</p>

Comparison of the `networkx`- and the plaquette-approach; 200 sample networks of size 4x4, with `psd=True`, `real=False` and bond dimension 4 contracted. The `networkx`-code and the plaquette code give equal results when `block_bp` is not used. When `block_bp` is used, the plaquette code performs one order of magnitude better.

# `loopy_graphs_bp.pdf`

<p align="center">
  <img width="800" height="480" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/plots/loopy_graphs_bp.jpeg">
</p>

The belief propagation algorithm on arbitary graphs with few short loops (lokally tree-like). `psd=True` and `real=False`, bond dimension 4. The relative error stays constant for a growing number of qubits.

# `loopy_graphs_bp.pdf`

<p align="center">
  <img width="800" height="480" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/plots/pq_without_block_bp.jpeg">
</p>

Belief propagation using the plaquette code, for different numbers of nodes[^1]. 100 samples each, bond dimension 4, `psd=True` and `real=False`. The relative error magnitude does not change when the number of nodes is increased.

[^1] `width,height in itertools.product(np.arange(2,6),repeat=2)`