# `loopy_graphs_bp.pdf`

<p align="center">
  <img width="800" height="480" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/plots/loopy_graphs_bp.jpeg">
</p>

The belief propagation algorithm on arbitary graphs with few short loops (lokally tree-like). `psd=True` and `real=False`, bond dimension 4. The relative error stays constant for a growing number of qubits.

# `pq_without_block_bp.pdf`

<p align="center">
  <img width="800" height="480" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/plots/pq_without_block_bp.jpeg">
</p>

Belief propagation using the plaquette code, for different numbers of nodes[^1]. 100 samples each, bond dimension 4, `psd=True` and `real=False`. The relative error magnitude does not change when the number of nodes is increased.

[^1]: `width,height in itertools.product(np.arange(2,6),repeat=2)`

# `tn_vs_pq_3x3_baseline.pdf`

<p align="center">
  <img width="800" height="480" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/plots/tn_vs_pq_3x3_baseline.jpeg">
</p>

Comparing the plaquette code and the tensor network code. For `nNodes != 16`, values for tensor networks with and without block belief propagation, and values for plaquettes without block belief propagation are displayed. Red `tn`-markers are always embedded within blue `pq`-markers, meaning both codes give the same results.

For `nNodes == 16`, both `pq` and `tn` are available with and without belief propagation. Block belief propagation achieves better results on average. Furthermore, grey lines show the best results from calculations where `psd=False`, and the worst results where `psd=True`. It becomes clear that `psd=True` is necessary for the belief propagation algorithm to work; there are sometimes many orders of magnitudes between the results and critically, the relative error is $\mathcal{O}(1)$ when `psd=False`.