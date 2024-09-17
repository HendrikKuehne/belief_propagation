# `tn_vs_pq.pdf`

<p align="center">
  <img width="400" height="240" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/plots/tn_vs_pq.pdf">
</p>

Comparison of the `networkx`- and te the plaquette-approach; 200 sample networks of size 4x4, with `psd=True` and bond dimension 4 contracted. The `networkx`-code and the plaquette code give equal results when `block_bp` is not used. When `block_bp` is used, the plaquette code performs one order of magnitude better.

# `loopy_graphs_bp.pdf`

<p align="center">
  <img width="400" height="240" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/plots/loopy_graphs_bp.pdf">
</p>

The belief propagation algorithm on arbitary graphs with few short loops (lokally tree-like). The relative error stays constant for a growing number of qubits.