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

# `contraction_runtimes.pdf`

<p align="center">
  <img width="300" height="224" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/plots/contraction_runtimes.jpeg">
</p>

Data generated using

```
    G = short_loop_graph(20,3,.6)
    construct_network(G,5,real=False,psd=False)
```

`np.einsum` combined with `np.einsum_path` yields the best results. I have not yet (as of 26.09.2024) compared the full suite of Cotengra tools against Numpy (because I can't get it to run), but for now I'll use Numpy.

# `BP_vs_loopyNBP_vs_blockBP.pdf`

<p align="center">
  <img width="300" height="224" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/plots/BP_vs_loopyNBP_vs_blockBP.jpeg">
</p>

Data generated using

```
    tensors,G = networks.grid_net(chi=4,width=4,height=4,real=False,psd=True)
```

Blocking into a 3 by 3 block.

# Spectra of a matrix chain

Why does the BP algorithm give better results for longer loops, as opposed to shorter ones? Recall [this section](https://github.com/HendrikKuehne/belief_propagation/tree/main?tab=readme-ov-file#open-questions), where I argue that BP on loops can be understood as a vector iteration, that converges to the largest eigenvalue. There is, a priori, no reason to expect that the largest eigenvalue corresponds to the network value; what is happening?

Consider a toy model, where a loop is formed as the product

$$
    \prod_{i=1}^L A_i
$$

of random matrices $A_i$[^2]. The network contraction value becomes

$$
    \text{cntr}\{A_i\}=\text{tr}\prod_{i=1}^L A_i = \sum\lambda,
$$

where the $\lambda$ are the eigenvalues of the matrix chain. What do the eigenvalue spectra of the matrix chains look like? As it turns out, for increasing length $L$, the magnitudes of the eigenvalues increase! The spectra come to be dominated by one large eigenvalue, and all others vanish in magnitude. The largest eigenvalue will thus be more and more close to the network contrcation value.[^3]

Plots that show histograms og eifenvalue magnitudes and the flatness of the spectrum are shown below.

<p align="center">
  <img width="800" height="480" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/plots/matrix_chain_spectra_hist.jpeg">
</p>

<p align="center">
  <img width="800" height="480" src="https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/plots/matrix_chain_spectra_flatness.jpeg">
</p>

[^2]: Each matrix is drawn from a normal distribution (centered around zero, std. one), e.g. `(rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)`.

[^3]: [The plot](https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/plots/matrix_chain_spectra_flatness.jpeg) doesn't really support this well - maybe look at this again.

# Sparse representations of PEPOs

* `sparse_dense_eig.pdf`: Sparse representation of PEPO using `scipy.sparse.bsr_array`, diagonalization using `scipy.sparse.linalg.eigsh`
* `sparse_lanczos_eig.pdf`: Diagonalization using `pytenet.eigh_krylov`, with `numiter = 10 * 2**N` (number of qubits `N`).[^4]
* `sparse_format_comp.pdf`: Sparse representation of PEPOs using the scipy array formats that are indicated in the plot.
* `sparse_format_comp_2.pdf`: Omitting `scipy.sparse.bsr_array` and repeating the calculation from `sparse_format_comp.pdf`, while also investigating sparsity.

[^4]: This is the default number of iterations in `scipy.sparse.linalg.eigsh`.