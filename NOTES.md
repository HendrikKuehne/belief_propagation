## ToDo

* Implement Belief Propagation algorithm from Kirkley, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211))
    * :white_check_mark: Expand the algorithm to work on arbitary graphs.
    * :white_check_mark: Implement `block_bp` for `nx.MultiGraph` grids. This necessitates code that merges parallel edges in a tensor network.
* :white_check_mark: Check if Bethe Free Energy is real if `psd=False` (eq. A12 in [Phys. Rev. Research 3, 023073 (2021)](https://doi.org/10.1103/PhysRevResearch.3.023073) ([arXiv:2008.04433](https://arxiv.org/abs/2008.04433))).
    * All data points with `psd=True` have (approximately) negative Bethe Free Energy!
* Improve contraction accuracy by treating short loops using Kirkley and long loops using Feynman Contraction.[^1]
    * :white_check_mark: Maybe contract small neighborhoods directly and use Feynman contraction to treat edges with large bond dimensions?
* Optimize exact contraction of tensor networks.
    * :white_check_mark: Exact contraction using a  `cotengra.HyperOptimizer` object
      * Implemented in `belief_proagation.sandwich_BP.braket.Braket.__contract_ctg_hyperopt()`, but the overhead is really big; I only use this when intermediate tensors might become too big.
    * :white_check_mark: Contraction using `np.einsum` and `np.einsum_path`.[^2]
    * :white_check_mark: Contraction using `cotengra.einsum` with objectives from [`cotengra.scoring`](https://cotengra.readthedocs.io/en/latest/autoapi/cotengra/scoring/index.html), or [`cotengra.array_contract`](https://cotengra.readthedocs.io/en/latest/autoapi/cotengra/index.html#cotengra.array_contract).
* Come up with a better way to construct neighborhoods; it seems like graphs created using `belief_propagation.graphs.short_loop_graph` still contain many short loops after `belief_propagation.loopyNBP.construct_neighborhoods` is used to contract neighborhoods.
    * Assume that we construct neighborhoods $N_a^{(r)}$, i.e. neighborhoods that contain loops up to length $r+2$. If the network contains loops that are only a little bit longer than $r+2$, say $r+2+\epsilon$, the neighborhood decomposition transforms these neighborhoods into loops of length $\epsilon$. The neighborhood decomposition (using the heuristic I have implemented) is only to be used if there is a gap in the loop length spectrum.
    * What I should do: Construct neighborhoods by moving outward from a root node; this is closer to what Kirkley et Al do.
* Documentation with [Sphinx documentation builder](https://docs.readthedocs.io/en/stable/intro/sphinx.html).[^3]
    * :white_check_mark: I should clean up my code anyways, a good guide might be the [PEP 8 style guide](https://peps.python.org/pep-0008/).
* Acceleration using GPUs and smarter implementation
    * Faster numerics with [Nvidia cuPyNumeric](https://developer.nvidia.com/cupynumeric)
        * Installation instructions [here](https://github.com/NVIDIA/accelerated-computing-hub/blob/main/Accelerated_Python_User_Guide/notebooks_v1/Chapter_11_Distributed_Computing_cuPyNumeric.ipynb)
        * I can't get it to use the GPUs on `sccs.homeone`; it seems to me like CUDA is not being installed correctly. The [CUDA toolkit downloads](https://developer.nvidia.com/cuda-downloads) require administrator privileges during the installation process, which I don't have. I tried [installing CUDA with conda](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#x86-64-conda) before installing cuPyNumeric, but that doesn't seem to work... I would think though that there is something wrong on my end, because [the `x-wing0` node does have NVIDIA GPUs](https://gitlab.lrz.de/tum-i05/public/home-one-cluster/-/blob/master/README.md?ref_type=heads#general-information).
    * Faster graph routines with [nx-cugraph](https://github.com/rapidsai/nx-cugraph)
        * Tutorial [here](https://developer.nvidia.com/blog/7-drop-in-replacements-to-instantly-speed-up-your-python-data-science-workflows/#scaling_graph_analytics_with_networkx)
    * Smarter implementation of the message update
        * Pancotti and Gray explain the implementation of the message update in Quimb in [arxiv:2306.15004](https://arxiv.org/abs/2306.15004). It is implemented [here](https://github.com/jcmgray/quimb/blob/main/quimb/tensor/belief_propagation/l2bp.py#L209), but it is difficult to understand since Quimb is very involved...
        * See also my notes [here](https://github.com/HendrikKuehne/belief_propagation/tree/main/belief_propagation#file-contents)
    * Port to GPU using PyTorch? [^7]
        * I could use PyTorch Geometric instead of NetworkX, but sparse diagonalization seems to be a problem... what about CuPy though.
    * Sparse linear algebra using [CuPy](https://cupy.dev)
        * A drop-in replacement for SciPy! It has a [sparse eigensolver](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.eigsh.html#cupyx.scipy.sparse.linalg.eigsh) and [linear operators](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.LinearOperator.html#cupyx.scipy.sparse.linalg.LinearOperator).
* Improve implementation of `Braket`, `PEPS`, `PEPO` and `DMRG` classes; see `README.md` in [`belief_propagation/`](https://github.com/HendrikKuehne/belief_propagation/tree/main/belief_propagation).

[^1]: Feynman contraction refers to contracting over an edgenot by summing over it and merging the tensors, but instead by inserting a resolution of the identity and summing over the different terms that arise. See [Huang et Al, 2022](https://arxiv.org/abs/2005.06787), Section three; and [Girolamo, 2023](https://mediatum.ub.tum.de/1747499).

[^2]: `np.einsum_path` cannot contract large networks (i.e. many edges) because the alphabet with which it creates it's equations is limited to 52 characters (lower- and uppercase letters). This seems a severe limitation to me, I don't understand why that's in there; `cotengra.einsum` does not have that limitation, so I'm using that instead (dated 30.09.2024).

[^3]: Refer to Christian's [pytenet](https://github.com/cmendl/pytenet/tree/master). The file [`pytenet/doc/conf.py`](https://github.com/cmendl/pytenet/blob/master/doc/conf.py) is especially relevant.

[^7]: Online ressources: [Performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) for PyTorch. How would this play with NetworkX? [NetworkX supports different backends](https://networkx.org/documentation/stable/tutorial.html#using-networkx-backends), among which is [nx-cugraph](https://github.com/rapidsai/nx-cugraph) (see above), but they don't natively interface with PyTorch. PyTorch-Geometric has graph routines, and it seems like a [`torch_geometric.Data`](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.data.Data.html) object represents a graph. One can even initialize it [from a NetworkX graph](https://pytorch-geometric.readthedocs.io/en/stable/modules/utils.html#torch_geometric.utils.from_networkx). But this would, as it seems, require much deeper modifications than I have time for now. Using SciPy would require CPU-synchronization, anyways - this is a little more subtle.

## Open questions

This will be updated continuously, as questions come to mind.

* Why does it work only if `psd=True` in `construct_network`?
    * Testing Christian's code on 11.9. without any modifications: rel. err. $\sim 10^{-3}$ for `psd=True`, and rel. err. $\mathcal{O}(1)$ if `psd=False`.
    * `psd=True` is necessary for tree tensor networks too (see testing with `lib.graph_creation.tree`).
    * Messages must not have positive entries; if `psd=True`, the algorithm works with messages that have negative entries (tested with messages generated from a normal distribution, and then normalized).
    * I suspected `psd=False` might simply introduce numerical inaccuracies (see [here](https://github.com/HendrikKuehne/belief_propagation/tree/main/doc#user-content-fn-2-8812249509624473e552f17db0b8f455)), but that is not what it does; see [here](https://github.com/HendrikKuehne/belief_propagation/tree/main/doc#the-effect-of-the-psd-option).
    * Does `psd=True` mimic the non-negativity of the factors of factor graphs, which seems to be required for the BP algorithm?
        * Not by default. Let $m_{x\rightarrow a}$ be messages incident to node $a$, where (for `psd=True`) the tensor $\text{tr}(T_aT_a^H)$ is located. It's local contraction value becomes $Z_a=\text{tr}\left(T_aT_a^*\prod_x m_{x\rightarrow a}\right)$. For arbitrary messages $\{m_{x\rightarrow a}\}$ the value $Z_a$ must not be positive; indeed, it is complex in general. If the messages are positive-semidefinite, however, it is clear that $Z_a\geq 0$.
        * :arrow_right: When `psd=True`, all messages are positive-semidefinite, meaning the factors will indeed all give non-negative results.
* How and why does the normalization of Kirkley et Al in `message_passing_step` work? Why is their BP algorithm exact on trees, although normalizations are incorporated?
    * :arrow_right: The function `message_passing_iteration` implements Kirkley's belief propagation for networks with loops (Kirkley, 2021: [Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)). It's messages correspond to marginal probabilities and, as such, need to be normalized; the normalization above is the one that this paper uses (see the discussion after Eq. 12).
        * This algorithm is significantly different from Message Passing on trees, which is exact. See [here](https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/README.md#belief-propagation-and-loopy-belief-propagation) for details.
    * :arrow_right: This normalization has the effect that one can associate with every node the value $Z$ and with every edge the inverse contraction value $1/Z$.[^4] Consider now that a tree with $N$ nodes has $N-1$ edges; this means that $\prod_i \text{cntr}_i = Z^NZ^{1-N}=Z$, i.e. the algorithm is exact.
    * Which part of the algorithm breaks down when moving from trees to loopy graphs? Kirkley et Al claim that the effect of long loops is negligible, how can this be understood in terms of messages and their normalization?
* Why are long loops negligible?
    * Many runs of the BP algorithm give exact results when only long loops are present, which is what Kirkley et Al claim in their paper; they do not give a source though.
        * Intuitive statistical explanation, where the error in loopy BP comes from and why it might be negligible for long loops in ch. 14.4.2 of [this book](https://doi.org/10.1093/acprof:oso/9780198570837.001.0001) (drafts available [online](https://web.stanford.edu/~montanar/RESEARCH/book.html) for free)
    * :arrow_right: Loops behave like vector iterations, which is not how Kirkleys algorithm works; it is in fact detrimental to the accuracy. Vector iterations require many iterations, however, and the longer the loop the more iterations one needs to reach vector iteration territory. Long loops will (probably - this is what I expect) introduce larger errors, when one does more iterations in the BP algorithm.
    * :arrow_right: Actually not! The eigenvalue spectra of long loops tend to feature one dominant eigenvalue, while all others are neglectable in magnitude. See [this section](https://github.com/HendrikKuehne/belief_propagation/tree/main/doc/plots#spectra-of-a-matrix-chain) for details.[^6]
* Why do we normalize by dividing by $\chi^{3/4}$ in `construct_network`?
* What does Christian mean when he refers to the second method of constracting the TN (`block_bp`) as "approximate contraction based on modified belief propagation"? That method is exact.
    * :arrow_right: This method is based on the "Block Belief Propagation" algorithm (Arad, 2023: [Phys. Rev. B 108, 125111 (2023)](https://doi.org/10.1103/PhysRevB.108.125111)), which is not exact in general.
    * :arrow_right: The relative error improves when `block_bp` is included in the plaquette routine; why is that the case? It is not because we are reducing the number of nodes (see [this section](https://github.com/HendrikKuehne/belief_propagation/tree/main/doc/plots#tn_vs_pq_3x3_baselinepdf)) - is it because we are able to model local interactions more faithfully if a large chunk of the network is contracted explicitly? That is the physical argument - in terms of graphs, we are treating many small loops exactly which could otherwise have introduced inaccuracies.
* Some iterations of the Belief Propagation algorithm take many orders of magnitude longer than others; do these still converge?
* What happens when we try Christian's idea of Orthogonal Belief Propagation?
    * After one iteration is finished and the messages are found, we attempt to find messages that are orthogonal to the previous ones.[^5] What is the result? Are we iteratively finding Schmidt bases of the edges? Is this related to the quasi-canonical form of PEPS networks that Arad (2021) introduces?
* Do different gauges have a (strong) effect on BPDMRG performance?
* What are BP Trapping sets?
    * I have seen the BP algorithm stagnate during imaginary time evolution, but in a very strange way: Messages oscillate s.t. the message epsilon stays constant. I have no idea where this might come from, but it seems like this behavior is not unheard of in the literature; [arXiv:2506.01779](https://arxiv.org/abs/2506.01779) talks about a thing called "trapping sets"
    * For the moment I'll simply check for this during `braket.BP`. If it happens, I'll initialize new messages and add damping to the BP iteration.
* How many BP fixed points are there?
    * There have been times when I thought there is only one
    * :arrow_right: There are multiple ones, though! Check e.g. [arxiv:2306.15004](https://arxiv.org/abs/2306.15004), beginning of section III
* What is the asymptotical complexity of the BP message update?
    * Compute it! The *n-mode unfolding* [[arXiv:2109.00626](https://arxiv.org/abs/2109.00626)] of tensors might be helpful here.
* How effective is QR-gauging?
    * I use a breadth-first search in [`belief_propagation.truncate_expand.QR_gauging`](https://github.com/HendrikKuehne/belief_propagation/blob/f40f9b761bc665018958d690a467f2e5b18ea266/belief_propagation/truncate_expand.py#L668), because this ensures that the edges that will be cut are far removed from the orthogonality center.[^9] This should make the approximate orthogonalization more accurate, and would (if the graph were actually a tree) cause the messages to evaluate to identity. In my code, this would translate to the local environment being (approximately)[^8] the identity! Is that what actually happens?

[^4]: In the code contained herein, only nodes contain values. The emphasis is here on *associate*; the $1/Z$ that we could associate with an edge is factorized, it's factors being distributed in the adjacent nodes.

[^5]: I can imagine this going two ways: Either we add projectors to the edges, always projecting out the part that is collinear to the previous messages; or we directly project out the previous messages from the tensors that are adjacent to that edge.

[^6]: This was, independently, also found by [Cao, Vontobel, 2017](10.1109/ITW.2017.8277985).

[^8]: Approximately, since - of course - my graphs are not actually trees.

[^9]: This is of course only a heuristic. I took inspiration from [Phys. Rev. X 14, 011009 (2024)](https://doi.org/10.1103/PhysRevX.14.011009) for this method, and the authors go into much more detail on the construction of gauging trees in appendix A.

## References

See my [Zotero](https://www.zotero.org/hendrikkuehne/collections/NKRUB7HK)