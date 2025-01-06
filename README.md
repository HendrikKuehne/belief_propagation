# Belief propagation for tensor network contraction

Forked on 11th of September from Mendl, so far (11.9.2024) just for initial exploration of the belief propagation algorithm.

## Contents

* `belief_propagation/`
    * `graphs.py` Creation of various graphs.
    * `BP.py` Belief propagation on graphs, i.e. on various geometries. Taken from Kirkley et Al, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)).
    * `loopyNBP` Belief propagation on graphs using neighbor regions. Inspired by Kirkley et Al, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)).
    * `plaquette.py` Code from Christian Mendl. Not to be modified in any substantial way, for reference.
    * `networks.py` Functions for network creation and handling.
    * `utils.py` Stuff that is useful here or there. `contract_edge` function for contracting an edge in a tensor network, sanity checks and test cases for `network_contraction.contract_network`.
    * `routines.py` All the different algorithms that are implemented herein, contained in single functions that accept a network and give back a contraction value.
* `doc/`
    * `plots/` Discussion of various plots that illustrate the behavior of the contents of this repo.

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

[^1]: Feynman contraction refers to contracting over an edgenot by summing over it and merging the tensors, but instead by inserting a resolution of the identity and summing over the different terms that arise. See [Huang et Al, 2022](https://arxiv.org/abs/2005.06787), Section three; and [Girolamo, 2023](https://mediatum.ub.tum.de/1747499).

[^2]: `np.einsum_path` cannot contract large networks (i.e. many edges) because the alphabet with which it creates it's equations is limited to 52 characters (lower- and uppercase letters). This seems a severe limitation to me, I don't understand why that's in there; `cotengra.einsum` does not have that limitation, so I'm usnig that instead (dated 30.09.2024).

[^3]: Refer to Christian's [pytenet](https://github.com/cmendl/pytenet/tree/master). The file [`pytenet/doc/conf.py`](https://github.com/cmendl/pytenet/blob/master/doc/conf.py) is especially relevant.

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
    * :arrow_right: Loops behave like vector iterations, which is not how Kirkleys algorithm works; it is in fact detrimental to the accuracy. Vector iterations require many iterations, however, and the longer the loop the more iterations one needs to reach vector iteration territory. Long loops will (probably - this is what I expect) introduce larger errors, when one does more iterations in the BP algorithm.
* Why do we normalize by dividing by $\chi^{3/4}$ in `construct_network`?
* What does Christian mean when he refers to the second method of constracting the TN (`block_bp`) as "approximate contraction based on modified belief propagation"? That method is exact.
    * :arrow_right: This method is based on the "Block Belief Propagation" algorithm (Arad, 2023: [Phys. Rev. B 108, 125111 (2023)](https://doi.org/10.1103/PhysRevB.108.125111)), which is not exact in general.
    * :arrow_right: The relative error improves when `block_bp` is included in the plaquette routine; why is that the case? It is not because we are reducing the number of nodes (see [this section](https://github.com/HendrikKuehne/belief_propagation/tree/main/doc/plots#tn_vs_pq_3x3_baselinepdf)) - is it because we are able to model local interactions more faithfully if a large chunk of the network is contracted explicitly? That is the physical argument - in terms of graphs, we are treating many small loops exactly which could otherwise have introduced inaccuracies.
* Some iterations of the Belief Propagation algorithm take many orders of magnitude longer than others; do these still converge?
* What happens when we try Christian's idea of Orthogonal Belief Propagation?
    * After one iteration is finished and the messages are found, we attempt to find messages that are orthogonal to the previous ones.[^5] What is the result? Are we iteratively finding Schmidt bases of the edges? Is this related to the quasi-canonical form of PEPS networks that Arad (2021) introduces?

[^4]: In the code contained herein, only nodes contain values. The emphasis is here on *associate*; the $1/Z$ that we could associate with an edge is factorized, it's factors being distributed in the adjacent nodes.

[^5]: I can imagine this going two ways: Either we add projectors to the edges, always projecting out the part that is collinear to the previous messages; or we directly project out the previous messages from the tensors that are adjacent to that edge.

[^6]: The data is not contained herein since it does not belong to the codebase, and it is too much anyways. It can, of course, be generated from this code however.

## References

* Tensor networks, many-body physics and variational methods, Belief Propagation
    * P. Hack, C. Mendl, A. Paler  
      Belief Propagation for general graphical models with loops  
      [arXiv:2411.04957](http://arxiv.org/abs/2411.04957) (Work in Progress)
    * R. Alkabetz, I. Arad  
      Tensor networks contraction and the belief propagation algorithm  
      [Phys. Rev. Research 3, 023073 (2021)](https://doi.org/10.1103/PhysRevResearch.3.023073) ([arXiv:2008.04433](https://arxiv.org/abs/2008.04433))
    * Alec Kirkley, George T. Cantwell, M. E. J. Newman  
      Belief propagation for networks with loops  
      [Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211) ([arXiv:2009.12246](https://arxiv.org/abs/2009.12246))
    * Chu Guo, Dario Poletti, Itai Arad  
      Block belief propagation algorithm for two-dimensional tensor networks  
      [Phys. Rev. B 108, 125111 (2023)](https://doi.org/10.1103/PhysRevB.108.125111) ([arXiv:2301.05844](https://arxiv.org/abs/2301.05844))
    * Yijia Wang, Yuwen Ebony Zhang, Feng Pan, Pan Zhang  
      Tensor network message passing  
      [Phys. Rev. Lett. 132, 117401 (2024)](https://doi.org/10.1103/PhysRevLett.132.117401) ([arXiv:2305.01874](https://arxiv.org/abs/2305.01874))
    * Subhayan Sahu, Brian Swingle  
      Efficient tensor network simulation of quantum many-body physics on sparse graphs  
      [arXiv:2206.04701](https://arxiv.org/abs/2206.04701)
    * Joseph Tindall, Matthew T. Fishman  
      Gauging tensor networks with belief propagation  
      [arXiv:2306.17837](https://arxiv.org/abs/2306.17837)
    * Nicola Pancotti, Johnnie Gray  
      One-step replica symmetry breaking in the language of tensor networks  
      [arxiv:2306.15004](https://arxiv.org/abs/2306.15004)
    * David Tellenbach  
      Canonicalization of Loop-free Tensor Networks  
      [MediaTUM](https://mediatum.ub.tum.de/1654468?style=full_text)
* Contraction of large tensor networks
    * Johnnie J., G. Kin-Lic Chan  
      Hyperoptimized Approximate Contraction of Tensor Networks with Arbitrary Geometry  
      [Phys. Rev. X 14, 011009 (24)](https://doi.org/10.1103/PhysRevX.14.011009) ([arXiv:2206.07044](https://arxiv.org/abs/2206.07044))
    * J. Gray, S. Kourtis  
      Hyper-optimized tensor network contraction  
      [Quantum 5, 410 (2021)](https://doi.org/10.22331/q-2021-03-15-410) ([arXiv:2002.01935](https://arxiv.org/abs/2002.01935))
* IBM kicked Ising experiment
  * Youngseok Kim, Andrew Eddins, Sajant Anand, Ken Xuan Wei, Ewout van den Berg, Sami Rosenblatt, Hasan Nayfeh, Yantao Wu, Michael Zaletel, Kristan Temme, Abhinav Kandala  
    Evidence for the utility of quantum computing before fault tolerance  
    [Nature 618, 500 - 505 (2023)](https://doi.org/10.1038/s41586-023-06096-3)
  * Joseph Tindall, Matthew Fishman, E. Miles Stoudenmire, Dries Sels  
    Efficient Tensor Network Simulation of IBM’s Eagle Kicked Ising Experiment  
    [PRX Quantum 5, 010308 (2024)](https://doi.org/10.1103/PRXQuantum.5.010308) ([arXiv:2306.14887](https://arxiv.org/abs/2306.14887))
  * Tomislav Begusic, Johnnie Gray, Garnet Kin-Lic Chan  
    Fast and converged classical simulations of evidence for the utility of quantum
computing before fault tolerance  
    [Sci. Adv. 10, eadk4321 (2024)](https://doi.org/10.1126/sciadv.adk4321) ([arXiv:2308.05077](https://arxiv.org/abs/2308.05077))