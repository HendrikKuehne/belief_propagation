# Belief propagation for tensor network contraction

Forked on 11th of September from Mendl, so far (11.9.2024) just for initial exploration of the belief propagation algorithm.

## Contents

* `lib`
    * `graph_creation.py` Creation of various graphs.
    * `network_contraction.py` Belief propagation on graphs, i.e. on various geometries.
    * `plaquette_contraction.py` Code from Christian Mendl. Not to be modified in any substantial way, for reference.
    * `utils.py` Stuff that is useful here or there. `contract_edge` function for contracting an edge in a tensor network, sanity checks and test cases for `network_contraction.contract_network`.
* `doc`
    * `plots` Discussion of various plots that illustrate the behavior of the contents of this repo.

## ToDo

* :white_check_mark: Expand the algorithm to work on arbitary graphs.
    * :white_check_mark: Implement `block_bp` for `nx.MultiGraph` grids. This necessitates code that merges parallel edges in a tensor network.
* Check if Bethe Free Energy is real if `psd=False` (eq. A12 in [Phys. Rev. Research 3, 023073 (2021)](https://doi.org/10.1103/PhysRevResearch.3.023073) ([arXiv:2008.04433](https://arxiv.org/abs/2008.04433))).
    * This seems to be necessary for the Belief Propagation algorithm to work, but I would not be surprised if it doesn't hold if `psd=False`.

## Open questions

This will be updated continuously, as questions come to mind.

* Why does it work only if `psd=True` in `construct_network`?
    * Testing Christian's code on 11.9. without any modifications: rel. err. $\sim 10^{-3}$ for `psd=True`, and rel. err. $\mathcal{O}(1)$ if `psd=False`.
    * `psd=True` is necessary for tree tensor networks too (see testing with `lib.graph_creation.tree`).
    * Messages must not have positive entries; if `psd=True`, the algorithm works with messages that have negative entries (tested with messages generated from a normal distribution, and then normalized).
    * I suspected `psd=False` might simply introduce numerical inaccuracies (see [here](https://github.com/HendrikKuehne/belief_propagation/tree/main/doc#user-content-fn-2-8812249509624473e552f17db0b8f455)), but that is not what it does; see [here](https://github.com/HendrikKuehne/belief_propagation/tree/main/doc#the-effect-of-the-psd-option).
* How and why does the normalization in `message_passing_step` work? We're simply dividing the message by the sum of it's elements; this is coming completely out of the blue for me.
    * :arrow_right: The function `message_passing_iteration` implements Kirkley's belief propagation for networks with loops (Kirkley, 2021: [Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)). It's messages correspond to marginal probabilities and, as such, need to be normalized; the normalization above is the one that this paper uses (see the discussion after Eq. 12).
    * This algorithm is significantly different from Message Passing on trees, which is exact. See [here](https://github.com/HendrikKuehne/belief_propagation/blob/main/doc/README.md#belief-propagation-and-loopy-belief-propagation) for details.
* Why do we normalize by dividing by $\chi^{3/4}$ in `construct_network`?
* What does Christian mean when he refers to the second method of constracting the TN (`block_bp`) as "approximate contraction based on modified belief propagation"? That method is exact.
    * :arrow_right: This method is based on the "Block Belief Propagation" algorithm (Arad, 2023: [Phys. Rev. B 108, 125111 (2023)](https://doi.org/10.1103/PhysRevB.108.125111)), which is not exact in general.
    * The relative error improves when `block_bp` is included in the plaquette routine; why is that the case? It is not because we are reducing the number of nodes (see [this section](https://github.com/HendrikKuehne/belief_propagation/tree/main/doc/plots#tn_vs_pq_3x3_baselinepdf)) - is it because we are able to model local interactions more faithfully if a large chunk of the network is contracted explicitly?

## References

- R. Alkabetz, I. Arad  
  Tensor networks contraction and the belief propagation algorithm  
  [Phys. Rev. Research 3, 023073 (2021)](https://doi.org/10.1103/PhysRevResearch.3.023073) ([arXiv:2008.04433](https://arxiv.org/abs/2008.04433))
- Alec Kirkley, George T. Cantwell, M. E. J. Newman  
  Belief propagation for networks with loops  
  [Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211) ([arXiv:2009.12246](https://arxiv.org/abs/2009.12246))
- Chu Guo, Dario Poletti, Itai Arad  
  Block belief propagation algorithm for two-dimensional tensor networks  
  [Phys. Rev. B 108, 125111 (2023)](https://doi.org/10.1103/PhysRevB.108.125111) ([arXiv:2301.05844](https://arxiv.org/abs/2301.05844))
- Yijia Wang, Yuwen Ebony Zhang, Feng Pan, Pan Zhang  
  Tensor network message passing  
  [Phys. Rev. Lett. 132, 117401 (2024)](https://doi.org/10.1103/PhysRevLett.132.117401) ([arXiv:2305.01874](https://arxiv.org/abs/2305.01874))
