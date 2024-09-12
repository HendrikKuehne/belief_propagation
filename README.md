# Belief propagation for tensor network contraction

Forked on 11th of September from Mendl, so far (11.9.2024) just for initial exploration of the belief propagation algorithm.

## ToDo

* Expand the algorithm to work on arbitary graphs.

## Open questions

This will be updated continuously, as questions come to mind.

* How exactly does Christian's algorithm work? Which source does it resemble most closely, if any?
    * How & why does the normalization work?
* Why does it work only if `psd = True` in `construct_network`?
    * Testing Christian's code on 11.9. without any modifications: rel. err. $\sim 10^{-3}$ for `psd=True`, and rel. err. $\mathcal{O}(1)$ if `psd=False`.
* How and why does the normalization in `message_passing_step` work? We're simply dividing the message by the sum of it's elements; this is coming completely out of the blue for me.
* Why do we normalize by dividing by $\chi^{3/4}$ in `construct_network`?
* What does Christian mean when he refers to the second method of constracting the TN (`coarse_grain`) as "approximate contraction based on modified belief propagation"? That method is exact.
    * :arrow_right:

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
