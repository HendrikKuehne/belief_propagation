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
    * I should clean up my code anyways, a good guide might be the [PEP 8 style guide](https://peps.python.org/pep-0008/).
* Acceleration using CUDA
    * Faster numerics with [Nvidia cuPyNumeric](https://developer.nvidia.com/cupynumeric)
        * Installation instructions [here](https://github.com/NVIDIA/accelerated-computing-hub/blob/main/Accelerated_Python_User_Guide/notebooks_v1/Chapter_11_Distributed_Computing_cuPyNumeric.ipynb)
        * I can't get it to use the GPUs on `sccs.homeone`; it seems to me like CUDA is not being installed correctly. The [CUDA toolkit downloads](https://developer.nvidia.com/cuda-downloads) require administrator privileges during the installation process, which I don't have. I tried [installing CUDA with conda](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#x86-64-conda) before installing cuPyNumeric, but that doesn't seem to work... I would think though that there is something wrong on my end, because [the `x-wing0` node does have NVIDIA GPUs](https://gitlab.lrz.de/tum-i05/public/home-one-cluster/-/blob/master/README.md?ref_type=heads#general-information).
    * Faster graph routines with [nx-cugraph](https://github.com/rapidsai/nx-cugraph)
        * Tutorial [here](https://developer.nvidia.com/blog/7-drop-in-replacements-to-instantly-speed-up-your-python-data-science-workflows/#scaling_graph_analytics_with_networkx)
* Improve implementation of `Braket`, `PEPS`, `PEPO` and `DMRG` classes; see `README.md` in [`belief_propagation/`](https://github.com/HendrikKuehne/belief_propagation/tree/main/belief_propagation).

[^1]: Feynman contraction refers to contracting over an edgenot by summing over it and merging the tensors, but instead by inserting a resolution of the identity and summing over the different terms that arise. See [Huang et Al, 2022](https://arxiv.org/abs/2005.06787), Section three; and [Girolamo, 2023](https://mediatum.ub.tum.de/1747499).

[^2]: `np.einsum_path` cannot contract large networks (i.e. many edges) because the alphabet with which it creates it's equations is limited to 52 characters (lower- and uppercase letters). This seems a severe limitation to me, I don't understand why that's in there; `cotengra.einsum` does not have that limitation, so I'm using that instead (dated 30.09.2024).

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
    * :arrow_right: Actually not! The eigenvalue spectra of long loops tend to feature one dominant eigenvalue, while all others are neglectable in magnitude. See [this section](https://github.com/HendrikKuehne/belief_propagation/tree/main/doc/plots#spectra-of-a-matrix-chain) for details.[^6]
* Why do we normalize by dividing by $\chi^{3/4}$ in `construct_network`?
* What does Christian mean when he refers to the second method of constracting the TN (`block_bp`) as "approximate contraction based on modified belief propagation"? That method is exact.
    * :arrow_right: This method is based on the "Block Belief Propagation" algorithm (Arad, 2023: [Phys. Rev. B 108, 125111 (2023)](https://doi.org/10.1103/PhysRevB.108.125111)), which is not exact in general.
    * :arrow_right: The relative error improves when `block_bp` is included in the plaquette routine; why is that the case? It is not because we are reducing the number of nodes (see [this section](https://github.com/HendrikKuehne/belief_propagation/tree/main/doc/plots#tn_vs_pq_3x3_baselinepdf)) - is it because we are able to model local interactions more faithfully if a large chunk of the network is contracted explicitly? That is the physical argument - in terms of graphs, we are treating many small loops exactly which could otherwise have introduced inaccuracies.
* Some iterations of the Belief Propagation algorithm take many orders of magnitude longer than others; do these still converge?
* What happens when we try Christian's idea of Orthogonal Belief Propagation?
    * After one iteration is finished and the messages are found, we attempt to find messages that are orthogonal to the previous ones.[^5] What is the result? Are we iteratively finding Schmidt bases of the edges? Is this related to the quasi-canonical form of PEPS networks that Arad (2021) introduces?
* Do different gauges have a (strong) effect on BPDMRG performance?
* BP Trapping sets
    * I have seen the BP algorithm stagnate during imaginary time evolution, but in a very strange way: Messages oscillate s.t. the message epsilon stays constant. I have no idea where this might come from, but it seems like this behavior is not unheard of in the literature; [arXiv:2506.01779](https://arxiv.org/abs/2506.01779) talks about a thing called "trapping sets"
    * For the moment I'll simply check for this during `braket.BP`. If it happens, I'll initialize new messages and add damping to the BP iteration.

[^4]: In the code contained herein, only nodes contain values. The emphasis is here on *associate*; the $1/Z$ that we could associate with an edge is factorized, it's factors being distributed in the adjacent nodes.

[^5]: I can imagine this going two ways: Either we add projectors to the edges, always projecting out the part that is collinear to the previous messages; or we directly project out the previous messages from the tensors that are adjacent to that edge.

[^6]: This was, independently, also found by [Cao, Vontobel, 2017](10.1109/ITW.2017.8277985).

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
    * Christian B. Mendl  
      PyTeNet: A concise Python implementation of quantum tensor network algorithms
      [Journal of Open Source Software, 3(30), 948](https://doi.org/10.21105/joss.00948) ([github.com/cmendl/pytenet](https://github.com/cmendl/pytenet))
    * Richard M. Milbradt and Qunsheng Huang and Christian B. Mendl  
      State diagrams to determine tree tensor network operators  
      [SciPost Phys. Core 7, 036 (2024)](https://doi.org/10.21468/SciPostPhysCore.7.2.036) ([arXiv:2311.13433](https://arxiv.org/abs/2311.13433))
    * David Tellenbach  
      Canonicalization of Loop-free Tensor Networks  
      [MediaTUM](https://mediatum.ub.tum.de/node?id=1654468)
    * Michael X. Cao, Pascal O. Vontobel  
      Double-Edge Factor Graphs: Definition, Properties and Examples  
      [2017 IEEE Information Theory Workshop (ITW)](https://doi.org/10.1109/ITW.2017.8277985)
    * Jonathan S. Yedidia, William T. Freeman, Yair Weiss  
      Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms  
      [IEEE Trans. Inf. Theory, vol. 51, no. 7, pp. 2282–2312, Jul. 2005.](https://doi.org/10.1109/TIT.2005.850085)
    * Glen Evenbly, Nicola Pancotti, Ashley Milsted, Johnnie Gray, Garnet Kin-Lic Chan  
      Loop Series Expansions for Tensor Networks  
      [arXiv:2409.03108](https://arxiv.org/abs/2409.03108)
        * Giorgio Parisi, František Slanina  
          Loop expansion around the Bethe–Peierls approximation for lattice models[^1]  
          [J. Stat. Mech. (2006) L02003](https://doi.org/10.1088/1742-5468/2006/02/L02003) ([arXiv:cond-mat/0512529](https://arxiv.org/abs/cond-mat/0512529))
    * Joris M. Mooij; Hilbert J. Kappen  
      Sufficient Conditions for Convergence of the Sum–Product Algorithm  
      [IEEE Transactions on Information Theory vol. 53, no. 12, 2007](https://doi.org/10.1109/TIT.2007.909166)
        * Sufficient condition based on the question whether the BP iteration represents a contraction in the space of messages
    * Christian Knoll, Dhagash Mehta, Tianran Chen, Franz Pernkopf  
      Fixed Points of Belief Propagation — An Analysis via Polynomial Homotopy Continuation  
      [IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 9, pp. 2124-2136, Sep. 2018](https://doi.org/10.1109/TPAMI.2017.2749575)
    * Christian Knoll, Franz Pernkopf  
      On Loopy Belief Propagation – Local Stability Analysis for Non-Vanishing Fields
      [Uncertainty in Artificial Intelligence Proceedings](https://auai.org/uai2017/proceedings/papers/160.pdf)
        * Exploration of the effect of an external field on BP on the TFI
    * Kevin Murphy, Yair Weiss, Michael I. Jordan  
      Loopy Belief Propagation for Approximate Inference: An Empirical Study  
      [UAI 1999 Proceedings 467 - 475](https://doi.org/10.48550/arXiv.1301.6725)
        * Dampening ("Momentum term") in BP
    * Yinchuan Li, Guangchen Lan, Xiaodong Wang  
      Tensor Generalized Approximate Message Passing  
      [arXiv:2504:00008](https://arxiv.org/abs/2504.00008)
        * Tensor ring decomposition using Belief Propagation
    * H. C. Jiang, Z. Y. Weng, T. Xiang  
      Accurate determination of tensor network state of quantum lattice models in two dimensions  
      [Phys. Rev. Lett. 101, 090603 (2008)](https://doi.org/10.1103/PhysRevLett.101.090603) ([arXiv:0806.3719](https://arxiv.org/abs/0806.3719))
    * Matthew Leifer, David Poulin  
      Quantum Graphical Models and Belief Propagation  
      [Ann. Phys. 323 1899 (2008)](https://doi.org/10.1016/j.aop.2007.10.001) ([arXiv:0708.1337](https://arxiv.org/abs/0708.1337))
    * Tristan Müller, Thomas Alexander, Michael E. Beverland, Markus Bühler, Blake R. Johnson, Thilo Maurer, Drew Vandeth  
      Improved belief propagation is sufficient for real-time decoding of quantum memory  
      [arXiv:2506.01779](https://arxiv.org/abs/2506.01779)
    * D. A. Millar, L. W. Anderson, E. Altamura, O. Wallis, M. E. Sahin, J. Crain, S. J. Thomson  
      Imaginary Time Spectral Transforms for Excited State Preparation  
      [arXiv:2508.00065](https://arxiv.org/abs/2508.00065)
* Contraction of large tensor networks
    * Johnnie J., G. Kin-Lic Chan  
      Hyperoptimized Approximate Contraction of Tensor Networks with Arbitrary Geometry  
      [Phys. Rev. X 14, 011009 (24)](https://doi.org/10.1103/PhysRevX.14.011009) ([arXiv:2206.07044](https://arxiv.org/abs/2206.07044))
    * J. Gray, S. Kourtis  
      Hyper-optimized tensor network contraction  
      [Quantum 5, 410 (2021)](https://doi.org/10.22331/q-2021-03-15-410) ([arXiv:2002.01935](https://arxiv.org/abs/2002.01935))
    * Johnnie Gray, Garnet Kin-Lic Chan  
      Hyper-optimized approximate contraction of tensor networks with arbitrary geometry  
      [Phys. Rev. X 14, 011009](https://doi.org/10.1103/PhysRevX.14.011009) ([arXiv:2206.07044](https://arxiv.org/abs/2206.07044))
* IBM kicked Ising experiment & TFI more generally
  * Youngseok Kim, Andrew Eddins, Sajant Anand, Ken Xuan Wei, Ewout van den Berg, Sami Rosenblatt, Hasan Nayfeh, Yantao Wu, Michael Zaletel, Kristan Temme, Abhinav Kandala  
    Evidence for the utility of quantum computing before fault tolerance  
    [Nature 618, 500 - 505 (2023)](https://doi.org/10.1038/s41586-023-06096-3)
  * Joseph Tindall, Matthew Fishman, E. Miles Stoudenmire, Dries Sels  
    Efficient Tensor Network Simulation of IBM’s Eagle Kicked Ising Experiment  
    [PRX Quantum 5, 010308 (2024)](https://doi.org/10.1103/PRXQuantum.5.010308) ([arXiv:2306.14887](https://arxiv.org/abs/2306.14887))
  * Tomislav Begusic, Johnnie Gray, Garnet Kin-Lic Chan  
    Fast and converged classical simulations of evidence for the utility of quantum computing before fault tolerance  
    [Sci. Adv. 10, eadk4321 (2024)](https://doi.org/10.1126/sciadv.adk4321) ([arXiv:2308.05077](https://arxiv.org/abs/2308.05077))
  * Geoffrey R. Grimmett, Tobias J. Osborne & Petra F. Scudo  
    Bounded Entanglement Entropy in the Quantum Ising Model  
    [J Stat Phys 178, 281–296 (2020)](https://doi.org/10.1007/s10955-019-02432-y) ([arXiv:1906.11954](https://arxiv.org/abs/1906.11954))
* Mathematics / Numerics
  * A. N. Tikhonov , A. V. Goncharsky , V. V. Stepanov , A. G. Yagola  
    Numerical Methods for the Solution of Ill-Posed Problems  
    Springer Dordrecht, 2013
  * Benyamin Ghojogh, Fakhri Karray, Mark Crowley  
    Eigenvalue and Generalized Eigenvalue Problems: Tutorial  
    [arXiv:1903.11240](https://arxiv.org/abs/1903.11240v3)
  * Silvia Martina  
    Spectra of Random Matrices  
    [hosted by Università degli Studi di Padova](https://thesis.unipd.it/retrieve/b409c829-d57d-491c-9c0f-7d5f21f6a756/Tesi_LM_Martina_Silvia.pdf)
  * Masuo Suzuki  
    General theory of fractal path integrals with applications to many‐body theories and statistical physics  
    [J. Math. Phys. 32, 400–407 (1991)](https://doi.org/10.1063/1.529425)
  * Andrew M. Childs, Yuan Su, Minh C. Tran, Nathan Wiebe, Shuchen Zhu  
    Theory of Trotter Error with Commutator Scaling  
    [Phys. Rev. X 11, 011020 (2021)](https://doi.org/10.1103/PhysRevX.11.011020) ([arXiv:1912.08854](https://arxiv.org/abs/1912.08854))
    * Explicit form of n-th order Trotteriation

[^1]: Could this be related to Gray's Loop Series Expansion?
