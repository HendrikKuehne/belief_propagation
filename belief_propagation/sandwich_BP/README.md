# Open questions

* **Stoudenmire vs Gray:** Kim et Al claimed utility of quantum computing before fault tolerance in 2023 ([Nature 618, 500 - 505 (2023)](https://doi.org/10.1038/s41586-023-06096-3)). Both Stoudenmire et Al ([PRX Quantum 5, 010308 (2024)](https://doi.org/10.1103/PRXQuantum.5.010308)) and Gray et Al ([Sci. Adv. 10, eadk4321 (2024)](https://doi.org/10.1126/sciadv.adk4321)) published results, where they simulate this experiment using BP on tensor networks - so what's the difference?
  * Both use the same gauge[^1]; Stoudenmire calls it the "Vidal gauge", while Gray calls it the "super-orthogonal gauge".
  * :arrow_right: Gray uses mixed time evolution, where they evolve both the PEPS and the PEPO "towards each other" to limit entanglement creation; Stoudenmire stays in the Schrödinger picture.
  * :arrow_right: Gray compresses each layer of the trotterized time evolution into a single PEPO, that he applies using what he calls "L2BP". Stoudenmire applies gates individually using the Simple-Update algorithm.

[^1]: Consult Tindall 2023 ([arXiv:2306.17837](https://arxiv.org/abs/2306.17837)) to see this; this gauge can be found using Belief Propagation.

# File contents

* **`PEPO.py`** Projector-entangled Pair Operators on arbitary graphs, where the Tensor Network structure is inherited from the main module (see [this file](https://github.com/HendrikKuehne/belief_propagation/blob/main/belief_propagation/README.md) for an introduction).
  * **ToDo**: Sparse matrices? Scipy only allows for two-dimensional sparse arrays, but the [sparse package](https://sparse.pydata.org/en/stable/) implements higher-dimensional sparse arrays.
    * Sparse matrices would be most useful in `PEPO.to_dense()`. There is hope that this could be done more or less easily, since the `sparse` package has an [einsum implementation](https://sparse.pydata.org/en/stable/generated/sparse.einsum.html#sparse.einsum) and a [reshape](https://sparse.pydata.org/en/stable/generated/sparse.reshape.html#sparse.reshape).
    * The catch is that their `einsum` has the same limitation that `np.einsum` has: The number of contractions is severely limited. Specifically, `sparse.einsum` chooses edge labels in a contraction from `np.core.einsumfunc.einsum_symbols` (check [`sparse._common.parse_einsum_input`](https://github.com/pydata/sparse/blob/main/sparse/numba_backend/_common.py#L1163)). For now I'll simply throw an error if I encounter a graph with too many edges; in the future, Feynman cuts could be used.
    * `sparse.einsum` does not seem to do any optimization - let's hope this does not become a problem.
  * **ToDo**: `PEPS` instances do not require a leg ordering on the underlying graph, but `PEPO` instances do; there is no reason why PEPOs should require a leg ordering. (except if a typical workflow is to generate a PEPS first and use it's graph to generate a PEPO; the leg ordering in PEPO intialisation should be optional)
  * **ToDo**: Overhaul PEPO initialisation. The current method defines site tensors without site-to-site coupling, then reshapes them such that the leg ordering is correct with respect to the graph. Site-to-site coupling is added afterwards. This, then, is very illegibile since I need to keep track of the leg ordering and since case distinctions are necessary. This could be done more elegantly by defining a tree along which coupling flows[^2]. The goal would be to define the site tensors without having to refer to the leg ordering of the graph, and re-shape afterwards.
  * **ToDo** Complete implementation of `PEPO.__add__`.
    * **ToDo** This necessitates handling the tree traversal; so far (5th of February), I had to disable the tree traversal tests in `PEPO.intact`, since the way I implemented summation of PEPOs is not compatible with the check I had so far. I don't think this would be hard to implement, it just requires some bookkeeping.
  * Implementation works; tested using `dummynet1`. Explicit construction of the Hamiltonian and `PEPO.to_dense()` yield the same eigenvalues. Tested against Christian's [pytenet](https://github.com/cmendl/pytenet/tree/master).
* **`PEPS.py`** PEPS on arbitrary graphs.
  * **ToDo** Smarter initialization of bond dimensions on loopy geometries. What I have so far prevents bond dimension bottlenecks, and is exact on edges that are not part of loops.
* **`braket.py`** Stacks of combinations of PEPS and PEPO on arbitrary graphs.
  * **ToDo** Accelerate the mesage uodate somehow
    * Sparse matrices? Scipy only allows for two-dimensional sparse arrays, but the [sparse package](https://sparse.pydata.org/en/stable/) implements higher-dimensional sparse arrays.
    * Pancotti & Gray stack all the tensors and the messages s.t. the BP algorithm becomes a vector iteration ([arxiv:2306.15004](https://arxiv.org/abs/2306.15004))
    * :arrows_counterclockwise: parallelize using [Ray](https://docs.ray.io/en/latest/ray-overview/getting-started.html) - What I have done so far is actually slower than the straightforward implementation. The overhead seems to be too large, maybe I should think this through a little more thoroughly.
  * :white_check_mark: Progress bar using [tqdm](https://tqdm.github.io)
  * **ToDo** Smarter discrimination between cases in `Braket.contract`.
  * **ToDo** Optimize local updates; Lanczos algorithm? (In python implemented, for example, in the [`pylanczos` package](https://pypi.org/project/pylanczos/))
  * **ToDo** The implementation of the DMRG class is wildly inefficient (at least in terms of memory) because the overlap $\braket{\psi|\psi}$ and the expectation value $\braket{\psi|H|\psi}$ are both stored as full `Braket` objects, although they contain essentially the same data
  * **ToDo** Normalize states after DMRG algorithm

[^2]: This other tree should be called `coupling_tree`, in contrast to the existing tree (`automaton_tree`, in the following). The automaton tree is contained in the coupling tree. This also necessitates a re-interpretation of inbound, passive and outbound legs: Inbound legs are upstream in the automaton tree, outbond legs are downstream in the coupling tree. Passive legs are upstream in the coupling tree, but are not contained in the automaton tree. Coupling goes out of every node along all outbound edges. This does not lead to double coupling along some edges because the coupling tree is directed; coupling flows downstream in the coupling tree.
