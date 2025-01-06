# Open questions

* **Stoudenmire vs Gray:** Kim et Al claimed utility of quantum computing before fault tolerance in 2023 ([Nature 618, 500 - 505 (2023)](https://doi.org/10.1038/s41586-023-06096-3)). Both Stoudenmire et Al ([PRX Quantum 5, 010308 (2024)](https://doi.org/10.1103/PRXQuantum.5.010308)) and Gray et Al ([Sci. Adv. 10, eadk4321 (2024)](https://doi.org/10.1126/sciadv.adk4321)) published results, where they simulate this experiment using BP on tensor networks - so what's the difference?
  * Both use the same gauge[^1]; Stoudenmire calls it the "Vidal gauge", while Gray calls it the "super-orthogonal gauge".
  * :arrow_right: Gray uses mixed time evolution, where they evolve both the PEPS and the PEPO "towards each other" to limit entanglement creation; Stoudenmire stays in the Schrödinger picture.
  * :arrow_right: Gray compresses each layer of the trotterized time evolution into a single PEPO, that he applies using what he calls "L2BP". Stoudenmire applies gates individually using the Simple-Update algorithm.

[^1]: Consult Tindall 2023 ([arXiv:2306.17837](https://arxiv.org/abs/2306.17837)) to see this; this gauge can be found using Belief Propagation.

# File contents

* **`PEPO.py`** Projector-entangled Pair Operators on arbitary graphs, where the Tensor Network structure is inherited from the main module (see [this file](https://github.com/HendrikKuehne/belief_propagation/blob/main/belief_propagation/README.md) for an introduction).
  * **ToDo**: Sparse matrices? Scipy only allows for two-dimensional sparse arrays, but the [sparse package](https://sparse.pydata.org/en/stable/) implements higher-dimensional sparse arrays.
    * Sparse matrices would be most useful in `PEPO.to_dense()`. There is hope that this could be done more or less easily, since the the `sparse` package has an [einsum implementation](https://sparse.pydata.org/en/stable/generated/sparse.einsum.html#sparse.einsum) and a [reshape](https://sparse.pydata.org/en/stable/generated/sparse.reshape.html#sparse.reshape).
  * **ToDo**: `PEPS` instances do not require a leg ordering on the underlying graph, but `PEPO` instances do; there is no reason why PEPOs should require a leg ordering. (except if a typical workflow is to generate a PEPS first and use it's graph to generate a PEPO; the leg ordering in PEPO intialisation should be optional)
  * Implementation works; tested using `dummynet1`. Explicit construction of the Hamiltonian and `PEPO.to_dense()` yield the same eigenvalues.
* **`PEPS.py`** PEPS on arbitrary graphs.
* **`braket.py`** Stacks of combinations of PEPS and PEPO on arbitrary graphs.
  * **ToDo** Accelerate the mesage uodate somehow
    * Sparse matrices? Scipy only allows for two-dimensional sparse arrays, but the [sparse package](https://sparse.pydata.org/en/stable/) implements higher-dimensional sparse arrays.
    * Pancotti & Gray stack all the tensors and the messages s.t. the BP algorithm becomes a vector iteration ([arxiv:2306.15004](https://arxiv.org/abs/2306.15004))
    * :arrows_counterclockwise: parallelize using [Ray](https://docs.ray.io/en/latest/ray-overview/getting-started.html) - What I have done so far is actually slower than the straightforward implementation. The overhead seems to be too large, maybe I should think this through a little more thoroughly.
  * :white_check_mark: Progress bar using [tqdm](https://tqdm.github.io)
  * **ToDo** Smarter discrimination between cases in `Braket.contract`.
  * **ToDo** Optimize local updates; Lanczos algorithm? (In python implemented, for example, in the [`pylanczos` package](https://pypi.org/project/pylanczos/))
  * **ToDo** The implementation of the DMRG class is wildly inefficient (at least in terms of memory) because the overlap $\braket{\psi|\psi}$ and the expectation value $\braket{\psi|H|\psi}$ are both stored as full `Braket` objects, although they contain essentially the same data
