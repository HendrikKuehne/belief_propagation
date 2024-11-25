# File contents

* **`PEPO.py`** Projector-entangled Pair Operators on arbitary graphs, where the Tensor Network structure is inherited from the main module (see [this file](https://github.com/HendrikKuehne/belief_propagation/blob/main/belief_propagation/README.md) for an introduction).
  * **ToDo**: Sparse matrices? Scipy only allows for two-dimensional sparse arrays, but the [sparse package](https://sparse.pydata.org/en/stable/) implements higher-dimensional sparse arrays.
  * **ToDo**: `MPS` instances do not require a leg ordering on the underlying graph, but `PEPO` instances do; there is no reason why PEPOs should require a leg ordering. (except if a typical workflow is to generate a MPS first and use it's graph to generate a PEPO; the leg ordering in PEPO intialisation should be optional)
  * Implementation works; tested using `dummynet1`. Explicit construction of the Hamiltonian and `PEPO.to_dense()` yield the same eigenvalues.
* **MPS.py** MPS on arbitrary graphs.
* **Sandwich.py** Stacks of combinations of MPS and PEPO on arbitrary graphs.