# File contents

* **`TPO.py`** Matrix Product Operators on arbitary graphs, where the Tensor Network structure is inherited from the main module (see [this file](https://github.com/HendrikKuehne/belief_propagation/blob/main/belief_propagation/README.md) for an introduction).
  * **ToDo**: Sparse matrices? Scipy only allows for two-dimensional sparse arrays, but the [sparse package](https://sparse.pydata.org/en/stable/) implements higher-dimensional sparse arrays.
  * Implementation works; tested using `dummynet1`. Explicit construction of the Hamiltonian and `TPO.to_dense()` yield the same eigenvalues.