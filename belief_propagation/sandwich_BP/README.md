# Open questions

* **Stoudenmire vs Gray:** Kim et Al claimed utility of quantum computing before fault tolerance in 2017 ([Nature 618, 500 - 505 (2023)](https://doi.org/10.1038/s41586-023-06096-3)). Both Stoudenmire et Al ([PRX Quantum 5, 010308 (2024)](https://doi.org/10.1103/PRXQuantum.5.010308)) and Gray et Al ([Sci. Adv. 10, eadk4321 (2024)](https://doi.org/10.1126/sciadv.adk4321)) published results, where they simulate this experiment using BP on tensor networks - so what's the difference?
  * Both use the same gauge[^1]; Stoudenmire calls it the "Vidal gauge", while Gray cals it the "super-orthogonal gauge"

[^1]: Consult Tindall 2023 ([arXiv:2306.17837](https://arxiv.org/abs/2306.17837)) to see this; this gauge can be found using Beleif Propagation.

# File contents

* **`PEPO.py`** Projector-entangled Pair Operators on arbitary graphs, where the Tensor Network structure is inherited from the main module (see [this file](https://github.com/HendrikKuehne/belief_propagation/blob/main/belief_propagation/README.md) for an introduction).
  * **ToDo**: Sparse matrices? Scipy only allows for two-dimensional sparse arrays, but the [sparse package](https://sparse.pydata.org/en/stable/) implements higher-dimensional sparse arrays.(https://arxiv.org/abs/2306.15004))
  * **ToDo**: `MPS` instances do not require a leg ordering on the underlying graph, but `PEPO` instances do; there is no reason why PEPOs should require a leg ordering. (except if a typical workflow is to generate a MPS first and use it's graph to generate a PEPO; the leg ordering in PEPO intialisation should be optional)
  * Implementation works; tested using `dummynet1`. Explicit construction of the Hamiltonian and `PEPO.to_dense()` yield the same eigenvalues.
* **MPS.py** MPS on arbitrary graphs.
* **braket.py** Stacks of combinations of MPS and PEPO on arbitrary graphs.
  * **ToDo**: Accelerate the mesage uodate somehow
    * Sparse matrices? Scipy only allows for two-dimensional sparse arrays, but the [sparse package](https://sparse.pydata.org/en/stable/) implements higher-dimensional sparse arrays.
    * Pancotti & Gray stack all the tensors and the messages s.t. the BP algorithm becomes a vector iteration ([arxiv:2306.15004])
    * :arrows_counterclockwise: parallelize using [Ray](https://docs.ray.io/en/latest/ray-overview/getting-started.html) - What I have done so far is actually slower than the straightforward implementation. The overhead seems to be too large, maybe I should think this through a little more thoroughly.
  * :white_check_mark: Progress bar using [tqdm](https://tqdm.github.io)

# Literature

* IBM kicked Ising experiment
  * Youngseok Kim, Andrew Eddins, Sajant Anand, Ken Xuan Wei, Ewout van den Berg, Sami Rosenblatt, Hasan Nayfeh, Yantao Wu, Michael Zaletel, Kristan Temme1, Abhinav Kandala 
    Evidence for the utility of quantum computing before fault tolerance  
    [Nature 618, 500 - 505 (2023)](https://doi.org/10.1038/s41586-023-06096-3)
  * Joseph Tindall, Matthew Fishman, E. Miles Stoudenmire, Dries Sels  
    Efficient Tensor Network Simulation of IBM’s Eagle Kicked Ising Experiment  
    [PRX Quantum 5, 010308 (2024)](https://doi.org/10.1103/PRXQuantum.5.010308) ([arXiv:2306.14887](https://arxiv.org/abs/2306.14887))
  * Tomislav Begusic, Johnnie Gray, Garnet Kin-Lic Chan  
    Fast and converged classical simulations of evidence for the utility of quantum
computing before fault tolerance  
    [Sci. Adv. 10, eadk4321 (2024)](https://doi.org/10.1126/sciadv.adk4321) ([arXiv:2308.05077](https://arxiv.org/abs/2308.05077))
* Tensor networks, many-body physics and variational methods
  * Subhayan Sahu, Brian Swingle  
    Efficient tensor network simulation of quantum many-body physics on sparse graphs  
    [arXiv:2206.04701](https://arxiv.org/abs/2206.04701)
  * Joseph Tindall, Matthew T. Fishman  
    Gauging tensor networks with belief propagation  
    [arXiv:2306.17837](https://arxiv.org/abs/2306.17837)
  * Nicola Pancotti, Johnnie Gray  
    One-step replica symmetry breaking in the language of tensor networks  
    [arxiv:2306.15004](https://arxiv.org/abs/2306.15004)