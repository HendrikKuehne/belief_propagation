# Belief propagation for tensor network contraction

The code contained herein uses Belief Propagation (BP )as a subroutine in ground state search, and implements many PEPS and PEPO routines around that. The goal is to facilitate ground state search on systems with arbitrary geometries. The typical workflow is as follows: Given a geometry of a system (i.e. a graph that represents coupling between spins), one creates a Hamiltonian and passes it to the desired algorithm.

# Basic usage of BP within this module

The centerpiece of this module's contents is the implementation of BP. Considering that, in the context of Quantum Information and classical simulation, BP is defined on what is known as "double-edged factor graphs" (see [here](https://doi.org/10.1109/ITW.2017.8277985)), the central objects on which one can run BP using this module are norms $\braket{\psi|\psi}$ and expectation values $\braket{\psi|O|\psi}$. These are instances of the `Braket` class, and running BP is as easy as calling `Braket.BP()`.

Consider the following example:

```python
    from belief_propagation.graphs import heavyhex
    from belief_propagation.braket import Braket
    from belief_propagation.PEPS import PEPS

    # Defining the geometry of this problem: A heavyhex-graph with four cells.
    G = heavyhex(2, 2)

    # Constructing a quantum state, and it's norm.
    psi = PEPS.init_random(G=G, D=2, chi=3)
    braket = Braket.Overlap(psi, psi)

    # BP on the norm.
    braket.BP(numiter=500, threshold=1e-10)
    cntr_BP = braket.cntr

    # Exact contraction.
    cntr_exact = braket.contract()
```

This code constructs a random quantum state $\ket{\psi}$ on a heavyhex-graph with four cells, and subsequently constructs the norm $\braket{\psi|\psi}$. Calling `Braket.BP` runs the Belief Propagation iteration on $\braket{\psi|\psi}$ for `numiter` message updates (or until the desired convergence threshold is reached). The fixed-point messages are contained in the `Braket.msg` attribute. `Braket.BP` calculates the BP contraction value, and saves it under `Braket.cntr`. `Braket` objects also admit exact contraction.

# Ground state search

Currently, two algorithms for ground state search are implemented and equipped with BP subroutines:

* DMRG, and
* imaginary time evolution.

## BP-DMRG

The BP-DMRG algorithm is in many respects a standard implementation of DMRG. Standard implementations in one dimension rely on the site-canonical form of a MPS, however, which is not available for PEPOs.[^1] This is where BP comes in: in absence of a canonical form, the left- and right block in the local Hamiltonian need to be obtained through partial contraction of the expectation value $\braket{\psi|H|\psi}$. This is very computationally intensive to do exactly for each local update, but contraction through BP is very cheap. Thus, this implementation of DMRG forms the local Hamiltonian from messages. Note also that since there is no canonical form available, the local update requires us to solve a generalized eigenvalue problem.

The algorithm, for every sweep, thus proceeds as follows (in pseudocode):

```
    for node in psi:
        psi = QR_gauging(psi)
        # gauging ensures numerical stability.

        Braket.Expval(H_pos, psi).BP()
        Braket.Expval(H_neg, psi).BP()
        Braket.Overlap(psi, psi).BP()
        # Obtaining messages.

        localH = assemble_local_hamiltonian()
        localN = assemble_local_environment()
        # Necessary objects for the local update.

        T = solve_gen_eigval_problem(localH, localN)
        # Solving the generalized eigenvalue problem yields the new site tensor.

        psi[node] = T
```

What does this look like in practice? All the above functionality is captured in the `run()` function of the `DMRG` class. The instantiation of one such object requires two things:

* A Hamiltonian. Since BP only converges on positive- or negative-semidefinite graphical models, it must be split up into a positive- and a negative-semidefinite part. Functions for obtaining definite splittings of the TFI model and the Heisenberg model come with this module.[^2]
* An initial state. It can be defined in two ways: Either by passing a bond dimension, in which case an initial state with that bond dimension is chosen at random, or by explicitly passing a state (e.g. an instance of the `PEPS` class).

Consider the following code snippet:

```python
    from belief_propagation.graphs import hex
    from belief_propagation.hamiltonians import TFI
    from belief_propagation.dmrg import DMRG

    # Defining the geometry of this problem: A hexagonal graph with four cells.
    G = hex(2, 2)

    # Constructing the Hamiltonian.
    TFI_pos, TFI_neg = TFI.posneg(G=G, J=1, g=3.5)

    # Instantiating a DMRG object. The state is here supplied through the bond
    # dimension.
    dmrg = DMRG(
        oplist=(TFI_pos, TFI_neg),
        chi=3,
        bond_dim_strategy="uniform"
    )

    dmrg.run(nSweeps=3, numiter=500)
```

The invocation of `dmrg.run()` runs the DMRG algorithm with three sweeps. Every BP iteration will run for a maximum of 500 message updates, but will stop early if a pre-determined accuracy is reached (by default: $10^{-10}$). The final state is then contained in `dmrg.psi`.

## Imaginary time evolution

The implementation of imaginary time evolution itself is the standard one; the state $\ket{\psi}$ is evolved by applying $e^{-\Delta\tau H}$ to it. Here as well the implementation is independent of the system geometry; $\ket{\psi}$ and $e^{-\Delta\tau H}$ are instances of the `PEPS` and `PEPO` classes, respectively, and may be defined on any graph. Applying an operator to a state follows familiar notation; $e^{-\Delta\tau H}\ket{\psi}$ is calculated through `psi = H_exp @ psi`.

BP enters the picture in two subroutines, that are necessary to make imaginary time evolution viable on classical hardware: compression and contraction.

Since bond dimensions grow during each application of the operator, compression is of the essence. For this we turn to L2BP compression (introduced [here](https://doi.org/10.1126/sciadv.adk4321)). This method constructs projectors through singular value decompositions of messages, which may be truncated, thusly compressing the state optimally.

Contraction is necessary to extract expectation values, once one has procured the ground state. Besides default BP, which represents an uncontrolled approximation, this module also implements Loop Series Expansion (introduced [here](https://arxiv.org/abs/2409.03108)). Loop Series Expansion controls the contraction accuracy of BP by incorporating loops up to a certain length.

Consider the following code snippet:

```python
    from belief_propagation.graphs import hex
    from belief_propagation.hamiltonians import Heisenberg
    from belief_propagation.time_evolution import operator_exponential
    from belief_propagation.truncate_expand import L2BP_compression, loop_series_contraction

    # Defining the geometry of this problem: A hexagonal graph with four cells.
    G = hex(2, 2)

    # Constructing the Hamiltonian.
    H = Heisenberg(G=G, g=3, Jx=.5, Jy=1, Jz=1.5)

    # Constructing the initial state.
    psi = PEPS.init_random(G=G, D=2, chi=3)

    # Constructing the time evolution operator.
    dtau = .05
    H_exp = operator_exponential(
        op=(-1) * dtau * H,
        contract=True,
        trotter_order=2
    )

    # Imaginary time evolution.
    for iStep in range(10):
        psi = H_exp @ psi

        psi = L2BP_compression(
            psi=psi,
            singval_threshold=1e-8,
            max_bond_dim=16
        )

    # Calculating the energy expectation value.
    expval = Braket.Expval(psi=psi, op=H)
    E0 = loop_series_contraction(braket=expval, max_order=6)
```

This code first defines the Heisenberg model Hamiltonian and an initial state with $\chi=3$ and $D=2$. The time evolution operator is then calculated, up to trotterization to second order. During imaginary time evolution, the state is compressed down to a maximum bond dimension of 16 after every application of the time evolution order. Afterwards, the ground state energy expectation value is calculated up to loop order six.

## Contents

* `belief_propagation/`
    * `old/` Legacy code, for reference.
    * `braket.py` Class `Braket`.
    * `cupy_utils.py` Matrix-free local Hamiltonian and local environment, that define the generalized eigenvalue problem during the local DMRG update. Parallelized implementation using CuPy.
    * `dmrg.py` Class `DMRG`.
    * `graphs.py` Creation of various graphs.
    * `hamiltonians.py` Transverse-field Ising model and Heisenberg model.
    * `networks.py` Manipulation of tensor networks.
    * `PEPO.py` Classes `OpChain`, `OpLayer`, `PEPO` and `PauliPEPO`. Zero-valued PEPO and identity PEPO.
    * `PEPS.py` Class `PEPS`.
    * `time_evolution.py` Trotterization, PEPO exponential and simple-update TEBD.
    * `truncate_expand.py` Manipulation and processing of brakets and PEPS. Inserting projectors on edges, compression, feynman cuts, gauging, and Loop Series Expansion.
    * `utils.py` Utility functions: Math routines, hermiticity checks, graph processing, operator chains and operator layer processing, sanity checks.
* `doc/`
    * `plots/` Discussion of various plots that illustrate the behavior of the contents of this module. Very messy and without explanations; beware.

[^1]: Exceptions exist, e.g. in two dimensions (see e.g. [arXiv:2507.08080](https://arxiv.org/abs/2507.08080)).

[^2]: To be more precise, BP only converges on graphs that preserve the positive- or negative-semidefiniteness of the messages. If the graph represents the tensor network that is $\braket{\psi|O|\psi}$, this is only the case if
$O$ is positive- or negative-semidefinite. The origin of this behavior lies in he fact that BP was originally used to marginalize over probability distributions.
