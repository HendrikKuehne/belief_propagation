"""
DMRG on Braket-objects. Contains the DMRG-class.
"""

__all__ = ["DMRG", "LoopSeriesDMRG"]

import itertools
from typing import Dict, Tuple, List, Iterator
import warnings
import copy

import numpy as np
import networkx as nx
import cotengra as ctg
import tqdm

from belief_propagation.utils import (
    is_hermitian,
    gen_eigval_problem,
    rel_err,
    same_legs,
    graph_compatible,
    check_msg_intact
)
from belief_propagation.PEPO import PEPO
from belief_propagation.PEPS import PEPS
from belief_propagation.braket import Braket, BP_excitations
from belief_propagation.truncate_expand import (
    L2BP_compression,
    QR_gauging,
    loop_series_contraction
)


class DMRG:
    """
    Single-site DMRG on graphs, with a sum hamiltonian. Environments are
    calculated using belief propagation.
    """

    nSweeps: int = 5
    """Default number of sweeps."""
    hermiticity_threshold: float = 1e-6
    """Allowed deviation from exact hermiticity."""
    tikhonov_regularization_eps: float = 1e-6
    """Epsilon for Tikhonov-regularization of singular messages."""

    def __assemble_messages(self, sanity_check: bool = False) -> None:
        """
        Evaluates the direct sum of messages, and writes to
        `self._msg`.
        """
        if sanity_check: assert self.intact

        self._msg = {
            sender: {
                receiver: None
                for receiver in self.overlap.G.adj[sender]
            }
            for sender in self.overlap.G.nodes()
        }

        for node1, node2 in self.overlap.G.edges():
            for sender, receiver in itertools.permutations((node1, node2)):
                chi = sum(
                    expval.op.G[sender][receiver][0]["size"]
                    for expval in self.expvals
                )
                msg = np.full(
                    shape=(
                        self.overlap.bra.G[sender][receiver][0]["size"],
                        chi,
                        self.overlap.ket.G[sender][receiver][0]["size"]
                    ),
                    fill_value=np.inf,
                    dtype=np.complex128
                )

                chi_filled = 0
                for expval in self.expvals:
                    # We are filling the slices that correspond to the expvals
                    # one after another.
                    fill_slice = (
                        slice(self.overlap.bra.G[sender][receiver][0]["size"]),
                        slice(
                            chi_filled,
                            chi_filled+expval.op.G[sender][receiver][0]["size"]
                        ),
                        slice(self.overlap.ket.G[sender][receiver][0]["size"])
                    )
                    msg[fill_slice] = expval.msg[sender][receiver]
                    chi_filled += expval.op.G[sender][receiver][0]["size"]

                assert np.all(np.isfinite(msg))
                self._msg[sender][receiver] = msg

        return

    def __assemble_T_totalH(self, sanity_check: bool = False) -> None:
        """
        Evaluates the direct product of PEPO tensors, and writes to
        `self._T_totalH`. Local hamiltonian tensors are block-diagonal
        and contain the local tensors of `self.expvals` on the diagonal.
        """
        if sanity_check: assert self.intact

        self._T_totalH = {node: None for node in self.overlap.G.nodes()}

        for node in self.overlap.G.nodes():
            # Total virtual bond dimension of the hamiltonian.
            chi = {
                neighbor: sum(
                    expval.op.G[node][neighbor][0]["size"]
                    for expval in self.expvals
                )
                for neighbor in self.overlap.G.adj[node]
            }

            H = np.zeros(
                shape=tuple(
                    chi[neighbor]
                    for neighbor in self.overlap.G.adj[node]
                ) + (
                    self.D[node], self.D[node]
                ),
                dtype=np.complex128
            )

            chi_filled = {neighbor: 0 for neighbor in self.overlap.G.adj[node]}
            # Filling the Hamiltonian.
            for expval in self.expvals:
                index = tuple(
                    slice(
                        chi_filled[neighbor],
                        (chi_filled[neighbor]
                         + expval.op.G[node][neighbor][0]["size"])
                    )
                    for neighbor in self.overlap.G.adj[node]
                ) + (
                    slice(self.D[node]), slice(self.D[node])
                )
                H[index] = expval.op[node]

                for neighbor in self.overlap.G.adj[node]:
                    edge_size = expval.op.G[node][neighbor][0]["size"]
                    chi_filled[neighbor] += edge_size

            self._T_totalH[node] = H

        return

    def __assemble_localH(
            self,
            node: int,
            threshold: float = hermiticity_threshold,
            sanity_check: bool = False
        ) -> np.ndarray:
        """
        Hamiltonian at `node`, calculated by taking inbound messages as
        environments. `threshold` is the absolute allowed error in the
        hermiticity of the hamiltonian obtained (checked if
        `sanity_check=True`).
        """
        # sanity check
        if sanity_check: assert self.intact

        # Construct total PEPO tensors, if necessary.
        if self._T_totalH is None:
            self.__assemble_T_totalH(sanity_check=sanity_check)

        # The hamiltonian at node is obtained by tracing out the rest of the
        # network. The environments are approximated by messages.
        nLegs = len(self.overlap.G.adj[node])
        args = ()
        """Arguments for einsum"""

        out_legs = (tuple(range(nLegs))
                    + (3 * nLegs,)
                    + tuple(range(2 * nLegs, 3 * nLegs))
                    + (3*nLegs + 1,))
        # Leg ordering of the local hamiltonian: (bra virtual dimensions, bra
        # physical dimension, ket virtual dimensions, ket physical dimension).
        # The order of the virtual legs is inherited from the "legs" indices on
        # the edges; the 0th leg ends up in the first dimension.
        vir_dim = 1

        for neighbor in self.overlap.G.adj[node]:
            # collecting einsum arguments
            args += (
                self.msg[neighbor][node],
                (
                    # bra leg:
                    self.overlap.bra.G[node][neighbor][0]["legs"][node],
                    # operator leg:
                    (nLegs
                     + self.expvals[0].op.G[node][neighbor][0]["legs"][node]),
                    # ket leg:
                    (2 * nLegs
                     + self.overlap.ket.G[node][neighbor][0]["legs"][node]),
                )
            )

            # Compiling virtual dimensions for later reshape.
            vir_dim *= self.overlap.ket.G[node][neighbor][0]["size"]

        args += (
            # operator tensor
            self._T_totalH[node],
            (tuple(nLegs + iLeg for iLeg in range(nLegs))
             + (3 * nLegs, 3*nLegs + 1)),
        )

        H = np.einsum(*args, out_legs, optimize=True)
        H = np.reshape(H, (vir_dim * self.D[node], vir_dim * self.D[node]))

        if sanity_check:
            assert is_hermitian(H, threshold=threshold, verbose=False)

        return H

    def __assemble_localN(
            self,
            node: int,
            threshold: float = hermiticity_threshold,
            sanity_check: bool = False
        ) -> np.ndarray:
        """
        Environment at node. Calculated from `self.overlap`. This
        amounts to stacking and re-shaping messages, that are inbound to
        `node`. `threshold` is the absolute allowed error in the
        hermiticity of the resulting matrix (checked if
        `sanity_check=True`).
        """
        # sanity check
        if sanity_check: assert self.intact

        nLegs = len(self.overlap.G.adj[node])
        args = ()
        """Arguments for einsum"""

        out_legs = (tuple(range(nLegs))
                    + (3 * nLegs,)
                    + tuple(range(2 * nLegs, 3 * nLegs))
                    + (3*nLegs + 1,))
        # Leg ordering of the local hamiltonian: (bra virtual dimensions, bra
        # physical dimension, ket virtual dimensions, ket physical dimension)
        # The order of the virtual dimensions is inherited from the "legs"
        # indices on the edges; the 0th leg ends up in the first dimension.
        vir_dim = 1

        for neighbor in self.overlap.G.adj[node]:
            if not self.overlap.msg[neighbor][node].shape[1] == 1:
                warnings.warn(
                    "".join((
                        f"Message {neighbor} -> {node} does not originate ",
                        "from an overlap!"
                    )),
                    RuntimeWarning
                )

            msg = self.overlap.msg[neighbor][node][:,0,:]

            # Collecting einsum arguments.
            args += (
                msg,
                (
                    # bra leg:
                    self.overlap.bra.G[node][neighbor][0]["legs"][node],
                    # ket leg:
                    (2 * nLegs
                     + self.overlap.ket.G[node][neighbor][0]["legs"][node]),
                )
            )

            # Compiling virtual dimensions for later reshape.
            vir_dim *= self.overlap.ket.G[node][neighbor][0]["size"]

        # Identity for the physical dimension.
        args += (self.overlap.op.I(node=node), (3 * nLegs, 3*nLegs + 1))

        N = np.einsum(*args, out_legs, optimize=True)
        N = np.reshape(N, (vir_dim * self.D[node], vir_dim * self.D[node]))

        if sanity_check: assert is_hermitian(N, threshold=threshold)

        return N

    def __BP(self, sanity_check: bool = False, **kwargs) -> None:
        """
        BP iteration on the overlap and all expvals. Total messages and
        local PEPO tensors are formed, if BP converged.
        """

        # Handling kwargs.
        if "iterator_desc_prefix" in kwargs.keys():
            iterator_desc_prefix = kwargs.pop("iterator_desc_prefix")
        else:
            iterator_desc_prefix = "DMRG"

        # Running BP on all Brakets that are not converged.
        if not self.overlap.converged:
            overlap_ = copy.deepcopy(self.overlap)
            overlap_.BP(
                iterator_desc_prefix="".join((
                    iterator_desc_prefix,
                    f" | overlap"
                )),
                sanity_check=sanity_check,
                **kwargs
            )
            self.overlap.msg = overlap_.msg
            self.overlap._converged = overlap_._converged
            self.overlap.cntr = overlap_.cntr
            self.overlap.normalize_messages(
                normalize_to="cntr",
                sanity_check=sanity_check
            )

        for i in range(len(self.expvals)):
            if not self.expvals[i].converged:
                expval_ = copy.deepcopy(self.expvals[i])
                expval_.BP(
                    iterator_desc_prefix="".join((
                        iterator_desc_prefix,
                        f" | expval {i}"
                    )),
                    sanity_check=sanity_check,
                    **kwargs
                )
                expval_.normalize_messages(
                    normalize_to="cntr",
                    sanity_check=sanity_check
                )
                self.expvals[i].msg = expval_.msg
                self.expvals[i]._converged = expval_._converged
                self.expvals[i].cntr = expval_.cntr

        if self.converged:
            self.__assemble_messages(sanity_check=sanity_check)
        else:
            warnings.warn("BP iteration not converged.", RuntimeWarning)
            self._msg = None

        return

    def __sweep(self, gauge: bool, sanity_check: bool, **kwargs) -> float:
        """
        Local update at all sites. `kwargs` are passed to `Braket.BP`.
        Returns the change in energy after the sweep.

        The graph is traversed in breadth-first manner. After each local
        update, new outgoing messages are calculated, thereby updating
        the environments.
        """
        if sanity_check: assert self.intact

        # Handling kwargs.
        if "iterator_desc_prefix" in kwargs.keys():
            iterator_desc_prefix = "".join((
                kwargs.pop("iterator_desc_prefix"),
                " | DMRG | "
            ))
        else:
            iterator_desc_prefix = "DMRG | "

        # Calculating environments and previous energy.
        self.__BP(sanity_check=sanity_check, **kwargs)
        Eprev = self.E0

        for node in nx.dfs_postorder_nodes(
            self.expvals[0].op.tree,
            source=self.expvals[0].op.root
        ):
            H = self.__assemble_localH(node, sanity_check=sanity_check)
            N = self.__assemble_localN(node, sanity_check=sanity_check)

            if False:
                warnings.warn(
                    "".join((
                        "This check does not work anymore due to changes in ",
                        "the normalization of messages."
                    )),
                    DeprecationWarning
                )
                # Are local hamiltonian and environment correctly defined?
                local_psi = self.ket[node].flatten()
                expval_local_cntr = ctg.einsum(
                    "i,ik,k", local_psi.conj(), H, local_psi
                )
                overlap_local_cntr = ctg.einsum(
                    "i,ik,k", local_psi.conj(), N, local_psi
                )
                expvals_total_cntr = sum(
                    expval.cntr for expval in self.expvals
                )

                if not np.isclose(expval_local_cntr, expvals_total_cntr):
                    rel_err_ = rel_err(expvals_total_cntr, expval_local_cntr)
                    warnings.warn(
                        "".join((
                            f"Local hamiltonian at node {node} does not ",
                            "reproduce expectation value. Relative error ",
                            f"{rel_err_:.3e}."
                        )),
                        RuntimeWarning
                    )
                if not np.isclose(overlap_local_cntr, self.overlap.cntr):
                    rel_err_ = rel_err(self.overlap.cntr, overlap_local_cntr)
                    warnings.warn(
                        "".join((
                            f"Local environment at node {node} does not ",
                            "reproduce overlap. Relative error ",
                            f"{rel_err_:.3e}."
                        )),
                        RuntimeWarning
                    )

            eigvals,eigvecs = gen_eigval_problem(
                H,
                N,
                eps=self.tikhonov_regularization_eps
            )

            # Finding the correct shape for the new site tensor.
            newshape = [np.nan for _ in self.overlap.G.adj[node]]
            for neighbor in self.overlap.G.adj[node]:
                leg = self.overlap.G[node][neighbor][0]["legs"][node]
                newshape[leg] = self.overlap.ket.G[node][neighbor][0]["size"]
            newshape += [self.D[node],]

            # Re-shaping statevector into site tensor.
            T = np.reshape(eigvecs[:, np.argmin(eigvals)], newshape)

            # Inserting the new tensor and gauging the current node.
            self[node] = T
            if gauge:
                self.gauge(
                    sanity_check=sanity_check,
                    tree=self.expvals[0].op.tree,
                    nodes=(node,)
                )

            # Calculating new environments.
            self.__BP(
                sanity_check=sanity_check,
                iterator_desc_prefix="".join((
                    iterator_desc_prefix,
                    f"node {node}"
                )),
                **kwargs
            )

        Enext = self.E0

        return np.abs(Eprev - Enext)

    def run(
            self,
            nSweeps: int = None,
            verbose: bool = False,
            gauge: bool = True,
            compress: bool = True,
            sanity_check: bool = False,
            **kwargs
        ) -> None:
        """
        Runs single-site DMRG on the underlying braket. `kwargs` are
        passed to BP iterations. The state is not normalized afterwards!
        """
        if sanity_check: assert self.intact

        # Handling kwargss
        kwargs["verbose"] = False

        nSweeps = nSweeps if nSweeps != None else self.nSweeps
        iterator = tqdm.tqdm(
            range(nSweeps),
            desc="DMRG sweeps",
            disable=not verbose
        )
        eps_list = ()

        if gauge: self.gauge(sanity_check=sanity_check, **kwargs)

        for iSweep in iterator:
            eps = self.__sweep(
                gauge=gauge, sanity_check=sanity_check, **kwargs
            )
            iterator.set_postfix_str(f"eps = {eps:.3e}")
            eps_list += (eps,)

            if compress: self.compress(sanity_check=sanity_check, **kwargs)

        return

    def contract(self, sanity_check: bool = False) -> float:
        """
        Exact calculation of the current expectation value.
        """
        cntr = 0
        for expval in self.expvals:
            cntr += expval.contract(sanity_check=sanity_check)
        cntr /= self.overlap.contract(sanity_check=sanity_check)

        return cntr

    def gauge(
            self,
            method: str = "QR",
            sanity_check: bool = False,
            **kwargs
        ) -> None:
        """
        Gauges bra and ket. `kwargs` are passed to the respective
        gauging methods.
        """

        if method == "QR":
            self.ket = QR_gauging(
                self.ket, sanity_check=sanity_check, **kwargs
            )
            return

        if method == "Schmidt":
            self.ket = L2BP_compression(
                self.ket,
                singval_threshold=0,
                return_singvals=False,
                sanity_check=sanity_check,
                **kwargs
            )
            return

        raise NotImplementedError(
            "Gauging method " + method + " not implemented."
        )

    def compress(
            self,
            method: str = "L2BP",
            sanity_check: bool = False,
            **kwargs
        ) -> None:
        """
        Compresses bra and ket. `kwargs` are passed to the respective
        compression methods.
        """

        if method == "L2BP":
            self.ket = L2BP_compression(
                self.ket,
                return_singvals=False,
                sanity_check=sanity_check,
                **kwargs
            )

            return

        raise NotImplementedError(
            "Compression method " + method + " not implemented."
        )

    def perturb_messages(
            self,
            d: float = 1e-3,
            msg_init: str = "zero-normal",
            sanity_check: bool = False,
            rng: np.random.Generator = np.random.default_rng()
        ) -> None:
        """
        Perturbing messages on the overlap and all expvals.
        """
        self.overlap.perturb_messages(
            real=False,
            d=d,
            msg_init=msg_init,
            sanity_check=sanity_check,
            rng=rng
        )
        for i in range(len(self.expvals)):
            self.expvals[i].perturb_messages(
                real=False,
                d=d,
                msg_init=msg_init,
                sanity_check=sanity_check,
                rng=rng
            )

        return

    @property
    def msg(self) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Messages on the total expval. These are formed as direct
        products of messages on the individual expvals.
        """
        if self._msg is None: self.__assemble_messages()
        return self._msg

    @property
    def D(self) -> Dict[int, int]:
        """Physical dimension at every node."""
        return self.overlap.D

    @property
    def converged(self) -> bool:
        """Whether the messages are converged."""
        return (self.overlap.converged
                and all(expval.converged for expval in self.expvals))

    @property
    def ket(self) -> PEPS:
        """The current state of the system."""
        return self.overlap.ket

    @ket.setter
    def ket(self, newket:PEPS) -> None:
        """
        Changing the state of the system requires inserting a new PEPS
        in all `Braket` objects. Convergence markers will be set to
        `False`.
        """
        self.overlap.ket = newket
        self.overlap.bra = newket.conj()
        for expval in self.expvals:
            expval.ket = newket
            expval.bra = newket.conj()

        return

    @property
    def E0(self) -> float:
        """Current estimate of the ground state energy."""
        return sum(expval.cntr for expval in self.expvals) / self.overlap.cntr

    @property
    def nsites(self) -> int:
        """
        Number of sites of the system.
        """
        return self.overlap.nsites

    @property
    def intact(self) -> bool:
        """
        Checks if the DMRG algrithm can be run. This amounts to:
        * Checking if the underlying braket is intact.
        + Checking if expval graphs and overlap graph are compatible.
        * Checking if bra and ket are adjoint to one another.
        * Checking if the leg orderings in `self.overlap` and
        `self.expvals` are the same.
        """
        # Are the brakets compatible?
        for expval in self.expvals:
            if not expval.intact: return False
            if expval.D != self.D: return False
        if not self.overlap.intact: return False

        # Are expval graphs and overlap graph compatible, in terms of geometry
        # and physical dimensions?
        for i, expval in enumerate(self.expvals):
            if not graph_compatible(expval.G, self.overlap.G):
                warnings.warn(
                    f"Graphs of overlap and expval {i} not compatible.",
                    UserWarning
                )
                return False

        # Are bra and ket in the overlap adjoint?
        for node in self.overlap:
            if not np.allclose(
                self.overlap.bra[node].conj(),
                self.overlap.ket[node]
            ):
                warnings.warn(
                    "".join((
                        f"Bra- and ket-tensors at node {node} in overlap not ",
                        "complex conjugates of one another."
                    )),
                    UserWarning
                )
                return False

        # Are bra and ket in the expvals adjoint?
        for i, expval in enumerate(self.expvals):
            for node in expval:
                if not np.allclose(expval.bra[node].conj(), expval.ket[node]):
                    warnings.warn(
                        "".join((
                            f"Bra- and ket-tensors at node {node} in ",
                            f"expval {i} not complex conjugates of one ",
                            "another."
                        )),
                        UserWarning
                    )
                    return False

        # Do overlap and expval contain the same tensors?
        for i, expval in enumerate(self.expvals):
            for node in expval:
                if not np.allclose(expval.ket[node], self.overlap.ket[node]):
                    warnings.warn(
                        "".join((
                            f"Tensor at node {node} is not the same in ",
                            f"overlap and expval {i}."
                        )),
                        UserWarning
                    )
                    return False

        # Are the leg orderings the same?
        for i, expval in enumerate(self.expvals):
            if not same_legs(self.overlap.G, expval.G):
                warnings.warn(
                    f"Leg orderings in overlap and expval {i} are different.",
                    UserWarning
                )
                return False

        return True

    def __getitem__(self, node: int) -> np.ndarray:
        """
        Subscripting with a node gives the ket tensor at this node.
        """
        return self.overlap.ket[node]

    def __setitem__(self, node: int, T: np.ndarray) -> None:
        """
        Changing local tensors in the state directly. Convergence
        markers will be set to `False`.
        """
        # Updating the tensor stacks in all braket objects.
        self.overlap[node] = (T.conj(), self.overlap[node][1], T)
        for i in range(len(self.expvals)):
            self.expvals[i][node] = (
                T.conj(),
                self.expvals[i][node][1],
                T
            )

        return

    def __repr__(self) -> str:
        return "".join(
            (
                f"DMRG problem on {self.nsites} sites.",
                "\nKet: " + str(self.overlap.ket) + "\nHamiltonians: "
            ) + (
                "".join(("\n",str(expval.op)))
                for expval in self.expvals
            ) + (
                "\nMessages are ",
                "converged." if self.converged else "not converged."
            ))

    def __init__(
            self,
            oplist: Tuple[PEPO],
            psi_init: PEPS = None,
            chi: int = None,
            sanity_check: bool = False,
            **kwargs
        ):
        """
        Initialisation of a `DMRG` object, where the state has bond
        dimension `chi`. The initial state is chosen randomly, if it is
        not given. `kwargs` are passed to `PEPS.init_random`.
        
        For `oplist = (H1, H2, ...)`, this object runs single-site DMRG
        on the Hamiltonian `H = H1 + H2 + ...`.
        """
        # sanity check
        if sanity_check:
            for op in oplist:
                assert op.intact
                if not oplist[0].D == op.D:
                    raise ValueError(
                        "All operators must have the same physical dimension."
                    )

            if psi_init is not None:
                assert psi_init.intact

            if (psi_init is None) and (chi is None):
                raise ValueError("".join((
                    "No instructions on how to define an initial state; ",
                    "psi_init and chi cannot both be None."
                )))

        # If not given, initial state is chosen randomly.
        if psi_init is None:
            psi_init = PEPS.init_random(
                G=op.G,
                D=oplist[0].D,
                chi=chi,
                **kwargs
            )

        self.expvals: Tuple[Braket] = tuple(
            Braket.Expval(psi=psi_init, op=op, sanity_check=sanity_check)
            for op in oplist
        )
        """`Braket`-objects for every operator."""
        self.overlap: Braket = Braket.Overlap(psi1=psi_init, psi2=psi_init)
        """Norm of the current state."""

        self._msg: Dict[int, Dict[int, np.ndarray]] = None
        """
        Messages on the total expval. Formed as direct products of
        expval messages.
        """
        self._T_totalH: Dict[int, np.ndarray] = None
        """
        Local tensors of the total hamiltonian. Formed as direct sums of
        constituent operator tensors.
        """

        if sanity_check: assert self.intact


class LoopSeriesDMRG:
    """
    Single-site DMRG on graphs, with a sum hamiltonian. Environments are
    calculated using Loop Series Expansion.
    """

    nSweeps: int = 5
    """Default number of sweeps."""
    hermiticity_threshold: float = 1e-6
    """Allowed deviation from exact hermiticity."""
    tikhonov_regularization_eps: float = 1e-6
    """Epsilon for Tikhonov-regularization of singular messages."""

    def __assemble_T_totalH(self, sanity_check: bool = False) -> None:
        """
        Evaluates the direct product of PEPO tensors, and writes to
        `self._T_totalH`. Local hamiltonian tensors are block-diagonal
        and contain the local tensors of `self.expvals` on the diagonal.
        """
        if sanity_check: assert self.intact

        self._T_totalH = {node: None for node in self._psi}

        for node in self:
            # Total virtual bond dimension of the hamiltonian.
            virt_bond_dims = tuple(
                sum(op[node].shape[leg] for op in self.oplist)
                for leg in range(len(self._psi.G.adj[node]))
            )

            # Scaffolding for the local hamiltonian tensor.
            H = np.zeros(
                shape=virt_bond_dims + (self.D[node], self.D[node]),
                dtype=np.complex128
            )

            # Starting indices for slices.
            chi_filled = [0 for _ in self._psi.G.adj[node]]

            # Filling the Hamiltonian in a block-diagonal fashion.
            for op in self.oplist:
                # Assembling the slice that this operator tensor belongs in.
                index = tuple(
                    slice(
                        chi_filled[leg],
                        chi_filled[leg] + op[node].shape[leg]
                    )
                    for leg in range(len(self._psi.G.adj[node]))
                ) + (
                    slice(self.D[node]), slice(self.D[node])
                )
                H[index] = op[node]

                # Updating the starting indices for the slices.
                for leg in range(len(self._psi.G.adj[node])):
                    chi_filled[leg] += op[node].shape[leg]

            self._T_totalH[node] = H

        return

    def __assemble_localH(
            self,
            node: int,
            threshold: float = hermiticity_threshold,
            sanity_check: bool = False
        ) -> np.ndarray:
        """
        Hamiltonian at `node`, calculated from inbound environments.
        `threshold` is the absolute allowed error in the hermiticity of
        the hamiltonian obtained (checked if `sanity_check = True`).
        """
        # sanity check
        if sanity_check: assert self.intact
        if self._env_totalH is None:
            raise RuntimeError(
                "No environments with which to calculate a local hamiltonian."
            )
        if not node in self._env_totalH.keys():
            raise RuntimeError(
                "No environments with which to calculate a local hamiltonian."
            )
        if not self.converged:
            warnings.warn(
                "Assembling local hamiltonian with non-converged messages.",
                RuntimeWarning
            )

        # Construct total PEPO tensors, if necessary.
        if self._T_totalH is None:
            self.__assemble_T_totalH(sanity_check=sanity_check)

        # The hamiltonian at node is obtained by tracing out the rest of the
        # network. The environments are approximated by messages.
        nLegs = len(self._psi.G.adj[node])
        args = ()
        """Arguments for einsum"""

        out_legs = (tuple(range(nLegs))
                    + (3 * nLegs,)
                    + tuple(range(2 * nLegs, 3 * nLegs))
                    + (3*nLegs + 1,))
        # Leg ordering of the local hamiltonian: (bra virtual dimensions, bra
        # physical dimension, ket virtual dimensions, ket physical dimension).
        # The order of the virtual legs is inherited from the "legs" indices on
        # the edges; the 0th leg ends up in the first dimension.
    
        args = (
            # Environment.
            self._env_totalH[node],
            range(3 * nLegs),
            # Operator tensor.
            self._T_totalH[node],
            tuple(range(nLegs, 2*nLegs)) + (3 * nLegs, 3*nLegs + 1),
        )

        H = np.einsum(*args, out_legs, optimize=True)

        # Compiling virtual dimensions for re-shape.
        vir_dim = np.prod(tuple(
            chi for chi in self._env_totalH[node].shape[:nLegs]
        ))

        H = np.reshape(H, (vir_dim * self.D[node], vir_dim * self.D[node]))

        if sanity_check:
            assert is_hermitian(H, threshold=threshold, verbose=True)

        return H

    def __assemble_localN(
            self,
            node: int,
            threshold: float = hermiticity_threshold,
            sanity_check: bool = False
        ) -> np.ndarray:
        """
        Environment at node. Calculated from `self.overlap`. This
        amounts to stacking and re-shaping messages, that are inbound to
        `node`. `threshold` is the absolute allowed error in the
        hermiticity of the resulting matrix (checked if
        `sanity_check=True`).
        """
        # sanity check
        if sanity_check: assert self.intact
        if self._env_overlap is None:
            raise RuntimeError(
                "No environments with which to calculate a local hamiltonian."
            )
        if not node in self._env_overlap.keys():
            raise RuntimeError(
                "No environments with which to calculate a local hamiltonian."
            )
        if not self.converged:
            warnings.warn(
                "".join((
                    "Assembling local overlap environment with non-converged ",
                    "messages."
                )),
                RuntimeWarning
            )

        nLegs = len(self._psi.G.adj[node])
        args = ()
        """Arguments for einsum"""

        out_legs = (tuple(range(nLegs))
                    + (3 * nLegs,)
                    + tuple(range(2 * nLegs, 3 * nLegs))
                    + (3*nLegs + 1,))
        # Leg ordering of the local hamiltonian: (bra virtual dimensions, bra
        # physical dimension, ket virtual dimensions, ket physical dimension)
        # The order of the virtual dimensions is inherited from the "legs"
        # indices on the edges; the 0th leg ends up in the first dimension.

        # Discarding dummy operator dimensions in the environment.
        relevant_slice = tuple(
            slice(chi) if (i < nLegs or i >= 2*nLegs) else 0
            for i, chi in enumerate(self._env_overlap[node].shape)
        )

        # Collecting einsum arguments.
        args = (
            # The environment in node.
            self._env_overlap[node][relevant_slice],
            tuple(range(nLegs)) + tuple(range(2*nLegs, 3*nLegs)),
            # Identity for the physical dimension.
            np.eye(self.D[node]),
            (3 * nLegs, 3*nLegs + 1)
        )

        N = np.einsum(*args, out_legs, optimize=True)

        # Compiling virtual dimensions for re-shape.
        vir_dim = np.prod(tuple(
            chi for chi in self._env_overlap[node].shape[:nLegs]
        ))

        N = np.reshape(N, (vir_dim * self.D[node], vir_dim * self.D[node]))

        if sanity_check: assert is_hermitian(N, threshold=threshold)

        return N

    def __BP(self, sanity_check: bool = False, **kwargs) -> None:
        """
        Runs BP iterations on the overlap and on every operator, if
        there is no converged set of messages available. Saves the
        messages under `self._msg_oplist` and `self._msg_overlap`.
        """
        if sanity_check: assert self.intact

        if self.converged:
            # We have a set of converged messages; returning.
            return

        # Handling kwargs.
        if "iterator_desc_prefix" in kwargs.keys():
            iterator_desc_prefix = kwargs.pop("iterator_desc_prefix")
        else:
            iterator_desc_prefix = "DMRG"

        self._converged = True

        allbrakets = self.brakets

        # Calculating messages on the operator brakets.
        for i, expval in enumerate(allbrakets[:-1]):
            expval.BP(
                iterator_desc_prefix="".join((
                    iterator_desc_prefix,
                    f" | expval {i}"
                )),
                sanity_check=sanity_check,
                **kwargs
            )

            if not expval.converged:
                self._converged = False
                warnings.warn(
                    f"BP iteration on expval {i} did not converge.",
                    RuntimeWarning
                )

            self._msg_oplist[i] = expval.msg

        # Calculating messages on the overlap.
        allbrakets[-1].BP(
            iterator_desc_prefix="".join((
                iterator_desc_prefix,
                f" | overlap"
            )),
            sanity_check=sanity_check,
            **kwargs
        )

        if not allbrakets[-1].converged:
            self._converged = False
            warnings.warn(
                "BP iteration on overlap did not converge.",
                RuntimeWarning
            )

        self._msg_overlap = allbrakets[-1].msg

        return

    def __calculate_environments(
            self,
            nodes: Tuple[int] = None,
            sanity_check: bool = False,
            **kwargs
        ) -> None:
        """
        Calculates the environments at the sites from `nodes`. If `nodes
        = None` (default), calculates environments at every site.
        `kwargs` are passed to BP iterations.

        The environment at a node is a tensor with `3 * len(adj[node])`
        legs. It's legs come in three groups: First the bra legs,
        followed by the operator legs, and finally the ket legs. Each
        group contains `len(adj[node])` legs, leading to the total of
        `3 * len(adj[node])` legs. The ordering within the groups
        follows the leg ordering of the tensor at the respective node in
        `self._psi`.
        """
        if sanity_check: assert self.intact

        # We need a converged set of messages; running BP iterations.
        self.__BP(sanity_check=sanity_check, **kwargs)

        if nodes is None: nodes = tuple(self.psi)

        allbrakets = self.brakets

        for i, msg_dict in enumerate(self._msg_oplist + [self._msg_overlap,]):
            allbrakets[i].msg = msg_dict
            allbrakets[i]._converged = self.converged
            allbrakets[i].write_cntr_value()

        # Calculating the environments in the overlap.
        self._env_overlap = loop_series_environments(
            braket=allbrakets[-1],
            excitations=self.excitations,
            nodes=nodes,
            max_order=self.max_order,
            sanity_check=sanity_check
        )

        # Calculating the environments in the constituent operators.
        envs_oplist = tuple(
            loop_series_environments(
                braket=braket,
                nodes=nodes,
                max_order=self.max_order,
                sanity_check=sanity_check
            )
            for braket in allbrakets[:-1]
        )

        self._env_totalH = {}
        # Assembling environments of the total hamiltonian.
        for node in nodes:
            envs = tuple(env_oplist[node] for env_oplist in envs_oplist)
            self._env_totalH[node] = environments_direct_sum(
                envs=envs
            )

        return

    def __sweep(self, gauge: bool, sanity_check: bool, **kwargs) -> float:
        """
        Local update at all sites. `kwargs` are passed to `Braket.BP`.
        Returns the change in energy after the sweep.

        The graph is traversed in breadth-first manner. After each local
        update, new outgoing messages are calculated, thereby updating
        the environments.
        """
        if sanity_check: assert self.intact

        # handling kwargs
        if "iterator_desc_prefix" in kwargs.keys():
            iterator_desc_prefix = "".join((
                kwargs.pop("iterator_desc_prefix"),
                " | DMRG | "
            ))
        else:
            iterator_desc_prefix = "DMRG | "

        # Calculating environments and previous energy.
        self.__calculate_environments(sanity_check=sanity_check, **kwargs)
        Eprev = self.E0

        for node in nx.dfs_postorder_nodes(
            self.oplist[0].tree,
            source=self.oplist[0].root
        ):
            H = self.__assemble_localH(node, sanity_check=sanity_check)
            N = self.__assemble_localN(node, sanity_check=sanity_check)

            eigvals,eigvecs = gen_eigval_problem(
                H,
                N,
                eps=self.tikhonov_regularization_eps
            )

            # Finding the correct shape for the new site tensor.
            newshape = [np.nan for _ in self.psi.G.adj[node]]
            for neighbor in self.psi.G.adj[node]:
                leg = self._psi.G[node][neighbor][0]["legs"][node]
                newshape[leg] = self.psi.G[node][neighbor][0]["size"]
            newshape += [self.D[node],]

            # Re-shaping the statevector into a site tensor.
            T = np.reshape(eigvecs[:, np.argmin(eigvals)], newshape)

            # Inserting the new tensor and gauging the current node.
            self[node] = T
            if gauge:
                self.gauge(
                    sanity_check=sanity_check,
                    tree=self.oplist[0].tree,
                    nodes=(node,)
                )

            # Calculating new environments.
            self.__calculate_environments(
                sanity_check=sanity_check,
                iterator_desc_prefix="".join((
                    iterator_desc_prefix,
                    f"node {node}"
                )),
                **kwargs
            )

        Enext = self.E0

        return np.abs(Eprev - Enext)

    def run(
            self,
            nSweeps: int = None,
            verbose: bool = False,
            gauge: bool = True,
            compress: bool = True,
            sanity_check: bool = False,
            **kwargs
        ) -> None:
        """
        Runs single-site DMRG on the underlying braket. `kwargs` are
        passed to BP iterations. The state is not normalized afterwards!
        """
        if sanity_check: assert self.intact

        # preparing kwargs
        kwargs["verbose"] = False

        nSweeps = nSweeps if nSweeps != None else self.nSweeps
        iterator = tqdm.tqdm(
            range(nSweeps),
            desc="DMRG sweeps",
            disable=not verbose
        )
        eps_list = ()

        if gauge: self.gauge(sanity_check=sanity_check, **kwargs)

        for iSweep in iterator:
            eps = self.__sweep(
                gauge=gauge,
                sanity_check=sanity_check,
                **kwargs
            )
            iterator.set_postfix_str(f"eps = {eps:.3e}")
            eps_list += (eps,)

            if compress: self.compress(sanity_check=sanity_check, **kwargs)

        return

    def contract(self, sanity_check: bool = False) -> float:
        """
        Exact calculation of the energy expectation value.
        """
        allbrakets = self.brakets
        cntr = 0

        # Operator brakets of the constituent operators.
        for expval in allbrakets[:-1]:
            cntr += expval.contract(sanity_check=sanity_check)

        # Braket of the norm of the current state.
        cntr /= allbrakets[-1].contract(sanity_check=sanity_check)

        return cntr

    def gauge(
            self,
            method: str = "QR",
            sanity_check: bool = False,
            **kwargs
        ) -> None:
        """
        Gauges the current state. `kwargs` are passed to the respective
        gauging methods.
        """

        if method == "QR":
            self.psi = QR_gauging(
                self.psi, sanity_check=sanity_check, **kwargs
            )
            return

        if method == "Schmidt":
            if self.converged:
                # We have a converged set of messages that we can use for
                # gauging, instead of having to run a new BP iteration.
                overlap = self.brakets[-1]
                #overlap.msg = self._msg_overlap
                #overlap._converged = self._converged
                #overlap.write_cntr_value(sanity_check=sanity_check)
            else:
                overlap = None

            self.psi = L2BP_compression(
                self.psi,
                singval_threshold=0,
                overlap=overlap,
                return_singvals=False,
                sanity_check=sanity_check,
                **kwargs
            )
            return

        raise NotImplementedError(
            "Gauging method " + method + " not implemented."
        )

    def compress(
            self,
            method: str = "L2BP",
            sanity_check: bool = False,
            **kwargs
        ) -> None:
        """
        Compresses the current state. `kwargs` are passed to the
        respective compression methods.
        """

        if method == "L2BP":
            self.psi = L2BP_compression(
                psi=self.psi,
                return_singvals=False,
                sanity_check=sanity_check,
                **kwargs
            )

            return

        raise NotImplementedError(
            "Compression method " + method + " not implemented."
        )

    @property
    def converged(self) -> bool:
        """Whether the messages are converged."""
        return self._converged

    @property
    def psi(self) -> PEPS:
        """The current state of the system."""
        return self._psi

    @psi.setter
    def psi(self, newpsi: PEPS) -> None:
        """
        Changing the state of the system. Convergence marker is set to
        `False`.
        """
        self._psi = newpsi
        self._converged = False

        return

    @property
    def E0(self) -> float:
        """
        Current estimate of the ground state energy, up to loop series
        order `self.max_order`.
        """
        allbrakets = self.brakets

        if self.converged:
            # Converged sets of messages can be re-used, accelerating the
            # computation of E0 drastically.
            for i, msg_dict in enumerate(self._msg_oplist + [self._msg_overlap,]):
                allbrakets[i].msg = msg_dict
                allbrakets[i]._converged = self.converged
                allbrakets[i].write_cntr_value()

        cntr = 0
        for expval in allbrakets[:-1]:
            cntr += loop_series_contraction(
                braket=expval,
                excitations=self.excitations,
                max_order=self.max_order
            )

        cntr /= loop_series_contraction(
            braket=allbrakets[-1],
            excitations=self.excitations,
            max_order=self.max_order
        )

        return cntr

    @property
    def D(self) -> Dict[int, int]:
        """Physical dimension at every node."""
        return self.psi.D

    @property
    def nsites(self) -> int:
        """
        Number of sites of the system.
        """
        return self._psi.nsites

    @property
    def intact(self) -> bool:
        """
        Checks if the DMRG algrithm can be run. This amounts to:
        * Checking if the underlying state `self.psi` and the operators
        in `self.oplist` are intact.
        * If the current state `self.psi` and the operators in
        `self.oplist` are compatible.
        * Checking if the messages have the correct shapes.
        * Checking if all excitations are below the maximum order.
        """
        # Is the state intact?
        if not self.psi.intact: return False

        for i, op in enumerate(self.oplist):
            # Is this operator intact?
            if not op.intact:
                warnings.warn(
                    f"Operator {i} not intact.",
                    UserWarning
                )
                return False

            # Are state graph and this operator graph compatible, in terms of
            # geometry and physical dimension?
            if not graph_compatible(self.psi.G, op.G):
                warnings.warn(
                    f"Graphs of state and operator {i} not compatible.",
                    UserWarning
                )
                return False

            # Are the leg orderings the same?
            if not same_legs(op.G, self._psi.G):
                warnings.warn(
                    f"Leg orderings in state and operator {i} are different.",
                    UserWarning
                )
                return False

        # Are the messages on the overlap intact?
        if self._msg_overlap is not None:
            for node1, node2 in self._psi.G.edges():
                for sender, receiver in itertools.permutations((node1, node2)):
                    size = self.psi.G[sender][receiver][0]["size"]

                    if not check_msg_intact(
                        msg=self._msg_overlap[sender][receiver],
                        target_shape=(size, 1, size),
                        sender=sender,
                        receiver=receiver
                    ):
                        return False

        # Are the messages on the operators intact?
        for msg_dict, op in zip(self._msg_oplist, self.oplist):
            if msg_dict is not None:
                for node1, node2 in op.G.edges():
                    for sender, receiver in itertools.permutations(
                        (node1, node2)
                    ):
                        op_size = op.G[sender][receiver][0]["size"]
                        psi_size = self._psi.G[sender][receiver][0]["size"]

                        if not check_msg_intact(
                            msg=msg_dict[sender][receiver],
                            target_shape=(psi_size, op_size, psi_size),
                            sender=sender,
                            receiver=receiver
                        ):
                            return False

        # Are all the excitations below the maximum order?
        if not all(
            exc.number_of_edges <= self.max_order
            for exc in self.excitations
        ):
            warnings.warn(
                "There are excitations with weight above maximum order.",
                RuntimeWarning
            )
            return False

        return True

    @property
    def brakets(self) -> Tuple[Braket]:
        """
        Assembles the brakets for the operators in `self.oplist` and the
        overlap. Returns a tuple of `Braket` objects, where the operator
        brakets are contained first and the last entry is the overlap
        braket.
        """
        # Assembling operator brakets.
        allbrakets = tuple(
            Braket.Expval(
                psi=self.psi,
                op=op,
            )
            for op in self.oplist
        )

        # Assembling overlap braket.
        allbrakets += (Braket.Overlap(
            psi1=self.psi,
            psi2=self.psi,
        ),)

        return allbrakets

    def __getitem__(self, node: int) -> np.ndarray:
        """
        Subscripting with a node gives the ket tensor at this node.
        """
        return self._psi[node]

    def __setitem__(self, node: int, T: np.ndarray) -> None:
        """
        Changing local tensors in the state directly. Convergence
        marker will be set to `False`.
        """
        # updating the tensor stacks in all braket objects
        self._psi[node] = T
        self._converged = False

        return

    def __repr__(self) -> str:
        return "".join(
            (
                f"LoopSeriesDMRG problem on {self.nsites} sites.",
                "\nKet: " + str(self._psi) + "\nHamiltonians: "
            ) + (
                "".join(("\n",str(op)))
                for op in self.oplist
            ) + (
                "\nMessages are ",
                "converged." if self.converged else "not converged."
            ))

    def __len__(self) -> int: return self.nsites

    def __iter__(self) -> Iterator[int]:
        """
        Iterator over the nodes in the graph `self.G`.
        """
        return iter(self._psi.G.nodes(data=False))

    def __contains__(self, node: int) -> bool:
        """Does this DMRG problem involve the node `node`?"""
        return (node in self.psi) and all(node in op for op in self.oplist)

    def __init__(
            self,
            oplist: Tuple[PEPO],
            psi_init: PEPS = None,
            chi: int = None,
            max_order: int = 0,
            sanity_check: bool = False,
            **kwargs
        ):
        """
        Initialisation of a `LoopSeriesDMRG` object, where the state has
        bond dimension `chi`. The initial state is chosen randomly, if
        it is not given. `kwargs` are passed to `PEPS.init_random`.
        
        For `oplist = (H1, H2, ...)`, this object runs single-site DMRG
        on the Hamiltonian `H = H1 + H2 + ...`.
        """
        # sanity check
        if sanity_check:
            for op in oplist:
                # Is this operator intact?
                assert op.intact

                # Does this operator have the correct physical dimension?
                if not oplist[0].D == op.D:
                    raise ValueError(
                        "All operators must have the same physical dimension."
                    )

            if (psi_init is None) and (chi is None):
                raise ValueError("".join((
                    "Insufficient instructions on how to define the initial ",""
                    "state; psi_init and chi cannot both be None."
                )))

            if psi_init is not None:
                assert psi_init.intact
                assert oplist[0].D == psi_init.D

        # If not given, initial state is chosen randomly.
        if psi_init is None:
            psi_init = PEPS.init_random(
                G=op.G,
                D=oplist[0].D,
                chi=chi,
                **kwargs
            )

        self.oplist: Tuple[PEPO] = oplist
        """Constituent operators of the Hamiltonian."""

        self._psi: PEPS = psi_init
        """The current state of the system."""

        self.max_order: int = max_order
        """The order of the expansion in loop series excitations."""

        self._converged: bool = False
        """Is the current set of messages a converged set?"""

        # We obtain these messages from the brakets <psi|oplist[i]|psi> of the
        # constituent operators. Their direct sums are the messages on the
        # total operator.
        self._msg_oplist: List[Dict[int, Dict[int, np.ndarray]]] = [
            None for _ in oplist
        ]
        """Messages on the constituent operator brakets."""

        # The messages on the overlap.
        self._msg_overlap: Dict[int, Dict[int, np.ndarray]] = None
        """Messages on the overlap braket."""

        # The environments on the total hamiltonian.
        self._env_totalH:  Dict[int, np.ndarray] = None
        """Environments on the total hamiltonian."""

        # The environments on the overlap.
        self._env_overlap: Dict[int, np.ndarray] = None
        """Environments on the overlap."""

        # Local PEPO tensors of the total hamiltonian. Formed as direct sums of
        # constituent PEPO tensors.
        self._T_totalH: Dict[int, np.ndarray] = None
        """Local tensors of the total hamiltonian."""

        # Computing loop excitations up to max_order.
        self.excitations: Tuple[nx.MultiGraph] = BP_excitations(
            G=self._psi.G, max_order=self.max_order, sanity_check=sanity_check
        )
        """Excitations up to order `self.max_order`."""

        if sanity_check: assert self.intact

        return


def __msg_direct_sum(
        braket: Braket,
        nodes: Tuple[int] = None,
        sanity_check: bool = False
    ) -> Dict[int, np.ndarray]:
    """
    Calculates the environments at the sites from `nodes` by taking the
    tensor product of the inbound messages.
    """
    # sanity check
    if sanity_check: assert braket.intact

    if nodes is None: nodes = tuple(braket)
    environments = {}

    for node in nodes:
        nLegs = len(braket.G.adj[node])

        if 3 * nLegs > np.MAXDIMS:
            # This node has too many neighbors; numpy would run out of
            # array dimensions.
            raise RuntimeError("".join((
                f"Node {node} has {nLegs} neighbors, which would lead to ",
                f"an environment with {3 * nLegs} dimensions. Numpy can ",
                f"only handle arrays with up to {np.MAXDIMS} dimensions."
            )))

        # Assembling einsum arguments.
        out_legs = tuple(range(3 * nLegs))
        args = ()
        for neighbor in braket.G.adj[node]:
            leg = braket.G[node][neighbor][0]["legs"][node]
            args += (
                braket.msg[neighbor][node],
                (leg, nLegs + leg, 2*nLegs + leg)
            )
        env = ctg.einsum(*args, out_legs)

        # Normalization to contraction value.
        env *= (braket.cntr / braket.G.nodes[node]["cntr"])

        environments[node] = env

    return environments


def loop_series_environments(
        braket: Braket,
        excitations: Tuple[nx.MultiGraph] = None,
        nodes: Tuple[int] = None,
        max_order: int = np.inf,
        verbose: bool = False,
        sanity_check: bool = False
    ) -> Dict[int, np.ndarray]:
    """
    Calculates the environments at the sites from `nodes`. If `nodes
    = None` (default), calculates environments at every site.

    The environment at a node is a tensor with `3 * len(adj[node])`
    legs. It's legs come in three groups: First the bra legs, followed
    by the operator legs, and finally the ket legs. Each group contains
    `len(adj[node])` legs, leading to the total of `3 * len(adj[node])`
    legs. The ordering within the groups follows the leg ordering of the
    tensor at the respective node in `self._psi`.
    """
    # sanity check
    if sanity_check: assert braket.intact
    if not braket.converged:
        warnings.warn(
            "Calculating environments from non-converged messages.",
            RuntimeWarning
        )

    if max_order == 0:
        # Loop series environments with zero-weight excitations involve only
        # the BP vacuum, i.e. the messages from the BP iteration. The
        # environments are the tensor products of the messages.

        return __msg_direct_sum(
            braket=braket,
            nodes=nodes,
            sanity_check=sanity_check
        )

    raise NotImplementedError


def environments_direct_sum(
        envs: Tuple[np.ndarray]
    ) -> np.ndarray:
    """
    Evaluates the direct sum of the environments in `envs`. The
    environments are composed by a direct sum in the operator virtual
    bond dimensions.
    """

    nLegs = envs[0].ndim // 3

    # sanity check
    if not all(env.ndim == 3 * nLegs for env in envs):
        raise ValueError("Environments have mismatches in numbers of legs.")
    if not all(
        all(env.shape[i] == env.shape[2*nLegs + i] for i in range(nLegs))
        for env in envs
    ):
        raise ValueError(
            "Environments have mismatches in physical dimensions."
        )
    if not all(
        all(envs[0].shape[i] == envs[j].shape[i] for j in range(len(envs)))
        for i in tuple(range(nLegs)) + tuple(range(2*nLegs, 3*nLegs))
    ):
        raise ValueError(
            "Environments have mismatches in physical dimensions."
        )

    # Summing over operator bond dimensions for every environment, to obtain
    # the total resulting operator bond dimensions.
    chi = tuple(
        sum(env.shape[nLegs + i] for env in envs)
        for i in range(nLegs)
    )

    # Scaffolding for the resulting environment.
    newshape = (tuple(envs[0].shape[:nLegs])
                + chi
                + tuple(envs[0].shape[2 * nLegs:]))
    sum_env = np.zeros(
        shape=newshape,
        dtype=np.complex128
    )

    full_chi = tuple(0 for _ in range(nLegs))
    fill_state_dims = tuple(
        slice(chi_)
        for chi_ in envs[0].shape[:nLegs]
    )

    # Filling the environment with the environments from the list.
    for env in envs:
        # Defining the slice that this environment will be inserted in.
        fill_op_dims = tuple(
            slice(full_chi_, full_chi_ + chi_)
            for full_chi_, chi_ in zip(full_chi, env.shape[nLegs: 2*nLegs])
        )
        fill_dims = fill_state_dims + fill_op_dims + fill_state_dims

        # Inserting this environment in the slice.
        sum_env[fill_dims] = env

        # Incrementing the filled operator dimensions.
        full_chi = tuple(
            full_chi_ + chi_
            for full_chi_, chi_ in zip(full_chi, env.shape[nLegs: 2*nLegs])
        )

    return sum_env

if __name__ == "__main__":
    pass
