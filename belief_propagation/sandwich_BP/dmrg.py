"""
DMRG on Braket-objects. Contains the DMRG-class.
"""

import numpy as np
import networkx as nx
import cotengra as ctg
import warnings
import tqdm
import copy

from belief_propagation.utils import is_hermitian,gen_eigval_problem,rel_err
from belief_propagation.sandwich_BP.PEPO import PEPO
from belief_propagation.sandwich_BP.PEPS import PEPS
from belief_propagation.sandwich_BP.braket import Braket,L2BP_compression

class DMRG:
    """
    Single-site DMRG on graphs. Environments are calculated using belief propagation.
    """

    nSweeps:int=5
    """Default number of sweeps."""
    hermiticity_threshold:float=1e-6
    """Allowed deviation from exact hermiticity."""
    tikhonov_regularization_eps:float=1e-6
    """Epsilon for Tikhonov-regularization of singular messages."""

    def local_H(self,node:int,threshold:float=hermiticity_threshold,sanity_check:bool=False) -> np.ndarray:
        """
        Hamiltonian at `node`, by taking inbound messages to be environments.
        `threshold` is the absolute allowed error in the hermiticity of the
        hamiltonian obtained (checked if `sanity_check=True`).
        """
        # sanity check
        if sanity_check: assert self.intact

        # The hamiltonian at node is obtained by tracing out the rest of the network. The environments are approximated by messages
        nLegs = len(self.expval.G.adj[node])
        args = ()
        """Arguments for einsum"""

        out_legs = tuple(range(nLegs)) + (3*nLegs,) + tuple(range(2*nLegs,3*nLegs)) + (3*nLegs+1,)
        # Leg ordering of the local hamiltonian: (bra virtual dimensions, bra physical dimension, ket virtual dimensions, ket physical dimension).
        # The order of the virtual legs is inherited from the "legs" indices on the edges
        vir_dim = 1

        for neighbor in self.expval.G.adj[node]:
            # collecting einsum arguments
            args += (
                self.expval.msg[neighbor][node],
                (
                    self.expval.bra.G[node][neighbor][0]["legs"][node], # bra leg
                    nLegs + self.expval.op.G[node][neighbor][0]["legs"][node], # operator leg
                    2 * nLegs + self.expval.ket.G[node][neighbor][0]["legs"][node], # ket leg
                )
            )

            # compiling virtual dimensions for later reshape
            vir_dim *= self.expval.ket.G[node][neighbor][0]["size"]

        args += (
            # operator tensor
            self.expval.op[node],
            tuple(nLegs + iLeg for iLeg in range(nLegs)) + (3*nLegs,3*nLegs+1),
        )

        H = np.einsum(*args,out_legs,optimize=True)
        H = np.reshape(H,(vir_dim * self.expval.D,vir_dim * self.expval.D))

        if sanity_check: assert is_hermitian(H,threshold=threshold,verbose=False)

        return H

    def local_env(self,node:int,threshold:float=hermiticity_threshold,regularize:bool=True,sanity_check:bool=False) -> np.ndarray:
        """
        Environment at node. Calculated from `self.overlap`. This amounts to
        stacking and re-shaping messages, that are inbound to `node`.
        `threshold` is the absolute allowed error in the hermiticity of the
        resulting matrix (checked if `sanity_check=True`).

        Messages from leafs are singular, in which case Tikhonov-regularization
        is performed (if `regularize=True`).
        """
        # sanity check
        if sanity_check: assert self.intact

        nLegs = len(self.overlap.G.adj[node])
        args = ()
        """Arguments for einsum"""

        out_legs = tuple(range(nLegs)) + (3*nLegs,) + tuple(range(2*nLegs,3*nLegs)) + (3*nLegs+1,)
        # Leg ordering of the local hamiltonian: (bra virtual dimensions, bra physical dimension, ket virtual dimensions, ket physical dimension)
        # The order of the virtual dimensions is inherited from the "legs" indices on the edges
        vir_dim = 1

        for neighbor in self.overlap.G.adj[node]:
            if not self.overlap.msg[neighbor][node].shape[1] == 1: warnings.warn(f"Message {neighbor} -> {node} does not correspond to an overlap!")
            msg = self.overlap.msg[neighbor][node][:,0,:]

            eigvals = np.linalg.eigvals(msg)
            nonzero_mask = np.logical_not(np.isclose(eigvals,0))
            rank = np.sum(nonzero_mask)
            edge_size = self.overlap.ket.G[node][neighbor][0]["size"]
            print(f"    Message {neighbor} -> {node} has rank {rank} on edge of size {edge_size}. Cond.number {np.linalg.cond(msg):.5e}")

            # collecting einsum arguments
            args += (
                msg,
                (
                    self.overlap.bra.G[node][neighbor][0]["legs"][node], # bra leg
                    2 * nLegs + self.overlap.ket.G[node][neighbor][0]["legs"][node], # ket leg
                )
            )

            # compiling virtual dimensions for later reshape
            vir_dim *= self.overlap.ket.G[node][neighbor][0]["size"]

        # identity for the physical dimension
        args += (self.overlap.op.I,(3*nLegs,3*nLegs+1))

        N = np.einsum(*args,out_legs,optimize=True)
        N = np.reshape(N,(vir_dim * self.overlap.D,vir_dim * self.overlap.D))

        if sanity_check: assert is_hermitian(N,threshold=threshold)

        return N

    def sweep(self,sanity_check:bool=False,normalize:bool=True,**kwargs) -> float:
        """
        Local update at all sites. `kwargs` are passed to `Braket.BP`.
        Returns the change in energy after the sweep.

        The graph is traversed in breadth-first manner. After each
        local update, new outgoing messages are calculated,
        thereby updating the environments.
        """
        if sanity_check: assert self.intact

        # calculating environments and previous energy
        if not self.overlap.converged: self.overlap.BP(normalize=normalize,sanity_check=sanity_check,**kwargs)
        if not self.expval.converged: self.expval.BP(normalize=normalize,sanity_check=sanity_check,**kwargs)
        Eprev = self.E0

        # we'll update tensors and matrices, so as a precaution, we'll set the convergence markers to False
        self.overlap.converged = False
        self.expval.converged = False

        for node in nx.dfs_postorder_nodes(self.expval.G,source=self.expval.op.root):
            H = self.local_H(node,sanity_check=sanity_check)
            N = self.local_env(node,sanity_check=sanity_check)
            print(f"Condition number node {node}: {np.linalg.cond(N):.5e}")

            if sanity_check: # are local hamiltonian and environment correctly defined?
                local_psi = self.overlap.ket[node].flatten()
                expval_local_cntr = ctg.einsum("i,ik,k",local_psi.conj(),H,local_psi)
                overlap_local_cntr = ctg.einsum("i,ik,k",local_psi.conj(),N,local_psi)

                if not np.isclose(expval_local_cntr,self.expval.cntr):
                    warnings.warn(f"Local hamiltonian at node {node} does not reproduce expectation value. Relative error {rel_err(self.expval.cntr,expval_local_cntr):.3e}.")
                if not np.isclose(overlap_local_cntr,self.overlap.cntr):
                    warnings.warn(f"Local environment at node {node} does not reproduce overlap. Relative error {rel_err(self.overlap.cntr,overlap_local_cntr):.3e}.")

            eigvals,eigvecs = gen_eigval_problem(H,N,eps=self.tikhonov_regularization_eps)

            # re-shaping new statevector
            newshape = [np.nan for neighbor in self.overlap.G.adj[node]]
            for neighbor in self.overlap.G.adj[node]: newshape[self.overlap.G[node][neighbor][0]["legs"][node]] = self.overlap.ket.G[node][neighbor][0]["size"]
            newshape += [self.overlap.D,]
            T = np.reshape(eigvecs[:,np.argmin(eigvals)],newshape)

            # inserting it into PEPS and PEPO
            self.overlap.ket[node] = T
            self.overlap.bra[node] = T.conj()
            self.expval.ket[node] = T
            self.expval.bra[node] = T.conj()

            # calculating new environments
            self.overlap.BP(sanity_check=sanity_check,**kwargs)
            self.expval.BP(sanity_check=sanity_check,**kwargs)
        Enext = self.E0

        return np.abs(Eprev - Enext)

    def run(self,nSweeps:int=None,verbose:bool=False,compress:bool=True,sanity_check:bool=False,**kwargs):
        """
        Runs single-site DMRG on the underlying braket.
        `kwargs` are passed to BP iterations. The state is not normalized afterwards!
        """
        if sanity_check: assert self.intact

        # preparing kwargs
        kwargs["numretries"] = np.inf
        kwargs["verbose"] = False

        nSweeps = nSweeps if nSweeps != None else self.nSweeps
        iterator = tqdm.tqdm(range(nSweeps),desc=f"DMRG sweeps",disable=not verbose)
        eps_list = ()

        for iSweep in iterator:
            eps = self.sweep(sanity_check=sanity_check,**kwargs)
            iterator.set_postfix_str(f"eps = {eps:.3e}")
            eps_list += (eps,)

            print(f"After sweep {iSweep}: ",self)

            if compress:
                # L2BP compression
                L2BP_compression(self.overlap.ket,sanity_check=sanity_check,**kwargs)
                self.expval.ket = self.overlap.ket
                self.overlap.bra = self.overlap.ket.conj(sanity_check=sanity_check)
                self.expval.bra = self.expval.ket.conj(sanity_check=sanity_check)
                self.overlap.converged = False
                self.expval.converged = False

        return

    def contract(self,sanity_check:bool=True) -> float:
        """
        Exact calculation of the current expectation value.
        """
        expval_cntr = self.expval.contract(sanity_check=sanity_check)
        overlap_cntr = self.overlap.contract(sanity_check=sanity_check)

        return expval_cntr / overlap_cntr

    @property
    def converged(self) -> bool:
        """Indicates whether the messages in `self.overlap.G` and `self.expval.G` are converged."""
        return self.overlap.converged and self.expval.converged

    @property
    def E0(self) -> float:
        """Current best guess of the ground state energy."""
        return self.expval.cntr / self.overlap.cntr

    @property
    def nsites(self) -> int:
        """
        Number of sites on which the braket is defined.
        """
        return self.overlap.nsites

    @property
    def intact(self) -> bool:
        """
        Checks if the DMRG algrithm can be run. This amounts to:
        * Checking if the underlying braket is intact.
        + Checking if expval graph and overlap graph are compatible.
        * Checking if bra and ket are adjoint to one another.
        * Checking if the leg orderings in `self.overlap` and `self.expval` are the same.
        """
        if not self.expval.intact: return False
        if not self.overlap.intact: return False

        # re the physical dimensions the same?
        if not self.overlap.D == self.expval.D:
            warnings.warn("Physical dimensions do not match.")
            return False

        # are expval graph and overlap graph compatible?
        if not Braket.graph_compatible(self.expval.G,self.overlap.G):
            warnings.warn("Graphs of overlap and expval not compatible.")
            return False

        # are bra and ket adjoint?
        for node in self.overlap.G.nodes():
            if not np.allclose(self.overlap.bra[node].conj(),self.overlap.ket[node]):
                warnings.warn(f"Bra- and ket-tensors at node {node} in overlap not complex conjugates of one another.")
                return False
        for node in self.expval.G.nodes():
            if not np.allclose(self.expval.bra[node].conj(),self.expval.ket[node]):
                warnings.warn(f"Bra- and ket-tensors at node {node} in expval not complex conjugates of one another.")
                return False

        # do overlap and expval contain the same tensors?
        for node in self.expval.G.nodes():
            if not np.allclose(self.expval.ket[node],self.overlap.ket[node]):
                warnings.warn(f"Tensor at node {node} is not the same in overlap and expval.")
                return False

        # are the leg orderings the same?
        for node1,node2,legs in self.expval.G.edges(data="legs"):
            if not self.overlap.G.has_edge(node1,node2):
                warnings.warn(f"Edge ({node1},{node2}) present in expval, but not present in overlap.")
                return False
            if self.expval.G[node1][node2][0]["legs"] != legs:
                warnings.warn(f"Leg indices of edge ({node1},{node2}) different in expval and overlap.")
                return False

        return True

    def __init__(self,op:PEPO,psi_init:PEPS=None,chi:int=None,sanity_check:bool=False,**kwargs):
        """
        Initialisation of a `DMRG` object. The initial is chosen randomly,
        if it is not given. `kwargs` are passed to `PEPS.init_random`.
        """
        # if not given, initial state is chosen randomly
        if psi_init == None:
            psi_init = PEPS.init_random(G=op.G,D=op.D,chi=chi,**kwargs)

        self.expval = Braket.Expval(psi=psi_init,op=op,sanity_check=sanity_check)
        """`Braket`-object that contains the operator."""
        self.overlap = Braket.Overlap(psi1=psi_init,psi2=psi_init)
        """Norm of the current state."""

        if sanity_check: assert self.intact

    def __repr__(self) -> str:
        out = f"----------------------   DMRG problem on {self.nsites} sites.   ----------------------\nKet: " + str(self.overlap.ket) + "\nHamiltonian: " + str(self.expval.op) + "\nMessages are " + "converged." if self.converged else "not converged."
        return out

if __name__ == "__main__":
    pass
