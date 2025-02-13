"""
DMRG on Braket-objects. Contains the DMRG-class.
"""

import numpy as np
import networkx as nx
import cotengra as ctg
import warnings
import tqdm
import itertools
import copy

from belief_propagation.utils import is_hermitian,gen_eigval_problem,rel_err,same_legs
from belief_propagation.PEPO import PEPO
from belief_propagation.PEPS import PEPS
from belief_propagation.braket import Braket,L2BP_compression,QR_gauging

class DMRG:
    """
    Single-site DMRG on graphs, with a sum hamiltonian. Environments are calculated using belief propagation.
    """

    nSweeps:int=5
    """Default number of sweeps."""
    hermiticity_threshold:float=1e-6
    """Allowed deviation from exact hermiticity."""
    tikhonov_regularization_eps:float=1e-6
    """Epsilon for Tikhonov-regularization of singular messages."""

    def __assemble_messages(self,sanity_check:bool=False) -> None:
        """
        Evaluates the direct product of messages, and writes to `self._msg`.
        """
        if sanity_check: assert self.intact

        self._msg = {node:{node_:None for node_ in self.overlap.G.nodes()} for node in self.overlap.G.nodes()}

        for node1,node2 in self.overlap.G.edges():
            for sending_node,receiving_node in itertools.permutations((node1,node2)):
                chi = sum(expval.op.chi for expval in self.expvals)
                msg = np.full(shape=(self.overlap.bra.G[sending_node][receiving_node][0]["size"],chi,self.overlap.ket.G[sending_node][receiving_node][0]["size"]),fill_value=np.inf) + 0j

                full_chi = 0
                for expval in self.expvals:
                    msg[:,full_chi:full_chi+expval.op.chi,:] = expval.msg[sending_node][receiving_node]
                    full_chi += expval.op.chi

                assert np.all(np.isfinite(msg))
                self._msg[sending_node][receiving_node] = msg

        return

    def __assemble_total_op_T(self,sanity_check:bool=False) -> None:
        """
        Evaluates the direct product of PEPO tensors,
        and writes to `self._total_op_T`.
        """
        if sanity_check: assert self.intact

        self._total_op_T = {node:None for node in self.overlap.G.nodes()}

        chi = sum(expval.op.chi for expval in self.expvals)
        for node in self.overlap.G.nodes():
            H = np.zeros(shape=tuple(chi for _ in self.overlap.G.adj[node]) + (self.D,self.D)) + 0j

            full_chi = 0
            # filling the Hamiltonian
            for expval in self.expvals:
                index = tuple(slice(full_chi,full_chi+expval.op.chi) for _ in self.overlap.G.adj[node]) + (slice(0,self.D),slice(0,self.D))
                H[index] = expval.op[node]
                full_chi += expval.op.chi

            self._total_op_T[node] = H

        return

    def local_H(self,node:int,threshold:float=hermiticity_threshold,sanity_check:bool=False) -> np.ndarray:
        """
        Hamiltonian at `node`, by taking inbound messages to be environments.
        `threshold` is the absolute allowed error in the hermiticity of the
        hamiltonian obtained (checked if `sanity_check=True`).
        """
        # sanity check
        if sanity_check: assert self.intact

        # construct total PEPO tensors, if necessary
        if self._total_op_T == None: self.__assemble_total_op_T(sanity_check=sanity_check)

        # The hamiltonian at node is obtained by tracing out the rest of the network. The environments are approximated by messages
        nLegs = len(self.overlap.G.adj[node])
        args = ()
        """Arguments for einsum"""

        out_legs = tuple(range(nLegs)) + (3*nLegs,) + tuple(range(2*nLegs,3*nLegs)) + (3*nLegs+1,)
        # Leg ordering of the local hamiltonian: (bra virtual dimensions, bra physical dimension, ket virtual dimensions, ket physical dimension).
        # The order of the virtual legs is inherited from the "legs" indices on the edges
        vir_dim = 1

        for neighbor in self.overlap.G.adj[node]:
            # collecting einsum arguments
            args += (
                self.msg[neighbor][node],
                (
                    self.overlap.bra.G[node][neighbor][0]["legs"][node], # bra leg
                    nLegs + self.expvals[0].op.G[node][neighbor][0]["legs"][node], # operator leg
                    2 * nLegs + self.overlap.ket.G[node][neighbor][0]["legs"][node], # ket leg
                )
            )

            # compiling virtual dimensions for later reshape
            vir_dim *= self.overlap.ket.G[node][neighbor][0]["size"]

        args += (
            # operator tensor
            self._total_op_T[node],
            tuple(nLegs + iLeg for iLeg in range(nLegs)) + (3*nLegs,3*nLegs+1),
        )

        H = np.einsum(*args,out_legs,optimize=True)
        H = np.reshape(H,(vir_dim * self.D,vir_dim * self.D))

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
            if not self.overlap.msg[neighbor][node].shape[1] == 1: warnings.warn(f"Message {neighbor} -> {node} does not originate from an overlap!")
            msg = self.overlap.msg[neighbor][node][:,0,:]

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
        N = np.reshape(N,(vir_dim * self.D,vir_dim * self.D))

        if sanity_check: assert is_hermitian(N,threshold=threshold)

        return N

    def BP(self,sanity_check:bool=False,**kwargs) -> None:
        """
        BP iteration on the overlap and all expvals. Total messages
        and local PEPO tensors are formed, if BP converged.
        """
        if not self.overlap.converged: self.overlap.BP(normalize_after=True,sanity_check=sanity_check,**kwargs)
        for i in range(len(self.expvals)):
            if not self.expvals[i].converged: self.expvals[i].BP(normalize_after=True,sanity_check=sanity_check,**kwargs)

        if self.converged:
            self.__assemble_messages(sanity_check=sanity_check)
        else:
            warnings.warn("BP iteration not converged.")
            self._msg = None

        return

    def __sweep(self,gauge:bool,sanity_check:bool,**kwargs) -> float:
        """
        Local update at all sites. `kwargs` are passed to `Braket.BP`.
        Returns the change in energy after the sweep.

        The graph is traversed in breadth-first manner. After each
        local update, new outgoing messages are calculated,
        thereby updating the environments.
        """
        if sanity_check: assert self.intact

        # calculating environments and previous energy
        self.BP(sanity_check=sanity_check,**kwargs)
        Eprev = self.E0

        for node in nx.dfs_postorder_nodes(self.expvals[0].op.tree,source=self.expvals[0].op.root):
            H = self.local_H(node,sanity_check=sanity_check)
            N = self.local_env(node,sanity_check=sanity_check)

            if sanity_check: # are local hamiltonian and environment correctly defined?
                local_psi = self.ket[node].flatten()
                expval_local_cntr = ctg.einsum("i,ik,k",local_psi.conj(),H,local_psi)
                overlap_local_cntr = ctg.einsum("i,ik,k",local_psi.conj(),N,local_psi)
                expvals_total_cntr = sum(expval.cntr for expval in self.expvals)

                if not np.isclose(expval_local_cntr,expvals_total_cntr):
                    warnings.warn(f"Local hamiltonian at node {node} does not reproduce expectation value. Relative error {rel_err(expvals_total_cntr,expval_local_cntr):.3e}.")
                if not np.isclose(overlap_local_cntr,self.overlap.cntr):
                    warnings.warn(f"Local environment at node {node} does not reproduce overlap. Relative error {rel_err(self.overlap.cntr,overlap_local_cntr):.3e}.")

            eigvals,eigvecs = gen_eigval_problem(H,N,eps=self.tikhonov_regularization_eps)

            # re-shaping new statevector
            newshape = [np.nan for neighbor in self.overlap.G.adj[node]]
            for neighbor in self.overlap.G.adj[node]: newshape[self.overlap.G[node][neighbor][0]["legs"][node]] = self.overlap.ket.G[node][neighbor][0]["size"]
            newshape += [self.overlap.D,]
            T = np.reshape(eigvecs[:,np.argmin(eigvals)],newshape)

            # inserting the new tensor and gauging the current node
            self[node] = T
            if gauge: self.gauge(sanity_check=sanity_check,tree=self.expvals[0].op.tree,nodes=(node,))

            # calculating new environments
            self.BP(sanity_check=sanity_check,**kwargs)
        Enext = self.E0

        return np.abs(Eprev - Enext)

    def run(self,nSweeps:int=None,verbose:bool=False,gauge:bool=True,compress:bool=True,sanity_check:bool=False,**kwargs):
        """
        Runs single-site DMRG on the underlying braket. `kwargs` are passed to
        BP iterations. The state is not normalized afterwards!
        """
        if sanity_check: assert self.intact

        # preparing kwargs
        kwargs["numretries"] = np.inf
        kwargs["verbose"] = False

        nSweeps = nSweeps if nSweeps != None else self.nSweeps
        iterator = tqdm.tqdm(range(nSweeps),desc=f"DMRG sweeps",disable=not verbose)
        eps_list = ()

        if gauge: self.gauge(sanity_check=sanity_check,**kwargs)

        for iSweep in iterator:
            eps = self.__sweep(gauge=gauge,sanity_check=sanity_check,**kwargs)
            iterator.set_postfix_str(f"eps = {eps:.3e}")
            eps_list += (eps,)

            if compress: self.compress(sanity_check=sanity_check,**kwargs)

        return

    def contract(self,sanity_check:bool=True) -> float:
        """
        Exact calculation of the current expectation value.
        """
        cntr = 0
        for expval in self.expvals: cntr += expval.contract(sanity_check=sanity_check)
        cntr /= self.overlap.contract(sanity_check=sanity_check)

        return cntr

    def gauge(self,method:str="QR",sanity_check:bool=False,**kwargs) -> None:
        """
        Gauges bra and ket. `kwargs` are passed to the respective gauging methods.
        """

        if method == "QR":
            ket_gauged = copy.deepcopy(self.ket)
            QR_gauging(ket_gauged,sanity_check=sanity_check,**kwargs)
            self.ket = ket_gauged

            return

        raise NotImplementedError("Gauging method " + method + " not implemented.")

    def compress(self,method:str="L2BP",sanity_check:bool=False,**kwargs) -> None:
        """
        Compresses bra and ket. `kwargs` are passed to the respective compression methods.
        """

        if method == "L2BP":
            ket_cmpr = copy.deepcopy(self.ket)
            L2BP_compression(ket_cmpr,sanity_check=sanity_check,**kwargs)
            self.ket = ket_cmpr

            return

        raise NotImplementedError("Compression method " + method + " not implemented.")

    @property
    def msg(self) -> dict[int,dict[int,np.ndarray]]:
        """
        Messages on the total expval. These are formed as
        direct products of messages on the individual expvals.
        """
        if self._msg == None: self.__assemble_messages()
        return self._msg

    @property
    def converged(self) -> bool:
        """Whether the messages are converged."""
        return self.overlap.converged and all(expval.converged for expval in self.expvals)

    @property
    def ket(self) -> PEPS:
        """The current state of the system."""
        return self.overlap.ket

    @ket.setter
    def ket(self,newket:PEPS) -> None:
        """
        Changing the state of the system requires inserting a new PEPS in all `Braket` objects.
        Convergence markers will be set to `False`.
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
        * Checking if the leg orderings in `self.overlap` and `self.expvals` are the same.
        """
        # are the brakets compatible?
        for expval in self.expvals:
            if not expval.intact: return False
            if expval.D != self.D: return False
        if not self.overlap.intact: return False

        # are the physical dimensions the same?
        if not self.overlap.D == self.D:
            warnings.warn("Physical dimensions do not match.")
            return False

        # are expval graphs and overlap graph compatible?
        for i,expval in enumerate(self.expvals):
            if not Braket.graph_compatible(expval.G,self.overlap.G):
                warnings.warn(f"Graphs of overlap and expval {i} not compatible.")
                return False

        # are bra and ket adjoint?
        for node in self.overlap.G.nodes():
            if not np.allclose(self.overlap.bra[node].conj(),self.overlap.ket[node]):
                warnings.warn(f"Bra- and ket-tensors at node {node} in overlap not complex conjugates of one another.")
                return False
        for i,expval in enumerate(self.expvals):
            for node in expval.G.nodes():
                if not np.allclose(expval.bra[node].conj(),expval.ket[node]):
                    warnings.warn(f"Bra- and ket-tensors at node {node} in expval {i} not complex conjugates of one another.")
                    return False

        # do overlap and expval contain the same tensors?
        for i,expval in enumerate(self.expvals):
            for node in expval.G.nodes():
                if not np.allclose(expval.ket[node],self.overlap.ket[node]):
                    warnings.warn(f"Tensor at node {node} is not the same in overlap and expval {i}.")
                    return False

        # are the leg orderings the same?
        for i,expval in enumerate(self.expvals):
            if not same_legs(self.overlap.G,expval.G):
                warnings.warn(f"Leg orderings in overlap and expval {i} are different.")
                return False

        return True

    def __init__(self,oplist:tuple[PEPO],psi_init:PEPS=None,chi:int=None,sanity_check:bool=False,**kwargs):
        """
        Initialisation of a `DMRG` object, where the state has bond dimension `chi`.
        The initial is chosen randomly, if it is not given. `kwargs` are passed to `PEPS.init_random`.
        
        For `oplist = (H1,H2,...)`, this object runs single-site DMRG on the Hamiltonian `H = H1 + H2 + ...`.
        """
        # sanity check
        if sanity_check:
            for op in oplist:
                assert op.intact
                if not oplist[0].D == op.D: raise ValueError("All operators must have the same physical dimension.")
            if psi_init != None:
                assert psi_init.intact
                assert oplist[0].D == psi_init.D

        # if not given, initial state is chosen randomly
        if psi_init == None:
            psi_init = PEPS.init_random(G=op.G,D=oplist[0].D,chi=chi,**kwargs)

        self.D = oplist[0].D
        """Physical dimension."""
        self.expvals = tuple(Braket.Expval(psi=psi_init,op=op,sanity_check=sanity_check) for op in oplist)
        """`Braket`-objects for every operator."""
        self.overlap:Braket = Braket.Overlap(psi1=psi_init,psi2=psi_init)
        """Norm of the current state."""

        self._msg:dict[int,dict[int,np.ndarray]] = None
        """Messages on the total expval. Formed as direct products of expval messages."""
        self._total_op_T:dict[int,np.ndarray] = None
        """Total local PEPO tensors. Formed as direct products of constituent PEPO tensors."""

        if sanity_check: assert self.intact

    def __getitem__(self,node:int) -> np.ndarray:
        """
        Subscripting with a node gives the ket tensor at this node.
        """
        return self.overlap.ket[node]

    def __setitem__(self,node:int,T:np.ndarray) -> None:
        """
        Changing local tensors in the state directly.
        Convergence markers will be set to `False`.
        """
        # updating the tensor stacks in all braket objects
        self.overlap[node] = (T.conj(),self.overlap[node][1],T)
        for i in range(len(self.expvals)):
            self.expvals[i][node] = (T.conj(),self.expvals[i][node][1],T)

        return

    def __repr__(self) -> str:
        out = f"DMRG problem on {self.nsites} sites.\nKet: " + str(self.overlap.ket) + "\nHamiltonians: "
        for expval in self.expvals: out += "\n" + str(expval.op)
        out += "\nMessages are "
        out += "converged." if self.converged else "not converged."
        return out

if __name__ == "__main__":
    pass
