"""
Functions for manipulation of PEPS, PEPO and Braket,
with the goal of cutting / truncating edges in the
graphs.
"""

__all__ = [
    "make_BP_informed",
    "project_down",
    "project_out",
    "QR_bottleneck",
    "L2BP_compression",
    "QR_gauging"
]

import itertools
import copy
from typing import Callable,Union
import warnings

import numpy as np
import networkx as nx
import scipy.linalg as scialg

from belief_propagation.braket import Braket
from belief_propagation.PEPO import PEPO
from belief_propagation.PEPS import PEPS
from belief_propagation.networks import expose_edge
from belief_propagation.utils import delta_tensor, graph_compatible


def make_BP_informed(truncation_func: Callable) -> Callable:
    """
    Gives back a version of the truncation function that executes a BP
    iteration beforehand, and uses the messages as arguments to
    `truncation_func`, thereby making it BP-informed. The call signature
    of `truncation_func` must be `truncation_func(braket: Braket,
    node1: int, node2: int, vec1: np.ndarray, vec2: np.ndarray,
    sanity_check: bool = False)`.

    Returns a function with call signature
    `BP_informed_truncation_func(braket: Braket, node1: int, node2: int,
    sanity_check: bool = False, **kwargs)`, where `kwargs` are passed to
    the BP iteration. `braket` contains the messages that will be used
    for truncation of the edge `(node1, node2)`.
    """

    def BP_informed_truncation_func(
            braket: Braket,
            node1: int,
            node2: int,
            sanity_check: bool = False,
            **kwargs
        ) -> None:
        # sanity check
        assert braket.G.has_edge(node1, node2, key=0)
        if sanity_check: assert braket.intact

        # BP iteration, to obtain messages for truncation.
        if not braket.converged:
            braket.BP(
                sanity_check=sanity_check, normalize_after=False, **kwargs
            )
        if not braket.converged:
            warnings.warn(
                "Truncation on edge with non-converged messages.",
                RuntimeWarning
            )

        # inserting truncation into the Braket
        truncation_func(
            braket=braket,
            node1=node1,
            node2=node2,
            vec1=braket.msg[node1][node2],
            vec2=braket.msg[node2][node1],
            sanity_check=sanity_check
        )

        return

    return BP_informed_truncation_func


@make_BP_informed
def project_down(
        braket: Braket,
        node1: int,
        node2: int,
        vec1: np.ndarray,
        vec2: np.ndarray,
        sanity_check: bool = False
    ) -> None:
    """
    Projects the edge `(node1, node2)` onto the given messages. `vec1`
    is the message from `node1` to `node2`, `vec2` is defined
    accordingly. Both vectors must have the shape `(braket.bra.size,
    braket.op.size, braket.ket.size)`. This effectively removes the
    edge.
    """
    # sanity check
    vec_size = (
        braket.bra.G[node1][node2][0]["size"],
        braket.op.G[node1][node2][0]["size"],
        braket.ket.G[node1][node2][0]["size"]
    )
    if not vec1.shape == vec_size:
        raise ValueError("".join((
            "vec1 has wrong size; expected ",
            str(vec_size),
            ", got ",
            str(vec1.shape),
            "."
        )))
    if not vec2.shape == vec_size:
        raise ValueError("".join((
            "vec2 has wrong size; expected ",
            str(vec_size),
            ", got ",
            str(vec2.shape),
            "."
        )))
    if sanity_check: assert braket.intact

    P1 = np.einsum("ijk, lmn -> ijklmn", vec2, vec1)
    P2 = np.einsum("ijk, lmn -> ijklmn", vec1, vec2)
    braket.insert_edge_T(P1, node1, node2, sanity_check=sanity_check)
    braket.insert_edge_T(P2, node2, node1, sanity_check=sanity_check)

    return


@make_BP_informed
def project_out(
        braket: Braket,
        node1: int,
        node2: int,
        vec1: np.ndarray,
        vec2: np.ndarray,
        sanity_check: bool = False
    ) -> None:
    """
    Projects the given messages out of the edge `(node1, node2)`. `vec1`
    is the message from `node1` to `node2`, `vec2` is defined
    accordingly. Both vectors must have the shape `(braket.bra.size,
    braket.op.size, braket.ket.size)`. Inserts the projector in the
    edge `(node1, node2)` in `braket`. `braket` is changed in-place.
    """
    # sanity check
    bra_size = braket.bra.G[node1][node2][0]["size"]
    op_size = braket.op.chi
    ket_size = braket.ket.G[node1][node2][0]["size"]
    vec_size = (bra_size, op_size, ket_size)
    if not vec1.shape == vec_size:
        raise ValueError("".join((
            "vec1 has wrong size; expected ",
            str(vec_size),
            ", got ",
            str(vec1.shape),
            "."
        )))
    if not vec2.shape == vec_size:
        raise ValueError("".join((
            "vec2 has wrong size; expected ",
            str(vec_size),
            ", got ",
            str(vec2.shape),
            "."
        )))
    if sanity_check: assert braket.intact

    # normalisation in every PEPO virtual bond.
    vec1 = np.einsum(
        "ijk, j -> ijk",
        vec1,
        1 / np.sqrt(np.einsum("ijk, ijk -> j", vec1.conj(), vec1))
    )
    vec2 = np.einsum(
        "ijk, j -> ijk",
        vec2,
        1 / np.sqrt(np.einsum("ijk, ijk -> j", vec2.conj(), vec2))
    )

    # constructing projectors
    P1 = np.einsum(
        "ij, kl, mn -> ikmjln",
        np.eye(bra_size),
        np.eye(op_size),
        np.eye(ket_size)
    )
    P1 -= np.einsum(
        "ijk, lmn, jmrh -> irklhn",
        vec1,
        vec1.conj(),
        delta_tensor(nLegs=4, chi=op_size)
    )

    P2 = np.einsum(
        "ij, kl, mn -> ikmjln",
        np.eye(bra_size),
        np.eye(op_size),
        np.eye(ket_size)
    )
    P2 -= np.einsum(
        "ijk, lmn, jmrh -> irklhn",
        vec2,
        vec2.conj(),
        delta_tensor(nLegs=4, chi=op_size)
    )

    braket.insert_edge_T(P1, node1, node2, sanity_check=sanity_check)
    braket.insert_edge_T(P2, node2, node1, sanity_check=sanity_check)

    return


@make_BP_informed
def QR_bottleneck(
        braket: Braket,
        node1: int,
        node2: int,
        vec1: np.ndarray,
        vec2: np.ndarray,
        sanity_check: bool = False
    ) -> None:
    """
    Projects edge `(node1, node2)` onto the two-dimensional subspace,
    that is spanned by `vec1` and `vec2`. Both vectors must have the
    shape `(braket.bra.size, braket.op.size, braket.ket.size)`.
    """
    # sanity check
    vec_size = (
        braket.bra.G[node1][node2][0]["size"],
        braket.op.G[node1][node2][0]["size"],
        braket.ket.G[node1][node2][0]["size"]
    )
    if not vec1.shape == vec_size:
        raise ValueError("".join((
            "vec1 has wrong size; expected ",
            str(vec_size),
            ", got ",
            str(vec1.shape),
            "."
        )))
    if not vec2.shape == vec_size:
        raise ValueError("".join((
            "vec2 has wrong size; expected ",
            str(vec_size),
            ", got ",
            str(vec2.shape),
            "."
        )))
    if sanity_check: assert braket.intact

    # QR-decomposition of the matrix (vec1, vec2).
    A = np.stack(arrays=(vec1.flatten(), vec2.flatten()), axis=-1)
    Q, R = np.linalg.qr(A, mode="reduced")

    # retaining only the hermitian part of Q
    Q = (Q + Q.conj()) / 2

    # assembling and inserting the projector
    Q = np.reshape(
        Q,
        newshape=(
            braket.bra.G[node1][node2][0]["size"],
            braket.op.G[node1][node2][0]["size"],
            braket.ket.G[node1][node2][0]["size"],
            Q.shape[-1]
        )
    )
    P1 = np.einsum("ijkx, xlmn -> ijklmn", Q, Q.conj().T)
    P2 = np.einsum("ijkx, xlmn -> ijklmn", Q.conj().T, Q)
    braket.insert_edge_T(P1, node1, node2, sanity_check=sanity_check)
    braket.insert_edge_T(P2, node2, node1, sanity_check=sanity_check)

    return


def L2BP_compression(
        psi: PEPS,
        singval_threshold: float = 1e-10,
        overlap: Braket = None,
        return_singvals: bool = True,
        min_bond_dim: Union[int, nx.MultiGraph] = 1,
        max_bond_dim: Union[int, nx.MultiGraph] = np.inf,
        verbose: bool = False,
        sanity_check: bool = False,
        **kwargs
    ) -> tuple[PEPS ,dict[frozenset, np.ndarray]]:
    """
    L2BP compression from [Sci. Adv. 10, eadk4321
    (2024)](https://doi.org/10.1126/sciadv.adk4321). Singular values
    below `singval_threshold` are discarded. A `Braket` object can be
    passed, in which case it's messages will be used. `kwargs` are
    passed to `Braket.BP`. Returns the gauged state `psi` and, if
    `return_singvals = True`, a dictionary containing the singular
    values over every edge.

    The minimum and maximum bond dimensions can be specified as
    `nx.MultiGraph`, in which case it is assumed that each edge contains
    the respective value for the edges in `psi`.

    The minimum magnitude of singular values (determined by
    `singval_threshold`) and the maximum bond dimension (determined by
    `max_bond_dim`) are combined using a logical and. The minimum size
    takes precedent over the maximum size.
    """
    if sanity_check: assert psi.intact

    # preparing target edge sizes
    if not isinstance(min_bond_dim, nx.MultiGraph):
        min_bond_dim = PEPO.prepare_graph(
            G=psi.G, chi=min_bond_dim, sanity_check=sanity_check
        )
    if not isinstance(max_bond_dim, nx.MultiGraph):
        max_bond_dim = PEPO.prepare_graph(
            G=psi.G, chi=max_bond_dim, sanity_check=sanity_check
        )

    if not graph_compatible(psi.G, min_bond_dim):
        raise ValueError(
            "Minimum bond dimension and state have non-compatible graphs."
        )
    if not graph_compatible(psi.G, max_bond_dim):
        raise ValueError(
            "Maximum bond dimension and state have non-compatible graphs."
        )

    # checking compatibility of sizes in min_bond_dim and max_bond_dim.
    for node1, node2 in psi.G.edges():
        if (min_bond_dim[node1][node2][0]["size"]
            > max_bond_dim[node1][node2][0]["size"]):
            warnings.warn(
                "".join((
                    f"Minimum bond dimension {min_bond_dim} on edge ",
                    f"({node1}, {node2}) is larger than maximum bond ",
                    f"dimension {max_bond_dim}. Setting bond dimension to ",
                    "unlimited. This may increase runtime drastically."
                )),
                RuntimeWarning
            )
            max_bond_dim[node1][node2][0]["size"] = np.inf

    # handling kwargs.
    if "iterator_desc_prefix" in kwargs.keys():
        kwargs["iterator_desc_prefix"] = "".join((
            kwargs["iterator_desc_prefix"],
            " | L2BP compression"
        ))
    else:
        kwargs["iterator_desc_prefix"] = "L2BP compression"

    if overlap is not None:
        if not isinstance(overlap, Braket):
            raise ValueError("Overlap is not a braket object!")
        if sanity_check: assert overlap.intact
        if not overlap.converged:
            warnings.warn(
                "Messages in overlap are not converged. Running BP iteration.",
                RuntimeWarning
            )
            overlap = None

    if overlap is None:
        # BP iteration
        overlap = Braket.Overlap(psi, psi, sanity_check=sanity_check)
        overlap.BP(**kwargs, verbose=verbose, sanity_check=sanity_check)

    all_singvals = {}

    # newly compressed state
    newpsi = copy.deepcopy(psi)

    # runtime diagnosis
    old_sizes = ()
    new_sizes = ()

    # compressing every edge.
    for node1, node2 in psi.G.edges():
        size = overlap.ket.G[node1][node2][0]["size"]
        ndim1 = psi.G.nodes[node1]["T"].ndim
        ndim2 = psi.G.nodes[node2]["T"].ndim

        # get messages
        msg_12 = np.reshape(
            overlap.msg[node1][node2][:,0,:], newshape=(size, size)
        )
        msg_21 = np.reshape(
            overlap.msg[node2][node1][:,0,:], newshape=(size, size)
        )

        # splitting the messages.
        eigvals1,W1 = scialg.eig(msg_12, overwrite_a=True)
        eigvals2,W2 = scialg.eig(msg_21, overwrite_a=True)
        R1 = np.diag(np.sqrt(eigvals1)) @ W1.conj().T
        R2 = np.diag(np.sqrt(eigvals2)) @ W2.conj().T

        # SVD over the bond, and truncation
        U, singvals, Vh = scialg.svd(
            R1 @ R2, full_matrices=False, overwrite_a=True
        )

        # saving singular values
        all_singvals[frozenset((node1, node2))] = singvals

        # for numerical stability, we have to drop all zero singular values.
        nonzero_mask = singvals != 0
        # singular values we want to keep, based on singval_threshold.
        threshold_mask = np.logical_not(
            np.isclose(singvals, 0, atol=singval_threshold)
        )
        # singular values we want to keep, based on the maximum size of this edge
        maxsize_mask = (np.arange(len(singvals))
                        < max_bond_dim[node1][node2][0]["size"])

        keep_mask = np.logical_and(
            np.logical_and(
                nonzero_mask,
                threshold_mask
            ),
            maxsize_mask
        )

        # saving values for diagnostics
        old_sizes += (len(keep_mask),)
        new_sizes += (sum(keep_mask),)

        if sum(nonzero_mask) == 0:
            raise RuntimeError(f"Edge ({node1}, {node2}) is zero-valued.")

        if np.sum(keep_mask) < min_bond_dim[node1][node2][0]["size"]:
            # the singular value threshold is too restrictive
            for i in range(min(
                min_bond_dim[node1][node2][0]["size"],
                sum(nonzero_mask)
            )):
                keep_mask[i] = True

        U = U[:,keep_mask]
        Vh = Vh[keep_mask,:]
        singvals = singvals[keep_mask]

        # projectors
        P1 = np.einsum(
            "ij, jk, kl -> il",
            R2,
            Vh.conj().T,
            np.diag(1 / np.sqrt(singvals)),
            optimize=True
        )
        P2 = np.einsum(
            "ij, jk, kl -> il",
            np.diag(1 / np.sqrt(singvals)),
            U.conj().T,
            R1,
            optimize=True
        )

        # absorbing projector 1 into node 1
        Tlegs = tuple(range(ndim1))
        Plegs = (overlap.ket.G[node1][node2][0]["legs"][node1], ndim1)
        outlegs = list(range(ndim1))
        outlegs[overlap.ket.G[node1][node2][0]["legs"][node1]] = ndim1
        newpsi[node1] = np.einsum(
            newpsi[node1], Tlegs,
            P1, Plegs,
            outlegs,
            optimize=True
        )

        # absorbing projector 2 into node 2
        Tlegs = tuple(range(ndim2))
        Plegs = (ndim2, overlap.ket.G[node1][node2][0]["legs"][node2])
        outlegs = list(range(ndim2))
        outlegs[overlap.ket.G[node1][node2][0]["legs"][node2]] = ndim2
        newpsi[node2] = np.einsum(
            P2, Plegs,
            newpsi[node2], Tlegs,
            outlegs,
            optimize=True
        )

        # updating size of edge
        newpsi.G[node1][node2][0]["size"] = P1.shape[1]

    if verbose:
        compression_ratios = tuple(
            new_sizes[i] / old_sizes[i]
            for i in range(len(old_sizes)))
        om = np.mean(old_sizes)
        os = np.std(old_sizes)
        nm = np.mean(new_sizes)
        ns = np.std(new_sizes)
        cm = np.mean(compression_ratios)
        cs = np.std(compression_ratios)
        print("L2BP compression diagnostics:")
        print(f"    Mean old size:          {om:7.3f} +- {os:7.3f}")
        print(f"    Mean new size:          {nm:7.3f} +- {ns:7.3f}")
        print(f"    Mean compression ratio: {cm:7.3f} +- {cs:7.3f}")

    return (newpsi, all_singvals) if return_singvals else newpsi


def QR_gauging(
        psi: PEPS,
        tree: nx.DiGraph = None,
        nodes: tuple[int] = None,
        sanity_check: bool = False,
        **kwargs
    ) -> PEPS:
    """
    Gauging of a state using QR decompositions. The root node of `tree`
    is the orthogonality center; if `tree` is not given, a breadth-first
    search spanning tree will be used. If given, only the nodes in
    `nodes` will be gauged.
    """
    if sanity_check: assert psi.intact

    if tree is None:
        # orthogonality center will be the node with the largest number of
        # neighborhoods.
        ortho_center = 0
        max_degree = 0
        for node in psi.G.nodes():
            if len(psi.G.adj[node]) > max_degree:
                ortho_center = node
                max_degree = len(psi.G.adj[node])
        tree = nx.bfs_tree(G=psi.G, source=ortho_center)
    else:
        if not isinstance(tree, nx.DiGraph):
            raise ValueError("tree must be an oriented graph.")
        if not nx.is_tree(tree):
            raise ValueError("Given spanning tree is not actually a tree.")
        # finding the orthogonality center.
        ortho_center = None
        for node in tree.nodes():
            if tree.in_degree(node) == 0:
                ortho_center = node
                break

    if nodes is None: nodes = tuple(psi.G.nodes())

    newpsi = copy.deepcopy(psi)

    # QR decompositions in upstream direction of the tree
    for node in nx.dfs_postorder_nodes(tree, source=ortho_center):
        if node not in nodes: continue

        if node == ortho_center:
            # we have reached the source
            continue

        # finding the upstream neighbor
        assert tree.in_degree(node) == 1
        pred = [_ for _ in tree.pred[node]][0]

        # exposing the upstream leg of the site tensor, and re-shaping
        T_exposed = np.moveaxis(
            newpsi[node],
            source=newpsi.G[pred][node][0]["legs"][node],
            destination=-1
        )
        oldshape = T_exposed.shape
        T_exposed = np.reshape(
            T_exposed,
            newshape=(-1, newpsi.G[pred][node][0]["size"])
        )

        # QR decomposition
        Q, R = np.linalg.qr(T_exposed, mode="reduced")

        # re-shaping Q, and inserting into the state.
        Q = np.reshape(Q, newshape=oldshape)
        Q = np.moveaxis(
            Q,
            source=-1,
            destination=newpsi.G[pred][node][0]["legs"][node]
        )
        newpsi[node] = Q

        # absorbing R into upstream node.
        upstream_legs = tuple(range(newpsi[pred].ndim))
        out_legs = tuple(
            newpsi[pred].ndim if i == newpsi.G[pred][node][0]["legs"][pred]
            else i
            for i in range(newpsi[pred].ndim)
        )
        newpsi[pred] = np.einsum(
            newpsi[pred], upstream_legs,
            R, (newpsi[pred].ndim, newpsi.G[pred][node][0]["legs"][pred]),
            out_legs,
        )

    return newpsi


def feynman_cut(
        obj: Union[PEPS, PEPO, Braket],
        node1: int,
        node2: int,
        sanity_check: bool = False
    ) -> Union[tuple[PEPS], tuple[PEPO], tuple[Braket]]:
    """
    Cuts the edge `(node1, node2)` in `obj`, and returns all resulting
    objects.
    """
    # sanity check
    assert obj.G.has_edge(node1, node2, 0)
    if sanity_check: assert obj.intact

    if isinstance(obj, PEPS):
        oldG = copy.deepcopy(obj. G)
        expose_edge(oldG, node1=node1, node2=node2, sanity_check=sanity_check)
        leg1 = oldG[node1][node2][0]["legs"][node1]
        leg2 = oldG[node1][node2][0]["legs"][node2]
        idx1 = lambda i: tuple(
            i if _ == leg1
            else slice(obj.G.nodes[node1]["T"].shape[_])
            for _ in range(obj.G.nodes[node1]["T"].ndim)
        )
        idx2 = lambda i: tuple(
            i if _ == leg2
            else slice(obj.G.nodes[node2]["T"].shape[_])
            for _ in range(obj.G.nodes[node2]["T"].ndim)
        )

        res_objs = ()
        for i in range(obj.G[node1][node2][0]["size"]):
            newG = copy.deepcopy(oldG)
            newG.remove_edge(node1,node2,key=0)
            newG.nodes[node1]["T"] = oldG.nodes[node1]["T"][idx1(i)]
            newG.nodes[node2]["T"] = oldG.nodes[node2]["T"][idx2(i)]
            res_objs += (PEPS(newG, sanity_check=sanity_check),)

        return res_objs

    if isinstance(obj, PEPO):
        oldG = copy.deepcopy(obj.G)
        expose_edge(oldG, node1=node1, node2=node2, sanity_check=sanity_check)
        leg1 = oldG[node1][node2][0]["legs"][node1]
        leg2 = oldG[node1][node2][0]["legs"][node2]
        idx1 = lambda i: tuple(
            i if _ == leg1
            else slice(obj.G.nodes[node1]["T"].shape[_])
            for _ in range(obj.G.nodes[node1]["T"].ndim)
        )
        idx2 = lambda i: tuple(
            i if _ == leg2
            else slice(obj.G.nodes[node2]["T"].shape[_])
            for _ in range(obj.G.nodes[node2]["T"].ndim)
        )

        res_objs = ()
        for i in range(obj.G[node1][node2][0]["size"]):
            newG = copy.deepcopy(oldG)
            newG.remove_edge(node1,node2,key=0)
            newG.nodes[node1]["T"] = oldG.nodes[node1]["T"][idx1(i)]
            newG.nodes[node2]["T"] = oldG.nodes[node2]["T"][idx2(i)]
            res_objs += (PEPO.from_graphs(
                newG,
                obj.tree,
                check_tree=False,
                sanity_check=sanity_check
            ),)

        return res_objs

    if isinstance(obj,Braket):
        bra_cuts = feynman_cut(
            obj.bra, node1, node2, sanity_check=sanity_check
        )
        op_cuts = feynman_cut(
            obj.op, node1, node2, sanity_check=sanity_check
        )
        ket_cuts = feynman_cut(
            obj.ket, node1, node2, sanity_check=sanity_check
        )

        res_objs = ()
        for bra, op, ket in itertools.product(bra_cuts, op_cuts, ket_cuts):
            res_objs += (Braket(bra, op, ket, sanity_check=sanity_check),)

        return res_objs

    raise NotImplementedError("".join((
        "feynman_cut not implemented for object of type ",str(type(obj)),"."
    )))


if __name__ == "__main__":
    pass
