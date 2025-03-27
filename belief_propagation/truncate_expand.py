"""
Functions for manipulation of PEPS, PEPO and Braket, with the goal of...
* cutting / truncating edges in the graphs.
* series expansion of brakets.
"""

__all__ = [
    "make_BP_informed",
    "project_down",
    "insert_excitation",
    "QR_bottleneck",
    "L2BP_compression",
    "QR_gauging",
    "BP_excitations",
    "loop_series_contraction"
]

import itertools
import copy
from typing import Callable, Union, Tuple, Dict
import warnings

import numpy as np
import networkx as nx
import scipy.linalg as scialg
import tqdm

from belief_propagation.braket import (
    Braket,
    contract_braket_physical_indices,
    BP_excitations
)
from belief_propagation.PEPO import PEPO
from belief_propagation.PEPS import PEPS
from belief_propagation.networks import expose_edge
from belief_propagation.utils import (
    delta_tensor,
    graph_compatible,
)

# -----------------------------------------------------------------------------
#                   Truncating brakets
# -----------------------------------------------------------------------------


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
    `BP_informed_truncation_func` runs a BP iteration, if messages in
    `braket` are not converged, and normalizes with respect to the dot
    product.
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

        # Inserting truncation into the Braket.
        truncation_func(
            braket,
            node1,
            node2,
            braket.msg[node1][node2],
            braket.msg[node2][node1],
            sanity_check=sanity_check
        )

        return

    return BP_informed_truncation_func


@make_BP_informed
def project_down(
        braket: Braket,
        node1: int,
        node2: int,
        msg12: np.ndarray,
        msg21: np.ndarray,
        sanity_check: bool = False
    ) -> None:
    """
    Projects the edge `(node1, node2)` onto the given messages. `msg12`
    is the message from `node1` to `node2`, `msg21` is defined
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
    if not msg12.shape == vec_size:
        raise ValueError("".join((
            "msg12 has wrong size; expected ",
            str(vec_size),
            ", got ",
            str(msg12.shape),
            "."
        )))
    if not msg21.shape == vec_size:
        raise ValueError("".join((
            "msg21 has wrong size; expected ",
            str(vec_size),
            ", got ",
            str(msg21.shape),
            "."
        )))
    if sanity_check: assert braket.intact

    P1 = np.einsum("ijk,lmn->ijklmn", msg21, msg12)
    P2 = np.einsum("ijk,lmn->ijklmn", msg12, msg21)
    braket.insert_edge_T(P1, node1, node2, sanity_check=sanity_check)
    braket.insert_edge_T(P2, node2, node1, sanity_check=sanity_check)

    return


@make_BP_informed
def insert_excitation(
        braket: Braket,
        node1: int,
        node2: int,
        msg12: np.ndarray,
        msg21: np.ndarray,
        sanity_check: bool = False,
        dtype=np.complex128
    ) -> None:
    """
    Inserts a projector onto the excited subspace on the edge `(node1,
    node2)`. `msg12` is the message from `node1` to `node2`, `msg21` is
    defined accordingly. Both vectors must have the shape
    `(braket.bra.size, braket.op.size, braket.ket.size)`. `braket` is
    changed in-place. Procedure from
    [arXiv:2409.03108](https://arxiv.org/abs/2409.03108).
    """
    # sanity check
    bra_size = braket.bra.G[node1][node2][0]["size"]
    op_size = braket.op.G[node1][node2][0]["size"]
    ket_size = braket.ket.G[node1][node2][0]["size"]
    vec_size = (bra_size, op_size, ket_size)
    if not msg12.shape == vec_size:
        raise ValueError("".join((
            "msg12 has wrong size; expected ",
            str(vec_size),
            ", got ",
            str(msg12.shape),
            "."
        )))
    if not msg21.shape == vec_size:
        raise ValueError("".join((
            "msg21 has wrong size; expected ",
            str(vec_size),
            ", got ",
            str(msg21.shape),
            "."
        )))
    if sanity_check: assert braket.intact

    # Constructing projectors.
    P1to2 = np.einsum(
        "ij,kl,mn->ikmjln",
        np.eye(bra_size, dtype=dtype),
        np.eye(op_size, dtype=dtype),
        np.eye(ket_size, dtype=dtype)
    )
    P1to2 -= np.einsum(
        "ijk,lmn->ijklmn",
        msg12,
        msg21
    )

    P2to1 = np.einsum(
        "ij,kl,mn->ikmjln",
        np.eye(bra_size, dtype=dtype),
        np.eye(op_size, dtype=dtype),
        np.eye(ket_size, dtype=dtype)
    )
    P2to1 -= np.einsum(
        "ijk,lmn->ijklmn",
        msg21,
        msg12
    )

    braket.insert_edge_T(P1to2, node1, node2, sanity_check=sanity_check)
    braket.insert_edge_T(P2to1, node2, node1, sanity_check=sanity_check)

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
    ) -> Tuple[PEPS ,Dict[frozenset, np.ndarray]]:
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
        # singular values we want to keep, based on the maximum size of this
        # edge.
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
            "ij,jk,kl->il",
            R2,
            Vh.conj().T,
            np.diag(1 / np.sqrt(singvals)),
            optimize=True
        )
        P2 = np.einsum(
            "ij,jk,kl->il",
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
            (old_sizes[i] - new_sizes[i]) / old_sizes[i]
            for i in range(len(old_sizes)))

        om = np.mean(old_sizes)
        os = np.std(old_sizes)
        nm = np.mean(new_sizes)
        ns = np.std(new_sizes)
        cm = np.mean(compression_ratios)
        cs = np.std(compression_ratios)
        print(
            f"L2BP compression: Sing.val. threshold = {singval_threshold:.3e}"
        )
        print(f"    Mean old size:          {om:7.3f} +- {os:7.3f}")
        print(f"    Mean new size:          {nm:7.3f} +- {ns:7.3f}")
        print(f"    Mean compression ratio: {cm:7.3f} +- {cs:7.3f}")

    if sanity_check: assert newpsi.intact

    return (newpsi, all_singvals) if return_singvals else newpsi


def QR_gauging(
        psi: PEPS,
        tree: nx.DiGraph = None,
        nodes: Tuple[int] = None,
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
    ) -> Union[Tuple[PEPS], Tuple[PEPO], Tuple[Braket]]:
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


# -----------------------------------------------------------------------------
#                   Loop series expansion
# -----------------------------------------------------------------------------


def loop_series_contraction(
        braket: Braket,
        excitations: Tuple[nx.MultiGraph] = None,
        max_order: int = np.inf,
        verbose: bool = False,
        sanity_check: bool = False,
        **kwargs
    ) -> float:
    """
    Loop series expansion of the value of `braket` from
    [arXiv:2409.03108](https://arxiv.org/abs/2409.03108). `max_order` is
    the maximum oder of loops that are taken into consideration; default
    is `np.inf`, where all loops are incorporated. `kwargs` are passed
    to `braket.BP`.
    """
    # sanity check
    if sanity_check: assert braket.intact

    # Handling kwargs.
    kwargs["verbose"] = verbose
    if "iterator_desc_prefix" in kwargs.keys():
        kwargs["iterator_desc_prefix"] = "".join((
            kwargs["iterator_desc_prefix"],
            " | Loop Series Contraction | "
        ))
    else:
        kwargs["iterator_desc_prefix"] = "Loop Series Contraction"

    # Finding the BP fixed point.
    if not braket.converged:
        braket.BP(**kwargs, sanity_check=sanity_check)

    # Inserting projectors in the edges.
    for node1, node2 in braket.G.edges():
        insert_excitation(
            braket=braket,
            node1=node1,
            node2=node2,
            sanity_check=sanity_check
        )

    # The total contraction value of the braket is factored out, s.t. the
    # calculation of higher-order contributions becomes easier.
    cntr_norm_factor = braket.cntr

    # Saving node contraction values, s.t. I can undo the node normalization
    # later.
    node_cntr = {node: cntr for node, cntr in braket.G.nodes(data="cntr")}

    # Normalizing braket s.t. the BP vacuum contribution is one.
    for node in braket:
        braket[node] = tuple(
            T / (node_cntr[node] ** (1/3))
            for T in braket[node]
        )
    braket.op.check_tree = False

    # During the computations of excitations, messages are inserted into the
    # network. Messages are dense tensors, not tensor stacks - we must thus
    # contract the physical dimensions.
    densebraket = contract_braket_physical_indices(
        braket=braket, sanity_check=sanity_check
    )

    if excitations is None:
        # Finding excitations in the graph.
        excitations = BP_excitations(
            braket.G, max_order=max_order, sanity_check=sanity_check
        )
    else:
        excitations = tuple(
            exc for exc in excitations
            if exc.number_of_edges() <= max_order
        )

    iterator = tqdm.tqdm(
        iterable=excitations,
        desc="BP excitations",
        disable=not verbose
    )

    cntr = 1
    for excitation in iterator:
        exc_weight = excitation.number_of_edges()
        iterator.set_postfix_str(f"Current exc. weight = {exc_weight}")

        cntr_ = __compute_BP_excitation(
            braket=densebraket,
            excitation=excitation,
            sanity_check=sanity_check
        )
        cntr += cntr_

    # Undoing BP vacuum contribution, so that the braket remains unchanged.
    for node in braket:
        braket[node] = tuple(
            T * (node_cntr[node] ** (1/3))
            for T in braket[node]
        )

    return cntr * cntr_norm_factor


def __check_contracted_physical_dims(
        braket: Braket,
        sanity_check: bool = False
    ) -> bool:
    """
    Checks if `braket` contains dummy networks in `braket.op` and
    `braket.bra`.
    """
    if sanity_check: assert braket.intact

    for node in braket:
        if not (np.allclose(braket.bra[node], np.ones(shape=1))
                and np.allclose(braket.op[node], np.ones(shape=1))):
            return False

    return True


def __compute_BP_excitation(
        braket: Braket,
        excitation: nx.MultiGraph,
        sanity_check: bool = False,
        **kwargs
    ) -> float:
    """
    Computes the contribution of the excitation `excitation`. Method
    taken from [arXiv:2409.03108](https://arxiv.org/abs/2409.03108).
    `excitation` contains the excited edges, `braket` contains the
    tensor network, the projectors and the messages. `kwargs` are passed
    to `Braket.contract`.
    """
    assert __check_contracted_physical_dims(braket, sanity_check=sanity_check)

    if not nx.is_connected(G=excitation):
        connected_excitations = tuple(
            excitation.subgraph(component)
            for component in nx.connected_components(excitation)
        )

        # For disjoint excitations, the contribution is the product of the
        # components.
        return np.prod(tuple(
            __compute_BP_excitation(
                braket=braket,
                excitation=exc,
                sanity_check=sanity_check,
                **kwargs
            )
            for exc in connected_excitations
        ))

    # How will this work under the hood? We truncate the network. All edges
    # that are not contained in the excitation will be removed, and new edges
    # will be added that connect the messages that flow into the excitation.
    # On the excitations edges we add the projectors.

    # The set of nodes inside the excitation, and the set of edges that are not
    # excited.
    excitation_nodes = set(excitation.nodes())
    non_excitation_nodes = set(braket.G.nodes()) - excitation_nodes
    non_excitation_edges = nx.MultiGraph(incoming_graph_data=braket.G.edges())
    non_excitation_edges.remove_edges_from(excitation.edges())

    # The graph that we will contract.
    G = copy.deepcopy(braket.ket.G)

    # Removing nodes and edges that are not present in the excitation.
    G.remove_edges_from(non_excitation_edges.edges())
    G.remove_nodes_from(non_excitation_nodes)

    for node in excitation_nodes:
        # Removing the dummy physical dimension from local tensors.
        G.nodes[node]["T"] = G.nodes[node]["T"][...,0]

        # Inserting messages as new sites in the graph.
        while G.nodes[node]["T"].ndim > len(G.adj[node]):
            # If this is the case, there are dangling tensor legs to which no
            # message is attached. Identifying the dangling neighbor, and the
            # tensor leg it connects to.
            next_node_label = max(node_ for node_ in G) + 1
            neighbor = (set(braket.G.adj[node])
                        - (set(G.adj[node]))).pop()
            leg = braket.G[node][neighbor][0]["legs"][node]

            # Inserting a new node that contains the respective message, and
            # connecting it to the graph.
            G.add_node(
                node_for_adding=next_node_label,
                T=braket.msg[neighbor][node][0,0,:]
            )
            G.add_edge(
                u_for_edge=node,
                v_for_edge=next_node_label,
                legs={node: leg, next_node_label: 0},
            )

    # Inserting projectors as nodes on the excited edges.
    for node1, node2 in excitation.edges():
        # Parameters of this edge, and the node we will add: edge legs, size,
        # and the label of the next node.
        leg1 = G[node1][node2][0]["legs"][node1]
        leg2 = G[node1][node2][0]["legs"][node2]
        next_node_label = max(node_ for node_ in G) + 1

        # Adding a node that contains the projector, and connecting it to the
        # graph.
        G.add_node(
            node_for_adding=next_node_label,
            T=braket.edge_T[node1][node2][0,0,:,0,0,:]
        )
        G.add_edge(
            u_for_edge=node1,
            v_for_edge=next_node_label,
            legs={node1: leg1, next_node_label: 1},
        )
        G.add_edge(
            u_for_edge=node2,
            v_for_edge=next_node_label,
            legs={node2: leg2, next_node_label: 0},
        )

        # Removing the old edge.
        G.remove_edge(node1, node2)

    # Constructing a new braket for this excitation, and contracting it to get
    # the contribution of this excitation.
    exc_braket = Braket.Cntr(G=G, sanity_check=sanity_check)
    cntr = exc_braket.contract(sanity_check=sanity_check, **kwargs)

    return cntr


if __name__ == "__main__":
    pass
