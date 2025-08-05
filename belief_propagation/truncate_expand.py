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
    "random_bond_gauging",
    "BP_excitations",
    "loop_series_contraction"
]

import itertools
import copy
from typing import Callable, Union, NoReturn
import warnings

import numpy as np
import networkx as nx
import scipy.linalg as scialg
import tqdm
import scipy.stats as scistats

from belief_propagation.braket import (
    Braket,
    BP_excitations,
    assemble_excitation_brakets
)
from belief_propagation.PEPO import PEPO
from belief_propagation.PEPS import PEPS
from belief_propagation.networks import expose_edge
from belief_propagation.utils import graph_compatible

# -----------------------------------------------------------------------------
#                   Truncating brakets
# -----------------------------------------------------------------------------


def make_BP_informed(
        truncation_func: Callable[
            [Braket, int, int, np.ndarray, np.ndarray, bool],
            NoReturn
        ]
    ) -> Callable[[Braket, int, int, bool, bool], NoReturn]:
    """
    Gives back a version of the truncation function that executes a BP
    iteration beforehand, and uses the messages as arguments to
    `truncation_func`, thereby making it BP-informed. The call signature
    of `truncation_func` must be `truncation_func(braket: Braket,
    node1: int, node2: int, vec1: np.ndarray, vec2: np.ndarray,
    sanity_check: bool = False)`.

    Returns a function with call signature
    `BP_informed_truncation_func(braket: Braket, node1: int, node2: int,
    skip_BP: bool = False, sanity_check: bool = False, **kwargs)`, where
    `kwargs` are passed to the BP iteration. `braket` contains the
    messages that will be used for truncation of the edge `(node1,
    node2)`. If `skip_BP = False` (default),
    `BP_informed_truncation_func` runs a BP iteration, if messages in
    `braket` are not converged, and normalizes with respect to the dot
    product.
    """

    def BP_informed_truncation_func(
            braket: Braket,
            node1: int,
            node2: int,
            skip_BP: bool = False,
            sanity_check: bool = False,
            **kwargs
        ) -> None:
        # sanity check
        assert braket.G.has_edge(node1, node2, key=0)
        if sanity_check: assert braket.intact

        # BP iteration, to obtain messages for truncation.
        if (not skip_BP) and (not braket.converged):
            braket.BP(sanity_check=sanity_check, **kwargs)

        if not braket.converged:
            with tqdm.tqdm.external_write_mode():
                warnings.warn(
                    "Modification on edge with non-converged messages.",
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
        shape=(
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
        overlap: Braket = None,
        singval_threshold: float = 1e-10,
        min_bond_dim: Union[int, nx.MultiGraph] = 1,
        max_bond_dim: Union[int, nx.MultiGraph] = np.inf,
        return_singvals: bool = False,
        verbose: bool = False,
        sanity_check: bool = False,
        **kwargs
    ) -> Union[PEPS, tuple[PEPS ,dict[frozenset, np.ndarray]]]:
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

    # Preparing target edge sizes.
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

    # Checking compatibility of sizes in min_bond_dim and max_bond_dim.
    for node1, node2 in psi.G.edges():
        if (min_bond_dim[node1][node2][0]["size"]
            > max_bond_dim[node1][node2][0]["size"]):
            with tqdm.tqdm.external_write_mode():
                warnings.warn(
                    "".join((
                        f"Minimum bond dimension {min_bond_dim} on edge ",
                        f"({node1}, {node2}) is larger than maximum bond ",
                        f"dimension {max_bond_dim}. Setting bond dimension ",
                        "to unlimited. This may increase runtime drastically."
                    )),
                    RuntimeWarning
                )
            max_bond_dim[node1][node2][0]["size"] = np.inf

    if singval_threshold <= 0:
        warnings.warn(
            "".join((
                "Singular value threshold is smaller than or equal to zero. ",
                "This leads to no truncation dependence on singular value ",
                "magnitude."
            )),
            RuntimeWarning
        )
        singval_threshold = 0

    # Handling kwargs.
    if "iterator_desc_prefix" in kwargs.keys():
        kwargs["iterator_desc_prefix"] = "".join((
            kwargs["iterator_desc_prefix"],
            " | L2BP compression"
        ))
    else:
        kwargs["iterator_desc_prefix"] = "L2BP compression"

    if overlap is not None:
        # Sanity check for braket.
        if not isinstance(overlap, Braket):
            raise ValueError("Overlap is not a braket object!")
        if sanity_check: assert overlap.intact
        if not overlap.converged:
            with tqdm.tqdm.external_write_mode():
                warnings.warn(
                    "".join((
                        "Messages in overlap are not converged. Running BP ",
                        "iteration."
                    )),
                    RuntimeWarning
                )
            overlap = None

    if overlap is None:
        # BP iteration, to obtain converged messages.
        overlap = Braket.Overlap(psi, psi, sanity_check=sanity_check)
        overlap.BP(**kwargs, verbose=verbose, sanity_check=sanity_check)

    all_singvals = {}

    # Newly compressed state, which we will return later.
    newpsi = copy.deepcopy(psi)

    # Runtime information.
    old_sizes = ()
    new_sizes = ()

    # Compressing every edge.
    for node1, node2 in psi.G.edges():
        size = overlap.ket.G[node1][node2][0]["size"]
        ndim1 = psi.G.nodes[node1]["T"].ndim
        ndim2 = psi.G.nodes[node2]["T"].ndim

        # Saving old size for diagnosis.
        old_sizes += (overlap.ket.G[node1][node2][0]["size"],)

        # Get messages.
        msg_12 = np.reshape(
            overlap.msg[node1][node2][:,0,:], shape=(size, size)
        )
        msg_21 = np.reshape(
            overlap.msg[node2][node1][:,0,:], shape=(size, size)
        )

        # Splitting the messages, using SVD.
        eigvals1, W1 = scialg.eig(msg_12, overwrite_a=True)
        eigvals2, W2 = scialg.eig(msg_21, overwrite_a=True)
        R1 = np.diag(np.sqrt(eigvals1)) @ W1.conj().T
        R2 = np.diag(np.sqrt(eigvals2)) @ W2.conj().T

        # SVD over the bond, and truncation
        U, singvals, Vh = scialg.svd(
            R1 @ R2, full_matrices=False, overwrite_a=True
        )

        # Saving singular values.
        all_singvals[frozenset((node1, node2))] = singvals

        # For numerical stability, we have to drop all zero singular values.
        nonzero_mask = singvals != 0
        # Singular values we want to keep, based on singval_threshold.
        threshold_mask = np.logical_not(
            np.isclose(singvals, 0, atol=singval_threshold)
        )
        # Singular values we want to keep, based on the maximum size of this
        # edge.
        maxsize_mask = (np.arange(len(singvals))
                        < max_bond_dim[node1][node2][0]["size"])

        # We only keep singular values where all of the above conditions are
        # met.
        keep_mask = np.logical_and(
            np.logical_and(
                nonzero_mask,
                threshold_mask
            ),
            maxsize_mask
        )

        if sum(nonzero_mask) == 0:
            raise RuntimeError(f"Edge ({node1}, {node2}) is zero-valued.")

        if np.sum(keep_mask) < min_bond_dim[node1][node2][0]["size"]:
            # The singular value threshold is too restrictive.
            for i in range(min(
                min_bond_dim[node1][node2][0]["size"],
                sum(nonzero_mask)
            )):
                keep_mask[i] = True

        U = U[:, keep_mask]
        Vh = Vh[keep_mask, :]
        singvals = singvals[keep_mask]

        # Projectors on the SVD subspace that we keep.
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

        # Absorbing projector 1 into node 1.
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

        # Absorbing projector 2 into node 2.
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

        # Updating size of edge.
        newpsi.G[node1][node2][0]["size"] = P1.shape[1]

        # Saving new size for diagnosis.
        new_sizes += (newpsi.G[node1][node2][0]["size"],)

    if verbose:
        # Printing runtime information.
        compression_ratios = tuple(
            (old_sizes[i] - new_sizes[i]) / old_sizes[i]
            for i in range(len(old_sizes)))

        om = np.mean(old_sizes)
        os = np.std(old_sizes)
        nm = np.mean(new_sizes)
        ns = np.std(new_sizes)
        cm = np.mean(compression_ratios)
        cs = np.std(compression_ratios)
        with tqdm.tqdm.external_write_mode():
            print("".join((
                "L2BP compression: ",
                f"Sing.val. threshold = {singval_threshold:.3e}"
            )))
            print(f"    Mean old size:          {om:7.3f} +- {os:7.3f}")
            print(f"    Mean new size:          {nm:7.3f} +- {ns:7.3f}")
            print(f"    Mean compression ratio: {cm:7.3f} +- {cs:7.3f}")

    if sanity_check: assert newpsi.intact

    return (newpsi, all_singvals) if return_singvals else newpsi


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

        # Exposing the edge for easier access.
        expose_edge(oldG, node1=node1, node2=node2, sanity_check=sanity_check)

        # Defining slice indices.
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
            newG.remove_edge(node1, node2, key=0)

            # Slicing edge.
            newG.nodes[node1]["T"] = oldG.nodes[node1]["T"][idx1(i)]
            newG.nodes[node2]["T"] = oldG.nodes[node2]["T"][idx2(i)]

            # Saving the new state.
            res_objs += (PEPS(G=newG, sanity_check=sanity_check),)

        return res_objs

    if isinstance(obj, PEPO):
        oldG = copy.deepcopy(obj.G)

        # Exposing the edge for easier access.
        expose_edge(oldG, node1=node1, node2=node2, sanity_check=sanity_check)

        # Defining slice indices.
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

            # Slicing edge.
            newG.nodes[node1]["T"] = oldG.nodes[node1]["T"][idx1(i)]
            newG.nodes[node2]["T"] = oldG.nodes[node2]["T"][idx2(i)]

            # Saving the new operator.
            res_objs += (PEPO.from_graphs(
                G=newG,
                tree=obj.tree,
                check_tree=False,
                sanity_check=sanity_check
            ),)

        return res_objs

    if isinstance(obj,Braket):
        # Cutting an edge is defined as cutting the respective edge in the
        # constituent bra, operator, and ket.
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
        # Combining all possible slices.
        for bra, op, ket in itertools.product(bra_cuts, op_cuts, ket_cuts):
            res_objs += (Braket(
                bra=bra,
                op=op,
                ket=ket,
                sanity_check=sanity_check
            ),)

        return res_objs

    raise NotImplementedError("".join((
        "feynman_cut not implemented for object of type ",str(type(obj)),"."
    )))


# -----------------------------------------------------------------------------
#                   Gauging PEPS
# -----------------------------------------------------------------------------


def QR_gauging(
        psi: PEPS,
        tree: nx.DiGraph = None,
        ortho_center: int = None,
        nodes: tuple[int] = None,
        sanity_check: bool = False,
        **kwargs
    ) -> PEPS:
    """
    Gauging of a state using QR decompositions. The root node of `tree`
    is the orthogonality center; if given, `ortho_center` will be the
    orthogonality center. If `tree` is not given, a breadth-first
    search spanning tree will be used. If given, only the nodes in
    `nodes` will be gauged.

    Uses the reduced mode of the QR decomposition, and can thus reduce
    bond dimensions. Thus this method may result in compression.

    Function signature includes `kwargs` for compatibility reasons in
    the gauging functions of the DMRG-classes.
    """
    if sanity_check: assert psi.intact

    if tree is None:
        if ortho_center is None:
            # Orthogonality center will be the node with the largest number of
            # neighborhoods.
            ortho_center = 0
            max_degree = 0
            for node in psi.G.nodes():
                if len(psi.G.adj[node]) > max_degree:
                    ortho_center = node
                    max_degree = len(psi.G.adj[node])
        else:
            if not ortho_center in psi: raise ValueError(
                f"Graph of state does not contain orho_center {ortho_center}."
            )

        tree = nx.bfs_tree(G=psi.G, source=ortho_center)
    else:
        # Sanity check for tree.
        if not isinstance(tree, nx.DiGraph):
            raise ValueError("tree must be an oriented graph.")
        if not nx.is_tree(tree):
            raise ValueError("Given spanning tree is not actually a tree.")

        # Finding the orthogonality center.
        ortho_center = None
        for node in tree.nodes():
            if tree.in_degree(node) == 0:
                ortho_center = node
                break

    if nodes is None: nodes = tuple(psi.G.nodes())

    newpsi = copy.deepcopy(psi)

    # QR decompositions in upstream direction of the tree.
    for node in nx.dfs_postorder_nodes(tree, source=ortho_center):
        if node not in nodes: continue

        if node == ortho_center:
            # We have reached the source.
            continue

        assert tree.in_degree(node) == 1
        # Finding the upstream neighbor.
        pred = [_ for _ in tree.pred[node]][0]

        # Exposing the upstream leg of the site tensor, and re-shaping.
        T_exposed = np.moveaxis(
            newpsi[node],
            source=newpsi.G[pred][node][0]["legs"][node],
            destination=-1
        )
        oldshape = list(T_exposed.shape)
        T_exposed = np.reshape(
            T_exposed,
            shape=(-1, newpsi.G[pred][node][0]["size"])
        )

        # QR decomposition.
        Q, R = np.linalg.qr(T_exposed, mode="reduced")

        # Adjusting bond dimension, if necessary.
        newpsi.G[pred][node][0]["size"] = min(
            newpsi.G[pred][node][0]["size"], Q.shape[-1]
        )

        # Re-shaping Q, and inserting into the state.
        oldshape[-1] = newpsi.G[pred][node][0]["size"]
        Q = np.reshape(Q, shape=oldshape)
        Q = np.moveaxis(
            Q,
            source=-1,
            destination=newpsi.G[pred][node][0]["legs"][node]
        )
        newpsi[node] = Q

        # Absorbing R into upstream node.
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

    if sanity_check: assert newpsi.intact
    return newpsi


def random_bond_gauging(
        psi: PEPS,
        method: str = "unitary",
        sanity_check: bool = False,
        rng: np.random.Generator = np.random.default_rng()
    ) -> PEPS:
    """
    Gauging of a state by inserting matrices on the virtual bonds. Three
    variants are implemented:
    * `ortho`: Inserting random orthogonal matrices.
    * `unitary`: Inserting random unitary matrices (fallback).
    * `invert`: Inserting random invertible matrices. Internally relies
    on [diagonally dominant
    matrices](https://en.wikipedia.org/wiki/Diagonally_dominant_matrix)
    (see also [this post](https://stackoverflow.com/a/73427048)).
    """
    if sanity_check: assert psi.intact

    if method not in ("ortho", "unitary", "invert"):
        warnings.warn("".join((
            "Unknown method ", method, ". Defaulting to unitary."
            )),
            UserWarning
        )
        method = "unitary"

    if method == "ortho":
        # Random matrix from the orthogonal group.
        matrixgen = lambda N: scistats.ortho_group.rvs(dim=N, size=1)
        inv = lambda M: M

    elif method == "unitary":
        # Random mattrix from the unitary group.
        matrixgen = lambda N: scistats.unitary_group.rvs(dim=N, size=1)
        inv = lambda M: M.conj()

    elif method == "invert":
        # Random invertible matrix.
        def matrixgen(N: int) -> np.ndarray:
            M = (rng.uniform(low=-1, high=1, size=(N, N))
                 + 1j * rng.uniform(low=-1, high=1, size=(N, N)))
            Mabssum = np.sum(np.abs(M), axis=1)
            np.fill_diagonal(a=M, val=Mabssum)
            rank = np.linalg.matrix_rank(M)
            if rank < N: warnings.warn("Matrix is not invertible!")
            return M
        inv = lambda M: np.linalg.inv(M).T

    else:
        raise ValueError("Unknown method for random matrix generation.")

    newpsi = copy.deepcopy(psi)

    for node1, node2 in newpsi.G.edges():
        leg1 = newpsi.G[node1][node2][0]["legs"][node1]
        leg2 = newpsi.G[node1][node2][0]["legs"][node2]
        nLegs1 = newpsi[node1].ndim
        nLegs2 = newpsi[node2].ndim
        size = newpsi.G[node1][node2][0]["size"]
        # Generating random matrix.
        randmat = matrixgen(size)

        # Absorbing matrix in node1.
        legs1_out = tuple(nLegs1 if i == leg1 else i for i in range(nLegs1))
        newpsi[node1] = np.einsum(
            newpsi[node1], tuple(range(nLegs1)),
            randmat, (leg1, nLegs1),
            legs1_out,
        )

        # Absorbing matrix inverse in node2.
        legs2_out = tuple(nLegs2 if i == leg2 else i for i in range(nLegs2))
        newpsi[node2] = np.einsum(
            newpsi[node2], tuple(range(nLegs2)),
            inv(randmat), (leg2, nLegs2),
            legs2_out,
        )

    return newpsi


# -----------------------------------------------------------------------------
#                   Loop series expansion
# -----------------------------------------------------------------------------


def loop_series_contraction(
        braket: Braket,
        excitations: tuple[nx.MultiGraph] = None,
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

    A braket with converged messages may be passed, in which case it's
    messages will be used. Changes to `braket` are reversed before this
    function returns, s.t. `braket` may be re-used.
    """
    # sanity check
    if sanity_check: assert braket.intact

    # Handling kwargs.
    if "iterator_desc_prefix" in kwargs.keys():
        kwargs["iterator_desc_prefix"] = "".join((
            kwargs["iterator_desc_prefix"],
            " | Loop Series Contraction"
        ))
    else:
        kwargs["iterator_desc_prefix"] = "Loop Series Contraction"

    # Finding the BP fixed point.
    if not braket.converged:
        braket.BP(**kwargs, sanity_check=sanity_check, verbose=verbose)

    # Inserting projectors in the edges.
    for node1, node2 in braket.G.edges():
        insert_excitation(
            braket=braket,
            node1=node1,
            node2=node2,
            skip_BP=True, # Skipping BP because we already executed it above.
            sanity_check=sanity_check
        )

    # The total contraction value of the braket is factored out, s.t. the
    # calculation of higher-order contributions becomes easier.
    cntr_norm_factor = braket.cntr

    # Saving node contraction values, s.t. I can undo the node normalization
    # later.
    node_cntr = {node: cntr for node, cntr in braket.G.nodes(data="cntr")}

    # Normalizing braket s.t. the BP vacuum contributions are one.
    for node in braket:
        braket[node] = tuple(
            T / (node_cntr[node] ** (1/3))
            for T in braket[node]
        )
    braket.op.check_tree = False

    if max_order == 0:
        # The function BP_excitations internally calculates all excitations
        # before returning only the ones that the user asked for. if max_order
        # zero, we do not need to incur this overhead.
        excitations = ()

    elif excitations is None:
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
        disable=not verbose,
        total=len(excitations)
    )

    cntr = 1
    for excitation in iterator:
        exc_weight = excitation.number_of_edges()
        iterator.set_postfix_str(f"Current exc. weight = {exc_weight}")

        exc_brakets = assemble_excitation_brakets(
            braket=braket,
            excitation=excitation,
            sanity_check=sanity_check
        )
        cntr += np.prod(tuple(
            braket_.contract(sanity_check=sanity_check)
            for braket_ in exc_brakets
        ))

    # Undoing BP vacuum contribution, so that the braket remains unchanged.
    for node in braket:
        braket[node] = tuple(
            T * (node_cntr[node] ** (1/3))
            for T in braket[node]
        )

    # Tensor stacks manipulations have been reversed, thus we may set the
    # converged flag again.
    braket._converged = True

    # Removing all edge transformations.
    braket.clear_edge_T(sanity_check=sanity_check)

    return cntr * cntr_norm_factor


if __name__ == "__main__":
    pass
