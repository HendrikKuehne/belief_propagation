"""
Random stuff that is useful here or there.
"""

__all__ = [
    # mathematics
    "crandn",
    "delta_tensor",
    "multi_kron",
    "proportional",
    "is_hermitian",
    "rel_err",
    "check_msg_psd",
    "gen_eigval_problem",
    "multi_tensor_rank",
    "entropy",
    "fidelity",
    # graphs
    "write_exp_bonddim_to_graph",
    "divide_graph",
    "cycle_cutnumber_ranking",
    "cycle_length_ranking",
    # operator chains and operator layers
    "is_disjoint_layer",
    "get_disjoint_subsets_from_opchains",
    # sanity checks and diagnosis
    "network_intact_check",
    "network_message_check",
    "op_layer_intact_check",
    "same_legs",
    "graph_compatible",
]

import warnings
import itertools
from typing import List, Dict, Tuple, FrozenSet

import numpy as np
import networkx as nx
import scipy.linalg as scialg
import scipy.sparse as scisparse

# -----------------------------------------------------------------------------
#                   math
# -----------------------------------------------------------------------------

def crandn(
        size: Tuple[int] = None,
        rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
    """
    Draw random samples from the standard complex normal (Gaussian)
    distribution.
    """
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)


def delta_tensor(
        nLegs: int,
        chi: int
    ) -> np.ndarray:
    T = np.zeros(shape = nLegs * [chi])
    idx = nLegs * (np.arange(chi),)
    T[idx] = 1
    return T


def multi_kron(*ops, create_using: str = "numpy"):
    """
    Tensor product of all the given operators.
    """
    if create_using == "numpy":
        res_op = 1
        for op in ops: res_op = np.kron(op,res_op)
        return res_op

    if "scipy" in create_using:
        if not all(scisparse.issparse(op) for op in ops):
            raise RuntimeError("There are non-sparse operators.")

        format = create_using.split(sep=".")[-1]
        if format not in ("bsr", "coo", "csc", "csr", "dia", "dok", "lil"):
            raise ValueError("".join((
                "multi_kron not implemented for method ",
                create_using,
                ". ",
                format,
                " is not a scipy.sparse array type."
            )))

        res_op = ops[0]
        for op in ops[1:]:
            res_op = scisparse.kron(op, res_op, format=format)
        return res_op

    raise ValueError(
        "".join(("multi_kron not implemented for method ", create_using, "."))
    )


def proportional(
        A: np.ndarray,
        B: np.ndarray,
        decimals: int = None,
        verbose: bool = False
    ) -> bool:
    """
    Returns `True` if `A` and `B` are proportional to each other.
    Zero is defined to be proportional to zero.

    This is accurate up to `decimals` decimal places.

    raises `ValueError` if `A` and `B` have different shapes.
    """
    if np.isnan(A).any() or np.isnan(B).any():
        warnings.warn(
            "A or B contain NaN, and I don't know what happes then.",
            UserWarning
        )

    if not A.shape == B.shape:
        raise ValueError("A and B must have the same shapes.")

    if np.allclose(A,0) and np.allclose(B,0):
        warnings.warn(
            "Assuming zero to be proportional to zero.",
            UserWarning
        )
        return True

    A0 = A.flatten()[np.logical_not(np.isclose(A.flatten(), 0))]
    B0 = B.flatten()[np.logical_not(np.isclose(B.flatten(), 0))]

    if not A0.shape == B0.shape:
        if verbose: print("A and B have different amounts of zeros.")
        return False

    div = (A0 / B0)

    if decimals != None:
        div = np.unique(np.round(div,decimals=decimals))
    else:
        div = np.unique(div)
    if len(div) != 1:
        if verbose: print("There is no unique proportionality factor.")
        return False

    return np.allclose(div[0] * B, A)


def is_hermitian(A,threshold: float = 1e-8,verbose: bool = False):
    if np.allclose(A,A.T.conj(),atol=threshold):
        return True
    else:
        if verbose:
            diff = np.linalg.norm(A - A.T.conj())
            warnings.warn(f"Difference from hermiticity: {diff:.5e}.")
        return False


def rel_err(ref: float, approx: float) -> float:
    """
    Relative error `||ref - approx|| / ||ref||`.
    Calls `np.linalg.norm` Internally.
    """
    return np.linalg.norm(ref - approx) / np.linalg.norm(ref)


def check_msg_psd(
        G: nx.MultiGraph,
        threshold: float = 1e-8,
        verbose: bool = False
    ) -> bool:
    """
    Checks whether all the messages in `G` are positive semi-definite.
    """
    # TODO: implement
    raise NotImplementedError


def gen_eigval_problem(
        A: np.ndarray,
        B: np.ndarray,
        maxcond: float = 1e6,
        eps: float = 1e-5
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves the generalized eigenvalue problem, using the
    workaround introduced [here](https://arxiv.org/abs/1903.11240v3).
    Tikhonov regularization is used if the condition number
    of `B` is larger than `maxcond`.
    The parameter `eps` is used for Tikhonov regularization,
    if `B` is singular. `A` is written over during computation.

    Returns `eigvals,eigvecs` as a tuple, where `eigvecs[:,i]` is
    the eigenvector to `eigvals[i]`.
    """
    cond = np.linalg.cond(B)

    if cond > maxcond: # B is singular, employing Tikhonov-regularization
        B_inverse = np.linalg.inv(B + eps * np.eye(B.shape[0]))
        return scialg.eig(B_inverse @ A, overwrite_a=True)
    else:
        return scialg.eig(
            a=A,
            b=B,
            overwrite_a=True,
            overwrite_b=True,
        )

    if is_hermitian(A) and is_hermitian(B):
        eig_solver = np.linalg.eigh
    else:
        eig_solver = np.linalg.eig

    lambda_B,U_B = eig_solver(B)

    if np.any(np.isclose(lambda_B,0)): lambda_B += eps
    U_B_tilde = U_B / np.diag(np.sqrt(lambda_B))

    A_tilde = U_B_tilde.conj().T @ A @ U_B_tilde

    lambda_A,U_A = eig_solver(A_tilde)

    return lambda_A,U_B_tilde @ U_A


def multi_tensor_rank(T: np.ndarray, threshold: float = 1e-8) -> Tuple[int]:
    """
    Computes the rank of all matricizations of `T`, where the
    matricizations are obtained by grouping all dimensions
    except one. Singular values below `threshold` are considered
    zero.

    Returns an array, where `rank[i]` is the rank of the `i`-mode
    matricization.
    """
    rank = ()
    newaxes = [_ for _ in range(1, T.ndim)] + [0,]
    for l in range(T.ndim):
        # l-mode matricization and it's singular value decomposition
        mat = np.reshape(T, newshape=(T.shape[0],-1))
        singvals = np.linalg.svd(mat, full_matrices=False, compute_uv=False)
        rank += (
            mat.shape[0] - np.sum(np.isclose(singvals, 0, atol=threshold)),
        )

        # rolling axes for next matricization
        T = np.transpose(T, axes=newaxes)

    return rank


def entropy(p: np.ndarray, alpha: int = 1) -> float:
    """
    Entanglement entropy of the distribution `p`. Returns Shannon
    entropy for `alpha = 1` (default), and Rényi-entropy
    otherwise.
    """
    if not alpha > 0:
        raise ValueError(f"Value {alpha} illegal; expected larger than zero.")
    if not p.ndim == 1 or not np.isclose(np.sum(p), 1):
        raise ValueError("p must be a probability distribution.")

    # dropping zeros
    p = p[p != 0]

    if alpha == 1:
        # shannon entropy
        return (-1) * np.sum(p * np.log2(p))

    return np.sum(p**alpha) / (1 - alpha)


def fidelity(psi: np.ndarray, subspace: Tuple[np.ndarray]) -> float:
    """
    Measuring expectation value of the projector on the subspace.
    The subspace is defined by the tuple `subspace`, containing
    the basis states. States are normalized during computation.
    """
    subspace_ = np.array(subspace)
    if not psi.ndim == 1:
        raise ValueError("psi must be a vector.")
    if not subspace_.shape[1] == psi.shape[0]:
        raise ValueError("Vectors in subspace do not match psi.")

    F = np.sum(
        np.abs(subspace_ @ psi)**2
        / np.diag(subspace_ @ subspace_.T.conj())
    )
    F /= np.dot(psi, psi.conj())

    # sanity check
    if not np.isclose(np.imag(F), 0):
        raise RuntimeError("Complex Fidelity; something went wrong.")

    return np.real_if_close(F)


# -----------------------------------------------------------------------------
#                   graph routines
# -----------------------------------------------------------------------------


def write_exp_bonddim_to_graph(
        G: nx.MultiGraph,
        D: int,
        max_chi: int = np.inf
    ) -> None:
    """
    Writes bond dimension to the edges of `G`, that is required for
    exact solutions. `G` is modified in-place. `D` is the physical
    dimension. Bond dimension is cut off at `chi_max`.

    On loopy networks, the required bond dimension is obtained from
    a spanning tree of a graph. This is a heuristic, that is designed
    to prevent bond dimension bottlenecks. Results are exact on edges
    that are not part of a loop.
    """
    N = G.number_of_nodes()

    if nx.is_tree(G):
        # if the graph is a tree, setting bond dimensions exactly is easy:
        # the tree is cut along the respective edge, and the sizes of the
        # resulting hilbert space determine the necessary bond dimension
        for node1,node2 in G.edges():
            # setting bond dimension using tree
            Gcopy = nx.MultiGraph(G.edges)
            Gcopy.remove_edge(node1, node2)
            comp = nx.node_connected_component(Gcopy, node1)
            N_subsystem = len(comp)

            # setting the bond dimension
            chi = min(D**N_subsystem, D**(N - N_subsystem))
            G[node1][node2][0]["size"] = min(chi, max_chi)

        return

    # if the graph is not a tree, we find a new breadth-first spanning tree
    # for every edge, and then cut the respective edge. This is only a
    # heuristic, but should prevent bond dimension bottlenecks. This
    # procedure becomes exact on trees, since the spanning tree of a tree
    # is unique

    for node1, node2 in G.edges():
        # spanning tree that contains edge (node1,node2)
        tree = nx.MultiGraph(nx.bfs_tree(G=G, source=node1).edges)
        # conversion because connected components are not implemented on
        # directed graphs
        tree.remove_edge(node1, node2)
        comp = nx.node_connected_component(tree, node1)
        N_subsystem = len(comp)

        # setting the bond dimension
        chi = min(D**N_subsystem, D**(N - N_subsystem))
        G[node1][node2][0]["size"] = min(chi, max_chi)

    return


def divide_graph(G: nx.MultiGraph) -> FrozenSet[FrozenSet[int]]:
    """
    Finds bipartition cuts of the graph `G`. A bipartition cut is a set
    of edges such that, if these edges are cut, the resulting graph is
    disjoint. **EXPONENTIAL RUNTiME** This could be used to find a good
    initial guess for the bond dimension on loopy geometries, but this
    method is prohibitively slow on all but the most basic graphs.
    """
    # The thinking is as follows: For every edge (u,v), I look for a cut of the
    # graph such that u is contained in one component, and v in the other.
    # Since a graph might have loops, in order to find a cut of the graph, I'll
    # need to cut all the paths between u and v. The cuts associated with edge
    # (u,v) are thus all possible ways of cutting all the paths between u and
    # v. All cuts are saved as sets since my algorithm generates many
    # duplicates.
    cuts = frozenset()

    for node1, node2 in G.edges():
        paths = nx.all_simple_paths(G,source=node1, target=node2)
        paths = tuple(
            tuple(
                frozenset((path[i], path[i+1]))
                for i in range(len(path)-1)
            )
            for path in paths
        )
        """a single path consists of the edges as sets"""

        for edge_collection in itertools.product(*paths):
            cut = frozenset(edge_collection)
            cuts = cuts.union(frozenset((cut,)))

    return cuts


# -----------------------------------------------------------------------------
#                   ranking edges in graphs
# -----------------------------------------------------------------------------


def cycle_cutnumber_ranking(
        G: nx.Graph,
        noisy: bool = True
    ) -> List[Tuple[int]]:
    """
    Returns a ranking of the edges in `G` based on the number of simple
    cycles that they appear in. Edges, that are present in many cycles,
    are sorted to the beginning. If `noisy = True`, gaussian noise is
    added to the scores s.t. the same graph can produce different
    orderings where scores are otherwise equal.
    """
    cycles = nx.cycle_basis(G)
    cycle_graphs = [
        nx.Graph([
            (cycle[i], cycle[i+1])
            for i in range(len(cycle)-1)
        ])
        for cycle in cycles
    ]

    edge_score = lambda edge: sum(tuple(
        cycle_graph.has_edge(*edge)
        for cycle_graph in cycle_graphs
    ))
    edges_ranked = sorted(
        G.edges(),
        key=lambda x:
            edge_score(x)
            + (np.random.normal(loc=0, scale=.1) if noisy else 0),
        reverse=True
    )

    return edges_ranked


def cycle_length_ranking(G: nx.Graph,noisy: bool = True) -> List[Tuple[int]]:
    """
    Returns a ranking of the edges in `G` based on the length of the
    simple cycles that they appear in. The edges that belong to the
    shortest loops are sorted to the front. If `noisy = True`, gaussian
    noise is added to the scores s.t. the same graph can produce
    different orderings where scores are otherwise equal.
    """
    cycles = nx.cycle_basis(G)
    cycle_graphs = [
        nx.Graph([
            (cycle[i], cycle[i+1])
            for i in range(len(cycle)-1)
        ])
        for cycle in cycles
    ]

    edge_score = lambda edge: min(
        cycle_graph.number_of_edges() if cycle_graph.has_edge(*edge)
        else np.inf
        for cycle_graph in cycle_graphs
    )
    edges_ranked = sorted(
        G.edges(),
        key=lambda x:
            edge_score(x)
            + (np.random.normal(loc=0, scale=.1) if noisy else 0),
        reverse=False
    )

    return edges_ranked


# -----------------------------------------------------------------------------
#                   Operator chains & operator layers
# -----------------------------------------------------------------------------


def is_disjoint_layer(
        layer: Tuple[Dict[int, tuple]],
        op_chain: Dict[int, tuple] = dict()
    ) -> bool:
    """
    Tests if the set `op_chains` of operator chains is disjoint,
    i.e. the operator chains in `op_chains` act on different sites.
    If `op_chain` is given, tests if `op_chains` is disjoint upon
    addition of `op_chain`.
    """
    new_sites = set(op_chain.keys())
    for op_chain_ in layer:
        if len(new_sites & set(op_chain_.keys())) > 0: return False

    return True


def get_disjoint_subsets_from_opchains(
        op_chains: Tuple[Dict[int, tuple]]
    ) -> Tuple[Tuple[Tuple[Dict[int, tuple]]]]:
    """
    Given operator chains, decomposes them into as many
    disjoint subsets as are necessary for a brick wall
    layout. Returns a tuple containing the single-site
    layers, and a tuple containing the multi-site layers
    (aka brick-wall layers).
    """
    # layers of the hamiltonian are sets of operators, s.t. all operators
    # within the layer commute. This is achieved by grouping spatially disjoint
    # operators together in the layers. Single-site operators will be grouped
    # separately.
    singlesite_layers = [(),]
    brick_wall_layers = [(),]

    for op_chain in op_chains:
        if len(op_chain.keys()) == 1:
            # single-site operators will be grouped in one chain
            iLayer = 0
            while not is_disjoint_layer(singlesite_layers[iLayer], op_chain):
                iLayer += 1
                if len(singlesite_layers) == iLayer: singlesite_layers += [(),]
            singlesite_layers[iLayer] += (op_chain,)

        else:
            # single-site operators will be grouped in one chain
            iLayer = 0
            while not is_disjoint_layer(brick_wall_layers[iLayer], op_chain):
                iLayer += 1
                if len(brick_wall_layers) == iLayer: brick_wall_layers += [(),]
            brick_wall_layers[iLayer] += (op_chain,)

    return tuple(singlesite_layers), tuple(brick_wall_layers)


# -----------------------------------------------------------------------------
#                   sanity checks & diagnosis
# -----------------------------------------------------------------------------


def network_intact_check(G: nx.MultiGraph) -> bool:
    """
    Checks if the given tensor network `G` is intact.
    """
    # this library only works with nx.MultiGraph
    if not isinstance(G, nx.MultiGraph):
        raise TypeError("G must be a MultiGraph.")

    # two legs in every edge's legs attribute?
    for node1,node2,key in G.edges(keys=True):
        if G[node1][node2][key]["trace"]:
            # trace edge
            if len(G[node1][node2][key]["indices"]) != 2:
                warnings.warn(
                    "".join((
                        "Wrong number of legs in trace edge",
                        f"({node1},{node2},{key})."
                    )),
                    UserWarning
                )
                return False
        else:
            # default edge
            if len(G[node1][node2][key]["legs"].keys()) != 2:
                warnings.warn(
                    f"Wrong number of legs in edge ({node1},{node2},{key}).",
                    UserWarning
                )
                return False

    # all edges in the graph accounted for?
    for node in G.nodes:
        legs = [leg for leg in range(len(G.adj[node]))]
        for node1,node2,key in G.edges(node,keys=True):
            try:
                if not G[node1][node2][key]["trace"]:
                    legs.remove(G[node1][node2][key]["legs"][node])
                else:
                    # trace edge
                    i1,i2 = G[node1][node2][key]["indices"]
                    legs.remove(i1)
                    legs.remove(i2)
            except ValueError:
                warnings.warn(f"Wrong leg in edge ({node1},{node2},{key}).")
                return False

    return True


def network_message_check(G: nx.MultiGraph) -> bool:
    """
    Checks if the network is intact, and verifies that there are
    no double edges and no trace edges. If messages are present,
    checks if there is one message in each direction on each edge.
    """
    if not network_intact_check(G): return False

    for node1,node2,key,data in G.edges(keys=True, data=True):
        if node1 == node2:
            warnings.warn(f"Trace edge from {node1} to {node2}.")
            return False
        if key != 0:
            warnings.warn(f"Multiple edges connecting {node1} and {node2}.")
            return False
        if "msg" in data.keys():
            if data["msg"] != {}:
                if (not node1 in data["msg"].keys()
                    or not node2 in data["msg"].keys()):
                    warnings.warn(
                        f"Wrong nodes in msg-value of edge ({node1},{node2}).",
                        UserWarning
                    )
                    return False
                if len(data["msg"].values()) != 2:
                    warnings.warn(
                        f"Wrong number of messages on edge ({node1},{node2}).",
                        UserWarning
                    )
                    return False

    return True


def op_layer_intact_check(
        G: nx.MultiGraph,
        layer: Tuple[Dict[int, np.ndarray]],
        target_chain_length: int = None,
        test_same_length: bool = False,
        test_disjoint: bool = False
    ) -> bool:
    """
    Tests if the operator chains in `op_chains` are intact.
    This amounts to:
    * Are all physical dimensions equal?
    * Are the chains disjoint? (tested only if `test_disjoint = True`)
    * Do all chains have the same length? (tested only if
    `test_same_length = True`)
        * Do all chains have the correct length? (tested only if the
        reference length is given under `target_chain_length`)
    """
    if test_disjoint:
        # are the operator chains disjoint?
        if not is_disjoint_layer(layer=layer):
            warnings.warn("The operator chains are not disjoint.", UserWarning)
            return False

    D_set = set()
    length_set = set()
    for iChain, op_chain in enumerate(layer):
        # saving operator chain length
        length_set.add(len(op_chain))

        for node, T in op_chain.items():
            # does the graph contain this node?
            if not G.has_node(node):
                warnings.warn(
                    "".join((
                        f"Node {node} in operator chain {iChain}",
                        "not contained in graph.",
                    )),
                    UserWarning
                )
                return False

            # is the operator square?
            if not T.shape[0] == T.shape[1]:
                warnings.warn(
                    "".join((
                        f"Non-square operator on node {node} in operator ",
                        f"chain {iChain}."
                    )),
                    UserWarning
                )
                return False

            # saving physical dimension
            D_set.add(T.shape[0])

    if not len(D_set) == 1:
        warnings.warn(
            "Multiple different physical dimensions in operator chains.",
            UserWarning
        )
        return False

    if target_chain_length is not None: test_same_length = True

    if test_same_length:
        if not len(length_set) == 1:
            warnings.warn(
                "Operator layer contains chains with varying length.",
                UserWarning
            )
            return False

        if target_chain_length is not None:
            chain_length = length_set.pop()
            if chain_length != target_chain_length:
                warnings.warn(
                    f"Wrong operator chain length; received {chain_length}, "
                    + f"expected {target_chain_length}.",
                    UserWarning
                )
                return False

    return True


def same_legs(G1: nx.MultiGraph,G2: nx.MultiGraph) -> bool:
    """
    Checks if the leg orderings in `G1` and `G2` are the same.
    """
    if not nx.utils.edges_equal(G1.edges(), G2.edges()):
        raise ValueError("G1 and G2 do not have equal edges.")
    if not nx.utils.nodes_equal(G1.nodes(), G2.nodes()):
        raise ValueError("G1 and G2 do not have equal nodes.")

    for node1, node2, legs in G1.edges(data="legs"):
        if G2[node1][node2][0]["legs"] != legs: return False

    return True


def graph_compatible(G1: nx.MultiGraph,G2: nx.MultiGraph) -> bool:
    """
    Tests if `G1` and `G2` can be combined into a sandwich, that is if
    their geometry is the same. This amounts to checking if every edge
    in `G1` is contained in `G2`.

    Throws `ValueError` if there are two edges between any two nodes in
    `G1` or `G2`.
    """
    # sanity check
    assert network_message_check(G1)
    assert network_message_check(G2)
    assert nx.utils.nodes_equal(G1.nodes(), G2.nodes())
    assert nx.utils.edges_equal(G1.edges(), G2.edges())

    return True


if __name__ == "__main__":
    pass
