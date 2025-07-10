"""
Random stuff that is useful here or there.
"""

__all__ = [
    # mathematics
    "crandn",
    "delta_tensor",
    "multi_kron",
    "proportional",
    "rel_err",
    "check_msg_psd",
    "gen_eigval_problem",
    "multi_tensor_rank",
    "entropy",
    "fidelity",
    "suzuki_recursion_coefficients",
    # hermiticity
    "is_hermitian_matrix",
    "is_hermitian_message",
    "is_hermitian_environment",
    # graphs
    "write_exp_bonddim_to_graph",
    "divide_graph",
    "write_callable_bonddim_to_graph",
    "cycle_cutnumber_ranking",
    "cycle_length_ranking",
    # operator chains and operator layers
    "is_disjoint_layer",
    "get_disjoint_subsets_from_opchains",
    # sanity checks and diagnosis
    "network_intact_check",
    "network_message_check",
    "same_legs",
    "graph_compatible",
    # runtime routines
    "dtau_to_nSteps",
]

import warnings
import itertools
from functools import cache
from typing import Callable, Union
import copy

import numpy as np
import networkx as nx
import scipy.linalg as scialg
import scipy.sparse as scisparse
import scipy.optimize as sciopt
import tqdm

# -----------------------------------------------------------------------------
#                   Math
# -----------------------------------------------------------------------------

def crandn(
        size: Union[int, tuple[int]] = None,
        rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
    """
    Draw random samples from the standard complex normal (Gaussian)
    distribution.
    """
    # 1/sqrt(2) is a normalization factor.
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)


def delta_tensor(
        nLegs: int,
        chi: int,
        dtype=np.complex128
    ) -> np.ndarray:
    T = np.zeros(shape=nLegs * [chi], dtype=dtype)
    idx = nLegs * (np.arange(chi),)
    T[idx] = 1
    return T


def multi_kron(*ops, create_using: str = "numpy"):
    """
    Tensor product of all the given operators.
    """
    if create_using == "numpy":
        res_op = np.ones(shape=(1,))
        for op in ops: res_op = np.kron(res_op, op)
        return res_op

    if "scipy" in create_using:
        if not all(scisparse.issparse(op) for op in ops):
            raise ValueError("There are non-sparse operators.")

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
            res_op = scisparse.kron(res_op, op, format=format)
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
        with tqdm.tqdm.external_write_mode():
            warnings.warn(
                "A or B contain NaN, and I don't know what happens then.",
                UserWarning
            )

    if not A.shape == B.shape:
        raise ValueError("A and B must have the same shapes.")

    if np.allclose(A,0) and np.allclose(B,0):
        with tqdm.tqdm.external_write_mode():
            warnings.warn(
                "Assuming zero to be proportional to zero.",
                UserWarning
            )
        return True

    A0 = A.flatten()[np.logical_not(np.isclose(A.flatten(), 0))]
    B0 = B.flatten()[np.logical_not(np.isclose(B.flatten(), 0))]

    if not A0.shape == B0.shape:
        with tqdm.tqdm.external_write_mode():
            if verbose: print("A and B have different amounts of zeros.")
        return False

    div = (A0 / B0)

    if decimals != None:
        div = np.unique(np.round(div,decimals=decimals))
    else:
        div = np.unique(div)
    if len(div) != 1:
        if verbose:
            with tqdm.tqdm.external_write_mode():
                print("There is no unique proportionality factor.")
        return False

    return np.allclose(div[0] * B, A)


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
    ) -> tuple[np.ndarray, np.ndarray]:
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

    if cond > maxcond:
        # B is singular; employing Tikhonov-regularization.
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


def multi_tensor_rank(T: np.ndarray, threshold: float = 1e-8) -> tuple[int]:
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
        # l-mode matricization and it's singular value decomposition.
        mat = np.reshape(T, newshape=(T.shape[0],-1))
        singvals = np.linalg.svd(mat, full_matrices=False, compute_uv=False)
        rank += (
            mat.shape[0] - np.sum(np.isclose(singvals, 0, atol=threshold)),
        )

        # Rolling axes for next matricization
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


def fidelity(psi: np.ndarray, subspace: tuple[np.ndarray]) -> float:
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
        np.abs(subspace_.conj() @ psi)**2
        / np.diag(subspace_ @ subspace_.T.conj())
    )
    F /= np.dot(psi, psi.conj())

    # sanity check
    if not np.isclose(np.imag(F), 0):
        raise RuntimeError("Complex Fidelity; something went wrong.")

    return np.real_if_close(F)


@cache
def suzuki_recursion_coefficients(
        m: int,
        r: int,
        max_retries: int = 1000,
        eps: float = 1e-10,
        verbose: bool = False,
        raise_err: bool = False
    ) -> np.ndarray:
    """
    Numerical calculation of the recursion coefficients in product
    formulas. Theorem 1.1 from
    [J. Math. Phys. 32, 400-407 (1991)](https://doi.org/10.1063/1.529425).
    This function tries a maximum of `numretries` parameter combinations
    and breaks when a solution is `eps` close to zero.
    """

    # Defining the function that is to be minimized and the constraint.
    def minfunc(p: np.ndarray) -> float:
        return np.abs(np.sum(p ** m))

    constraint = {
        "type": "eq",
        "fun": lambda p: np.sum(p) - 1,
        "jac": lambda p: np.ones(shape=p.shape)
    }

    iterator = tqdm.tqdm(
        np.arange(max_retries),
        desc="Minimization trials",
        disable=not verbose
    )

    for iTrial in iterator:
        # Initial guess chosen randomly.
        p0 = np.random.normal(loc=0, scale=10, size=r)
        p0 /= np.sum(p0)

        res = sciopt.minimize(
            fun=minfunc,
            x0=p0,
            method="SLSQP",
            constraints=constraint
        )
        popt = res.x

        minval = minfunc(popt)
        if np.isclose(a=minval, b=0, rtol=0, atol=eps):
            iterator.close()
            break

    if not np.isclose(a=minval, b=0, rtol=0, atol=eps):
        msg = "".join((
            f"Parameters found give min. func value of {minfunc(popt):.3e}. ",
            f"Did not reach eps value {eps:.3e}."
        ))

        if raise_err:
            raise RuntimeError(msg)
        else:
            with tqdm.tqdm.external_write_mode():
                warnings.warn(msg, RuntimeWarning)

    if not res.success:
        raise RuntimeError("SciPy did not return successfully.")

    return popt


# -----------------------------------------------------------------------------
#                   Hermiticity of Different Objects
# -----------------------------------------------------------------------------


def hermitian_wrapper(
        diff_func: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray, float, bool], bool]:
    """
    Wraps functions that test if certain objects are hermitian. `func`
    must accept the object that is to be tested, and must return the
    distance between the object and it's complex conjugate as a numpy
    array. `kwargs` are passed to `diff_func`.
    """

    def herm_test_func(
            obj: np.ndarray,
            threshold: float = 1e-8,
            verbose: bool = False,
            **kwargs
        ) -> bool:
        # Calculating the distance between the object and it's conjugate.
        diff = diff_func(obj, **kwargs)

        if np.allclose(diff, 0, atol=threshold):
            return True

        else:
            if verbose:
                diff_norm = np.linalg.norm(diff)
                with tqdm.tqdm.external_write_mode():
                    warnings.warn(
                        f"Difference from hermiticity: {diff_norm:.5e}.",
                        RuntimeWarning
                    )
            return False

    return herm_test_func


@hermitian_wrapper
def is_hermitian_matrix(A: np.ndarray) -> np.ndarray:
    """Is the matrix `A` hermitian?"""
    # sanity check
    if not A.ndim == 2:
        raise ValueError("A is not a matrix.")

    return A - A.T.conj()


@hermitian_wrapper
def is_hermitian_message(
        msg: np.ndarray,
        antihermitian: bool = False
    ) -> np.ndarray:
    """
    Is the message `msg` hermitian?

    A message is conjugated by switching its bra- and ket-leg, while
    leaving the operator leg in its place, and taking the complex
    conjugate. Tetsts for anti-hermiticity, if `antihermitian = True`.
    """
    # sanity check
    if not msg.ndim == 3:
        raise ValueError("Messages must have three legs.")
    if not msg.shape[0] == msg.shape[2]:
        raise ValueError(
            "Physical dimensions in bra and ket legs have different sizes."
        )

    if antihermitian:
        return msg + np.transpose(msg, axes=(2, 1, 0)).conj()
    else:
        return msg - np.transpose(msg, axes=(2, 1, 0)).conj()


@hermitian_wrapper
def is_antihermitian_message(msg: np.ndarray) -> np.ndarray:
    """
    Is the message `msg` hermitian?

    A message is conjugated by switching its bra- and ket-leg, while
    leaving the operator leg in its place, and taking the complex
    conjugate.
    """
    # sanity check
    if not msg.ndim == 3:
        raise ValueError("Messages must have three legs.")
    if not msg.shape[0] == msg.shape[2]:
        raise ValueError(
            "Physical dimensions in bra and ket legs have different sizes."
        )

    return msg + np.transpose(msg, axes=(2, 1, 0)).conj()


@hermitian_wrapper
def is_hermitian_environment(env: np.ndarray) -> np.ndarray:
    """
    Is the environment `env` hermitian?

    An environment at a node is a tensor with `3 * len(adj[node])` legs.
    It's legs come in three groups: First the bra legs, followed by the
    operator legs, and finally the ket legs. Each group contains
    `len(adj[node])` legs, leading to the total of `3 * len(adj[node])`
    legs.
    """
    nLegs = env.ndim // 3

    # sanity check
    if not env.ndim == 3 * nLegs:
        raise ValueError("Environment has mismatches in numbers of legs.")
    if not all(env.shape[i] == env.shape[2*nLegs + i] for i in range(nLegs)):
        raise ValueError(
            "Environment has mismatches in physical dimensions."
        )

    bra_legs = tuple(range(nLegs))
    op_legs = tuple(nLegs + i for i in range(nLegs))
    ket_legs = tuple(2*nLegs + i for i in range(nLegs))
    env_conj = np.transpose(env, axes=ket_legs + op_legs + bra_legs).conj()

    return env -  env_conj


# -----------------------------------------------------------------------------
#                   Graph Routines
# -----------------------------------------------------------------------------


def write_exp_bonddim_to_graph(
        G: nx.MultiGraph,
        D: dict[int, int],
        max_chi: int = np.inf
    ) -> None:
    """
    Writes bond dimension to the edges of `G`, that is required for
    exact solutions. Saves bond dimension in `size` attribute of edges
    of `G`. `D` contains the physical dimension for every node in `G`.
    `G` is modified in-place.

    On loopy networks, the required bond dimension is obtained from
    a spanning tree of a graph. This is a heuristic, that is designed
    to prevent bond dimension bottlenecks. Results are exact on edges
    that are not part of a loop.
    """
    # sanity check
    if not all(node in D.keys() for node in G.nodes()):
        raise ValueError("There are nodes without physical dimension.")

    # If the graph is a tree, setting bond dimensions exactly is easy: the tree
    # is cut along the respective edge, and the sizes of the hilbert spaces on
    # the left and right of the cut determine the necessary bond dimension.
    # If the graph is not a tree, we find a breadth-first spanning tree that
    # contains this edge, and then cut the respective edge, thereby getting a
    # bipartition. This is only a heuristic, but should prevent bond dimension
    # bottlenecks. This procedure becomes exact on trees, since the spanning
    # tree of a tree is unique.

    if nx.is_tree(G):
        get_divide_tree = lambda G_, node_: nx.MultiGraph(G_.edges())
    else:
        get_divide_tree = lambda G_, node_: nx.MultiGraph(
            nx.bfs_tree(G=G_, source=node_)
        )

    for node1, node2 in G.edges():
        # Bipartition of the system using a tree.
        Gcopy = get_divide_tree(G_=G, node_=node1)
        Gcopy.remove_edge(node1, node2)
        comp1 = nx.node_connected_component(Gcopy, node1)
        comp2 = nx.node_connected_component(Gcopy, node2)

        # Hilbert space sizes on each side of the cut.
        D_left = np.prod([D[node] for node in comp1])
        D_right = np.prod([D[node] for node in comp2])

        # Setting the bond dimension.
        chi = min(D_left, D_right)
        G[node1][node2][0]["size"] = min(chi, max_chi)

    return


def divide_graph(G: nx.MultiGraph) -> frozenset[frozenset[int]]:
    """
    Finds bipartition cuts of the graph `G`. A bipartition cut is a set
    of edges such that, if these edges are cut, the resulting graph is
    disjoint. **EXPONENTIAL RUNTIME** This could be used to find a good
    initial guess for the bond dimension on loopy geometries, but this
    method is prohibitively slow on all but the most basic graphs.
    """
    # My thinking is as follows: For every edge (u, v), I look for a cut of the
    # graph such that u is contained in one component, and v in the other.
    # Since a graph might have loops, in order to find a cut of the graph, I'll
    # need to cut all the paths between u and v. The cuts associated with edge
    # (u, v) are thus all possible ways of cutting all the paths between u and
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
        """a single path consists of the edges as sets."""

        for edge_collection in itertools.product(*paths):
            cut = frozenset(edge_collection)
            cuts = cuts.union(frozenset((cut,)))

    return cuts


def write_callable_bonddim_to_graph(
        bond_dim: Union[int, Callable[[int], int]],
        G: nx.MultiGraph,
        sanity_check: bool = False
    ) -> nx.MultiGraph:
    """
    Given a bond dimension `bond_dim`, adds it to every edge in `G`.
    Intended to be used during runs of imaginary time evolution or
    compression. `bond_dim` can be an integer or a callable, where the
    callable takes an integer and returns an integer.

    Bond dimensions are saved as attributes of the graph edges, with the
    key `size`.
    """
    if not callable(bond_dim):
        bond_dim_val = copy.deepcopy(bond_dim)
        bond_dim = lambda _: bond_dim_val

    newG = nx.MultiGraph(G.edges())
    for node1, node2 in newG.edges():
        newG[node1][node2][0]["size"] = bond_dim

    if sanity_check: assert network_message_check(G=newG)

    return newG

# -----------------------------------------------------------------------------
#                   Ranking Edges in Graphs
# -----------------------------------------------------------------------------


def cycle_cutnumber_ranking(
        G: nx.Graph,
        noisy: bool = True
    ) -> list[tuple[int]]:
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


def cycle_length_ranking(G: nx.Graph,noisy: bool = True) -> list[tuple[int]]:
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
#                   Sanity Checks & Diagnosis
# -----------------------------------------------------------------------------


def network_intact_check(G: nx.MultiGraph) -> bool:
    """
    Checks if the given tensor network `G` is intact.
    """
    # This library only works with nx.MultiGraph objects.
    if not isinstance(G, nx.MultiGraph):
        raise TypeError("G must be a MultiGraph.")

    # Is the graph connected?
    if not nx.is_connected(G=G):
        with tqdm.tqdm.external_write_mode():
            warnings.warn("The graph G must be connected.", UserWarning)
        return False

    # Two legs in every edge's legs attribute?
    for node1, node2, key in G.edges(keys=True):
        if G[node1][node2][key]["trace"]:
            # Trace edge.
            if len(G[node1][node2][key]["indices"]) != 2:
                with tqdm.tqdm.external_write_mode():
                    warnings.warn(
                        "".join((
                            "Wrong number of legs in trace edge",
                            f"({node1}, {node2}, {key})."
                        )),
                        UserWarning
                    )
                return False
        else:
            # Default edge.
            if len(G[node1][node2][key]["legs"].keys()) != 2:
                with tqdm.tqdm.external_write_mode():
                    warnings.warn(
                        f"Wrong number of legs in edge ({node1}, {node2}, {key}).",
                        UserWarning
                    )
                return False

    # All edges in the graph accounted for?
    for node in G.nodes:
        legs = [leg for leg in range(len(G.adj[node]))]
        for node1,node2,key in G.edges(node,keys=True):
            try:
                if not G[node1][node2][key]["trace"]:
                    legs.remove(G[node1][node2][key]["legs"][node])
                else:
                    # Trace edge.
                    i1,i2 = G[node1][node2][key]["indices"]
                    legs.remove(i1)
                    legs.remove(i2)
            except ValueError:
                with tqdm.tqdm.external_write_mode():
                    warnings.warn(
                        f"Wrong leg in edge ({node1},{node2},{key})."
                    )
                return False

    return True


def network_message_check(G: nx.MultiGraph) -> bool:
    """
    Checks if the network is intact, and verifies that there are
    no double edges and no trace edges. If messages are present,
    checks if there is one message in each direction on each edge.
    """
    if not network_intact_check(G): return False

    for node1, node2, key, data in G.edges(keys=True, data=True):
        if node1 == node2:
            with tqdm.tqdm.external_write_mode():
                warnings.warn(
                    f"Trace edge from {node1} to {node2}.",
                    UserWarning
                )
            return False

        if key != 0:
            with tqdm.tqdm.external_write_mode():
                warnings.warn(
                    f"Multiple edges connecting {node1} and {node2}.",
                    UserWarning
                )
            return False

        if "msg" in data.keys():
            if data["msg"] != {}:
                if (not node1 in data["msg"].keys()
                    or not node2 in data["msg"].keys()):
                    with tqdm.tqdm.external_write_mode():
                        warnings.warn(
                            "".join((
                                "Wrong node in msg-value of edge ",
                                f"({node1}, {node2})."
                            )),
                            UserWarning
                        )
                    return False

                if len(data["msg"].values()) != 2:
                    with tqdm.tqdm.external_write_mode():
                        warnings.warn(
                            "".join((
                                "Wrong number of messages on edge ",
                                f"({node1}, {node2})."
                            )),
                            UserWarning
                        )
                    return False

    return True


def same_legs(G1: nx.MultiGraph, G2: nx.MultiGraph) -> bool:
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


def graph_compatible(
        G1: nx.MultiGraph,
        G2: nx.MultiGraph,
        sanity_check: bool = False
    ) -> bool:
    """
    Tests if `G1` and `G2` can be combined into a sandwich, that is if
    their geometry is the same. This amounts to checking if every edge
    in `G1` is contained in `G2` and, if present, if the physical
    dimensions match.
    """
    # sanity check
    if sanity_check:
        assert network_message_check(G1)
        assert network_message_check(G2)

    # Do nodes and edges match?
    if not nx.utils.nodes_equal(G1.nodes(), G2.nodes()): return False
    if not nx.utils.edges_equal(G1.edges(), G2.edges()): return False

    # Do the physical dimensions match?
    if (all("D" in data for _, data in G1.nodes(data=True))
        and all("D" in data for _, data in G2.nodes(data=True))):
        for node in G1:
            if G1.nodes[node]["D"] != G2.nodes[node]["D"]:
                return False

    return True


def check_msg_intact(
        msg: np.ndarray,
        target_shape: tuple[int],
        sender: int = None,
        receiver: int = None
    ) -> bool:
    """
    Checks if `msg` is intact. This amounts to:
    * Checking if `msg` has the correct shape.
    * Checking if `msg` contains only finite values (non-infinite and
    non-nan). Only checked if `check_finite = True` (default).
    """
    if not target_shape[0] == target_shape[2]:
        raise ValueError("".join((
            "Target shape must have the form (bra_size, op_size, ket_size), ",
            "where bra_size = ket_size."
        )))

    if not msg.shape == target_shape:
        with tqdm.tqdm.external_write_mode():
            warnings.warn(
                "".join((
                    f"Message from {sender} to {receiver} has ", "wrong ",
                    "shape. Expected ",
                    str(target_shape),
                    " got ",
                    str(msg.shape),
                    "."
                )),
                UserWarning
            )
        return False

    if not np.isfinite(msg).all():
        with tqdm.tqdm.external_write_mode():
            warnings.warn(
                "".join((
                    f"Message from {sender} to {receiver} contains non-",
                    "finite values."
                )),
                UserWarning
            )
        return False

    return True


# -----------------------------------------------------------------------------
#                   Runtime routines.
# -----------------------------------------------------------------------------


def dtau_to_nSteps(
        dtau: Union[float, Callable[[int], float]],
        tau_total: float,
        max_nSteps: int = 1e6
    ) -> int:
    """
    Given the time step and the total time elapsed, returns the number
    of steps. If `dtau` is callable, it is assumed that it accepts an
    integer step number and returns the time step for that step.
    """
    if not callable(dtau): return int(tau_total / dtau)

    nSteps = 0
    tau = 0
    while tau < tau_total:
        tau += dtau(nSteps)
        nSteps += 1

        if nSteps > max_nSteps:
            with tqdm.tqdm.external_write_mode():
                warnings.warn(
                    "".join((
                        "Number of steps exceeds 1 million. Stopping ",
                        "to avoid infinite loop."
                    )),
                    RuntimeWarning
                )
            break

    return nSteps


if __name__ == "__main__":
    pass
