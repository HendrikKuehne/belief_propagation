"""
Random stuff that is useful here or there.
"""
import numpy as np
import networkx as nx
import warnings
import scipy.linalg as scialg
import itertools

# -------------------------------------------------------------------------------
#                   math
# -------------------------------------------------------------------------------

def crandn(size=None,rng:np.random.Generator=np.random.default_rng()) -> np.ndarray:
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)

def delta_tensor(nLegs:int,chi:int) -> np.ndarray:
    T = np.zeros(shape = nLegs * [chi])
    idx = nLegs * (np.arange(chi),)
    T[idx] = 1
    return T

def multi_kron(*ops):
    """
    Tensor product of all the given operators.
    """
    res_op = 1
    for op in ops: res_op = np.kron(op,res_op)
    return res_op

def proportional(A:np.ndarray,B:np.ndarray,decimals:int=None,verbose:bool=False) -> bool:
    """
    Returns `True` if `A` and `B` are proportional to each other.
    Zero is defined to be proportional to zero.

    This is accurate up to `decimals` decimal places.

    raises `ValueError` if `A` and `B` have different shapes.
    """
    if np.isnan(A).any() or np.isnan(B).any():
        warnings.warn("A or B contain NaN, and I don't know what happes then.")

    if not A.shape == B.shape:
        raise ValueError("A and B must have the same shapes.")

    if np.allclose(A,0) and np.allclose(B,0):
        warnings.warn("Assuming zero to be proportional to zero.")
        return True

    A0 = A.flatten()[np.logical_not(np.isclose(A.flatten(),0))]
    B0 = B.flatten()[np.logical_not(np.isclose(B.flatten(),0))]

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

    return np.allclose(div[0] * B,A)

def is_hermitian(A,threshold:float=1e-8,verbose:bool=False):
    if np.allclose(A,A.T.conj(),atol=threshold):
        return True
    else:
        if verbose:
            diff = np.linalg.norm(A - A.T.conj())
            warnings.warn(f"Difference from hermiticity: {diff:.5e}.")
        return False

def rel_err(ref:float,approx:float) -> float:
    """
    Relative error `||ref - approx|| / ||ref||`. For vectors,
    euclidean distance is used. For matrices, the Frobenius
    norm is used.
    """
    return np.linalg.norm(ref - approx) / np.linalg.norm(ref)

def check_msg_psd(G:nx.MultiGraph,threshold:float=1e-8,verbose:bool=False) -> bool:
    """
    Checks whether all the messages in `G` are positive semi-definite.
    """
    # TODO: implement

def gen_eigval_problem(A:np.ndarray,B:np.ndarray,eps:float=1e-5) -> tuple[np.ndarray,np.ndarray]:
    """
    Solves the generalized eigenvalue problem, using the
    workaround introduced [here](https://arxiv.org/abs/1903.11240v3).
    The parameter `eps` is used for Tikhonov regularization,
    if `B` is singular. `A` is written over during computation.

    Returns `eigvals,eigvecs` as a tuple, where `eigvecs[:,i]` is
    the eigenvector to `eigvals[i]`.
    """
    cond = np.linalg.cond(B)

    if cond > 1e6: # B is singular, let's employ Tikhonov-regularization
        B_inverse = np.linalg.inv(B + eps * np.eye(B.shape[0]))
        return scialg.eig(B_inverse @ A,overwrite_a=True)
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

def multi_tensor_rank(T:np.ndarray,threshold:float=1e-8) -> tuple[int]:
    """
    Computes the rank of all matricizations of `T`, where the
    matricizations are obtained by grouping all dimensions
    except one. Singular values below `threshold` are considered
    zero.

    Returns an array, where `rank[i]` is the rank of the `i`-mode
    matricization.
    """
    rank = ()
    newaxes = [_ for _ in range(1,T.ndim)] + [0,]
    for l in range(T.ndim):
        # l-mode matricization and it's singular value decomposition
        mat = np.reshape(T,newshape=(T.shape[0],-1))
        singvals = np.linalg.svd(mat,full_matrices=False,compute_uv=False)
        rank += (mat.shape[0] - np.sum(np.isclose(singvals,0,atol=threshold)),)

        # rolling axes for next matricization
        T = np.transpose(T,axes=newaxes)

    return rank

# -------------------------------------------------------------------------------
#                   graph routines
# -------------------------------------------------------------------------------

def write_exp_bonddim_to_graph(G:nx.MultiGraph,D:int,max_chi:int=np.inf) -> None:
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
            Gcopy.remove_edge(node1,node2)
            comp = nx.node_connected_component(Gcopy,node1)
            N_subsystem = len(comp)

            # setting the bond dimension
            chi = min(D**N_subsystem,D**(N - N_subsystem))
            G[node1][node2][0]["size"] = min(chi,max_chi)

        return

    # if the graph is not a tree, we find a new breadth-first spanning tree
    # for every edge, and then cut the respective edge. This is only a
    # heuristic, but should prevent bond dimension bottlenecks. This
    # procedure becomes exact on trees, since the spanning tree of a tree
    # is unique

    for node1,node2 in G.edges():
        # spanning tree that contains edge (node1,node2)
        tree = nx.MultiGraph(nx.bfs_tree(G,source=node1).edges) # conversion because connected components are not implemented on directed graphs
        tree.remove_edge(node1,node2)
        comp = nx.node_connected_component(tree,node1)
        N_subsystem = len(comp)

        # setting the bond dimension
        chi = min(D**N_subsystem,D**(N - N_subsystem))
        G[node1][node2][0]["size"] = min(chi,max_chi)

    return

def divide_graph(G:nx.MultiGraph) -> frozenset[frozenset[int]]:
    """
    Finds bipartition cuts of the graph `G`. A bipartition cut is a set of edges
    such that, if these edges are cut, the resulting graph is disjoint.
    **EXPONENTIAL RUNTiME** This could be used to find a good initial guess for
    the bond dimension on loopy geometries, but this method is prohibitively
    slow on all but the most basic graphs.
    """
    # The thinking is as follows: For every edge (u,v), I look for a cut of the graph such that
    # u is contained in one component, and v in the other. Since a graph might have loops, in
    # order to find a cut of the graph, I'll need to cut all the paths between u and v. The cuts
    # associated with edge (u,v) are thus all possible ways of cutting all the paths between
    # u and v. All cuts are saved as sets since my algorithm generates many duplicates.
    cuts = frozenset()

    for node1,node2 in G.edges():
        paths = nx.all_simple_paths(G,source=node1,target=node2)
        paths = tuple(
            tuple(
                frozenset((path[i],path[i+1]))
                for i in range(len(path)-1)
            )
            for path in paths
        )
        """a single path consists of the edges as sets"""

        for edge_collection in itertools.product(*paths):
            cut = frozenset(edge_collection)
            cuts = cuts.union(frozenset((cut,)))

    return cuts

# -------------------------------------------------------------------------------
#                   sanity checks & diagnosis
# -------------------------------------------------------------------------------

def network_intact_check(G:nx.MultiGraph) -> bool:
    """
    Checks if the given tensor network `G` is intact.
    """
    # two legs in every edge's legs attribute?
    for node1,node2,key in G.edges(keys=True):
        if G[node1][node2][key]["trace"]:
            # trace edge
            if len(G[node1][node2][key]["indices"]) != 2:
                warnings.warn(f"Wrong number of legs in trace edge ({node1},{node2},{key}).")
                return False
        else:
            # default edge
            if len(G[node1][node2][key]["legs"].keys()) != 2:
                warnings.warn(f"Wrong number of legs in edge ({node1},{node2},{key}).")
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

def network_message_check(G:nx.MultiGraph) -> bool:
    """
    Checks if the network is intact, and verifies that there are
    no double edges and no trace edges. If messages are present,
    checks if there is one message in each direction on each edge.
    """
    if not network_intact_check(G): return False

    for node1,node2,key,data in G.edges(keys=True,data=True):
        if node1 == node2:
            warnings.warn(f"Trace edge from {node1} to {node2}.")
            return False
        if key != 0:
            warnings.warn(f"Multiple edges connecting {node1} and {node2}.")
            return False
        if "msg" in data.keys():
            if data["msg"] != {}:
                if not node1 in data["msg"].keys() or not node2 in data["msg"].keys():
                    warnings.warn(f"Wrong nodes in msg-value of edge ({node1},{node2}).")
                    return False
                if len(data["msg"].values()) != 2:
                    warnings.warn(f"Wrong number of messages on edge ({node1},{node2}).")
                    return False

    return True

if __name__ == "__main__":
    pass
