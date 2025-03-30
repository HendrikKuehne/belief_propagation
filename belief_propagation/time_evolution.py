"""
Time evolution of PEPO operators.
"""

__all__ = ["get_brick_wall_layers", "operator_exponential"]

from typing import Union, Tuple, Dict

import numpy as np
import networkx as nx
import scipy.linalg as scialg

from belief_propagation.PEPO import PEPO
from belief_propagation.hamiltonians import Identity
from belief_propagation.utils import (
    get_disjoint_subsets_from_opchains,
    op_layer_intact_check
)


def get_brick_wall_layers(
        op: PEPO,
        sanity_check: bool = False
    ) -> Tuple[Tuple[Dict[int, tuple]]]:
    """
    Decomposes a PEPO into multiple layers based on the brick wall
    layout. This is accomplished by decomposing the PEPO into operator
    chains, and choosing (spatially) disjoint subsets.
    """
    # sanity check
    if sanity_check: assert op.intact

    op_chains = op.operator_chains(sanity_check=sanity_check)

    singlesite_layers, brick_wall_layers = get_disjoint_subsets_from_opchains(
        op_chains
    )

    all_layers = ()
    if singlesite_layers != ((),): all_layers += (*singlesite_layers,)
    if brick_wall_layers != ((),): all_layers += (*brick_wall_layers,)

    return all_layers


def operator_exponential(
        op: PEPO,
        trotter_order: int = 1,
        contract: bool = False,
        sanity_check: bool = False
    ) -> Union[PEPO, Tuple[PEPO]]:
    """
    Time evolution operator from trotterization. If `contract=True`,
    multiple layers are multiplied together afterwards.
    """
    if trotter_order == 1:
        op_list = __operator_exponential_first_order_trotter(
            op=op, sanity_check=sanity_check
        )

    else:
        raise NotImplementedError("".join((
            f"Time evolution from trotterization order {trotter_order} not ",
            "implemented."
        )))

    if not contract:
        # no contraction; returning the unitaries separately in a list.
        return op_list

    # contracting all unitaries.
    contracted_op = op_list[-1]
    for op in reversed(op_list[:-1]): contracted_op = op @ contracted_op

    return contracted_op


def __operator_exponential_first_order_trotter(
        op: PEPO,
        sanity_check: bool = False
    ) -> Tuple[PEPO]:
    """
    Time evolution operator from first-order trotterization.
    """
    # decomposing PEPO into layers.
    layers = get_brick_wall_layers(op=op, sanity_check=sanity_check)

    op_list = ()

    # trotterization: operator exponential for each layer separately.
    for layer in layers:
        num_sites_in_chain = len(layer[0])

        if num_sites_in_chain == 1:
            op_list += (__PEPO_exp_single_site_op_chains(
                G=op.G, op_chain_sum=layer, sanity_check=sanity_check
            ),)

        elif num_sites_in_chain == 2:
            op_list += (__PEPO_exp_two_site_op_chains(
                G=op.G, op_chain_sum=layer, sanity_check=sanity_check
            ),)

        else:
            raise NotImplementedError("".join((
                "Operator exponential for operator chains longer than three ",
                "sites is not implemented."
            )))

    return op_list


def __PEPO_exp_single_site_op_chains(
        G: nx.MultiGraph,
        op_chain_sum: Tuple[Dict[int, np.ndarray]],
        sanity_check: bool = False
    ) -> PEPO:
    """
    PEPO exponential of the sum of all operator chains in
    `op_chain_sum`. All operator chains must have length one, and must
    be disjoint.
    """
    # sanity check
    assert op_layer_intact_check(
        G=G, layer=op_chain_sum, target_chain_length=1, test_disjoint=True
    )
    if not all("D" in data.keys() for node, data in G.nodes(data=True)):
        raise ValueError("No physical dimensions saved in graph.")

    # Getting physical dimensions.
    D = {node: D for node, D in G.nodes(data="D")}

    op = Identity(G=G, D=D, sanity_check=sanity_check)
    op.check_tree = False

    for op_chain in op_chain_sum:
        for node in op_chain:
            op[node][...,:,:] = scialg.expm(op_chain[node])

    if sanity_check: assert op.intact

    return op


def __PEPO_exp_two_site_op_chains(
        G: nx.MultiGraph,
        op_chain_sum: Tuple[Dict[int, np.ndarray]],
        sanity_check: bool = False
    ) -> PEPO:
    """
    PEPO exponential of the sum of all operator chains in
    `op_chain_sum`. All operator chains must have length two, and must
    be disjoint.
    """
    # sanity check
    assert op_layer_intact_check(
        G=G, layer=op_chain_sum, target_chain_length=2, test_disjoint=True
    )
    if not all("D" in data.keys() for node, data in G.nodes(data=True)):
        raise ValueError("No physical dimensions saved in graph.")

    # Getting physical dimensions.
    D = {node: D for node, D in G.nodes(data="D")}

    op = Identity(G=G, D=D, sanity_check=sanity_check)

    # Inserting matrix exponentials into op.
    for op_chain in op_chain_sum:
        node1, node2 = op_chain.keys()
        exp_op = scialg.expm(np.kron(op_chain[node1], op_chain[node2]))

        # Re-shaping and transposing.
        exp_op = exp_op.reshape(
            D[node1], D[node2], D[node1], D[node2]
        ).transpose(
            0, 2, 1, 3
        ).reshape(
            D[node1]**2, D[node2]**2
        )

        # SVD to separate legs from different nodes. U will be inserted into
        # node1, Vh will be inserted into node2.
        U, singvals, Vh = scialg.svd(
            exp_op, full_matrices=False, overwrite_a=True
        )
        U = U @ np.diag(np.sqrt(singvals))
        Vh = np.diag(np.sqrt(singvals)) @ Vh

        # New bond dimension.
        chi = len(singvals)

        # Enlarging PEPO tensor in node1.
        shape1 = list(op[node1].shape)
        leg1 = op.G[node1][node2][0]["legs"][node1]
        shape1[leg1] = chi
        op[node1] = np.resize(op[node1], new_shape=shape1)
        # Enlarging PEPO tensor in node2.
        shape2 = list(op[node2].shape)
        leg2 = op.G[node1][node2][0]["legs"][node2]
        shape2[leg2] = chi
        op[node2] = np.resize(op[node2], new_shape=shape2)

        U = U.reshape(D[node1], D[node1], chi)
        Vh = Vh.reshape(chi, D[node2], D[node2])

        # Defining the indices at which matrix exponentials will be inserted.
        index1 = lambda i: tuple(
            i if iAdj == leg1 else 0
            for iAdj, _ in enumerate(op.G.adj[node1])
        ) + (
            slice(D[node1]), slice(D[node1])
        )
        index2 = lambda i: tuple(
            i if iAdj == leg2 else 0 for
            iAdj, _ in enumerate(op.G.adj[node2])
        ) + (
            slice(D[node2]), slice(D[node2])
        )

        # inserting matrix expoentials.
        for i in range(chi):
            op[node1][index1(i)] = U[:,:,i]
            op[node2][index2(i)] = Vh[i,:,:]

        # adjusting bond dimension
        op.G[node1][node2][0]["size"] = chi

    # the tree traversal checks are not applicable
    op.check_tree = False

    if sanity_check: assert op.intact

    return op


if __name__ == "__main__":
    pass
