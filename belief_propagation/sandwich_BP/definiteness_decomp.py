"""
Decomposition of a PEPO in a positive-semidefinite and a negative-definite part.
"""

import numpy as np
import itertools
import copy

from belief_propagation.sandwich_BP.PEPO import PEPO

def get_operator_chains(op:PEPO,sanity_check:bool=False) -> tuple[dict[int:tuple]]:
    """
    Returns the operator chains. An operator chain is a collection
    of operators, whose tensor product forms part of the operator
    that is `op`.

    Operator chains are returned as a dict, where nodes are keys
    and indices are values. The index for a given node is the index
    to the PEPO-tensor, that gives the respective factor in the
    operator chain.
    """
    # sanity checks
    if sanity_check: assert op.intact

    operator_chains = []

    def chain_construction_recursion(node:int,i_upstream:int,chain:dict[int,tuple],chains:list[dict[int,tuple]]) -> None:
        """
        Traversing the PEPO either downstream along the tree, or hopping
        from one branch (of the tree) to another, collecting the operator
        chains.
        """
        # this recursion breaks off eventually because the finite state automaton
        # that defines the tree does not have feedback loops

        if node != op.root:
            assert len(op.tree.pred[node]) == 1
            # parameters of the upstream node (parent)
            parent = tuple(op.tree.pred[node])[0]
            upstream_leg = op.G[node][parent][0]["legs"][node]
        else:
            # an upstream node only exists if node is not the root node
            parent = np.nan
            upstream_leg = np.nan

        # checking if the operator chain that terminates in node is non-zero
        terminal_index = tuple(i_upstream if _ == upstream_leg else op.chi - 1 for _ in range(op[node].ndim - 2)) + (slice(0,op.D),slice(0,op.D))
        if not np.allclose(op[node][terminal_index],0):
            chain = copy.deepcopy(chain)
            chain[node] = terminal_index
            chains.append(chain)

        for child,i_downstream in itertools.product(op.G.adj[node],range(op.chi)):
            if child == parent:
                # this leg is the upstream leg in the PEPO tree; this is not where the operator chain continues
                continue

            downstream_leg = op.G[node][child][0]["legs"][node]
            # assembling an index that takes us downstream in the operator chain
            index = tuple(
                i_downstream if _ == downstream_leg else
                i_upstream if _ == upstream_leg else
                op.chi - 1
                for _ in range(op[node].ndim-2)
            ) + (slice(0,op.D),slice(0,op.D))

            if np.allclose(op[node][index],0):
                # not part of any operator chain
                continue

            if index == terminal_index:
                # this operator chain terminates here
                continue

            nextchain = copy.deepcopy(chain)
            nextchain[node] = index

            chain_construction_recursion(node=child,i_upstream=i_downstream,chain=nextchain,chains=chains)

    chain_construction_recursion(node=op.root,i_upstream=np.nan,chain=dict(),chains=operator_chains)

    return tuple(operator_chains)

def print_operator_chain(op:PEPO,chain:dict[int:tuple],sanity_check:bool=False) -> None:
    """
    Prints the given operator chain.
    """
    if sanity_check: assert op.intact

    for node in chain.keys():
        print(f"op[{node}] : \n",op[node][chain[node]])

    return

if __name__ == "__main__":
    pass
