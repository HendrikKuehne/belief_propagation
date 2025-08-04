"""
Time evolution of PEPO operators.
"""

__all__ = [
    "get_brick_wall_layers",
    "operator_exponential",
    "simple_update_TEBD"
]

from typing import Union, Callable
import warnings
import copy

import numpy as np
import networkx as nx
import scipy.linalg as scialg
import tqdm

from belief_propagation.PEPO import PEPO, OpChain, OpLayer, Identity
from belief_propagation.PEPS import PEPS
from belief_propagation.braket import Braket
from belief_propagation.utils import (
    graph_compatible,
    multi_kron,
    suzuki_recursion_coefficients,
    write_callable_bonddim_to_graph,
    dtau_to_nSteps
)

# -----------------------------------------------------------------------------
#                   Trotterization utilities
# -----------------------------------------------------------------------------


def __disjoint_oplayers_from_opchains(
        op_chains: tuple[OpChain],
        sanity_check: bool = False
    ) -> tuple[tuple[OpLayer]]:
    """
    Given operator chains, decomposes them into as many disjoint subsets
    as are necessary for a brick wall layout. Returns a tuple containing
    the single-site layers, and a tuple containing the multi-site layers
    (aka brick-wall layers).
    """
    # Layers of the hamiltonian are sets of operators, s.t. all operators
    # within the layer commute. This is achieved by grouping spatially disjoint
    # operators together in the layers. Single-site operators will be grouped
    # separately.
    if sanity_check: assert all(chain.intact for chain in op_chains)

    singlesite_layers = [OpLayer(),]
    brick_wall_layers = [OpLayer(),]

    for op_chain in op_chains:
        if len(op_chain) == 1:
            # Single-site operators will be grouped in one chain.
            iLayer = 0
            while not (singlesite_layers[iLayer]
                       + OpLayer(iterable=(op_chain,))).disjoint:
                iLayer += 1
                if len(singlesite_layers) == iLayer:
                    singlesite_layers += [OpLayer(),]

            singlesite_layers[iLayer] += OpLayer(iterable=(op_chain,))

        else:
            # Multiple-site operators will be grouped in one chain.
            iLayer = 0
            while not (brick_wall_layers[iLayer]
                       + OpLayer(iterable=(op_chain,))).disjoint:
                iLayer += 1
                if len(brick_wall_layers) == iLayer:
                    brick_wall_layers += [OpLayer(),]

            brick_wall_layers[iLayer] += OpLayer(iterable=(op_chain,))

    return tuple(singlesite_layers), tuple(brick_wall_layers)


def __assemble_layers_in_trotter_order(
        layers: tuple[OpLayer],
        trotter_order: int,
        sanity_check: bool = False
    ) -> tuple[OpLayer]:
    """
    Given the brick wall layers in `layers`, returns the trotterization
    to order `trotter_order`.

    `layers` contains the brick wall layers of `t*H`, where `H` is an
    operator. Let the layers be denoted as operators `O1`, `O2`, and so
    on. This function solves the problem of approximating
    `exp(O1 + O2 + ...)` to a given order in `t`, by returning the
    layers in the correct order with correct prefactors incorporated
    into the layers. Recursion rules from
    [Phys. Rev. X 11, 011020 (2021)](https://doi.org/10.1103/PhysRevX.11.011020)
    and
    [J. Math. Phys. 32, 400-407 (1991)](https://doi.org/10.1063/1.529425).
    """
    if sanity_check: assert all(layer.intact for layer in layers)

    if (not int(trotter_order) == trotter_order) or trotter_order <= 0:
        raise ValueError("".join((
            f"Received trotter_order = {trotter_order}. Trotterization order ",
            "must be a larger-than-zero integer."
        )))

    if trotter_order == 1:
        # Nothing to be done for first-order trotterization.
        return copy.deepcopy(layers)

    if trotter_order == 2:
        # Multiplying every layer by 1/2.
        newlayers = (layers[0],) + tuple(layer * .5 for layer in layers[1:])

        # Assembling the correct layer order for second-order trotterization.
        ordered_layers = newlayers[:0:-1] + newlayers

        return ordered_layers

    if trotter_order % 2 == 0:
        # Method from https://doi.org/10.1103/PhysRevX.11.011020.
        u = 1 / (4 - 4 ** (1 / (trotter_order - 1)))
        outer_layers = __assemble_layers_in_trotter_order(
            layers=tuple(layer * u for layer in layers),
            trotter_order=trotter_order - 2,
            sanity_check=sanity_check
        )
        middle_layers = __assemble_layers_in_trotter_order(
            layers=tuple(layer * (1 - 4 * u) for layer in layers),
            trotter_order=trotter_order - 2,
            sanity_check=sanity_check
        )

        return 2 * outer_layers + middle_layers + 2 * outer_layers

    if trotter_order % 2 == 1:
        # Method from https://doi.org/10.1063/1.529425.
        # Choosing m = r is a heuristic that seems to work reasonably well for
        # uneven trotterization orders; there might be more reasonable
        # choices. It is desirable to choose r as small as possible, s.t. this
        # function returns as few operator layers as necessary.
        popt = suzuki_recursion_coefficients(
            m=trotter_order,
            r=trotter_order,
            eps=1e-10,
            max_retries=10000,
            raise_err=True
        )

        newlayers: tuple[OpLayer] = ()
        for p in popt:
            newlayers += __assemble_layers_in_trotter_order(
                layers=tuple(layer * p for layer in layers),
                trotter_order=trotter_order - 1,
                sanity_check=sanity_check
            )

        return newlayers

    raise NotImplementedError


def get_brick_wall_layers(
        op: PEPO,
        t: float = 1,
        trotter_order: int = 1,
        sanity_check: bool = False
    ) -> tuple[OpLayer]:
    """
    Decomposes a PEPO into multiple layers based on the brick wall
    layout. This is accomplished by decomposing the PEPO into operator
    chains, and choosing (spatially) disjoint subsets. Returns the
    oerator layers in correct trotterization order, multiplied with the
    factor `t`.
    """
    # sanity check
    if sanity_check: assert op.intact

    op_chains: tuple[OpChain] = op.operator_chains(
        sanity_check=sanity_check,
        save_tensors=True
    )

    singlesite_layers, brick_wall_layers = __disjoint_oplayers_from_opchains(
        op_chains=op_chains, sanity_check=sanity_check
    )

    all_layers: tuple[tuple[OpChain]] = ()
    if singlesite_layers != ((),): all_layers += (*singlesite_layers,)
    if brick_wall_layers != ((),): all_layers += (*brick_wall_layers,)

    ordered_layers = __assemble_layers_in_trotter_order(
        layers=all_layers,
        trotter_order=trotter_order,
        sanity_check=sanity_check
    )

    ordered_scaled_layers = tuple(layer * t for layer in ordered_layers)

    return ordered_scaled_layers


# -----------------------------------------------------------------------------
#                   PEPO operator exponential & trotterization
# -----------------------------------------------------------------------------


def operator_exponential(
        op: PEPO,
        t: float = 1,
        trotter_order: int = 1,
        contract: bool = False,
        sanity_check: bool = False
    ) -> Union[PEPO, tuple[PEPO]]:
    """
    Time evolution operator from trotterization. If `contract=True`,
    multiple layers are multiplied together afterwards.

    Layers are returned in notational order, i.e. the returned tuple
    `(O1, O2, ..., On)` is applied to a quantum state like
    `|out> = On * ... * O2 * O1 * |in>`.
    """
    op_list = __trotter_operator_exponential(
        op=op, t=t, trotter_order=trotter_order, sanity_check=sanity_check
    )

    if not contract:
        # no contraction; returning the PEPOs separately in a list.
        return op_list[::-1]

    # Contracting all PEPOs.
    contracted_op = op_list[-1]
    for op in reversed(op_list[:-1]): contracted_op = op @ contracted_op

    return contracted_op


def __trotter_operator_exponential(
        op: PEPO,
        t: float,
        trotter_order: int,
        sanity_check: bool
    ) -> tuple[PEPO]:
    """
    Decomposes the operator into brick wall layers based on
    `trotter_order`, and calculates the operator exponential of each
    layer.
    """
    # Trotterization: decomposing PEPO into brick wall layers.
    layers = get_brick_wall_layers(
        op=op, t=t, trotter_order=trotter_order, sanity_check=sanity_check
    )

    # Operators that will be returned.
    op_list = ()

    # Operator exponential for each brick wall layer.
    for layer in layers:
        chain_length_set = layer.chain_length_set

        if chain_length_set == set((1,)):
            op_list += (__PEPO_exp_single_site_brick_wall_layer(
                G=op.G, brick_wall_layer=layer, sanity_check=sanity_check
            ),)

        elif chain_length_set == set((2,)):
            op_list += (__PEPO_exp_two_site_brick_wall_layer(
                G=op.G, brick_wall_layer=layer, sanity_check=sanity_check
            ),)

        else:
            raise NotImplementedError("".join((
                "Operator exponential for operator chains longer than three ",
                "sites, or with chains of different length, is not ",
                "implemented."
            )))

    return op_list


def __PEPO_exp_single_site_brick_wall_layer(
        G: nx.MultiGraph,
        brick_wall_layer: OpLayer,
        sanity_check: bool = False
    ) -> PEPO:
    """
    PEPO exponential of the sum of all operator chains in
    `brick_wall_layer`. All operator chains must have length one, and
    must be disjoint. In other words, this method computes
    `exp(sum(brick_wall_layer))`.
    """
    if sanity_check:
        assert brick_wall_layer.intact
        assert brick_wall_layer.disjoint
        assert brick_wall_layer.chain_length_set == set((1,))

    if not all("D" in data.keys() for node, data in G.nodes(data=True)):
        raise ValueError("No physical dimensions saved in graph.")

    # Getting physical dimensions.
    D = {node: D for node, D in G.nodes(data="D")}

    op = Identity(G=G, D=D, sanity_check=sanity_check)
    op.check_tree = False

    for op_chain in brick_wall_layer:
        for node in op_chain:
            op[node][...,:,:] = scialg.expm(op_chain[node])

    if sanity_check: assert op.intact

    return op


def __PEPO_exp_two_site_brick_wall_layer(
        G: nx.MultiGraph,
        brick_wall_layer: OpLayer,
        singval_eps: float = None,
        sanity_check: bool = False
    ) -> PEPO:
    """
    PEPO exponential of the sum of all operator chains in
    `brick_wall_layer`. All operator chains must have length two, and
    must be disjoint.

    `brick_wall_layer` is taken to be a brick wall layer of an operator
    (as it occurs during trotterization). This method then takes all
    operator chains in `brick_wall_layer`, computes their operator
    exponentials, and separates them using SVDs. All operator chain
    exponentials are inserted into one PEPO. In other words, this method
    computes `exp(sum(brick_wall_layer))`.

    Singular values close to zero are truncated. If
    `singval_eps = None`, the absolute tolerance for closeness to zero
    is the Numpy machine epsilon of the datatype.
    """
    if sanity_check:
        assert brick_wall_layer.intact
        assert brick_wall_layer.disjoint
        assert brick_wall_layer.chain_length_set == set((2,))

    if not all("D" in data.keys() for node, data in G.nodes(data=True)):
        raise ValueError("No physical dimensions saved in graph.")

    # Handling singular value epsilon.
    if singval_eps is None:
        singval_eps = lambda dtype: np.finfo(dtype).eps
        # In choosing the machine epsilon as truncation threshold, I am
        # assuming that the singular values I want to keep are reasonably close
        # to one (compare relatice precision of floating point types). After
        # looking at examples, this in fact seems to be the case.
    else:
        singval_eps__ = copy.deepcopy(singval_eps)
        singval_eps = lambda x: singval_eps__

    # Getting physical dimensions.
    D = {node: D for node, D in G.nodes(data="D")}

    op = Identity(G=G, D=D, sanity_check=sanity_check)

    # Inserting matrix exponentials into op.
    for op_chain in brick_wall_layer:
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

        # Truncating vanishing singular values.
        mask = np.logical_not(np.isclose(
            a=singvals,
            b=0,
            rtol=0,
            atol=singval_eps(singvals.dtype)
        ))
        U = U[:, mask]
        singvals = singvals[mask]
        Vh = Vh[mask, :]

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

        # Constructing the indices at which matrix exponentials will be
        # inserted.
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

        # Inserting matrix expoentials.
        for i in range(chi):
            op[node1][index1(i)] = U[:,:,i]
            op[node2][index2(i)] = Vh[i,:,:]

        # Adjusting bond dimension.
        op.G[node1][node2][0]["size"] = chi

    # The tree traversal checks are not applicable.
    op.check_tree = False

    if sanity_check: assert op.intact

    return op


# -----------------------------------------------------------------------------
#                   Simple update imaginary time evolution
# -----------------------------------------------------------------------------


def simple_update_TEBD(
        psi: PEPS,
        H: PEPO,
        dtau: Union[float, Callable[[int], float]] = 0.05,
        nSteps: int = None,
        tau_total: float = 5,
        trotter_order: int = 1,
        singval_threshold: float = None,
        min_bond_dim: Union[int, nx.MultiGraph, Callable[[int], int]] = 1,
        max_bond_dim: Union[int, nx.MultiGraph, Callable[[int], int]] = np.inf,
        return_all_states: bool = False,
        normalize_every: int = np.inf,
        normalize_with: str = "BP",
        verbose: bool = False,
        sanity_check: bool = False,
    ) -> Union[PEPS, tuple[PEPS]]:
    """
    TEBD using the simple-update method from
    [Phys. Rev. Lett. 101, 090603 (2008)](https://doi.org/10.1103/PhysRevLett.101.090603).
    Implements imaginary time evolution with a timestep of `dtau`, for
    a total (imaginary) time `tau_total`.

    The arguments `singval_threshold`, `min_bond_dim` and `max_bond_dim`
    govern the behavior of the SVDs on the bonds. During truncation of
    singular values, values with magnitude below `singval_threshold` are
    discarded. If given, `min_bond_dim` and `max_bond_dim` enforce
    strict limits on the size of the bond dimension. Accepts either an
    integer, or a `nx.MultiGraph`, in which case the `"size"` argument
    on the edges determines the bond dimension on that edge.

    If `singval_threshold is None`, the Numpy machine epsilon for the
    respective data type will be used.

    Normalizes the state every `normalize_every` steps, if given. Method
    is determined by `normalize_with` argument; currently, the choice is
    between `"BP"` and `"exact"`.

    Returns the final state if `return_all_states == False`. Otherwise,
    returns a tuple of the initial state and all states that were
    encountered during computation.
    """
    if sanity_check:
        assert psi.intact and H.intact

    # Checking compatibility of state and operator.
    if not graph_compatible(psi.G, H.G):
        raise ValueError(
            "State and operator are not compatible."
        )

    # Preparing target edge sizes.
    min_bond_dim = write_callable_bonddim_to_graph(
        G=psi.G, bond_dim=min_bond_dim, sanity_check=sanity_check
    )
    max_bond_dim = write_callable_bonddim_to_graph(
        G=psi.G, bond_dim=max_bond_dim, sanity_check=sanity_check
    )

    if not graph_compatible(psi.G, min_bond_dim):
        raise ValueError(
            "Minimum bond dimension and state have non-compatible graphs."
        )
    if not graph_compatible(psi.G, max_bond_dim):
        raise ValueError(
            "Maximum bond dimension and state have non-compatible graphs."
        )

    if singval_threshold is None:
        with tqdm.tqdm.external_write_mode():
            warnings.warn(
                "".join((
                    "No singular value threshold given; machine epsilon will ",
                    "be used. This may result in unconstrained growth of ",
                    "virtual bond dimensions."
                )),
                RuntimeWarning
            )
    elif singval_threshold <= 0:
        with tqdm.tqdm.external_write_mode():
            warnings.warn(
                "".join((
                    "Singular value threshold is smaller than or equal to ",
                    "zero. Setting to machine epsilon. This may result in ",
                    "unconstrained growth of virtual bond dimensions."
                )),
                RuntimeWarning
            )
            singval_threshold = None

    if normalize_with not in ("BP", "exact"):
        with tqdm.tqdm.external_write_mode():
            warnings.warn(
                "".join((
                    f"Normalization method {normalize_with} not implemented. ",
                    "Defaulting to normalization by BP contraction."
                )),
                UserWarning
            )
        normalize_with = "BP"

    # Inferring number of steps.
    if nSteps is None: nSteps = dtau_to_nSteps(dtau=dtau, tau_total=tau_total)

    # Handling dtau.
    if not callable(dtau):
        dtau_val = copy.deepcopy(dtau)
        dtau = lambda _: dtau_val
    dtau = tuple(dtau(iStep) for iStep in range(nSteps))

    # Trotterization of the operator.
    layers = get_brick_wall_layers(
        op=H,
        t=1,
        trotter_order=trotter_order,
        sanity_check=sanity_check
    )

    if return_all_states: all_states = (copy.deepcopy(psi),)

    for iStep in tqdm.tqdm(
        np.arange(nSteps),
        desc="TEBD itime steps",
        disable=not verbose
    ):
        # Svaling the operator layers by the timestep.
        scaled_layers = tuple(layer * (-dtau[iStep]) for layer in layers)

        for layer in scaled_layers:
            layer_magnitude = 1
            for op_chain in layer:
                if False:
                    raise NotImplementedError("".join((
                        "I tried to normalize the state without contracting ",
                        "it explicitly. This doesn't seem to work yet, so",
                        "normalization is done with explicit contraction so ",
                        "far."
                    )))
                    # Estimating the magnitude of this layer s.t. we can
                    # normalize the state later.
                    exp_layer = scialg.expm(multi_kron(*op_chain.values()))
                    singvals = scialg.svdvals(exp_layer, overwrite_a=True)
                    layer_magnitude *= np.max(np.abs(singvals))

                # How many operators in the operator chain?
                if len(op_chain.keys()) == 2:
                    # Two operators; we must extract the relevant parameters
                    # for the truncated SVD at this edge.
                    node1, node2 = op_chain.keys()
                    min_chi = min_bond_dim[node1][node2][0]["size"](iStep)
                    max_chi = max_bond_dim[node1][node2][0]["size"](iStep)

                    __simple_update_apply_op_chain_two_site(
                        psi=psi,
                        op_chain=op_chain,
                        singval_threshold=singval_threshold,
                        min_bond_dim=min_chi,
                        max_bond_dim=max_chi,
                        sanity_check=sanity_check
                    )

                elif len(op_chain.keys()) == 1:
                    # One operator; no additional parameters needed to apply
                    # the operator chain.
                    __simple_update_apply_op_chain_single_site(
                        psi=psi,
                        op_chain=op_chain,
                        sanity_check=sanity_check
                    )

                else:
                    raise NotImplementedError("".join((
                        "Encountered operator chain with ",
                        f"{len(op_chain.keys())} operators during ",
                        "simple update TEBD."
                    )))

        if iStep % normalize_every == 0:
            # Normalizing psi.
            overlap = Braket.Overlap(
                psi1=psi,
                psi2=psi,
                sanity_check=sanity_check
            )

            if normalize_with == "BP":
                norm = overlap.BP(
                    verbose=verbose,
                    iterator_desc_prefix="Normalization",
                    sanity_check=sanity_check
                )

            elif normalize_with == "exact":
                norm = overlap.contract(sanity_check=sanity_check)

            else:
                raise ValueError(
                    f"Normalization method {normalize_with} not implemented."
                )

            psi = psi * (1 / np.sqrt(norm))

        if return_all_states: all_states += (copy.deepcopy(psi),)

    if return_all_states: return all_states
    else: return psi


def __simple_update_apply_op_chain_single_site(
        psi: PEPS,
        op_chain: OpChain,
        sanity_check: bool,
    ) -> None:
    """
    Applies `exp(op_chain)` to the state `psi`. `psi` is modified
    in-place.
    """
    if sanity_check:
        assert psi.intact
        assert op_chain.intact

    if len(op_chain.keys()) != 1:
        raise ValueError("".join((
            "__simple_update_apply_op_chain_single_site received operator ",
            f"chain with {len(op_chain.keys())} operators."
        )))

    for node, op in op_chain.items():
        # Assembling indices for einsum.
        idx_in = tuple(_ for _ in range(psi[node].ndim))
        idx_out = idx_in[:-1] + (psi[node].ndim,)

        # Applying the operator exponential.
        psi[node] = np.einsum(
            psi[node], idx_in,
            scialg.expm(op), (psi[node].ndim, psi[node].ndim - 1),
            idx_out
        )

    return


def __simple_update_apply_op_chain_two_site(
        psi: PEPS,
        op_chain: OpChain,
        singval_threshold: float,
        min_bond_dim: int,
        max_bond_dim: int,
        sanity_check: bool,
    ) -> None:
    """
    Applies `exp(op_chain)` to the state `psi`. `psi` is modified
    in-place. The arguments `singval_threshold`, `min_bond_dim` and
    `max_bond_dim` determine the behavior of the truncated SVD.

    If `singval_threshold is None`, the Numpy machine epsilon for the
    respective data type will be used.
    """
    if sanity_check:
        assert psi.intact
        assert op_chain.intact

    if min_bond_dim > max_bond_dim:
        with tqdm.tqdm.external_write_mode():
            warnings.warn(
                "".join((
                    F"Minimum bond dimension {min_bond_dim} is larger than ",
                    f"maximum bond dimension {max_bond_dim}. Setting minimum ",
                    "bond dimension equal to maximum bond dimension."
                )),
                RuntimeWarning
            )
        min_bond_dim = max_bond_dim

    # Handling singular value threshold.
    if singval_threshold is None:
        singval_threshold = lambda dtype: np.finfo(dtype).eps
        # By choosing the machine epsilon as truncation threshold, one is
        # assuming that the singular values one wants to keep are reasonably
        # close to one (compare relatice precision of floating point types).
        # This might not be the case, depending on the value of dtau! Choosing
        # a threshold that is too large leads to a loss of precision.
    else:
        singval_threshold__ = copy.deepcopy(singval_threshold)
        singval_threshold = lambda x: singval_threshold__

    if len(op_chain.keys()) != 2:
        raise ValueError("".join((
            "__simple_update_apply_op_chain_single_site received operator ",
            f"chain with {len(op_chain.keys())} operators."
        )))

    node1, node2 = op_chain.keys()
    op1, op2 = op_chain.values()

    leg1 = psi.G[node1][node2][0]["legs"][node1]
    leg2 = psi.G[node1][node2][0]["legs"][node2]
    nLegs1 = psi[node1].ndim
    nLegs2 = psi[node2].ndim

    # Calculating operator exponential, and re-shaping.
    op_exp = scialg.expm(np.kron(op1, op2))
    op_exp = op_exp.reshape(
        (op1.shape[0], op2.shape[0], op1.shape[1], op2.shape[1])
    )

    # Assembling indices for einsum.
    idx_node1 = tuple(
        nLegs1 + nLegs2 if i == leg1 else i
        for i in range(nLegs1)
    )
    idx_node2 = tuple(
        nLegs1 + nLegs2 if i == leg2 else nLegs1 + i
        for i in range(nLegs2)
    )
    idx_exp = (
        nLegs1 + nLegs2 + 1, nLegs1 + nLegs2 + 2, # output legs
        nLegs1 - 1, nLegs1 + nLegs2 - 1 # input legs
    )
    idx_out = list(
        idx_node1[:-1] # remaining legs node1
        + (nLegs1 + nLegs2 + 1,) # physical leg node1
        + idx_node2[:-1] # remaining legs node2
        + (nLegs1 + nLegs2 + 2,) # physical leg node2
    )
    idx_out.remove(nLegs1 + nLegs2)
    idx_out.remove(nLegs1 + nLegs2)

    # Absorbing operator exponential into the state tensors.
    block_tensor = np.einsum(
        psi[node1], idx_node1,
        psi[node2], idx_node2,
        op_exp, idx_exp,
        idx_out
    )

    # Re-shaping block_tensor in preparation for SVD.
    size1 = np.prod([
        1 if neighbor == node2 else psi.G[node1][neighbor][0]["size"]
        for neighbor in psi.G.adj[node1]
    ]) * psi.D[node1]
    size2 = np.prod([
        1 if neighbor == node1 else psi.G[node2][neighbor][0]["size"]
        for neighbor in psi.G.adj[node2]
    ]) * psi.D[node2]
    block_tensor = block_tensor.reshape((size1, size2))

    # SVD of re-shaped block tensor. U will be inserted into node1,
    # Vh will be inserted into node2.
    U, singvals, Vh = np.linalg.svd(block_tensor, full_matrices=False)

    # Truncation.
    threshold_mask = singvals > singval_threshold(singvals.dtype)
    min_dim_mask = tuple(
        True if i < min_bond_dim else False
        for i in range(singvals.shape[-1])
    )
    max_dim_mask = tuple(
        True if i < max_bond_dim else False
        for i in range(singvals.shape[-1])
    )
    mask = np.logical_and(
        np.logical_or(threshold_mask, min_dim_mask),
        max_dim_mask
    )
    U = U[:, mask]
    singvals = singvals[mask]
    Vh = Vh[mask, :]

    # Absorbing singular values.
    U = U @ np.diag(np.sqrt(singvals))
    Vh = np.diag(np.sqrt(singvals)) @ Vh

    newsize = np.sum(mask)

    # Constructing the new shape for U.
    newshape1 = list(None for neighbor in psi.G.adj[node1])
    for neighbor in psi.G.adj[node1]:
        if neighbor == node2: continue
        leg = psi.G[node1][neighbor][0]["legs"][node1]
        size = psi.G[node1][neighbor][0]["size"]
        newshape1[leg] = size
    newshape1.remove(None)
    newshape1 += [psi.D[node1], newsize]

    # Constructing the new shape for Vh.
    newshape2 = list(None for neighbor in psi.G.adj[node2])
    for neighbor in psi.G.adj[node2]:
        if neighbor == node1: continue
        leg = psi.G[node2][neighbor][0]["legs"][node2]
        size = psi.G[node2][neighbor][0]["size"]
        newshape2[leg] = size
    newshape2.remove(None)
    newshape2 = [newsize,] + newshape2 + [psi.D[node2],]

    # Re-shaping U and Vh.
    U = U.reshape(newshape1)
    Vh = Vh.reshape(newshape2)

    # Transposing U and Vh, s.t. the axes corresponding to the edge (node1,
    # node2) are at the correct places.
    axes1 = tuple(range(0, leg1)) + (nLegs1 - 1,) + tuple(range(leg1, nLegs1 - 1))
    axes2 = tuple(range(1, leg2 + 1)) + (0,) + tuple(range(leg2 + 1, nLegs2))
    U = np.transpose(U, axes=axes1)
    Vh = np.transpose(Vh, axes=axes2)

    # Updating size of the edge, and inserting U and Vh.
    psi.G[node1][node2][0]["size"] = newsize
    psi[node1] = U
    psi[node2] = Vh

    if sanity_check: assert psi.intact

    return


if __name__ == "__main__":
    pass
