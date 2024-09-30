"""
Code from Christian Mendl - the base on which I will build. Not to be modified in any substantial way, for reference.
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
from lib.utils import crandn

def construct_network(chi:int,width:int,height:int,rng:np.random.Generator,real:bool=True,psd:bool=False):
    r"""
    Construct random tensors forming a network on a two-dimensional square lattice. `chi` is the bonding dimension.

    Tensor axis ordering convention:

        __|__
       /  2  \
     --|0   1|--
       \__3__/
          |
    """
    # random number generation
    if real:
        randn = lambda size: rng.standard_normal(size)
    else:
        randn = lambda size: crandn(size, rng)

    tensors = []
    """Tuple that contains the grid."""

    for i in range(height):
        row = []
        for j in range(width):
            dim = (
                1 if j == 0        else chi,
                1 if j == width-1  else chi,
                1 if i == 0        else chi,
                1 if i == height-1 else chi)
            if psd:
                h = int(np.sqrt(chi))
                assert h**2 == chi, "if psd=True, chi must have an integer root."
                s = 0.3 * randn((
                    1 if j == 0        else h,
                    1 if j == width-1  else h,
                    1 if i == 0        else h,
                    1 if i == height-1 else h,
                    chi))
                t = np.einsum(s, (0, 2, 4, 6, 8), s.conj(), (1, 3, 5, 7, 8), (0, 1, 2, 3, 4, 5, 6, 7)).reshape(dim) / chi**(3/4)
            else:
                t = randn(dim) / chi**(3/4)
            row.append(t)
        tensors.append(row)
    return tensors

def contract_network(tensors:list):
    r"""
    Contract a tensor network on a two-dimensional square lattice. The first row is interpreted as an MPO,
    and subsequent rows are contracted into it.

    Tensor axis ordering convention:

        __|__
       /  2  \
     --|0   1|--
       \__3__/
          |
    """
    # first row
    mpo = tensors[0]
    for row in tensors[1:]:
        mpo = [np.einsum(a, (0, 2, 4, 6), t, (1, 3, 6, 5), (0, 1, 2, 3, 4, 5)).reshape(
            (a.shape[0]*t.shape[0], a.shape[1]*t.shape[1], a.shape[2], t.shape[3]))
                for a, t in zip(mpo, row)]

    # contract the MPO
    a = mpo[0]
    for t in mpo[1:]:
        s = (a.shape[0], t.shape[1], a.shape[2]*t.shape[2], a.shape[3]*t.shape[3])
        a = np.einsum(a, (0, 6, 2, 4), t, (6, 1, 3, 5), (0, 1, 2, 3, 4, 5)).reshape(s)

    if not a.shape == (1, 1, 1, 1): warnings.warn("Unexpected shape after contracting the network; continuing.")
    return a.flatten()[0]

def block_bp(tensors:list,verbose:bool=False):
    """
    A kind of coarse-grainig inspired by the Block Belief Propagation
    algorithm (Arad, 2023: [Phys. Rev. B 108, 125111 (2023)](https://doi.org/10.1103/PhysRevB.108.125111)), which is the
    initialization of said algorithm.
    """
    # s0: Turning the upper-left 3x3 plaquettes into one plaquette; first, we construct an MPO
    mpo = tensors[0][:3]
    for row in tensors[1:3]:
        mpo = [np.einsum(a, (0, 2, 4, 6), t, (1, 3, 6, 5), (0, 1, 2, 3, 4, 5)).reshape(
            (a.shape[0]*t.shape[0], a.shape[1]*t.shape[1], a.shape[2], t.shape[3]))
                for a, t in zip(mpo, row[:3])]
    # contract virtual bonds of MPO
    s0 = mpo[0]
    for a in mpo[1:]:
        s0 = np.einsum(s0, (0, 6, 2, 4), a, (6, 1, 3, 5), (0, 1, 2, 3, 4, 5)).reshape(
            (s0.shape[0], a.shape[1], s0.shape[2]*a.shape[2], s0.shape[3]*a.shape[3]))
    if verbose: print("s0.shape:", s0.shape)

    # s1: 3x1 column at upper-right boundary
    s1 = tensors[0][3]
    for a in (tensors[1][3], tensors[2][3]):
        s1 = np.einsum(s1, (0, 1, 2, 6), a, (3, 4, 6, 5), (0, 1, 2, 3, 4, 5)).reshape(
            (s1.shape[0]*a.shape[0], s1.shape[1]*a.shape[1], s1.shape[2], a.shape[3]))
    if verbose: print("s1.shape:", s1.shape)

    # s2: 1x3 row at lower-left boundary
    s2 = tensors[3][0]
    for a in tensors[3][1:3]:
        s2 = np.einsum(s2, (0, 6, 2, 4), a, (6, 1, 3, 5), (0, 1, 2, 3, 4, 5)).reshape(
            (s2.shape[0], a.shape[1], s2.shape[2]*a.shape[2], s2.shape[3]*a.shape[3]))
    if verbose: print("s2.shape:", s2.shape)

    # s3: tensor at lower-right corner boundary
    s3 = tensors[3][3]
    if verbose: print("s3.shape:", s3.shape)

    return [[s0, s1], [s2, s3]]

def construct_initial_messages(tensors:list):
    """
    Construct initial message vectors for a tensor network on a two-dimensional square lattice.

    The returned lists each contain one message vector for every tensor in the network. The message lists contain incoming messages for the respective grid point;
    `msg_in_l[i][j]` contains the message that is passed to `tensors[i][j]` from the left. The message vectors themselves are initialised as containing only ones, and normalized
    to unity.
    """
    # initialization
    height = len(tensors)
    width  = len(tensors[0])
    msg_in_l = []
    msg_in_r = []
    msg_in_u = []
    msg_in_d = []

    for i in range(height):
        row_in_l = []
        row_in_r = []
        row_in_u = []
        row_in_d = []
        for j in range(width):
            row_in_l.append(np.ones(tensors[i][j].shape[0]) / tensors[i][j].shape[0])
            row_in_r.append(np.ones(tensors[i][j].shape[1]) / tensors[i][j].shape[1])
            row_in_u.append(np.ones(tensors[i][j].shape[2]) / tensors[i][j].shape[2])
            row_in_d.append(np.ones(tensors[i][j].shape[3]) / tensors[i][j].shape[3])
        msg_in_l.append(row_in_l)
        msg_in_r.append(row_in_r)
        msg_in_u.append(row_in_u)
        msg_in_d.append(row_in_d)

    return msg_in_l, msg_in_r, msg_in_u, msg_in_d

def message_passing_step(tensors:list,msg_in_l:list,msg_in_r:list,msg_in_u:list,msg_in_d:list):
    """
    Perform a message passing update step, using the provided incoming messages from the respective directions.
    Messages from the boundary have to be set to the dummy vector [1].
    """
    # initialization
    height = len(tensors)
    width  = len(tensors[0])
    msg_out_l = []
    msg_out_r = []
    msg_out_u = []
    msg_out_d = []

    for i in range(height):
        row_out_l = []
        row_out_r = []
        row_out_u = []
        row_out_d = []
        for j in range(width):
            # The outcoming message on one side is the result of absorbing all incoming messages on all other sides into the tensor
            msg_l = np.einsum(tensors[i][j], (0, 1, 2, 3), msg_in_r[i][j], (1,), msg_in_u[i][j], (2,), msg_in_d[i][j], (3,), (0,))
            msg_r = np.einsum(tensors[i][j], (0, 1, 2, 3), msg_in_l[i][j], (0,), msg_in_u[i][j], (2,), msg_in_d[i][j], (3,), (1,))
            msg_u = np.einsum(tensors[i][j], (0, 1, 2, 3), msg_in_l[i][j], (0,), msg_in_r[i][j], (1,), msg_in_d[i][j], (3,), (2,))
            msg_d = np.einsum(tensors[i][j], (0, 1, 2, 3), msg_in_l[i][j], (0,), msg_in_r[i][j], (1,), msg_in_u[i][j], (2,), (3,))

            # normalization
            msg_l /= np.sum(msg_l)
            msg_r /= np.sum(msg_r)
            msg_u /= np.sum(msg_u)
            msg_d /= np.sum(msg_d)
            row_out_l.append(msg_l)
            row_out_r.append(msg_r)
            row_out_u.append(msg_u)
            row_out_d.append(msg_d)
        msg_out_l.append(row_out_l)
        msg_out_r.append(row_out_r)
        msg_out_u.append(row_out_u)
        msg_out_d.append(row_out_d)

    # propagate messages along bonds
    msg_in_l_next = []
    msg_in_r_next = []
    msg_in_u_next = []
    msg_in_d_next = []
    for i in range(height):
        row_in_l = []
        row_in_r = []
        # new incoming messages from "left" direction
        row_in_l.append(np.array([1.]))
        for j in range(1, width):
            row_in_l.append(msg_out_r[i][j - 1])
        # new incoming messages from "right" direction
        for j in range(width - 1):
            row_in_r.append(msg_out_l[i][j + 1])
        row_in_r.append(np.array([1.]))
        msg_in_l_next.append(row_in_l)
        msg_in_r_next.append(row_in_r)

    # new incoming messages from "up" direction
    msg_in_u_next.append(width * [np.array([1.])])
    for i in range(1, height):
        msg_in_u_next.append(msg_out_d[i - 1])

    # new incoming messages from "down" direction
    for i in range(height - 1):
        msg_in_d_next.append(msg_out_u[i + 1])
    msg_in_d_next.append(width * [np.array([1.])])

    return msg_in_l_next, msg_in_r_next, msg_in_u_next, msg_in_d_next

def message_passing_iteration(tensors:list,numiter:int,verbose:bool=False):
    """
    Perform a message passing iteration. Algorithm taken from Kirkley, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)).
    """
    if verbose: print(f"Message passing: {numiter} iterations.")
    # initialization
    height = len(tensors)
    width  = len(tensors[0])
    msg_in_l, msg_in_r, msg_in_u, msg_in_d = construct_initial_messages(tensors)
    eps_iter = []
    """Change in message entries, to determine if the iteration converged."""

    for n in range(1, numiter + 1):
        msg_in_l_next, msg_in_r_next, msg_in_u_next, msg_in_d_next = message_passing_step(tensors, msg_in_l, msg_in_r, msg_in_u, msg_in_d)
        eps = 0
        # absolute change compared to previous iteration
        for i in range(height):
            for j in range(width):
                eps = max(eps, np.linalg.norm(msg_in_l_next[i][j] - msg_in_l[i][j]))
                eps = max(eps, np.linalg.norm(msg_in_r_next[i][j] - msg_in_r[i][j]))
                eps = max(eps, np.linalg.norm(msg_in_u_next[i][j] - msg_in_u[i][j]))
                eps = max(eps, np.linalg.norm(msg_in_d_next[i][j] - msg_in_d[i][j]))
        eps_iter.append(eps)

        msg_in_l = msg_in_l_next
        msg_in_r = msg_in_r_next
        msg_in_u = msg_in_u_next
        msg_in_d = msg_in_d_next

        if verbose: print("    iteration {:3}: eps = {:.3e}".format(n,eps))
    return msg_in_l, msg_in_r, msg_in_u, msg_in_d, eps_iter

def normalize_messages(msg_in_l:list,msg_in_r:list,msg_in_u:list,msg_in_d:list):
    """
    Normalize messages such that the inner product between messages traveling
    along the same bond but in opposite directions is one.
    """
    # initialization
    height = len(msg_in_l)
    width  = len(msg_in_l[0])
    msg_in_l_norm = []
    msg_in_r_norm = []
    msg_in_u_norm = []
    msg_in_d_norm = []

    for i in range(height):
        row_in_l = []
        row_in_r = []

        # normalize incoming messages from "left" direction
        row_in_l.append(np.array([1.]))
        for j in range(1, width):
            row_in_l.append(msg_in_l[i][j] / np.sqrt(np.abs(np.dot(msg_in_l[i][j], msg_in_r[i][j - 1]))))

        # normalize incoming messages from "right" direction
        for j in range(width - 1):
            row_in_r.append(msg_in_r[i][j] / np.sqrt(np.abs(np.dot(msg_in_r[i][j], msg_in_l[i][j + 1]))))
        row_in_r.append(np.array([1.]))
        msg_in_l_norm.append(row_in_l)
        msg_in_r_norm.append(row_in_r)

    # normalize incoming messages from "up" direction
    msg_in_u_norm.append(width * [np.array([1.])])
    for i in range(1, height):
        row_in_u = []
        for j in range(width):
            row_in_u.append(msg_in_u[i][j] / np.sqrt(np.abs(np.dot(msg_in_u[i][j], msg_in_d[i - 1][j]))))
        msg_in_u_norm.append(row_in_u)

    # normalize incoming messages from "down" direction
    for i in range(height - 1):
        row_in_d = []
        for j in range(width):
            row_in_d.append(msg_in_d[i][j] / np.sqrt(np.abs(np.dot(msg_in_d[i][j], msg_in_u[i + 1][j]))))
        msg_in_d_norm.append(row_in_d)
    msg_in_d_norm.append(width * [np.array([1.])])

    return msg_in_l_norm, msg_in_r_norm, msg_in_u_norm, msg_in_d_norm

def contract_tensors_messages(tensors:list,msg_in_l:list,msg_in_r:list,msg_in_u:list,msg_in_d:list):
    """
    Fully contract each tensor with the incoming messages.
    """
    height = len(tensors)
    width  = len(tensors[0])
    cntr = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(
                np.einsum( tensors[i][j], (0, 1, 2, 3),
                          msg_in_l[i][j], (0,),
                          msg_in_r[i][j], (1,),
                          msg_in_u[i][j], (2,),
                          msg_in_d[i][j], (3,), ()))
        cntr.append(row)
    return np.array(cntr)

def main():
    # random number generator
    rng = np.random.default_rng()

    # construct network
    chi = 9
    tensors = construct_network(chi, 4, 4, rng, real=False, psd=True)

    # reference contraction value
    c_ref = contract_network(tensors)
    print("c_ref:", c_ref)

    # approximate contraction based on modified belief propagation
    s_tensors = block_bp(tensors)
    c_s = contract_network(s_tensors)
    print("c_s:", c_s)
    assert np.isclose(c_s, c_ref)

    # message passing iteration
    numiter = 30
    msg_in_l, msg_in_r, msg_in_u, msg_in_d, eps_iter = message_passing_iteration(s_tensors, numiter)
    plt.semilogy(range(1, numiter + 1), eps_iter)
    plt.xlabel("iteration")
    plt.ylabel("absolute change of message entries")
    # plt.show()

    msg_in_l, msg_in_r, msg_in_u, msg_in_d = normalize_messages(msg_in_l, msg_in_r, msg_in_u, msg_in_d)

    # contract incoming messages with tensors
    cntr = contract_tensors_messages(s_tensors, msg_in_l, msg_in_r, msg_in_u, msg_in_d)
    print("cntr:")
    print(cntr)
    print("np.prod(cntr):", np.prod(cntr))

    # relative error
    err = abs(np.prod(cntr) - c_ref) / abs(c_ref)
    print("relative error: {:.3e}".format(err))

if __name__ == "__main__":
    main()
