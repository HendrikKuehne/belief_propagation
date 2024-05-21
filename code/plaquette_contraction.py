import numpy as np
import matplotlib.pyplot as plt


def construct_network(chi: int, width: int, height: int, rng: np.random.Generator, psd: bool=False):
    r"""
    Construct random tensors forming a network on a two-dimensional square lattice.

    Tensor axis ordering convention:

        __|__
       /  2  \
     --|0   1|--
       \__3__/
          |
    """
    tensors = []
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
                assert h**2 == chi
                s = 0.3 * crandn((
                    1 if j == 0        else h,
                    1 if j == width-1  else h,
                    1 if i == 0        else h,
                    1 if i == height-1 else h,
                    chi), rng)
                t = np.einsum(s, (0, 2, 4, 6, 8), s.conj(), (1, 3, 5, 7, 8), (0, 1, 2, 3, 4, 5, 6, 7)).reshape(dim) / chi**(3/4)
            else:
                t = crandn(dim, rng) / chi**(3/4)
            row.append(t)
        tensors.append(row)
    return tensors


def contract_network(tensors):
    r"""
    Contract a tensor network on a two-dimensional square lattice.

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
    assert a.shape == (1, 1, 1, 1)
    return a[0, 0, 0, 0]



def construct_initial_messages(tensors):
    """
    Construct initial message vectors for a tensor network on a two-dimensional square lattice.
    """
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


def message_passing_step(tensors, msg_in_l, msg_in_r, msg_in_u, msg_in_d):
    """
    Perform a message passing update step, using the provided incoming messages from the respective directions.
    Messages from the boundary have to be set to the dummy vector [1].
    """
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


def message_passing_iteration(tensors, numiter: int):
    """
    Perform a message passing iteration.
    """
    height = len(tensors)
    width  = len(tensors[0])
    msg_in_l, msg_in_r, msg_in_u, msg_in_d = construct_initial_messages(tensors)
    eps_iter = []
    for n in range(1, numiter + 1):
        print("iteration", n)
        msg_in_l_next, msg_in_r_next, msg_in_u_next, msg_in_d_next = message_passing_step(tensors, msg_in_l, msg_in_r, msg_in_u, msg_in_d)
        # absolute change compared to previous iteration
        eps = 0
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
    return msg_in_l, msg_in_r, msg_in_u, msg_in_d, eps_iter


def contract_tensors_messages(tensors, msg_in_l, msg_in_r, msg_in_u, msg_in_d):
    """
    Fully contract each tensor with the incoming messages.
    """
    height = len(tensors)
    width  = len(tensors[0])
    cntr = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(complex(
                np.einsum( tensors[i][j], (0, 1, 2, 3),
                          msg_in_l[i][j], (0,),
                          msg_in_r[i][j], (1,),
                          msg_in_u[i][j], (2,),
                          msg_in_d[i][j], (3,), ())))
        cntr.append(row)
    return cntr


def crandn(size=None, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)


def main():

    # random number generator
    rng = np.random.default_rng()

    # construct network
    chi = 9
    tensors = construct_network(chi, 4, 4, rng, True)

    # reference contraction value
    c_ref = contract_network(tensors)
    print("c_ref:", c_ref)

    # approximate contraction based on modified belief propagation
    # s0: upper-left 3x3 plaquette
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
    print("s0.shape:", s0.shape)
    # s1: 3x1 column at upper-right boundary
    s1 = tensors[0][3]
    for a in (tensors[1][3], tensors[2][3]):
        s1 = np.einsum(s1, (0, 1, 2, 6), a, (3, 4, 6, 5), (0, 1, 2, 3, 4, 5)).reshape(
            (s1.shape[0]*a.shape[0], s1.shape[1]*a.shape[1], s1.shape[2], a.shape[3]))
    print("s1.shape:", s1.shape)
    # s2: 1x3 row at lower-left boundary
    s2 = tensors[3][0]
    for a in tensors[3][1:3]:
        s2 = np.einsum(s2, (0, 6, 2, 4), a, (6, 1, 3, 5), (0, 1, 2, 3, 4, 5)).reshape(
            (s2.shape[0], a.shape[1], s2.shape[2]*a.shape[2], s2.shape[3]*a.shape[3]))
    print("s2.shape:", s2.shape)
    # s3: tensor at lower-right corner boundary
    s3 = tensors[3][3]
    print("s3.shape:", s3.shape)
    s_tensors = [[s0, s1], [s2, s3]]
    c_s = contract_network(s_tensors)
    print("c_s:", c_s)
    assert np.isclose(c_s, c_ref)

    # message passing iteration
    numiter = 30
    msg_in_l, msg_in_r, msg_in_u, msg_in_d, eps_iter = message_passing_iteration(s_tensors, numiter)
    print("eps_iter:", eps_iter)
    plt.semilogy(range(1, numiter + 1), eps_iter)
    plt.xlabel("iteration")
    plt.ylabel("absolute change of message entries")

    # contract incoming messages with tensors
    cntr = np.array(contract_tensors_messages(s_tensors, msg_in_l, msg_in_r, msg_in_u, msg_in_d))
    print("cntr:")
    print(cntr)
    print("np.sum(cntr):", np.sum(cntr))


if __name__ == "__main__":
    main()
