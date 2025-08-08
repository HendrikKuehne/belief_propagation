"""
Linear operators for GPUs. The contained classes are only functional if
an installation of cupy can be found.
"""

__all__ = [
    "CUPY_AVAILABLE",
    "LocalOperator",
    "LocalHamiltonianOperator",
    "LocalEnvironmentOperator",
]

from typing import Union

import numpy as np
try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as culinalg
    CUPY_AVAILABLE = True

    try:
        n_devices = cp.cuda.runtime.getDeviceCount()
    except:
        # If this fails, there is no CUDA device available.
        raise ModuleNotFoundError("No CUDA device available.")

except ModuleNotFoundError:
    CUPY_AVAILABLE = False


class _SumLocalOperator(
    culinalg._interface._SumLinearOperator if CUPY_AVAILABLE else object
):
    """
    In order to preserve the benefits of the cupy implementation of
    linear operators, yet be able to use my own implementation of
    `LocalOperator.toarray`, this class inherits from cupys
    [_SumLinearOperator](https://github.com/cupy/cupy/blob/main/cupyx/scipy/sparse/linalg/_interface.py#L364).
    """

    def toarray(self) -> np.ndarray:
        """
        The dense version of a sum of matrix-free operators is, of
        course, the sum of the dense versions of the summands.
        """
        return (self.__toarray_backend_agnostic(self.args[0])
                + self.__toarray_backend_agnostic(self.args[1]))

    @staticmethod
    def __toarray_backend_agnostic(
        A: culinalg.LinearOperator
    ) -> np.ndarray:
        """
        Provides a method to get the dense matrix of linear operators,
        independently of whether they are in fact `LocalOperator`
        instances (and thus possess a `toarray` method) or not. This is
        needed s.t. I can sum instances of SciPys `LinearOperator` and
        my own `LocalOperator`.
        """
        if hasattr(A, "toarray"):
            return A.toarray()
        else:
            # The dense version of any matrix-free operator can be obtained by
            # multiplying it with the identity matrix.
            return A.matmat(cp.eye(A.shape[0]))

    def __add__(
            self,
            B: culinalg.LinearOperator
        ) -> "_SumLocalOperator":
        return self.__class__(self, B)

    def __init__(self, A: "LocalOperator", B: "LocalOperator"):
        super().__init__(A, B)


class LocalOperator(
    culinalg.LinearOperator if CUPY_AVAILABLE else object
):
    """
    Base class for matrix-free operators, defined via tensor network
    contraction, on GPU.
    """

    def toarray(self) -> cp.ndarray:
        """
        Contracts the tensor network and returns the matrix that
        represents the operator.

        Leg ordering of the local operator: (bra virtual dimensions, bra
        physical dimension, ket virtual dimensions, ket physical
        dimension). The order of the virtual legs is inherited from the
        "legs" indices on the edges; the 0th leg ends up in the first
        dimension.
        """
        mat = cp.einsum(*self.args, self.out_legs_array, optimize=True)
        mat = cp.reshape(mat, newshape=self.shape)

        return mat

    def _matvec(self, vec: cp.ndarray) -> cp.ndarray:
        """
        Matrix-free action of `self` on a vector.
        """
        # Sanity check
        if not np.prod(vec.shape) == self.shape[1]:
            raise ValueError("".join((
                "Vector has wrong number of elements; expected ",
                f"{np.prod(self.in_shape_vec)}, got ",
                f"{np.prod(vec.shape)}."
            )))

        # Re-shaping the vector.
        vec = cp.reshape(vec, newshape=self.in_shape_vec)

        # Adding vector to einsum arguments.
        allargs = self.args + (vec, self.in_legs_vec + (...,))

        res_vec = cp.einsum(
            *allargs,
            self.out_legs_vec + (...,),
            optimize=True
        )

        return res_vec.flatten()

    def __add__(self, B: culinalg.LinearOperator) -> _SumLocalOperator:
        return _SumLocalOperator(A=self, B=B)

    def __init__(self, nLegs: int, dtype: cp.dtype, shape: tuple[int]):
        self.nLegs: int = nLegs
        """Number of constituent messages."""

        # Output leg ordering in self.toarray().
        self.out_legs_array = (tuple(range(nLegs))
                               + (3 * nLegs,)
                               + tuple(range(2 * nLegs, 3 * nLegs))
                               + (3*nLegs + 1,))

        # Input leg ordering in self._matvec().
        self.in_legs_vec = tuple(range(2 * nLegs, 3 * nLegs)) + (3*nLegs + 1,)

        # Output leg ordering in self._matvec().
        self.out_legs_vec = tuple(range(nLegs)) + (3 * nLegs,)

        # Verifying against SciPy.
        super().__init__(shape=shape, dtype=dtype)


class LocalHamiltonianOperator(LocalOperator):
    """
    Local Hamiltonian in DMRG optimization step as matrix-free linear
    operator, on GPU.
    """

    def _adjoint(self) -> "LocalHamiltonianOperator":
        """
        Hermitian conjugate, by conjugation of the constituent messages
        and the operator tensor.
        """
        # Assembling message data for hermitian conjugate.
        msgdata = tuple(
            (
                self.args[2 * i].conj().T,
                (
                    self.args[2 * i + 1][0],
                    self.args[2 * i + 1][1] - self.nLegs,
                    self.args[2 * i + 1][2] - 2 * self.nLegs
                )
            )
            for i in range(self.nLegs)
        )

        # Retrieving operator tensor for hermitian conjugate.
        W = self.args[-2]
        axes = list(range(W.ndim))
        axes[-1] = W.ndim - 2
        axes[-2] = W.ndim - 1
        W_conj = W.transpose(axes)

        return self.__class__(
            nLegs=self.nLegs,
            W=W_conj,
            msgdata=msgdata
        )

    def __init__(
            self,
            nLegs: int,
            W: Union[cp.ndarray, np.ndarray],
            msgdata: tuple[tuple[Union[cp.ndarray, np.ndarray], tuple[int]]]
        ) -> None:
        """
        Initializing the local Hamiltonian operator involves collecting
        the data that is necessary for contraction.

        Arguments:
        * `nLegs`: Number of neighbors in the graph.
        * `D`: Physical dimension.
        * `W`: Hamiltonian PEPO tensor.
        * `msgdata`: Message data. One tuple for each incoming message,
        which consists of the mesage itself, and the legs
        (`(bra_leg, op_leg, ket_leg)`). These legs are the legs of the
        respective node, i.e. they denote which leg of the site tensor
        the respective message leg connects to.
        """
        # Sanity checks.
        if W.ndim != nLegs + 2: raise ValueError("".join((
            f"Operator PEPO tensor has wring shape. Expected {nLegs + 2} ",
            f"legs, received {W.ndim}. W should have one dimension per ",
            "neighbor, plus two physical dimensions."
        )))
        if len(msgdata) != nLegs: raise ValueError("".join((
            f"Received wrong amount of message data. Expected {nLegs} message",
            f"-value pairs, got {len(msgdata)}."
        )))

        # Inferring data type.
        dtype = cp.result_type(W.dtype, *[datum[0].dtype for datum in msgdata])

        self.args = ()
        """Contraction data, in sublist format, to be passed to einsum."""

        self.in_shape_vec: tuple[int] = [None for _ in range(nLegs)]
        """
        During `self._matvec`, incoming vectors will be re-shaped into this
        shape.
        """

        self.D: int = W.shape[-1]

        for datum in msgdata:
            # Assembling contraction arguments: messages.
            self.args += (
                # Message
                cp.asarray(datum[0]),
                # (bra_leg, op_leg, ket_leg)
                tuple(i * nLegs + leg for i, leg in enumerate(datum[1]))
            )

            # Compiling vector re-shape data.
            self.in_shape_vec[datum[1][2]] = datum[0].shape[-1]

        self.in_shape_vec = tuple(self.in_shape_vec + [self.D,])

        # Assembling contraction arguments: operator tensor.
        self.args += (
            cp.asarray(W),
            (tuple(nLegs + iLeg for iLeg in range(nLegs))
             + (3 * nLegs, 3*nLegs + 1)),
        )

        super().__init__(
            nLegs=nLegs,
            shape=(np.prod(self.in_shape_vec), np.prod(self.in_shape_vec)),
            dtype=dtype
        )


class LocalEnvironmentOperator(LocalOperator):
    """
    Local environment in DMRG optimization step as matrix-free linear
    operator, on GPU.
    """

    @property
    def inv(self) -> "LocalEnvironmentOperator":
        """Matrix inverse, by inversion of the constituent messages."""
        msgdata = tuple(
            (
                cp.linalg.inv(self.args[2 * i]),
                (self.args[2*i + 1][0], self.args[2*i + 1][1] - 2*self.nLegs))
            for i in range(self.nLegs)
        )

        return self.__class__(nLegs=self.nLegs, D=self.D, msgdata=msgdata)

    def _adjoint(self) -> "LocalEnvironmentOperator":
        """
        Hermitian conjugate, by conjugation of the constituent messages.
        """
        # Assembling message data for hermitian conjugate.
        msgdata = tuple(
            (
                self.args[2 * i].conj().T,
                (self.args[2*i + 1][0], self.args[2*i + 1][1] - 2*self.nLegs))
            for i in range(self.nLegs)
        )
        return self.__class__(nLegs=self.nLegs, D=self.D, msgdata=msgdata)

    def __init__(
            self,
            nLegs: int,
            D: int,
            msgdata: tuple[tuple[Union[cp.ndarray, np.ndarray], tuple[int]]]
        ) -> None:
        """
        Initializing the local Hamiltonian operator involves collecting
        the data that is necessary for contraction.

        Arguments:
        * `nLegs`: Number of neighbors in the graph.
        * `D`: Physical dimension.
        * `msgdata`: Message data. One tuple for each incoming message,
        which consists of the mesage itself and the legs
        (`(bra_leg, ket_leg)`).
        """
        # Sanity checks.
        if len(msgdata) != nLegs: raise ValueError("".join((
            f"Received wrong amount of message data. Expected {nLegs} message",
            f"-value pairs, got {len(msgdata)}."
        )))

        # Inferring data type.
        dtype = cp.result_type(*[datum[0].dtype for datum in msgdata])

        self.args = ()
        """Contraction data, in sublist format, to be passed to einsum."""

        self.in_shape_vec: tuple[int] = [None for _ in range(nLegs)]
        """
        During `self._matvec`, incoming vectors will be re-shaped into this
        shape.
        """

        self.D = D

        for datum in msgdata:
            if not datum[0].ndim == 2: raise ValueError("".join((
                "Constituent messages of a local environment tensor must ",
                f"have two legs; received {datum[0].ndim} legs."
            )))

            # Assembling contraction arguments: messages.
            self.args += (
                # Message
                cp.asarray(datum[0]),
                # (bra_leg, ket_leg)
                (datum[1][0], 2*nLegs + datum[1][1])
            )

            # Compiling vector re-shape data.
            self.in_shape_vec[datum[1][1]] = datum[0].shape[-1]

        self.in_shape_vec = tuple(self.in_shape_vec + [self.D,])

        # Assembling contraction arguments: Identity for the physical legs.
        self.args += (cp.eye(self.D), (3 * nLegs, 3*nLegs + 1))

        super().__init__(
            nLegs=nLegs,
            shape=(np.prod(self.in_shape_vec), np.prod(self.in_shape_vec)),
            dtype=dtype
        )


if __name__ == "__main__":
    pass
