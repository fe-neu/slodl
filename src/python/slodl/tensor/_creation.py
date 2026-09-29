"""Creating tensors from a shape, rather than from data."""

from collections.abc import Sequence

from slodl import _core
from slodl.tensor._tensor import Tensor


def zeros(dims: Sequence[int]) -> Tensor:
    """Create a tensor of the given shape, filled with zeros.

    Parameters
    ----------
    dims : sequence of int
        Size of each dimension. An empty sequence gives a 0-dimensional
        tensor holding a single zero.

    Returns
    -------
    Tensor
        A new tensor of shape ``dims``, every element 0.0.

    See Also
    --------
    ones, full : Fill with one, or with a value of your choosing.
    slodl.Tensor : Create a tensor from data instead of from a shape.

    Examples
    --------
    >>> from slodl import zeros
    >>> zeros([2, 3])
    Tensor([[0, 0, 0],
            [0, 0, 0]])
    """
    return full(dims, 0.0)


def ones(dims: Sequence[int]) -> Tensor:
    """Create a tensor of the given shape, filled with ones.

    Parameters
    ----------
    dims : sequence of int
        Size of each dimension. An empty sequence gives a 0-dimensional
        tensor holding a single one.

    Returns
    -------
    Tensor
        A new tensor of shape ``dims``, every element 1.0.

    See Also
    --------
    zeros, full : Fill with zero, or with a value of your choosing.

    Examples
    --------
    >>> from slodl import ones
    >>> ones([2, 2])
    Tensor([[1, 1],
            [1, 1]])
    """
    return full(dims, 1.0)


def full(dims: Sequence[int], value: float) -> Tensor:
    """Create a tensor of the given shape, filled with ``value``.

    Parameters
    ----------
    dims : sequence of int
        Size of each dimension. An empty sequence gives a 0-dimensional
        tensor holding ``value``.
    value : float
        The value every element is set to.

    Returns
    -------
    Tensor
        A new tensor of shape ``dims``, every element ``value``.

    See Also
    --------
    zeros, ones : The common cases, spelled more briefly.

    Examples
    --------
    >>> from slodl import full
    >>> full([2, 2], 7.0)
    Tensor([[7, 7],
            [7, 7]])
    """
    return Tensor._from_impl(_core.Tensor([int(d) for d in dims], float(value)))
