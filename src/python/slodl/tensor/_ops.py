"""Operations on tensors."""

from __future__ import annotations

from slodl import _core
from slodl.tensor._tensor import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    """Add two tensors element by element.

    Parameters
    ----------
    a, b : Tensor
        Tensors of the same shape. Shapes are not broadcast against each
        other yet.

    Returns
    -------
    Tensor
        A new tensor holding the sums. It requires a gradient, and records
        the addition, if either input requires one.

    Raises
    ------
    ValueError
        If the two shapes differ.

    See Also
    --------
    slodl.Tensor.__add__ : The same operation, spelled ``a + b``.

    Examples
    --------
    >>> from slodl import Tensor, add
    >>> add(Tensor([1, 2]), Tensor([10, 20]))
    Tensor([11, 22])

    The result carries the history of the addition:

    >>> a = Tensor([1.0, 2.0]).requires_grad_()
    >>> add(a, Tensor([3.0, 4.0])).grad_fn
    <AddBackward>
    """
    return Tensor._from_impl(_core.add(a._impl, b._impl))
