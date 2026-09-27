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


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Multiply two tensors element by element.

    This is the element-wise (Hadamard) product, not a matrix product.

    Parameters
    ----------
    a, b : Tensor
        Tensors of the same shape. Shapes are not broadcast against each
        other yet, so multiplying by a single number is not supported.

    Returns
    -------
    Tensor
        A new tensor holding the products. It requires a gradient, and
        records the multiplication, if either input requires one.

    Raises
    ------
    ValueError
        If the two shapes differ.

    See Also
    --------
    slodl.Tensor.__mul__ : The same operation, spelled ``a * b``.

    Examples
    --------
    >>> from slodl import Tensor, mul
    >>> mul(Tensor([2, 3]), Tensor([10, 20]))
    Tensor([20, 60])

    Each input's gradient is the other input's value:

    >>> a = Tensor(3.0).requires_grad_()
    >>> b = Tensor(4.0).requires_grad_()
    >>> (a * b).backward()
    >>> a.grad
    Tensor(4)
    >>> b.grad
    Tensor(3)
    """
    return Tensor._from_impl(_core.mul(a._impl, b._impl))


def sum(a: Tensor) -> Tensor:
    """Add up every element of a tensor.

    This is how a tensor becomes the single value that
    :meth:`slodl.Tensor.backward` can start from.

    Parameters
    ----------
    a : Tensor
        Tensor to add up, which may be a view.

    Returns
    -------
    Tensor
        A 0-dimensional tensor holding the total, zero for an empty tensor.
        It requires a gradient, and records the sum, if ``a`` does.

    See Also
    --------
    slodl.Tensor.sum : The same operation, spelled ``a.sum()``.

    Notes
    -----
    Every element contributes to the total equally, so a backward pass gives
    each one the same gradient. Python's built-in ``sum`` does not work on a
    tensor; use this instead.

    Examples
    --------
    >>> import slodl
    >>> from slodl import Tensor
    >>> slodl.sum(Tensor([[1, 2], [3, 4]]))
    Tensor(10)

    Summing is what makes a non-scalar computation differentiable:

    >>> a = Tensor([2.0, 3.0]).requires_grad_()
    >>> b = Tensor([10.0, 20.0]).requires_grad_()
    >>> slodl.sum(a * b).backward()
    >>> a.grad
    Tensor([10, 20])
    >>> b.grad
    Tensor([2, 3])
    """
    return Tensor._from_impl(_core.sum(a._impl))
