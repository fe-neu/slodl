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


def sub(a: Tensor, b: Tensor) -> Tensor:
    """Subtract one tensor from another, element by element.

    Parameters
    ----------
    a : Tensor
        Tensor to subtract from.
    b : Tensor
        Tensor to subtract, of the same shape as ``a``. Shapes are not
        broadcast against each other yet.

    Returns
    -------
    Tensor
        A new tensor holding the differences. It requires a gradient, and
        records the subtraction, if either input requires one.

    Raises
    ------
    ValueError
        If the two shapes differ.

    See Also
    --------
    slodl.Tensor.__sub__ : The same operation, spelled ``a - b``.

    Examples
    --------
    >>> from slodl import Tensor, sub
    >>> sub(Tensor([10, 3]), Tensor([4, 8]))
    Tensor([6, -5])

    Raising the right operand lowers the difference, so its gradient is
    negated:

    >>> a = Tensor(5.0).requires_grad_()
    >>> b = Tensor(3.0).requires_grad_()
    >>> (a - b).backward()
    >>> a.grad, b.grad
    (Tensor(1), Tensor(-1))
    """
    return Tensor._from_impl(_core.sub(a._impl, b._impl))


def neg(a: Tensor) -> Tensor:
    """Flip the sign of every element of a tensor.

    Parameters
    ----------
    a : Tensor
        Tensor to negate.

    Returns
    -------
    Tensor
        A new tensor holding the negated elements. It requires a gradient,
        and records the negation, if ``a`` does.

    See Also
    --------
    slodl.Tensor.__neg__ : The same operation, spelled ``-a``.

    Examples
    --------
    >>> from slodl import Tensor, neg
    >>> neg(Tensor([1, -2, 3]))
    Tensor([-1, 2, -3])

    The gradient comes back negated too:

    >>> a = Tensor(3.0).requires_grad_()
    >>> (-a).backward()
    >>> a.grad
    Tensor(-1)
    """
    return Tensor._from_impl(_core.neg(a._impl))


def div(a: Tensor, b: Tensor) -> Tensor:
    """Divide one tensor by another, element by element.

    Parameters
    ----------
    a : Tensor
        The dividend.
    b : Tensor
        The divisor, of the same shape as ``a``. Shapes are not broadcast
        against each other yet.

    Returns
    -------
    Tensor
        A new tensor holding the quotients. It requires a gradient, and
        records the division, if either input requires one.

    Raises
    ------
    ValueError
        If the two shapes differ.

    See Also
    --------
    slodl.Tensor.__truediv__ : The same operation, spelled ``a / b``.

    Notes
    -----
    Dividing by zero follows IEEE 754 and gives an infinity, or a NaN for
    ``0 / 0``, rather than raising.

    Examples
    --------
    >>> from slodl import Tensor, div
    >>> div(Tensor([6, 9]), Tensor([2, 3]))
    Tensor([3, 3])

    The divisor's gradient is negative, because raising it lowers the
    result:

    >>> a = Tensor(6.0).requires_grad_()
    >>> b = Tensor(2.0).requires_grad_()
    >>> (a / b).backward()
    >>> a.grad, b.grad
    (Tensor(0.5), Tensor(-1.5))
    """
    return Tensor._from_impl(_core.div(a._impl, b._impl))


def mean(a: Tensor) -> Tensor:
    """Average every element of a tensor.

    Parameters
    ----------
    a : Tensor
        Tensor to average, which may be a view.

    Returns
    -------
    Tensor
        A 0-dimensional tensor holding the average, or NaN for an empty
        tensor, since that divides zero by zero. It requires a gradient, and
        records the operation, if ``a`` does.

    See Also
    --------
    slodl.Tensor.mean : The same operation, spelled ``a.mean()``.
    slodl.sum : The total rather than the average.

    Notes
    -----
    A backward pass gives every element a gradient of ``1 / n``, since each
    one contributes that much to the average.

    Examples
    --------
    >>> from slodl import Tensor, mean
    >>> mean(Tensor([[1, 2], [3, 4]]))
    Tensor(2.5)

    Averaging a squared error gives a loss to differentiate:

    >>> prediction = Tensor([3.0, 5.0]).requires_grad_()
    >>> error = prediction - Tensor([1.0, 1.0])
    >>> (error * error).mean().backward()
    >>> prediction.grad
    Tensor([2, 4])
    """
    return Tensor._from_impl(_core.mean(a._impl))
