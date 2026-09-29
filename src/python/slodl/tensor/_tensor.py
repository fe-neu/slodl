from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from slodl import _core


class Tensor:
    """A dense, n-dimensional array of ``float64`` values.

    A tensor owns a flat block of memory plus the shape and strides that
    describe how to read it. Indexing a tensor does not copy that memory: it
    returns a *view* that points into the same storage, so writing through a
    view is visible from the tensor it came from. Indexing every dimension
    yields a single element, which is returned as a Python ``float``.

    Parameters
    ----------
    data : array_like
        The tensor's contents. Nested sequences give a tensor of the
        corresponding shape, a NumPy array gives a tensor of that array's
        shape, and a bare number gives a 0-dimensional tensor. The elements
        are copied and converted to ``float64``.

    Raises
    ------
    ValueError
        If ``data`` is ragged, such as ``[[1, 2], [3]]``, or holds something
        that cannot be read as a number.

    See Also
    --------
    slodl.zeros, slodl.ones, slodl.full : Create a tensor from a shape
        instead of from data.
    Tensor.item : Read the value out of a 0-dimensional tensor.

    Notes
    -----
    Like ``numpy.array`` and ``torch.tensor``, this takes the *values* of the
    tensor, not its dimensions. ``Tensor([2, 3])`` is a 1-dimensional tensor
    holding 2.0 and 3.0; for a 2x3 tensor of zeros use ``zeros([2, 3])``.

    Examples
    --------
    >>> from slodl import Tensor
    >>> t = Tensor([[1, 2], [3, 4]])
    >>> t
    Tensor([[1, 2],
            [3, 4]])
    >>> t.shape
    [2, 2]
    >>> t[1][0]
    3.0

    Views share storage with the tensor they came from:

    >>> row = t[0]
    >>> row[1] = 50.0
    >>> t[0][1]
    50.0

    A NumPy array or a bare number works too:

    >>> import numpy as np
    >>> Tensor(np.eye(2)).shape
    [2, 2]
    >>> Tensor(5.0).item()
    5.0
    """

    def __init__(self, data: npt.ArrayLike | Tensor) -> None:
        if isinstance(data, Tensor):
            self._impl = data._impl.clone()
        else:
            self._impl = _core.Tensor.from_numpy(np.asarray(data, dtype=np.float64))

    @classmethod
    def from_numpy(cls, array: npt.ArrayLike) -> Tensor:
        """Create a tensor holding a copy of ``array``'s elements.

        Parameters
        ----------
        array : array_like
            Any array NumPy can turn into ``float64``. Arrays that are not
            contiguous or not row-major, such as a transpose or a stepped
            slice, are converted; the tensor always ends up row-major.

        Returns
        -------
        Tensor
            A tensor with the array's shape, owning its own copy of the data.

        See Also
        --------
        Tensor.__array__ : The reverse direction, which shares memory.

        Notes
        -----
        The elements are copied, so later changes to ``array`` are not
        reflected in the tensor.

        Examples
        --------
        >>> import numpy as np
        >>> from slodl import Tensor
        >>> t = Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
        >>> t.shape
        [2, 2]
        >>> t[1][0]
        3.0
        """
        return cls._from_impl(_core.Tensor.from_numpy(np.asarray(array)))

    def __array__(self, dtype: npt.DTypeLike = None, copy: bool | None = None):
        """Expose the tensor to NumPy, sharing memory where possible.

        Called by ``np.asarray(t)`` and anything built on it. The returned
        array is a view of this tensor's storage, so writing to it writes
        through to the tensor, unless ``dtype`` or ``copy`` force a copy.

        Parameters
        ----------
        dtype : data-type, optional
            Requested dtype. Anything other than ``float64`` forces a copy.
        copy : bool, optional
            If True, always copy. If False, never copy and raise instead.

        Returns
        -------
        ndarray
            An array of shape :attr:`shape`, sharing storage when it can.

        Examples
        --------
        >>> import numpy as np
        >>> from slodl import Tensor
        >>> t = Tensor([[1, 2], [3, 4]])
        >>> np.asarray(t)
        array([[1., 2.],
               [3., 4.]])

        The array shares the tensor's memory:

        >>> np.asarray(t)[0, 0] = 99.0
        >>> t[0][0]
        99.0
        """
        array = np.asarray(memoryview(self._impl))
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        if copy is True:
            array = array.copy()
        elif copy is False and array.base is None:
            raise ValueError("cannot avoid a copy for this conversion")
        return array

    def clone(self) -> Tensor:
        """Return a copy with its own storage.

        The copy holds the same values but shares nothing with this tensor,
        so writing to either leaves the other unchanged. Cloning a view
        produces a tensor of the view's shape, laid out contiguously.

        Returns
        -------
        Tensor
            An independent tensor with the same shape and values.

        Examples
        --------
        >>> from slodl import Tensor
        >>> t = Tensor([[1, 2], [3, 4]])
        >>> copy = t.clone()
        >>> copy[0][0] = 99.0
        >>> t[0][0]
        1.0
        """
        return Tensor._from_impl(self._impl.clone())

    @classmethod
    def _from_impl(cls, impl: _core.Tensor) -> Tensor:
        """Wrap a compiled tensor without constructing new storage."""
        wrapper = cls.__new__(cls)
        wrapper._impl = impl
        return wrapper

    @property
    def shape(self) -> list[int]:
        """list of int : Size of each dimension, outermost first.

        Examples
        --------
        >>> from slodl import zeros
        >>> zeros([2, 3]).shape
        [2, 3]
        """
        return self._impl.shape

    def item(self) -> float:
        """Return the value of a 0-dimensional tensor as a ``float``.

        Indexing already returns a ``float`` once every dimension has been
        indexed, so this is only needed for a tensor that was created with no
        dimensions.

        Returns
        -------
        float
            The single value held by the tensor.

        Raises
        ------
        IndexError
            If the tensor has one or more dimensions.

        Examples
        --------
        >>> from slodl import Tensor
        >>> Tensor(5.0).item()
        5.0
        """
        return self._impl.item()

    def __len__(self) -> int:
        """Size of the outermost dimension.

        Raises
        ------
        TypeError
            If the tensor is 0-dimensional, and so has no dimension to size.
        """
        return len(self._impl)

    def __getitem__(self, index: int) -> Tensor | float:
        """Index the outermost dimension.

        Parameters
        ----------
        index : int
            Position along the outermost dimension. Negative values count
            back from the end.

        Returns
        -------
        Tensor or float
            A view sharing this tensor's storage, or a ``float`` if the
            result is a single element.

        Raises
        ------
        IndexError
            If ``index`` is out of range, or the tensor is 0-dimensional.

        Examples
        --------
        >>> from slodl import Tensor
        >>> t = Tensor([[1, 2], [3, 4]])
        >>> t[0]
        Tensor([1, 2])
        >>> t[0][1]
        2.0
        >>> t[-1][-1]
        4.0
        """
        result = self._impl[index]
        if isinstance(result, _core.Tensor):
            return Tensor._from_impl(result)
        return result

    def __setitem__(self, index: int, value: float | Tensor) -> None:
        """Write into the outermost dimension, in place.

        Parameters
        ----------
        index : int
            Position along the outermost dimension. Negative values count
            back from the end.
        value : float or Tensor
            A float, which is written to every element of the slice. A
            tensor, whose values are copied into the slice; it must have the
            same shape as the slice.

        Raises
        ------
        IndexError
            If ``index`` is out of range.
        ValueError
            If ``value`` is a tensor whose shape differs from the slice.

        Notes
        -----
        Assignment always copies values into this tensor's existing storage;
        it never makes the tensor refer to ``value``'s storage.

        Examples
        --------
        >>> from slodl import Tensor
        >>> t = Tensor([[1, 2], [3, 4]])
        >>> t[0][0] = 9.0
        >>> t[0][0]
        9.0
        >>> t[1] = t[0]
        >>> [t[1][j] for j in range(2)]
        [9.0, 2.0]

        A float fills the whole slice:

        >>> t[0] = 7.0
        >>> [t[0][j] for j in range(2)]
        [7.0, 7.0]
        """
        if isinstance(value, Tensor):
            self._impl[index] = value._impl
        else:
            self._impl[index] = float(value)

    def __add__(self, other: Tensor) -> Tensor:
        """Add two tensors element by element.

        Parameters
        ----------
        other : Tensor or float
            A tensor whose shape broadcasts against this one's, or a number,
            which is added to every element.

        Returns
        -------
        Tensor
            A new tensor holding the sums, recording the addition if either
            operand requires a gradient.

        Raises
        ------
        ValueError
            If the shapes cannot be broadcast together.

        Examples
        --------
        >>> from slodl import Tensor
        >>> Tensor([1, 2]) + Tensor([10, 20])
        Tensor([11, 22])

        A row is added to every row of a matrix, and a number to every
        element:

        >>> Tensor([[1, 2], [3, 4]]) + Tensor([10, 20])
        Tensor([[11, 22],
                [13, 24]])
        >>> 100 + Tensor([1, 2])
        Tensor([101, 102])
        """
        if isinstance(other, Tensor):
            return Tensor._from_impl(self._impl + other._impl)
        if isinstance(other, (int, float)):
            return Tensor._from_impl(self._impl + float(other))
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other: Tensor) -> Tensor:
        """Multiply two tensors element by element.

        This is the element-wise (Hadamard) product, not a matrix product.

        Parameters
        ----------
        other : Tensor or float
            A tensor whose shape broadcasts against this one's, or a number,
            which scales every element.

        Returns
        -------
        Tensor
            A new tensor holding the products, recording the multiplication
            if either operand requires a gradient.

        Raises
        ------
        ValueError
            If the shapes cannot be broadcast together.

        Examples
        --------
        >>> from slodl import Tensor
        >>> Tensor([2, 3]) * Tensor([10, 20])
        Tensor([20, 60])
        >>> Tensor([1, 2]) * 3
        Tensor([3, 6])
        """
        if isinstance(other, Tensor):
            return Tensor._from_impl(self._impl * other._impl)
        if isinstance(other, (int, float)):
            return Tensor._from_impl(self._impl * float(other))
        return NotImplemented

    __rmul__ = __mul__

    def __sub__(self, other: Tensor) -> Tensor:
        """Subtract another tensor, element by element.

        Parameters
        ----------
        other : Tensor or float
            A tensor whose shape broadcasts against this one's, or a number,
            which is subtracted from every element.

        Returns
        -------
        Tensor
            A new tensor holding the differences, recording the subtraction
            if either operand requires a gradient.

        Raises
        ------
        ValueError
            If the shapes cannot be broadcast together.

        Examples
        --------
        >>> from slodl import Tensor
        >>> Tensor([10, 3]) - Tensor([4, 8])
        Tensor([6, -5])
        >>> 10 - Tensor([1, 2])
        Tensor([9, 8])
        """
        if isinstance(other, Tensor):
            return Tensor._from_impl(self._impl - other._impl)
        if isinstance(other, (int, float)):
            return Tensor._from_impl(self._impl - float(other))
        return NotImplemented

    def __rsub__(self, other: float) -> Tensor:
        """Subtract this tensor from a number, element by element."""
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Tensor._from_impl(float(other) - self._impl)

    def __truediv__(self, other: Tensor) -> Tensor:
        """Divide by another tensor, element by element.

        Parameters
        ----------
        other : Tensor or float
            A tensor whose shape broadcasts against this one's, or a number,
            which divides every element.

        Returns
        -------
        Tensor
            A new tensor holding the quotients, recording the division if
            either operand requires a gradient.

        Raises
        ------
        ValueError
            If the shapes cannot be broadcast together.

        Notes
        -----
        Dividing by zero follows IEEE 754 and gives an infinity, or a NaN
        for ``0 / 0``, rather than raising.

        Examples
        --------
        >>> from slodl import Tensor
        >>> Tensor([6, 9]) / Tensor([2, 3])
        Tensor([3, 3])
        >>> Tensor([6, 9]) / 3
        Tensor([2, 3])
        """
        if isinstance(other, Tensor):
            return Tensor._from_impl(self._impl / other._impl)
        if isinstance(other, (int, float)):
            return Tensor._from_impl(self._impl / float(other))
        return NotImplemented

    def __rtruediv__(self, other: float) -> Tensor:
        """Divide a number by this tensor, element by element."""
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Tensor._from_impl(float(other) / self._impl)

    def __neg__(self) -> Tensor:
        """Flip the sign of every element.

        Returns
        -------
        Tensor
            A new tensor holding the negated elements, recording the
            negation if this tensor requires a gradient.

        Examples
        --------
        >>> from slodl import Tensor
        >>> -Tensor([1, -2, 3])
        Tensor([-1, 2, -3])
        """
        return Tensor._from_impl(-self._impl)

    def expand(self, shape: Sequence[int]) -> Tensor:
        """Read this tensor as though it had a larger shape.

        Dimensions of size 1 are stretched, and dimensions this tensor does
        not have are added on the left. Nothing is copied: the stretched
        positions all read the same element.

        Parameters
        ----------
        shape : sequence of int
            The shape to read this tensor as. This tensor's shape must
            broadcast to it.

        Returns
        -------
        Tensor
            A view sharing this tensor's storage.

        Raises
        ------
        ValueError
            If this tensor's shape does not broadcast to ``shape``.

        Notes
        -----
        The operators broadcast on their own, so this is rarely needed
        directly. In a backward pass the gradient is summed back down to the
        original shape, because one element was read in several places.

        Examples
        --------
        >>> from slodl import Tensor
        >>> Tensor([1, 2, 3]).expand([2, 3])
        Tensor([[1, 2, 3],
                [1, 2, 3]])
        """
        return Tensor._from_impl(self._impl.expand([int(d) for d in shape]))

    def mean(self) -> Tensor:
        """Average every element of this tensor.

        Returns
        -------
        Tensor
            A 0-dimensional tensor holding the average, or NaN for an empty
            tensor, since that divides zero by zero. It requires a gradient,
            and records the operation, if this tensor does.

        See Also
        --------
        slodl.mean : The same operation, spelled ``slodl.mean(a)``.
        Tensor.sum : The total rather than the average.

        Notes
        -----
        A backward pass gives every element a gradient of ``1 / n``, since
        each one contributes that much to the average.

        Examples
        --------
        >>> from slodl import Tensor
        >>> Tensor([[1, 2], [3, 4]]).mean()
        Tensor(2.5)

        >>> a = Tensor([1.0, 2.0, 3.0, 4.0]).requires_grad_()
        >>> a.mean().backward()
        >>> a.grad
        Tensor([0.25, 0.25, 0.25, 0.25])
        """
        return Tensor._from_impl(self._impl.mean())

    def sum(self) -> Tensor:
        """Add up every element of this tensor.

        This is how a tensor becomes the single value that :meth:`backward`
        can start from.

        Returns
        -------
        Tensor
            A 0-dimensional tensor holding the total, zero for an empty
            tensor. It requires a gradient, and records the sum, if this
            tensor does.

        See Also
        --------
        slodl.sum : The same operation, spelled ``slodl.sum(a)``.

        Notes
        -----
        Every element contributes to the total equally, so a backward pass
        gives each one the same gradient. Python's built-in ``sum`` does not
        work on a tensor; use this instead.

        Examples
        --------
        >>> from slodl import Tensor
        >>> Tensor([[1, 2], [3, 4]]).sum()
        Tensor(10)

        Each element's gradient is the gradient of the total:

        >>> a = Tensor([1.0, 2.0, 3.0]).requires_grad_()
        >>> a.sum().backward()
        >>> a.grad
        Tensor([1, 1, 1])
        """
        return Tensor._from_impl(self._impl.sum())

    @property
    def requires_grad(self) -> bool:
        """bool : Whether operations on this tensor are recorded for autograd.

        Setting it is only allowed on a leaf: a tensor produced by a recorded
        operation already inherits its answer from that operation's inputs.

        Examples
        --------
        >>> from slodl import Tensor
        >>> t = Tensor([1.0, 2.0])
        >>> t.requires_grad
        False
        >>> t.requires_grad = True
        >>> t.requires_grad
        True
        """
        return self._impl.requires_grad

    @requires_grad.setter
    def requires_grad(self, flag: bool) -> None:
        self._impl.requires_grad = bool(flag)

    def requires_grad_(self, flag: bool = True) -> Tensor:
        """Turn gradient tracking on or off, in place.

        Parameters
        ----------
        flag : bool, default True
            Whether to track gradients.

        Returns
        -------
        Tensor
            This tensor, so the call can be chained.

        Raises
        ------
        ValueError
            If this tensor is not a leaf.

        See Also
        --------
        Tensor.requires_grad : The same setting, as a property.

        Examples
        --------
        >>> from slodl import Tensor
        >>> t = Tensor([1.0, 2.0]).requires_grad_()
        >>> t.requires_grad
        True
        """
        self._impl.requires_grad_(bool(flag))
        return self

    @property
    def is_leaf(self) -> bool:
        """bool : Whether this tensor was not produced by a recorded operation.

        Tensors you create are leaves, and only leaves accumulate a
        :attr:`grad`. Results of recorded operations are not.

        Examples
        --------
        >>> from slodl import Tensor
        >>> a = Tensor([1.0, 2.0]).requires_grad_()
        >>> a.is_leaf
        True
        >>> (a + a).is_leaf
        False
        """
        return self._impl.is_leaf

    @property
    def grad(self) -> Tensor | None:
        """Tensor or None : The gradient accumulated by ``backward()``.

        None until a backward pass has accumulated one. Only leaves that
        require a gradient ever get one.

        Notes
        -----
        The returned tensor shares the gradient's storage, so writing to it
        writes through to the gradient.
        """
        gradient = self._impl.grad
        if gradient is None:
            return None
        return Tensor._from_impl(gradient)

    @property
    def grad_fn(self) -> _core.Node | None:
        """Node or None : The operation that produced this tensor.

        None for a leaf. Otherwise the node that a backward pass would call
        to push gradients back to this operation's inputs; it prints as its
        own name, such as ``<AddBackward>``.

        Examples
        --------
        >>> from slodl import Tensor
        >>> a = Tensor([1.0, 2.0]).requires_grad_()
        >>> (a + a).grad_fn
        <AddBackward>
        >>> a.grad_fn is None
        True
        """
        return self._impl.grad_fn

    def detach(self) -> Tensor:
        """Return this tensor's data without its autograd history.

        Returns
        -------
        Tensor
            A tensor sharing this one's storage that requires no gradient and
            records no history, so gradients do not flow through it. Writes
            through either tensor are visible in the other.

        See Also
        --------
        Tensor.clone : An independent copy, which does not share storage.
        slodl.no_grad : Stop recording for a whole block instead.

        Examples
        --------
        >>> from slodl import Tensor
        >>> a = Tensor([1.0, 2.0]).requires_grad_()
        >>> b = a.detach()
        >>> b.requires_grad
        False
        >>> b[0] = 9.0
        >>> a[0]
        9.0
        """
        return Tensor._from_impl(self._impl.detach())

    def backward(self) -> None:
        """Compute gradients back through the graph that produced this tensor.

        Starts from a gradient of 1 for this tensor and works backwards,
        adding into the :attr:`grad` of every leaf that requires one.

        Raises
        ------
        ValueError
            If this tensor has any dimensions, since a starting gradient is
            only obvious for a scalar, or if it does not require a gradient.

        See Also
        --------
        Tensor.grad : Where the results end up.

        Notes
        -----
        Gradients accumulate, so calling this twice without clearing
        :attr:`grad` in between adds to what is already there.

        Examples
        --------
        >>> from slodl import Tensor
        >>> a = Tensor(2.0).requires_grad_()
        >>> b = Tensor(3.0).requires_grad_()
        >>> (a + b).backward()
        >>> a.grad
        Tensor(1)

        A tensor used twice collects a gradient from each use:

        >>> c = Tensor(1.0).requires_grad_()
        >>> (c + c).backward()
        >>> c.grad
        Tensor(2)
        """
        self._impl.backward()

    def __repr__(self) -> str:
        return repr(self._impl)
