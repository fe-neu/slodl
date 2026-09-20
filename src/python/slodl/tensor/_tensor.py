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
    dims : sequence of int or ndarray
        Size of each dimension. An empty sequence creates a 0-dimensional
        tensor holding a single value. A NumPy array is instead taken as the
        tensor's contents, equivalent to :meth:`Tensor.from_numpy`; a list is
        always read as dimensions.
    data : float or sequence of float, optional
        A float fills every element with that value. A sequence is taken as
        the elements themselves, in row-major order, and must contain exactly
        ``prod(dims)`` of them. If omitted, the tensor is filled with zeros.

    Raises
    ------
    IndexError
        If ``data`` is a sequence whose length does not match ``dims``. (The
        core raises ``std::out_of_range`` here, which pybind11 maps to
        ``IndexError``; ``ValueError`` would fit better.)

    See Also
    --------
    Tensor.item : Read the value out of a 0-dimensional tensor.

    Examples
    --------
    >>> from slodl import Tensor
    >>> t = Tensor([2, 2], [1, 2, 3, 4])
    >>> t
    Tensor(shape=[2, 2])
    >>> t.shape
    [2, 2]
    >>> t[1][0]
    3.0

    Views share storage with the tensor they came from:

    >>> row = t[0]
    >>> row[1] = 50.0
    >>> t[0][1]
    50.0

    A NumPy array can be used directly, and shares no memory with the tensor:

    >>> import numpy as np
    >>> Tensor(np.eye(2)).shape
    [2, 2]
    """

    def __init__(
        self,
        dims: Sequence[int] | np.ndarray,
        data: float | Sequence[float] | None = None,
    ) -> None:
        # Composition, not inheritance: _impl is the compiled tensor.
        if isinstance(dims, np.ndarray):
            if data is not None:
                raise TypeError(
                    "data cannot be given when constructing from an array")
            self._impl = _core.Tensor.from_numpy(dims)
        elif data is None:
            self._impl = _core.Tensor(list(dims))
        elif isinstance(data, (int, float)):
            self._impl = _core.Tensor(list(dims), float(data))
        else:
            self._impl = _core.Tensor(list(dims), [float(x) for x in data])

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
        >>> t = Tensor([2, 2], [1, 2, 3, 4])
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
        >>> from slodl import Tensor
        >>> Tensor([2, 3]).shape
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
        >>> Tensor([], 5.0).item()
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
        >>> t = Tensor([2, 2], [1, 2, 3, 4])
        >>> t[0]
        Tensor(shape=[2])
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
            A float, which requires that ``self[index]`` is a single element.
            A tensor, whose values are copied into that slice; it must have
            the same shape as the slice.

        Raises
        ------
        IndexError
            If ``index`` is out of range, or a float is assigned to a slice
            that is not a single element.
        ValueError
            If ``value`` is a tensor whose shape differs from the slice.

        Notes
        -----
        Assignment always copies values into this tensor's existing storage;
        it never makes the tensor refer to ``value``'s storage.

        Examples
        --------
        >>> from slodl import Tensor
        >>> t = Tensor([2, 2], [1, 2, 3, 4])
        >>> t[0][0] = 9.0
        >>> t[0][0]
        9.0
        >>> t[1] = t[0]
        >>> [t[1][j] for j in range(2)]
        [9.0, 2.0]
        """
        if isinstance(value, Tensor):
            self._impl[index] = value._impl
        else:
            self._impl[index] = float(value)

    def __repr__(self) -> str:
        return repr(self._impl)
