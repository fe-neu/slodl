"""Turning autograd recording on and off."""

from __future__ import annotations

from contextlib import ContextDecorator
from types import TracebackType

from slodl import _core


def is_grad_enabled() -> bool:
    """Report whether operations are currently recorded for autograd.

    Returns
    -------
    bool
        True unless recording has been turned off, for instance by
        :class:`no_grad`.

    See Also
    --------
    no_grad : Turn recording off for a block of code.

    Examples
    --------
    >>> from slodl import is_grad_enabled, no_grad
    >>> is_grad_enabled()
    True
    >>> with no_grad():
    ...     is_grad_enabled()
    False
    """
    return _core.is_grad_enabled()


class no_grad(ContextDecorator):
    """Stop operations from being recorded for autograd.

    Inside the block, operations still compute their values but build no
    graph, so their results require no gradient. Use it for evaluating a
    model, and for updating parameters, where recording the update itself
    would be wrong.

    See Also
    --------
    is_grad_enabled : Report whether recording is on.
    slodl.Tensor.detach : Drop the history of a single tensor.

    Notes
    -----
    The previous setting is restored on exit, so blocks nest. Recording is
    per thread: a thread started inside the block is unaffected.

    Examples
    --------
    >>> from slodl import Tensor, no_grad
    >>> a = Tensor([1.0, 2.0]).requires_grad_()
    >>> (a + a).requires_grad
    True
    >>> with no_grad():
    ...     (a + a).requires_grad
    False

    It also works as a decorator:

    >>> @no_grad()
    ... def predict(x):
    ...     return (x + x).requires_grad
    >>> predict(a)
    False
    """

    def __enter__(self) -> None:
        self._previous = _core.is_grad_enabled()
        _core.set_grad_enabled(False)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _core.set_grad_enabled(self._previous)
