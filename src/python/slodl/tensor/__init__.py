"""Tensor, the n-dimensional array type at the centre of slodl."""

from ._creation import full, ones, zeros
from ._ops import add
from ._tensor import Tensor

__all__ = [
    "Tensor",
    "add",
    "full",
    "ones",
    "zeros",
    ]
