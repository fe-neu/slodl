"""Tensor, the n-dimensional array type at the centre of slodl."""

from ._creation import full, ones, zeros
from ._tensor import Tensor

__all__ = [
    "Tensor",
    "full",
    "ones",
    "zeros",
    ]
