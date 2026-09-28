"""Tensor, the n-dimensional array type at the centre of slodl."""

from ._creation import full, ones, zeros
from ._ops import add, mul, neg, sub, sum
from ._tensor import Tensor

__all__ = [
    "Tensor",
    "add",
    "mul",
    "neg",
    "full",
    "ones",
    "sub",
    "sum",
    "zeros",
    ]
