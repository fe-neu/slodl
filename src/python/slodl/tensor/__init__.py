"""Tensor, the n-dimensional array type at the centre of slodl."""

from ._creation import full, ones, zeros
from ._ops import add, div, mean, mul, neg, sub, sum
from ._tensor import Tensor

__all__ = [
    "Tensor",
    "add",
    "div",
    "mean",
    "mul",
    "neg",
    "full",
    "ones",
    "sub",
    "sum",
    "zeros",
    ]
