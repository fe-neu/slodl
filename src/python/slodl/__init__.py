"""slodl: a small deep-learning package backed by a compiled C++ core."""

from slodl.autograd import is_grad_enabled, no_grad
from slodl.tensor import (
    Tensor,
    add,
    div,
    full,
    mean,
    mul,
    neg,
    ones,
    sub,
    sum,
    transpose,
    zeros,
)

from importlib.metadata import PackageNotFoundError, version


__all__ = [
    "Tensor",
    "add",
    "div",
    "full",
    "is_grad_enabled",
    "mean",
    "mul",
    "neg",
    "no_grad",
    "ones",
    "sub",
    "sum",
    "transpose",
    "zeros",
    ]

try:
    __version__ = version("slodl")
except PackageNotFoundError:
    __version__ = "0+unknown"
