"""slodl: a small deep-learning package backed by a compiled C++ core."""

from slodl.autograd import is_grad_enabled, no_grad
from slodl.tensor import Tensor, add, div, full, mul, neg, ones, sub, sum, zeros

from importlib.metadata import PackageNotFoundError, version


__all__ = [
    "Tensor",
    "add",
    "div",
    "full",
    "is_grad_enabled",
    "mul",
    "neg",
    "no_grad",
    "ones",
    "sub",
    "sum",
    "zeros",
    ]

try:
    __version__ = version("slodl")
except PackageNotFoundError:
    __version__ = "0+unknown"
