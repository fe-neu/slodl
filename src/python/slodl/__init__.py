"""slodl: a small deep-learning package backed by a compiled C++ core."""

from slodl.tensor import Tensor, full, ones, zeros

from importlib.metadata import PackageNotFoundError, version


__all__ = [
    "Tensor",
    "full",
    "ones",
    "zeros",
    ]

try:
    __version__ = version("slodl")
except PackageNotFoundError:
    __version__ = "0+unknown"
