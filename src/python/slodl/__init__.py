"""slodl: a small deep-learning package backed by a compiled C++ core."""

from slodl.tensor import Tensor

from importlib.metadata import PackageNotFoundError, version


__all__ = [
    "Tensor",
    ]

try:
    __version__ = version("slodl")
except PackageNotFoundError:
    __version__ = "0+unknown"
