"""slodl: a small deep-learning package backed by a compiled C++ core."""

from importlib.metadata import PackageNotFoundError, version

from slodl._core import Tensor

__all__ = [
    "Tensor",
    ]

try:
    __version__ = version("slodl")
except PackageNotFoundError:
    __version__ = "0+unknown"
