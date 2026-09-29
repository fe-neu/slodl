"""Automatic differentiation: recording operations and computing gradients."""

from ._grad_mode import is_grad_enabled, no_grad

__all__ = [
    "is_grad_enabled",
    "no_grad",
    ]
