"""Search strategies: how to explore the configuration space."""

from .ddmin import PrecisionDDMin, ddmin
from .uniform import UniformBisect

__all__ = ["PrecisionDDMin", "ddmin", "UniformBisect"]
