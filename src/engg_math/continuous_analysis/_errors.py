"""Exception types."""

from __future__ import annotations

from engg_math.core import EnggMathError


class DomainError(EnggMathError, ValueError):
    """An argument lies outside the domain of the operation."""


class IntegrationError(EnggMathError, RuntimeError):
    """The ODE integrator failed to reach the end of the interval."""
