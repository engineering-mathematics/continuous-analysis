"""Polynomials, transfer functions, state-space models and linear ODE solvers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._domain import Domain
from ._errors import DomainError, IntegrationError
from ._ode import LinearODE, Trajectory
from ._polynomial import Polynomial
from ._statespace import StateSpace
from ._transfer import TransferFunction

try:
    __version__ = version("engg-math-continuous-analysis")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "Domain",
    "DomainError",
    "IntegrationError",
    "LinearODE",
    "Polynomial",
    "StateSpace",
    "Trajectory",
    "TransferFunction",
    "__version__",
]
