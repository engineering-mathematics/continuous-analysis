"""Linear ODE ``x' = A x + B u(t)`` with exact and adaptive solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import integrate as _scipy_integrate
from scipy import linalg as _scipy_linalg

from engg_math.core import ShapeError

from ._errors import DomainError, IntegrationError
from ._validate import matrix

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from ._domain import Domain
    from ._types import Array, FloatArray

_integrate: Any = _scipy_integrate  # scipy-stubs differ between releases
_linalg: Any = _scipy_linalg

# DOP853 tolerances for forced responses; tight enough that the integrator error is far below
# typical modelling error, loose enough to avoid step-size collapse near stiff modes.
RTOL = 1e-9
ATOL = 1e-12


@dataclass(frozen=True, slots=True, eq=False)
class Trajectory:
    """States sampled at times ``t``: ``x[k]`` is the state at ``t[k]``."""

    t: FloatArray
    x: FloatArray


class LinearODE:
    """Linear time-invariant system ``x' = A x + B u``."""

    __slots__ = ("a", "b")

    def __init__(self, a: npt.ArrayLike, b: npt.ArrayLike | None = None) -> None:
        """Build from ``A`` (n x n) and optional ``B`` (n x m) for autonomous systems."""
        self.a = matrix("A", a)
        n = self.a.shape[0]
        if self.a.shape != (n, n):
            raise ShapeError(f"A must be square, got {self.a.shape}")
        self.b: Array | None = None if b is None else matrix("B", b)
        if self.b is not None and self.b.shape[0] != n:
            raise ShapeError(f"B must have {n} rows, got {self.b.shape}")

    def _initial(self, x0: npt.ArrayLike) -> FloatArray:
        x = np.asarray(x0, dtype=float)
        if x.shape != (self.a.shape[0],):
            raise ShapeError(f"x0 must have shape ({self.a.shape[0]},), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise DomainError("x0 must be finite")
        return x

    def state_transition(self, dt: float) -> Array:
        """``expm(A dt)``: maps the state at ``t`` to the state at ``t + dt`` when ``u = 0``."""
        phi: Array = _linalg.expm(self.a * dt)
        return phi

    def solve(
        self,
        x0: npt.ArrayLike,
        domain: Domain,
        u: Callable[[float], npt.ArrayLike] | npt.ArrayLike | None = None,
    ) -> Trajectory:
        """Integrate from ``domain.start`` and sample on ``domain.grid()``.

        Without input the result is exact up to rounding (matrix exponential). With input ``u``
        (a constant vector or a function of time) it uses adaptive DOP853.
        """
        start = self._initial(x0)
        t = domain.grid()
        if u is None:
            phi = self.state_transition(domain.step)
            x = np.empty((t.size, start.size))
            x[0] = start
            for k in range(1, t.size):
                x[k] = phi @ x[k - 1]
            return Trajectory(t, x)
        return self._forced(start, t, u)

    def _forced(
        self, x0: FloatArray, t: FloatArray, u: Callable[[float], npt.ArrayLike] | npt.ArrayLike
    ) -> Trajectory:
        if self.b is None:
            raise DomainError("an input was given but the system has no B matrix")
        b = self.b
        n_inputs = b.shape[1]

        def input_at(time: float) -> Array:
            value = np.atleast_1d(np.asarray(u(time) if callable(u) else u, dtype=float))
            if value.shape != (n_inputs,):
                raise ShapeError(f"input must have shape ({n_inputs},), got {value.shape}")
            return value

        def rhs(time: float, state: Array) -> Array:
            derivative: Array = self.a @ state + b @ input_at(time)
            return derivative

        result = _integrate.solve_ivp(
            rhs, (t[0], t[-1]), x0, method="DOP853", t_eval=t, rtol=RTOL, atol=ATOL
        )
        if not result.success:
            raise IntegrationError(f"integration failed at t={result.t[-1]}: {result.message}")
        return Trajectory(t, np.ascontiguousarray(result.y.T))
