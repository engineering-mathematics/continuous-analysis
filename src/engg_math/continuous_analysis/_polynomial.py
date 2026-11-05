"""Real or complex polynomials in one variable."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._errors import DomainError

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._types import Array


def trim(coefficients: npt.ArrayLike) -> Array:
    """Return descending coefficients without leading zeros (zero polynomial -> ``[0]``)."""
    c = np.atleast_1d(np.asarray(coefficients))
    if c.ndim != 1 or c.size == 0:
        raise DomainError(f"coefficients must be a non-empty 1-D sequence, got shape {c.shape}")
    if not np.all(np.isfinite(c)):
        raise DomainError("coefficients must be finite")
    c = c.astype(complex) if np.iscomplexobj(c) else c.astype(float)
    nonzero = np.flatnonzero(c)
    return c[nonzero[0] :].copy() if nonzero.size else c[:1] * 0.0


class Polynomial:
    """Polynomial stored as descending coefficients: ``[a, b, c]`` is ``a x^2 + b x + c``."""

    __slots__ = ("_c",)

    def __init__(self, coefficients: npt.ArrayLike) -> None:
        """Build from descending coefficients; leading zeros are dropped."""
        self._c = trim(coefficients)
        self._c.flags.writeable = False

    @classmethod
    def from_roots(cls, roots: npt.ArrayLike, leading: float = 1.0) -> Polynomial:
        """Monic polynomial (scaled by ``leading``) with the given roots."""
        r = np.atleast_1d(np.asarray(roots))
        return cls(leading * np.poly(r)) if r.size else cls([leading])

    @property
    def coefficients(self) -> Array:
        """Descending coefficients (read-only view)."""
        return self._c

    @property
    def degree(self) -> int:
        """Degree; the zero polynomial has degree 0."""
        return int(self._c.size - 1)

    def __call__(self, x: npt.ArrayLike) -> Array:
        """Evaluate at scalar or array ``x`` (Horner)."""
        xs = np.asarray(x)
        result: Array = np.zeros(xs.shape, dtype=np.result_type(self._c, xs))
        for c in self._c:
            result = result * xs + c
        return result

    def __add__(self, other: Polynomial) -> Polynomial:
        """Sum."""
        return Polynomial(np.polyadd(self._c, other._c))

    def __sub__(self, other: Polynomial) -> Polynomial:
        """Difference."""
        return Polynomial(np.polysub(self._c, other._c))

    def __mul__(self, other: Polynomial) -> Polynomial:
        """Product."""
        return Polynomial(np.convolve(self._c, other._c))

    def __eq__(self, other: object) -> bool:
        """Exact coefficient equality."""
        if not isinstance(other, Polynomial):
            return NotImplemented
        return bool(np.array_equal(self._c, other._c))

    __hash__ = None  # type: ignore[assignment]  # mutable-looking value type; equality is by value

    def __repr__(self) -> str:
        """Unambiguous form."""
        return f"Polynomial({self._c.tolist()})"

    def derivative(self) -> Polynomial:
        """First derivative."""
        if self._c.size == 1:
            return Polynomial([0.0])
        return Polynomial(np.polyder(self._c))

    def antiderivative(self, constant: float = 0.0) -> Polynomial:
        """Antiderivative with the given constant term."""
        return Polynomial(np.polyint(self._c, k=constant))

    def roots(self) -> Array:
        """Roots with multiplicity; empty for constants."""
        if self._c.size == 1:
            return np.empty(0, dtype=complex)
        roots: Array = np.roots(self._c)
        return roots
