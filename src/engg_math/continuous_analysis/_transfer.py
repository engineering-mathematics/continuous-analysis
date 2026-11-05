"""SISO continuous-time transfer function ``N(s) / D(s)``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import signal as _scipy_signal

from ._errors import DomainError
from ._polynomial import Polynomial, trim

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._statespace import StateSpace
    from ._types import Array

_signal: Any = _scipy_signal  # scipy-stubs differ between releases


class TransferFunction:
    """Rational function in ``s``: descending-power coefficients, monic denominator."""

    __slots__ = ("_den", "_num")

    def __init__(self, num: npt.ArrayLike, den: npt.ArrayLike) -> None:
        """Build from numerator and denominator coefficients (descending powers of ``s``)."""
        n, d = trim(num), trim(den)
        if d[0] == 0:
            raise DomainError("denominator must not be the zero polynomial")
        self._num = Polynomial(n / d[0])
        self._den = Polynomial(d / d[0])

    @property
    def num(self) -> Array:
        """Numerator coefficients."""
        return self._num.coefficients

    @property
    def den(self) -> Array:
        """Denominator coefficients (leading coefficient 1)."""
        return self._den.coefficients

    @property
    def order(self) -> int:
        """Denominator degree."""
        return self._den.degree

    @property
    def is_proper(self) -> bool:
        """Whether the numerator degree does not exceed the denominator degree."""
        return self._num.degree <= self._den.degree

    def __repr__(self) -> str:
        """Unambiguous form."""
        return f"TransferFunction(num={self.num.tolist()}, den={self.den.tolist()})"

    def __call__(self, s: npt.ArrayLike) -> Array:
        """Evaluate ``N(s) / D(s)``; a pole at ``s`` gives ``inf``/``nan`` per IEEE rules."""
        with np.errstate(divide="ignore", invalid="ignore"):
            value: Array = self._num(s) / self._den(s)
        return value

    def poles(self) -> Array:
        """Roots of the denominator."""
        return self._den.roots()

    def zeros(self) -> Array:
        """Roots of the numerator."""
        return self._num.roots()

    def dc_gain(self) -> float:
        """``H(0)``; undefined (raises) when there is a pole at the origin."""
        if self.den[-1] == 0:
            raise DomainError("dc gain is undefined: pole at the origin")
        return float(np.real(self.num[-1] / self.den[-1]))

    def is_stable(self, margin: float = 0.0) -> bool:
        """Whether every pole has real part below ``-margin``."""
        return bool(np.all(self.poles().real < -margin))

    def series(self, other: TransferFunction) -> TransferFunction:
        """Cascade ``self * other``."""
        other_num, other_den = Polynomial(other.num), Polynomial(other.den)
        return TransferFunction(
            (self._num * other_num).coefficients, (self._den * other_den).coefficients
        )

    def parallel(self, other: TransferFunction) -> TransferFunction:
        """Sum ``self + other``."""
        other_num, other_den = Polynomial(other.num), Polynomial(other.den)
        num = self._num * other_den + other_num * self._den
        return TransferFunction(num.coefficients, (self._den * other_den).coefficients)

    def feedback(
        self, other: TransferFunction | None = None, *, positive: bool = False
    ) -> TransferFunction:
        """Return ``G / (1 -/+ G H)`` with ``self`` as ``G`` and ``other`` as ``H`` (default 1)."""
        h = other if other is not None else TransferFunction([1.0], [1.0])
        h_num, h_den = Polynomial(h.num), Polynomial(h.den)
        loop = self._num * h_num
        den = self._den * h_den - loop if positive else self._den * h_den + loop
        return TransferFunction((self._num * h_den).coefficients, den.coefficients)

    def to_state_space(self) -> StateSpace:
        """Return a controllable canonical realisation; needs a proper transfer function."""
        from ._statespace import StateSpace  # noqa: PLC0415 - breaks the import cycle

        if not self.is_proper:
            raise DomainError("state-space realisation needs a proper transfer function")
        if self.order == 0:
            return StateSpace(np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), [[self.num[0]]])
        a, b, c, d = _signal.tf2ss(self.num, self.den)
        return StateSpace(a, b, c, d)

    def time_constant(self) -> float:
        """Dominant time constant ``-1 / max Re(p)`` of a stable system."""
        if not self.is_stable():
            raise DomainError("time constant is defined only for stable systems")
        return -1.0 / float(np.max(self.poles().real))
