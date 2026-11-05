"""State-space model ``x' = A x + B u``, ``y = C x + D u``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import signal as _scipy_signal

from engg_math.core import ShapeError

from ._transfer import TransferFunction
from ._validate import matrix

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._types import Array

_signal: Any = _scipy_signal  # scipy-stubs differ between releases


class StateSpace:
    """Continuous-time linear time-invariant model."""

    __slots__ = ("a", "b", "c", "d")

    def __init__(
        self, a: npt.ArrayLike, b: npt.ArrayLike, c: npt.ArrayLike, d: npt.ArrayLike
    ) -> None:
        """Validate shapes: ``A`` n x n, ``B`` n x m, ``C`` p x n, ``D`` p x m."""
        self.a = matrix("A", a)
        self.b = matrix("B", b)
        self.c = matrix("C", c)
        self.d = matrix("D", d)
        n = self.a.shape[0]
        if self.a.shape != (n, n):
            raise ShapeError(f"A must be square, got {self.a.shape}")
        if self.b.shape[0] != n:
            raise ShapeError(f"B must have {n} rows, got {self.b.shape}")
        if self.c.shape[1] != n:
            raise ShapeError(f"C must have {n} columns, got {self.c.shape}")
        if self.d.shape != (self.c.shape[0], self.b.shape[1]):
            raise ShapeError(f"D must be {self.c.shape[0]} x {self.b.shape[1]}, got {self.d.shape}")

    def __repr__(self) -> str:
        """Unambiguous summary."""
        return (
            f"StateSpace(states={self.n_states}, inputs={self.n_inputs}, outputs={self.n_outputs})"
        )

    @property
    def n_states(self) -> int:
        """State dimension."""
        return int(self.a.shape[0])

    @property
    def n_inputs(self) -> int:
        """Input dimension."""
        return int(self.b.shape[1])

    @property
    def n_outputs(self) -> int:
        """Output dimension."""
        return int(self.c.shape[0])

    def poles(self) -> Array:
        """Eigenvalues of ``A``."""
        poles: Array = np.linalg.eigvals(self.a)
        return poles

    def is_stable(self, margin: float = 0.0) -> bool:
        """Whether every pole has real part below ``-margin`` (asymptotic stability)."""
        return bool(np.all(self.poles().real < -margin))

    def to_transfer_function(self) -> TransferFunction:
        """Equivalent transfer function; single-input single-output models only."""
        if self.n_inputs != 1 or self.n_outputs != 1:
            raise ShapeError(
                "transfer function needs 1 input and 1 output, "
                f"got {self.n_inputs} and {self.n_outputs}"
            )
        num, den = _signal.ss2tf(self.a, self.b, self.c, self.d)
        return TransferFunction(np.asarray(num)[0], np.asarray(den))
