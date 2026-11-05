"""Input validation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engg_math.core import ShapeError

from ._errors import DomainError

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._types import Array


def matrix(name: str, value: npt.ArrayLike) -> Array:
    m = np.asarray(value, dtype=float)
    if m.ndim != 2:  # noqa: PLR2004 - matrices are 2-D
        raise ShapeError(f"{name} must be 2-D, got shape {m.shape}")
    if not np.all(np.isfinite(m)):
        raise DomainError(f"{name} must be finite")
    out: Array = np.array(m, order="F")
    out.flags.writeable = False
    return out
