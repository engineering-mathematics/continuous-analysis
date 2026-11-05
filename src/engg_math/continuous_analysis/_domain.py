"""Uniform sampling grid on an interval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._errors import DomainError

if TYPE_CHECKING:
    from ._types import FloatArray

MIN_POINTS = 2


@dataclass(frozen=True, slots=True)
class Domain:
    """Closed interval ``[start, stop]`` sampled at ``num_points`` uniform points."""

    start: float
    stop: float
    num_points: int

    def __post_init__(self) -> None:
        """Validate the interval and point count."""
        if not (math.isfinite(self.start) and math.isfinite(self.stop)) or self.stop <= self.start:
            raise DomainError(f"need finite start < stop, got [{self.start}, {self.stop}]")
        if self.num_points < MIN_POINTS:
            raise DomainError(f"num_points must be at least {MIN_POINTS}, got {self.num_points}")

    @property
    def step(self) -> float:
        """Spacing between adjacent points."""
        return (self.stop - self.start) / (self.num_points - 1)

    def grid(self) -> FloatArray:
        """Return the sample points, endpoints included."""
        grid: FloatArray = np.linspace(self.start, self.stop, self.num_points)
        return grid
