"""Unit and property tests for Polynomial and Domain. REQ-CA-001."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from engg_math.continuous_analysis import Domain, DomainError, Polynomial

coeffs = st.lists(st.floats(-5, 5), min_size=1, max_size=5)


def test_leading_zeros_are_dropped_and_zero_polynomial_is_valid() -> None:
    assert Polynomial([0, 0, 2, 1]).degree == 1
    assert Polynomial([0, 0]).degree == 0
    assert Polynomial([0.0])(3.0) == 0.0


def test_evaluation_derivative_integral() -> None:
    p = Polynomial([1, -3, 2])  # x^2 - 3x + 2
    assert p(0.0) == 2.0
    assert p.derivative() == Polynomial([2, -3])
    assert Polynomial([5]).derivative() == Polynomial([0.0])
    assert p.antiderivative(1.0).derivative() == p


def test_roots_and_from_roots() -> None:
    p = Polynomial.from_roots([1.0, 2.0], leading=3.0)
    assert_allclose(np.sort(p.roots().real), [1.0, 2.0], atol=1e-12)
    assert p.coefficients[0] == 3.0
    assert Polynomial([4.0]).roots().size == 0
    assert Polynomial.from_roots([]) == Polynomial([1.0])


def test_arithmetic() -> None:
    a, b = Polynomial([1, 1]), Polynomial([1, -1])
    assert a * b == Polynomial([1, 0, -1])
    assert a + b == Polynomial([2, 0])
    assert a - a == Polynomial([0.0])


def test_equality_and_repr() -> None:
    assert Polynomial([1, 2]) != [1, 2]
    assert repr(Polynomial([1, 2])) == "Polynomial([1.0, 2.0])"


@pytest.mark.parametrize("bad", [[], [[1, 2]], [float("nan")], [float("inf"), 1]])
def test_invalid_coefficients_rejected(bad: list[object]) -> None:
    with pytest.raises(DomainError):
        Polynomial(bad)  # type: ignore[arg-type]


def test_coefficients_are_read_only() -> None:
    with pytest.raises(ValueError, match="read-only"):
        Polynomial([1, 2]).coefficients[0] = 9.0


@given(coeffs, coeffs, st.floats(-3, 3))
def test_product_evaluates_as_product(a: list[float], b: list[float], x: float) -> None:
    pa, pb = Polynomial(a), Polynomial(b)
    assert_allclose((pa * pb)(x), pa(x) * pb(x), rtol=1e-9, atol=1e-9)


@given(coeffs, coeffs)
def test_addition_commutes(a: list[float], b: list[float]) -> None:
    assert Polynomial(a) + Polynomial(b) == Polynomial(b) + Polynomial(a)


@given(st.lists(st.floats(-3, 3), min_size=1, max_size=4))
def test_from_roots_vanishes_at_roots(roots: list[float]) -> None:
    p = Polynomial.from_roots(roots)
    assert_allclose(p(np.array(roots)), 0.0, atol=1e-6)


def test_domain_grid_and_validation() -> None:
    d = Domain(0.0, 1.0, 5)
    assert_allclose(d.grid(), [0, 0.25, 0.5, 0.75, 1.0])
    assert d.step == 0.25
    for args in [(1.0, 1.0, 5), (0.0, float("inf"), 5), (0.0, 1.0, 1)]:
        with pytest.raises(DomainError):
            Domain(*args)
