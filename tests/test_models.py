"""Unit and property tests for TransferFunction and StateSpace. REQ-CA-002."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from engg_math.continuous_analysis import DomainError, StateSpace, TransferFunction
from engg_math.core import ShapeError


def sorted_poles(poles: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return poles[np.lexsort((poles.imag, poles.real))]


def test_first_order_lag() -> None:
    g = TransferFunction([2.0], [1.0, 4.0])  # 2 / (s + 4)
    assert_allclose(g.poles(), [-4.0])
    assert g.zeros().size == 0
    assert g.dc_gain() == 0.5
    assert g.is_stable()
    assert g.time_constant() == 0.25
    assert_allclose(g(0.0), 0.5)
    assert g.order == 1
    assert g.is_proper


def test_normalisation_makes_denominator_monic() -> None:
    g = TransferFunction([2.0], [2.0, 8.0])
    assert_allclose(g.den, [1.0, 4.0])
    assert_allclose(g.num, [1.0])
    assert "TransferFunction" in repr(g)


def test_stability_margin_and_unstable() -> None:
    assert not TransferFunction([1.0], [1.0, -1.0]).is_stable()
    assert TransferFunction([1.0], [1.0, 1.0]).is_stable()
    assert not TransferFunction([1.0], [1.0, 1.0]).is_stable(margin=2.0)
    with pytest.raises(DomainError, match="stable"):
        TransferFunction([1.0], [1.0, -1.0]).time_constant()


def test_pole_at_origin() -> None:
    integrator = TransferFunction([1.0], [1.0, 0.0])
    with pytest.raises(DomainError, match="origin"):
        integrator.dc_gain()
    assert np.isinf(integrator(0.0))


def test_zero_denominator_rejected() -> None:
    with pytest.raises(DomainError):
        TransferFunction([1.0], [0.0])


def test_interconnections() -> None:
    g = TransferFunction([1.0], [1.0, 1.0])
    assert_allclose(g.series(g).den, [1.0, 2.0, 1.0])
    assert_allclose(g.parallel(g)(0.0), 2.0)
    # Unity negative feedback of 1/(s+1) is 1/(s+2).
    assert_allclose(g.feedback().poles(), [-2.0])
    assert_allclose(g.feedback(positive=True).poles(), [0.0], atol=1e-12)
    # Feedback through a gain H.
    assert_allclose(g.feedback(TransferFunction([3.0], [1.0])).poles(), [-4.0])


def test_state_space_round_trip_preserves_poles_and_dc_gain() -> None:
    g = TransferFunction([1.0, 3.0], [1.0, 3.0, 2.0])  # (s+3) / ((s+1)(s+2))
    ss = g.to_state_space()
    assert ss.n_states == 2
    back = ss.to_transfer_function()
    assert_allclose(sorted_poles(back.poles()), [-2.0, -1.0], atol=1e-9)
    assert_allclose(back.dc_gain(), g.dc_gain(), rtol=1e-9)
    assert_allclose(sorted_poles(ss.poles()), [-2.0, -1.0], atol=1e-9)


def test_static_gain_realisation() -> None:
    ss = TransferFunction([5.0], [1.0]).to_state_space()
    assert ss.n_states == 0
    assert_allclose(ss.to_transfer_function().dc_gain(), 5.0)


def test_improper_has_no_realisation() -> None:
    g = TransferFunction([1.0, 0.0], [1.0])
    assert not g.is_proper
    with pytest.raises(DomainError, match="proper"):
        g.to_state_space()


def test_state_space_shapes_and_stability() -> None:
    ss = StateSpace([[0, 1], [-2, -3]], [[0], [1]], [[1, 0]], [[0]])
    assert (ss.n_states, ss.n_inputs, ss.n_outputs) == (2, 1, 1)
    assert ss.is_stable()
    assert not StateSpace([[1.0]], [[1.0]], [[1.0]], [[0.0]]).is_stable()
    assert repr(ss) == "StateSpace(states=2, inputs=1, outputs=1)"


@pytest.mark.parametrize(
    ("a", "b", "c", "d"),
    [
        ([[1, 2]], [[1]], [[1]], [[0]]),
        ([[1]], [[1], [2]], [[1]], [[0]]),
        ([[1]], [[1]], [[1, 2]], [[0]]),
        ([[1]], [[1]], [[1]], [[0, 0]]),
        ([1.0], [[1]], [[1]], [[0]]),
    ],
)
def test_state_space_shape_errors(a: object, b: object, c: object, d: object) -> None:
    with pytest.raises(ShapeError):
        StateSpace(a, b, c, d)  # type: ignore[arg-type]


def test_non_finite_matrix_rejected_and_mimo_has_no_transfer_function() -> None:
    with pytest.raises(DomainError):
        StateSpace([[float("nan")]], [[1]], [[1]], [[0]])
    mimo = StateSpace(np.eye(2) * -1, np.eye(2), np.eye(2), np.zeros((2, 2)))
    with pytest.raises(ShapeError):
        mimo.to_transfer_function()


@given(st.lists(st.floats(0.2, 5.0), min_size=1, max_size=4, unique=True))
def test_real_negative_poles_round_trip(magnitudes: list[float]) -> None:
    poles = -np.array(sorted(magnitudes))
    if len(poles) > 1 and np.min(np.diff(poles)) < 0.05:
        return  # near-repeated poles are ill-conditioned; not the property under test
    den = np.poly(poles)
    g = TransferFunction([1.0], den)
    assert g.is_stable()
    assert_allclose(sorted_poles(g.to_state_space().poles()), np.sort(poles), atol=1e-6)
