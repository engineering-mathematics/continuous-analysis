"""Unit, property and reference tests for LinearODE. REQ-CA-003."""

from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from engg_math.continuous_analysis import Domain, DomainError, IntegrationError, LinearODE
from engg_math.core import ShapeError

GRID = Domain(0.0, 5.0, 101)


def test_exponential_decay_is_exact() -> None:
    traj = LinearODE([[-2.0]]).solve([3.0], GRID)
    assert_allclose(traj.x[:, 0], 3.0 * np.exp(-2.0 * traj.t), rtol=1e-12)


def test_harmonic_oscillator_matches_closed_form_and_conserves_energy() -> None:
    ode = LinearODE([[0.0, 1.0], [-4.0, 0.0]])  # omega = 2
    traj = ode.solve([1.0, 0.0], Domain(0.0, 20.0, 2001))
    assert_allclose(traj.x[:, 0], np.cos(2.0 * traj.t), atol=1e-10)
    energy = traj.x[:, 1] ** 2 + 4.0 * traj.x[:, 0] ** 2
    assert_allclose(energy, 4.0, rtol=1e-10)


def test_forced_step_response_matches_analytic() -> None:
    # x' = -x + u, u = 1  ->  x = 1 - exp(-t)
    traj = LinearODE([[-1.0]], [[1.0]]).solve([0.0], GRID, u=[1.0])
    assert_allclose(traj.x[:, 0], 1.0 - np.exp(-traj.t), atol=1e-7)


def test_time_varying_input() -> None:
    # x' = u(t) = 2t  ->  x = t^2
    traj = LinearODE([[0.0]], [[1.0]]).solve([0.0], GRID, u=lambda t: [2.0 * t])
    assert_allclose(traj.x[:, 0], traj.t**2, atol=1e-7)


def test_state_transition_composes() -> None:
    ode = LinearODE([[0.0, 1.0], [-1.0, -0.3]])
    assert_allclose(
        ode.state_transition(0.7) @ ode.state_transition(0.5), ode.state_transition(1.2), atol=1e-12
    )


def test_input_errors() -> None:
    with pytest.raises(DomainError, match="no B"):
        LinearODE([[-1.0]]).solve([1.0], GRID, u=[1.0])
    with pytest.raises(ShapeError, match="input"):
        LinearODE([[-1.0]], [[1.0]]).solve([1.0], GRID, u=[1.0, 2.0])


def test_construction_and_initial_state_errors() -> None:
    with pytest.raises(ShapeError):
        LinearODE([[1.0, 2.0]])
    with pytest.raises(ShapeError):
        LinearODE([[1.0]], [[1.0], [2.0]])
    with pytest.raises(ShapeError, match="x0"):
        LinearODE([[1.0]]).solve([1.0, 2.0], GRID)
    with pytest.raises(DomainError, match="finite"):
        LinearODE([[1.0]]).solve([float("nan")], GRID)


@given(st.floats(-3.0, 3.0), st.floats(-5.0, 5.0))
def test_scalar_solution_matches_exponential(rate: float, x0: float) -> None:
    traj = LinearODE([[rate]]).solve([x0], Domain(0.0, 1.0, 11))
    assert_allclose(traj.x[:, 0], x0 * np.exp(rate * traj.t), rtol=1e-9, atol=1e-12)


@given(st.floats(-2.0, 2.0), st.floats(-2.0, 2.0))
def test_linearity_in_initial_state(p: float, q: float) -> None:
    ode = LinearODE([[0.0, 1.0], [-2.0, -0.5]])
    grid = Domain(0.0, 2.0, 21)
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    combined = ode.solve(p * a + q * b, grid).x
    assert_allclose(combined, p * ode.solve(a, grid).x + q * ode.solve(b, grid).x, atol=1e-10)


def test_integrator_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = SimpleNamespace(success=False, t=np.array([0.5]), message="step size too small")
    monkeypatch.setattr(
        "engg_math.continuous_analysis._ode._integrate",
        SimpleNamespace(solve_ivp=lambda *_a, **_k: failed),
    )
    with pytest.raises(IntegrationError, match="step size too small"):
        LinearODE([[-1.0]], [[1.0]]).solve([0.0], GRID, u=[1.0])
