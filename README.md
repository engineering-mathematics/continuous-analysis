# engg-math-continuous-analysis

Polynomials, transfer functions, state-space models and linear ODE solvers for `engg_math`.
Validated inputs, read-only coefficient and matrix storage, typed errors, no printing.

```bash
pip install engg-math-continuous-analysis
```

```python
from engg_math.continuous_analysis import TransferFunction, LinearODE, Domain

g = TransferFunction([2.0], [1.0, 4.0])  # 2 / (s + 4)
g.poles(), g.dc_gain(), g.is_stable()  # [-4.], 0.5, True
ss = g.to_state_space()  # controllable canonical form

ode = LinearODE([[-1.0]], [[1.0]])
ode.solve([0.0], Domain(0.0, 5.0, 101), u=[1.0]).x  # step response of x' = -x + u
```

## API

| Area | Names |
|---|---|
| Sampling | `Domain` (`grid`, `step`) |
| Polynomials | `Polynomial` (evaluate, derivative, antiderivative, roots, `+ - *`, `from_roots`) |
| Models | `TransferFunction` (poles, zeros, dc gain, stability, `series`, `parallel`, `feedback`, `to_state_space`), `StateSpace` (poles, stability, `to_transfer_function`) |
| ODEs | `LinearODE` (`state_transition`, `solve`), `Trajectory` |
| Errors | `DomainError`, `IntegrationError` (subclass `EnggMathError`) |

## Conventions

- Transfer-function coefficients are in descending powers of `s`; the denominator is normalised to be monic.
  Common factors are not cancelled.
- Stability means every pole has real part strictly below `-margin` (default 0).
- `LinearODE.solve` without input uses the matrix exponential (exact up to rounding on the uniform grid).
  With input it uses adaptive DOP853 (`rtol=1e-9`, `atol=1e-12`).
- State-space matrices are stored read-only in Fortran order.

## Scope of 0.1

The legacy fixed-step Euler solver, symbolic-variable classes, LaTeX and Rich rendering are not included.
Time and frequency responses and controllability/observability live in `engg-math-dynamical-systems`.

## Development

```bash
uv sync
uv run pytest && uv run ruff check && uv run mypy && uv run lint-imports
```

Licensed under Apache-2.0.
