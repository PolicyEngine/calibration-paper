"""Numerics of the classical calibration solvers, against known solutions.

Every method is exercised on tiny synthetic problems whose answer is known:

* feasible surfaces -> exact target reproduction (to Newton tolerance),
* bounded methods -> g-weights inside their bounds always,
* raking / logit -> strictly positive weights always,
* GREG -> negative weights when a target demands an extreme shift (the reported
  pathology),
* infeasible surfaces -> ``converged=False`` and a documented residual,
* raking == entropy == exponential tilting (the Deville-Sarndal identity).

These pin the *methods*, not statistical quality; the R-parity suite
(``test_r_parity.py``) grounds them against R ``survey`` where it is installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from calibration_paper.classical import (
    calibrate_distance,
    chi_square_weights,
    linear_weights,
    logit_weights,
    raking_weights,
)
from calibration_paper.synthetic import (
    feasible_problem,
    infeasible_problem,
    margins_problem,
)

#: The smooth (unbounded) methods reproduce a feasible target exactly.
SMOOTH_SOLVERS = {
    "raking": lambda p: raking_weights(p.matrix, p.target, p.initial_weights),
    "greg": lambda p: linear_weights(p.matrix, p.target, p.initial_weights),
    "chi_square": lambda p: chi_square_weights(p.matrix, p.target, p.initial_weights),
}

#: Bounds used by the bounded-method tests; they contain the feasible problem's
#: true g-weight range so a feasible solve stays interior and still hits targets.
BOUNDS = (0.05, 20.0)


@pytest.mark.parametrize("name", sorted(SMOOTH_SOLVERS))
def test_smooth_methods_reproduce_feasible_targets_exactly(name: str) -> None:
    """On a feasible surface the smooth methods drive ``A@w`` to ``b`` exactly."""
    problem = feasible_problem(n_records=200, n_targets=8, seed=0).problem
    solution = SMOOTH_SOLVERS[name](problem)
    assert solution.converged, name
    achieved = problem.estimates(solution.weights)
    np.testing.assert_allclose(achieved, problem.target, rtol=1e-6, atol=1e-4)


def test_raking_weights_are_strictly_positive() -> None:
    """Raking's multiplicative g = exp(x'lambda) is positive by construction.

    Checked on a demanding surface (a big shift) where GREG would go negative --
    raking must not.
    """
    problem = _extreme_shift_problem()
    solution = raking_weights(problem.matrix, problem.target, problem.initial_weights)
    assert (solution.weights > 0.0).all()
    assert (solution.g_weights > 0.0).all()


def test_greg_produces_negative_weights_on_an_extreme_shift() -> None:
    """GREG's affine g = 1 + x'lambda goes negative when a target pulls hard.

    This is the reported GREG pathology the paper compares against the bounded
    and gradient methods; the test makes sure the surface actually triggers it
    (so the comparison is not vacuous).
    """
    problem = _extreme_shift_problem()
    solution = linear_weights(problem.matrix, problem.target, problem.initial_weights)
    assert (solution.weights < 0.0).any()


@pytest.mark.parametrize("distance", ["chi_square", "logit"])
def test_bounded_methods_respect_their_bounds_even_when_infeasible(
    distance: str,
) -> None:
    """Bounded g-weights stay in ``[L, U]`` on a surface that forces the bound.

    On the extreme-shift surface the targets are unreachable within the bounds,
    so the method cannot converge -- but it must *never* leave the bounds, which
    is the guarantee the paper's failure-mode section relies on.
    """
    lower, upper = 0.2, 5.0
    problem = _extreme_shift_problem()
    solution = calibrate_distance(
        distance,
        problem.matrix,
        problem.target,
        problem.initial_weights,
        bounds=(lower, upper),
    )
    # Truncated-linear attains the bounds; logit approaches them. Allow a tiny
    # float tolerance at the closed end.
    assert solution.g_weights.min() >= lower - 1e-9
    assert solution.g_weights.max() <= upper + 1e-9


def test_logit_weights_stay_strictly_interior_to_the_bounds() -> None:
    """Logit calibration confines g-weights to the *open* interval ``(L, U)``."""
    lower, upper = 0.2, 5.0
    problem = _extreme_shift_problem()
    solution = logit_weights(
        problem.matrix, problem.target, problem.initial_weights, bounds=(lower, upper)
    )
    assert solution.g_weights.min() > lower
    assert solution.g_weights.max() < upper


def test_bounded_methods_still_hit_feasible_targets() -> None:
    """With wide bounds containing the solution, bounded methods reproduce ``b``."""
    problem = feasible_problem(n_records=200, n_targets=8, seed=1).problem
    for distance in ("chi_square", "logit"):
        solution = calibrate_distance(
            distance, problem.matrix, problem.target, problem.initial_weights,
            bounds=BOUNDS,
        )
        assert solution.converged, distance
        np.testing.assert_allclose(
            problem.estimates(solution.weights), problem.target, rtol=1e-5, atol=1e-3
        )


def test_raking_equals_entropy_equals_exponential_tilting() -> None:
    """Raking, entropy balancing, and exponential tilting are the same solve.

    Deville-Sarndal: the raking distance ``G(g) = g log g - g + 1`` is the KL
    divergence from the design weights, so its minimizer *is* the
    entropy-balancing / exponential-tilting reweighting. Both registry keys map
    to the same solver, so their weights must be bit-identical.
    """
    from calibration_paper.methods import run_method

    problem = feasible_problem(n_records=150, n_targets=6, seed=2).problem
    raking = run_method("raking", problem)
    entropy = run_method("entropy", problem)
    np.testing.assert_array_equal(raking.outcome.weights, entropy.outcome.weights)


def test_margins_problem_is_the_raking_regime() -> None:
    """Two-way margin counts are reproduced by raking to machine precision."""
    problem = margins_problem(n_rows=3, n_cols=4, per_cell=30, seed=3).problem
    solution = raking_weights(problem.matrix, problem.target, problem.initial_weights)
    assert solution.converged
    assert (solution.weights > 0.0).all()
    np.testing.assert_allclose(
        problem.estimates(solution.weights), problem.target, rtol=1e-8
    )


@pytest.mark.parametrize(
    "solver",
    [
        lambda p: raking_weights(p.matrix, p.target, p.initial_weights),
        lambda p: linear_weights(p.matrix, p.target, p.initial_weights),
        lambda p: logit_weights(p.matrix, p.target, p.initial_weights, bounds=(0.2, 5.0)),
    ],
)
def test_infeasible_surface_does_not_converge(solver) -> None:
    """No reweighting satisfies contradictory targets: ``converged`` is False.

    The failure-mode signal: every method reports it could not hit the surface,
    and returns a finite best-effort weight vector (not NaN, not a crash).
    """
    problem = infeasible_problem(n_records=200, seed=4).problem
    solution = solver(problem)
    assert not solution.converged
    assert np.isfinite(solution.weights).all()
    assert solution.max_scaled_residual > 1e-3


def test_solvers_accept_a_sparse_matrix() -> None:
    """The classical solvers densify a sparse ``A`` and give the same answer."""
    problem = feasible_problem(n_records=120, n_targets=5, seed=5).problem
    dense_solution = raking_weights(
        problem.matrix, problem.target, problem.initial_weights
    )
    sparse_solution = raking_weights(
        sparse.csr_array(problem.matrix), problem.target, problem.initial_weights
    )
    np.testing.assert_allclose(
        dense_solution.weights, sparse_solution.weights, rtol=1e-9
    )


def test_logit_requires_bounds_via_dispatch() -> None:
    """``calibrate_distance('logit', ...)`` without bounds is a clear error."""
    problem = feasible_problem(n_records=50, n_targets=3, seed=6).problem
    with pytest.raises(ValueError, match="logit calibration requires bounds"):
        calibrate_distance("logit", problem.matrix, problem.target, problem.initial_weights)


def test_invalid_bounds_are_rejected() -> None:
    """Bounds must bracket the design ratio 1 (L < 1 < U)."""
    problem = feasible_problem(n_records=50, n_targets=3, seed=7).problem
    with pytest.raises(ValueError, match="bracket the design ratio"):
        logit_weights(problem.matrix, problem.target, problem.initial_weights, bounds=(1.2, 5.0))
    with pytest.raises(ValueError, match="L < U"):
        chi_square_weights(
            problem.matrix, problem.target, problem.initial_weights, bounds=(5.0, 0.2)
        )


def _extreme_shift_problem():
    """A tiny surface demanding a large concentrated shift (forces the pathologies).

    One total target (feasible at w0) plus one component target set far from its
    baseline, so hitting it needs extreme g-weights: GREG goes negative, and the
    bounded methods hit their bounds.
    """
    rng = np.random.default_rng(11)
    n = 50
    w0 = np.full(n, 100.0)
    x = rng.normal(size=n)
    matrix = np.vstack([np.ones(n), x])
    baseline = float((x * w0).sum())
    spread = float(np.abs(x * w0).sum())
    target = np.array([w0.sum(), baseline + 30.0 * spread], dtype=np.float64)
    from calibration_paper.problem import CalibrationProblem

    return CalibrationProblem(matrix=matrix, target=target, initial_weights=w0)
