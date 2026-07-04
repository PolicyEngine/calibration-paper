"""The gradient calibrator's contract, when torch is installed.

Skipped on the base install (``importorskip``), so CI stays torch-free. These
pin the gradient method's *contract* -- positivity, the hard ratio bound,
seed reproducibility, and soft-loss behavior on an infeasible surface -- not its
accuracy against the classical methods (that is the sweep's job).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="methods extra (torch) not installed")

from calibration_paper.methods import run_method  # noqa: E402
from calibration_paper.problem import target_relative_errors  # noqa: E402
from calibration_paper.sgd import capped_weighted_mape, sgd_calibrate  # noqa: E402
from calibration_paper.synthetic import (  # noqa: E402
    feasible_problem,
    infeasible_problem,
)


def test_sgd_weights_are_strictly_positive() -> None:
    """Adam on log-weights keeps every weight positive by construction."""
    problem = feasible_problem(n_records=200, n_targets=8, seed=0).problem
    solution = sgd_calibrate(problem.matrix, problem.target, problem.initial_weights)
    assert (solution.weights > 0.0).all()


def test_sgd_reduces_the_loss_below_the_starting_point() -> None:
    """Calibration lowers the capped-MAPE loss from its input-weight value."""
    problem = feasible_problem(n_records=200, n_targets=8, seed=1).problem
    solution = sgd_calibrate(problem.matrix, problem.target, problem.initial_weights)
    start_loss = capped_weighted_mape(
        problem.estimates(problem.initial_weights), problem.target
    )
    assert solution.final_loss < start_loss
    assert solution.loss_trajectory[0] == pytest.approx(start_loss, rel=1e-5)


def test_sgd_bounded_respects_the_hard_ratio_bound() -> None:
    """No calibrated weight exceeds ``max_weight_ratio * w0`` -- exactly."""
    problem = _extreme_shift_problem()
    solution = sgd_calibrate(
        problem.matrix, problem.target, problem.initial_weights, max_weight_ratio=3.0
    )
    ratios = solution.weights / problem.initial_weights
    assert ratios.max() <= 3.0 + 1e-9
    assert (solution.weights > 0.0).all()


def test_sgd_is_seed_reproducible() -> None:
    """Same seed, identical weights (the objective is deterministic given it)."""
    problem = feasible_problem(n_records=150, n_targets=6, seed=2).problem
    a = sgd_calibrate(problem.matrix, problem.target, problem.initial_weights, seed=7)
    b = sgd_calibrate(problem.matrix, problem.target, problem.initial_weights, seed=7)
    np.testing.assert_array_equal(a.weights, b.weights)


def test_sgd_compromises_on_an_infeasible_surface() -> None:
    """The soft loss settles between contradictory targets -- finite, no crash.

    The failure-mode contrast with the hard-constraint classical methods: where
    a bounded classical method reports non-convergence at the boundary, the
    gradient method returns a finite compromise weight vector.
    """
    problem = infeasible_problem(n_records=200, seed=3).problem
    solution = sgd_calibrate(problem.matrix, problem.target, problem.initial_weights)
    assert np.isfinite(solution.weights).all()
    errors = target_relative_errors(problem, solution.weights)
    contradiction = [
        i
        for i, fam in enumerate(problem.target_families)
        if fam == "contradiction"
    ]
    # The two contradiction rows are the SAME functional (identical matrix rows)
    # demanding +20% and -20% of the baseline, so A@w produces one common value:
    # no weight vector satisfies both. The failure-mode signal is therefore not
    # "each keeps a residual" (the loss may drive the achieved value to one side,
    # nearly hitting that target while missing the other by ~40%) but that the
    # contradiction is never *jointly* resolved and a real combined residual
    # remains -- versus a hard-constraint method, which reports non-convergence.
    assert not all(abs(errors[i]) < 1e-2 for i in contradiction), (
        "the soft loss cannot satisfy contradictory targets simultaneously"
    )
    combined_residual = sum(abs(errors[i]) for i in contradiction)
    assert combined_residual > 0.1, (
        f"a compromise on ±20% contradictory targets must leave a substantial "
        f"combined residual, got {combined_residual:.4f}"
    )


def test_run_method_dispatches_the_gradient_methods() -> None:
    """The registry's ``sgd`` / ``sgd_bounded`` run through the uniform path."""
    problem = feasible_problem(n_records=120, n_targets=5, seed=4).problem
    for key in ("sgd", "sgd_bounded"):
        result = run_method(key, problem, seed=0)
        assert np.isfinite(result.outcome.weights).all(), key
        assert (result.outcome.weights > 0.0).all(), key
        assert result.outcome.iterations > 0, key


def _extreme_shift_problem():
    """A surface demanding a large concentrated shift (to stress the bound)."""
    from calibration_paper.problem import CalibrationProblem

    rng = np.random.default_rng(11)
    n = 50
    w0 = np.full(n, 100.0)
    x = rng.normal(size=n)
    matrix = np.vstack([np.ones(n), x])
    baseline = float((x * w0).sum())
    spread = float(np.abs(x * w0).sum())
    target = np.array([w0.sum(), baseline + 30.0 * spread], dtype=np.float64)
    return CalibrationProblem(matrix=matrix, target=target, initial_weights=w0)
