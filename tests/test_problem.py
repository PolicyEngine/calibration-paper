"""The CalibrationProblem value type and the weight diagnostics.

Base-install tests: pure numpy/scipy. They pin the ``(A, b, w0)`` contract's
validation and the diagnostics' closed forms against hand-computable cases.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from calibration_paper.problem import (
    CalibrationProblem,
    target_relative_errors,
    weight_diagnostics,
)


def _simple_problem() -> CalibrationProblem:
    matrix = np.array([[1.0, 1.0, 1.0], [2.0, 0.0, 1.0]])
    w0 = np.array([1.0, 2.0, 3.0])
    target = matrix @ w0  # feasible at w0 itself
    return CalibrationProblem(matrix=matrix, target=target, initial_weights=w0)


def test_estimates_are_matrix_times_weights() -> None:
    """``problem.estimates(w)`` is ``A @ w`` (dense)."""
    problem = _simple_problem()
    w = np.array([2.0, 2.0, 2.0])
    np.testing.assert_allclose(problem.estimates(w), problem.matrix @ w)


def test_estimates_match_between_dense_and_sparse() -> None:
    """A sparse and dense encoding of the same ``A`` give identical estimates."""
    dense = _simple_problem()
    sparse_problem = CalibrationProblem(
        matrix=sparse.csr_array(dense.matrix),
        target=dense.target,
        initial_weights=dense.initial_weights,
    )
    assert sparse_problem.is_sparse
    w = np.array([1.5, 0.5, 2.0])
    np.testing.assert_allclose(sparse_problem.estimates(w), dense.estimates(w))
    np.testing.assert_allclose(sparse_problem.dense(), dense.matrix)


def test_default_names_and_families_are_filled() -> None:
    """Omitted names/families default to ``t{i}`` and a single ``all`` family."""
    problem = _simple_problem()
    assert problem.target_names == ("t0", "t1")
    assert problem.target_families == ("all", "all")


def test_shape_mismatches_are_rejected() -> None:
    """A target/weight length that disagrees with the matrix is an error."""
    matrix = np.ones((2, 3))
    with pytest.raises(ValueError, match="target must have shape"):
        CalibrationProblem(matrix=matrix, target=np.ones(3), initial_weights=np.ones(3))
    with pytest.raises(ValueError, match="initial_weights must have shape"):
        CalibrationProblem(matrix=matrix, target=np.ones(2), initial_weights=np.ones(2))


def test_nonpositive_initial_weights_are_rejected() -> None:
    """Calibration needs a strictly positive design weight."""
    matrix = np.ones((1, 2))
    with pytest.raises(ValueError, match="strictly positive"):
        CalibrationProblem(
            matrix=matrix, target=np.array([1.0]), initial_weights=np.array([1.0, 0.0])
        )


def test_nonfinite_inputs_are_rejected() -> None:
    """NaN/inf in the matrix or target is a construction error, not a big miss."""
    with pytest.raises(ValueError, match="target must be finite"):
        CalibrationProblem(
            matrix=np.ones((1, 2)),
            target=np.array([np.nan]),
            initial_weights=np.ones(2),
        )
    bad = np.array([[1.0, np.inf]])
    with pytest.raises(ValueError, match="matrix must be finite"):
        CalibrationProblem(
            matrix=bad, target=np.array([1.0]), initial_weights=np.ones(2)
        )


def test_subset_targets_keeps_the_selected_rows() -> None:
    """``subset_targets`` selects matrix rows, target, names, and families together."""
    problem = CalibrationProblem(
        matrix=np.arange(6.0).reshape(3, 2),
        target=np.array([1.0, 2.0, 3.0]),
        initial_weights=np.array([1.0, 1.0]),
        target_names=("a", "b", "c"),
        target_families=("x", "y", "x"),
    )
    kept = problem.subset_targets([True, False, True])
    assert kept.n_targets == 2
    assert kept.target_names == ("a", "c")
    assert kept.target_families == ("x", "x")
    np.testing.assert_allclose(kept.target, [1.0, 3.0])
    np.testing.assert_allclose(kept.dense(), [[0.0, 1.0], [4.0, 5.0]])


def test_weight_diagnostics_on_uniform_weights() -> None:
    """Uniform weights: ESS = n, ratio percentiles all 1, no negatives."""
    w0 = np.array([2.0, 2.0, 2.0, 2.0])
    diag = weight_diagnostics(w0.copy(), w0)
    assert diag.effective_sample_size == pytest.approx(4.0)
    assert diag.ess_ratio == pytest.approx(1.0)
    assert diag.negative_weight_share == 0.0
    assert diag.max_weight_ratio == pytest.approx(1.0)
    assert diag.mass_ratio == pytest.approx(1.0)
    for value in diag.weight_ratio_percentiles.values():
        assert value == pytest.approx(1.0)


def test_weight_diagnostics_reports_concentration_and_negatives() -> None:
    """Concentrated mass lowers ESS; a negative weight is counted."""
    w0 = np.array([1.0, 1.0, 1.0, 1.0])
    w = np.array([10.0, 0.1, 0.1, -0.2])
    diag = weight_diagnostics(w, w0)
    # ESS uses |w|; it is far below the record count when mass concentrates.
    assert 0.0 < diag.effective_sample_size < 4.0
    assert diag.negative_weight_share == pytest.approx(0.25)
    assert diag.min_weight == pytest.approx(-0.2)
    assert diag.max_weight_ratio == pytest.approx(10.0)


def test_weight_diagnostics_flatten_includes_percentile_keys() -> None:
    """The flat form names each percentile ``weight_ratio_p{pct}``."""
    w0 = np.ones(10)
    flat = weight_diagnostics(np.arange(1.0, 11.0), w0).as_flat()
    assert "weight_ratio_p100" in flat
    assert "effective_sample_size" in flat
    assert "negative_weight_share" in flat


def test_target_relative_errors_use_target_defined_scale() -> None:
    """Relative error is ``(A@w - b) / max(|b|, 1)``; zero at exact reproduction."""
    problem = _simple_problem()
    exact = target_relative_errors(problem, problem.initial_weights)
    np.testing.assert_allclose(exact, 0.0, atol=1e-12)
    # A zero-valued target uses a unit scale (no divide-by-zero).
    problem_zero = CalibrationProblem(
        matrix=np.array([[1.0, -1.0]]),
        target=np.array([0.0]),
        initial_weights=np.array([1.0, 1.0]),
    )
    err = target_relative_errors(problem_zero, np.array([3.0, 1.0]))
    assert err[0] == pytest.approx(2.0)  # (3 - 1 - 0) / max(0, 1) = 2
