"""The pre/post-calibration population-view delta, against the pinned popdgp.

Skipped unless the ``popdgp`` extra is installed (``importorskip``), so the base
install and the synthetic sweep never require it. These pin the *wiring*: the
candidate/holdout tables build from a ``(A, b, w0)`` problem, the harness runs
pre and post against the same holdout, and the delta behaves as the harness's
invariances predict -- coverage delta exactly zero below the resample cap
(reweight-invariant support geometry), a real, nonzero energy-distance delta
when calibration moved the weights.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("popdgp", reason="popdgp extra not installed")

from calibration_paper.methods import run_method  # noqa: E402
from calibration_paper.population_view import (  # noqa: E402
    WEIGHT_COLUMN,
    ViewDeltaRow,
    candidate_frame_from_problem,
    default_views_for_problem,
    population_view_delta,
)
from calibration_paper.synthetic import feasible_problem  # noqa: E402


def _calibrated(seed: int = 0, *, n_records: int = 200, n_targets: int = 6):
    """A feasible problem plus a raking-calibrated weight vector for it."""
    problem = feasible_problem(
        n_records=n_records, n_targets=n_targets, seed=seed
    ).problem
    result = run_method("raking", problem, seed=seed)
    return problem, problem.initial_weights, result.outcome.weights


def test_candidate_frame_carries_columns_and_weight() -> None:
    """The candidate frame has one ``m{i}`` per continuous row plus the weight."""
    problem, w0, _ = _calibrated()
    frame = candidate_frame_from_problem(problem, w0)
    assert WEIGHT_COLUMN in frame.columns
    measure_cols = [c for c in frame.columns if c != WEIGHT_COLUMN]
    assert measure_cols  # at least one continuous measure
    assert len(frame) == problem.n_records
    np.testing.assert_array_equal(frame[WEIGHT_COLUMN].to_numpy(), w0)


def test_view_columns_match_the_candidate_frame() -> None:
    """The view's columns are exactly the candidate frame's measure columns."""
    problem, w0, _ = _calibrated()
    frame = candidate_frame_from_problem(problem, w0)
    views = default_views_for_problem(problem)
    assert len(views) == 1
    measure_cols = tuple(c for c in frame.columns if c != WEIGHT_COLUMN)
    assert views[0].columns == measure_cols
    assert views[0].weight_column == WEIGHT_COLUMN


def test_delta_returns_a_row_per_view_metric() -> None:
    """The delta is a nonempty list of ``ViewDeltaRow`` with pre/post/delta set."""
    problem, w0, w = _calibrated()
    rows = population_view_delta(problem, w0, w, seed=0)
    assert rows and all(isinstance(r, ViewDeltaRow) for r in rows)
    for row in rows:
        assert row.delta == pytest.approx(row.post - row.pre)
    metrics = {r.metric for r in rows}
    # The harness's joint + coverage blocks are present.
    assert "energy_distance" in metrics
    assert "prdc_coverage" in metrics


def test_coverage_delta_is_zero_below_the_resample_cap() -> None:
    """Coverage is reweight-invariant: its delta is exactly 0 (no resample).

    The candidate (200 records) is below ``max_points`` so it is not
    weight-resampled; PRDC coverage uses unweighted support geometry on the same
    point set, so recalibrating the candidate cannot change it. This is the
    wiring's built-in sanity check.
    """
    problem, w0, w = _calibrated(n_records=200)
    rows = population_view_delta(problem, w0, w, seed=0, max_points=4096)
    coverage = next(r for r in rows if r.metric == "prdc_coverage")
    assert coverage.delta == 0.0


def test_energy_distance_delta_is_nonzero_when_weights_moved() -> None:
    """A real calibration shift shows up in the weight-sensitive joint block.

    Raking on a feasible-but-perturbed surface moves the weights off ``w0``, so
    the energy distance to the (design-weighted) holdout must change -- the delta
    is not a no-op.
    """
    problem, w0, w = _calibrated(n_records=200)
    # Confirm the weights actually moved (else the test is vacuous).
    assert not np.allclose(w, w0)
    rows = population_view_delta(problem, w0, w, seed=0, max_points=4096)
    energy = next(r for r in rows if r.metric == "energy_distance")
    assert energy.delta != 0.0


def test_delta_is_seed_reproducible() -> None:
    """Same problem, weights, and seed => identical delta rows."""
    problem, w0, w = _calibrated(n_records=200, seed=1)
    first = population_view_delta(problem, w0, w, seed=2, max_points=4096)
    second = population_view_delta(problem, w0, w, seed=2, max_points=4096)
    assert [(r.view, r.metric, r.delta) for r in first] == [
        (r.view, r.metric, r.delta) for r in second
    ]


def test_all_indicator_surface_has_no_population_view() -> None:
    """A surface with no continuous measure raises a clear error, not a crash."""
    from calibration_paper.problem import CalibrationProblem

    # Only a constant row and a 0/1 indicator row: no continuous joint.
    matrix = np.vstack([np.ones(20), (np.arange(20) % 2).astype(float)])
    problem = CalibrationProblem(
        matrix=matrix,
        target=np.array([20.0, 10.0]),
        initial_weights=np.ones(20),
    )
    with pytest.raises(ValueError, match="no population view|continuous"):
        population_view_delta(problem, problem.initial_weights, problem.initial_weights)
