"""Pre/post-calibration population-view delta, via popdgp (pinned).

PLAN.md's metric list includes the **pre/post-calibration population-view
delta**: project the candidate file through each survey view and score it
against that view's holdout with the population-view harness, *before* and
*after* calibration, and report the change. Because PRDC **coverage** is
invariant to any reweighting of the candidate -- its neighbourhood radii are
unweighted support geometry and it is the weighted fraction of *holdout* points
covered, so only the (fixed) holdout weights enter -- the coverage delta is a
clean wiring check: it is exactly zero when the candidate is scored without a
weighted resample, and only the weight-sensitive blocks (energy distance, and
the point-weighted precision/density) move. That isolates what a calibrator does
to the *weighted joint*, separate from whether it hit the targets.

**Exactness of the coverage check.** popdgp resamples a side down to
``max_points`` *weight-proportionally* when it exceeds the cap, which turns the
weighted measure into a uniform one and (because pre and post use different
weights) perturbs even coverage by Monte Carlo noise. So the coverage-delta
sanity check is exact only when the candidate stays at or below ``max_points``
(no resample: identical point set, unweighted radii). Above the cap the delta is
still meaningful but carries resample noise; :func:`population_view_delta`
defaults ``max_points`` high enough that the synthetic sweep's per-point views
stay in the exact regime, and the real frozen surface sets it per its record
budget.

The harness itself is **imported from popdgp, pinned by commit SHA** (never
copied): :func:`popdgp.views.harness_scorecard`, extracted per popdgp#1. This
module only builds the candidate/holdout tables from a
:class:`~calibration_paper.problem.CalibrationProblem` and a weight vector, calls
the harness twice (design weights vs calibrated weights against the *same*
holdout), and differences the two long-format scorecards.

popdgp is an optional dependency (the ``popdgp`` extra); the import is lazy so
the base install and the synthetic sweep never require it. When popdgp is
absent, :func:`population_view_delta` raises a clear ``ModuleNotFoundError`` that
the sweep records as a skip.

**On the candidate/holdout construction.** A ``CalibrationProblem`` is an
abstract ``(A, b, w0)`` system with no named survey variables, so for the
synthetic sweep we derive a view's variable space from the problem itself: the
continuous rows of ``A`` are per-record measures, i.e. record-level feature
columns. The candidate table carries those columns plus its weight column; the
holdout is an independent reference draw from the same generative structure,
fixed across the pre/post calls (a survey's holdout does not change when we
recalibrate the candidate). On the *real* frozen surface (issue #2's blocked
half) the views come from the Ledger/populace variable taxonomy instead; this
synthetic bridge lets the delta wiring be built and tested against the real
popdgp API now.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from calibration_paper.problem import CalibrationProblem

__all__ = [
    "ViewDeltaRow",
    "candidate_frame_from_problem",
    "default_views_for_problem",
    "population_view_delta",
    "WEIGHT_COLUMN",
]

#: The weight column name the candidate tables carry (the harness reads it by
#: name; kept constant so callers and holdouts agree).
WEIGHT_COLUMN = "weight"

#: Max continuous rows of ``A`` used as synthetic view columns. Keeps the view
#: dimensionality modest (the harness's pairwise metrics are the cost) while
#: still exercising a multivariate joint.
_MAX_VIEW_COLUMNS = 6


@dataclass(frozen=True)
class ViewDeltaRow:
    """One (view, metric) entry of the pre/post-calibration delta.

    Attributes:
        view: The survey view's name.
        metric: The harness metric name (``energy_distance``, ``c2st_auc``,
            ``prdc_*``, or a per-target tail metric).
        pre: The metric on the candidate at the design weights ``w0``.
        post: The metric on the candidate at the calibrated weights ``w``.
        delta: ``post - pre`` -- the calibrator's effect on this metric. For
            ``prdc_coverage`` this is ~0 by construction (coverage is
            reweight-invariant), which doubles as a wiring sanity check.
    """

    view: str
    metric: str
    pre: float
    post: float
    delta: float


def _continuous_rows(problem: CalibrationProblem) -> list[int]:
    """Indices of ``A`` rows that are continuous measures (not 0/1 indicators).

    A view's variable space is a *continuous* joint; pure 0/1 indicator rows
    (subpopulation-count targets) collapse to two mass points and make the
    pairwise geometry degenerate, so they are excluded from the synthetic view.
    """
    dense = problem.dense()
    rows: list[int] = []
    for i in range(dense.shape[0]):
        row = dense[i]
        unique = np.unique(row)
        is_indicator = set(np.round(unique, 12)).issubset({0.0, 1.0})
        is_constant = unique.size == 1
        if not is_indicator and not is_constant:
            rows.append(i)
    return rows


def candidate_frame_from_problem(
    problem: CalibrationProblem,
    weights: np.ndarray,
    *,
    columns: list[int] | None = None,
) -> pd.DataFrame:
    """Build a candidate weighted table from a problem and a weight vector.

    Each selected continuous row of ``A`` becomes a record-level feature column
    (``m{row_index}``); the weight vector becomes :data:`WEIGHT_COLUMN`. This is
    the "candidate population file" the harness projects through each view.

    Args:
        problem: The calibration problem (supplies the per-record measures).
        weights: The weight vector to attach (design ``w0`` for the pre score,
            calibrated ``w`` for the post score).
        columns: Row indices of ``A`` to use as feature columns; defaults to the
            continuous rows (capped at :data:`_MAX_VIEW_COLUMNS`).

    Returns:
        A ``DataFrame`` with one ``m{i}`` column per selected row and a
        :data:`WEIGHT_COLUMN` column, ``n_records`` rows long.

    Raises:
        ValueError: If ``weights`` is the wrong length, or no continuous row is
            available to form a view.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (problem.n_records,):
        raise ValueError(
            f"weights must have shape ({problem.n_records},), got {w.shape}."
        )
    if columns is None:
        columns = _continuous_rows(problem)[:_MAX_VIEW_COLUMNS]
    if not columns:
        raise ValueError(
            "No continuous rows in the constraint matrix to form a population "
            "view; the population-view delta needs at least one continuous "
            "measure (an all-indicator/constant surface has no continuous joint)."
        )
    dense = problem.dense()
    data = {f"m{i}": dense[i].astype(np.float64) for i in columns}
    data[WEIGHT_COLUMN] = w
    return pd.DataFrame(data)


def default_views_for_problem(
    problem: CalibrationProblem,
    *,
    columns: list[int] | None = None,
):
    """A single synthetic :class:`popdgp.views.SurveyView` over the problem.

    Wraps the continuous feature columns as one view named ``"synthetic"``, with
    no imputed-target tail block (the synthetic measures are all "carried"
    columns). On the real frozen surface this is replaced by the Ledger/populace
    view taxonomy.

    Args:
        problem: The calibration problem.
        columns: Row indices used as feature columns (must match the candidate
            frame's columns); defaults to the continuous rows.

    Returns:
        A one-element list ``[SurveyView(...)]``.

    Raises:
        ModuleNotFoundError: If popdgp is not installed (the ``popdgp`` extra).
        ValueError: If there is no continuous row to view.
    """
    from popdgp.views import SurveyView

    if columns is None:
        columns = _continuous_rows(problem)[:_MAX_VIEW_COLUMNS]
    if not columns:
        raise ValueError(
            "No continuous rows to form a population view (see "
            "candidate_frame_from_problem)."
        )
    view_columns = tuple(f"m{i}" for i in columns)
    return [
        SurveyView(
            name="synthetic",
            columns=view_columns,
            weight_column=WEIGHT_COLUMN,
            target_columns=(),
        )
    ]


def _holdout_frame(
    problem: CalibrationProblem,
    columns: list[int],
    *,
    seed: int,
) -> pd.DataFrame:
    """An independent reference draw for the view's holdout.

    A survey's holdout is a weighted sample from the population view that was
    *not* used upstream. For the synthetic bridge we draw a fresh record set of
    the same size from the problem's own generative structure by bootstrap
    resampling the candidate's continuous measures under the design weights, then
    carrying uniform holdout weights. It is built once and shared by the pre and
    post scores (recalibrating the candidate does not change the survey holdout),
    which is what makes the coverage delta ~0 and the joint delta meaningful.

    Args:
        problem: The calibration problem.
        columns: The feature-column row indices (same as the candidate).
        seed: Seed for the holdout resample (distinct from the method seed).

    Returns:
        A holdout ``DataFrame`` with the same ``m{i}`` columns and a
        :data:`WEIGHT_COLUMN` of ones.
    """
    rng = np.random.default_rng(seed)
    dense = problem.dense()
    w0 = problem.initial_weights
    probabilities = w0 / w0.sum()
    n = problem.n_records
    # Weighted bootstrap: the design-weighted population is the reference the
    # holdout samples from, so the holdout reflects the pre-calibration joint.
    draw = rng.choice(n, size=n, replace=True, p=probabilities)
    data = {f"m{i}": dense[i][draw].astype(np.float64) for i in columns}
    data[WEIGHT_COLUMN] = np.ones(n, dtype=np.float64)
    return pd.DataFrame(data)


def population_view_delta(
    problem: CalibrationProblem,
    initial_weights: np.ndarray,
    calibrated_weights: np.ndarray,
    *,
    seed: int = 0,
    max_points: int = 4096,
    columns: list[int] | None = None,
) -> list[ViewDeltaRow]:
    """The pre/post-calibration population-view delta for one calibrated file.

    Scores the candidate at the design weights (pre) and at the calibrated
    weights (post) against the *same* holdout, using
    :func:`popdgp.views.harness_scorecard` (pinned), and returns the per-(view,
    metric) change. Coverage is reweight-invariant, so ``prdc_coverage`` deltas
    ~0 -- a built-in check that the wiring changed only the weights.

    Args:
        problem: The calibration problem the weights solve.
        initial_weights: The design weights ``w0`` (the pre condition).
        calibrated_weights: The calibrated weights ``w`` (the post condition).
        seed: Seed for the harness (resampling / C2ST folds) and the holdout
            draw; paired across methods.
        max_points: Pairwise-metric size cap passed to the harness. Kept above
            the synthetic sweep's per-view record counts so the candidate is not
            weight-resampled and the coverage delta is exactly zero (the wiring
            check); lower it only knowingly (it trades the exact coverage check
            for speed on large frames).
        columns: Row indices to use as view columns; defaults to the continuous
            rows (shared by candidate and holdout).

    Returns:
        One :class:`ViewDeltaRow` per (view, metric).

    Raises:
        ModuleNotFoundError: If popdgp is not installed (the ``popdgp`` extra).
        ValueError: On a weight-length mismatch or a view-less (all-indicator)
            surface.
    """
    from popdgp.views import harness_scorecard

    if columns is None:
        columns = _continuous_rows(problem)[:_MAX_VIEW_COLUMNS]
    if not columns:
        raise ValueError(
            "population_view_delta needs at least one continuous measure row; "
            "this surface is all-indicator/constant and has no population view."
        )

    views = default_views_for_problem(problem, columns=columns)
    holdout = _holdout_frame(problem, columns, seed=seed)
    holdouts = {"synthetic": holdout}

    pre_frame = candidate_frame_from_problem(problem, initial_weights, columns=columns)
    post_frame = candidate_frame_from_problem(
        problem, calibrated_weights, columns=columns
    )
    pre_rows = harness_scorecard(
        pre_frame, WEIGHT_COLUMN, views, holdouts, max_points=max_points, seed=seed
    )
    post_rows = harness_scorecard(
        post_frame, WEIGHT_COLUMN, views, holdouts, max_points=max_points, seed=seed
    )

    pre_by_key = {(r["view"], r["metric"]): float(r["value"]) for r in pre_rows}
    post_by_key = {(r["view"], r["metric"]): float(r["value"]) for r in post_rows}
    delta_rows: list[ViewDeltaRow] = []
    for key in pre_by_key:
        if key not in post_by_key:  # pragma: no cover - same views => same keys
            continue
        view, metric = key
        pre_value = pre_by_key[key]
        post_value = post_by_key[key]
        delta_rows.append(
            ViewDeltaRow(
                view=view,
                metric=metric,
                pre=pre_value,
                post=post_value,
                delta=post_value - pre_value,
            )
        )
    return delta_rows
