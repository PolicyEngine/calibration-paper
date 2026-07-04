"""The sweep config layer: scale curves and infeasibility surfaces.

Base-install tests (pure numpy via the synthetic builders): the configs
materialize the right problems at the right scales, feasible where they claim to
be and infeasible where they claim to be, so the sweep regenerates from a config
plus a seed.
"""

from __future__ import annotations

import numpy as np

from calibration_paper.configs import (
    DEFAULT_CONFIGS,
    INFEASIBILITY_SWEEP,
    SCALE_SWEEP,
    ConfigSuite,
    InfeasibilityConfig,
    ScaleSweepConfig,
    SweepPoint,
)
from calibration_paper.problem import CalibrationProblem


def test_scale_sweep_covers_two_to_four_decades() -> None:
    """The default scale sweep spans 10^2 -> 10^4 with the endpoints present."""
    counts = SCALE_SWEEP.target_counts()
    assert counts[0] == 100
    assert counts[-1] == 10_000
    # Strictly increasing and distinct.
    assert list(counts) == sorted(set(counts))
    # Three points per decade over two decades => 7 grid points.
    assert len(counts) == 7


def test_scale_sweep_points_are_feasible_and_shaped() -> None:
    """Each scale point is a feasible problem with the declared shape."""
    config = ScaleSweepConfig(start_pow=2, stop_pow=2, per_decade=1, seeds=(0,))
    points = list(config.points())
    assert len(points) == 1
    point = points[0]
    assert isinstance(point, SweepPoint)
    assert isinstance(point.problem, CalibrationProblem)
    assert point.feasible is True
    assert point.n_targets == 100
    assert point.n_records == 100 * config.records_per_target
    assert point.problem.n_targets == point.n_targets
    assert point.problem.n_records == point.n_records


def test_scale_sweep_pairs_every_count_with_every_seed() -> None:
    """The sweep is the product of the count grid and the seed set."""
    config = ScaleSweepConfig(start_pow=2, stop_pow=3, per_decade=1, seeds=(0, 1))
    points = list(config.points())
    # 2 decade marks (100, 1000) x 2 seeds.
    assert len(points) == 4
    seeds = {p.seed for p in points}
    counts = {p.n_targets for p in points}
    assert seeds == {0, 1}
    assert counts == {100, 1000}
    # Labels are unique.
    assert len({p.label for p in points}) == len(points)


def test_scale_sweep_is_seed_reproducible() -> None:
    """The same config yields identical problems across two materializations."""
    config = ScaleSweepConfig(start_pow=2, stop_pow=2, per_decade=1, seeds=(3,))
    a = list(config.points())[0]
    b = list(config.points())[0]
    np.testing.assert_array_equal(a.problem.dense(), b.problem.dense())
    np.testing.assert_array_equal(a.problem.target, b.problem.target)


def test_scaled_down_lowers_the_top_decade() -> None:
    """``scaled_down`` caps the sweep at a lower decade for CI runs."""
    small = SCALE_SWEEP.scaled_down(stop_pow=3)
    assert small.target_counts()[-1] == 1000
    assert small.start_pow == SCALE_SWEEP.start_pow


def test_infeasibility_sweep_points_are_infeasible() -> None:
    """Each infeasibility point carries contradictory targets (no true weights)."""
    config = InfeasibilityConfig(record_counts=(200,), seeds=(0,))
    points = list(config.points())
    assert len(points) == 1
    point = points[0]
    assert point.feasible is False
    # The contradiction: two identical functional rows, different targets.
    dense = point.problem.dense()
    np.testing.assert_array_equal(dense[1], dense[2])
    assert point.problem.target[1] != point.problem.target[2]


def test_infeasibility_sweep_scales_records() -> None:
    """The record count sweeps while the target count stays at the 3-row surface."""
    config = InfeasibilityConfig(record_counts=(200, 2000), seeds=(0,))
    points = list(config.points())
    assert {p.n_records for p in points} == {200, 2000}
    assert all(p.n_targets == 3 for p in points)


def test_default_config_suite_yields_both_sweeps() -> None:
    """The committed suite emits the scale points then the infeasibility points."""
    assert isinstance(DEFAULT_CONFIGS, ConfigSuite)
    suite = ConfigSuite(
        scale=ScaleSweepConfig(start_pow=2, stop_pow=2, per_decade=1, seeds=(0,)),
        infeasibility=InfeasibilityConfig(record_counts=(200,), seeds=(0,)),
    )
    points = list(suite.all_points())
    names = [p.config_name for p in points]
    assert names == ["scale_sweep", "infeasibility_sweep"]
    assert points[0].feasible and not points[1].feasible


def test_default_singletons_have_expected_shape() -> None:
    """The module-level defaults match PLAN.md (10^2->10^4, three record scales)."""
    assert SCALE_SWEEP.start_pow == 2 and SCALE_SWEEP.stop_pow == 4
    assert INFEASIBILITY_SWEEP.record_counts == (200, 2_000, 20_000)
