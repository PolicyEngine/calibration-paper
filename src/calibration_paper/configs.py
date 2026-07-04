"""Declarative sweep configurations for the calibration-method comparison.

The paper's experiments are driven by *committed run configs* so every number
regenerates from a config plus a seed (PLAN.md "Honesty rules"). This module is
the config layer: small frozen dataclasses that describe a family of calibration
problems, and builders that materialize them into
:class:`~calibration_paper.problem.CalibrationProblem` instances the method
registry runs.

Two families are defined here, both unblocked because they need only the
synthetic builders in :mod:`calibration_paper.synthetic` (no frozen populace
surface, no sparsity-paper machinery):

* :class:`ScaleSweepConfig` -- the **scale sweep**: target-count curves
  ``10^2 -> 10^4`` (PLAN.md "Scale sweep") for accuracy, runtime, and
  feasibility behavior. Each grid point is a feasible synthetic problem at a
  fixed target count, so every method *can* hit the surface and the curves
  isolate how cost and quality move with scale.
* :class:`InfeasibilityConfig` -- the **deliberately infeasible surfaces** for
  the failure-mode comparison (hard-constraint infeasibility vs soft-loss
  compromise). Each grid point is an infeasible synthetic problem (contradictory
  targets) at a fixed record count.

When the frozen populace target surface lands (issue #2's blocked half, gated on
sparsity-paper#17), the real Ledger targets replace the synthetic builders here
behind the same config contract; see :mod:`calibration_paper.frozen_targets`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace

from calibration_paper.problem import CalibrationProblem
from calibration_paper.synthetic import (
    feasible_problem,
    infeasible_problem,
)

__all__ = [
    "SweepPoint",
    "ScaleSweepConfig",
    "InfeasibilityConfig",
    "SCALE_SWEEP",
    "INFEASIBILITY_SWEEP",
    "DEFAULT_CONFIGS",
]


@dataclass(frozen=True)
class SweepPoint:
    """One materialized problem in a sweep, tagged with the axes it varies on.

    Attributes:
        config_name: The owning config's :attr:`name` (groups points in the
            long-format artifacts).
        problem: The :class:`~calibration_paper.problem.CalibrationProblem` to
            run every method on.
        n_targets: The problem's target count (the scale-sweep x-axis).
        n_records: The problem's record count.
        seed: The seed this point was drawn with (paired across methods).
        feasible: Whether a weight vector reproduces the targets exactly (the
            scale sweep is feasible; the infeasibility sweep is not) -- so a
            scorer knows whether ``converged=False`` is expected.
        label: A stable human/tabular label, unique within the config.
    """

    config_name: str
    problem: CalibrationProblem
    n_targets: int
    n_records: int
    seed: int
    feasible: bool
    label: str


def _log_grid(start_pow: int, stop_pow: int, per_decade: int) -> tuple[int, ...]:
    """Integer log-spaced grid ``10^start .. 10^stop`` with ``per_decade`` points.

    Rounds to the nearest integer and de-duplicates (so a coarse grid on a short
    span never emits the same count twice). Endpoints are always included.

    Args:
        start_pow: Lower decade exponent (e.g. ``2`` for ``10^2``).
        stop_pow: Upper decade exponent (e.g. ``4`` for ``10^4``).
        per_decade: Points per decade (``1`` gives just the decade marks).

    Returns:
        A sorted tuple of distinct integer grid values.
    """
    if stop_pow < start_pow:
        raise ValueError(f"stop_pow ({stop_pow}) must be >= start_pow ({start_pow}).")
    if per_decade < 1:
        raise ValueError(f"per_decade must be >= 1, got {per_decade}.")
    n_steps = (stop_pow - start_pow) * per_decade
    values: list[int] = []
    for i in range(n_steps + 1):
        exponent = start_pow + i / per_decade
        values.append(int(round(10.0**exponent)))
    return tuple(sorted(set(values)))


@dataclass(frozen=True)
class ScaleSweepConfig:
    """The scale sweep: feasible problems over a target-count curve.

    The target count sweeps ``10^start_pow -> 10^stop_pow``; the record count is
    a fixed multiple of the target count (``records_per_target``) so the problem
    stays under-determined (the calibration regime) at every scale. Each
    ``(n_targets, seed)`` cell is a :func:`~calibration_paper.synthetic.feasible_problem`.

    Attributes:
        name: Config identifier (the artifact group key).
        start_pow: Lower target-count decade exponent. Default ``2`` (``10^2``).
        stop_pow: Upper target-count decade exponent. Default ``4`` (``10^4``).
        per_decade: Grid points per decade. Default ``3`` (``~2.15x`` steps).
        records_per_target: Records per target (keeps the system
            under-determined). Default ``50``.
        seeds: Seeds drawn at every grid point (paired across methods).
    """

    name: str = "scale_sweep"
    start_pow: int = 2
    stop_pow: int = 4
    per_decade: int = 3
    records_per_target: int = 50
    seeds: tuple[int, ...] = (0, 1, 2)

    def target_counts(self) -> tuple[int, ...]:
        """The target-count grid this config sweeps."""
        return _log_grid(self.start_pow, self.stop_pow, self.per_decade)

    def points(self) -> Iterator[SweepPoint]:
        """Materialize the sweep: one feasible problem per ``(count, seed)``.

        Yields:
            :class:`SweepPoint` in ascending target count, then seed. The record
            count is ``records_per_target * n_targets``.
        """
        for n_targets in self.target_counts():
            n_records = self.records_per_target * n_targets
            for seed in self.seeds:
                synthetic = feasible_problem(
                    n_records=n_records, n_targets=n_targets, seed=seed
                )
                yield SweepPoint(
                    config_name=self.name,
                    problem=synthetic.problem,
                    n_targets=n_targets,
                    n_records=n_records,
                    seed=seed,
                    feasible=True,
                    label=f"{self.name}/t{n_targets}/s{seed}",
                )

    def scaled_down(self, *, stop_pow: int = 3) -> ScaleSweepConfig:
        """A cheaper variant capped at a lower top decade (for CI / smoke runs).

        The full ``10^4`` sweep is expensive; tests and quick local runs use a
        ``10^2 -> 10^3`` variant with the same shape.
        """
        return replace(self, stop_pow=stop_pow)


@dataclass(frozen=True)
class InfeasibilityConfig:
    """Deliberately infeasible surfaces for the failure-mode comparison.

    Each grid point is a :func:`~calibration_paper.synthetic.infeasible_problem`
    (two identical target functionals demanding different values) at a fixed
    record count, so no reweighting satisfies it: a hard-constraint method
    reports non-convergence, a soft-loss method returns a compromise. Sweeping
    the record count checks the failure mode is scale-stable.

    Attributes:
        name: Config identifier.
        record_counts: The record counts to build infeasible surfaces at.
        seeds: Seeds per record count (paired across methods).
    """

    name: str = "infeasibility_sweep"
    record_counts: tuple[int, ...] = (200, 2_000, 20_000)
    seeds: tuple[int, ...] = (0, 1, 2)

    def points(self) -> Iterator[SweepPoint]:
        """Materialize the sweep: one infeasible problem per ``(records, seed)``.

        Yields:
            :class:`SweepPoint` with ``feasible=False`` and a fixed target count
            of 3 (the contradiction surface: one total + two contradictory
            component rows).
        """
        for n_records in self.record_counts:
            for seed in self.seeds:
                synthetic = infeasible_problem(n_records=n_records, seed=seed)
                problem = synthetic.problem
                yield SweepPoint(
                    config_name=self.name,
                    problem=problem,
                    n_targets=problem.n_targets,
                    n_records=n_records,
                    seed=seed,
                    feasible=False,
                    label=f"{self.name}/n{n_records}/s{seed}",
                )


#: The default scale sweep the paper runs (``10^2 -> 10^4``, three points/decade,
#: three seeds).
SCALE_SWEEP = ScaleSweepConfig()

#: The default infeasibility sweep (three record scales, three seeds).
INFEASIBILITY_SWEEP = InfeasibilityConfig()


@dataclass(frozen=True)
class ConfigSuite:
    """The committed set of sweep configs the paper regenerates from.

    Attributes:
        scale: The scale sweep.
        infeasibility: The infeasibility sweep.
    """

    scale: ScaleSweepConfig = field(default_factory=ScaleSweepConfig)
    infeasibility: InfeasibilityConfig = field(default_factory=InfeasibilityConfig)

    def all_points(self) -> Iterator[SweepPoint]:
        """Every point across both sweeps (scale first, then infeasibility)."""
        yield from self.scale.points()
        yield from self.infeasibility.points()


#: The committed config suite: import this to regenerate every problem the
#: paper's sweeps run.
DEFAULT_CONFIGS = ConfigSuite()
