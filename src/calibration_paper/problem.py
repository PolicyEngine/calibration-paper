"""The linear calibration problem ``(A, b, w0)`` and the diagnostics on a solve.

Every calibrator the paper compares -- classical (raking, GREG, chi-square,
entropy, logit-bounded) and gradient (SGD, SGD-bounded) -- reduces to the same
object: a design/constraint matrix ``A`` of shape ``(n_targets, n_records)``,
the administrative target vector ``b`` of length ``n_targets``, and the input
(design) weights ``w0`` of length ``n_records``. Calibration searches for a new
weight vector ``w`` with ``A @ w`` close to ``b``.

This is the linear form the whole calibration literature works in: R
``survey``'s ``grake`` takes exactly ``(mm=A.T, ww=w0, population=b)``, and
populace-calibrate compiles a Frame + declared facts into precisely this
``(A, b, w0)`` sparse system. Fixing the comparison at this level means every
method sees an *identical* problem -- no method wins by a different compilation
-- which is the frozen-input discipline the sibling papers use for the fill and
selection operators.

Two representations of ``A`` are supported behind one type: a dense
``numpy.ndarray`` (small problems, exact) and a ``scipy.sparse`` matrix (the
national-scale hierarchical surface, where ``A`` is mostly zeros). Adapters read
``problem.matrix`` (native) or the ``dense``/``estimates`` helpers and never
special-case the layout.

The diagnostics here are the paper's weight-quality axis (PLAN.md "Weight
diagnostics"): effective sample size, weight-ratio percentiles, and
negative-weight incidence. They are pure functions of ``(w, w0)`` so they score
any calibrator's output -- including ones (GREG) that produce negative weights,
which is the point of reporting them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

__all__ = [
    "CalibrationProblem",
    "WeightDiagnostics",
    "weight_diagnostics",
    "target_relative_errors",
]

#: Weight-ratio percentiles reported by :func:`weight_diagnostics` (the tail of
#: the ``w / w0`` distribution -- how far calibration stretched its most
#: inflated records, the "landmine" exposure the bounded methods guard).
DEFAULT_RATIO_PERCENTILES: tuple[float, ...] = (50.0, 90.0, 99.0, 100.0)


@dataclass(frozen=True)
class CalibrationProblem:
    """One linear calibration problem: reproduce ``b`` from ``A`` by reweighting.

    Attributes:
        matrix: The constraint matrix ``A`` of shape ``(n_targets, n_records)``,
            either a dense ``numpy.ndarray`` or a ``scipy.sparse`` matrix. Row
            ``i`` is the per-record contribution to target ``i`` (e.g. the value
            of a component for a sum target, or an indicator for a count target),
            so ``A @ w`` is the vector of achieved aggregates.
        target: The right-hand side ``b`` of length ``n_targets`` -- the
            administrative values the aggregates should match.
        initial_weights: The input (design) weights ``w0`` of length
            ``n_records``, the calibration's starting point and mass reference.
        target_names: One label per target row (for diagnostics and held-out
            family grouping). Defaults to ``t0, t1, ...``.
        target_families: One family label per target row, so the held-out
            evaluation can fit on some families and score on the rest. Defaults
            to a single ``"all"`` family.

    Raises:
        ValueError: If the shapes are inconsistent, or any input is non-finite,
            or the initial weights are not all strictly positive (calibration is
            defined relative to a positive design weight).
    """

    matrix: np.ndarray | sparse.spmatrix
    target: np.ndarray
    initial_weights: np.ndarray
    target_names: tuple[str, ...] = ()
    target_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        target = np.asarray(self.target, dtype=np.float64)
        w0 = np.asarray(self.initial_weights, dtype=np.float64)
        n_targets, n_records = self._matrix_shape()
        if target.shape != (n_targets,):
            raise ValueError(
                f"target must have shape ({n_targets},) to match the matrix's "
                f"{n_targets} rows, got {target.shape}."
            )
        if w0.shape != (n_records,):
            raise ValueError(
                f"initial_weights must have shape ({n_records},) to match the "
                f"matrix's {n_records} columns, got {w0.shape}."
            )
        if not np.isfinite(target).all():
            raise ValueError("target must be finite.")
        if not np.isfinite(w0).all():
            raise ValueError("initial_weights must be finite.")
        if (w0 <= 0.0).any():
            raise ValueError(
                "initial_weights must be strictly positive; calibration is "
                "defined relative to a positive design weight."
            )
        if not _matrix_is_finite(self.matrix):
            raise ValueError("matrix must be finite (no NaN/inf entries).")
        # Freeze normalized copies so downstream reads are dtype-stable and the
        # caller's arrays are never aliased/mutated.
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "initial_weights", w0)
        names = self.target_names or tuple(f"t{i}" for i in range(n_targets))
        if len(names) != n_targets:
            raise ValueError(
                f"target_names must have {n_targets} entries, got {len(names)}."
            )
        families = self.target_families or ("all",) * n_targets
        if len(families) != n_targets:
            raise ValueError(
                f"target_families must have {n_targets} entries, got {len(families)}."
            )
        object.__setattr__(self, "target_names", tuple(names))
        object.__setattr__(self, "target_families", tuple(families))

    def _matrix_shape(self) -> tuple[int, int]:
        shape = self.matrix.shape
        if len(shape) != 2:
            raise ValueError(f"matrix must be 2-D, got shape {shape}.")
        return int(shape[0]), int(shape[1])

    @property
    def n_targets(self) -> int:
        """Number of target rows (constraints)."""
        return int(self.matrix.shape[0])

    @property
    def n_records(self) -> int:
        """Number of records (columns / weights)."""
        return int(self.matrix.shape[1])

    @property
    def is_sparse(self) -> bool:
        """Whether the matrix is stored as a ``scipy.sparse`` matrix."""
        return sparse.issparse(self.matrix)

    def dense(self) -> np.ndarray:
        """Return ``A`` as a dense float64 array (materializes a sparse matrix)."""
        if sparse.issparse(self.matrix):
            return np.asarray(self.matrix.toarray(), dtype=np.float64)
        return np.asarray(self.matrix, dtype=np.float64)

    def estimates(self, weights: np.ndarray) -> np.ndarray:
        """Achieved aggregates ``A @ weights`` for a weight vector.

        Args:
            weights: A weight vector of length :attr:`n_records`.

        Returns:
            The length-:attr:`n_targets` vector of achieved target aggregates.
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (self.n_records,):
            raise ValueError(
                f"weights must have shape ({self.n_records},), got {w.shape}."
            )
        result = self.matrix @ w
        return np.asarray(result, dtype=np.float64).ravel()

    def subset_targets(self, keep: Sequence[bool] | np.ndarray) -> CalibrationProblem:
        """Return a problem keeping only the target rows where ``keep`` is true.

        Used by the held-out evaluation: fit on the in-family rows, then score
        the fitted weights against the held-out rows (a separate subset). The
        matrix rows, target vector, names, and families are all subset together.

        Args:
            keep: A boolean mask of length :attr:`n_targets`.

        Returns:
            A new :class:`CalibrationProblem` over the kept rows and the same
            records / initial weights.
        """
        mask = np.asarray(keep, dtype=bool)
        if mask.shape != (self.n_targets,):
            raise ValueError(
                f"keep must have shape ({self.n_targets},), got {mask.shape}."
            )
        rows = np.flatnonzero(mask)
        sub_matrix = self.matrix[rows, :]
        return CalibrationProblem(
            matrix=sub_matrix,
            target=self.target[rows],
            initial_weights=self.initial_weights,
            target_names=tuple(self.target_names[i] for i in rows),
            target_families=tuple(self.target_families[i] for i in rows),
        )


def _matrix_is_finite(matrix: np.ndarray | sparse.spmatrix) -> bool:
    """Whether every stored entry of ``A`` is finite (sparse: only stored data)."""
    if sparse.issparse(matrix):
        return bool(np.isfinite(matrix.data).all())
    return bool(np.isfinite(np.asarray(matrix)).all())


@dataclass(frozen=True)
class WeightDiagnostics:
    """Weight-quality diagnostics for one solved weight vector.

    These are the paper's "Weight diagnostics" axis: they describe what a
    calibrator *did to the weights*, independent of how well it hit the targets.

    Attributes:
        effective_sample_size: Kish ESS ``(sum w)^2 / sum(w^2)`` -- the number
            of equal-weight records carrying the same information. Falls as mass
            concentrates on few records.
        ess_ratio: ESS as a fraction of the record count, in ``[0, 1]``; ``1``
            is uniform weights.
        weight_ratio_percentiles: The requested percentiles of the ``w / w0``
            distribution, keyed by percentile. The top percentile is the most
            inflated record's ratio -- the tail the bounded methods cap.
        negative_weight_share: Fraction of records with a negative calibrated
            weight (nonzero only for methods, e.g. GREG, that permit it).
        min_weight: The smallest calibrated weight (negative for GREG in some
            regimes).
        max_weight_ratio: The largest ``w / w0`` ratio realized.
        mass_ratio: Calibrated total over input total ``sum(w) / sum(w0)`` --
            ``1`` under mass conservation.
    """

    effective_sample_size: float
    ess_ratio: float
    weight_ratio_percentiles: dict[float, float]
    negative_weight_share: float
    min_weight: float
    max_weight_ratio: float
    mass_ratio: float

    def as_flat(self) -> dict[str, float]:
        """Flatten to a ``metric -> value`` dict for the long-format artifacts."""
        flat: dict[str, float] = {
            "effective_sample_size": self.effective_sample_size,
            "ess_ratio": self.ess_ratio,
            "negative_weight_share": self.negative_weight_share,
            "min_weight": self.min_weight,
            "max_weight_ratio": self.max_weight_ratio,
            "mass_ratio": self.mass_ratio,
        }
        for pct, value in self.weight_ratio_percentiles.items():
            flat[f"weight_ratio_p{pct:g}"] = value
        return flat


def weight_diagnostics(
    weights: np.ndarray,
    initial_weights: np.ndarray,
    *,
    ratio_percentiles: Sequence[float] = DEFAULT_RATIO_PERCENTILES,
) -> WeightDiagnostics:
    """Compute weight-quality diagnostics for a calibrated weight vector.

    Args:
        weights: The calibrated weights ``w``.
        initial_weights: The input weights ``w0`` (must be strictly positive).
        ratio_percentiles: Percentiles of ``w / w0`` to report.

    Returns:
        A :class:`WeightDiagnostics`.

    Raises:
        ValueError: On a shape mismatch, non-finite weights, or non-positive
            ``w0`` (the ratio is undefined there).

    Notes:
        The effective sample size uses ``|w|`` so a method that produces some
        negative weights still returns a finite, comparable ESS; the
        negative-weight incidence is reported separately rather than folded in.
    """
    w = np.asarray(weights, dtype=np.float64)
    w0 = np.asarray(initial_weights, dtype=np.float64)
    if w.shape != w0.shape:
        raise ValueError(
            f"weights and initial_weights must align, got {w.shape} vs {w0.shape}."
        )
    if not np.isfinite(w).all():
        raise ValueError("weights must be finite.")
    if not np.isfinite(w0).all() or (w0 <= 0.0).any():
        raise ValueError("initial_weights must be finite and strictly positive.")

    abs_w = np.abs(w)
    sum_sq = float(np.square(abs_w).sum())
    ess = float(abs_w.sum() ** 2 / sum_sq) if sum_sq > 0.0 else 0.0
    ratios = w / w0
    percentiles = {float(p): float(np.percentile(ratios, p)) for p in ratio_percentiles}
    negative_share = float(np.mean(w < 0.0))
    return WeightDiagnostics(
        effective_sample_size=ess,
        ess_ratio=ess / w.size if w.size else 0.0,
        weight_ratio_percentiles=percentiles,
        negative_weight_share=negative_share,
        min_weight=float(w.min()),
        max_weight_ratio=float(ratios.max()),
        mass_ratio=float(w.sum() / w0.sum()),
    )


def target_relative_errors(
    problem: CalibrationProblem, weights: np.ndarray
) -> np.ndarray:
    """Signed relative error ``(A@w - b) / scale`` per target.

    The per-target miss the paper's accuracy metrics reduce (median/mean
    absolute relative error, in-fit and held-out). The scale is
    ``max(|b|, 1)`` -- the target's own value, or one unit for a zero-valued
    target -- matching the gradient calibrator's fixed loss scale so the two
    method families are scored on the same denominator.

    Args:
        problem: The calibration problem (supplies ``A`` and ``b``).
        weights: The calibrated weights.

    Returns:
        The length-``n_targets`` signed relative-error vector.
    """
    estimates = problem.estimates(weights)
    scale = np.maximum(np.abs(problem.target), 1.0)
    return (estimates - problem.target) / scale
