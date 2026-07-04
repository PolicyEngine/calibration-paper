"""calibration-paper: the calibrate operator's validation dossier.

This package benchmarks the gradient-descent survey calibrator (the reweighter
populace uses) against the survey-statistics standard -- classical calibration:
raking/IPF, GREG/linear calibration, entropy balancing, chi-square distance, and
their bounded variants -- on one frozen target surface at modern scale.

The public surface is small and mirrors the sibling papers (imputation-paper,
sparsity-paper):

* :mod:`calibration_paper.problem` -- the :class:`~calibration_paper.problem.CalibrationProblem`
  value type ``(A, b, w0)`` that every calibrator consumes, and the diagnostics
  computed on a solved weight vector.
* :mod:`calibration_paper.methods` -- the calibrator registry: every method under
  one ``calibrate(problem, *, seed) -> CalibrationOutcome`` adapter contract, with
  skip accounting for methods whose packages are absent.
* :mod:`calibration_paper.classical` -- the pure-NumPy classical-calibration
  estimators (Deville-Sarndal 1992), checked against R's ``survey`` package where
  it is installed.
"""

from __future__ import annotations

__version__ = "0.1.0"
