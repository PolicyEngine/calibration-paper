"""Frozen-target machinery import points -- STUBBED pending sparsity-paper#17.

The paper's protocol pins a hierarchical administrative target surface (Ledger
facts: national + state families) and reuses sparsity-paper's frozen-target
machinery *wholesale* -- the precalibration freeze, the target registry, the
held-out-family split, and the sweep checkpointing (PLAN.md "Protocol"). That
machinery is being factored into an importable export in sparsity-paper by
``sparsity-paper#17`` (the scope-split refactor). Until that export lands, this
paper cannot depend on it.

**This module is a stub, by design.** The portfolio rule is *pin, never copy*:
we do not vendor sparsity-paper's precalibration/registry code into this repo.
So each integration point below is a thin function that raises
:class:`FrozenTargetsError` with a message pointing at the blocking issue.
When sparsity-paper#17 merges, the plan is:

1. Add ``frozen-targets @ git+https://github.com/PolicyEngine/sparsity-paper.git@<sha>``
   (the exported package/subdirectory) to the ``methods``/data extra in
   ``pyproject.toml``, pinned by commit SHA (mirroring the populace and popdgp
   pins).
2. Replace each ``raise FrozenTargetsError(...)`` body below with a lazy
   ``import`` of the exported symbol and a thin adapter to this repo's
   :class:`~calibration_paper.problem.CalibrationProblem` -- so the scale sweep
   and held-out-family evaluation run on the *real* Ledger surface with no
   change to their call sites.

The scale-sweep and infeasibility configs in :mod:`calibration_paper.configs`
are deliberately independent of this module (they use the synthetic builders),
so the unblocked half of issue #2 is fully functional now; only the frozen
populace/Ledger surface waits here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from calibration_paper.problem import CalibrationProblem

__all__ = [
    "FrozenTargetsError",
    "SPARSITY_PAPER_ISSUE",
    "load_frozen_target_surface",
    "held_out_family_split",
    "pinned_candidate_frame",
]

#: The blocking issue; every stub points the caller here.
SPARSITY_PAPER_ISSUE = "sparsity-paper#17"


class FrozenTargetsError(RuntimeError):
    """Raised when a frozen-target integration point is used before it is wired.

    The frozen-target machinery is imported by pin from sparsity-paper once
    ``sparsity-paper#17`` exports it; until then these entry points are stubs.
    Catch this to fall back to the synthetic builders
    (:mod:`calibration_paper.configs`), which need no frozen surface.
    """


def _unavailable(what: str) -> FrozenTargetsError:
    """Build the standard 'blocked on the export refactor' error for ``what``."""
    return FrozenTargetsError(
        f"{what} depends on sparsity-paper's frozen-target machinery, which is "
        f"not yet an importable export (blocked on {SPARSITY_PAPER_ISSUE}). "
        "Pin the exported package by commit SHA and wire this function once the "
        "refactor lands; until then use the synthetic configs in "
        "calibration_paper.configs. The portfolio rule is pin-not-copy, so this "
        "machinery is never vendored here."
    )


def load_frozen_target_surface(*_args: Any, **_kwargs: Any) -> CalibrationProblem:
    """Load the pinned hierarchical Ledger target surface -- STUB.

    TODO(sparsity-paper#17): replace this body with a lazy import of the
    exported precalibration-freeze / target-registry symbols and an adapter
    returning a :class:`~calibration_paper.problem.CalibrationProblem` over the
    real national + state target families. Do not copy the machinery; depend on
    the pinned export.

    Raises:
        FrozenTargetsError: Always, until the export lands.
    """
    raise _unavailable("Loading the frozen Ledger target surface")


def held_out_family_split(*_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
    """Split the target families into fit / held-out sets -- STUB.

    TODO(sparsity-paper#17): replace with the exported held-out-family split
    (the generalization axis: fit on K families, score on the rest). The
    synthetic builders already tag two families for a stand-in split via
    :meth:`~calibration_paper.problem.CalibrationProblem.subset_targets`; this
    entry point is for the *real* Ledger family taxonomy.

    Raises:
        FrozenTargetsError: Always, until the export lands.
    """
    raise _unavailable("The held-out target-family split")


def pinned_candidate_frame(*_args: Any, **_kwargs: Any) -> Any:
    """Return the pinned candidate frame artifact (hash-recorded) -- STUB.

    TODO(sparsity-paper#17): the paper pins the same candidate-frame artifact
    sparsity-paper uses, by hash. Wire the pinned-artifact loader here (via the
    exported registry) once available, recording the artifact hash alongside the
    run config for regeneration.

    Raises:
        FrozenTargetsError: Always, until the export lands.
    """
    raise _unavailable("The pinned candidate-frame artifact")
