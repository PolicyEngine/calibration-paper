"""The frozen-target integration points are honest stubs (sparsity-paper#17).

Base-install tests: the frozen-target machinery is imported by pin from
sparsity-paper once its export refactor (sparsity-paper#17) lands; until then
every entry point raises :class:`FrozenTargetsError` pointing at the
blocking issue. These tests pin *that* contract, so the stub cannot silently
start returning fake data, and a reviewer sees the boundary is deliberate.
"""

from __future__ import annotations

import pytest

from calibration_paper.frozen_targets import (
    SPARSITY_PAPER_ISSUE,
    FrozenTargetsError,
    held_out_family_split,
    load_frozen_target_surface,
    pinned_candidate_frame,
)


def test_issue_marker_is_sparsity_paper_17() -> None:
    """The stubs point at sparsity-paper#17 (the export-refactor blocker)."""
    assert SPARSITY_PAPER_ISSUE == "sparsity-paper#17"


@pytest.mark.parametrize(
    "entry_point",
    [load_frozen_target_surface, held_out_family_split, pinned_candidate_frame],
)
def test_every_entry_point_raises_unavailable(entry_point) -> None:
    """Each frozen-target entry point is a stub that raises, not fake data."""
    with pytest.raises(FrozenTargetsError) as excinfo:
        entry_point()
    # The error names the blocking issue so the caller knows what unblocks it.
    assert SPARSITY_PAPER_ISSUE in str(excinfo.value)


def test_unavailable_is_a_runtime_error() -> None:
    """``FrozenTargetsError`` is catchable as a standard error type.

    So a sweep can ``except FrozenTargetsError`` (or ``RuntimeError``) and
    fall back to the synthetic configs while the export is pending.
    """
    assert issubclass(FrozenTargetsError, RuntimeError)


def test_error_message_states_the_pin_not_copy_rule() -> None:
    """The stub message records the portfolio rule (pin, never copy)."""
    with pytest.raises(FrozenTargetsError, match="pin-not-copy"):
        load_frozen_target_surface()
