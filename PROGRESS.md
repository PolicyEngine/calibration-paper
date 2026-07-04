# PROGRESS — calibration-frozen-inputs (issue #2)

## SESSION 2026-07-04 (new agent)
Issue #2: frozen inputs — scale/infeasibility configs + popdgp population-view delta + sparsity-paper stubs.

### Spec (verified from `gh issue view 2` + PLAN.md)
- (1) UNBLOCKED: scale-sweep configs (target counts 10^2 -> 10^4) + deliberately-infeasible-surface configs, using the synthetic builders already in synthetic.py.
- (2) UNBLOCKED NOW (lead decision): popdgp population-view delta wiring. popdgp#1 CLOSED today; pin popdgp by git URL + current main SHA = 402fb235ff116629fdb50304cc41cb2381a656f0. Wire pre/post-calibration scorecard delta against real API popdgp.views.harness_scorecard.
- (3) BLOCKED: sparsity-paper frozen-target import stays blocked on sparsity-paper#17 (still OPEN). STUB import points behind clear `TODO(sparsity-paper#17)` markers. DO NOT copy that machinery (pin-only rule).

### KEY DECISION — branch base (stacking)
- Lead's literal instruction: `-b frozen-inputs origin/main`. BUT origin/main @ 107373f has ONLY PLAN.md + README.md — none of synthetic.py/methods.py/problem.py (those live on `method-surface`, unmerged PR #3). #2's deliverables REQUIRE those builders.
- RESOLUTION: stacked PR. `frozen-inputs` re-pointed onto `method-surface` @ 2e545d6 (= PR #3 head). This satisfies the intent (separate worktree, separate branch, second PR) and makes the code actually import/run. When PR #3 merges to main, this rebases cleanly onto main. Flagged for lead in final report.

### popdgp API (read from views.py @ pinned SHA)
- `harness_scorecard(candidate: pd.DataFrame, candidate_weight_column: str, views: Iterable[SurveyView], holdouts: Mapping[str, pd.DataFrame], *, k=5, max_points=2048, seed=0) -> list[{"view","metric","value"}]`
- `SurveyView(name, columns: tuple[str,...], weight_column: str, target_columns: tuple[str,...]=())`
- DELTA = harness_scorecard(candidate@w) - harness_scorecard(candidate@w0), per (view, metric). Coverage (prdc) is reweight-invariant so its delta ~ 0 by construction — a correctness check.

### Plan (files to add)
- `src/calibration_paper/configs.py` — declarative sweep config dataclasses (ScaleSweepConfig, InfeasibilityConfig) + builders that emit CalibrationProblem instances from synthetic.py across the 10^2->10^4 grid.
- `src/calibration_paper/population_view.py` — popdgp delta wiring: build a candidate DataFrame from (problem, weights), call harness_scorecard pre/post, return the delta. Lazy popdgp import (behind methods/popdgp extra).
- `src/calibration_paper/frozen_targets.py` — STUB module: the sparsity-paper#17 import points, all raising/deferring behind TODO(sparsity-paper#17). No copied machinery.
- pyproject: add popdgp git pin (extra), matplotlib already in viz.
- tests: test_configs.py, test_population_view.py (skip w/o popdgp), test_frozen_targets_stub.py.

### Progress log (updated every push)
- [DONE] Worktree created + re-pointed onto method-surface. Spec re-checked. popdgp API read.
- [DONE] configs.py — ScaleSweepConfig (10^2->10^4, log grid, 3/decade, 3 seeds), InfeasibilityConfig (record scales 200/2k/20k), SweepPoint, ConfigSuite/DEFAULT_CONFIGS. Uses synthetic.py builders only (unblocked).
- [DONE] frozen_targets.py — STUB module, all 3 entry points raise FrozenTargetsError w/ TODO(sparsity-paper#17); NO machinery copied (pin-only rule). Renamed exc to satisfy N818.
- [DONE] population_view.py — pre/post popdgp delta via REAL popdgp.views.harness_scorecard (pinned @402fb235). Builds candidate/holdout from (A,b,w0); lazy popdgp import.
  - CORRECTNESS FINDING (verified): coverage delta is EXACTLY 0 only below the resample cap (candidate not weight-resampled; PRDC coverage = unweighted support geometry + fixed holdout weights). Above cap, weighted resample adds MC noise. Raised default max_points to 4096 and documented the exactness condition. precision/density are point-weighted so their deltas are nonzero (only coverage is the invariance check); energy delta nonzero when weights move. Empirically confirmed.
- [DONE] pyproject: added `popdgp` extra pinned by git URL @ main SHA 402fb235ff116629fdb50304cc41cb2381a656f0.
- [DONE] Tests: test_configs.py (9), test_frozen_targets_stub.py (6), test_population_view.py (7, skips w/o popdgp). 
- [DONE] VERIFIED: full suite w/ popdgp+methods extras = 87 passed (65 inherited + 22 new), exit 0. ruff check + format CLEAN (exit 0).
- [next] Verify base-install skip-clean (no popdgp/torch) = the 3 popdgp/torch tests skip, everything else passes. Then PR #2.
