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
- [DONE] Base-install verified: 71 passed, 3 skipped (exit 0). Skips = test_gradient (torch), test_sgd_parity (torch), test_population_view (popdgp). configs + frozen_targets_stub tests RUN on base install. torch/popdgp/populace all absent — lazy-import discipline holds for the new popdgp module too.
- [DONE] End-to-end integration verified: a 100-target/5000-record scale point runs through every classical method (all converge, 3e-11..7e-9); an infeasible point reports converged=False across methods. Configs produce real, method-runnable problems.
- [DONE] Self-review of the 3 new modules: no real defects. Tightened one population_view docstring line (coverage delta "exactly zero below cap", not "~0"). Column-set consistency across candidate/holdout/view is airtight (resolved once, threaded to all 4 builders).
- [next] Final cleanup: delete PROGRESS.md, then PR #2 (stacked on method-surface / PR #3). DO NOT merge.

### VERIFICATION SUMMARY (issue #2 branch)
- Base install (dev only): 71 passed, 3 skipped, exit 0.
- popdgp + methods extras: 87 passed, exit 0 (1 benign torch sparse-CSR warning).
- ruff check + format: exit 0.
- popdgp pinned @ 402fb235ff116629fdb50304cc41cb2381a656f0 (main SHA, popdgp#1 merged).

### OPEN QUESTIONS FOR LEAD
1. STACKING: frozen-inputs is based on method-surface (PR #3), not origin/main, because origin/main lacks the synthetic builders #2 needs. PR #2 will show only the 3 new modules + tests IF GitHub bases it on method-surface; if #3 merges first, rebase frozen-inputs onto main and the PR is clean. Confirm the merge order (merge #3, then #2).
2. sparsity-paper#17 still OPEN — frozen-target import stays stubbed (frozen_targets.py, TODO markers). Unblocks the held-out-family split + real Ledger surface + pinned candidate-frame artifact.
3. Synthetic population-view bridge: population_view.py derives views from the problem's continuous matrix rows (a stand-in for the real Ledger/populace view taxonomy). When the frozen surface lands, swap default_views_for_problem for the real views; the delta wiring itself is unchanged.
