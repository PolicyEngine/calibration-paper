# PROGRESS — calibration-method-surface (issue #1)

## State
- Resuming killed predecessors. Worktree branch `method-surface` @ dc14c01.
- Salvaged src is HIGH QUALITY and near-complete: problem/classical/methods/sgd/synthetic/smoke all solid.
- Baseline: BASE-install `uv run --no-sync pytest` = 44 passed (CI path, no torch). GREEN.
- The two failures seen with torch present are (a) test-ordering artifact (torch in sys.modules) — resolves once base env has no torch; (b) ONE over-strict SGD assertion (real, analyzed below).

## Gaps to finish #1 (declared in docstrings/pyproject but NOT implemented)
1. tests/test_r_parity.py — the R-parity Rscript-subprocess test. CORE inherited deliverable. Driver scripts/survey_calibrate.R EXISTS; nothing calls it yet.
2. tests/test_sgd_parity.py — SGD vs populace-calibrate parity (referenced in sgd.py docstring).
3. src/calibration_paper/cli.py — `cal` console entry (pyproject [project.scripts] cal = calibration_paper.cli:main; smoke.py + synthetic.py mention `cal demo`). Currently a BROKEN entry point.
4. pyproject rparity=["rpy2>=3.5"] extra + "when rpy2 installed" docstrings CONTRADICT the inherited Rscript-subprocess decision. Reconcile to R-toolchain-not-rpy2.

## Done
- Full orientation: spec (#1/#2), PLAN.md, README, all src + tests + R driver read.
- Baseline established (44 base-install tests green).

## Next
1. [DONE] test_r_parity.py — Rscript subprocess, skip-clean. 4 tests PASS vs real R survey here.
2. [DONE] test_sgd_parity.py — populace parity, skip-clean. 3 tests PASS (bit-identical to populace.calibrate).
3. [DONE] Fixed over-strict SGD infeasible assertion (test_gradient.py). 6 gradient tests PASS.
4. [DONE] cli/ package (cal methods, cal demo) — fixes broken entry point. 8 CLI tests PASS. Mirrors sparsity-paper cli/ dispatcher.
5. [DONE] Removed dead rparity=[rpy2] extra; reconciled classical.py + test_classical.py docstrings to Rscript-subprocess.
6. IN PROGRESS: verify full green both envs → PR on method-surface.
   - BASE install: 56 passed, 2 skipped (gradient + sgd_parity skip w/o torch; r_parity RAN, R present).
   - TODO: run full suite WITH methods extra (all should pass).

## VERIFIED empirically this session
- R-parity (Rscript+survey present here): raking 6e-15, linear/GREG 1.3e-10, logit 6e-15 g-weight agreement. Ours converge TIGHTER than R's default epsilon 1e-7 — confirms inherited note.
- SGD parity: our sgd_calibrate reproduces populace.calibrate.calibrate BIT-IDENTICALLY (0.0 rel diff) on populace's own compiled system, seed 0. populace defaults epochs=256/lr=0.02/cap=10.0/mass=free == sgd.py DEFAULTS. Faithful.
- populace API: module is `populace` (namespace); populace.calibrate.calibrate(frame, targets, weight_entity=..., seed=...) -> CalibrationResult with .problem.matrix/.target_vector, .initial_weights, .weights. Frame needs person+household schema (EntitySchema(person_entity, group_entities), Weights(values, WeightKind.DESIGN)).

## Decisions (inherited, KEEP)
- Python solvers validated vs R `survey` g-weights to 1e-8 (ours converge tighter than R default 1e-6).
- R-parity test = Rscript SUBPROCESS (driver in repo + Python test that runs it, skips cleanly if Rscript/survey absent). NO rpy2.
- sparsity-paper machinery BY PIN (git URL + SHA), never copied.

## Decisions (this session)
- SGD infeasible-surface test (test_gradient.py:81) assertion `all(|err_i|>1e-2 for contradiction rows)` is WRONG, not the solver. The two contradiction rows ARE the same matrix row → A@w gives ONE common value v. With per-target scale s=|b|, the low target (0.8B, smaller denom) gets more gradient weight, so SGD settles v≈0.8B: low row err≈0.002, high row err≈-0.33, ACROSS ALL SEEDS 0-5 (verified). Correct compromise. Fix the assertion to the true invariant: contradiction NOT jointly satisfied (not both <1e-2) AND summed abs residual bounded below (~0.33) AND finite weights. Solver untouched.

## Open questions
- PLAN.md says "sparsity-paper's frozen-target machinery"; issue #2 says "l0-paper". Same repo renamed (sparsity-paper ← l0-paper per commit 107373f "Point references at the renamed sparsity-paper"). #2 body still says l0-paper#17 / "l0-paper's frozen-target machinery". For #2 I'll pin sparsity-paper (the current name) and note l0-paper as the old name. FLAG for lead.
- Issue #1 text says "pinned rpy2 bridge, OR a vetted Python equivalent with R-parity tests" — predecessor+lead chose the Python-equivalent + Rscript-subprocess path. Consistent with the issue's "or". Keeping.
