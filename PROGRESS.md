# PROGRESS — calibration-method-surface (issue #1)

## SESSION 2026-07-04 (new agent, resuming @ f4312eb)
- Spec re-checked: issue #1 + #2 text MATCHES this file's understanding (the "or a vetted Python equivalent with R-parity tests" clause legitimizes the Rscript path). popdgp#1 now CLOSED; popdgp main SHA = 402fb235ff116629fdb50304cc41cb2381a656f0. sparsity-paper#17 still OPEN (stub-only).
- TASK: (a) re-verify BOTH envs green myself (pipefail on any piped pytest); (b) bounded self-review of full diff vs origin/main; (c) delete PROGRESS.md + open PR via gh --body-file. Do NOT merge.
- Progress log below (updated every push):
  - [DONE] BOTH ENVS RE-VERIFIED MYSELF (no pipe, explicit exit code):
    - BASE (uv sync --group dev; torch+populace ABSENT): PYTEST_EXIT=0 -> 56 passed, 2 skipped. Skips = test_gradient.py + test_sgd_parity.py, reason "methods extra (torch) not installed". R present (Rscript + survey TRUE) so r_parity RAN.
    - METHODS (uv sync --extra methods --group dev; torch 2.12.1 + populace-calibrate/frame 0.1.0): PYTEST_EXIT=0 -> 65 passed, 0 failed, 0 skipped, 1 warning. The warning is the torch sparse-CSR beta notice from sgd.py:139 (torch-internal, one-time, numerics unchanged) -- benign.
    - Note: dist `populace-calibrate` installs into the `populace` namespace; import path is `populace.calibrate` (NOT `populace_calibrate`). Parity test uses importorskip("populace.calibrate").
  - [in progress] Bounded self-review of full diff vs origin/main (origin/main @ 107373f has only PLAN.md+README.md; all 24 files are new additions).

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
6. [DONE] Full green BOTH envs:
   - BASE install: 56 passed, 2 skipped (gradient + sgd_parity skip w/o torch; r_parity RAN, R present).
   - METHODS extra: 65 passed, 0 failed.
   - Fixed test_registry lazy-import invariant to run in a SUBPROCESS (was a latent failure: full-suite-with-methods polluted sys.modules so the in-process check failed — real isolation bug in inherited test).
   - Quieted torch sparse-invariant UserWarning via check_sparse_tensor_invariants(False) (numerics unchanged; the "beta state" notice is torch-internal, unavoidable, populace emits it too).
   - ruff check + format CLEAN across all files (removed 2 unused imports in problem.py; noqa B007 on the Newton loop counter whose terminal value is used).
7. NEXT: /cycle review of the full diff, then PR.

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

## Open questions / handoff for issue #2
- RENAME: l0-paper → sparsity-paper (GitHub redirects old name). Issue #2 "l0-paper#17" == sparsity-paper#17 (VERIFIED via gh: OPEN, "Scope split: this paper = sparse selection under a fixed calibrator; the reweighter's own dossier moves to calibration-paper"). Its export refactor (frozen-target machinery as an importable export) has NOT landed. So #2's "import machinery by pin" IS still blocked.
- popdgp#1 (harness package) is CLOSED (extracted). popdgp is a proper package (pyproject/src) but NOT on PyPI → #2's popdgp delta wiring uses a git pin+SHA (consistent w/ issue "pinned"). So THIS half of #2's blocker is resolved.
- sparsity-paper src has precalibration.py + experiments/ — the machinery EXISTS in-repo but is not yet exported per #17.
- #2 SPLIT for the lead to weigh: UNBLOCKED now = scale-sweep configs (target counts 10^2→10^4) + deliberately-infeasible-surface configs, which only need the synthetic builders already in synthetic.py. BLOCKED = "import frozen-target machinery by pin" (needs sparsity-paper#17 export) + popdgp population-view delta wiring (needs a chosen popdgp pin SHA + its extraction API). Recommend: do #2 on a NEW worktree/branch `frozen-inputs` from origin/main; land the unblocked config scaffolding; stub the pinned-import points behind clear TODOs referencing sparsity-paper#17; DO NOT copy sparsity-paper machinery (pin-only rule). FLAG: confirm with lead whether to proceed with the unblocked half now or wait for #17.
- Issue #1 text says "pinned rpy2 bridge, OR a vetted Python equivalent with R-parity tests" — predecessor+lead chose the Python-equivalent + Rscript-subprocess path. Consistent with the issue's "or". Keeping. VERIFIED empirically: g-weights match R survey to 6e-15 (raking/logit), 1.3e-10 (GREG).
