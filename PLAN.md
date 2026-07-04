# Experiment plan

## Question

Modern policy-microsimulation calibration reweights a candidate frame against
thousands of hierarchical administrative targets with stochastic gradient
descent on flexible losses. The survey-statistics standard remains classical
calibration: raking/IPF, GREG/linear calibration (Deville–Särndal), entropy
balancing / exponential tilting, chi-square distance minimization, and their
bounded variants. No published comparison runs these families on a modern
target surface at modern scale. This paper is that comparison — the
"calibrate" operator's validation dossier, symmetric to the fill operator's
(imputation-paper).

## Methods surface

| Key | Family | Notes |
| --- | --- | --- |
| `raking` | IPF / multiplicative | classical margins-only baseline |
| `greg` | Linear calibration (GREG) | closed-form; can go negative — report |
| `chi_square` | Quadratic distance calibration | with and without L/U bounds |
| `entropy` | Entropy balancing / exponential tilting | KL-from-design |
| `logit_bounded` | Bounded calibration (logit link) | hard weight-ratio bounds |
| `sgd` | Gradient-descent calibration (populace-calibrate) | capped relative-error loss, torch |
| `sgd_bounded` | SGD + hard weight-ratio bound | the landmine guard variant |

Classical implementations from established libraries where possible (R
`survey`/`sampling` via a pinned bridge, or a vetted Python port), so the
comparison is against the field's actual tools, not reimplementations.

## Protocol

- Frozen inputs, one operator varies: a pinned candidate frame (populace US
  support artifact, hash-recorded) and a pinned hierarchical target surface
  (Ledger facts: national + state families; the l0-paper's frozen-target
  machinery reused wholesale).
- **Held-out target families** (fit on K families, score on the rest) — the
  generalization axis classical papers rarely report.
- Scale sweep: target-count curves (10^2 → 10^4) for accuracy, runtime, and
  feasibility behavior; deliberately infeasible surfaces to compare failure
  modes (hard-constraint infeasibility vs soft-loss compromise).
- Repeated seeds where stochastic; paired across methods.

## Metrics

- In-fit and held-out target absolute relative error (median/mean).
- Weight diagnostics: effective sample size, max/percentile weight ratios,
  negative-weight incidence (GREG).
- **Reweight fragility** of the calibrated file (closed form, κ=5) and the
  **pre/post-calibration population-view delta** (energy/coverage/tail blocks
  vs survey holdouts — coverage is invariant by construction, so the delta
  isolates what each calibrator does to the weighted joint). Machinery
  imported from popdgp.
- Runtime and memory vs target count.

## Honesty rules

Same as the sibling papers: every number regenerates from committed run
configs; skips and caps recorded; where classical methods win a regime, the
paper says so. No subjective claims; metric-defined statements only.

## Relationship to l0-paper

l0-paper evaluates *record selection* under a fixed gradient calibrator; this
paper evaluates the *calibrator itself* with selection held fixed (full
frame). The l0-paper cites this paper for the calibrator's validity; see the
scope-split issue in that repo.
