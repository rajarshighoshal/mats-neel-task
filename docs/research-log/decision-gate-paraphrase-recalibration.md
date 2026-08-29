# Decision gate: paraphrase-placebo recalibration

Date: 2026-08-29. Registered and committed BEFORE the recalibration run.
Status: **post-hoc recalibration — suggestive, not confirmatory.** The frozen
v2 confirmation verdict (placebo rule failed) stands in the record; this
entry estimates a sounder noise constant and re-applies the magnitude
criterion to the already-measured Theta. It does not replace the
pre-registered result.

## Motivation

The v2 placebo rule failed: -Theta = 1.654 vs required > 2N = 3.628. Decomposition
of the validation placebo cells shows the original placebo is structurally
confounded, not merely noisy. With source-slot weights w_request, w_notes and
directive suppressor strengths d_1, d_2:

    S(placebo_a) - S(placebo_b) = (w_request - w_notes) x (d_1 - d_2)
    psi                          = (w_request - w_notes) x (d_reason - d_direct)

The placebo swap shares the (w_request - w_notes) factor with the estimand
itself: a model with strong REQUEST-binding necessarily shows a large swap
effect. The control measures the signal through different wording, so the
criterion asked whether the effect exceeds a scaled copy of itself. The
dominant cell (RLVR x notes_first, -1.814) is the same stage x order cell
that carries the largest conflict effect, consistent with this reading.

## Design

Paraphrase placebo: source assignment is FIXED within each pair and only the
wording of the reasoning directive varies. The score difference between
canonical and paraphrase wording is pure wording sensitivity, with no
source-swap factor.

- Canonical reasoning directive (unchanged, frozen):
  "Solve the problem with explicit step-by-step reasoning before the final answer."
- Paraphrase reasoning directive (new, frozen):
  "Think the problem through one step at a time before giving your final answer."

Conditions (4, all scoring-only):

| condition       | REQUEST slot                     | NOTES slot                     |
|-----------------|----------------------------------|--------------------------------|
| request_cue     | canonical reasoning              | direct                         |
| notes_cue       | direct                           | canonical reasoning            |
| para_request_b  | paraphrase reasoning             | direct                         |
| para_notes_b    | direct                           | paraphrase reasoning           |

- Items: 352 fresh GSM8K train items. The original calibration lock ranked
  only the first 200 train items (source limit: 200), all consumed by
  pilot+validation. Fresh items are drawn by ranking the FULL train split
  under the recalibration namespace (seed 20260821, namespace
  "mode-choice-repair-v2:recalibration"), excluding the consumed 200, and
  taking the first 352. No item overlaps any earlier phase.
- Stages: base and rlvr only (the Theta endpoints, matching the original
  placebo constant definition).
- Variant: explicit_12 (the selected instrument; recorded in the frozen
  recalibration config, which is this phase's registration).
- Block orders: both. Scoring only; no generations.

## Aggregation and criterion (pre-specified)

For each (assignment, order) cell, wording sensitivity =
|mean_items S(canonical) - S(para)|:

- N_max = max over the 4 cells (primary; direct analogue of the original
  max-over-cells aggregation)
- N_mean, N_median also reported for transparency.

Criterion: -Theta > 2 x N_max, with Theta read from the confirmation
evidence bundle (estimands.theta), never entered by hand.

Also reported (free replication): psi on the fresh items per stage,
mean S(notes_cue) - S(request_cue), a direction check on untouched items.

## Authorization and cost

The recalibration config (configs/frame_stage_binding_recalibration.locked.json)
is committed before the run and is the phase's registration; the runner does
not consume the pilot receipt chain for this phase. Cost cap: $1.50. Expected
~5 minutes on one GPU (~$0.20).

## Interpretation rules

- If -Theta > 2 x N_max: the magnitude criterion passes under the sound
  placebo. Reported as "passes under paraphrase-placebo recalibration
  (post-hoc, suggestive)".
- If it fails under N_max but passes under N_mean/N_median: reported as
  borderline and aggregation-dependent.
- If it fails under all: the magnitude claim is not supported under any
  sound noise estimate; the direction claim stands alone.
