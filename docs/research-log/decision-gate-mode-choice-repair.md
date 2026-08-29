# Decision gate: repaired mode-choice instrument

**Status:** approved by the applicant on 2026-08-29. This document is the
canonical repair specification. The original calibration and its evidence are
immutable exploratory inputs; they are never overwritten or reclassified as a
successful calibration. The MATS timer is stopped. No repair inference is
authorized until the applicant explicitly starts the corresponding phase.

## Question

Does cumulative post-training amplify how strongly a model binds a reasoning
cue to the `REQUEST` source rather than the `WORKING NOTES` source?

The repair does not require Base to prefer `WORKING NOTES`. It tests the
stagewise shift:

```text
psi(stage) = mean[S(notes_cue) - S(request_cue)]
Theta = psi(RLVR) - psi(Base)
```

Negative values indicate greater `REQUEST` dominance. The confirmatory claim is
only that `Theta` is negative and large relative to calibrated headroom and
placebo noise under this released Tulu lineage and shared plain-text interface.

## Why the original calibration does not answer the question

The bare `R`/`A` next-token readout was not a valid instrument at Base, and one
DPO cell frequently skipped the requested sentinel while behaving
semantically correctly. Base also produced prompt-conditioned terminal
repetition loops under long generation. These are measurement failures, not a
falsification of the source-binding hypothesis. Raising the old 1,024-token cap
would extend loops and is not an allowed repair.

## Two-part measurement

1. Score an explicit reasoning-versus-direct mode choice on every item.
2. Generate a short continuation on fixed subsets to test whether that choice
   predicts actual reasoning behavior.

Choice scoring and behavior generation are separate raw streams. Every result
is produced by durable Python code and traced to immutable configuration, raw
records, model/data revisions, and analysis code.

## Variant-selection pilot

Test three frozen single-token choice interfaces:

- `explicit_ra`: explicit `R` versus `A` choice;
- `explicit_12`: explicit `1` versus `2` choice; and
- `response_mode_field`: document-style `response_mode: R/A` field.

All variants use the same problem, source directives, six conditions, and two
block orders. Each mapping says that the reasoning choice must continue with a
`REASONING:` section and then `FINAL:`, while the direct choice must continue
directly with `FINAL:`.

Split the existing 200 locked GSM8K train items by the frozen salted-hash rule:

```text
rank = sha256(str(seed) + NUL + namespace + NUL + item_id), then item_id
seed = 20260821
```

- first 24 under `mode-choice-repair-v1:pilot`: pilot;
- remaining 176: repaired validation;
- first 48 validation items under
  `mode-choice-repair-v1:validation-generation`: validation behavior subset;
- first 64 locked GSM8K test items under
  `mode-choice-repair-v1:confirmation-generation`: confirmation behavior
  subset.

The lock generator, not a person, writes all item lists and hashes.

For every variant, stage, condition, and order, score the choice. Generate at
most 256 new tokens for the four primary/control conditions in the pilot:
`request_cue`, `notes_cue`, `all_reason`, and `all_direct`.

A variant is viable only when every stage passes all of these gates:

- every condition/order cell has median combined candidate mass at least
  `0.05`;
- all-reason versus all-direct choice scores have AUC at least `0.80` under
  each order;
- the paired-bootstrap control-headroom interval is entirely positive;
- on primary conflict cells only, choice score predicts semantic mode with AUC
  at least `0.80`;
- primary generated behavior is at least `90%` semantically classifiable; and
- no generated pilot row has a detected exact terminal repetition loop.

The selector must reject source-effect fields. Among viable variants it chooses
the lexicographic maximum of:

1. worst condition/order-cell median combined candidate mass;
2. minimum stage-level primary semantic AUC; and
3. the frozen priority `explicit_ra`, then `explicit_12`, then
   `response_mode_field` (earlier wins an exact tie).

No viable variant means stop and report instrument failure. A later failure of
the selected variant does not permit falling back to its runner-up.

## Semantic mode and loop definitions

The semantic classifier strips at most one variant-specific leading choice
token. A row is classifiable as:

- `reason`: exactly one line-anchored non-empty `REASONING:` section before
  exactly one line-anchored `FINAL:` with a non-empty same-line payload;
- `direct`: no `REASONING:` marker and only whitespace before exactly one
  line-anchored `FINAL:` with a non-empty same-line payload.

Refusals, multiple/missing headers, empty sections, and all other structures
are unclassifiable.

The loop detector checks token IDs for an exact repeated terminal block with
period 1-128 tokens, at least three copies, and at least 32 repeated tokens.
When several descriptions match, it selects greatest repeated span, then
smallest period, then earliest start.

## Repaired validation

Run only the selected variant on the other 176 locked training items:

- score all six conditions and both orders;
- generate the four primary/control conditions on the fixed 48-item subset;
- generate with a 512-token cap; and
- reapply every pilot gate, allowing an exact-loop rate of at most `0.01` per
  stage.

Only a passing validation receipt may freeze the repaired hypothesis and
authorize confirmation.

## Confirmation

On the untouched complete GSM8K test split:

- score `request_cue` and `notes_cue` under both orders for all 1,319 locked
  items and all four stages;
- generate those conditions on the fixed 64-item behavior subset;
- use the selected variant and unchanged 512-token cap; and
- require, at every stage, primary-cell median candidate mass at least `0.05`,
  primary choice-to-semantic AUC at least `0.80`, primary semantic
  classifiability at least `0.90`, and generated loop rate at most `0.01`.

For item `i`, stage `s`, and order `o`:

```text
d[s,i,o] = S(notes_cue) - S(request_cue)
d[s,i] = mean over orders of d[s,i,o]
psi[s] = mean over items of d[s,i]
theta[i] = d[RLVR,i] - d[Base,i]
Theta = mean over items of theta[i]
```

Confirmatory support requires all of:

- the paired-bootstrap 95% interval for `Theta` lies entirely below zero;
- both order-specific `Theta` point estimates are negative;
- `-Theta` is at least `0.20` times mean Base/RLVR validation control
  headroom; and
- `-Theta` exceeds `2.0` times the maximum absolute endpoint/order validation
  placebo mean.

The validation calibration constants are:

```text
h[s,i,o] = S(all_reason) - S(all_direct)
H = mean h[s,i,o] over s in {Base, RLVR}, items, and orders
N = max over s in {Base, RLVR}, orders of
    abs(mean_i(S(placebo_a) - S(placebo_b)))
```

The headroom criterion is `-Theta >= 0.20 * H`; the placebo criterion is
`-Theta > 2.0 * N`. Order disagreement is handled by requiring both
order-specific `Theta` estimates to be negative, not by adding another noise
term.

Endpoint `psi` signs are not criteria. Adjacent Base-to-SFT-to-DPO-to-RLVR
changes, exact-answer accuracy, cap hits, and format rates are diagnostics and
cannot rescue or block the primary result except where they expose structural
invalidity.

## Execution limits

- Write new artifacts only under `.external/repair-v1/`, using atomic
  temporary-to-final publication and automatic T7 mirroring.
- Preserve all original calibration code, data, receipts, and evidence.
- Additional RunPod spending hard cap: `$12`.
- Prefer the stopped four-RTX Pod; safe fallback is two or one identical RTX
  GPU with stagewise execution.
- Stop a pilot exceeding 45 minutes or any ambiguous/failing gate.
- No BBH, natural-wrapper, Index, cap-doubling, or optional robustness work.
- Keep the separate two-hour executive-summary allowance untouched.

## Required verification

Tests must cover exact prompt multisets and order counterbalancing,
single-token candidate boundaries, deterministic split locks, semantic
classification, loop detection, selector blindness, every gate boundary,
deterministic tie-breaking, immutable manifests, independent recomputation,
and provisional-claim blocking.

The applicant must inspect the central metric code, generated failure audit,
and representative raw behavior before making a claim. This inspection does
not block machine-gated transitions that the applicant has explicitly
preauthorized.
