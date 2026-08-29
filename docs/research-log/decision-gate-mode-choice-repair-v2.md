# Decision gate: mode-choice instrument repair v2

Date: 2026-08-29. Applicant-approved same day (all three gate decisions
selected by the applicant after reviewing pilot diagnostics).

## Motivation

The repair-v1 pilot (24 frozen train items, all three variants, four Tulu
stages) failed its machine gate for every variant. Offline diagnosis against
the preserved raw generations found three causes:

1. **Classifier stricter than the prompt contract.** The v1 classifier
   required a line-anchored `FINAL:` marker and a `REASONING:` header. The
   prompt never required either. 165 of 267 RLVR primary failures were
   fully compliant responses with a mid-line `FINAL:`. Reclassifying all
   2,304 v1 generations with the v2 classifier raises classifiability to
   0.95–1.00 at DPO/RLVR.
2. **Self-contradictory prompt.** "Write exactly R or A as the next token"
   licenses stopping after one token. SFT obeyed literally: 123 primary
   rows are the choice token followed by EOS.
3. **Structural absence of direct-mode behavior.** Direct-mode behavior
   occurs in 3–7 of 576 generations per stage — models essentially always
   reason on GSM8K, even under `all_direct`. The choice→semantic AUC gate
   is therefore uncomputable, not merely failed. Choice *scores* still
   separate controls (response_mode_field control AUC 0.986–1.000 at all
   four stages).

Base cannot follow the response format at all (established in both the
original calibration and repair-v1): classifiability 0.62–0.89 under the v2
classifier, terminal repetition loops at 4–24%. This is a property of the
base model, not of the instrument.

## Changes (all direction-blind)

Neither change touches the request_cue/notes_cue contrast, the scoring
position, the candidate tokens, or the ψ/Θ definitions.

1. **Prompt v2.** "Write exactly R or A as the next token." → "Start your
   response with R or A, then continue in that mode." (analogous wording for
   1/2 and response_mode). The scored position (immediately after the final
   `CHOICE:` / `response_mode:` line) and the candidate tokens are unchanged.
2. **Classifier v2.** Semantic mode is read from the last `FINAL:` marker
   anywhere in the response (case-insensitive, not line-anchored): reason iff
   substantive non-whitespace text precedes it, direct iff not. The answer is
   the same-line content after the marker, else the next non-blank line.
   Refusal detection and terminal-loop detection are unchanged.

## Gate decisions (applicant-selected)

1. **Behavior gates** (classifiability ≥ 0.90, loop-rate limits,
   concordance) apply to the post-training stages (SFT, DPO, RLVR) only.
   Scoring gates (choice mass, control AUC, positive control headroom)
   apply to all four stages. Base non-compliance is reported as a finding.
2. **Concordance ≥ 0.90** (leading choice token matches semantic mode among
   classifiable generations) replaces the structurally uncomputable
   choice→semantic AUC gate.
3. **Control AUC ≥ 0.80 per block order** is required at all four stages.

Variant selection remains blind to source effects: highest worst-cell median
choice mass, then highest minimum control AUC across stages and orders, then
fixed variant priority. No fallback to a runner-up variant.

## Blinding and integrity note

During repair design, pilot source-effect medians (ψ by stage) were
inspected for one variant as a repair-feasibility check. Variant selection
does not consume source-effect information, and the confirmatory inference
runs on the untouched 1,319-item GSM8K test split, so the confirmatory claim
is unaffected. This inspection is recorded here so the provenance is honest.

## Execution plan and limits

- Pilot v2: same 24 frozen pilot items, all three variants, four stages.
- Validation v2: 176 frozen validation items, selected variant only.
- Confirmation v2: 1,319 frozen test items, selected variant, primary
  conditions only.
- Confirmation support requires: item-paired bootstrap 95% CI for Θ entirely
  below zero, Θ negative under both block orders, −Θ ≥ 0.20·H (validation
  control headroom), and −Θ > 2.0·N (validation placebo noise).
- Budget: ≤ $12 additional RunPod spend (≈ $1 pilot + ≈ $3 validation +
  ≈ $5 confirmation expected); pilot wall cap 45 minutes.
- Artifacts under `.external/repair-v2/`. The repair-v1 pilot remains
  immutable failed-instrument evidence under `.external/repair-v1/`.

The v1 instrument, its frozen specification
([decision-gate-mode-choice-repair.md](decision-gate-mode-choice-repair.md)),
and its evidence are retired historical record. The v1 analysis code remains
available at commit 99f25fc and earlier.
