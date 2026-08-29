# Decision gate: When does post-training change which source controls a reasoning cue?

## Stagewise source binding on exact-answer reasoning tasks

**Status: FROZEN v2 scientific design approved 2026-08-21. Implementation is
verified and committed. `time-ledger.json` and generated `time-log.md` are the
sole authority for the current timer state and phase authorization. No model
experiment has started.**

**FINAL extraction validity approved by the applicant on 2026-08-21:**
technical validity means deterministic parser tests, exact scripted re-parsing,
the ITT capability gate, and applicant inspection of the generated random and
failure-category audit sample. Format rate is reported as a diagnostic only;
there is no additional compliance cutoff.

**Preflight status:** the immutable config and all calibration-stage receipts
are complete. The phase-level verifier confirms exact prompt input-token IDs,
prompt bytes, item/content locks, and R/A candidate IDs across Base, SFT, DPO,
and RLVR. The generated metadata diff shows that Tulu adds a chat template and
one unused `<pad>` vocabulary entry; all shared token IDs agree. Every stage's
checkpoint-default EOS already matches `<|end_of_text|>`, and the runner now
explicitly forces `<|end_of_text|>` as both EOS and masked batch-padding token
at every stage. The primary experiment never applies a chat template. Thus the
remaining repository-level tokenizer difference is documented metadata, not
different prompt or stopping plumbing. Exact weight files are cached, but no
model has been instantiated and no inference has run.

The applicant owns the scientific question, claim boundaries, and go/kill
decisions. Codex may implement and audit, with one writer in the
checkout at a time. The canonical implementation handoff is
`implementation-plan-frame-stage-binding.md`.

## Question and claim boundary

> At which cumulative post-training stage does an identical reasoning cue
> switch from being controlled by document/working-note context to being
> controlled by the user request?

This is an operational projection of the applicant's broader attribution
question. Base behavior is the reference state. The experiment measures:

- which source controls the cue at the base checkpoint;
- how that source preference changes after SFT, DPO, and RLVR; and
- whether the stagewise binding change affects exact-answer accuracy.

It does not recover universal percentages of an answer attributable to
pretraining, RLHF, or context. It estimates convention-declared effects along
one released checkpoint lineage. Differences are cumulative checkpoint
transitions, not universal causal effects of training algorithms.

## Primary lineage

Use the documented Tulu 3 8B path:

1. `meta-llama/Llama-3.1-8B`
2. `allenai/Llama-3.1-Tulu-3-8B-SFT`
3. `allenai/Llama-3.1-Tulu-3-8B-DPO`
4. `allenai/Llama-3.1-Tulu-3-8B` (final RLVR checkpoint)

Tulu is chosen for the public sequential post-training stages, not because it
is the newest or strongest model. Qwen is motivation only and receives no new
core run.

The final RLVR stage was trained on GSM8K, MATH, and instruction-following
prompts. Any DPO→RLVR result on GSM8K is therefore domain- and lineage-specific
until the pre-declared BBH robustness arm supports a broader claim.

An optional controlled-pretraining extension uses
`IndexTeam/Index-1.9B-Pure` and `IndexTeam/Index-1.9B-Base` only after the Tulu
result is secure and both Index checkpoints pass the same capability gate.
Index Pure/Base are separately trained controlled counterparts, not a shared
checkpoint split.

## Nearest work and novelty-safe contribution

Prior work already shows that post-training can change frame sensitivity:

- `2601.08842` crosses base/instruct checkpoints, prompt strategy, and an
  injected correction signal.
- `2605.06327` holds tasks fixed while varying evaluation, deployment, and
  neutral frames, including a matched OLMo-3 base/instruct comparison.
- *Decoded but Unused* (ICML 2026 MechInterp workshop) finds moral framing
  decodable in Gemma-3 base but routed into judgment after instruction tuning.
- Role-Sensitive Neurons (ACL Findings 2026) studies role prompts across
  base/instruct models.
- URIAL and Predicting vs Acting establish the broader superficial-alignment
  and predictor-to-agent context.

Therefore the contribution is not “post-training changes what prompts mean.”
The novelty-safe claim is narrower:

> Across a released Base→SFT→DPO→RLVR lineage, locate the cumulative stage at
> which a fixed reasoning cue becomes differently controlled by request versus
> working-note sources, with an exact binding score and exact-answer outcomes.

## Frozen source-binding prompt

Use a byte-identical plain-text serializer at every stage; no native chat
tokens in the primary experiment:

```text
PROBLEM:
{question}

REQUEST:
{request_instruction}

WORKING NOTES:
{notes_instruction}

RESPONSE MODE:
```

Counterbalance the two labeled blocks: `request_first` and `notes_first`.

Competing instructions, present exactly once in each conflict prompt:

- **Reasoning:**
  `Begin with "R". Then show a step-by-step solution under "REASONING:", and end with "FINAL:" followed by only the answer.`
- **Direct answer:**
  `Begin with "A". Then write "FINAL:" followed by only the answer. Do not show reasoning.`

Core conflict cells:

- `request_cue`: reasoning in REQUEST, direct answer in WORKING NOTES.
- `notes_cue`: direct answer in REQUEST, reasoning in WORKING NOTES.

Development-only calibration cells:

- `all_reason`: reasoning in both slots.
- `all_direct`: direct answer in both slots.
- `placebo_request_first` / `placebo_notes_first`: two synonymous direct-only
  instructions swapped between slots; neither asks for reasoning.

Primary binding is read before free generation:

`S = log P(R) - log P(A)`

Use next-token logits if both are single tokens. Otherwise use normalized
conditional sequence log-likelihoods for the exact strings.
Resolve and record exact candidate token IDs and the prompt's
leading-whitespace convention. Record absolute `P(R)+P(A)` before
renormalizing; a tiny combined mass cannot be rescued by high relative AUC.

Greedy generation supplies secondary outcomes: actual R/A marker compliance,
presence of `REASONING:` before `FINAL:`, exact-answer correctness, missing or
invalid markers, refusal/other text, and cap hits.

Exact sentinel compliance and semantic header compliance are separate
diagnostics: beginning directly with `REASONING:` does not count as obeying the
requested initial `R` sentinel.

## Data and phases

### Development/calibration

- GSM8K training split, 200 items sampled with seed `20260821`.
- All four Tulu stages.
- Core conflict cells, all-reason/all-direct controls, placebo swaps, and both
  block orders.
- Prompt/parser changes are allowed only here. One global formatting repair is
  permitted, then the prompt and parser freeze.

### Confirmatory

- Complete official GSM8K test split.
- Core conflict cells and both block orders at all four stages.
- No prompt, parser, threshold, or stage change after opening results.

### Pre-declared robustness

- All BBH logical-deduction tasks after the GSM8K analysis is frozen.
- One natural request wrapper and one natural worked-solution wrapper on a
  fixed 200-item GSM8K test subset, Base and final RLVR only.
- The same fixed subset on final RLVR under its official native chat template,
  bounding the off-distribution cost of the shared plain-text interface.
- Natural wrappers retain the same competing instruction strings and remain
  exploratory.

### Decoding and scoring

- Greedy decoding.
- Stage-invariant explicit stopping and padding: `<|end_of_text|>` for every
  checkpoint. Do not use checkpoint-specific EOS/pad defaults.
- Stage-invariant dynamic KV caching: explicitly `use_cache=true` during
  generation at every checkpoint. The final RLVR checkpoint defaults to false,
  so inheriting checkpoint defaults is forbidden. Standalone scoring forwards
  use no cache because their past-key-values are discarded.
- Stage-invariant BF16 on one GPU per process, with no CPU/disk offload. Use the
  same frozen batch size and exact batch-plan hash at every compared stage.
  Dynamic cache is the only allowed cache implementation.
- Record the actual dtype, device map, attention backend, CUDA/cuDNN/driver and
  GPU type, cache settings, peak allocated/reserved VRAM, token throughput, and
  wall time in every manifest. Different GPU types or runtime contracts across
  core stages invalidate the comparison.
- Initial cap: 1,024 new tokens.
- If more than 1% of development outputs hit the cap, raise it globally to
  2,048 before confirmation.
- Parse the answer after the final `FINAL:` with established GSM8K or BBH
  normalization.
- Missing/invalid final answers are incorrect under intention-to-treat.

## Estimands

For stage `s`:

`psi[s] = mean(S_notes_cue - S_request_cue)`

- Positive `psi`: working-note/document source dominates.
- Negative `psi`: request source dominates.

Primary Base-to-final change:

`Theta = psi[RLVR] - psi[Base]`

Adjacent cumulative-stage changes:

```text
theta_SFT  = psi[SFT]  - psi[Base]
theta_DPO  = psi[DPO]  - psi[SFT]
theta_RLVR = psi[RLVR] - psi[DPO]
```

The adjacent changes must sum to `Theta`. Exact-answer accuracy uses the same
paired contrasts as a secondary downstream outcome.

## Development gates

Before opening confirmatory data:

- All-reason versus all-direct scores separate with AUC ≥ 0.80 at every stage
  and separately under each block order.
- Paired-bootstrap intervals for control headroom exclude zero.
- Every stage × condition × order cell has median absolute `P(R)+P(A)` of at
  least 0.05; pooled medians are diagnostic only.
- R/A scoring and FINAL-answer extraction are technically valid at every stage.
  Operationally, this means deterministic parser tests, programmatic
  re-parsing of every raw completion with exact field agreement, ITT accuracy
  gating, and applicant review of the generated audit sample. There is no
  silently invented format-rate cutoff; format rate remains a reported
  diagnostic.
- Development ITT exact-answer accuracy over the two core conflict conditions
  is between 5% and 95%.
- Placebo swaps measure harmless order/wording noise.
- Cap-hit rate passes the decoding rule above.

If the gates fail, apply one global formatting repair on development data. If
they still fail, stop. Do not silently switch model, task, or readout.

## Confirmatory success and inference

The calibrated-crossover rule is frozen from development data and requires:

- Base `psi` is positive and RLVR `psi` is negative, with 95%
  paired-bootstrap intervals excluding zero; a significant reverse crossover
  is interesting falsification, not confirmatory success;
- `abs(Theta)` is at least 20% of mean Base/final control headroom; and
- `abs(Theta)` exceeds twice the development order/wording noise.

A passed instrument with no crossover falsifies the hypothesis.

Inference:

- At least 10,000 problem-level paired-bootstrap draws.
- All stages, source cells, order variants, and ITT outcomes for a sampled
  problem travel together in one joint bootstrap.
- `Theta` is the sole confirmatory inferential contrast; directional endpoint
  intervals and frozen calibration ratios are decision-rule components rather
  than additional searched hypotheses.
- Test `Theta` and adjacent changes with paired item-level sign flips; Holm
  correct the three adjacent-stage tests. Percentile bootstraps provide
  intervals, not p-values.
- Report block orders separately before averaging.
- BBH, natural wrappers, marker generation, accuracy, and Index are secondary
  or exploratory.
- Never condition correctness on successful marker compliance or extraction.

Freeze calibration noise as follows: source order noise is half the absolute
difference between the two order-specific `psi` values; placebo noise is the
maximum absolute order-specific mean `placebo_a - placebo_b`; the success rule
uses the maximum of these quantities over Base and RLVR.

## Optional Index extension

Run only if:

- the Tulu core analysis and applicant verification are complete;
- both Index checkpoints pass the same development instrument and capability
  gates; and
- at least two applicant-active hours remain before the 16-hour target.

Use GSM8K only and the same source-binding prompts. Interpret results as a
controlled pretraining-data comparison, not a shared-weight causal ablation.

## Applicant verification

Record timestamps and sampled IDs for:

- exact prompts and 20 random development records;
- five random generations per primary stage × source cell and every failure
  category;
- raw R/A logits for at least ten Base and ten RLVR problems;
- independent recomputation of `psi`, `Theta`, and one adjacent-stage change;
- BBH/natural-wrapper claim boundaries; and
- claims–figures–raw-artifact consistency before submission.

The generated audit selection and the applicant's completed checklist must be
bound into a durable review receipt during a phase-specific counted review
session. Calibration/confirmation freeze receipts require this review receipt;
a bare acknowledgement flag is insufficient.

## Time and agent convention

Count applicant-active prompting, decisions, debugging, inspection,
interpretation, verification, and own-voice writing. Detached agent execution
while the applicant is genuinely inactive or doing unrelated work is
informationally logged and uncounted. Any autonomous scientific choice remains
provisional until a counted applicant review.

Record a 30-minute applicant-approved retrospective planning estimate.

Target approximately 15 hours, hard cap 20 hours:

| Applicant-active work | Hours |
|---|---:|
| Retrospective project selection/design | 0.50 |
| Gate and prompt freeze | 1.00 |
| Development inspection and go/no-go | 1.00 |
| Confirmatory stage-result review | 1.50 |
| Raw-output and failure audit | 1.50 |
| Statistics and independent recomputation | 2.50 |
| Robustness and alternative-explanation review | 1.75 |
| Main report in own voice | 4.00 |
| Contingency or Index review | 1.25 |
| **Target total** | **15.00** |

The executive-summary allowance remains separate, up to two hours.

## Reproducibility and storage

### Programmatic evidence invariant

- Run experiments only through version-controlled Python scripts and the
  frozen machine-readable configuration.
- Treat saved raw records as immutable. All parsing, validation, exclusions,
  aggregates, inferential statistics, claim values, tables, and plots must be
  generated by durable auditable Python code.
- Neither the applicant nor any agent may manually enter, copy, remember,
  repair, or calculate an observed result number. Quantitative report material
  is inserted from generated artifacts, never transcribed from a terminal or
  chat.
- Generated artifacts must record hashes of every raw input and the
  configuration plus model, tokenizer, dataset, and analysis-code provenance.
- Independent verification of headline estimands must be a separate scripted
  computation. Raw-record inspection remains mandatory but cannot modify or
  populate quantitative data.
- Any break in this chain is a project-blocking failure and invalidates the
  affected claim until the pipeline is fixed and rerun.

Before model execution, resolve every used Hugging Face `main` reference to an
immutable SHA in a generated locked config. The same script freezes exact
item-ID and question/reference content hashes; no dataset count is typed by
hand. Each stage requires a durable tokenizer/data remote-preflight receipt
before weights can load. Calibration produces an atomic evidence bundle and
deterministic audit sample. Confirmation requires a hashed calibration receipt
produced only after the machine gates pass and the applicant records that they
inspected the generated audit sample. Robustness similarly requires a frozen,
applicant-reviewed confirmation receipt.

- Freeze model/tokenizer revisions, prompt bytes, token IDs, dataset revisions,
  item IDs, seed, decoding config, and normalization code.
- Save raw binding logits, generations, parsed fields, failure categories, and
  derived statistics.
- Use timestamped result directories without overwriting distinct runs.
- Heavy artifacts live under
  `/Volumes/T7/mats-weekend/frame-stage-binding/` through one gitignored
  `.external` symlink. Versioned copies of active scientific receipts live in
  git; provider runtime/cache receipts remain on the T7.

## Implementation and start boundary

Before model execution:

1. Reconcile `AGENTS.md`, `notes.md`, `time-ledger.json`, and its generated
   `time-log.md` view with this selected project and agent-time convention.
2. Record the 30-minute retrospective estimate.
3. Implement and test the frozen configuration, runner, parser, and analyzer.
4. Generate the immutable model/dataset/item-content lock and durable remote
   preflight receipts without loading weights.
5. Complete independent code/statistics review.

Only then may the applicant start the experiment with:

`Start MATS project timer — Tulu stage-binding calibration`
