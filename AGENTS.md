# Repository instructions

## Scope

This repository contains the Tulu 3 stagewise reasoning-cue source-binding
experiment. Start with `README.md`. The scientific audit trail is in
`docs/research-log/`.

Keep the public repository focused on this experiment. Do not add personal
notes, unrelated projects, agent transcripts, infrastructure logs, secrets, or
machine-specific paths.

## Scientific integrity

- Treat frozen configs and immutable raw records as the experiment authority.
- Produce every observed number, table, figure, and quantitative report fragment
  programmatically from raw records. Never transcribe or repair results by hand.
- Preserve failed instruments and negative controls honestly. Do not relabel a
  failed registered criterion as a pass after inspecting results.
- Regenerate downstream artifacts after upstream changes instead of editing
  derived evidence.
- Keep confirmatory, exploratory, and post-hoc findings clearly distinguished.
- The applicant owns the interpretation and final prose. Agents may implement,
  audit, calculate, and critique, but must not silently change the claim.

## Data safety

`.external` is a symlink to `/Volumes/T7/mats-weekend/frame-stage-binding`.
Raw runs and evidence bundles live on that external SSD and are not tracked by
Git.

- Never recursively delete `.external` or anything beneath it.
- Before deleting any path, check whether it is a symlink and resolve the exact
  target.
- Verify copied run files against their manifests before stopping a GPU pod.
- Preserve raw records; regenerate evidence bundles from them when needed.

## Implementation style

- Prefer the smallest readable implementation that answers the named question.
- Do not turn hypothetical edge cases into architecture.
- Keep functions direct and comments minimal. Comment only non-obvious
  scientific choices, invariants, or numerical subtleties.
- Use durable Python entry points and frozen machine-readable configs for runs.
- Keep generated artifacts out of Git unless they are intentionally published
  figures or compact evidence summaries.
- Preserve unrelated user changes and use one writer in the checkout at a time.

## Verification

Before committing or publishing:

1. Run `python -m unittest discover -s tests`.
2. Verify relevant generated evidence through its scripted analysis path.
3. Check Markdown links and the tracked file list.
4. Confirm no credentials, private paths, or operational logs are tracked.

## Time records

Application bookkeeping remains under `docs/research-log/admin/`.
`time-ledger.json` is authoritative and `time-log.md` is generated. Start and
stop counted sessions only on the applicant's explicit instruction; never edit
durations or totals manually.
