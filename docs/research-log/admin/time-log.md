<!-- GENERATED from time-ledger.json by frame_stage_binding.time_ledger; do not edit. -->
# MATS application time log

Raw timestamps in `time-ledger.json` are authoritative. This Markdown view and all
durations/totals are generated programmatically. Times use Asia/Kolkata unless noted.

## Timer state

- State: **STOPPED**
- Active session: none
- Current official project: Tulu stagewise reasoning-cue source binding

## Totals

| Budget | Used | Target/cap | Remaining |
|---|---:|---:|---:|
| Counted project work | 13:50:13 | target 16:00:00; hard cap 20:00:00 | 02:09:47 to target; 06:09:47 to cap |
| Executive-summary extension | 00:00:00 | cap 02:00:00 | 02:00:00 |
| Uncounted preparation (informational) | 00:00:00 | no cap | — |

Open running time is added only when the session is stopped from a raw timestamp.

## Counted project sessions

| Start timestamp | Stop timestamp | Duration | Activity | Evidence/output | Human verification |
|---|---|---:|---|---|---|
| Retrospective estimate; exact timestamps unavailable | Retrospective estimate; exact timestamps unavailable | 00:30:00 | Selecting and shaping the frame/stage attribution project | Frozen v2 gate and canonical implementation plan | Applicant supplied a 30-minute retrospective estimate on 2026-08-21 |
| 2026-08-21 17:41:17 +05:30 | 2026-08-21 23:11:18 +05:30 | 00:20:00 | Project-specific evidence-pipeline implementation, analysis safeguards, and verification | Core evidence pipeline implemented, verified, and independently audited; applicant-active duration reconciled from applicant report | Applicant explicitly stopped the timer after being away, initially described the departure as immediate, then corrected the estimate to approximately 20 active minutes; the corrected estimate is counted and the remainder is detached |
| 2026-08-22 11:28:20 +05:30 | 2026-08-22 11:46:59 +05:30 | 00:18:39 | Tulu stage-binding calibration | Calibration launch review, GPU sizing, tokenizer/stopping audit, and performance-plan decisions; no model weights loaded | Applicant explicitly stopped the timer before detached optimization work |
| 2026-08-23 02:41:46 +05:30 | 2026-08-23 15:21:36 +05:30 | 00:10:00 | Tulu stage-binding calibration | Calibration coordination only; applicant went away before GPU provisioning and no model weights were loaded; applicant-active duration reconciled from applicant report | Applicant reported being away after starting the timer and estimated approximately 10 active minutes; the corrected estimate is counted and the remainder is excluded |
| 2026-08-28 21:39:57 +05:30 | 2026-08-28 22:48:56 +05:30 | 01:08:59 | Code walkthrough and verification | Applicant completed code review; prompt, analysis, review-receipt, runner guard, and preflight audit finalized through commit 895f1a0 | Applicant explicitly stopped the timer with: Stop MATS timer — code review and final audit complete |
| 2026-08-29 00:55:39 +05:30 | 2026-08-29 01:15:44 +05:30 | 00:20:05 | Tulu stage-binding calibration | No model inference or raw output occurred; Base remained in network-volume virtual-environment import scanning, the infrastructure bottleneck was diagnosed, and the Pod was stopped before migrating GPU/storage topology | Applicant explicitly requested that buffering, migration, and provisioning time be recorded as project-specific infrastructure setup |
| 2026-08-29 01:15:51 +05:30 | 2026-08-29 01:30:14 +05:30 | 00:14:23 | Tulu cross-datacenter GPU and storage setup | Provisioned four RTX PRO 6000 GPUs with a 200 GB local volume, restored the audited four-process launcher, built a fast local Python environment, downloaded all four locked snapshots in parallel, and passed final runtime/cache/plan checks | Applicant directed the four-GPU fallback and explicitly requested infrastructure setup time be tracked separately |
| 2026-08-29 01:30:26 +05:30 | 2026-08-29 03:27:10 +05:30 | 01:56:44 | Tulu stage-binding calibration | All four calibration stages completed and were mirrored to T7; automated analysis then blocked confirmation because marker_mass disagreed with candidate sequence probabilities, so the RTX Pod was stopped without launching confirmation | Applicant preauthorized automatic continuation only on a clear machine-gate pass and required stop-and-wait on any failure or ambiguity |
| 2026-08-29 03:29:20 +05:30 | 2026-08-29 03:32:15 +05:30 | 00:02:55 | Tulu stage-binding calibration | Both float32 consistency-tolerance bugs were repaired and regression-tested; calibration analysis completed, but the frozen machine gate failed because instrument_pass was false and cap_increase_required was true, so confirmation remained blocked | Applicant approved the numerical validator repairs and requested approximately one hour be recorded for post-run bug fixing and analysis |
| Retrospective applicant estimate; exact timestamps unavailable | Retrospective applicant estimate; exact timestamps unavailable | 01:00:00 | Tulu infrastructure setup and GPU migration | Applicant-directed RunPod provisioning, storage migration, four-GPU RTX fallback, environment setup, model caching, and launch preparation performed today | Applicant explicitly requested that one hour be added for today’s infrastructure setup |
| Retrospective applicant estimate; exact timestamps unavailable | Retrospective applicant estimate; exact timestamps unavailable | 01:00:00 | Post-run validator repair and calibration analysis | Applicant-directed diagnosis, correction, regression testing, evidence regeneration, and machine-gate analysis after calibration execution | Applicant explicitly requested approximately one hour be recorded for post-run bug fixes and analysis |
| 2026-08-29 18:28:12 +05:30 | 2026-08-29 18:31:51 +05:30 | 00:03:39 | repair-v1 code review | Applicant reviewed the frozen prompt, next-token scoring and candidate-mass validity, blinded pilot selector, validation and confirmation estimators, independent recomputation, execution budget and failure handling, atomic T7 mirroring, generated evidence, and human-review/report gate; final evidence-chain blockers were repaired and locally verified | Applicant explicitly said: Stop MATS timer — repair-v1 code review complete |
| 2026-08-29 18:50:02 +05:30 | 2026-08-29 19:04:51 +05:30 | 00:14:49 | Tulu mode-choice repair pilot | Repair-v1 four-stage pilot inference completed once and immutable raw outputs were preserved; post-run evidence verification exposed deterministic CSV row-order drift after JSON reload; Pod stopped and no model rerun is required; offline verifier replay and independent audit remain | Applicant explicitly said: Stop MATS timer — pilot completed; offline verifier repair required |
| Retrospective applicant estimate; exact timestamps unavailable | Retrospective applicant estimate; exact timestamps unavailable | 00:30:00 | Repair-v2 instrument implementation and codebase cleanup supervision | Codebase simplified 13.2k->5.1k lines (commit 99f25fc); repair-v2 instrument implemented, tested, committed (9950d01); v2 classifier validated offline on v1 pilot raw data (0.993 classifiable on RLVR primary) | Applicant directed the cleanup and repair-v2 gate decisions (behavior gates post-train only, concordance>=0.90, control AUC all stages) and estimated 30 minutes of active supervision for this session |
| Retrospective applicant estimate; exact timestamps unavailable | Retrospective applicant estimate; exact timestamps unavailable | 03:00:00 | Experiment supervision, infrastructure recovery, validation, confirmation, and result interpretation | Applicant actively directed the repair-v2 pilot, validation and confirmation workflow; reviewed instrument decisions and limitations; supervised GPU execution and T7 recovery; and interpreted the final stagewise results. | Applicant explicitly estimated approximately three hours of active work and instructed that it be added. |
| Retrospective applicant estimate; exact timestamps unavailable | Retrospective applicant estimate; exact timestamps unavailable | 01:00:00 | Post-experiment reconciliation and public artifact preparation | Applicant actively reviewed the completed experiment, reconciled the final interpretation and limitations, directed the paraphrase-placebo follow-up, and supervised preparation and cleanup of the public repository. | Applicant explicitly estimated one additional hour for post-experiment reconciliation work. |
| Retrospective applicant estimate; exact timestamps unavailable | Retrospective applicant estimate; exact timestamps unavailable | 02:00:00 | Project selection and research scoping | Applicant actively explored candidate questions, rejected directions that did not match genuine curiosity or scope, selected the post-training source-binding question, and refined it into the staged Tulu experiment. | Applicant explicitly estimated two hours for determining and scoping the project. |

## Executive-summary extension sessions

| Start timestamp | Stop timestamp | Duration | Activity | Evidence/output | Human verification |
|---|---|---:|---|---|---|

## Uncounted preparation sessions

| Start timestamp | Stop timestamp | Duration | Activity | Evidence/output | Human verification |
|---|---|---:|---|---|---|

## Abandoned/reset project sessions

| Start timestamp | Stop timestamp | Duration | Activity | Evidence/output | Human verification |
|---|---|---:|---|---|---|

## Counting convention for agent-assisted work

Count all applicant-active work: decisions, design, project-specific reading,
prompting/guiding/debugging agents, code inspection, experiment supervision,
analysis, verification, and own-voice writing. Generic environment setup is
uncounted. Detached agent execution and passive runtime are uncounted only while
the applicant is genuinely inactive or doing unrelated work, and are logged below.

## Agent execution sessions (informational, uncounted)

| Start timestamp | Stop timestamp | Duration | Agent | Activity | Output/artifact |
|---|---|---:|---|---|---|
| 2026-08-21 17:27:31 +05:30 | 2026-08-21 17:41:17 +05:30 | 00:13:46 | Codex and read-only reviewers | Detached implementation and audit before the applicant's explicit time-block instruction | Initial runner, analyzer, and reviewer findings |
| 2026-08-21 18:01:17 +05:30 | 2026-08-21 23:11:18 +05:30 | 05:10:01 | Codex and read-only reviewers | Detached implementation, testing, and audits while the applicant was away | Core evidence pipeline implemented, verified, and independently audited; applicant-active duration reconciled from applicant report |
| 2026-08-22 11:46:59 +05:30 | 2026-08-22 12:05:22 +05:30 | 00:18:23 | Codex and read-only reviewers | Detached GPU-efficiency optimization, cache/stopping hardening, relocking, and receipt regeneration | Safe performance patch implemented and independently audited; optimized immutable lock and all calibration preflight receipts regenerated; no model weights loaded |
| 2026-08-23 02:51:46 +05:30 | 2026-08-23 15:21:36 +05:30 | 12:29:50 | No active agent | Applicant away; waiting for RunPod provisioning | No code, model, or experiment work occurred during the excluded interval |
| 2026-08-29 18:31:51 +05:30 | 2026-08-29 18:47:56 +05:30 | 00:16:05 | Codex | Detached repair-v1 GPU-host verification and weight-free preflight | Persistent Pod environment rebuilt; complete Torch-enabled suite passed; pilot preflight passed across Base/SFT/DPO/RLVR; verified preflight-only snapshot atomically published to T7; Pod stopped without model inference |

## Logging protocol

- Use `frame-stage-time start` and `frame-stage-time stop`; never calculate or type durations/totals manually.
- Start only on the applicant's explicit instruction and stop on an explicit stop instruction.
- Do not overlap counted sessions. Stop for breaks.
- Model execution additionally requires the phase-specific active-session label enforced by the runner.
- The retrospective estimate is the sole timestamp-free exception and is explicitly marked in the raw ledger.
