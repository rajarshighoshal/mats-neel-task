# Neel Nanda MATS 12.0 application — local brief

This is the local source of truth for choosing, executing, and writing up the
application project. Keep it concise. Raw application material belongs in the
public Google Doc; experiment evidence belongs beside the code/results.

## Application outcome and constraints

- Deadline: **Friday, September 4, 2026 at 11:59pm PT**. Extensions are
  available until September 11.
- Task: make research progress on an interesting AI-safety problem.
  Interpretability is optional, but the question must fit Neel's current
  interests.
- Deliverables:
  - Application-form answers. These are read first and act as a preliminary
    filter, so concrete communication is a first-class deliverable.
  - A public, anyone-with-the-link Google Doc explaining the work and evidence.
  - An executive summary at the beginning of the Doc: ideally about one page,
    at most three pages and 600 words, with graphs.
  - Code is optional but useful as supporting context.
- Selection signal: teach the reviewer something new through a small,
  self-contained investigation. A carefully explained negative result is
  better than a shaky positive result.

## What the reviewer rewards

1. **Clarity:** state the claim, experiment, metric, models, controls, result,
   and limitations concretely. Specific numbers beat vibes.
2. **Taste and fit:** choose a novel, safety-relevant question aligned with
   current interests, then go deep on one or two insights.
3. **Truth-seeking:** actively search for alternative explanations and ways the
   exciting result could be false.
4. **Technical depth and practicality:** understand the experiment and get
   direct feedback from models/data rather than blindly applying a recipe.
5. **Simplicity and prioritisation:** start with the obvious cheap method; add
   complexity only when it answers a specific unresolved question.
6. **Show your work:** explain why each major decision and experiment was made,
   especially pivots, failures, and sanity checks.

## Current research-interest filter

Strong fits include pragmatic/applied interpretability, model biology and
forensics, reasoning-model behaviour, science of model character,
post-training, alignment training, generalisation, and useful or rigorously
evaluated interpretability methods.

Avoid unless there is an unusually compelling application or twist:

- Grokking, circuit finding for its own sake, SAE hill-climbing/basic SAE
  science, toy models trained on algorithmic tasks, or very theoretical work.
- Generic demonstrations that a safety concept is linearly represented,
  generic head/layer patching, or merely showing that chain of thought affects
  the answer.
- Projects using only old or weak models such as GPT-2, Pythia, or Gemma 2.
- Fancy methods without comparison to simple baselines such as prompting,
  random choice/directions, linear probes, or direct behavioural tests.

Useful default subject models are current open Qwen models, especially dense
Qwen 3.5/3.6 4B or 9B when they are capable enough for the task. Raw PyTorch
hooks or nnsight are preferred over unnecessary machinery.

## Selected application project

The selected question is **whether cumulative post-training amplifies how
strongly a model binds a reasoning cue to the request rather than to
document/working-note context**. The primary object is the public Tulu 3 8B
Base→SFT→DPO→RLVR lineage. The original bare-`R`/`A` calibration failed its
instrument gate and remains immutable exploratory evidence. The approved
repair scores clearer mode choices and validates them against short generated
behavior. No optional model or benchmark extension remains in scope.

**Status (2026-08-29):** the repair-v1 pilot ran once on the GPU pod; raw
outputs are mirrored and hash-verified at `.external/repair-v1/pilot/`. The
machine gate failed for all three variants (Base: repetition loops and
near-zero format compliance; post-train stages: the semantic classifier
rejected mid-line `FINAL:` markers and header-less reasoning that the prompt
never actually forbade, plus SFT emitting the choice token then EOS). An
offline replay with the cleaned code reproduces the verdict exactly. The
codebase was simplified on 2026-08-29 (retired calibration stack, RunPod
orchestration, timer enforcement, and receipt ceremony deleted; ~13.2k → ~5.1k
lines) while preserving all raw evidence, the time ledger, and replay
verdict-identity. Next decision (applicant's): repair the instrument's
prompt/classifier (direction-blind fixes) and rerun the pilot, or report the
honest negative result.

The frozen repair specification is in
[decision-gate-mode-choice-repair.md](../decision-gate-mode-choice-repair.md).
The original gate and implementation plan remain the historical record of the
failed instrument.
Earlier geometry, J-space, arbitration, and CoT-factorization candidates are
parked and must not be revived without an explicit applicant decision.

## Non-overlap with the existing Lie-geometry project

The applicant already has a large, active project on deception under
conversational pressure in `lie-geometry-probes`. It studies geometric detection
and structured control of deceptive commitment using relational residual and
attention structure, charts/atlases, local and source-conditioned control,
Riemannian/optimal-transport metrics, and a manifold-controller program.

The MATS application is deliberately separate:

- Use none of that project's data, code, unpublished results, or artifacts.
- Do not make deception, evaluation awareness, or steering deceptive internal
  states the application target.
- Do not reproduce the chart/atlas, local-control-field,
  source-conditioned-displacement, relational-metric, curvature/holonomy, or
  manifold-controller architecture on a smaller benchmark.
- Generic steering on a clearly unrelated phenomenon remains possible, but is
  not the default. A descriptive or predictive geometric question is preferred
  when it produces an equally informative result with less execution risk.
- Reuse only research-craft lessons: explicit baselines, frozen decision rules,
  retained failures, raw-data checks, and honest claim boundaries.

## Non-negotiable human verification

Agents may help with coding, analysis, graph production, and critical review,
but the applicant owns experimental design, interpretation, and final prose.

For every load-bearing result:

- Read raw prompts, completions/transcripts, and randomly selected examples.
- Read the code that generated the result and graph.
- Recompute at least one headline number through the independent scripted path;
  hand inspection is reserved for raw records and never creates a number.
- Run a cheap baseline or negative control capable of falsifying the claim.
- Check the dumbest plausible failure mode: leakage, metric mismatch, an
  incapable model, trivial heuristics, or a grader/model gaming the setup.
- Record exactly what was personally verified. Never treat an agent's report
  that an experiment worked as evidence that it worked.

Final application-form answers and executive-summary prose must be written in
the applicant's own voice. LLMs can critique, interrogate, and help organise a
human draft; raw LLM prose should not be submitted.

## Time policy

Use `time-ledger.json` as the raw timestamp authority and
[time-log.md](time-log.md) as its programmatically generated readable view.
Use `frame-stage-time` for every start/stop; never hand-calculate durations.

### Counted project budget

- Target: **about 16 hours**.
- Hard cap: **20 hours**.
- Count all applicant-active work toward the chosen project, including
  planning/thinking, project-specific reading, prompting/guiding/debugging
  agents, inspecting code and outputs, experiments, data analysis,
  verification, and the main Google Doc write-up.
- Detached agent execution and passive model runtime are uncounted only while
  the applicant is genuinely inactive or doing unrelated work. Log them
  informationally. Any autonomous scientific choice remains provisional until
  reviewed during counted applicant time.
- Bias toward at most about five hours of project-specific papers/tutorials;
  get feedback from data early.

### Additional executive-summary allowance

- Up to **2 additional hours** may be used for the executive summary so it is
  not rushed.
- During this extension, do not edit the rest of the report or write new
  experiment code. New graphs from existing data are allowed when they improve
  presentation.

### Not counted

- General learning/preparation done before deciding on a project.
- Generic technical setup needed by most projects, such as provisioning a GPU.
- Breaks and passive waiting while doing something else.
- Writing the MATS application-form answers (the detailed time-limit section
  explicitly excludes these, even though they should be prioritised).

Project-relevant novelty searches, papers, or planning after a candidate has
been chosen should conservatively be counted. If the project is genuinely
abandoned and the previous work is not useful for the replacement, the project
timer may be reset; preserve the abandoned log rather than deleting it.

## Research loop

1. **Explore:** maximise information gained per unit time. Inspect prompts,
   outputs, and data early. Ask every 60–90 minutes whether anything was
   learned and whether the direction remains fruitful.
2. **Understand:** maintain explicit hypotheses and alternate between a
   discriminating experiment, the result, and skeptical analysis.
3. **Distil:** organise the report around one or two supported insights, not a
   chronological experiment dump.

## Executive-summary skeleton

- What problem was tested, and why is it interesting for AI safety?
- What are the one or two high-level takeaways?
- For each takeaway: one short paragraph plus one graph explaining the key
  experiment, result, and why it supports the claim.
- What are the strongest alternative explanations, negative results, and
  limitations?
- What raw data and code were personally checked?
- If data quality or model/LLM judgement is load-bearing, show randomly chosen
  qualitative examples immediately after the summary.

## Archived sketch: premise-permutation orbit geometry

**Status:** unselected sketch retained for reference. It is not the primary
candidate and agents must not implement it without a new explicit decision.

**Question:** For deductive problems with logically exchangeable premises,
does a mid-to-late layer band form a representation that converges across
premise permutations, and does failure to converge predict answer flips?

- Proposed model: Qwen 3.5 4B, with 9B only as a replication if time permits.
- Data: templated transitive-inference chains; eight premise permutations per
  problem; a matched order-sensitive control task.
- Capture: final-prompt-token residual stream using simple hooks.
- Main measurement: normalised within-problem orbit diameter by layer.
- Behavioural measurement: answer flip rate under premise permutation.
- Required baselines: matched order-sensitive task, shuffled-problem null, raw
  versus normalised distance, and behavioural base rate.
- Earliest feasibility check: verify that the model solves the task and that
  enough answers flip to test the proposed relationship.
- Pre-specify a decisive effect-size/AUC threshold before inspecting the full
  results.
- Kill criterion: no layer separates exchangeable invariance from the
  order-sensitive control at a meaningful pre-specified effect size, and orbit
  diameter does not predict flips above the shuffled null.
- Main fit risk: this may look like generic representation geometry on a toy
  algorithmic task. It needs a credible safety-relevant use case or a stronger
  link to current interests before selection.

## Archived sketch: one-hop-per-layer causal propagation

**Status:** unselected sketch retained for reference. It is not the default
pivot and agents must not implement it without a new explicit decision.

**Question:** In in-context graph reasoning, does the causal effect of patching
a node's edge-list tokens reach other node representations one graph hop per
layer, independent of sequence distance?

- Data: shuffled serialized paths/trees, including disconnected components.
- Intervention: patch edge-list tokens at one node and measure when effects
  arrive at other node representations.
- Pre-specify the arrival-layer estimator using a threshold relative to the
  random-token noise floor, with per-layer bootstrap uncertainty.
- Required baselines: sequence-distance-only prediction, disconnected-component
  negative control, and random-token patch floor.
- Kill criterion: arrival layer is flat in graph distance or sequence distance
  subsumes graph distance.
- Main fit risk: this is close to circuit/algorithm analysis on synthetic toy
  tasks, an area the application explicitly deprioritises. It should not be the
  default pivot without a concrete pragmatic application.

## Decision gate before starting the official timer

Do not start counted research until all of these are written down:

- The safety-relevant phenomenon or downstream use case.
- Why the answer is not obvious without evidence.
- The simplest first experiment and capability sanity check.
- The decisive metric, baseline, negative control, and kill criterion.
- A scoped plan that can yield an honest result inside 16 hours.

Current timer state is generated from `time-ledger.json`; consult
[time-log.md](time-log.md). Selected project: Tulu stagewise reasoning-cue
source binding. Model calibration has not started.
