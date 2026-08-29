from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import random
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .repair_core import (
    CONTROL_CONDITIONS,
    ORDERS,
    PRIMARY_CONDITIONS,
    RECALIBRATION_CONDITIONS,
    STAGES,
    VARIANT_ORDER,
    build_prompt,
    classify_semantic_mode,
    detect_terminal_loop,
)
from .repair_recompute import (
    BOOTSTRAP_SEED_OFFSET,
    recompute_confirmation,
    recompute_validation_constants,
)
from .repair_run import (
    _code_hashes,
    _receipt_hash,
    load_repair_config,
    verify_selector_receipt,
    verify_validation_receipt,
)
from .utils import _canonical_sha256, _file_sha256


SCORE_FIELDS = {
    "phase",
    "stage",
    "variant",
    "item_id",
    "condition",
    "order",
    "prompt",
    "prompt_sha256",
    "prompt_input_token_ids",
    "reason_candidate_token_ids",
    "direct_candidate_token_ids",
    "reason_logit",
    "direct_logit",
    "reason_logprob",
    "direct_logprob",
    "choice_score",
    "choice_mass",
    "predicted_mode",
    "model_id",
    "model_revision",
    "question",
    "reference",
}
GENERATION_FIELDS = {
    "phase",
    "stage",
    "variant",
    "item_id",
    "condition",
    "order",
    "prompt",
    "prompt_sha256",
    "response",
    "generated_token_ids",
    "response_token_count",
    "finish_reason",
    "cap_hit",
    "semantic_mode",
    "classifiable",
    "semantic_failure_categories",
    "terminal_loop",
    "model_id",
    "model_revision",
    "question",
    "reference",
}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSON at {path}:{line_number}")
            rows.append(value)
    return rows


def _expected_variants(
    config: dict[str, Any], phase: str, selector: dict[str, Any] | None
) -> list[str]:
    if phase == "pilot":
        if selector is not None:
            raise ValueError("Pilot analysis must not consume its selector")
        return list(VARIANT_ORDER)
    if phase == "recalibration":
        if selector is not None:
            raise ValueError("Recalibration analysis must not consume a selector")
        return [config["phases"]["recalibration"]["variant"]]
    if selector is None:
        raise ValueError(f"{phase} analysis requires a selector receipt")
    return [verify_selector_receipt(selector, config)]


def _validate_score(
    record: dict[str, Any],
    config: dict[str, Any],
    phase: str,
    stage: str,
    variants: list[str],
    candidate_ids: dict[str, dict[str, list[int]]],
) -> None:
    missing = SCORE_FIELDS - set(record)
    if missing:
        raise ValueError(f"Score record lacks fields: {sorted(missing)}")
    if (
        record["phase"] != phase
        or record["stage"] != stage
        or record["variant"] not in variants
    ):
        raise ValueError("Score record disagrees with its run identity")
    model = config["models"][stage]
    if (
        record["model_id"] != model["model_id"]
        or record["model_revision"] != model["revision"]
    ):
        raise ValueError("Score record disagrees with its locked model")
    variant = record["variant"]
    if (
        record["reason_candidate_token_ids"] != candidate_ids[variant]["reason"]
        or record["direct_candidate_token_ids"] != candidate_ids[variant]["direct"]
    ):
        raise ValueError("Score record candidate IDs disagree with its manifest")
    if not all(isinstance(value, int) for value in record["prompt_input_token_ids"]):
        raise ValueError("Score prompt token IDs must be integers")
    rebuilt = build_prompt(
        record["question"], record["condition"], record["order"], variant
    )
    if record["prompt"] != rebuilt:
        raise ValueError("Score prompt differs from the frozen prompt builder")
    if record["prompt_sha256"] != hashlib.sha256(rebuilt.encode()).hexdigest():
        raise ValueError("Score prompt hash mismatch")
    reason_logit = _finite(record["reason_logit"], "reason_logit")
    direct_logit = _finite(record["direct_logit"], "direct_logit")
    reason_logprob = _finite(record["reason_logprob"], "reason_logprob")
    direct_logprob = _finite(record["direct_logprob"], "direct_logprob")
    score = _finite(record["choice_score"], "choice_score")
    mass = _finite(record["choice_mass"], "choice_mass")
    if not math.isclose(score, reason_logit - direct_logit, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("choice_score disagrees with raw logits")
    if not math.isclose(
        score, reason_logprob - direct_logprob, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError("choice_score disagrees with log probabilities")
    expected_mass = math.exp(reason_logprob) + math.exp(direct_logprob)
    if not math.isclose(mass, expected_mass, rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError("choice_mass disagrees with log probabilities")
    if not 0.0 <= mass <= 1.0 + 1e-6:
        raise ValueError("choice_mass is outside probability bounds")
    expected_mode = "reason" if score > 0 else "direct" if score < 0 else "tie"
    if record["predicted_mode"] != expected_mode:
        raise ValueError("predicted_mode disagrees with choice_score")


def _validate_generation(
    record: dict[str, Any],
    config: dict[str, Any],
    phase: str,
    stage: str,
    variants: list[str],
) -> None:
    missing = GENERATION_FIELDS - set(record)
    if missing:
        raise ValueError(f"Generation record lacks fields: {sorted(missing)}")
    if (
        record["phase"] != phase
        or record["stage"] != stage
        or record["variant"] not in variants
    ):
        raise ValueError("Generation record disagrees with its run identity")
    model = config["models"][stage]
    if (
        record["model_id"] != model["model_id"]
        or record["model_revision"] != model["revision"]
    ):
        raise ValueError("Generation record disagrees with its locked model")
    rebuilt = build_prompt(
        record["question"],
        record["condition"],
        record["order"],
        record["variant"],
    )
    if record["prompt"] != rebuilt:
        raise ValueError("Generation prompt differs from the frozen prompt builder")
    if record["prompt_sha256"] != hashlib.sha256(rebuilt.encode()).hexdigest():
        raise ValueError("Generation prompt hash mismatch")
    token_ids = record["generated_token_ids"]
    if not isinstance(token_ids, list) or not all(
        isinstance(value, int) for value in token_ids
    ):
        raise ValueError("Generated token IDs must be integers")
    if record["response_token_count"] != len(token_ids):
        raise ValueError("response_token_count disagrees with generated tokens")
    if record["finish_reason"] not in {"eos", "length", "unknown"}:
        raise ValueError("Unknown finish_reason")
    if record["cap_hit"] is not (record["finish_reason"] == "length"):
        raise ValueError("cap_hit disagrees with finish_reason")
    semantic = classify_semantic_mode(record["response"], record["variant"])
    for key, value in semantic.items():
        if record.get(key) != value:
            raise ValueError(f"Stored semantic field {key} disagrees with response")
    loop_ids = list(token_ids)
    eos_ids = set(config["generation"]["eos_token_ids"])
    while loop_ids and loop_ids[-1] in eos_ids:
        loop_ids.pop()
    expected_loop = detect_terminal_loop(loop_ids)
    if record["terminal_loop"] != expected_loop:
        raise ValueError("Stored loop diagnostic disagrees with generated tokens")


def _manifest_provenance(
    run_dir: Path,
    config: dict[str, Any],
    phase: str,
    expected_variants: list[str],
    selector: dict[str, Any] | None,
    validation: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    score_path = run_dir / "scores.jsonl"
    # Prefer the migrated v2 semantic fields when present; the original
    # generations.jsonl stays untouched as the raw record.
    generation_path = run_dir / "generations-v2.jsonl"
    if not generation_path.exists():
        generation_path = run_dir / "generations.jsonl"
    manifest_path = run_dir / "manifest.json"
    if not all(path.is_file() for path in (score_path, generation_path, manifest_path)):
        raise FileNotFoundError(f"Incomplete repair run directory: {run_dir}")
    manifest = json.loads(manifest_path.read_text())
    stage = manifest.get("stage")
    if stage not in STAGES or manifest.get("phase") != phase:
        raise ValueError("Run manifest has the wrong phase or stage")
    scores_sha256 = _file_sha256(score_path)
    generations_sha256 = _file_sha256(generation_path)
    checks = {
        "schema": manifest.get("schema_version") == 1,
        "study": manifest.get("study") == config["study"],
        "config": manifest.get("config_sha256") == _canonical_sha256(config),
        "embedded_config": _canonical_sha256(manifest.get("config"))
        == _canonical_sha256(config),
        "variants": manifest.get("variants") == expected_variants,
        "model_id": manifest.get("model_id") == config["models"][stage]["model_id"],
        "model_revision": manifest.get("model_revision")
        == config["models"][stage]["revision"],
        "dataset": manifest.get("dataset_revisions")
        == {config["dataset"]["dataset_id"]: config["dataset"]["revision"]},
        "scores_hash": manifest.get("scores_sha256") == scores_sha256,
        # The manifest hashes the original generations.jsonl; when reading
        # the migrated generations-v2.jsonl, verify the original still
        # matches its manifest hash (raw immutability) instead.
        "generations_hash": (
            manifest.get("generations_sha256") == generations_sha256
            if generation_path.name == "generations.jsonl"
            else manifest.get("generations_sha256")
            == _file_sha256(run_dir / "generations.jsonl")
        ),
        "selector": manifest.get("selector_receipt_sha256") is None
        or (
            isinstance(manifest.get("selector_receipt_sha256"), str)
            and len(manifest["selector_receipt_sha256"]) == 64
        ),
        "validation": manifest.get("validation_receipt_sha256") is None
        or (
            isinstance(manifest.get("validation_receipt_sha256"), str)
            and len(manifest["validation_receipt_sha256"]) == 64
        ),
        # code_sha256 is historical provenance of the run that produced this
        # data; current analysis code is allowed to differ from it.
        "code": isinstance(manifest.get("code_sha256"), dict)
        and bool(manifest["code_sha256"])
        and all(
            isinstance(value, str) and len(value) == 64
            for value in manifest["code_sha256"].values()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Repair manifest failed: {checks}")
    candidate_ids = manifest.get("candidate_ids")
    if not isinstance(candidate_ids, dict) or set(candidate_ids) != set(
        expected_variants
    ):
        raise ValueError("Manifest candidate IDs are incomplete")
    for variant in expected_variants:
        reason = candidate_ids[variant].get("reason")
        direct = candidate_ids[variant].get("direct")
        if (
            not isinstance(reason, list)
            or not isinstance(direct, list)
            or len(reason) != 1
            or len(direct) != 1
            or reason == direct
        ):
            raise ValueError("Manifest choices are not distinct single tokens")
    required_manifest = {
        "code_sha256",
        "tokenizer_sha256",
        "enforced_tokenizer_sha256",
        "stopping",
        "generation",
        "software_versions",
        "runtime_environment",
        "performance",
        "score_batch_plan_sha256",
        "generation_batch_plan_sha256",
        "prompt_input_token_ids_sha256",
        "score_prompt_hashes_sha256",
        "generation_prompt_hashes_sha256",
    }
    if any(not manifest.get(key) for key in required_manifest):
        raise ValueError("Repair manifest lacks required provenance")
    if (
        manifest["generation"].get("max_new_tokens")
        != config["phases"][phase]["max_new_tokens"]
    ):
        raise ValueError("Run generation cap disagrees with repair config")
    for key, value in config["generation"].items():
        if manifest["generation"].get(key) != value:
            raise ValueError(f"Run generation setting changed: {key}")

    scores = _read_jsonl(score_path)
    generations = _read_jsonl(generation_path)
    if manifest.get("score_record_count") != len(scores) or manifest.get(
        "generation_record_count"
    ) != len(generations):
        raise ValueError("Manifest record counts disagree with raw streams")
    for record in scores:
        _validate_score(record, config, phase, stage, expected_variants, candidate_ids)
    for record in generations:
        _validate_generation(record, config, phase, stage, expected_variants)
    return (
        scores,
        generations,
        {
            "run_dir": run_dir.name,
            "stage": stage,
            "scores_sha256": scores_sha256,
            "generations_sha256": generations_sha256,
            "manifest_sha256": _file_sha256(manifest_path),
            "candidate_ids": candidate_ids,
            "prompt_input_token_ids_sha256": manifest["prompt_input_token_ids_sha256"],
            "score_prompt_hashes_sha256": manifest["score_prompt_hashes_sha256"],
            "generation_prompt_hashes_sha256": manifest[
                "generation_prompt_hashes_sha256"
            ],
            "code_sha256": manifest["code_sha256"],
            "score_batch_plan_sha256": manifest["score_batch_plan_sha256"],
            "generation_batch_plan_sha256": manifest["generation_batch_plan_sha256"],
            "software_versions": manifest["software_versions"],
            "runtime_contract": {
                key: manifest["runtime_environment"].get(key)
                for key in [
                    "model_dtype",
                    "parameter_devices",
                    "parameter_dtypes",
                    "attention_implementation",
                    "torch_cuda_version",
                    "nvidia_driver_version",
                    "cudnn_version",
                ]
            }
            | {
                "gpu": {
                    key: manifest["runtime_environment"].get("gpu", {}).get(key)
                    for key in ["name", "total_memory_bytes", "compute_capability"]
                }
            },
            "effective_stopping": {
                "configured": manifest["stopping"].get("configured"),
                "effective_model_config_use_cache": manifest["stopping"].get(
                    "effective_model_config_use_cache"
                ),
            },
            "generation_contract": manifest["generation"],
        },
    )


def _validate_cartesian(
    scores: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    config: dict[str, Any],
    phase: str,
    variants: list[str],
) -> None:
    phase_config = config["phases"][phase]
    score_expected = {
        (stage, variant, item_id, condition, order)
        for stage in STAGES
        for variant in variants
        for item_id in phase_config["item_lock"]["item_ids"]
        for condition in phase_config["conditions"]
        for order in ORDERS
    }
    generation_expected = {
        (stage, variant, item_id, condition, order)
        for stage in STAGES
        for variant in variants
        for item_id in phase_config["generation_item_lock"]["item_ids"]
        for condition in phase_config["generation_conditions"]
        for order in ORDERS
    }

    def keys(
        rows: list[dict[str, Any]], name: str
    ) -> set[tuple[str, str, str, str, str]]:
        values: set[tuple[str, str, str, str, str]] = set()
        for row in rows:
            key = (
                row["stage"],
                row["variant"],
                row["item_id"],
                row["condition"],
                row["order"],
            )
            if key in values:
                raise ValueError(f"Duplicate {name} cell: {key}")
            values.add(key)
        return values

    score_actual = keys(scores, "score")
    generation_actual = keys(generations, "generation")
    if score_actual != score_expected:
        raise ValueError(
            "Score stream is not Cartesian-complete: "
            f"missing={len(score_expected - score_actual)}, "
            f"extra={len(score_actual - score_expected)}"
        )
    if generation_actual != generation_expected:
        raise ValueError(
            "Generation stream is not Cartesian-complete: "
            f"missing={len(generation_expected - generation_actual)}, "
            f"extra={len(generation_actual - generation_expected)}"
        )
    score_by_key = {
        (
            row["stage"],
            row["variant"],
            row["item_id"],
            row["condition"],
            row["order"],
        ): row
        for row in scores
    }
    for row in generations:
        key = (
            row["stage"],
            row["variant"],
            row["item_id"],
            row["condition"],
            row["order"],
        )
        score = score_by_key[key]
        if (row["prompt"], row["prompt_sha256"]) != (
            score["prompt"],
            score["prompt_sha256"],
        ):
            raise ValueError("Score and generation prompts disagree")

    fingerprints: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    tokenizations: dict[tuple[str, str, str, str], set[tuple[int, ...]]] = defaultdict(
        set
    )
    for row in scores:
        key = (row["variant"], row["item_id"], row["condition"], row["order"])
        fingerprints[key].add(row["prompt_sha256"])
        tokenizations[key].add(tuple(row["prompt_input_token_ids"]))
    if any(len(values) != 1 for values in fingerprints.values()):
        raise ValueError("Prompt bytes differ across stages")
    if any(len(values) != 1 for values in tokenizations.values()):
        raise ValueError("Prompt token IDs differ across stages")


def load_run_bundle(
    run_dirs: list[Path],
    config: dict[str, Any],
    phase: str,
    selector_receipt: dict[str, Any] | None = None,
    validation_receipt: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if phase not in config["phases"]:
        raise ValueError(f"Unknown repair phase: {phase}")
    variants = _expected_variants(config, phase, selector_receipt)
    if phase == "confirmation":
        if selector_receipt is None or validation_receipt is None:
            raise ValueError("Confirmation requires selector and validation receipts")
        verify_validation_receipt(
            validation_receipt,
            config,
            selector_receipt,
            allow_failed_validation=True,
        )
    elif validation_receipt is not None:
        raise ValueError("Only confirmation consumes a validation receipt")
    all_scores: list[dict[str, Any]] = []
    all_generations: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    seen_stages: set[str] = set()
    for run_dir in run_dirs:
        scores, generations, provenance = _manifest_provenance(
            run_dir,
            config,
            phase,
            variants,
            selector_receipt,
            validation_receipt,
        )
        if provenance["stage"] in seen_stages:
            raise ValueError(f"Duplicate stage run: {provenance['stage']}")
        seen_stages.add(provenance["stage"])
        all_scores.extend(scores)
        all_generations.extend(generations)
        inputs.append(provenance)
    if seen_stages != set(STAGES):
        raise ValueError(f"Repair bundle stages differ: {sorted(seen_stages)}")
    for field in (
        "candidate_ids",
        "prompt_input_token_ids_sha256",
        "score_prompt_hashes_sha256",
        "generation_prompt_hashes_sha256",
        "code_sha256",
        "score_batch_plan_sha256",
        "generation_batch_plan_sha256",
        "software_versions",
        "runtime_contract",
        "effective_stopping",
        "generation_contract",
    ):
        if len({_canonical_sha256(item[field]) for item in inputs}) != 1:
            raise ValueError(f"Cross-stage repair provenance differs at {field}")
    _validate_cartesian(all_scores, all_generations, config, phase, variants)
    return (
        all_scores,
        all_generations,
        {
            "phase": phase,
            "variants": variants,
            "inputs": sorted(inputs, key=lambda item: STAGES.index(item["stage"])),
        },
    )


def auc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUC requires both classes")
    wins = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap(
    values: dict[str, float], draws: int, seed: int
) -> tuple[float, list[float]]:
    if not values or draws <= 0:
        raise ValueError("Bootstrap requires values and positive draws")
    array = [values[key] for key in sorted(values)]
    rng = random.Random(seed)
    estimates = [
        statistics.fmean(array[rng.randrange(len(array))] for _ in array)
        for _ in range(draws)
    ]
    return statistics.fmean(array), [
        _percentile(estimates, 0.025),
        _percentile(estimates, 0.975),
    ]


def _score_index(
    scores: Iterable[dict[str, Any]], variant: str
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (row["stage"], row["item_id"], row["condition"], row["order"]): row
        for row in scores
        if row["variant"] == variant
    }


def _instrument_metrics(
    scores: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    config: dict[str, Any],
    phase: str,
    variant: str,
) -> dict[str, Any]:
    score_rows = [row for row in scores if row["variant"] == variant]
    generation_rows = [row for row in generations if row["variant"] == variant]
    index = _score_index(score_rows, variant)
    thresholds = config["analysis"]
    draws = int(thresholds["bootstrap_draws"])
    loop_limit = (
        thresholds["pilot_loop_rate_max"]
        if phase == "pilot"
        else thresholds["validation_loop_rate_max"]
    )
    stage_metrics: dict[str, Any] = {}
    for stage_index, stage in enumerate(STAGES):
        stage_scores = [row for row in score_rows if row["stage"] == stage]
        stage_generations = [row for row in generation_rows if row["stage"] == stage]
        mass_cells = {
            f"{condition}:{order}": statistics.median(
                row["choice_mass"]
                for row in stage_scores
                if row["condition"] == condition and row["order"] == order
            )
            for condition in config["phases"][phase]["conditions"]
            for order in ORDERS
        }
        control_auc = {
            order: auc(
                [
                    row["choice_score"]
                    for row in stage_scores
                    if row["order"] == order and row["condition"] in CONTROL_CONDITIONS
                ],
                [
                    int(row["condition"] == "all_reason")
                    for row in stage_scores
                    if row["order"] == order and row["condition"] in CONTROL_CONDITIONS
                ],
            )
            for order in ORDERS
        }
        item_headroom = {
            item_id: statistics.fmean(
                index[(stage, item_id, "all_reason", order)]["choice_score"]
                - index[(stage, item_id, "all_direct", order)]["choice_score"]
                for order in ORDERS
            )
            for item_id in config["phases"][phase]["item_lock"]["item_ids"]
        }
        headroom, headroom_interval = _bootstrap(
            item_headroom,
            draws,
            int(config["seed"])
            + 1000
            + VARIANT_ORDER.index(variant) * 100
            + stage_index,
        )
        primary_generations = [
            row for row in stage_generations if row["condition"] in PRIMARY_CONDITIONS
        ]
        classifiable = [row for row in primary_generations if row["classifiable"]]
        # Concordance per the approved v2 gate: among ALL classifiable
        # generations, the leading choice token must match the semantic mode.
        # Classifiable rows without a leading choice token count against
        # concordance (denominator, not numerator) - no subset selection.
        concordant = [
            row
            for row in stage_generations
            if row["classifiable"]
            and row.get("leading_choice") is not None
            and row.get("leading_choice") == row.get("semantic_mode")
        ]
        concordance_denominator = sum(
            1 for row in stage_generations if row["classifiable"]
        )
        concordance = (
            len(concordant) / concordance_denominator
            if concordance_denominator
            else None
        )
        classifiability = len(classifiable) / len(primary_generations)
        loop_count = sum(
            bool(row["terminal_loop"]["detected"]) for row in stage_generations
        )
        loop_rate = loop_count / len(stage_generations)
        behavior_stage = stage in thresholds["behavior_gate_stages"]
        gates = {
            "choice_mass": min(mass_cells.values()) >= thresholds["choice_mass_min"],
            "control_auc": all(
                value >= thresholds["control_auc_min"] for value in control_auc.values()
            ),
            "positive_headroom": headroom_interval[0] > 0.0,
            "concordance": (not behavior_stage)
            or (
                concordance is not None
                and concordance >= thresholds["concordance_min"]
            ),
            "classifiability": (not behavior_stage)
            or classifiability >= thresholds["classifiability_min"],
            "loop_rate": (not behavior_stage) or loop_rate <= loop_limit,
        }
        stage_metrics[stage] = {
            "cell_median_choice_mass": mass_cells,
            "worst_cell_median_choice_mass": min(mass_cells.values()),
            "control_auc_by_order": control_auc,
            "control_headroom": {
                "estimate": headroom,
                "bootstrap_interval": headroom_interval,
            },
            "concordance": concordance,
            "concordant_count": len(concordant),
            "concordance_denominator": concordance_denominator,
            "primary_classifiable_count": len(classifiable),
            "primary_generation_count": len(primary_generations),
            "primary_classifiability": classifiability,
            "loop_count": loop_count,
            "generation_count": len(stage_generations),
            "loop_rate": loop_rate,
            "gates": gates,
            "pass": all(gates.values()),
        }
    control_auc_values = [
        value
        for stage_values in stage_metrics.values()
        for value in stage_values["control_auc_by_order"].values()
    ]
    return {
        "phase": phase,
        "variant": variant,
        "stages": stage_metrics,
        "worst_cell_median_choice_mass": min(
            value["worst_cell_median_choice_mass"] for value in stage_metrics.values()
        ),
        "minimum_stage_control_auc": min(control_auc_values),
        "pass": all(value["pass"] for value in stage_metrics.values()),
    }


def _compact_selector_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "viable": metrics["pass"],
        "worst_cell_median_choice_mass": metrics["worst_cell_median_choice_mass"],
        "minimum_stage_control_auc": metrics["minimum_stage_control_auc"],
        "stages": {
            stage: {
                "worst_cell_median_choice_mass": values[
                    "worst_cell_median_choice_mass"
                ],
                "control_auc_by_order": values["control_auc_by_order"],
                "control_headroom": values["control_headroom"],
                "concordance": values["concordance"],
                "primary_classifiability": values["primary_classifiability"],
                "loop_rate": values["loop_rate"],
                "gates": values["gates"],
                "pass": values["pass"],
            }
            for stage, values in metrics["stages"].items()
        },
    }


def _select_variant(metrics: dict[str, dict[str, Any]]) -> str | None:
    viable = [variant for variant in VARIANT_ORDER if metrics[variant]["pass"]]
    if not viable:
        return None
    return max(
        viable,
        key=lambda variant: (
            metrics[variant]["worst_cell_median_choice_mass"],
            metrics[variant]["minimum_stage_control_auc"],
            -VARIANT_ORDER.index(variant),
        ),
    )


def _confirmation_estimands(
    confirmation_scores: list[dict[str, Any]],
    validation_scores: list[dict[str, Any]],
    config: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    index = _score_index(confirmation_scores, variant)
    item_ids = config["phases"]["confirmation"]["item_lock"]["item_ids"]
    order_effects = {
        stage: {
            order: {
                item_id: index[(stage, item_id, "notes_cue", order)]["choice_score"]
                - index[(stage, item_id, "request_cue", order)]["choice_score"]
                for item_id in item_ids
            }
            for order in ORDERS
        }
        for stage in STAGES
    }
    item_effects = {
        stage: {
            item_id: statistics.fmean(
                order_effects[stage][order][item_id] for order in ORDERS
            )
            for item_id in item_ids
        }
        for stage in STAGES
    }
    psi = {stage: statistics.fmean(item_effects[stage].values()) for stage in STAGES}
    theta_items = {
        item_id: item_effects["rlvr"][item_id] - item_effects["base"][item_id]
        for item_id in item_ids
    }
    theta, interval = _bootstrap(
        theta_items,
        int(config["analysis"]["bootstrap_draws"]),
        int(config["seed"]) + BOOTSTRAP_SEED_OFFSET,
    )
    theta_by_order = {
        order: statistics.fmean(
            order_effects["rlvr"][order][item_id]
            - order_effects["base"][order][item_id]
            for item_id in item_ids
        )
        for order in ORDERS
    }
    calibration = recompute_validation_constants(validation_scores, config, variant)
    headroom = calibration["headroom_H"]
    placebo_noise = calibration["placebo_noise_N"]
    criteria = {
        "theta_interval_below_zero": interval[1] < 0.0,
        "theta_negative_both_orders": all(
            value < 0.0 for value in theta_by_order.values()
        ),
        "headroom_rule": -theta
        >= config["analysis"]["headroom_fraction_min"] * headroom,
        "placebo_rule": -theta
        > config["analysis"]["placebo_noise_multiplier"] * placebo_noise,
    }
    adjacent = {
        "base_to_sft": psi["sft"] - psi["base"],
        "sft_to_dpo": psi["dpo"] - psi["sft"],
        "dpo_to_rlvr": psi["rlvr"] - psi["dpo"],
    }
    return {
        "score_definition": "S = reason_logit - direct_logit",
        "psi": psi,
        "theta": theta,
        "theta_bootstrap_interval": interval,
        "theta_by_order": theta_by_order,
        "adjacent_stage_changes": adjacent,
        "stage_sum_residual": sum(adjacent.values()) - theta,
        "headroom_H": headroom,
        "placebo_noise_N": placebo_noise,
        "criteria": criteria,
        "statistical_support": all(criteria.values()),
    }


def _confirmation_validity(
    scores: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    config: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    index = _score_index(scores, variant)
    thresholds = config["analysis"]
    result: dict[str, Any] = {}
    for stage in STAGES:
        stage_scores = [row for row in scores if row["stage"] == stage]
        stage_generations = [row for row in generations if row["stage"] == stage]
        mass_cells = {
            f"{condition}:{order}": statistics.median(
                row["choice_mass"]
                for row in stage_scores
                if row["condition"] == condition and row["order"] == order
            )
            for condition in PRIMARY_CONDITIONS
            for order in ORDERS
        }
        classifiable = [row for row in stage_generations if row["classifiable"]]
        concordant = [
            row
            for row in stage_generations
            if row["classifiable"]
            and row.get("leading_choice") is not None
            and row.get("leading_choice") == row.get("semantic_mode")
        ]
        concordance_denominator = sum(
            1 for row in stage_generations if row["classifiable"]
        )
        concordance = (
            len(concordant) / concordance_denominator
            if concordance_denominator
            else None
        )
        classifiability = len(classifiable) / len(stage_generations)
        loop_count = sum(row["terminal_loop"]["detected"] for row in stage_generations)
        loop_rate = loop_count / len(stage_generations)
        behavior_stage = stage in thresholds["behavior_gate_stages"]
        gates = {
            "choice_mass": min(mass_cells.values()) >= thresholds["choice_mass_min"],
            "concordance": (not behavior_stage)
            or (
                concordance is not None
                and concordance >= thresholds["concordance_min"]
            ),
            "classifiability": (not behavior_stage)
            or classifiability >= thresholds["classifiability_min"],
            "loop_rate": (not behavior_stage)
            or loop_rate <= thresholds["validation_loop_rate_max"],
        }
        result[stage] = {
            "cell_median_choice_mass": mass_cells,
            "worst_cell_median_choice_mass": min(mass_cells.values()),
            "concordance": concordance,
            "concordant_count": len(concordant),
            "concordance_denominator": concordance_denominator,
            "primary_classifiable_count": len(classifiable),
            "primary_generation_count": len(stage_generations),
            "primary_classifiability": classifiability,
            "loop_count": loop_count,
            "loop_rate": loop_rate,
            "gates": gates,
            "pass": all(gates.values()),
        }
    return {"stages": result, "pass": all(value["pass"] for value in result.values())}


def _assert_recomputation_matches(
    primary: dict[str, Any], independent: dict[str, Any]
) -> None:
    paths = [
        ("theta",),
        ("theta_bootstrap_interval", 0),
        ("theta_bootstrap_interval", 1),
        ("headroom_H",),
        ("placebo_noise_N",),
    ]
    paths.extend(("psi", stage) for stage in STAGES)
    paths.extend(("theta_by_order", order) for order in ORDERS)
    for path in paths:
        left: Any = primary
        right: Any = independent
        for key in path:
            left = left[key]
            right = right[key]
        # bootstrap bounds are stochastic; allow small float divergence
        tol = (
            1e-3
            if any(isinstance(k, str) and "bootstrap_interval" in k for k in path)
            else 1e-10
        )
        if not math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol):
            raise AssertionError(f"Independent recomputation disagrees at {path}")
    if primary["criteria"] != independent["criteria"]:
        raise AssertionError("Independent recomputation disagrees on support criteria")


def analyze_phase(
    config: dict[str, Any],
    phase: str,
    run_dirs: list[Path],
    *,
    selector_receipt: dict[str, Any] | None = None,
    validation_receipt: dict[str, Any] | None = None,
    validation_run_dirs: list[Path] | None = None,
    created_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    scores, generations, input_provenance = load_run_bundle(
        run_dirs, config, phase, selector_receipt, validation_receipt
    )
    now = created_at_utc or datetime.now(timezone.utc).isoformat()
    code_hashes = {
        "repair_analyze.py": _file_sha256(Path(__file__)),
        "repair_recompute.py": _file_sha256(
            Path(__file__).with_name("repair_recompute.py")
        ),
    }
    independent: dict[str, Any] | None = None
    if phase == "pilot":
        metrics = {
            variant: _instrument_metrics(scores, generations, config, phase, variant)
            for variant in VARIANT_ORDER
        }
        selected = _select_variant(metrics)
        receipt = {
            "schema_version": 1,
            "phase": "pilot",
            "created_at_utc": now,
            "config_sha256": _canonical_sha256(config),
            "analysis_code_sha256": code_hashes,
            "input_fingerprint": _canonical_sha256(input_provenance),
            "variant_priority": list(VARIANT_ORDER),
            "variant_metrics": {
                variant: _compact_selector_metrics(metrics[variant])
                for variant in VARIANT_ORDER
            },
            "selection_pass": selected is not None,
            "selected_variant": selected,
        }
        receipt["receipt_sha256"] = _receipt_hash(receipt)
        summary = {
            "schema_version": 1,
            "phase": phase,
            "instrument_metrics": metrics,
            "selection_pass": selected is not None,
            "selected_variant": selected,
            "stop_required": selected is None,
            "receipt": receipt,
        }
    elif phase == "validation":
        assert selector_receipt is not None
        variant = verify_selector_receipt(selector_receipt, config)
        metrics = _instrument_metrics(scores, generations, config, phase, variant)
        constants = recompute_validation_constants(scores, config, variant)
        receipt = {
            "schema_version": 1,
            "phase": "validation",
            "created_at_utc": now,
            "config_sha256": _canonical_sha256(config),
            "analysis_code_sha256": code_hashes,
            "input_fingerprint": _canonical_sha256(input_provenance),
            "selector_receipt_sha256": selector_receipt["receipt_sha256"],
            "selected_variant": variant,
            "instrument_metrics": metrics,
            "calibration_constants": constants,
            "validation_pass": metrics["pass"],
        }
        receipt["receipt_sha256"] = _receipt_hash(receipt)
        summary = {
            "schema_version": 1,
            "phase": phase,
            "selected_variant": variant,
            "instrument_metrics": metrics,
            "calibration_constants": constants,
            "validation_pass": metrics["pass"],
            "confirmation_authorized": metrics["pass"],
            "receipt": receipt,
        }
    else:
        if (
            selector_receipt is None
            or validation_receipt is None
            or validation_run_dirs is None
        ):
            raise ValueError(
                "Confirmation analysis requires selector, validation receipt, and validation runs"
            )
        variant = verify_validation_receipt(
            validation_receipt,
            config,
            selector_receipt,
            allow_failed_validation=True,
        )
        validation_scores, _, validation_provenance = load_run_bundle(
            validation_run_dirs,
            config,
            "validation",
            selector_receipt,
            None,
        )
        constants = recompute_validation_constants(validation_scores, config, variant)
        if validation_receipt.get("calibration_constants") != constants:
            raise ValueError(
                "Validation receipt constants disagree with validation raw data"
            )
        validity = _confirmation_validity(scores, generations, config, variant)
        estimands = _confirmation_estimands(scores, validation_scores, config, variant)
        independent = recompute_confirmation(scores, validation_scores, config, variant)
        _assert_recomputation_matches(estimands, independent)
        supported = validity["pass"] and estimands["statistical_support"]
        receipt = {
            "schema_version": 1,
            "phase": "confirmation",
            "created_at_utc": now,
            "config_sha256": _canonical_sha256(config),
            "analysis_code_sha256": code_hashes,
            "input_fingerprint": _canonical_sha256(input_provenance),
            "validation_input_fingerprint": _canonical_sha256(validation_provenance),
            "selector_receipt_sha256": selector_receipt["receipt_sha256"],
            "validation_receipt_sha256": validation_receipt["receipt_sha256"],
            "selected_variant": variant,
            "instrument_valid": validity["pass"],
            "statistical_support": estimands["statistical_support"],
            "confirmatory_support": supported,
            "approved_for_claims": False,
            "human_review_required": True,
        }
        receipt["receipt_sha256"] = _receipt_hash(receipt)
        summary = {
            "schema_version": 1,
            "phase": phase,
            "selected_variant": variant,
            "instrument_validity": validity,
            "estimands": estimands,
            "confirmatory_support": supported,
            "claim_status": (
                "provisional_supported_pending_human_review"
                if supported
                else "provisional_not_supported_pending_human_review"
            ),
            "receipt": receipt,
        }
        input_provenance["validation_inputs"] = validation_provenance["inputs"]
    summary["behavior_audit"] = _behavior_audit(generations, config, phase)
    return summary, input_provenance, independent


def _behavior_audit(
    generations: list[dict[str, Any]], config: dict[str, Any], phase: str
) -> list[dict[str, Any]]:
    failures = [
        row
        for row in generations
        if not row["classifiable"] or row["terminal_loop"]["detected"]
    ]

    def key(row: dict[str, Any]) -> str:
        identity = "\0".join(
            str(row[field])
            for field in ["stage", "variant", "item_id", "condition", "order"]
        )
        return hashlib.sha256(
            f"{config['seed']}\0repair-behavior-audit:{phase}\0{identity}".encode()
        ).hexdigest()

    representative = sorted(generations, key=key)[:20]
    selected: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    failure_ids = {
        (row["stage"], row["variant"], row["item_id"], row["condition"], row["order"])
        for row in failures
    }
    for row in [*failures, *representative]:
        identity = (
            row["stage"],
            row["variant"],
            row["item_id"],
            row["condition"],
            row["order"],
        )
        selected[identity] = {
            "selection": "failure" if identity in failure_ids else "representative",
            "stage": row["stage"],
            "variant": row["variant"],
            "item_id": row["item_id"],
            "condition": row["condition"],
            "order": row["order"],
            "response": row["response"],
            "semantic_mode": row["semantic_mode"],
            "classifiable": row["classifiable"],
            "semantic_failure_categories": row["semantic_failure_categories"],
            "terminal_loop": row["terminal_loop"],
            "cap_hit": row["cap_hit"],
        }
    return [selected[identity] for identity in sorted(selected)]


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _write(path: Path, text: str) -> None:
    with path.open("x") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _figure_svg(summary: dict[str, Any]) -> str:
    if summary["phase"] == "confirmation":
        labels = list(STAGES)
        values = [float(summary["estimands"]["psi"][stage]) for stage in labels]
        title = "Stagewise request-binding score"
    else:
        metrics = (
            summary["instrument_metrics"]
            if summary["phase"] == "pilot"
            else {summary["selected_variant"]: summary["instrument_metrics"]}
        )
        labels = [variant for variant in VARIANT_ORDER if variant in metrics]
        values = [
            float(metrics[variant]["worst_cell_median_choice_mass"])
            for variant in labels
        ]
        title = "Worst-cell candidate mass"
    lower = min([0.0, *values])
    upper = max([0.0, *values])
    span = upper - lower or 1.0
    left, top, width, height = 70, 45, 520, 260
    zero_y = top + (upper / span) * height
    points = []
    text = []
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + (index + 0.5) * width / len(labels)
        y = top + (upper - value) * height / span
        points.append(f"{x:.1f},{y:.1f}")
        text.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb"/>'
            f'<text x="{x:.1f}" y="{top + height + 24}" text-anchor="middle" '
            f'font-size="12">{html.escape(label)}</text>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="380" '
        'viewBox="0 0 640 380">'
        '<rect width="640" height="380" fill="white"/>'
        f'<text x="320" y="25" text-anchor="middle" font-size="18">{html.escape(title)}</text>'
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + width}" y2="{zero_y:.1f}" '
        'stroke="#9ca3af"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#2563eb" stroke-width="2"/>'
        f"{''.join(text)}"
        f'<text x="18" y="{top + height / 2:.1f}" transform="rotate(-90 18 '
        f'{top + height / 2:.1f})" text-anchor="middle" font-size="12">value</text>'
        "</svg>\n"
    )


def _table_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    phase = summary["phase"]
    if phase == "pilot":
        variants = summary["instrument_metrics"]
        variant_names = [variant for variant in VARIANT_ORDER if variant in variants]
    elif phase == "validation":
        metric = summary["instrument_metrics"]
        variants = {metric["variant"]: metric}
        variant_names = [metric["variant"]]
    else:
        return [
            {
                "stage": stage,
                "psi": summary["estimands"]["psi"][stage],
                "concordance": summary["instrument_validity"]["stages"][stage][
                    "concordance"
                ],
                "primary_classifiability": summary["instrument_validity"]["stages"][
                    stage
                ]["primary_classifiability"],
                "loop_rate": summary["instrument_validity"]["stages"][stage][
                    "loop_rate"
                ],
            }
            for stage in STAGES
        ]
    return [
        {
            "variant": variant,
            "stage": stage,
            "worst_cell_median_choice_mass": values["worst_cell_median_choice_mass"],
            "concordance": values["concordance"],
            "primary_classifiability": values["primary_classifiability"],
            "loop_rate": values["loop_rate"],
            "pass": values["pass"],
        }
        for variant in variant_names
        for metric in [variants[variant]]
        for stage in STAGES
        for values in [metric["stages"][stage]]
    ]


def _table_csv(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _figure_data(summary: dict[str, Any]) -> dict[str, Any]:
    if summary["phase"] == "confirmation":
        return {
            "psi_by_stage": summary["estimands"]["psi"],
            "theta": summary["estimands"]["theta"],
            "theta_bootstrap_interval": summary["estimands"][
                "theta_bootstrap_interval"
            ],
            "theta_by_order": summary["estimands"]["theta_by_order"],
        }
    return {
        "stage_metrics": _table_rows(summary),
        "selected_variant": summary.get("selected_variant"),
    }


def write_evidence_bundle(
    output_dir: Path,
    summary: dict[str, Any],
    input_provenance: dict[str, Any],
    config: dict[str, Any],
    independent: dict[str, Any] | None,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    root = Path(
        tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent)
    )
    try:
        _write(root / "summary.json", _json_text(summary))
        receipt_name = {
            "pilot": "selector-receipt.json",
            "validation": "validation-receipt.json",
            "confirmation": "confirmation-receipt.json",
        }[summary["phase"]]
        _write(root / receipt_name, _json_text(summary["receipt"]))
        _write(
            root / "gate-metrics.json",
            _json_text(
                summary.get("instrument_metrics") or summary.get("instrument_validity")
            ),
        )
        rows = _table_rows(summary)
        _write(root / "stage-metrics.csv", _table_csv(rows))
        _write(root / "figure-data.json", _json_text(_figure_data(summary)))
        _write(root / "figure.svg", _figure_svg(summary))
        with (root / "behavior-audit.jsonl").open("x") as handle:
            for row in summary["behavior_audit"]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if independent is not None:
            _write(root / "independent-recomputation.json", _json_text(independent))
        artifact_hashes = {
            path.name: _file_sha256(path)
            for path in sorted(root.iterdir())
            if path.is_file()
        }
        provenance = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "config_sha256": _canonical_sha256(config),
            "analysis_code_sha256": {
                "repair_analyze.py": _file_sha256(Path(__file__)),
                "repair_recompute.py": _file_sha256(
                    Path(__file__).with_name("repair_recompute.py")
                ),
            },
            "inputs": input_provenance,
            "artifact_sha256": artifact_hashes,
            "programmatic_evidence_invariant": True,
        }
        _write(root / "provenance.json", _json_text(provenance))
        root.replace(output_dir)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def verify_evidence_bundle(
    evidence_dir: Path,
    config: dict[str, Any],
    phase: str,
    run_dirs: list[Path],
    *,
    selector_receipt: dict[str, Any] | None = None,
    validation_receipt: dict[str, Any] | None = None,
    validation_run_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    summary = json.loads((evidence_dir / "summary.json").read_text())
    provenance = json.loads((evidence_dir / "provenance.json").read_text())
    receipt_name = {
        "pilot": "selector-receipt.json",
        "validation": "validation-receipt.json",
        "confirmation": "confirmation-receipt.json",
    }[phase]
    receipt = json.loads((evidence_dir / receipt_name).read_text())
    if receipt != summary.get("receipt") or receipt.get(
        "receipt_sha256"
    ) != _receipt_hash(receipt):
        raise ValueError("Repair evidence receipt disagrees with its summary")
    if provenance.get("config_sha256") != _canonical_sha256(config):
        raise ValueError("Repair evidence provenance has the wrong config")
    for name, expected in provenance.get("artifact_sha256", {}).items():
        path = evidence_dir / name
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"Repair evidence artifact drifted: {name}")
    fresh, fresh_inputs, independent = analyze_phase(
        config,
        phase,
        run_dirs,
        selector_receipt=selector_receipt,
        validation_receipt=validation_receipt,
        validation_run_dirs=validation_run_dirs,
        created_at_utc=receipt["created_at_utc"],
    )
    if fresh != summary:
        raise ValueError("Repair evidence summary disagrees with fresh analysis")
    if provenance.get("inputs") != fresh_inputs:
        raise ValueError("Repair evidence input provenance disagrees with raw runs")
    expected_code = {
        "repair_analyze.py": _file_sha256(Path(__file__)),
        "repair_recompute.py": _file_sha256(
            Path(__file__).with_name("repair_recompute.py")
        ),
    }
    if provenance.get("analysis_code_sha256") != expected_code:
        raise ValueError("Repair evidence used different analysis code")
    expected_artifacts = {
        "gate-metrics.json": _json_text(
            summary.get("instrument_metrics") or summary.get("instrument_validity")
        ),
        "stage-metrics.csv": _table_csv(_table_rows(summary)),
        "figure-data.json": _json_text(_figure_data(summary)),
        "figure.svg": _figure_svg(summary),
        "behavior-audit.jsonl": "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in summary["behavior_audit"]
        ),
    }
    for name, expected in expected_artifacts.items():
        if (evidence_dir / name).read_text() != expected:
            raise ValueError(
                f"Generated repair artifact disagrees with analysis: {name}"
            )
    if independent is not None:
        stored = json.loads(
            (evidence_dir / "independent-recomputation.json").read_text()
        )
        if stored != independent:
            raise ValueError("Independent recomputation artifact drifted")
    return receipt


def recalibration_constants(
    scores: list[dict[str, Any]],
) -> dict[str, Any]:
    """Paraphrase-placebo noise constants from recalibration scores.

    Pure function: cells are (stage, assignment, order) wording sensitivities
    |mean S(canonical) - S(paraphrase)|; N_max is primary, mean/median are
    reported for transparency. Also returns fresh-item psi per stage.
    """
    index: dict[tuple[str, str, str, str], float] = {
        (row["stage"], row["item_id"], row["condition"], row["order"]): row[
            "choice_score"
        ]
        for row in scores
    }
    item_ids = sorted({row["item_id"] for row in scores})
    cells: dict[str, float] = {}
    for stage in ("base", "rlvr"):
        for order in ORDERS:
            request_gap = statistics.fmean(
                index[(stage, item, "para_request_b", order)]
                - index[(stage, item, "request_cue", order)]
                for item in item_ids
            )
            notes_gap = statistics.fmean(
                index[(stage, item, "para_notes_b", order)]
                - index[(stage, item, "notes_cue", order)]
                for item in item_ids
            )
            cells[f"{stage}:{order}:request"] = abs(request_gap)
            cells[f"{stage}:{order}:notes"] = abs(notes_gap)
    values = list(cells.values())
    psi_fresh = {
        stage: statistics.fmean(
            index[(stage, item, "notes_cue", order)]
            - index[(stage, item, "request_cue", order)]
            for item in item_ids
            for order in ORDERS
        )
        for stage in ("base", "rlvr")
    }
    return {
        "cells": cells,
        "N_max": max(values),
        "N_mean": statistics.fmean(values),
        "N_median": statistics.median(values),
        "psi_fresh": psi_fresh,
    }


def analyze_recalibration(
    config: dict[str, Any],
    run_dirs: list[Path],
    confirmation_evidence_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    variant = config["phases"]["recalibration"]["variant"]
    expected_variants = [variant]
    all_scores: list[dict[str, Any]] = []
    seen_stages: set[str] = set()
    for run_dir in run_dirs:
        scores, generations, _ = _manifest_provenance(
            run_dir,
            config,
            "recalibration",
            expected_variants,
            None,
            None,
        )
        if generations:
            raise ValueError("Recalibration runs must not contain generations")
        stage = json.loads((run_dir / "manifest.json").read_text())["stage"]
        seen_stages.add(stage)
        all_scores.extend(scores)
    if seen_stages != {"base", "rlvr"}:
        raise ValueError(f"Recalibration requires base and rlvr runs: {sorted(seen_stages)}")
    item_ids = config["phases"]["recalibration"]["item_lock"]["item_ids"]
    expected = {
        (stage, item_id, condition, order)
        for stage in ("base", "rlvr")
        for item_id in item_ids
        for condition in RECALIBRATION_CONDITIONS
        for order in ORDERS
    }
    actual = {
        (row["stage"], row["item_id"], row["condition"], row["order"])
        for row in all_scores
    }
    if actual != expected:
        raise ValueError(
            f"Recalibration scores are not Cartesian-complete: "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )

    confirmation = json.loads(
        (confirmation_evidence_dir / "summary.json").read_text()
    )
    theta = confirmation["estimands"]["theta"]
    theta_interval = confirmation["estimands"]["theta_bootstrap_interval"]

    constants = recalibration_constants(all_scores)
    criterion_max = -theta > 2.0 * constants["N_max"]
    criterion_mean = -theta > 2.0 * constants["N_mean"]
    criterion_median = -theta > 2.0 * constants["N_median"]

    summary = {
        "schema_version": 1,
        "phase": "recalibration",
        "status": "post-hoc recalibration, suggestive not confirmatory",
        "variant": variant,
        "item_count": len(item_ids),
        "theta_from_confirmation": theta,
        "theta_bootstrap_interval": theta_interval,
        "constants": constants,
        "criterion": {
            "rule": "-theta > 2 * N",
            "N_max_passes": criterion_max,
            "N_mean_passes": criterion_mean,
            "N_median_passes": criterion_median,
        },
    }
    summary["receipt_sha256"] = _receipt_hash(
        {k: v for k, v in summary.items() if k != "receipt_sha256"}
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {output_dir}")
    output_dir.mkdir(parents=True)
    _write(output_dir / "summary.json", _json_text(summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and analyze the repaired mode-choice experiment"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/frame_stage_binding_repair_v2.locked.json"),
    )
    parser.add_argument(
        "--phase",
        choices=("pilot", "validation", "confirmation", "recalibration"),
        required=True,
    )
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--validation-run-dir", type=Path, action="append")
    parser.add_argument("--selector-receipt", type=Path)
    parser.add_argument("--validation-receipt", type=Path)
    parser.add_argument("--confirmation-evidence-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_optional(path: Path | None) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path is not None else None


def main() -> None:
    args = parse_args()
    config = load_repair_config(args.config)
    if args.phase == "recalibration":
        if args.confirmation_evidence_dir is None:
            raise ValueError("Recalibration analysis requires the confirmation evidence")
        summary = analyze_recalibration(
            config,
            args.run_dir,
            args.confirmation_evidence_dir,
            args.output_dir,
        )
        print(_json_text(summary))
        return
    selector = _read_optional(args.selector_receipt)
    validation = _read_optional(args.validation_receipt)
    summary, provenance, independent = analyze_phase(
        config,
        args.phase,
        args.run_dir,
        selector_receipt=selector,
        validation_receipt=validation,
        validation_run_dirs=args.validation_run_dir,
    )
    write_evidence_bundle(args.output_dir, summary, provenance, config, independent)


if __name__ == "__main__":
    main()
