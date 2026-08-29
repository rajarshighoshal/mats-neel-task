from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .repair_core import (
    CANONICAL_TEMPLATE,
    CONDITIONS,
    DIRECT_DIRECTIVE,
    GENERATION_CONDITIONS,
    ORDERS,
    PARAPHRASE_REASONING_DIRECTIVE,
    PLACEBO_DIRECTIVE,
    PRIMARY_CONDITIONS,
    REASONING_DIRECTIVE,
    RECALIBRATION_CONDITIONS,
    STAGES,
    VARIANT_ORDER,
    VARIANTS,
    build_prompt,
    candidate_texts,
    classify_semantic_mode,
    detect_terminal_loop,
)
from .utils import (
    _candidate_ids,
    _canonical_sha256,
    _checkpoint_stopping_fields,
    _chunks,
    _configure_shared_stopping,
    _file_sha256,
    _item_content_sha256,
    _normalized_dtype_name,
    _software_versions,
    _tokenizer_hash,
    _trim_generated_ids,
    _validate_process_topology,
    _write_json_atomic,
    answers_equal,
    extract_final_answer,
    extract_gsm8k_reference,
)


SOURCE_CONFIG_SHA256 = (
    "b946df28e8e6eab3adc6b739c39d9a37ebe7e3c181067cdd1cf69564c7572d68"
)
MODEL_REVISIONS = {
    "base": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "sft": "f2a0b46b0cfda21003c6141b1ff837b7e165524d",
    "dpo": "a7beb67e33ffd01cc87ac3b46cadc1000985b8db",
    "rlvr": "666943798adbde0b1aff34626007e26986a3c107",
}
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
LOCK_HASHES = {
    "pilot": (
        "57f8da4818fdd00c8d633084dc8a29d794a489588a60a43b49e09d60bb137947",
        "1bdc889d5eaf83550562dad422a280ec4c14328203d1a4fb89fcd205ebb98608",
    ),
    "validation": (
        "7e9d5abf5c18e9525089d5498d3a41f72f5cd5aa1fc65c581a78480d16f2778f",
        "bb46df4c46ff8c47adc4ddbf157e953200e2c281a4d8ca17d41c42cb5e43aea8",
    ),
    "validation_generation": (
        "95962ba9a1b3256aeb3072c96176597a0ab507f7f3da0ad6b5280be7a0c5c0c9",
        "f55f06dbc14a57ef3ddf1be6ecf21edd3166c1d8715140cfd9815dea319b77e2",
    ),
    "confirmation": (
        "95aaf06aeab89d88caa3c1111f01e528f5b114957df252d57067628bcf4a03dc",
        "6d9a2ff6cf344760e6a081409e0c4711ce622a9cf8c5175f4c991803bea28f94",
    ),
    "confirmation_generation": (
        "b04e3d385f87341572435f8364009e797030c08cca473501db1e88630c2cc6b2",
        "6bc267c82fd653c9839b90c01f9f24a5e24590bccd2c8e5214e539d1407fed44",
    ),
}


def load_repair_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    validate_repair_config(config)
    return config


def validate_repair_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported repair config schema")
    if config.get("study") != "tulu-stage-binding-mode-choice-repair-v2":
        raise ValueError("Unexpected repair study")
    if config.get("seed") != 20260821:
        raise ValueError("Repair seed changed")
    expected_models = {
        "base": "meta-llama/Llama-3.1-8B",
        "sft": "allenai/Llama-3.1-Tulu-3-8B-SFT",
        "dpo": "allenai/Llama-3.1-Tulu-3-8B-DPO",
        "rlvr": "allenai/Llama-3.1-Tulu-3-8B",
    }
    if set(config.get("models", {})) != set(STAGES):
        raise ValueError("Repair config must contain exactly the four Tulu stages")
    for stage, model_id in expected_models.items():
        model = config["models"][stage]
        if (
            model.get("model_id") != model_id
            or model.get("revision") != MODEL_REVISIONS[stage]
        ):
            raise ValueError(f"Invalid locked model at {stage}")
    dataset = config.get("dataset", {})
    if (
        dataset.get("dataset_id") != "openai/gsm8k"
        or dataset.get("config") != "main"
        or dataset.get("revision") != DATASET_REVISION
    ):
        raise ValueError("Repair dataset is not immutable GSM8K main")
    if config.get("source_config_sha256") != SOURCE_CONFIG_SHA256:
        raise ValueError("Repair source config changed")
    expected_protocol = {
        "canonical_template": CANONICAL_TEMPLATE,
        "reasoning_directive": REASONING_DIRECTIVE,
        "direct_directive": DIRECT_DIRECTIVE,
        "placebo_directive": PLACEBO_DIRECTIVE,
        "conditions": list(CONDITIONS),
        "orders": list(ORDERS),
        "variant_order": list(VARIANT_ORDER),
        "variant_tie_priority": list(VARIANT_ORDER),
        "variants": VARIANTS,
    }
    if config.get("prompt_protocol") != expected_protocol:
        raise ValueError("Repair prompt protocol changed")
    generation = config.get("generation", {})
    required_generation = {
        "batch_size": 4,
        "do_sample": False,
        "eos_token_ids": [128001],
        "eos_token_texts": ["<|end_of_text|>"],
        "pad_token_id": 128001,
        "pad_token_text": "<|end_of_text|>",
        "use_cache": True,
        "cache_implementation": "dynamic",
        "model_dtype": "bfloat16",
    }
    if generation != required_generation:
        raise ValueError("Repair generation contract changed")
    expected_analysis = {
        "bootstrap_draws": 10000,
        "choice_mass_min": 0.05,
        "control_auc_min": 0.8,
        "concordance_min": 0.9,
        "classifiability_min": 0.9,
        "pilot_loop_rate_max": 0.0,
        "validation_loop_rate_max": 0.01,
        "headroom_fraction_min": 0.2,
        "placebo_noise_multiplier": 2.0,
        "behavior_gate_stages": ["sft", "dpo", "rlvr"],
    }
    if config.get("analysis") != expected_analysis:
        raise ValueError("Repair analysis contract changed")
    expected_phases = {
        "pilot": (24, 24, list(VARIANT_ORDER), list(CONDITIONS), 256),
        "validation": (176, 48, "selected", list(CONDITIONS), 512),
        "confirmation": (
            1319,
            64,
            "selected",
            list(PRIMARY_CONDITIONS),
            512,
        ),
    }
    if set(config.get("phases", {})) != set(expected_phases) and set(
        config.get("phases", {})
    ) != set(expected_phases) | {"recalibration"}:
        raise ValueError("Repair phases changed")
    for phase_name, (
        item_count,
        generation_count,
        variants,
        conditions,
        cap,
    ) in expected_phases.items():
        phase = config["phases"][phase_name]
        expected_split = "test" if phase_name == "confirmation" else "train"
        if (
            phase.get("split") != expected_split
            or phase.get("variants") != variants
            or phase.get("conditions") != conditions
            or phase.get("orders") != list(ORDERS)
            or phase.get("max_new_tokens") != cap
        ):
            raise ValueError(f"Repair phase contract changed at {phase_name}")
        expected_generation_conditions = (
            list(PRIMARY_CONDITIONS)
            if phase_name == "confirmation"
            else list(GENERATION_CONDITIONS)
        )
        if phase.get("generation_conditions") != expected_generation_conditions:
            raise ValueError(f"Generation conditions changed at {phase_name}")
        _validate_item_lock(phase.get("item_lock"), item_count, dataset["revision"])
        _validate_item_lock(
            phase.get("generation_item_lock"), generation_count, dataset["revision"]
        )
        if not set(phase["generation_item_lock"]["item_ids"]) <= set(
            phase["item_lock"]["item_ids"]
        ):
            raise ValueError(f"Generation subset escapes phase items at {phase_name}")
    recalibration = config["phases"].get("recalibration")
    if recalibration is not None:
        expected_recalibration = {
            "split": "train",
            "variant": "explicit_12",
            "conditions": list(RECALIBRATION_CONDITIONS),
            "orders": list(ORDERS),
            "max_new_tokens": 512,
            "generation_conditions": [],
            "selection_namespace": "mode-choice-repair-v2:recalibration",
            "paraphrase_directive": PARAPHRASE_REASONING_DIRECTIVE,
        }
        for field, expected in expected_recalibration.items():
            if recalibration.get(field) != expected:
                raise ValueError(f"Recalibration contract changed at {field}")
        _validate_item_lock(recalibration.get("item_lock"), 352, dataset["revision"])
        _validate_item_lock(
            recalibration.get("generation_item_lock"), 0, dataset["revision"]
        )
    expected_namespaces = {
        "pilot": {
            "selection_namespace": "mode-choice-repair-v1:pilot",
        },
        "validation": {
            "selection_namespace": "mode-choice-repair-v1:pilot",
            "generation_namespace": "mode-choice-repair-v1:validation-generation",
        },
        "confirmation": {
            "generation_namespace": "mode-choice-repair-v1:confirmation-generation",
        },
    }
    for phase_name, fields in expected_namespaces.items():
        for field, expected in fields.items():
            if config["phases"][phase_name].get(field) != expected:
                raise ValueError(f"Repair namespace changed at {phase_name}.{field}")
    locks = config["phases"]
    observed_hashes = {
        "pilot": locks["pilot"]["item_lock"],
        "validation": locks["validation"]["item_lock"],
        "validation_generation": locks["validation"]["generation_item_lock"],
        "confirmation": locks["confirmation"]["item_lock"],
        "confirmation_generation": locks["confirmation"]["generation_item_lock"],
    }
    for name, lock in observed_hashes.items():
        if (lock["item_ids_sha256"], lock["item_content_sha256"]) != LOCK_HASHES[name]:
            raise ValueError(f"Frozen repair item lock changed at {name}")
    pilot_ids = set(locks["pilot"]["item_lock"]["item_ids"])
    validation_ids = set(locks["validation"]["item_lock"]["item_ids"])
    if pilot_ids & validation_ids or _canonical_sha256(
        sorted(pilot_ids | validation_ids)
    ) != ("df44eb106d453b7148e9fe714afcfb4f7311d839285590f748c92cdbaa53a719"):
        raise ValueError(
            "Pilot and validation do not partition the inherited 200 items"
        )
    hypothesis = config.get("hypothesis", {})
    if hypothesis != {
        "name": "request_binding_amplification",
        "primary_shift": "psi_rlvr_minus_psi_base",
        "supported_direction": "negative",
        "requires_endpoint_signs": False,
    }:
        raise ValueError("Repair hypothesis changed")
    if config.get("limits") != {
        "additional_runpod_spend_usd": 12.0,
        "pilot_wall_seconds": 2700,
    }:
        raise ValueError("Repair execution limits changed")


def _validate_item_lock(lock: Any, count: int, revision: str) -> None:
    if not isinstance(lock, dict) or lock.get("item_count") != count:
        raise ValueError("Repair item lock count changed")
    ids = lock.get("item_ids")
    if not isinstance(ids, list) or len(ids) != count or len(set(ids)) != count:
        raise ValueError("Repair item lock has invalid IDs")
    if _canonical_sha256(sorted(ids)) != lock.get("item_ids_sha256"):
        raise ValueError("Repair item ID hash mismatch")
    if lock.get("dataset_revisions") != {"openai/gsm8k": revision}:
        raise ValueError("Repair item lock dataset revision mismatch")
    if not isinstance(lock.get("item_content_sha256"), str):
        raise ValueError("Repair item lock lacks content hash")


def load_phase_items(
    config: dict[str, Any], phase_name: str
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    dataset = config["dataset"]
    info = HfApi().dataset_info(dataset["dataset_id"], revision=dataset["revision"])
    if info.sha != dataset["revision"]:
        raise ValueError("Resolved GSM8K revision differs from repair lock")
    phase = config["phases"][phase_name]
    rows = load_dataset(
        dataset["dataset_id"],
        dataset["config"],
        revision=dataset["revision"],
        split=phase["split"],
    )
    all_items = {
        f"gsm8k:{phase['split']}:{index}": {
            "benchmark": "gsm8k",
            "item_id": f"gsm8k:{phase['split']}:{index}",
            "question": row["question"],
            "reference": extract_gsm8k_reference(row["answer"]),
        }
        for index, row in enumerate(rows)
    }
    items = _select_locked_items(all_items, phase["item_lock"])
    generation_items = _select_locked_items(all_items, phase["generation_item_lock"])
    return items, generation_items, {dataset["dataset_id"]: info.sha}


def _select_locked_items(
    all_items: dict[str, dict[str, str]], lock: dict[str, Any]
) -> list[dict[str, str]]:
    try:
        items = [all_items[item_id] for item_id in lock["item_ids"]]
    except KeyError as error:
        raise ValueError(f"Locked item is missing from dataset: {error}") from error
    checks = {
        "count": len(items) == lock["item_count"],
        "ids": _canonical_sha256(sorted(item["item_id"] for item in items))
        == lock["item_ids_sha256"],
        "content": _item_content_sha256(items) == lock["item_content_sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"Repair item lock validation failed: {checks}")
    return items


def _receipt_hash(receipt: dict[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def _analysis_code_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        name: _file_sha256(package / name)
        for name in ["repair_analyze.py", "repair_recompute.py"]
    }


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _expected_instrument_gates(
    values: dict[str, Any], config: dict[str, Any], loop_limit: float, stage: str
) -> dict[str, bool]:
    behavior_stage = stage in config["analysis"]["behavior_gate_stages"]
    return {
        "choice_mass": values["worst_cell_median_choice_mass"]
        >= config["analysis"]["choice_mass_min"],
        "control_auc": all(
            value >= config["analysis"]["control_auc_min"]
            for value in values["control_auc_by_order"].values()
        ),
        "positive_headroom": values["control_headroom"]["bootstrap_interval"][0] > 0.0,
        "concordance": (not behavior_stage)
        or (
            values["concordance"] is not None
            and values["concordance"] >= config["analysis"]["concordance_min"]
        ),
        "classifiability": (not behavior_stage)
        or values["primary_classifiability"]
        >= config["analysis"]["classifiability_min"],
        "loop_rate": (not behavior_stage) or values["loop_rate"] <= loop_limit,
    }


def _verify_compact_variant_metrics(
    metrics: Any, config: dict[str, Any]
) -> tuple[bool, float, float]:
    if not isinstance(metrics, dict) or set(metrics) != {
        "viable",
        "worst_cell_median_choice_mass",
        "minimum_stage_control_auc",
        "stages",
    }:
        raise ValueError("Selector variant metrics have an unexpected schema")
    if set(metrics["stages"]) != set(STAGES):
        raise ValueError("Selector metrics lack a Tulu stage")
    stage_passes: list[bool] = []
    stage_masses: list[float] = []
    stage_aucs: list[float] = []
    expected_gates = {
        "choice_mass",
        "control_auc",
        "positive_headroom",
        "concordance",
        "classifiability",
        "loop_rate",
    }
    for stage, values in metrics["stages"].items():
        required = {
            "worst_cell_median_choice_mass",
            "control_auc_by_order",
            "control_headroom",
            "concordance",
            "primary_classifiability",
            "loop_rate",
            "gates",
            "pass",
        }
        if not isinstance(values, dict) or set(values) != required:
            raise ValueError("Selector stage metrics have an unexpected schema")
        if set(values["control_auc_by_order"]) != set(ORDERS):
            raise ValueError("Selector control AUC lacks a block order")
        gates = values["gates"]
        if set(gates) != expected_gates or not all(
            isinstance(value, bool) for value in gates.values()
        ):
            raise ValueError("Selector gate fields are invalid")
        expected = _expected_instrument_gates(
            values, config, config["analysis"]["pilot_loop_rate_max"], stage
        )
        if gates != expected or values["pass"] is not all(expected.values()):
            raise ValueError("Selector stage pass disagrees with its gates")
        interval = values["control_headroom"].get("bootstrap_interval", [])
        numeric = [
            values["worst_cell_median_choice_mass"],
            *values["control_auc_by_order"].values(),
            values["control_headroom"].get("estimate"),
            *interval,
            values["primary_classifiability"],
            values["loop_rate"],
        ]
        if len(interval) != 2 or not all(_finite_number(value) for value in numeric):
            raise ValueError("Selector stage metrics contain an invalid number")
        concordance = values["concordance"]
        if concordance is not None and not _finite_number(concordance):
            raise ValueError("Selector concordance is invalid")
        stage_passes.append(values["pass"])
        stage_masses.append(float(values["worst_cell_median_choice_mass"]))
        stage_aucs.extend(float(value) for value in values["control_auc_by_order"].values())
    viable = all(stage_passes)
    minimum_auc = min(stage_aucs)
    if (
        metrics["viable"] is not viable
        or not math.isclose(
            float(metrics["worst_cell_median_choice_mass"]),
            min(stage_masses),
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(metrics["minimum_stage_control_auc"]),
            minimum_auc,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Selector aggregate metrics disagree with stage metrics")
    return viable, min(stage_masses), minimum_auc


def verify_selector_receipt(receipt: dict[str, Any], config: dict[str, Any]) -> str:
    required = {
        "schema_version",
        "phase",
        "created_at_utc",
        "config_sha256",
        "analysis_code_sha256",
        "input_fingerprint",
        "variant_priority",
        "variant_metrics",
        "selection_pass",
        "selected_variant",
        "receipt_sha256",
    }
    if set(receipt) != required:
        raise ValueError("Selector receipt has an unexpected schema")
    if receipt.get("schema_version") != 1 or receipt.get("phase") != "pilot":
        raise ValueError("Invalid selector receipt schema")
    if receipt.get("config_sha256") != _canonical_sha256(config):
        raise ValueError("Selector receipt config mismatch")
    # analysis_code_sha256 is historical provenance (which code produced this
    # receipt); current analysis code is allowed to differ.
    if not isinstance(receipt.get("analysis_code_sha256"), dict) or not (
        receipt["analysis_code_sha256"]
        and all(
            isinstance(value, str) and len(value) == 64
            for value in receipt["analysis_code_sha256"].values()
        )
    ):
        raise ValueError("Selector receipt lacks valid analysis code provenance")
    if receipt.get("receipt_sha256") != _receipt_hash(receipt):
        raise ValueError("Selector receipt hash mismatch")
    if receipt.get("variant_priority") != list(VARIANT_ORDER):
        raise ValueError("Selector receipt changed tie priority")
    forbidden = re.compile(
        r"(?:psi|theta|source.effect|request.dominance)", re.IGNORECASE
    )
    if any(forbidden.search(str(key)) for key in _walk_keys(receipt)):
        raise ValueError("Selector receipt contains a forbidden source-effect field")
    if set(receipt.get("variant_metrics", {})) != set(VARIANT_ORDER):
        raise ValueError("Selector receipt lacks variant metrics")
    summaries = {
        variant: _verify_compact_variant_metrics(
            receipt["variant_metrics"][variant], config
        )
        for variant in VARIANT_ORDER
    }
    viable = [variant for variant in VARIANT_ORDER if summaries[variant][0]]
    selected = (
        max(
            viable,
            key=lambda variant: (
                summaries[variant][1],
                summaries[variant][2],
                -VARIANT_ORDER.index(variant),
            ),
        )
        if viable
        else None
    )
    if receipt.get("selection_pass") is not bool(viable):
        raise ValueError("Selector pass disagrees with viable variants")
    if receipt.get("selected_variant") != selected:
        raise ValueError("Selector chose a non-deterministic variant")
    if selected is None:
        raise ValueError("Selector receipt did not pass")
    return selected


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _only_documented_base_control_auc_miss(receipt: dict[str, Any]) -> bool:
    """The applicant-authorized exception: validation failed ONLY because
    Base missed the control-AUC gate. Every other gate at every stage must
    have passed."""
    metrics = receipt.get("instrument_metrics", {})
    stages = metrics.get("stages", {})
    if not isinstance(stages, dict) or set(stages) != set(STAGES):
        return False
    for stage, values in stages.items():
        gates = values.get("gates", {})
        if not isinstance(gates, dict):
            return False
        failed = {name for name, passed in gates.items() if not passed}
        if not failed:
            continue
        if stage == "base" and failed == {"control_auc"}:
            continue
        return False
    return True


def verify_validation_receipt(
    receipt: dict[str, Any],
    config: dict[str, Any],
    selector: dict[str, Any],
    *,
    allow_failed_validation: bool = False,
) -> str:
    selected = verify_selector_receipt(selector, config)
    required = {
        "schema_version",
        "phase",
        "created_at_utc",
        "config_sha256",
        "analysis_code_sha256",
        "input_fingerprint",
        "selector_receipt_sha256",
        "selected_variant",
        "instrument_metrics",
        "calibration_constants",
        "validation_pass",
        "receipt_sha256",
    }
    checks = {
        "keys": set(receipt) == required,
        "schema": receipt.get("schema_version") == 1,
        "phase": receipt.get("phase") == "validation",
        "config": receipt.get("config_sha256") == _canonical_sha256(config),
        "pass": receipt.get("validation_pass") is True
        or (
            allow_failed_validation
            and _only_documented_base_control_auc_miss(receipt)
        ),
        "variant": receipt.get("selected_variant") == selected,
        "selector": receipt.get("selector_receipt_sha256")
        == selector["receipt_sha256"],
        "hash": receipt.get("receipt_sha256") == _receipt_hash(receipt),
        "analysis_code": isinstance(receipt.get("analysis_code_sha256"), dict)
        and bool(receipt["analysis_code_sha256"])
        and all(
            isinstance(value, str) and len(value) == 64
            for value in receipt["analysis_code_sha256"].values()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Validation receipt failed: {checks}")
    metrics = receipt["instrument_metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("Validation instrument metrics must be an object")
    required_stage_fields = {
        "cell_median_choice_mass",
        "worst_cell_median_choice_mass",
        "control_auc_by_order",
        "control_headroom",
        "concordance",
        "concordant_count",
        "concordance_denominator",
        "primary_classifiable_count",
        "primary_generation_count",
        "primary_classifiability",
        "loop_count",
        "generation_count",
        "loop_rate",
        "gates",
        "pass",
    }
    stage_passes: list[bool] = []
    for stage, values in metrics.get("stages", {}).items():
        if set(values) != required_stage_fields:
            raise ValueError("Validation stage metrics have an unexpected schema")
        expected_gates = _expected_instrument_gates(
            values, config, config["analysis"]["validation_loop_rate_max"], stage
        )
        if values["gates"] != expected_gates or values["pass"] is not all(
            expected_gates.values()
        ):
            raise ValueError("Validation stage gates disagree with frozen thresholds")
        stage_passes.append(values["pass"])
    if (
        not isinstance(metrics, dict)
        or metrics.get("phase") != "validation"
        or metrics.get("variant") != selected
        or set(metrics.get("stages", {})) != set(STAGES)
        or metrics.get("pass") is not all(stage_passes)
        or (
            metrics.get("pass") is not True
            and not (
                allow_failed_validation
                and _only_documented_base_control_auc_miss(receipt)
            )
        )
    ):
        raise ValueError("Validation receipt does not contain a passing instrument")
    constants = receipt["calibration_constants"]
    if (
        not isinstance(constants, dict)
        or not _finite_number(constants.get("headroom_H"))
        or not _finite_number(constants.get("placebo_noise_N"))
        or constants["headroom_H"] <= 0
        or constants["placebo_noise_N"] < 0
    ):
        raise ValueError("Validation calibration constants are invalid")
    return selected


def variants_for_phase(
    config: dict[str, Any],
    phase_name: str,
    selector_receipt: dict[str, Any] | None,
    validation_receipt: dict[str, Any] | None,
    *,
    allow_failed_validation: bool = False,
) -> list[str]:
    if phase_name == "pilot":
        if selector_receipt is not None or validation_receipt is not None:
            raise ValueError("Pilot must not consume later receipts")
        return list(VARIANT_ORDER)
    if phase_name == "recalibration":
        if selector_receipt is not None or validation_receipt is not None:
            raise ValueError("Recalibration must not consume receipts")
        return [config["phases"]["recalibration"]["variant"]]
    if selector_receipt is None:
        raise ValueError(f"{phase_name} requires a selector receipt")
    selected = verify_selector_receipt(selector_receipt, config)
    if phase_name == "confirmation":
        if validation_receipt is None:
            raise ValueError("Confirmation requires a validation receipt")
        selected = verify_validation_receipt(
            validation_receipt,
            config,
            selector_receipt,
            allow_failed_validation=allow_failed_validation,
        )
    elif validation_receipt is not None:
        raise ValueError("Validation must not consume its own receipt")
    return [selected]


def build_phase_records(
    items: list[dict[str, str]],
    generation_items: list[dict[str, str]],
    config: dict[str, Any],
    phase_name: str,
    stage: str,
    variants: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    phase = config["phases"][phase_name]
    generation_ids = {item["item_id"] for item in generation_items}
    score_records: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []
    for variant in variants:
        for item in items:
            for condition in phase["conditions"]:
                for order in phase["orders"]:
                    prompt = build_prompt(item["question"], condition, order, variant)
                    record = {
                        **item,
                        "phase": phase_name,
                        "stage": stage,
                        "variant": variant,
                        "condition": condition,
                        "order": order,
                        "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    }
                    score_records.append(record)
                    if (
                        item["item_id"] in generation_ids
                        and condition in phase["generation_conditions"]
                    ):
                        generation_records.append(dict(record))
    return score_records, generation_records


def _code_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        name: _file_sha256(package / name)
        for name in ["repair_run.py", "repair_core.py", "utils.py"]
    }


def _batch_plan(records: list[dict[str, Any]], size: int) -> str:
    return _canonical_sha256(
        [
            [
                {
                    key: record[key]
                    for key in [
                        "phase",
                        "variant",
                        "item_id",
                        "condition",
                        "order",
                        "prompt_sha256",
                    ]
                }
                for record in batch
            ]
            for batch in _chunks(records, size)
        ]
    )


def _prepare(
    config: dict[str, Any],
    phase_name: str,
    stage: str,
    variants: list[str],
) -> dict[str, Any]:
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    model_spec = config["models"][stage]
    info = HfApi().model_info(model_spec["model_id"], revision=model_spec["revision"])
    if info.sha != model_spec["revision"]:
        raise ValueError("Resolved model revision differs from repair lock")
    tokenizer = AutoTokenizer.from_pretrained(model_spec["model_id"], revision=info.sha)
    tokenizer_sha = _tokenizer_hash(tokenizer)
    stopping = _configure_shared_stopping(config, tokenizer)
    items, generation_items, dataset_revisions = load_phase_items(config, phase_name)
    score_records, generation_records = build_phase_records(
        items, generation_items, config, phase_name, stage, variants
    )
    candidate_ids: dict[str, dict[str, list[int]]] = {}
    for variant in variants:
        prompts = [
            record["prompt"] for record in score_records if record["variant"] == variant
        ]
        reason, direct = candidate_texts(variant)
        reason_ids = _candidate_ids(tokenizer, prompts[0], reason, True)
        direct_ids = _candidate_ids(tokenizer, prompts[0], direct, True)
        if len(reason_ids) != 1 or len(direct_ids) != 1 or reason_ids == direct_ids:
            raise ValueError(
                f"Repair variant {variant} does not have distinct single-token choices"
            )
        for prompt in prompts[1:]:
            if _candidate_ids(tokenizer, prompt, reason, True) != reason_ids:
                raise ValueError(f"Reason candidate tokenization varies for {variant}")
            if _candidate_ids(tokenizer, prompt, direct, True) != direct_ids:
                raise ValueError(f"Direct candidate tokenization varies for {variant}")
        candidate_ids[variant] = {"reason": reason_ids, "direct": direct_ids}
    prompt_token_ids = [tokenizer.encode(record["prompt"]) for record in score_records]
    return {
        "model_spec": model_spec,
        "model_revision": info.sha,
        "tokenizer": tokenizer,
        "tokenizer_sha256": tokenizer_sha,
        "enforced_tokenizer_sha256": _tokenizer_hash(tokenizer),
        "stopping": stopping,
        "items": items,
        "generation_items": generation_items,
        "dataset_revisions": dataset_revisions,
        "score_records": score_records,
        "generation_records": generation_records,
        "candidate_ids": candidate_ids,
        "prompt_input_token_ids_sha256": _canonical_sha256(prompt_token_ids),
        "score_prompt_hashes_sha256": _canonical_sha256(
            [record["prompt_sha256"] for record in score_records]
        ),
        "generation_prompt_hashes_sha256": _canonical_sha256(
            [record["prompt_sha256"] for record in generation_records]
        ),
    }


def _score_batch(
    model: Any,
    encoded: Any,
    reason_id: int,
    direct_id: int,
) -> list[dict[str, float | str]]:
    import torch

    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits[:, -1, :].float()
    normalization = torch.logsumexp(logits, dim=-1)
    reason_logits = logits[:, reason_id]
    direct_logits = logits[:, direct_id]
    reason_logprobs = reason_logits - normalization
    direct_logprobs = direct_logits - normalization
    choice_mass = reason_logprobs.exp() + direct_logprobs.exp()
    if not all(
        torch.isfinite(value).all().item()
        for value in [
            reason_logits,
            direct_logits,
            reason_logprobs,
            direct_logprobs,
            choice_mass,
        ]
    ):
        raise ValueError("Non-finite repair choice scores")
    result: list[dict[str, float | str]] = []
    for r_logit, d_logit, r_logprob, d_logprob, mass in zip(
        reason_logits.tolist(),
        direct_logits.tolist(),
        reason_logprobs.tolist(),
        direct_logprobs.tolist(),
        choice_mass.tolist(),
    ):
        score = float(r_logit - d_logit)
        result.append(
            {
                "reason_logit": float(r_logit),
                "direct_logit": float(d_logit),
                "reason_logprob": float(r_logprob),
                "direct_logprob": float(d_logprob),
                "choice_score": score,
                "choice_mass": float(mass),
                "predicted_mode": "reason"
                if score > 0
                else "direct"
                if score < 0
                else "tie",
            }
        )
    return result


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def run_stage(
    config: dict[str, Any],
    phase_name: str,
    stage: str,
    output_path: Path,
    selector_receipt: dict[str, Any] | None,
    validation_receipt: dict[str, Any] | None,
    *,
    allow_failed_validation: bool = False,
) -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Repair inference requires BF16 CUDA")
    _validate_process_topology(
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("RANK", "0")),
        torch.cuda.device_count(),
    )
    variants = variants_for_phase(
        config,
        phase_name,
        selector_receipt,
        validation_receipt,
        allow_failed_validation=allow_failed_validation,
    )
    temporary = output_path.with_name(output_path.name + ".tmp")
    if output_path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite repair run {output_path}")
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_started = time.perf_counter()
    prepared = _prepare(config, phase_name, stage, variants)
    model_spec = prepared["model_spec"]
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["model_id"],
        revision=prepared["model_revision"],
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    model.eval()
    parameter_devices = {str(parameter.device) for parameter in model.parameters()}
    parameter_dtypes = {
        str(parameter.dtype)
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    if any(not device.startswith("cuda") for device in parameter_devices):
        raise RuntimeError(f"Model offload is forbidden: {parameter_devices}")
    if parameter_dtypes != {str(torch.bfloat16)}:
        raise RuntimeError(f"Repair model is not uniformly BF16: {parameter_dtypes}")
    checkpoint_stopping = _checkpoint_stopping_fields(
        model.generation_config, model.config
    )
    prepared["stopping"].update(checkpoint_stopping)
    model.config.use_cache = config["generation"]["use_cache"]
    prepared["stopping"]["effective_model_config_use_cache"] = model.config.use_cache
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "--id=0"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    batch_size = config["generation"]["batch_size"]
    device = next(model.parameters()).device
    temporary.mkdir(parents=True)
    score_path = temporary / "scores.jsonl"
    generation_path = temporary / "generations.jsonl"
    score_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    prompt_tokens = 0
    generated_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    score_started = time.perf_counter()
    for batch in tqdm(
        _chunks(prepared["score_records"], batch_size),
        total=math.ceil(len(prepared["score_records"]) / batch_size),
        desc=f"repair:{phase_name}:{stage}:score",
    ):
        variants_in_batch = {record["variant"] for record in batch}
        if len(variants_in_batch) != 1:
            raise RuntimeError("A scoring batch crossed repair variants")
        variant = next(iter(variants_in_batch))
        encoded = prepared["tokenizer"](
            [record["prompt"] for record in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(device)
        ids = prepared["candidate_ids"][variant]
        scores = _score_batch(model, encoded, ids["reason"][0], ids["direct"][0])
        for index, (record, score) in enumerate(zip(batch, scores)):
            prompt_ids = [
                int(value)
                for value in encoded["input_ids"][index][
                    encoded["attention_mask"][index].bool()
                ].tolist()
            ]
            score_rows.append(
                {
                    **record,
                    **score,
                    "prompt_input_token_ids": prompt_ids,
                    "reason_candidate_token_ids": ids["reason"],
                    "direct_candidate_token_ids": ids["direct"],
                    "model_id": model_spec["model_id"],
                    "model_revision": prepared["model_revision"],
                }
            )
            prompt_tokens += len(prompt_ids)
        if (
            phase_name == "pilot"
            and time.perf_counter() - run_started
            > config["limits"]["pilot_wall_seconds"]
        ):
            raise TimeoutError("Repair pilot exceeded its frozen wall-time limit")
    score_seconds = time.perf_counter() - score_started
    generation_started = time.perf_counter()
    cap = config["phases"][phase_name]["max_new_tokens"]
    eos_ids = list(config["generation"]["eos_token_ids"])
    for batch in tqdm(
        _chunks(prepared["generation_records"], batch_size),
        total=math.ceil(len(prepared["generation_records"]) / batch_size),
        desc=f"repair:{phase_name}:{stage}:generate",
    ):
        encoded = prepared["tokenizer"](
            [record["prompt"] for record in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=cap,
                pad_token_id=prepared["tokenizer"].pad_token_id,
                eos_token_id=eos_ids,
                use_cache=config["generation"]["use_cache"],
                cache_implementation=config["generation"]["cache_implementation"],
            )
        prompt_width = encoded["input_ids"].shape[1]
        for index, record in enumerate(batch):
            padded = [int(value) for value in generated[index, prompt_width:].tolist()]
            new_ids, finish_reason = _trim_generated_ids(padded, cap, eos_ids)
            response = prepared["tokenizer"].decode(new_ids, skip_special_tokens=True)
            semantic = classify_semantic_mode(response, record["variant"])
            loop_ids = list(new_ids)
            while loop_ids and loop_ids[-1] in eos_ids:
                loop_ids.pop()
            loop = detect_terminal_loop(loop_ids)
            generation_rows.append(
                {
                    **record,
                    **semantic,
                    "terminal_loop": loop,
                    "response": response,
                    "generated_token_ids": new_ids,
                    "response_token_count": len(new_ids),
                    "finish_reason": finish_reason,
                    "cap_hit": finish_reason == "length",
                    "correct": answers_equal(
                        semantic["final_answer"], record["reference"], "gsm8k"
                    ),
                    "model_id": model_spec["model_id"],
                    "model_revision": prepared["model_revision"],
                }
            )
            generated_tokens += len(new_ids)
        if (
            phase_name == "pilot"
            and time.perf_counter() - run_started
            > config["limits"]["pilot_wall_seconds"]
        ):
            raise TimeoutError("Repair pilot exceeded its frozen wall-time limit")
    generation_seconds = time.perf_counter() - generation_started
    score_count = _write_jsonl(score_path, score_rows)
    generation_count = _write_jsonl(generation_path, generation_rows)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - run_started
    manifest = {
        "schema_version": 1,
        "study": config["study"],
        "phase": phase_name,
        "stage": stage,
        "variants": variants,
        "model_id": model_spec["model_id"],
        "model_revision": prepared["model_revision"],
        "dataset_revisions": prepared["dataset_revisions"],
        "config": config,
        "config_sha256": _canonical_sha256(config),
        "code_sha256": _code_hashes(),
        "selector_receipt_sha256": selector_receipt.get("receipt_sha256")
        if selector_receipt
        else None,
        "validation_receipt_sha256": validation_receipt.get("receipt_sha256")
        if validation_receipt
        else None,
        "tokenizer_sha256": prepared["tokenizer_sha256"],
        "enforced_tokenizer_sha256": prepared["enforced_tokenizer_sha256"],
        "candidate_ids": prepared["candidate_ids"],
        "stopping": prepared["stopping"],
        "generation": {
            **config["generation"],
            "max_new_tokens": cap,
            "effective_batch_size": batch_size,
        },
        "score_record_count": score_count,
        "generation_record_count": generation_count,
        "score_batch_plan_sha256": _batch_plan(prepared["score_records"], batch_size),
        "generation_batch_plan_sha256": _batch_plan(
            prepared["generation_records"], batch_size
        ),
        "prompt_input_token_ids_sha256": prepared["prompt_input_token_ids_sha256"],
        "score_prompt_hashes_sha256": prepared["score_prompt_hashes_sha256"],
        "generation_prompt_hashes_sha256": prepared["generation_prompt_hashes_sha256"],
        "scores_sha256": _file_sha256(score_path),
        "generations_sha256": _file_sha256(generation_path),
        "software_versions": _software_versions(),
        "runtime_environment": {
            "model_dtype": _normalized_dtype_name(model.dtype),
            "parameter_devices": sorted(parameter_devices),
            "parameter_dtypes": sorted(parameter_dtypes),
            "attention_implementation": getattr(
                model.config, "_attn_implementation", None
            ),
            "torch_cuda_version": torch.version.cuda,
            "nvidia_driver_version": driver,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu": {
                "name": torch.cuda.get_device_properties(0).name,
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "compute_capability": [
                    torch.cuda.get_device_properties(0).major,
                    torch.cuda.get_device_properties(0).minor,
                ],
                "uuid": str(
                    getattr(torch.cuda.get_device_properties(0), "uuid", "unknown")
                ),
            },
        },
        "performance": {
            "wall_seconds": wall_seconds,
            "score_seconds": score_seconds,
            "generation_seconds": generation_seconds,
            "score_records_per_second": score_count / score_seconds,
            "generated_tokens_per_second": generated_tokens / generation_seconds
            if generation_seconds
            else 0.0,
            "prompt_token_count": prompt_tokens,
            "generated_token_count": generated_tokens,
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "started_at_utc": run_started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
    }
    manifest_path = temporary / "manifest.json"
    with manifest_path.open("x") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the repaired mode-choice instrument"
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
    parser.add_argument("--stage", choices=list(STAGES), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selector-receipt", type=Path)
    parser.add_argument("--validation-receipt", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--allow-failed-validation",
        action="store_true",
        help="Applicant-authorized override: proceed to confirmation with a "
        "documented validation limitation",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _read_optional(path: Path | None) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path else None


def _verify_cli_prior_evidence(
    config: dict[str, Any],
    phase: str,
    artifact_root: Path | None,
    selector_path: Path | None,
    validation_path: Path | None,
    selector: dict[str, Any] | None,
    validation: dict[str, Any] | None,
) -> None:
    if phase in ("pilot", "recalibration"):
        return
    if artifact_root is None:
        raise ValueError("Repair follow-up requires --artifact-root")
    from .repair_analyze import verify_evidence_bundle

    expected_selector = artifact_root / "evidence/pilot/selector-receipt.json"
    if selector_path is None or selector_path.resolve() != expected_selector.resolve():
        raise ValueError("Repair follow-up requires the canonical selector receipt")
    pilot_runs = [artifact_root / "raw" / f"pilot-{stage}" for stage in STAGES]
    if selector != verify_evidence_bundle(
        artifact_root / "evidence/pilot", config, "pilot", pilot_runs
    ):
        raise ValueError("Selector receipt differs from canonical pilot evidence")
    if phase == "confirmation":
        expected_validation = (
            artifact_root / "evidence/validation/validation-receipt.json"
        )
        if (
            validation_path is None
            or validation_path.resolve() != expected_validation.resolve()
        ):
            raise ValueError("Confirmation requires the canonical validation receipt")
        validation_runs = [
            artifact_root / "raw" / f"validation-{stage}" for stage in STAGES
        ]
        if validation != verify_evidence_bundle(
            artifact_root / "evidence/validation",
            config,
            "validation",
            validation_runs,
            selector_receipt=selector,
        ):
            raise ValueError("Validation receipt differs from canonical evidence")


def main() -> None:
    args = parse_args()
    config = load_repair_config(args.config)
    selector = _read_optional(args.selector_receipt)
    validation = _read_optional(args.validation_receipt)
    _verify_cli_prior_evidence(
        config,
        args.phase,
        args.artifact_root,
        args.selector_receipt,
        args.validation_receipt,
        selector,
        validation,
    )
    variants = variants_for_phase(
        config,
        args.phase,
        selector,
        validation,
        allow_failed_validation=args.allow_failed_validation,
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "stage": args.stage,
                    "variants": variants,
                    "mode": "plan",
                },
                indent=2,
            )
        )
        return
    if args.output is None:
        raise ValueError("Execution requires --output")
    run_stage(
        config,
        args.phase,
        args.stage,
        args.output,
        selector,
        validation,
        allow_failed_validation=args.allow_failed_validation,
    )


if __name__ == "__main__":
    main()
