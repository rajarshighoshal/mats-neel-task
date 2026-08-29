from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from .repair_core import (
    CANONICAL_TEMPLATE,
    CONDITIONS,
    DIRECT_DIRECTIVE,
    GENERATION_CONDITIONS,
    ORDERS,
    PLACEBO_DIRECTIVE,
    REASONING_DIRECTIVE,
    STAGES,
    VARIANT_ORDER,
    VARIANTS,
    ranked_items,
)
from .utils import (
    _canonical_sha256,
    _item_content_sha256,
    extract_gsm8k_reference,
)


def _load_source_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


PILOT_NAMESPACE = "mode-choice-repair-v1:pilot"
VALIDATION_GENERATION_NAMESPACE = "mode-choice-repair-v1:validation-generation"
CONFIRMATION_GENERATION_NAMESPACE = "mode-choice-repair-v1:confirmation-generation"


def _item_lock(items: list[dict[str, Any]], dataset_revision: str) -> dict[str, Any]:
    return {
        "item_count": len(items),
        "item_ids": [item["item_id"] for item in items],
        "item_ids_sha256": _canonical_sha256(sorted(item["item_id"] for item in items)),
        "item_content_sha256": _item_content_sha256(items),
        "dataset_revisions": {"openai/gsm8k": dataset_revision},
    }


def _load_locked_items(source: dict[str, Any], phase_name: str) -> list[dict[str, str]]:
    phase = source["phases"][phase_name]
    dataset = source["datasets"][phase["dataset"]]
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/datasets--openai--gsm8k/snapshots"
        / dataset["revision"]
        / dataset["config"]
    )
    parquet_paths = sorted(snapshot.glob(f"{phase['split']}-*.parquet"))
    if parquet_paths:
        import pyarrow.parquet as parquet

        rows = parquet.read_table(parquet_paths).to_pylist()
    else:
        from datasets import load_dataset

        rows = load_dataset(
            dataset["dataset_id"],
            dataset["config"],
            revision=dataset["revision"],
            split=phase["split"],
        )
    items = [
        {
            "benchmark": "gsm8k",
            "item_id": f"gsm8k:{phase['split']}:{index}",
            "question": row["question"],
            "reference": extract_gsm8k_reference(row["answer"]),
        }
        for index, row in enumerate(rows)
    ]
    limit = phase.get("limit")
    if limit is not None and limit < len(items):
        indices = sorted(random.Random(source["seed"]).sample(range(len(items)), limit))
        items = [items[index] for index in indices]
    lock = phase["item_lock"]
    checks = {
        "item_count": len(items) == lock["item_count"],
        "item_ids_sha256": _canonical_sha256(sorted(item["item_id"] for item in items))
        == lock["item_ids_sha256"],
        "item_content_sha256": _item_content_sha256(items)
        == lock["item_content_sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"Inherited {phase_name} item lock failed: {checks}")
    return items


def build_repair_lock(source: dict[str, Any]) -> dict[str, Any]:
    train_items = _load_locked_items(source, "calibration")
    test_items = _load_locked_items(source, "confirm")
    dataset_revision = source["datasets"]["gsm8k"]["revision"]
    ranked_train = ranked_items(train_items, source["seed"], PILOT_NAMESPACE)
    pilot_items = ranked_train[:24]
    validation_items = ranked_train[24:]
    validation_generation = ranked_items(
        validation_items, source["seed"], VALIDATION_GENERATION_NAMESPACE
    )[:48]
    confirmation_generation = ranked_items(
        test_items, source["seed"], CONFIRMATION_GENERATION_NAMESPACE
    )[:64]
    if len(pilot_items) != 24 or len(validation_items) != 176:
        raise ValueError("The inherited 200-item calibration lock did not split 24/176")
    return {
        "schema_version": 1,
        "study": "tulu-stage-binding-mode-choice-repair-v2",
        "seed": source["seed"],
        "source_config_sha256": _canonical_sha256(source),
        "models": {stage: source["models"][stage] for stage in STAGES},
        "dataset": source["datasets"]["gsm8k"],
        "prompt_protocol": {
            "canonical_template": CANONICAL_TEMPLATE,
            "reasoning_directive": REASONING_DIRECTIVE,
            "direct_directive": DIRECT_DIRECTIVE,
            "placebo_directive": PLACEBO_DIRECTIVE,
            "conditions": list(CONDITIONS),
            "orders": list(ORDERS),
            "variant_order": list(VARIANT_ORDER),
            "variant_tie_priority": list(VARIANT_ORDER),
            "variants": VARIANTS,
        },
        "generation": {
            key: source["generation"][key]
            for key in [
                "batch_size",
                "do_sample",
                "eos_token_ids",
                "eos_token_texts",
                "pad_token_id",
                "pad_token_text",
                "use_cache",
                "cache_implementation",
                "model_dtype",
            ]
        },
        "analysis": {
            "bootstrap_draws": 10000,
            "choice_mass_min": 0.05,
            "control_auc_min": 0.80,
            "concordance_min": 0.90,
            "classifiability_min": 0.90,
            "pilot_loop_rate_max": 0.0,
            "validation_loop_rate_max": 0.01,
            "headroom_fraction_min": 0.20,
            "placebo_noise_multiplier": 2.0,
            "behavior_gate_stages": ["sft", "dpo", "rlvr"],
        },
        "hypothesis": {
            "name": "request_binding_amplification",
            "primary_shift": "psi_rlvr_minus_psi_base",
            "supported_direction": "negative",
            "requires_endpoint_signs": False,
        },
        "phases": {
            "pilot": {
                "split": "train",
                "variants": list(VARIANT_ORDER),
                "conditions": list(CONDITIONS),
                "generation_conditions": list(GENERATION_CONDITIONS),
                "orders": list(ORDERS),
                "item_lock": _item_lock(pilot_items, dataset_revision),
                "generation_item_lock": _item_lock(pilot_items, dataset_revision),
                "max_new_tokens": 256,
                "selection_namespace": PILOT_NAMESPACE,
            },
            "validation": {
                "split": "train",
                "variants": "selected",
                "conditions": list(CONDITIONS),
                "generation_conditions": list(GENERATION_CONDITIONS),
                "orders": list(ORDERS),
                "item_lock": _item_lock(validation_items, dataset_revision),
                "generation_item_lock": _item_lock(
                    validation_generation, dataset_revision
                ),
                "max_new_tokens": 512,
                "selection_namespace": PILOT_NAMESPACE,
                "generation_namespace": VALIDATION_GENERATION_NAMESPACE,
            },
            "confirmation": {
                "split": "test",
                "variants": "selected",
                "conditions": ["request_cue", "notes_cue"],
                "generation_conditions": ["request_cue", "notes_cue"],
                "orders": list(ORDERS),
                "item_lock": _item_lock(test_items, dataset_revision),
                "generation_item_lock": _item_lock(
                    confirmation_generation, dataset_revision
                ),
                "max_new_tokens": 512,
                "generation_namespace": CONFIRMATION_GENERATION_NAMESPACE,
            },
        },
        "limits": {
            "additional_runpod_spend_usd": 12.0,
            "pilot_wall_seconds": 2700,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the mode-choice repair design")
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path("configs/frame_stage_binding.locked.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    config = build_repair_lock(_load_source_config(args.source_config))
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite {temporary}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x") as handle:
        handle.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
