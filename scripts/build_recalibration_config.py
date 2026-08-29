"""Build the frozen paraphrase-recalibration config.

Derives 352 fresh GSM8K train items (ranked positions 200:552 under the
pilot selection namespace; positions 0:200 are the consumed pilot and
validation items) and emits the recalibration config: the v2 config plus
the recalibration phase. The v2 config file itself is not modified.

Usage:
    python -m scripts.build_recalibration_config
"""

from __future__ import annotations

import json
from pathlib import Path

from frame_stage_binding.repair_core import (
    PARAPHRASE_REASONING_DIRECTIVE,
    RECALIBRATION_CONDITIONS,
    ranked_items,
)
from frame_stage_binding.repair_lock import _item_lock
from frame_stage_binding.utils import _canonical_sha256, extract_gsm8k_reference

V2_CONFIG = Path("configs/frame_stage_binding_repair_v2.locked.json")
OUTPUT = Path("configs/frame_stage_binding_recalibration.locked.json")


def main() -> None:
    v2 = json.loads(V2_CONFIG.read_text())
    from datasets import load_dataset

    rows = load_dataset(
        v2["dataset"]["dataset_id"],
        v2["dataset"]["config"],
        revision=v2["dataset"]["revision"],
        split="train",
    )
    all_items = [
        {
            "benchmark": "gsm8k",
            "item_id": f"gsm8k:train:{index}",
            "question": row["question"],
            "reference": extract_gsm8k_reference(row["answer"]),
        }
        for index, row in enumerate(rows)
    ]
    # The original calibration lock ranked only the first 200 train items
    # (source limit: 200). Fresh items come from ranking the FULL train
    # split under the recalibration namespace, excluding the consumed 200.
    consumed = set(v2["phases"]["pilot"]["item_lock"]["item_ids"]) | set(
        v2["phases"]["validation"]["item_lock"]["item_ids"]
    )
    ranked = ranked_items(
        [item for item in all_items if item["item_id"] not in consumed],
        v2["seed"],
        "mode-choice-repair-v2:recalibration",
    )
    recalibration_items = ranked[:352]

    config = json.loads(json.dumps(v2))
    config["phases"]["recalibration"] = {
        "split": "train",
        "variant": "explicit_12",
        "conditions": list(RECALIBRATION_CONDITIONS),
        "orders": list(v2["phases"]["pilot"]["orders"]),
        "max_new_tokens": 512,
        "generation_conditions": [],
        "item_lock": _item_lock(recalibration_items, v2["dataset"]["revision"]),
        "generation_item_lock": _item_lock([], v2["dataset"]["revision"]),
        "selection_namespace": "mode-choice-repair-v2:recalibration",
        "paraphrase_directive": PARAPHRASE_REASONING_DIRECTIVE,
    }
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT} | recalibration config sha256: {_canonical_sha256(config)}")


if __name__ == "__main__":
    main()
