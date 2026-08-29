"""One-time migration: recompute stored semantic fields in repair-v2 pilot
generation records using the v2 classifier.

The pod ran the pilot with code that stored final_answer from the old
line-anchored parser while all other semantic fields came from the v2
classifier. The response text is the immutable raw truth; this script
recomputes every semantic field from it and writes a new versioned
generations file. The original file is preserved untouched.

Usage:
    python -m scripts.migrate_v2_semantics --run-dir <pilot-stage-dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frame_stage_binding.repair_core import classify_semantic_mode
from frame_stage_binding.utils import answers_equal, _file_sha256


def migrate(run_dir: Path) -> dict[str, int]:
    generations_path = run_dir / "generations.jsonl"
    rows = [json.loads(line) for line in generations_path.read_text().splitlines()]
    changed = 0
    for row in rows:
        semantic = classify_semantic_mode(row["response"], row["variant"])
        new_correct = answers_equal(
            semantic["final_answer"], row["reference"], "gsm8k"
        )
        if (
            row.get("final_answer") != semantic["final_answer"]
            or row.get("correct") != new_correct
        ):
            changed += 1
        row.update(semantic)
        row["correct"] = new_correct
    migrated_path = run_dir / "generations-v2.jsonl"
    if migrated_path.exists():
        raise FileExistsError(f"Refusing to overwrite {migrated_path}")
    with migrated_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "rows": len(rows),
        "changed": changed,
        "migrated_sha256": _file_sha256(migrated_path),
        "original_sha256": _file_sha256(generations_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    args = parser.parse_args()
    for run_dir in args.run_dir:
        print(run_dir, json.dumps(migrate(run_dir), indent=2))


if __name__ == "__main__":
    main()
