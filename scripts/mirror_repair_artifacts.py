from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from frame_stage_binding.repair_core import STAGES
from frame_stage_binding.repair_run import _receipt_hash
from frame_stage_binding.utils import _file_sha256, _write_json_atomic


def _verify_snapshot(root: Path, phase: str) -> dict[str, Any]:
    """Verify a mirrored phase snapshot against the per-stage run manifests.

    Each stage's manifest.json carries the sha256 of its own scores.jsonl and
    generations.jsonl, so the snapshot is self-verifying without any launcher
    receipt.
    """
    for stage in STAGES:
        run = root / "raw" / f"{phase}-{stage}"
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest.get("phase") != phase or manifest.get("stage") != stage:
            raise ValueError(f"Mirrored manifest mismatch at {stage}")
        checks = {
            "manifest_scores": _file_sha256(run / "scores.jsonl")
            == manifest["scores_sha256"],
            "manifest_generations": _file_sha256(run / "generations.jsonl")
            == manifest["generations_sha256"],
        }
        if not all(checks.values()):
            raise ValueError(f"Mirrored repair output failed at {stage}: {checks}")
    summary = root / "evidence" / phase / "summary.json"
    if not summary.exists():
        raise ValueError("Mirrored repair evidence summary is missing")
    files = {
        str(path.relative_to(root)): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    return {
        "schema_version": 1,
        "phase": phase,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "file_sha256": files,
        "atomic_t7_snapshot": True,
    }


def mirror(source: str, target_root: Path, phase: str) -> Path:
    target_root.mkdir(parents=True, exist_ok=True)
    final = target_root / phase
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite T7 repair snapshot {final}")
    staging = Path(tempfile.mkdtemp(prefix=f".{phase}.tmp-", dir=target_root))
    try:
        subprocess.run(
            [
                "rsync",
                "--archive",
                "--partial",
                source.rstrip("/") + "/",
                str(staging) + "/",
            ],
            check=True,
        )
        receipt = _verify_snapshot(staging, phase)
        receipt["receipt_sha256"] = _receipt_hash(receipt)
        _write_json_atomic(staging / "mirror-receipt.json", receipt)
        staging.replace(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atomically mirror a completed repair phase to the T7"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Local path or rsync SSH source for the Pod repair root",
    )
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("pilot", "validation", "confirmation"), required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(mirror(args.source, args.target_root, args.phase))


if __name__ == "__main__":
    main()
