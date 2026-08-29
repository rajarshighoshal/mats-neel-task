from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .repair_core import STAGES
from .repair_review import verify_review_receipt
from .repair_run import load_repair_config
from .utils import _canonical_sha256, _file_sha256, _write_json_atomic


def render(summary: dict[str, Any], review: dict[str, Any]) -> str:
    estimands = summary["estimands"]
    rows = [
        "<!-- PROGRAMMATICALLY GENERATED; do not edit observed numbers manually. -->",
        "# Repaired mode-choice results",
        "",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Confirmatory support: `{str(summary['confirmatory_support']).lower()}`",
        f"- Human review receipt: `{review['receipt_sha256']}`",
        "",
        "| Stage | psi |",
        "|---|---:|",
    ]
    rows.extend(
        f"| {stage} | {float(estimands['psi'][stage]):.6g} |" for stage in STAGES
    )
    rows.extend(
        [
            "",
            f"Theta: `{float(estimands['theta']):.6g}`.",
            "",
            "Theta 95% paired-bootstrap interval: "
            f"`[{float(estimands['theta_bootstrap_interval'][0]):.6g}, "
            f"{float(estimands['theta_bootstrap_interval'][1]):.6g}]`.",
            "",
            "![Programmatically generated result figure](figure.svg)",
            "",
        ]
    )
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render reviewed repair results without manual number entry"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/frame_stage_binding_repair_v2.locked.json"),
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_repair_config(args.config)
    review = verify_review_receipt(args.review_receipt, config, args.artifact_root)
    summary_path = args.artifact_root / "evidence/confirmation/summary.json"
    summary = json.loads(summary_path.read_text())
    if not review["approved_for_reporting"]:
        raise ValueError("Repair results are not approved for reporting")
    text = render(summary, review)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("x") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)
    provenance = {
        "schema_version": 1,
        "config_sha256": _canonical_sha256(config),
        "summary_sha256": _file_sha256(summary_path),
        "review_receipt_sha256": review["receipt_sha256"],
        "renderer_sha256": _file_sha256(Path(__file__)),
        "output_sha256": _file_sha256(args.output),
    }
    _write_json_atomic(args.provenance_output, provenance)


if __name__ == "__main__":
    main()
