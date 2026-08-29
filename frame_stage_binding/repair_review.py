from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .repair_analyze import verify_evidence_bundle
from .repair_core import STAGES
from .repair_run import _receipt_hash, load_repair_config
from .utils import _canonical_sha256, _file_sha256, _write_json_atomic


REVIEW_ACTIVITY = "Tulu mode-choice repair evidence review"


def create_review_receipt(
    config: dict[str, Any],
    artifact_root: Path,
    checks: dict[str, bool],
) -> dict[str, Any]:
    if set(checks) != {
        "inspected_behavior_audit",
        "inspected_metric_code",
        "inspected_controls",
        "inspected_independent_recomputation",
        "understands_limitations",
    } or not all(checks.values()):
        raise ValueError("Every applicant review check must be explicit and true")
    selector = json.loads(
        (artifact_root / "evidence/pilot/selector-receipt.json").read_text()
    )
    validation = json.loads(
        (artifact_root / "evidence/validation/validation-receipt.json").read_text()
    )
    confirmation_runs = [
        artifact_root / "raw" / f"confirmation-{stage}" for stage in STAGES
    ]
    validation_runs = [
        artifact_root / "raw" / f"validation-{stage}" for stage in STAGES
    ]
    pilot_runs = [artifact_root / "raw" / f"pilot-{stage}" for stage in STAGES]
    if selector != verify_evidence_bundle(
        artifact_root / "evidence/pilot", config, "pilot", pilot_runs
    ):
        raise ValueError("Pilot evidence does not match its selector receipt")
    if validation != verify_evidence_bundle(
        artifact_root / "evidence/validation",
        config,
        "validation",
        validation_runs,
        selector_receipt=selector,
    ):
        raise ValueError("Validation evidence does not match its receipt")
    confirmation = verify_evidence_bundle(
        artifact_root / "evidence/confirmation",
        config,
        "confirmation",
        confirmation_runs,
        selector_receipt=selector,
        validation_receipt=validation,
        validation_run_dirs=validation_runs,
    )
    receipt = {
        "schema_version": 1,
        "activity": REVIEW_ACTIVITY,
        "config_sha256": _canonical_sha256(config),
        "confirmation_receipt_sha256": confirmation["receipt_sha256"],
        "behavior_audit_sha256": _file_sha256(
            artifact_root / "evidence/confirmation/behavior-audit.jsonl"
        ),
        "independent_recomputation_sha256": _file_sha256(
            artifact_root / "evidence/confirmation/independent-recomputation.json"
        ),
        "review_code_sha256": _file_sha256(Path(__file__)),
        "checks": checks,
        "review_complete": True,
        "approved_for_reporting": True,
        "supported_claim_approved": confirmation.get("confirmatory_support") is True,
    }
    receipt["receipt_sha256"] = _receipt_hash(receipt)
    return receipt


def verify_review_receipt(
    path: Path, config: dict[str, Any], artifact_root: Path
) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    required = {
        "schema_version",
        "activity",
        "config_sha256",
        "confirmation_receipt_sha256",
        "behavior_audit_sha256",
        "independent_recomputation_sha256",
        "review_code_sha256",
        "checks",
        "review_complete",
        "approved_for_reporting",
        "supported_claim_approved",
        "receipt_sha256",
    }
    if set(receipt) != required or receipt.get("receipt_sha256") != _receipt_hash(
        receipt
    ):
        raise ValueError("Repair review receipt has an invalid schema or hash")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("activity") != REVIEW_ACTIVITY
        or receipt.get("config_sha256") != _canonical_sha256(config)
        or receipt.get("review_code_sha256") != _file_sha256(Path(__file__))
        or receipt.get("review_complete") is not True
        or receipt.get("approved_for_reporting") is not True
        or set(receipt.get("checks", {}))
        != {
            "inspected_behavior_audit",
            "inspected_metric_code",
            "inspected_controls",
            "inspected_independent_recomputation",
            "understands_limitations",
        }
        or not all(receipt["checks"].values())
    ):
        raise ValueError("Repair review receipt failed its fixed checks")
    selector = json.loads(
        (artifact_root / "evidence/pilot/selector-receipt.json").read_text()
    )
    validation = json.loads(
        (artifact_root / "evidence/validation/validation-receipt.json").read_text()
    )
    confirmation = verify_evidence_bundle(
        artifact_root / "evidence/confirmation",
        config,
        "confirmation",
        [artifact_root / "raw" / f"confirmation-{stage}" for stage in STAGES],
        selector_receipt=selector,
        validation_receipt=validation,
        validation_run_dirs=[
            artifact_root / "raw" / f"validation-{stage}" for stage in STAGES
        ],
    )
    checks = {
        "confirmation": receipt["confirmation_receipt_sha256"]
        == confirmation["receipt_sha256"],
        "audit": receipt["behavior_audit_sha256"]
        == _file_sha256(artifact_root / "evidence/confirmation/behavior-audit.jsonl"),
        "recompute": receipt["independent_recomputation_sha256"]
        == _file_sha256(
            artifact_root / "evidence/confirmation/independent-recomputation.json"
        ),
        "claim": receipt["supported_claim_approved"]
        is (confirmation.get("confirmatory_support") is True),
    }
    if not all(checks.values()):
        raise ValueError(f"Repair review receipt drifted: {checks}")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record the applicant's repair-evidence review"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/frame_stage_binding_repair_v2.locked.json"),
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspected-behavior-audit", action="store_true")
    parser.add_argument("--inspected-metric-code", action="store_true")
    parser.add_argument("--inspected-controls", action="store_true")
    parser.add_argument("--inspected-independent-recomputation", action="store_true")
    parser.add_argument("--understands-limitations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = {
        "inspected_behavior_audit": args.inspected_behavior_audit,
        "inspected_metric_code": args.inspected_metric_code,
        "inspected_controls": args.inspected_controls,
        "inspected_independent_recomputation": args.inspected_independent_recomputation,
        "understands_limitations": args.understands_limitations,
    }
    receipt = create_review_receipt(
        load_repair_config(args.config),
        args.artifact_root,
        checks,
    )
    _write_json_atomic(args.output, receipt)


if __name__ == "__main__":
    main()
