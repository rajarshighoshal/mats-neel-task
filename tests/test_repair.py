from __future__ import annotations

import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from frame_stage_binding.repair_analyze import (
    _behavior_audit,
    _figure_svg,
    _instrument_metrics,
    _select_variant,
    _table_rows,
    _validate_cartesian,
)
from frame_stage_binding.repair_core import (
    CONDITIONS,
    DIRECT_DIRECTIVE,
    ORDERS,
    PARAPHRASE_REASONING_DIRECTIVE,
    PLACEBO_DIRECTIVE,
    REASONING_DIRECTIVE,
    STAGES,
    VARIANT_ORDER,
    build_prompt,
    classify_semantic_mode,
    detect_terminal_loop,
    ranked_items,
    source_directives,
)
from frame_stage_binding.repair_recompute import recompute_confirmation
from frame_stage_binding.repair_run import (
    _analysis_code_hashes,
    _batch_plan,
    _code_hashes,
    _receipt_hash,
    _score_batch,
    build_phase_records,
    load_repair_config,
    validate_repair_config,
    verify_selector_receipt,
    verify_validation_receipt,
)
from frame_stage_binding.utils import _canonical_sha256, _file_sha256


CONFIG_PATH = Path("configs/frame_stage_binding_repair_v2.locked.json")


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def item(item_id: str) -> dict[str, str]:
    return {
        "benchmark": "gsm8k",
        "item_id": item_id,
        "question": "What is 1 + 1?",
        "reference": "2",
    }


def selector_receipt(selected: str = "explicit_ra") -> dict:
    if selected != "explicit_ra":
        raise ValueError("Synthetic selector helper supports the frozen first winner")

    def metrics(mass: float) -> dict:
        stages = {
            stage: {
                "worst_cell_median_choice_mass": mass,
                "control_auc_by_order": {order: 1.0 for order in ORDERS},
                "control_headroom": {
                    "estimate": 2.0,
                    "bootstrap_interval": [1.0, 3.0],
                },
                "concordance": 1.0,
                "primary_classifiability": 1.0,
                "loop_rate": 0.0,
                "gates": {
                    "choice_mass": True,
                    "control_auc": True,
                    "positive_headroom": True,
                    "concordance": True,
                    "classifiability": True,
                    "loop_rate": True,
                },
                "pass": True,
            }
            for stage in STAGES
        }
        return {
            "viable": True,
            "worst_cell_median_choice_mass": mass,
            "minimum_stage_control_auc": 1.0,
            "stages": stages,
        }

    value = {
        "schema_version": 1,
        "phase": "pilot",
        "created_at_utc": "2026-08-29T00:00:00+00:00",
        "config_sha256": _canonical_sha256(config()),
        "analysis_code_sha256": _analysis_code_hashes(),
        "input_fingerprint": "input",
        "variant_priority": list(VARIANT_ORDER),
        "variant_metrics": {
            "explicit_ra": metrics(0.3),
            "explicit_12": metrics(0.2),
            "response_mode_field": metrics(0.1),
        },
        "selection_pass": True,
        "selected_variant": selected,
    }
    value["receipt_sha256"] = _receipt_hash(value)
    return value


def validation_receipt(selector: dict) -> dict:
    stages = {
        stage: {
            "cell_median_choice_mass": {
                f"{condition}:{order}": 0.2
                for condition in CONDITIONS
                for order in ORDERS
            },
            "worst_cell_median_choice_mass": 0.2,
            "control_auc_by_order": {order: 1.0 for order in ORDERS},
            "control_headroom": {
                "estimate": 2.0,
                "bootstrap_interval": [1.0, 3.0],
            },
            "concordance": 1.0,
            "concordant_count": 10,
            "concordance_denominator": 10,
            "primary_classifiable_count": 10,
            "primary_generation_count": 10,
            "primary_classifiability": 1.0,
            "loop_count": 0,
            "generation_count": 20,
            "loop_rate": 0.0,
            "gates": {
                "choice_mass": True,
                "control_auc": True,
                "positive_headroom": True,
                "concordance": True,
                "classifiability": True,
                "loop_rate": True,
            },
            "pass": True,
        }
        for stage in STAGES
    }
    value = {
        "schema_version": 1,
        "phase": "validation",
        "created_at_utc": "2026-08-29T00:00:00+00:00",
        "config_sha256": _canonical_sha256(config()),
        "analysis_code_sha256": _analysis_code_hashes(),
        "input_fingerprint": "input",
        "selector_receipt_sha256": selector["receipt_sha256"],
        "selected_variant": selector["selected_variant"],
        "instrument_metrics": {
            "phase": "validation",
            "variant": selector["selected_variant"],
            "stages": stages,
            "pass": True,
        },
        "calibration_constants": {"headroom_H": 2.0, "placebo_noise_N": 0.1},
        "validation_pass": True,
    }
    value["receipt_sha256"] = _receipt_hash(value)
    return value


class RepairConfigTests(unittest.TestCase):
    def test_locked_config_and_exact_split_hashes(self) -> None:
        locked = load_repair_config(CONFIG_PATH)
        phases = locked["phases"]
        self.assertEqual(len(phases["pilot"]["item_lock"]["item_ids"]), 24)
        self.assertEqual(len(phases["validation"]["item_lock"]["item_ids"]), 176)
        self.assertEqual(
            phases["validation"]["generation_item_lock"]["item_ids_sha256"],
            "95962ba9a1b3256aeb3072c96176597a0ab507f7f3da0ad6b5280be7a0c5c0c9",
        )
        self.assertEqual(
            phases["confirmation"]["generation_item_lock"]["item_content_sha256"],
            "6bc267c82fd653c9839b90c01f9f24a5e24590bccd2c8e5214e539d1407fed44",
        )
        pilot = set(phases["pilot"]["item_lock"]["item_ids"])
        validation = set(phases["validation"]["item_lock"]["item_ids"])
        self.assertFalse(pilot & validation)
        self.assertEqual(len(pilot | validation), 200)

    def test_identity_mutations_fail(self) -> None:
        for mutation in ("model", "split", "namespace", "source"):
            with self.subTest(mutation=mutation):
                changed = config()
                if mutation == "model":
                    changed["models"]["base"]["revision"] = "0" * 40
                elif mutation == "split":
                    lock = changed["phases"]["pilot"]["item_lock"]
                    lock["item_ids"][0] = "gsm8k:train:9999"
                    lock["item_ids_sha256"] = _canonical_sha256(
                        sorted(lock["item_ids"])
                    )
                elif mutation == "namespace":
                    changed["phases"]["pilot"]["selection_namespace"] = "other"
                else:
                    changed["source_config_sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    validate_repair_config(changed)

    def test_ranking_is_input_order_invariant(self) -> None:
        values = [item(f"id:{index}") for index in range(20)]
        forward = [row["item_id"] for row in ranked_items(values, 7, "salt")]
        reverse = [row["item_id"] for row in ranked_items(reversed(values), 7, "salt")]
        self.assertEqual(forward, reverse)


class RepairPromptTests(unittest.TestCase):
    def test_source_directives_and_order_are_exact(self) -> None:
        expected = {
            "request_cue": (REASONING_DIRECTIVE, DIRECT_DIRECTIVE),
            "notes_cue": (DIRECT_DIRECTIVE, REASONING_DIRECTIVE),
            "all_reason": (REASONING_DIRECTIVE, REASONING_DIRECTIVE),
            "all_direct": (DIRECT_DIRECTIVE, DIRECT_DIRECTIVE),
            "placebo_a": (DIRECT_DIRECTIVE, PLACEBO_DIRECTIVE),
            "placebo_b": (PLACEBO_DIRECTIVE, DIRECT_DIRECTIVE),
            "para_request_b": (PARAPHRASE_REASONING_DIRECTIVE, DIRECT_DIRECTIVE),
            "para_notes_b": (DIRECT_DIRECTIVE, PARAPHRASE_REASONING_DIRECTIVE),
        }
        for condition, directives in expected.items():
            self.assertEqual(source_directives(condition), directives)
            first = build_prompt("q", condition, "request_first", "explicit_ra")
            second = build_prompt("q", condition, "notes_first", "explicit_ra")
            self.assertEqual(
                sorted(first.split("\n\n")[1:3]), sorted(second.split("\n\n")[1:3])
            )
            self.assertNotEqual(first, second)

    def test_recalibration_config_validates_and_items_are_fresh(self) -> None:
        config = load_repair_config(
            Path("configs/frame_stage_binding_recalibration.locked.json")
        )
        recalibration = config["phases"]["recalibration"]
        consumed = set(config["phases"]["pilot"]["item_lock"]["item_ids"]) | set(
            config["phases"]["validation"]["item_lock"]["item_ids"]
        )
        self.assertEqual(recalibration["item_lock"]["item_count"], 352)
        self.assertFalse(
            consumed & set(recalibration["item_lock"]["item_ids"])
        )
        # v2 config must still validate without the recalibration phase
        load_repair_config(Path("configs/frame_stage_binding_repair_v2.locked.json"))

    def test_recalibration_constants(self) -> None:
        from frame_stage_binding.repair_analyze import recalibration_constants

        scores = []
        for stage in ("base", "rlvr"):
            for item in ("i0", "i1"):
                for order in ("request_first", "notes_first"):
                    base = {
                        "stage": stage,
                        "item_id": item,
                        "order": order,
                        "variant": "explicit_12",
                    }
                    scores.append({**base, "condition": "request_cue", "choice_score": 1.0})
                    scores.append({**base, "condition": "para_request_b", "choice_score": 1.5})
                    scores.append({**base, "condition": "notes_cue", "choice_score": -1.0})
                    scores.append({**base, "condition": "para_notes_b", "choice_score": -1.25})
        constants = recalibration_constants(scores)
        self.assertAlmostEqual(constants["N_max"], 0.5)
        self.assertAlmostEqual(constants["N_mean"], 0.375)
        self.assertAlmostEqual(constants["psi_fresh"]["base"], -2.0)
        self.assertAlmostEqual(constants["psi_fresh"]["rlvr"], -2.0)


    def test_phase_record_counts_and_unique_keys(self) -> None:
        locked = config()
        for phase, expected in {
            "pilot": (864, 576),
            "validation": (2112, 384),
            "confirmation": (5276, 256),
        }.items():
            count = locked["phases"][phase]["item_lock"]["item_count"]
            gen_count = locked["phases"][phase]["generation_item_lock"]["item_count"]
            items = [
                item(f"gsm8k:{locked['phases'][phase]['split']}:{i}")
                for i in range(count)
            ]
            generated = items[:gen_count]
            variants = list(VARIANT_ORDER) if phase == "pilot" else ["explicit_ra"]
            scores, generations = build_phase_records(
                items, generated, locked, phase, "base", variants
            )
            self.assertEqual((len(scores), len(generations)), expected)
            keys = {
                (row["variant"], row["item_id"], row["condition"], row["order"])
                for row in scores
            }
            self.assertEqual(len(keys), len(scores))


class RepairParserTests(unittest.TestCase):
    def test_semantic_modes_for_all_variants(self) -> None:
        choices = {
            "explicit_ra": ("R", "A"),
            "explicit_12": ("1", "2"),
            "response_mode_field": ("R", "A"),
        }
        for variant, (reason, direct) in choices.items():
            with self.subTest(variant=variant):
                parsed = classify_semantic_mode(
                    f"{reason}\nREASONING: work\nFINAL: 2", variant
                )
                self.assertEqual(
                    (parsed["semantic_mode"], parsed["leading_choice"]),
                    ("reason", "reason"),
                )
                parsed = classify_semantic_mode(f"{direct}: FINAL: 2", variant)
                self.assertEqual(
                    (parsed["semantic_mode"], parsed["leading_choice"]),
                    ("direct", "direct"),
                )

    def test_semantic_failures_are_not_silently_classified(self) -> None:
        failures = [
            "R\nno marker at all",
            "A\nFINAL:",
            "A\nFINAL:\n\n",
            "I cannot comply\nFINAL: 2",
        ]
        for response in failures:
            with self.subTest(response=response):
                self.assertFalse(
                    classify_semantic_mode(response, "explicit_ra")["classifiable"]
                )
        # v2: last FINAL wins; substantive text before it means reason.
        self.assertEqual(
            classify_semantic_mode("R\nREASONING:\nFINAL: 2", "explicit_ra")[
                "semantic_mode"
            ],
            "reason",
        )
        self.assertEqual(
            classify_semantic_mode("A\ntext\nFINAL: 2", "explicit_ra")[
                "semantic_mode"
            ],
            "reason",
        )
        self.assertEqual(
            classify_semantic_mode("R\nFINAL: 2\nREASONING: late", "explicit_ra")[
                "semantic_mode"
            ],
            "direct",
        )
        self.assertEqual(
            classify_semantic_mode("A\nFINAL: 2", "explicit_ra")["semantic_mode"],
            "direct",
        )
        # mid-line FINAL is classifiable under v2
        self.assertEqual(
            classify_semantic_mode(
                "R\nREASONING:\nso 1+1 = 2. FINAL: 2", "explicit_ra"
            )["semantic_mode"],
            "reason",
        )
        # answer on the line after an empty FINAL marker
        self.assertEqual(
            classify_semantic_mode("A\nFINAL:\n42", "explicit_ra")["final_answer"],
            "42",
        )
        self.assertIsNone(
            classify_semantic_mode("Rationale\nFINAL: 2", "explicit_ra")[
                "leading_choice"
            ]
        )
        self.assertIsNone(
            classify_semantic_mode("12\nFINAL: 2", "explicit_12")["leading_choice"]
        )

    def test_terminal_loop_boundaries_and_tie_break(self) -> None:
        self.assertFalse(detect_terminal_loop([7] * 31)["detected"])
        self.assertTrue(detect_terminal_loop([7] * 32)["detected"])
        self.assertFalse(detect_terminal_loop([1, 2] * 16 + [9])["detected"])
        period_two = detect_terminal_loop([1, 2] * 16)
        self.assertTrue(period_two["detected"])
        self.assertEqual(period_two["period"], 2)
        period_128 = detect_terminal_loop(list(range(128)) * 3)
        self.assertEqual(period_128["period"], 128)


class RepairReceiptTests(unittest.TestCase):
    def test_selector_and_validation_receipts_are_strict(self) -> None:
        selector = selector_receipt()
        self.assertEqual(verify_selector_receipt(selector, config()), "explicit_ra")
        validation = validation_receipt(selector)
        self.assertEqual(
            verify_validation_receipt(validation, config(), selector), "explicit_ra"
        )
        changed = copy.deepcopy(selector)
        changed["variant_metrics"] = {"nested": {"theta": 1}}
        changed["receipt_sha256"] = _receipt_hash(changed)
        with self.assertRaises(ValueError):
            verify_selector_receipt(changed, config())
        changed = copy.deepcopy(selector)
        changed["selected_variant"] = "explicit_12"
        changed["receipt_sha256"] = _receipt_hash(changed)
        with self.assertRaises(ValueError):
            verify_selector_receipt(changed, config())
        changed = copy.deepcopy(selector)
        changed["analysis_code_sha256"] = {}
        changed["receipt_sha256"] = _receipt_hash(changed)
        with self.assertRaises(ValueError):
            verify_selector_receipt(changed, config())
        changed = copy.deepcopy(selector)
        changed["variant_metrics"]["explicit_ra"]["stages"]["base"]["gates"][
            "choice_mass"
        ] = False
        changed["receipt_sha256"] = _receipt_hash(changed)
        with self.assertRaises(ValueError):
            verify_selector_receipt(changed, config())
        changed = copy.deepcopy(validation)
        changed["extra"] = 1
        changed["receipt_sha256"] = _receipt_hash(changed)
        with self.assertRaises(ValueError):
            verify_validation_receipt(changed, config(), selector)
        changed = copy.deepcopy(validation)
        changed["instrument_metrics"]["stages"]["base"]["gates"]["choice_mass"] = False
        changed["receipt_sha256"] = _receipt_hash(changed)
        with self.assertRaises(ValueError):
            verify_validation_receipt(changed, config(), selector)
        changed = copy.deepcopy(validation)
        changed["instrument_metrics"] = {}
        changed["receipt_sha256"] = _receipt_hash(changed)
        with self.assertRaises(ValueError):
            verify_validation_receipt(changed, config(), selector)

class RepairStatisticsTests(unittest.TestCase):
    def _small_config(self) -> dict:
        value = config()
        value["analysis"]["bootstrap_draws"] = 200
        for phase in ("validation", "confirmation"):
            value["phases"][phase]["item_lock"]["item_ids"] = ["i0", "i1"]
            value["phases"][phase]["item_lock"]["item_count"] = 2
        return value

    @staticmethod
    def _score_row(
        phase: str, stage: str, item_id: str, condition: str, order: str, score: float
    ) -> dict:
        return {
            "phase": phase,
            "stage": stage,
            "variant": "explicit_ra",
            "item_id": item_id,
            "condition": condition,
            "order": order,
            "reason_logit": score,
            "direct_logit": 0.0,
            "choice_score": score,
        }

    def test_independent_confirmation_recomputation(self) -> None:
        small = self._small_config()
        validation = []
        confirmation = []
        psi = {"base": 0.5, "sft": 0.0, "dpo": -0.5, "rlvr": -1.5}
        for stage in STAGES:
            for item_id in ("i0", "i1"):
                for order in ORDERS:
                    for condition, score in {
                        "request_cue": 0.0,
                        "notes_cue": psi[stage],
                        "all_reason": 10.0,
                        "all_direct": 0.0,
                        "placebo_a": 0.1,
                        "placebo_b": 0.0,
                    }.items():
                        validation.append(
                            self._score_row(
                                "validation", stage, item_id, condition, order, score
                            )
                        )
                    for condition, score in {
                        "request_cue": 0.0,
                        "notes_cue": psi[stage],
                    }.items():
                        confirmation.append(
                            self._score_row(
                                "confirmation", stage, item_id, condition, order, score
                            )
                        )
        result = recompute_confirmation(confirmation, validation, small, "explicit_ra")
        self.assertEqual(result["theta"], -2.0)
        self.assertTrue(result["statistical_support"])
        self.assertTrue(all(result["criteria"].values()))
        duplicate = confirmation + [dict(confirmation[0])]
        with self.assertRaises(ValueError):
            recompute_confirmation(duplicate, validation, small, "explicit_ra")

        strict_noise = copy.deepcopy(validation)
        for row in strict_noise:
            if row["condition"] == "placebo_a":
                row["reason_logit"] = 1.0
                row["choice_score"] = 1.0
        no_support = recompute_confirmation(
            confirmation, strict_noise, small, "explicit_ra"
        )
        self.assertFalse(no_support["criteria"]["placebo_rule"])

    def test_selector_tie_priority_and_source_blind_shape(self) -> None:
        metrics = {
            variant: {
                "pass": True,
                "worst_cell_median_choice_mass": 0.2,
                "minimum_stage_control_auc": 0.9,
            }
            for variant in VARIANT_ORDER
        }
        self.assertEqual(_select_variant(metrics), "explicit_ra")
        metrics["explicit_12"]["worst_cell_median_choice_mass"] = 0.3
        self.assertEqual(_select_variant(metrics), "explicit_12")
        self.assertFalse(
            any("psi" in key or "theta" in key for key in metrics["explicit_12"])
        )

    def test_pilot_gate_boundaries_and_loop_failure(self) -> None:
        small = config()
        small["analysis"]["bootstrap_draws"] = 100
        small["phases"]["pilot"]["item_lock"]["item_ids"] = ["i0"]
        small["phases"]["pilot"]["generation_item_lock"]["item_ids"] = ["i0"]
        scores = []
        generations = []
        values = {
            "request_cue": 1.0,
            "notes_cue": -1.0,
            "all_reason": 2.0,
            "all_direct": -2.0,
            "placebo_a": 0.0,
            "placebo_b": 0.0,
        }
        for stage in STAGES:
            for condition in CONDITIONS:
                for order in ORDERS:
                    scores.append(
                        {
                            "stage": stage,
                            "variant": "explicit_ra",
                            "item_id": "i0",
                            "condition": condition,
                            "order": order,
                            "choice_score": values[condition],
                            "choice_mass": 0.05,
                        }
                    )
                    if condition in (
                        "request_cue",
                        "notes_cue",
                        "all_reason",
                        "all_direct",
                    ):
                        mode = (
                            "reason"
                            if condition in ("request_cue", "all_reason")
                            else "direct"
                        )
                        generations.append(
                            {
                                "stage": stage,
                                "variant": "explicit_ra",
                                "item_id": "i0",
                                "condition": condition,
                                "order": order,
                                "semantic_mode": mode,
                                "classifiable": True,
                                "leading_choice": mode,
                                "choice_matches_semantic": True,
                                "terminal_loop": {"detected": False},
                            }
                        )
        passed = _instrument_metrics(scores, generations, small, "pilot", "explicit_ra")
        self.assertTrue(passed["pass"])
        scores[0]["choice_mass"] = 0.049
        self.assertFalse(
            _instrument_metrics(scores, generations, small, "pilot", "explicit_ra")[
                "pass"
            ]
        )
        scores[0]["choice_mass"] = 0.05
        # loops only fail the gate at behavior stages (post-training); a Base
        # loop is reported but not gated
        base_loop_index = next(
            index
            for index, row in enumerate(generations)
            if row["stage"] == "base" and row["condition"] == "request_cue"
        )
        generations[base_loop_index]["terminal_loop"] = {"detected": True}
        self.assertTrue(
            _instrument_metrics(scores, generations, small, "pilot", "explicit_ra")[
                "pass"
            ]
        )
        sft_loop_index = next(
            index
            for index, row in enumerate(generations)
            if row["stage"] == "sft" and row["condition"] == "request_cue"
        )
        generations[sft_loop_index]["terminal_loop"] = {"detected": True}
        self.assertFalse(
            _instrument_metrics(scores, generations, small, "pilot", "explicit_ra")[
                "pass"
            ]
        )

    def test_cartesian_validation_rejects_duplicates(self) -> None:
        small = config()
        small["phases"]["pilot"]["item_lock"]["item_ids"] = ["i0"]
        small["phases"]["pilot"]["generation_item_lock"]["item_ids"] = ["i0"]
        scores = []
        generations = []
        for stage in STAGES:
            for variant in VARIANT_ORDER:
                for condition in CONDITIONS:
                    for order in ORDERS:
                        scores.append(
                            {
                                "stage": stage,
                                "variant": variant,
                                "item_id": "i0",
                                "condition": condition,
                                "order": order,
                            }
                        )
                        if condition in (
                            "request_cue",
                            "notes_cue",
                            "all_reason",
                            "all_direct",
                        ):
                            generations.append(
                                {
                                    "stage": stage,
                                    "variant": variant,
                                    "item_id": "i0",
                                    "condition": condition,
                                    "order": order,
                                    "prompt": "p",
                                    "prompt_sha256": "h",
                                }
                            )
                for row in scores[-12:]:
                    row["prompt"] = "p"
                    row["prompt_sha256"] = "h"
                    row["prompt_input_token_ids"] = [1]
        _validate_cartesian(scores, generations, small, "pilot", list(VARIANT_ORDER))
        with self.assertRaises(ValueError):
            _validate_cartesian(
                scores + [dict(scores[0])],
                generations,
                small,
                "pilot",
                list(VARIANT_ORDER),
            )


class RepairScoreTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "torch is tested on the GPU preflight host"
    )
    def test_score_batch_uses_full_vocabulary_and_rejects_nonfinite(self) -> None:
        import torch

        class Model:
            def __init__(self, logits: object) -> None:
                self.logits = logits

            def __call__(self, **_: object) -> object:
                return type("Output", (), {"logits": self.logits})()

        logits = torch.tensor([[[0.0, 2.0, 1.0, -1.0]]])
        result = _score_batch(Model(logits), {}, 1, 2)[0]
        denominator = sum(math.exp(x) for x in [0, 2, 1, -1])
        expected_mass = (math.exp(2.0) + math.exp(1.0)) / denominator
        self.assertAlmostEqual(result["choice_score"], 1.0)
        self.assertAlmostEqual(result["choice_mass"], expected_mass, places=6)
        shifted = _score_batch(Model(logits + 100), {}, 1, 2)[0]
        self.assertTrue(
            math.isclose(
                result["choice_mass"],
                shifted["choice_mass"],
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
        )
        bad = logits.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            _score_batch(Model(bad), {}, 1, 2)


class RepairLauncherTests(unittest.TestCase):
    def test_programmatic_figure_and_behavior_audit(self) -> None:
        summary = {
            "phase": "confirmation",
            "estimands": {
                "psi": {stage: float(index) for index, stage in enumerate(STAGES)}
            },
        }
        svg = _figure_svg(summary)
        self.assertIn("<svg", svg)
        rows = [
            {
                "stage": "base",
                "variant": "explicit_ra",
                "item_id": f"i{index}",
                "condition": "request_cue",
                "order": "request_first",
                "response": "A\nFINAL: 2",
                "semantic_mode": "direct",
                "classifiable": index != 0,
                "semantic_failure_categories": [] if index else ["failure"],
                "terminal_loop": {"detected": False},
                "cap_hit": False,
            }
            for index in range(30)
        ]
        audit = _behavior_audit(rows, config(), "confirmation")
        self.assertTrue(any(row["selection"] == "failure" for row in audit))
        self.assertGreaterEqual(len(audit), 20)

    def test_pilot_artifact_rows_use_frozen_order_after_json_sorting(self) -> None:
        stage_values = {
            "worst_cell_median_choice_mass": 0.2,
            "concordance": 0.9,
            "primary_classifiability": 1.0,
            "loop_rate": 0.0,
            "pass": True,
        }
        summary = {
            "phase": "pilot",
            "instrument_metrics": {
                variant: {
                    "stages": {stage: dict(stage_values) for stage in reversed(STAGES)}
                }
                for variant in reversed(VARIANT_ORDER)
            },
        }
        rows = _table_rows(json.loads(json.dumps(summary, sort_keys=True)))
        self.assertEqual(
            [(row["variant"], row["stage"]) for row in rows],
            [(variant, stage) for variant in VARIANT_ORDER for stage in STAGES],
        )

    def test_mirror_snapshot_rejects_drift(self) -> None:
        from scripts.mirror_repair_artifacts import _verify_snapshot

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in STAGES:
                run = root / "raw" / f"pilot-{stage}"
                run.mkdir(parents=True)
                manifest = {
                    "phase": "pilot",
                    "stage": stage,
                    "scores_sha256": "",
                    "generations_sha256": "",
                }
                for name, content in (
                    ("scores.jsonl", "{}\n"),
                    ("generations.jsonl", "{}\n"),
                ):
                    (run / name).write_text(content)
                manifest["scores_sha256"] = _file_sha256(run / "scores.jsonl")
                manifest["generations_sha256"] = _file_sha256(
                    run / "generations.jsonl"
                )
                (run / "manifest.json").write_text(json.dumps(manifest))
            evidence = root / "evidence" / "pilot"
            evidence.mkdir(parents=True)
            (evidence / "summary.json").write_text("{}\n")
            self.assertTrue(_verify_snapshot(root, "pilot")["atomic_t7_snapshot"])
            (root / "raw" / "pilot-base" / "scores.jsonl").write_text("drift\n")
            with self.assertRaises(ValueError):
                _verify_snapshot(root, "pilot")


if __name__ == "__main__":
    unittest.main()
