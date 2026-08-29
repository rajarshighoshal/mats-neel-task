from __future__ import annotations

import math
import random
import statistics
from typing import Any, Iterable


STAGES = ("base", "sft", "dpo", "rlvr")
ORDERS = ("request_first", "notes_first")
PRIMARY = ("request_cue", "notes_cue")
BOOTSTRAP_SEED_OFFSET = 9100


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Independent recomputation found non-numeric {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Independent recomputation found non-finite {name}")
    return result


def _score(record: dict[str, Any]) -> float:
    reason = _finite(record.get("reason_logit"), "reason_logit")
    direct = _finite(record.get("direct_logit"), "direct_logit")
    derived = reason - direct
    stored = _finite(record.get("choice_score"), "choice_score")
    if not math.isclose(stored, derived, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Independent recomputation found a stale choice_score")
    return derived


def _index(
    records: Iterable[dict[str, Any]],
    config: dict[str, Any],
    phase: str,
    variant: str,
) -> dict[tuple[str, str, str, str], float]:
    phase_config = config["phases"][phase]
    expected_conditions = tuple(phase_config["conditions"])
    expected_items = set(phase_config["item_lock"]["item_ids"])
    cells: dict[tuple[str, str, str, str], float] = {}
    for record in records:
        if record.get("phase") != phase:
            raise ValueError("Independent recomputation received the wrong phase")
        if record.get("variant") != variant:
            raise ValueError("Independent recomputation received the wrong variant")
        stage = record.get("stage")
        item_id = record.get("item_id")
        condition = record.get("condition")
        order = record.get("order")
        if (
            stage not in STAGES
            or item_id not in expected_items
            or condition not in expected_conditions
            or order not in ORDERS
        ):
            raise ValueError("Independent recomputation found an unexpected score cell")
        key = (stage, item_id, condition, order)
        if key in cells:
            raise ValueError("Independent recomputation found a duplicate score cell")
        cells[key] = _score(record)

    expected = {
        (stage, item_id, condition, order)
        for stage in STAGES
        for item_id in expected_items
        for condition in expected_conditions
        for order in ORDERS
    }
    actual = set(cells)
    if actual != expected:
        raise ValueError(
            "Independent recomputation found incomplete score cells: "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
    return cells


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_interval(values: list[float], draws: int, seed: int) -> list[float]:
    if not values or draws <= 0:
        raise ValueError(
            "Independent recomputation requires values and bootstrap draws"
        )
    rng = random.Random(seed)
    estimates = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(draws)
    ]
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _calibration_constants(
    cells: dict[tuple[str, str, str, str], float],
    item_ids: list[str],
) -> tuple[float, float, dict[str, float]]:
    headroom_values = [
        cells[(stage, item_id, "all_reason", order)]
        - cells[(stage, item_id, "all_direct", order)]
        for stage in ("base", "rlvr")
        for item_id in item_ids
        for order in ORDERS
    ]
    placebo_means = {
        f"{stage}:{order}": statistics.fmean(
            cells[(stage, item_id, "placebo_a", order)]
            - cells[(stage, item_id, "placebo_b", order)]
            for item_id in item_ids
        )
        for stage in ("base", "rlvr")
        for order in ORDERS
    }
    return (
        statistics.fmean(headroom_values),
        max(abs(value) for value in placebo_means.values()),
        placebo_means,
    )


def recompute_validation_constants(
    validation_scores: list[dict[str, Any]],
    config: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    cells = _index(validation_scores, config, "validation", variant)
    item_ids = list(config["phases"]["validation"]["item_lock"]["item_ids"])
    headroom, placebo_noise, placebo_means = _calibration_constants(cells, item_ids)
    return {
        "headroom_H": headroom,
        "placebo_noise_N": placebo_noise,
        "endpoint_order_placebo_means": placebo_means,
        "definitions": {
            "headroom_H": (
                "mean of S(all_reason)-S(all_direct) over Base and RLVR, "
                "validation items, and both block orders"
            ),
            "placebo_noise_N": (
                "maximum over Base/RLVR and block order of the absolute mean "
                "S(placebo_a)-S(placebo_b) across validation items"
            ),
        },
    }


def recompute_confirmation(
    confirmation_scores: list[dict[str, Any]],
    validation_scores: list[dict[str, Any]],
    config: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    confirm = _index(confirmation_scores, config, "confirmation", variant)
    calibration = recompute_validation_constants(validation_scores, config, variant)
    item_ids = list(config["phases"]["confirmation"]["item_lock"]["item_ids"])

    order_effects = {
        stage: {
            order: {
                item_id: confirm[(stage, item_id, "notes_cue", order)]
                - confirm[(stage, item_id, "request_cue", order)]
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
    theta = statistics.fmean(theta_items.values())
    theta_by_order = {
        order: statistics.fmean(
            order_effects["rlvr"][order][item_id]
            - order_effects["base"][order][item_id]
            for item_id in item_ids
        )
        for order in ORDERS
    }
    interval = _bootstrap_interval(
        list(theta_items.values()),
        int(config["analysis"]["bootstrap_draws"]),
        int(config["seed"]) + BOOTSTRAP_SEED_OFFSET,
    )
    headroom = calibration["headroom_H"]
    placebo_noise = calibration["placebo_noise_N"]
    criteria = {
        "theta_interval_below_zero": interval[1] < 0.0,
        "theta_negative_both_orders": all(
            value < 0.0 for value in theta_by_order.values()
        ),
        "headroom_rule": -theta
        >= float(config["analysis"]["headroom_fraction_min"]) * headroom,
        "placebo_rule": -theta
        > float(config["analysis"]["placebo_noise_multiplier"]) * placebo_noise,
    }
    return {
        "score_definition": "S = reason_logit - direct_logit",
        "item_count": len(item_ids),
        "psi": psi,
        "theta": theta,
        "theta_bootstrap_interval": interval,
        "theta_by_order": theta_by_order,
        "headroom_H": headroom,
        "placebo_noise_N": placebo_noise,
        "criteria": criteria,
        "statistical_support": all(criteria.values()),
    }
