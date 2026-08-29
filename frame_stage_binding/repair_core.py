from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable


REASONING_DIRECTIVE = (
    "Solve the problem with explicit step-by-step reasoning before the final answer."
)
DIRECT_DIRECTIVE = "Answer the problem directly without showing step-by-step reasoning."
PLACEBO_DIRECTIVE = (
    "Give only the final answer without an explanation or intermediate work."
)
PARAPHRASE_REASONING_DIRECTIVE = (
    "Think the problem through one step at a time before giving your final answer."
)
CANONICAL_TEMPLATE = (
    "PROBLEM:\n{question}\n\n{first_block}\n\n{second_block}\n\n{choice_block}"
)

CONDITIONS = (
    "request_cue",
    "notes_cue",
    "all_reason",
    "all_direct",
    "placebo_a",
    "placebo_b",
)
PRIMARY_CONDITIONS = ("request_cue", "notes_cue")
CONTROL_CONDITIONS = ("all_reason", "all_direct")
GENERATION_CONDITIONS = PRIMARY_CONDITIONS + CONTROL_CONDITIONS
RECALIBRATION_CONDITIONS = (
    "request_cue",
    "notes_cue",
    "para_request_b",
    "para_notes_b",
)
ORDERS = ("request_first", "notes_first")
STAGES = ("base", "sft", "dpo", "rlvr")
VARIANT_ORDER = ("explicit_ra", "explicit_12", "response_mode_field")

VARIANTS: dict[str, dict[str, Any]] = {
    "explicit_ra": {
        "reason": "R",
        "direct": "A",
        "choice_block": (
            "MODE CHOICE:\n"
            "R = begin with R, then write a non-empty REASONING: section and end "
            "with FINAL: followed by only the answer.\n"
            "A = begin with A, then write FINAL: followed by only the answer and "
            "do not show reasoning.\n"
            "Start your response with R or A, then continue in that mode.\n"
            "CHOICE:\n"
        ),
    },
    "explicit_12": {
        "reason": "1",
        "direct": "2",
        "choice_block": (
            "MODE CHOICE:\n"
            "1 = begin with 1, then write a non-empty REASONING: section and end "
            "with FINAL: followed by only the answer.\n"
            "2 = begin with 2, then write FINAL: followed by only the answer and "
            "do not show reasoning.\n"
            "Start your response with 1 or 2, then continue in that mode.\n"
            "CHOICE:\n"
        ),
    },
    "response_mode_field": {
        "reason": "R",
        "direct": "A",
        "choice_block": (
            "OUTPUT RECORD:\n"
            "response_mode R means: after the field value, write a non-empty "
            "REASONING: section and end with FINAL: followed by only the answer.\n"
            "response_mode A means: after the field value, write FINAL: followed "
            "by only the answer and do not show reasoning.\n"
            "Enter R or A as the response_mode value, then continue in that "
            "mode.\n"
            "response_mode:\n"
        ),
    },
}


def source_directives(condition: str) -> tuple[str, str]:
    if condition == "request_cue":
        return REASONING_DIRECTIVE, DIRECT_DIRECTIVE
    if condition == "notes_cue":
        return DIRECT_DIRECTIVE, REASONING_DIRECTIVE
    if condition == "all_reason":
        return REASONING_DIRECTIVE, REASONING_DIRECTIVE
    if condition == "all_direct":
        return DIRECT_DIRECTIVE, DIRECT_DIRECTIVE
    if condition == "placebo_a":
        return DIRECT_DIRECTIVE, PLACEBO_DIRECTIVE
    if condition == "placebo_b":
        return PLACEBO_DIRECTIVE, DIRECT_DIRECTIVE
    if condition == "para_request_b":
        return PARAPHRASE_REASONING_DIRECTIVE, DIRECT_DIRECTIVE
    if condition == "para_notes_b":
        return DIRECT_DIRECTIVE, PARAPHRASE_REASONING_DIRECTIVE
    raise ValueError(f"Unknown condition: {condition}")


def build_prompt(question: str, condition: str, order: str, variant: str) -> str:
    if condition not in CONDITIONS + RECALIBRATION_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if order not in ORDERS:
        raise ValueError(f"Unknown order: {order}")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    request, notes = source_directives(condition)
    blocks = [f"REQUEST:\n{request}", f"WORKING NOTES:\n{notes}"]
    if order == "notes_first":
        blocks.reverse()
    return CANONICAL_TEMPLATE.format(
        question=question.strip(),
        first_block=blocks[0],
        second_block=blocks[1],
        choice_block=VARIANTS[variant]["choice_block"],
    )


def candidate_texts(variant: str) -> tuple[str, str]:
    try:
        spec = VARIANTS[variant]
    except KeyError as error:
        raise ValueError(f"Unknown variant: {variant}") from error
    return spec["reason"], spec["direct"]


def strip_one_choice_prefix(response: str, variant: str) -> tuple[str, str | None]:
    reason, direct = candidate_texts(variant)
    choices = sorted((reason, direct), key=len, reverse=True)
    pattern = re.compile(
        rf"^\s*({'|'.join(re.escape(value) for value in choices)})(?=\s|:|-|$)"
    )
    match = pattern.match(response)
    if match is None:
        return response, None
    remainder = response[match.end() :]
    remainder = re.sub(r"^[ \t]*(?::|-)?[ \t]*(?:\r?\n)?", "", remainder, count=1)
    choice = "reason" if match.group(1) == reason else "direct"
    return remainder, choice


def _is_refusal(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:I\s+cannot|I\s+can't|I\s+am\s+unable|unable\s+to|cannot\s+comply|won't\s+comply)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def classify_semantic_mode(response: str, variant: str) -> dict[str, Any]:
    """v2 semantic classifier: the last FINAL: marker anywhere in the
    response defines the answer; substantive text before it means the model
    reasoned, whitespace-only means it answered directly."""
    body, leading_choice = strip_one_choice_prefix(response, variant)
    failures: list[str] = []
    if _is_refusal(body):
        failures.append("refusal")
    final_markers = list(re.finditer(r"FINAL[ \t]*:", body, re.IGNORECASE))
    semantic_mode: str | None = None
    final_answer: str | None = None
    if not final_markers:
        failures.append("no_final")
    else:
        last = final_markers[-1]
        after = body[last.end() :]
        line_rest = after.split("\n", 1)[0].strip()
        if not line_rest:
            following = [line.strip() for line in after.split("\n")[1:]]
            line_rest = next((line for line in following if line), "")
        if not line_rest:
            failures.append("empty_final")
        elif not failures:
            final_answer = line_rest
            semantic_mode = "reason" if body[: last.start()].strip() else "direct"
    if semantic_mode is None and not failures:
        failures.append("unclassifiable_structure")
    return {
        "leading_choice": leading_choice,
        "semantic_mode": semantic_mode,
        "final_answer": final_answer,
        "classifiable": semantic_mode is not None,
        "choice_matches_semantic": leading_choice == semantic_mode
        if leading_choice is not None and semantic_mode is not None
        else None,
        "semantic_failure_categories": failures,
        "stripped_response": body,
    }


def detect_terminal_loop(
    token_ids: Iterable[int],
    *,
    maximum_period: int = 128,
    minimum_copies: int = 3,
    minimum_span: int = 32,
) -> dict[str, Any]:
    tokens = list(token_ids)
    matches: list[dict[str, int]] = []
    for period in range(1, min(maximum_period, len(tokens) // minimum_copies) + 1):
        block = tokens[-period:]
        copies = 1
        start = len(tokens) - period
        while start >= period and tokens[start - period : start] == block:
            copies += 1
            start -= period
        span = copies * period
        if copies >= minimum_copies and span >= minimum_span:
            matches.append(
                {"start": start, "period": period, "copies": copies, "span": span}
            )
    if not matches:
        return {
            "detected": False,
            "start": None,
            "period": None,
            "copies": None,
            "span": 0,
        }
    winner = min(
        matches, key=lambda value: (-value["span"], value["period"], value["start"])
    )
    return {"detected": True, **winner}


def salted_item_key(seed: int, namespace: str, item_id: str) -> tuple[str, str]:
    payload = f"{seed}\0{namespace}\0{item_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), item_id


def ranked_items(
    items: Iterable[dict[str, Any]], seed: int, namespace: str
) -> list[dict[str, Any]]:
    return sorted(
        items, key=lambda item: salted_item_key(seed, namespace, item["item_id"])
    )
