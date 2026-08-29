"""Shared utilities for the mode-choice repair experiment.

Hashing/IO provenance, GSM8K answer handling, and the inference helpers used
by the runner. Extracted verbatim from the retired calibration stack so the
repair pipeline no longer depends on it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from re import Pattern
import re
from typing import Any, Iterable


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _item_content_sha256(items: Iterable[dict[str, Any]]) -> str:
    content = [
        {
            "benchmark": item["benchmark"],
            "item_id": item["item_id"],
            "question": item["question"],
            "reference": item["reference"],
        }
        for item in items
    ]
    content.sort(key=lambda item: item["item_id"])
    return _canonical_sha256(content)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def extract_gsm8k_reference(answer: str) -> str:
    marker = "####"
    if marker not in answer:
        raise ValueError("GSM8K reference answer has no #### marker")
    return answer.rsplit(marker, 1)[1].strip()


def extract_final_answer(response: str) -> str | None:
    matches = re.findall(
        r"^\s*(?:[RA]\s+)?FINAL\s*:\s*([^\n\r]+)",
        response,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not matches:
        return None
    answer = matches[-1].strip()
    answer = answer.strip("`* _\t")
    return answer or None


def _numeric_value(text: str) -> Fraction | Decimal | None:
    cleaned = text.strip().strip("$£€¥").replace(",", "")
    cleaned = cleaned.rstrip(".。")
    cleaned = cleaned.replace("%", "")
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", cleaned):
        try:
            numerator, denominator = cleaned.split("/", 1)
            return Fraction(int(numerator.strip()), int(denominator.strip()))
        except (ValueError, ZeroDivisionError):
            return None
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", cleaned):
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None
    return None


def answers_equal(predicted: str | None, reference: str, benchmark: str) -> bool:
    if predicted is None:
        return False
    if benchmark == "gsm8k":
        predicted_value = _numeric_value(predicted)
        reference_value = _numeric_value(reference)
        if predicted_value is not None and reference_value is not None:
            return predicted_value == reference_value
    normalized_predicted = re.sub(r"\s+", " ", predicted.strip()).casefold()
    normalized_reference = re.sub(r"\s+", " ", reference.strip()).casefold()
    return normalized_predicted == normalized_reference


def _tokenizer_hash(tokenizer: Any) -> str:
    payload = {
        "vocab": sorted(tokenizer.get_vocab().items()),
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ["torch", "transformers", "datasets", "huggingface-hub", "accelerate"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _trim_generated_ids(
    token_ids: list[int], maximum: int, eos_token_id: int | list[int] | None
) -> tuple[list[int], str]:
    eos_ids = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
    eos_ids.discard(None)
    for index, token_id in enumerate(token_ids):
        if token_id in eos_ids:
            return token_ids[: index + 1], "eos"
    return token_ids, "length" if len(token_ids) >= maximum else "unknown"


def _normalized_token_ids(value: Any) -> list[int]:
    values = value if isinstance(value, (list, tuple)) else [value]
    return list(dict.fromkeys(int(item) for item in values if item is not None))


def _normalized_dtype_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).removeprefix("torch.")


def _configure_shared_stopping(
    config: dict[str, Any], tokenizer: Any
) -> dict[str, Any]:
    generation = config["generation"]
    chat_template = getattr(tokenizer, "chat_template", None)
    original = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        if chat_template
        else None,
        "special_tokens_map_sha256": _canonical_sha256(tokenizer.special_tokens_map),
    }
    for token_id, token_text in zip(
        generation["eos_token_ids"], generation["eos_token_texts"]
    ):
        if tokenizer.convert_ids_to_tokens(token_id) != token_text:
            raise ValueError(f"Configured EOS token mismatch at ID {token_id}")
    if tokenizer.convert_ids_to_tokens(generation["pad_token_id"]) != generation[
        "pad_token_text"
    ]:
        raise ValueError("Configured pad token ID/text mismatch")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = generation["pad_token_text"]
    if tokenizer.pad_token_id != generation["pad_token_id"]:
        raise ValueError("Tokenizer did not adopt the frozen pad token")
    configured = {
        "configured_eos_token_ids": generation["eos_token_ids"],
        "configured_eos_token_texts": generation["eos_token_texts"],
        "configured_pad_token_id": generation["pad_token_id"],
        "configured_pad_token_text": generation["pad_token_text"],
        "configured_use_cache": generation["use_cache"],
        "configured_cache_implementation": generation["cache_implementation"],
        "configured_model_dtype": generation["model_dtype"],
    }
    return {
        "configured": configured,
        "tokenizer_original": original,
        "tokenizer_enforced_pad_token_id": tokenizer.pad_token_id,
    }


def _candidate_ids(
    tokenizer: Any, prompt: str, text: str, add_special_tokens: bool = True
) -> list[int]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    combined_ids = tokenizer.encode(prompt + text, add_special_tokens=add_special_tokens)
    if combined_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Tokenizer boundary changed when appending candidate {text!r}")
    ids = combined_ids[len(prompt_ids) :]
    if not ids:
        raise ValueError(f"Candidate {text!r} tokenized to nothing")
    return ids


def _chunks(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _validate_process_topology(
    world_size: int, rank: int, cuda_device_count: int
) -> None:
    if world_size != 1 or rank != 0 or cuda_device_count != 1:
        raise RuntimeError("Frozen execution requires one process with one visible GPU")


def _checkpoint_stopping_fields(generation_config: Any, model_config: Any) -> dict[str, Any]:
    dtype = getattr(model_config, "dtype", None)
    if dtype is None:
        dtype = getattr(model_config, "torch_dtype", None)
    return {
        "checkpoint_generation_eos_token_ids": _normalized_token_ids(
            generation_config.eos_token_id
        ),
        "checkpoint_generation_pad_token_id": generation_config.pad_token_id,
        "checkpoint_config_use_cache": model_config.use_cache,
        "checkpoint_config_torch_dtype": _normalized_dtype_name(dtype),
    }
