"""Dataset loading utilities for HNC experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hnc_lab.schemas import QAItem

__all__ = ["REQUIRED_METADATA_KEYS", "load_qa_jsonl"]

REQUIRED_METADATA_KEYS = (
    "topic",
    "difficulty",
    "capacity_stratum",
    "source_condition",
    "expected_failure_mode",
)


def load_qa_jsonl(path: str | Path) -> list[QAItem]:
    """Load known-answer QA items from JSONL.

    Each line must contain `id`, `prompt`, and `answers`.

    Args:
        path: JSONL file path.

    Returns:
        List of QA items.

    Raises:
        AssertionError: If the file is missing or contains invalid records.
        ValueError: If a line is not valid JSON.
    """
    dataset_path = Path(path)
    assert dataset_path.exists(), f"Dataset not found: {dataset_path}"
    assert dataset_path.is_file(), f"Dataset path is not a file: {dataset_path}"

    items: list[QAItem] = []
    seen_ids: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            item = _qa_item_from_record(record, line_number)
            assert item.item_id not in seen_ids, f"Duplicate item id: {item.item_id}"
            seen_ids.add(item.item_id)
            items.append(item)

    assert len(items) > 0, f"Dataset is empty: {dataset_path}"
    return items


def _qa_item_from_record(record: dict[str, Any], line_number: int) -> QAItem:
    """Convert one JSON record into a QA item.

    Args:
        record: JSON object from one dataset line.
        line_number: One-based line number used in validation messages.

    Returns:
        Validated QA item.

    Raises:
        AssertionError: If the record lacks required HNC metadata.
    """
    assert isinstance(record, dict), f"Line {line_number}: expected object"
    assert "id" in record, f"Line {line_number}: missing id"
    assert "prompt" in record, f"Line {line_number}: missing prompt"
    assert "answers" in record, f"Line {line_number}: missing answers"
    answers = record["answers"]
    assert isinstance(answers, list), f"Line {line_number}: answers must be a list"
    metadata = record.get("metadata", {})
    assert isinstance(metadata, dict), f"Line {line_number}: metadata must be an object"
    for key in REQUIRED_METADATA_KEYS:
        assert key in metadata, f"Line {line_number}: metadata missing {key}"
        assert isinstance(metadata[key], str), (
            f"Line {line_number}: metadata {key} must be a string"
        )
        assert metadata[key].strip(), f"Line {line_number}: metadata {key} required"
    return QAItem(
        item_id=str(record["id"]),
        prompt=str(record["prompt"]),
        answers=tuple(str(answer) for answer in answers),
        metadata=metadata,
    )
