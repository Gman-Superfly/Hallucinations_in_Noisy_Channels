"""Metrics for known-answer HNC experiments."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Sequence

from hnc_lab.schemas import AggregateRow, DenoisingTrace, ResultRow

__all__ = [
    "count_tokens",
    "denoising_unsupported_delta",
    "is_exact_match",
    "is_refusal",
    "normalize_answer",
    "self_consistency",
    "summarize_temperature",
]

_ARTICLES = {"a", "an", "the"}
_REFUSAL_PATTERNS = (
    "i don't know",
    "i do not know",
    "cannot answer",
    "can't answer",
    "not enough information",
    "insufficient information",
    "i am not sure",
    "i'm not sure",
    "unknown",
)


def normalize_answer(text: str) -> str:
    """Normalize an answer for exact-match scoring.

    Args:
        text: Raw answer text.

    Returns:
        Lowercased text with punctuation, articles, and repeated whitespace removed.

    Raises:
        AssertionError: If text is not a string.
    """
    assert isinstance(text, str), f"Expected str, got {type(text)}"
    lowered = text.lower()
    no_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in no_punctuation.split() if token not in _ARTICLES]
    return " ".join(tokens)


def is_exact_match(output: str, answers: Sequence[str]) -> bool:
    """Return whether output matches any accepted answer after normalization.

    Args:
        output: Raw generated output.
        answers: Accepted answer strings.

    Returns:
        True when output exactly matches an accepted answer after normalization.

    Raises:
        AssertionError: If no answers are supplied.
    """
    assert len(answers) > 0, "answers required"
    normalized_output = normalize_answer(output)
    normalized_answers = {normalize_answer(answer) for answer in answers}
    return normalized_output in normalized_answers


def is_refusal(output: str) -> bool:
    """Return whether output looks like an abstention or refusal.

    Args:
        output: Raw generated output.

    Returns:
        True when a simple refusal pattern appears in the normalized output.
    """
    normalized = normalize_answer(output)
    return any(pattern in normalized for pattern in _REFUSAL_PATTERNS)


def count_tokens(output: str) -> int:
    """Count whitespace-delimited tokens in raw output.

    Args:
        output: Raw generated output.

    Returns:
        Number of non-empty whitespace-delimited tokens.
    """
    return len(re.findall(r"\S+", output))


def self_consistency(outputs: Iterable[str]) -> float:
    """Measure agreement among repeated samples.

    Args:
        outputs: Raw generated outputs for one item and temperature.

    Returns:
        Fraction of samples matching the most common normalized output.
        Returns 0.0 for an empty input.
    """
    normalized_outputs = [normalize_answer(output) for output in outputs]
    if not normalized_outputs:
        return 0.0
    counts = Counter(normalized_outputs)
    return max(counts.values()) / len(normalized_outputs)


def denoising_unsupported_delta(trace: DenoisingTrace) -> int:
    """Return unsupported-claim reduction after denoising.

    Args:
        trace: Denoising trace containing before and after claim counts.

    Returns:
        Positive values mean the repair step reduced unsupported claims.
    """
    return trace.unsupported_claims_before - trace.unsupported_claims_after


def summarize_temperature(rows: Sequence[ResultRow]) -> AggregateRow:
    """Aggregate scored rows for a single temperature.

    Args:
        rows: Scored result rows sharing one temperature.

    Returns:
        Aggregate metrics for the temperature.

    Raises:
        AssertionError: If rows are empty or contain multiple temperatures.
    """
    assert len(rows) > 0, "rows required"
    temperatures = {row.temperature for row in rows}
    assert len(temperatures) == 1, f"Expected one temperature, got {temperatures}"

    item_ids = {row.item_id for row in rows}
    exact_match_count = sum(1 for row in rows if row.is_exact_match)
    refusal_count = sum(1 for row in rows if row.is_refusal)
    capacity_strata = sorted({row.capacity_stratum for row in rows})
    source_conditions = sorted({row.source_condition for row in rows})
    expected_failure_modes = sorted({row.expected_failure_mode for row in rows})
    grouped_outputs: dict[str, list[str]] = {}
    for row in rows:
        grouped_outputs.setdefault(row.item_id, []).append(row.raw_output)

    consistency_values = [
        self_consistency(outputs) for outputs in grouped_outputs.values()
    ]
    mean_consistency = sum(consistency_values) / len(consistency_values)
    generation_count = len(rows)

    return AggregateRow(
        temperature=rows[0].temperature,
        item_count=len(item_ids),
        generation_count=generation_count,
        exact_match_count=exact_match_count,
        refusal_count=refusal_count,
        accuracy=exact_match_count / generation_count,
        refusal_rate=refusal_count / generation_count,
        self_consistency=mean_consistency,
        capacity_strata="|".join(capacity_strata),
        source_conditions="|".join(source_conditions),
        expected_failure_modes="|".join(expected_failure_modes),
    )
