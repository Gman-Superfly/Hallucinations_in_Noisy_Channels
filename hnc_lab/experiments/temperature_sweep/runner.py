"""Run a temperature sweep for known answer QA prompts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from hnc_lab.backends import build_backend
from hnc_lab.datasets import load_qa_jsonl
from hnc_lab.metrics import (
    count_tokens,
    is_exact_match,
    is_refusal,
    normalize_answer,
    summarize_temperature,
)
from hnc_lab.schemas import AggregateRow, GenerationRequest, QAItem, ResultRow

__all__ = ["main", "run_temperature_sweep"]


def run_temperature_sweep(
    items: Sequence[QAItem],
    backend_name: str,
    temperatures: Sequence[float],
    stochastic_samples: int,
) -> tuple[list[ResultRow], list[AggregateRow]]:
    """Run the temperature sweep and return raw and aggregate rows.

    Args:
        items: Known answer QA items.
        backend_name: Generation backend name.
        temperatures: Temperatures to evaluate.
        stochastic_samples: Samples per item for temperatures greater than zero.

    Returns:
        Raw result rows and aggregate rows.

    Raises:
        AssertionError: If inputs are empty or invalid.
    """
    assert len(items) > 0, "items required"
    assert len(temperatures) > 0, "temperatures required"
    assert all(temperature >= 0.0 for temperature in temperatures), (
        "temperatures must be non-negative"
    )
    assert stochastic_samples > 0, "stochastic_samples must be positive"

    backend = build_backend(backend_name, list(items))
    raw_rows: list[ResultRow] = []

    for temperature in temperatures:
        sample_count = 1 if temperature == 0.0 else stochastic_samples
        for item in items:
            for sample_index in range(sample_count):
                request = GenerationRequest(
                    item_id=item.item_id,
                    prompt=item.prompt,
                    temperature=temperature,
                    sample_index=sample_index,
                    metadata=item.metadata,
                )
                generation = backend.generate(request)
                raw_rows.append(
                    _score_generation(
                        item=item,
                        raw_output=generation.text,
                        backend_name=generation.backend_name,
                        model_name=generation.model_name,
                        request=request,
                    )
                )

    grouped_rows: dict[float, list[ResultRow]] = defaultdict(list)
    for row in raw_rows:
        grouped_rows[row.temperature].append(row)

    aggregate_rows = [
        summarize_temperature(grouped_rows[temperature])
        for temperature in sorted(grouped_rows)
    ]
    return raw_rows, aggregate_rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line temperature sweep."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    items = load_qa_jsonl(args.input)
    temperatures = _parse_temperatures(args.temperatures)
    raw_rows, aggregate_rows = run_temperature_sweep(
        items=items,
        backend_name=args.backend,
        temperatures=temperatures,
        stochastic_samples=args.samples,
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_prefix.with_name(f"{output_prefix.name}_raw.csv")
    aggregate_path = output_prefix.with_name(f"{output_prefix.name}_aggregate.csv")
    metadata_path = output_prefix.with_name(f"{output_prefix.name}_metadata.log")

    _write_csv(raw_path, raw_rows)
    _write_csv(aggregate_path, aggregate_rows)
    _write_metadata(
        metadata_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": "temperature_sweep",
            "main_hnc_sections": [
                "Prediction 12",
                "Prediction 15",
                "Prediction 16",
                "Section 8.5",
                "Section 8.6",
            ],
            "backend": args.backend,
            "input": str(args.input),
            "item_count": len(items),
            "models": sorted({row.model_name for row in raw_rows}),
            "temperatures": temperatures,
            "stochastic_samples": args.samples,
            "raw_output": str(raw_path),
            "aggregate_output": str(aggregate_path),
            "required_item_metadata": [
                "topic",
                "difficulty",
                "capacity_stratum",
                "source_condition",
                "expected_failure_mode",
            ],
            "claim_boundary": (
                "Fixture backend runs test the pipeline only. They are not evidence "
                "for HNC predictions."
            ),
        },
    )

    print(f"Wrote raw rows: {raw_path}")
    print(f"Wrote aggregate rows: {aggregate_path}")
    print(f"Wrote metadata: {metadata_path}")
    return 0


def _score_generation(
    item: QAItem,
    raw_output: str,
    backend_name: str,
    model_name: str | None,
    request: GenerationRequest,
) -> ResultRow:
    """Score one generation against known answers."""
    metadata_json = json.dumps(item.metadata, sort_keys=True)
    return ResultRow(
        item_id=item.item_id,
        temperature=request.temperature,
        sample_index=request.sample_index,
        prompt=item.prompt,
        expected_answers="|".join(item.answers),
        raw_output=raw_output,
        normalized_output=normalize_answer(raw_output),
        is_exact_match=is_exact_match(raw_output, item.answers),
        is_refusal=is_refusal(raw_output),
        token_count=count_tokens(raw_output),
        backend_name=backend_name,
        model_name=model_name or backend_name,
        topic=item.topic,
        difficulty=item.difficulty,
        capacity_stratum=item.capacity_stratum,
        source_condition=item.source_condition,
        expected_failure_mode=item.expected_failure_mode,
        metadata_json=metadata_json,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run an HNC temperature sweep over known answer QA items."
    )
    parser.add_argument(
        "--input",
        default="data/temperature_probe_seed.jsonl",
        help="Path to known answer QA JSONL.",
    )
    parser.add_argument(
        "--backend",
        default="fixture",
        help="Generation backend. Supported: fixture, openai_compatible.",
    )
    parser.add_argument(
        "--temperatures",
        default="0,0.3,0.7,1.0,1.5",
        help="Comma-separated non-negative temperatures.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Samples per item for temperatures greater than zero.",
    )
    parser.add_argument(
        "--output-prefix",
        default="figures/temperature_sweep",
        help="Output prefix for generated CSV and metadata log files.",
    )
    return parser


def _parse_temperatures(raw_temperatures: str) -> list[float]:
    """Parse a comma-separated temperature list."""
    temperatures = [
        float(value.strip())
        for value in raw_temperatures.split(",")
        if value.strip()
    ]
    assert len(temperatures) > 0, "at least one temperature required"
    assert all(temperature >= 0.0 for temperature in temperatures), (
        "temperatures must be non-negative"
    )
    return temperatures


def _write_csv(path: Path, rows: Sequence[ResultRow] | Sequence[AggregateRow]) -> None:
    """Write dataclass rows to CSV."""
    assert len(rows) > 0, f"No rows to write: {path}"
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(asdict(rows[0]).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_safe_dict(asdict(row)))


def _csv_safe_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Convert nested values to stable JSON for CSV output."""
    safe_row: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            safe_row[key] = json.dumps(value, sort_keys=True)
        else:
            safe_row[key] = value
    return safe_row


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    """Write run metadata as JSON lines in an ignored log file."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, indent=2, sort_keys=True))
        handle.write("\n")
