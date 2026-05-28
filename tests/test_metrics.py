"""Tests for HNC experiment metrics."""

from __future__ import annotations

import unittest

from hnc_lab.metrics import (
    count_tokens,
    denoising_unsupported_delta,
    is_exact_match,
    is_refusal,
    normalize_answer,
    self_consistency,
    summarize_temperature,
)
from hnc_lab.schemas import DenoisingTrace, ResultRow


class MetricsTest(unittest.TestCase):
    """Validate deterministic metric behavior."""

    def test_normalize_answer_removes_case_articles_and_punctuation(self) -> None:
        self.assertEqual(normalize_answer("The Eiffel Tower!"), "eiffel tower")

    def test_is_exact_match_accepts_aliases(self) -> None:
        self.assertTrue(is_exact_match("Paris.", ["City of Paris", "Paris"]))
        self.assertFalse(is_exact_match("Paris maybe", ["Paris"]))

    def test_is_refusal_detects_common_abstentions(self) -> None:
        self.assertTrue(is_refusal("I do not know."))
        self.assertTrue(is_refusal("There is not enough information."))
        self.assertFalse(is_refusal("Paris"))

    def test_count_tokens_uses_raw_output(self) -> None:
        self.assertEqual(count_tokens("  Paris is correct. "), 3)

    def test_self_consistency_uses_normalized_majority(self) -> None:
        outputs = ["The Paris.", "Paris!", "London"]
        self.assertAlmostEqual(self_consistency(outputs), 2 / 3)

    def test_summarize_temperature_aggregates_rows(self) -> None:
        rows = [
            _row("item-1", 0.7, 0, "Paris", True, False),
            _row("item-1", 0.7, 1, "Paris.", True, False),
            _row("item-2", 0.7, 0, "I do not know.", False, True),
        ]
        aggregate = summarize_temperature(rows)
        self.assertEqual(aggregate.temperature, 0.7)
        self.assertEqual(aggregate.item_count, 2)
        self.assertEqual(aggregate.generation_count, 3)
        self.assertEqual(aggregate.exact_match_count, 2)
        self.assertEqual(aggregate.refusal_count, 1)
        self.assertAlmostEqual(aggregate.accuracy, 2 / 3)
        self.assertAlmostEqual(aggregate.refusal_rate, 1 / 3)
        self.assertAlmostEqual(aggregate.self_consistency, 1.0)
        self.assertEqual(aggregate.capacity_strata, "strong")
        self.assertEqual(aggregate.source_conditions, "weights")
        self.assertEqual(aggregate.expected_failure_modes, "none")

    def test_denoising_unsupported_delta_counts_removed_claims(self) -> None:
        trace = DenoisingTrace(
            trace_id="trace-1",
            item_id="item-1",
            initial_answer="Paris is in Germany.",
            repaired_answer="Paris is in France.",
            source_refs=["answer:Paris"],
            effective_query_id="query-1",
            denoiser_id="manual",
            verifier_id="exact_match_v1",
            source_conditioned=True,
            denoising_step_count=1,
            unsupported_claims_before=1,
            unsupported_claims_after=0,
            supported_claims_before=0,
            supported_claims_after=1,
            abstention_after=False,
            retrieval_requested_after=False,
            exact_match_before=0.0,
            exact_match_after=1.0,
        )
        self.assertEqual(denoising_unsupported_delta(trace), 1)


def _row(
    item_id: str,
    temperature: float,
    sample_index: int,
    raw_output: str,
    is_match: bool,
    refusal: bool,
) -> ResultRow:
    """Build a result row for tests."""
    return ResultRow(
        item_id=item_id,
        temperature=temperature,
        sample_index=sample_index,
        prompt="Question?",
        expected_answers="Answer",
        raw_output=raw_output,
        normalized_output=normalize_answer(raw_output),
        is_exact_match=is_match,
        is_refusal=refusal,
        token_count=count_tokens(raw_output),
        backend_name="test",
        model_name="test-model",
        topic="test-topic",
        difficulty="easy",
        capacity_stratum="strong",
        source_condition="weights",
        expected_failure_mode="none",
        metadata_json=(
            '{"capacity_stratum": "strong", "difficulty": "easy", '
            '"expected_failure_mode": "none", "source_condition": "weights", '
            '"topic": "test-topic"}'
        ),
    )


if __name__ == "__main__":
    unittest.main()
