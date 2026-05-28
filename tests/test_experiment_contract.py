"""Tests for HNC experiment contract records."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hnc_lab.datasets import load_qa_jsonl
from hnc_lab.schemas import (
    ApproximationGap,
    CompressedContextTrace,
    DenoisingTrace,
    SinkCompressionProbe,
)
from hnc_lab.experiments.temperature_sweep import run_temperature_sweep
from hnc_lab.verification import build_denoising_trace, verify_exact_match


class ExperimentContractTest(unittest.TestCase):
    """Validate that hnc_lab preserves HNC experiment metadata."""

    def test_load_qa_jsonl_requires_hnc_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "items.jsonl"
            dataset_path.write_text(
                (
                    '{"id":"item-1","prompt":"Answer only Paris.",'
                    '"answers":["Paris"],'
                    '"metadata":{"topic":"geography","difficulty":"easy",'
                    '"capacity_stratum":"strong","source_condition":"weights",'
                    '"expected_failure_mode":"none"}}\n'
                ),
                encoding="utf-8",
            )
            items = load_qa_jsonl(dataset_path)
        self.assertEqual(items[0].capacity_stratum, "strong")
        self.assertEqual(items[0].source_condition, "weights")
        self.assertEqual(items[0].expected_failure_mode, "none")

    def test_temperature_sweep_preserves_metadata(self) -> None:
        items = load_qa_jsonl("data/temperature_probe_seed.jsonl")
        raw_rows, aggregate_rows = run_temperature_sweep(
            items=items[:1],
            backend_name="fixture",
            temperatures=[0.0],
            stochastic_samples=1,
        )
        self.assertEqual(raw_rows[0].capacity_stratum, "strong")
        self.assertEqual(raw_rows[0].source_condition, "weights")
        self.assertEqual(raw_rows[0].expected_failure_mode, "none")
        self.assertIn('"topic": "geography"', raw_rows[0].metadata_json)
        self.assertEqual(aggregate_rows[0].capacity_strata, "strong")

    def test_theory_schema_stubs_validate(self) -> None:
        ApproximationGap(
            item_id="item-1",
            architecture_profile="api_text",
            ideal_predictor_reference=None,
            neural_predictor_id="model-1",
            log_loss_proxy=1.2,
            code_length_proxy=3.4,
            support_condition="in_support",
            amortization_gap_proxy=None,
            model_class_regret_proxy=None,
            approximation_notes="smoke test",
        )
        CompressedContextTrace(
            trace_id="trace-1",
            item_id="item-1",
            thinker_model="thinker",
            answerer_model="answerer",
            original_context_tokens=100,
            trace_tokens=20,
            target_compression_ratio=0.2,
            actual_compression_ratio=0.2,
            trace_text="Relevant evidence only.",
            trace_utility_score=1.0,
            trace_faithfulness_score=1.0,
            answer_leakage_flag=False,
            unsupported_claim_rate=0.0,
            downstream_exact_match=1.0,
            downstream_f1=1.0,
        )
        DenoisingTrace(
            trace_id="trace-2",
            item_id="item-1",
            initial_answer="Paris maybe.",
            repaired_answer="Paris.",
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
        SinkCompressionProbe(
            probe_id="probe-1",
            model_name="open-model",
            layer_index=4,
            token_position=0,
            beginning_of_sequence_norm_ratio=12.0,
            sink_rate=0.8,
            matrix_entropy=0.2,
            anisotropy=0.9,
            mixing_score=0.3,
            colsum_concentration=0.8,
            sink_vs_identity_index=0.75,
            layer_phase="middle_compression",
            downstream_failure_rate=0.4,
            downstream_exact_match=0.6,
            downstream_unsupported_claim_rate=0.2,
        )

    def test_verification_and_denoising_helpers(self) -> None:
        item = load_qa_jsonl("data/temperature_probe_seed.jsonl")[0]
        verification = verify_exact_match(item, "Paris")
        self.assertTrue(verification.passed)
        trace = build_denoising_trace(
            trace_id="trace-1",
            item=item,
            initial_answer="Paris maybe.",
            repaired_answer="Paris",
            source_refs=["answer:Paris"],
            denoiser_id="manual",
            source_conditioned=True,
            unsupported_claims_before=1,
            unsupported_claims_after=0,
            supported_claims_before=0,
            supported_claims_after=1,
        )
        self.assertEqual(trace.exact_match_after, 1.0)
        self.assertFalse(trace.abstention_after)


if __name__ == "__main__":
    unittest.main()
