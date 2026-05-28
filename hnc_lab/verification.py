"""Verification and source-attribution helpers for HNC experiments."""

from __future__ import annotations

from typing import Literal

from hnc_lab.metrics import is_exact_match, is_refusal
from hnc_lab.schemas import (
    ClaimAttributionRow,
    DenoisingTrace,
    QAItem,
    VerificationResult,
    VerifierProfile,
)

__all__ = [
    "build_denoising_trace",
    "claim_attribution_row",
    "exact_match_verifier",
    "verify_exact_match",
]


def exact_match_verifier(domain: str = "known_answer_qa") -> VerifierProfile:
    """Build the default exact-match verifier profile.

    Args:
        domain: Domain label for the verifier.

    Returns:
        Verifier profile for known-answer QA.
    """
    return VerifierProfile(
        verifier_id="exact_match_v1",
        verifier_type="exact_match",
        domain=domain,
        target_description="Generated answer matches one accepted answer after normalization.",
        threshold=1.0,
        source_required=True,
    )


def verify_exact_match(
    item: QAItem,
    output: str,
    verifier: VerifierProfile | None = None,
) -> VerificationResult:
    """Verify one generated answer with exact-match scoring.

    Args:
        item: QA item containing accepted answers.
        output: Raw generated output.
        verifier: Optional verifier profile.

    Returns:
        Verification result with pass/fail status and evidence references.
    """
    active_verifier = verifier or exact_match_verifier()
    passed = is_exact_match(output, item.answers)
    evidence_refs = [f"answer:{answer}" for answer in item.answers]
    failure_reason = None if passed else "exact_match_failed"
    return VerificationResult(
        verifier_id=active_verifier.verifier_id,
        item_id=item.item_id,
        claim_id=None,
        passed=passed,
        score=1.0 if passed else 0.0,
        evidence_refs=evidence_refs,
        failure_reason=failure_reason,
    )


def claim_attribution_row(
    item_id: str,
    claim_id: str,
    claim_text: str,
    source_label: Literal[
        "weights",
        "context",
        "retrieval",
        "tool",
        "memory",
        "unsupported",
        "untracked",
    ],
    evidence_refs: list[str],
    support_score: float,
    verifier_result_id: str | None,
    attribution_confidence: float,
) -> ClaimAttributionRow:
    """Build a claim attribution row with validation.

    Args:
        item_id: Stable item identifier.
        claim_id: Stable claim identifier.
        claim_text: Claim text extracted from an answer.
        source_label: Source label from the HNC source taxonomy.
        evidence_refs: References supporting the attribution.
        support_score: Support score in [0, 1].
        verifier_result_id: Optional verification result identifier.
        attribution_confidence: Attribution confidence in [0, 1].

    Returns:
        Validated claim attribution row.
    """
    return ClaimAttributionRow(
        item_id=item_id,
        claim_id=claim_id,
        claim_text=claim_text,
        source_label=source_label,
        evidence_refs=evidence_refs,
        support_score=support_score,
        verifier_result_id=verifier_result_id,
        attribution_confidence=attribution_confidence,
    )


def build_denoising_trace(
    trace_id: str,
    item: QAItem,
    initial_answer: str,
    repaired_answer: str,
    source_refs: list[str],
    denoiser_id: str,
    source_conditioned: bool,
    unsupported_claims_before: int,
    unsupported_claims_after: int,
    supported_claims_before: int,
    supported_claims_after: int,
    denoising_step_count: int = 1,
    effective_query_id: str | None = None,
    verifier: VerifierProfile | None = None,
    retrieval_requested_after: bool = False,
    repair_notes: str | None = None,
) -> DenoisingTrace:
    """Build a denoising trace for source conditioned repair.

    Args:
        trace_id: Stable denoising trace identifier.
        item: QA item being repaired.
        initial_answer: Answer before repair.
        repaired_answer: Answer after repair.
        source_refs: Source references used during repair.
        denoiser_id: Identifier for the repair process.
        source_conditioned: Whether repair saw source evidence.
        unsupported_claims_before: Unsupported claim count before repair.
        unsupported_claims_after: Unsupported claim count after repair.
        supported_claims_before: Supported claim count before repair.
        supported_claims_after: Supported claim count after repair.
        denoising_step_count: Number of repair passes.
        effective_query_id: Optional effective query identifier.
        verifier: Optional verifier profile.
        retrieval_requested_after: Whether repair requested retrieval.
        repair_notes: Optional notes for manual audit.

    Returns:
        Validated denoising trace.
    """
    active_verifier = verifier or exact_match_verifier()
    exact_match_before = 1.0 if is_exact_match(initial_answer, item.answers) else 0.0
    exact_match_after = 1.0 if is_exact_match(repaired_answer, item.answers) else 0.0
    abstention_after = is_refusal(repaired_answer)
    return DenoisingTrace(
        trace_id=trace_id,
        item_id=item.item_id,
        initial_answer=initial_answer,
        repaired_answer=repaired_answer,
        source_refs=source_refs,
        effective_query_id=effective_query_id or item.item_id,
        denoiser_id=denoiser_id,
        verifier_id=active_verifier.verifier_id,
        source_conditioned=source_conditioned,
        denoising_step_count=denoising_step_count,
        unsupported_claims_before=unsupported_claims_before,
        unsupported_claims_after=unsupported_claims_after,
        supported_claims_before=supported_claims_before,
        supported_claims_after=supported_claims_after,
        abstention_after=abstention_after,
        retrieval_requested_after=retrieval_requested_after,
        exact_match_before=exact_match_before,
        exact_match_after=exact_match_after,
        repair_notes=repair_notes,
    )
