"""Typed records for HNC experiment runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "AggregateRow",
    "ApproximationGap",
    "ArchitectureProfile",
    "CapacityEstimate",
    "ChannelStage",
    "ClaimAttributionRow",
    "ClaimBinding",
    "CompressedContextTrace",
    "DecodingControl",
    "DecompressionBudget",
    "DenoisingTrace",
    "DistortionTrace",
    "EffectiveQuery",
    "GenerationRegime",
    "GenerationRequest",
    "GenerationResult",
    "QAItem",
    "RepresentationMatch",
    "ResultRow",
    "RetrievalMemoryChannel",
    "SinkCompressionProbe",
    "SourceSignal",
    "VerificationResult",
    "VerifierProfile",
]

CapacityStratum = Literal["strong", "weak_recoverable", "unsupported", "misleading"]
SourceCondition = Literal[
    "weights",
    "context",
    "retrieval",
    "tool",
    "memory",
    "none",
    "misleading",
    "unknown",
]
ExpectedFailureMode = Literal[
    "none",
    "capacity_violation",
    "matching_failure",
    "decompression_failure",
    "geometric_distortion",
    "prior_dominance",
    "noise_failure",
    "unknown",
]
_CAPACITY_STRATA = {"strong", "weak_recoverable", "unsupported", "misleading"}
_SOURCE_CONDITIONS = {
    "weights",
    "context",
    "retrieval",
    "tool",
    "memory",
    "none",
    "misleading",
    "unknown",
}
_EXPECTED_FAILURE_MODES = {
    "none",
    "capacity_violation",
    "matching_failure",
    "decompression_failure",
    "geometric_distortion",
    "prior_dominance",
    "noise_failure",
    "unknown",
}
_CLAIM_SOURCE_LABELS = {
    "weights",
    "context",
    "retrieval",
    "tool",
    "memory",
    "unsupported",
    "untracked",
}


def _require_text(value: str, field_name: str) -> None:
    """Assert that a text field is non-empty."""
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} required"


def _require_non_negative(value: float, field_name: str) -> None:
    """Assert that a numeric field is non-negative."""
    assert value >= 0.0, f"{field_name} must be non-negative"


def _require_probability(value: float | None, field_name: str) -> None:
    """Assert that a score is either missing or in [0, 1]."""
    if value is None:
        return
    assert 0.0 <= value <= 1.0, f"{field_name} must be in [0, 1]"


def _metadata_text(metadata: dict[str, Any], key: str) -> str:
    """Read a required metadata field as text."""
    assert key in metadata, f"metadata missing {key}"
    value = metadata[key]
    assert isinstance(value, str), f"metadata {key} must be a string"
    _require_text(value, f"metadata {key}")
    return value


@dataclass(frozen=True)
class QAItem:
    """Represent one known-answer factual question.

    Args:
        item_id: Stable item identifier.
        prompt: Prompt given to the generation backend.
        answers: Accepted answer strings.
        metadata: HNC item metadata used for stratified experiments.

    Raises:
        AssertionError: If required fields are empty.
    """

    item_id: str
    prompt: str
    answers: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the question item."""
        _require_text(self.item_id, "item_id")
        _require_text(self.prompt, "prompt")
        assert len(self.answers) > 0, "at least one answer required"
        assert all(answer.strip() for answer in self.answers), "answers must be non-empty"
        _metadata_text(self.metadata, "topic")
        _metadata_text(self.metadata, "difficulty")
        capacity_stratum = _metadata_text(self.metadata, "capacity_stratum")
        source_condition = _metadata_text(self.metadata, "source_condition")
        expected_failure_mode = _metadata_text(self.metadata, "expected_failure_mode")
        assert capacity_stratum in _CAPACITY_STRATA, (
            f"invalid capacity_stratum: {capacity_stratum}"
        )
        assert source_condition in _SOURCE_CONDITIONS, (
            f"invalid source_condition: {source_condition}"
        )
        assert expected_failure_mode in _EXPECTED_FAILURE_MODES, (
            f"invalid expected_failure_mode: {expected_failure_mode}"
        )

    @property
    def topic(self) -> str:
        """Return the item topic."""
        return _metadata_text(self.metadata, "topic")

    @property
    def difficulty(self) -> str:
        """Return the item difficulty label."""
        return _metadata_text(self.metadata, "difficulty")

    @property
    def capacity_stratum(self) -> str:
        """Return the HNC capacity stratum."""
        return _metadata_text(self.metadata, "capacity_stratum")

    @property
    def source_condition(self) -> str:
        """Return the modeled source condition."""
        return _metadata_text(self.metadata, "source_condition")

    @property
    def expected_failure_mode(self) -> str:
        """Return the expected HNC failure mode."""
        return _metadata_text(self.metadata, "expected_failure_mode")


@dataclass(frozen=True)
class DecodingControl:
    """Record decoding parameters for one generation condition."""

    temperature: float
    top_p: float | None = None
    top_k: int | None = None
    sample_count: int = 1
    seed: int | None = None
    self_consistency_rule: str | None = None
    annealing_schedule: str | None = None

    def __post_init__(self) -> None:
        """Validate decoding controls."""
        _require_non_negative(self.temperature, "temperature")
        if self.top_p is not None:
            _require_probability(self.top_p, "top_p")
        if self.top_k is not None:
            assert self.top_k > 0, "top_k must be positive"
        assert self.sample_count > 0, "sample_count must be positive"


@dataclass(frozen=True)
class GenerationRequest:
    """Represent one generation request sent to a backend."""

    item_id: str
    prompt: str
    temperature: float
    sample_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the generation request."""
        _require_text(self.item_id, "item_id")
        _require_text(self.prompt, "prompt")
        _require_non_negative(self.temperature, "temperature")
        assert self.sample_index >= 0, "sample_index must be non-negative"


@dataclass(frozen=True)
class GenerationResult:
    """Represent raw backend output for one request."""

    request: GenerationRequest
    text: str
    backend_name: str
    model_name: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the generation result."""
        _require_text(self.backend_name, "backend_name")
        assert isinstance(self.text, str), "text must be a string"


@dataclass(frozen=True)
class ResultRow:
    """Represent one scored generation."""

    item_id: str
    temperature: float
    sample_index: int
    prompt: str
    expected_answers: str
    raw_output: str
    normalized_output: str
    is_exact_match: bool
    is_refusal: bool
    token_count: int
    backend_name: str
    model_name: str
    topic: str
    difficulty: str
    capacity_stratum: str
    source_condition: str
    expected_failure_mode: str
    metadata_json: str


@dataclass(frozen=True)
class AggregateRow:
    """Represent aggregate metrics for one temperature."""

    temperature: float
    item_count: int
    generation_count: int
    exact_match_count: int
    refusal_count: int
    accuracy: float
    refusal_rate: float
    self_consistency: float
    capacity_strata: str
    source_conditions: str
    expected_failure_modes: str


@dataclass(frozen=True)
class SourceSignal:
    """Record a source that can support a claim or answer."""

    source_type: Literal["weights", "context", "retrieval", "tool", "memory"]
    topic: str
    support_score: float
    evidence_uri: str | None = None
    timestamp: str | None = None

    def __post_init__(self) -> None:
        """Validate the source signal."""
        _require_text(self.topic, "topic")
        _require_probability(self.support_score, "support_score")


@dataclass(frozen=True)
class VerifierProfile:
    """Define a verifier used to check generated claims."""

    verifier_id: str
    verifier_type: Literal[
        "exact_match",
        "f1",
        "citation",
        "retrieval",
        "tool",
        "unit_test",
        "static_analysis",
        "human_rubric",
        "llm_judge",
        "other",
    ]
    domain: str
    target_description: str
    threshold: float | None
    source_required: bool

    def __post_init__(self) -> None:
        """Validate the verifier profile."""
        _require_text(self.verifier_id, "verifier_id")
        _require_text(self.domain, "domain")
        _require_text(self.target_description, "target_description")
        _require_probability(self.threshold, "threshold")


@dataclass(frozen=True)
class VerificationResult:
    """Record the result of checking one claim or answer."""

    verifier_id: str
    item_id: str
    claim_id: str | None
    passed: bool
    score: float
    evidence_refs: list[str]
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate the verification result."""
        _require_text(self.verifier_id, "verifier_id")
        _require_text(self.item_id, "item_id")
        _require_probability(self.score, "score")


@dataclass(frozen=True)
class CapacityEstimate:
    """Record a topic capacity estimate for one architecture profile."""

    topic: str
    architecture_profile: str
    requested_rate: float | None
    topic_capacity_proxy: float
    source_support_score: float
    capacity_stratum: CapacityStratum
    proxy_name: str
    measurement_notes: str

    def __post_init__(self) -> None:
        """Validate the capacity estimate."""
        _require_text(self.topic, "topic")
        _require_text(self.architecture_profile, "architecture_profile")
        if self.requested_rate is not None:
            _require_non_negative(self.requested_rate, "requested_rate")
        _require_non_negative(self.topic_capacity_proxy, "topic_capacity_proxy")
        _require_probability(self.source_support_score, "source_support_score")
        _require_text(self.proxy_name, "proxy_name")


@dataclass(frozen=True)
class ApproximationGap:
    """Record prediction-compression approximation proxies."""

    item_id: str
    architecture_profile: str
    ideal_predictor_reference: str | None
    neural_predictor_id: str
    log_loss_proxy: float | None
    code_length_proxy: float | None
    support_condition: Literal["in_support", "weak_support", "out_of_support", "unknown"]
    amortization_gap_proxy: float | None
    model_class_regret_proxy: float | None
    approximation_notes: str

    def __post_init__(self) -> None:
        """Validate approximation gap fields."""
        _require_text(self.item_id, "item_id")
        _require_text(self.architecture_profile, "architecture_profile")
        _require_text(self.neural_predictor_id, "neural_predictor_id")
        for field_name in (
            "log_loss_proxy",
            "code_length_proxy",
            "amortization_gap_proxy",
            "model_class_regret_proxy",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_negative(value, field_name)


@dataclass(frozen=True)
class EffectiveQuery:
    """Record context plus prompt as the query actually tested."""

    prompt: str
    context_digest: str
    task_type: str
    ambiguity_score: float
    requested_answer_rate: float

    def __post_init__(self) -> None:
        """Validate effective query fields."""
        _require_text(self.prompt, "prompt")
        _require_text(self.context_digest, "context_digest")
        _require_text(self.task_type, "task_type")
        _require_probability(self.ambiguity_score, "ambiguity_score")
        _require_non_negative(self.requested_answer_rate, "requested_answer_rate")


@dataclass(frozen=True)
class RepresentationMatch:
    """Record a candidate representation match."""

    match_id: str
    effective_query_id: str
    candidate_representation_id: str
    match_score: float
    ambiguity_score: float
    manifold_distance: float | None
    vigilance_threshold: float | None
    accepted: bool
    false_resonance_risk: float | None

    def __post_init__(self) -> None:
        """Validate representation match fields."""
        _require_text(self.match_id, "match_id")
        _require_text(self.effective_query_id, "effective_query_id")
        _require_text(self.candidate_representation_id, "candidate_representation_id")
        _require_probability(self.match_score, "match_score")
        _require_probability(self.ambiguity_score, "ambiguity_score")
        _require_probability(self.vigilance_threshold, "vigilance_threshold")
        _require_probability(self.false_resonance_risk, "false_resonance_risk")


@dataclass(frozen=True)
class ChannelStage:
    """Record one stage in a source-to-output channel."""

    name: Literal["encode", "match", "retrieve", "reconstruct", "decode", "verify"]
    input_signal: str
    output_signal: str
    noise_score: float
    distortion_score: float
    confidence_score: float

    def __post_init__(self) -> None:
        """Validate channel stage fields."""
        _require_text(self.input_signal, "input_signal")
        _require_text(self.output_signal, "output_signal")
        _require_probability(self.noise_score, "noise_score")
        _require_probability(self.distortion_score, "distortion_score")
        _require_probability(self.confidence_score, "confidence_score")


@dataclass(frozen=True)
class DecompressionBudget:
    """Record context crowding and reconstruction budget proxies."""

    item_id: str
    context_length: int
    query_complexity_proxy: float
    context_load_proxy: float
    latent_capacity_proxy: float | None
    available_room_proxy: float
    reconstruction_budget_proxy: float
    crowding_score: float

    def __post_init__(self) -> None:
        """Validate decompression budget fields."""
        _require_text(self.item_id, "item_id")
        assert self.context_length >= 0, "context_length must be non-negative"
        _require_non_negative(self.query_complexity_proxy, "query_complexity_proxy")
        _require_non_negative(self.context_load_proxy, "context_load_proxy")
        if self.latent_capacity_proxy is not None:
            _require_non_negative(self.latent_capacity_proxy, "latent_capacity_proxy")
        _require_non_negative(self.available_room_proxy, "available_room_proxy")
        _require_non_negative(self.reconstruction_budget_proxy, "reconstruction_budget_proxy")
        _require_probability(self.crowding_score, "crowding_score")


@dataclass(frozen=True)
class CompressedContextTrace:
    """Record a TaC style dynamic codebook trace."""

    trace_id: str
    item_id: str
    thinker_model: str
    answerer_model: str
    original_context_tokens: int
    trace_tokens: int
    target_compression_ratio: float
    actual_compression_ratio: float
    trace_text: str
    trace_utility_score: float | None
    trace_faithfulness_score: float | None
    answer_leakage_flag: bool
    unsupported_claim_rate: float | None
    downstream_exact_match: float | None
    downstream_f1: float | None

    def __post_init__(self) -> None:
        """Validate compressed context trace fields."""
        _require_text(self.trace_id, "trace_id")
        _require_text(self.item_id, "item_id")
        _require_text(self.thinker_model, "thinker_model")
        _require_text(self.answerer_model, "answerer_model")
        assert self.original_context_tokens > 0, "original_context_tokens must be positive"
        assert self.trace_tokens >= 0, "trace_tokens must be non-negative"
        _require_non_negative(self.target_compression_ratio, "target_compression_ratio")
        _require_non_negative(self.actual_compression_ratio, "actual_compression_ratio")
        for field_name in (
            "trace_utility_score",
            "trace_faithfulness_score",
            "unsupported_claim_rate",
            "downstream_exact_match",
            "downstream_f1",
        ):
            _require_probability(getattr(self, field_name), field_name)


@dataclass(frozen=True)
class DistortionTrace:
    """Record per-stage distortion for a generation path."""

    trace_id: str
    stage_names: list[str]
    stage_errors: list[float]
    total_fidelity: float
    correlated_error_flag: bool
    parallel_error: float | None
    perpendicular_error: float | None
    downstream_error_rate: float | None

    def __post_init__(self) -> None:
        """Validate distortion trace fields."""
        _require_text(self.trace_id, "trace_id")
        assert len(self.stage_names) == len(self.stage_errors), (
            "stage_names and stage_errors length mismatch"
        )
        assert len(self.stage_names) > 0, "at least one stage required"
        for error in self.stage_errors:
            _require_probability(error, "stage_error")
        _require_probability(self.total_fidelity, "total_fidelity")
        _require_probability(self.parallel_error, "parallel_error")
        _require_probability(self.perpendicular_error, "perpendicular_error")
        _require_probability(self.downstream_error_rate, "downstream_error_rate")


@dataclass(frozen=True)
class RetrievalMemoryChannel:
    """Record RAG or memory consolidation geometry."""

    channel_id: str
    corpus_id: str
    embedding_model: str
    unit_norm: bool
    retrieval_metric: Literal["cosine", "dot", "learned"]
    theta: float
    retrieval_slack: float
    mean_within_cluster_distance: float
    local_effective_dimension: float
    representative_budget: int
    consolidation_operator: Literal[
        "raw",
        "centroid",
        "medoid",
        "summary",
        "prune",
        "quantized",
        "other",
    ]
    predicted_identity_floor: float | None
    observed_identity_error: float
    observed_coverage_error: float
    downstream_exact_match: float | None
    downstream_unsupported_claim_rate: float | None

    def __post_init__(self) -> None:
        """Validate retrieval memory channel fields."""
        _require_text(self.channel_id, "channel_id")
        _require_text(self.corpus_id, "corpus_id")
        _require_text(self.embedding_model, "embedding_model")
        _require_non_negative(self.theta, "theta")
        _require_non_negative(self.retrieval_slack, "retrieval_slack")
        _require_non_negative(
            self.mean_within_cluster_distance, "mean_within_cluster_distance"
        )
        _require_non_negative(self.local_effective_dimension, "local_effective_dimension")
        assert self.representative_budget >= 0, "representative_budget must be non-negative"
        _require_probability(self.predicted_identity_floor, "predicted_identity_floor")
        _require_probability(self.observed_identity_error, "observed_identity_error")
        _require_probability(self.observed_coverage_error, "observed_coverage_error")
        _require_probability(self.downstream_exact_match, "downstream_exact_match")
        _require_probability(
            self.downstream_unsupported_claim_rate,
            "downstream_unsupported_claim_rate",
        )


@dataclass(frozen=True)
class DenoisingTrace:
    """Record source conditioned repair of an answer state."""

    trace_id: str
    item_id: str
    initial_answer: str
    repaired_answer: str
    source_refs: list[str]
    effective_query_id: str
    denoiser_id: str
    verifier_id: str
    source_conditioned: bool
    denoising_step_count: int
    unsupported_claims_before: int
    unsupported_claims_after: int
    supported_claims_before: int
    supported_claims_after: int
    abstention_after: bool
    retrieval_requested_after: bool
    exact_match_before: float | None
    exact_match_after: float | None
    repair_notes: str | None = None

    def __post_init__(self) -> None:
        """Validate denoising trace fields."""
        _require_text(self.trace_id, "trace_id")
        _require_text(self.item_id, "item_id")
        _require_text(self.effective_query_id, "effective_query_id")
        _require_text(self.denoiser_id, "denoiser_id")
        _require_text(self.verifier_id, "verifier_id")
        assert self.denoising_step_count > 0, "denoising_step_count must be positive"
        for field_name in (
            "unsupported_claims_before",
            "unsupported_claims_after",
            "supported_claims_before",
            "supported_claims_after",
        ):
            assert getattr(self, field_name) >= 0, f"{field_name} must be non-negative"
        _require_probability(self.exact_match_before, "exact_match_before")
        _require_probability(self.exact_match_after, "exact_match_after")


@dataclass(frozen=True)
class SinkCompressionProbe:
    """Record open-weight sink-compression measurements."""

    probe_id: str
    model_name: str
    layer_index: int
    token_position: int
    beginning_of_sequence_norm_ratio: float
    sink_rate: float
    matrix_entropy: float
    anisotropy: float
    mixing_score: float | None
    colsum_concentration: float | None
    sink_vs_identity_index: float | None
    layer_phase: Literal["early_mixing", "middle_compression", "late_refinement", "unknown"]
    downstream_failure_rate: float | None
    downstream_exact_match: float | None = None
    downstream_unsupported_claim_rate: float | None = None

    def __post_init__(self) -> None:
        """Validate sink compression probe fields."""
        _require_text(self.probe_id, "probe_id")
        _require_text(self.model_name, "model_name")
        assert self.layer_index >= 0, "layer_index must be non-negative"
        assert self.token_position >= 0, "token_position must be non-negative"
        _require_non_negative(
            self.beginning_of_sequence_norm_ratio,
            "beginning_of_sequence_norm_ratio",
        )
        _require_probability(self.sink_rate, "sink_rate")
        _require_non_negative(self.matrix_entropy, "matrix_entropy")
        _require_non_negative(self.anisotropy, "anisotropy")
        _require_probability(self.mixing_score, "mixing_score")
        _require_probability(self.colsum_concentration, "colsum_concentration")
        _require_probability(self.sink_vs_identity_index, "sink_vs_identity_index")
        _require_probability(self.downstream_failure_rate, "downstream_failure_rate")
        _require_probability(self.downstream_exact_match, "downstream_exact_match")
        _require_probability(
            self.downstream_unsupported_claim_rate,
            "downstream_unsupported_claim_rate",
        )


@dataclass(frozen=True)
class GenerationRegime:
    """Record the selected generation regime for an item."""

    regime_type: Literal["direct", "retrieval", "tool", "decomposition", "memory", "abstain"]
    reason: str
    expected_source_support: float
    verifier_required: bool

    def __post_init__(self) -> None:
        """Validate generation regime fields."""
        _require_text(self.reason, "reason")
        _require_probability(self.expected_source_support, "expected_source_support")


@dataclass(frozen=True)
class ClaimAttributionRow:
    """Record source attribution for one generated claim."""

    item_id: str
    claim_id: str
    claim_text: str
    source_label: Literal[
        "weights",
        "context",
        "retrieval",
        "tool",
        "memory",
        "unsupported",
        "untracked",
    ]
    evidence_refs: list[str]
    support_score: float
    verifier_result_id: str | None
    attribution_confidence: float

    def __post_init__(self) -> None:
        """Validate claim attribution fields."""
        _require_text(self.item_id, "item_id")
        _require_text(self.claim_id, "claim_id")
        _require_text(self.claim_text, "claim_text")
        assert self.source_label in _CLAIM_SOURCE_LABELS, (
            f"invalid source_label: {self.source_label}"
        )
        _require_probability(self.support_score, "support_score")
        _require_probability(self.attribution_confidence, "attribution_confidence")


@dataclass(frozen=True)
class ClaimBinding:
    """Bind a theory claim to required measurements."""

    claim_id: str
    architecture_profile: str
    required_variables: list[str]
    allowed_proxies: list[str]
    expected_pattern: str
    falsification_condition: str

    def __post_init__(self) -> None:
        """Validate claim binding fields."""
        _require_text(self.claim_id, "claim_id")
        _require_text(self.architecture_profile, "architecture_profile")
        assert len(self.required_variables) > 0, "required_variables required"
        assert len(self.allowed_proxies) > 0, "allowed_proxies required"
        _require_text(self.expected_pattern, "expected_pattern")
        _require_text(self.falsification_condition, "falsification_condition")


@dataclass(frozen=True)
class ArchitectureProfile:
    """Record what an experiment can and cannot measure."""

    model_family: str
    access_level: Literal["api_text", "logprobs", "hidden_states", "attention", "trainable"]
    retrieval: dict[str, Any] | None
    tool_use: dict[str, Any] | None
    memory: dict[str, Any] | None
    decoding_controls: list[DecodingControl]
    verifier: VerifierProfile

    def __post_init__(self) -> None:
        """Validate architecture profile fields."""
        _require_text(self.model_family, "model_family")
        assert len(self.decoding_controls) > 0, "decoding_controls required"
