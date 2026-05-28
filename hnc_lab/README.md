# HNC lab

This package contains shared lab utilities and experiment folders for turning HNC claims into typed rows. Shared infrastructure lives at the package root. Each experiment lives under `hnc_lab/experiments/<experiment_name>/`.

The lab should be read as an experiment harness. It does not make a run into evidence by itself. A run becomes usable evidence only when the backend, model, prompts, item strata, raw outputs, scoring rules, and negative results are recorded.

## Folder map

| Folder | Status | Main HNC relation |
|---|---|---|
| `experiments/temperature_sweep` | Runnable | Predictions 12, 15, and 16; Sections 8.5 and 8.6 |
| `experiments/sink_compression_probe` | Planned | Section 4.6; Prediction 21; attention sink research lead |
| `experiments/source_conditioned_denoising` | Planned | Section 8.7; Conjecture 9; Predictions 27 and 28 |
| `experiments/compressed_context_trace` | Planned | Section 4.5.2; Section 7.6; Thinking as Compression research lead |
| `experiments/claim_attribution` | Planned | Section 8.1; source accounting and verification |
| `experiments/compression_proxy_pass` | Planned | Section 2.1.0; Section 11.5.0; algorithmic compression review |

## Shared infrastructure

The package root contains shared code used by experiment folders:

- `schemas.py`: typed rows for HNC experiment records.
- `datasets.py`: JSONL loading and metadata validation.
- `backends.py`: fixture and OpenAI compatible generation backends.
- `metrics.py`: exact match, refusal, self consistency, and denoising helper metrics.
- `verification.py`: verifier, claim attribution, and denoising trace helpers.

## Current runnable experiment

The temperature sweep in `experiments/temperature_sweep` tests whether answer behavior changes across decoding temperature. It uses known answer factual QA so the first baseline can use exact match rather than an LLM judge. It writes raw generation rows, aggregate rows, and metadata logs under `figures/`, which is ignored by git.

Run a pipeline check from the repository root:

```powershell
python -m hnc_lab.experiments.temperature_sweep --backend fixture
```

The `fixture` backend tests the pipeline only. It simulates outputs from the seed answers so metrics, CSV writing, metadata preservation, and aggregation can be checked without calling a model. Fixture runs are not evidence for HNC predictions.

## Real backend hook

The `openai_compatible` backend uses the standard chat completions shape and requires no extra Python package. Set these environment variables before running:

```powershell
$env:HNC_OPENAI_API_KEY = "..."
$env:HNC_OPENAI_MODEL = "..."
$env:HNC_OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"
python -m hnc_lab.experiments.temperature_sweep --backend openai_compatible --samples 3
```

`HNC_OPENAI_BASE_URL` is optional when using the OpenAI endpoint. Use a local or compatible server by changing that URL. The output rows record `backend_name`, `model_name`, raw output, normalized output, exact match, refusal status, token count, and HNC item metadata.

## Required item metadata

Each JSONL item must include:

```json
{
  "topic": "geography",
  "difficulty": "easy",
  "capacity_stratum": "strong",
  "source_condition": "weights",
  "expected_failure_mode": "none"
}
```

The required fields let later experiments separate strong source, weak recoverable, unsupported, and misleading items. They also let the project test whether a failure matches the expected HNC mechanism.

## Typed records

`hnc_lab.schemas` includes the current experiment objects from the paper and project status:

- `CapacityEstimate` for topic capacity and source support.
- `ApproximationGap` for log loss, code length, and support condition records.
- `ClaimAttributionRow`, `VerifierProfile`, and `VerificationResult` for source accounting.
- `CompressedContextTrace` for TaC style dynamic codebook experiments.
- `DenoisingTrace` for source conditioned repair experiments.
- `DecodingControl`, `DistortionTrace`, `RetrievalMemoryChannel`, and `SinkCompressionProbe` for later architecture specific paths. `SinkCompressionProbe` records beginning of sequence norm ratio, matrix entropy, anisotropy, sink rate, mixing score, column sum concentration, sink versus identity index, layer phase, and downstream HNC outcome fields.

These are schema stubs, not proof. They make each run explicit about what it measured and what it did not measure.

## Experiment folder status

Each planned experiment folder has its own README with the HNC document mapping, planned measurements, and evidence boundary. The next low cost work is:

1. Add weak recoverable, unsupported, and misleading items to the seed data for `temperature_sweep`.
2. Run `temperature_sweep` with a real backend and preserve the raw CSV rows.
3. Add manual `ClaimAttributionRow` labels in `claim_attribution`.
4. Run a draft and repair pass in `source_conditioned_denoising`.
5. Add a prompt only compressed trace pass in `compressed_context_trace`.
6. Add a small open weight probe run in `sink_compression_probe`.
