# Temperature sweep experiment

This folder contains the temperature sweep experiment for the HNC lab.

## Relation to the main HNC document

This experiment maps to:

- Prediction 12, temperature and hallucination relationship.
- Prediction 15, optimal noise existence.
- Prediction 16, stochastic resonance.
- Section 8.5, thermodynamic interpretation.
- Section 8.6, functional role of noise.

The experiment tests whether decoding temperature changes exact match, refusal, answer variance, and later unsupported claim labels across source support strata.

## Current status

This folder contains a runnable experiment runner. With the `fixture` backend, it only checks the pipeline. A fixture run is not evidence for HNC claims.

Evidence requires:

- a real backend,
- a stated model,
- fixed prompts,
- preserved raw outputs,
- item metadata for capacity stratum and expected failure mode,
- scoring rules,
- negative or flat results if they occur.

## Run

Run from the repository root:

```powershell
python -m hnc_lab.experiments.temperature_sweep --backend fixture
```

For an OpenAI compatible backend:

```powershell
$env:HNC_OPENAI_API_KEY = "..."
$env:HNC_OPENAI_MODEL = "..."
$env:HNC_OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"
python -m hnc_lab.experiments.temperature_sweep --backend openai_compatible --samples 3
```

The runner writes raw rows, aggregate rows, and metadata logs under `figures/`.
