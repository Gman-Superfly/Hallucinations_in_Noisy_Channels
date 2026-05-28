# Source conditioned denoising experiment

This folder is reserved for the source conditioned denoising experiment.

## Relation to the main HNC document

This experiment maps to:

- Section 8.7, diffusion interpretation.
- Conjecture 9, source conditioned denoising.
- Prediction 27, denoising benefit.
- Prediction 28, no source limit.
- Project status Section 3.4, diffusion as the testable dynamics layer.

## Planned measurements

The first implementation should generate an initial answer, apply a source conditioned repair step, and write `DenoisingTrace` rows with:

- unsupported claims before and after repair,
- supported claims before and after repair,
- source references,
- exact match before and after repair,
- abstention after repair,
- retrieval request after repair.

## Evidence boundary

Denoising supports HNC only when repaired claims are tied to a modeled source. If repair improves an unsupported item without retrieval, tool output, or supplied context, then the result should be treated as untracked source signal, benchmark leakage, or verifier failure.
