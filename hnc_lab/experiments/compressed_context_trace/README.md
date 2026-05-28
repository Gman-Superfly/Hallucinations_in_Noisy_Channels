# Compressed context trace experiment

This folder is reserved for the TaC style compressed context trace experiment.

## Relation to the main HNC document

This experiment maps to:

- Section 4.5.2, Kolmogorov garbage.
- Section 4.5.5, context and decompression implications.
- Section 7.6, verifiable generation regimes.
- Project status Section 1.2, what the Thinking as Compression paper changes.
- Research lead Experiment 7, thinking traces as compressed context.

## Planned measurements

The first implementation should compare full context, generic summaries, token pruning, and query conditioned traces. It should write `CompressedContextTrace` rows with:

- original context tokens,
- trace tokens,
- target and actual compression ratio,
- trace utility,
- trace faithfulness,
- answer leakage flag,
- unsupported claim rate,
- downstream exact match or F1.

## Evidence boundary

A trace that improves answer score can still hide source accounting errors. The trace must be audited against the original context before it counts as support for HNC.
