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
- payload retention,
- routing payload mismatch,
- answer leakage flag,
- unsupported claim rate,
- downstream exact match or F1.

## Payload audit

This experiment should separate routing from payload preservation.

Routing asks whether the trace points to the right region of the source context. Payload preservation asks whether the exact claim supporting content survived inside the trace. A trace can route correctly and still lose the value like payload needed to support the final claim.

The first audit should mark each generated claim with:

- the source span that should support it,
- whether that source span was selected or summarized,
- whether the trace retained the claim supporting payload,
- whether the answer used the retained payload accurately.

## Evidence boundary

A trace that improves answer score can still hide source accounting errors. The trace must be audited against the original context before it counts as support for HNC. It should count as source preserving only when the claim supporting payload survives the compression step.
