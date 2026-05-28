# Compression proxy pass experiment

This folder is reserved for compression proxy and approximation gap experiments.

## Relation to the main HNC document

This experiment maps to:

- Section 2.1.0, source coding as compression.
- Section 11.5.0, approximation gap.
- Project status Section 1.3, what the algorithmic compression review changes.
- Research lead Experiment 8, compression proxies and source support.

## Planned measurements

The first implementation should record `ApproximationGap` rows or compatible result rows with:

- log loss proxy,
- code length proxy,
- support condition,
- amortization gap proxy when available,
- model class regret proxy when available,
- exact match,
- refusal quality,
- source attribution labels,
- unsupported claim rate.

## Evidence boundary

Compression proxies are support measurements. They do not prove hallucination mechanisms unless they predict unsupported output under defined source support strata.
