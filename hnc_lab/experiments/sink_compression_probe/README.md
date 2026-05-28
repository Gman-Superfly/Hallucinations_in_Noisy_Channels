# Sink compression probe experiment

This folder is reserved for the open weight sink compression experiment.

## Relation to the main HNC document

This experiment maps to:

- Section 4.6, attention sinks and anchoring.
- Prediction 21, sink limited capacity.
- Project status Section 1.1, the attention sink paper.
- Research lead Experiment 6, massive activations, sink compression, and late context failure.

## Planned measurements

The first implementation should record `SinkCompressionProbe` rows with:

- beginning of sequence norm ratio,
- sink rate,
- matrix entropy,
- anisotropy,
- mixing score,
- column sum concentration,
- sink versus identity index,
- layer phase,
- downstream exact match,
- downstream unsupported claim rate.

## Evidence boundary

This experiment requires hidden states and attention weights from an open weight decoder only model. API text runs can test downstream behavior, but they cannot measure the internal sink compression variables.
