# Hallucinations in Noisy Channels — Theory Index

**Version**: 1.2.1  
**Last Updated**: November 2025  
**Main Document**: `Hallucinations_in_Noisy_Channels_v1.2.1.md`

---

## Quick Reference

| Core Principle | Statement |
|----------------|-----------|
| **The Hallucination Equation** | $K(\text{output}) \leq K(\text{weights}) + K(\text{context})$ — violation = hallucination |
| **Intelligence** | Compression (finding minimal programs that capture structure) |
| **Teaching** | Rate-matched decompression through a noisy channel |
| **Hallucination** | Information creation — output contains more than source provides |
| **Form Prior** | Maximum entropy attractor (thermal bath) when knowledge constraints fail |
| **Effective Query** | $Q_{eff} = \text{Attention}(p, S_{ctx})$ — context colors the prompt |

---

## Table of Contents

1. [Definitions](#definitions)
2. [Theorems](#theorems)
3. [Propositions](#propositions)
4. [Predictions](#predictions)
5. [Key Concepts & Intuitions](#key-concepts--intuitions)
6. [Hallucination Mechanisms](#hallucination-mechanisms)
7. [Mitigations](#mitigations)
8. [Section Map](#section-map)

---

## Definitions

| # | Name | Section | Description |
|---|------|---------|-------------|
| 1 | Learning as Compression | 2.1 | Training ≡ finding minimal description of regularities in data |
| 2 | Inference as Multi-Stage Reconstruction | 2.2 | Generation as teaching through cascaded noisy channel |
| 3 | Knowledge Capacity | 2.3 | $C_T$ = max rate of reliable information about topic $T$ |
| 4 | Hallucination | 3.1 | Output with $H(O \mid \mathcal{F}) > H(O \mid \mathcal{F}, \mathcal{C}_T)$ |
| 5 | In-Context Learning | 4.1 | Temporary capacity boost via examples in context |
| 6 | Representation Matching | 4.4.1 | Activation based on structural similarity to prompt |
| 7 | Decompression Room | 4.5.4 | Latent capacity budget for reconstructing compressed knowledge |
| 8 | Sink Severity | 4.6 | Fraction of attention mass in first $k$ tokens |
| 9 | Information Atom | 4.7 | Compressed pattern from training — irreducible knowledge unit |
| 10 | Test-Time Atom | 4.7.6 | Atom created during inference via test-time learning |
| 11 | Manifold-Based Capacity Estimator | 7.4 | $\hat{C}_T$ via embedding density, translation fidelity, confidence |
| 12 | Distortion Operator | 8.4 | Per-stage error characteristic in pipeline |
| 13 | Adaptive Resonance Condition | 8.6.8 | Resonance when match exceeds adaptive threshold $\rho$ |
| 14 | The Universal Manifold | 11.5 | Shared geometric structure for truthful representations |
| — | Teaching | 2.2.1 | Rate-matched decompression + redundancy coding |
| — | Operational Intelligence | 1.2 | Teaching capacity — max reliable knowledge transmission rate |
| — | Effective Query | 4.4.0 | Holistic combination of prompt + context state |

---

## Theorems

| # | Name | Section | Statement |
|---|------|---------|-----------|
| 1 | Hallucination Threshold | 2.3 | If $R_T > C_T$, hallucinations unavoidable regardless of decoding |
| 2 | Geometric Matching | 4.4.4 | Retrieval accuracy ∝ softmax of manifold distances |
| 3 | Information Conservation | 8.3 | $H(O \mid S, T) = 0$ for truthful generation; $> 0$ for hallucination |
| 4 | Geometric Distortion Accumulation | 8.4.3 | Fidelity = $\prod_i (1 - \epsilon_i)$ — multiplicative error cascade |
| 5 | Thermodynamic Hallucination | 8.5.5 | $P(\text{hallucination}) \propto \exp(\Delta S)$ where $\Delta S = S_{form} - S_{knowledge}$ |
| 6 | Optimal Noise Principle | 8.6.5 | ∃ optimal $\sigma^*$ maximizing correction − hallucination trade-off |
| 7 | Adaptive Resonance Optimality | 8.6.8 | ∃ optimal vigilance $\rho^*$ minimizing match failures + false resonance |
| 8 | Model-Specific Sampling Limit | 11.6 | Nyquist-style: constraint sampling rate $s > 2B_{M,T}$ required |

---

## Propositions

| # | Name | Section | Statement |
|---|------|---------|-----------|
| 1 | Compression = Understanding | 2.1 | Higher useful compression = deeper abstraction |
| 2 | Hallucination as Entropy Maximization | 3.3 | Hallucination maximizes $H(O \mid \mathcal{F})$ subject to form |
| 3 | Confidence-Accuracy Decoupling | 3.3 | On OOD topics, confidence tracks fluency, not accuracy |
| 4 | Ambiguity-Induced Hallucination | 4.4.4 | Multiple similar-activation representations → composite output |
| 5 | Context Crowding | 4.5.4 | Insufficient decompression room → Kolmogorov garbage |
| 6 | Decompression-Compression Asymmetry | 4.5.4 | $K(\text{reconstruct}) \gg K(\text{store})$ |
| 7 | Sink-Limited Capacity | 4.6.2 | $\partial C_{ctx}/\partial s \leq 0$ — sinks reduce capacity |
| 8 | Test-Time Learning as Capacity Extension | 4.7.6 | $C_T^{effective} = C_T^{static} + \Delta C_T(ctx)$ |
| 9 | Compression-Transmission Trade-off | 5.1 | Pareto frontier between compression efficiency and transmission reliability |
| 10 | Detectability | 6.2 | Conservation violations are detectable via complexity comparison |
| 11 | Manifold Departure | 8.4.3 | Distortion = on-manifold (recoverable) + off-manifold (hallucination) |
| — | Hallucination as Teaching Failure | 2.2.2 | Decompression failure, not compression failure |

---

## Predictions

| # | Name | Section | Testable Claim |
|---|------|---------|----------------|
| 1 | Frequency-Accuracy Correlation | 9.1 | Higher training frequency → lower hallucination rate |
| 2 | Few-Shot Logarithmic Improvement | 9.1 | $P(\text{hall}) \propto 1/\log(1+k)$ with $k$ examples |
| 3 | Confidence-Grounding Decoupling | 9.1 | Corr(confidence, fluency) > Corr(confidence, accuracy) on OOD |
| 4 | Prompt Specificity Effect | 9.1 | $P(\text{hall}) \approx 1 - \exp(-d_{\mathcal{M}}^2/2\sigma^2)$ |
| 5 | Context Crowding Effect | 9.1 | Hallucination ∝ $1/(K_{latent} - K_{query} - K_{context})$ |
| 6 | Decompression Asymmetry | 9.1 | Complex topics need disproportionately more context room |
| 7 | Information Conservation Violation | 9.1 | $K(\text{output}) > K(\text{source})$ indicates hallucination |
| 8 | Excess Information Source | 9.1 | Excess comes from form prior, not knowledge |
| 9 | Geometric Distortion Accumulation | 9.1 | Error compounds multiplicatively through pipeline |
| 10 | First-Stage Dominance | 9.1 | Early errors have outsized impact |
| 11 | Multi-Hop Degradation | 9.1 | Accuracy degrades with reasoning chain length |
| 12 | Temperature-Hallucination Relationship | 9.1 | U-shaped: low T brittle, high T hallucinates |
| 13 | Entropy Ratio Prediction | 9.1 | $P(\text{hall}) \propto \exp(S_{form} - S_{knowledge})$ |
| 14 | Free Energy Minimization | 9.1 | Generation minimizes $F = E - TS$ |
| 15 | Optimal Noise Existence | 9.1 | ∃ $\sigma^*$ where noise helps retrieval |
| 16 | Stochastic Resonance | 9.1 | Weak memories retrievable with optimal noise |
| 17 | Self-Consistency Optimality | 9.1 | Multiple samples + voting improves accuracy |
| 18 | Goldilocks Context Window | 9.1 | U-shaped error: too little context = no constraints, too much = crowding |
| 19 | Geometry-Aligned Warm-Start | 9.1 | Initializing near truth manifold improves training |
| 20 | Geometry-Driven Training Diagnostics | 9.1 | Manifold distance predicts validation accuracy |
| 21 | Position Primacy | 4.6 | Late-context evidence degrades with sink severity |
| 22 | Atom Coverage | 4.7 | Fewer activated atoms → higher hallucination rate |
| 23 | Adaptive Resonance Peak | 8.6.8 | Weak knowledge benefits from joint (σ, ρ) tuning |
| 24 | Knowledge-Contingent Optimum | 8.6.8 | Optimal (σ*, ρ*) varies with topic capacity |
| 25 | Test-Time Learning Reduces Hallucination | 11.7 | Titans-style architectures have lower hallucination on partial-coverage topics |
| 26 | Memory Hierarchy Optimality | 11.7 | Multi-tier memory outperforms monolithic architectures |

---

## Key Concepts & Intuitions

### Intuition Blocks (New in v1.2.1)
| # | Name | Section | Analogy |
|---|------|---------|---------|
| 1 | The Bayesian Prior | 1.6 | Car colors: default expectations before evidence |
| 2 | The Thermal Bath | 1.7 | Ice melting: knowledge (ice) vs form prior (room temp) |
| 3 | Compression IS Understanding | 2.1.0 | Sequence prediction: memorizing vs. learning the rule |
| 4 | The Teacher's Dilemma | 2.2.1 | Teaching a topic you don't know = faking it |
| 5 | The Confabulation Mechanism | 3.2.1 | Why the model "must" lie to complete the pattern |
| 6 | The Effective Query | 4.4.0 | Lens and Light: context filters the prompt |
| 7 | The Library Paradox | 8.3.0 | You can't summarize a 100pg book into 200pgs |
| 8 | The Telephone Game | 8.4.0 | Geometric distortion accumulation |
| 9 | The Stuck Lock | 8.6.0 | Stochastic resonance: noise jiggles the key |
| 10 | Plato's Cave | 11.5.0 | Universal manifold: shared geometry of truth |
| 11 | The Open-Book Exam | 11.7.0 | Test-time learning: writing notes during the test |

---

## Hallucination Mechanisms

### The Six Mechanisms

| # | Mechanism | Section | Cause | Result |
|---|-----------|---------|-------|--------|
| 1 | **Capacity Violation** | 3 | $R_T > C_T$ — asking beyond knowledge | Nothing to retrieve → max entropy |
| 2 | **Matching Failure** | 4.4 | Ambiguous effective query → wrong representation | Composite/wrong retrieval |
| 3 | **Decompression Failure** | 4.5 | Insufficient context room | Kolmogorov garbage |
| 4 | **Geometric Distortion** | 8.4 | Multiplicative error cascade | Accumulated corruption |
| 5 | **Thermodynamic Equilibration** | 8.5 | Constraints fail → form prior | Fluent but empty (thermalization) |
| 6 | **Noise Paradox** | 8.6 | Too much/little stochasticity | Brittle or hallucinating |

---

## Section Map

| Section | Title | Key Content |
|---------|-------|-------------|
| **1** | Introduction | Core insight, **Bayesian Prior**, **Thermal Bath** |
| **2** | Theoretical Framework | Compression/transmission, **Teacher's Dilemma** |
| **3** | Hallucinations as Capacity Violations | **Confabulation Mechanism** |
| **4** | Reconstruction Failures | **Effective Query**, **Lens vs Load**, Atoms |
| **5** | Compression-Transmission Duality | LLMs as teachers |
| **6** | The Conservation Law | Information cannot be created |
| **7** | Capacity Estimation | Practical estimators |
| **8** | Thermodynamic & Geometric Views | **Library Paradox**, **Telephone Game**, **Stuck Lock** |
| **9** | Experimental Predictions | 26 testable predictions |
| **10** | Mathematical Foundations | Rigorous derivations |
| **11** | Conclusion & Extensions | **Plato's Cave**, **Open-Book Exam** |
| **App A** | Duality Table | Quick reference mapping |
| **App B** | Hallucination as Constraint Absence | Visual summary |

---

## Citation

```bibtex
@misc{goldman2025hallucinations,
  author = {Oscar Goldman},
  title = {Hallucinations in Noisy Channels: Information-Theoretic Framework for Understanding LLM Hallucination Errors},
  year = {2025},
  institution = {Shogu Research Group @ Datamutant.ai},
  note = {Theoretical Framework v1.2.1}
}
```

---

*This index is auto-generated from Hallucinations_in_Noisy_Channels_v1.2.1.md*
