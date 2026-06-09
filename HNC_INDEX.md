# Hallucinations in Noisy Channels: theory index

**Version**: 1.2.2  
**Last updated**: May 2026
**Main Document**: `Hallucinations_in_Noisy_Channels_v1.2.2.md`

---

## Quick reference

| Core Principle | Statement |
|----------------|-----------|
| **Source-accounting equation** | $K(\text{output}) \leq K(\text{weights}) + K(\text{context})$ as a proxy; excess is a hallucination candidate |
| **Intelligence** | Compression that preserves task-relevant structure |
| **Teaching** | Rate-matched decompression through a noisy channel |
| **Hallucination** | Grounded-source failure: output contains information not supported by stored or supplied source signal |
| **Form prior** | High-entropy learned prior that can dominate when content constraints fail |
| **Effective query** | $Q_{eff} = \text{Attention}(p, S_{ctx})$; context shapes the prompt |
| **Verifiable representation** | Representation that supports outputs passing a task-specific check against sources, logic, tools, retrieval, or experiment |

---

## Table of contents

1. [Definitions](#definitions)
2. [Formal claims](#formal-claims)
3. [Propositions](#propositions)
4. [Predictions](#predictions)
5. [Key Concepts & Intuitions](#key-concepts--intuitions)
6. [Hallucination mechanisms](#hallucination-mechanisms)
7. [Mitigations](#mitigations)
8. [Section map](#section-map)

---

## Definitions

| # | Name | Section | Description |
|---|------|---------|-------------|
| 1 | Learning as compression | 2.1 | Training preserves useful structure at lower description cost |
| N/A | Verifiable representation | 2.1.1 | Representation that can support task-specific verification |
| 2 | Inference as multi-stage reconstruction | 2.2 | Generation as teaching through cascaded noisy channel |
| 3 | Knowledge capacity | 2.3 | $C_T$ = max rate of reliable information about topic $T$ |
| 4 | Hallucination | 3.1 | Output with $H(O \mid \mathcal{F}) > H(O \mid \mathcal{F}, \mathcal{C}_T)$ |
| 5 | In-Context Learning | 4.1 | Temporary capacity boost via examples in context |
| 6 | Representation Matching | 4.4.1 | Activation based on structural similarity to prompt |
| 7 | Decompression Room | 4.5.4 | Latent capacity budget for reconstructing compressed knowledge |
| 8 | Sink Severity | 4.6 | Fraction of attention mass in first $k$ tokens |
| 9 | Information Atom | 4.7 | Compressed pattern from training; irreducible knowledge unit |
| 10 | Test-Time Atom | 4.7.6 | Atom created during inference via test-time learning |
| 11 | Manifold-based capacity estimator | 7.4 | $\hat{C}_T$ via embedding density, translation fidelity, confidence |
| 12 | Distortion operator | 8.4 | Per-stage error characteristic in pipeline |
| 13 | Adaptive resonance condition | 8.6.8 | Resonance when match exceeds adaptive threshold $\rho$ |
| 14 | Universal manifold hypothesis | 11.5 | Hypothesized shared or overlapping geometry for task-relevant verifiable representations |
| N/A | Teaching | 2.2.1 | Rate-matched decompression + redundancy coding |
| N/A | Operational Intelligence | 1.2 | Teaching capacity: max reliable knowledge transmission rate |
| N/A | Effective Query | 4.4.0 | Holistic combination of prompt + context state |

---

## Formal claims

| # | Name | Section | Statement |
|---|------|---------|-----------|
| Corollary 1 | Hallucination threshold | 2.3 | If $R_T > C_T$, reliable source-supported generation exceeds the modeled channel limit |
| Model 2 | Geometric matching proxy | 4.4.4 | Retrieval accuracy is modeled by a softmax over candidate manifold distances |
| Conjecture 3 | Regime-aligned generation | 7.6 | A router should select a generation regime whose source signal and verifier support the requested rate |
| 3 | Information conservation | 8.3 | Unsupported entropy remains when the modeled source cannot explain the output |
| 4 | Geometric distortion accumulation | 8.4.3 | Fidelity = $\prod_i (1 - \epsilon_i)$; multiplicative error cascade |
| Conjecture 5 | Maximum-entropy hallucination model | 8.5.5 | Prior relaxation risk rises under weak content constraints |
| Conjecture 6 | Optimal noise principle | 8.6.5 | Intermediate noise can improve correction when recoverable signal exists |
| Conjecture 7 | Adaptive resonance optimality | 8.6.8 | There may be an optimal vigilance $\rho^*$ that balances false rejections and false acceptances |
| Conjecture 8 | Model-specific sampling limit | 11.6 | Nyquist-style analogy: constraint sampling rate $s > 2B_{M,T}$ may be required |

---

## Propositions

| # | Name | Section | Statement |
|---|------|---------|-----------|
| 1 | Compression as understanding | 2.1 | Useful compression can indicate abstraction when it preserves task distinctions |
| 2 | Hallucination as entropy maximization | 3.3 | Under weak content constraints, form-valid output remains available |
| 3 | Confidence-accuracy decoupling | 3.3 | On OOD topics, confidence can track fluency more than accuracy |
| 4 | Ambiguity-induced hallucination | 4.4.4 | Multiple similar-activation representations can produce composite output |
| 5 | Context crowding | 4.5.4 | Insufficient decompression room can produce Kolmogorov garbage |
| 6 | Decompression-compression asymmetry | 4.5.4 | $K(\text{reconstruct}) \gg K(\text{store})$ |
| 7 | Sink-limited capacity | 4.6.2 | $\partial C_{ctx}/\partial s \leq 0$; sinks reduce capacity |
| 8 | Test-time learning as capacity extension | 4.7.6 | $C_T^{effective} = C_T^{static} + \Delta C_T(ctx)$ |
| 9 | Compression-transmission trade-off | 5.1 | Compression efficiency and transmission reliability can trade off |
| 10 | Detectability | 6.2 | Conservation violations are detectable via complexity comparison |
| 11 | Manifold departure | 8.4.3 | Distortion = on-manifold (recoverable) + off-manifold (hallucination) |
| N/A | Hallucination as teaching failure | 2.2.2 | Failure during matching, decompression, or transmission |

---

## Predictions

| # | Name | Section | Testable Claim |
|---|------|---------|----------------|
| 1 | Frequency-accuracy correlation | 9.1 | Higher training frequency should reduce hallucination rate |
| 2 | Few-shot logarithmic improvement | 9.1 | $P(\text{hall}) \propto 1/\log(1+k)$ with $k$ examples |
| 3 | Confidence-grounding decoupling | 9.1 | Corr(confidence, fluency) > Corr(confidence, accuracy) on OOD |
| 4 | Prompt specificity effect | 9.1 | $P(\text{hall}) \approx 1 - \exp(-d_{\mathcal{M}}^2/2\sigma^2)$ |
| 5 | Context crowding effect | 9.1 | Hallucination risk should rise as decompression room vanishes |
| 6 | Decompression asymmetry | 9.1 | Complex topics need disproportionately more context room |
| 7 | Information conservation violation | 9.1 | $K(\text{output}) > K(\text{source})$ is an unsupported-output signal |
| 8 | Excess information source | 9.1 | Excess content should trace to another source or to unsupported completion |
| 9 | Geometric distortion accumulation | 9.1 | Error compounds multiplicatively through pipeline |
| 10 | First-stage dominance | 9.1 | Early errors can have outsized impact |
| 11 | Multi-hop degradation | 9.1 | Accuracy can degrade with reasoning chain length |
| 12 | Temperature-hallucination relationship | 9.1 | The curve should be U-shaped in some weak-signal regimes |
| 13 | Entropy ratio prediction | 9.1 | $P(\text{hall}) \propto \exp(S_{form} - S_{knowledge})$ as a model target |
| 14 | Free energy minimization | 9.1 | Generation can be modeled with $F = E - TS$ |
| 15 | Optimal noise existence | 9.1 | There may exist $\sigma^*$ where noise helps retrieval |
| 16 | Stochastic resonance | 9.1 | Weak recoverable memories may improve with intermediate noise |
| 17 | Self-consistency benefit | 9.1 | Multiple samples + voting can improve accuracy under independent-error assumptions |
| 18 | Balanced context window | 9.1 | U-shaped error: too little context = no constraints, too much = crowding |
| 19 | Geometry-aligned warm start | 9.1 | Geometry-aligned initialization should improve training under the proxy |
| 20 | Geometry-driven training diagnostics | 9.1 | Manifold distance should predict validation accuracy |
| 21 | Position primacy | 4.6 | Late-context evidence degrades with sink severity |
| 22 | Atom coverage | 4.7 | Fewer activated atoms should correlate with higher hallucination rate |
| 23 | Adaptive resonance peak | 8.6.8 | Weak knowledge can benefit from joint $(\sigma, \rho)$ tuning |
| 24 | Knowledge-contingent optimum | 8.6.8 | Candidate optimal $(\sigma^*, \rho^*)$ varies with topic capacity |
| 25 | Test-time learning reduces hallucination | 11.7 | Titans-style architectures should reduce hallucination on partial-coverage topics |
| 26 | Memory hierarchy advantage | 11.7 | Multi-tier memory should outperform monolithic architectures under partial-coverage conditions |

---

## Key concepts and intuitions

### Intuition blocks
| # | Name | Section | Analogy |
|---|------|---------|---------|
| 1 | Bayesian prior | 1.6 | Car colors: default expectations before evidence |
| 2 | Thermal bath | 8.5.0 | Ice melting: content constraints vs. form prior |
| 3 | Compression as understanding | 2.1.0 | Sequence prediction: memorizing vs. learning the rule |
| 4 | Teacher's dilemma | 2.2.1 | Teaching a topic without enough source signal |
| 5 | Confabulation mechanism | 3.2.1 | Form pressure when content constraints are weak |
| 6 | Effective query | 4.4.0 | Lens and light: context filters the prompt |
| 7 | Library paradox | 8.3.0 | Output needs enough source support |
| 8 | Telephone game | 8.4.0 | Geometric distortion accumulation |
| 9 | Stuck lock | 8.6.0 | Stochastic resonance: noise jiggles the key |
| 10 | Shared objects and shared geometry | 11.5.0 | Universal manifold as a bounded geometry hypothesis |
| 11 | Open-book exam | 11.7.0 | Test-time learning: writing notes during the test |

---

## Hallucination mechanisms

### The six mechanisms

| # | Mechanism | Section | Cause | Result |
|---|-----------|---------|-------|--------|
| 1 | **Capacity violation** | 3 | $R_T > C_T$; asking beyond source support | Nothing reliable to retrieve |
| 2 | **Matching failure** | 4.4 | Ambiguous effective query | Composite or wrong retrieval |
| 3 | **Decompression Failure** | 4.5 | Insufficient context room | Kolmogorov garbage |
| 4 | **Geometric Distortion** | 8.4 | Multiplicative error cascade | Accumulated corruption |
| 5 | **Maximum-entropy prior relaxation** | 8.5 | Weak content constraints | Fluent but weakly supported output |
| 6 | **Noise paradox** | 8.6 | Too much or too little stochasticity | Brittle or unsupported output |

---

## Section map

| Section | Title | Key Content |
|---------|-------|-------------|
| **1** | Introduction | Source accounting, Bayesian prior |
| **2** | Theoretical framework | Compression, verifiable representations, teaching |
| **3** | Hallucinations as capacity violations | Confabulation mechanism |
| **4** | Reconstruction failures | Effective query, context pressure, atoms |
| **5** | Compression-transmission duality | LLMs as teachers |
| **6** | Hallucination taxonomy | Severity and detection |
| **7** | Mitigation strategies | Capacity estimation, verification, regime routing |
| **8** | Complexity from constraints | Conservation, distortion, maximum-entropy relaxation, noise |
| **9** | Experimental predictions | 26 testable predictions |
| **10** | Related work | Prior work and boundaries |
| **11** | Conclusion and extensions | Limitations, sampling conjecture, memory hierarchy |
| **App A** | Duality Table | Quick reference mapping |
| **App B** | Hallucination as constraint absence | Visual summary |

---

## Citation

If you use this repository in your research, please cite it. This is ongoing work, and we would like to know your opinions and experiments. Thank you.

Oscar Goldman - Shogu Research Group @ Datamutant.ai (subsidiary of 温心重工業)

Goldman, O. (2025). *Hallucinations in Noisy Channels: An information-theoretic framework for LLM hallucination errors* (Version 1.2.2). Shogu Research Group @ Datamutant.ai. https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels

---

*This index is aligned with Hallucinations_in_Noisy_Channels_v1.2.2.md.*
