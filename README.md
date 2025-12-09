# Hallucinations in Noisy Channels

**Information-Theoretic and Thermodynamic Informed Framework for Understanding LLM Hallucination Errors**


[![Status: Theoretical Framework](https://img.shields.io/badge/Status-Theoretical%20Framework-blue.svg)]()


**Author:** Oscar Goldman - Shogu Research Group @ Datamutant.ai  
**Date:** November 2025

---

## Read the Paper

**→ [Hallucinations in Noisy Channels v1.2.1 (Working Document)](Hallucinations_in_Noisy_Channels_v1.2.1.md)**


This repo is work in progress, there is a lot to do here, the experiments are ongoing, some you can find in the AKIRA repo now made public, we estimate a years or more work until this is finalized, maybe longer, depends on experiments wins/failures, some statements might change, but the overall formalization is here.

---

## Overview

We model LLMs as **teachers, not just generators**. During inference, they must first **reconstruct** knowledge from compressed weights, then **transmit** it reliably.

The framework establishes a fundamental duality:
- **Training = Compression = Learning** (source coding)
- **Inference = Transmission = Teaching** (channel coding)

Hallucinations emerge when the teaching process fails: when the model cannot correctly reconstruct and transmit stored knowledge.

Currently identified process failures through six mechanisms:



| # | Mechanism | Description |
|---|-----------|-------------|
| 1 | **Capacity Violations** | Asking about topics never learned |
| 2 | **Matching Failures** | Ambiguous prompts activate wrong representations |
| 3 | **Decompression Failures** | Insufficient context to unfold compressed knowledge |
| 4 | **Geometric Distortion** | Errors compound multiplicatively through the pipeline |
| 5 | **Thermodynamic Equilibration** | System relaxes to maximum entropy (fluent noise) |
| 6 | **The Noise Paradox** | Too little noise prevents self-correction; too much causes hallucination |

### The Unifying Principle

> **Information cannot be created; it can only be transmitted or lost.**
>
> When output contains more information than was stored or provided, the excess was hallucinated from the **form prior**; the model knows *how* to write but not *what* is true.

---

## Key contributions

### Multi-mechanism framework

```
Training (Compression) → Matching → Reconstruction (Context) → Transmission (Teaching)
    ↓                         ↓                 ↓                         ↓
Capacity Violation       Matching Failure   Decompression Failure    Geometric Distortion
    └──────────────────────────────┬──────────────────────────────────────────────┘
                          Constraints fail → THERMALIZATION TO FORM PRIOR
                                           ↓
           P(hallucination) ∝ Ω_form / Ω_knowledge = exp(ΔS);  F = E − T·S
                                           ↓
                                     HALLUCINATION
```

### Seven theorems

| Theorem | Name | Key Result |
|---------|------|------------|
| **1** | Hallucination Threshold | R_T > C_T ⟹ hallucinations unavoidable (Shannon limit) |
| **2** | Geometric Matching | P(correct) ∝ exp(-d_M) / Σexp(-d_M) in universal manifold |
| **3** | Information Conservation | K(output) ≤ K(weights) + K(context); excess = hallucination |
| **4** | Geometric Distortion | Fidelity = ∏(1 - εᵢ); errors compound multiplicatively |
| **5** | Thermodynamic Hallucination | P(hallucination) ∝ exp(ΔS) = Ω_form / Ω_knowledge |
| **6** | Optimal Noise Principle | T* > 0 required for self-correction; greedy is suboptimal |
| **7** | Nyquist–Shannon Analogy | s > 2·B_{M,T} for reliable reconstruction *(conjecture)* |

### Twenty-one testable predictions

Empirically falsifiable hypotheses spanning:
- Capacity-accuracy correlations (Predictions 1-3)
- Prompt specificity and matching effects (Prediction 4)
- Context crowding curves (Predictions 5-6)
- Information conservation violations (Predictions 7-8)
- Geometric distortion accumulation (Predictions 9-11)
- Temperature-hallucination relationships (Predictions 12-14)
- Optimal noise existence (Predictions 15-17)
- Goldilocks context window (Prediction 18)
- Geometry-aligned training (Predictions 19-20)
- Attention sink effects (Prediction 21)

See [Section 9: Experimental predictions](Hallucinations_in_Noisy_Channels_v1.2.md#9-experimental-predictions) for full mathematical formulations.

---


---

## Core concepts

### The teaching framework

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  INFERENCE = RECONSTRUCTION + TRANSMISSION = TEACHING       │
│                                                              │
│  Query ──▶ MATCH to internal representation                 │
│        ──▶ RECONSTRUCT knowledge in context                 │
│        ──▶ TRANSMIT to output                               │
│                                                              │
│  Hallucination = Teaching Failure                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### The conservation law

```
Information cannot be created; it can only be transmitted or lost.

K(output | topic) ≤ K(source | topic)

When violated: Information was "created" from the form prior
                → Definitional hallucination
```

### The thermodynamic view

```
KNOWLEDGE        ←→        FORM PRIOR
Potential energy           Kinetic/thermal energy
Low entropy                High entropy
Few microstates            Many microstates
Constrained                Unconstrained
Grounded                   Hallucinated

Hallucination = Thermalization to maximum entropy
P(hallucination) ∝ exp(S_form - S_knowledge)
```

### The noise paradox

```
T → 0:   Frozen, deterministic, cannot self-correct
T = T*:  Goldilocks zone, explores and corrects
T → ∞:   Pure entropy, complete hallucination

Optimal noise T* > 0 is REQUIRED for error correction
```

### Geometric distortion cascade

```
Fidelity decays EXPONENTIALLY with chain length:

n=3 stages, ε=0.1:  (0.9)³  = 73% fidelity
n=5 stages, ε=0.1:  (0.9)⁵  = 59% fidelity
n=10 stages, ε=0.1: (0.9)¹⁰ = 35% fidelity
n=20 stages, ε=0.1: (0.9)²⁰ = 12% fidelity

This is why long reasoning chains and multi-hop retrieval degrade.
```

---


---

## Practical mitigation strategies

Principled techniques grounded in theory:

| Strategy | Addresses | Section |
|----------|-----------|---------|
| **Unambiguous prompts** | Matching failures | §4.4 |
| **Context budget management** | Decompression crowding | §4.5 |
| **Chain-of-thought** | Distribute reconstruction load | §4.2 |
| **Optimal temperature calibration** | Enable self-correction | §8.6 |
| **Information accounting** | Detect conservation violations | §8.3 |
| **First-stage quality** | Training > prompting (Friis analogy) | §8.4 |
| **Semantic anchors** | Counter attention sink drift | §4.6 |

---

We hypothesize that > "Hallucination is thermalization to the form prior bath. 
When knowledge constraints fail, the system equilibrates to maximum entropy: fluent text, empty content."
The temperature parameter in LLM sampling IS analogous to the Boltzmann temperature. 
The framework tries to reveal that hallucination control is fundamentally about managing the balance between:
Potential energy (stored knowledge, constraints)
Kinetic energy (form prior, entropy)
Temperature (exploration vs. exploitation)

## Experimental status

### Theory: Complete Working document (v1.2.1)
- [x] Six-mechanism framework formalized
- [x] Eleven intuition blocks added (Analogies + ASCII diagrams)
- [x] Seven theorems with proof sketches (Theorem 7 = conjecture)
- [x] Twenty-one testable predictions defined
- [x] Mitigation strategies derived


### Experiments: In Progress
- [ ] Prediction 1: Frequency-accuracy correlation (in progress)
- [ ] Prediction 4: Prompt specificity effect (in progress)
- [ ] Prediction 12: Temperature-hallucination relationship (in progress)
- [ ] Predictions 5-6: Context crowding effects
- [ ] Predictions 9-11: Geometric distortion accumulation (in progress)
- [ ] Predictions 15-17: Optimal noise existence

---



### Citation

If you use this repository in your research, please cite it. This is ongoing work; we would like to know your opinions and experiments. Thank you.

Oscar Goldman - Shogu Research Group @ Datamutant.ai (subsidiary of 温心重工業)

Goldman, O. (2025). *Hallucinations in Noisy Channels: An Information-Theoretic and Thermodynamic Informed Framework for Understanding LLM Hallucination Errors* (Version 1.2.1). Shogu Research Group @ Datamutant.ai. https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels

---

## License

### Paper (Theoretical Content)

The theoretical framework and documentation are licensed under [**Creative Commons Attribution 4.0 International (CC-BY-4.0)**](https://creativecommons.org/licenses/by/4.0/).

**You are free to:** Share, Adapt, use commercially  
**Required:** Attribution

### Code (Experiments & Scripts)

Code in `experiments/`, `scripts/`, and `THX/` is licensed under the [**MIT License**](LICENSE).

---


---

## Related work

This framework builds on:

- **Information Theory:** Shannon (1948), Kolmogorov (1965), Tishby (2000)
- **Statistical Mechanics:** Boltzmann (1877), Jaynes (1957), Hopfield (1982)
- **Representation Learning:** Huh et al. (2024), Jha et al. (2025)
- **Hallucination Studies:** Ji et al. (2023), Huang et al. (2023)

See [Section 10: Related Work](Hallucinations_in_Noisy_Channels_v1.2.md#10-related-work) for full citations.

---

## Contact

**Oscar Goldman**  
Shogu Research Group Datamutant.ai  
[GitHub](https://github.com/Gman-Superfly) · [Issues](https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels/issues) · [Discussions](https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels/discussions)

---

> **Hallucinations are not bugs; they are information-theoretic necessities when you transmit beyond capacity.**
>
> **When constraints fail, systems thermalize to maximum entropy: fluent form, empty content.**
>
> **Information cannot be created; it can only be transmitted or lost. The excess is hallucination.**

---

*Oscar Goldman - Shogu Research Group @ Datamutant.ai - November 2025*
