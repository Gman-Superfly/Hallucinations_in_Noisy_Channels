# Hallucinations in Noisy Channels

**An information-theoretic framework for LLM hallucination errors**


[![Status: Theoretical Framework](https://img.shields.io/badge/Status-Theoretical%20Framework-blue.svg)]()


**Author:** Oscar Goldman - Shogu Research Group @ Datamutant.ai  
**Date:** November 2025

---

## Read the paper

**Read [Hallucinations in Noisy Channels v1.2.2 (working document)](Hallucinations_in_Noisy_Channels_v1.2.2.md)**


This repository is work in progress. The experiments are ongoing, and some are now public in the AKIRA repository. We estimate at least a year of further work before the framework is stable, depending on experimental results. Some statements may change as evidence changes, and the current formalization records the working structure.

---

## Overview

We model LLMs as teachers during inference. They must reconstruct usable information from compressed weights, context, retrieval, tools, or adaptive memory, then transmit it through decoding.

The framework uses a compression-transmission duality:
- Training as compression and learning (source coding)
- Inference as reconstruction and teaching (channel coding)

Hallucinations can arise when this teaching process fails: when the model lacks source signal, selects the wrong representation, lacks decompression room, or transmits through a noisy path.

The current paper separates six process failures:



| # | Mechanism | Description |
|---|-----------|-------------|
| 1 | Capacity violations | The request exceeds the available source signal. |
| 2 | Matching failures | The effective query selects the wrong or composite representation. |
| 3 | Decompression failures | The model lacks room to unfold compressed information. |
| 4 | Geometric distortion | Small errors compound through sequential transformations. |
| 5 | Maximum-entropy prior relaxation | Weak content constraints allow prior-dominated fluent text. |
| 6 | Noise paradox | Some stochasticity can help correction, while too much can destroy signal. |

### Source-accounting principle

> Grounded information must come from a modeled source; decoding alone should not be treated as a source of topic information.
>
> When output contains more topic information than the modeled sources can explain, the excess is evidence that learned priors, untracked signal, or unsupported completion filled the gap.

---

## Key contributions

### Multi-mechanism framework

```
Training (compression) -> Matching -> Reconstruction (context) -> Transmission (teaching)
    ↓                         ↓                 ↓                         ↓
Capacity Violation       Matching Failure   Decompression Failure    Geometric Distortion
    └──────────────────────────────┬──────────────────────────────────────────────┘
                          Weak constraints -> prior relaxation risk
                                           ↓
           P(hallucination) ∝ Ω_form / Ω_knowledge = exp(ΔS);  F = E - T·S
                                           ↓
                                  unsupported output risk
```

### Core formal claims

| Type | Name | Key Result |
|------|------|------------|
| Corollary 1 | Hallucination threshold | $R_T > C_T$ implies reliable source-supported generation exceeds the modeled channel limit. |
| Model 2 | Geometric matching proxy | Retrieval accuracy is modeled by a softmax over candidate representation distances. |
| Theorem 3 | Information conservation | Grounded output should be traceable to weights, context, retrieval, tools, or adaptive memory. |
| Theorem 4 | Geometric distortion | Fidelity decays as $\prod_i(1 - \epsilon_i)$ in the modeled cascade. |
| Conjecture 3 | Regime-aligned generation | A router should select a regime whose source signal and verifier support the requested answer rate. |
| Conjecture 5 | Maximum-entropy hallucination model | Prior relaxation risk rises under weak content constraints. |
| Conjecture 6 | Optimal noise principle | Intermediate noise can improve correction when recoverable signal exists. |
| Conjecture 7 | Adaptive resonance optimality | Match threshold and noise should vary with knowledge certainty. |
| Conjecture 8 | Model-specific sampling limit | $s > 2B_{M,T}$ is a Nyquist-style reconstruction analogy. |

### Twenty-six testable predictions

Empirically falsifiable hypotheses spanning:
- Capacity-accuracy correlations (Predictions 1-3)
- Prompt specificity and matching effects (Prediction 4)
- Context crowding curves (Predictions 5-6)
- Information conservation violations (Predictions 7-8)
- Geometric distortion accumulation (Predictions 9-11)
- Temperature-hallucination relationships (Predictions 12-14)
- Optimal noise existence (Predictions 15-17)
- Balanced context window (Prediction 18)
- Geometry-aligned training (Predictions 19-20)
- Attention sink effects (Prediction 21)
- Atom coverage and adaptive resonance effects (Predictions 22-24)
- Test-time learning and memory hierarchy effects (Predictions 25-26)

See [Section 9: Experimental predictions](Hallucinations_in_Noisy_Channels_v1.2.2.md#9-experimental-predictions) for full mathematical formulations.

---


---

## Core concepts

### The teaching framework

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Inference = reconstruction + transmission = teaching       │
│                                                              │
│  Query ──▶ MATCH to internal representation                 │
│        ──▶ RECONSTRUCT knowledge in context                 │
│        ──▶ TRANSMIT to output                               │
│                                                              │
│  Hallucination risk rises when teaching fails               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### The conservation law

```
Grounded information must come from modeled source signal.

K(output | topic) ≤ K(source | topic)

When violated: unsupported completion or untracked source signal
                may have filled the gap.
```

### The maximum-entropy view

```
Knowledge        <->       Form prior
Potential energy           Kinetic/thermal energy
Low entropy                High entropy
Few microstates            Many microstates
Constrained                Unconstrained
Grounded                   Hallucinated

Hallucination risk rises when generation relaxes toward form prior
P(hallucination) ∝ exp(S_form - S_knowledge)
```

### The noise paradox

```
T -> 0:  Frozen, deterministic, cannot self-correct
T = T*:  Intermediate region, explores while preserving signal
T -> ∞:  High noise, signal loss and prior relaxation risk

Intermediate noise can support error correction when recoverable signal exists
```

### Geometric distortion cascade

```
Fidelity decays EXPONENTIALLY with chain length:

n=3 stages, ε=0.1:  (0.9)³  = 73% fidelity
n=5 stages, ε=0.1:  (0.9)⁵  = 59% fidelity
n=10 stages, ε=0.1: (0.9)¹⁰ = 35% fidelity
n=20 stages, ε=0.1: (0.9)²⁰ = 12% fidelity

Long reasoning chains and multi-hop retrieval can degrade when stage errors are correlated or compounding.
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
| **Temperature calibration** | Balance exploration against signal preservation | §8.6 |
| **Information accounting** | Detect conservation violations | §8.3 |
| **First-stage quality** | Training > prompting (Friis analogy) | §8.4 |
| **Semantic anchors** | Counter attention sink drift | §4.6 |

---

> **Working hypothesis:** hallucination can be modeled, in part, as relaxation toward the form prior.
>
> When knowledge constraints fail, generation can relax toward high-entropy fluent text with weak or unsupported content.

The temperature parameter in LLM sampling is an algorithmic analogue of thermodynamic temperature. In this framing, hallucination control means managing stored or supplied source signal, form-prior pressure, and exploration noise.

## Experimental status

### Theory: complete working document (v1.2.2)
- [x] Six-mechanism framework formalized
- [x] Eleven intuition blocks added (Analogies + ASCII diagrams)
- [x] Formal claims relabeled by evidence status: corollaries, models, and conjectures
- [x] Twenty-six testable predictions defined
- [x] Mitigation strategies derived


### Experiments: in progress
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

Goldman, O. (2025). *Hallucinations in Noisy Channels: An information-theoretic framework for LLM hallucination errors* (Version 1.2.2). Shogu Research Group @ Datamutant.ai. https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels

---

## License

### Paper (theoretical content)

The theoretical framework and documentation are licensed under [**Creative Commons Attribution 4.0 International (CC-BY-4.0)**](https://creativecommons.org/licenses/by/4.0/).

**You are free to:** Share, Adapt, use commercially  
**Required:** Attribution

### Code (experiments and scripts)

Code in `experiments/`, `scripts/`, and `THX/` is licensed under the [**MIT License**](LICENSE).

---


---

## Related work

This framework builds on:

- **Information Theory:** Shannon (1948), Kolmogorov (1965), Tishby (2000)
- **Statistical Mechanics:** Boltzmann (1877), Jaynes (1957), Hopfield (1982)
- **Representation Learning:** Huh et al. (2024), Jha et al. (2025)
- **Hallucination Studies:** Ji et al. (2023), Huang et al. (2023)

See [Section 10: Related Work](Hallucinations_in_Noisy_Channels_v1.2.2.md#10-related-work) for full citations.

---

## Contact

**Oscar Goldman**  
Shogu Research Group Datamutant.ai  
[GitHub](https://github.com/Gman-Superfly) · [Issues](https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels/issues) · [Discussions](https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels/discussions)

---

> Hallucination risk rises when generation exceeds available topic capacity under the channel model.
>
> When content constraints fail, generation can relax toward high-entropy form-prior text.
>
> Grounded information should trace to modeled source signal. Unsupported excess is a hallucination candidate.

---

*Oscar Goldman - Shogu Research Group @ Datamutant.ai - May 2026*
