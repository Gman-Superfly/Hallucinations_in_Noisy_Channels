# Hallucinations in Noisy Channels

**Information-Theoretic and Thermodynamic Framework for Understanding LLM Hallucination Errors**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status: Theoretical Framework](https://img.shields.io/badge/Status-Theoretical%20Framework-blue.svg)]()
[![Version: 1.2](https://img.shields.io/badge/Version-1.2-green.svg)]()

**Author:** Oscar Goldman - Shogu Research Group @ Datamutant.ai  
**Date:** November 2025

---

## Read the Paper

**→ [Hallucinations in Noisy Channels v1.2 (Full Framework)](Hallucinations_in_Noisy_Channels_v1.2.md)**

---

## Overview

LLMs are **teachers, not just generators**. During inference, they must first **reconstruct** knowledge from compressed weights, then **transmit** it reliably. Hallucinations occur when this reconstruction-transmission process fails through six mechanisms:

| # | Mechanism | Description |
|---|-----------|-------------|
| 1 | **Capacity Violations** | Asking about topics never learned |
| 2 | **Matching Failures** | Ambiguous prompts activate wrong representations |
| 3 | **Decompression Failures** | Insufficient context to unfold compressed knowledge |
| 4 | **Geometric Distortion** | Errors compound multiplicatively through the pipeline |
| 5 | **Thermodynamic Equilibration** | System relaxes to maximum entropy (fluent noise) |
| 6 | **The Noise Paradox** | Too little noise prevents self-correction; too much causes hallucination |

### The Unifying Principle

> **Information cannot be created, only transmitted or lost.**
>
> When output contains more information than was stored or provided, the excess was hallucinated from the **form prior**—the model knows *how* to write but not *what* is true.

---

## Key Contributions

### Multi-Mechanism Framework

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

### Seven Theorems

| Theorem | Name | Key Result |
|---------|------|------------|
| **1** | Hallucination Threshold | R_T > C_T ⟹ hallucinations unavoidable (Shannon limit) |
| **2** | Geometric Matching | P(correct) ∝ exp(-d_M) / Σexp(-d_M) in universal manifold |
| **3** | Information Conservation | K(output) ≤ K(weights) + K(context); excess = hallucination |
| **4** | Geometric Distortion | Fidelity = ∏(1 - εᵢ) — errors compound multiplicatively |
| **5** | Thermodynamic Hallucination | P(hallucination) ∝ exp(ΔS) = Ω_form / Ω_knowledge |
| **6** | Optimal Noise Principle | T* > 0 required for self-correction; greedy is suboptimal |
| **7** | Nyquist–Shannon Analogy | s > 2·B_{M,T} for reliable reconstruction *(conjecture)* |

### Twenty-One Testable Predictions

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

See [Section 9: Experimental Predictions](Hallucinations_in_Noisy_Channels_v1.2.md#9-experimental-predictions) for full mathematical formulations.

---

## Repository Structure

```
Hallucinations_Noisy_Channels/
│
├── Hallucinations_in_Noisy_Channels_v1.2.md   # Main theoretical paper
├── README.md                                   # This file
├── LICENSE                                     # MIT (code) + CC-BY-4.0 (paper)
├── CITATION.cff                                # Citation metadata
│
├── experiments/                                # Empirical validation notebooks
│   ├── rope_accumulation.ipynb                 # RoPE drift experiments
│   ├── sampling_reconstruction.ipynb           # Nyquist comparison
│   ├── simpleLM_drift.ipynb                    # Latent drift analysis
│   ├── prompt_ablation_threshold.ipynb         # Context threshold tests
│   ├── cot_vs_direct_channel.ipynb             # CoT redundancy coding
│   ├── prompt_noise_tradeoff.ipynb             # Noise/signal tradeoffs
│   ├── semantic_redundancy_real_prompts.ipynb  # Real-world ρ estimation
│   ├── geometric_alignment_metrics.ipynb       # Manifold alignment
│   └── control_diagnostics.ipynb               # Controllability analysis
│
├── scripts/
│   ├── semantic_redundancy_metric.py           # Redundancy calculator
│   └── generate_figures.py                     # Figure generation
│
├── figures/                                    # Generated visualizations
│   ├── semantic_redundancy_heatmap.png
│   ├── nyquist_comparison.png
│   ├── rope_drift.png
│   └── ...
│
├── docs/                                       # Supplementary documentation
│   ├── GLOSSARY.md                             # Term definitions
│   ├── VALIDATION_CHECKLIST.md                 # Experiment checklist
│   └── ...                                     # Additional notes
│
├── working/                                    # Development files
│   ├── CORE_FRAMEWORK_SIGNAL_PROCESSING.md     # Signal processing perspective
│   ├── requirements.txt                        # Python dependencies
│   └── ...
│
└── THX/                                        # Test harness & experiments
```

---

## Core Concepts

### The Teaching Framework

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

### The Conservation Law

```
Information cannot be created—only transmitted or lost.

K(output | topic) ≤ K(source | topic)

When violated: Information was "created" from the form prior
                → Definitional hallucination
```

### The Thermodynamic View

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

### The Noise Paradox

```
T → 0:   Frozen, deterministic, cannot self-correct
T = T*:  Goldilocks zone, explores and corrects
T → ∞:   Pure entropy, complete hallucination

Optimal noise T* > 0 is REQUIRED for error correction
```

### Geometric Distortion Cascade

```
Fidelity decays EXPONENTIALLY with chain length:

n=3 stages, ε=0.1:  (0.9)³  = 73% fidelity
n=5 stages, ε=0.1:  (0.9)⁵  = 59% fidelity
n=10 stages, ε=0.1: (0.9)¹⁰ = 35% fidelity
n=20 stages, ε=0.1: (0.9)²⁰ = 12% fidelity

This is why long reasoning chains and multi-hop retrieval degrade.
```

---

## Quick Start

### Read the Theory

```bash
# Clone the repository
git clone https://github.com/Gman-Superfly/Hallucinations_Noisy_Channels.git
cd Hallucinations_Noisy_Channels

# Read the main paper (markdown)
cat Hallucinations_in_Noisy_Channels_v1.2.md
```

### Run Experiments

```bash
# Install dependencies
pip install -r working/requirements.txt

# Run semantic redundancy analysis
python scripts/semantic_redundancy_metric.py --backend tfidf

# Launch Jupyter for notebooks
jupyter notebook experiments/
```

---

## Practical Mitigation Strategies

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

## Experimental Status

### Theory: Complete (v1.2)
- [x] Six-mechanism framework formalized
- [x] Seven theorems with proof sketches (Theorem 7 = conjecture)
- [x] Twenty-one testable predictions defined
- [x] Mitigation strategies derived
- [x] Thermodynamic unification complete

### Experiments: In Progress
- [ ] Prediction 1: Frequency-accuracy correlation (in progress)
- [ ] Prediction 4: Prompt specificity effect (in progress)
- [ ] Prediction 12: Temperature-hallucination relationship (in progress)
- [ ] Predictions 5-6: Context crowding effects
- [ ] Predictions 9-11: Geometric distortion accumulation
- [ ] Predictions 15-17: Optimal noise existence

---

## Citation

If you use this framework in your research, please cite:

### BibTeX

```bibtex
@techreport{goldman2025hallucinations,
  title={Hallucinations in Noisy Channels: An Information-Theoretic and Thermodynamic Framework for Understanding LLM Hallucination Errors},
  author={Goldman, Oscar},
  institution={Shogu Research Group @ Datamutant.ai, subsidiary of 温心重工業},
  year={2025},
  month={November},
  version={1.2},
  url={https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels},
  license={CC-BY-4.0}
}
```

### APA

Goldman, O. (2025). *Hallucinations in Noisy Channels: An Information-Theoretic and Thermodynamic Framework for Understanding LLM Hallucination Errors* (Version 1.2). Shogu Research Group @ Datamutant.ai. https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels

---

## License

### Paper (Theoretical Content)

The theoretical framework and documentation are licensed under [**Creative Commons Attribution 4.0 International (CC-BY-4.0)**](https://creativecommons.org/licenses/by/4.0/).

**You are free to:** Share, Adapt, use commercially  
**Required:** Attribution

### Code (Experiments & Scripts)

Code in `experiments/`, `scripts/`, and `THX/` is licensed under the [**MIT License**](LICENSE).

---

## Contributing

We welcome:
- Theoretical critiques and proof improvements
- Experimental validation results
- Suggestions for additional predictions
- Clarity improvements to exposition

**How to contribute:**
- Open an [Issue](https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels/issues) for bugs or questions
- Start a [Discussion](https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels/discussions) for ideas
- Submit a Pull Request with experiments or fixes

---

## Related Work

This framework builds on:

- **Information Theory:** Shannon (1948), Kolmogorov (1965), Tishby (2000)
- **Statistical Mechanics:** Boltzmann (1877), Jaynes (1957), Hopfield (1982)
- **Representation Learning:** Huh et al. (2024), Jha et al. (2025)
- **Hallucination Studies:** Ji et al. (2023), Huang et al. (2023)

See [Section 10: Related Work](Hallucinations_in_Noisy_Channels_v1.2.md#10-related-work) for full citations.

---

## Contact

**Oscar Goldman**  
Shogu Research Group @ Datamutant.ai  
[GitHub](https://github.com/Gman-Superfly) · [Issues](https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels/issues) · [Discussions](https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels/discussions)

---

> **Hallucinations are not bugs—they are information-theoretic necessities when you transmit beyond capacity.**
>
> **When constraints fail, systems thermalize to maximum entropy: fluent form, empty content.**
>
> **Information cannot be created—only transmitted or lost. The excess is hallucination.**

---

*Oscar Goldman - Shogu Research Group @ Datamutant.ai - November 2025*
