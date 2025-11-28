# Hallucinations in Noisy Channels

**An Information-Theoretic and Thermodynamic Framework for Understanding LLM Hallucination Errors**

**Author:** Oscar Goldman - Shogu Research Group @ Datamutant.ai  
**Version:** 1.1  
**Date:** November 2025  
**Status:** Theoretical Framework (Empirical Validation In Progress)

---

## Read the Paper

**Main Paper:** [Hallucinations in Noisy Channels v1.1 (Full Framework)](Hallucinations_in_Noisy_Channels_v1.1.md)

**Supplementary Documents:**
- [Core Framework: Signal Processing Perspective](CORE_FRAMEWORK_SIGNAL_PROCESSING.md)
- [Hallucination and Noisy Channels: Extended Analysis](HALLUCINATION_AND_NOISY_CHANNELS.md)

---

## Overview

LLMs are **teachers, not just generators**. During inference, they must first **reconstruct** knowledge from compressed weights, then **transmit** it reliably. Hallucinations occur when this reconstruction-transmission process fails through six mechanisms:

1. **Capacity Violations** – Asking about topics never learned
2. **Matching Failures** – Ambiguous prompts activate wrong representations
3. **Decompression Failures** – Insufficient context to unfold compressed knowledge
4. **Geometric Distortion** – Errors compound multiplicatively through the pipeline
5. **Thermodynamic Equilibration** – System relaxes to maximum entropy (fluent noise)
6. **The Noise Paradox** – Too little noise prevents self-correction; too much causes hallucination

**The Unifying Principle:** Information cannot be created, only transmitted or lost. When output contains more information than was stored, the excess was hallucinated from the **form prior**—the model knows *how* to write but not *what* is true.

---

## Key Contributions

### 1. Unified Multi-Mechanism Framework

Integrates six distinct failure modes under a single information-theoretic and thermodynamic foundation:

```
Training (Compression) → Matching → Decompression → Transmission (Teaching)
    ↓                      ↓            ↓               ↓
Capacity               Retrieval    Context         Distortion
Violation             Failure      Crowding        Accumulation
    ↓                      ↓            ↓               ↓
                 THERMALIZATION TO FORM PRIOR
                          ↓
                   HALLUCINATION
```

### 2. Novel Theoretical Contributions

**Theorem 3 (Information Conservation):** Output cannot contain more information about a topic than was stored. When K(output) > K(stored), the excess is definitionally hallucinated.

**Theorem 4 (Geometric Distortion Accumulation):** Errors compound multiplicatively: Fidelity = ∏(1 - εᵢ), not additively. Each pipeline stage degrades truth exponentially.

**Theorem 5 (Thermodynamic Hallucination):** Hallucination probability scales exponentially with entropy gap: P(hallucination) ∝ exp(ΔS) = Ωform / Ωknowledge.

**Theorem 6 (Optimal Noise Principle):** Systems require optimal noise T* > 0 for self-correction. Greedy decoding (T=0) is suboptimal—cannot escape bad attractors.

### 3. Twenty Testable Predictions

Empirically falsifiable hypotheses spanning:
- Capacity-accuracy correlations
- Prompt specificity effects
- Context crowding curves
- Temperature-hallucination relationships
- Multi-hop reasoning degradation
- Optimal noise existence
- Geometric alignment diagnostics

See [Section 9: Experimental Predictions](paper/Hallucinations_in_Noisy_Channels_v1.1.md#9-experimental-predictions) for full list.

### 4. Practical Mitigation Strategies

Principled techniques grounded in theory:
- Unambiguous prompts (reduce matching failures)
- Context budget management (avoid decompression crowding)
- Chain-of-thought (distribute reconstruction load)
- Optimal temperature calibration (enable self-correction)
- Information accounting (detect conservation violations)
- First-stage quality prioritization (training > prompting)

---

## Repository Structure

```
hallucinations-noisy-channels/
│
├── paper/
│   ├── Hallucinations_in_Noisy_Channels_v1.1.md    # Main theoretical paper
│   ├── Hallucinations_in_Noisy_Channels_v1.1.pdf   # PDF version
│   └── draft.md                                     # Submission template
│
├── experiments/                                     # Empirical validation (In Progress)
│   ├── rope_accumulation.ipynb                      # RoPE drift experiments
│   ├── sampling_reconstruction.ipynb                # Nyquist comparison
│   ├── simpleLM_drift.ipynb                         # Latent drift analysis
│   ├── prompt_ablation_threshold.ipynb              # Context threshold tests
│   ├── cot_vs_direct_channel.ipynb                  # CoT redundancy coding
│   ├── prompt_noise_tradeoff.ipynb                  # Semantic redundancy
│   ├── semantic_redundancy_real_prompts.ipynb       # Real-world ρ estimation
│   ├── geometric_alignment_metrics.ipynb            # Manifold alignment
│   └── control_diagnostics.ipynb                    # Controllability analysis
│
├── scripts/
│   ├── semantic_redundancy_metric.py                # Redundancy calculator
│   └── ...                                          # Additional utilities
│
├── docs/
│   ├── GLOSSARY.md                                  # Term definitions
│   ├── VALIDATION_CHECKLIST.md                      # Experiment checklist
│   └── ...                                          # Additional documentation
│
├── figures/                                         # Generated visualizations
│   ├── semantic_redundancy_heatmap.png
│   ├── control_observability_scatter.png
│   └── ...
│
├── THX/                                             # Test harness & experiments
│
├── CORE_FRAMEWORK_SIGNAL_PROCESSING.md              # Signal processing perspective
├── HALLUCINATION_AND_NOISY_CHANNELS.md              # Extended analysis
├── CITATION.cff                                     # Citation metadata
├── CHANGELOG.md                                     # Version history
├── LICENSE                                          # MIT (code)
└── README.md                                        # This file
```

---

## Experimental Status

### Theory: Complete (v1.1)
- Six-mechanism framework formalized
- Four main theorems with proof sketches
- Twenty testable predictions defined
- Mitigation strategies derived

### Experiments: In Progress

**Validated Predictions:**
- Prediction 1: Frequency-accuracy correlation (in progress)
- Prediction 4: Prompt specificity effect (in progress)
- Prediction 12: Temperature-hallucination relationship (in progress)

**Upcoming:**
- Predictions 5-6: Context crowding effects
- Predictions 9-11: Geometric distortion accumulation
- Predictions 15-17: Optimal noise existence

**Notebooks Currently Functional:**
1. `rope_accumulation.ipynb` – RoPE trajectory analysis
2. `sampling_reconstruction.ipynb` – Nyquist baseline
3. `simpleLM_drift.ipynb` – Latent drift t-tests
4. Others under active development

---

## Quick Start

### Read the Theory

```bash
# Clone the repository
git clone https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels.git
cd Hallucinations_in_Noisy_Channels

# Read the main paper
cat paper/Hallucinations_in_Noisy_Channels_v1.1.md
# or open the PDF in paper/
```

### Run Experiments

```bash
# Install dependencies
pip install -r requirements.txt

# Run semantic redundancy analysis
python scripts/semantic_redundancy_metric.py --backend tfidf

# Launch Jupyter for notebooks
jupyter notebook experiments/
```

### Generate Figures

```bash
# Run all experiments and generate figures
make figures

# Individual experiments
jupyter nbconvert --execute experiments/rope_accumulation.ipynb
```

---

## Key Concepts

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

K(output | topic) ≤ K(stored | topic)

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
P(hallucination) ∝ exp(Sform - Sknowledge)
```

### The Noise Paradox

```
T → 0:   Frozen, deterministic, cannot self-correct
T = T*:  Goldilocks zone, explores and corrects
T → ∞:   Pure entropy, complete hallucination

Optimal noise T* > 0 is REQUIRED for error correction
```

---

## Testable Predictions (Sample)

**Prediction 1:** Hallucination rate inversely correlates with topic frequency in training.

**Prediction 4:** Hallucination rate decreases with prompt specificity: P(hallucination) ∝ 1/|K(prompt) - K(representation)|

**Prediction 7:** Conservation violations are detectable: When K(output) > K(stored), hallucination occurred.

**Prediction 12:** Hallucination rate follows Boltzmann statistics with temperature: P(hallucination|T) ∝ exp(ΔS/k) · f(T)

**Prediction 15:** Optimal temperature exists: T* = argmax[P(correction) - P(hallucination)] > 0

[See full list of 20 predictions in the paper](paper/Hallucinations_in_Noisy_Channels_v1.1.md#91-testable-hypotheses)

---

## Feedback and Contributions

We welcome:
- Theoretical critiques and proof improvements
- Suggestions for experimental designs
- Validation results on predictions
- Related work citations
- Clarity improvements to exposition

**How to contribute:**
- Open an [Issue](https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels/issues) for bugs or questions
- Start a [Discussion](https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels/discussions) for ideas
- Submit a Pull Request with experiments or fixes

### Collaboration Opportunities

Interested in:
- Running experiments on the 20 predictions
- Extending the framework to specific domains
- Building practical hallucination detectors
- Validating on large-scale models

Reach out via Issues or Discussions.

---

## Citation

If you use this framework in your research, please cite:

### BibTeX

```bibtex
@techreport{goldman2025hallucinations,
  title={Hallucinations in Noisy Channels: A Unified Information-Theoretic 
         and Thermodynamic Framework for Understanding LLM Hallucination Errors},
  author={Goldman, Oscar},
  institution={Shogu Research Group @ Datamutant.ai, subsidiary of 温心重工業},
  year={2025},
  month={November},
  version={1.1},
  note={Available at \url{https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels}},
  license={CC-BY-4.0}
}
```

### APA

Goldman, O. (2025). *Hallucinations in Noisy Channels: A Unified Information-Theoretic and Thermodynamic Framework for Understanding LLM Hallucination Errors* (Version 1.1). Shogu Research Group @ Datamutant.ai. https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels

### Chicago

Goldman, Oscar. "Hallucinations in Noisy Channels: A Unified Information-Theoretic and Thermodynamic Framework for Understanding LLM Hallucination Errors." Shogu Research Group @ Datamutant.ai, November 2025. https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels.

---

## License

### Paper (Theoretical Content)

The theoretical framework, paper, and documentation in `paper/`, `docs/`, and markdown files are licensed under [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).

**You are free to:**
- Share — copy and redistribute the material
- Adapt — remix, transform, and build upon
- Commercial use is permitted

**Under these terms:**
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

### Code (Experiments & Scripts)

Code in `experiments/`, `scripts/`, and `THX/` is licensed under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2025 Oscar Goldman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Roadmap

### Phase 1: Theory (Complete)
- [x] Six-mechanism framework
- [x] Four main theorems
- [x] Twenty testable predictions
- [x] Mitigation strategies

### Phase 2: Validation (In Progress)
- [x] Experimental harness setup
- [x] Baseline notebooks (Nyquist, RoPE)
- [ ] Validate Predictions 1-5
- [ ] Validate Predictions 12, 15-17
- [ ] Full geometric alignment metrics

### Phase 3: Publication (Upcoming)
- [ ] Complete empirical validation
- [ ] Submit to arXiv (cs.AI or cs.LG)
- [ ] Workshop paper (ICML Theory, NeurIPS Workshop)
- [ ] Journal submission (JMLR, Entropy)

### Phase 4: Ecosystem (Future)
- [ ] Hallucination detection toolkit
- [ ] Prompt optimization library
- [ ] Integration with RAG systems
- [ ] Real-time monitoring dashboards

---

## Related Work

This framework builds on and extends:

- **Information Theory:** Shannon (1948), Kolmogorov (1965), Tishby (2000)
- **Statistical Mechanics:** Boltzmann (1877), Jaynes (1957), Hopfield (1982)
- **Hallucination Studies:** Ji et al. (2023), Huang et al. (2023), Manakul et al. (2023)
- **In-Context Learning:** Xie et al. (2022), Akyürek et al. (2023), Olsson et al. (2022)

See [Section 10: Related Work](paper/Hallucinations_in_Noisy_Channels_v1.1.md#10-related-work) for comprehensive citations.

---

## Acknowledgments

This work is part of a broader research program on information structures for reliable LLMs at the Shogu Research Group @ Datamutant.ai (subsidiary of 温心重工業).

Special thanks to the open-source community for tools that made this research possible: Jupyter, NumPy, PyTorch, Transformers, and countless others.

---

## Contact

**Oscar Goldman**  
Shogu Research Group @ Datamutant.ai  
[GitHub Profile](https://github.com/Gman-Superfly)

For questions, collaborations, or feedback:
- [Issues](https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels/issues)
- [Discussions](https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels/discussions)

---

**Hallucinations are not bugs—they are information-theoretic necessities when you transmit beyond capacity.**

**When constraints fail, systems thermalize to maximum entropy: fluent form, empty content.**

**Information cannot be created—only transmitted or lost. The excess is hallucination.**

---

*Oscar Goldman - Shogu Research Group @ Datamutant.ai - November 2025*

