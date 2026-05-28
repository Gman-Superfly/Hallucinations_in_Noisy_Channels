# Notes on terminology clarifications

This document records decisions about terminology in the Hallucinations in Noisy Channels framework, explaining why specific terms were chosen and how they relate to each other.

---

## 1. "Kolmogorov garbage" vs "Zipf/form prior"

### The question

Should "Kolmogorov garbage" be renamed to "Zipf garbage" given that Zipf distributions represent the empirical manifestation of compression in language?

### The distinction

| Term | What it describes | Type |
|------|-------------------|------|
| **Kolmogorov garbage** | The *output* from decompression failure: fragments that look plausible individually but do not cohere into a source-supported whole | Process failure output |
| **Zipf / Form prior** | The *distribution* you sample from when content constraints fail: the null model arising from combinatorics | Attractor state |

### Why "Kolmogorov garbage" is correct

**Kolmogorov garbage** describes a *process failure*:

```
Insufficient complexity in -> truncated reconstruction -> incoherent fragments out
```

This is Section 4.5 (Decompression Failure). The mechanism is:
1. Context is over-filled OR query lacks discriminating structure
2. Decompression room is insufficient: $K_{\text{available}} < K_{\text{reconstruct}}(r)$
3. Reconstruction is truncated mid-process
4. Output consists of structurally valid fragments that fail to cohere

The term "Kolmogorov" is appropriate here because:
- It references the complexity mismatch that causes the failure
- The garbage is the result of incomplete algorithmic reconstruction
- The fragments have valid local structure but invalid global coherence

### Why "Zipf garbage" doesn't work

**Zipf distribution** describes an *attractor state*:

```
Content constraints fail -> system thermalizes -> samples from form prior (Zipf distribution)
```

This is Section 8.5 (Thermodynamic Equilibration). The mechanism is:
1. Knowledge constraints are absent or fail
2. System relaxes toward high-entropy form-consistent output
3. Output samples from the Zipf distribution over tokens

"Zipf garbage" is incorrect because:
1. **Zipf describes a distribution** (power-law over tokens), not fragmented output
2. **The garbage is not Zipf-distributed**: it is the result of truncated reconstruction
3. **Zipf is the null model**: what you get from combinatorics alone, not a failure mode

### The conceptual relationship

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  Two distinct failure modes                                             │
│                                                                          │
│  1. Decompression failure -> Kolmogorov garbage                         │
│     Knowledge: stored                                                   │
│     Match: correct                                                      │
│     Problem: insufficient room to reconstruct                           │
│     Result: Fragments that individually look correct but don't cohere  │
│                                                                          │
│     Analogy: Trying to do long division in your head without paper     │
│              You have the method, but cannot complete the process       │
│                                                                          │
│  2. Thermalization -> form prior (Zipf) sampling                        │
│     Knowledge: absent or inaccessible                                  │
│     Match: failed or no target                                         │
│     Problem: no content constraints                                    │
│     Result: Fluent text sampled from maximum-entropy distribution      │
│                                                                          │
│     Analogy: Asked about something you don't know                      │
│              You speak fluently but say nothing grounded               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Summary

| Concept | Term | Mechanism | Section |
|---------|------|-----------|---------|
| Process failure (truncated reconstruction) | **Kolmogorov garbage** | Complexity mismatch | 4.5 |
| Attractor state (null model) | **Zipf / form prior** | Thermalization | 8.5 |

Both produce hallucinations, but through different mechanisms:
- Kolmogorov garbage: the source signal exists, but reconstruction cannot unfold it.
- Form prior sampling: the source signal is absent, inaccessible, or too weak to constrain output.

### Decision

**Keep "Kolmogorov garbage"** for decompression failures.
**Use "form prior" or "Zipf distribution"** for the thermalization attractor.

The terms serve different purposes and describe different phenomena in the framework.

---

## 2. Field alignment after Ouyang et al. (2026)

### The question

Ouyang et al. (2026) use Shannon-channel language for LLM scaling laws. Should HNC adopt their terms?

### The decision

HNC should use their terms when discussing their paper and keep their training-time variables separate from HNC's inference-time variables. The two frameworks share a channel vocabulary, yet they model different stages.

| Term | Use in Ouyang et al. (2026) | Use in HNC | Decision |
|------|-----------------------------|------------|----------|
| **Shannon Scaling Law** | Empirical law for training-time loss as a function of model size, token count, and noise | External term | Use only when citing Ouyang et al. |
| **$C_{\text{LLM}}(N,D)$** | Global fitted capacity from parameters $N$ and tokens $D$ | Different from topic capacity $C_T$ | Call it global LLM capacity or fitted scaling capacity |
| **Bandwidth** | Model size term, usually $aN^{\alpha}$ | Possible analogy for global model resources | Keep topic capacity and source capacity as the HNC terms |
| **Signal power** | Training-token signal, usually $bD^{\beta}$ | HNC source signal can come from weights, context, retrieval, tools, or adaptive memory | Use "source signal" in HNC unless discussing their law directly |
| **Noise** | Data-induced noise, model-interaction noise, irreducible noise, and perturbation noise | Noise can appear in training, matching, decompression, retrieval, context, and decoding | Use stage-specific noise labels where possible |
| **SNR** | Signal-to-noise ratio in the fitted scaling law | Effective signal-to-noise ratio of a source-support path | Adopt SNR as field-aligned shorthand, but define the signal and noise source each time |
| **Loss basin** | Closed low-loss region surrounded by degradation under perturbation | HNC predicts U-shaped hallucination or error curves in some regimes | Use "loss basin" only for loss surfaces or loss curves |
| **Training as channel modulation** | Training modulates information into weights | HNC usually says training compresses source structure into weights | Can be cited as compatible language, but keep "compression" for HNC |
| **Inference as transmission** | Input context $\mathcal{X}$ transmits to output $\mathcal{Y}$ | HNC says inference is reconstruction plus transmission, called teaching | Compatible, but HNC should keep teaching when source accounting matters |

### Why the variables must stay separate

Ouyang et al. define a global capacity proxy:

$$
C_{\text{LLM}}(N,D)
$$

Here, $N$ is model size and $D$ is training tokens. The quantity is fitted to loss curves across model sizes, token budgets, and perturbation settings.

HNC defines topic capacity:

$$
C_T \approx I(\text{Query}; \text{Accurate Answer} \mid T)
$$

Here, $T$ is the topic and the effective query includes prompt and context state. The quantity targets whether a model can transmit grounded information for a particular topic under a particular inference setup.

These quantities may relate, and they answer different measurement questions. A high $C_{\text{LLM}}(N,D)$ can still leave low $C_T$ for a rare topic, ambiguous prompt, crowded context, missing retrieval source, or unsupported tool-free request. Conversely, a smaller model can have enough $C_T$ for a narrow topic when the context supplies the missing source signal.

### Terms to adopt carefully

**Adopt SNR with explicit accounting.** HNC can say "effective SNR" when the document names the signal path and the noise source. For example, "retrieval SNR" can mean source-support signal from retrieved context divided by ambiguity, contradiction, or irrelevant-token pressure. "Decoding SNR" can mean content-token support relative to sampling noise and form-prior pressure.

**Adopt global LLM capacity for scaling discussions.** When discussing Ouyang et al., use $C_{\text{LLM}}(N,D)$ or "global LLM capacity." When discussing hallucination, use $C_T$, "topic capacity," or "source-supported topic capacity."

**Adopt loss basin only for loss.** HNC should use "U-shaped hallucination curve," "U-shaped error curve," or "balanced context curve" for inference experiments. Use "loss basin" only when the measured quantity is loss.

### Terms to avoid importing

Reserve "signal power" for Ouyang et al.'s scaling law. In their paper, signal power comes from training tokens. In HNC, the source signal can come from several modeled sources, including context and tools. Calling all of that "signal power" would blur the source-accounting principle.

Reserve $C_{\text{LLM}}$ for fitted global scaling capacity. $C_T$ is a topic-conditioned inference quantity.

Treat the Shannon Scaling Law as scaling-law evidence for capacity and SNR boundary conditions. It leaves source accounting, matching failure, decompression room, and thermalization as separate HNC claims requiring direct tests.

### Reference

Ouyang, X., Liu, D., Cai, Y., Liu, J., Yang, Y., Zheng, C., Hartvigsen, T., & Ma, Y. (2026). *LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws.* arXiv:2605.23901. https://arxiv.org/abs/2605.23901

---

*Last updated: May 2026*

