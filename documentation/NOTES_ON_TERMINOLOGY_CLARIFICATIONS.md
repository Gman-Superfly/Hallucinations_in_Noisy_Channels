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

*Last updated: December 2025*

