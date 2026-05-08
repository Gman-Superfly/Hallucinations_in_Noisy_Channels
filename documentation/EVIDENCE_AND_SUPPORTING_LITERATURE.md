# Evidence and supporting literature

This document catalogs empirical evidence and independent research that supports the Hallucinations in Noisy Channels framework. Each finding links to specific theoretical claims.

---

## 1. From theory to observation: compression in language

### 1.1 The theoretical foundation

The framework begins with a simple insight from algorithmic information theory: **learning is compression**. The shortest program that generates data captures its essential structure. This is the Kolmogorov complexity $K(x)$, a theoretical limit that tells us compression is possible, but is itself uncomputable.

We use this foundation sparingly: it establishes that understanding = finding the rule that generates the data.

### 1.2 The empirical manifestation: Zipf's law

What does compression *look like* in practice? This is where Zipf distributions become central.

Berman (2025a, 2025b) proves that Zipf's law, the power-law rank-frequency distribution observed in all natural languages, arises from **pure combinatorics**:

$$
p(r) \propto r^{-\alpha}, \quad \alpha \approx 1.1 - 1.5
$$

**Key point:** Zipf distributions emerge from the interaction of two exponentials:
1. **Exponential growth** of possible word types with length
2. **Exponential decay** of probability for each word type with length

This yields a power-law without any semantic content, optimization, or linguistic organization.

### 1.3 The connection: form prior = Zipf distribution

We identify the **form prior** in this framework with the Zipf distribution:

| Concept | Definition | Evidence |
|---------|------------|----------|
| **Kolmogorov** | Theoretical: shortest program that generates data | Foundation (uncomputable) |
| **Zipf** | Empirical: power-law distribution over tokens | Berman (2025): arises from combinatorics |
| **Form prior** | The distribution LLMs thermalize to when content fails | = Zipf distribution |

When content constraints are absent, the model samples from the maximum-entropy distribution consistent with linguistic form. In this framework, that distribution corresponds to the Zipf distribution: the statistical signature of symbolic combinatorics.

**Implication:** The form prior does not require semantics. It is the **null model** (what you get from structure alone). Content knowledge distinguishes truthful generation from this baseline.

### 1.4 Why this matters

This connection resolves a key question: **What exactly is the "form prior" that models thermalize to?**

Answer: It is the Zipf distribution over tokens/words that arises from combinatorial structure, independent of meaning. Berman proves this mathematically and validates it empirically across English, Russian, and mixed corpora.

The form is "free" (arises from combinatorics). The content is what costs information. Hallucination = generating from form without content.

---

## 2. Summary of evidence status

| Core Claim | Status | Primary Source |
|------------|--------|----------------|
| LMs preserve information (injective) | **Proven** | Nikolaou et al. (2025) |
| Organization requires training | **Proven** | Teoh et al. (2025) - NextLat |
| Universal manifold exists | **Strong support** | Jha et al. (2025), Huh et al. (2024) |
| Form prior has mathematical support | **Proven for the cited model** | Berman (2025a, 2025b) |
| Verification-First improves accuracy | **Empirical** | Wu & Yao (2025) |
| Optimal noise exists ($T^* > 0$) | **Theoretical and empirical** | Gammaitoni et al. (1998), Wu & Yao (2025) |
| Test-time learning extends capacity | **Architectural** | Behrouz et al. (2025) - Titans |

---

## 3. Information preservation vs organization

### 3.1 LMs are provably injective

**Source:** Nikolaou, G., et al. (2025). *Language Models are Injective and Hence Invertible.* [arXiv:2510.15511](https://arxiv.org/abs/2510.15511)

**Key Findings:**
- Mathematical proof that transformer LMs mapping discrete sequences to continuous representations are injective
- Empirical confirmation: billions of collision tests across six state-of-the-art models → zero collisions
- Property established at initialization and preserved during training
- Introduces SipIt algorithm for exact input reconstruction from hidden states

**Implications for framework:**
- The framework question shifts from preservation to **organization**: whether preserved information is structured usefully
- Hallucination is a failure of **information access**, not information storage

**Framework connection:** Sections 11.5, Glossary (injectivity establishes matching and decompression failures as the mechanisms)

---

### 3.2 Training creates organization (belief-state convergence)

**Source:** Teoh, J., et al. (2025). *Next-Latent Prediction Transformers Learn Compact World Models.* [arXiv:2511.05963](https://arxiv.org/abs/2511.05963)

**Key Findings:**
- NextLat trains transformers with self-supervised predictions in latent space
- **Latents provably converge to belief states**; compressed information of history necessary to predict future
- Significant gains in representation compression and downstream accuracy
- Standard transformers "lack an inherent incentive to compress history into compact latent states"

**Implications for framework:**
- **Organization requires training**; injectivity alone does not provide useful structure
- Structured compression (not degenerate collapse) emerges from proper training objectives
- Belief-state manifolds are learnable with appropriate training pressure

**Framework connection:** Sections 4.4, 4.5 (matching and decompression); Documentation: `STRUCTURED_COMPRESSION_VS_DEGENERATE_COLLAPSE.md`, `INFORMATION_VS_ORGANIZATION.md`

---

## 4. The universal manifold

### 4.1 Unsupervised embedding translation

**Source:** Jha, R., et al. (2025). *Harnessing the Universal Geometry of Embeddings.* [arXiv:2505.12540](https://arxiv.org/abs/2505.12540)

**Key Findings:**
- vec2vec method translates embeddings between models with completely different architectures, parameter counts, and training data, **without paired data**
- **>0.92 cosine similarity** between translated embeddings and ground truth
- **Perfect matching on 8000+ embeddings** without knowing possible match set in advance
- Preservation of semantic information sufficient for classification and attribute inference

**Implications for framework:**
- The universal manifold hypothesis is measurable with embedding translation methods
- All capable models converge to the same underlying geometric structure
- Translation succeeds by learning the shared latent representation all models approximate
- Hallucinations can be detected as geometric outliers (off-manifold representations)

**Framework connection:** Sections 7.4 (Capacity Estimation via Universal Manifold), 11.5 (Practical Implementation)

---

### 4.2 Platonic representation hypothesis

**Source:** Huh, M., et al. (2024). *The Platonic Representation Hypothesis.* [arXiv:2405.07987](https://arxiv.org/abs/2405.07987)

**Key Findings:**
- Different architectures trained on different data converge to the same geometric shapes for representing concepts
- The internal geometry of "truth" is imposed by reality being modeled, not arbitrary model choice
- Representations are projections of a shared underlying manifold

**Implications for framework:**
- The manifold geometry is **determined by the object, not the model**
- Provides theoretical foundation for why vec2vec translation works
- Supports Definition 14 (The Universal Manifold)

**Framework connection:** Sections 11.5.0 (Plato's Cave intuition), Definition 14

---

## 5. The form prior (Zipf evidence)

### 5.1 Form prior as null model (Zipf statistics)

**Source:** Berman, V. (2025a). *Random Text, Zipf's Law, Critical Length, and Implications for Large Language Models.* [arXiv:2511.17575](https://arxiv.org/abs/2511.17575)

**Key Findings:**
- Zipf distributions in both natural language and **LLM token statistics** arise purely from combinatorics and segmentation
- No optimization, semantics, or linguistic organization required
- Provides a "structurally grounded null model" for token statistics
- Clarifies which phenomena require deeper explanation beyond random-text structure

**Implications for framework:**
- **The form prior is mathematically real**; it's the null model structure
- Form is "free" (arises from combinatorics); content is what costs information
- Hallucination = relaxation to the null model when content constraints fail

**Framework connection:** Glossary (Form prior definition), Section 8.5 (Thermodynamic Interpretation)

---

### 5.2 Stability under lexical filtering

**Source:** Berman, V. (2025b). *Zipf Distributions from Two-Stage Symbolic Processes: Stability Under Stochastic Lexical Filtering.* [arXiv:2511.21060](https://arxiv.org/abs/2511.21060)

**Key Findings:**
- Stochastic Lexical Filter (SLF) selects tiny subset of combinatorial word space
- **Power-law tail is preserved** under a wide class of filters
- Head of distribution becomes flatter (few high-frequency types)
- Zipf exponents in range [1.1, 1.5] match empirical corpora (English, Russian, mixed-genre)

**Implications for framework:**
- The form prior structure persists across languages and filtering mechanisms
- "Flat head + power-law tail" is a universal geometric signature
- Linguistic constraints filter the space but don't change asymptotic structure

**Framework connection:** Glossary (Form prior), Theorem 5 (Thermodynamic Hallucination)

---

## 6. Noise and error correction

### 6.1 Stochastic resonance (physical foundation)

**Source:** Gammaitoni, L., et al. (1998). *Stochastic Resonance.* Reviews of Modern Physics, 70(1), 223–287. [DOI](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.70.223)

**Key Findings:**
- Adding noise to a weak signal can make it **more detectable**
- Requires three ingredients: energetic barrier, weak coherent signal, noise source
- Optimal noise level exists; too little or too much degrades performance

**Implications for framework:**
- Maps directly to LLM generation: logit threshold (barrier), weak knowledge (signal), temperature (noise)
- Explains why $T=0$ (greedy decoding) is suboptimal for weak knowledge retrieval
- Provides physical basis for Theorem 6 (Optimal Noise Principle)

**Framework connection:** Section 8.6.3, 8.6.5 (Three Ingredients); Documentation: `NOISE_AND_ERROR_CORRECTION.md`

---

### 6.2 Verification-first improves reasoning

**Source:** Wu, S., & Yao, Q. (2025). *Asking LLMs to Verify First is Almost Free Lunch.* [arXiv:2511.21734](https://arxiv.org/abs/2511.21734)

**Key Findings:**
- Asking LLMs to verify a candidate answer improves accuracy
- Works **even if the candidate is random or wrong**
- Verification is easier than generation (discrimination vs generation)

**Implications for framework:**
- **Random answers act as "thermal shock"**: kicks system out of local minima
- Validates Optimal Noise Principle: noise can improve performance
- Verification is reverse reasoning that detects geometric distortion
- Supports Prediction 16 (Stochastic Resonance)

**Framework connection:** Section 7.5 (Verification-First), Theorem 6 (Optimal Noise)

---

## 7. Architectural validation

### 7.1 Titans memory hierarchy

**Source:** Behrouz, A., et al. (2025). *Titans: Learning to Memorize at Test Time.* [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)

**Key Findings:**
- Neural long-term memory module that learns at test time
- Explicit three-tier memory hierarchy: long-term (weights), working (attention), adaptive (test-time learning)
- Forgetting gate manages capacity
- Momentum-based updates capture token flow structure

**Implications for framework:**
- Validates memory duality (static/dynamic codebook)
- Confirms compression paradox: long context cannot fit in small state
- Test-time learning = dynamic atom creation = capacity extension
- Forgetting gate = sink severity control

**Framework connection:** Section 11.7 (Titans validation), Proposition 8 (Test-Time Capacity Extension)

---

## 8. Cross-reference matrix

| Evidence | Framework Section | Theorem/Definition | Documentation |
|----------|------------------|-------------------|---------------|
| Nikolaou (Injectivity) | 11.5 | N/A | INFORMATION_VS_ORGANIZATION.md |
| Teoh (NextLat) | 4.4, 4.5 | N/A | STRUCTURED_COMPRESSION.md |
| Jha (vec2vec) | 7.4, 11.5 | Def 14 | N/A |
| Huh (Platonic) | 11.5 | Def 14 | N/A |
| Berman (Zipf/Form Prior) | 3, 8.5 | Thm 5 | N/A |
| Gammaitoni (Stochastic Resonance) | 8.6 | Thm 6 | NOISE_AND_ERROR_CORRECTION.md |
| Wu & Yao (Verify-First) | 7.5, 8.6 | Thm 6, Pred 16 | NOISE_AND_ERROR_CORRECTION.md |
| Behrouz (Titans) | 11.7 | Prop 8, Pred 25-26 | N/A |

---

## 9. Open questions requiring further evidence

| Claim | Current Status | Needed Evidence |
|-------|---------------|-----------------|
| Hallucination rate scales as $e^{\Delta S}$ | Theoretical | Empirical measurement of entropy gap vs hallucination rate |
| Optimal $T^*$ varies with topic capacity | Conjectured (Pred 24) | Temperature sweep experiments across topics |
| Multi-hop accuracy decays as $(1-\epsilon)^n$ | Theoretical (Thm 4) | Chain-length experiments |
| Context crowding U-curve (Pred 18) | Theoretical | Context length vs accuracy measurements |
| Atom coverage correlates with accuracy (Pred 22) | Theoretical | SAE/probing experiments |

---

## 10. Summary

The framework's foundational claims now have external support:

1. **Information preservation and organization differ**: Supported by Nikolaou (injectivity) and Teoh (training creates organization)

2. **Universal manifold**: Supported by Jha (vec2vec achieves >0.92 similarity) and Huh (Platonic hypothesis)

3. **Form prior has a concrete candidate distribution**: Berman shows Zipf distributions arise from combinatorics and provide a null model for LLM tokens

4. **Optimal noise exists**: Supported by Gammaitoni (physics) and Wu & Yao (verification-first empirics)

5. **Memory hierarchy has architectural support**: Supported by Titans architecture

The remaining predictions (Sections 9.1-9.7) await systematic experimental validation.

---

## References

1. Nikolaou, G., et al. (2025). Language Models are Injective and Hence Invertible. arXiv:2510.15511
2. Teoh, J., et al. (2025). Next-Latent Prediction Transformers Learn Compact World Models. arXiv:2511.05963
3. Jha, R., et al. (2025). Harnessing the Universal Geometry of Embeddings. arXiv:2505.12540
4. Huh, M., et al. (2024). The Platonic Representation Hypothesis. arXiv:2405.07987
5. Berman, V. (2025a). Random Text, Zipf's Law, Critical Length, and Implications for LLMs. arXiv:2511.17575
6. Berman, V. (2025b). Zipf Distributions from Two-Stage Symbolic Processes. arXiv:2511.21060
7. Gammaitoni, L., et al. (1998). Stochastic Resonance. Rev. Mod. Phys. 70(1), 223–287
8. Wu, S., & Yao, Q. (2025). Asking LLMs to Verify First is Almost Free Lunch. arXiv:2511.21734
9. Behrouz, A., et al. (2025). Titans: Learning to Memorize at Test Time. arXiv:2501.00663

