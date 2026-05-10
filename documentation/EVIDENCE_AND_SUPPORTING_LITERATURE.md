# Evidence and supporting literature

This document catalogs empirical evidence and independent research relevant to the Hallucinations in Noisy Channels framework. Each finding links to a specific theoretical claim and keeps the evidence status separate from the interpretation.

---

## 1. From theory to observation: compression in language

### 1.1 The theoretical foundation

The framework begins with an idea from algorithmic information theory: learning can be modeled as compression that preserves useful structure. The shortest program that generates data captures a limiting notion of structure. This is Kolmogorov complexity $K(x)$, a theoretical quantity that guides the argument but is not computable in general.

HNC uses this foundation as an accounting language. In experiments, the framework relies on computable proxies rather than exact Kolmogorov complexity.

### 1.2 The empirical manifestation: Zipf's law

Zipf distributions provide one candidate signature for form structure in language.

Berman (2025a, 2025b) reports that Zipf-like rank-frequency behavior can arise from combinatorics and segmentation:

$$
p(r) \propto r^{-\alpha}, \quad \alpha \approx 1.1 - 1.5
$$

The mechanism uses the interaction of two exponentials:
1. Exponential growth of possible word types with length.
2. Exponential decay of probability for each word type with length.

This yields a power-law without any semantic content, optimization, or linguistic organization.

### 1.3 The connection: form prior = Zipf distribution

HNC treats Zipf-like structure as evidence for a form-prior null model:

| Concept | Definition | Evidence |
|---------|------------|----------|
| **Kolmogorov** | Theoretical: shortest program that generates data | Foundation (uncomputable) |
| **Zipf** | Empirical: power-law distribution over tokens | Berman (2025): arises from combinatorics |
| **Form prior** | The distribution that can dominate when content constraints fail | Candidate null model |

When content constraints are absent or weak, generation can move toward a high-entropy distribution consistent with linguistic form. Zipf-like statistics are one candidate signature of that form structure.

Implication: the form prior does not require semantic grounding. It can act as a null model for fluent structure. Source-supported content distinguishes grounded output from this baseline.

### 1.4 Why this matters

This connection gives a concrete candidate for the form prior:

Answer: Zipf-like token or word statistics can arise from combinatorial structure independent of meaning. Berman gives mathematical and empirical support for this candidate across English, Russian, and mixed corpora.

Form structure can arise from combinatorics. Content support still requires source signal. Hallucination risk rises when generation satisfies form constraints without enough content support.

---

## 2. Summary of evidence status

| Core Claim | Status | Primary Source |
|------------|--------|----------------|
| LMs preserve information under the cited assumptions | Formal result | Nikolaou et al. (2025) |
| Organization requires training pressure in the cited setting | Formal and empirical support | Teoh et al. (2025) - NextLat |
| Shared representation geometry is measurable in some settings | Bounded support | Jha et al. (2025), Huh et al. (2024) |
| Form-prior null model has mathematical support | Formal support for the cited model | Berman (2025a, 2025b) |
| Verification-First improves accuracy | **Empirical** | Wu & Yao (2025) |
| Optimal noise exists ($T^* > 0$) | **Theoretical and empirical** | Gammaitoni et al. (1998), Wu & Yao (2025) |
| Test-time learning extends capacity | **Architectural** | Behrouz et al. (2025) - Titans |

---

## 3. Information preservation vs organization

### 3.1 LMs are injective under the cited assumptions

**Source:** Nikolaou, G., et al. (2025). *Language Models are Injective and Hence Invertible.* [arXiv:2510.15511](https://arxiv.org/abs/2510.15511)

**Findings:**
- Mathematical proof that transformer LMs mapping discrete sequences to continuous representations are injective under the paper's assumptions.
- Empirical check: billions of collision tests across six models reported zero collisions.
- Property established at initialization and preserved during training in the cited setting.
- Introduces SipIt algorithm for exact input reconstruction from hidden states.

**Implications for framework:**
- The framework question shifts from preservation to organization: whether preserved information is structured usefully.
- Some hallucinations can be modeled as failures of information access rather than storage.

**Framework connection:** Sections 11.5, Glossary (injectivity establishes matching and decompression failures as the mechanisms)

---

### 3.2 Training creates organization (belief-state convergence)

**Source:** Teoh, J., et al. (2025). *Next-Latent Prediction Transformers Learn Compact World Models.* [arXiv:2511.05963](https://arxiv.org/abs/2511.05963)

**Findings:**
- NextLat trains transformers with self-supervised predictions in latent space
- Latents converge to belief states under the paper's assumptions; these states compress history needed to predict future observations.
- Significant gains in representation compression and downstream accuracy
- Standard transformers "lack an inherent incentive to compress history into compact latent states"

**Implications for framework:**
- Organization requires training pressure in the cited setting; injectivity alone does not provide useful structure.
- Structured compression (not degenerate collapse) emerges from proper training objectives
- Belief-state manifolds are learnable with appropriate training pressure

**Framework connection:** Sections 4.4, 4.5 (matching and decompression); Documentation: `STRUCTURED_COMPRESSION_VS_DEGENERATE_COLLAPSE.md`, `INFORMATION_VS_ORGANIZATION.md`

---

## 4. The universal manifold

Current status: supported as shared local, semantic, or domain-specific geometry; not established as a single global manifold across all modalities and models.

### 4.1 Unsupervised embedding translation

**Source:** Jha, R., et al. (2025). *Harnessing the Universal Geometry of Embeddings.* [arXiv:2505.12540](https://arxiv.org/abs/2505.12540)

**Findings:**
- vec2vec method translates embeddings between models with completely different architectures, parameter counts, and training data, **without paired data**
- Greater than 0.92 cosine similarity between translated embeddings and ground truth in the reported setting.
- Perfect matching on 8000+ embeddings in the reported setting without knowing the possible match set in advance.
- Preservation of semantic information sufficient for classification and attribute inference

**Implications for framework:**
- The universal manifold hypothesis is measurable with embedding translation methods
- The cited result supports substantial shared geometry in the reported text-embedding setting
- Translation succeeds by learning a useful cross-model map between related latent spaces
- Some hallucinations may be detectable as geometric outliers or poor cross-model translations, but this remains an empirical prediction

**Framework connection:** Sections 7.4 (Capacity Estimation via Universal Manifold), 11.5 (Practical Implementation)

---

### 4.2 Platonic representation hypothesis

**Source:** Huh, M., et al. (2024). *The Platonic Representation Hypothesis.* [arXiv:2405.07987](https://arxiv.org/abs/2405.07987)

**Key Findings:**
- Different architectures trained on different data can develop geometrically similar representations
- The internal geometry of a representation is partly constrained by the reality being modeled
- The strongest defensible claim is convergence toward overlapping or partially shared structures, not universal identity of all representations

**Implications for framework:**
- Manifold geometry is constrained by the represented object and shaped by modality, architecture, data, and task
- Provides a theoretical basis for why vec2vec-style translation can work
- Supports Definition 14 (Universal Manifold Hypothesis), with the caveat that full topology remains open

### 4.3 Domain support and counter-evidence

**Supporting source:** Li, Z., & Walsh, A. (2026). *Platonic representation of foundation machine learning interatomic potentials.* Nature Machine Intelligence. https://www.nature.com/articles/s42256-026-01235-7

**Key implication:** In physically constrained domains, independently trained models can project into a common latent organization preserving chemical periodicity and structural invariants. This strengthens the claim that shared geometry can emerge when models are constrained by the same external object.

**Cautionary sources:**
- Koepke, A. S., Zverev, D., Ginosar, S., & Efros, A. A. (2026). *Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale.* arXiv:2604.18572. https://arxiv.org/abs/2604.18572
- Gröger, F., Wen, S., & Brbić, M. (2026). *Revisiting the Platonic Representation Hypothesis: An Aristotelian View.* arXiv:2602.14486. https://arxiv.org/abs/2602.14486

**Key implication:** Cross-modal convergence appears weaker under larger and less constrained evaluations, and some representational-similarity metrics may be inflated by model scale. HNC should therefore treat the universal manifold as an operational hypothesis and measurement target, not a settled global fact.

**Framework connection:** Section 11.5.0 (shared objects and shared geometry), Definition 14

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

**Framework connection:** Glossary (Form prior), Conjecture 5 (Thermodynamic Hallucination Model)

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
- Explains why $T=0$ (greedy decoding) can be suboptimal for weak but recoverable knowledge retrieval
- Provides physical basis for Conjecture 6 (Optimal Noise Principle)

**Framework connection:** Section 8.6.3, 8.6.5 (Three Ingredients); Documentation: `NOISE_AND_ERROR_CORRECTION.md`

---

### 6.2 Verification-first improves reasoning

**Source:** Wu, S., & Yao, Q. (2025). *Asking LLMs to Verify First is Almost Free Lunch.* [arXiv:2511.21734](https://arxiv.org/abs/2511.21734)

**Key Findings:**
- Asking LLMs to verify a candidate answer improves accuracy
- Works **even if the candidate is random or wrong**
- Verification is easier than generation (discrimination vs generation)

**Implications for framework:**
- **Random answers may act as "thermal shock"**: a perturbation that can kick the system out of local minima
- Supports the Optimal Noise Principle: noise can improve performance when recoverable signal exists
- Verification is reverse reasoning that detects geometric distortion
- Supports Prediction 16 (Stochastic Resonance)

**Framework connection:** Section 7.5 (Verification-First), Conjecture 6 (Optimal Noise)

---

## 7. Architectural evidence

### 7.1 Titans memory hierarchy

**Source:** Behrouz, A., et al. (2025). *Titans: Learning to Memorize at Test Time.* [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)

**Findings:**
- Neural long-term memory module that learns at test time
- Explicit three-tier memory hierarchy: long-term (weights), working (attention), adaptive (test-time learning)
- Forgetting gate manages capacity
- Momentum-based updates capture token flow structure

**Implications for framework:**
- Supports memory duality as an architectural pattern.
- Provides an example of compression pressure in long contexts.
- Test-time learning maps to dynamic atom creation and capacity extension in HNC.
- Forgetting gates map to capacity management and sink severity control.

**Framework connection:** Section 11.7 (Titans architectural evidence), Proposition 8 (test-time capacity extension)

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
| Hallucination rate scales as $e^{\Delta S}$ | Conjectural | Empirical measurement of entropy gap vs hallucination rate |
| Optimal $T^*$ varies with topic capacity | Conjectured (Pred 24) | Temperature sweep experiments across topics |
| Multi-hop accuracy decays as $(1-\epsilon)^n$ | Theoretical (Thm 4) | Chain-length experiments |
| Context crowding U-curve (Pred 18) | Theoretical | Context length vs accuracy measurements |
| Atom coverage correlates with accuracy (Pred 22) | Theoretical | SAE/probing experiments |

---

## 10. Summary

The framework's main claims have several forms of external support, with important limits:

1. **Information preservation and organization differ**: supported by Nikolaou (injectivity) and Teoh (training creates organization in the cited setting).

2. **Universal manifold**: supported as a bounded geometry hypothesis by Jha (vec2vec reports >0.92 similarity) and Huh (Platonic representation hypothesis).

3. **Form prior has a concrete candidate distribution**: Berman shows Zipf-like distributions can arise from combinatorics and provide a null model for token statistics.

4. **Optimal noise has a plausible basis**: supported by Gammaitoni (physics) and motivated by Wu and Yao (verification-first empirics).

5. **Memory hierarchy has architectural support**: supported by Titans as a useful test case.

The remaining predictions (Sections 9.1-9.7) await systematic empirical testing.

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
