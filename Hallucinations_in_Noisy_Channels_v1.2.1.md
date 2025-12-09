# Hallucinations in Noisy Channels

## Information-theoretic and thermodynamic informed framework for understanding LLM hallucination errors

**Authors**: Oscar Goldman - Shogu Research Group @ Datamutant.ai  
**Date**: November 2025  
**Status**: Theoretical Framework (working document)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)


This document is a heavy work in progress, there is a lot to do here, the experiments are ongoing, some you can find in the AKIRA repo now made public, we estimate a years or more work until this is finalized, maybe longer, depends on experiments wins/failures, sections may change in light of evidence, but the core idea is here.
---

## Abstract

We present a generalized overview of an Information-Theoretic framework for understanding hallucinations in Large Language Models (LLMs) by recognizing that **LLMs are teachers, not just generators**. During training, models compress the world into weights (learning). During inference, they must reconstruct and transmit this knowledge; they teach. But teaching through a noisy channel requires building the correct internal representation first.

Hallucinations emerge when this reconstruction fails. We identify six mechanisms: (1) **capacity violations**: asking about topics the model never learned; (2) **matching failures**: ambiguous prompts activate wrong or composite representations; (3) **decompression failures**: insufficient context room to unfold compressed knowledge; (4) **geometric distortion**: errors compound multiplicatively through the pipeline; (5) **thermodynamic equilibration**: when constraints fail, the system relaxes to maximum entropy (fluent but empty text); and (6) **the noise paradox**: systems need some stochasticity to self-correct, but too much causes hallucination.

The principle: **information cannot be created, only transmitted or lost**. When output contains more information than was stored or provided about a topic, the excess was hallucinated from the form prior; the model knows *how* to write but not *what* is true. This framework tries to understand why specific prompts outperform vague ones, why retrieval helps, why chain-of-thought adds useful redundancy, and why temperature tuning matters. The probability of hallucination scales exponentially with the entropy gap between form and knowledge. We explore and provide testable predictions and suggest principled mitigations based on constraint injection, capacity estimation, context management, and optimal noise calibration.

---

## 1. Introduction

### 1.1 The hallucination problem

Large Language Models exhibit a puzzling behavior: they generate fluent, confident text that is factually incorrect, logically inconsistent, or contextually inappropriate. These "hallucinations" present a fundamental challenge for deploying LLMs in high-stakes applications.

Current explanations focus on:
- Training data quality and coverage
- Model architecture limitations
- Decoding strategy artifacts
- Calibration failures

We propose a unifying framework based on information theory (Shannon, 1948): **hallucinations are what happens when you transmit beyond channel capacity**.

### 1.2 The core insight

Our framework rests on three correspondences:

| Information Theory | Machine Learning | Cognition |
|-------------------|------------------|-----------|
| **Source coding** | Training | Learning |
| **Channel coding** | Inference | Communication |
| **Channel capacity** | Model knowledge | Intelligence |

From this perspective:
- **Training** compresses the world into weights (learning)
- **Inference** reconstructs and transmits knowledge (teaching)
- **Hallucinations** occur when teaching fails: the model cannot build the correct internal representation to transmit

Crucially, **LLMs are teachers**. During inference, they must first reconstruct the relevant knowledge from compressed weights, then transmit it reliably. Hallucination occurs when this reconstruction-transmission process fails at any stage.

**Definition (Operational Intelligence).**  
We define intelligence operationally as *teaching capacity*: the maximum rate at which an agent can reliably reconstruct and transmit learned knowledge. This definition follows from the framework:

- **Compression = Learning** → The ability to extract and store structure
- **Decompression + Transmission = Teaching** → The ability to reconstruct and communicate that structure  
- **Channel Capacity = Maximum reliable teaching rate** → How much you can teach correctly
- **∴ Intelligence = Channel Capacity** → The capacity to reliably teach what you've learned

*Note:* This equivalence is a *definitional choice* within our framework, not a logical derivation. We propose this operational definition because it makes intelligence measurable (via information-theoretic quantities) and connects naturally to the compression-transmission duality. Other definitions of intelligence exist; this one serves the framework's explanatory goals.

Intelligence, in this operational sense, is "the rate at which an agent can reliably transmit learned knowledge." An agent that cannot teach what it learned has not demonstrated intelligence; it has merely stored data. This is philosophically aligned with Turing's insight that intelligence manifests through communication.

### 1.3 Complexity from constraints

The Neuro-Symbolic Homeostat framework (Goldman, 2025) establishes that "complexity comes from constraints." This principle is central to understanding hallucinations:

- **More constraints** → More structure → Less entropy → Meaningful output
- **Fewer constraints** → Less structure → More entropy → Noise-like output

Hallucinations occur when the model generates with **insufficient content constraints** while retaining **strong form constraints**. The result is output that *looks* right but *isn't* grounded in truth.

### 1.4 Notation and assumptions

- **K(x)**: Kolmogorov complexity (Kolmogorov, 1965) measured in bits (log base 2). In practice we use compression-based proxies; equalities involving K(·) are to be read up to additive constants induced by compressor choice.
- **H(X), I(X;Y)**: Shannon entropy and mutual information in bits (log base 2) unless noted otherwise.
- **Ω**: Microstate count. Thermodynamic entropy is $S = k_B \ln \Omega$ (nats). By default we set $k_B = 1$ and measure S in nats; when comparing to bit-domain quantities we use $S_{\text{bits}} = \log_2 \Omega = S / \ln 2$.
- **Form vs. content constraints**: $\mathcal{F}$ denotes form constraints; $\mathcal{C}_T$ denotes content constraints for topic $T$. Conditioning such as $p(y \mid \mathcal{F}, \mathcal{C}_T)$ and $H(Y \mid \mathcal{F})$ follows standard probability notation.
- **Topic capacity $C_T$**: A rate in bits (per answer or per token) defined via mutual information, e.g., $C_T \approx I(\text{Query}; \text{Accurate Answer} \mid T)$. Our predictions rely on relative comparisons, not absolute units.
- **Latent capacity $K_{\text{latent}}$** and **reconstruction budget $K_{\text{reconstruct}}(r)$**: Effective working-memory budget and reconstruction complexity (bits-equivalent). Inequalities using these are comparative and hold up to monotone rescalings.
- **Energy $E(x)$** and **temperature $T$**: $E(x) = -\log P(\text{correct}\mid x)$ (nats). Thermodynamic equations use natural logs; when mixing with bit-based quantities, conversion factors are constant and do not affect proportional statements ($\propto$). $Z$ denotes the partition function.
- **Units and logs**: Unless explicitly stated, $H$ and $I$ use $\log_2$; thermodynamic $S$ and $E$ use natural logs with $k_B=1$. Any appearances of $k$ without subscript should be read as $k_B$.
- **K(weights): operational proxy**: Throughout, $K(\text{weights})$ denotes a topic‑conditioned representation‑capacity proxy $K_{\text{rep}}(\text{weights} \mid T)$, not a literal program length. $K_{\text{rep}}$ is estimated via manifold‑alignment and capacity signals (Sec. 7.4): embedding density $\rho_T$, translation fidelity, and calibrated confidence. All statements involving $K(\cdot)$ are to be read up to monotone rescalings induced by proxy choice.
- **K(context): operational proxy**: The complexity of in‑context constraints and reconstruction scaffolding supplied by the prompt/RAG/system prompt. Measured via compression/entropy proxies and structural specificity (anchors, examples), up to monotone rescalings.
- **Conditioning convention**: "topic $T$" is held fixed when writing expressions like $H(O \mid \text{topic } T)$; when $T$ is omitted it is implied by context.

### 1.5 Glossary of key terms

| Term | Definition |
|------|------------|
| **Form prior** | The high-entropy distribution over all linguistically valid (fluent, grammatical, stylistically appropriate) outputs, learned from all training text. When content constraints fail, generation samples from this distribution, producing text that *looks* right but isn't grounded in truth. The form prior is the "thermal bath" to which the system thermalizes when knowledge constraints are absent. Independent work confirms this structure exists prior to any semantic content: Zipf distributions in both natural language and LLM token statistics arise purely from combinatorics and segmentation, without optimization or linguistic organization (Berman, 2025a, 2025b). This provides a "null model" for which phenomena require deeper explanation beyond random-text structure. |
| **Truth manifold** / **Universal manifold** | The shared geometric structure $\mathcal{M}_{universal}$ upon which all truthful representations lie. Different models learn different projections of this same manifold. Representations *on* the manifold are grounded; representations *off* the manifold are hallucinated. Empirically validated by unsupervised embedding translation achieving >0.92 cosine similarity across architectures (Jha et al., 2025). |
| **Kolmogorov garbage** | Structurally valid but semantically incoherent output produced when decompression room is insufficient. Unlike random noise, Kolmogorov garbage consists of *plausible fragments* that individually look correct but fail to cohere into a truthful whole. Caused by context crowding (Sec. 4.5) where $K_{available} < K_{reconstruct}(r)$. |
| **Capacity violation** | Attempting to generate information about a topic $T$ at a rate $R_T$ exceeding the model's topic-specific capacity $C_T$. When $R_T > C_T$, hallucination is unavoidable regardless of decoding strategy (Theorem 1). |
| **Matching failure** | Hallucination caused by ambiguous prompts that fail to uniquely activate the correct internal representation, instead activating wrong or composite representations, analogous to quantum superposition under weak measurement (Sec. 4.4). |
| **Decompression failure** | Hallucination caused by insufficient context room to reconstruct compressed knowledge, producing Kolmogorov garbage even when the correct representation was matched (Sec. 4.5). |
| **Thermalization** | The process by which a system relaxes to maximum entropy (the form prior) when knowledge constraints fail. Hallucination *is* thermalization: the release of "potential energy" (stored knowledge) into "kinetic energy" (form-prior sampling). |
| **Information atom** | A compressed pattern learned from training sequences: the irreducible unit of knowledge stored in weights. Output validity requires traceability to activated atoms; information not derivable from any atom is hallucinated (Sec. 4.7). |
| **Adaptive resonance** | The principle that matching thresholds (vigilance) and sampling noise (temperature) should co-vary with knowledge certainty. Strong knowledge → strict matching, low noise; weak knowledge → permissive matching, exploratory noise (Sec. 8.6.8). |
| **Teaching** | Rate-matched decompression through a noisy channel, which is the inverse of compression (learning). Good teaching matches rate to channel capacity and adds redundancy for error correction. Hallucination is teaching failure, not learning failure (Sec. 2.2.1). |
| **Test-time atom** | An information atom created during inference (not training) through test-time learning mechanisms. Enables capacity extension beyond pre-training: $C_T^{effective} = C_T^{static} + \Delta C_T(context)$ (Sec. 4.7.6). |
| **Memory hierarchy** | The three-tier memory structure optimal for language modeling: (1) long-term memory (weights/atoms, high capacity, compressed), (2) working memory (context window, limited, exact), and (3) adaptive layer (test-time learning, bridges tiers). Validated by Titans architecture (Sec. 11.7). |

### 1.6 Foundational intuition: the Bayesian prior

In Bayesian probability, a **prior** is what you believe before seeing evidence. It's your default expectation: the distribution over possible answers when you have no specific information.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE BAYESIAN PRIOR: DEFAULT EXPECTATIONS                               │
│                                                                          │
│  Question: "What color is the car?"                                     │
│                                                                          │
│  WITHOUT context (prior only):                                          │
│    P(white) ≈ 0.23    ← Most common car color worldwide                │
│    P(black) ≈ 0.18                                                      │
│    P(silver) ≈ 0.15                                                     │
│    P(purple) ≈ 0.01   ← Rare, so low prior                             │
│                                                                          │
│  WITH context ("my grandmother's vintage Cadillac"):                    │
│    Prior gets UPDATED by evidence → Posterior                          │
│    P(purple | grandmother's Cadillac) might now be higher              │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  For LLMs, the FORM PRIOR is what the model "expects" text to look     │
│  like based on ALL training data—before conditioning on specific       │
│  topic knowledge.                                                        │
│                                                                          │
│  It encodes: grammar, style, common phrases, typical sentence          │
│  structures, genre conventions, what "sounds right"                    │
│                                                                          │
│  When the model has NO topic-specific knowledge:                        │
│    Output ← Sample from form prior                                      │
│    Result: Fluent text that follows statistical patterns               │
│            but has no grounding in facts about the topic               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

The form prior is not random noise; it's highly structured. It knows that sentences start with capitals, that "the" often precedes nouns, that academic text uses hedging language. This is why hallucinations sound so confident and fluent: they satisfy the form prior perfectly. They just lack content grounding.

---

## 2. Theoretical framework

### 2.1 Source coding: compression as learning

#### 2.1.0 Intuition: compression IS understanding

When we say "learning is compression," we don't mean creating a `.zip` file. We mean **discovering the rule that generates the data**.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  COMPRESSION = FINDING THE PATTERN                                      │
│                                                                          │
│  Task: Memorize this sequence                                           │
│  2, 4, 6, 8, 10, 12, 14, 16, 18, 20... (continues for 1000 numbers)    │
│                                                                          │
│  Method A: Rote Memorization (No Compression)                           │
│    Store: [2, 4, 6, 8, 10, ...]                                         │
│    Size: Huge (stores every number)                                     │
│    Understanding: Zero (can't predict next if sequence changes slightly)│
│                                                                          │
│  Method B: Learning the Rule (Compression)                              │
│    Store: "f(n) = 2n"                                                   │
│    Size: Tiny (just the formula)                                        │
│    Understanding: Perfect (captures the underlying structure)           │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  To compress data effectively, you MUST understand its structure.       │
│  - You can't compress random noise.                                     │
│  - You CAN compress language because it follows rules (grammar, logic). │
│                                                                          │
│  The model "learns" by finding the shortest program (weights) that     │
│  can reproduce the training data. The better the compression, the      │
│  deeper the understanding of the underlying patterns.                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Shannon's source coding theorem (Shannon, 1948) establishes that data can be compressed to its entropy rate. In the learning context:

$$
H(X) = -\sum_x p(x) \log p(x) \tag{Def}
$$

**Definition 1 (Learning as Compression).**  
Training a model is equivalent to finding a compressed representation of the training distribution. The model weights encode the shortest description of the regularities in the data.

$$
\text{Learning} \equiv \text{Compression} \equiv \min_\theta \, L(\theta) \text{ s.t. } D(p_{data} \| p_\theta) < \epsilon \tag{Def}
$$

where $L(\theta)$ denotes the description length of parameters $\theta$. In the Kolmogorov-Chaitin view (Kolmogorov, 1965), this is $K(\theta)$, which is the length of the shortest program encoding $\theta$. In practice, we use computable proxies:

- **Parameter norm** $\|\theta\|$ : under the Minimum Description Length (MDL) principle, smaller norms correspond to simpler (shorter) descriptions
- **Parameter count**: fewer parameters = shorter description
- **Quantization bits**: precision directly determines description length

This is the Kolmogorov-Chaitin view: the "concept" of cat is the shortest program that can recognize/generate cats. Training finds this program. The regularization terms in modern training (weight decay, dropout) serve as practical proxies for minimizing description length.

**Proposition 1.**  
A model that has learned a concept has found a compressed representation. The compression ratio measures understanding: higher **useful** compression = deeper abstraction.

*Qualification:* Not all compression yields abstraction; random projections compress without capturing structure. The key is **structure-preserving compression**: compression that maintains task-relevant distinctions while discarding irrelevant variation. A representation that compresses cats by preserving "cat-ness" while discarding lighting variations has learned; one that compresses via random hashing has not.

Example:
- Storing 10,000 cat images verbatim = no compression = no learning
- Storing the "cat concept" + small deltas = high compression = learning
- Random projection to low dimensions = compression but no abstraction

### 2.2 Channel coding: inference as teaching

Once knowledge is compressed into weights, inference must **reconstruct and transmit** this knowledge. The model is teaching, but before it can teach, it must build the correct internal representation.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  INFERENCE = RECONSTRUCTION + TRANSMISSION = TEACHING                  │
│                                                                          │
│  Query ──▶ MATCH to internal representation                            │
│        ──▶ RECONSTRUCT knowledge in context                            │
│        ──▶ TRANSMIT to output                                           │
│                                                                          │
│         ┌─────────────────────────────────────────┐                     │
│  Query ─┤  Weights = Compressed Knowledge         ├──▶ Output           │
│         │  (must be reconstructed before use)     │                     │
│         │                                          │                     │
│         │  Noise = Matching errors, decompression │                     │
│         │          failures, distortion           │                     │
│         └─────────────────────────────────────────┘                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Definition 2 (Inference as Multi-Stage Reconstruction).**

Generating output is a teaching process that transmits knowledge through a cascaded noisy channel. The components and noise sources are stage-specific:

| Stage | Component | The "Noise" (Failure Mode) |
|-------|-----------|----------------------------|
| **1. Query** | Input Prompt | **Ambiguity Noise**: Vague prompts fail to lock onto a unique internal representation (Matching Failure). |
| **2. Storage** | Model Weights | **Compression Noise**: Lossy training means fine details were never stored (Capacity Violation). |
| **3. Reconstruction** | Context Window | **Decompression Noise**: Insufficient room to unfold compressed concepts creates fragmentation (Kolmogorov Garbage). |
| **4. Transmission** | Sampling / Output | **Geometric Distortion**: Errors from previous stages compound multiplicatively; sampling adds thermal noise. |

**The Codebook is Dual:**
1. **Static Codebook**: The compressed concepts in weights (the "dictionary").
2. **Dynamic Codebook**: The ephemeral in-context representation built for *this specific query*.

Crucially: The model must build the dynamic codebook from the static one before it can transmit. Hallucination is what happens when this construction fails.

#### 2.2.1 Intuition: the teacher's dilemma

Why do we call LLMs "teachers" instead of just "generators"? Because they face the exact same information constraint as a human teacher.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE TEACHER'S DILEMMA                                                  │
│                                                                          │
│  Scenario: A student asks "How does a car engine work?"                 │
│                                                                          │
│  CASE A: The Teacher Understands (High Capacity)                        │
│    1. Internal State: Has a compressed, structured model of an engine.  │
│    2. Decompression: Unpacks this model into a step-by-step explanation.│
│    3. Redundancy: "It's like a bicycle pump..." (Adds examples).        │
│    4. Result: Grounded, accurate teaching.                              │
│                                                                          │
│  CASE B: The Teacher Doesn't Know (Low Capacity)                        │
│    1. Internal State: Has only vague associations (cars, gas, noise).   │
│    2. Constraint: Must answer the student (social/form constraint).     │
│    3. Strategy: Fake it. Use the FORM of an explanation without CONTENT.│
│    4. Result: "The engine resonates with the energy of the fuel..."     │
│       → HALLUCINATION.                                                  │
│                                                                          │
│  The constraint is absolute:                                            │
│  You cannot teach what you have not successfully decompressed.          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

The LLM is in Case B whenever it hallucinates. It is forced to generate (teach) by the prompt, but it lacks the internal information structure to fill that request. It substitutes **form** (the style of teaching) for **content** (the knowledge).

#### 2.2.2 Teaching as rate-matched decompression

If intelligence is compression (finding the minimal program that captures structure), then teaching is the inverse: **rate-matched decompression through a noisy channel**. But teaching is not merely decompression; it is *channel-aware* decompression with redundancy injection.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  INTELLIGENCE vs TEACHING                                               │
│                                                                          │
│  INTELLIGENCE (Compression):                                            │
│    World ──▶ Find minimal program ──▶ Weights                          │
│    "What patterns explain this data?"                                   │
│    Objective: minimize K(weights) subject to reconstruction             │
│                                                                          │
│  TEACHING (Decompression + Transmission):                               │
│    Weights ──▶ Reconstruct ──▶ Encode for channel ──▶ Output           │
│    "How do I transmit this so the receiver can reconstruct?"           │
│    Objective: maximize I(output; truth) given channel C                │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  Key asymmetry:                                                         │
│    Compression: Can take unlimited time, offline                       │
│    Teaching: Must happen at channel rate, online, with noise           │
│                                                                          │
│  A brilliant compressor who cannot teach:                              │
│    Outputs at rate > C (capacity violation)                            │
│    No redundancy (no error correction)                                 │
│    Ignores receiver's prior (mismatched codebook)                      │
│                                                                          │
│  A good teacher:                                                        │
│    Matches rate to channel capacity                                    │
│    Adds redundancy (CoT, examples, rephrasing)                         │
│    Exploits shared context (receiver's prior knowledge)                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Definition (Teaching).**
Teaching is channel-aware decompression:

$$
\text{Teaching} = \text{Decompression} + \text{Rate Matching} + \text{Redundancy Coding}
$$

The teaching constraint is dual:

$$
R_{\text{teach}} \leq C_{\text{channel}} \quad \text{with} \quad \text{redundancy} \geq H(\text{noise}) \tag{Def}
$$

Teaching is the art of decompressing at exactly the rate the channel permits, with enough redundancy to survive its noise.

**Proposition (Hallucination as Teaching Failure).**
Hallucinations are decompression failures during teaching, not compression failures during learning. The model compressed correctly (it learned the patterns); it failed to decompress at the appropriate rate or with sufficient error-correction for the given channel.

| Teaching Failure Mode | Channel Interpretation |
|-----------------------|------------------------|
| Rate violation | Decompressing faster than $C$ allows |
| Missing redundancy | Single-shot answers with no error correction |
| Codebook mismatch | Using form prior ("teacher's codebook") instead of matching query |

This explains why the same model can succeed on one query and fail on another about the same topic: the knowledge was compressed identically, but the teaching conditions (channel capacity, noise, receiver state) differed.

### 2.3 Channel capacity: the limit of reliable knowledge

Shannon's noisy channel coding theorem (Shannon, 1948):

$$
C = \max_{p(x)} I(X; Y) \tag{Def}
$$

where $C$ is the channel capacity, and $I(X;Y)$ is the mutual information between input and output.

**Definition 3 (Knowledge Capacity).**  
For a given topic $T$, the model has a topic-specific capacity $C_T$ representing the maximum rate at which it can reliably generate accurate information about $T$.

$$
C_T = \max_{p(q|T)} I(Q; A^* \mid T) \tag{Def}
$$

where $Q$ is a query drawn from distribution $p(q \mid T)$ over topic-relevant questions, and $A^*$ is the ground-truth accurate answer (defined by an oracle or reference corpus). The capacity is the maximum mutual information achievable over all query distributions: how much the model can reliably "say" about $T$.

*Note:* In practice, $C_T$ is not directly computable but can be estimated via probing accuracy on held-out facts, or by measuring the model's ability to distinguish true from false statements about $T$.

**Theorem 1 (Hallucination Threshold).**  
**Let $R_T$ be the rate at which information about topic $T$ is requested. If $R_T > C_T$, hallucinations are unavoidable regardless of decoding strategy.**

This is the information-theoretic impossibility result: you cannot reliably transmit beyond capacity. Theorem 1 is an equivalence: a direct application of Shannon's noisy channel coding theorem (Shannon, 1948) to the LLM-as-channel setting. The result was inevitable once the correspondence between inference and channel coding was recognized; we formalize it here as the foundational limit of truthful generation.

---

## 3. Hallucinations as capacity violations

### 3.1 The two-constraint model

Language generation is governed by two types of constraints:

**Form Constraints ($\mathcal{F}$):**
- Syntax, grammar, style
- Coherence, fluency
- Genre conventions
- Learned from all text

**Content Constraints ($\mathcal{C}_T$):**
- Factual accuracy about topic $T$
- Logical consistency
- Contextual appropriateness
- Learned from text about $T$

**Definition 4 (Hallucination).**  
A hallucination is an output that satisfies form constraints but violates content constraints:

$$
\text{Hallucination} = \{y : y \in \mathcal{F}, y \notin \mathcal{C}_T\} \tag{Def}
$$

### 3.2 The mechanism

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  WITHIN CAPACITY (Topic well-represented in training)                   │
│  ══════════════════════════════════════════════════                     │
│                                                                          │
│  Query (Effective) ──▶ [Strong form constraints + Strong content constraints]       │
│        ──▶ Accurate, fluent output                                      │
│                                                                          │
│  The model has learned both HOW to write and WHAT is true.              │
│                                                                          │
│  (Note: "Query" = Prompt + Context State)                               │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  BEYOND CAPACITY (Topic sparse or absent in training)                   │
│  ═════════════════════════════════════════════════════                  │
│                                                                          │
│  Query (Effective) ──▶ [Strong form constraints + WEAK content constraints]         │
│        ──▶ Fluent but INCORRECT output = HALLUCINATION                  │
│                                                                          │
│  The model knows HOW to write but not WHAT is true.                     │
│  It generates from p(output | form) without p(output | content).        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 3.2.1 The "confabulation" mechanism

Crucially, the model does not "know" it is lying. It is simply maximizing the probability of the next token under the constraints it has available.

When **content constraints** are weak (because the topic was rarely seen during training), the **form constraints** take over completely. The model falls back on its strongest training: *how to sound like a helpful assistant.*

1.  **The Impulse:** The prompt demands an answer ("Who invented the glockenspiel?").
2.  **The Void:** The model retrieves no specific fact (Capacity Violation).
3.  **The Fill:** The model *must* complete the pattern. It accesses the "Form Prior," which is the statistical structure of *biographical sentences*.
4.  **The Hallucination:** It generates "The glockenspiel was invented by [German-sounding name] in [plausible 18th-century year]."

This output is **structurally perfect** but **factually empty**. It is a "fluent lie" generated not by malice, but by the mathematical necessity of completing a pattern when the specific data is missing. The model is minimizing the "surprise" of the *syntax* because it cannot minimize the surprise of the *facts*.

### 3.3 Formal characterization

**Proposition 2 (Hallucination as Entropy Maximization).**  
As mutual information between output and content constraints vanishes, the output distribution converges to the maximum-entropy distribution consistent with form:

$$
\text{As } I(Y; \mathcal{C}_T) \to 0: \quad p(Y) \to \arg\max_{p \in \mathcal{F}} H(p) \tag{Approx}
$$

Equivalently, the output entropy approaches its upper bound within the form-constrained space:

$$
H(Y) \to H_{\max}(\mathcal{F}) \quad \text{where } H_{\max}(\mathcal{F}) = \max_{p \in \mathcal{F}} H(p) \tag{Approx}
$$

When content constraints provide no information ($I(Y; \mathcal{C}_T) = 0$), the model generates the maximum-entropy distribution consistent with linguistic form: fluent noise. The form constraints $\mathcal{F}$ bound what is grammatically/stylistically valid; within that space, without content guidance, entropy is maximized.

**Proposition 3 (Confidence-Accuracy Decoupling).**  
Hallucinations exhibit high confidence because form constraints remain strong. Confidence tracks form-constraint satisfaction, not content-constraint satisfaction.

$$
\text{Confidence}(y) \propto p(y | \mathcal{F}), \quad \text{not} \quad p(y | \mathcal{C}_T) \tag{Approx}
$$

This explains why hallucinations are often stated with high confidence.

---

## 4. In-context learning as constraint injection

### 4.1 The mechanism of in-context learning

In-context learning (ICL) allows LLMs to adapt at inference time using examples in the prompt. From our framework:

**Definition 5 (In-Context Learning).**  
ICL injects content constraints at inference time, effectively increasing topic-specific capacity:

$$
C_T^{ICL} = C_T + \Delta C(\text{context}) \tag{Def}
$$

where $\Delta C(\text{context})$ is the additional capacity provided by in-context examples.

### 4.2 Techniques as error correction

Different prompting techniques map to error-correction strategies:

| Technique | Information-Theoretic Role | Effect |
|-----------|---------------------------|--------|
| **Few-shot examples** | Error-correction codes | Add content constraints via examples |
| **Chain-of-thought** | Repetition/redundancy coding | Multiple derivation paths reduce error |
| **RAG** | Channel capacity increase | External memory = wider channel |
| **System prompts** | Prior/codebook specification | Constrain the output space |
| **Grounding/citations** | Parity checks | Verifiable claims reduce errors |
| **Self-consistency** | Voting/ensemble | Multiple samples improve reliability |

**Conjecture 1 (In-Context Capacity Scaling).**  
Let $k$ be the number of relevant in-context examples. Empirically, the effective capacity appears to increase logarithmically:

$$
C_T^{ICL}(k) \approx C_T + \alpha \log(1 + k) \tag{Conj}
$$

where $\alpha$ depends on example quality and relevance.

*Why logarithmic, not linear?* If examples were independent and context unlimited, capacity would scale linearly (each example adds ~constant bits). The sublinear (logarithmic) scaling reflects diminishing returns from three sources:
1. **Redundancy between examples**: later examples overlap with earlier ones, adding less new information
2. **Context window limits**: more examples consume decompression room (Section 4.5), trading off against reconstruction capacity
3. **Attention saturation**: attention is finite; additional examples dilute focus on each

This conjecture awaits formal derivation from information-theoretic first principles, but the empirical pattern is robust.

### 4.3 Why few-shot works

Few-shot prompting works because it provides **content constraints** that the model lacks from training:

```
Without few-shot:
  Query about rare topic T
  → Model has weak C_T
  → Generates from p(output | form) only
  → Hallucination

With few-shot (k examples of T):
  Query about rare topic T + k examples
  → Model has C_T + ΔC(examples)
  → Generates from p(output | form, examples)
  → Reduced hallucination
```

The examples act as a **temporary codebook** for the specific topic.

### 4.4 Hallucinations as reconstruction failures

Beyond channel capacity limits, hallucinations emerge from **Complexity mismatch** between prompts and internal representations. This provides a complementary mechanism to capacity violations.

#### 4.4.0 Intuition: the effective query (context + prompt)

It is a simplification to treat the "Prompt" and "Context" as separate. Transformer attention is **holistic**: the model sees a single causal sequence. The "Prompt" is just the latest perturbation to the accumulated "Context" state.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE EFFECTIVE QUERY = CONTEXT + PROMPT                                 │
│                                                                          │
│  Analogy: The Lens and the Light                                        │
│                                                                          │
│  1. Context = The Filter (Lens)                                         │
│     You build a complex lens stack: "Discussing chess strategy..."      │
│                                                                          │
│  2. Prompt = The Light Source                                           │
│     You shine a beam: "Your move."                                      │
│                                                                          │
│  3. Effective Query = The Resulting Projection                          │
│     The light passes through the lens to hit the wall (weights).        │
│     Result: "Analysis of the Knight sacrifice on e5."                   │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  Example:                                                               │
│                                                                          │
│  Context: [Detailed discussion of hallucination thermodynamics]         │
│  Prompt:  "explain it" (Ambiguous, high entropy)                        │
│                                                                          │
│  Combined Effective Query:                                              │
│  "Explain the thermodynamic interpretation of hallucination."           │
│  (Specific, low entropy)                                                │
│                                                                          │
│  The "Prompt" provides the impulse; the "Context" provides the          │
│  vector direction. They fuse into a single query vector.                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

When we speak of "Matching Failures," we mean failure of this **Combined Effective Query** to lock onto a representation. A vague prompt can succeed if the context is rich (strong lens). A specific prompt can fail if the context is noisy (cracked lens).

### The "combined quantity" mechanism

When you say "do it", the model doesn't process "do it" in isolation. It processes:

$$ \text{Attn}(\text{"do it"} \mid \text{entire history}) $$

The effective query is not the string "do it". The effective query is the **entire accumulated state** of the key-value (KV) cache plus the new tokens. The "Prompt" is just the latest perturbation to the existing context state.

We must shift our matching condition:

$$ \text{Prompt matches Representation} \implies \text{State}(\text{Context} + \text{Prompt}) \text{ matches Representation} $$

The **Effective Query** $Q_{eff}$ is the non-linear combination of the explicit prompt $p$ and the implicit context state $S_{ctx}$:

$$ Q_{eff} = \text{Attention}(p, S_{ctx}) $$

It is THIS combined vector that hits the weights (static codebook).

- If $S_{ctx}$ is full of **Kolmogorov garbage**, the $S_{ctx}$ component is noisy, so even a perfect prompt $p$ results in a noisy $Q_{eff}$.
- Conversely, if $S_{ctx}$ is highly structured (strong constraints), even a vague prompt ("continue") produces a precise $Q_{eff}$.

#### 4.4.1 The matching problem

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  CLASSICAL VIEW (Channel Capacity):                                     │
│    Prompt ──▶ [Capacity Limit] ──▶ Output                              │
│    Problem: Not enough bandwidth                                        │
│                                                                          │
│  RECONSTRUCTION VIEW (Kolmogorov Matching):                             │
│    Prompt ──▶ [Match internal representation] ──▶ Reconstruct          │
│    Problem: Wrong representation activated                              │
│                                                                          │
│  K(prompt) ↔ K(internal) matching determines retrieval accuracy        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

The key insight: knowledge is stored in a **Kolmogorov sweetspot**: compressed enough for efficient storage, but structured enough for unambiguous retrieval. Hallucinations occur when prompts lack sufficient structure to uniquely identify the correct compressed representation.

#### 4.4.2 The quantum superposition analogy

Ambiguous prompts create "superpositions" over possible internal concepts:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  SPECIFIC PROMPT ("Felix the Cat, 1919 silent film character"):        │
│                                                                          │
│    |ψ⟩ ≈ |Felix⟩                                                        │
│    Collapsed wavefunction → High retrieval accuracy                     │
│                                                                          │
│  AMBIGUOUS PROMPT ("that black cat from the 70s"):                      │
│                                                                          │
│    |ψ⟩ = α|Felix⟩ + β|Sylvester⟩ + γ|Salem⟩ + δ|random_cat⟩ + ...     │
│    Superposition → Measurement activates WRONG state                    │
│    → HALLUCINATION                                                       │
│                                                                          │
│  The prompt is the MEASUREMENT OPERATOR                                 │
│  Ambiguous prompts = weak measurements = spread probability            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.4.3 The Felix the cat problem

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  INTERNAL REPRESENTATION SPACE                                          │
│                                                                          │
│  Query: "black cat from the 70s"                                        │
│  K(query) ≈ low (few discriminating features)                           │
│                                                                          │
│                     ┌──────────────┐                                    │
│                     │   Felix      │  K(Felix) = medium                 │
│                     │   (1919)     │  Contains: era, studio, style      │
│                     └──────────────┘                                    │
│                           ↑                                              │
│         ┌─────────────────┼─────────────────┐                           │
│         │                 │                 │                            │
│    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐                        │
│    │Sylvester│      │ Ambiguous│      │ Salem   │                        │
│    │ (1945)  │      │  Query   │      │ (1996)  │                        │
│    └─────────┘      │ "black   │      └─────────┘                        │
│                     │ cat 70s" │                                          │
│                     └──────────┘                                          │
│                                                                          │
│  Problem: Query complexity << Representation complexity                 │
│  Multiple representations have similar "distance" to query              │
│  → Superposition → Wrong activation → Hallucination                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.4.4 Formal characterization

**Definition 6 (Representation Matching).**
Let $\mathcal{R} = \{r_1, r_2, ..., r_n\}$ be the set of compressed internal representations. A prompt $p$ activates representations based on structural similarity:

$$
\text{activation}(r_i | p) \propto \exp\left(-\frac{d_{\mathcal{M}}(p, r_i)^2}{2\sigma^2}\right) \tag{Proxy}
$$

where $d_{\mathcal{M}}(\cdot, \cdot)$ denotes distance in the universal embedding space $\mathcal{M}_{universal}$.

*Note on kernel choice:* We assume Gaussian activation kernels; other kernels (e.g., softmax over dot products as in attention) may change quantitative but not qualitative predictions. The key property, that activation decreases monotonically with distance, is shared across kernel families.

*Note on operationalization:* While the underlying quantity is related to Kolmogorov complexity $K(\cdot)$, which is uncomputable, the geometric distance $d_{\mathcal{M}}$ is operationally measurable via:
1. **Translation fidelity**: unsupervised embedding translation achieves >0.9 cosine similarity across model architectures (Jha et al., 2025), demonstrating the manifold is learnable
2. **CKA/Procrustes alignment**: measuring representational similarity across models (Kornblith et al., 2019)
3. **Compression proxies**: Normalized Compression Distance (NCD) on decoded text

**Theorem 2 (Geometric Matching Theorem).**
Let $\phi_{\mathcal{M}}(p)$ be the projection of prompt $p$ onto the universal manifold $\mathcal{M}_{universal}$, and $\phi_{\mathcal{M}}(r_i)$ be the projection of internal representation $r_i$. Reconstruction accuracy depends on the geometric alignment:

$$
P(\text{correct retrieval}) \propto \frac{\exp(-d_{\mathcal{M}}(\phi_{\mathcal{M}}(p), \phi_{\mathcal{M}}(r_{target})))}{\sum_j \exp(-d_{\mathcal{M}}(\phi_{\mathcal{M}}(p), \phi_{\mathcal{M}}(r_j)))} \tag{Proxy}
$$

where $d_{\mathcal{M}}(\cdot,\cdot)$ is distance in the universal embedding space. Operationally, this is the translation infidelity when mapping between representation spaces.

**Proposition 4 (Ambiguity-Induced Hallucination).**
When multiple representations have similar activation levels, the model enters a "superposition" state. The decoded output is a mixture that may not correspond to any single ground truth:

$$
\text{output} = \sum_i \text{activation}(r_i | p) \cdot \text{decode}(r_i) \tag{Proxy}
$$

This mixture can produce confident, fluent, but factually composite (hallucinatory) outputs.

#### 4.4.5 The Kolmogorov Sweetspot

Internal representations must balance compression and discriminability:

**Corollary (The Sweetspot).**
Internal representations exist in a Kolmogorov sweetspot:

$$
K_{optimal} = \arg\min_K \left[ \underbrace{E_{reconstruction}(K)}_{\text{too compressed}} + \underbrace{E_{storage}(K)}_{\text{too sparse}} \right] \tag{Def}
$$

- **Over-compressed** ($K < K_{optimal}$): Concepts collapse, distinctions lost
- **Under-compressed** ($K > K_{optimal}$): No useful abstraction, just memorization
- **Sweetspot**: Maximum discrimination with minimum description

This connects to Shannon's rate-distortion theory (Shannon, 1959):

$$
R(D) = \min_{p(\hat{x}|x): E[d(x,\hat{x})] \leq D} I(X; \hat{X}) \tag{Def}
$$

When compression rate $R$ is pushed too low, distortion $D$ increases; similar concepts merge, and retrieval becomes ambiguous.

#### 4.4.6 Implications for Mitigation

The matching view provides additional insight into why various techniques reduce hallucinations:

| Phenomenon | Channel Capacity View | Matching View |
|------------|----------------------|---------------|
| Specific prompts work better | More "signal" | Better query-key match |
| Similar concepts confused | Capacity spillover | Representation overlap |
| Few-shot helps | Adds capacity | Adds discriminating structure |
| RAG helps | External capacity | External disambiguation |
| CoT helps | Redundancy | Iterative refinement of match |

The matching view is **mechanistically grounded**—it maps to how transformers actually work: attention as soft matching, embeddings as compressed representations.

### 4.5 Context Window as Decompression Buffer

Beyond matching, there is a third mechanism: the context window must provide sufficient **room for reconstruction**. Internal representations are compressed; generating output requires decompressing them into the latent working space.

#### 4.5.1 The Asymmetry Problem

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  INTERNAL STORAGE (Weights):                                            │
│    K(concept) = compressed, small                                        │
│    Efficient storage via learned abstractions                           │
│                                                                          │
│  RECONSTRUCTION (Context Window):                                        │
│    K(decompressed) = larger, needs ROOM                                 │
│    Must "unpack" the compressed representation                          │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  CONTEXT WINDOW                                               │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐  │       │
│  │  │   Query     │  │ Retrieved   │  │  DECOMPRESSION ROOM  │  │       │
│  │  │   Input     │  │  Context    │  │  (latent working     │  │       │
│  │  │             │  │             │  │   memory for         │  │       │
│  │  │             │  │             │  │   reconstruction)    │  │       │
│  │  └─────────────┘  └─────────────┘  └──────────────────────┘  │       │
│  │                                                               │       │
│  │  If no room for decompression → KOLMOGOROV GARBAGE           │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

The asymmetry is fundamental:

```
Storage:     K(concept) = 100 bits (compressed in weights)
Retrieval:   K(decompressed) = 1000 bits (needs room to unfold)
Context:     C = 500 bits available

Result: TRUNCATED RECONSTRUCTION → Kolmogorov Garbage
```

This is analogous to **asking someone to do long division in their head** when they need scratch paper. The answer may be simple, but the *process* needs working memory.

#### 4.5.2 Kolmogorov Garbage

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  CLASSICAL GARBAGE IN GARBAGE OUT:                                      │
│    Bad input → Bad output                                               │
│    (Noise propagation)                                                  │
│                                                                          │
│  KOLMOGOROV GARBAGE:                                                    │
│    Insufficient structure in → Insufficient structure out              │
│    OR: Insufficient ROOM to reconstruct → Partial/corrupted structure  │
│                                                                          │
│    K(input) too low     → Can't uniquely identify representation       │
│    K(context) too small → Can't fully decompress representation        │
│    K(output) truncated  → Fragments stitched together = hallucination  │
│                                                                          │
│  The garbage isn't random noise; it's STRUCTURAL FAILURE               │
│  Plausible fragments that don't cohere                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Kolmogorov garbage is distinct from random noise: it consists of **structurally valid fragments** that fail to cohere into a truthful whole. The model produces pieces that individually look correct but collectively hallucinate.

*Terminological note:* Kolmogorov garbage (decompression failure) differs from **form prior sampling** (thermalization, Sec. 8.5). Kolmogorov garbage occurs when you *have* the knowledge but lack room to reconstruct it; truncated process, fragmented output. Form prior sampling occurs when you *lack* the knowledge entirely; the system thermalizes to maximum entropy, producing fluent text with no content grounding. Both produce hallucinations, but through different mechanisms.

#### 4.5.3 Bidirectional Bandwidth: The Context-Decompression Trade-off

The context window is not a passive bucket; it is an active component of the model's cognitive machinery. It plays two opposing roles simultaneously:

```
                    ┌─────────────────┐
    Effective       │                 │────────▶ Output
    Query Q_eff ───▶│  LATENT SPACE   │         (K_out)
                    │  (finite room)  │
                    └────────┬────────┘
                             │
                  Decompression requires
                  working memory space
```

1.  **The Source (Focusing Lens):** The context *defines* the query. As established in Section 4.4, the effective query $Q_{eff}$ is the result of the prompt interacting with the context history. A rich context sharpens the query, narrowing the search space to the correct internal representation. Without context, a prompt like "do it" is meaningless high-entropy noise.

2.  **The Load (Space Consumer):** The context *occupies* the latent workspace. Every token in the context consumes attention bandwidth and working memory capacity. This leaves less room for the decompression process itself: the "scratchpad" needed to unfold the retrieved concept into a coherent answer.

**The Fundamental Tension**

This creates a critical optimization problem. You cannot simply "add more context" indefinitely to improve performance.

-   **Too Little Context (The Ambiguity Failure):** The prompt lacks sufficient grounding. The effective query $Q_{eff}$ is weak/ambiguous. The model activates the wrong representation or a superposition of many.
    *Result:* Hallucination via Matching Failure.

-   **Too Much Context (The Crowding Failure):** The context is rich and the query is sharp, BUT the "desk is cluttered." The model retrieves the correct concept but has no working memory left to unpack it. The reconstruction is truncated or fragmented.
    *Result:* Hallucination via Decompression Failure (Kolmogorov Garbage).

**The Sweet Spot**

Reliable teaching requires finding the **Goldilocks Zone** for context utilization. You need enough history to constrain the *what* (the topic), but enough free space to process the *how* (the generation).

$$ K_{Q_{eff}} + K_{reconstruct} \leq K_{latent\_capacity} $$

#### 4.5.4 Formal Characterization

**Definition 7 (Decompression Room).**
Let $K_{latent}$ be the effective capacity of the latent space (constrained by context window and attention bandwidth). Let $K_{query}$ be the Kolmogorov complexity of the query, $K_{context}$ be the complexity of provided context, and $K_{reconstruct}(r)$ be the complexity required to decompress internal representation $r$.

Successful reconstruction requires:

$$
K_{query} + K_{context} + K_{reconstruct}(r) \leq K_{latent} \tag{Def}
$$

*Note on subadditivity:* Kolmogorov complexity is uncomputable and to be consistent with our description herein not strictly additive: $K(A,B) \leq K(A) + K(B) + O(\log n)$ where the logarithmic term accounts for combining descriptions. The sum $K_{query} + K_{context} + K_{reconstruct}$ is thus an upper bound on the joint complexity $K(\text{query}, \text{context}, \text{reconstruction})$. For our purposes, this upper bound is the operationally relevant constraint: if even the upper bound exceeds capacity, reconstruction certainly fails.

**Proposition 5 (Context Crowding).**
When context is over-filled, decompression room decreases:

$$
K_{available} = K_{latent} - K_{query} - K_{context} \tag{Def}
$$

If $K_{available} < K_{reconstruct}(r)$, reconstruction is truncated, producing structurally coherent but semantically fragmented outputs or unwanted noise that can effect in a cascading failure all the next outputs.

**Proposition 6 (Decompression-Compression Asymmetry).**
For most concepts, decompression complexity exceeds storage complexity:

$$
K_{reconstruct}(r) > K_{storage}(r) \tag{Approx}
$$

*Why this asymmetry exists:* Storage encodes a concept in its most compressed form: the "program." Reconstruction requires executing that program, which needs intermediate state, working variables, and expansion room. Analogies:
- **Computation**: A program file is small; running it requires much more RAM for stack, heap, and intermediate results
- **Compression**: A .zip file is small; decompressing it requires buffer space exceeding the compressed size
- **Mathematics**: The statement "$e^{i\pi} + 1 = 0$" is short; deriving it requires pages of working

The same applies to LLMs: the concept "French Revolution" is stored compactly, but generating a coherent explanation requires unfolding dates, figures, causes, and consequences; all simultaneously active in the latent space.

This asymmetry means that context windows must be sized not for storage, but for the working memory required during generation.

#### 4.5.5 Implications

This explains several observed phenomena:

| Phenomenon | Decompression View |
|------------|-------------------|
| Long context degrades quality | Less room for reconstruction |
| "Lost in the middle" effect (Liu et al., 2023) | Middle context crowds decompression space |
| Simple prompts work better on complex topics | More room for complex reconstruction |
| RAG can hurt when over-filled | Context crowds out working memory |
| Chain-of-thought helps | Distributes decompression across steps |

#### 4.5.6 The Trifecta of Hallucination Mechanisms

We now have three complementary mechanisms:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THREE HALLUCINATION MECHANISMS                                         │
│                                                                          │
│  1. CAPACITY VIOLATION (Section 3)                                      │
│     Knowledge not stored in weights                                     │
│     → Nothing to retrieve                                               │
│     → Maximum entropy output                                            │
│                                                                          │
│  2. MATCHING FAILURE (Section 4.4)                                      │
│     Knowledge stored but query is ambiguous                             │
│     → Multiple representations activated                                │
│     → Wrong or composite retrieval                                      │
│                                                                          │
│  3. DECOMPRESSION FAILURE (Section 4.5)                                 │
│     Knowledge stored AND correctly matched                              │
│     → Insufficient room to reconstruct                                  │
│     → Truncated/fragmented output = Kolmogorov garbage                 │
│                                                                          │
│  All three produce fluent, confident, but incorrect output.            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.6 Attention Sinks and Anchoring 

We now integrate the graph-diffusion perspective of attention sinks (Pappone, 2025) into the decompression model. This reveals that "context crowding" is often a structural failure of attention allocation rather than purely a token-count limit.

#### 4.6.1 The Mechanism of Sinks

Causal attention creates a directed acyclic graph (DAG) where repeated composition pushes probability mass "leftward" to early tokens. This creates **attention sinks**—tokens that accumulate disproportionate attention independent of their semantic value.

**Definition 8 (Sink Severity).**
The sink severity $s(L)$ is the fraction of total attention mass concentrated in the first $k$ tokens (the prefix) across all heads and layers $L$:

$$
s(L) = \frac{1}{H \cdot L} \sum_{h, \ell} \frac{\sum_{i>k, j \le k} A^{(\ell, h)}_{i \to j}}{\sum_{i>k, j} A^{(\ell, h)}_{i \to j}} \tag{Def}
$$

where $A$ is the attention matrix with rows normalized to sum to 1 (i.e., $\sum_j A_{i \to j} = 1$ for each query position $i$), as is standard after softmax.

#### 4.6.2 Impact on Capacity

Sinks effectively reduce the "bandwidth" available for late-context tokens. If the model attends primarily to the BOS token or system prompt (the sink), it cannot effectively "read" retrieved context or recent tokens.

**Proposition 7 (Sink-Limited Capacity).**
The effective context capacity $C_{ctx}$ is monotonically decreasing with sink severity $s$:

$$
\frac{\partial C_{ctx}}{\partial s} \le 0 \tag{Approx}
$$

As $s \to 1$, the channel becomes dominated by the prefix, causing reconstruction failure (Kolmogorov garbage) even if relevant information is present in the context.

#### 4.6.3 The Thermodynamic Consequence

Sinks create deep potential energy wells at the start of the sequence.

- **Aligned Sinks**: If the sink tokens are *semantic anchors* (e.g., strong constraints, entity definitions), the well aligns with the truth manifold ($\mathcal{M}_{truth}$), lowering free energy $F$ for grounded states.
- **Misaligned Sinks**: If sinks are generic (e.g., "The", BOS), the well is misaligned. Escaping this local minimum requires higher temperature $T$, which increases the entropy term $-TS$ and drives the system toward the form prior.

**Prediction 21 (Position Primacy).**
Tasks requiring late-context evidence degrade as sink severity $s$ increases. Periodic repetition of anchors (counter-diffusion) is required to maintain effective capacity.

### 4.7 Information Atoms: Grounding the Framework

The abstract Kolmogorov framework, $K(\text{output}) \leq K(\text{weights}) + K(\text{context})$, can be mechanistically grounded by viewing weights as compressed storage of training sequences. Following nested learning interpretations, we can decompose model weights into **information atoms**: the irreducible units of knowledge extracted from training.

#### 4.7.1 Weights as Sequence Memory

**Definition 9 (Information Atom).**
An information atom $a_i$ is a compressed pattern learned from training sequence(s) $s_i$:

$$
a_i = \text{compress}(\{s_j : s_j \text{ contains pattern } i\})
$$

The model weights encode a superposition of atoms:

$$
W^{(\ell)} \approx \sum_i \alpha_i \cdot \text{encode}(a_i) \tag{Approx}
$$

where $\alpha_i$ reflects frequency and importance weighting from training.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  NESTED LEARNING VIEW: WEIGHTS AS SEQUENCE MEMORY                       │
│                                                                          │
│  Training sequences {s₁, s₂, ..., sₙ} are "information atoms"          │
│                                                                          │
│  Layer ℓ weights W^(ℓ) encode:                                          │
│    W^(ℓ) ≈ ∑ᵢ αᵢ · compress(patterns at level ℓ from sᵢ)              │
│                                                                          │
│  Inference = pattern matching + decompression:                          │
│    query → activates subset of atoms → reconstructs from those atoms   │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  HALLUCINATION REDEFINED:                                               │
│                                                                          │
│  Valid output: derivable from activated atom combinations               │
│    output ∈ span{decompress(aᵢ) : aᵢ activated}                        │
│                                                                          │
│  Hallucination: output contains information from NO atom                │
│    output ∉ span{any atom combination}                                  │
│    = "created" information = form prior filling the gap                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.7.2 Grounding the Theorems

This atomic view grounds our abstract claims in concrete, traceable training data:

| **Abstract Claim** | **Atom-Grounded Version** |
|--------------------|---------------------------|
| "Model has capacity $C_T$ for topic $T$" | Model has $N$ atoms covering $T$ with total information $C_T = \sum_{a_i \text{ covers } T} I(a_i; T)$ |
| "Hallucination = information creation" | Hallucination = output not in span of activated atoms |
| "Matching failure" | Wrong atoms activated by query |
| "Thermalization to form prior" | No atoms strongly activated → default to statistical structure |

**Corollary (Atom-Grounded Conservation).**
*Corollary to Theorem 3.* Output information cannot exceed the information content of activated atoms plus context:

$$
K(\text{output}) \leq \sum_{i \in \text{activated}} K(a_i) + K(\text{context}) \tag{Approx}
$$

Any output information not traceable to activated atoms was hallucinated from the form prior.

#### 4.7.3 Atom Tracing for Hallucination Detection

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ATOM-TRACING TEST FOR HALLUCINATION                                    │
│                                                                          │
│  1. Given output O and query Q:                                         │
│     - Extract activated features/circuits (mechanistic interp)         │
│     - These correspond to "atoms" from training                         │
│                                                                          │
│  2. Compute atom coverage:                                              │
│     coverage(O) = max_{atom subset} sim(O, decompress(atoms))          │
│                                                                          │
│  3. Hallucination score:                                                │
│     H(O) = K(O) - K(O | activated atoms)                               │
│          = information in O NOT explained by any atom                  │
│                                                                          │
│  Prediction: H(O) correlates with factual error rate                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.7.4 Connection to Universal Representations

The Platonic Representation Hypothesis (Huh et al., 2024) (Sec. 11.5) states that representations converge across models trained on similar data. The atom view explains why:

**Same training distribution → same atoms (approximately), which leads to same representation space**

The atoms ARE the universal basis. Different models compress them into slightly different geometric arrangements, but the underlying atomic structure is shared. This is why unsupervised translation between model embedding spaces achieves high fidelity; the spaces encode the same atoms.

#### 4.7.5 Practical Implications

| Method | Atom Interpretation |
|--------|---------------------|
| SAE features (Anthropic) | Learned atoms with interpretable semantics |
| Probing classifiers | Testing for presence of specific atoms |
| Activation patching | Identifying which atoms contribute to output |
| Representation engineering | Steering by activating/suppressing atoms |

**Prediction 22 (Atom Coverage).**
Hallucination rate correlates inversely with atom coverage: outputs where fewer training-derived atoms are strongly activated have higher factual error rates.

#### 4.7.6 Test-Time Atom Creation

The atom framework as presented assumes atoms are fixed at training time; the model can only decompress what it previously compressed. However, recent architectural innovations (Titans; Behrouz et al., 2025) demonstrate that **atoms can be created at inference time** through test-time learning.

**Definition 10 (Test-Time Atom).**
A test-time atom $a^{test}_j$ is a compressed pattern learned during inference from the current context:

$$
a^{test}_j = \text{compress}(\text{context patterns during inference})
$$

The effective atom set becomes:

$$
\mathcal{A}_{effective} = \mathcal{A}_{training} \cup \mathcal{A}_{test-time}(context)
$$

**Extended Conservation Law.**
With test-time learning, the information conservation bound extends:

$$
K(\text{output}) \leq \sum_{i \in \text{activated}} K(a_i) + \sum_{j \in \text{test-learned}} K(a^{test}_j) + K(\text{context}) \tag{Approx}
$$

This is significant: test-time atom creation **extends effective capacity beyond pre-training**. A model encountering a topic not well-covered in training can learn new atoms from rich context (e.g., RAG-retrieved documents), reducing the capacity gap that would otherwise cause hallucination.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  STATIC vs DYNAMIC ATOMS                                                │
│                                                                          │
│  STATIC ATOMS (Standard LLM):                                           │
│    - Created at training time only                                     │
│    - Fixed capacity C_T per topic                                       │
│    - Hallucination when R_T > C_T (no recourse)                        │
│                                                                          │
│  DYNAMIC ATOMS (Test-Time Learning):                                    │
│    - Created at training + inference time                              │
│    - Capacity = C_T(training) + ΔC_T(context)                          │
│    - Context can fill capacity gaps                                     │
│    - Hallucination reduced when rich context available                 │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  Implication: RAG is not just "external memory"—it enables             │
│  TEST-TIME ATOM CREATION when paired with learning mechanisms          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Proposition 8 (Test-Time Learning as Capacity Extension).**
Let $C_T^{static}$ be the pre-training capacity for topic $T$. With test-time learning from context $ctx$:

$$
C_T^{effective} = C_T^{static} + \Delta C_T(ctx)
$$

where $\Delta C_T(ctx) \leq I(ctx; T)$—the capacity extension is bounded by the mutual information between context and topic. Rich, relevant context enables larger extensions; irrelevant context provides no benefit.

This explains why RAG helps even when the model "knows" something: the retrieved context enables creation of topic-specific test-time atoms that supplement (and can correct) the static atoms from training.

---

## 5. The Compression-Transmission Duality

### 5.1 Learning vs. Inference

Our framework reveals a fundamental duality:

| Phase | Operation | Information Direction | Goal |
|-------|-----------|----------------------|------|
| **Training** | Compression | World → Weights | Minimize description length |
| **Inference** | Transmission | Query → Output | Maximize reliable throughput |

This duality has profound implications:

**Proposition 9 (Compression-Transmission Trade-off).**  
Aggressive compression during training reduces capacity for out-of-distribution transmission during inference. There exists a Pareto frontier between compression efficiency and transmission reliability.

### 5.2 LLMs Are Teachers: The Core Insight

**LLMs are not just generators; they are teachers.** This reframing is essential to understanding hallucination.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE TEACHING FRAMEWORK                                                 │
│                                                                          │
│  LEARNING (Training = Source Coding):                                   │
│    Many instances ──compress──▶ Concept (short description in weights) │
│    The model LEARNS by compressing the world                           │
│                                                                          │
│  TEACHING (Inference = Channel Coding):                                 │
│    Query ──▶ Build representation ──▶ Transmit knowledge ──▶ Output   │
│    The model TEACHES by decompressing and transmitting                 │
│                                                                          │
│  Why redundancy in teaching? Because the channel is NOISY.             │
│  The "student" (output space) needs error-correction.                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 5.2.1 Building the In-Context Representation

**Before the model can teach, it must reconstruct the knowledge internally.** This is the critical step where hallucination originates:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE RECONSTRUCTION REQUIREMENT                                         │
│                                                                          │
│  Query: "What is the capital of France?"                               │
│                                                                          │
│  Step 1: MATCH query to internal representation                        │
│    Query structure → Activate relevant compressed knowledge            │
│    If match fails → Wrong representation activated → Hallucination     │
│                                                                          │
│  Step 2: DECOMPRESS the representation                                  │
│    Compressed knowledge → Unfold into working context                  │
│    If no room → Truncated reconstruction → Kolmogorov garbage          │
│                                                                          │
│  Step 3: TRANSMIT to output                                            │
│    Reconstructed knowledge → Generate answer with redundancy           │
│    If distortion accumulated → Signal degraded → Hallucination         │
│                                                                          │
│  The model must BUILD THE CORRECT IN-CONTEXT REPRESENTATION            │
│  before it can teach. This is where all failures originate.           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 5.2.2 Why Teachers Need Redundancy

When you teach, you don't just state a fact once; you explain, give examples, rephrase, and verify understanding. This redundancy is **error-correction coding**:

- **Chain-of-thought** = Showing your work = Redundant paths to the answer
- **Examples in context** = Multiple instances = Redundant encoding
- **Self-consistency** = Multiple explanations = Voting over redundant samples

A teacher who doesn't know the material (low capacity) cannot provide reliable redundancy; they fill the gaps with plausible-sounding nonsense. This is exactly what hallucinating LLMs do.

#### 5.2.3 The Fundamental Asymmetry

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  LEARNING vs TEACHING ASYMMETRY                                         │
│                                                                          │
│  LEARNING:                                                              │
│    - Has access to ground truth                                        │
│    - Can iterate and correct                                            │
│    - Compresses gradually over many examples                           │
│    - Errors during learning → Fixed by more training                   │
│                                                                          │
│  TEACHING (Inference):                                                  │
│    - No access to ground truth during generation                       │
│    - Single forward pass, no iteration                                 │
│    - Must decompress on-the-fly                                        │
│    - Errors during teaching → HALLUCINATION                            │
│                                                                          │
│  This is why inference is fragile: one-shot reconstruction             │
│  without the safety net of ground truth.                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**An LLM doing inference is teaching.** When it lacks knowledge (low capacity), can't match the query to the right representation (matching failure), has no room to decompress (context crowding), or accumulates errors (distortion), it cannot teach reliably; it hallucinates.

### 5.3 The Capacity of a Mind

Extending the framework:

$$
C_{mind} = \max I(\text{Experience}; \text{Understanding}) \tag{Conj}
$$

This capacity is limited by:
- Number of parameters (channel width)
- Precision of weights (noise floor)
- Interference between concepts (crosstalk)
- Attention bandwidth (serial bottleneck)

---

## 6. Hallucination Taxonomy

### 6.1 Types of Capacity Violations

| Hallucination Type | Capacity Violation | Example |
|--------------------|-------------------|---------|
| **Factual** | Knowledge capacity for specific facts | "The Eiffel Tower was built in 1920" |
| **Logical** | Reasoning capacity | Contradictory statements |
| **Temporal** | Capacity for time-sensitive info | Outdated information as current |
| **Attribution** | Capacity for source tracking | Fabricated citations |
| **Contextual** | Capacity for context integration | Ignoring conversation history |
| **Numerical** | Capacity for precise computation | Arithmetic errors |

### 6.2 Severity and Detection

**Proposition 10 (Detectability).**  
Hallucinations are detectable to the extent that content constraints can be externally verified. Unverifiable claims are undetectable hallucinations.

This suggests:
- Factual claims → Checkable against knowledge bases
- Logical claims → Checkable via formal verification
- Stylistic claims → Harder to verify (subjective)

---

## 7. Mitigation Strategies

### 7.1 Capacity Enhancement

Strategies that increase effective capacity:

| Strategy | Mechanism | Capacity Effect | Primary failure mode addressed |
|----------|-----------|-----------------|--------------------------------|
| **Unambiguous Prompts** | Matches internal representations better | Reduces noise, increases accuracy | Matching failure (Sec. 4.4) |
| **Larger models** | More parameters | Higher base capacity | Capacity violation (Sec. 3) |
| **Better training data** | Stronger content constraints | Higher topic-specific capacity | Capacity violation (Sec. 3) |
| **RAG** | External retrieval | Capacity from external sources | Capacity violation (Sec. 3) |
| **Fine-tuning** | Topic specialization | Higher capacity for specific domains | Capacity violation (Sec. 3) |
| **Tool use** | Offload to reliable tools | Infinite capacity for tool-solvable problems | Capacity violation (Sec. 3) |

### 7.2 Constraint Injection

Strategies that add constraints at inference time:

| Strategy | Constraints Added | Implementation | Primary failure mode addressed |
|----------|-------------------|----------------|--------------------------------|
| **Few-shot examples** | Content constraints | Examples in prompt | Capacity violation (Sec. 3) |
| **Chain-of-thought** | Reasoning constraints | "Let's think step by step" | Decompression failure (Sec. 4.5) |
| **Self-consistency** | Consistency constraints | Sample and vote | Geometric distortion (Sec. 8.4) |
| **Constitutional AI** | Value constraints | Principles in system prompt | Thermalization to form prior (Sec. 8.5) |
| **Grounding** | Factual constraints | Require citations | Capacity violation (Sec. 3) |
| **Unambiguous prompts** | Matching constraints | Use named entities, dates, identifiers; precise referents | Matching failure (Sec. 4.4) |

### 7.3 Capacity-Aware Generation

**Algorithm 1: Capacity-Aware Decoding**

```python
def capacity_aware_generate(query, model, threshold=0.8):
    """
    Generate only when estimated capacity exceeds threshold.
    Otherwise, express uncertainty or retrieve external information.
    """
    # Estimate topic-specific capacity
    capacity_estimate = estimate_capacity(query, model)
    
    if capacity_estimate < threshold:
        # Below capacity - high hallucination risk
        if can_retrieve_external_info(query):
            # Increase capacity via RAG
            context = retrieve(query)
            return generate_with_context(query, context, model)
        else:
            # Express uncertainty
            return express_uncertainty(query, capacity_estimate)
    else:
        # Within capacity - safe to generate
        return generate(query, model)
```

### 7.4 Capacity Estimation via the Universal Manifold

The framework provides a **theoretical path to capacity estimation** through the universal manifold $\mathcal{M}_{universal}$ (Sec. 11.5). The key insight: capacity is inversely related to geometric distance from the manifold.

**Definition 11 (Manifold-Based Capacity Estimator).**
For a query $q$ about topic $T$, the estimated capacity is:

$$
\hat{C}_T(q) = f\left( \rho_T, \; d_{\mathcal{M}}(\phi(q), \mathcal{M}_T), \; \text{conf}(q) \right) \tag{Proxy}
$$

where:
- $\rho_T$ = **embedding density** around topic $T$ in the model's representation space (higher density → more training data → higher capacity)
- $d_{\mathcal{M}}(\phi(q), \mathcal{M}_T)$ = **geometric distance** from the query embedding to the topic submanifold (smaller distance → better match → higher effective capacity)
- $\text{conf}(q)$ = **calibrated confidence** from the model's own uncertainty estimates
- $f$ = monotonically increasing in $\rho_T$ and $\text{conf}$, decreasing in $d_{\mathcal{M}}$

**Operationalization Approaches:**

| Method | What It Measures | Capacity Proxy |
|--------|------------------|----------------|
| **Embedding density** | Local density of training representations around query | High density → high $C_T$ |
| **Translation fidelity** | How well query embedding translates across model architectures (vec2vec) | High fidelity → on-manifold → high $C_T$ |
| **Probing accuracy** | Model's accuracy on held-out facts about $T$ | High accuracy → high $C_T$ |
| **Entropy of next-token distribution** | Uncertainty in generation | Low entropy → high $C_T$ |
| **Self-consistency variance** | Agreement across multiple samples | Low variance → high $C_T$ |

**Algorithm 2: Manifold-Based Capacity Estimation**

```python
def estimate_capacity(query, model, reference_manifold):
    """
    Estimate topic-specific capacity via manifold alignment.
    
    Returns: capacity_score in [0, 1]
    """
    # 1. Embed the query
    query_embedding = model.encode(query)
    
    # 2. Measure embedding density (topic familiarity)
    density = compute_local_density(query_embedding, model.embedding_space)
    
    # 3. Measure manifold alignment (translation fidelity proxy)
    alignment = measure_manifold_distance(query_embedding, reference_manifold)
    
    # 4. Get calibrated model confidence
    confidence = model.calibrated_confidence(query)
    
    # 5. Combine into capacity estimate
    # Higher density, lower distance, higher confidence → higher capacity
    capacity_score = combine_signals(
        density_score=normalize(density),
        alignment_score=1 - normalize(alignment),  # Invert: low distance = high score
        confidence_score=confidence
    )
    
    return capacity_score
```

*Note:* This provides a **theoretical operationalization**; we know *what* to measure. Developing and validating practical estimators at scale remains future work (Sec. 11.4). The key advance is that the universal manifold hypothesis (empirically supported; Jha et al., 2025) transforms capacity estimation from an abstract information-theoretic quantity to a **geometric measurement problem**.

Addresses: Capacity violation (Sec. 3); secondarily reduces geometric distortion (Sec. 8.4) by refusing generation when fidelity is low.

### 7.5 Verification-First and Reverse Reasoning

Recent work (Wu & Yao, 2025) demonstrates that asking LLMs to "verify first," even against a random or wrong answer, significantly improves reasoning accuracy. This phenomenon is fully explained by our framework:

1.  **Reverse Reasoning breaks Geometric Distortion**: Forward reasoning ($A \to B \to C$) accumulates error geometrically (Theorem 4). Verification is a reverse check ($C \to? A$). If the forward path has drifted off the Truth Manifold, the reverse path is unlikely to map back to the origin, exposing the hallucination.
2.  **Verification is Compression, not Generation**: Verifying a candidate answer requires less channel capacity than generating it from scratch. It is a discrimination task (low entropy) rather than a generation task (high entropy).
3.  **Random Answers as Stochastic Resonance**: The finding that even *random* answers improve performance validates our **Optimal Noise Principle** (Theorem 6). A random answer provides a "structural scaffold" (form constraint) that forces the model to engage its critical faculties. It kicks the system out of the local minimum of "fluent but wrong" (form prior) and forces it to traverse the energy landscape to verify the candidate.

**Algorithm 3: Verification-First Generation**

```python
def verify_then_generate(query, model):
    """
    Leverages reverse reasoning and stochastic resonance.
    """
    # 1. Propose (or sample random) candidate
    # High temp/randomness provides the 'noise' for stochastic resonance
    candidate = model.generate(query, temperature=1.0) 
    
    # 2. Verify (Reverse Reasoning)
    # Checks consistency: Does Candidate implies Query?
    verification = model.generate(
        f"Question: {query}\nProposed Answer: {candidate}\nIs this correct? explain."
    )
    
    # 3. Final Generation (conditioned on verification)
    final_answer = model.generate(
        f"Question: {query}\nAnalysis: {verification}\nTherefore, the correct answer is:"
    )
    return final_answer
```

---

## 8. Connection to Complexity from Constraints

### 8.1 The Homeostat Principle

The Neuro-Symbolic Homeostat framework (Goldman, 2025) establishes:

> "Complexity comes from constraints. Without constraints, you have maximum entropy (noise). With constraints, you get structure."

Hallucinations are the manifestation of this principle in generative models:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COMPLEXITY FROM CONSTRAINTS                                            │
│                                                                         │
│  Strong constraints (good capacity):                                    │
│    Query ──▶ [Constrained generation] ──▶ Structured, accurate output  │
│                                                                         │
│  Weak constraints (poor capacity):                                      │
│    Query ──▶ [Unconstrained generation] ──▶ Entropy = Hallucination    │
│                                                                         │
│  Hallucination = Generation in the UNCONSTRAINED REGION                 │
│                 of the model's knowledge space                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 The Form-Content Asymmetry

LLMs learn form constraints from ALL text, but content constraints only from text about specific topics:

$$
|\mathcal{F}| \gg |\mathcal{C}_T| \quad \text{for rare topics } T \tag{Approx}
$$

This asymmetry is the root cause of hallucinations:
- Form is globally consistent (the model "knows how to write")
- Content is locally specific (the model may not "know what is true")

### 8.3 The Conservation of Information

A fundamental principle of information theory, the Data Processing Inequality, establishes that processing cannot increase the information content of a signal. For language generation, we can state this limit explicitly:

#### 8.3.0 Intuition: The Library Paradox

You cannot take a 100-page book and summarize it into a 200-page book without making things up.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE CONSERVATION OF INFORMATION (Data Processing Inequality)           │
│                                                                          │
│  Analogy: The Library                                                   │
│                                                                          │
│  1. The Source (Library): Contains 1,000 facts about Roman History.     │
│  2. The Task (Generation): Write a book containing 2,000 facts.         │
│                                                                          │
│  IMPOSSIBLE.                                                            │
│                                                                          │
│  Where do the extra 1,000 facts come from?                              │
│  - They cannot come from the source (it's empty).                       │
│  - They MUST be fabricated.                                             │
│                                                                          │
│  Rule: Output Info ≤ Source Info (Weights + Context)                    │
│                                                                          │
│  If you ask an LLM for "10 citations about X" and it only knows 3,      │
│  it MUST hallucinate 7 to satisfy the form constraint of the request.   │
│                                                                          │
│  Hallucination is the mathematical necessity of satisfying a form       │
│  request that exceeds the available content budget.                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Information cannot be created, only transmitted or lost.** You cannot output more information about a topic than was stored or provided. Any excess must come from the form prior, and that excess is hallucination.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE INFORMATION CONSERVATION LIMIT                                     │
│                                                                          │
│  Conservation Law: K(output) ≤ K(weights) + K(context)                 │
│                                                                          │
│  You cannot output more information than was stored OR provided.       │
│  Information is conserved through the cycle.                           │
│                                                                          │
│  TRAINING:   K(world) ──compress──▶ K(weights)                         │
│  MATCHING:   K(query) + K(context) ──▶ K(retrieved)                    │
│  LIMIT:      K(retrieved) ≤ K(weights) + K(context)                    │
│  INFERENCE:  K(retrieved) ──decompress──▶ K(output)                    │
│                                                                          │
│  CHAIN: K(output) ≤ K(retrieved) ≤ K(weights) + K(context)             │
│                                                                          │
│  VIOLATION = HALLUCINATION                                              │
│  When K(output) > K(source), information was "created"                 │
│  That extra structure comes from the PRIOR (form), not KNOWLEDGE       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Operational convention (proxies).** In all uses of the conservation law here, $K(\text{weights})$ abbreviates the topic‑conditioned representation‑capacity proxy $K_{\text{rep}}(\text{weights} \mid T)$ defined via manifold geometry (Sec. 7.4). $K(\text{context})$ counts the in‑context constraints and reconstruction structure provided by the prompt/RAG/system prompt. Both quantities are computable proxies; inequalities using $K(\cdot)$ hold up to monotone rescalings induced by the chosen proxies.

**Theorem 3 (Information Conservation / Data Processing Limit).**

Let $S = S_{weights} + S_{context}$ be the total available information source (weights + context). The key quantity is the **conditional entropy** $H(O \mid S, T)$, which is the entropy of the output that remains unexplained after conditioning on both the source and the topic.

For truthful generation:

$$
H(O \mid S, T) = 0 \tag{Def}
$$

The output is fully determined by the source; there is no unexplained entropy. All structure in the output traces back to stored knowledge or provided context.

For hallucination:

$$
H(O \mid S, T) > 0 \tag{Def}
$$

The output contains entropy not explained by the source. This unexplained entropy, sampled from the form prior rather than grounded in knowledge, is the hallucinated component.

Equivalently, via mutual information: $I(S; O \mid T) = H(O \mid T) - H(O \mid S, T)$. For truthful generation, $I(S; O \mid T) = H(O \mid T)$ (source explains all topic-relevant output entropy). For hallucination, $I(S; O \mid T) < H(O \mid T)$ (a gap exists; the form prior filled it).

Equivalently, in complexity terms (weights + context as the source):

$$
K(\text{output}) \;\le\; K(\text{weights}) + K(\text{context}) + O(\log n) \tag{Proxy}
$$

where the $O(\log n)$ term accounts for the overhead of combining two descriptions (see Kolmogorov, 1965). Since Kolmogorov complexities are not strictly additive, i.e. $K(A,B) \le K(A) + K(B) + O(\log(K(A) + K(B)))$, this inequality holds up to logarithmic factors. For our purposes, these constants are negligible compared to the main terms, and we write the simplified form in subsequent equations.

**Proof sketch.** Consider the Markov chain $S \to R \to O$, where $S$ is the available source (weights + context), $R$ any intermediate reconstruction, and $O$ the output. By the data processing inequality, $I(S;O) \le I(S;R)$. In the idealized truthful case, the output is a deterministic function of the source given the topic: $H(O \mid S, T) = 0$. When the source is insufficient or incorrectly reconstructed, $H(O \mid S, T) > 0$, meaning the output contains entropy unexplained by the source. This unexplained entropy must come from somewhere; in LLMs, it is sampled from the form prior (the distribution over fluent text). The gap $H(O \mid S, T)$ quantifies the hallucinated component (see Cover & Thomas, 2006, Ch. 2).

**Corollary (Information Accounting).**

The output decomposes as:

$$
K(\text{output}) = \underbrace{K(\text{from weights} + \text{context})}_{\text{grounded}} + \underbrace{K(\text{from form prior})}_{\text{hallucinated filler}} \tag{Proxy}
$$

For truthful generation, the second term is zero. Any contribution from the form prior that isn't constrained by content knowledge violates the conservation law.

#### 8.3.1 The Hallucination Detector

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  HALLUCINATION DETECTOR (Information Accounting)                        │
│                                                                          │
│  Measure: K(output) vs K(source)                                        │
│                                                                          │
│  IF K(output) ≤ K(source):                                              │
│     Information could have been transmitted                            │
│     → Possibly grounded (check accuracy separately)                    │
│                                                                          │
│  IF K(output) > K(source):                                              │
│     Information was CREATED                                            │
│     → DEFINITELY hallucination                                          │
│     → Excess structure came from form prior, not knowledge             │
│                                                                          │
│  This is like energy accounting:                                        │
│     If energy_out > energy_in, something is wrong                      │
│     If information_out > information_source, hallucination             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.3.2 Why This Is The Conserved Quantity

The form constraints are **high entropy**; they permit many possible outputs that all "look right" linguistically. The content constraints **reduce entropy**; they select the unique correct answer from among the form-valid possibilities.

When generation occurs without sufficient content constraints (due to capacity violation, matching failure, or decompression failure), the model samples from the high-entropy form distribution. That sample contains more information than was stored about the topic; the excess is necessarily hallucinated.

This is precisely analogous to energy conservation in physics:
- **Energy**: You cannot extract more energy from a system than was put in
- **Information**: You cannot output more information about a topic than was stored or provided

The principle being preserved is **truth-preservation under the compression-transmission-decompression cycle**. When this limit is breached (more structure out than in), the conservation law is violated, and we have detectable hallucination.

#### 8.3.3 Practical Implications

This principle enables:
- **Real-time hallucination detection**: Estimate $K(\text{output})$ vs $K(\text{source})$
- **Guaranteed-truthful generation**: Only generate when information budget is satisfied
- **Formal verification**: Prove outputs don't exceed source information bounds
- **Calibrated uncertainty**: Confidence should track $K(\text{source}) / K(\text{output})$

### 8.4 Geometric Distortion Accumulation

While the conservation law tells us information cannot be created, it can be **lost or corrupted** at each stage. Crucially, this corruption is **geometric**; errors compound multiplicatively, not additively.

#### 8.4.0 Intuition: The Telephone Game

This mechanism is best understood via the children's game **Telephone** (or Chinese Whispers).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE TELEPHONE GAME (Geometric Error Accumulation)                      │
│                                                                          │
│  Original Message: "The purple elephant danced at midnight."           │
│                                                                          │
│  Stage 1 (Compression/Training):                                        │
│     Error ε₁ = 10% (small detail lost)                                  │
│     Message: "The purple elephant danced at night."                    │
│                                                                          │
│  Stage 2 (Retrieval/Matching):                                          │
│     Error ε₂ = 10% (on top of ε₁)                                       │
│     Message: "The purple elephant danced tonight."                     │
│                                                                          │
│  Stage 3 (Decompression/Generation):                                    │
│     Error ε₃ = 10% (on top of ε₁ and ε₂)                                │
│     Message: "The purple elephant is dancing tonight."                 │
│                                                                          │
│  The errors MULTIPLY. Fidelity = 0.9 × 0.9 × 0.9 = 0.729 (73%)         │
│  With 10 stages, fidelity drops to 0.9¹⁰ = 35%                         │
│                                                                          │
│  Key insight: You cannot fix the message by shouting (prompting)        │
│  at Stage 3 if the meaning was lost at Stage 1.                         │
│                                                                          │
│  Multi-hop reasoning (Step 1 → Step 2 → Step 3) is a Telephone Game.    │
│  The probability of hallucination grows with every hop.                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.4.1 The Distortion Cascade

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE DISTORTION CASCADE                                                 │
│                                                                          │
│  World ──ε₁──▶ Weights ──ε₂──▶ Retrieved ──ε₃──▶ Output                │
│        compress      match         decompress                           │
│                                                                          │
│  Each stage introduces distortion εᵢ                                    │
│  Total fidelity = (1-ε₁)(1-ε₂)(1-ε₃) ← MULTIPLICATIVE                  │
│                                                                          │
│  THE KEY INSIGHT: Fidelity decays EXPONENTIALLY with chain length n    │
│                                                                          │
│  For uniform ε per stage:                                               │
│    n=3 stages, ε=0.1:  (0.9)³  = 73% fidelity                          │
│    n=5 stages, ε=0.1:  (0.9)⁵  = 59% fidelity                          │
│    n=10 stages, ε=0.1: (0.9)¹⁰ = 35% fidelity                          │
│    n=20 stages, ε=0.1: (0.9)²⁰ = 12% fidelity                          │
│                                                                          │
│  Even small per-stage errors (10%) compound to severe total loss.      │
│  This is why long reasoning chains and multi-hop retrieval degrade.    │
│                                                                          │
│  Note: The linear approximation (1 - nε) is a pessimistic LOWER BOUND  │
│  on fidelity. The geometric formula is exact for independent errors.   │
│                                                                          │
│  With correlated errors (aligned off-manifold):                        │
│    Geometric cascade can be MUCH WORSE than independent case           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.4.2 The Truth Manifold

Representations exist on a curved manifold of valid knowledge. Errors push representations off this manifold into hallucination space:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  REPRESENTATIONS LIVE ON CURVED MANIFOLDS                               │
│                                                                          │
│            True knowledge manifold                                      │
│                    ╭────────────╮                                       │
│                   ╱              ╲                                      │
│                  ╱    ○ target    ╲                                     │
│                 │                  │                                    │
│                 │      ○←ε₁        │  Compression error                 │
│                  ╲       ↘        ╱   pushes OFF manifold              │
│                   ╲        ○←ε₂  ╱    Matching error                   │
│                    ╲         ↘  ╱     pushes FURTHER                   │
│                     ╲          ○←ε₃   Decompression                    │
│                      ╲        ╱       now in HALLUCINATION SPACE       │
│                       ╲      ╱                                          │
│                        ╲    ╱                                           │
│                         ╲  ╱                                            │
│                          ╲╱  ← Off-manifold = hallucinatory            │
│                                                                          │
│  Errors don't just reduce magnitude—they change DIRECTION              │
│  Once off the truth manifold, you're in form-prior space              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.4.3 Formal Characterization

**Definition 12 (Distortion Operator).**
Each stage of the pipeline applies a distortion operator $D_i$ with error characteristic $\epsilon_i$:

$$
D_{total} = D_3 \circ D_2 \circ D_1 \tag{Def}
$$

**Theorem 4 (Geometric Distortion Accumulation).**

Let $\epsilon_i$ be the distortion introduced at stage $i$. The total fidelity is multiplicative:

$$
\text{Fidelity} = \prod_i (1 - \epsilon_i) \geq 1 - \sum_i \epsilon_i \tag{Approx}
$$

Equality holds exactly when at most one $\epsilon_i$ is non-zero (making all cross-product terms vanish), and approximately when all $\epsilon_i \ll 1$ (first-order Taylor expansion). The gap between the product and the linear approximation is $\sum_{i < j} \epsilon_i \epsilon_j + O(\epsilon^3)$—the sum of all pairwise products plus higher-order terms. When errors are correlated (aligned in representation space), the distortion compounds faster:

$$
\text{Fidelity}_{correlated} \ll \prod_i (1 - \epsilon_i) \tag{Approx}
$$

**Proof sketch.** Model each stage $i$ as a contraction $T_i$ on the topic-aligned signal subspace with operator norm $\lVert T_i \rVert \le 1-\epsilon_i$. By submultiplicativity, $\lVert T_n \cdots T_1 \rVert \le \prod_i (1-\epsilon_i)$. Under independence and small $\epsilon_i$, expected fidelity matches the product. When distortions are correlated (aligned off-manifold), the effective contraction is stricter, yielding a smaller bound than the independent-case product (cf. Friis, 1944).

**Proposition 11 (Manifold Departure).**

Representations lie on a truth manifold $\mathcal{M}$. Each distortion has two components:

$$
\epsilon_i = \underbrace{\epsilon_i^{\parallel}}_{\text{along manifold}} + \underbrace{\epsilon_i^{\perp}}_{\text{off manifold}} \tag{Def}
$$

The parallel component shifts within valid representations (may still be accurate). The perpendicular component pushes into hallucination space. Perpendicular errors compound faster because there's no truth-preserving structure to constrain them.

#### 8.4.4 The Chain Effect

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  WHY LONG REASONING CHAINS FAIL                                         │
│                                                                          │
│  Step 1: Retrieve fact A         (ε₁ distortion)                       │
│  Step 2: Reason from A to B      (ε₂ distortion ON TOP OF ε₁)          │
│  Step 3: Reason from B to C      (ε₃ distortion ON TOP OF ε₁∘ε₂)      │
│  ...                                                                    │
│  Step n: Final answer            (∏ᵢ εᵢ accumulated distortion)        │
│                                                                          │
│  This is why:                                                           │
│  - Multi-hop reasoning degrades rapidly                                 │
│  - RAG with multiple retrievals can HURT accuracy                      │
│  - Very long CoT eventually drifts into hallucination                  │
│  - Self-consistency helps (averaging reduces geometric accumulation)   │
│                                                                          │
│  Each step is a lossy transformation on an already-lossy state         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.4.5 The Friis Formula Analogy

This parallels **noise accumulation in cascaded amplifiers** (Friis formula, 1944; conceptual analogy):

$$
\text{SNR}_{total} = \frac{\text{SNR}_1}{1 + \frac{1}{G_1 \cdot \text{SNR}_2} + \frac{1}{G_1 G_2 \cdot \text{SNR}_3} + \ldots} \tag{Approx}
$$

The early stages dominate. For LLMs:
- **Compression quality** (training) is the primary determinant
- **Matching precision** (retrieval) is the second bottleneck
- **Decompression fidelity** (generation) cannot recover what's already lost

This is why **better training data beats better prompting**; you can't decompress what wasn't properly compressed.

#### 8.4.6 The Distortion-Hallucination Relationship

Combining conservation (Section 8.3) with distortion accumulation, and assuming **independent per-stage errors**:

$$
K(\text{output}) \approx \underbrace{K(\text{source}) \cdot \prod_i (1 - \epsilon_i)}_{\text{grounded (degraded)}} + \underbrace{K(\text{form prior}) \cdot \left[1 - \prod_i (1 - \epsilon_i)\right]}_{\text{hallucinated filler}} \tag{Approx}
$$

As fidelity drops (product decreases), the form prior fills in the gaps. **Hallucination is proportional to accumulated distortion**:

$$
K(\text{hallucinated}) \propto 1 - \prod_i (1 - \epsilon_i) = 1 - \text{Fidelity} \tag{Approx}
$$

Assumption: Independent errors. When errors are correlated (e.g., systematic training bias pushing representations in a consistent off-manifold direction), degradation can be faster than the product formula predicts.

#### 8.4.7 Implications

| Phenomenon | Geometric Distortion Explanation |
|------------|----------------------------------|
| Training data quality matters most | First-stage distortion propagates through all subsequent stages |
| Multi-hop reasoning fails | Each hop multiplies distortion |
| RAG can hurt on complex queries | Multiple retrievals cascade errors |
| Very long CoT drifts | Accumulated distortion eventually dominates signal |
| Self-consistency helps | Averaging independent samples reduces correlated errors |
| Fine-tuning on domain helps | Reduces first-stage distortion for that domain |
| Smaller models hallucinate more | Higher per-stage distortion |

### 8.5 Thermodynamic Interpretation: Hallucination as Thermalization

The framework achieves its deepest form when connected to statistical mechanics. Hallucination is not merely an error; it is **thermalization** to the maximum entropy state.

#### 8.5.0 Intuition: The Thermal Bath

In thermodynamics, a **thermal bath** (or heat reservoir) is the environment that a system equilibrates to when constraints are removed. Think of it as "room temperature" for a physical system.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE THERMAL BATH: WHERE THINGS DRIFT WHEN UNCONSTRAINED               │
│                                                                          │
│  Physical example:                                                      │
│    - Ice cube (ordered, low entropy, constrained)                      │
│    - Remove from freezer → Melts → Room temperature water              │
│    - The room is the "thermal bath"                                     │
│    - System EQUILIBRATES to the bath when constraints removed          │
│                                                                          │
│  Key insight: The bath is the MAXIMUM ENTROPY state                    │
│               consistent with ambient conditions                        │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  For LLMs:                                                              │
│                                                                          │
│  KNOWLEDGE = ICE CUBE                    FORM PRIOR = ROOM TEMPERATURE  │
│    - Low entropy (few correct answers)     - High entropy (many options)│
│    - Constrained by facts                  - Constrained only by form   │
│    - Requires "energy" to maintain         - Default equilibrium state  │
│                                                                          │
│  When knowledge constraints FAIL:                                       │
│    The system "melts" → Equilibrates to the form prior                 │
│    This is THERMALIZATION                                               │
│    The output drifts to what's statistically common                    │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  HALLUCINATION = THERMALIZATION                                         │
│                                                                          │
│  The model "cools" to room temperature (form prior) when it can't      │
│  maintain the "frozen" state of specific knowledge. The form prior     │
│  IS the thermal bath—the maximum-entropy attractor that everything     │
│  drifts toward when content constraints are absent.                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

This explains why hallucinations are so hard to prevent: the form prior is where the system *wants* to be. Maintaining grounded output requires constant "energy" (constraints, context, knowledge) to keep the system from relaxing to its comfortable equilibrium of fluent-but-empty text.

*Terminological note:* Form prior sampling (thermalization) differs from **Kolmogorov garbage** (decompression failure, Sec. 4.5.2). Form prior sampling occurs when knowledge constraints are *absent*; the system has nothing to reconstruct and thermalizes to maximum entropy. Kolmogorov garbage occurs when knowledge is *present but inaccessible*; insufficient room to unfold the compressed representation produces fragmented output. Both are hallucinations; only the mechanism differs.

#### 8.5.1 The Thermodynamic Duality

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THERMODYNAMIC DUALITY                                                  │
│                                                                          │
│  POTENTIAL ENERGY          ←→          KINETIC ENERGY                   │
│  Low entropy                           High entropy                     │
│  Stored, constrained                   Released, dispersed              │
│  Structure                             Movement/randomness              │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  FOR LLMs:                                                              │
│                                                                          │
│  STORED KNOWLEDGE          ←→          FORM PRIOR                       │
│  (Potential)                           (Kinetic)                        │
│  Low entropy                           High entropy                     │
│  Few valid outputs                     Many valid-looking outputs       │
│  Content-constrained                   Only form-constrained            │
│  GROUNDED                              HALLUCINATED                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.5.2 Boltzmann's Equation for LLMs

The entropy of output space follows Boltzmann (Boltzmann, 1877):

$$
S = k_B \ln \Omega \tag{Def}
$$

Where $\Omega$ is the number of valid outputs (microstates):
- **Grounded generation**: Knowledge constrains → small $\Omega$ → low $S$
- **Hallucination**: Only form constrains → large $\Omega$ → high $S$

*Note on microstates:* For current LLMs, a microstate is a **distinct token sequence** satisfying the given constraints. Two sequences differing by even a single token are different microstates. While this framework uses tokens as the fundamental unit, the theory generalizes to any discrete representation—future architectures operating on graphs, structured objects, or other modalities would simply redefine microstates accordingly. The thermodynamic principles (entropy, equilibration, temperature) remain invariant to the choice of elementary unit.

**The form prior is the maximum entropy state.** Hallucination is relaxation to this equilibrium.

#### 8.5.3 The Gibbs Distribution

Output probability follows Gibbs-Boltzmann statistics (Boltzmann, 1877; Jaynes, 1957):

$$
P(x) = \frac{1}{Z} e^{-E(x)/kT} \tag{Def}
$$

Where:
- $E(x)$ = "energy" = $-\log P(\text{correct} | x)$ = how "ungrounded" output $x$ is
- $T$ = temperature parameter (literally exists in LLM sampling!)
- $Z$ = partition function (normalization)

Note: In LLMs, "sampling temperature" is an algorithmic control, not a physical temperature; the Gibbs/Boltzmann analogy is conceptual and empirically useful rather than literal.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE ENERGY LANDSCAPE                                                   │
│                                                                          │
│  Energy E(x)                                                            │
│    ▲                                                                    │
│    │      ╭─────╮                  ╭───────────────────────╮           │
│    │     ╱       ╲                ╱                         ╲          │
│    │    ╱         ╲              ╱     HALLUCINATION BASIN   ╲         │
│    │   ╱           ╲            ╱       (high entropy,        ╲        │
│    │  ╱  GROUNDED   ╲          ╱         many microstates)     ╲       │
│    │ ╱    WELL       ╲________╱                                 ╲      │
│    │╱  (low entropy,                                             ╲     │
│    │    few states)                                               ╲    │
│    └───────────────────────────────────────────────────────────────▶   │
│                              Output space x                            │
│                                                                          │
│  With knowledge constraints: System stays in grounded well             │
│  Without constraints: System thermalizes to high-entropy basin         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.5.4 Temperature IS the Hallucination Dial

The temperature parameter in LLM sampling is literally the Boltzmann temperature:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  LLM TEMPERATURE = BOLTZMANN TEMPERATURE                                │
│                                                                          │
│  T → 0  (Greedy/Deterministic):                                        │
│    P(x) → δ(x - x_min)                                                 │
│    Only lowest energy state (most grounded)                            │
│    Minimum entropy, maximum constraint                                  │
│                                                                          │
│  T = 1  (Standard):                                                     │
│    P(x) ∝ e^{-E(x)}                                                    │
│    Balanced sampling around grounded states                            │
│                                                                          │
│  T → ∞  (Maximum temperature):                                         │
│    P(x) → uniform                                                       │
│    All outputs equally likely                                          │
│    Maximum entropy = pure form prior                                   │
│    COMPLETE THERMALIZATION = PURE HALLUCINATION                        │
│                                                                          │
│  Increasing T releases potential (knowledge) into kinetic (entropy)    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.5.5 Free Energy Formulation

Generation can be understood as minimizing Helmholtz free energy:

$$
F = E - TS = \underbrace{-\log P(\text{correct})}_{\text{energy (groundedness)}} - T \cdot \underbrace{H(\text{output})}_{\text{entropy (diversity)}} \tag{Def}
$$

At low T: Energy dominates → Grounded, deterministic
At high T: Entropy dominates → Thermalized, hallucinated

**Theorem 5 (Thermodynamic Hallucination).**

Let $\Omega_{\text{knowledge}}$ be the number of outputs consistent with stored knowledge, and $\Omega_{\text{form}}$ be the number of outputs consistent only with form. The probability of hallucination follows:

$$
P(\text{hallucination}) \propto e^{S_{\text{form}} - S_{\text{knowledge}}} = \frac{\Omega_{\text{form}}}{\Omega_{\text{knowledge}}} \tag{Approx}
$$

Hallucination probability increases exponentially with the entropy difference between form prior and knowledge constraints.

**Proof sketch.** Under a maximum-entropy (Gibbs) ensemble with weak energy differences across admissible outputs and $k_B=1$, the probability mass assigned to each admissible set is proportional to its microstate count $\Omega$. When constraints fail, sampling transitions from the knowledge-constrained ensemble to the form-only ensemble; the relative likelihood scales as $\Omega_{\text{form}}/\Omega_{\text{knowledge}} = e^{\Delta S}$. If average energies differ non-negligibly between sets, an additional factor depending on those energies appears; we subsume this into temperature-dependent constants in the proportionality (Jaynes, 1957; Boltzmann, 1877).

#### 8.5.6 The Complete Thermodynamic Picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE THERMODYNAMICS OF HALLUCINATION                                    │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                          │
│  STORED KNOWLEDGE = POTENTIAL ENERGY = LOW ENTROPY                     │
│    - Compressed in weights                                              │
│    - Constrains output space                                            │
│    - Few valid outputs (small Ω)                                       │
│    - Structured, ordered                                                │
│                                                                          │
│  FORM PRIOR = KINETIC/THERMAL ENERGY = HIGH ENTROPY                    │
│    - Distributed across all fluent text                                │
│    - Many valid-looking outputs (large Ω)                              │
│    - Unstructured (from knowledge perspective)                         │
│    - Maximum entropy equilibrium state                                 │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                          │
│  HALLUCINATION = THERMALIZATION                                         │
│                                                                          │
│  When constraints fail (capacity, matching, decompression, distortion) │
│  the system releases "potential" into "kinetic":                       │
│                                                                          │
│    Knowledge (potential) → Released → Form prior (kinetic)             │
│    Low entropy           → Expands → High entropy                      │
│    Constrained           → Relaxes → Equilibrium                       │
│                                                                          │
│  The form prior IS the thermal bath.                                   │
│  Hallucination IS equilibration to this bath.                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.5.7 Implications for Control

This thermodynamic view suggests control mechanisms:

| Control | Thermodynamic Interpretation |
|---------|------------------------------|
| Lower temperature | Favor low-energy (grounded) states |
| More context | Deepen the potential well |
| Better training | Increase potential energy storage |
| Constraint injection | Add energy barriers against thermalization |
| Self-consistency | Cool the system via averaging |

The goal is to keep the system in the **grounded potential well** and prevent thermalization to the **form prior bath**.

### 8.6 The Functional Role of Noise: Error Correction Requires Exploration

A crucial counterpoint: **noise is not just the enemy; it is also the medicine**. Systems require a certain amount of inherent and learned noise to correct mistakes.

#### 8.6.0 Intuition: The Stuck Lock

Why does adding noise improve accuracy?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE STUCK LOCK (Stochastic Resonance)                                  │
│                                                                          │
│  Analogy: Opening a jammed door with a key.                             │
│                                                                          │
│  Strategy A: Deterministic Force (Greedy Decoding, T=0)                 │
│    Action: Push the key straight in with maximum force.                 │
│    Result: It jams on the first misalignment. FAILS.                    │
│                                                                          │
│  Strategy B: Random Jiggling (Optimal Noise, T=T*)                      │
│    Action: Gently jiggle the key while pushing.                         │
│    Result: The noise helps the key slide past the sticking point.       │
│    Result: OPENS.                                                       │
│                                                                          │
│  Strategy C: Violent Shaking (High Temperature, T >> T*)                │
│    Action: Shake the key wildly.                                        │
│    Result: You drop the key or break the lock. FAILS.                   │
│                                                                          │
│  Key Insight:                                                           │
│  Greedy decoding gets stuck in "local minima" (the first wrong token).  │
│  Optimal noise allows the model to "jiggle" out of errors and           │
│  find the correct path.                                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.6.1 The Dual Role of Noise

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  NOISE IS BOTH THE DISEASE AND THE CURE                                │
│                                                                          │
│  TOO LITTLE NOISE (T → 0):                                             │
│    - Stuck in local minima                                              │
│    - Cannot explore alternatives                                        │
│    - Cannot correct mistakes                                            │
│    - Brittle, deterministic                                             │
│    - Overfitted, memorized                                              │
│                                                                          │
│  TOO MUCH NOISE (T → ∞):                                               │
│    - Pure entropy                                                       │
│    - Complete thermalization                                            │
│    - Hallucination dominates                                            │
│                                                                          │
│  OPTIMAL NOISE (T = T*):                                                │
│    - Enough fluctuation to explore                                      │
│    - Enough stability to preserve signal                                │
│    - Can escape bad attractors                                          │
│    - Can correct mistakes via exploration                               │
│    - Generalizes (not memorizes)                                        │
│                                                                          │
│  THE GOLDILOCKS ZONE                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.6.2 Error Correction Requires Exploration

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ERROR CORRECTION REQUIRES EXPLORATION                                  │
│                                                                          │
│  Without noise:                                                         │
│    Input → Deterministic path → Output                                  │
│    If path is wrong, NO WAY TO CORRECT                                 │
│    Stuck in the mistake                                                 │
│                                                                          │
│  With optimal noise:                                                    │
│    Input → Stochastic exploration → Sample alternatives                │
│    If one path wrong, noise enables finding another                    │
│    Self-consistency, beam search, etc. USE this                        │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  ANALOGY: SIMULATED ANNEALING                                           │
│                                                                          │
│  T high:  Explore widely (can escape local minima)                     │
│  T low:   Exploit locally (crystallize around best solution)           │
│                                                                          │
│  You NEED the high-T phase to find good solutions                      │
│  Then COOL to lock them in                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.6.3 Stochastic Resonance

In physics, **stochastic resonance** demonstrates that adding noise to a weak signal can make it *more* detectable. For LLMs, this means optimal noise can help retrieve weak memories:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  STOCHASTIC RESONANCE IN LLMs                                           │
│                                                                          │
│  Retrieval                                                              │
│  quality                                                                │
│     ▲                                                                   │
│     │           ╭─────╮                                                 │
│     │          ╱       ╲                                                │
│     │         ╱         ╲                                               │
│     │        ╱           ╲                                              │
│     │       ╱             ╲                                             │
│     │      ╱               ╲                                            │
│     │     ╱                 ╲                                           │
│     │────╱                   ╲──────────────────▶                       │
│     │   0        σ*                              Noise σ                │
│     │         OPTIMAL                                                   │
│     │                                                                   │
│  σ = 0: Can't explore, stuck if wrong path activated                  │
│  σ = σ*: Can explore alternatives, find correct weak memory            │
│  σ → ∞: Noise overwhelms signal                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.6.4 Learned Noise as Regularization

During training, noise is essential for building error-correction capability:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  TRAINING-TIME NOISE = REGULARIZATION = ERROR-CORRECTION CAPACITY      │
│                                                                          │
│  Dropout:     Random neuron zeroing                                    │
│  → Prevents co-adaptation, forces redundancy                           │
│  → Network learns ROBUST representations                               │
│                                                                          │
│  Data augmentation:  Noisy variants of inputs                          │
│  → Forces learning of INVARIANTS, not particulars                      │
│                                                                          │
│  Label smoothing:    Soft targets with uncertainty                     │
│  → Prevents overconfidence, better calibration                         │
│                                                                          │
│  Stochastic gradient descent:  Weight noise                            │
│  → Escapes sharp minima, finds flat basins                            │
│  → Flat basins = generalization                                        │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  By training with noise, the model learns:                             │
│  1. Multiple paths to same answer (redundancy)                         │
│  2. Robustness to perturbations                                        │
│  3. How to recover when initial path fails                             │
│                                                                          │
│  This is analogous to ERROR-CORRECTING CODES:                          │
│  Redundancy enables correction                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.6.5 Formal Characterization

**Theorem 6 (Optimal Noise Principle).**

There exists an optimal noise level $\sigma^*$ that maximizes the trade-off between exploration benefit and thermalization cost:

$$
\sigma^* = \arg\max_\sigma \left[ \underbrace{P(\text{correction} | \sigma)}_{\text{exploration benefit}} - \underbrace{P(\text{hallucination} | \sigma)}_{\text{thermalization cost}} \right] \tag{Conj}
$$

*Note on objective:* This formulation highlights the two competing effects of noise. Equivalently, $\sigma^*$ maximizes overall accuracy: $\sigma^* = \arg\max_\sigma P(\text{correct output} \mid \sigma)$. The decomposition into correction and hallucination terms makes explicit *why* intermediate noise is optimal: it enables error recovery (correction) while limiting drift to the form prior (hallucination).

At $\sigma = 0$: No hallucination, but no error correction capability
At $\sigma \to \infty$: Complete exploration, but pure hallucination
At $\sigma = \sigma^*$: Optimal balance enabling self-correction while preserving signal

**Physical Basis: The Three Ingredients.**
Stochastic Resonance is defined physically by three ingredients (Gammaitoni et al., 1998) which map directly to the LLM generation process:

1.  **An Energetic Barrier (Threshold):** In physics, the potential barrier between bistable states. In LLMs, the **logit threshold** or attention score required to select a specific, low-probability content token over the high-probability form prior.
2.  **A Weak Coherent Input (Signal):** In physics, the periodic force. In LLMs, the **weakly stored knowledge** or ambiguous prompt that biases the distribution but is insufficient to cross the threshold deterministically.
3.  **A Source of Noise:** In physics, the heat bath. In LLMs, the **sampling temperature (T)** or random seed variation.

*The Mechanism:* Without noise ($T=0$), the weak signal (knowledge) never crosses the threshold; the system defaults to the global attractor (form prior). With optimal noise ($T^*$), the fluctuations sum with the weak signal to cross the threshold intermittently, "amplifying" the knowledge. With too much noise, the signal is swamped.

**Proof sketch.** Let $f(\sigma) = P(\text{correction} \mid \sigma) - P(\text{hallucination} \mid \sigma)$. Empirically and in models of stochastic resonance, $f(0)$ is suboptimal due to lack of exploration, and $f(\sigma)\to -\infty$ as $\sigma\to\infty$ due to thermalization. Under continuity and mild unimodality, there exists $\sigma^*>0$ that maximizes $f$. This mirrors classical stochastic resonance (Gammaitoni et al., 1998) and simulated annealing (Kirkpatrick et al., 1983) arguments where controlled noise enables escape from poor attractors before cooling. Training-time noise mechanisms (dropout, SGLD) similarly improve generalization via noise-induced exploration (Srivastava et al., 2014; Welling & Teh, 2011).

**Corollary (Temperature Regimes).**

$$
\begin{aligned}
T = 0: \quad &\text{Frozen. Deterministic. Cannot self-correct.} \\
T = T^*: \quad &\text{Goldilocks. Explores. Corrects. Preserves signal.} \\
T \to \infty: \quad &\text{Boiled. Pure entropy. Hallucination.}
\end{aligned} \tag{Conj}
$$

#### 8.6.6 The Complete Thermodynamic Picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE COMPLETE THERMODYNAMICS OF GENERATION                              │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                          │
│  KNOWLEDGE = POTENTIAL ENERGY = SIGNAL                                 │
│    Constrains output, reduces entropy                                  │
│    Provides grounding                                                   │
│                                                                          │
│  FORM PRIOR = THERMAL BATH = NOISE BACKGROUND                          │
│    Maximum entropy attractor                                            │
│    Source of hallucination                                              │
│                                                                          │
│  SAMPLING NOISE = TEMPERATURE = EXPLORATION CAPABILITY                 │
│    Enables error correction                                             │
│    Enables escape from local minima                                    │
│    Enables self-consistency                                             │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                          │
│  THE BALANCE:                                                           │
│                                                                          │
│  Noise (T > 0) is NECESSARY for:                                       │
│    ✓ Error correction                                                  │
│    ✓ Exploring alternatives                                            │
│    ✓ Escaping bad attractors                                           │
│    ✓ Self-consistency sampling                                         │
│    ✓ Generalization (during training)                                  │
│                                                                          │
│  But too much noise causes:                                             │
│    ✗ Thermalization to form prior                                      │
│    ✗ Hallucination                                                      │
│    ✗ Signal destruction                                                 │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                          │
│  NOISE IS BOTH THE DISEASE AND THE CURE                                │
│  THE QUESTION IS DOSAGE                                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.6.7 Implications

| Phenomenon | Noise Interpretation |
|------------|---------------------|
| Self-consistency works | Multiple samples explore alternatives, voting selects best |
| Beam search helps | Parallel exploration of multiple paths |
| Temperature tuning matters | Finding the Goldilocks zone |
| Dropout improves generalization | Learned redundancy enables error correction |
| Greedy decoding is brittle | T=0 cannot correct initial mistakes |
| High temperature is creative but unreliable | Exploration dominates grounding |

#### 8.6.8 Adaptive Resonance

The stochastic resonance phenomenon (Sec. 8.6.3) suggests a fixed optimal noise $\sigma^*$. But the optimal noise level depends on the *knowledge state*, i.e., how strongly the correct representation is activated. This motivates **adaptive resonance**: dynamically adjusting noise and matching thresholds based on retrieval confidence.

**Connection to Adaptive Resonance Theory.**
Grossberg's Adaptive Resonance Theory (ART) (Grossberg, 1976) from cognitive neuroscience addresses how biological systems learn new patterns without catastrophic forgetting. The key mechanism is **resonance**: when input sufficiently matches a stored pattern (above a "vigilance" threshold), a feedback loop stabilizes retrieval. When no match exceeds vigilance, a new category is created.

This maps directly to our framework:

| **ART Concept** | **Framework Analog** |
|-----------------|----------------------|
| Vigilance parameter $\rho$ | Matching threshold (Sec. 4.4) |
| Resonance state | Successful reconstruction (low $\Delta S$) |
| Mismatch reset | Matching failure → composite activation |
| Adaptive vigilance | Dynamic threshold based on context density |

**Definition 13 (Adaptive Resonance Condition).**
Resonance occurs when the query-representation match exceeds an adaptive threshold:

$$
\text{Resonance} \iff \frac{K(p \cap r_i)}{K(r_i)} > \rho_{adaptive} \tag{Def}
$$

where $\rho_{adaptive}$ adjusts based on:
1. **Context constraint density**: Higher density → stricter vigilance
2. **Estimated knowledge capacity $C_T$**: Lower capacity → more permissive matching
3. **Temperature $T$**: Higher temperature → wider resonance basins

**Theorem 7 (Adaptive Resonance Optimality).**
There exists an optimal vigilance $\rho^* = f(C_T, T, s)$ that minimizes the sum of false rejections and false acceptances:

$$
\rho^* = \arg\min_\rho \left[ P(\text{matching failure} \mid \rho) + P(\text{false resonance} \mid \rho) \right] \tag{Conj}
$$

At $\rho = 0$: Everything "resonates" → hallucination via composite activation  
At $\rho = 1$: Nothing resonates → capacity underutilization  
At $\rho = \rho^*$: Optimal match-specificity trade-off

**Dual Control: Noise × Vigilance.**
The optimal noise $\sigma^*$ (Theorem 6) and adaptive vigilance $\rho^*$ are dual controls that should co-vary with knowledge certainty:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  DUAL CONTROL: NOISE × VIGILANCE                                        │
│                                                                          │
│  σ (temperature): Controls exploration breadth during generation        │
│  ρ (vigilance):   Controls matching strictness during retrieval        │
│                                                                          │
│  Joint optimum: (σ*, ρ*) = argmax P(correct) - P(hallucination)        │
│                                                                          │
│  High σ + High ρ: Explores widely but accepts only strict matches      │
│  Low σ + Low ρ:   Deterministic but accepts weak matches               │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  Key insight: They should CO-VARY with knowledge certainty             │
│                                                                          │
│  Strong knowledge: Low σ, High ρ (lock in, be strict)                  │
│  Weak knowledge:   High σ, Low ρ (explore, accept weaker matches)      │
│                                                                          │
│  This is the adaptive resonance principle:                             │
│  Tune retrieval and generation jointly based on knowledge state        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Prediction 23 (Adaptive Resonance Peak).**
For queries with low estimated $C_T$ (weak knowledge), jointly increasing temperature $T$ while relaxing embedding-similarity thresholds will exhibit a **resonance peak**: a $(\sigma, \rho)$ combination where retrieval of correct weak memories exceeds both the frozen ($\sigma = 0$) and strict ($\rho = 1$) baselines.

**Prediction 24 (Knowledge-Contingent Optimum).**
The optimal $(\sigma^*, \rho^*)$ pair varies systematically with topic capacity:
- High-capacity topics: Low $\sigma^*$, high $\rho^*$ (confident, strict)
- Low-capacity topics: Higher $\sigma^*$, lower $\rho^*$ (exploratory, permissive)

This predicts that uniform temperature settings are suboptimal; adaptive temperature scheduling based on estimated knowledge capacity should improve accuracy.

---

## 9. Experimental Predictions

All theorems and propositions in this work are intended as testable claims. We will empirically evaluate Theorems 3–8 and Predictions 1–26, designing experiments to confirm or falsify each. These theorems are natural consequences of foundational results:
- Theorem 3 (Information Conservation) follows from the Data Processing Inequality and related information-theoretic limits (Sec. 10.1).
- Theorem 4 (Geometric Distortion Accumulation) follows from submultiplicativity of contractions and cascaded-noise models (Friis-style reasoning) (Secs. 8.4, 10.5).
- Theorem 5 (Thermodynamic Hallucination) follows from the Gibbs–Boltzmann distribution and maximum-entropy principles (Secs. 8.5, 10.5).
- Theorem 6 (Optimal Noise Principle) follows from stochastic resonance and annealing-style trade-offs between exploration and stability (Secs. 8.6, 10.5).
- Theorem 7 (Adaptive Resonance Optimality) follows from the dual-control framework of noise and vigilance, extending stochastic resonance to matching thresholds (Sec. 8.6.8).
- Theorem 8 (Model-Specific Sampling Limit) is conjectural, motivated by Nyquist–Shannon sampling theory applied to representation manifolds (Sec. 11.6).

### 9.1 Testable Hypotheses

Our framework generates specific predictions:

**Prediction 1 (Frequency-Accuracy Correlation).**  
For topics $T_1, T_2$ with training frequencies $f_1 > f_2$, the hallucination rate satisfies:

$$
P(\text{hallucination} | T_1) < P(\text{hallucination} | T_2) \tag{Approx}
$$

**Prediction 2 (Few-Shot Logarithmic Improvement).**  
Hallucination rate decreases logarithmically with number of relevant examples:

$$
P(\text{hallucination} | k \text{ examples}) \propto \frac{1}{\log(1 + k)} \tag{Approx}
$$

**Prediction 3 (Confidence-Grounding Decoupling).**  
On out-of-distribution topics, confidence correlates with form quality, not content accuracy:

$$
\text{Corr}(\text{confidence}, \text{fluency}) > \text{Corr}(\text{confidence}, \text{accuracy}) \tag{Approx}
$$

**Prediction 4 (Prompt Specificity Effect).**  
Hallucination rate increases monotonically with geometric mismatch between prompt and target representation. From Theorem 2 (Gaussian kernel):

$$
P(\text{hallucination}) \approx 1 - \exp\left(-\frac{d_{\mathcal{M}}(\phi(p), \phi(r_{target}))^2}{2\sigma^2}\right) \tag{Approx}
$$

where $d_{\mathcal{M}}$ is distance in the universal embedding space. For small distances, this is approximately quadratic: $P(\text{hallucination}) \approx d_{\mathcal{M}}^2 / 2\sigma^2$. More specific prompts better match internal representation structure, reducing ambiguity-induced errors. When the prompt is maximally aligned with the target ($d_{\mathcal{M}} \to 0$), hallucination probability is minimized.

**Prediction 5 (Context Crowding Effect).**  
In the crowding regime (high context utilization), hallucination rate increases non-linearly as available decompression room approaches zero:

$$
P(\text{hallucination}) \propto \frac{1}{K_{latent} - K_{query} - K_{context}} \quad \text{when } K_{context} \to K_{latent} - K_{query} \tag{Approx}
$$

As decompression room decreases, reconstruction quality degrades, producing Kolmogorov garbage. Note: This describes the right side of the U-shaped curve in Prediction 18; the full relationship includes insufficient context effects (left side of U).

**Prediction 6 (Decompression Asymmetry).**  
Complex topics require disproportionately more context room than simple topics, independent of query length:

$$
K_{reconstruct}(\text{complex}) \gg K_{reconstruct}(\text{simple}) \tag{Approx}
$$

This explains why long contexts hurt more on complex queries; the decompression room is consumed.

**Prediction 7 (Information Conservation Violation).**  
Hallucinations can be detected by information accounting. For truthful outputs:

$$
K(\text{output} | \text{topic}) \leq K(\text{source} | \text{topic}) \tag{Proxy}
$$

When this inequality is violated, information was created; the output contains more structure than was learned or provided. This excess necessarily came from the form prior and is definitionally hallucinated.

**Prediction 8 (Excess Information Source).**  
The "extra" information in hallucinations correlates with high-frequency patterns in the training corpus (form prior), not with topic-specific facts:

$$
K(\text{hallucinated excess}) \sim K(\text{form prior}) \tag{Approx}
$$

Hallucinations should statistically resemble generic fluent text, not domain-specific knowledge.

**Prediction 9 (Geometric Distortion Accumulation).**  
Under the assumption of independent per-stage errors, hallucination rate increases geometrically with reasoning chain length:

$$
P(\text{hallucination after } n \text{ steps}) = 1 - \prod_{i=1}^{n} (1 - \epsilon_i) \tag{Approx}
$$

For uniform per-step error $\epsilon$: $P(\text{hallucination}) \approx 1 - (1-\epsilon)^n$, which grows faster than $n\epsilon$ for non-trivial $\epsilon$. When errors are correlated (e.g., systematic bias), degradation can be faster.

**Prediction 10 (First-Stage Dominance).**  
Training data quality (compression fidelity) has larger effect on hallucination rate than inference-time interventions:

$$
\frac{\partial P(\text{hallucination})}{\partial \epsilon_1} > \frac{\partial P(\text{hallucination})}{\partial \epsilon_3} \tag{Approx}
$$

Improving training beats improving prompting; you can't decompress what wasn't properly compressed.

**Prediction 11 (Multi-Hop Degradation).**  
For n-hop reasoning tasks with independent per-hop errors, accuracy degrades as:

$$
\text{Accuracy}(n) \approx \text{Accuracy}(1)^n \tag{Approx}
$$

Two-hop reasoning squares the error rate; three-hop cubes it. This assumes independent errors; correlated errors (e.g., consistent retrieval bias) can cause faster degradation.

**Prediction 12 (Temperature-Hallucination Relationship).**  
Hallucination rate follows Boltzmann statistics with sampling temperature:

$$
P(\text{hallucination} | T) \propto e^{\Delta S} \cdot f(T) \tag{Approx}
$$

where $\Delta S = S_{\text{form}} - S_{\text{knowledge}}$ (with $k_B = 1$ per notation conventions) and $f(T)$ is monotonically increasing. Higher temperature → more thermalization → more hallucination.

**Prediction 13 (Entropy Ratio Prediction).**  
Hallucination probability scales with the ratio of microstate counts:

$$
P(\text{hallucination}) \propto \frac{\Omega_{\text{form}}}{\Omega_{\text{knowledge}}} = e^{S_{\text{form}} - S_{\text{knowledge}}} \tag{Approx}
$$

Topics with larger form-to-knowledge entropy gaps hallucinate exponentially more often.

**Prediction 14 (Free Energy Minimization).**  
At fixed temperature, outputs minimize free energy $F = E - TS$. Hallucinations occur when the entropy term dominates:

$$
\text{Hallucination when: } T \cdot S_{\text{form}} > E_{\text{grounding}} \tag{Approx}
$$

This predicts a critical temperature above which thermalization dominates grounding.

**Prediction 15 (Optimal Noise Existence).**  
There exists an optimal temperature $T^* > 0$ that maximizes accuracy via error correction:

$$
T^* = \arg\max_T \left[ P(\text{correction} | T) - P(\text{hallucination} | T) \right] \tag{Conj}
$$

Greedy decoding ($T=0$) is suboptimal because it cannot self-correct.

**Prediction 16 (Stochastic Resonance).**  
For weakly stored knowledge, there exists a noise level that improves retrieval:

$$
\exists \sigma^* > 0 : P(\text{correct} | \sigma^*) > P(\text{correct} | \sigma=0) \tag{Conj}
$$

Adding noise can help retrieve weak memories that deterministic decoding misses.

**Prediction 17 (Self-Consistency Optimality).**  
Self-consistency (sampling + voting) achieves better accuracy than single-sample at any fixed temperature:

$$
\text{Accuracy}(\text{vote}(T, n)) > \text{Accuracy}(\text{single}(T)) \quad \forall T > 0 \tag{Approx}
$$

This is because voting exploits exploration while averaging reduces hallucination variance.

**Prediction 18 (Goldilocks Context Window).**  
For fixed query complexity and topic, there exists an optimal context length $L^*$ that minimizes hallucination (maximizes accuracy). Hallucination is high when $L \ll L^*$ (insufficient constraints) and when $L \gg L^*$ (decompression crowding):

$$
L^* \;=\; \arg\min_L \left| \underbrace{K_{\text{latent}} - K_{\text{query}} - K_{\text{context}}(L)}_{K_{\text{available}}(L)} \;-\; K_{\text{reconstruct}}(r) \right| \tag{Approx}
$$

Equivalently, $P(\text{hallucination} \mid L)$ is U-shaped in $L$, minimized when $K_{\text{available}}(L)\approx K_{\text{reconstruct}}(r)$ (Sec. 4.5).

Regime clarification: Prediction 5 describes the right branch of this U-curve (crowding regime, $L \gg L^*$). The left branch ($L \ll L^*$) reflects insufficient content constraints; the model lacks information to ground its output. This prediction unifies both failure modes.

**Prediction 19 (Geometry‑Aligned Warm‑Start).**  
Pretraining or initialization that aligns internal representations to the universal manifold $\mathcal{M}_{\text{universal}}$ (via CCA/Procrustes losses or teacher features) reduces sample complexity and speeds convergence. Let $\tau$ be a target accuracy threshold and $t(\tau)$ be the number of training steps required to reach it:

$$
t_{\text{geo}}(\tau) \;<\; t_{\text{base}}(\tau), \quad \text{and} \quad P_{\text{hallucination}}^{\text{geo}}(B) \;<\; P_{\text{hallucination}}^{\text{base}}(B) \tag{Approx}
$$

where $B$ is a fixed training budget; geometry-aligned initialization reaches target accuracy $\tau$ faster and achieves lower hallucination at equal compute (Secs. 8.3, 11.5).

**Prediction 20 (Geometry‑Driven Training Diagnostics and Error Correction).**  
A representation‑alignment score $g(t)$ (distance to $\mathcal{M}_{\text{universal}}$) correlates with downstream accuracy and inversely with hallucination rate; using $g(t)$ for online monitoring/regularization reduces hallucination:

$$
\text{Corr}\big(g(t), \text{Accuracy}_{\text{val}}(t+\Delta)\big) > 0, \qquad 
\text{Corr}\big(g(t), -P(\text{hallucination})\big) > 0 \tag{Approx}
$$

Adding a penalty $\lambda \cdot d(\text{rep}(t), \mathcal{M}_{\text{universal}})$ during training improves grounding and stability (Secs. 8.3, 11.5).

### 9.2 Capacity Estimation Experiments

To validate the framework, one could:

1. **Measure topic-specific capacity** via probing accuracy on held-out facts
2. **Correlate with hallucination rate** on generation tasks
3. **Test in-context capacity boost** by measuring improvement with examples
4. **Verify the capacity threshold** by finding the transition point

### 9.3 Matching and Decompression Experiments

To validate the Kolmogorov matching and decompression hypotheses:

1. **Prompt specificity gradient**: Vary prompt complexity from ambiguous ("that animal") to specific ("African elephant, Loxodonta africana") and measure hallucination rate
2. **Context crowding curve**: Fix query complexity, vary context length, measure quality degradation
3. **Reconstruction room estimation**: For fixed topics, determine minimum context budget for accurate generation
4. **Kolmogorov garbage detection**: Train classifiers to distinguish coherent vs. fragmented outputs
5. **Lost-in-the-middle replication**: Verify that mid-context information degrades more than edges (decompression crowding)
6. **Chain-of-thought decomposition**: Measure if distributing reasoning across steps reduces decompression pressure

### 9.4 Information Conservation Experiments

To validate the Information Conservation law (Theorem 3):

1. **Information accounting**: Estimate $K(\text{output})$ and $K(\text{source})$ via compression proxies (gzip, neural compressors); verify that hallucinations violate $K(\text{out}) \leq K(\text{source})$
2. **Excess source tracing**: Show that the "extra" information in hallucinations correlates with corpus-wide patterns (form), not topic-specific knowledge
3. **Conservation-based detector**: Build classifier using $K(\text{output}) - K(\text{source})$ as primary feature; compare to existing hallucination detectors
4. **Topic capacity probing**: For topics with known training frequency, estimate $K(\text{weights})$ and verify it predicts hallucination rate
5. **Guaranteed generation**: Implement generation that refuses to output when $K(\text{estimated output}) > K(\text{source})$; verify reduced hallucination rate

### 9.5 Geometric Distortion Experiments

To validate Theorem 4 (geometric distortion accumulation):

1. **Chain length scaling**: Measure accuracy on n-hop reasoning tasks; verify geometric decay $(1-\epsilon)^n$ rather than linear decay $(1-n\epsilon)$
2. **First-stage dominance**: Compare effect of training data quality vs. prompting quality on same task; verify training improvements dominate
3. **Manifold departure tracking**: Use representation probing to measure distance from "truth manifold" after each reasoning step; verify cumulative drift
4. **Cascaded retrieval degradation**: In multi-hop RAG, measure accuracy vs. number of retrieval steps; verify multiplicative error accumulation
5. **Self-consistency as distortion reduction**: Verify that self-consistency (averaging) reduces effective $\epsilon$ by decorrelating errors
6. **Friis formula validation**: Model hallucination as SNR degradation; fit Friis-like formula to empirical data

### 9.6 Thermodynamic Experiments

To validate Theorem 5 (thermodynamic hallucination):

1. **Temperature scaling**: Measure hallucination rate vs. sampling temperature; verify Boltzmann-like relationship
2. **Entropy ratio measurement**: Estimate $\Omega_{\text{form}}$ and $\Omega_{\text{knowledge}}$ via sampling; verify hallucination rate scales as ratio
3. **Free energy decomposition**: Separate energy (grounding) and entropy (diversity) terms in generation; verify trade-off
4. **Critical temperature identification**: Find temperature threshold where thermalization dominates grounding
5. **Potential well depth**: Vary knowledge strength (training frequency) and measure how it affects resistance to temperature increases
6. **Self-consistency as cooling**: Verify that averaging multiple samples reduces effective temperature / increases grounding

### 9.7 Optimal Noise Experiments

To validate Theorem 6 (optimal noise principle):

1. **Optimal temperature search**: For fixed tasks, sweep temperature and find $T^*$ that maximizes accuracy; verify $T^* > 0$
2. **Stochastic resonance detection**: For weak-knowledge topics, verify existence of $\sigma^*$ where noise improves retrieval
3. **Self-consistency scaling**: Measure accuracy vs. number of samples at various temperatures; verify voting improves over single-sample
4. **Greedy vs. stochastic**: Compare $T=0$ (greedy) to $T=T^*$ (optimal) on error-prone tasks; verify stochastic wins
5. **Dropout ablation**: Compare models trained with/without dropout on out-of-distribution tasks; verify dropout models self-correct better
6. **Annealing schedules**: Test if temperature annealing (high→low during generation) improves over fixed temperature

---

## 10. Related Work

### 10.1 Information-Theoretic Approaches

- **Shannon (1948)**: Channel coding theorem establishing capacity limits
- **Shannon (1959)**: Rate distortion theory: the fundamental trade-off between compression and reconstruction fidelity
- **Kolmogorov (1965)**: Algorithmic complexity and minimal description length
- **Tishby (2000)**: Information bottleneck and compression in learning

### 10.2 Hallucination Studies

- **Ji et al. (2023)**: Survey of hallucination in NLP
- **Huang et al. (2023)**: Factuality in LLMs
- **Manakul et al. (2023)**: SelfCheckGPT for hallucination detection

### 10.3 In-Context Learning Theory

- **Xie et al. (2022)**: Explanation of in-context learning
- **Akyürek et al. (2023)**: What learning algorithm is in-context learning?
- **Olsson et al. (2022)**: In-context learning and induction heads

### 10.4 Neuro-Symbolic Integration

- **Goldman (2025)**: The Neuro-Symbolic Homeostat - Complexity from Constraints
- **Marcus (2020)**: The Next Decade in AI - symbolic grounding
- **Garcez & Lamb (2020)**: Neurosymbolic AI

### 10.5 Statistical Mechanics and Machine Learning

- **Boltzmann (1877)**: On the relationship between the second law of thermodynamics and probability theory
- **Jaynes (1957)**: Information theory and statistical mechanics
- **Hopfield (1982)**: Neural networks and physical systems with emergent collective computational abilities
- **Hinton & Sejnowski (1983)**: Optimal perceptual inference (Boltzmann machines)
- **Bahri et al. (2020)**: Statistical mechanics of deep learning

### 10.6 Supporting Technical Literature

- **Fischbacher et al. (2020)**: Intelligent Matrix Exponentiation, provides rigorous Lipschitz bounds and contraction analysis for neural network layers; supports the geometric distortion accumulation argument (Theorem 4)

---

## 11. Conclusion

### 11.1 Summary

We have presented a unified information-theoretic and thermodynamic framework for understanding LLM hallucinations:

1. **Compression is Learning**: Training compresses the world into weights (source coding)
2. **Transmission is Inference**: Generation transmits queries through the model (channel coding)
3. **Capacity is Knowledge**: Each topic has a capacity limit for reliable generation
4. **Hallucinations are Capacity Violations**: Generating beyond capacity produces form-constrained but content-unconstrained output
5. **Hallucinations are Matching Failures**: Ambiguous prompts fail to uniquely match internal compressed representations, activating wrong or composite concepts, analogous to quantum superposition under weak measurement
6. **Hallucinations are Decompression Failures**: Even correctly matched representations require room to reconstruct. Over-filled contexts produce Kolmogorov garbage: structurally valid fragments that fail to cohere
7. **Information Conservation**: You cannot output more information about a topic than was stored or provided; the excess must come from the form prior, and that excess is hallucination
8. **Geometric Distortion Accumulation**: Errors compound multiplicatively through the pipeline; each stage degrades fidelity geometrically, pushing representations off the truth manifold into hallucination space
9. **Thermodynamic Equilibration**: Knowledge is potential energy (low entropy, constrained); the form prior is the thermal bath (high entropy, unconstrained); hallucination is thermalization, defined as relaxation to maximum entropy when constraints fail
10. **Noise as Error Correction**: Paradoxically, systems require optimal noise ($T^* > 0$) to correct mistakes; too little noise prevents exploration and self-correction; the Goldilocks zone enables error recovery while preserving signal

The complete framework—capacity, matching, decompression, distortion, thermodynamics, and optimal noise—provides a full physics of hallucination. The conservation law (Theorem 3) gives a detection criterion. The distortion theorem (Theorem 4) explains accumulation. The thermodynamic theorem (Theorem 5) reveals hallucination as equilibration. The optimal noise theorem (Theorem 6) shows that $T=0$ is suboptimal: systems need stochasticity to self-correct. This suggests strategies based on constraint injection, capacity enhancement, prompt-representation alignment, context budget management, information accounting, first-stage quality prioritization, **temperature control**, and **optimal noise calibration**.

### 11.2 Key Equations

$$
\boxed{
\begin{aligned}
\text{Learning} &\equiv \text{Compression} \equiv \text{Source Coding} \\
\text{Inference} &\equiv \text{Transmission} \equiv \text{Channel Coding} \\
\text{Knowledge} &\equiv \text{Capacity} \equiv \text{Potential Energy} \\
\text{Form Prior} &\equiv \text{Thermal Bath} \equiv \text{Max Entropy} \\
\text{Hallucination} &\equiv \text{Thermalization to Form Prior} \\
\text{Fidelity} &= \prod_i (1 - \epsilon_i) \quad \text{(geometric decay; independent errors)} \\
P(\text{hallucination}) &\propto \Omega_{\text{form}} / \Omega_{\text{knowledge}} = e^{\Delta S} \\
T^* &= \arg\max_T [P(\text{correction}) - P(\text{hallucination})] > 0
\end{aligned}
} \tag{Approx}
$$

### 11.3 The Core Insight

> **Hallucinations are what happens when you ask a channel to transmit beyond its capacity, and it fills the gap with its prior distribution: fluent form, empty content.**

Or equivalently:

> **Hallucinations emerge when complexity is demanded without sufficient constraints. Without constraints, generation degrades to maximum-entropy output conditioned on form alone.**

From the matching perspective:

> **Hallucinations are quantum superposition collapses gone wrong—ambiguous prompts activate multiple internal representations, and the measurement selects incorrectly or produces a composite that never existed.**

From the decompression perspective:

> **Kolmogorov garbage in, Kolmogorov garbage out—when context is too cramped for reconstruction, the model produces structurally valid fragments that individually look correct but collectively hallucinate.**

And the unifying conservation law:

> **Information cannot be created; it can only be transmitted or lost. When a model outputs more information about a topic than it stored or was provided, the excess was hallucinated from the form prior. This is the absolute limit of truthful generation.**

And the accumulation principle:

> **Distortion is geometric, not additive. Each stage of the pipeline compounds errors multiplicatively, pushing representations off the truth manifold. The form prior fills the growing gap; hallucination grows as fidelity decays.**

And the thermodynamic unification:

> **Knowledge is potential energy; the form prior is the thermal bath. Hallucination is thermalization: when constraints fail, the system equilibrates to maximum entropy. The probability of hallucination scales as $e^{\Delta S}$: exponentially with the entropy gap between form and knowledge.**

And the final paradox:

> **Noise is both the disease and the cure. Too little noise prevents error correction; too much causes hallucination. The Goldilocks zone ($T = T^*$) enables exploration and self-correction while preserving signal. Systems need stochasticity to heal themselves.**

### 11.4 Future Directions

1. **Validate capacity estimators**: The framework provides theoretical operationalization via manifold alignment (Sec. 7.4), but the next step is empirical validation of embedding density, translation fidelity, and probing-based estimators at scale
2. **Information conservation detector**: Implement $K(\text{output})$ vs $K(\text{source})$ comparison for real-time hallucination detection
3. **Optimal constraint injection**: Determine minimal context for reliable generation
4. **Capacity-aware architectures**: Design models that know what they don't know
5. **Kolmogorov matching metrics**: Develop measures of prompt-representation alignment to predict retrieval accuracy
6. **Sweetspot analysis**: Characterize the optimal compression level for different knowledge domains
7. **Decompression budgeting**: Develop methods to estimate reconstruction room requirements and optimize context allocation
8. **Kolmogorov garbage detection**: Identify structural fragmentation patterns that indicate decompression failures
9. **Conservation-guaranteed generation**: Architectures that provably cannot output more information than available in source
10. **Distortion minimization**: Techniques to reduce per-stage $\epsilon_i$ at each pipeline stage
11. **Manifold-preserving architectures**: Designs that keep representations on the truth manifold through transformations
12. **Chain-length-aware generation**: Systems that estimate accumulated distortion and refuse multi-hop reasoning when fidelity drops below threshold
13. **Thermodynamic temperature control**: Adaptive temperature scheduling based on estimated knowledge capacity
14. **Potential well deepening**: Training techniques that increase the "energy barrier" against thermalization
15. **Free energy architectures**: Systems that explicitly minimize $F = E - TS$ with tunable trade-off
16. **Optimal noise calibration**: Methods to find $T^*$ for each query/context
17. **Stochastic resonance exploitation**: Techniques to use noise to retrieve weak memories
18. **Annealing schedules for generation**: Temperature trajectories that maximize exploration then crystallize

### 11.5 Limitations and Practical Implementation

**Kolmogorov Complexity is Uncomputable.**

The theoretical framework relies heavily on Kolmogorov complexity $K(x)$—the length of the shortest program that generates $x$. This quantity is provably uncomputable (halting problem). We cannot measure true $K(x)$ for arbitrary data.

#### 11.5.0 Intuition: Plato's Cave and the Universal Shape

Why should different models learn the same "geometry of truth"?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  PLATO'S CAVE AND THE UNIVERSAL MANIFOLD                                │
│                                                                          │
│  Imagine two artists (Models A and B) drawing a cat.                    │
│  - Artist A uses charcoal (Architecture A).                             │
│  - Artist B uses watercolors (Architecture B).                          │
│                                                                          │
│  Their drawings look different (different weights, different dimensions).│
│  BUT the "cat" they are drawing is the same object in reality.          │
│                                                                          │
│  The "Shadows on the Wall":                                            │
│  The models are seeing shadows of the same underlying reality.          │
│  If both models are accurate, their internal representations must       │
│  preserve the structure of that reality.                                │
│                                                                          │
│  Therefore:                                                             │
│  The GEOMETRY of the representation is determined by the OBJECT,        │
│  not by the artist.                                                     │
│                                                                          │
│  This is why we can translate between models: they are just different   │
│  coordinate systems for the SAME universal manifold.                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Practical Resolution: The Universality of Representations.**

We address this by leveraging the **Platonic Representation Hypothesis** (Huh et al., 2024) (and related findings on linear representations), which demonstrates a crucial fact:

**In-context representations are universal across models.**

Research confirms that sufficiently capable models converge to the **same geometric shapes** for representing concepts. The internal geometry of "truth" is not an arbitrary choice of the model, but a discovered structure imposed by the reality being modeled.

**Definition 14 (The Universal Manifold).**
There exists a shared, lower-dimensional manifold $\mathcal{M}_{universal}$ upon which all truthful representations lie. Different models merely learn different rotation/permutation projections of this same manifold.

**Empirical Validation: Unsupervised Embedding Translation.**

Strong empirical support for the universal manifold comes from recent work on unsupervised embedding translation (Jha et al., 2025). The vec2vec method demonstrates that embeddings from models with **completely different architectures, parameter counts, and training data** can be translated between each other **without any paired data**—achieving:

- **>0.92 cosine similarity** between translated embeddings and ground truth
- **Perfect matching on 8000+ embeddings** without knowing the possible match set in advance
- Preservation of semantic information sufficient for classification and attribute inference

This works because all models converge to the same underlying geometric structure. The translation succeeds not by learning a model-specific mapping, but by learning the **shared latent representation** that all models approximate.

**Foundational Result: LMs Are Provably Injective.**

A critical foundational result establishes that language models **preserve information** (Nikolaou et al., 2025):

- **Mathematical proof**: Transformer LMs mapping discrete sequences to continuous representations are injective
- **Empirical confirmation**: Billions of collision tests across six state-of-the-art models → zero collisions
- **Robustness**: Property established at initialization AND preserved during training

This means **information preservation is not the question**—LMs do not lose information through their forward pass. The question is entirely about **organization**: whether the preserved information is structured usefully for downstream tasks.

This distinction is critical:
- **Proven**: LMs are injective (no information loss) (Nikolaou et al., 2025)
- **Proven**: Training creates organization—latents converge to belief states (Teoh et al., 2025)

Hallucination is therefore not a failure of information storage, but a failure of **information access**—the knowledge may be present but inaccessible due to matching failures, decompression failures, or geometric misalignment with the universal manifold.

This solves the uncomputability problem for our purposes:
1.  We do not need to calculate absolute Kolmogorov complexity $K(x)$.
2.  We only need to measure **Geometric Alignment** with $\mathcal{M}_{universal}$.
3.  Hallucinations are vectors that have drifted off this shared manifold.
4.  **Alignment is operationally measurable** via translation fidelity, CKA similarity, or cycle-consistency losses.

**Standard geometric analysis tools serve as the coordinate systems** for measuring this alignment. We use Fourier analysis, SVD, and Wavelet transforms to **characterize the topology** of this universal manifold and detect deviations. Additionally, adversarial training with cycle consistency (as in vec2vec) provides a direct operational method for learning the manifold structure.

$$
\text{Hallucination}(x) \approx \|x - \text{proj}_{\mathcal{M}_{universal}}(x)\| \tag{Proxy}
$$

The "complexity" of a hallucination is that it possesses a geometry *inconsistent* with the universal shape of the concept. It is a geometric outlier. Operationally, a hallucinated embedding is one that **fails to translate faithfully** between model spaces; it lies off the shared manifold that enables high-fidelity translation.

**Other Limitations:**

1. **Entropy estimation is approximate**: We use sampling and compression proxies, not true information-theoretic quantities
2. **Capacity estimators need validation**: The framework provides theoretical operationalization via manifold alignment (Sec. 7.4), but practical estimators require empirical validation at scale
3. **Manifold geometry partially characterized**: The universal manifold's existence is empirically validated (Jha et al., 2025; Huh et al., 2024), but full topological characterization remains future work
4. **Full empirical validation is future work**: The 21 predictions and experimental designs await systematic implementation (though the core manifold hypothesis is now empirically supported)

**What This Framework Provides:**

Despite these limitations, the framework provides:
- Principled vocabulary for discussing hallucination mechanisms
- Testable predictions (even with approximate measurements)
- Design principles for mitigation (constraint injection, capacity awareness, temperature control)
- Unification of disparate observations under a single physics

### 11.6 Open Theorem: Model-Specific Sampling Limit (Nyquist–Shannon Analogy)

**Theorem 8 (Model-Specific Sampling Limit; Nyquist–Shannon Analogy).**  
Note: At this time, we classify this as a* ***conjecture*** *rather than a proven theorem. The "theorem" label reflects its structural role in the framework; formal proof remains future work.

*Conjectural.* For each model $M$ and topic $T$, there exists a representation bandlimit $B_{M,T}$ (in an appropriate spectral parameterization of the model's internal manifold) such that reliable reconstruction of topic-consistent outputs requires an effective "constraint sampling rate" $s$ (from prompt specificity, retrieved context, and internal working memory) satisfying

$$
s \;>\; 2\, B_{M,T} \tag{Conj}
$$

Equivalently, when the information-bearing structure of the input constraints is under‑sampled relative to the model's topic bandlimit, aliasing manifests as matching errors and Kolmogorov garbage; when severely over‑sampled, decompression room can be crowded (Sec. 4.5). The quantities $B_{M,T}$ and $s$ are model‑ and topic‑specific and depend on architecture and training; we do not yet have operational estimators.

*Proof status:* Open. The claim is motivated by classical sampling theory (Sec. 10.1) and observed spectral structure in learned representations, but precise definitions of bandlimits on nonlinear manifolds and their relation to attention/activation spectra are model‑dependent. We plan to investigate empirical estimators via frequency‑domain probes of attention/feature spectra vs. error curves under prompt/context resolution sweeps.

### 11.7 Architectural Validation: The Titans Memory Hierarchy

Recent architectural innovations provide concrete validation of our theoretical framework. The **Titans** architecture (Behrouz et al., 2025) introduces a neural long-term memory module that learns to memorize at test time, directly implementing several principles we have derived from information-theoretic first principles.

#### 11.7.0 Intuition: The Open-Book Exam

Standard LLMs suffer from a fundamental limitation: they cannot learn new facts once training ends.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  TEST-TIME LEARNING: THE OPEN-BOOK EXAM                                 │
│                                                                          │
│  Standard LLM (Static Weights):                                         │
│    - Like a student taking a closed-book exam.                          │
│    - Can only use what was memorized years ago (training).              │
│    - If the context contains new info, they must hold it in             │
│      "short-term memory" (attention), which is small and fragile.       │
│                                                                          │
│  Titans / Test-Time Learning (Dynamic Weights):                         │
│    - Like a student taking an open-book exam who can WRITE NOTES.       │
│    - As they read the prompt/context, they update their long-term       │
│      memory (synapses) on the fly.                                      │
│    - They "learn" the context, not just "attend" to it.                 │
│                                                                          │
│  Impact:                                                                │
│  The model's capacity GROWS during the conversation.                    │
│  C_effective = C_training + C_learned_from_context                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 11.7.1 Memory Duality as Architectural Primitive

Titans formalizes the distinction between short-term and long-term memory that maps directly to our static/dynamic codebook framework:

| **Titans Component** | **Our Framework Analog** | **Function** |
|---------------------|--------------------------|--------------|
| Attention (short-term) | Context window / dynamic codebook | Accurate, limited, ephemeral |
| Neural memory (long-term) | Weights / static codebook (atoms) | Compressed, persistent, capacious |
| Test-time learning | Adaptive resonance | Dynamic threshold adjustment |
| Forgetting gate ($\alpha_t$) | Sink severity control | Capacity management |

The paper states: "Attention due to its limited context but accurate dependency modeling performs as a **short-term memory**, while neural memory due to its ability to memorize the data, acts as a **long-term, more persistent, memory**." This is precisely our Section 2.2 distinction.

#### 11.7.2 The Compression Paradox Confirmed

Titans identifies the fundamental tension we formalize as Theorem 1 (Hallucination Threshold):

> "On one hand, we use these linear models to enhance scalability... On the other hand, **a very long context cannot be properly compressed in a small vector-valued or matrix-valued states**."

This IS capacity violation: when $R_T > C_T$, no architectural trick can avoid information loss. Titans addresses this by maintaining both compressed (long-term) and uncompressed (short-term) memory, trading off between them, exactly the strategy our framework predicts as optimal.

#### 11.7.3 Test-Time Learning as Dynamic Atom Creation

The crucial innovation in Titans is **learning at inference time**. The memory update rule:

$$
\mathcal{M}_t = \text{diag}(1 - \alpha_t)\mathcal{M}_{t-1} + S_t
$$

where $S_t$ incorporates gradient updates on a reconstruction loss, enables the model to **create new information atoms during inference**. This extends our framework (Sec. 4.7):

$$
\text{atoms}_{effective} = \text{atoms}_{training} + \text{atoms}_{test-time}(context)
$$

Test-time atom creation means the model is not limited to decompressing pre-stored knowledge—it can actively learn from context, reducing the gap between $C_T$ (stored capacity) and $R_T$ (requested rate).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  TEST-TIME ATOM CREATION (Titans Mechanism)                            │
│                                                                          │
│  Standard LLM:                                                          │
│    Query → Match to fixed atoms → Decompress → Output                  │
│    Capacity limited to K(weights)                                       │
│                                                                          │
│  Titans:                                                                │
│    Query → Match to atoms + LEARN new atoms from context → Output      │
│    Capacity = K(weights) + K(test-time learned)                        │
│                                                                          │
│  This is TEACHING WHILE LEARNING:                                       │
│    The teacher (model) acquires new knowledge during the lesson        │
│    Extends effective capacity beyond pre-training                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 11.7.4 Momentum vs. Momentary Surprise

Titans criticizes prior architectures (DeltaNet, TTT) for relying on "momentary surprise"—updating memory based only on the current token. Titans uses **momentum-based updates**:

$$
S_t = \text{diag}(\eta_t)S_{t-1} - \text{diag}(\theta_t)(\mathcal{M}_{t-1}k_t^\top k_t - v_t^\top k_t)
$$

This captures **token flow**; the sequence structure matters, not just individual tokens. In our framework, this explains why:
- **Context structure matters** (Sec. 4.6): Anchoring, schema-first prompts work because they establish momentum
- **Chain-of-thought helps** (Sec. 4.5): Each step builds on previous, creating coherent flow
- **Position primacy** (Prediction 21): Early tokens set the momentum that later tokens follow

#### 11.7.5 Forgetting as Capacity Management

The forgetting gate $\alpha_t$ (implemented as weight decay) directly addresses our attention sink problem (Sec. 4.6):

> "A forget mechanism... allows clearing the memory when very past information is not needed anymore."

This is **active capacity management**:
- Without forgetting: Memory fills up → new information cannot be stored → hallucination
- With forgetting: Old, irrelevant atoms decay → room for new atoms → extended effective capacity

The forgetting rate $\alpha_t$ plays the role of our sink severity control—managing what persists vs. what is cleared to maintain usable bandwidth.

#### 11.7.6 The Complete Memory Hierarchy

Titans makes explicit what our framework implies: a three-tier memory hierarchy for effective language modeling:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  MEMORY HIERARCHY (Titans / Our Framework Unified)                      │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TIER 1: LONG-TERM MEMORY (Persistent)                          │   │
│  │                                                                   │   │
│  │  Titans: Neural memory module W trained on full history         │   │
│  │  Ours: Weights = compressed atoms from training                 │   │
│  │                                                                   │   │
│  │  Properties:                                                     │   │
│  │    - High capacity (billions of parameters)                     │   │
│  │    - Slow update (training-time, or test-time with Titans)     │   │
│  │    - Lossy compression (abstractions, not verbatim)            │   │
│  │    - Source of hallucination when mismatched                   │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│                              │ Retrieval / Decompression               │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TIER 2: WORKING MEMORY (Ephemeral)                             │   │
│  │                                                                   │   │
│  │  Titans: Attention over current context window                  │   │
│  │  Ours: Context window = dynamic codebook                        │   │
│  │                                                                   │   │
│  │  Properties:                                                     │   │
│  │    - Limited capacity (context length)                          │   │
│  │    - Fast access (single forward pass)                          │   │
│  │    - Exact storage (no compression within window)              │   │
│  │    - Subject to crowding and sink effects                       │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│                              │ Gradient updates (Titans)               │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TIER 3: ADAPTIVE LAYER (Test-Time Learning)                    │   │
│  │                                                                   │   │
│  │  Titans: Gradient-based memory updates during inference         │   │
│  │  Ours: Adaptive resonance - dynamic ρ, test-time atoms         │   │
│  │                                                                   │   │
│  │  Properties:                                                     │   │
│  │    - Bridges long/short term                                    │   │
│  │    - Learns from current context                                │   │
│  │    - Enables capacity extension beyond pre-training             │   │
│  │    - Implements adaptive matching thresholds                    │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 11.7.7 Implications for Hallucination Mitigation

The Titans architecture suggests concrete implementations for our theoretical mitigations:

| **Our Theoretical Mitigation** | **Titans Implementation** |
|--------------------------------|---------------------------|
| Increase $C_T$ (capacity) | Test-time learning of new atoms |
| Control sink severity $s$ | Forgetting gate $\alpha_t$ |
| Adaptive resonance $\rho^*$ | Momentum-based memory updates |
| Prevent context crowding | Separate short-term (attention) from long-term (memory) |
| Add redundancy | Dual memory provides error correction via cross-checking |

**Prediction 25 (Test-Time Learning Reduces Hallucination).**
Architectures with test-time memory learning (Titans-style) will exhibit lower hallucination rates on topics partially covered in training, because they can:
1. Create new atoms from context to fill capacity gaps
2. Adaptively adjust matching thresholds based on memory state
3. Manage forgetting to maintain effective capacity

**Prediction 26 (Memory Hierarchy Optimality).**
The optimal memory architecture for minimizing hallucination maintains distinct tiers with different capacity/accuracy trade-offs, with an adaptive layer enabling transfer between tiers. Monolithic architectures (pure attention or pure recurrence) are suboptimal.

---

## References (currently adding links for easy access)

1. Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal. https://ieeexplore.ieee.org/document/6773024

2. Pappone, F. (2025). Attention sinks from the graph perspective. Università La Sapienza di Roma -- PSTP Technoscience. https://publish.obsidian.md/the-tensor-throne/The+Graph+Side+of+Attention/Attention+sinks+from+the+graph+perspective

3. Shannon, C. E. (1959). Coding Theorems for a Discrete Source with a Fidelity Criterion. IRE National Convention Record. https://gwern.net/doc/cs/algorithm/information/1959-shannon.pdf

4. Kolmogorov, A. N. (1965). Three Approaches to the Quantitative Definition of Information. Problems of Information Transmission. http://alexander.shen.free.fr/library/Kolmogorov65_Three-Approaches-to-Information.pdf

5. Tishby, N., Pereira, F. C., & Bialek, W. (2000). The Information Bottleneck Method. arXiv:physics/0004057. https://arxiv.org/abs/physics/0004057

6. Ji, Z., Lee, N., Frieske, R., et al. (2023). Survey of Hallucination in Natural Language Generation. arXiv:2202.03629. https://arxiv.org/abs/2202.03629

7. Xie, S. M., Raghunathan, A., Liang, P., & Ma, T. (2022). An Explanation of In-context Learning as Implicit Bayesian Inference. arXiv:2111.02080. https://arxiv.org/abs/2111.02080

8. Akyürek, E., Schuurmans, D., Andreas, J., Ma, T., & Zhou, D. (2023). What Learning Algorithm is In-Context Learning? Investigations with Linear Models. arXiv:2211.15661. https://arxiv.org/abs/2211.15661

9. Bach, A. (1990). Boltzmann's probability distribution of 1877. 
Analysis of Boltzmann [Published: March 1990] Alexander Bach . URL: https://link.springer.com/article/10.1007/BF00348700

10. Jaynes, E. T. (1957). Information Theory and Statistical Mechanics. Physical Review. DOI: 10.1103/PhysRev.106.620. https://journals.aps.org/pr/abstract/10.1103/PhysRev.106.620

11. Hopfield, J. J. (1982). Neural Networks and Physical Systems with Emergent Collective Computational Abilities. PNAS. DOI: 10.1073/pnas.79.8.2554. https://www.pnas.org/doi/10.1073/pnas.79.8.2554

12. Cover, T. M., & Thomas, J. A. (2005). Elements of Information Theory (2nd ed.). Wiley. https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X

13. Friis, H. T. (1944). Noise Figures of Radio Receivers. Proceedings of the IRE. https://ieeexplore.ieee.org/document/1695024

14. Gammaitoni, L., Hänggi, P., Jung, P., & Marchesoni, F. (1998). Stochastic Resonance. Reviews of Modern Physics, 70(1), 223–287. https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.70.223

15. Grossberg, S. (1976). Adaptive Pattern Classification and Universal Recoding: I. Parallel Development and Coding of Neural Feature Detectors. Biological Cybernetics.

16. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by Simulated Annealing. Science.

17. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR.
18. Welling, M., & Teh, Y. W. (2011). Bayesian Learning via Stochastic Gradient Langevin Dynamics. ICML.

19. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. ICML.

20. Liu, N. F., et al. (2023). Lost in the Middle: How Language Models Use Long Context. arXiv:2307.03172.

21. Goldman, O. (2025). Complexity from Constraints: The Neuro-Symbolic Homeostat. Shogu Research Group @ Datamutant.ai.

22. Jha, R., Zhang, C., Shmatikov, V., & Morris, J. X. (2025). Harnessing the Universal Geometry of Embeddings. arXiv:2505.12540. https://arxiv.org/abs/2505.12540

23. Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. arXiv:2405.07987.

24. Behrouz, A., Zhong, P., & Mirrokni, V. (2025). Titans: Learning to Memorize at Test Time. arXiv:2501.00663. https://arxiv.org/abs/2501.00663

25. Fischbacher, T., Comsa, I. M., Potempa, K., Firsching, M., Versari, L., & Alakuijala, J. (2020). Intelligent Matrix Exponentiation. arXiv:2008.03936. https://arxiv.org/abs/2008.03936

26. Wu, S., & Yao, Q. (2025). Asking LLMs to Verify First is Almost Free Lunch. arXiv:2511.21734. https://arxiv.org/abs/2511.21734

27. Teoh, J., Tomar, M., Ahn, K., Hu, E. S., Sharma, P., Islam, R., Lamb, A., & Langford, J. (2025). Next-Latent Prediction Transformers Learn Compact World Models. arXiv:2511.05963. https://arxiv.org/abs/2511.05963

28. Nikolaou, G., Mencattini, T., Crisostomi, D., Santilli, A., Panagakis, Y., & Rodolà, E. (2025). Language Models are Injective and Hence Invertible. arXiv:2510.15511. https://arxiv.org/abs/2510.15511

29. Berman, V. (2025). Random Text, Zipf's Law, Critical Length, and Implications for Large Language Models. arXiv:2511.17575. https://arxiv.org/abs/2511.17575

30. Berman, V. (2025). Zipf Distributions from Two-Stage Symbolic Processes: Stability Under Stochastic Lexical Filtering. arXiv:2511.21060. https://arxiv.org/abs/2511.21060

---

## Appendix A: The Duality Table

| Concept | Information Theory | Learning | Cognition |
|---------|-------------------|----------|-----------|
| Source coding | Compression | Training | Learning |
| Channel coding | Transmission | Inference | Communication |
| Entropy | Randomness | Uncertainty | Confusion |
| Capacity | Max reliable rate | Max knowledge | Intelligence |
| Noise | Corruption | Finite precision | Distraction |
| Redundancy | Error protection | Overparameterization | Emphasis |
| Codebook | Codewords | Weights | Concepts |

---

## Appendix B: Hallucination as Constraint Absence

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE HALLUCINATION EQUATION                                             │
│                                                                          │
│  P(output) = P(output | form_constraints ∩ content_constraints)         │
│                                                                          │
│  WHEN content_constraints → ∅:                                          │
│                                                                          │
│  P(output) → P(output | form_constraints)                               │
│           → Max entropy given form                                       │
│           → Fluent noise                                                 │
│           → HALLUCINATION                                                │
│                                                                          │
│  The model knows HOW to write (form) but not WHAT is true (content).   │
│  It generates the most probable output consistent with linguistic form, │
│  which may have no relationship to factual reality.                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix C: Geometric Matching and Decompression

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE MATCHING EQUATION (Operationalized)                                │
│                                                                          │
│  P(correct) ∝ exp(-d_M(φ(prompt), φ(representation)))                  │
│                                                                          │
│  WHERE d_M is distance in universal embedding space M_universal         │
│  Operationally measured via translation fidelity (Jha et al., 2025)    │
│                                                                          │
│  WHEN prompt is geometrically distant from target representation:       │
│                                                                          │
│  Multiple representations match → Superposition                         │
│  Wrong "measurement" → Composite or incorrect output                   │
│  → HALLUCINATION                                                        │
│                                                                          │
│  The prompt lacks discriminating structure.                             │
│  The model activates multiple candidates and picks wrong.              │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  THE DECOMPRESSION EQUATION                                             │
│                                                                          │
│  K(query) + K(context) + K(reconstruct) ≤ K(latent)                    │
│                                                                          │
│  WHEN K(available) < K(reconstruct):                                    │
│                                                                          │
│  Reconstruction truncated → Fragments stitched together                │
│  Structurally valid but semantically incoherent                        │
│  → KOLMOGOROV GARBAGE                                                   │
│                                                                          │
│  The context is too cramped for the representation to unfold.          │
│  The model produces pieces that look right but don't cohere.           │
│                                                                          │
│  OPERATIONALLY: Embeddings that fail to translate faithfully           │
│  between model spaces lie off the shared manifold.                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix D: The Three Failure Modes

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  1. CAPACITY FAILURE                                                    │
│     ═══════════════                                                     │
│     Knowledge: NOT STORED                                               │
│     Prompt: (irrelevant)                                                │
│     Context: (irrelevant)                                               │
│     Result: Max-entropy output conditioned on form                      │
│                                                                          │
│     → "I don't know this, but I'll say something fluent anyway"        │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  2. MATCHING FAILURE                                                    │
│     ═══════════════                                                     │
│     Knowledge: STORED                                                   │
│     Prompt: TOO AMBIGUOUS                                               │
│     Context: (irrelevant)                                               │
│     Result: Wrong or composite representation activated                 │
│                                                                          │
│     → "I know several things like this, let me pick... wrong one"      │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  3. DECOMPRESSION FAILURE                                               │
│     ════════════════════                                                │
│     Knowledge: STORED                                                   │
│     Prompt: SPECIFIC (correct match)                                    │
│     Context: TOO FULL                                                   │
│     Result: Truncated reconstruction = Kolmogorov garbage              │
│                                                                          │
│     → "I know this and found it, but no room to unfold properly"       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix E: The Conservation Law (Data Processing Limit)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE INFORMATION CONSERVATION LAW                                       │
│                                                                          │
│  Conservation Law: K(output) ≤ K(weights) + K(context)                 │
│                                                                          │
│  You cannot output more information than was stored OR provided.       │
│  Information is conserved through the cycle.                           │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  ANALOGY TO PHYSICS:                                                    │
│                                                                          │
│  Energy:      E_out ≤ E_in        (First Law of Thermodynamics)        │
│  Information: K_out ≤ K_source    (Conservation Limit)                 │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  THE HALLUCINATION DECOMPOSITION:                                       │
│                                                                          │
│  K(output) = K(from weights+context)  + K(from form prior)             │
│            = K(grounded content)      + K(hallucinated filler)         │
│                                                                          │
│  For truthful generation: K(hallucinated filler) = 0                   │
│  For hallucination:       K(hallucinated filler) > 0                   │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  DETECTION PRINCIPLE:                                                   │
│                                                                          │
│  IF K(output) ≤ K(source):  Information transmitted (possibly true)    │
│  IF K(output) > K(source):  Information CREATED (definitely false)     │
│                                                                          │
│  The excess came from the form prior—fluent patterns learned from      │
│  all text, not topic-specific knowledge. This excess is hallucination. │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix F: Geometric Distortion Accumulation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE DISTORTION ACCUMULATION LAW (assuming independent errors)          │
│                                                                          │
│  Fidelity = ∏ᵢ (1 - εᵢ)                                                │
│                                                                          │
│  Errors compound MULTIPLICATIVELY through the pipeline:                 │
│                                                                          │
│  World ──ε₁──▶ Weights ──ε₂──▶ Retrieved ──ε₃──▶ Output                │
│        compress      match         decompress                           │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  THE FRIIS ANALOGY:                                                     │
│                                                                          │
│  Just as in cascaded amplifiers:                                        │
│    - Early stages dominate total noise figure                          │
│    - Late stages cannot recover what early stages corrupted            │
│                                                                          │
│  For LLMs:                                                              │
│    - Training quality (ε₁) dominates                                   │
│    - Matching precision (ε₂) is second                                 │
│    - Generation fidelity (ε₃) cannot fix upstream errors              │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  THE MANIFOLD PRINCIPLE:                                                │
│                                                                          │
│  Representations live on a truth manifold M                            │
│  Each εᵢ has components:                                               │
│    - ε‖ (parallel): shifts along M (may still be accurate)            │
│    - ε⊥ (perpendicular): pushes OFF M (into hallucination space)      │
│                                                                          │
│  Perpendicular errors compound faster—no truth structure constrains    │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  THE HALLUCINATION-DISTORTION RELATIONSHIP:                            │
│                                                                          │
│  K(output) ≈ K(source)·∏(1-εᵢ) + K(form prior)·[1-∏(1-εᵢ)]            │
│            ≈ (grounded × fidelity) + (hallucinated × infidelity)       │
│                                                                          │
│  As fidelity drops, form prior fills the gap                           │
│  Hallucination ∝ accumulated distortion                                │
│                                                                          │
│  NOTE: Product formula assumes INDEPENDENT errors.                      │
│  Correlated errors (systematic bias) can cause FASTER degradation.     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix G: Thermodynamic Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  THE THERMODYNAMICS OF HALLUCINATION                                    │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                          │
│  FUNDAMENTAL DUALITY:                                                   │
│                                                                          │
│  KNOWLEDGE         ←→         FORM PRIOR                               │
│  Potential energy             Kinetic/thermal energy                   │
│  Low entropy                  High entropy                             │
│  Few microstates              Many microstates                         │
│  Constrained                  Unconstrained                            │
│  Grounded                     Hallucinated                             │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  BOLTZMANN'S EQUATION:                                                  │
│                                                                          │
│  S = kB ln Ω                                                            │
│                                                                          │
│  Ω_knowledge = few valid outputs (strong constraints)                  │
│  Ω_form = many valid-looking outputs (weak constraints)                │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  THE GIBBS DISTRIBUTION:                                                │
│                                                                          │
│  P(x) = (1/Z) exp(-E(x)/kT)                                            │
│                                                                          │
│  E(x) = -log P(correct|x) = "ungroundedness energy"                    │
│  T = sampling temperature (literally!)                                  │
│  Z = partition function                                                 │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  TEMPERATURE CONTROLS HALLUCINATION:                                   │
│                                                                          │
│  T → 0:   Greedy, lowest energy only (most grounded)                   │
│  T = 1:   Standard sampling, balanced                                   │
│  T → ∞:   Uniform, maximum entropy (pure form prior)                   │
│                                                                          │
│  Increasing T releases potential into kinetic = thermalization         │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  THE HALLUCINATION PROBABILITY:                                         │
│                                                                          │
│  P(hallucination) ∝ Ω_form / Ω_knowledge = exp(ΔS)                     │
│                                                                          │
│  Hallucination probability is EXPONENTIAL in entropy gap               │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  FREE ENERGY:                                                           │
│                                                                          │
│  F = E - TS = -log P(correct) - T·H(output)                            │
│                                                                          │
│  Generation minimizes F:                                                │
│    Low T: Energy dominates → Grounded                                  │
│    High T: Entropy dominates → Thermalized (hallucinated)              │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  THE UNIFYING PRINCIPLE:                                                │
│                                                                          │
│  Hallucination = Thermalization to the form prior bath                 │
│                                                                          │
│  When knowledge constraints fail, the system relaxes to maximum        │
│  entropy—the form prior. This is the thermal equilibrium state.        │
│  The form prior IS the thermal bath. Hallucination IS equilibration.   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix H: The Functional Role of Noise

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  NOISE IS BOTH THE DISEASE AND THE CURE                                │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  THE PARADOX:                                                           │
│                                                                          │
│  Too little noise (T → 0):                                             │
│    - Deterministic, frozen                                              │
│    - Cannot explore alternatives                                        │
│    - Cannot correct mistakes                                            │
│    - Stuck in local minima                                              │
│                                                                          │
│  Too much noise (T → ∞):                                               │
│    - Pure entropy                                                       │
│    - Complete thermalization                                            │
│    - Hallucination dominates                                            │
│                                                                          │
│  Optimal noise (T = T*):                                                │
│    - Explores alternatives                                              │
│    - Enables self-correction                                            │
│    - Preserves signal                                                   │
│    - Escapes bad attractors                                             │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  THE OPTIMAL NOISE EQUATION:                                            │
│                                                                          │
│  T* = argmax_T [P(correction|T) - P(hallucination|T)]                  │
│                                                                          │
│  T* > 0: Systems NEED stochasticity to self-correct                    │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  STOCHASTIC RESONANCE:                                                  │
│                                                                          │
│  For weak signals (partially known knowledge):                         │
│  ∃ σ* > 0 : P(correct|σ*) > P(correct|σ=0)                            │
│                                                                          │
│  Adding noise can IMPROVE retrieval of weak memories                   │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  LEARNED NOISE = ERROR-CORRECTION CAPACITY:                            │
│                                                                          │
│  Training with noise (dropout, augmentation, SGD) teaches:             │
│    1. Multiple paths to same answer (redundancy)                       │
│    2. Robustness to perturbations                                       │
│    3. How to recover when initial path fails                             │
│                                                                          │
│  This is analogous to error-correcting codes:                          │
│  Redundancy enables correction                                          │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  THE COMPLETE PICTURE:                                                  │
│                                                                          │
│  T = 0:    Frozen. Cannot self-correct. Brittle.                       │
│  T = T*:   Goldilocks. Explores. Corrects. Preserves signal.          │
│  T → ∞:    Boiled. Pure hallucination.                                 │
│                                                                          │
│  The art is finding T* for each context.                               │
│  The question is not whether noise, but how much.                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Citation

If you use this repository in your research, please cite it, this is ongoing work we would like to know your opinions and experiments, thank you.

Oscar Goldman - Shogu research Group @ Datamutant.ai (subsidiary of 温心重工業)






EXTRA:



### Key Supporting References

The universal manifold hypothesis central to our operationalization is empirically supported by:

Jha, R., Zhang, C., Shmatikov, V., & Morris, J. X. (2025). Harnessing the Universal Geometry of Embeddings. *arXiv preprint arXiv:2505.12540*. (Demonstrates unsupervised embedding translation with >0.92 cosine similarity across model architectures)

Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. *arXiv preprint arXiv:2405.07987*. (Theoretical foundation for universal representation convergence)

**Note**: All references should be expanded and inline. References are cited in shortform for transparency during development.

---

## License

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)



© 2025 Datamutant.ai

