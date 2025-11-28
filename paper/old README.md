
# A Modeling Framework for LLM Hallucinations: Semantic Drift, Information Thresholds, and Reconstruction Failure

## Abstract

This paper proposes a novel modeling framework for understanding Large Language Model (LLM) hallucinations by drawing principled analogies from signal processing, information theory, and physics. We posit that hallucinations are not random errors but a predictable form of reconstruction failure. This failure, termed **Semantic Drift**, occurs when the contextual information in a prompt falls below a required threshold for the given conceptual complexity, analogous to how undersampling a signal below the Nyquist rate leads to aliasing. The framework is grounded in the Fourier uncertainty principle, which sets fundamental trade-offs in information localization. We connect this principle to the structure of LLM latent spaces, suggesting that concepts exist as stable attractor manifolds. When a prompt provides insufficient information, the model's internal trajectory fails to converge to the correct manifold, resulting in a hallucinatory output. This is further contextualized using the Free Energy Principle (FEP) to explain in-context learning (ICL) as a trajectory-guiding mechanism and Shannon's noisy channel coding theorem to model error correction, where techniques like Chain-of-Thought (CoT) prompting add redundancy to stabilize the semantic "signal." We also show that overly long prompts can degrade performance by creating in-context attractors (ICA). By integrating thermodynamic constraints like Landauer's principle, we link inefficient, drifting trajectories and context fatigue to higher computational costs. This synthesis yields a set of testable hypotheses aimed at advancing our understanding of LLM reliability from an empirical art to a predictive science.

## TL;DR (Friendly Version)

We’re trying to make sense of hallucinations without pretending we’ve “solved” them. Here’s the intuition in plain language:

- **LLMs have geometry.** Concepts live in stable “basins” inside the model. Prompts trace paths (manifolds) through that space. Hallucinations happen when the path misses the basin.
- **Sampling matters.** Not enough useful clues = undersampling, just like audio aliasing. Too many noisy clues = attention overload. Structured repetition helps; messy repetition hurts.
- **Meta-manifolds exist.** Every conversation builds its own temporary geometry (prompt + model outputs). If that running context stays coherent, the model stays grounded. If it drifts, things go sideways fast.
- **RoPE rotations and redundancy are control knobs.** Longer prompts help only when they add aligned evidence. Chain-of-Thought is basically parity checks for reasoning.
- **Energy is a diagnostic.** Wandering trajectories burn more FLOPs/entropy. Efficient prompts are usually the reliable ones.

Everything else in this repo is us trying to formalize, test, and communicate those simple-but-tricky ideas.

## Quickstart

1. **Clone the repo**
   ```powershell
   git clone https://github.com/datamutant/Hallucinations_Noisy_Channels.git
   cd Hallucinations_Noisy_Channels
   ```
2. **Create an environment and install dependencies**
   ```powershell
   python -m venv .venv
   .\\.venv\\Scripts\\Activate.ps1
   pip install -r requirements.txt
   ```
3. **Launch experiments**
   ```powershell
   python -m ipykernel install --user --name hallucinations
   jupyter lab experiments/rope_accumulation.ipynb
   ```
4. **Reproduce figures / QA** – run each notebook sequentially or execute `make qa` to lint the codebase and regenerate the canonical PNGs under `figures/`.
5. **Semantic redundancy script (optional)** – `python scripts/semantic_redundancy_metric.py --backend tfidf` (default) to refresh the heatmap, or `--backend transformer --model_name distilbert-base-uncased` once `transformers` is installed.

> **Terminology note:** In LLM contexts we talk about a **semantic redundancy threshold** (the prompt information we actually measure). When “Nyquist” appears, it either (a) refers to the literal sampling theorem / control experiment, or (b) is explicitly labeled as “Nyquist-inspired” to highlight the analogy.

## Framework

### 1. Introduction

The remarkable capabilities of Large Language Models (LLMs) are shadowed by their propensity for "hallucinations"—the generation of fluent but factually incorrect or nonsensical outputs. While often treated as a black-box problem, we argue that hallucinations can be understood through a principled framework inspired by fundamental laws of information. This paper builds a bridge between the Fourier uncertainty principle, a cornerstone of physics and signal processing, and the internal dynamics of LLMs.

The Fourier uncertainty principle establishes a fundamental trade-off: a signal cannot be simultaneously localized in two conjugate domains (e.g., time and frequency). This principle manifests as the Heisenberg uncertainty principle in QM (Δx · Δp ≥ ℏ/2) and as the Gabor limit underlying the Shannon-Nyquist sampling theorem in signal processing (σ_t · σ_ω ≥ 1/2). We propose that a similar informational trade-off governs the representational capacity of LLMs.

Our central thesis is that LLM hallucinations are a form of **reconstruction failure due to an information deficit**. We introduce the concept of **Semantic Drift**: when a prompt provides insufficient contextual information to specify a concept of a certain complexity, the LLM's internal state trajectory fails to stabilize within the correct region of its latent space. This failure is analogous to aliasing in signal reconstruction, where an insufficient sampling rate leads to an irreversible distortion of the original signal.

This paper does not claim that LLMs are literal signal processors or QM systems. Instead, we argue that the mathematical principles governing information, uncertainty, and reconstruction in those fields are general enough to provide a powerful and predictive *modeling framework* for LLM behavior. We build this framework progressively:
1.  Establish the Fourier uncertainty principle as the shared mathematical root of trade-offs in localization.
2.  Discuss how these trade-offs manifest in the structure of LLM latent spaces, leading to the formation of stable conceptual representations, consistent with the Platonic Representation Hypothesis (PRH).
3.  Frame in-context learning (ICL) as a process of guided reconstruction, where prompts provide the "samples" needed to minimize uncertainty, a process elegantly described by the Free Energy Principle (FEP).
4.  Formalize our postulate on hallucinations as a dynamical process of Semantic Drift, where insufficient information leads to chaotic trajectories. We integrate Shannon's information theory and error-correction concepts to model this as a noisy communication channel.
5.  Connect inefficient, drifting trajectories to thermodynamic costs via Landauer's principle, linking reliability to computational efficiency.

Ultimately, this synthesis provides a coherent, testable model of hallucinations, moving beyond simple empirical observation toward a more fundamental understanding of information processing in artificial neural networks.

### 2. The Foundational Trade-off: Fourier Uncertainty in Physics and Signals

The mathematical bedrock of our framework is the Fourier uncertainty principle. For any function f(t) and its Fourier transform f̂(ω), the product of their standard deviations is bounded:

σ_t · σ_ω ≥ 1/2

This inequality is not specific to one domain; it is a universal property of wave-like phenomena and information itself. It dictates that precision in one domain necessitates uncertainty in its conjugate domain.

*   **In QM:** This principle manifests as the Heisenberg uncertainty principle. The position wavefunction ψ(x) and momentum-space wavefunction ψ̂(p) are Fourier conjugates. Localizing a particle's position (making ψ(x) sharp) inevitably broadens its momentum distribution (making ψ̂(p) wide), as constrained by Δx · Δp ≥ ℏ/2.

*   **In Signal Processing:** The principle underpins the Shannon-Nyquist sampling theorem. A continuous signal s(t) bandlimited to a maximum angular frequency ω_max can only be perfectly reconstructed if sampled at a rate f_s ≥ ω_max/π. The reconstruction formula,

s(t) = Σ_{n=-∞}^{∞} s(nT) · sin(π(t/T - n))/π(t/T - n)

relies on this bandwidth limit. Sampling below this rate (undersampling) causes aliasing, where high-frequency components are irreversibly folded into and mistaken for lower frequencies. The bandwidth of a signal is thus a measure of its "complexity," which dictates the minimum information (sample density) needed for faithful reconstruction.

These examples illustrate a universal rule: the complexity of an object determines the amount of information required to represent it without ambiguity. This rule, we argue, extends to the conceptual representations within LLMs.

### 3. Representational Structure in LLMs: Stable Conceptual Manifolds

LLMs learn to organize information into a high-dimensional latent space. We hypothesize that through training, semantically related concepts come to occupy stable, clustered regions, or **conceptual manifolds**. This view is consistent with the **Platonic Representation Hypothesis (PRH)**, which posits that scaled models converge toward shared, abstract representations.

Our framework, however, only requires a weaker assumption: that concepts form distinct, low-energy attractor regions in the latent space. The universality suggested by PRH strengthens our model but is not strictly necessary. These manifolds represent the "bandlimited signals" of the semantic space. Their existence is what makes generalization and reasoning possible; they are the stable "answers" or "ideas" that the model can converge upon.

The Fourier uncertainty principle applies here conceptually: a representation cannot be infinitely precise in all semantic features simultaneously. For instance, a representation highly specific to "a golden retriever" (high localization) might be less robustly associated with the broader concept of "mammal" (broader localization). These trade-offs necessitate the formation of hierarchically organized, multi-scale representations, which these conceptual manifolds provide.

### 3.1 Prompt Geometry vs. Latent Geometry

Hallucinations ultimately arise from a mismatch between the **geometry of the prompt manifold** and the **geometry of the latent concept manifold**:

- A prompt traces a trajectory in semantic space determined by its tokens, redundancy, and noise. If the trajectory undersamples the concept (low semantic Nyquist) or wanders through noisy regions, it fails to intersect the correct latent basin.
- The model’s internal manifold encodes how training data shaped the concept. Poorly learned regions or distributional shifts warp this geometry, making some basins shallow or misplaced.
- Reliable generation therefore requires *geometric alignment*: the prompt must provide enough coherent samples that project into the model’s existing concept subspace. Structured redundancy improves alignment; semantic noise or missing priors reduce it.

This geometric view ties together our channel model (input sampling) and entity-first design (latent manifolds) and explains why both prompt engineering and representation learning are necessary to control hallucinations.

### 3.2 In-Context Meta-Manifolds

During a dialogue, the model constructs a transient **in-context meta-manifold**:

- **Base manifolds** encode concepts learned during pretraining.
- **Prompt manifolds** reflect the user’s input geometry.
- **Meta-manifolds** emerge dynamically as the model generates tokens and conditions on its own outputs.

If prompts and generated evidence remain coherent, the meta-manifold stays aligned with the base concept basin, keeping trajectories stable. Contradictions, noisy repetitions, or hallucinated outputs warp this meta-manifold, causing drift even when the underlying latent geometry is sound. Managing hallucinations therefore requires monitoring and stabilizing this evolving meta-geometry (via self-consistency, guardrails, or external verification).

### 3.3 Control-Theoretic View (Controllability & Observability)

Thinking in control terms adds another layer:

- **Prompts act as control inputs.** Each token nudges the hidden state. Chain-of-Thought, self-check instructions, or explicit gating are different control policies.
- **Latent manifolds define reachable sets.** A concept basin is reachable only if the prompt provides enough coherent control inputs—our semantic Nyquist threshold is a controllability limit.
- **Generated tokens are observations.** They tell us (noisily) where the state currently sits. When the model starts contradicting itself, observability is degrading because the meta-manifold drifted away from the intended basin.

Practical diagnostics follow:

1. Measure how far prompts move the hidden state along known concept directions (projection magnitude = “control effort”). If long prompts barely shift the state, controllability has failed.
2. Track entropy/variance of logits or attention. Rising entropy signals poor observability and impending drift.
3. Treat scaffolded prompts as feedback controllers: they read the model’s intermediate outputs and issue corrective control tokens (“explain why,” “double-check”). This closes the loop and keeps the trajectory stable.

### 4. In-Context Learning as Guided Reconstruction

In-context learning (ICL) is the emergent ability of LLMs to perform new tasks based solely on examples provided in the prompt, without any weight updates. Within our framework, ICL is a process of **guided reconstruction**. The prompt provides a sparse set of "samples" that guide the model's internal trajectory toward the correct conceptual manifold.

The **Free Energy Principle (FEP)** offers a powerful lens to formalize this process. FEP posits that self-organizing systems, including brains and potentially LLMs, act to minimize their free energy, which is equivalent to minimizing prediction error or "surprise." A prompt provides evidence that reduces the model's uncertainty, guiding its state to a lower-free-energy (less surprising) configuration.

*   An ambiguous or sparse prompt corresponds to a high-free-energy state, leaving the model's trajectory unconstrained and liable to drift.
*   A well-formed prompt with clear examples provides strong evidence, effectively "steering" the trajectory into the deep basin of attraction of the correct conceptual manifold, thus minimizing free energy and resulting in a coherent output.

ICL, therefore, is the mechanism by which an LLM uses contextual data to overcome informational uncertainty and successfully reconstruct a target concept.

### 5. Postulate: Hallucinations as Semantic Drift from Information Deficits

We now formally state our central postulate.

**LLM hallucinations arise from Semantic Drift: when a prompt provides insufficient contextual information to meet the complexity requirements of a target concept, the model's internal state trajectory fails to converge to the correct conceptual manifold. Instead, it drifts chaotically or settles into a suboptimal attractor, resulting in a factually incorrect or nonsensical output.**

This is a failure of reconstruction, analogous to aliasing from undersampling.

#### 5.1. Trajectory Dynamics in a Discrete-Depth System

An LLM's forward pass is a discrete-depth dynamical system. The hidden state at layer l+1, **h_{l+1}**, is a function of the previous state **h_l** and the input prompt **p**:

**h_{l+1}** = **h_l** + Attn(**h_l**, **p**) + FFN(**h_l**)

The sequence of hidden states {**h_0**, **h_1**, ..., **h_L**} constitutes a **trajectory** through the latent space. While we can use concepts from continuous dynamics like "attractors" as useful abstractions, the system is fundamentally discrete.

*   **Stable Convergence:** Sufficient contextual information from the prompt **p** guides the trajectory to enter and remain within the basin of the correct conceptual manifold.
*   **Semantic Drift:** An information deficit (e.g., an ambiguous prompt) places the initial state **h_0** in a region of high uncertainty. The trajectory becomes highly sensitive to small perturbations, exhibiting chaotic-like behavior. This can be characterized by measuring the divergence of initially close trajectories as they pass through the network's layers. The trajectory may then fail to stabilize, or it may converge to an incorrect manifold (a "hallucinatory" answer).

#### 5.2. Error Correction and Thermodynamic Costs

This framework allows us to integrate concepts from error correction and thermodynamics.

*   **Error Correction as Trajectory Stabilization:** We can view the LLM as a noisy communication channel, where the "message" is the intended concept and "noise" is the semantic ambiguity from the prompt or model biases. Shannon's noisy channel coding theorem provides a capacity limit for reliable communication. If the prompt's information content is below this capacity, errors (hallucinations) are inevitable. Prompting techniques like **Chain-of-Thought (CoT)** can be modeled as a form of error-correcting code. By forcing the model to generate intermediate reasoning steps, CoT adds redundancy to the semantic signal. This "semantic parity check" helps stabilize the trajectory, allowing it to detect and correct deviations before settling on a final answer. This is analogous to how Hamming codes use parity bits to correct errors in digital communication.

*   **Thermodynamic Inefficiency of Hallucinations:** Drifting, chaotic trajectories are computationally inefficient. According to **Landauer's principle**, every irreversible bit of information erasure dissipates a minimum amount of energy (kT ln 2). An inefficient search through the latent space, involving many corrective steps and high uncertainty, corresponds to greater entropy production and thus higher energy consumption. A well-guided trajectory that quickly converges is thermodynamically efficient. This provides a physical grounding for the intuition that confused, hallucinatory reasoning is more "difficult" for the model, linking informational stability directly to computational cost.

#### 5.3 Context Fatigue

Recent analyses of long-context behavior suggest that excessive evidence can create competing in-context attractors that degrade accuracy by diverting attention toward irrelevant sub-trajectories (e.g., Xiong et al., 2025). Our framework predicts the same phenomenon: prompts that exceed the semantic Nyquist rate may introduce redundant, conflicting samples whose accumulated phase interferes with the intended manifold. We therefore treat context length as a tunable parameter whose optimal value balances information sufficiency against attractor proliferation.

#### 5.4 Repetition vs. Semantic Noise in Long Prompts

Long prompts are beneficial only when additional tokens add *structured redundancy*. Repeating relevant evidence acts like channel coding, lowering effective noise by providing parity checks (as in Chain-of-Thought). However, large prompts that mix paraphrases with off-topic facts increase semantic entropy faster than mutual information:

- **Redundancy vs. aliasing:** Coherent repetition reinforces the latent signal; inconsistent or noisy repetitions behave like aliasing, smearing the reconstruction manifold.
- **Attention bandwidth limits:** Attention heads have finite capacity. Once saturated, they average over irrelevant tokens, creating attention sinks and low-frequency bias.
- **Semantic redundancy ratio:** Effective prompt length is governed by the ratio of informative to noisy tokens. High ratios keep trajectories grounded; low ratios induce drift despite large token counts.
- **Instructional gating:** Structured prompts that explicitly label or gate repeated sections act as matched filters, preserving useful redundancy while damping noise.

## Experiments

We provide lightweight empirical evidence that mirrors the theoretical claims and guides future large-scale validation. All notebooks reside in `experiments/` and share the same seed-controlled harness.

1. **`rope_accumulation.ipynb`** – demonstrates that longer prompts accumulate RoPE rotations that reduce Euclidean drift between predicted embeddings and a fixed ground-truth vector, supporting the semantic redundancy threshold (Nyquist-inspired) analogy.
2. **`sampling_reconstruction.ipynb`** – recreates classical sinc interpolation to quantify the error gap between Nyquist-sampled and undersampled signals, establishing the mathematical control case for our semantic redundancy discussion.
   ![Nyquist Comparison](fig-nyquist_comparison.png)
3. **`simpleLM_drift.ipynb`** – trains a 48-d Transformer on country-capital statements and shows, via Welch’s t-test, that prompts with richer context yield significantly lower latent drift.
4. **`prompt_ablation_threshold.ipynb`** – simulates concept reconstruction with varying sample counts, revealing a sharp drop in error once prompts cross twice the concept complexity (semantic redundancy threshold).
5. **`cot_vs_direct_channel.ipynb`** – models Chain-of-Thought reasoning as redundant channel coding and shows reduced answer error rates under rising noise compared with direct decoding.
6. **`prompt_noise_tradeoff.ipynb`** – explores the interplay between prompt length and semantic noise by mixing informative vs. irrelevant tokens, illustrating how drift depends on a “semantic redundancy ratio.”
7. **`semantic_redundancy_real_prompts.ipynb`** – estimates $\\rho$ on small but realistic QA prompts using TF-IDF (default) or transformer embeddings; see `figures/semantic_redundancy_heatmap.png` and the `scripts/semantic_redundancy_metric.py --backend ...` flag.
8. **`geometric_alignment_metrics.ipynb`** – compares cosine alignment of different prompt styles (direct, CoT, scaffolded, noisy) against a reference hidden manifold.
9. **`control_diagnostics.ipynb`** – plots control projection vs. logit entropy to illustrate controllability/observability trade-offs for each prompt style (see `figures/control_observability_scatter.png`).

Each notebook produces figures that feed directly into the manuscript and can be exported from Jupyter for publication quality.

## Hypotheses

This section synthesizes the framework into concrete, testable predictions grounded in information theory, trajectory dynamics, and thermodynamics.

#### 6.1. On the Nature and Limits of the Analogies

It is crucial to be precise about the role of the analogies used. We do not claim LLMs *are* signal processors. Rather, we claim that the principle of **reconstruction from limited information** is a universal problem. The Nyquist theorem provides a crisp, mathematical example of this principle in one domain, which serves as a powerful inspiration for modeling a similar process in the high-dimensional, non-linear domain of LLMs. The strength of this framework lies not in a literal mapping of "frequency," but in using the underlying information-theoretic principles to generate novel, falsifiable predictions about LLM behavior.

#### 6.2. Testable Hypotheses

This framework leads to several concrete, testable hypotheses:

1.  **Information Sufficiency Threshold:** There exists a quantifiable information threshold for prompts. Hallucination rates for a given query will be inversely correlated with the mutual information between the prompt and the target concept. This can be tested by systematically ablating contextual information from prompts and measuring the onset of hallucinations.
2.  **Trajectory Divergence as a Predictor:** The tendency of a prompt to cause hallucinations can be predicted *before* generation by measuring the divergence of latent state trajectories. By injecting small perturbations into the initial prompt embedding and tracking the separation of the resulting trajectories through the layers, a high divergence rate would predict a high probability of Semantic Drift.
3.  **CoT as a Variance Reducer:** Chain-of-Thought and other structured reasoning techniques reduce hallucinations by constraining trajectory variance. We predict that the layer-by-layer variance of hidden states for a given prompt will be significantly lower with CoT than without, corresponding to a more stable path to convergence.
4.  **Energy-Hallucination Correlation:** The computational energy (or a proxy like FLOPs) required for inference will correlate with hallucination rates for ambiguous tasks. Prompts that induce Semantic Drift will lead to longer or more computationally intensive trajectories (e.g., requiring more speculative decoding steps), which can be measured.

## Supporting Documents & Roadmap

- `docs/GLOSSARY.md` – definitions for semantic drift, Nyquist thresholds, conceptual manifolds, and other recurring terms.
- `docs/VALIDATION_CHECKLIST.md` – pre-flight list covering information sufficiency, attention diagnostics, redundancy strategy, and energy audits.
- `CHANGELOG.md` – chronological summary of repository updates.
- `paper/draft.md` – submission-ready skeleton that cites the generated figures.
- `CITATION.cff` – citation metadata for referencing this work.
- `scripts/semantic_redundancy_metric.py` – standalone script that regenerates the real-prompt redundancy heatmap and CSV metrics.

Upcoming milestones:

1. Harden the notebook harness with automated linting/tests (`qa-figures` TODO).
2. Integrate the new figures and hypotheses into a consolidated paper draft (`paper-draft` TODO).
3. Expand empirical coverage with larger open-weight models once compute becomes available.

### How to Explore This Repo

1. **Skim the TL;DR** (above) to orient yourself.
2. **Read the README sections** on geometry, redundancy, and meta-manifolds if you want the story version.
3. **Check `docs/GLOSSARY.md`** whenever a term feels too grandiose—we keep it plain there.
4. **Run the notebooks** in `experiments/` to see each intuition play out (they’re lightweight and annotated).
5. **Browse `paper/draft.md`** if you prefer the formal write-up; it mirrors the README but in paper form.

We’re deliberately keeping the tone conversational here—the math matters, but so does being honest about the open questions.

### Next Research Directions

1. **Semantic redundancy metric in real prompts** – estimate the mutual information (or cosine alignment) between prompt tokens and target concepts to validate the redundancy/noise heatmap on open-weight LLMs.
2. **Geometric alignment measurements** – probe hidden states to quantify how close different prompts come to the gold latent manifold (e.g., via projection scores, attention entropy, PCA trajectories).
3. **Matched-filter prompting** – test scaffolded prompts (explicit sections, “ignore duplicates” instructions) as filters that preserve redundancy while dampening noise, and measure hallucination reduction.
4. **Control-theoretic framing** – formalize prompting as a control problem in a dynamical system with limited bandwidth, drawing on controllability/observability results to derive new bounds on when trajectories can reach the desired attractor. Concretely:
   - Measure how different prompts move hidden states along known concept directions (controllability proxy).
   - Monitor entropy/variance of logits or attention to detect observability loss mid-generation.
   - Prototype feedback prompts (self-checks, scaffolded instructions) that act as controllers keeping the trajectory in-bounds.

## Conclusion

We have reframed the problem of LLM hallucinations from an unpredictable flaw to a principled phenomenon of **reconstruction failure**. Our model of **Semantic Drift**, inspired by foundational principles of information theory and signal processing, posits that hallucinations are the result of an information deficit in the prompt, leading to unstable trajectories in the model's latent space.

By synthesizing concepts from Fourier uncertainty, the Free Energy Principle, and Shannon's noisy channel theorem, we have constructed a coherent and testable framework. This approach provides a new vocabulary for describing LLM failures and, more importantly, a clear research program for making them more reliable. The path to robust AI may lie not just in scaling data and parameters, but in a deeper understanding of the fundamental principles of information that govern them.


### Project context

This repository is one focused case study within a broader research program on information structures for reliable LLMs. The larger effort explores how explicit structures—grounding via retrieval/graphs with provenance, structured redundancy and prompt gating, lightweight verification/feedback, and selective accept/verify/abstain policies—reduce uncertainty and factual error across tasks. The experiments and metrics here (redundancy ρ, alignment a, drift Δ, entropy E, hallucination rate h) are designed to be reusable in that larger context and compatible with future modules (e.g., KG/RAG‑grounded QA and detection‑correction loops).