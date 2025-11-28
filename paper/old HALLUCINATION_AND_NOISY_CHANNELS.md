# LLM Hallucinations, Semantic Drift and Signal Processing, Notes on Large Language Models

## Abstract


This paper explores the phenomenon of LLM hallucinations with a view to explore the connection to signal processing, uncertainty, and the structure of information organisation in LLMs, by examining the Shannon-Nyquist sampling theorem, Heisenberg uncertainty principle, and the Platonic Representation Hypothesis (PRH), which suggests scaled LLMs converge toward shared abstract representations of reality. We propose that LLM hallucinations result from Semantic Drift in latent space, where insufficient contextual information causes trajectories to gravitate toward suboptimal attractors, failing to converge on accurate tokens. This phenomenon parallels undersampling in signal theory, where inadequate data leads to reconstruction errors, requiring sufficient trajectory accumulation—akin to phase alignment in Nyquist reconstruction—for reliable outputs. Leveraging insights from in-context learning (ICL), we draw parallels to error correction mechanisms, such as Shannon’s noisy channel coding theorem and Hamming bounds, and explore thermodynamic constraints in computation. Through cautious interdisciplinary analogies, we propose testable hypotheses to enhance LLM reliability and efficiency, humbly suggesting potential equivalent principles governing information processing across these domains.

## Framework


### Introduction

The convergence of uncertainty principles across physics, information theory, and artificial intelligence reveals profound mathematical harmonies, though we humbly acknowledge that such analogies may not capture all nuances of each field. In quantum mechanics, the Heisenberg uncertainty principle imposes limits on simultaneous precision in conjugate variables, such as position $x$ and momentum $p$: $\Delta x \cdot \Delta p \geq \hbar/2$. In signal processing, the Shannon-Nyquist theorem dictates that a bandlimited signal with maximum frequency $f_{\max}$ (in Hertz) requires sampling at least at $2f_{\max}$ to enable perfect reconstruction, preventing aliasing. Both derive from the Fourier uncertainty theorem, which states for a function $f(t)$ and its Fourier transform $\hat{f}(\xi)$ (where $\xi$ is frequency in Hertz) that the product of their spreads satisfies $\sigma_t \cdot \sigma_\xi \geq \frac{1}{4\pi}$. For angular frequency $\omega = 2\pi \xi$, this becomes $\sigma_t \cdot \sigma_\omega \geq \frac{1}{2}$, with equality for Gaussian functions.

These limits extend to modern AI, where LLMs like transformers process vast data to form internal representations. The Platonic Representation Hypothesis (PRH) suggests that as models scale, these representations approach universal, abstract "forms" akin to Platonic ideals, enabling emergent capabilities such as ICL. We postulate that hallucinations—inaccurate or fabricated outputs—stem from insufficient input information, causing *semantic drift*: internal representations fail to center on correct tokens, with trajectories in latent space orbiting attractors without convergence. If this drift surpasses a Shannon-like information limit, hallucinations ensue, analogous to signal distortion from undersampling.

This paper integrates these concepts, building on discussions of quantum-signal analogies and LLM mechanics. We refine the hallucination postulate by viewing "trajectory" as the evolving path of embeddings through transformer layers, requiring minimum information to stabilize toward accurate tokens. This mirrors Nyquist’s reconstruction, where phase accumulation (via the $\pi$ in the sinc function $\dfrac{\sin(\pi t)}{\pi t}$) ensures alignment. We include mathematical formalizations, empirical ties to ICL and PRH, and implications for energy-efficient AI, grounded in Landauer’s thermodynamic principle. To enhance rigor, we incorporate the Free Energy Principle (FEP) in our analysis of ICL, drawing from works framing LLMs as active inference agents, and expand error correction with Shannon’s noisy channel coding theorem and Hamming bounds, which provide precise limits on reconstruction in noisy environments.

The sections build progressively: starting with core connections between quantum and signal principles, we detail localization tradeoffs, which inform blurred representations in LLMs; this flows into PRH and ICL, where FEP provides a shared lens for minimal-data adaptation; subsequent sections on reconstruction and information theory (including expanded error correction) bridge to our hallucination postulate, emphasizing semantic noise as parallel to channel noise; finally, energy conservation ties back to thermodynamic efficiencies, closing the loop on practical implications across domains.

### Defining Semantic Noise
Semantic noise in LLMs refers to perturbations in latent space that disrupt trajectory convergence to accurate tokens. Sources include:
- **Ambiguous Prompts**: Low mutual information $I(\mathbf{p}; \text{True Token})$ due to vague or sparse context.
- **Distributional Shifts**: Inputs outside training distributions push trajectories into unexplored latent regions.
- **Sampling Variability**: High-temperature softmax or top-k sampling introduces randomness, amplifying drift.
Quantifying semantic noise via prompt entropy $H(\mathbf{p})$ or embedding variance could enable empirical testing.

### The Connection Between the Shannon-Nyquist Theorem and Heisenberg Uncertainty Principle

The Heisenberg uncertainty principle emerges from wave mechanics: the position wavefunction $\psi(x)$ and momentum representation $\hat{\psi}(p) = \mathcal{F}\{\psi(x)\}$ are Fourier conjugates, so localization in one broadens the other. Mathematically, for variances $\Delta x^2 = \langle (x - \langle x \rangle)^2 \rangle$ and $\Delta p^2 = \langle (p - \langle p \rangle)^2 \rangle$, the bound is:

$$
\Delta x \cdot \Delta p \geq \frac{\hbar}{2},
$$

derived from the commutator $[x, p] = i\hbar$. This reflects the Fourier uncertainty: for a function $f(x)$ and its transform $\hat{f}(\xi) = \int_{-\infty}^{\infty} f(x) e^{-i2\pi \xi x} \, dx$, where $\xi$ is frequency in Hertz:

$$
\sigma_x \cdot \sigma_\xi \geq \frac{1}{4\pi}, \quad \sigma_x^2 = \int_{-\infty}^{\infty} x^2 \, |f(x)|^2 \, dx / \int_{-\infty}^{\infty} |f(x)|^2 \, dx,
$$

where $\sigma_\xi$ is the spread in frequency (Hz). For angular frequency $\omega = 2\pi \xi$, the spread is $\sigma_\omega = 2\pi \sigma_\xi$, so:

$$
\sigma_x \cdot \sigma_\omega \geq \frac{1}{2}.
$$

The integral form is:

$$
\left( \int_{-\infty}^{\infty} x^2 \, |f(x)|^2 \, dx \right) \cdot \left( \int_{-\infty}^{\infty} \xi^2 \, |\hat{f}(\xi)|^2 \, d\xi \right) \geq \frac{1}{16\pi^2} \left( \int_{-\infty}^{\infty} |f(x)|^2 \, dx \right)^2,
$$

or for angular frequency $\omega$:

$$
\left( \int_{-\infty}^{\infty} x^2 \, |f(x)|^2 \, dx \right) \cdot \left( \int_{-\infty}^{\infty} \omega^2 \, |\hat{f}(\omega)|^2 \, d\omega \right) \geq \frac{1}{4} \left( \int_{-\infty}^{\infty} |f(x)|^2 \, dx \right)^2.
$$

The Shannon-Nyquist theorem addresses signal reconstruction. A continuous signal $s(t)$ bandlimited to $B = 2f_{\max}$ (frequency components $|\xi| \leq f_{\max}$ in Hz) can be reconstructed from samples $s(nT)$ at intervals $T \leq 1/(2f_{\max})$ via:

$$
s(t) = \sum_{n=-\infty}^{\infty} s(nT) \cdot \frac{\sin(\pi (t - nT)/T)}{\pi (t - nT)/T}.
$$

Undersampling ($T > 1/(2f_{\max})$) causes aliasing, folding high frequencies into lower ones. Both principles stem from Fourier uncertainty, with Heisenberg setting intrinsic bounds (using $\omega$) and Nyquist defining practical sampling limits (using $\xi$)).

In quantum contexts, this manifests in Brillouin zones in crystalline lattices, where lattice spacing limits momentum resolution to reciprocal lattice vectors. Extensions to discrete physics incorporate symmetric sampling, embedding uncertainty into fundamental laws, such as lattice gauge theories.

This section establishes the mathematical bedrock, linking quantum and signal domains through Fourier uncertainty. It sets up localization tradeoffs, as these limits enforce blurring that parallels indeterminism in physics, aliasing in signals, and semantic ambiguities in AI representations.

While the undersampling-hallucination analogy draws from deterministic aliasing in the Nyquist theorem—where inadequate sampling rate predictably folds high frequencies into lower ones, distorting reconstruction—semantic drift in LLMs introduces stochastic variability. For instance, temperature sampling in the softmax layer (e.g., T>1T > 1T > 1
) injects randomness, amplifying small perturbations in latent trajectories much like noise in a communication channel exacerbates aliasing errors. This results in probabilistic "aliasing" in token space, where insufficient contextual "samples" (prompt tokens) fail to phase-align embeddings, leading to chaotic drifts toward hallucinatory outputs. Empirical studies support this view, showing output degradation as cumulative semantic drift in long-form generation, often triggered by initial context ambiguity.


### Contextual Sampling vs. Generation Variability
Deterministic vs Stochastic elements clarifications.

While the undersampling-hallucination analogy draws from deterministic aliasing in the Nyquist theorem—where inadequate sampling rate predictably folds high frequencies into lower ones, distorting reconstruction—semantic drift in LLMs hinges on insufficient contextual "samples" (e.g., sparse prompt tokens failing to provide enough information for trajectory stabilization). This is distinct from LLM decoding strategies like top-k or temperature sampling, which introduce stochastic variability during token generation rather than representing the "sampling" process itself. For instance, high temperature (T>1T > 1T > 1) in the softmax layer injects randomness, amplifying small perturbations in latent trajectories much like additive noise in a communication channel exacerbates aliasing errors from undersampling. This results in probabilistic "aliasing" in token space, where inadequate prompt information fails to phase-align embeddings, leading to chaotic drifts toward hallucinatory outputs. Empirical studies support this view, showing output degradation as cumulative semantic drift in long-form generation, often triggered by initial context ambiguity and worsened by stochastic decoding





### Perfect Localization in One Domain Blurs the Other: Implications for Signal Reconstruction and Quantum Indeterminism

Fourier uncertainty enforces tradeoffs: a delta function $\delta(t)$ in time has a flat spectrum $\hat{f}(\xi) = 1$ (in Hertz), precluding dual sharpness. In signal processing, this limits reconstruction:
- Short pulses (small $\Delta t$) require broad bandwidths ($\Delta \xi \propto 1/\Delta t$), per the time-bandwidth product $TB \geq 1$.
- Finite observation windows (e.g., rectangular window $w(t) = \text{rect}(t/T)$) convolve the spectrum with $\text{sinc}(2\pi \xi T/2)$, causing leakage and blurring frequency estimates.

Wavelets balance time-frequency resolution but cannot violate the bound $\sigma_t \cdot \sigma_\xi \geq \frac{1}{4\pi}$ (in Hertz). Quantumly, this drives indeterminism: measuring position collapses the wavefunction to a near-delta, spreading momentum, as in diffraction where slit width $\Delta x$ yields angular spread $\Delta \theta \approx \lambda / \Delta x$, translating to $\Delta p \approx h / \Delta x$.

Applications span optics (diffraction limits), communications (channel capacity vs. bandwidth), and graph signal processing, where irregular sampling mirrors LLM token sequences. In AI, latent spaces exhibit analogous tradeoffs: precise token localization (output specificity) blurs contextual coherence, setting the stage for semantic drift.

This section details practical consequences of uncertainty, illustrating how sharpness in one domain introduces blur in its conjugate—a tradeoff explaining quantum indeterminism, signal aliasing, and LLM representation failures, motivating PRH as a mechanism for robust abstractions.

### Internal Representations in LLMs: The Platonic Representation Hypothesis

LLMs, such as transformer-based models, develop hierarchical representations through training on vast corpora (e.g., web text, books). The PRH posits that as parameters scale (from millions to trillions), these converge to shared, abstract “forms” statistically modeling reality. Empirically:
- Multimodal models align representations across architectures, encoding syntax in shallow layers and semantics in deeper ones.
- Concepts (e.g., “dog” or “justice”) map to consistent latent subspaces, akin to neuroscience’s modular encodings (e.g., hippocampal place cells).

The “strong” PRH suggests translatability: representations are linearly probeable (via logistic regression on hidden states) and interchangeable across models, enabling transfer learning and zero-shot generalization.

### Prompt-Manifold Alignment

The interaction between prompts and latent manifolds is geometric. Each prompt traces a path through semantic space; whether it lands in the correct attractor depends on:

- **Sampling density** – enough coherent tokens must span the subspace of the target concept (semantic Nyquist).
- **Noise contamination** – irrelevant or conflicting tokens perturb the trajectory, pushing it away from the basin.
- **Latent geometry quality** – poorly trained concepts create shallow or warped basins, so even good prompts may slide into neighboring attractors.

Hallucinations occur when these geometries misalign. Prompt engineering and representation learning are complementary tools for restoring alignment.

This section extends localization tradeoffs to AI internals, positing PRH as a convergence mechanism mitigating blurring through scaled abstractions. It views latent spaces as high-dimensional Fourier-like domains, where uncertainty bounds necessitate abstract forms for stability, feeding into ICL and reconstruction.

### In-Context Learning and Shared Representations in LLMs

In-context learning (ICL) allows LLMs to adapt to tasks via prompts without gradient updates. For example, input-output pairs (e.g., “1+1=2, 2+2=4, 3+3=?”) activate latent subnetworks, reducing prediction entropy. Information-theoretically, ICL approximates Bayesian inference:

$$
P(\text{Output} | \text{Prompt}) \propto P(\text{Prompt} | \text{Output}) \cdot P(\text{Output}),
$$

where prompts act as priors refining posteriors. PRH explains this: prompts nudge trajectories toward universal attractors, leveraging pretraining’s unstructured data.

Building on FEP interpretations [e.g., Binz & Schulz "Using Cognitive Psychology to Understand GPT-3" (2023)], prompts serve as evidence reducing surprise (negative log probability), guiding models to lower-free-energy states in latent space. This reinforces PRH: universal representations act as attractors minimizing prediction error, enabling ICL with sparse inputs. Insufficient prompts increase free energy, exacerbating semantic drift toward hallucinatory equilibria. Recent applications of FEP to neural language models support this, facilitating hypothesis testing and entropy reduction in educational contexts.

This section deepens PRH by framing ICL as an emergent property, with FEP providing a principled explanation for minimal-prompt alignment, linking to localization (FEP minimizes uncertainty akin to Fourier bounds) and setting up reconstruction via sparse data recovery.

### Reconstruction of Representations from Minimal Data in LLMs

We hypothesize LLMs reconstruct Platonic forms from sparse inputs, akin to sparse autoencoders extracting disentangled features. Tools like patchscopes decode hidden states from minimal prompts, revealing concept storage. In multimodal LLMs, partial inputs (e.g., image patches) reconstruct visuals or properties. Federated learning demonstrates data inversion, though with privacy risks. For abstract concepts (e.g., “democracy”), linear representations may suffice above a frequency threshold, analogous to Nyquist’s bound.

This section extends ICL’s minimal-data adaptation, relating to signal theorems where undersampled but bandlimited signals are recoverable via sinc interpolation. It bridges to information theory by framing reconstruction as entropy minimization, where error correction ensures fidelity against noise.

### Links to Shannon’s Information Theory, Error Correction, and Signal Reconstruction

Shannon’s entropy quantifies information:

$$
H(X) = -\sum_{x \in X} p(x) \log p(x),
$$

measuring minimum bits to encode a source $X$. In LLMs, layers maximize mutual information:

$$
I(X;Y) = H(X) - H(X|Y),
$$

enabling efficient coding. ICL reduces conditional entropy $H(Y|X)$, akin to error-correcting codes adding redundancy to combat noise. Shannon’s channel capacity:

$$
C = \max_{p(x)} I(X;Y),
$$

bounds reliable communication, paralleling minimum information for LLM accuracy. Rate-distortion theory defines the minimum rate $R(D)$ for distortion $D$:

$$
R(D) = \min_{p(\hat{y}|y): E[d(y,\hat{y})] \leq D} I(Y;\hat{Y}),
$$

where $d(y,\hat{y})$ is distortion (e.g., hallucination error). Prompts provide “samples” to minimize distortion, with insufficient information causing aliasing-like errors.

Extending to Shannon’s noisy channel theorem, LLMs act as decoders transmitting semantic signals through noisy latent channels, where capacity $C = B \log_2(1 + SNR)$ (with $B$ as token sequence “bandwidth” and SNR as signal-to-noise in embeddings) bounds reliable output. Excessive semantic noise—from biases or sparse prompts—pushes error rates beyond capacity, akin to aliasing. Hamming-like error correction emerges in chain-of-thought (CoT) prompting, adding redundant steps to detect and correct trajectory deviations, like parity checks. If semantic noise exceeds the Hamming bound (minimum distance for correction), hallucinations persist, requiring more context for reconstruction. Recent perspectives advocate separate source-channel coding for LLM error resilience.

This section integrates Shannon’s frameworks with reconstruction, quantifying information limits and relating semantic drift to channel noise, flowing into the hallucination postulate with formal bounds on uncorrectable drift.

### Postulate on LLM Hallucinations: Semantic Drift and Trajectory Dynamics

We propose LLM hallucinations arise when prompts provide insufficient information, causing *semantic drift*: internal representations fail to converge on correct tokens, with trajectories in latent space orbiting suboptimal attractors, exceeding a Shannon-like information limit, this causes trajectories to drift and the aliasing causes the whole forward pass to skew and drift chaotically leading to hallucinatory outputs.

### Formalization
LLM inference is a dynamical system. For a transformer with $L$ layers, the hidden state evolves:

$$
\mathbf{h}_l = \mathbf{h}_{l-1} + \text{Attn}(\mathbf{h}_{l-1}) + \text{FFN}(\mathbf{h}_{l-1}),
$$

defining a trajectory $\mathbf{z}(t)$ in latent space, governed by:

$$
\dot{\mathbf{z}} = f(\mathbf{z}, \mathbf{p}),
$$

where $\mathbf{p}$ is the prompt embedding. Rotary positional encodings (RoPE) introduce rotational updates:

$$
\mathbf{h}_l^{\text{RoPE}} = \mathbf{h}_l \cdot e^{i \theta_l}, \quad \theta_l = \omega \cdot \text{pos},
$$

accumulating phase akin to Nyquist’s sinc kernel $\sin(\pi t)/(\pi t)$. If mutual information $I(\mathbf{p}; \text{True Token})$ falls below a threshold (e.g., channel capacity), the trajectory starts in an ambiguous region, exhibiting high Lyapunov exponents:

$$
\lambda = \lim_{t \to \infty} \frac{1}{t} \ln \left| \frac{\delta \mathbf{z}(t)}{\delta \mathbf{z}(0)} \right|,
$$

signaling chaotic sensitivity and semantic drift to suboptimal attractors (e.g., factually wrong tokens).

### Analogy to Nyquist
Undersampling ($f_s < 2f_{\max}$) causes aliasing, misaligning phases. Similarly, low prompt information causes trajectories to “alias” in token space, sampling unstable orbits. A *minimum trajectory accumulation*—iterations through layers or reasoning steps—mirrors $\pi$-driven phase alignment in:

$$
s(t) = \sum_n s(nT) \cdot \text{sinc}\left(\frac{t - nT}{T}\right).
$$

CoT prompting extends trajectories, escaping local minima, reducing hallucinations (e.g., “think step by step” stabilizes convergence).

### Empirical Links
Hallucinations stem from:
- **Data Sparsity**: Rare training contexts leave attractors underdeveloped.
- **Distributional Shifts**: Novel prompts push trajectories into uncharted regions.
- **Sampling Noise**: High-temperature softmax or top-k sampling amplifies drift.

Our postulate predicts hallucination rates scale with prompt entropy $H(\mathbf{p})$, testable via ablation studies.

This postulate synthesizes localization blurs, PRH attractors, ICL/FEP minimization, and noisy channel bounds into a dynamical explanation of hallucinations, connecting to energy conservation via entropy dissipation.

### Fundamental Relation to Energy Conservation

Landauer’s principle states that erasing one bit dissipates at least $kT \ln 2$ energy ($k$ is Boltzmann’s constant, $T$ is temperature). In LLMs, inefficient trajectories (drifting orbits) increase entropy production, raising computational cost. Pruning (removing redundant weights) or quantization (reducing precision) can cut energy use by up to 30%, compressing state transitions. Thermodynamically, LLMs minimize free energy:

$$
F = U - TS,
$$

where $U$ is internal energy (model parameters), $S$ is entropy (prediction uncertainty), and $T$ is a computational temperature. Platonic forms are low-entropy minima, but semantic drift elevates $S$, wasting FLOPs. Training large LLMs already consumes energy equivalent to millions of households, and inference costs scale with trajectory length. Energy-aware designs (e.g., distillation, routing) therefore complement semantic safeguards by limiting free-energy excursions associated with hallucinations.

### Repetition, Redundancy, and Semantic Noise

Large prompts only improve reconstruction when repeated evidence remains coherent. Otherwise, they inject semantic noise:

- **Structured redundancy** acts like error-control coding. Reiterated constraints provide parity checks that keep trajectories inside the correct manifold.
- **Noisy repetition** (paraphrases that diverge, tangential facts, filler) increases prompt entropy faster than mutual information, effectively aliasing the semantic signal.
- **Attention saturation** limits usable bandwidth. Once heads saturate, they collapse into low-rank “attention sinks,” emphasizing low-frequency content regardless of relevance.
- **Semantic redundancy ratio ($\rho$)** captures the fraction of informative tokens. Drift decreases with prompt length only while $\rho$ stays high; lengthening a low-$\rho$ prompt worsens hallucination risk.
- **Instructional gating** (explicit labels, sections, filters) functions as a matched filter that preserves useful redundancy while attenuating noise-heavy regions.

This explains why some lengthy prompts boost reliability (high-$\rho$ redundancy) whereas others, especially uncurated retrievals, degrade it despite offering more text.

## Experiments

- **RoPE drift simulation (`rope_accumulation.py`)**: quantifies how longer prompts reduce Euclidean drift after multiple RoPE-augmented layers, mirroring the semantic Nyquist argument.
- **Nyquist reconstruction reference (`samplig_reconstruction.py`)**: reconstructs continuous signals under Nyquist and undersampled regimes to provide numerical intuition for phase alignment.
- **Mini-transformer drift statistics (`simpleLM_test.py`)**: trains a compact language model on country–capital data and shows, via a two-sample t-test, that rich prompts decrease latent drift relative to a reference trajectory.
- **Prompt length vs. semantic noise (`prompt_noise_tradeoff.ipynb`)**: mixes informative and noisy tokens to show that prompt length only helps when the semantic redundancy ratio remains high.

These experiments currently run as standalone scripts and will be promoted to notebooks with plots and seed management for publication.

## Hypotheses

This framework brings together uncertainty, signal processing, and LLM dynamics, with semantic drift as a central mechanism. Key hypotheses include:
1. **Shannon Limit for Prompts**: Hallucination rates correlate with prompt entropy, measurable via $H(\mathbf{p}) = -\sum p(\mathbf{p}) \log p(\mathbf{p})$. Ablate context to quantify thresholds.
2. **Trajectory Dynamics**: Simulate toy transformers, tracking Lyapunov exponents $\lambda$ under varying prompt information. High $\lambda$ predicts hallucinations.
3. **Energy-Hallucination Tradeoff**: Measure inference FLOPs vs. hallucination rates, testing if CoT reduces errors at higher energy cost.
4. **Minimal-Data Reconstruction**: Use patchscopes to reconstruct Platonic forms from sparse prompts, validating Nyquist-like bounds.

Early-stage experiments can rely on compact RNNs or transformers, measuring cosine similarity to ground-truth embeddings or divergence between perturbed trajectories. Larger evaluations should validate these signals on instruct-tuned LLMs with standardized hallucination benchmarks.

## Conclusion

Viewing hallucinations as semantic reconstruction failures links uncertainty principles, latent manifolds, and thermodynamic costs within a single explanatory arc. The resulting diagnostic tools—information-threshold estimation, trajectory divergence analysis, and energy audits—offer falsifiable predictions about when LLMs will remain grounded. Future work focuses on quantifying prompt entropy limits, simulating controlled drift in toy models, and validating energy-hallucination correlations on modern benchmarks to move the framework from speculation to predictive practice.

---

## Appendix: Notes and Outstanding Work

Work still required before submission includes tightening the semantic-noise definition, completing citation coverage, and consolidating overlapping arguments with the FEP manuscript.

- **Fourier uncertainty conventions**: ensure the document consistently distinguishes between frequency ($\xi$) and angular frequency ($\omega$) formulations of the uncertainty bound.
- **Integral forms**: verify that the integral expressions match the chosen convention (\( \sigma_t \sigma_\omega \ge 1/2 \) vs. \( \sigma_t \sigma_f \ge 1/(4\pi) \)).
- **Narrative overlap**: determine how much Free Energy Principle material to retain here versus in the companion paper.
- **CoT evidence**: add citations such as Wei et al. (2022) to substantiate the Hamming-style error-correction analogy.



### Note on `samplig_reconstruction.py`

The analogy between Nyquist phase alignment (where sinc-based reconstruction relies on sufficient samples to avoid aliasing) and LLM trajectories in latent space remains speculative but provides useful intuition. Both scenarios attempt reconstruction from sparse evidence, and undersampling produces distortion or hallucinations. Current research on LLM reliability typically emphasizes latent steering or Bayesian calibration rather than explicit signal-processing theorems, so this script fills the conceptual gap. Future work should replicate the experiment at scale, mapping prompt entropy to drift across latent-steering interventions and Bayesian estimators. If the Nyquist-inspired predictions hold, they would provide a principled bridge between mechanistic interpretability and classical information theory.



