# A Modeling Framework for LLM Hallucinations

## Abstract

Hallucinations in large language models (LLMs) are modeled as **semantic reconstruction failures**. When prompts fall below a concept-dependent information threshold, latent trajectories drift away from the correct manifold, analogous to aliasing under the Shannon–Nyquist theorem. By combining Fourier uncertainty, the Free Energy Principle (FEP), and Shannon’s noisy-channel coding, we derive diagnostic metrics and error-control strategies (e.g., Chain-of-Thought (CoT) prompting) that predict and mitigate these failures. Toy experiments validate the framework via RoPE drift analysis, Nyquist reconstruction baselines, semantic threshold simulations, and redundancy-as-CoT channels.

## 1. Framework

### 1.1 Fourier and Information-Theoretic Roots
- Fourier uncertainty imposes $\sigma_t \sigma_\omega \ge 1/2$, creating a trade-off between temporal (prompt) localization and frequency (concept) fidelity.
- Shannon–Nyquist sampling requires at least twice the maximum semantic “frequency” to avoid aliasing; undersampled prompts inevitably misreconstruct the concept.

### 1.2 Conceptual Manifolds and Platonic Forms
- Training organizes latent space into attractor basins representing stable concepts (Platonic Representation Hypothesis).
- Semantic drift occurs when trajectories never reach or exit the intended basin due to insufficient evidence or conflicting cues.

### 1.2.1 Prompt vs. Latent Geometry
- Prompts themselves trace manifolds determined by their sampled evidence; their geometry can align (or misalign) with latent basins.
- Hallucinations emerge when the prompt manifold fails to intersect the target attractor due to undersampling, noise, or warped latent geometry.
- Ensuring alignment requires both better prompts (structured redundancy, low entropy) and well-trained latent manifolds.

### 1.2.2 In-Context Meta-Manifolds
- Once generation starts, the model builds a transient meta-manifold that includes both the prompt and its own outputs.
- Coherent interactions keep this meta-manifold aligned with the base concept basin; hallucinated or noisy outputs warp it, causing further drift.
- Stabilizing the meta-manifold (e.g., via self-consistency checks or retrieval grounding) is crucial for multi-turn reliability.

### 1.2.3 Control-Theoretic Framing
- View prompts as control inputs driving a discrete dynamical system; latent manifolds are reachable subsets.
- Semantic Nyquist thresholds become controllability limits: without enough coherent control inputs, the desired basin cannot be reached.
- Generated tokens are observations. If their entropy spikes or they contradict prior context, observability is degrading and the controller (the prompt) must inject corrective signals.
- Chain-of-Thought, self-critique, and sectioned prompts can be interpreted as feedback controllers that keep the trajectory within safe bounds.

### 1.3 In-Context Learning and FEP
- Prompts act as evidence that lowers free energy; sparse prompts leave the model in high-entropy states prone to chaotic divergence.
- CoT adds redundant “parity checks,” guiding trajectories toward low-free-energy manifolds.

### 1.4 Semantic Drift Postulate
- Forward passes are discrete dynamical systems; drift correlates with Lyapunov-style divergence between nearby prompts.
- Context fatigue (excess tokens) introduces competing in-context attractors, mirroring long-context failure modes in recent RoPE studies.

### 1.5 Prompt Length, Redundancy, and Noise
- Length helps only when repeated evidence is coherent—structured redundancy acts like channel coding.
- Once the *semantic redundancy ratio* drops, additional tokens inject noise that saturates attention bandwidth and induces aliasing.
- Instructional gating (explicit sections, labels) behaves like a matched filter, preserving useful redundancy while suppressing noisy repetition.

## 2. Experiments

1. **RoPE phase accumulation** (`experiments/rope_accumulation.ipynb`): long prompts reduce Euclidean drift vs. short prompts.  
   ![RoPE Drift](../figures/rope_drift.png)
2. **Nyquist reconstruction control** (`experiments/sampling_reconstruction.ipynb`): perfect vs. undersampled sinc interpolation.  
   ![Nyquist Comparison](../figures/nyquist_comparison.png)
3. **Simple Transformer drift benchmark** (`experiments/simpleLM_drift.ipynb`): richer prompts yield statistically lower drift relative to a reference trajectory.
4. **Semantic Nyquist ablation** (`experiments/prompt_ablation_threshold.ipynb`): reconstruction error forms a sharp boundary at twice the concept complexity.  
   ![Semantic Threshold](../figures/semantic_threshold.png)
5. **Prompt length vs. semantic noise** (`experiments/prompt_noise_tradeoff.ipynb`): shows that longer prompts help only when semantic redundancy remains high; otherwise reconstruction error plateaus or worsens.  
   ![Prompt Noise Heatmap](../figures/prompt_noise_heatmap.png)
6. **Semantic redundancy on real prompts** (`experiments/semantic_redundancy_real_prompts.ipynb`): TF-IDF cosine similarities yield a redundancy ratio ordering `relevant > direct > noisy > contradictory`, matching observed hallucination risk.  
   ![Semantic Redundancy](../figures/semantic_redundancy_heatmap.png)
7. **Geometric alignment metrics** (`experiments/geometric_alignment_metrics.ipynb`): Chain-of-Thought and scaffolded prompts show higher cosine alignment with the reference manifold than direct or noisy prompts.
8. **Control diagnostics** (`experiments/control_diagnostics.ipynb`): scatter plots of control projection vs. logit entropy demonstrate that feedback prompts are both controllable and observable, whereas noisy prompts fail both tests.  
   ![Control Observability](../figures/control_observability_scatter.png)
9. **CoT as redundancy** (`experiments/cot_vs_direct_channel.ipynb`): repeating reasoning tokens reduces answer error rates on a noisy channel.

## 3. Testable Hypotheses

1. **Prompt information threshold** – hallucination rate is inversely correlated with prompt entropy or mutual information.
2. **Trajectory divergence predictor** – latent divergence measured pre-decoding forecasts hallucination probability.
3. **CoT variance reduction** – CoT reduces hidden-state variance and acts as semantic error correction.
4. **Energy–hallucination coupling** – prompts inducing drift consume more FLOPs / energy, measurable via inference profiling.

## 4. Discussion and Conclusion

The semantic reconstruction view unifies aliasing, latent manifolds, event-driven reasoning, and thermodynamic costs. Lightweight experiments already match theoretical predictions; future work targets larger open-weight models, automated drift diagnostics, and integration with human-in-the-loop evaluation.

## 5. Research Directions

1. **Semantic redundancy metrics** – Estimate prompt mutual information or cosine alignment to quantify redundancy/noise ratios on real LLM prompts, validating the heatmap behavior beyond toy models.
2. **Geometric alignment probes** – Measure how closely different prompts approach the gold latent manifold using projection scores, attention entropy, or PCA trajectories.
3. **Matched-filter prompting** – Evaluate scaffolded prompts (explicit labeling, “ignore duplicates” gating) as semantic filters that preserve redundancy while suppressing noise.
4. **Control-theoretic formalization** – Treat prompting as controlling a bandwidth-limited dynamical system and apply controllability/observability theory to derive bounds on when trajectories can reach the desired attractor.

## References

1. Wei et al., 2022 – *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*.
2. Su et al., 2024 – *RoPE: Rotary Position Embeddings for LLMs*.
3. Xiong et al., 2025 – *DoPE: Denoising Rotary Position Embedding* (arXiv:2511.09146).
