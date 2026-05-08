# Research leads under review

This document tracks papers that appear relevant to the Hallucinations in Noisy Channels framework but need careful checking before they move into the main evidence file. A paper belongs here when it suggests a mechanism, tool, or experiment that could support the framework, while the exact claim still needs reproduction or closer reading.

The goal is to keep the research process honest. We separate promising leads from confirmed support, record caveats early, and define the work needed before a citation becomes part of the main argument.

Last link check: 2026-05-08.

## Status labels

- **Lead**: The paper appears relevant, and the connection has not been checked in detail.
- **Checked**: The paper has been read enough to identify the exact claim, tool, and caveat.
- **Reproduction needed**: The paper should be tested locally before it supports a strong claim.
- **Ready for evidence file**: The relevant result has been reproduced or checked deeply enough to cite with confidence.

## Current leads

### GOAT: trainable attention priors

**Source status:** Checked, reproduction needed.

**Paper:** Litman, E., & Guo, G. (2026). *You Need Better Attention Priors.* arXiv:2601.15380. https://arxiv.org/abs/2601.15380

**Code:** `elonlit/goat`, Generalized Optimal Transport Attention with Trainable Priors. https://github.com/elonlit/goat

**Package:** `goat-attention` on PyPI. https://pypi.org/project/goat-attention/

**Relevant claim for this framework:** GOAT frames scaled dot-product attention as entropic optimal transport. Standard attention corresponds to a transport problem with an implicit uniform prior over key positions. GOAT replaces that prior with a learned continuous prior and keeps compatibility with optimized attention kernels.

**Why it matters here:** GOAT gives a precise attention-level version of prior relaxation. The attention solution has the form $p^*=\mathrm{softmax}(s/\tau+\log \pi)$, where $s$ contains content scores, $\tau$ controls entropy pressure, and $\pi$ is the attention prior over key positions. When the content scores carry little discriminating signal, or when score differences are small relative to $\tau$, $p^*$ moves toward $\pi$. If the scores are flat, then $p^*=\pi$ exactly. This is a concrete token-matching mechanism for temperature-driven relaxation toward a prior. GOAT also gives a mathematical treatment of attention sinks, which connects to sink-limited context capacity.

**HNC interpretation:** GOAT operationalizes local form-prior pressure inside attention. The entropy term supplies attention-level equilibration pressure, the prior $\pi$ supplies the local key-position prior, and weak content signal causes attention to relax toward that prior. This matches the HNC thermalization story at the token-matching level. It does not by itself prove the output-level form prior or the full thermodynamic theorem.

**Framework mapping:**

1. **Capacity violation:** GOAT uses Shannon entropy and KL divergence inside attention. It does not directly use Shannon's noisy-channel coding theorem. When the content signal in $s$ is weak, the attention distribution moves toward the prior. This gives a local mechanism that can be tested against the HNC capacity-violation story.
2. **Thermodynamic equilibration:** The entropy term in the attention objective gives a concrete attention-level pressure toward higher-entropy matching. The prior $\pi$ defines the distribution that attention relaxes toward under weak content scores.
3. **Sink-limited capacity:** GOAT gives a formal account of attention sinks and includes a key-only sink bias. This is directly relevant to HNC sink severity and late-context degradation.
4. **Optimal noise:** The attention temperature $\tau$ controls the balance between content scores and entropy pressure. This creates a direct experiment for the HNC U-shaped noise prediction.
5. **Information conservation diagnostics:** GOAT exposes attention entropy and prior dominance as measurable layer-level signals. These signals may predict unsupported output content, but that connection needs a hallucination benchmark.

**Tool for our work:** The `goat-attention` PyTorch package exposes `GoatAttention` with Fourier positional features, relative-position priors, and a key-only sink bias. This can test whether learned priors reduce sink severity, improve long-context use, or reduce hallucination on weak-knowledge prompts.

**GOAT-specific experiment queue:**

1. **Thermodynamic equilibration test:** Compare baseline attention and GOAT on capacity-stressed prompts. Measure attention entropy, prior dominance, unsupported claims, and refusal accuracy.
2. **Sink-limited capacity test:** Measure attention mass on beginning-of-sequence and early tokens with and without GOAT's sink bias. Correlate this with late-context retrieval and multi-hop accuracy.
3. **Temperature sweep:** Vary $\tau$ and measure hallucination rate, self-consistency, and recovery from misleading context. Test whether the curve has a minimum at an intermediate value.
4. **Information conservation diagnostic:** Use per-layer attention entropy and prior dominance as early-warning signals for outputs that contain unsupported claims.
5. **Prior-shaping test:** Train or initialize priors for factual retrieval contexts and test whether this reduces composite answers or Kolmogorov garbage.

**Caveats:** GOAT does not prove the HNC form-prior theorem, the output-level thermodynamic theorem, or the information conservation claim. It supplies an attention-level mechanism that may instantiate part of the theory. Treat it as a stronger mechanism lead than the others, but still require experiments before moving it into confirmed evidence.

### Co-Tok: compute optimal tokenization

**Source status:** Checked, reproduction needed.

**Paper:** Limisiewicz, T., Pagnoni, A., Iyer, S., Lewis, M., Mehta, S., Liu, A., Li, M., Ghosh, G., & Zettlemoyer, L. (2026). *Compute Optimal Tokenization.* Project PDF: https://co-tok.github.io/paper.pdf

**Project page:** https://co-tok.github.io/

**Archival link:** arXiv:2605.01188. https://arxiv.org/abs/2605.01188

**Code:** No official implementation repository verified yet. The project page appears to be hosted through GitHub Pages, and it should be treated as a project page unless an implementation repository is linked.

**Relevant claim for this framework:** Co-Tok studies how token compression rate, measured as average bytes per token, affects scaling behavior. The paper reports that compute-optimal data size scales better in bytes than in tokens, and that the optimal compression rate changes with compute and language.

**Why it matters here:** Tokenization is the first source-coding step before the model performs matching, reconstruction, and transmission. A poor compression rate may reduce useful input structure or create inefficient sequence lengths. That makes Co-Tok relevant to HNC capacity questions because the effective query depends on how the source was encoded before attention sees it.

**Tool for our work:** The paper uses Byte Latent Transformer style latent tokenization and entropy-threshold segmentation to control compression rate. This creates a way to test whether upstream compression changes hallucination rates, context crowding, or sink severity.

**Framework mapping:**

1. **Capacity violation:** Co-Tok defines compression rate as average bytes per token and shows that validation loss depends on this rate. This gives a source-coding variable that can be tested against HNC topic capacity. The paper measures loss, so the hallucination-capacity connection remains our experiment to run.
2. **Information conservation:** Off-optimal compression may remove or obscure recoverable input structure before attention and reconstruction begin. HNC can test whether this increases unsupported output content or excess complexity proxies.
3. **Thermodynamic equilibration:** Entropy-threshold segmentation gives a controlled way to change the strength of the input signal. If compression is too aggressive, then useful boundaries or high-entropy regions may be missed. HNC can test whether weaker input signal increases relaxation toward form-prior output.
4. **Sink-limited capacity:** Changing compression rate changes sequence length and the structure seen by attention. This can test whether upstream compression changes sink severity and late-context retrieval.
5. **GOAT complement:** Co-Tok gives an upstream source-coding control, while GOAT gives an attention-prior control. Together they can test whether source encoding and attention priors jointly reduce context failure.

**Co-Tok-specific experiment queue:**

1. **Compression-hallucination curve:** Sweep compression rate in a BLT-style or subword setup. Measure factual accuracy, hallucination rate, self-consistency, and unsupported-claim count on capacity-stressed prompts.
2. **Source-coding capacity test:** Keep facts fixed and change only encoding granularity. Measure whether the effective query becomes easier or harder to reconstruct.
3. **Sink severity vs. compression:** Track attention mass on beginning-of-sequence and early tokens across compression rates. Compare late-context retrieval at each rate.
4. **Cross-language capacity test:** Use languages with different byte parity. Test whether mismatched compression rates increase hallucination or refusal errors.
5. **Co-Tok plus GOAT ablation:** Compare baseline tokenization plus vanilla attention, Co-Tok-style compression alone, GOAT alone, and the combined setup. The outcome should be measured with hallucination-specific metrics.

**Caveats:** Co-Tok measures language modeling loss and scaling behavior. It does not directly measure hallucination. Any HNC claim needs an added hallucination benchmark, with the tokenizer or latent segmentation treated as the independent variable.

### Generalization theory: signal channel and reservoir

**Source status:** Checked, reproduction needed.

**Paper:** Litman, E., & Guo, G. (2026). *A Theory of Generalization in Deep Learning.* arXiv:2605.01172. https://arxiv.org/abs/2605.01172

**Code:** No official code link verified yet.

**Relevant claim for this framework:** The paper describes generalization through an empirical neural tangent kernel decomposition. Output directions split into a signal channel, where error dissipates quickly, and a reservoir, where residual error can remain test-invisible. The paper also derives a population-risk objective that becomes an SNR preconditioner on top of Adam.

**Why it matters here:** The signal-channel and reservoir language is close to the HNC distinction between accessible knowledge and inaccessible or noisy structure. It may give a training-time analogue of HNC capacity failure: signal that enters useful channels generalizes, while noise or weak structure remains trapped in directions that do not transfer well.

**Tool for our work:** The proposed SNR preconditioner is a candidate optimizer intervention. It can test whether suppressing low-signal updates reduces memorization, improves robustness under noisy labels, or reduces hallucination after preference tuning.

**Framework mapping:**

1. **Hallucination threshold:** The signal-channel and reservoir split gives a training-time way to ask whether useful signal enters transferable directions. HNC can test whether weak topic signal later produces capacity-violation behavior at inference.
2. **Thermodynamic equilibration:** The reservoir may act as a high-dimensional holding region for noise or idiosyncratic memorization. This is relevant to form-prior drift, but it is not identical to the HNC output-level form prior until hallucination measurements are added.
3. **Information conservation:** The population-risk objective and SNR quantities may help quantify noise that remains in transferable directions. HNC can test whether that noise predicts unsupported output content or excess complexity proxies.
4. **Optimal noise:** The SNR preconditioner suppresses low-signal updates. This gives a practical training intervention for testing whether better signal/noise control reduces hallucination after noisy supervision or preference tuning.
5. **Relation to GOAT and Co-Tok:** Co-Tok controls upstream source encoding, GOAT controls attention priors, and the SNR gate controls training updates. Together they define a source-coding, attention-matching, and training-dynamics experiment stack.

**Generalization-theory experiment queue:**

1. **SNR gate under noisy labels:** Implement the SNR preconditioner on a small controlled task with label noise. Compare memorization, calibration, and validation error against AdamW.
2. **Noisy preference fine-tuning:** Run a small DPO or preference-tuning task with noisy labels. Add HNC metrics: false-premise refusal, unknown-answer calibration, and unsupported-claim count.
3. **Reservoir diagnostic:** Track the population-risk or SNR signal during training. Test whether high noise in transferable directions predicts later hallucination or overconfidence.
4. **SNR plus GOAT test:** Combine SNR-gated training with GOAT attention. Compare against each method alone on capacity-stressed prompts.
5. **Grokking and sink severity:** In tasks with delayed generalization, measure whether SNR gating changes attention sink severity or late-evidence use.

**Caveats:** The paper studies generalization broadly. It does not directly study hallucination. The HNC connection is a hypothesis about shared signal/noise structure and needs experiments that measure hallucination-specific behavior.

### H-Neurons: hallucination-associated neurons

**Source status:** Checked, reproduction needed.

**Paper:** Gao, C., Chen, H., Xiao, C., Chen, Z., Liu, Z., & Sun, M. (2025). *H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs.* arXiv:2512.01797. https://arxiv.org/abs/2512.01797

**Code:** `thunlp/H-Neurons`. https://github.com/thunlp/H-Neurons

**Relevant claim for this framework:** The paper reports that a sparse subset of feed-forward neurons can predict hallucinated responses and that scaling those neurons changes over-compliance behavior.

**Why it matters here:** H-Neurons may provide a microscopic signature for the HNC claim that hallucination corresponds to movement away from grounded knowledge and toward form-prior behavior. The result is especially relevant to off-manifold drift and over-compliance.

**Tool for our work:** The repository provides a pipeline for collecting responses, extracting answer tokens, computing CETT neuron contributions, training a sparse logistic regression classifier, and intervening on selected neurons.

**Reproduction tasks:**

1. Reproduce the classifier on one small supported model and dataset.
2. Verify the label convention in the code: false answer tokens should receive label 1, and true answer tokens should receive label 0.
3. Test whether the selected neurons predict HNC-style hallucination categories, including capacity violation, false premise compliance, and fabricated entities.
4. Test whether intervention changes factual refusal behavior without harming normal QA accuracy.

**Caveats:** The paper contains a label inconsistency in Section 6.1.3. The prose and official code define hallucinated or false answer tokens as the positive class, but the displayed equation appears reversed. This must be checked locally before using the result as strong evidence.

## Cross-paper experiment queue

### Experiment 1: attention prior and sink severity

**Question:** Do trainable attention priors reduce attention sinks and improve late-context use?

**Methods:** Compare a baseline attention module with GOAT. Measure attention mass on early tokens, retrieval accuracy for late evidence, and hallucination rate on prompts where late context contradicts an earlier distractor.

**Framework target:** Sink-limited capacity, context crowding, and matching failure.

### Experiment 2: compression rate and hallucination

**Question:** Does token compression rate change hallucination rate when the same facts are supplied in context?

**Methods:** Use multiple tokenization or latent segmentation settings. Keep content fixed. Measure output accuracy, context length, self-consistency, and excess unsupported claims.

**Framework target:** Capacity violation, decompression failure, and information conservation.

### Experiment 3: SNR gating under noisy supervision

**Question:** Does suppressing low-signal updates reduce hallucination after noisy training?

**Methods:** Train or fine-tune with noisy labels or noisy preferences using AdamW and the SNR preconditioner. Evaluate false-premise refusal, unknown-answer calibration, and factual QA.

**Framework target:** Optimal noise, form-prior drift, and confidence-accuracy decoupling.

### Experiment 4: H-Neurons and off-manifold drift

**Question:** Do H-Neurons activate when outputs leave grounded knowledge and move into over-compliant form completion?

**Methods:** Reproduce H-Neuron extraction. Evaluate on HNC categories: absent knowledge, misleading context, context crowding, and fabricated entities. Compare neuron activation with embedding-distance or translation-fidelity proxies.

**Framework target:** Off-manifold drift, form-prior sampling, and geometric misalignment.

## Promotion rule

A paper should move from this file into `EVIDENCE_AND_SUPPORTING_LITERATURE.md` only when at least one of the following is true:

1. The exact claim has been reproduced locally.
2. The paper directly measures a quantity already used in HNC.
3. The paper supplies a method that we used in an HNC experiment.
4. The caveats are documented clearly enough that the citation cannot be mistaken for stronger evidence than it is.

Until then, these papers remain research leads under review.
