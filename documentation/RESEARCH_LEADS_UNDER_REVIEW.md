# Research leads under review

This document tracks papers that appear relevant to the Hallucinations in Noisy Channels framework but need careful checking before they move into the main evidence file. A paper belongs here when it suggests a mechanism, tool, or experiment that could support the framework, while the exact claim still needs reproduction or closer reading.

The goal is to keep the research process honest. We separate promising leads from confirmed support, record caveats early, and define the work needed before a citation becomes part of the main argument.

Last link check: 2026-05-10.

## Status labels

- **Lead**: The paper appears relevant, and the connection has not been checked in detail.
- **Checked**: The paper has been read enough to identify the exact claim, tool, and caveat.
- **Reproduction needed**: The paper should be tested locally before it supports a strong claim.
- **Ready for evidence file**: The relevant result has been reproduced or checked deeply enough to cite with confidence.

## Current leads

### Algorithmic compression via pretrained neural networks

**Source status:** Checked, claim binding needed.

**Paper:** Genewein, T., Grau-Moya, J., Wenliang, L. K., Orseau, L., & Hutter, M. (2026). *Algorithmic Compression via Pretrained Neural Networks.* Entropy, 28(6), 596. https://doi.org/10.3390/e28060596

**Code:** Review article. Implementation sources belong to the individual works it surveys when those works provide code.

**Relevant claim for this framework:** The article reviews the argument that sequential prediction and lossless compression are formally linked: a sequential predictor can be converted into a compressor through arithmetic coding, and minimizing log-loss improves compression. It also reviews memory-based meta-learning, where training a sequence model across a distribution of tasks can produce an amortized Bayesian predictor that adapts in context through hidden state or attention.

**Why it matters here:** This is direct foundation work for HNC's source-coding side. HNC says training stores usable structure in weights and inference reconstructs from that compressed source. Genewein et al. give a more formal basis for the first half of that claim: log-loss training over diverse sequential data can be read as training an algorithmic compressor or amortized Bayesian predictor. The article also gives HNC a precise way to discuss finite-model limits through the approximation or amortization gap.

**HNC interpretation:** The paper strengthens the learning-as-compression branch and the in-context learning branch. It supports the idea that a trained model can carry a compressed prior over tasks and use context to infer the current task. In HNC terms, the static codebook comes from the meta-training distribution, and the dynamic codebook comes from context-conditioned posterior-like adaptation.

The support condition matters. The article states that Bayesian guarantees depend on the target task lying within the support of the meta-training distribution and on realizability, capacity, data, and optimization. HNC can use this as a clean statement of source support: when the target task has weak or absent support under the learned mixture, the model's answer relies more on approximation behavior, architecture bias, retrieval, tools, or supplied context.

**Framework mapping:**

1. **Learning as compression:** Minimizing log-loss is equivalent to improving sequential compression under the prediction-compression correspondence.
2. **Static codebook:** The learned weights can be interpreted as an amortized predictor over the task mixture encountered during training.
3. **In-context learning:** Context helps identify the current task or source within that mixture.
4. **Capacity violation:** A task outside the learned support gives no Bayesian guarantee and should be treated as low source support under HNC.
5. **Matching failure:** Prompting can fail when the target task cannot be concentrated by a prefix or when the posterior geometry resists steering.
6. **Approximation gap:** Finite models differ from the ideal Bayesian or universal predictor because of capacity, finite data, and optimization limits.
7. **Agency boundary:** Passive prediction leaves intervention handling as a separate requirement. Agentic HNC claims should track the difference between conditioning on observations and acting in an environment.

**Useful quantities for HNC:**

1. **Sequential code length:** $-\log_2 \rho(x_{1:n}) + O(1)$, a practical link between prediction quality and compression.
2. **Amortization gap:** $\Delta_{\text{amort}}(x_{\lt t}) = D_{KL}(\xi(\cdot \mid x_{\lt t}) \,\|\, \pi_\theta(\cdot \mid x_{\lt t}))$, the gap between an exact Bayesian mixture and the neural approximation.
3. **Loss decomposition:** total loss can be decomposed into irreducible entropy, model-class regret against an ideal predictor, and approximation gap.
4. **Support condition:** task support under the pretraining or meta-training distribution controls whether the Bayesian account applies.

**Experiment queue:**

1. **Claim binding:** Add an `ApproximationGap` or `AmortizationGap` field to future theory schemas before using this paper as evidence for hallucination behavior.
2. **Support-stratified QA:** Label examples by whether source support is strong, weak recoverable, unsupported, or misleading, then measure whether errors follow the support condition.
3. **Compression proxy test:** Compare log-loss, code-length proxies, exact match, refusal quality, and unsupported-claim rate on the same items.
4. **Prompt steering test:** Construct tasks where the target is inside support, outside support, or a mixture of supported modes. Test whether prompts can concentrate behavior as predicted.
5. **Architecture gap test:** Compare LSTM-style, decoder-only transformer, and RAG-backed predictors on length generalization or OOD context to separate learned prior support from architecture bias.

**Caveats:** This article reviews foundational work on algorithmic compression, amortized Bayesian prediction, and universal prediction. The HNC use should stay precise: cite it for prediction-compression equivalence, amortized Bayesian interpretation of in-context learning, support limits, and approximation gaps. Hallucination-specific claims still need experiments that measure unsupported output, refusal behavior, source attribution, and failure mode labels.

### Neural weight norm as Kolmogorov complexity

**Source status:** Checked, theory lead, reproduction needed for HNC use.

**Paper:** Musat, T. (2026). *Neural Weight Norm = Kolmogorov Complexity.* arXiv:2605.10878. https://arxiv.org/abs/2605.10878

**Code:** No official implementation repository verified yet.

**Relevant claim for this framework:** The paper proves a two sided asymptotic bound between the minimum number of nonzero fixed precision neural parameters needed for a looped network to emit a binary string $s$ and the Kolmogorov complexity $K(s)$ of that string:

$$
\mathcal{N}(s) \leq K(s) \leq c \mathcal{N}(s)\log \mathcal{N}(s) + c.
$$

Here $\mathcal{N}(s)$ is the minimum nonzero parameter count of a fixed precision looped neural network that halts and outputs $s$. The paper then argues that, in fixed precision, $L_p$ weight norms collapse to nonzero parameter count up to constants. Under that condition, a weight norm regularizer can be read as a description length proxy for emitted strings. The paper also derives an induced output prior that matches Solomonoff's universal prior up to a logarithmic factor in the exponent.

**Why it matters here:** This is relevant to HNC's compression prior layer. HNC already treats learned weights as a compressed source and treats unsupported generation as relaxation toward a form prior when source support is weak. Musat gives a theoretical route from fixed precision weights, sparsity, weight norm, and output description length to a Solomonoff style prior over computable strings. This helps explain why simple high prior continuations can dominate when the effective query lacks enough source signal.

**HNC interpretation:** The paper strengthens the static codebook side of HNC. In HNC language, fixed precision weights carry a finite description length, and regularization biases the model toward shorter descriptions. This gives a sharper vocabulary for the form prior: part of the form prior can be treated as the output prior induced by the regularized compressed source. The result also supports the planned compression proxy path in `hnc_lab`, where `ApproximationGap` rows can record log loss proxies, code length proxies, support condition, and unsupported claim rate.

The distinction from Genewein et al. matters. Genewein et al. focus on prediction-compression equivalence and amortized Bayesian prediction. Musat focuses on the description length of fixed precision neural parameters and the induced prior over emitted strings. Together they suggest one testable HNC question: when source support is weak, do compression proxy variables predict whether the output falls toward short, common, unsupported continuations?

**Framework mapping:**

1. **Static codebook:** Fixed precision weights can be treated as a finite information store whose norm or sparsity relates to description length under the paper's assumptions.
2. **Form prior:** Weight regularization can induce a Solomonoff style output prior, up to the paper's logarithmic factor.
3. **Capacity violation:** If a prompt asks for content outside the usable source support, then output may be drawn more strongly from the induced prior.
4. **Approximation gap:** The theorem concerns minimum norm emitters and induced priors, while trained models found by optimization may differ. HNC should track this gap rather than assuming trained weights achieve the theoretical minimum.
5. **Architecture profile:** The theorem is stated for fixed precision looped networks emitting strings. API only text experiments cannot measure the internal norm condition directly.
6. **Quantization and sparsity:** The fixed precision condition makes int4, int8, sparsity, pruning, and weight decay natural experiment surfaces for HNC compression proxy work.

**Useful quantities for HNC:**

1. **Nonzero parameter count:** $\|\theta\|_0$, measured after pruning or thresholding under a stated precision regime.
2. **Log augmented norm proxy:** $\|\theta\|_2^2 \log \|\theta\|_2^2$, the paper's suggested effective complexity proxy for generalization.
3. **Quantization regime:** fp16, bf16, int8, int4, or another fixed precision format.
4. **Sparsity level:** fraction of zero or near zero parameters after a defined pruning rule.
5. **Compression proxy row:** a record joining norm proxies, log loss proxies, source support labels, exact match, refusal quality, and unsupported claim rate.

**Experiment queue:**

1. **Claim binding:** Add the paper to the `compression_proxy_pass` path before citing it as evidence. The first use should record what norm or sparsity proxy was measured and what precision regime was used.
2. **Small model comparison:** Train or fine tune small fixed precision models with different weight decay or sparsity settings on controlled tasks with low complexity and random labels. Test whether the log augmented norm proxy separates them.
3. **Source support test:** On the same generated outputs, record support condition, exact match, refusal quality, source attribution labels, and unsupported claim rate. Test whether weaker source support increases dependence on short high prior continuations.
4. **Quantization test:** Compare fp16, int8, and int4 variants when available. Measure whether stronger quantization and sparsity make norm proxies more predictive of unsupported output.
5. **SGD gap test:** Compare the theoretical minimum norm story against trained model behavior. Treat the gap between minimum norm existence and learned weights as an approximation or optimization gap.

**Caveats:** The result is theoretical and asymptotic. It applies to fixed precision looped networks that emit strings, not directly to ordinary feedforward supervised learning or chat models. The induced prior statement concerns a prior over weights and outputs, not what gradient descent necessarily finds. The paper does not run hallucination experiments. HNC should use it as a compression prior lead until local experiments connect norm or sparsity proxies to unsupported output under source support strata.

### Thinking as Compression: reasoning traces as compressed context

**Source status:** Checked, reproduction needed.

**Paper:** Ma, G., Liu, Y., Li, C., Liang, Y., Wang, Y., Zhang, Y., Chen, K., Zhang, Z., Sun, Z., & Shi, D. (2026). *Thinking as Compression: Your Reasoning Model is Secretly a Context Compressor.* arXiv:2605.28713. https://arxiv.org/abs/2605.28713

**Code:** No official implementation repository verified yet.

**Relevant claim for this framework:** The paper proposes Thinking as Compression (TaC), where a reasoning model converts a long context and query into a shorter thinking trace that serves as compressed context for a downstream answer model. The constrained version, TaC-C, trains the thinker with utility, budget, and anti-hacking rewards. The answer model receives only the query and compressed trace; the original long context stays hidden.

**Why it matters here:** TaC-C gives an experimental surface for HNC's claim that context must be reorganized into a usable dynamic codebook before reliable generation. It also tests the HNC context trade-off directly: raw context supplies source signal, but it can contain distractors and consume working capacity. A query-conditioned trace can preserve task-relevant evidence while reducing context load.

**HNC interpretation:** TaC-C operationalizes a context-management regime in which the model compresses supplied source signal into a compact intermediate representation before answering. In HNC terms, the thinker builds a dynamic codebook from $(q, C)$ under a budget $\mathcal{B}$, and the answerer tests whether that codebook carries enough source support for the requested answer. The budget reward maps to decompression-room control. The utility reward maps to verifier-guided preservation of source signal. The anti-hacking gate maps to source-accounting discipline because direct answer leakage can satisfy a benchmark while failing to act as reusable compressed context.

The paper reports that TaC-C can outperform full-context prompting on several long-context QA settings. HNC should treat that result as evidence that unfiltered context can harm downstream generation when it contains distractors or imposes excess load. The supported mechanism is selective preservation of task-relevant source signal under a budget. Claims about shorter context improving grounding should stay tied to that mechanism.

**Framework mapping:**

1. **Dynamic codebook:** The thinking trace is a query-conditioned representation built at inference time from context.
2. **Context-decompression trade-off:** The budget reward gives a direct way to study how much compressed context the answerer needs.
3. **Kolmogorov garbage:** Fragmented or unfaithful compressed traces give a practical failure mode where pieces of context remain but no longer cohere.
4. **Verifiable generation:** The utility reward uses downstream EM and F1 as task verifiers for whether the trace preserved answer-supporting information.
5. **Information conservation:** The answerer has access only to the query and trace, so unsupported answer content can be checked against the compressed trace and original source.
6. **Regime routing:** TaC-C fits HNC's decomposition regime: reduce requested working load by building a compact intermediate state before final generation.

**Experiment queue:**

1. **Local reproduction:** Reproduce a small TaC-style pipeline on one long-context QA dataset with a small thinker and frozen answerer.
2. **Unsupported-claim audit:** Measure whether TaC-C reduces unsupported claims relative to full context, token pruning, and summary compression.
3. **Budget curve:** Sweep compression budgets and test whether answer quality follows an HNC-style balanced region rather than a monotone curve.
4. **Trace faithfulness:** Compare each trace against the original context to detect omitted evidence, answer leakage, and invented connecting facts.
5. **Distractor load test:** Hold answer evidence fixed and vary distractor count. Test whether TaC-style traces preserve evidence better than full-context prompting under high distractor load.
6. **Cross-answerer transfer:** Reuse the same traces across answer models and measure whether natural-language compressed context transfers more reliably than model-specific soft prompts.

**Caveats:** The paper studies long-context QA compression and efficiency. The full HNC hallucination taxonomy, thermodynamic prior relaxation, stochastic resonance, and universal-manifold claims remain separate test targets. The benchmark reward uses EM and F1, so a trace can improve answer scores while still needing a source faithfulness audit. The anti-hacking constraint is important for HNC because direct answer disclosure changes the trace from compressed context into a shortcut. HNC should cite this paper as support for dynamic context compression and decompression budgeting, with hallucination-specific reproduction still required.

### Geometry of Consolidation: spectral limits for embedding memory

**Source status:** Checked, reproduction needed.

**Paper:** Vangara, A. B., & Gopinath, A. (2026). *The Geometry of Consolidation.* GitHub preprint source. https://github.com/niashwin/geometry-of-consolidation/blob/main/paper/arxiv/main.pdf

**Code and paper source:** `niashwin/geometry-of-consolidation`. https://github.com/niashwin/geometry-of-consolidation

**Relevant claim for this framework:** The paper studies consolidation of unit-norm embedding clusters under cosine-threshold retrieval. It proves a lower bound on identity-retrieval error for any consolidator that maps $n$ cluster members to $m < n$ representatives:

$$
\varepsilon_{\mathrm{id}} \geq 1 - c_1 m \left(\frac{\theta'}{\bar{d}}\right)^{d_{\mathrm{eff}}/2},
$$

where $\varepsilon_{\mathrm{id}}$ is identity-retrieval error, $m$ is the number of representatives retained, $\theta' = 1 - \theta$ is retrieval slack, $\bar{d}$ is mean within-cluster cosine distance, and $d_{\mathrm{eff}}$ is the local effective dimension, measured as the participation ratio of the cluster covariance spectrum:

$$
d_{\mathrm{eff}}(X) = \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}.
$$

**Why it matters here:** This supplies a concrete spectral memory-channel limit for RAG and agent memory. If consolidation erases identity before retrieval, then the downstream generator receives weaker or wrong content constraints. In HNC terms, the result gives a precise pre-generation failure mechanism. A RAG memory system can convert stored source signal into ambiguous, compressed, or corrupted context before the LLM begins reconstruction.

**HNC interpretation:** The result gives an operational version of the Nyquist-style intuition in HNC Section 11.6 for embedding consolidation. The spectral quantity is the local effective dimension of a unit-norm embedding cluster. The temporal-bandlimit reading should remain an analogy. The representative budget $m$ plays the role of a sampling or reconstruction budget. When the budget is too small relative to $(\bar{d}/\theta')^{d_{\mathrm{eff}}/2}$, identity cannot be preserved. That identity loss weakens retrieval grounding and can increase capacity violation, matching failure, or decompression failure in the final answer.

The paper also separates two retrieval quantities that HNC should keep separate. *Identity* asks whether a stored item retrieves itself after consolidation. *Coverage* asks whether a new query lands inside a representative's retrieval cap. Downstream hallucination can depend on either quantity. Identity loss says the memory has forgotten what it stored. Coverage loss says the memory can still contain the right source while failing to surface it for the effective query.

The downstream RAG experiment is useful as a research lead because it links the geometry to reader behavior. With Llama-3.1-70B-Instruct, centroid consolidation hurts Natural Questions by 4.2 exact-match points, is neutral on HotpotQA within the reported uncertainty, and improves PopQA by 8.4 exact-match points. Keep the citation specific: cluster geometry predicts when consolidation helps, hurts, or has no measurable reader effect.

**Framework mapping:**

1. **Model-specific sampling limit:** Provides a concrete spectral lower bound for one memory subsystem: vector-store consolidation under cosine retrieval.
2. **Capacity violation:** Consolidated memories can lose distinguishability, reducing the usable content capacity supplied by RAG.
3. **Matching failure:** Identity loss makes retrieved representatives ambiguous, so the effective query may select a wrong or composite context.
4. **Coverage failure:** A stored source can remain represented while new queries fail to land inside the representative's cap.
5. **Geometric distortion:** Consolidation can move memory points away from recoverable identity regions on the embedding sphere.
6. **Information conservation:** Once identity information is lost during consolidation, later generation cannot recover it from the retrieved context.

**Experiment queue:**

1. **Local reproduction:** Run the repository's synthetic grid and confirm the boundary using the inequalities directly: safe one-vector consolidation when $\bar{d} < \theta'$ and lower-bounded identity loss when $\theta' < \bar{d}$.
2. **Identity and coverage split:** Measure identity error and coverage error separately before passing retrieved passages to a reader.
3. **RAG hallucination test:** Build matched RAG indexes with raw chunks, L2-normalized centroids, medoids, and LLM summaries. Measure unsupported-claim rate, exact match, retrieval identity error, and coverage error.
4. **HNC connection test:** Correlate $\varepsilon_{\mathrm{id}}$, coverage error, $d_{\mathrm{eff}}$, $\bar{d}$, and $\theta'$ with downstream hallucination rate on Natural Questions, HotpotQA, and PopQA-style tasks.
5. **Centroid vs. LLM summary:** Test whether direct geometric centroids reduce consolidation latency and avoid summarization noise without losing answer quality.
6. **Cluster-level diagnostic:** Add a preflight warning for clusters where the bound predicts high identity loss.

**Caveats:** The theorem applies to unit-norm embedding clusters under cosine-threshold retrieval. Treat raw LLM hidden states, non-normalized memory, multimodal memory, biological memory, and learned reranking heads as separate test surfaces. The paper reports downstream QA behavior, but the HNC-specific link to hallucination needs local reproduction with unsupported-claim metrics. Cite the regime conditions by their inequalities because the prose labels around tight and spread regimes are easy to confuse. Social-media claims that this explains every RAG hallucination should stay outside HNC.

### GOAT: trainable attention priors

**Source status:** Checked, reproduction needed.

**Paper:** Litman, E., & Guo, G. (2026). *You Need Better Attention Priors.* arXiv:2601.15380. https://arxiv.org/abs/2601.15380

**Code:** `elonlit/goat`, Generalized Optimal Transport Attention with Trainable Priors. https://github.com/elonlit/goat

**Package:** `goat-attention` on PyPI. https://pypi.org/project/goat-attention/

**Relevant claim for this framework:** GOAT frames scaled dot-product attention as entropic optimal transport. Standard attention corresponds to a transport problem with an implicit uniform prior over key positions. GOAT replaces that prior with a learned continuous prior and keeps compatibility with optimized attention kernels.

**Why it matters here:** GOAT gives a precise attention-level version of prior relaxation. The attention solution has the form $p^{\ast}=\mathrm{softmax}(s/\tau+\log \pi)$, where $s$ contains content scores, $\tau$ controls entropy pressure, and $\pi$ is the attention prior over key positions. When the content scores carry little discriminating signal, or when score differences are small relative to $\tau$, $p^{\ast}$ moves toward $\pi$. If the scores are flat, then $p^{\ast}=\pi$ exactly. This is a concrete token-matching mechanism for temperature-driven relaxation toward a prior. GOAT also gives a mathematical treatment of attention sinks, which connects to sink-limited context capacity.

**HNC interpretation:** GOAT operationalizes local form-prior pressure inside attention. The entropy term supplies attention-level equilibration pressure, the prior $\pi$ supplies the local key-position prior, and weak content signal causes attention to relax toward that prior. This matches the HNC thermalization story at the token-matching level. It does not by itself prove the output-level form prior or the full thermodynamic theorem.

**Framework mapping:**

1. **Capacity violation:** GOAT uses Shannon entropy and KL divergence inside attention. It does not directly use Shannon's noisy-channel coding theorem. When the content signal in $s$ is weak, the attention distribution moves toward the prior. This gives a local mechanism that can be tested against the HNC capacity-violation story.
2. **Maximum-entropy prior relaxation:** The entropy term in the attention objective gives a concrete attention-level pressure toward higher-entropy matching. The prior $\pi$ defines the distribution that attention relaxes toward under weak content scores.
3. **Sink-limited capacity:** GOAT gives a formal account of attention sinks and includes a key-only sink bias. This is directly relevant to HNC sink severity and late-context degradation.
4. **Optimal noise:** The attention temperature $\tau$ controls the balance between content scores and entropy pressure. This creates a direct experiment for the HNC U-shaped noise prediction.
5. **Information conservation diagnostics:** GOAT exposes attention entropy and prior dominance as measurable layer-level signals. These signals may predict unsupported output content, but that connection needs a hallucination benchmark.

**Tool for our work:** The `goat-attention` PyTorch package exposes `GoatAttention` with Fourier positional features, relative-position priors, and a key-only sink bias. This can test whether learned priors reduce sink severity, improve long-context use, or reduce hallucination on weak-knowledge prompts.

**GOAT-specific experiment queue:**

1. **Maximum-entropy prior relaxation test:** Compare baseline attention and GOAT on capacity-stressed prompts. Measure attention entropy, prior dominance, unsupported claims, and refusal accuracy.
2. **Sink-limited capacity test:** Measure attention mass on beginning-of-sequence and early tokens with and without GOAT's sink bias. Correlate this with late-context retrieval and multi-hop accuracy.
3. **Temperature sweep:** Vary $\tau$ and measure hallucination rate, self-consistency, and recovery from misleading context. Test whether the curve has a minimum at an intermediate value.
4. **Information conservation diagnostic:** Use per-layer attention entropy and prior dominance as early-warning signals for outputs that contain unsupported claims.
5. **Prior-shaping test:** Train or initialize priors for factual retrieval contexts and test whether this reduces composite answers or Kolmogorov garbage.

**Caveats:** GOAT does not prove the HNC form-prior theorem, the output-level thermodynamic theorem, or the information conservation claim. It supplies an attention-level mechanism that may instantiate part of the theory. Treat it as a stronger mechanism lead than the others, but still require experiments before moving it into confirmed evidence.

### Attention sinks and compression valleys: massive activations

**Source status:** Checked, reproduction needed.

**Paper:** Queipo-de-Llano, E., Arroyo, A., Barbero, F., Dong, X., Bronstein, M., LeCun, Y., & Shwartz-Ziv, R. (2025). *Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin.* arXiv:2510.06477. https://arxiv.org/abs/2510.06477

**Code:** No official implementation repository verified yet.

**Relevant claim for this framework:** The paper argues that attention sinks and compression valleys share one mechanism: massive activations in the residual stream, usually on the beginning of sequence token. When the beginning of sequence token develops an extreme norm, the representation matrix develops a dominant singular value. This lowers matrix based entropy and coincides with attention sink behavior. The paper also proposes a Mix Compress Refine account of depthwise computation: early layers mix broadly, middle layers compress and reduce mixing, and late layers restore token specific refinement.

**Why it matters here:** HNC already uses attention sinks as a candidate mechanism for sink-limited context capacity. This paper gives a more concrete measurement surface. Instead of measuring only attention mass on prefix tokens, HNC can also measure residual stream norm dominance, singular value entropy, anisotropy, mixing score, column sum concentration, sink versus identity index, and layer phase. That makes sink-limited capacity an architecture specific hidden state probe when model internals are available.

**HNC interpretation:** The result strengthens the mechanism lead for Section 4.6 of the main paper. It suggests that some sinks can reflect a residual stream compression event. In HNC language, the middle layers may enter a compressed state where broad mixing is reduced, then late layers restore token specific refinement. This maps naturally to context crowding, decompression room, and the distinction between embedding style tasks and generation tasks.

**Framework mapping:**

1. **Sink-limited capacity:** The paper supplies measurable variables for the sink mechanism: beginning of sequence norm ratio, sink rate, anisotropy, matrix based entropy, mixing score, column sum concentration, and sink versus identity index.
2. **Decompression failure:** The reported Mix Compress Refine pattern suggests that generation needs late layer refinement after compressed middle layer states. HNC can test whether failures occur when source signal is not recovered after the compression phase.
3. **Geometric distortion:** Layerwise entropy and singular value dominance can become stage specific distortion signals in hidden state experiments.
4. **Architecture profiles:** The paper applies to decoder only transformers with internal access. It supports adding a hidden state sink compression probe to the architecture dependent branch of HNC.
5. **Thermodynamic direction:** Matrix-based entropy gives a concrete entropy proxy inside representations. It gives HNC a measurable entropy term for layerwise information flow, while leaving the output-level thermodynamic model as a separate experiment target.

**Experiment queue:**

1. **Local reproduction:** Reproduce the paper's sink rate, beginning of sequence norm ratio, anisotropy, matrix entropy, mixing score, column sum concentration, and sink versus identity curves on one small open decoder only model.
2. **Late-context degradation test:** Correlate sink-compression metrics with accuracy on prompts where the needed evidence appears late in context.
3. **Context crowding test:** Vary context length and measure whether middle-layer compression metrics predict the right branch of the HNC context-crowding curve.
4. **Mechanism attribution test:** Compare failures labeled as matching failure, decompression failure, and prior dominance against the layer phase where sink-compression metrics peak.
5. **Ablation test:** If model access allows it, ablate or damp the massive activation and measure whether late context use, exact match, refusal behavior, or unsupported claim rate changes.
6. **Task phase test:** Compare embedding style probes and generation style outputs across layers. Test whether HNC should use different layer probes for retrieval, classification, and generation claims.

**Reduced first pass setup work:** HNC can use the paper's candidate sink metrics for open weight decoder only models: sink rate, beginning of sequence norm ratio, anisotropy, matrix entropy, mixing score, column sum concentration, sink versus identity index, and layer phase. HNC still needs to test whether these metrics predict hallucination, refusal, late context use, and unsupported output.

**Caveats:** The paper studies internal mechanics and downstream performance. Hallucination remains an HNC specific test. The paper supports a mechanism lead for sink-limited capacity and decompression. Its hidden state measurements require model access, so API only experiments can only test downstream behavioral signatures.

### TCC context and payload separation

**Source status:** Checked internal design lead, reproduction needed for HNC use.

**Source:** `C:\Git\TCC`, especially `ONE_FIELD_NOTE.md`, `M2_INTEGRATION_DESIGN.md`, `M2_RESULTS.md`, `M2_SPEC.md`, and `model/shared_readout.py`.

**Related unpublished lead:** He, M., Liu, Y., Huang, Q., & Zhang, L. (unpublished). *Do value vectors in deep layers need context from the residual stream?* This is cited in TCC as Bank of Values. Treat it as an unpublished supporting pointer until a paper or reproducible artifact is available.

**Relevant claim for this framework:** TCC separates context routing from payload transport. The carried state `m_t` supplies a context path:

```text
m_t -> shared_read_trunk(m_t) -> h_t
```

The candidate code, observed present code dimensions, future teacher code, and candidate action window remain direct payload paths. In attention terms, the context path is QK-like because it decides which constraints, memories, futures, or completions matter. The payload paths are V-like because they carry the content being denoised, completed, scored, or ranked.

**Why it matters here:** HNC source accounting requires more than correct routing. A model can route to a relevant place in context while losing the value like content needed to support a claim. This matters for compressed context traces, attention sinks, late context failure, and source conditioned repair. If compression preserves where to look but drops what was actually stated, then the final answer can remain coherent while exceeding the available source.

**HNC interpretation:** TCC gives HNC a useful design rule: keep separate measurements for routing success and payload preservation. Routing success asks whether the effective query or attention path selects the right region, source, or constraint. Payload preservation asks whether the content needed for the claim remains available after compression, attention allocation, settling, or repair. A compressed trace, hidden state, retrieval summary, or denoising step should count as grounded only when it preserves the payload evidence that later claims require.

The TCC results support a conservative intermediate architecture. M2a and M2b pass with shared context, separate heads, and direct payload inputs. The results leave one field unification unproved. The TCC notes explicitly warn that forcing all payload through `m_t` or through one scalar field can blur the experiment because payload may be erased or smuggled into the compressed state.

**Framework mapping:**

1. **Source accounting:** A claim needs a payload path to source content, not only a route to a relevant context region.
2. **Compressed context traces:** A trace can improve answer score while dropping payload details. HNC should audit trace faithfulness at the claim level, not only downstream exact match.
3. **Sink-limited capacity:** Attention sinks may preserve global routing or positional bias while starving late value content. This creates a concrete measurement split between attention allocation and content transport.
4. **Decompression failure:** If the compressed state carries constraints without enough payload detail, the answerer may reconstruct a plausible form from the prior.
5. **Source conditioned denoising:** A repair step should receive explicit source payload or retrieval evidence. A repair that only receives a compressed context state may improve fluency while leaving source support untracked.
6. **Experiment design discipline:** Shared trunks and unified objectives should be tested against baselines that receive the same evidence and target. Direct payload access should be declared before the run.

**Useful quantities for HNC:**

1. **Routing accuracy:** whether the selected context, retrieved passage, attention region, or trace span contains the required evidence.
2. **Payload retention:** whether the exact claim supporting content survives compression, summarization, attention routing, or denoising.
3. **Payload attribution:** which source text, retrieved passage, tool output, or memory record supports each generated claim.
4. **Routing-payload mismatch:** cases where routing points to the right source region but the generated claim is unsupported by the retained payload.
5. **Direct-payload flag:** whether an experiment gives the answerer direct access to source text, only a compressed trace, only retrieved summaries, or hidden state features.

**Experiment queue:**

1. **Trace payload audit:** In `compressed_context_trace`, compare full context, summaries, token pruning, and query conditioned traces. For each generated claim, record whether the claim's supporting payload survived in the trace.
2. **Routing versus payload split:** On open weight models, measure attention routing to evidence tokens and separately test whether the value-side content influences the output. Use late evidence prompts where routing and content can diverge.
3. **Sink payload test:** In sink compression probes, measure whether beginning of sequence sink strength predicts loss of late value content, not only late-token attention mass.
4. **Denoising evidence test:** In source conditioned repair, compare repair with direct source payload, repair with a compressed trace, and repair with no source payload. Measure unsupported claims after repair.
5. **Unification falsification rule:** Any future unified read or compressed state experiment should preregister which payload remains direct, which payload is intentionally compressed, and which metric would falsify the unification.

**Caveats:** TCC is a 2D latent control system and architecture study, not an LLM attention experiment. Its positive results support shared context with separate heads and direct payload inputs. They do not establish one scalar field for all reads, and they do not establish temporal memory repair because M2c failed its honesty checks. The Bank of Values reference remains unpublished in the inspected repo. HNC should use this as a design and experiment lead until an LLM probe tests routing and payload preservation directly.

### Co-Tok: compute optimal tokenization

**Source status:** Checked, reproduction needed.

**Paper:** Limisiewicz, T., Pagnoni, A., Iyer, S., Lewis, M., Mehta, S., Liu, A., Li, M., Ghosh, G., & Zettlemoyer, L. (2026). *Compute Optimal Tokenization.* Project PDF: https://co-tok.github.io/paper.pdf

**Project page:** https://co-tok.github.io/

**Archival link:** arXiv:2605.01188. https://arxiv.org/abs/2605.01188

**Code:** No official implementation repository verified yet. The project page appears to be hosted through GitHub Pages, and it should be treated as a project page unless an implementation repository is linked.

**Relevant claim for this framework:** Co-Tok studies how token compression rate, measured as average bytes per token, affects scaling behavior. The paper reports that compute-optimal data size scales better in bytes than in tokens, and that the optimal compression rate changes with compute and language.

**Project-page findings of interest:**

1. In compute-optimal scaling, bytes of training data increase proportionally to parameter count. The project page states this as a bytes-based rule for scaling recipes.
2. At each training compute budget, there is an optimal compression rate. The reported optimum decreases at larger scales.
3. The optimal compression rate varies across languages and correlates with parity, defined as the byte-length ratio needed to express comparable content relative to English.

**Why this is especially interesting for HNC:** HNC treats training as source coding and inference as reconstruction plus transmission. Co-Tok makes the first source-coding step measurable: raw text is compressed into tokens or latent segments before the model can learn, match, retrieve, or reconstruct anything. If bytes per token changes the scaling loss surface, then tokenization can change the effective signal seen by the model and the sequence length burden carried by attention.

This suggests a direct HNC question: when the source facts are held fixed, does changing compression rate change hallucination, refusal, self-consistency, context crowding, or late-context retrieval? If the answer is yes, then tokenization belongs inside the source-support path as a theory object.

**Why it matters here:** Tokenization is the first source-coding step before the model performs matching, reconstruction, and transmission. A poor compression rate may reduce useful input structure or create inefficient sequence lengths. That makes Co-Tok relevant to HNC capacity questions because the effective query depends on how the source was encoded before attention sees it. The reported U-shaped loss profiles over compression rate also resemble HNC's balanced-region logic: too coarse an encoding can merge distinctions, while too fine an encoding can increase sequence burden.

**Tool for our work:** The paper uses Byte Latent Transformer style latent tokenization and entropy-threshold segmentation to control compression rate. This creates a way to test whether upstream compression changes hallucination rates, context crowding, or sink severity.

**Framework mapping:**

1. **Capacity violation:** Co-Tok defines compression rate as average bytes per token and shows that validation loss depends on this rate. This gives a source-coding variable that can be tested against HNC topic capacity. The paper measures loss, so the hallucination-capacity connection remains our experiment to run.
2. **Information conservation:** Off-optimal compression may remove or obscure recoverable input structure before attention and reconstruction begin. HNC can test whether this increases unsupported output content or excess complexity proxies.
3. **Maximum-entropy prior relaxation:** Entropy-threshold segmentation gives a controlled way to change the strength of the input signal. If compression is too aggressive, then useful boundaries or high-entropy regions may be missed. HNC can test whether weaker input signal increases relaxation toward form-prior output.
4. **Sink-limited capacity:** Changing compression rate changes sequence length and the structure seen by attention. This can test whether upstream compression changes sink severity and late-context retrieval.
5. **GOAT complement:** Co-Tok gives an upstream source-coding control, while GOAT gives an attention-prior control. Together they can test whether source encoding and attention priors jointly reduce context failure.
6. **Language parity and topic capacity:** Co-Tok reports that optimal compression varies across languages and correlates with parity. HNC can test whether language-specific encoding burden changes topic capacity, refusal quality, or unsupported-answer rate when facts are matched across translated prompts.

**Co-Tok-specific experiment queue:**

1. **Compression-hallucination curve:** Sweep compression rate in a BLT-style or subword setup. Measure factual accuracy, hallucination rate, self-consistency, and unsupported-claim count on capacity-stressed prompts.
2. **Source-coding capacity test:** Keep facts fixed and change only encoding granularity. Measure whether the effective query becomes easier or harder to reconstruct.
3. **Sink severity vs. compression:** Track attention mass on beginning-of-sequence and early tokens across compression rates. Compare late-context retrieval at each rate.
4. **Cross-language capacity test:** Use languages with different byte parity. Test whether mismatched compression rates increase hallucination or refusal errors.
5. **Co-Tok plus GOAT ablation:** Compare baseline tokenization plus vanilla attention, Co-Tok-style compression alone, GOAT alone, and the combined setup. The outcome should be measured with hallucination-specific metrics.
6. **Bytes versus tokens accounting:** Report every experiment in both bytes and tokens. Test whether bytes predict source support, context load, and hallucination rate more consistently than token count under tokenizer changes.

**Caveats:** Co-Tok measures language modeling loss and scaling behavior. Hallucination remains an HNC experiment. Any HNC claim needs an added hallucination benchmark, with the tokenizer or latent segmentation treated as the independent variable. The current HNC use should therefore stay at the research-lead level: Co-Tok supplies an upstream source-coding knob, and HNC must test whether that knob affects unsupported output.

### Generalization theory: signal channel and reservoir

**Source status:** Checked, reproduction needed.

**Paper:** Litman, E., & Guo, G. (2026). *A Theory of Generalization in Deep Learning.* arXiv:2605.01172. https://arxiv.org/abs/2605.01172

**Code:** No official code link verified yet.

**Relevant claim for this framework:** The paper describes generalization through an empirical neural tangent kernel decomposition. Output directions split into a signal channel, where error dissipates quickly, and a reservoir, where residual error can remain test-invisible. The paper also derives a population-risk objective that becomes an SNR preconditioner on top of Adam.

**Why it matters here:** The signal-channel and reservoir language is close to the HNC distinction between accessible knowledge and inaccessible or noisy structure. It may give a training-time analogue of HNC capacity failure: signal that enters useful channels generalizes, while noise or weak structure remains trapped in directions that do not transfer well.

**Tool for our work:** The proposed SNR preconditioner is a candidate optimizer intervention. It can test whether suppressing low-signal updates reduces memorization, improves perturbation stability under noisy labels, or reduces hallucination after preference tuning.

**Framework mapping:**

1. **Hallucination threshold:** The signal-channel and reservoir split gives a training-time way to ask whether useful signal enters transferable directions. HNC can test whether weak topic signal later produces capacity-violation behavior at inference.
2. **Maximum-entropy prior relaxation:** The reservoir may act as a high-dimensional holding region for noise or idiosyncratic memorization. This is relevant to form-prior drift, but it is not identical to the HNC output-level form prior until hallucination measurements are added.
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

### Edge of stability: random attractors and sharpness dimension

**Source status:** Checked, reproduction needed.

**Paper:** Tuci, M., Korkmaz, C., Simsekli, U., & Birdal, T. (2026). *Generalization at the Edge of Stability.* arXiv:2604.19740. https://arxiv.org/abs/2604.19740

**Code:** Project page announced in the paper: https://circle-group.github.io/research/GATES. No local reproduction has been run for HNC.

**Relevant claim for this framework:** The paper models stochastic optimization as a random dynamical system. In the edge-of-stability regime, optimizer trajectories need not converge to a single parameter vector. They can explore a lower-dimensional random attractor. The paper introduces Sharpness Dimension, a Hessian-spectrum-based measure of the effective dimension of this attractor, and gives a generalization bound in terms of that dimension.

**Why it matters here:** HNC treats training as source coding into weights. This paper suggests a more precise training-side object: the geometry of the optimizer's long-run attractor. If the trained model's useful structure depends on an attractor with lower effective dimension, then `K_rep(weights | T)` may need to account for training dynamics, Hessian spectrum, attractor geometry, parameter count, and final validation loss.

**HNC interpretation:** This lead belongs to the training-capacity branch. It supports the idea that controlled instability and stochasticity can shape useful structure rather than merely corrupt it. It is adjacent to HNC's noise paradox, but at training time rather than decoding time. It also gives a formal path for discussing grokking and delayed generalization as changes in the geometry of the training dynamics.

**Framework mapping:**

1. **Learning as compression:** Sharpness Dimension provides a candidate complexity proxy for the effective dimension of the learned attractor.
2. **Topic capacity:** HNC can test whether lower or better-structured Sharpness Dimension correlates with stronger topic capacity after training or fine-tuning.
3. **Optimal noise:** The edge-of-stability regime gives a training-time example where instability and stochasticity can coexist with better generalization.
4. **Geometric distortion:** Hessian-spectrum structure may help identify training runs that produce fragile representations before inference-time distortion begins.
5. **Grokking:** The paper reports that Sharpness Dimension tracks grokking transitions in controlled settings. HNC can use this as a lead for delayed acquisition of source-support structure.

**Experiment queue:**

1. **Local reproduction on a small task:** Reproduce Sharpness Dimension on a small MLP or small transformer setup before using it in HNC claims.
2. **Noisy supervision test:** Train matched models with different learning rates, batch sizes, and label-noise levels. Measure Sharpness Dimension, validation gap, refusal calibration, and unsupported-answer rate.
3. **Topic-capacity test:** Fine-tune small models on topics with known coverage. Test whether Sharpness Dimension predicts held-out QA accuracy or false-premise refusal better than Hessian trace or top eigenvalue alone.
4. **Grokking source-support test:** On an algorithmic task with delayed generalization, measure whether changes in Sharpness Dimension precede improved source-supported generation.
5. **Training-to-inference connection:** Compare training-side Sharpness Dimension with inference-side self-consistency, entropy, and hallucination rate on the same model checkpoints.

**Caveats:** The paper studies generalization bounds, grokking, and training dynamics. It does not measure hallucination. The HNC use should stay at the research-lead level until a local experiment tests whether Sharpness Dimension predicts source-supported output, refusal quality, or topic capacity. Full Hessian-spectrum estimation is also computationally expensive for large models, so early HNC tests should use small controlled models.

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

### Experiment 5: consolidation identity loss and RAG hallucination

**Question:** Does spectral identity loss during embedding consolidation predict downstream unsupported claims?

**Methods:** Build matched RAG indexes over the same source documents using raw chunks, L2-normalized centroids, medoids, LLM summaries, and pruning. For each cluster, measure $d_{\mathrm{eff}}$, $\bar{d}$, retrieval slack $\theta'$, predicted identity-error floor, observed retrieval identity error, exact match, and unsupported-claim rate.

**Framework target:** Model-specific sampling limits, RAG capacity loss, matching failure, and information conservation.

### Experiment 6: massive activations, sink compression, and late context failure

**Question:** Do massive activation and compression valley metrics predict downstream late context failure or unsupported output?

**Methods:** On a small open decoder only model, measure beginning of sequence norm ratio, sink rate, anisotropy, matrix entropy, mixing score, column sum concentration, sink versus identity index, and layer phase. Run matched prompts where the answer appears early, middle, or late in context. Correlate the hidden state metrics with exact match, refusal, self-consistency, and unsupported claim rate.

**Framework target:** Sink-limited capacity, context crowding, decompression failure, and architecture profiles.

### Experiment 7: thinking traces as compressed context

**Question:** Can query conditioned thinking traces reduce unsupported output by preserving source payload while lowering context load?

**Methods:** Reproduce a small TaC style thinker answerer pipeline. Compare full context, token pruning, generic summaries, query conditioned traces, and budget trained traces. Measure exact match, F1, unsupported claim rate, trace faithfulness, payload retention, routing payload mismatch, answer leakage, actual compression ratio, and cross answerer transfer.

**Framework target:** Dynamic codebooks, context decompression tradeoffs, Kolmogorov garbage, QK/V context and payload separation, verifiable generation, and source accounting.

### Experiment 7b: routing versus payload audit

**Question:** Can an experiment separate correct routing from preserved source payload?

**Methods:** Build prompts where the required evidence appears in a known context span. Record whether the model, trace, or retrieval system selects the right span. Separately record whether the generated claim is supported by the exact content retained from that span. Count routing success, payload retention, unsupported claims, and routing payload mismatch.

**Framework target:** Source accounting, sink limited capacity, compressed context traces, and decompression failure.

### Experiment 8: compression proxies and source support

**Question:** Do prediction-compression quantities and fixed precision norm proxies help predict unsupported output under HNC source support strata?

**Methods:** For the same prompt set, record log loss or available probability proxies, code length proxies, fixed precision norm proxies when model weights are available, quantization regime, sparsity level, exact match, refusal quality, source attribution labels, and unsupported claim rate. Stratify items by strong source, weak recoverable source, unsupported source, and misleading source.

**Framework target:** Learning as compression, static codebooks, form prior, capacity violation, approximation gap, in-context support, and source accounting.

## Promotion rule

A paper should move from this file into `EVIDENCE_AND_SUPPORTING_LITERATURE.md` only when at least one of the following is true:

1. The exact claim has been reproduced locally.
2. The paper directly measures a quantity already used in HNC.
3. The paper supplies a method that we used in an HNC experiment.
4. The caveats are documented clearly enough that the citation cannot be mistaken for stronger evidence than it is.

Until then, these papers remain research leads under review.
