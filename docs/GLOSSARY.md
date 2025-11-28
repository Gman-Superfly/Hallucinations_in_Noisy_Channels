# Glossary of Core Concepts

| Term | Definition |
| --- | --- |
| **Semantic Drift** | Deviation of the latent trajectory away from the intended concept manifold when contextual information is insufficient. |
| **Semantic Nyquist Rate** | Minimum prompt information required to reconstruct a concept of given complexity without aliasing; inspired by the Shannon-Nyquist theorem. |
| **Conceptual Manifold** | Stable region of the model’s latent space representing a coherent concept or answer, analogous to an attractor basin. |
| **Prompt Manifold** | Trajectory traced by a prompt in semantic space; must intersect the relevant conceptual manifold to avoid hallucination. |
| **In-Context Attractor (ICA)** | Spurious latent basin created by overly long or conflicting prompts, which can capture the trajectory and induce hallucinations. |
| **Chain-of-Thought Parity** | Redundant reasoning tokens that act like parity checks, detecting or correcting semantic errors during generation. |
| **Semantic Redundancy Ratio ($\\rho$)** | Fraction of prompt tokens that convey informative evidence; high $\\rho$ indicates redundancy helps, low $\\rho$ signals noise-dominated prompts. |
| **Energy Drift** | Increase?  in computational free energy, maybe shift of energy distribution (e.g., FLOPs, entropy production) caused by unstable trajectories, linked to Landauer’s principle. |
| **Attention SNR** | Ratio between concept clarity and attention ambiguity; predicts whether the channel capacity is sufficient for faithful reconstruction. |

