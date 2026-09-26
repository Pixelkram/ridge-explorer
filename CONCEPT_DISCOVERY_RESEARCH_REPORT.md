# Concept Discovery in Generative AI Models: A Comprehensive Research Report

**Prepared for: Phase Boundary Ridge Research Program**
**Date: April 2026**

---

## Executive Summary

This report surveys the rapidly evolving landscape of concept discovery in generative AI models, with particular emphasis on techniques applicable to understanding phase boundary ridges in text-to-image diffusion model latent spaces. The field has undergone a remarkable acceleration in 2024-2025, driven by three converging threads: (1) sparse autoencoders moving from LLMs into vision and diffusion models, (2) Riemannian/information-geometric analysis of latent spaces revealing fractal phase boundary structure, and (3) mechanistic interpretability techniques being adapted from language to image generation. Several of these developments connect directly to your ridge detection work.

---

## 1. Concept Discovery in Diffusion Models

### 1.1 Unsupervised Discovery of Semantic Directions

**NoiseCLR** (Dalva et al., CVPR 2024) is the leading unsupervised method for discovering latent semantics in diffusion models. It employs a contrastive learning objective: similar edits in noise space attract each other while different edits repel. It requires no text prompts, labeled data, or fine-tuning, making it domain-agnostic (faces, cats, art, medicine). Discovered directions are highly disentangled and composable across domains.
- Paper: https://arxiv.org/abs/2312.05390
- Code: https://github.com/gemlab-vt/NoiseCLR

**Self-Discovering Interpretable Diffusion Latent Directions** (Li et al., CVPR 2024) learns interpretable latent vectors in h-space for user-defined concepts by minimizing reconstruction loss when the text prompt omits the target concept. The frozen diffusion model's semantic knowledge drives the discovery process.
- Paper: https://arxiv.org/html/2311.17216v2

**Discovering Interpretable Directions via Jacobian Analysis** (Haas et al., 2024) uses spectral analysis of the denoiser Jacobian to find both global principal component directions and image-specific local semantic directions. By projecting the Jacobian of masked regions into an orthogonal subspace, it enables precise local semantic control without annotations.
- Paper: https://arxiv.org/abs/2303.11073

**Discovering Concept Directions from Diffusion-based Counterfactuals via Latent Clustering (CDLC)** (2025) extracts global, class-specific concept directions by clustering latent difference vectors from factual/counterfactual image pairs. Reduces storage by ~4.6% and accelerates discovery by ~5.3% vs. baselines, requiring no GPU for clustering.
- Paper: https://arxiv.org/abs/2505.07073

### 1.2 Compositional Concept Discovery

**Unsupervised Compositional Concepts Discovery** (Liu et al., ICCV 2023) discovers generative concepts from unlabeled images using score function decomposition. Each concept gets a score function; these compose to form a joint score for denoising. Successfully disentangles art styles, objects, and lighting, with concepts recombineable for hybrid generation.
- Paper: https://arxiv.org/abs/2306.05357
- Code: https://github.com/nanlliu/Unsupervised-Compositional-Concepts-Discovery

### 1.3 Concept Sliders and Concept Control

**Concept Sliders** (Gandikota et al., ECCV 2024) learns low-rank parameter directions (LoRA adaptors) corresponding to specific concepts while minimizing interference with other attributes. Sliders are composable, continuously modulable, and work from prompts or sample images.
- Paper: https://dl.acm.org/doi/10.1007/978-3-031-73661-2_10
- Code: https://github.com/rohitgandikota/sliders

**Prompt Sliders** (Sridhar & Vasconcelos, ECCVW 2024) learns concepts through text embeddings via textual inversion, rather than LoRA. Generalizable across models sharing the same text encoder. Simpler optimization, supports concept erasure (styles, nudity, objects).
- Paper: https://arxiv.org/abs/2409.16535
- Code: https://github.com/DeepakSridhar/promptsliders

### 1.4 Concept Probing and Erasure

**"When Are Concepts Erased From Diffusion Models?"** (Lu & Kriplani, NeurIPS 2025) introduces a comprehensive probing suite (visual context, trajectory modification, classifier guidance, alternative generation analysis). Key finding: current erasure methods suppress concepts at the prompt level, functioning as input filtering rather than genuine parameter-space erasure.
- Paper: https://arxiv.org/abs/2505.17013

**"Erased or Dormant?"** (2025) rethinks concept erasure through reversibility, showing that supposedly erased concepts often remain dormant in model parameters.
- Paper: https://arxiv.org/html/2505.16174v2

**ConceptPrune** (ICLR 2025) identifies critical weight regions responsible for undesirable concepts and prunes just ~0.12% of weights for training-free multi-concept erasure with adversarial robustness.
- Paper: https://arxiv.org/abs/2405.19237

### 1.5 Connection to Ridge Detection

These concept discovery methods reveal that diffusion models organize semantic knowledge along discoverable directions in activation space. Your ridge detection work identifies *where* semantic transitions occur during interpolation; these methods characterize *what* concepts define each side of those transitions. Combining NoiseCLR-style direction discovery with Hessian ridge detection could label ridges by the concepts they separate.

---

## 2. Mechanistic Interpretability of Image Generators

### 2.1 Circuit-Level Analysis of Diffusion Models

**"Mechanistic Interpretability of Diffusion Models: Circuit-Level Analysis and Causal Validation"** (2025) presents the first comprehensive circuit-level analysis of diffusion models. Key findings:
- Eight functionally distinct attention mechanisms identified: edge detection, texture analysis, semantic understanding
- Real-world face processing requires circuits with measurably higher computational complexity
- Intervention analysis reveals critical bottlenecks where targeted ablations produce 25.6%-128.3% performance degradation
- Paper: https://arxiv.org/abs/2506.17237

### 2.2 ConceptAttention: Interpretable Features in DiT

**ConceptAttention** (Helbling et al., ICML 2025) demonstrates that Diffusion Transformer (DiT) attention layers learn highly interpretable features. Linear projections in DiT output space yield significantly sharper saliency maps than cross-attention maps. Achieves SOTA on zero-shot ImageNet-Segmentation, outperforming 15 other methods. Works with Flux DiT, generalizes to video. First evidence that multi-modal DiT representations transfer well to vision tasks.
- Paper: https://arxiv.org/abs/2502.04320
- Code: https://github.com/helblazer811/ConceptAttention

### 2.3 Cross-Attention Head Analysis

**Head Relevance Vectors** (OpenReview, 2024-2025) construct per-concept importance vectors over cross-attention heads, aligning head-level computations with human visual concepts. Studies show only certain attention layers generate specific visual concepts, and deeper layers encode semantics more strongly.
- Paper: https://openreview.net/forum?id=1vggIT5vvj
- Related survey: https://arxiv.org/pdf/2504.03738

### 2.4 Sparse Autoencoders for Diffusion Model Interpretability

**SAeUron** (Cywinski et al., ICML 2025) trains SAEs on diffusion model activations across multiple denoising timesteps. Captures sparse, interpretable features per concept. Enables simultaneous multi-concept unlearning with high adversarial robustness. Outperforms existing methods on UnlearnCanvas benchmark.
- Paper: https://arxiv.org/abs/2501.18052
- Code: https://github.com/cywinski/SAeUron

**Steering Diffusion Transformers with Sparse Autoencoders** (OpenReview, 2025) conducts the first systematic SAE study on DiTs. Introduces multi-layer steering for improved causal effectiveness and a similarity-based criterion for feature detection in the residual stream.
- Paper: https://openreview.net/forum?id=J48XM0au4u

**TIDE: Temporal-Aware Sparse Autoencoders** (AAAI 2025) extracts sparse features across timesteps in DiTs. Reveals that DiTs naturally learn hierarchical semantics (3D structure, object class, fine-grained concepts) during pretraining. Enables safe image editing and style transfer.
- Paper: https://arxiv.org/abs/2503.07050

**Concept Steerers** (2025) leverages k-sparse autoencoders for test-time controllable generation, enabling efficient and interpretable concept manipulation.
- Paper: https://arxiv.org/html/2501.19066

**Emergence and Evolution of Interpretable Concepts** (CVPR Workshop, 2025) uses SAEs to probe how human-interpretable concepts evolve through the diffusion generative process. Finds that even before the first reverse step completes, the final scene composition can be predicted from spatial distribution of activated concepts.
- Paper: https://arxiv.org/abs/2504.15473

### 2.5 Connection to Ridge Detection

Circuit-level analysis and SAE-based feature extraction provide the mechanistic foundation for understanding *why* ridges exist. If specific attention circuits encode specific concepts, then phase boundaries should align with transitions between dominant circuit activations. TIDE's temporal-aware approach is particularly relevant: ridges may correspond to timestep-specific concept boundaries that the denoising process navigates.

---

## 3. Latent Space Structure & Topology

### 3.1 Phase Transitions in Diffusion Models

**"A Phase Transition in Diffusion Models Reveals the Hierarchical Nature of Data"** (Sclocchi et al., PNAS, January 2025) is the foundational paper on phase transitions in diffusion. Key findings:
- The backward diffusion process exhibits a phase transition at a threshold "speciation time" t_s
- High-level features (image class) reconstruct via sharp transition; low-level features evolve smoothly
- Hierarchical data structure is reflected in hierarchical phase transitions
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11725779/

**"Caffarelli Regularity and Hierarchical Phase Boundaries in Diffusion Models"** (NeurIPS ML4PS Workshop, 2025) provides mathematical foundations using optimal transport theory (JKO scheme, Caffarelli regularity). Observes hierarchical emergence: coarse boundaries in early denoising, finer boundaries progressively within regions. Diffusion distillation shifts boundaries earlier and reduces final complexity.
- Paper: https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_71.pdf
- OpenReview: https://openreview.net/forum?id=HadqLI0x1V

**"Hessian Geometry of Latent Space in Generative Models"** (Lobashev et al., ICML 2025) -- this is the paper closest to your own work. Reconstructs the Fisher information metric on the latent space. Reveals fractal structure of phase transitions characterized by abrupt changes in the Fisher metric. Geodesic interpolations are approximately linear within phases but break down at boundaries where the Lipschitz constant diverges. Validated on Ising model, TASEP, and diffusion models.
- Paper: https://arxiv.org/abs/2506.10632
- Code: https://github.com/alobashev/hessian-geometry-of-diffusion-models

### 3.2 Riemannian Geometry of Diffusion Latent Spaces

**"The Spacetime of Diffusion Models: An Information Geometry Perspective"** (Karczewski et al., ICLR 2026 Oral) critiques the standard pullback approach using deterministic probability flow ODE decoders, proving it forces geodesics to decode as straight segments. Instead introduces a latent spacetime z=(x_t, t) indexing denoising distributions p(x_0|x_t), endowed with the Fisher-Rao metric. Derives simulation-free length estimators. Introduces "Diffusion Edit Distance." Demonstrates applications in molecular transition path sampling.
- Paper: https://arxiv.org/abs/2505.17517
- Code: https://github.com/rafalkarczewski/spacetime-geometry

**"Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry"** (Park et al., NeurIPS 2023) derives local latent bases via pullback metrics from encoding feature maps, enabling image editing by moving along basis vectors at specific timesteps. The CLIP pullback metric tracks where generated image features change abruptly.
- Paper: https://arxiv.org/abs/2307.12868

**"Be Tangential to Manifold"** (2025) proposes a Riemannian metric for diffusion noise space requiring no retraining. Yields perceptually more natural transitions than density-based methods.
- Paper: https://arxiv.org/abs/2510.05509

**"What's Inside Your Diffusion Model? A Score-Based Riemannian Metric"** (Azeglio et al., 2025) uses the Stein score function to characterize data manifold geometry without explicit parameterization. Metric stretches distances perpendicular to the manifold while preserving tangential ones.
- Paper: https://arxiv.org/abs/2505.11128

**"Score-based Pullback Riemannian Geometry"** (Diepeveen et al., 2024) provides closed-form expressions for geodesics and distances on data manifolds using score-based pullback metrics.
- Paper: https://arxiv.org/abs/2410.01950

### 3.3 Smooth Latent Spaces

**Smooth Diffusion** (Guo et al., CVPR 2024) enforces step-wise variation regularization so that perturbations in latent space produce proportional output changes. Introduces the ISTD metric for latent space smoothness. Implemented as plug-and-play Smooth-LoRA.
- Paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_Smooth_Diffusion_Crafting_Smooth_Latent_Spaces_in_Diffusion_Models_CVPR_2024_paper.pdf
- Code: https://github.com/SHI-Labs/Smooth-Diffusion

### 3.4 Topological Data Analysis

**Persistent Homology for LLM Latent Spaces** (2025) uses persistent homology to characterize multi-scale dynamics within model activations under adversarial conditions.
- Paper: https://arxiv.org/abs/2505.20435

**Evaluating Generative Models via Cubical Homology** (KDD 2025) proposes a scalable framework using persistent entropy, tracking topological transformation of the generated data manifold during training.
- Proceedings: https://dl.acm.org/doi/10.1145/3711896.3736943

### 3.5 Connection to Ridge Detection

This section contains the work most directly related to yours. Lobashev et al.'s Hessian geometry paper demonstrates that phase boundaries in diffusion latent spaces have fractal structure with divergent Lipschitz constants -- your ridge detection may be capturing these same phenomena. The Caffarelli regularity paper shows these boundaries emerge hierarchically during denoising, suggesting that your Hessian ridges should exhibit timestep-dependent structure. The spacetime geometry paper (ICLR 2026 Oral) argues that the correct metric is Fisher-Rao on denoising distributions rather than deterministic pullback, which could provide a more principled basis for ridge computation. Smooth Diffusion's work on enforcing Lipschitz continuity is the complement: they try to *remove* the very discontinuities you are trying to *detect*.

---

## 4. Concept Discovery in LLMs (Transferable Techniques)

### 4.1 Sparse Autoencoders: The Core Method

**Anthropic's Scaling Monosemanticity** (May 2024) extracted high-quality features from Claude 3 Sonnet using SAEs, finding highly abstract, multilingual, multimodal features. Features span safety concerns (deception, sycophancy, bias, dangerous content). Features can steer model behavior. Scaling laws guide SAE training.
- Paper: https://transformer-circuits.pub/2024/scaling-monosemanticity/

**OpenAI's Extracting Concepts from GPT-4** (2024) uses k-sparse autoencoders to directly control sparsity, simplifying tuning and improving the reconstruction-sparsity Pareto frontier.
- Coverage: https://arize.com/blog/llm-interpretability-and-sparse-autoencoders-openai-anthropic/

**BatchTopK SAE** (2024) significantly improves the sparsity/reconstruction trade-off, allows directly setting desired sparsity without tuning a penalty, with good training stability.
- Referenced in: https://arxiv.org/html/2503.05613v3

### 4.2 Circuit Tracing and Attribution Graphs

**Anthropic's Circuit Tracing** (May 2025) produces graph descriptions of model computation by tracing individual steps in a "replacement model" using cross-layer transcoders. Attribution graphs show features as nodes and causal interactions as edges. Applied to Claude 3.5 Haiku, revealing planning behavior in poem generation.
- Methods: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Biology: https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- Open-source tools: https://www.anthropic.com/research/open-source-circuit-tracing

### 4.3 Crosscoders for Cross-Layer and Cross-Model Analysis

**Sparse Crosscoders** (Anthropic, October 2024) extend SAEs to read/write across multiple layers simultaneously, resolving cross-layer superposition. Enable "model diffing" by training a single crosscoder on two models to discover shared vs. unique features. Model-exclusive features tend to be more polysemantic.
- Paper: https://transformer-circuits.pub/2024/crosscoders/index.html
- Update: https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html

### 4.4 Representation Engineering

**Representation Engineering** (2024-2025) learns vectors that can be added to activations to manipulate concept detection. Multiple approaches exist:
- Linear probes learn weight vectors separating concept-positive from concept-negative activations
- Concept Activation Vectors (CAVs) use SVM hyperplanes in activation space
- Recent work (Beaglehole et al., 2025) trains probes learning non-linear features, deriving sensitivity matrices with eigenvector concept operators
- Nguyen et al. (2025) use Maximum Mean Discrepancy loss to match activation distributions
- Key critique: Probing-based removal can be counter-productive (NeurIPS 2022)
- Recent CAV paper: https://proceedings.iclr.cc/paper_files/paper/2025/file/852870d56e20a22fcd52ea14523fb820-Paper-Conference.pdf

### 4.5 Connection to Ridge Detection

SAEs are the single most transferable technique from LLM interpretability to your work. Training SAEs on diffusion model activations at points along your interpolation paths could reveal which features activate on each side of a ridge, providing semantic labels for phase boundaries. Crosscoders could identify which features are *shared* vs. *unique* across different phases, effectively characterizing what changes at a ridge. Circuit tracing could map the computational pathway from text prompt through to the generation of a particular semantic feature, showing *how* ridges arise mechanistically.

---

## 5. Concept Discovery in Other Generative Models

### 5.1 GANs: Latent Space Analysis

**Local Intrinsic Dimension Estimation** (Pattern Recognition, 2025) estimates local intrinsic dimensions in GAN latent spaces, analyzing StyleGAN mapping network layers through a Distortion metric.
- Paper: https://www.sciencedirect.com/science/article/abs/pii/S0020025519302993

**Semantic Distribution Descriptions** (Knowledge-Based Systems, 2025) uses high-dimensional Gaussian distributions to describe semantics learned from GAN latent spaces, applied to StyleGAN and PGGAN.
- Paper: https://www.sciencedirect.com/science/article/abs/pii/S0950705124015284

**OTUSD** (Optimal Transport-based Unsupervised Semantic Disentanglement) uses manifold learning and optimal transport theory to uncover semantic directions in StyleGAN, StyleGAN2, and BigGAN latent spaces.
- Paper: https://www.sciencedirect.com/science/article/abs/pii/S0141938223001932

### 5.2 VAEs: Disentangled Representations

**L-VAE** (Machine Vision and Applications, 2025) extends beta-VAE by learning the hyperparameter beta itself, mitigating the reconstruction-disentanglement trade-off.
- Paper: https://arxiv.org/abs/2507.02619

**CL-Dis** (ECCV 2024) uses Diffusion-AE as a backbone with beta-VAE as a co-pilot, introducing a self-supervised Navigation strategy for finding interpretable semantic directions.
- Paper: https://arxiv.org/abs/2402.02346

### 5.3 NeRFs and 3D Scenes

**Decompositional Neural Scene Reconstruction** (CVPR 2025) decomposes implicit 3D representations into individual objects using diffusion priors.
- Paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Ni_Decompositional_Neural_Scene_Reconstruction_with_Generative_Diffusion_Prior_CVPR_2025_paper.pdf

**GP-NeRF** (CVPR 2024 Highlight) achieves generalized perception in NeRFs for context-aware 3D scene understanding, with strong instance and semantic segmentation.
- Paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_GP-NeRF_Generalized_Perception_NeRF_for_Context-Aware_3D_Scene_Understanding_CVPR_2024_paper.pdf

### 5.4 Video Generation

**Sora 2 / Video Diffusion Models** process video as sequences of visual patches over time, learning spatial and temporal relationships. The field is primarily focused on capabilities rather than interpretability, but emerging work on temporal attention analysis could inform temporal concept discovery.
- Video models paper: https://openai.com/index/video-generation-models-as-world-simulators/

### 5.5 Music and Audio Generation

**"Discovering and Steering Interpretable Concepts in Large Generative Music Models"** (2025) applies SAEs to autoregressive music generators, extracting interpretable features from transformer residual streams. Discovered concepts align with traditional music constructs like chord progressions while also revealing novel statistical structures.
- Paper: https://arxiv.org/abs/2505.18186

### 5.6 Connection to Ridge Detection

The GAN latent space analysis methods (intrinsic dimension estimation, optimal transport-based disentanglement) provide mature techniques that could be adapted for diffusion model latent space analysis. The key insight from GANs is that semantic boundaries in latent space correspond to changes in local intrinsic dimensionality -- this maps directly to ridge detection via Hessian eigenvalue analysis.

---

## 6. Cross-Modal Concept Transfer

### 6.1 Universal Sparse Autoencoders

**Universal Sparse Autoencoders (USAEs)** (ICML 2025) is the landmark paper on cross-model concept alignment. A single overcomplete SAE encodes activations from multiple models, reconstructing any model's activations from any other's. Discovers semantically coherent universal concepts ranging from low-level features (colors, textures) to higher-level structures (parts, objects). Enables coordinated activation maximization across models.
- Paper: https://arxiv.org/abs/2502.03714

**SPARC: Concept-Aligned SAEs for Cross-Model and Cross-Modal Interpretability** (2025) extends the universal SAE approach to cross-modal settings.
- Paper: https://arxiv.org/html/2507.06265

### 6.2 SAEs for Vision-Language Models

**"Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models"** (NeurIPS 2025) extends SAEs to CLIP, with comprehensive monosemanticity evaluation. Proposes MonoSemanticity (MS) score for vision tasks. SAE interventions on CLIP's vision encoder directly steer multimodal LLM outputs (e.g., LLaVA) without language model modifications.
- Paper: https://arxiv.org/abs/2504.02821
- Code: https://github.com/ExplainableML/sae-for-vlm

### 6.3 CLIP Cross-Modal Alignment

Recent work at ICLR 2025 shares parameter space between vision and language encoders and adds intra-modality separation objectives to reduce the cross-modal alignment gap. Research shows that cross-modal representations converge in deeper model layers.
- Paper: https://proceedings.iclr.cc/paper_files/paper/2025/file/cc1de06a58ba1db43538a37e076e466d-Paper-Conference.pdf

### 6.4 Connection to Ridge Detection

USAEs provide a direct framework for asking: "Do ridges in different models' latent spaces correspond to the same semantic boundaries?" If universal concepts exist across models, then phase boundaries should be structurally similar. Your CLIP pullback metric analysis already connects to this -- CLIP encodes universal visual concepts, so ridges detected via CLIP similarity should correspond to universal semantic transitions.

---

## 7. Practical Tools & Frameworks

### 7.1 SAE Training and Analysis

| Tool | Purpose | URL |
|------|---------|-----|
| **SAELens** | Training and analyzing SAEs on any PyTorch model | https://github.com/jbloomAus/SAELens |
| **SAE-Vis / SAEDashboard** | Feature dashboard visualization | https://github.com/jbloomAus/SAEDashboard |
| **SAEBench** | LLM SAE benchmark suite | Works with SAELens |
| **Language-Model-SAEs** | Training/analyzing SAEs and frontier variants | https://github.com/OpenMOSS/Language-Model-SAEs |
| **Neuronpedia** | Interactive SAE feature exploration (50M+ features) | https://www.neuronpedia.org/ |

### 7.2 Interpretability Frameworks

| Tool | Purpose | URL |
|------|---------|-----|
| **Anthropic Circuit Tracing** | Attribution graph generation for LLMs | https://www.anthropic.com/research/open-source-circuit-tracing |
| **ConceptAttention** | DiT attention-based concept saliency maps | https://github.com/helblazer811/ConceptAttention |
| **NoiseCLR** | Unsupervised diffusion direction discovery | https://github.com/gemlab-vt/NoiseCLR |
| **Concept Sliders** | LoRA-based concept control | https://github.com/rohitgandikota/sliders |
| **SAeUron** | SAE-based concept unlearning in diffusion | https://github.com/cywinski/SAeUron |

### 7.3 Latent Space Exploration

| Tool | Purpose | URL |
|------|---------|-----|
| **Smooth Diffusion** | Plug-and-play latent smoothness (Smooth-LoRA) | https://github.com/SHI-Labs/Smooth-Diffusion |
| **Hessian Geometry** | Fisher metric / phase boundary analysis | https://github.com/alobashev/hessian-geometry-of-diffusion-models |
| **Spacetime Geometry** | Fisher-Rao geodesics on denoising distributions | https://github.com/rafalkarczewski/spacetime-geometry |
| **SCALEX** | Scalable H-space concept extraction via NL prompts | (WACV 2026) |
| **Autolume 2.0** | No-code GAN latent space exploration | NeurIPS 2024 |
| **PAIR SAE Explorer** | Google's interactive SAE visualization | https://www.pair.withgoogle.com/explorables/sae/ |

### 7.4 Benchmarks

| Benchmark | Evaluates | Reference |
|-----------|-----------|-----------|
| **MIB** | Circuit localization, causal variable localization | Hugging Face Mechanistic Interpretability Benchmark |
| **UnlearnCanvas** | Concept/style unlearning in diffusion models | Used by SAeUron |
| **I2P** | Inappropriate image generation detection | Used by multiple concept erasure papers |
| **ISTD** | Latent space smoothness (interpolation standard deviation) | Smooth Diffusion (CVPR 2024) |
| **MonoSemanticity Score** | Vision SAE feature interpretability | NeurIPS 2025 |

---

## 8. Emerging & Speculative Directions

### 8.1 World Models and Concept Discovery

World models are converging with diffusion models. OpenAI's Sora 2 (September 2025), Google's Genie 3 (August 2025), and World Labs' Marble (November 2025) all learn implicit physical simulations. The compositional approach -- building world models from smaller expert modules rather than monolithic networks -- suggests concepts should emerge naturally as module boundaries.
- Overview: https://www.scientificamerican.com/article/world-models-could-unlock-the-next-revolution-in-artificial-intelligence/

### 8.2 Temporal Concept Evolution During Training

**"Emergence of Hidden Capabilities"** (NeurIPS 2024) studies concept learning dynamics in concept space, revealing ordered emergence (shape before color) and geometric structure underlying capability emergence.
- Paper: https://proceedings.neurips.cc/paper_files/paper/2024/file/99e6bcf460ea36818cf236da29311e73-Paper-Conference.pdf

### 8.3 Safety-Relevant Concept Discovery

The entire concept unlearning field (SAeUron, ConceptPrune, CURE, MACE) is driven by safety concerns. Key open question: how do we discover *all* harmful concepts a model has learned, rather than testing known ones? This connects to your ridge work: unexpected ridges in latent space might reveal undocumented concept boundaries, including potentially harmful ones.

**Meta-Unlearning** (ICCV 2025) addresses concept relearning prevention, using meta-learning to prevent fine-tuning from recovering erased concepts.
- Paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_Meta-Unlearning_on_Diffusion_Models_Preventing_Relearning_Unlearned_Concepts_ICCV_2025_paper.pdf

### 8.4 Multi-Scale Concept Hierarchies

The Caffarelli regularity paper establishes a mathematical framework for hierarchical boundaries: coarse boundaries form first, finer ones emerge within them. This is consistent with TIDE's finding that DiTs learn hierarchical semantics (3D structure -> object class -> fine-grained details) and the PNAS paper on hierarchical data producing hierarchical phase transitions.

### 8.5 Latent Space as Spacetime

The ICLR 2026 Oral on spacetime geometry of diffusion models (Karczewski et al.) is perhaps the most conceptually radical recent paper. By treating (x_t, t) as a spacetime manifold with Fisher-Rao metric, it unifies the temporal dimension of denoising with the spatial dimension of the latent space. This suggests that your ridges should be understood not just as spatial boundaries at a fixed timestep, but as hypersurfaces in the full spacetime.

---

## 9. Research Gaps and Opportunities

### 9.1 Direct Gaps Relevant to Your Work

1. **No one has combined Hessian ridge detection with SAE feature labeling.** This is a natural next step: use SAEs trained on diffusion model activations to semantically annotate the phases separated by your ridges.

2. **The fractal structure of phase boundaries (Lobashev ICML 2025) lacks empirical characterization at scale.** Your ridge detection tools could provide the first systematic catalog of phase boundary geometry across diverse prompt pairs.

3. **The spacetime geometry framework (ICLR 2026) has not been connected to concept discovery.** Computing Fisher-Rao geodesics and then identifying where concept transitions occur along them would unify geometric and semantic views.

4. **No existing work maps ridge topology to concept taxonomy.** TDA methods (persistent homology) could characterize the topological structure of your ridge networks, potentially revealing hierarchical concept organization.

5. **The relationship between CFG strength and ridge sharpness is unexplored.** CFG pushes distributions toward modes; this should sharpen ridges. Understanding this relationship could inform better generation strategies.

### 9.2 Broader Open Questions

- Do all diffusion models learn the same phase boundaries, or are boundaries architecture-specific? (USAEs could test this)
- How do phase boundaries evolve during model training? (TIDE + training checkpoints)
- Can ridge detection serve as a safety audit tool, revealing concept boundaries the model has learned that developers did not intend?
- What is the relationship between ridge structure and model capability? Do more capable models have more complex ridge topologies?

---

## 10. Recommended Reading Priority

For researchers working on phase boundary ridges, the following papers are most directly relevant, ordered by priority:

1. **Lobashev et al. (ICML 2025)** - Hessian geometry, Fisher metric, fractal phase transitions -- the closest existing work to yours
2. **Karczewski et al. (ICLR 2026 Oral)** - Spacetime geometry with Fisher-Rao metric -- the theoretical framework your work should engage with
3. **Sclocchi et al. (PNAS 2025)** - Phase transitions revealing hierarchical data structure -- the phenomenology your ridges capture
4. **Caffarelli Regularity paper (NeurIPS ML4PS 2025)** - Mathematical foundations for hierarchical phase boundaries
5. **TIDE (AAAI 2025)** - Temporal-aware SAEs for DiTs -- the tool for understanding timestep-dependent ridges
6. **SAeUron (ICML 2025)** - Practical SAE application to diffusion models -- the method for labeling ridge-separated phases
7. **ConceptAttention (ICML 2025)** - What DiT attention layers encode -- mechanistic understanding of ridge origins
8. **NoiseCLR (CVPR 2024)** - Unsupervised direction discovery -- complementary to ridge detection
9. **Universal SAEs (ICML 2025)** - Cross-model concept alignment -- for testing ridge universality
10. **Smooth Diffusion (CVPR 2024)** - The inverse problem: smoothing away the ridges you detect

---

## Sources

- [NoiseCLR - CVPR 2024](https://arxiv.org/abs/2312.05390)
- [Self-Discovering Diffusion Latent Directions - CVPR 2024](https://arxiv.org/html/2311.17216v2)
- [Discovering Interpretable Directions via Jacobian](https://arxiv.org/abs/2303.11073)
- [CDLC - Concept Directions via Latent Clustering](https://arxiv.org/abs/2505.07073)
- [Unsupervised Compositional Concepts Discovery - ICCV 2023](https://arxiv.org/abs/2306.05357)
- [Concept Sliders - ECCV 2024](https://github.com/rohitgandikota/sliders)
- [Prompt Sliders - ECCVW 2024](https://arxiv.org/abs/2409.16535)
- [When Are Concepts Erased? - NeurIPS 2025](https://arxiv.org/abs/2505.17013)
- [Erased or Dormant? 2025](https://arxiv.org/html/2505.16174v2)
- [ConceptPrune - ICLR 2025](https://arxiv.org/abs/2405.19237)
- [Mechanistic Interpretability of Diffusion Models 2025](https://arxiv.org/abs/2506.17237)
- [ConceptAttention - ICML 2025](https://arxiv.org/abs/2502.04320)
- [Cross-Attention Head Patterns](https://openreview.net/forum?id=1vggIT5vvj)
- [SAeUron - ICML 2025](https://arxiv.org/abs/2501.18052)
- [Steering DiTs with SAEs](https://openreview.net/forum?id=J48XM0au4u)
- [TIDE - AAAI 2025](https://arxiv.org/abs/2503.07050)
- [Concept Steerers 2025](https://arxiv.org/html/2501.19066)
- [Emergence and Evolution of Concepts in Diffusion Models 2025](https://arxiv.org/abs/2504.15473)
- [Phase Transition Reveals Hierarchical Nature - PNAS 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725779/)
- [Caffarelli Regularity - NeurIPS ML4PS 2025](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_71.pdf)
- [Hessian Geometry of Latent Space - ICML 2025](https://arxiv.org/abs/2506.10632)
- [Spacetime of Diffusion Models - ICLR 2026 Oral](https://arxiv.org/abs/2505.17517)
- [Understanding Latent Space via Riemannian Geometry - NeurIPS 2023](https://arxiv.org/abs/2307.12868)
- [Be Tangential to Manifold 2025](https://arxiv.org/abs/2510.05509)
- [Score-Based Riemannian Metric 2025](https://arxiv.org/abs/2505.11128)
- [Score-based Pullback Geometry 2024](https://arxiv.org/abs/2410.01950)
- [Smooth Diffusion - CVPR 2024](https://github.com/SHI-Labs/Smooth-Diffusion)
- [Persistent Homology for LLM Latent Spaces 2025](https://arxiv.org/abs/2505.20435)
- [Cubical Homology for Generative Models - KDD 2025](https://dl.acm.org/doi/10.1145/3711896.3736943)
- [Anthropic Scaling Monosemanticity 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [Anthropic Circuit Tracing 2025](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
- [Sparse Crosscoders 2024](https://transformer-circuits.pub/2024/crosscoders/index.html)
- [SAE Survey 2025](https://arxiv.org/html/2503.05613v3)
- [CAV Revisited - ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/852870d56e20a22fcd52ea14523fb820-Paper-Conference.pdf)
- [Universal SAEs - ICML 2025](https://arxiv.org/abs/2502.03714)
- [SPARC 2025](https://arxiv.org/html/2507.06265)
- [SAEs for VLMs - NeurIPS 2025](https://arxiv.org/abs/2504.02821)
- [CLIP Cross-Modal Alignment - ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/cc1de06a58ba1db43538a37e076e466d-Paper-Conference.pdf)
- [Local Dimension in GAN Latent Spaces 2025](https://www.sciencedirect.com/science/article/abs/pii/S0031320324006654)
- [L-VAE 2025](https://arxiv.org/abs/2507.02619)
- [CL-Dis - ECCV 2024](https://arxiv.org/abs/2402.02346)
- [Decompositional Neural Scene Reconstruction - CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Ni_Decompositional_Neural_Scene_Reconstruction_with_Generative_Diffusion_Prior_CVPR_2025_paper.pdf)
- [GP-NeRF - CVPR 2024 Highlight](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_GP-NeRF_Generalized_Perception_NeRF_for_Context-Aware_3D_Scene_Understanding_CVPR_2024_paper.pdf)
- [Discovering Concepts in Music Models 2025](https://arxiv.org/abs/2505.18186)
- [Decoding Diffusion 2024](https://arxiv.org/abs/2410.21314)
- [SCALEX - WACV 2026](https://arxiv.org/abs/2511.13750)
- [Emergence of Hidden Capabilities - NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/99e6bcf460ea36818cf236da29311e73-Paper-Conference.pdf)
- [Meta-Unlearning - ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_Meta-Unlearning_on_Diffusion_Models_Preventing_Relearning_Unlearned_Concepts_ICCV_2025_paper.pdf)
- [Neuronpedia](https://www.neuronpedia.org/)
- [SAELens](https://github.com/jbloomAus/SAELens)
- [Anthropic Open-Source Circuit Tracing](https://www.anthropic.com/research/open-source-circuit-tracing)
- [MIB Benchmark](https://www.corti.ai/stories/gim-a-new-standard-for-mechanistic-interpretability)
- [Concept-Based XAI Metrics and Benchmarks 2025](https://arxiv.org/html/2501.19271v1)
- [AttnDreamBooth - NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/465a13a95741fab2e912f98adb07df1d-Paper-Conference.pdf)
- [Sander Dieleman: Generative Modelling in Latent Space](https://sander.ai/2025/04/15/latents.html)
- [CFG++ Manifold-Constrained Guidance](https://openreview.net/forum?id=E77uvbOTtp)
