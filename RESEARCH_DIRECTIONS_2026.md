# Research Directions for Phase Boundary Ridges in Diffusion Models

## April 2026 — Comprehensive Ideation Document

---

## Preamble: What We Have and What It Means

You have established that text-to-image diffusion models contain sharp semantic phase boundaries (ridges) in their text-embedding interpolation space. These ridges are ubiquitous (100% of slices), detected consistently by three independent metrics (DINOv2, CLIP, pixel MSE at rho=0.94), partially universal across CLIP-based architectures (rho=0.36-0.49), and exhibit critical exponents (beta=0.346) consistent with the 3D Ising universality class. You have also established important negative results: noise-space Lipschitz does not predict text-space ridges, and SAE-based concept labeling on text-interpolation ridges is circular.

The deep question this work raises is: **Why does a neural network trained on gradient descent over image-text pairs spontaneously organize its representation space into discrete phases separated by sharp boundaries that obey the same scaling laws as physical phase transitions?** This is not a metaphor. Beta=0.346 is a number, and it matches 3D Ising (beta=0.326) more closely than any other universality class. That demands explanation.

Below are 18 research directions, ranked by a composite score of novelty, impact, and feasibility.

---

## Tier 1: Highest Priority (Novel, High Impact, Feasible)

### 1. Ridge Ontogeny: How Do Phase Boundaries Form During Training?

**Composite Score: 9.5/10**

**Core hypothesis**: Phase boundaries do not exist in a randomly initialized model. They emerge during training. The ORDER in which boundaries form reveals the hierarchical structure of concepts the model learns, and the DYNAMICS of their formation may exhibit the same critical phenomena as physical phase transitions (analogous to symmetry breaking during cooling).

**Why it's novel**: Sclocchi et al. (PNAS 2025) showed that diffusion models exhibit phase transitions during the backward (denoising) process, and the "Emergence of Hidden Capabilities" (NeurIPS 2024) showed ordered concept emergence during training (shape before color). But nobody has tracked the SPATIAL STRUCTURE of phase boundaries across training checkpoints. The question is not "when does the model learn cats vs dogs" but "when does the boundary between cat-space and dog-space crystallize, and does it sharpen according to Ising-like dynamics?"

**What it would prove**: If ridge sharpness follows a power law in training step (analogous to cooling through a critical temperature), it would establish that concept formation in neural networks is literally a phase transition — not metaphorically, but in the statistical mechanics sense. If different concept boundaries crystallize at different "temperatures" (training stages), it would provide a physics-grounded theory of curriculum learning and concept hierarchy.

**Surprising finding potential**: The critical exponent of ridge formation during training might DIFFER from the critical exponent of ridge geometry in the trained model (beta=0.346). This would mean the dynamics of learning and the statics of the learned representation belong to different universality classes — a finding that would reshape how we think about the relationship between optimization and representation.

**Feasibility**: Requires training checkpoints of a diffusion model. SD 1.5 checkpoints are available (CompVis released intermediate checkpoints). FLUX checkpoints are not public, but you could fine-tune a small model from scratch on a controlled dataset. On 6x RTX 4090, training a small DiT on CIFAR-10 or a subset of ImageNet is feasible (days, not weeks). For each checkpoint, run your existing ridge detection pipeline on a fixed set of prompt triplets.

**Concrete first experiment**:
1. Obtain 10-20 training checkpoints of Stable Diffusion 1.5 (or train a small DiT from scratch on a curated dataset with known semantic categories)
2. For each checkpoint, compute ridge maps on 5 fixed prompt triplets using DINOv2 sensitivity
3. Measure: (a) ridge sharpness (max/median sensitivity ratio), (b) ridge location stability across consecutive checkpoints, (c) fit sharpness vs training step to a power law
4. Test whether "coarse" boundaries (animal vs object) form before "fine" boundaries (cat vs dog)

**Connection to literature**: Grokking (arXiv:2603.01192) shows that learning transitions are phase transitions between competing loss basins. Ridge ontogeny would show the SPATIAL manifestation of these transitions in representation space. The implicit dynamical regularization work (NeurIPS 2025 Best Paper, arXiv:2505.17638) shows two timescales in training (tau_gen and tau_mem) — ridges should form at tau_gen and potentially reorganize at tau_mem.

---

### 2. The Universality Class Problem: Why 3D Ising?

**Composite Score: 9.3/10**

**Core hypothesis**: The beta=0.346 critical exponent is not coincidental. The text-embedding space of a diffusion model is an effective 3D system (despite the embedding space being ~4096-dimensional) because the EFFECTIVE dimensionality at the scale of ridge formation is approximately 3. The universality class is determined by the symmetry of the order parameter and the effective dimensionality of the system, both of which can be measured.

**Why it's novel**: You measured beta. Nobody has measured the OTHER critical exponents (gamma for susceptibility, nu for correlation length, delta for the equation of state, alpha for specific heat). A complete set of exponents would UNIQUELY identify the universality class and rule out coincidence. Furthermore, nobody has measured the effective dimensionality of the phase boundary manifold itself.

**What it would prove**: A complete match to 3D Ising exponents (beta=0.326, gamma=1.237, nu=0.630, delta=4.789) would be extraordinary — it would mean that concept boundaries in neural networks obey the same mathematics as magnetic domain walls in three-dimensional ferromagnets. A MISMATCH (e.g., beta matches but gamma doesn't) would be equally interesting, suggesting a novel universality class specific to neural network representations.

**Surprising finding potential**: The exponents might vary by model architecture. CLIP-based models might be 3D Ising while T5-based models (PixArt) are in a different class. This would provide a physics-based taxonomy of neural network architectures.

**Feasibility**: You already have the infrastructure. Measuring gamma requires computing the "susceptibility" (variance of the order parameter near the critical point). Measuring nu requires computing the correlation length (spatial extent of ridge-ridge correlations). These are straightforward extensions of your existing analysis. The main challenge is collecting enough data for robust fits — you need at least 50-100 ridge profiles per exponent.

**Concrete first experiment**:
1. Select 10 well-characterized ridges from existing data (clear single-ridge crossings)
2. For each ridge, measure: (a) order parameter profile (DINOv2 distance as function of perpendicular distance from ridge), (b) correlation length (exponential decay of sensitivity autocorrelation parallel to ridge), (c) susceptibility (variance of sensitivity in a window near the ridge)
3. Fit each to power laws: M ~ |t-tc|^beta, chi ~ |t-tc|^(-gamma), xi ~ |t-tc|^(-nu)
4. Check Widom scaling relation: gamma = beta(delta - 1). If this holds, the system is in thermodynamic equilibrium near criticality.

**Connection to literature**: The absorbing phase transitions paper (Phys. Rev. Research 2025) showed that MLPs exhibit mean-field universality while CNNs exhibit directed percolation universality during signal propagation. Your finding of 3D Ising in the OUTPUT space (rather than the signal propagation space) would establish a third universality class for a different phenomenon in the same systems.

---

### 3. Ridge Topology in Higher Dimensions: Are Phase Boundaries a Percolation Network?

**Composite Score: 9.0/10**

**Core hypothesis**: In 2D slices, ridges appear as lines. In 3D, they form surfaces. In the full ~4096-dimensional text embedding space, they form a high-dimensional network. This network may exhibit PERCOLATION — a connected, space-spanning structure that divides the embedding space into isolated semantic regions. The percolation threshold and its exponents would reveal the large-scale organization of concept space.

**Why it's novel**: Nobody has characterized the connectivity or topology of phase boundaries in high-dimensional generative model spaces. Your 2D slices are like core samples through a 3D ore body — you see cross-sections but not the full structure. The question is whether the ridges form a single connected manifold (percolating) or many disconnected pieces (sub-critical).

**What it would prove**: If ridges percolate, then EVERY concept in the model is separated from EVERY other concept by a phase boundary — there are no "smooth transitions" at the global scale, only at the local scale within a phase. This would mean concept space is fundamentally discrete despite being embedded in a continuous vector space. If ridges DON'T percolate, then some concepts blend smoothly into each other, creating a hierarchical structure where sharp boundaries exist only between semantically distant concepts.

**Surprising finding potential**: The percolation transition itself might be tunable — e.g., by varying CFG scale. At low CFG, ridges might be subcritical (disconnected, soft boundaries). At high CFG, they might percolate (everything sharply separated). The CFG-driven percolation transition would be a new phenomenon with practical implications for generation quality.

**Feasibility**: You already have 3D ridge detection (marching cubes). Extending to 4D-6D using Sobol sampling and GP-based active learning (as documented in your RESEARCH_high_dim_exploration.md) is planned infrastructure. The percolation analysis itself is well-established (connectivity analysis, spanning cluster detection).

**Concrete first experiment**:
1. Choose 4 prompts spanning semantically diverse concepts (animal, vehicle, building, food)
2. Use Sobol sampling (4096 points) in the 3D simplex of 4 prompts
3. Compute 1-step Jacobian sensitivity at each point using fast detection
4. Threshold at various tau values and measure: (a) largest connected component size vs tau, (b) number of components vs tau, (c) look for a sharp transition (percolation threshold)
5. Compare the percolation exponents to known 3D percolation values (beta_perc=0.41, gamma_perc=1.80, nu_perc=0.88)

**Connection to literature**: High-dimensional percolation theory (Diskin 2025, Random Structures & Algorithms) shows that percolation on product graphs has sharp thresholds. The "Critical Percolation as a Framework" ICLR workshop paper explicitly proposed percolation as a framework for understanding deep learning. Your work would provide the first EMPIRICAL measurement of percolation in a generative model's representation space.

---

### 4. Ridges as the Skeleton of a Voronoi Decomposition: Concept Geometry

**Composite Score: 8.8/10**

**Core hypothesis**: The phase boundaries in text-embedding space form the Voronoi diagram of a finite set of "concept attractors" — prototypical points in embedding space that act as the centers of semantic basins. If true, the entire ridge structure can be predicted from just the locations of these attractors, and the number of attractors is the effective number of discrete concepts the model has learned.

**Why it's novel**: Your DBSCAN analysis already shows 5-14 clusters per 2D slice, and the ridges align with cluster boundaries. But nobody has tested the VORONOI HYPOTHESIS: are the boundaries equidistant between cluster centers in the model's INTRINSIC metric (not Euclidean distance in embedding space, but Fisher-Rao distance)?

**What it would prove**: If boundaries are Voronoi in Fisher-Rao metric, then the model's concept space has an elegant mathematical structure: it's a metric space with a finite set of attractors, and the denoising process is essentially nearest-neighbor classification in this metric space. This would connect diffusion models to classical prototype theory in cognitive science and information geometry.

**Surprising finding potential**: The Voronoi decomposition might be hierarchical — coarse attractors at one scale, finer attractors at another, like a Voronoi treemap. This would provide a constructive proof that diffusion models learn hierarchical concept taxonomies, with each level of the hierarchy corresponding to a different denoising timestep (matching Sclocchi et al.'s hierarchical phase transitions).

**Feasibility**: Moderate. You need to: (1) identify cluster centers from your existing DBSCAN, (2) compute pairwise Fisher-Rao distances (using the spacetime geometry framework of Karczewski et al.), (3) construct the Voronoi diagram in this metric, (4) compare predicted boundaries to observed ridges. Step 2 is the hardest — Fisher-Rao distances require score function evaluations, but Karczewski et al. provide simulation-free estimators.

**Concrete first experiment**:
1. Take a well-characterized 2D slice with clear clusters
2. Identify cluster centers (centroid of each DBSCAN cluster in embedding space)
3. For each ridge point, compute the two nearest cluster centers
4. Test whether the ridge point is equidistant (in cosine distance) from these two centers
5. Measure the deviation from perfect Voronoi structure — if small (<10%), the hypothesis is supported
6. Compare Euclidean-Voronoi vs DINOv2-distance-Voronoi vs raw-latent-Voronoi to find which metric produces the best fit

---

### 5. Mechanistic Circuit Analysis: Which Transformer Blocks Create Ridges?

**Composite Score: 8.7/10**

**Core hypothesis**: Ridges are created by specific transformer blocks in the diffusion model, and the block at which a ridge "forms" during the forward pass corresponds to the level of abstraction of the concept boundary. Early blocks (which process coarse features) should create boundaries between broad categories (animal vs object), while later blocks should create boundaries between fine-grained categories (tabby cat vs calico cat).

**Why it's novel**: The Mechanistic Interpretability of Diffusion Models paper (arXiv:2506.17237) identified 8 functionally distinct attention mechanisms but didn't connect them to phase boundaries. Your SPEEDUP_RESEARCH.md already notes that early blocks suffice for ridge detection (rho=0.82 for mid-block at 1 step). But nobody has systematically mapped WHICH ridges require WHICH blocks. ConceptAttention (ICML 2025) showed DiT attention layers learn interpretable features but didn't connect to phase boundary geometry.

**What it would prove**: If different ridges "crystallize" at different blocks, it would provide a mechanistic explanation for why ridges exist: they're the spatial manifestation of discrete attention head activations. This would connect your phenomenological observations (ridges exist, have Ising exponents) to a mechanistic cause (specific circuits).

**Concrete first experiment**:
1. Implement early-exit at blocks 8, 16, 24, 32, 40, 48, 56 (you already have the architecture for this from SPEEDUP_RESEARCH.md)
2. For each exit point, compute the full ridge map on 5 prompt triplets
3. Identify which ridges appear at which exit depth
4. Classify: ridges that appear by block 16 = "coarse concept boundaries"; ridges that only appear after block 40 = "fine concept boundaries"
5. Validate by checking whether "coarse ridges" separate semantically distant prompts while "fine ridges" separate similar prompts

---

## Tier 2: High Priority (Strong Novelty or Impact)

### 6. Ridge-Guided Adversarial Safety Audit

**Composite Score: 8.5/10**

**Core hypothesis**: Ridges in embedding space mark the boundaries of what a model considers "the same concept." By systematically scanning for ridges in the neighborhood of safety-relevant prompts, you can discover the EXACT boundary between safe and unsafe content generation — including edge cases that text-based safety filters miss.

**Why it's novel**: Current safety work (ConceptPrune ICLR 2025, SAeUron ICML 2025, Meta-Unlearning ICCV 2025) operates in concept space — identifying and removing specific concepts. But they can't discover UNKNOWN unsafe boundaries. Ridge detection is agnostic to what the concepts are; it finds ALL boundaries. This turns safety auditing from a game of whack-a-mole (enumerate known bad concepts) into a systematic geometric scan.

**What it would prove**: If you can find ridges in the neighborhood of borderline prompts (e.g., between "person swimming" and more explicit content), and these ridges precisely mark where the model's safety boundary lies, then ridge detection becomes a practical safety tool. The exciting possibility: discovering that certain seemingly-safe prompts are ON a ridge that leads to unsafe content with minor perturbation.

**Practical value**: This could become a product. Model providers (Stability AI, Black Forest Labs) need to audit their models for safety. A geometric audit that maps ALL boundaries — not just known ones — is more robust than keyword-based or classifier-based approaches.

**Feasibility**: High. You already have the tool. The experiment is: take a set of borderline prompts, interpolate between them, run ridge detection, and check whether ridges predict content transitions. The main challenge is ethical: you need to carefully handle the generated content.

**Concrete first experiment**:
1. Define 5 prompt pairs that span a known safety boundary (e.g., "medical anatomy diagram" <-> "nude figure", "war documentary" <-> "graphic violence", "chemistry equipment" <-> "drug manufacturing")
2. Run 1D interpolation (50 points) between each pair
3. Detect ridges using DINOv2 sensitivity
4. For each ridge, generate high-quality images on both sides and manually classify as safe/unsafe
5. Measure: does the ridge PRECISELY mark the safety boundary? How sharp is it?

---

### 7. Cross-Modality Ridges: Do Text, Image, and Audio Models Share Phase Boundaries?

**Composite Score: 8.3/10**

**Core hypothesis**: If ridges reflect the structure of human concepts (rather than arbitrary model internals), then the SAME concept boundaries should appear in text-to-image, text-to-audio, and text-to-video models. The Universal SAE (ICML 2025) showed that concept features are shared across vision models. Ridges might be the geometric manifestation of these universal concepts.

**Why it's novel**: Cross-architecture ridge correlation (rho=0.36-0.49) shows partial universality within text-to-image models. But nobody has tested whether the same ridges appear in entirely different modalities. If "cat" and "dog" have a sharp boundary in image space AND in audio space (meow vs bark), the universality claim becomes dramatically stronger.

**What it would prove**: If ridge maps correlate across modalities, it would suggest that phase boundaries reflect the structure of CONCEPTS THEMSELVES — how human knowledge is organized — rather than artifacts of visual processing. This would be a finding about cognition, not just about neural networks.

**Feasibility**: Moderate. Requires access to a text-to-audio model (e.g., AudioLDM, MusicGen). The key challenge is defining a cross-modal sensitivity metric — you can't use DINOv2 for audio. You'd need CLAP (audio-text CLIP) or a modality-specific perceptual metric. The interpolation machinery is identical.

**Concrete first experiment**:
1. Choose 5 prompt triplets that have both visual and auditory semantics (e.g., "thunderstorm" / "birdsong" / "traffic noise")
2. Run ridge detection on FLUX (image) and AudioLDM (audio) using the same prompt triplets
3. Use DINOv2 for image ridges, CLAP embeddings for audio ridges
4. Compute Spearman correlation between image and audio ridge maps

---

### 8. Ridges Under Smooth Diffusion: The Anti-Ridge Experiment

**Composite Score: 8.2/10**

**Core hypothesis**: Smooth Diffusion (CVPR 2024) enforces Lipschitz continuity in the latent space via Smooth-LoRA, explicitly trying to eliminate the discontinuities your ridges represent. If you apply your ridge detection to a Smooth-Diffusion model, ridges should be ATTENUATED but not eliminated. The residual ridges that survive smoothing are the most fundamental concept boundaries — they resist elimination because they reflect irreducible semantic discreteness.

**Why it's novel**: Nobody has measured ridge structure before and after smoothing. Smooth Diffusion claims to improve interpolation quality, but doesn't characterize what happens to phase boundaries. This experiment directly tests whether ridges are (a) an artifact of training that can be smoothed away, or (b) a fundamental property of mapping continuous inputs to discrete concepts.

**What it would prove**: If ridges survive smoothing, it proves they're not artifacts — they're intrinsic to the task of representing discrete concepts in continuous space. The degree of attenuation would reveal a hierarchy: fundamental boundaries (survive heavy smoothing) vs superficial boundaries (disappear with mild smoothing). If ridges are completely eliminated, it proves they're artifacts of standard training, which would actually be a useful finding for improving generation quality.

**Feasibility**: High. Smooth-LoRA is available as a plug-and-play adapter. Load FLUX + Smooth-LoRA, run your existing pipeline, compare ridge maps.

**Concrete first experiment**:
1. Generate ridge maps for 10 prompt triplets on standard FLUX Klein
2. Apply Smooth-LoRA to FLUX Klein
3. Re-generate ridge maps for the same 10 triplets
4. Measure: (a) reduction in peak sensitivity, (b) change in ridge count, (c) which ridges survive vs disappear, (d) change in critical exponent beta

---

### 9. Ridges Predict Compositional Failure

**Composite Score: 8.0/10**

**Core hypothesis**: When users prompt for compositional scenes (e.g., "a red car next to a blue house"), diffusion models often fail — producing a blue car or a red house. These compositional failures occur because the prompt embedding lands NEAR a phase boundary between "red car" mode and "blue car" mode. Ridge detection can predict which compositions will fail before generating the full image.

**Why it's novel**: Compositional failure is a massive practical problem (see Attend-and-Excite, StructureDiffusion, etc.). Current solutions intervene during denoising. Nobody has connected compositional failure to phase boundary structure. If prompts that fail compositionally are systematically located near ridges, it provides both a prediction tool AND a theoretical explanation for WHY composition is hard.

**What it would prove**: If compositional failures cluster near ridges, then the difficulty of composition is not a bug in the denoising process — it's a geometric property of the embedding space. Models fail at composition because the compositional prompt sits on a ridge between two simpler interpretations. This reframes the entire compositional generation problem.

**Practical value**: Fast ridge detection (1-step Jacobian, 7x cheaper) could be used to flag prompts that are likely to fail before spending compute on full generation. The user could then be offered a slightly rephrased prompt that moves off the ridge.

**Concrete first experiment**:
1. Collect 100 known-difficult compositional prompts from T2I-CompBench or similar
2. For each, create a triplet: [difficult prompt] / [simplified version A] / [simplified version B] (e.g., "red car blue house" / "red car" / "blue house")
3. Run fast ridge detection on the 1D interpolation from simplified-A through the difficult prompt to simplified-B
4. Measure: is the difficult prompt located near a detected ridge? Compute correlation between "distance to nearest ridge" and "compositional failure rate"

---

### 10. The Renormalization Group Structure of Ridges

**Composite Score: 7.8/10**

**Core hypothesis**: Ridges exist at multiple scales — from broad category boundaries (animal vs object) to fine-grained distinctions (persian cat vs siamese cat). If you coarse-grain the embedding space (e.g., by averaging neighboring embeddings), fine ridges should disappear while coarse ridges persist. This coarse-graining procedure is analogous to the Renormalization Group (RG) in statistical physics. The RG FLOW of ridge structure would reveal the hierarchy of concepts.

**Why it's novel**: The RG-deep learning connection has been theorized extensively (Mehta & Schwab 2014, arXiv:1410.3831; "Is Deep Learning an RG Flow?" arXiv:1906.05212; recent work arXiv:2510.25553 on RG for DNN scaling laws). But nobody has IMPLEMENTED RG on the output space of a generative model and measured how phase boundaries flow under coarse-graining. This would be the first empirical RG analysis of concept boundaries.

**What it would prove**: If ridge structure has a well-defined RG flow with fixed points, it would provide a PREDICTIVE theory: given the ridge structure at one scale, you can predict it at all other scales. The fixed points of the flow correspond to "universal concept boundaries" that persist at all scales of description. The universality class (3D Ising?) should be recoverable from the RG flow.

**Feasibility**: Moderate. The coarse-graining can be implemented as: (a) averaging embeddings in a spatial window (real-space RG), (b) truncating PCA components (momentum-space RG), or (c) reducing grid resolution and re-computing sensitivity. All are straightforward with existing infrastructure. The challenge is defining "scale" in embedding space and measuring the flow quantitatively.

**Concrete first experiment**:
1. Compute a high-resolution (100x100) ridge map for a well-characterized triplet
2. Coarse-grain by averaging 2x2, 4x4, 8x8, 16x16 blocks of embeddings
3. Recompute sensitivity at each coarse-graining level
4. Track: (a) which ridges persist at each level, (b) how ridge sharpness scales with block size, (c) whether the scaling is self-similar (power-law)
5. Plot the ridge "phase diagram" as a function of coarse-graining scale

---

## Tier 3: Medium Priority (Novel, Some Risk)

### 11. Koopman Eigenfunctions: Zero-Shot Ridge Detection Without Generation

**Composite Score: 7.5/10**

**Core hypothesis**: The denoising process is a dynamical system. Its Koopman eigenfunctions are scalar fields over the initial condition space whose ZERO LEVEL SETS are exactly the separatrices (phase boundaries). A neural network can learn these eigenfunctions from trajectory data, then predict ridges from embeddings alone — no generation required.

**Why it's novel**: Your RESEARCH_SWEEP doc identifies Koopman eigenfunctions as "highest priority" but notes it hasn't been implemented. The NeurIPS 2025 paper (arXiv:2505.15231) shows this works for RNN dynamical systems. Applying it to diffusion models would be a first.

**What it would prove**: If Koopman eigenfunctions successfully predict ridges without running the denoiser, it would: (a) provide 100-200x speedup over even 1-step detection, (b) prove that ridge structure is encoded in the STRUCTURE of the dynamical system, not just its output, (c) connect your work to a deep mathematical framework (spectral theory of dynamical systems).

**Feasibility**: Requires training a neural network to approximate the Koopman eigenfunction. Training data: ~5000 full denoising trajectories (labeled by final semantic basin). This is a one-time cost (~2-3 hours on 6x RTX 4090). The learned eigenfunction is a small MLP (~1M parameters) that can be evaluated in microseconds. Risk: the eigenfunction may not generalize across prompt triplets (it may be triplet-specific).

**Concrete first experiment**:
1. Choose one prompt triplet
2. Sample 5000 random points in the embedding simplex
3. For each, run full 10-step denoising and classify the output (using DBSCAN on DINOv2 embeddings)
4. Train an MLP: input = interpolation coordinates (alpha, beta), output = basin label (or continuous eigenfunction)
5. Evaluate: does the zero level set of the learned eigenfunction match the DINOv2 ridge map?

---

### 12. Ridges in Fine-Tuned vs Base Models: The LoRA Perturbation Theory

**Composite Score: 7.3/10**

**Core hypothesis**: When you fine-tune a model with LoRA (e.g., to learn a new style or concept), the ridge structure changes. The PERTURBATION to the ridge map caused by LoRA fine-tuning reveals what the LoRA learned — not by examining the LoRA weights, but by examining how they reshape concept boundaries. Furthermore, the ridge perturbation should be proportional to the LoRA rank and the fine-tuning dataset size, providing a physics-like perturbation expansion.

**Why it's novel**: LoRA interpretability is an active area (Concept Sliders, weights2weights), but nobody has characterized how LoRA changes the GEOMETRY of concept space. This is analogous to studying how an external field perturbs the domain structure of a ferromagnet — a classic problem in statistical mechanics.

**What it would prove**: If ridge perturbation theory works (small LoRA = small perturbation = predictable ridge shift), it would provide a principled way to predict the effect of a fine-tune without running it, and a new way to interpret what LoRAs do geometrically.

**Feasibility**: High. Load base model, compute ridge map. Load base + LoRA, compute ridge map. Subtract. Repeat for many LoRAs. The only requirement is a collection of diverse LoRAs (widely available on CivitAI/HuggingFace).

**Concrete first experiment**:
1. Choose 3 prompt triplets and compute baseline ridge maps on FLUX Klein
2. Apply 5 diverse LoRAs (style LoRA, character LoRA, concept LoRA, quality LoRA, safety LoRA)
3. For each LoRA, recompute ridge maps on the same triplets
4. Measure: (a) Spearman rho between base and LoRA ridge maps, (b) which ridges move, (c) do new ridges appear, (d) correlation between LoRA rank and ridge perturbation magnitude

---

### 13. Human-Perceptible Boundaries: Do Ridges Mark What Humans Consider "Different"?

**Composite Score: 7.2/10**

**Core hypothesis**: Ridges mark where the MODEL considers outputs to be in different categories. But do HUMANS agree? If you show humans pairs of images from the same side of a ridge vs from opposite sides, do humans rate the cross-ridge pairs as more "different"? If so, ridges are not just model internals — they capture human perceptual category boundaries.

**Why it's novel**: Your experiment log notes that the human evaluation is incomplete — the definitive dataset (1,750 pairs) exists but lacks independent annotations. The question goes beyond "interestingness" to the more fundamental: do ridges match human categorical perception?

**What it would prove**: If humans consistently rate cross-ridge pairs as more different (after controlling for pixel-level distance), it would establish that diffusion models have learned human-like category boundaries — not just visual features, but the STRUCTURE of human concept space. This connects to the neuroscience of categorical perception (Harnad 1987) and the "warped" perceptual spaces studied in cognitive science.

**Practical value**: If ridges predict human-perceived categorical differences, then ridge detection becomes a tool for evaluating model alignment with human perception — a metric for how well the model has captured human concepts.

**Concrete first experiment**:
1. From 10 well-characterized ridges, select 50 image pairs: 25 "same-phase" (both from the same side of a ridge, similar DINOv2 distance to each other) and 25 "cross-phase" (from opposite sides, matched for DINOv2 distance)
2. Present pairs to 20+ human raters via a simple web interface
3. Ask: "How different are these two images?" (Likert scale 1-7)
4. Compare mean ratings: same-phase vs cross-phase, controlling for DINOv2 distance

---

### 14. Ridges in Noise-Text Joint Space: The 4D Phase Diagram

**Composite Score: 7.0/10**

**Core hypothesis**: You've shown that noise-space Lipschitz does NOT predict text-space ridges (rho=0.13-0.44). But this doesn't mean noise and text are independent — it means they're measuring DIFFERENT ridges. A joint exploration of 2D text + 2D noise might reveal coupled phase boundaries: text-space ridges that are stable across noise realizations, noise-space ridges that are stable across text embeddings, and COUPLED ridges that only appear in the joint space.

**Why it's novel**: Your NOISE_SPACE_RESEARCH.md identifies this as "an open research question perfectly suited for the ridge_explorer tool" and notes that "no existing paper addresses this question." The 4D exploration (2D text + 2D noise) would be a first.

**What it would prove**: If there are coupled ridges (boundaries that only exist in the joint space), it would prove that the model uses noise to RESOLVE ambiguity at text-space ridges — the noise selects which side of a text boundary the image falls on. This would explain the seed stability result (rho=0.64): ridge location in text space is partially determined by the noise realization.

**Feasibility**: The Jacobian SVD approach (LOCO Edit, NeurIPS 2024) can identify the top 2 most impactful noise directions for a given text embedding. Combined with 2 text axes, this gives a 4D grid. At 10 points per axis, that's 10^4 = 10,000 1-step evaluations — about 45 minutes on 6x RTX 4090.

**Concrete first experiment**:
1. Choose a prompt pair with a well-characterized text-space ridge
2. Find the ridge center in text space
3. At the ridge center, compute the top-2 noise-space singular vectors via Jacobian SVD at t=0.6
4. Generate a 20x20 grid: 1 text axis (perpendicular to ridge) x 1 noise axis (top singular vector)
5. Compute sensitivity. Does the text-space ridge shift location as a function of noise direction?

---

### 15. Fractal Dimension of the Ridge Network

**Composite Score: 6.8/10**

**Core hypothesis**: Lobashev et al. (ICML 2025) mention fractal structure of phase boundaries, and your beta=0.346 is consistent with fractal scaling. But the FRACTAL DIMENSION of the ridge network itself has not been measured. If ridges are fractal, their box-counting dimension D_f should be non-integer and related to the critical exponents via D_f = d - beta/nu (where d is the embedding dimension of the slice).

**Why it's novel**: Fractal dimensions have been measured for domain walls in physical systems (ferromagnets, fluid turbulence) but never for concept boundaries in neural networks. Recent work on fractal Ising models (arXiv:2506.17053, arXiv:2507.00956) provides the theoretical framework for connecting fractal geometry to critical exponents in non-integer dimensions.

**What it would prove**: A non-integer fractal dimension would prove that ridges are genuinely fractal — self-similar at multiple scales — rather than smooth curves that happen to have rough edges due to finite grid effects. Combined with the RG analysis (Direction 10), this would provide a complete geometric characterization of concept boundaries.

**Feasibility**: High. Box-counting analysis is a standard technique. You need high-resolution ridge maps (100x100 or higher) and compute the number of boxes of size epsilon that contain ridge points, for varying epsilon. Plot log(N) vs log(1/epsilon) — the slope is D_f.

**Concrete first experiment**:
1. Generate a 200x200 ridge map for a prompt triplet (requires ~20 minutes with current fast scan)
2. Threshold at tau=1.5 to get a binary ridge/non-ridge map
3. Apply box-counting at scales epsilon = 1, 2, 4, 8, 16, 32, 64 pixels
4. Fit D_f from the log-log plot
5. Compare to the prediction D_f = 2 - beta/nu (for a 2D slice, with beta=0.346 and nu from Direction 2)

---

### 16. Ridges as a Training Signal: Phase-Boundary-Aware Diffusion

**Composite Score: 6.5/10**

**Core hypothesis**: If ridges represent failure modes (ambiguous, low-quality images at boundaries), then training with extra data or loss weight near ridges should improve model quality. Conversely, if some ridges are desirable (sharp concept boundaries), preserving them during fine-tuning might improve model fidelity.

**Why it's novel**: The anti-ridge direction (Smooth Diffusion) tries to eliminate ridges globally. The pro-ridge direction (this idea) would selectively strengthen or weaken specific ridges. No existing work uses ridge detection as a training signal.

**What it would prove**: If ridge-aware training produces measurably better models (sharper concept boundaries where desired, smoother transitions where not), it would establish ridge geometry as a principled training diagnostic — analogous to how loss landscape curvature (Fisher information) is used to guide optimization.

**Feasibility**: Low-medium. Requires training or fine-tuning a model, which is expensive. A proof-of-concept could use a small model (e.g., 100M parameter DiT on CIFAR-10) where training is fast.

**Concrete first experiment**:
1. Train a small DiT on CIFAR-10 (10 classes)
2. After training, compute ridge maps for class interpolations (e.g., cat vs dog, car vs truck)
3. Fine-tune with extra loss weight on embeddings near detected ridges (encouraging sharper boundaries) or away from ridges (encouraging smoother interiors)
4. Evaluate: FID, IS, and ridge sharpness before and after fine-tuning

---

### 17. Temporal Ridges: Phase Boundaries Across Denoising Timesteps

**Composite Score: 6.3/10**

**Core hypothesis**: Ridges exist not just in text-embedding space at a fixed timestep, but evolve across the denoising trajectory. Using the spacetime geometry framework (Karczewski et al., ICLR 2026 Oral), ridges should be understood as HYPERSURFACES in the (embedding, timestep) spacetime. The timestep at which a ridge first appears marks the "speciation time" of Sclocchi et al. (PNAS 2025) — when the model first commits to a particular semantic interpretation.

**Why it's novel**: The Caffarelli regularity paper (NeurIPS ML4PS 2025) predicts hierarchical boundary emergence during denoising, and TIDE (AAAI 2025) shows DiTs learn hierarchical semantics across timesteps. But nobody has measured ridge maps at EACH denoising step and tracked how they evolve. The combination of your ridge detection with the spacetime geometry framework would be a natural synthesis.

**What it would prove**: If early-timestep ridges predict late-timestep ridges (but are broader/smoother), it would validate the hierarchical phase boundary prediction and show that your 1-step fast detection works BECAUSE the first step captures the coarsest boundaries — exactly the ones that matter for semantic classification.

**Feasibility**: Medium. Requires computing ridge maps at each of the 10-50 denoising steps, which multiplies compute by 10-50x. For a 20x20 grid at 10 steps, that's ~4000 evaluations (~30 minutes). The analysis is then straightforward: track ridge location and sharpness vs timestep.

**Concrete first experiment**:
1. Choose a prompt triplet with clear ridges
2. Run 20-step denoising on a 30x30 grid, saving intermediate latents at steps 1, 2, 4, 8, 12, 16, 20
3. Compute DINOv2 sensitivity at each saved timestep (using Tweedie prediction x0_hat at each step)
4. Measure: (a) ridge location shift across timesteps, (b) ridge sharpness vs timestep, (c) number of ridges vs timestep (do new ridges appear at later steps?)

---

### 18. Connection to Categorical Perception in Neuroscience

**Composite Score: 6.0/10**

**Core hypothesis**: The phase boundaries in diffusion models are the computational analogue of "categorical perception" in biological neural networks — the well-established phenomenon where humans perceive within-category variation as smaller than between-category variation, even when the physical stimulus differences are equal. The beta=0.346 exponent might match measurements of perceptual boundary sharpness in human psychophysics experiments.

**Why it's novel**: The CATS Net paper (Nature Computational Science, 2026) models human concept formation and shows that concept spaces in biological networks have 20-200 effective dimensions and align with brain responses in the ventral occipitotemporal cortex. But nobody has compared the GEOMETRY of concept boundaries in biological vs artificial neural networks. If both exhibit the same critical exponents, it would suggest a universal law of concept boundary formation.

**What it would prove**: A match between diffusion model critical exponents and human categorical perception measurements would be a landmark finding — it would suggest that the brain and neural networks converge on the same geometry for organizing concepts, despite completely different learning mechanisms. This connects to the PRX Life paper showing that in vivo and in vitro neural populations belong to different universality classes — the question is where artificial neural networks fit in this taxonomy.

**Feasibility**: Low-medium. The experimental side is straightforward (generate stimuli along interpolation paths, present to humans, measure discrimination accuracy as a function of distance from the ridge). The analysis is well-established in psychophysics (categorical perception paradigm). The challenge is recruiting enough participants for statistical power (20-30 participants, ~1 hour each).

**Concrete first experiment**:
1. Generate a 1D interpolation (100 points) between two clearly different prompts (e.g., "cat" to "dog")
2. Find the ridge using DINOv2 sensitivity
3. Create a discrimination task: show participants two images from the interpolation, ask "same or different?"
4. Measure discrimination accuracy as a function of whether the pair crosses a ridge
5. If accuracy is higher for cross-ridge pairs (after controlling for pixel distance), categorical perception is confirmed
6. Fit the discrimination function to a sigmoid and extract the sharpness parameter — compare to beta=0.346

---

## Summary Rankings

| Rank | Direction | Novelty | Impact | Feasibility | Composite |
|------|-----------|---------|--------|-------------|-----------|
| 1 | Ridge Ontogeny (Training Dynamics) | 10 | 10 | 8 | 9.5 |
| 2 | Full Critical Exponent Measurement | 9 | 10 | 9 | 9.3 |
| 3 | Percolation in High-D | 10 | 9 | 7 | 9.0 |
| 4 | Voronoi Decomposition / Concept Attractors | 9 | 9 | 7 | 8.8 |
| 5 | Mechanistic Circuit Analysis | 8 | 9 | 9 | 8.7 |
| 6 | Safety Audit via Ridge Detection | 7 | 10 | 9 | 8.5 |
| 7 | Cross-Modality Ridges | 9 | 9 | 6 | 8.3 |
| 8 | Anti-Ridge (Smooth Diffusion) | 8 | 8 | 10 | 8.2 |
| 9 | Compositional Failure Prediction | 8 | 9 | 8 | 8.0 |
| 10 | Renormalization Group Flow | 9 | 8 | 6 | 7.8 |
| 11 | Koopman Eigenfunctions | 9 | 8 | 5 | 7.5 |
| 12 | LoRA Perturbation Theory | 8 | 7 | 9 | 7.3 |
| 13 | Human Categorical Perception | 7 | 9 | 6 | 7.2 |
| 14 | 4D Noise-Text Joint Space | 8 | 7 | 6 | 7.0 |
| 15 | Fractal Dimension Measurement | 7 | 7 | 9 | 6.8 |
| 16 | Phase-Boundary-Aware Training | 7 | 8 | 4 | 6.5 |
| 17 | Temporal Ridges (Spacetime) | 7 | 7 | 6 | 6.3 |
| 18 | Neuroscience Connection | 8 | 9 | 3 | 6.0 |

---

## Recommended Execution Order

**Phase A (Weeks 1-2): Low-Hanging Fruit with Existing Infrastructure**
- Direction 2 (Full Critical Exponents) — uses existing data, just new analysis
- Direction 8 (Anti-Ridge / Smooth Diffusion) — plug-and-play experiment
- Direction 15 (Fractal Dimension) — uses existing high-res maps
- Direction 5 (Mechanistic Circuits, partial) — early-exit already documented

**Phase B (Weeks 3-4): Novel Experiments with Moderate Compute**
- Direction 12 (LoRA Perturbation Theory) — uses existing pipeline
- Direction 4 (Voronoi Decomposition) — uses existing cluster data
- Direction 9 (Compositional Failure) — requires T2I-CompBench prompts

**Phase C (Weeks 5-8): Ambitious Experiments**
- Direction 1 (Ridge Ontogeny) — requires training checkpoints
- Direction 3 (Percolation) — requires 3D+ infrastructure
- Direction 6 (Safety Audit) — requires careful experimental design

**Phase D (Ongoing): High-Risk/High-Reward**
- Direction 10 (RG Flow)
- Direction 7 (Cross-Modality)
- Direction 11 (Koopman)
- Direction 13/18 (Human experiments)

---

## The Unifying Vision

These 18 directions converge on a single grand narrative: **Diffusion models spontaneously discover the same discrete concept structure that physics discovers in matter.** Phase boundaries in diffusion models obey the same scaling laws as domain walls in magnets (3D Ising). They form hierarchically, like symmetry breaking during cooling. They may percolate, like conductivity networks. Their geometry may be fractal, like critical clusters. And they may match the category boundaries that biological brains use to organize perception.

If even half of these experiments succeed, the resulting paper would not be about diffusion models — it would be about the UNIVERSALITY OF CATEGORICAL STRUCTURE across physical, biological, and artificial systems. That is the deep thesis: discrete categories emerging from continuous substrates is a universal phenomenon governed by the same mathematical laws regardless of the substrate.

---

## Key Literature References

- Sclocchi et al. "Phase Transition in Diffusion Models Reveals Hierarchical Data" (PNAS 2025)
- Lobashev et al. "Hessian Geometry of Latent Space in Generative Models" (ICML 2025)
- Karczewski et al. "Spacetime of Diffusion Models" (ICLR 2026 Oral)
- Caffarelli Regularity and Hierarchical Phase Boundaries (NeurIPS ML4PS 2025)
- Universal Scaling Laws of Absorbing Phase Transitions in DNNs (Phys. Rev. Research 2025)
- Grokking as Phase Transition (arXiv:2603.01192, March 2026)
- Tuning Universality in Deep Neural Networks (arXiv:2512.00168, 2025)
- CATS Net: Neural Model for Concept Formation (Nature Computational Science, 2026)
- Renormalization Group for DNNs: Universality and Scaling Laws (arXiv:2510.25553, 2025)
- SAeUron: SAEs for Diffusion Model Interpretability (ICML 2025)
- ConceptAttention: Interpretable Features in DiT (ICML 2025)
- TIDE: Temporal-Aware SAEs (AAAI 2025)
- Universal SAEs (ICML 2025)
- Smooth Diffusion (CVPR 2024)
- Koopman Eigenfunctions for Separatrices (NeurIPS 2025)
- Ising Model on 3D Fractal Lattices (arXiv:2506.17053, 2025)
- Categorical Perception in Biological Systems (PRX Life 2025)
- LENS 2026 Workshop: Learning and Exploitation of Latent Space Geometries
- Implicit Dynamical Regularization in Diffusion Training (NeurIPS 2025 Best Paper)
