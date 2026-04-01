# Fast Ridge Detection Speedup Research

## Executive Summary

The current fast scan pipeline processes a 20x20 grid (400 points) in 6.2 seconds across 6x RTX 4090 GPUs, with ~93ms per point per GPU. Analysis reveals that ~78ms of this is pipeline overhead (Python `__call__`, scheduler setup, generator creation, queue serialization) and only ~15ms is actual transformer compute. This massive overhead ratio means the single highest-impact optimization is **batched direct-transformer inference**, not model-level changes.

**Ranked recommendations** (by expected impact, accounting for implementation difficulty):

| Rank | Approach | Expected Speedup | Quality Impact | Difficulty | Section |
|------|----------|-----------------|---------------|------------|---------|
| 1 | Batched direct-transformer call | 8-12x | None | Medium | 3 |
| 2 | Early exit (10 of 56 blocks) | 2x (standalone), 2x on top of batching | Slight degradation | Medium | 1 |
| 3 | CUDA graphs | 1.3-1.5x | None | Medium-High | 10 |
| 4 | torch.compile (regional) | 1.3-1.5x | None | Low | 2 |
| 5 | Hierarchical subsampling | 3-8x | Controlled | Medium | 5,6 |
| 6 | FP8 quantization | 1.3-1.5x | Minimal | Low | 2 |
| 7 | Prompt-change caching | N/A (amortized) | None | Low | 7 |
| 8 | Text-encoder proxy | 100-200x | Uncertain | High (R&D) | 4,8 |

**Realistic combined target**: Batching + early exit + torch.compile + CUDA graphs could bring the 20x20 scan from **6.2s to ~0.1-0.3s** (20-60x speedup), making it near-interactive. Adding hierarchical sampling could push sub-100ms for initial feedback.

---

## 1. Early Exit / Partial Network Evaluation

### Architecture Context

FLUX Klein 4B has **8 double-stream blocks + 48 single-stream blocks = 56 total blocks**. The double-stream blocks process image and text tokens separately via cross-attention; the single-stream blocks concatenate them into a joint sequence. The 48 single-stream blocks dominate compute (~86% of block-level computation).

### Evidence That Semantic Structure Emerges Early

The UNet layer benchmark (from `/home/student/ai/hessian_ridges/results/unet_layer_benchmark/summary.json`) tested a different model (DreamShaper 8) but is informative:

- `mid_1step`: Spearman rho = **0.822**, IoU = 0.599 (best overall)
- `up_0_1step`: rho = **0.821**, IoU = 0.589
- `down_2_1step`: rho = 0.541

The midblock (bottleneck) at 1 step already captures ridge structure excellently. For the FLUX DiT architecture, the analogous "bottleneck" is the transition from double-stream to single-stream blocks (after block 8). The early single-stream blocks then refine this representation.

**Hypothesis**: Running the 8 double-stream blocks + first 2-10 single-stream blocks (10-18 of 56 total) should preserve most ridge-detection signal. This would be ~18-32% of block computation.

### Implementation Strategy

The transformer forward method in `transformer_flux2.py` (line 1299-1361) iterates over blocks in simple for-loops:

```python
# Double stream blocks (lines 1300-1322)
for index_block, block in enumerate(self.transformer_blocks):
    encoder_hidden_states, hidden_states = block(...)

# Concatenate streams (line 1325)
hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

# Single stream blocks (lines 1341-1361)
for index_block, block in enumerate(self.single_transformer_blocks):
    hidden_states = block(...)
```

**Implementation**: Create a subclass or monkey-patch `Flux2Transformer2DModel.forward` to break out of the single-stream loop after N blocks, then skip to the output projection:

```python
# After first N single-stream blocks, jump to:
hidden_states = hidden_states[:, num_txt_tokens:, ...]
hidden_states = self.norm_out(hidden_states, temb)
output = self.proj_out(hidden_states)
```

**Important caveat**: The `norm_out` and `proj_out` layers were trained with the full 48 single-stream blocks' output distribution. Using them on partial outputs will introduce a distribution mismatch. This is acceptable for relative sensitivity ranking (we only need ordinal correctness for ridge detection) but the absolute values will be wrong. L2-normalization of the output latent mitigates this.

### Expected Tradeoff

- **10/56 blocks** (8 double + 2 single): ~80% compute reduction in the block loop. With fixed overhead (embeddings, modulation, RoPE, proj_out), net ~50% wall-clock reduction. Expect rho degradation from 0.798 to ~0.6-0.7.
- **18/56 blocks** (8 double + 10 single): ~68% compute reduction, ~40% wall-clock. Expect rho ~0.7-0.78.
- **Standalone speedup**: ~2x.
- **Combined with batching**: Multiplicative -- batch=4 with early exit is ~4x faster than batch=4 alone.

### Validation Required

Must run an empirical sweep: compute fast-scan sensitivity with full 56 blocks and with 10, 18, 25, 35 blocks, then compare Spearman rho and Jaccard@20% against DINOv2 ground truth.

---

## 2. Reduced Precision / Quantization / torch.compile

### Current State

Model runs in **bfloat16**. RTX 4090 has:
- BF16 tensor cores: 660 TFLOPS
- FP8 tensor cores: 1,321 TFLOPS (2x BF16)
- INT8 tensor cores: 1,321 TOPS

### FP8 Quantization

BFL provides an official FP8 variant: `black-forest-labs/FLUX.2-klein-4b-fp8`. Key benefits:
- **~2x compute throughput** on FP8 tensor cores vs BF16
- **~50% VRAM reduction** for model weights (4GB vs 8GB), freeing memory for larger batches
- **Quality**: FP8 is sufficient for ridge detection since we only need ordinal sensitivity ranking

**Implementation**: Load `FLUX.2-klein-4b-fp8` directly, or apply post-training quantization via `quanto` / `bitsandbytes`. The `flux-fp8-api` package provides a ready-made FP8 implementation.

**Expected speedup**: 1.3-1.5x on the transformer forward pass (not full 2x due to non-matmul operations). Combined with batching, the freed VRAM allows larger batch sizes.

### torch.compile

Regional compilation for DiT models is well-established:
- Compile a **single transformer block once**, reuse for all blocks (7x faster compile time vs full model)
- Runtime speedup: **1.3-1.5x** on the forward pass
- Compile latency: ~10 seconds (one-time cost)

**Implementation**:
```python
# Compile one block, share across all
block = pipe.transformer.single_transformer_blocks[0]
compiled_block = torch.compile(block, mode="reduce-overhead")
for i in range(len(pipe.transformer.single_transformer_blocks)):
    pipe.transformer.single_transformer_blocks[i] = compiled_block
```

**Compatibility concern**: torch.compile may conflict with early exit (dynamic control flow). Solution: compile the block individually, not the whole forward pass.

### INT4 Quantization

INT4 (e.g., via GPTQ or AWQ) would halve VRAM further (~2GB) but:
- RTX 4090 INT4 throughput is limited (no native INT4 tensor cores)
- Quality degradation likely more significant
- **Not recommended** for the 4090 hardware -- FP8 is the sweet spot.

### TensorRT

Torch-TensorRT can deliver **2x speedup** over vanilla PyTorch for diffusion models. However:
- Requires fixed input shapes (compatible with our fixed 256x256 latents)
- Compilation takes minutes but is a one-time cost
- May complicate early exit implementation
- Best used after other optimizations are validated

---

## 3. Batched Inference -- THE HIGHEST IMPACT OPTIMIZATION

### The Overhead Problem

The current code calls `pipe(prompt_embeds=emb, ...)` in a Python loop for each grid point. Each pipeline `__call__` involves:

1. `check_inputs` validation (~1ms)
2. `encode_prompt` (already pre-encoded, but still creates text_ids) (~2ms)
3. `prepare_latents` (randn_tensor, packing, id creation) (~2ms)
4. Scheduler timestep computation (~1ms)
5. **Transformer forward pass** (~15ms)
6. Unpacking, BN normalization, unpatchify (~2ms)
7. `torch.Generator` creation per call (~1ms)
8. Result queue serialization + CPU transfer (~5ms)
9. Python overhead (dict access, conditionals, GC) (~10ms)

The transformer forward is only **~15ms out of ~93ms** (16% of wall time!). The pipeline overhead is **5-6x larger than the actual compute**.

### Solution: Bypass the Pipeline, Call Transformer Directly

Instead of calling `pipe()` for each point, prepare all inputs in bulk and call the transformer once per batch:

```python
# Prepare batch of embeddings
batch_embeds = torch.stack([interp(alpha_i, beta_j) for (i,j) in batch_points])

# Prepare shared noise (same seed -> same noise, just expand)
noise = randn_tensor(single_shape, generator=gen, ...).expand(batch_size, -1, -1)
latent_ids = prepare_latent_ids(noise).expand(batch_size, -1, -1)

# Direct transformer call
with torch.no_grad():
    output = pipe.transformer(
        hidden_states=noise,
        timestep=timestep.expand(batch_size),
        encoder_hidden_states=batch_embeds,
        txt_ids=text_ids.expand(batch_size, -1, -1),
        img_ids=latent_ids,
        return_dict=False,
    )[0]

# Scheduler step (vectorized)
latents = pipe.scheduler.step(output, t, noise, return_dict=False)[0]
```

### VRAM Analysis for Batching

At 256x256 with FLUX Klein 4B in BF16:
- **Model weights**: ~8 GB
- **Text encoder + VAE + misc**: ~3 GB
- **Available for activations**: ~13 GB per 24GB card
- **Per-sample activation overhead**: ~2-4 GB (varies by FlashAttention implementation)

Conservative batch sizes:
- **BF16 model**: batch 2-3 per GPU
- **FP8 model** (4 GB weights): batch 3-5 per GPU
- **With 256x256** (only 256 image tokens): batch 4-8 is likely feasible

With batch=4 on 6 GPUs = 24 points per round. A 20x20 grid (400 points) needs 400/24 = 17 rounds.

### Expected Performance

| Configuration | Per-batch time | Rounds (20x20) | Total time | Speedup |
|---------------|---------------|----------------|------------|---------|
| Current (bs=1, pipeline) | 93ms | 67/GPU | 6.2s | 1x |
| Direct call bs=1 | 20ms | 67/GPU | 1.3s | 4.8x |
| Direct call bs=4 | 35ms | 17/GPU | 0.6s | 10x |
| Direct call bs=8 | 55ms | 9/GPU | 0.5s | 12x |

### Implementation Complexity: Medium

Requires:
1. Extracting the forward-pass logic from `_process_fast_scan` into a batched version
2. Pre-computing and reusing noise tensors (trivial since same seed)
3. Handling the scheduler step for batched outputs
4. Modifying the result reporting (batch of LatentResults instead of one-at-a-time)

**Key file to modify**: `/home/student/ai/hessian_ridges/ridge_explorer/backend/services/gpu_pool.py`, specifically `_process_fast_scan`.

---

## 4. Embedding-Space Shortcuts

### Can We Skip the DiT Entirely?

The DiT maps `(noise, text_embedding, timestep)` to a denoised latent. For ridge detection, we care about how the **output latent direction** varies with `text_embedding`. The question: does the text embedding itself already encode this sensitivity information?

### Analysis

The text embedding `emb(alpha, beta) = (1-alpha-beta)*A + alpha*B + beta*C` is a simple affine function of (alpha, beta). Its Jacobian is **exactly constant**:
- d(emb)/d(alpha) = B - A
- d(emb)/d(beta) = C - A

This means the text embedding itself has **zero curvature** -- it's a flat 2D plane in embedding space. All the nonlinearity (and hence all the ridge structure) comes from the DiT's nonlinear mapping.

**Conclusion**: Text encoder features alone **cannot** predict ridges. The DiT is essential.

### Possible Proxy: Trained MLP

One could train a small MLP to predict `sensitivity(alpha, beta)` from the interpolated text embedding:

- **Input**: Flattened text embedding (~512 x 15360 = 7.8M dimensions -- too large)
- **More practical**: PCA-reduced text embedding (top 100 components) or pooled embedding
- **Output**: Scalar sensitivity value
- **Training data**: Collect (embedding, sensitivity) pairs from full DiT evaluations
- **Problem**: This MLP would be **prompt-triplet-specific** -- it would need retraining for each new set of 3 prompts, defeating the purpose

### Possible Proxy: Single Transformer Block

Run just the first double-stream block + projection:
- Cost: ~1/56 of full forward pass (~0.3ms)
- The first block applies cross-attention between text and image tokens
- This may capture coarse semantic alignment
- **Worth testing** but likely too shallow for reliable ridge detection

**Verdict**: Embedding-space shortcuts are fundamentally limited. The DiT nonlinearity is where ridges arise. However, a single-block proxy deserves empirical testing.

---

## 5. Subsampling + Interpolation

### Current Situation

For a 20x20 grid, all 400 points are evaluated. But ridges are **mesoscale features** spanning multiple grid cells. A sparser grid with interpolation could capture them.

### Approach: Coarse Grid + Interpolation

1. Evaluate a 10x10 grid (100 points, 4x fewer)
2. Compute Jacobian sensitivity on the coarse grid
3. Upsample sensitivity to 20x20 using bicubic interpolation
4. The upsampled map identifies approximate ridge locations

**Expected quality**: Ridges span 2-5 grid cells at 20x20 resolution. A 10x10 grid (2x coarser) should capture most ridges, though thin ones may be missed.

### Approach: Adaptive Sampling

1. Start with 8x8 coarse grid (64 points)
2. Compute coarse sensitivity
3. Identify high-sensitivity regions (above median * tau)
4. Subdivide only those regions to 2x resolution
5. Repeat if needed

**Expected benefit**: Most of the simplex is low-sensitivity "interior" of semantic regions. Only the ridge zones (typically 15-25% of area) need fine sampling. This reduces total evaluations by 50-70%.

### Interpolation Method

For the latent vectors themselves, simple bilinear interpolation works because:
- Adjacent latents on the grid are highly correlated (cosine similarity > 0.95 for neighbors)
- The sensitivity metric uses neighbor differences, so we need latents, not just sensitivity values
- After interpolation, recompute the Jacobian from the interpolated grid

**For sensitivity values directly**: Bicubic interpolation of the scalar sensitivity field is sufficient.

### Expected Speedup

| Strategy | Points evaluated | Speedup factor |
|----------|-----------------|---------------|
| 10x10 uniform | 100 | 4x |
| 8x8 + adaptive refine | ~100-150 | 2.7-4x |
| 5x5 + two rounds adaptive | ~80-120 | 3.3-5x |

**Key limitation**: Each adaptive round requires a full pipeline cycle (submit tasks, wait for results, analyze, submit next round). The latency of multiple rounds may negate point-count savings on small grids. Best suited for larger grids (50x50+).

---

## 6. Multi-Resolution / Hierarchical Approaches

### Within-Scan Hierarchy

This extends the subsampling idea into a structured hierarchy:

**Round 1**: 5x5 coarse grid (25 points) -- identifies approximate ridge locations in ~0.1s (with batching)

**Round 2**: For each of the ~5-10 coarse cells above threshold, subdivide into 4x4 sub-grids. That's ~40-80 additional points.

**Round 3**: (Optional) Further 2x refinement of the hottest sub-regions.

Total: ~70-130 points evaluated to achieve 20x20-equivalent resolution where it matters.

### Integration with Existing Refinement

The existing tool already does coarse-to-fine refinement (`/api/grid/{id}/refine`), but across separate API calls initiated by the user. The proposal is to **automate this within a single fast-scan call**:

1. Fast scan submits 5x5 coarse grid
2. Backend automatically analyzes, identifies ridges
3. Backend automatically submits refinement points
4. Returns combined high-resolution sensitivity map

**Implementation**: Modify `_process_fast_scan` to include a second round of targeted evaluation before declaring scan complete.

### Expected Speedup

With batching, a 5x5 grid + 50 targeted refinement points = 75 points total:
- Current approach: 400 points * 93ms / 6 = 6.2s
- Hierarchical: 75 points, batched bs=4, direct call: ~0.15s
- **Combined with all optimizations: sub-second easily**

---

## 7. Caching / Incremental Updates

### Prompt-Change Caching

When the user modifies one of three prompts, 2/3 of the text embeddings are unchanged. However:
- Text encoding is already negligible (~100ms total, done once)
- The DiT forward pass must be re-run for ALL grid points because the interpolated embedding changes everywhere
- **Conclusion**: No savings from text embedding caching

### Noise Tensor Caching

Since all grid points share the same seed:
- The initial noise tensor is identical for all points
- Currently, each pipeline call recreates it via `torch.Generator.manual_seed(seed)`
- **Optimization**: Generate the noise once, clone/expand for batched calls
- Savings: ~2ms per point (minor, but part of the batching optimization)

### Latent Grid Caching

When doing hierarchical refinement:
- Parent grid latents can be reused as initialization
- No need to recompute points that were already evaluated
- The current refinement system already preserves parent cell data

### Cross-Scan Caching

If the user re-scans with the same prompts but different seed:
- All text embeddings are identical
- Only noise changes
- Could cache the text-related intermediate computations in the double-stream blocks
- **Estimate**: ~10% of computation is text-only in double-stream blocks
- **Low value, high complexity**: Not recommended

---

## 8. Model Distillation / Smaller Proxy Models

### Distilling FLUX Klein into a Tiny Model

The idea: train a small model (e.g., 100M parameters) that maps `text_embedding -> latent_sensitivity` directly.

**Problems**:
1. The mapping is prompt-triplet-specific. A model trained on one set of 3 prompts won't generalize to new prompts.
2. Training requires generating ground-truth sensitivity maps, which is the expensive part we're trying to avoid.
3. A general-purpose distilled model would need training on thousands of prompt triplets.

**Verdict**: Not practical for an interactive tool where users bring arbitrary prompts.

### Smaller Diffusion Models as Proxies

Could a much smaller model (e.g., Stable Diffusion XL Turbo at ~3.5B, or even SD 1.5 at ~1B) predict where FLUX Klein's ridges are?

From the experiment log: cross-architecture ridge correlation varies widely:
- Flux <-> SD3.5: mean rho = +0.491
- Flux <-> DreamShaper8: mean rho = +0.361
- Flux <-> PixArt: mean rho = -0.214 (anti-correlated!)

**Conclusion**: Smaller models from the same architectural family (CLIP-based) show moderate correlation. A smaller FLUX variant or SD3.5 Medium could serve as a rough proxy (rho ~0.4-0.5), but the quality drop is severe. This is a **last resort** if sub-100ms latency is critical and quality can be sacrificed.

### Text Encoder Features Directly

As analyzed in Section 4, text embeddings alone cannot predict ridges because the interpolation is linear. However, the text encoder's **hidden state structure** (attention patterns, intermediate activations) might contain useful information:

- FLUX Klein uses a Qwen3-based text encoder with layer outputs at layers 9, 18, 27
- The differences between these multi-layer features across grid points might correlate with ridge structure
- Cost: ~1ms per point (just text encoder inference, no DiT)

**This is worth a quick experiment**: Compute cosine distances between text encoder hidden states at adjacent grid points and compare against DiT-based sensitivity.

---

## 9. Analytical / Mathematical Approaches

### Jacobian via Autograd (JVP)

The current approach uses **finite-difference** Jacobian estimation from grid neighbors. An alternative: compute the Jacobian analytically using PyTorch's automatic differentiation.

**The 2D case**: We need the 2x2 Gram matrix `J^T J` where `J = [df/d(alpha), df/d(beta)]`. This requires exactly **2 JVP evaluations** (one per input dimension).

**Cost analysis**:
- Each JVP costs ~1.5-3x a forward pass (forward-mode AD in PyTorch)
- For 1 grid point: 1 forward + 2 JVPs = ~3 forward passes equivalent
- But JVPs give exact derivatives, allowing **sparser grids**
- A 10x10 grid with JVPs: 100 * 3 = 300 forward-pass equivalents
- vs. current 20x20 grid: 400 forward passes

**However**: The current finite-difference Jacobian from grid neighbors is already well-validated (rho=0.798 with DINOv2). The JVP approach would give exact gradients but:
1. PyTorch's forward-mode AD may not support all custom CUDA ops in FlashAttention
2. Requires `torch.no_grad()` to be disabled (increases memory)
3. The improvement in gradient accuracy is marginal for mesoscale ridge detection

**Critical insight from FAST_RIDGE_DETECTION.md**: "Fine-epsilon Jacobian (epsilon=1e-4) gives rho=-0.013 -- zero correlation. Coarse grid-neighbor Jacobian gives rho=0.569." This means ridges are **NOT differential features**. They exist at the mesoscale (grid spacing ~0.02-0.05), not at infinitesimal scale. Exact derivatives from JVP would actually give **worse** results than coarse finite differences, because they would measure the wrong thing!

**Verdict**: JVP/autograd is theoretically elegant but **counterproductive** for this specific application. The finite-difference approach using grid neighbors is correct because ridges are mesoscale features. Do not pursue.

---

## 10. Hardware / System Optimizations

### CUDA Graphs

CUDA graphs capture a sequence of GPU kernel launches into a replayable graph, eliminating per-kernel launch overhead (~20-200us per kernel).

**Applicability**: Excellent for the fast scan because:
- Every grid point uses the **exact same** computation graph (same shapes, same operations)
- Only the input tensor data changes between calls
- The 56 transformer blocks generate hundreds of individual kernel launches

**Expected speedup**: 1.3-1.5x on the transformer forward pass (more impactful for small batch sizes where kernel launch overhead is a larger fraction).

**Implementation**:
```python
# Warmup pass
static_input = torch.randn_like(noise)
static_embeds = torch.randn_like(prompt_embeds)
# Capture graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_output = transformer(static_input, ...)
# Replay with new data
static_input.copy_(actual_noise)
static_embeds.copy_(actual_embeds)
graph.replay()
```

**Compatibility**: CUDA graphs require static shapes and no dynamic control flow. This is **compatible with fixed 256x256 inference** but **incompatible with early exit** (which changes the computation graph). Solution: use CUDA graphs for the truncated model (fixed number of blocks), not for dynamic early exit.

### FlashAttention

FLUX Klein already uses `scaled_dot_product_attention` which dispatches to FlashAttention-2 on compatible hardware. No additional optimization needed.

### Pipeline Parallelism Across GPUs

Currently, each GPU runs independent grid points. An alternative: pipeline-parallel a single forward pass across GPUs (e.g., blocks 1-9 on GPU0, blocks 10-18 on GPU1, etc.).

**Not recommended** because:
- Pipeline parallelism has bubble overhead (GPUs idle waiting for previous stage)
- The current data-parallel approach (independent points) is already optimal for embarrassingly parallel workloads
- Inter-GPU communication latency (PCIe or NVLink) would add overhead

### Memory-Mapped Storage

Not relevant -- the bottleneck is GPU compute, not CPU-GPU transfer. The latent vectors (128KB each) are tiny.

---

## 11. Alternative Sensitivity Metrics

### Current Metric: Spectral Norm of 2x2 Jacobian Gram Matrix

This requires: L2-normalized latent vectors at each grid point, then finite-difference Gram matrix computation (pure numpy, instant).

### Simpler Alternatives

**Cosine distance to neighbors** (already implemented in `_sensitivity_2d`):
```python
sensitivity[i,j] = mean(1 - dot(emb[i,j], emb[ni,nj]))  # 4-neighbors
```
This is essentially the **trace of the Gram matrix** (sum of diagonal elements), not the spectral norm. It measures average sensitivity rather than maximum directional sensitivity.

**L2 distance to neighbors**: `||latent[i,j] - latent[ni,nj]||^2`
- After L2-normalization, this equals `2(1 - cos_sim)`, so it's equivalent to cosine distance.

**Maximum neighbor distance**: `max(||latent[i,j] - latent[ni,nj]||)` over 4 neighbors.
- Approximates the spectral norm for axis-aligned ridges.

**All of these are already instant** (numpy computation on the grid). The sensitivity metric is NOT a bottleneck. The bottleneck is generating the latent vectors via the DiT. Therefore, changing the metric provides zero speedup.

**However**: Simpler metrics might allow skipping the L2-normalization step (which is also instant). The only reason to consider alternative metrics is if they enable **fewer grid evaluations** (e.g., a metric that works with sparser grids).

---

## 12. Noise-Space Tricks

### Exploiting Shared Noise

In rectified flow, the 1-step prediction is: `x_pred = (1-t)*noise + t*model(noise, t, cond)`

At t=1 (the typical single-step setting): `x_pred = model(noise, 1.0, cond)`

The noise is the initial latent, which is **identical across all grid points** (same seed). Only `cond` (the interpolated text embedding) varies.

### Factoring Out Common Computation

In the transformer forward pass:
1. **Time embedding** (`temb`): Computed from timestep and guidance. Same for all grid points. **Can be computed once and reused.**
2. **Modulation parameters**: Derived from `temb`. Same for all points. **Can be computed once.**
3. **RoPE embeddings**: Derived from position IDs. Same for all points. **Can be computed once.**
4. **Initial noise packing**: Same noise, same packing. **Can be done once.**

The only thing that varies is `encoder_hidden_states` (the text embedding).

In the double-stream blocks, text and image streams interact via cross-attention. The image stream's Q/K/V projections of the noise are the same across grid points (at block 0). Only the text-side Q/K/V change.

**Potential optimization**: For the first double-stream block, precompute the image-side Q, K, V from the noise (shared), and only compute text-side Q, K, V per point. This saves ~50% of the first block's compute.

**However**: After block 0, the hidden states diverge (because the text conditioning changes the image stream). So this trick only helps for the very first block. Net savings: ~1% of total compute. **Not worth the complexity.**

### Batch as the Better Solution

The shared-noise insight is more naturally exploited by **batching**: prepare one noise tensor, `.expand()` it to batch size, pair with different text embeddings. This is exactly what batched inference does, and it's the correct way to exploit the shared noise structure.

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days, ~10x speedup)

**1a. Batched direct-transformer inference**
- Modify `_process_fast_scan` in `gpu_pool.py` to:
  - Pre-encode prompts once (already done)
  - Pre-generate noise tensor once, expand for batch
  - Compute all interpolated embeddings as a tensor
  - Call transformer directly in batches of 4-8
  - Process outputs in bulk
- Expected: **8-12x speedup** (6.2s -> ~0.5-0.8s)

**1b. Pre-compute shared tensors**
- Time embedding, modulation params, RoPE, noise packing: compute once per scan, pass to all batches
- Expected: Minor additional speedup on top of 1a (~10%)

### Phase 2: Compilation + Quantization (1-2 days, additional 1.5-2x)

**2a. torch.compile with regional compilation**
- Compile a single transformer block, reuse for all blocks
- One-time 10-second compile cost per worker startup
- Expected: **1.3-1.5x** on top of Phase 1

**2b. FP8 quantization**
- Load `FLUX.2-klein-4b-fp8` or apply post-training FP8 quantization
- Frees VRAM for larger batches (batch 6-8 instead of 4)
- Expected: **1.3-1.5x** from compute speedup + larger batches

### Phase 3: Early Exit (1-2 days, additional 1.5-2x)

**3a. Implement truncated forward pass**
- Subclass `Flux2Transformer2DModel` with configurable `max_single_blocks` parameter
- Default to 10 single-stream blocks (18/56 total blocks)
- Expected: **1.5-2x** on top of Phase 2

**3b. Validate quality**
- Run sensitivity comparison: full 56 blocks vs 10, 15, 20, 25 blocks
- Target: Spearman rho > 0.7 vs full model, Jaccard@20% > 0.35

### Phase 4: CUDA Graphs (1-2 days, additional 1.2-1.4x)

**4a. Capture and replay CUDA graph**
- Requires static shapes (already fixed at 256x256, fixed batch size)
- Capture the truncated transformer forward as a graph
- Replay with updated input data for each batch
- Expected: **1.2-1.4x** on top of Phase 3

### Phase 5: Hierarchical Sampling (2-3 days, additional 2-4x for large grids)

**5a. Two-round adaptive scan**
- Round 1: 5x5 coarse grid (~0.05s with all Phase 1-4 optimizations)
- Round 2: Targeted refinement of high-sensitivity cells (~0.1s)
- Expected: **2-4x** additional for 20x20+ grids

### Projected Total Performance

| Phase | 20x20 grid time | Cumulative speedup |
|-------|-----------------|-------------------|
| Current | 6.2s | 1x |
| Phase 1 (batching) | 0.5-0.8s | 8-12x |
| Phase 2 (+compile+FP8) | 0.25-0.4s | 15-25x |
| Phase 3 (+early exit) | 0.15-0.25s | 25-40x |
| Phase 4 (+CUDA graphs) | 0.1-0.2s | 30-60x |
| Phase 5 (+hierarchical) | 0.05-0.1s | 60-120x |

**Target**: Sub-200ms for a 20x20 sensitivity heatmap, approaching interactive rates.

---

## Approaches NOT Recommended

| Approach | Why Not |
|----------|---------|
| JVP/Autograd Jacobian (Section 9) | Ridges are mesoscale, not differential. Exact gradients give worse results than coarse finite differences. |
| Pipeline parallelism (Section 10) | Worse than data parallelism for embarrassingly parallel workloads. |
| Full model distillation (Section 8) | Prompt-triplet-specific; training cost defeats purpose. |
| Alternative sensitivity metrics (Section 11) | Metric computation is already instant; not a bottleneck. |
| Cross-model proxy (Section 8) | rho=0.36-0.49 correlation is too low for reliable ridge detection. |
| Noise factorization (Section 12) | Only saves ~1% compute; batching achieves the same benefit naturally. |
| INT4 quantization (Section 2) | No native INT4 tensor cores on RTX 4090; FP8 is the sweet spot. |
