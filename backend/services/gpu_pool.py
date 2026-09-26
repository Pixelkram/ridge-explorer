import time
"""
Multi-GPU worker pool for FLUX.2 Klein 4B.

Single-phase workflow: generate image + DINOv2 embed in one pass.
Each worker loads FLUX Klein + DINOv2 on its GPU.
"""
import torch
import torch.multiprocessing as mp
import numpy as np
import hashlib
import io
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms

from backend import config


@dataclass
class GenerateTask:
    """Generate images for grid rows and compute DINOv2 embeddings."""
    job_id: str
    row_indices: list[int]
    alphas: np.ndarray
    betas: np.ndarray
    prompt_a: str
    prompt_b: str
    prompt_c: str  # empty string = 2-prompt mode
    grid_size: int
    seed: int
    height: int
    width: int
    steps: int
    guidance_scale: float
    active_cells: set | None = None
    # 3D mode
    prompt_d: str = ""
    gammas: np.ndarray | None = None
    grid_size_z: int = 0  # 0 = 2D mode
    use_slerp: bool = False
    epoch: int = 0  # cancel epoch stamped by GPUPool.submit; see _cancelled()


@dataclass
class FastScanTask:
    """Generate 1-step latents for fast ridge detection (no images, no DINOv2)."""
    job_id: str
    row_indices: list[int]
    alphas: np.ndarray
    betas: np.ndarray
    prompt_a: str
    prompt_b: str
    prompt_c: str
    grid_size: int
    seed: int
    height: int
    width: int
    guidance_scale: float = 1.0  # 1.0 = no CFG, single forward pass
    # 3D mode
    prompt_d: str = ""
    gammas: np.ndarray | None = None
    grid_size_z: int = 0  # 0 = 2D mode
    # None = follow the 2D/3D default the generation pass uses (see _ScanContext).
    # The scan and the generation MUST mix embeddings the same way, or the Jacobian
    # proxy selects cells on a different field than the images then sample.
    use_slerp: bool | None = None
    epoch: int = 0


@dataclass
class HikeTask:
    """Generate an arbitrary set of simplex points whose VERTICES are coefficient
    vectors over a basis of prompt strings.

    Edge-hopping needs this: after the first hop a vertex is no longer a prompt but an
    interpolated coordinate that no text names. Because every mixture here is affine, a
    blend-of-blends flattens to one coefficient vector over the base prompts, so the
    representation stays flat however far the chain travels.
    """
    job_id: str
    basis: list            # prompt strings
    vertex_coefs: list     # 3 x len(basis)
    points: list           # list of (i, j, w0, w1, w2)
    seed: int
    height: int
    width: int
    steps: int
    guidance_scale: float
    epoch: int = 0


@dataclass
class DiscoverTask:
    """Generate a batch of conditioning points given as ARBITRARY weight vectors over a
    prompt basis of any size.

    HikeTask cannot express this: it carries three vertices and points of the form
    (i, j, w0, w1, w2), so it is limited to triangles. The discovery loop draws its points
    from Dirichlet(alpha) over k prompts, where k is 3-8 and every point has its own
    k-vector of weights, so the weights travel with the point rather than with a fixed set
    of vertices.

    `points` is a list of (index, weights) with len(weights) == len(basis). Mixing stays
    affine, so this is the same arithmetic the rest of the system uses -- only the arity
    changes.
    """
    job_id: str
    basis: list            # k prompt strings
    points: list           # list of (int index, sequence of k floats)
    seed: int
    height: int
    width: int
    steps: int
    guidance_scale: float
    epoch: int = 0


@dataclass
class ProbeTask:
    """Exact-JVP ridge probe at one barycentric point of a job (services/jvp_probe.ridge_normal)."""
    probe_id: str
    job_id: str
    prompts: list            # k prompts, prompts[0] = remainder prompt (prompt_a)
    weights: list            # k barycentric weights, sum 1
    seed: int
    height: int
    width: int
    steps: int
    guidance_scale: float
    use_slerp: bool = False
    epoch: int = 0


@dataclass
class TokenProbeTask:
    """Per-token exact-JVP sensitivity of one prompt (services/jvp_probe.token_sensitivity)."""
    probe_id: str
    job_id: str
    prompt: str
    seed: int
    height: int
    width: int
    steps: int
    guidance_scale: float
    epoch: int = 0


@dataclass
class ProbeResult:
    probe_id: str
    kind: str                # "jvp" | "token"
    result: dict
    error: str = ""


@dataclass
class HQTask:
    """Re-render specific cells at high quality."""
    job_id: str
    cells: list[tuple[int, int]]
    alphas: np.ndarray
    betas: np.ndarray
    prompt_a: str
    prompt_b: str
    prompt_c: str
    seed: int
    epoch: int = 0


@dataclass
class LatentResult:
    """Result for one grid cell from fast scan — normalized latent vector."""
    job_id: str
    gpu_id: int
    row: int
    col: int
    latent_vector: np.ndarray  # normalized flattened latent


@dataclass
class LatentBatchResult:
    """Batch result for an entire row from fast scan — reduces queue overhead."""
    job_id: str
    gpu_id: int
    row: int
    cols: list[int]
    latent_vectors: np.ndarray  # (n_cols, latent_dim) normalized
    depths: list[int] | None = None  # z-indices for 3D mode


@dataclass
class CellResult:
    """Result for one grid cell."""
    job_id: str
    gpu_id: int
    row: int
    col: int
    thumbnail_bytes: bytes
    thumbnail_hash: str
    dino_embedding: np.ndarray  # (768,)
    is_hq: bool = False
    depth: int = 0  # z-index for 3D grids


# Exceptions whose text means this process's CUDA context is unusable: every later task
# on this worker fails in microseconds, and because the task queue is shared that turns
# the worker into a blackhole that out-races the healthy ones for work.
_CUDA_FATAL = ("CUDA error", "device-side assert", "illegal memory access",
               "CUDA_ERROR", "cuDNN error")


def _cancelled(task, cancel_epoch) -> bool:
    """True when a cancel was issued after `task` was submitted.

    A shared boolean could not express this: every start endpoint clears the flag a
    millisecond after /cancel sets it, so work already dispatched never observed it.
    The epoch is stamped on the task at submit time, so a later start cannot un-cancel
    an older task, and reading it is a plain shared-memory load — cheap enough to poll
    per generated image rather than per row.
    """
    return cancel_epoch is not None and task.epoch < cancel_epoch.value


def worker_main(gpu_id: int, task_queue, result_queue, cancel_epoch=None):
    """Main loop for a GPU worker process."""
    device = torch.device(f"cuda:{gpu_id}")

    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(
        config.MODEL_ID, torch_dtype=torch.bfloat16,
    ).to(device)

    # DINOv2 for embedding
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(device)
    dino.eval()
    dino_transform = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # The pipeline encodes the empty NEGATIVE prompt on every __call__: a full 36-layer
    # Qwen3-4B forward over 512 padded tokens, producing the same tensor every time. At
    # one pipe() call per simplex point that is 38.5 ms/image, ~8.9% of a hop. Precompute
    # it once; passing negative_prompt_embeds makes __call__ skip encode_prompt entirely,
    # so output is bit-identical (verified: max abs channel diff 0 over 24 image pairs).
    try:
        with torch.no_grad():
            _neg, _ = pipe.encode_prompt(prompt="")
        pipe._cached_neg_embeds = _neg
        print(f"[GPU {gpu_id}] cached empty negative-prompt embedding "
              f"{tuple(_neg.shape)}", flush=True)
    except Exception as exc:                 # never let an optimisation break startup
        pipe._cached_neg_embeds = None
        print(f"[GPU {gpu_id}] negative-embed cache unavailable ({exc}); "
              f"falling back to per-call encode", flush=True)

    print(f"[GPU {gpu_id}] Worker ready (FLUX Klein + DINOv2)", flush=True)
    result_queue.put({"type": "ready", "gpu_id": gpu_id})

    while True:
        task = task_queue.get()
        if task is None:
            break

        # one bad task must not kill the worker: the process would exit while the
        # parent kept dispatching to a queue nobody reads, and every later job on that
        # GPU would hang until its deadline with no error anywhere
        try:
            if isinstance(task, GenerateTask):
                _process_generate(gpu_id, device, pipe, dino, dino_transform, task, result_queue, cancel_epoch)
            elif isinstance(task, FastScanTask):
                _process_fast_scan(gpu_id, device, pipe, task, result_queue, cancel_epoch)
            elif isinstance(task, HikeTask):
                _process_hike(gpu_id, device, pipe, dino, dino_transform, task, result_queue, cancel_epoch)
            elif isinstance(task, DiscoverTask):
                _process_discover(gpu_id, device, pipe, dino, dino_transform, task, result_queue, cancel_epoch)
            elif isinstance(task, HQTask):
                _process_hq(gpu_id, device, pipe, dino, dino_transform, task, result_queue, cancel_epoch)
            elif isinstance(task, (ProbeTask, TokenProbeTask)):
                _process_probe(gpu_id, device, pipe, dino, task, result_queue)
        except Exception as exc:
            import traceback
            print(f"[GPU {gpu_id}] task {getattr(task, 'job_id', '?')} failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            result_queue.put({"type": "task_error", "gpu_id": gpu_id,
                              "job_id": getattr(task, "job_id", None),
                              "error": f"{type(exc).__name__}: {exc}"})
            msg = str(exc)
            if any(m in msg for m in _CUDA_FATAL):
                # Nothing on this device will succeed again, so stop consuming the
                # shared queue; leaving the process up starves the healthy workers.
                print(f"[GPU {gpu_id}] CUDA context lost, exiting worker", flush=True)
                result_queue.put({"type": "worker_died", "gpu_id": gpu_id, "error": msg})
                import os
                os._exit(1)


def _encode_prompts(pipe, prompt_a, prompt_b, prompt_c, prompt_d=""):
    """Encode 2, 3, or 4 prompts and return embeddings."""
    with torch.no_grad():
        emb_a, _ = pipe.encode_prompt(prompt=prompt_a)
        emb_b, _ = pipe.encode_prompt(prompt=prompt_b)
        emb_c = None
        emb_d = None
        if prompt_c:
            emb_c, _ = pipe.encode_prompt(prompt=prompt_c)
        if prompt_d:
            emb_d, _ = pipe.encode_prompt(prompt=prompt_d)
    return emb_a, emb_b, emb_c, emb_d


def _nlerp(embs, weights):
    """Normalized linear interpolation (NLERP) — LERP then normalize to preserve norm.
    Approximates SLERP for embeddings on a hypersphere."""
    result = sum(w * e for w, e in zip(weights, embs))
    # The target norm has to be the barycentric mixture of the input norms, not their
    # plain mean: with the mean, a simplex vertex (weights [1,0,0]) came back as emb_a
    # rescaled by the *other* prompts' norms, so the pure-prompt corner of the grid did
    # not reproduce the pure prompt and editing prompt C moved the image at corner A.
    target_norm = sum(float(w) * e.norm() for w, e in zip(weights, embs))
    result_norm = result.norm()
    # On the 3D cube the weights are affine but not convex (the first one goes negative
    # past the simplex), so the mixed norm can reach 0 or flip sign; leave the plain
    # LERP alone there rather than reflecting the embedding through the origin.
    if result_norm > 1e-8 and target_norm > 1e-8:
        result = result * (target_norm / result_norm)
    return result


def _mix_embeddings(embs, weights, use_slerp):
    """The single place prompt embeddings are mixed.

    The fast scan's Jacobian proxy only predicts the field the images sample if both
    phases interpolate identically. They used to diverge (the scan was always LERP, 3D
    generation NLERP), so the ridge map selected cells on one field and the thumbnails
    it labelled came from another."""
    if use_slerp:
        return _nlerp(embs, weights)
    return sum(w * e for w, e in zip(weights, embs))


def _image_to_thumbnail(image):
    """Convert PIL image to JPEG bytes at full resolution.

    Despite the name these are the full render, not config.THUMBNAIL_SIZE tiles: the UI
    serves them as the per-cell image and only the montage downsamples. The `size`
    argument this used to take was never applied, which read as if it did resize.
    """
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=90)
    return buf.getvalue()


class _ScanContext:
    """Shared state for batched 1-step latent evaluation. Supports 2D and 3D."""
    __slots__ = ('emb_a', 'emb_b', 'emb_c', 'emb_d', 'noise_t', 'text_ids_t',
                 'latent_ids_t', 'batch_t', 'transformer', 'img_tokens', 'use_slerp')

    def __init__(self, pipe, device, task):
        from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu

        with torch.no_grad():
            self.emb_a, text_ids = pipe.encode_prompt(prompt=task.prompt_a)
            self.emb_b, _ = pipe.encode_prompt(prompt=task.prompt_b)
            self.emb_c = None
            self.emb_d = None
            if task.prompt_c:
                self.emb_c, _ = pipe.encode_prompt(prompt=task.prompt_c)
            if getattr(task, 'prompt_d', '') and task.prompt_d:
                self.emb_d, _ = pipe.encode_prompt(prompt=task.prompt_d)

        # Every submitter now states this (grid.py records use_slerp on the job and
        # passes the same value to the generation that follows). The fallback mirrors
        # grid.py's use_slerp=is_3d rule for a caller that does not, because a scan run
        # on a different interpolation measures a field the generation never samples.
        want_slerp = getattr(task, 'use_slerp', None)
        self.use_slerp = (self.emb_d is not None) if want_slerp is None else want_slerp

        gen = torch.Generator(device=device).manual_seed(task.seed)
        in_ch = pipe.transformer.config.in_channels
        noise, latent_ids = pipe.prepare_latents(
            batch_size=1, num_latents_channels=in_ch // 4,
            height=task.height, width=task.width,
            dtype=self.emb_a.dtype, device=device, generator=gen,
        )

        self.img_tokens = noise.shape[1]
        mu = compute_empirical_mu(image_seq_len=self.img_tokens, num_steps=1)
        pipe.scheduler.set_timesteps(num_inference_steps=1, device=device, mu=mu)
        t_val = pipe.scheduler.timesteps[0]

        self.transformer = pipe.transformer
        t_dtype = self.transformer.dtype
        self.noise_t = noise.to(t_dtype)
        self.text_ids_t = text_ids
        self.latent_ids_t = latent_ids
        self.batch_t = (t_val / 1000).to(t_dtype)

    def evaluate_points(self, points, batch_size=8, should_abort=None):
        """Evaluate a list of (alpha, beta[, gamma]) points. Returns (N, D) normalized
        latents, or None if `should_abort` asked to stop part-way."""
        all_latents = []
        for b_start in range(0, len(points), batch_size):
            # one poll per transformer forward: fine-grained enough that a cancel is
            # acted on within a batch, cheap enough not to show up next to the forward
            if should_abort is not None and should_abort():
                return None
            batch = points[b_start:b_start + batch_size]
            bs = len(batch)

            embeds = []
            for pt in batch:
                if len(pt) == 3 and self.emb_d is not None:
                    alpha, beta, gamma = pt
                    e = _mix_embeddings(
                        [self.emb_a, self.emb_b, self.emb_c, self.emb_d],
                        [1 - alpha - beta - gamma, alpha, beta, gamma], self.use_slerp)
                elif self.emb_c is not None:
                    alpha, beta = pt[0], pt[1]
                    e = _mix_embeddings([self.emb_a, self.emb_b, self.emb_c],
                                        [1 - alpha - beta, alpha, beta], self.use_slerp)
                else:
                    alpha, beta = pt[0], pt[1]
                    e = _mix_embeddings([self.emb_a, self.emb_b],
                                        [1 - alpha, alpha], self.use_slerp)
                embeds.append(e)
            batch_embeds = torch.cat(embeds, dim=0).to(self.noise_t.dtype)

            batch_noise = self.noise_t.expand(bs, -1, -1)
            with torch.no_grad():
                velocity = self.transformer(
                    hidden_states=batch_noise,
                    timestep=self.batch_t.expand(bs),
                    guidance=None,
                    encoder_hidden_states=batch_embeds,
                    txt_ids=self.text_ids_t.expand(bs, -1, -1),
                    img_ids=self.latent_ids_t.expand(bs, -1, -1),
                    return_dict=False,
                )[0]

            velocity = velocity[:, :self.img_tokens, :]
            denoised = batch_noise - velocity

            for idx in range(bs):
                flat = denoised[idx].flatten().float()
                norm = flat.norm()
                if norm > 1e-8:
                    flat = flat / norm
                all_latents.append(flat.cpu().numpy())

        return np.stack(all_latents)


def _process_fast_scan(gpu_id, device, pipe, task, result_queue, cancel_epoch=None):
    """Batched fast scan: bypass pipeline overhead, call transformer directly.

    Supports both 2D (3 prompts) and 3D (4 prompts) grids.
    """
    BATCH_SIZE = 8

    abort = lambda: _cancelled(task, cancel_epoch)
    ctx = _ScanContext(pipe, device, task)
    gs = task.grid_size
    alphas = task.alphas
    betas = task.betas
    is_3d = task.grid_size_z > 0 and task.gammas is not None

    mode = "3D" if is_3d else "2D"
    print(f"[GPU {gpu_id}] Fast scan setup done: {mode}, {ctx.img_tokens} img tokens, "
          f"batch={BATCH_SIZE}", flush=True)

    if is_3d:
        gammas = task.gammas
        gs_z = task.grid_size_z
        for i in task.row_indices:
            alpha_i = float(alphas[i])
            for j in range(gs):
                beta_j = float(betas[j])
                # Evaluate all z-slices for this (i, j) column
                points = [(alpha_i, beta_j, float(gammas[k])) for k in range(gs_z)]
                latents = ctx.evaluate_points(points, BATCH_SIZE, abort)
                if latents is None:
                    print(f"[GPU {gpu_id}] Cancelled mid-fastscan (row {i})", flush=True)
                    return
                result_queue.put(LatentBatchResult(
                    job_id=task.job_id, gpu_id=gpu_id,
                    row=i, cols=[j] * gs_z,
                    latent_vectors=latents,
                    depths=list(range(gs_z)),
                ))
            print(f"[GPU {gpu_id}] Fast scan {task.job_id} row {i+1}/{gs}", flush=True)
    else:
        for i in task.row_indices:
            points = [(float(alphas[i]), float(betas[j])) for j in range(gs)]
            latents = ctx.evaluate_points(points, BATCH_SIZE, abort)
            if latents is None:
                print(f"[GPU {gpu_id}] Cancelled mid-fastscan (row {i})", flush=True)
                return
            result_queue.put(LatentBatchResult(
                job_id=task.job_id, gpu_id=gpu_id,
                row=i, cols=list(range(gs)),
                latent_vectors=latents,
            ))
            print(f"[GPU {gpu_id}] Fast scan {task.job_id} row {i+1}/{gs}", flush=True)



def _process_generate(gpu_id, device, pipe, dino, dino_transform, task, result_queue, cancel_epoch=None):
    """Generate images for assigned rows and compute DINOv2 embeddings."""
    emb_a, emb_b, emb_c, emb_d = _encode_prompts(
        pipe, task.prompt_a, task.prompt_b, task.prompt_c, task.prompt_d)

    is_3d = task.grid_size_z > 0 and task.gammas is not None

    for i in task.row_indices:
        alpha = task.alphas[i]
        z_range = range(task.grid_size_z) if is_3d else [0]

        for j in range(task.grid_size):
            for k in z_range:
                # Skip cells not in active set
                if task.active_cells is not None:
                    key = (i, j, k) if is_3d else (i, j)
                    if key not in task.active_cells:
                        continue

                # Poll per cell, not per row: a row of a large grid is minutes of
                # uninterruptible work, so a cancel issued mid-row used to be a no-op
                # for every task already dispatched.
                if _cancelled(task, cancel_epoch):
                    print(f"[GPU {gpu_id}] Cancelled mid-generate (row {i}, col {j})",
                          flush=True)
                    return

                beta = task.betas[j]
                gamma = task.gammas[k] if is_3d else 0.0

                # Interpolated embedding
                if is_3d and emb_d is not None:
                    # 4-prompt 3D: (1-α-β-γ)*A + α*B + β*C + γ*D
                    emb = _mix_embeddings(
                        [emb_a, emb_b, emb_c, emb_d],
                        [1 - alpha - beta - gamma, alpha, beta, gamma], task.use_slerp)
                elif emb_c is not None:
                    emb = _mix_embeddings([emb_a, emb_b, emb_c],
                                          [1 - alpha - beta, alpha, beta], task.use_slerp)
                else:
                    emb = _mix_embeddings([emb_a, emb_b], [1 - alpha, alpha],
                                          task.use_slerp)

                gen = torch.Generator(device=device).manual_seed(task.seed)
                with torch.no_grad():
                    image = pipe(
                        prompt_embeds=emb,
                        height=task.height, width=task.width,
                        num_inference_steps=task.steps,
                        guidance_scale=task.guidance_scale,
                        generator=gen,
                    ).images[0]

                # DINOv2 embedding
                dino_input = dino_transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb_dino = dino(dino_input)
                    emb_dino = emb_dino / emb_dino.norm(dim=-1, keepdim=True)

                # Thumbnail
                thumb_bytes = _image_to_thumbnail(image)
                thumb_hash = hashlib.md5(thumb_bytes).hexdigest()

                result_queue.put(CellResult(
                    job_id=task.job_id,
                    gpu_id=gpu_id,
                    row=i, col=j, depth=k,
                    thumbnail_bytes=thumb_bytes,
                    thumbnail_hash=thumb_hash,
                    dino_embedding=emb_dino.cpu().numpy().flatten(),
                ))

        print(f"[GPU {gpu_id}] Job {task.job_id} row {i+1}/{task.grid_size}", flush=True)


def _process_hike(gpu_id, device, pipe, dino, dino_transform, task, result_queue,
                  cancel_epoch=None):
    """Generate the points of one hiker grid. Vertices come from `vertex_coefs` over
    `basis`, so this handles both ordinary prompt triangles (one-hot coefficients) and
    carried exit coordinates."""
    with torch.no_grad():
        base = [pipe.encode_prompt(prompt=t)[0] for t in task.basis]
    verts = [sum(float(c) * base[n] for n, c in enumerate(row) if c != 0.0)
             for row in task.vertex_coefs]

    for (i, j, w0, w1, w2) in task.points:
        if _cancelled(task, cancel_epoch):
            print(f"[GPU {gpu_id}] Cancelled mid-hike", flush=True)
            return
        emb = w0 * verts[0] + w1 * verts[1] + w2 * verts[2]
        gen = torch.Generator(device=device).manual_seed(task.seed)
        with torch.no_grad():
            image = pipe(prompt_embeds=emb, height=task.height, width=task.width,
                         num_inference_steps=task.steps,
                         negative_prompt_embeds=getattr(pipe, "_cached_neg_embeds", None),
                         guidance_scale=task.guidance_scale, generator=gen).images[0]
            e = dino(dino_transform(image).unsqueeze(0).to(device))
            e = e / e.norm(dim=-1, keepdim=True)
        tb = _image_to_thumbnail(image)
        result_queue.put(CellResult(
            job_id=task.job_id, gpu_id=gpu_id, row=int(i), col=int(j),
            thumbnail_bytes=tb, thumbnail_hash=hashlib.md5(tb).hexdigest(),
            dino_embedding=e.cpu().numpy().flatten()))
    print(f"[GPU {gpu_id}] hike {task.job_id}: {len(task.points)} points", flush=True)


def _process_discover(gpu_id, device, pipe, dino, dino_transform, task, result_queue,
                      cancel_epoch=None):
    """Generate one batch of the discovery loop: k-way affine mixtures of a prompt basis.

    The basis is encoded once per batch, so the per-image cost is the diffusion call and
    the DINOv2 pass, exactly as for a grid. Results are reported as CellResult with
    row = the point's index and col = 0, so the existing thumbnail cache and status
    plumbing need no changes.
    """
    with torch.no_grad():
        base = [pipe.encode_prompt(prompt=t)[0] for t in task.basis]

    for idx, weights in task.points:
        if _cancelled(task, cancel_epoch):
            print(f"[GPU {gpu_id}] Cancelled mid-discover", flush=True)
            return
        emb = sum(float(w) * base[n] for n, w in enumerate(weights) if w != 0.0)
        gen = torch.Generator(device=device).manual_seed(task.seed)
        with torch.no_grad():
            image = pipe(prompt_embeds=emb, height=task.height, width=task.width,
                         num_inference_steps=task.steps,
                         negative_prompt_embeds=getattr(pipe, "_cached_neg_embeds", None),
                         guidance_scale=task.guidance_scale, generator=gen).images[0]
            e = dino(dino_transform(image).unsqueeze(0).to(device))
            e = e / e.norm(dim=-1, keepdim=True)
        tb = _image_to_thumbnail(image)
        result_queue.put(CellResult(
            job_id=task.job_id, gpu_id=gpu_id, row=int(idx), col=0,
            thumbnail_bytes=tb, thumbnail_hash=hashlib.md5(tb).hexdigest(),
            dino_embedding=e.cpu().numpy().flatten()))
    print(f"[GPU {gpu_id}] discover {task.job_id}: {len(task.points)} points", flush=True)


def _process_probe(gpu_id, device, pipe, dino, task, result_queue):
    """Run a JVP probe on the resident pipe. jvp_probe patches/unpatches the pipe around the call, so the
    worker's ordinary generation stays bit-identical. Errors are returned as a ProbeResult, not raised, so a
    failed probe never counts as a job failure."""
    from . import jvp_probe
    t0 = time.time()
    try:
        if isinstance(task, ProbeTask):
            with torch.no_grad():
                encoded = [pipe.encode_prompt(prompt=p, device=device) for p in task.prompts]
            embs = [e for e, _ in encoded]
            text_ids = encoded[0][1]
            res = jvp_probe.ridge_normal(pipe, dino, embs, text_ids, task.weights, seed=task.seed, steps=task.steps,
                                         guidance=task.guidance_scale, height=task.height, width=task.width,
                                         use_slerp=task.use_slerp)
            kind = "jvp"
        else:
            res = jvp_probe.token_sensitivity(pipe, dino, task.prompt, seed=task.seed, steps=task.steps,
                                              guidance=task.guidance_scale, height=task.height, width=task.width)
            kind = "token"
        res["gpu_id"] = gpu_id
        result_queue.put(ProbeResult(probe_id=task.probe_id, kind=kind, result=res))
        print(f"[GPU {gpu_id}] probe {task.probe_id} ({kind}) done in {time.time() - t0:.1f}s", flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result_queue.put(ProbeResult(probe_id=task.probe_id, kind="jvp" if isinstance(task, ProbeTask) else "token",
                                     result={}, error=f"{type(exc).__name__}: {exc}"))
        torch.cuda.empty_cache()


def _process_hq(gpu_id, device, pipe, dino, dino_transform, task, result_queue,
                cancel_epoch=None):
    """Re-render specific cells at high quality (512px, 20 steps)."""
    emb_a, emb_b, emb_c, _ = _encode_prompts(pipe, task.prompt_a, task.prompt_b, task.prompt_c)

    for i, j in task.cells:
        # HQ cells are the slowest renders in the pool and used to ignore cancellation
        # entirely (the worker never passed the flag down here)
        if _cancelled(task, cancel_epoch):
            print(f"[GPU {gpu_id}] Cancelled mid-HQ render", flush=True)
            return
        alpha = task.alphas[i]
        beta = task.betas[j]

        if emb_c is not None:
            emb = (1 - alpha - beta) * emb_a + alpha * emb_b + beta * emb_c
        else:
            emb = (1 - alpha) * emb_a + alpha * emb_b

        gen = torch.Generator(device=device).manual_seed(task.seed)
        with torch.no_grad():
            image = pipe(
                prompt_embeds=emb,
                height=config.HQ_HEIGHT, width=config.HQ_WIDTH,
                num_inference_steps=config.HQ_NUM_INFERENCE_STEPS,
                guidance_scale=config.DEFAULT_GUIDANCE_SCALE,
                generator=gen,
            ).images[0]

        # DINOv2
        dino_input = dino_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb_dino = dino(dino_input)
            emb_dino = emb_dino / emb_dino.norm(dim=-1, keepdim=True)

        # Full-size JPEG (not thumbnail)
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=92)
        hq_bytes = buf.getvalue()
        hq_hash = hashlib.md5(hq_bytes).hexdigest()

        result_queue.put(CellResult(
            job_id=task.job_id,
            gpu_id=gpu_id,
            row=i, col=j,
            thumbnail_bytes=hq_bytes,
            thumbnail_hash=hq_hash,
            dino_embedding=emb_dino.cpu().numpy().flatten(),
            is_hq=True,
        ))

    print(f"[GPU {gpu_id}] HQ render {len(task.cells)} cells done", flush=True)


class GPUPool:
    """Manages N GPU worker processes."""

    def __init__(self, n_gpus: int = config.N_GPUS):
        self.n_gpus = n_gpus
        self.ctx = mp.get_context('spawn')
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        # Cancellation is a monotonic epoch, not a flag: a task carries the epoch it was
        # submitted under and aborts once the shared epoch moves past it. A flag could
        # not work here because /cancel and the next /start are milliseconds apart, so
        # the flag was always cleared before any running worker looked at it.
        self.cancel_epoch = self.ctx.Value('q', 0)
        self.task_errors = []      # undrained; pop_task_errors() empties this
        self.recent_errors = []    # capped history for /health
        self.workers = []
        self.ready_count = 0

    def start(self):
        for gpu_id in range(self.n_gpus):
            p = self.ctx.Process(
                target=worker_main,
                args=(gpu_id, self.task_queue, self.result_queue, self.cancel_epoch),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def wait_ready(self, timeout=300):
        """Wait for all workers to signal ready. Returns True if the whole pool is up.

        Liveness is polled as well: a worker that dies during model load never sends its
        ready message, and the loop used to spin out the full deadline and then let the
        server come up with a pool that could not finish a job.
        """
        import time
        deadline = time.time() + timeout
        ready_ids = set()
        while self.ready_count < self.n_gpus and time.time() < deadline:
            try:
                msg = self.result_queue.get(timeout=1)
                if isinstance(msg, dict) and msg.get("type") == "ready":
                    ready_ids.add(msg["gpu_id"])
                    self.ready_count = len(ready_ids)
                    print(f"Worker {msg['gpu_id']} ready ({self.ready_count}/{self.n_gpus})", flush=True)
            except Exception:
                pass
            lost = [i for i, w in enumerate(self.workers)
                    if i not in ready_ids and not w.is_alive()]
            if lost and self.ready_count + len(lost) >= self.n_gpus:
                print(f"Workers {lost} exited during startup "
                      f"(exit codes {[self.workers[i].exitcode for i in lost]})", flush=True)
                break

        if self.ready_count < self.n_gpus:
            live = sum(1 for w in self.workers if w.is_alive())
            print(f"WARNING: only {self.ready_count}/{self.n_gpus} GPU workers ready "
                  f"({live} processes alive)", flush=True)
            # Callers shard rows across range(pool.n_gpus); leaving n_gpus at the
            # requested count hands whole row chunks to workers that do not exist, and
            # the job then stalls partway with no error.
            if live:
                self.n_gpus = min(self.n_gpus, live)
        return self.ready_count >= self.n_gpus

    def submit(self, task):
        # Stamp the current epoch so a cancel issued after this point can be told apart
        # from one issued before it.
        task.epoch = self.cancel_epoch.value
        self.task_queue.put(task)

    def drain_pending(self):
        """Drain all pending tasks from the queue and signal workers to abort current task."""
        with self.cancel_epoch.get_lock():
            self.cancel_epoch.value += 1
        drained = 0
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
                drained += 1
            except Exception:
                break
        if drained:
            print(f"Drained {drained} pending tasks from queue", flush=True)
        return drained

    def collect_results(self) -> list:
        results = []
        while not self.result_queue.empty():
            try:
                r = self.result_queue.get_nowait()
                if isinstance(r, (CellResult, LatentResult, LatentBatchResult, ProbeResult)):
                    results.append(r)
                elif isinstance(r, dict) and r.get("type") in ("task_error", "worker_died"):
                    # keep worker-side failures reachable: without this a crashed task
                    # is only visible in the GPU process's stdout
                    self.task_errors.append(r)
                    del self.task_errors[:-50]
            except Exception:
                break
        return results

    def pop_task_errors(self) -> list:
        """Take and clear the worker-side failures seen since the last call.

        collect_results() cannot return these alongside results (callers read r.job_id
        off a dataclass), so they need their own drain — otherwise a failed task is a
        silent hole in a job's cell count and the job never completes.
        """
        errors, self.task_errors = self.task_errors, []
        # the collector consumes each entry exactly once, so keep a capped copy for
        # /health -- otherwise the only reader would empty the list before it was seen
        self.recent_errors = (self.recent_errors + errors)[-20:]
        return errors

    def shutdown(self):
        # Ask in-flight work to stop before waiting on it: the sentinel is only seen at
        # the top of the worker loop, so without this every join() burned its full
        # timeout against a worker still rendering, and then returned with it alive.
        with self.cancel_epoch.get_lock():
            self.cancel_epoch.value += 1
        for _ in self.workers:
            self.task_queue.put(None)
        # one shared deadline, not 10s per worker: the waits used to serialise
        import time
        deadline = time.time() + 10
        for w in self.workers:
            w.join(timeout=max(0.1, deadline - time.time()))
        for w in self.workers:
            if w.is_alive():
                w.terminate()
                w.join(timeout=5)
            if w.is_alive():
                print(f"Worker {w.pid} did not exit, killing", flush=True)
                w.kill()
                w.join(timeout=5)
