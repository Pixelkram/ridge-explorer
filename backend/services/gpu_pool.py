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


def worker_main(gpu_id: int, task_queue, result_queue):
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

    print(f"[GPU {gpu_id}] Worker ready (FLUX Klein + DINOv2)", flush=True)
    result_queue.put({"type": "ready", "gpu_id": gpu_id})

    while True:
        task = task_queue.get()
        if task is None:
            break

        if isinstance(task, GenerateTask):
            _process_generate(gpu_id, device, pipe, dino, dino_transform, task, result_queue)
        elif isinstance(task, FastScanTask):
            _process_fast_scan(gpu_id, device, pipe, task, result_queue)
        elif isinstance(task, HQTask):
            _process_hq(gpu_id, device, pipe, dino, dino_transform, task, result_queue)


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
    # Normalize to the average norm of the inputs
    avg_norm = sum(e.norm() for e in embs) / len(embs)
    result_norm = result.norm()
    if result_norm > 1e-8:
        result = result * (avg_norm / result_norm)
    return result


def _interpolate_2d(emb_a, emb_b, emb_c, alpha, beta, use_slerp=False):
    """2D interpolation: emb = (1-α-β)*A + α*B + β*C"""
    if emb_c is not None:
        if use_slerp:
            return _nlerp([emb_a, emb_b, emb_c], [1 - alpha - beta, alpha, beta])
        return (1 - alpha - beta) * emb_a + alpha * emb_b + beta * emb_c
    else:
        return (1 - alpha) * emb_a + alpha * emb_b + beta * emb_c


def _image_to_thumbnail(image, size=None):
    """Convert PIL image to JPEG bytes at full resolution."""
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=90)
    return buf.getvalue()


class _ScanContext:
    """Shared state for batched 1-step latent evaluation. Supports 2D and 3D."""
    __slots__ = ('emb_a', 'emb_b', 'emb_c', 'emb_d', 'noise_t', 'text_ids_t',
                 'latent_ids_t', 'batch_t', 'transformer', 'img_tokens')

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

    def evaluate_points(self, points, batch_size=8):
        """Evaluate a list of (alpha, beta[, gamma]) points. Returns (N, D) normalized latents."""
        all_latents = []
        for b_start in range(0, len(points), batch_size):
            batch = points[b_start:b_start + batch_size]
            bs = len(batch)

            embeds = []
            for pt in batch:
                if len(pt) == 3 and self.emb_d is not None:
                    alpha, beta, gamma = pt
                    e = ((1 - alpha - beta - gamma) * self.emb_a +
                         alpha * self.emb_b + beta * self.emb_c + gamma * self.emb_d)
                elif self.emb_c is not None:
                    alpha, beta = pt[0], pt[1]
                    e = (1 - alpha - beta) * self.emb_a + alpha * self.emb_b + beta * self.emb_c
                else:
                    alpha, beta = pt[0], pt[1]
                    e = (1 - alpha) * self.emb_a + alpha * self.emb_b
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


def _process_fast_scan(gpu_id, device, pipe, task, result_queue):
    """Batched fast scan: bypass pipeline overhead, call transformer directly.

    Supports both 2D (3 prompts) and 3D (4 prompts) grids.
    """
    BATCH_SIZE = 8

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
                latents = ctx.evaluate_points(points, BATCH_SIZE)
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
            latents = ctx.evaluate_points(points, BATCH_SIZE)
            result_queue.put(LatentBatchResult(
                job_id=task.job_id, gpu_id=gpu_id,
                row=i, cols=list(range(gs)),
                latent_vectors=latents,
            ))
            print(f"[GPU {gpu_id}] Fast scan {task.job_id} row {i+1}/{gs}", flush=True)



def _process_generate(gpu_id, device, pipe, dino, dino_transform, task, result_queue):
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

                beta = task.betas[j]
                gamma = task.gammas[k] if is_3d else 0.0

                # Interpolated embedding
                if is_3d and emb_d is not None:
                    # 4-prompt 3D: (1-α-β-γ)*A + α*B + β*C + γ*D
                    if task.use_slerp:
                        emb = _nlerp([emb_a, emb_b, emb_c, emb_d],
                                     [1 - alpha - beta - gamma, alpha, beta, gamma])
                    else:
                        emb = ((1 - alpha - beta - gamma) * emb_a +
                               alpha * emb_b + beta * emb_c + gamma * emb_d)
                elif emb_c is not None:
                    if task.use_slerp:
                        emb = _nlerp([emb_a, emb_b, emb_c],
                                     [1 - alpha - beta, alpha, beta])
                    else:
                        emb = (1 - alpha - beta) * emb_a + alpha * emb_b + beta * emb_c
                else:
                    emb = (1 - alpha) * emb_a + alpha * emb_b

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


def _process_hq(gpu_id, device, pipe, dino, dino_transform, task, result_queue):
    """Re-render specific cells at high quality (512px, 20 steps)."""
    emb_a, emb_b, emb_c = _encode_prompts(pipe, task.prompt_a, task.prompt_b, task.prompt_c)

    for i, j in task.cells:
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
        self.workers = []
        self.ready_count = 0

    def start(self):
        for gpu_id in range(self.n_gpus):
            p = self.ctx.Process(
                target=worker_main,
                args=(gpu_id, self.task_queue, self.result_queue),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def wait_ready(self, timeout=300):
        """Wait for all workers to signal ready."""
        import time
        deadline = time.time() + timeout
        while self.ready_count < self.n_gpus and time.time() < deadline:
            try:
                msg = self.result_queue.get(timeout=1)
                if isinstance(msg, dict) and msg.get("type") == "ready":
                    self.ready_count += 1
                    print(f"Worker {msg['gpu_id']} ready ({self.ready_count}/{self.n_gpus})", flush=True)
            except Exception:
                pass

    def submit(self, task):
        self.task_queue.put(task)

    def collect_results(self) -> list:
        results = []
        while not self.result_queue.empty():
            try:
                r = self.result_queue.get_nowait()
                if isinstance(r, (CellResult, LatentResult, LatentBatchResult)):
                    results.append(r)
            except Exception:
                break
        return results

    def shutdown(self):
        for _ in self.workers:
            self.task_queue.put(None)
        for w in self.workers:
            w.join(timeout=10)
