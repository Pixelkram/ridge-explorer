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
    active_cells: set | None = None  # if set, only generate these (row, col) pairs


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
        elif isinstance(task, HQTask):
            _process_hq(gpu_id, device, pipe, dino, dino_transform, task, result_queue)


def _encode_prompts(pipe, prompt_a, prompt_b, prompt_c):
    """Encode 2 or 3 prompts and return embeddings."""
    with torch.no_grad():
        emb_a, _ = pipe.encode_prompt(prompt=prompt_a)
        emb_b, _ = pipe.encode_prompt(prompt=prompt_b)
        if prompt_c:
            emb_c, _ = pipe.encode_prompt(prompt=prompt_c)
        else:
            emb_c = None
    return emb_a, emb_b, emb_c


def _interpolate_embedding(emb_a, emb_b, emb_c, alpha, beta):
    """Compute interpolated embedding for grid position (alpha, beta).

    3-prompt mode: emb = (1-α-β)*A + α*B + β*C
    2-prompt mode: emb = (1-α)*A + α*B + β*perp  (perp computed externally)
    """
    if emb_c is not None:
        return (1 - alpha - beta) * emb_a + alpha * emb_b + beta * emb_c
    else:
        # 2-prompt: alpha interpolates A→B, beta is perpendicular
        # For 2-prompt, caller should pass perp_dir as emb_c
        return (1 - alpha) * emb_a + alpha * emb_b + beta * emb_c


def _image_to_thumbnail(image, size=None):
    """Convert PIL image to JPEG bytes at full resolution."""
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=90)
    return buf.getvalue()


def _process_generate(gpu_id, device, pipe, dino, dino_transform, task, result_queue):
    """Generate images for assigned rows and compute DINOv2 embeddings."""
    emb_a, emb_b, emb_c = _encode_prompts(pipe, task.prompt_a, task.prompt_b, task.prompt_c)

    for i in task.row_indices:
        alpha = task.alphas[i]
        for j in range(task.grid_size):
            # Skip cells not in active set (for refine tasks)
            if task.active_cells is not None and (i, j) not in task.active_cells:
                continue

            beta = task.betas[j]

            # Interpolated embedding
            if emb_c is not None:
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
                row=i, col=j,
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
                if isinstance(r, CellResult):
                    results.append(r)
            except Exception:
                break
        return results

    def shutdown(self):
        for _ in self.workers:
            self.task_queue.put(None)
        for w in self.workers:
            w.join(timeout=10)
